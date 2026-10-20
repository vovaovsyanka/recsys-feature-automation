"""Модель над историей событий: внимание к истории пользователя, зависящее от кандидата (GPU, PyTorch).

Вход - только сырые события пользователя до якоря (последние L прослушиваний: трек, артист, время до якоря, доля
прослушивания, органика) и кандидат (трек, артист); инженерных агрегатов нет. Строки, кандидаты и метки те же, что у бустинга.

Окна подбирает сама модель: вклад событий, совпадающих с кандидатом по треку или артисту, взвешивается обучаемыми ядрами
от времени до якоря и доли прослушивания. Это аналог набора признаков UI.cnt.<окно> и UA.cnt.<окно> с окнами, найденными обучением.
Идея в духе DIN (arXiv:1706.06978) и ETA (arXiv:2108.04468); реализация упрощённая и не воспроизводит эти работы.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from recsys.protocol import VAL

DAY = 86_400
N_TB = 28   # корзины времени (лог-шкала)


def time_bucket(dt_sec: torch.Tensor) -> torch.Tensor:
    h = dt_sec.clamp(min=0).float() / 3600.0
    return (torch.log2(h + 1.0) * 2.2).long().clamp(0, N_TB - 1)


def ratio_bucket(r: torch.Tensor) -> torch.Tensor:
    # r: uint8 доля прослушивания (0..255); корзины: <20, 20-50, 50-90, 90-100, >100
    r = r.long()
    return (r >= 20).long() + (r >= 50).long() + (r >= 90).long() + (r > 100).long()


@dataclass
class HistArrays:
    """Прослушивания, отсортированные по (uid, ts), на GPU."""
    key: torch.Tensor      # int64 uid<<32 | ts  (для searchsorted)
    uid: torch.Tensor
    item: torch.Tensor     # int32 исходный item_id
    art: torch.Tensor      # int32 исходный artist (0 = нет)
    ts: torch.Tensor       # int32
    ratio: torch.Tensor    # uint8
    org: torch.Tensor      # uint8
    ustart: torch.Tensor   # int64 по uid: начало блока пользователя
    item_vocab: torch.Tensor   # item_id -> индекс эмбеддинга (0=OOV)
    art_vocab: torch.Tensor
    n_item: int
    n_art: int


def build_hist(ev: pl.DataFrame, device: str = "cuda", min_item_cnt: int = 3, min_art_cnt: int = 5) -> HistArrays:
    L = ev.filter(pl.col("et") == 0).select("uid", "item_id", "artist", "ts", "ratio", "org").sort("uid", "ts")
    uid = L["uid"].to_numpy().astype(np.int64)
    ts = L["ts"].to_numpy().astype(np.int64)
    key = uid * (1 << 32) + ts
    item = L["item_id"].to_numpy().astype(np.int32)
    art = L["artist"].fill_null(0).to_numpy().astype(np.int32)
    ratio = np.clip(L["ratio"].fill_null(0).to_numpy(), 0, 255).astype(np.uint8)
    org = L["org"].to_numpy().astype(np.uint8)
    # начало блока пользователя
    maxu = int(uid.max()) + 1
    first = np.full(maxu, -1, dtype=np.int64)
    idx = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    first[uid[idx]] = idx
    # словари строятся по частотам до первого якоря val, чтобы не опираться на события val и test
    train_mask = ts < VAL[0] * DAY
    ic = np.bincount(item[train_mask], minlength=int(item.max()) + 1)
    ac = np.bincount(art[train_mask], minlength=int(art.max()) + 1)
    iv = np.zeros(len(ic), dtype=np.int64); keep_i = np.flatnonzero(ic >= min_item_cnt); iv[keep_i] = np.arange(1, len(keep_i) + 1)
    av = np.zeros(len(ac), dtype=np.int64); keep_a = np.flatnonzero(ac >= min_art_cnt); keep_a = keep_a[keep_a > 0]
    av[keep_a] = np.arange(1, len(keep_a) + 1)
    t = lambda a, dt=None: torch.as_tensor(a, device=device) if dt is None else torch.as_tensor(a, device=device, dtype=dt)
    return HistArrays(t(key), t(uid), t(item), t(art), t(ts.astype(np.int32)), t(ratio), t(org), t(first),
                      t(iv), t(av), len(keep_i) + 1, len(keep_a) + 1)


def gather_history(H: HistArrays, uids: torch.Tensor, anchor_ts: torch.Tensor, L: int):
    """Последние L прослушиваний пользователей строго до anchor_ts. Возвращает тензоры [B, L] и маску."""
    q = uids.long() * (1 << 32) + anchor_ts.long()
    end = torch.searchsorted(H.key, q, right=False)          # события с key < q, т.е. ts < a
    start = H.ustart[uids.long()].clamp(min=0)
    pos = end.unsqueeze(1) - 1 - torch.arange(L, device=end.device).unsqueeze(0)     # [B, L], 0 = самое свежее
    valid = pos >= start.unsqueeze(1)
    pos = pos.clamp(min=0)
    return (H.item[pos].long(), H.art[pos].long(), (anchor_ts.unsqueeze(1) - H.ts[pos]).long(), H.ratio[pos], H.org[pos].long(), valid)


class SeqRanker(nn.Module):
    def __init__(self, n_item, n_art, d=32, heads=4, n_kern=8, n_extra=0):
        super().__init__()
        self.n_extra = n_extra
        self.d, self.heads, self.n_kern = d, heads, n_kern
        D = 2 * d
        self.item = nn.Embedding(n_item, d, padding_idx=0)
        self.art = nn.Embedding(n_art, d, padding_idx=0)
        self.tb = nn.Embedding(N_TB, d); self.rb = nn.Embedding(5, d); self.ob = nn.Embedding(2, d)
        self.hproj = nn.Linear(2 * d + 3 * d, D)
        self.cproj = nn.Linear(2 * d, D)
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        # смещения внимания: по корзине времени, совпадению трека/артиста (на голову)
        self.tbias = nn.Embedding(N_TB, heads); self.mi = nn.Parameter(torch.zeros(heads)); self.ma = nn.Parameter(torch.zeros(heads))
        # обучаемые ядра: (корзина времени × корзина доли) -> K весов, отдельно для совпадений трека и артиста
        self.kern_i = nn.Embedding(N_TB * 5, n_kern); self.kern_a = nn.Embedding(N_TB * 5, n_kern)
        nn.init.normal_(self.kern_i.weight, 0, 0.5); nn.init.normal_(self.kern_a.weight, 0, 0.5)
        self.mlp = nn.Sequential(nn.Linear(D + D + 2 * n_kern + 2 + n_extra, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, h, c_item, c_art, c_mask, c_item_raw, c_art_raw, iv, av, extra=None):
        """h = gather_history(...); c_*: [B, N]. Возвращает логиты [B, N]."""
        h_item, h_art, dt, ratio, org, valid = h
        B, L = h_item.shape
        N = c_item.shape[1]
        tbk, rbk = time_bucket(dt), ratio_bucket(ratio)
        he = torch.cat([self.item(iv[h_item]), self.art(av[h_art]), self.tb(tbk), self.rb(rbk), self.ob(org)], -1)
        hv = torch.relu(self.hproj(he))                                  # [B, L, D]
        ce = torch.relu(self.cproj(torch.cat([self.item(iv[c_item]), self.art(av[c_art])], -1)))   # [B, N, D]
        H_, dh = self.heads, hv.shape[-1] // self.heads
        q = self.q(ce).view(B, N, H_, dh).transpose(1, 2)               # [B,H,N,dh]
        k = self.k(hv).view(B, L, H_, dh).transpose(1, 2)
        v = self.v(hv).view(B, L, H_, dh).transpose(1, 2)
        mi = (c_item_raw.unsqueeze(2) == h_item.unsqueeze(1)) & valid.unsqueeze(1)           # [B,N,L]
        ma = (c_art_raw.unsqueeze(2) == h_art.unsqueeze(1)) & valid.unsqueeze(1) & (c_art_raw.unsqueeze(2) > 0)
        s = q @ k.transpose(-1, -2) / dh ** 0.5                          # [B,H,N,L]
        bias = self.tbias(tbk).permute(0, 2, 1).unsqueeze(2)             # [B,H,1,L]
        s = s + bias + self.mi.view(1, H_, 1, 1) * mi.unsqueeze(1) + self.ma.view(1, H_, 1, 1) * ma.unsqueeze(1)
        s = s.masked_fill(~valid.view(B, 1, 1, L), -1e4)
        a = torch.softmax(s, -1)
        pooled = (a @ v).transpose(1, 2).reshape(B, N, -1)               # [B,N,D]
        kid = tbk * 5 + rbk                                              # [B,L]
        wi = torch.nn.functional.softplus(self.kern_i(kid))              # [B,L,K]
        wa = torch.nn.functional.softplus(self.kern_a(kid))
        ki = torch.log1p(mi.float() @ wi)                                # [B,N,K]
        ka = torch.log1p(ma.float() @ wa)
        cnt = torch.stack([torch.log1p(mi.sum(-1).float()), torch.log1p(ma.sum(-1).float())], -1)
        parts = [ce, pooled, ki, ka, cnt] + ([extra] if self.n_extra else [])
        z = torch.cat(parts, -1)
        return self.mlp(z).squeeze(-1)


class GroupData:
    """Строки якорей, сгруппированные по (якорь, пользователь): для батчера."""

    EXTRA = ['s1_rank', 's1_score', 's2_rank', 's2_score', 's3_rank', 's3_score', 's4_rank', 's4_score']

    def __init__(self, store, days: list[int], neg_keep: float = 1.0, seed: int = 0, max_cand: int = 400, extra: bool = False):
        rng = np.random.default_rng(seed)
        self.use_extra = extra
        X_ = []
        self.max_cand = max_cand
        I, A, Y, U, ANC, G0, G1, NP = [], [], [], [], [], [], [], []
        off = 0
        for d in days:
            r = store.rows(d)
            gr = pl.read_parquet(store.rows_dir / "groups" / f"groups_{d:03d}.parquet").select("uid", "n_pos")
            uid = r["uid"].to_numpy(); y = r["label"].to_numpy().astype(np.int8)
            keep = np.ones(len(y), bool)
            if neg_keep < 1.0: keep = (y == 1) | (rng.random(len(y)) < neg_keep)
            r2 = r.filter(pl.Series(keep)) if not keep.all() else r
            uid = uid[keep]; y = y[keep]
            n_pos = r2.select("uid").join(gr, on="uid", how="left")["n_pos"].to_numpy()
            start = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]]); end = np.r_[start[1:], len(uid)]
            if extra:
                ex = np.stack([r2[c].to_numpy().astype(np.float32) for c in self.EXTRA], 1)
                miss = np.isnan(ex)
                ex = np.where(miss, 0.0, ex)
                for j, c in enumerate(self.EXTRA):
                    ex[:, j] = np.log1p(ex[:, j]) if c.endswith('score') else ex[:, j] / 100.0
                X_.append(np.concatenate([ex, miss[:, ::2].astype(np.float32)], 1))
            I.append(r2["item_id"].to_numpy().astype(np.int32)); A.append(r2["artist"].fill_null(0).to_numpy().astype(np.int32))
            Y.append(y); U.append(uid[start]); ANC.append(np.full(len(start), d * DAY, dtype=np.int64))
            G0.append(start + off); G1.append(end + off); NP.append(n_pos[start]); off += len(uid)
            store.drop_rows_cache()
        self.item = np.concatenate(I); self.art = np.concatenate(A); self.y = np.concatenate(Y)
        self.uid = np.concatenate(U); self.anchor_ts = np.concatenate(ANC); self.g0 = np.concatenate(G0)
        self.g1 = np.concatenate(G1); self.npos = np.concatenate(NP)
        self.day = (self.anchor_ts // DAY).astype(np.int32)
        self.extra = np.concatenate(X_) if extra else None
        self.n_extra = 0 if not extra else self.extra.shape[1]

    def __len__(self):
        return len(self.uid)

    def batch(self, gidx: np.ndarray, device="cuda"):
        sizes = self.g1[gidx] - self.g0[gidx]
        N = int(min(sizes.max(), self.max_cand))
        B = len(gidx)
        ci = np.zeros((B, N), np.int32); ca = np.zeros((B, N), np.int32); cy = np.zeros((B, N), np.float32); cm = np.zeros((B, N), bool)
        ce = np.zeros((B, N, self.n_extra), np.float32) if self.use_extra else None
        for b, g in enumerate(gidx):
            s, e = self.g0[g], min(self.g1[g], self.g0[g] + N)
            n = e - s
            ci[b, :n] = self.item[s:e]; ca[b, :n] = self.art[s:e]; cy[b, :n] = self.y[s:e]; cm[b, :n] = True
            if ce is not None: ce[b, :n] = self.extra[s:e]
        T = lambda x: torch.as_tensor(x, device=device)
        return (T(self.uid[gidx].astype(np.int64)), T(self.anchor_ts[gidx]), T(ci).long(), T(ca).long(), T(cy), T(cm),
                T(ce) if ce is not None else None)


def run_epoch(model, H, data: GroupData, order: np.ndarray, bs: int, L: int, opt=None, scaler=None, pos_w: float = 1.0,
              collect: bool = False, device="cuda", loss_type: str = "bce"):
    train = opt is not None
    model.train(train)
    tot, nb = 0.0, 0
    scores = []
    for i in range(0, len(order), bs):
        gidx = order[i:i + bs]
        uid, anc, ci, ca, cy, cm, cex = data.batch(gidx, device)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.float16):
            h = gather_history(H, uid, anc, L)
            logit = model(h, H.item_vocab[ci], H.art_vocab[ca], cm, ci, ca, H.item_vocab, H.art_vocab, cex)
        logit = logit.float()
        if train:
            if loss_type == "listwise":
                lg = logit.masked_fill(~cm, -1e4)
                logp = torch.log_softmax(lg, -1)
                npos = cy.sum(1)
                per = -(cy * logp).sum(1) / npos.clamp(min=1)
                loss = (per * (npos > 0)).sum() / (npos > 0).sum().clamp(min=1)
            else:
                w = (cy * (pos_w - 1.0) + 1.0) * cm
                loss = (Fn.binary_cross_entropy_with_logits(logit, cy, reduction="none") * w).sum() / cm.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        if collect:
            lg = logit.detach().cpu().numpy(); m = cm.cpu().numpy()
            for b in range(len(gidx)):
                n = int(m[b].sum()); scores.append(lg[b, :n])
    return (tot / max(nb, 1)), scores
