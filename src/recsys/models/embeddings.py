"""Векторные признаки из обучаемых представлений: ALS и LightGCN по графу «пользователь - трек - артист».

Временная честность. Модель на контрольной точке c обучается только на событиях с ts < c * сутки (рёбра - Listen+, вес
ребра затухает с давностью). Для якоря a берётся модель с наибольшей точкой c <= a: она устарела на 0-6 суток, как при
регулярном переобучении в продакшене. Метки окна после якоря модель не видит никогда.

Идеи: LightGCN (He et al., arXiv:2002.02126), PinSage (Ying et al., arXiv:1806.01973), ALS для неявной обратной связи
(Hu, Koren, Volinsky, ICDM 2008).
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from recsys.protocol import DAY, LISTEN_PLUS

CHECKPOINTS = list(range(145, 286, 7)) + [292]


def checkpoint_for(anchor_day: int) -> int:
    return max(c for c in CHECKPOINTS if c <= anchor_day)


def edges_before(events: pl.DataFrame, checkpoint_day: int, halflife_days: float = 21.0, min_item_users: int = 2):
    """Рёбра пользователь - трек (Listen+) до контрольной точки с весом - затухающим числом прослушиваний.

    Возвращает индексы (user_idx, item_idx), веса, отсортированные массивы id пользователей и треков и артиста каждого трека (-1, если нет).
    """
    cut = checkpoint_day * DAY
    history = events.filter((pl.col("et") == 0) & (pl.col("ratio") >= LISTEN_PLUS) & (pl.col("ts") < cut))
    history = history.with_columns((0.5 ** ((cut - pl.col("ts").cast(pl.Float64)) / (halflife_days * DAY))).alias("w"))
    pairs = history.group_by("uid", "item_id").agg(pl.col("w").sum())
    popular = pairs.group_by("item_id").agg(pl.len().alias("n_users")).filter(pl.col("n_users") >= min_item_users)["item_id"]
    pairs = pairs.filter(pl.col("item_id").is_in(popular.implode()))
    users = np.sort(pairs["uid"].unique().to_numpy())
    items = np.sort(pairs["item_id"].unique().to_numpy())
    user_idx = np.searchsorted(users, pairs["uid"].to_numpy())
    item_idx = np.searchsorted(items, pairs["item_id"].to_numpy())
    artists = events.select("item_id", "artist").unique("item_id").filter(pl.col("item_id").is_in(pl.Series(items).implode()))
    artist_of = dict(zip(artists["item_id"].to_list(), artists["artist"].to_list()))
    item_artist = np.array([artist_of.get(int(x)) if artist_of.get(int(x)) is not None else -1 for x in items], dtype=np.int64)
    return user_idx, item_idx, pairs["w"].to_numpy().astype(np.float32), users, items, item_artist


def train_als(user_idx, item_idx, weights, n_users, n_items, factors=64, iterations=12, regularization=0.05, alpha=10.0, seed=0):
    from implicit.als import AlternatingLeastSquares
    confidence = sp.csr_matrix(((1.0 + alpha * np.log1p(weights)).astype(np.float32), (user_idx, item_idx)),
                               shape=(n_users, n_items))
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, random_state=seed,
                                    use_gpu=False, calculate_training_loss=False)
    model.fit(confidence, show_progress=False)
    return np.asarray(model.user_factors), np.asarray(model.item_factors)


def train_lightgcn(user_idx, item_idx, weights, n_users, n_items, item_artist, dim=64, layers=2, steps=600, batch=16384,
                   lr=0.01, reg=1e-4, seed=0, device="cuda"):
    """LightGCN по графу из пользователей, треков и артистов; обучение BPR с равномерными негативами.

    Возвращает эмбеддинги пользователей, треков, артистов и словарь артист -> строка в матрице артистов.
    """
    torch.manual_seed(seed)
    artist_ids = np.unique(item_artist[item_artist >= 0])
    artist_row = {int(a): k for k, a in enumerate(artist_ids)}
    n_nodes = n_users + n_items + len(artist_ids)
    linked = np.flatnonzero(item_artist >= 0)
    linked_artist = np.array([artist_row[int(item_artist[k])] for k in linked], dtype=np.int64)
    # граф неориентированный: рёбра пользователь-трек (с весами) и трек-артист (вес 1)
    src = np.concatenate([user_idx, n_users + item_idx, n_users + linked, n_users + n_items + linked_artist])
    dst = np.concatenate([n_users + item_idx, user_idx, n_users + n_items + linked_artist, n_users + linked])
    w = np.concatenate([weights, weights, np.ones(len(linked), np.float32), np.ones(len(linked), np.float32)])
    degree = np.bincount(src, weights=w, minlength=n_nodes).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
    values = (w * inv_sqrt[src] * inv_sqrt[dst]).astype(np.float32)
    adj = torch.sparse_coo_tensor(torch.as_tensor(np.stack([src, dst])), torch.as_tensor(values), (n_nodes, n_nodes)).coalesce().to(device)
    emb = torch.nn.Parameter(torch.randn(n_nodes, dim, device=device) * 0.1)
    opt = torch.optim.Adam([emb], lr=lr)
    cum_weights = torch.as_tensor(np.cumsum(weights, dtype=np.float64), device=device)   # выбор положительных рёбер пропорционально весу
    users_t, items_t = torch.as_tensor(user_idx, device=device), torch.as_tensor(item_idx, device=device)

    def propagate():
        e, out = emb, [emb]
        for _ in range(layers):
            e = torch.sparse.mm(adj, e)
            out.append(e)
        return torch.stack(out).mean(0)

    for _ in range(steps):
        r = torch.rand(batch, device=device, dtype=torch.float64) * cum_weights[-1]
        edge = torch.searchsorted(cum_weights, r).clamp(max=len(weights) - 1)
        u, pos = users_t[edge], items_t[edge]
        neg = torch.randint(0, n_items, (batch,), device=device)
        f = propagate()
        eu, ep, en = f[u], f[n_users + pos], f[n_users + neg]
        bpr = -torch.nn.functional.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        l2 = (emb[u].pow(2).sum() + emb[n_users + pos].pow(2).sum() + emb[n_users + neg].pow(2).sum()) / batch
        (bpr + reg * l2).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        f = propagate().cpu().numpy()
    return f[:n_users], f[n_users:n_users + n_items], f[n_users + n_items:], artist_row


def score_rows(rows: pl.DataFrame, users, items, U, I, A=None, artist_row=None, chunk=1_000_000):
    """Скалярные произведения для строк якоря: пользователь-трек (ui) и, если есть эмбеддинги артистов, пользователь-артист (ua).
    NaN там, где пользователь, трек или артист не вошли в словарь модели."""
    uid, iid = rows["uid"].to_numpy(), rows["item_id"].to_numpy()
    u_pos = np.searchsorted(users, uid)
    u_ok = (u_pos < len(users)) & (users[np.minimum(u_pos, len(users) - 1)] == uid)
    i_pos = np.searchsorted(items, iid)
    i_ok = (i_pos < len(items)) & (items[np.minimum(i_pos, len(items) - 1)] == iid)
    ui = np.full(len(uid), np.nan, np.float32)
    ua = np.full(len(uid), np.nan, np.float32)
    ok = np.flatnonzero(u_ok & i_ok)
    for s in range(0, len(ok), chunk):
        idx = ok[s:s + chunk]
        ui[idx] = np.einsum("ij,ij->i", U[u_pos[idx]], I[i_pos[idx]])
    if A is not None:
        artist = rows["artist"].to_numpy()
        a_row = np.array([artist_row.get(int(a), -1) if a is not None and a == a else -1
                          for a in np.nan_to_num(artist.astype(np.float64), nan=-1)], dtype=np.int64)
        idx = np.flatnonzero(u_ok & (a_row >= 0))
        ua[idx] = np.einsum("ij,ij->i", U[u_pos[idx]], A[a_row[idx]])
    return ui, ua
