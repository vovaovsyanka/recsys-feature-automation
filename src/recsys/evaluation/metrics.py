"""Метрики ранжирования по группам (якорь, пользователь) + бутстреп по пользователям.

Знаменатели считаются по ВСЕМ позитивам группы (в т.ч. не попавшим в кандидаты), поэтому метрики
напрямую отражают потолок кандидатной стадии. Recall@K = hits@K / n_pos; NDCG@K с идеальным DCG по min(K, n_pos).
"""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

KS = (10, 100)
_DISC = 1.0 / np.log2(np.arange(2, 5002))


def group_metrics(gid: np.ndarray, score: np.ndarray, label: np.ndarray, n_pos: np.ndarray, ks=KS, seed: int = 0):
    """gid - целочисленный ключ группы (непрерывные блоки не обязательны). n_pos - по строке: всего позитивов группы.
    Возвращает (unique_gid, dict metric-> array по группам)."""
    rng = np.random.default_rng(seed)
    tie = rng.random(len(score))                      # случайное разрешение ничьих (детерминированно по seed)
    order = np.lexsort((tie, -score.astype(np.float64), gid))
    g, y, npos = gid[order], label[order].astype(np.float64), n_pos[order]
    start = np.r_[True, g[1:] != g[:-1]]
    idx = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    rank = np.arange(len(g)) - first[idx]
    ug = g[first]
    npos_g = npos[first].astype(np.float64)
    out = {}
    for k in ks:
        m = rank < k
        hits = np.bincount(idx[m], weights=y[m], minlength=len(first))
        dcg = np.bincount(idx[m], weights=y[m] * _DISC[np.minimum(rank[m], len(_DISC) - 1)], minlength=len(first))
        cum = np.r_[0.0, np.cumsum(_DISC)]
        idcg = cum[np.minimum(npos_g, k).astype(int)]
        out[f"ndcg@{k}"] = dcg / idcg
        out[f"recall@{k}"] = hits / npos_g
    return ug, out


def summarize(per_group: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(v.mean()) for k, v in per_group.items()}


def user_level(ug: np.ndarray, per_group: dict[str, np.ndarray], user_of_group: np.ndarray):
    """Суммы метрик и число групп по пользователям - для быстрого бутстрепа."""
    uu, inv = np.unique(user_of_group, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sums = {k: np.bincount(inv, weights=v) for k, v in per_group.items()}
    return uu, cnt, sums


def bootstrap_ci(cnt, sums_a, sums_b=None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """ДИ среднего по группам с ресэмплингом пользователей; при sums_b - ДИ разности (a - b), парный."""
    rng = np.random.default_rng(seed)
    n = len(cnt)
    res = {}
    for k in sums_a:
        vals = np.empty(n_boot)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n)
            den = (w * cnt).sum()
            num_a = (w * sums_a[k]).sum() / den
            vals[b] = num_a - ((w * sums_b[k]).sum() / den if sums_b is not None else 0.0)
        pt = sums_a[k].sum() / cnt.sum() - (sums_b[k].sum() / cnt.sum() if sums_b is not None else 0.0)
        res[k] = dict(mean=float(pt), lo=float(np.quantile(vals, alpha / 2)), hi=float(np.quantile(vals, 1 - alpha / 2)))
    return res


def pointwise(label: np.ndarray, p: np.ndarray | None, score: np.ndarray):
    out = {"auc": float(roc_auc_score(label, score))}
    if p is not None:
        out["logloss"] = float(log_loss(label, np.clip(p, 1e-7, 1 - 1e-7)))
    return out


def make_gid(anchor_idx: np.ndarray, uid: np.ndarray) -> np.ndarray:
    return anchor_idx.astype(np.int64) * (1 << 32) + uid.astype(np.int64)


def evaluate(anchor_idx, uid, score, label, n_pos, p=None, seed=0, boot: int = 0):
    gid = make_gid(anchor_idx, uid)
    ug, pg = group_metrics(gid, score, label, n_pos, seed=seed)
    res = summarize(pg)
    res.update(pointwise(label, p, score))
    res["n_groups"] = int(len(ug))
    pack = None
    if boot:
        uoff = (ug & ((1 << 32) - 1))
        pack = user_level(ug, pg, uoff)
    return res, pg, ug, pack
