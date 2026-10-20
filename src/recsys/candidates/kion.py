"""KION: протокол, кандидаты и метки. Время - сутки (индекс от 2021-03-13), окно метки - 7 суток.

Для якоря a (индекс суток): история - взаимодействия с day < a, метка - пара (пользователь, фильм) встречается в [a, a + 7).
Группа - пользователь с непустой историей и хотя бы одним позитивом; берётся 10% пользователей (по хэшу) со всей их историей.
Глобальные признаки (популярность, жанры) считаются по всем пользователям, но тоже только по дням до a.

Кандидаты (уже просмотренное исключается):
  s1 - популярные за 14 суток;   s2 - популярные за 60 суток в трёх любимых жанрах пользователя;
  s3 - item-kNN: косинус по совместным просмотрам за 90 суток от пяти последних просмотров пользователя.
Столбцы источников называются так же, как у Yambda (s1_*, s2_*, s3_*; s4_* пусты), чтобы использовать общий код.

Разбиение: test a = 156 (окно 156-162), val a = 149 (окно 149-155), train a <= 142 (окна заканчиваются к 149).
"""
import numpy as np
import polars as pl
import scipy.sparse as sp

HORIZON_DAYS = 7
A_TEST, A_VAL, A_TRAIN_LAST, A_TRAIN_FIRST = 156, 149, 142, 60
N_DAYS = 163
DAY = 86400


def check_protocol(train: list[int]) -> None:
    assert max(train) + HORIZON_DAYS <= A_VAL, "метки train пересекаются с якорем val"
    assert A_VAL + HORIZON_DAYS <= A_TEST, "метки val пересекаются с якорем test"
    assert A_TEST + HORIZON_DAYS <= N_DAYS, "окно test выходит за пределы данных"


def train_anchors(step: int = 3) -> list[int]:
    days = list(range(A_TRAIN_FIRST, A_TRAIN_LAST + 1, step))
    check_protocol(days)
    return days


def user_hash_mask(uid: np.ndarray, frac: float) -> np.ndarray:
    return ((uid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(10007)) < int(frac * 10007)


def knn_matrix(events: pl.DataFrame, a_day: int, n_items: int, window_days: int = 90) -> np.ndarray:
    """Косинусная близость фильмов по совместным просмотрам за `window_days` суток до a."""
    h = events.filter((pl.col("ts") >= (a_day - window_days) * DAY) & (pl.col("ts") < a_day * DAY)).select("uid", "item_id")
    _, user_idx = np.unique(h["uid"].to_numpy(), return_inverse=True)
    X = sp.csr_matrix((np.ones(h.height, np.float32), (user_idx, h["item_id"].to_numpy())), shape=(user_idx.max() + 1, n_items))
    X.data[:] = 1.0
    norm = np.sqrt(np.maximum(np.asarray(X.sum(0)).ravel(), 1.0))
    C = (X.T @ X).toarray().astype(np.float32)
    C /= norm[:, None]
    C /= norm[None, :]
    np.fill_diagonal(C, 0.0)
    return C


def _knn_candidates(events_before: pl.DataFrame, users: pl.DataFrame, seen: pl.DataFrame, a_day: int, n_items: int,
                    last_k: int, top_k: int) -> pl.DataFrame:
    C = knn_matrix(events_before, a_day, n_items)
    last = (events_before.join(users, on="uid", how="semi").sort(["uid", "ts"], descending=[False, True])
            .with_columns(pl.int_range(pl.len()).over("uid").alias("recency")).filter(pl.col("recency") < last_k))
    uids = users["uid"].to_numpy()
    user_pos = {int(u): j for j, u in enumerate(uids)}
    rows = np.array([user_pos[int(u)] for u in last["uid"].to_numpy()])
    weights = 1.0 / (1.0 + last["recency"].to_numpy().astype(np.float32))
    W = sp.csr_matrix((weights, (rows, last["item_id"].to_numpy())), shape=(len(uids), n_items))
    seen_rows = np.array([user_pos[int(u)] for u in seen["uid"].to_numpy()])
    seen_csr = sp.csr_matrix((np.ones(seen.height, np.float32), (seen_rows, seen["item_id"].to_numpy())), shape=(len(uids), n_items))

    out = {"uid": [], "item_id": [], "s3_score": [], "s3_rank": []}
    chunk = 4000
    for s in range(0, len(uids), chunk):
        scores = np.asarray(W[s:s + chunk] @ C)
        scores[seen_csr[s:s + chunk].nonzero()] = -1.0               # просмотренное не рекомендуем
        top = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, top, 1)
        order = np.argsort(-top_scores, 1)
        top, top_scores = np.take_along_axis(top, order, 1), np.take_along_axis(top_scores, order, 1)
        ok = top_scores > 0
        out["uid"].append(np.repeat(uids[s:s + chunk], top_k).reshape(-1, top_k)[ok])
        out["item_id"].append(top[ok])
        out["s3_score"].append(top_scores[ok])
        out["s3_rank"].append(np.broadcast_to(np.arange(top_k), top.shape)[ok])
    return pl.DataFrame({"uid": np.concatenate(out["uid"]).astype(np.uint32),
                         "item_id": np.concatenate(out["item_id"]).astype(np.uint32),
                         "s3_score": np.concatenate(out["s3_score"]).astype(np.float64),
                         "s3_rank": np.concatenate(out["s3_rank"]).astype(np.int64)})


def build_anchor(events: pl.DataFrame, ts_sorted: np.ndarray, items: pl.DataFrame, a_day: int, user_frac: float,
                 n_items: int, k1=150, k2_genres=3, k2_per=40, k3_last=5, k3_top=100):
    """Возвращает (rows, groups) для якоря a_day; см. описание модуля."""
    i_a = int(np.searchsorted(ts_sorted, a_day * DAY, "left"))
    i_b = int(np.searchsorted(ts_sorted, (a_day + HORIZON_DAYS) * DAY, "left"))
    hist, window = events.slice(0, i_a), events.slice(i_a, i_b - i_a)
    pos = window.select("uid", "item_id").unique().with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    users = pos.select("uid").unique().join(hist.select("uid").unique(), on="uid", how="inner")
    if user_frac < 1.0:
        users = users.filter(pl.Series(user_hash_mask(users["uid"].to_numpy(), user_frac)))
    pos = pos.join(users, on="uid", how="semi")
    seen = hist.join(users, on="uid", how="semi").select("uid", "item_id").unique()

    # s1: популярные за 14 суток по числу пользователей
    popular = (hist.filter(pl.col("ts") >= (a_day - 14) * DAY).group_by("item_id")
               .agg(pl.col("uid").n_unique().alias("s1_score")).sort("s1_score", descending=True).head(k1)
               .with_columns(pl.int_range(pl.len()).alias("s1_rank")))
    s1 = users.join(popular, how="cross")

    # s2: любимые жанры пользователя и популярные за 60 суток фильмы этих жанров
    genres = (hist.join(users, on="uid", how="semi").group_by("uid", "artist").agg(pl.len().alias("n"))
              .sort(["uid", "n"], descending=[False, True])
              .with_columns(pl.int_range(pl.len()).over("uid").alias("genre_rank")).filter(pl.col("genre_rank") < k2_genres))
    by_genre = (hist.filter(pl.col("ts") >= (a_day - 60) * DAY).group_by("artist", "item_id")
                .agg(pl.col("uid").n_unique().alias("pop")).sort(["artist", "pop"], descending=[False, True])
                .with_columns(pl.int_range(pl.len()).over("artist").alias("pop_rank")).filter(pl.col("pop_rank") < k2_per))
    s2 = (genres.join(by_genre, on="artist", how="inner")
          .with_columns((pl.col("genre_rank") * k2_per + pl.col("pop_rank")).alias("s2_rank"),
                        pl.col("pop").cast(pl.Float64).alias("s2_score"))
          .select("uid", "item_id", "s2_rank", "s2_score").sort("uid", "s2_rank")
          .unique(["uid", "item_id"], keep="first", maintain_order=True))

    s3 = _knn_candidates(hist, users, seen, a_day, n_items, k3_last, k3_top)

    sources = [("s1", s1), ("s2", s2), ("s3", s3)]
    rows = pl.concat([s.select("uid", "item_id") for _, s in sources]).unique().join(seen, on=["uid", "item_id"], how="anti")
    for name, s in sources:
        rows = rows.join(s.select("uid", "item_id", f"{name}_rank", f"{name}_score"), on=["uid", "item_id"], how="left")
    rows = rows.with_columns(pl.lit(None, dtype=pl.Int64).alias("s4_rank"), pl.lit(None, dtype=pl.Float64).alias("s4_score"))
    rows = (rows.join(items.select("item_id", "artist", "album"), on="item_id", how="left")
            .join(pos, on=["uid", "item_id"], how="left").with_columns(pl.col("label").fill_null(0)).sort("uid", "item_id"))
    groups = (pos.group_by("uid").agg(pl.len().alias("n_pos"))
              .join(rows.group_by("uid").agg(pl.col("label").sum().alias("n_pos_in_cand"), pl.len().alias("n_cand")),
                    on="uid", how="left")
              .with_columns(pl.col("n_pos_in_cand").fill_null(0), pl.col("n_cand").fill_null(0)))
    return rows, groups
