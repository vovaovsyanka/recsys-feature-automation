"""Кандидаты и метки для Yambda. Кандидаты общие для всех сравниваемых моделей.

Для якоря a (секунды):
  история    - события с ts < a; из неё строятся и кандидаты, и признаки;
  окно метки - [a + GAP, a + GAP + сутки); используется только для меток;
  группа     - пользователь с непустой историей и хотя бы одним Listen+ в окне (условная оценка, как в статье Yambda).

Источники кандидатов (в скобках имя столбцов):
  s1 - топ по затухающему личному счётчику Listen+;   s2 - топ по полному личному счётчику Listen+;
  s3 - популярные за 30 суток треки любимых артистов пользователя;   s4 - глобальные хиты недели.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from recsys.config import DATA
from recsys.protocol import DAY, GAP, HORIZON, LISTEN_PLUS


@dataclass(frozen=True)
class CandidateConfig:
    s1_n: int = 200
    s1_halflife_days: float = 14.0
    s2_n: int = 100
    s3_artists: int = 5
    s3_per_artist: int = 8
    s4_n: int = 30


def history_before(events: pl.DataFrame, ts_sorted: np.ndarray, anchor: int) -> pl.DataFrame:
    """События строго до якоря (таблица отсортирована по ts)."""
    return events.slice(0, int(np.searchsorted(ts_sorted, anchor, side="left")))


def positives(events: pl.DataFrame, ts_sorted: np.ndarray, anchor: int) -> pl.DataFrame:
    lo, hi = anchor + GAP, anchor + GAP + HORIZON
    i0, i1 = int(np.searchsorted(ts_sorted, lo, "left")), int(np.searchsorted(ts_sorted, hi, "left"))
    window = events.slice(i0, i1 - i0)
    return (window.filter((pl.col("et") == 0) & (pl.col("ratio") >= LISTEN_PLUS))
            .select("uid", "item_id").unique().with_columns(pl.lit(1, dtype=pl.UInt8).alias("label")))


def _top_n(df: pl.DataFrame, by: str, score: str, n: int, rank_name: str) -> pl.DataFrame:
    ranked = df.sort([by, score], descending=[False, True])
    return ranked.with_columns(pl.int_range(pl.len()).over(by).alias(rank_name)).filter(pl.col(rank_name) < n)


def build_anchor(events: pl.DataFrame, ts_sorted: np.ndarray, items: pl.DataFrame, anchor: int, cfg: CandidateConfig):
    """Возвращает (rows, groups): кандидаты с метками и источниками; число позитивов и их попадание в кандидаты по пользователям."""
    pos = positives(events, ts_sorted, anchor)
    hist = history_before(events, ts_sorted, anchor)
    listens = hist.filter(pl.col("et") == 0)
    users = pos.select("uid").unique().join(listens.select("uid").unique(), on="uid", how="inner")
    pos = pos.join(users, on="uid", how="semi")
    plus = listens.join(users, on="uid", how="semi").filter(pl.col("ratio") >= LISTEN_PLUS)

    # s1, s2: личные счётчики по парам (пользователь, трек)
    weighted = plus.with_columns((0.5 ** ((anchor - pl.col("ts").cast(pl.Float64)) / (cfg.s1_halflife_days * DAY))).alias("w"))
    pairs = weighted.group_by("uid", "item_id").agg(pl.col("w").sum().alias("s1_score"), pl.len().alias("s2_score"))
    s1 = _top_n(pairs.select("uid", "item_id", "s1_score"), "uid", "s1_score", cfg.s1_n, "s1_rank")
    s2 = _top_n(pairs.select("uid", "item_id", pl.col("s2_score").cast(pl.Float64)), "uid", "s2_score", cfg.s2_n, "s2_rank")

    # s3: любимые артисты пользователя -> популярные за 30 суток треки этих артистов
    artists = weighted.filter(pl.col("artist").is_not_null()).group_by("uid", "artist").agg(pl.col("w").sum().alias("ua_score"))
    artists = _top_n(artists, "uid", "ua_score", cfg.s3_artists, "ua_rank")
    recent = listens.filter((pl.col("ts") >= anchor - 30 * DAY) & (pl.col("ratio") >= LISTEN_PLUS) & pl.col("artist").is_not_null())
    popular = _top_n(recent.group_by("artist", "item_id").agg(pl.len().alias("pop30")), "artist", "pop30",
                     cfg.s3_per_artist, "ap_rank")
    s3 = (artists.join(popular, on="artist", how="inner")
          .with_columns((pl.col("ua_rank") * cfg.s3_per_artist + pl.col("ap_rank")).alias("s3_rank"),
                        pl.col("pop30").cast(pl.Float64).alias("s3_score"))
          .select("uid", "item_id", "s3_rank", "s3_score")
          .sort("uid", "s3_rank").unique(["uid", "item_id"], keep="first", maintain_order=True))

    # s4: хиты недели одинаковы для всех пользователей
    hits = (listens.filter((pl.col("ts") >= anchor - 7 * DAY) & (pl.col("ratio") >= LISTEN_PLUS))
            .group_by("item_id").agg(pl.len().alias("s4_score")).sort("s4_score", descending=True).head(cfg.s4_n)
            .with_columns(pl.int_range(pl.len()).alias("s4_rank")))
    s4 = users.join(hits, how="cross")

    sources = [("s1", s1), ("s2", s2), ("s3", s3), ("s4", s4)]
    rows = pl.concat([s.select("uid", "item_id") for _, s in sources]).unique()
    for name, s in sources:
        rows = rows.join(s.select("uid", "item_id", f"{name}_rank", f"{name}_score"), on=["uid", "item_id"], how="left")
    rows = (rows.join(items, on="item_id", how="left").join(pos, on=["uid", "item_id"], how="left")
            .with_columns(pl.col("label").fill_null(0)).sort("uid", "item_id"))
    groups = (pos.group_by("uid").agg(pl.len().alias("n_pos"))
              .join(rows.group_by("uid").agg(pl.col("label").sum().alias("n_pos_in_cand"), pl.len().alias("n_cand")),
                    on="uid", how="left")
              .with_columns(pl.col("n_pos_in_cand").fill_null(0), pl.col("n_cand").fill_null(0)))
    return rows, groups


def rows_dir(tag: str):
    d = DATA / "rows" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(tag: str, cfg: CandidateConfig) -> None:
    (rows_dir(tag) / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
