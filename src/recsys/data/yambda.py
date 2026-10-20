"""Единая таблица событий Yambda-50M.

Колонки: uid, item_id, ts (секунды), et (тип события), org (1 - органическое), ratio (доля прослушивания, %), tlen (длина трека, с),
artist, album. Типы событий: 0 прослушивание, 1 лайк, 2 дизлайк, 3 снятие лайка, 4 снятие дизлайка. Таблица отсортирована по (ts, uid).
Для треков с несколькими артистами/альбомами берётся минимальный id - так результат детерминирован.
"""
import polars as pl

from recsys.config import DATA

RAW = DATA / "raw" / "yambda_50m"
EVENTS = DATA / "processed" / "yambda50m_events.parquet"
ITEMS = DATA / "processed" / "yambda50m_items.parquet"

EVENT_TYPES = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}
_COLUMNS = ["uid", "item_id", "timestamp", "et", "is_organic", "ratio", "tlen"]


def build() -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    listens = pl.read_parquet(RAW / "listens.parquet").with_columns(
        pl.lit(EVENT_TYPES["listen"], dtype=pl.UInt8).alias("et"),
        pl.col("played_ratio_pct").cast(pl.Float32).alias("ratio"),
        pl.col("track_length_seconds").alias("tlen"))
    parts = [listens.select(_COLUMNS)]
    for name in ["like", "dislike", "unlike", "undislike"]:
        d = pl.read_parquet(RAW / f"{name}s.parquet").with_columns(
            pl.lit(EVENT_TYPES[name], dtype=pl.UInt8).alias("et"),
            pl.lit(None, dtype=pl.Float32).alias("ratio"),
            pl.lit(None, dtype=pl.UInt32).alias("tlen"))
        parts.append(d.select(_COLUMNS))
    events = pl.concat(parts).rename({"timestamp": "ts", "is_organic": "org"})
    artist = pl.read_parquet(RAW / "artist_item_mapping.parquet").group_by("item_id").agg(pl.col("artist_id").min().alias("artist"))
    album = pl.read_parquet(RAW / "album_item_mapping.parquet").group_by("item_id").agg(pl.col("album_id").min().alias("album"))
    items = artist.join(album, on="item_id", how="full", coalesce=True)
    events = events.join(items, on="item_id", how="left").sort("ts", "uid", maintain_order=True)
    events.write_parquet(EVENTS, compression="zstd")
    items.write_parquet(ITEMS)


def load_events() -> pl.DataFrame:
    return pl.read_parquet(EVENTS)


def load_items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS)
