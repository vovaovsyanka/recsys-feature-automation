"""KION в схеме событий движка признаков (та же, что у Yambda): так можно использовать общий код без изменений.

Соответствие колонок:
  uid, item_id        <- user_id, item_id
  ts                  <- (last_watch_dt - 2021-03-13) в сутках * 86400; время известно только с точностью до дня
  et = 0, org = 1     просмотр; рекомендаций в данных нет
  ratio, tlen         <- watched_pct (0..100), total_dur (с)
  artist, album       <- id первого жанра и id первого режиссёра (косвенные связи «жанр», «режиссёр»)
При повторных просмотрах в данных оставлена только последняя запись (длительность и процент просуммированы; Petrov et al., 2022),
поэтому пара (пользователь, фильм) встречается один раз и датирована последним просмотром.
"""
import polars as pl

from recsys.config import DATA

RAW = DATA / "raw" / "kion" / "data_en"
EVENTS = DATA / "processed" / "kion_events.parquet"
ITEMS = DATA / "processed" / "kion_items.parquet"
D0 = "2021-03-13"      # первый день данных
N_DAYS = 163           # 2021-03-13 ... 2021-08-22 включительно


def build() -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    interactions = pl.read_csv(RAW / "interactions.csv").with_columns(pl.col("last_watch_dt").str.to_date("%Y-%m-%d"))
    raw_items = pl.read_csv(RAW / "items_en.csv", infer_schema_length=10000).drop("")
    first_genre = raw_items["genres"].str.split(",").list.first().str.strip_chars().fill_null("unk")
    first_director = raw_items["directors_translated"].str.split(",").list.first().str.strip_chars()
    genre_ids = {g: i + 1 for i, g in enumerate(sorted(first_genre.unique().to_list()))}
    director_ids = {d: i + 1 for i, d in enumerate(sorted(first_director.drop_nulls().unique().to_list()))}
    items = pl.DataFrame({
        "item_id": raw_items["item_id"].cast(pl.UInt32),
        "artist": first_genre.replace_strict(genre_ids, default=0).cast(pl.UInt32),
        "album": first_director.replace_strict(director_ids, default=None, return_dtype=pl.UInt32),
        "content_type": raw_items["content_type"], "release_year": raw_items["release_year"],
        "age_rating": raw_items["age_rating"], "genres": raw_items["genres"], "countries": raw_items["countries"]})
    day0 = pl.lit(D0).str.to_date("%Y-%m-%d")
    events = (interactions.with_columns((pl.col("last_watch_dt") - day0).dt.total_days().cast(pl.UInt32).alias("day"))
              .select(pl.col("user_id").cast(pl.UInt32).alias("uid"), pl.col("item_id").cast(pl.UInt32),
                      (pl.col("day") * 86400).cast(pl.UInt32).alias("ts"),
                      pl.lit(0, dtype=pl.UInt8).alias("et"), pl.lit(1, dtype=pl.UInt8).alias("org"),
                      pl.col("watched_pct").fill_null(0.0).clip(0, 100).cast(pl.Float32).alias("ratio"),
                      pl.col("total_dur").clip(0, 4_000_000).cast(pl.UInt32).alias("tlen"))
              .join(items.select("item_id", "artist", "album"), on="item_id", how="left")
              .sort("ts", "uid", maintain_order=True))
    events.write_parquet(EVENTS, compression="zstd")
    items.write_parquet(ITEMS)


def load_events() -> pl.DataFrame:
    return pl.read_parquet(EVENTS)


def load_items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS)
