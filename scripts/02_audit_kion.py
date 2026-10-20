"""E02. Аудит данных KION: схема, гранулярность времени, уникальность пар, покрытие метаданных, целостность файлов.
Запуск: python scripts/02_audit_kion.py
"""
import polars as pl

from recsys.config import DATA
from recsys.tracking import Experiment

RAW = DATA / "raw" / "kion"
files = [RAW / "data_en" / "interactions.csv", RAW / "data_en" / "items_en.csv", RAW / "data_en" / "users_en.csv",
         RAW / "interactions.csv"]

with Experiment("E02", "audit_kion", "Аудит данных KION",
                goal="Проверить схему, временную гранулярность, уникальность пар (user,item), покрытие метаданных "
                     "и целостность файлов (отдельный interactions.csv выглядит обрезанным).",
                hypothesis="(1) Корневой interactions.csv обрезан по 50 МиБ; (2) время только на уровне даты; "
                           "(3) короткие истории (~5.7 события на пользователя) - ручные агрегаты слабее, чем в Yambda.",
                config={"dataset": "KION data_en"}, data_files=files) as exp:
    with exp.stage("load"):
        I = pl.read_csv(RAW / "data_en" / "interactions.csv").with_columns(
            pl.col("last_watch_dt").str.to_date("%Y-%m-%d"))
        Iroot = pl.read_csv(RAW / "interactions.csv", ignore_errors=True)
        It = pl.read_csv(RAW / "data_en" / "items_en.csv", infer_schema_length=10000).drop("")
        U = pl.read_csv(RAW / "data_en" / "users_en.csv").drop("")
    exp.record("integrity", dict(
        data_en_rows=I.height, repo_claim_rows=5476251, root_file_rows=Iroot.height,
        root_file_bytes=(RAW / "interactions.csv").stat().st_size,
        root_is_exactly_50MiB=(RAW / "interactions.csv").stat().st_size == 50 * 2**20))
    with exp.stage("stats"):
        exp.record("basic", dict(rows=I.height, users=I["user_id"].n_unique(), items=I["item_id"].n_unique(),
                                 repo_claim=dict(users=962179, items=15706),
                                 date_min=str(I["last_watch_dt"].min()), date_max=str(I["last_watch_dt"].max()),
                                 days=(I["last_watch_dt"].max() - I["last_watch_dt"].min()).days))
        pairs = I.select("user_id", "item_id").unique().height
        exp.record("pair_uniqueness", dict(unique_pairs=pairs, rows=I.height, duplicates=I.height - pairs))
        exp.record("nulls", {c: int(I[c].null_count()) for c in I.columns})
        exp.record("watched_pct_quantiles", {q: float(I["watched_pct"].quantile(q)) for q in [.1, .25, .5, .75, .9]})
        exp.record("total_dur_quantiles", {q: float(I["total_dur"].quantile(q)) for q in [.1, .25, .5, .75, .9]})
        exp.record("share_watched_pct_ge_50", float((I["watched_pct"] >= 50).mean()))
        exp.record("share_watched_pct_null_or_neg", float(((I["watched_pct"] < 0) | I["watched_pct"].is_null()).mean()))
        ul = I.group_by("user_id").len()
        exp.record("events_per_user", ul.select("len").describe().to_dicts())
        exp.record("users_with_ge_5_events", int((ul["len"] >= 5).sum()))
        exp.record("users_with_ge_20_events", int((ul["len"] >= 20).sum()))
        perday = I.group_by("last_watch_dt").len().sort("last_watch_dt")
        exp.record("events_per_day", perday.select("len").describe().to_dicts())
        exp.record("last_14_days", perday.tail(14).with_columns(pl.col("last_watch_dt").cast(str)).to_dicts())
    with exp.stage("metadata"):
        exp.record("items_file", dict(rows=It.height, unique=It["item_id"].n_unique(),
                                      interaction_items_covered=float(
                                          I.select("item_id").unique().join(It.select("item_id"), on="item_id", how="semi").height
                                          / I["item_id"].n_unique())))
        exp.record("users_file", dict(rows=U.height, unique=U["user_id"].n_unique(),
                                      interaction_users_covered=float(
                                          I.select("user_id").unique().join(U.select("user_id"), on="user_id", how="semi").height
                                          / I["user_id"].n_unique())))
        exp.record("item_null_share", {c: round(float(It[c].null_count() / It.height), 3) for c in It.columns})
        exp.record("user_null_share", {c: round(float(U[c].null_count() / U.height), 3) for c in U.columns})
        exp.record("content_type", It["content_type"].value_counts().to_dicts())
        exp.record("item_popularity_top_share", (lambda p: {f"top{k}": float(p.head(k).sum() / p.sum())
                                                            for k in (10, 100, 1000)})(
            I.group_by("item_id").len().sort("len", descending=True)["len"]))
