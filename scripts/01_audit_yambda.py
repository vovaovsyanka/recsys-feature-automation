"""E01. Аудит данных Yambda-50M перед постановкой задачи: схема, шкала времени, смысл is_organic, повторы, покрытие артистов.
Запуск: python scripts/01_audit_yambda.py
"""
import polars as pl

from recsys.config import DATA
from recsys.tracking import Experiment

RAW = DATA / "raw" / "yambda_50m"
files = sorted(RAW.glob("*.parquet"))
cfg = {"dataset": "yambda-50m flat", "files": [f.name for f in files]}

with Experiment("E01", "audit_yambda", "Аудит данных Yambda-50M",
                goal="Проверить схему, диапазон времени, семантику is_organic, долю повторных прослушиваний "
                     "и покрытие маппингов артист/альбом - до выбора постановки задачи.",
                hypothesis="(1) is_organic=0 соответствует рекомендациям (Table 2 статьи: 48,7% прослушиваний рекомендательные); "
                           "(2) доля повторных прослушиваний высока (музыка), что делает прямые user-item признаки полезными.",
                config=cfg, data_files=files) as exp:
    with exp.stage("load"):
        L = pl.read_parquet(RAW / "listens.parquet")
        LK = pl.read_parquet(RAW / "likes.parquet")
        DL = pl.read_parquet(RAW / "dislikes.parquet")
        UL = pl.read_parquet(RAW / "unlikes.parquet")
        UD = pl.read_parquet(RAW / "undislikes.parquet")
        AR = pl.read_parquet(RAW / "artist_item_mapping.parquet")
        AL = pl.read_parquet(RAW / "album_item_mapping.parquet")
    print(L.schema)
    exp.record("schemas", {n: {k: str(v) for k, v in d.schema.items()} for n, d in
                           dict(listens=L, likes=LK, dislikes=DL, unlikes=UL, undislikes=UD, artist=AR, album=AL).items()})

    with exp.stage("basic_stats"):
        ev = {"listens": L, "likes": LK, "dislikes": DL, "unlikes": UL, "undislikes": UD}
        basic = {}
        for n, d in ev.items():
            basic[n] = dict(rows=d.height, users=d["uid"].n_unique(), items=d["item_id"].n_unique(),
                            ts_min=int(d["timestamp"].min()), ts_max=int(d["timestamp"].max()),
                            days=round((d["timestamp"].max() - d["timestamp"].min()) * 5 / 86400, 2),
                            organic_share=round(float(d["is_organic"].mean()), 4))
        exp.record("basic", basic)
        print(basic)
        # Table 2 статьи (5B): доля рекомендательных: listen .4874, like .4230, dislike .4847, unlike .0501, undislike .0982
        exp.record("paper_table2_recommendation_ratio_5B",
                   dict(listens=0.4874, likes=0.4230, dislikes=0.4847, unlikes=0.0501, undislikes=0.0982))

    with exp.stage("timeline"):
        # timestamp: uint32, шаг 5 с. День = 17280 тиков
        D = 86400 // 5
        ts_max = int(L["timestamp"].max())
        exp.record("n_days_total", ts_max / D)
        per_day = (L.with_columns((pl.col("timestamp") // D).alias("day"))
                   .group_by("day").agg(pl.len().alias("n"), pl.col("uid").n_unique().alias("users"))
                   .sort("day"))
        exp.record("listens_per_day_describe", per_day.select("n", "users").describe().to_dicts())
        exp.record("first_days", per_day.head(5).to_dicts())
        exp.record("last_days", per_day.tail(10).to_dicts())
        # проверка монотонности внутри пользователя
        srt = L.select("uid", "timestamp").with_columns(pl.col("timestamp").diff().over("uid").alias("d"))
        exp.record("listens_nonmonotonic_within_user_in_file_order", int((srt["d"] < 0).sum()))

    with exp.stage("timestamp_unit_check"):
        # Если timestamp в секундах, то при полном прослушивании подряд (ratio 95-100) разрыв до следующего
        # события ~ длине трека. Если бы единица была 5 с, отношение было бы ~ 1/5.
        T = (L.sort("uid", "timestamp")
             .with_columns((pl.col("timestamp").shift(-1).over("uid") - pl.col("timestamp")).alias("gap"))
             .filter((pl.col("played_ratio_pct").is_between(95, 100)) & pl.col("gap").is_not_null()
                     & (pl.col("gap") > 0) & (pl.col("gap") < 3 * pl.col("track_length_seconds"))))
        ratio = (T["gap"].cast(pl.Float64) / T["track_length_seconds"].cast(pl.Float64))
        exp.record("gap_over_track_length_median_if_ts_in_seconds", float(ratio.median()))
        exp.record("gap_over_track_length_n", T.height)
        exp.record("share_ts_multiple_of_5", float((L["timestamp"] % 5 == 0).mean()))
        exp.record("share_track_length_multiple_of_5", float((L["track_length_seconds"] % 5 == 0).mean()))
        exp.record("max_timestamp_in_days_if_seconds", float(L["timestamp"].max() / 86400))
        exp.record("max_timestamp_in_days_if_5s_ticks", float(L["timestamp"].max() * 5 / 86400))

    with exp.stage("played_ratio"):
        pr = L["played_ratio_pct"]
        exp.record("played_ratio_quantiles", {q: float(pr.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]})
        exp.record("share_listen_plus_ge50", float((pr >= 50).mean()))
        exp.record("share_ratio_gt100", float((pr > 100).mean()))
        exp.record("share_listen_plus_by_organic", L.group_by("is_organic").agg(
            (pl.col("played_ratio_pct") >= 50).mean().alias("ge50"), pl.len().alias("n")).sort("is_organic").to_dicts())

    with exp.stage("repeat_consumption"):
        # доля Listen+ событий, где пара (uid,item) уже встречалась ранее (любым листеном)
        Ls = L.sort("uid", "timestamp")
        Ls = Ls.with_columns((pl.int_range(pl.len()).over(["uid", "item_id"])).alias("k"))  # номер повторения пары
        rep = Ls.group_by(pl.col("played_ratio_pct") >= 50).agg((pl.col("k") > 0).mean().alias("repeat_share"),
                                                                pl.len().alias("n"))
        exp.record("repeat_share_by_listen_plus", rep.to_dicts())
        exp.record("repeat_share_by_organic", Ls.group_by("is_organic").agg((pl.col("k") > 0).mean().alias("repeat_share"),
                                                                           pl.len().alias("n")).to_dicts())
        hist = L.group_by("uid").agg(pl.len().alias("n"), pl.col("item_id").n_unique().alias("uniq"))
        exp.record("user_history_listens", hist.select("n").describe().to_dicts())
        exp.record("user_unique_items_ratio_median", float((hist["uniq"] / hist["n"]).median()))

    with exp.stage("mappings"):
        exp.record("artist_map", dict(rows=AR.height, items=AR["item_id"].n_unique(), artists=AR["artist_id"].n_unique(),
                                      dup_items=int(AR.height - AR["item_id"].n_unique())))
        exp.record("album_map", dict(rows=AL.height, items=AL["item_id"].n_unique(), albums=AL["album_id"].n_unique(),
                                     dup_items=int(AL.height - AL["item_id"].n_unique())))
        items_l = L.select("item_id").unique()
        exp.record("coverage_listen_items_with_artist", float(items_l.join(AR.select("item_id").unique(), on="item_id",
                                                                           how="semi").height / items_l.height))
        exp.record("coverage_listen_items_with_album", float(items_l.join(AL.select("item_id").unique(), on="item_id",
                                                                          how="semi").height / items_l.height))
        # покрытие по событиям
        j = L.join(AR.unique("item_id"), on="item_id", how="left")
        exp.record("coverage_listen_events_with_artist", float(j["artist_id"].is_not_null().mean()))

    with exp.stage("popularity_concentration"):
        pop = L.group_by("item_id").len().sort("len", descending=True)
        tot = pop["len"].sum()
        exp.record("top_share", {f"top{k}": float(pop["len"].head(k).sum() / tot) for k in [10, 100, 1000, 10000]})
        exp.record("items_with_le2_listens_share", float((pop["len"] <= 2).mean()))
