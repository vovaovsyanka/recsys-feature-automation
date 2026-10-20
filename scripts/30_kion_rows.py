"""E30. KION: кандидаты и метки для всех якорей (10% пользователей со всей историей).

Запуск: python scripts/30_kion_rows.py
"""
import time

import polars as pl

from recsys.candidates import kion as P
from recsys.config import DATA, KION_TAG
from recsys.data import kion as K
from recsys.search import setup_kion as C
from recsys.tracking import Experiment

DAYS = C.TRAIN_DAYS + [C.VAL_DAY, C.TEST_DAY]

with Experiment("E30", "kion_rows", "KION: кандидаты и метки",
                goal="Построить для второго домена те же сущности, что в E04: общие кандидаты и метки для всех якорей, измерить recall.",
                hypothesis="Recall кандидатов 0,3-0,5 (короткие истории, много новых фильмов); основной вклад дают популярность и kNN.",
                config={"user_frac": C.ROWS_USER_FRAC, "train": C.TRAIN_DAYS, "val": C.VAL_DAY, "test": C.TEST_DAY,
                        "horizon_days": P.HORIZON_DAYS}, data_files=[K.EVENTS]) as exp:
    with exp.stage("load"):
        events, items = K.load_events(), K.load_items()
        ts = events["ts"].to_numpy()
        n_items = int(events["item_id"].max()) + 1
    out = DATA / "rows" / KION_TAG
    (out / "groups").mkdir(parents=True, exist_ok=True)
    stats = []
    with exp.stage("build"):
        for day in DAYS:
            t = time.time()
            rows, groups = P.build_anchor(events, ts, items, day, C.ROWS_USER_FRAC, n_items)
            rows.write_parquet(out / f"rows_{day:03d}.parquet", compression="zstd")
            groups.write_parquet(out / "groups" / f"groups_{day:03d}.parquet")
            stats.append(dict(day=day, users=groups.height, rows=rows.height, pos=int(groups["n_pos"].sum()),
                              pos_in=int(groups["n_pos_in_cand"].sum()), seconds=time.time() - t))
            print(stats[-1])
    table = pl.DataFrame(stats)
    summary = {}
    for name, part in {"train": table.filter(pl.col("day") <= P.A_TRAIN_LAST), "val": table.filter(pl.col("day") == C.VAL_DAY),
                       "test": table.filter(pl.col("day") == C.TEST_DAY)}.items():
        summary[name] = dict(anchors=part.height, groups=int(part["users"].sum()), rows=int(part["rows"].sum()),
                             recall=float(part["pos_in"].sum() / part["pos"].sum()),
                             positive_rate=float(part["pos_in"].sum() / part["rows"].sum()),
                             candidates_per_group=float(part["rows"].sum() / part["users"].sum()))
    exp.record("summary", summary)
    exp.record("per_anchor", stats)
    print(summary)
