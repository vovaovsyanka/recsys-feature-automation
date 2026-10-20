"""E04. Строки (якорь, пользователь, кандидат) и метки для всех якорей.

Состав кандидатов выбран в E03. Строки фиксируются один раз, и все модели и наборы признаков используют ровно их:
сравнение идёт при одинаковых кандидатах.
Запуск: python scripts/04_build_rows.py
"""
import polars as pl

from recsys.candidates import yambda as C
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.protocol import DAY, TEST, TRAIN_FIRST, TRAIN_LAST, VAL, check_protocol, train_anchors
from recsys.tracking import Experiment

CFG = C.CandidateConfig(s1_n=200, s1_halflife_days=14.0, s2_n=100, s3_artists=5, s3_per_artist=8, s4_n=30)
DAYS = list(range(TRAIN_FIRST, TRAIN_LAST + 1)) + VAL + TEST
check_protocol(train_anchors())

with Experiment("E04", "build_rows", "Построение кандидатных строк и меток",
                goal="Зафиксировать набор строк и меток для всех якорей: общие кандидаты - условие честного сравнения моделей.",
                hypothesis="Около 4 тысяч групп и 1 млн строк на якорь; recall кандидатов около 0,40 (по E03).",
                config={"candidates": vars(CFG), "tag": YAMBDA_TAG, "days": [DAYS[0], DAYS[-1], len(DAYS)]},
                data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        events, items = D.load_events(), D.load_items()
        ts = events["ts"].to_numpy()
    out = C.rows_dir(YAMBDA_TAG)
    C.save_config(YAMBDA_TAG, CFG)
    (out / "groups").mkdir(exist_ok=True)
    stats = []
    with exp.stage("build"):
        for day in DAYS:
            rows, groups = C.build_anchor(events, ts, items, day * DAY, CFG)
            rows.write_parquet(out / f"rows_{day:03d}.parquet", compression="zstd")
            groups.write_parquet(out / "groups" / f"groups_{day:03d}.parquet")
            stats.append(dict(day=day, users=groups.height, rows=rows.height, pos=int(groups["n_pos"].sum()),
                              pos_in=int(groups["n_pos_in_cand"].sum())))
            if len(stats) % 10 == 0:
                print(f"  {len(stats)}/{len(DAYS)} якорей")
    table = pl.DataFrame(stats)
    summary = {}
    for name, (lo, hi) in {"train": (TRAIN_FIRST, TRAIN_LAST), "val": (VAL[0], VAL[-1]), "test": (TEST[0], TEST[-1])}.items():
        part = table.filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
        summary[name] = dict(anchors=part.height, groups=int(part["users"].sum()), rows=int(part["rows"].sum()),
                             recall=float(part["pos_in"].sum() / part["pos"].sum()),
                             positive_rate=float(part["pos_in"].sum() / part["rows"].sum()))
    exp.record("summary", summary)
    exp.record("per_anchor", stats)
    print(summary)
