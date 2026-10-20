"""E05. Проверка отсутствия утечек: признаки якоря не должны зависеть от событий после якоря.

На подвыборке пользователей считаем признаки, затем заменяем все события с ts >= a случайным мусором (другие треки, пользователи,
доли и типы событий) и считаем снова. Любое расхождение - утечка. Чувствительность проверки подтверждает «канарейка»:
признаки, у которых окно захватывает сутки якоря, должны расходиться.
Запуск: python scripts/05_leakage_tests.py
"""
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from recsys import features as F
from recsys.candidates import yambda as C
from recsys.data import yambda as D
from recsys.protocol import DAY
from recsys.tracking import Experiment

A_DAYS = [120, 200, 288]
N_USERS = 400
SEED = 7


def corrupt_future(ev: pl.DataFrame, a_ts: int, rng: np.random.Generator) -> pl.DataFrame:
    """Все события с ts >= a_ts заменяются случайными (сохраняя ts >= a_ts)."""
    past = ev.filter(pl.col("ts") < a_ts)
    fut = ev.filter(pl.col("ts") >= a_ts)
    n = fut.height
    uids = past["uid"].unique().to_numpy()
    items = ev["item_id"].unique().to_numpy()
    items_tab = ev.select("item_id", "artist", "album").unique("item_id")
    new_items = rng.choice(items, n)
    bad = (pl.DataFrame({"item_id": new_items.astype(np.uint32)}).join(items_tab, on="item_id", how="left"))
    fut2 = fut.with_columns(
        pl.Series("uid", rng.choice(uids, n).astype(np.uint32)),
        bad["item_id"], bad["artist"], bad["album"],
        pl.Series("et", rng.integers(0, 5, n).astype(np.uint8)),
        pl.Series("org", rng.integers(0, 2, n).astype(np.uint8)),
        pl.Series("ratio", np.where(rng.random(n) < 0.5, rng.integers(1, 101, n), 100).astype(np.float32)),
        pl.Series("tlen", rng.integers(30, 400, n).astype(np.uint32)),
        pl.Series("ts", (a_ts + rng.integers(0, 2 * DAY, n)).astype(np.uint32)))
    fut2 = fut2.with_columns(pl.when(pl.col("et") == 0).then(pl.col("ratio")).otherwise(None).alias("ratio"),
                             pl.when(pl.col("et") == 0).then(pl.col("tlen")).otherwise(None).alias("tlen"))
    return pl.concat([past, fut2.select(past.columns)]).sort("ts", "uid", maintain_order=True)


def spec_battery() -> list[str]:
    names = []
    for key in F.KEYS:
        for chan in F.CHANS:
            for win in ["1", "7", "30", "all"]:
                names.append(f"{key}.{chan}.cnt.{win}")
        for chan in F.SR_CH:
            names += [f"{key}.{chan}.mean.{w}" for w in ["7", "all"]]
        names += [f"{key}.listen.sec.{w}" for w in ["7", "all"]]
        names += [f"{key}.{c}.last.{w}" for c in F.TMAX_CH for w in ["7", "all"]]
        names += [f"{key}.{c}.first.all" for c in F.TMIN_CH]
        names += [f"{key}.{c}.decay.h{h}" for c in ["listen", "plus", "like"] for h in [3, 14]]
        if key in F.PARTNER:
            names += [f"{key}.{c}.nuniq.{w}" for c in ["listen", "plus"] for w in ["7", "30", "all"]]
    names += ["d:ratio(UI.plus.cnt.7|U.plus.cnt.7)", "d:diff(UI.like.cnt.all|UI.unlike.cnt.all)",
              "d:urank(UI.plus.cnt.30)", "d:urank(d:ratio(UI.plus.cnt.30|U.plus.cnt.30))"]
    assert all(F.Spec.parse(n).valid() for n in names if not F.is_derived(n)), "в батарее есть невалидная спецификация"
    return names


def compute_all(eng: F.Engine, rows: pl.DataFrame, a_day: int, names: list[str]) -> dict[str, np.ndarray]:
    base = sorted({b for n in names for b in F.base_deps(n)})
    df = eng.compute_base([F.Spec.parse(b) for b in base], rows, a_day)
    cols = {c: df[c].to_numpy() for c in df.columns}
    uid = rows["uid"].to_numpy(); dc: dict = {}
    return {n: F.derived_value(n, cols, dc, uid) for n in names}


with Experiment("E05", "leakage_tests", "Проверка на утечки: замена будущего мусором и «канарейка»",
                goal="Автоматически показать, что ни один признак грамматики не использует события с ts >= якоря, "
                     "и убедиться, что проверка способна поймать утечку.",
                hypothesis="При замене будущего мусором все признаки совпадают побитово; канарейка расходится.",
                config={"anchors": A_DAYS, "n_users": N_USERS, "seed": SEED}, data_files=[D.EVENTS]) as exp:
    rng = np.random.default_rng(SEED)
    with exp.stage("prepare"):
        ev = D.load_events(); it = D.load_items()
        users = rng.choice(ev["uid"].unique().to_numpy(), N_USERS, replace=False)
        ev_s = ev.filter(pl.col("uid").is_in(users.tolist()))
        print("subsample events:", ev_s.height)
        names = spec_battery()
        print("features in battery:", len(names))
    results = []
    for a_day in A_DAYS:
        a_ts = a_day * DAY
        with exp.stage(f"anchor_{a_day}"):
            ts_s = ev_s["ts"].to_numpy()
            rows, groups = C.build_anchor(ev_s, ts_s, it, a_ts, C.CandidateConfig(s1_n=100, s2_n=50))
            ev_p = corrupt_future(ev_s, a_ts, rng)
            with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
                eA, eB = F.Engine(ev_s, Path(t1)), F.Engine(ev_p, Path(t2))
                XA = compute_all(eA, rows, a_day, names)
                XB = compute_all(eB, rows, a_day, names)
                bad = [n for n in names if not np.array_equal(XA[n], XB[n], equal_nan=True)]
                # канарейка: «окно» сдвинуто на сутки вперёд -> включает сутки якоря (т.е. будущее)
                canary = compute_all(eA, rows, a_day + 1, names)
                canary_p = compute_all(eB, rows, a_day + 1, names)
                caught = [n for n in names if not np.array_equal(canary[n], canary_p[n], equal_nan=True)]
            results.append(dict(anchor=a_day, rows=rows.height, features=len(names), leaking=bad,
                                canary_detected_features=len(caught), canary_total=len(names)))
            print(f"anchor {a_day}: rows={rows.height} leaking_features={len(bad)} canary_caught={len(caught)}/{len(names)}")
    exp.record("per_anchor", results)
    exp.record("PASSED", all(len(r["leaking"]) == 0 for r in results) and all(r["canary_detected_features"] > 0 for r in results))
    assert exp.results["PASSED"], "тест на утечки не пройден"
