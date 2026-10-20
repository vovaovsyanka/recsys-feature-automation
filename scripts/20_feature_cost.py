"""E20. Стоимость расчёта признаков по ключам и агрегатам: микробенчмарк на одном якоре с полным набором строк.

Нужен для анализа затрат (какие признаки дорогие, как стоимость зависит от пакета и окна). Запускать при свободном процессоре.
Запуск: python scripts/20_feature_cost.py
"""
import time

import numpy as np

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import grammar as G
from recsys.tracking import Experiment

DAY = 288
N_PER = 12
rng = np.random.default_rng(0)

with Experiment("E20", "feature_cost", "Стоимость расчёта признаков по ключу и агрегату",
                goal="Измерить секунды на признак (изолированно, на полном якоре ~1.1 млн строк) по ключам U/I/UI/UA/UB/A/B и "
                     "агрегатам; оценить, какие признаки дороги и насколько растёт стоимость с числом признаков в пакете.",
                hypothesis="UI/UA/UB (ключи по парам) дороже U/I/A/B; окно 'all' и nuniq дороже коротких окон; пакетирование "
                           "амортизирует чтение/semi-join.",
                config={"anchor_day": DAY, "n_per_cell": N_PER}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        eng = F.Engine(D.load_events())
        rows = F.FeatureStore(YAMBDA_TAG, eng).rows(DAY)
        for key in F.KEYS:
            eng.daily(key)       # прогрев: посуточные таблицы строятся один раз, их стоимость в замер не входит
    pool = [n for n in G.grammar_pool() if not F.is_derived(n)]
    table = {}
    with exp.stage("by_key_agg"):
        for key in F.KEYS:
            for agg in F.AGGS:
                cell = [n for n in pool if F.Spec.parse(n).key == key and F.Spec.parse(n).agg == agg]
                if not cell: continue
                pick = list(rng.choice(cell, min(N_PER, len(cell)), replace=False))
                t = time.time()
                eng.compute_base([F.Spec.parse(n) for n in pick], rows, DAY)
                dt = time.time() - t
                # одиночные расчёты (без амортизации)
                t1 = time.time()
                for n in pick[:3]:
                    eng.compute_base([F.Spec.parse(n)], rows, DAY)
                single = (time.time() - t1) / 3
                table[f"{key}.{agg}"] = dict(batch_per_feature=dt / len(pick), single_feature=single, n=len(pick))
                print(f"{key:3s} {agg:6s} batch {dt/len(pick):.2f}s/feature  single {single:.2f}s/feature")
    exp.record("seconds_per_feature_by_key_agg", table)
    with exp.stage("by_window"):
        win = {}
        for w in ["1", "7", "30", "all"]:
            names = [f"UI.{c}.cnt.{w}" for c in ["listen", "plus", "skip", "full", "org", "rec"]]
            t = time.time(); eng.compute_base([F.Spec.parse(n) for n in names], rows, DAY); win[w] = (time.time() - t) / len(names)
        exp.record("UI_cnt_seconds_per_feature_by_window", win)
        print(win)
    with exp.stage("batch_scaling"):
        sc = {}
        base = [n for n in pool if F.Spec.parse(n).key == "UA" and F.Spec.parse(n).agg == "cnt"]
        for b in [1, 5, 20, 80]:
            pick = list(rng.choice(base, min(b, len(base)), replace=False))
            t = time.time(); eng.compute_base([F.Spec.parse(n) for n in pick], rows, DAY); sc[b] = (time.time() - t)
            print("batch", b, f"{sc[b]:.1f}s total, {sc[b]/b:.2f}s/feature")
        exp.record("UA_cnt_batch_total_seconds", sc)
