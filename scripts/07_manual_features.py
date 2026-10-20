"""E07. Расчёт ручного набора признаков (123) для обучающих, val и test якорей с замером времени.

Запуск: python scripts/07_manual_features.py
"""
import time

import numpy as np

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import manual_yambda as MF
from recsys.protocol import FULL_TRAIN_DAYS, TEST, VAL, check_protocol, train_anchors
from recsys.tracking import Experiment

check_protocol(train_anchors())
names = MF.manual_all()

with Experiment("E07", "manual_features", "Расчёт ручных признаков (123) и их стоимость",
                goal="Посчитать и сохранить ручной набор для всех якорей и измерить, сколько стоит расчёт на якорь и на признак.",
                hypothesis="Порядка 15-45 секунд на якорь; суммарно меньше часа на 48 якорей.",
                config={"tag": YAMBDA_TAG, "train_days": FULL_TRAIN_DAYS, "val": VAL, "test": TEST, "n_features": len(names),
                        "groups": {k: len(v) for k, v in MF.manual_groups().items()}}, data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
    seconds = {}
    with exp.stage("compute"):
        for part, days in [("val", VAL), ("train", FULL_TRAIN_DAYS), ("test", TEST)]:
            for day in days:
                t = time.time()
                store.ensure(day, names)
                seconds[day] = time.time() - t
                store.drop_rows_cache()
                print(f"  {part}, сутки {day}: {seconds[day]:.1f} с")
    exp.record("seconds_per_anchor", seconds)
    exp.record("mean_seconds_per_anchor_per_feature", float(np.mean([v for v in seconds.values() if v > 1]) / len(names)))
