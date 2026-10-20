"""E17. Финальная модель LightGBM (lambdarank) на полных данных для заданного набора признаков, оценка на val.

Гиперпараметры, обучающие якоря и отбор негативов те же, что у базлайна E08, поэтому наборы признаков сравниваются честно.
Признаки считаются и для test-якорей (метки при этом не используются), но качество на test здесь не оценивается:
это делает scripts/18_evaluate_models.py.
Запуск: python scripts/17_final_model.py --set=A2_B [--seed=1]
"""
import sys
import time

import numpy as np

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import sets
from recsys.models import gbdt as T
from recsys.protocol import FULL_TRAIN_DAYS, TEST, VAL, check_protocol, train_anchors
from recsys.tracking import Experiment


def arg(name: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


SET = arg("set")
SEED = int(arg("seed", "0"))
NEG_KEEP = 0.3
check_protocol(train_anchors())
names = sets.get_set(SET)
assert len(names) == len(set(names)), "в наборе есть дубликаты"
label = SET if SEED == 0 else f"{SET}_s{SEED}"

with Experiment("E17", f"final_{label}", f"Полная модель LightGBM: набор {SET} ({len(names)} признаков), val",
                goal=f"Обучить набор {SET} на полных данных теми же гиперпараметрами, что базлайн E08, и оценить на val.",
                hypothesis="Порядок наборов на полных данных повторяет порядок в песочнице (E11-E14).",
                config={"set": SET, "seed": SEED, "n_features": len(names), "train_days": FULL_TRAIN_DAYS, "neg_keep": NEG_KEEP,
                        "params": T.DEFAULT_PARAMS, "features": names}, data_files=[D.EVENTS]) as exp:
    with exp.stage("features"):
        store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
        t = time.time()
        for day in list(VAL) + FULL_TRAIN_DAYS + list(TEST):
            store.ensure(day, names)
            store.drop_rows_cache()
        exp.record("feature_compute_seconds_new_features", time.time() - t)
    with exp.stage("load_matrices"):
        train = T.build_mat(store, FULL_TRAIN_DAYS, names, neg_keep=NEG_KEEP, seed=SEED, verbose=False)
        val = T.build_mat(store, list(VAL), names, verbose=False)
    with exp.stage("train"):
        booster, seconds = T.train_lgb(train, val, "lambdarank", params={"seed": SEED})
    with exp.stage("evaluate_val"):
        res, per_group, groups, _ = T.eval_mat(val, T.predict(booster, val))
        res.update(best_iteration=int(booster.best_iteration), train_seconds=seconds, n_features=len(names))
        gain = booster.feature_importance("gain")
        res["top_features_gain"] = [(names[i], float(gain[i] / gain.sum())) for i in np.argsort(-gain)[:30]]
        exp.record("val_metrics", res)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "top_features_gain"})
        (exp.dir / "model_lambdarank.txt").write_text(booster.model_to_string(), encoding="utf-8")
