"""E31. KION: эвристики и LightGBM на ручных признаках (val). Test не используется.

Двухстадийная схема «кандидаты + бустинг-реранкер» - та же, что у победителей KION Challenge (Petrov et al., 2022).
Запуск: python scripts/31_kion_baseline.py
"""
import time

import numpy as np

from recsys import features as F
from recsys.config import KION_TAG
from recsys.data import kion as K
from recsys.models import gbdt as T
from recsys.search import setup_kion as C
from recsys.tracking import Experiment

DAYS = [C.VAL_DAY] + C.TRAIN_DAYS + [C.TEST_DAY]
HEURISTICS = {
    "случайный порядок": None,
    "популярность за 14 суток (s1_score)": ("s1_score", +1),
    "жанровое сродство (s2_score)": ("s2_score", +1),
    "item-kNN (s3_score)": ("s3_score", +1),
    "жанр пользователя (UA.listen.cnt.all)": ("UA.listen.cnt.all", +1),
    "популярность за 7 суток (I.listen.cnt.7)": ("I.listen.cnt.7", +1),
    "популярность за 30 суток (I.listen.cnt.30)": ("I.listen.cnt.30", +1),
    "тренд популярности (I.listen.decay.h3)": ("I.listen.decay.h3", +1),
}
names = C.MAN

with Experiment("E31", "kion_baseline", "KION: эвристики и LightGBM на ручных признаках (val)",
                goal="Сильный базлайн для второго домена и нижняя граница эвристик.",
                hypothesis="Бустинг превзойдёт лучшую эвристику, но выигрыш меньше, чем на Yambda: истории короткие.",
                config={"train": C.TRAIN_DAYS, "val": C.VAL_DAY, "test": C.TEST_DAY, "neg_keep": C.NEG_KEEP,
                        "n_features": len(names), "params": T.DEFAULT_PARAMS}, data_files=[K.EVENTS]) as exp:
    store = F.FeatureStore(KION_TAG, F.Engine(K.load_events()))
    heuristic_names = sorted({h[0] for h in HEURISTICS.values() if h})
    seconds = {}
    with exp.stage("features"):
        for day in DAYS:
            t = time.time()
            store.ensure(day, names + heuristic_names)
            seconds[day] = time.time() - t
            store.drop_rows_cache()
        exp.record("feature_seconds_per_anchor", seconds)
    with exp.stage("heuristics"):
        mat = T.build_mat(store, [C.VAL_DAY], heuristic_names, verbose=False)
        rng = np.random.default_rng(0)
        results = {}
        for title, spec in HEURISTICS.items():
            score = rng.random(len(mat.y)) if spec is None else np.nan_to_num(mat.X[:, mat.names.index(spec[0])] * spec[1], nan=-1e9)
            results[title], *_ = T.eval_mat(mat, score)
            r = results[title]
            print(f"{title:46s} ndcg@10={r['ndcg@10']:.4f}  ndcg@100={r['ndcg@100']:.4f}  recall@100={r['recall@100']:.4f}  auc={r['auc']:.4f}")
        results["оракул внутри кандидатов"], *_ = T.eval_mat(mat, mat.y + rng.random(len(mat.y)) * 1e-3)
        exp.record("heuristics_val", results)
    with exp.stage("load_matrices"):
        train = T.build_mat(store, C.TRAIN_DAYS, names, neg_keep=C.NEG_KEEP, seed=0, verbose=False)
        val = T.build_mat(store, [C.VAL_DAY], names, verbose=False)
        print("train", train.X.shape, "доля позитивов %.4f" % train.y.mean(), "val", val.X.shape)
    with exp.stage("train"):
        booster, seconds_train = T.train_lgb(train, val, "lambdarank", rounds=1500)
    with exp.stage("evaluate_val"):
        res, *_ = T.eval_mat(val, T.predict(booster, val))
        res.update(best_iteration=int(booster.best_iteration), train_seconds=seconds_train)
        gain = booster.feature_importance("gain")
        res["top_features_gain"] = [(names[i], float(gain[i] / gain.sum())) for i in np.argsort(-gain)[:25]]
        exp.record("gbdt_val", res)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "top_features_gain"})
        (exp.dir / "model_lambdarank.txt").write_text(booster.model_to_string(), encoding="utf-8")
