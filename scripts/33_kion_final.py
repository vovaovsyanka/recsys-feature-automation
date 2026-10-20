"""E33. KION: финальные модели на полных данных и оценка на test.

Для ручного набора и найденных в E32 автоматических наборов обучается LightGBM (lambdarank) на 28 обучающих якорях, качество
сравнивается на val; затем каждая модель один раз оценивается на test (журнал experiments/test_access_log.jsonl) с бутстреп-интервалами
по пользователям относительно ручного набора.
Запуск: python scripts/33_kion_final.py
"""
import time

import numpy as np

from recsys import features as F
from recsys.config import KION_TAG
from recsys.data import kion as K
from recsys.evaluation import metrics as M
from recsys.features import sets
from recsys.models import gbdt as T
from recsys.search import setup_kion as C
from recsys.tracking import Experiment

N_BOOT = 1000
FINALISTS = sets.KION_FINALISTS

with Experiment("E33", "kion_final", "KION: финальные модели и оценка на test",
                goal="Подтвердить на втором домене выводы Yambda на полных данных: ручной набор против автоматических наборов.",
                hypothesis="Автоматический набор не хуже ручного; разница мала из-за коротких историй пользователей.",
                config={"finalists": FINALISTS, "train": C.TRAIN_DAYS, "val": C.VAL_DAY, "test": C.TEST_DAY, "neg_keep": C.NEG_KEEP,
                        "params": T.DEFAULT_PARAMS}, data_files=[K.EVENTS]) as exp:
    store = F.FeatureStore(KION_TAG, F.Engine(K.load_events()))
    val_metrics, boosters, feature_names = {}, {}, {}
    for name in FINALISTS:
        names = sets.get_set(name)
        feature_names[name] = names
        with exp.stage(f"{name}: признаки"):
            t = time.time()
            for day in [C.VAL_DAY] + C.TRAIN_DAYS + [C.TEST_DAY]:
                store.ensure(day, names)
                store.drop_rows_cache()
            feature_seconds = time.time() - t
        with exp.stage(f"{name}: обучение"):
            train = T.build_mat(store, C.TRAIN_DAYS, names, neg_keep=C.NEG_KEEP, seed=0, verbose=False)
            val = T.build_mat(store, [C.VAL_DAY], names, verbose=False)
            booster, seconds = T.train_lgb(train, val, "lambdarank", rounds=1500)
            res, *_ = T.eval_mat(val, T.predict(booster, val))
            res.update(n_features=len(names), train_seconds=seconds, feature_seconds=feature_seconds,
                       best_iteration=int(booster.best_iteration))
            val_metrics[name], boosters[name] = res, booster
            exp.record("val_metrics", val_metrics)
            print(f"{name:12s} val ndcg@10={res['ndcg@10']:.4f}  ndcg@100={res['ndcg@100']:.4f}  auc={res['auc']:.4f}  ({seconds:.0f} с)")
            del train, val
    with exp.stage("test"):
        test_metrics, packs = {}, {}
        for name in FINALISTS:
            T.log_test_access(f"kion:{name}", "финальная оценка")
            test = T.build_mat(store, [C.TEST_DAY], feature_names[name], verbose=False)
            res, _, _, pack = T.eval_mat(test, T.predict(boosters[name], test), boot=1)
            test_metrics[name], packs[name] = res, pack
            print(f"{name:12s} test ndcg@10={res['ndcg@10']:.4f}  ndcg@100={res['ndcg@100']:.4f}  recall@100={res['recall@100']:.4f}")
        exp.record("test_metrics", test_metrics)
        reference, intervals = packs["kion:manual"], {}
        for name, (users, count, sums) in packs.items():
            intervals[name] = {"abs": M.bootstrap_ci(count, sums, None, N_BOOT)}
            if name != "kion:manual":
                assert np.array_equal(users, reference[0])
                intervals[name]["vs_manual"] = M.bootstrap_ci(count, sums, reference[2], N_BOOT)
                d = intervals[name]["vs_manual"]["ndcg@10"]
                print(f"{name:12s} ΔNDCG@10 к ручному: {d['mean']:+.4f} [{d['lo']:+.4f}; {d['hi']:+.4f}]")
        exp.record("bootstrap_ci", intervals)
