"""E08. Сильный базлайн: LightGBM на ручных признаках (123). Оценка на val, test не используется.

Сравниваются две цели обучения (lambdarank и binary) при одинаковых гиперпараметрах. Время обучения здесь - часть результата,
поэтому запускать на свободной машине.
Запуск: python scripts/08_baseline_gbdt.py
"""
import numpy as np

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import manual_yambda as MF
from recsys.models import gbdt as T
from recsys.protocol import FULL_TRAIN_DAYS, VAL
from recsys.tracking import Experiment

NEG_KEEP = 0.3        # в обучении оставляем 30% негативов
names = MF.manual_all()

with Experiment("E08", "baseline_gbdt", "LightGBM на ручных признаках: базлайн, val",
                goal="Получить сильный базлайн: бустинг на содержательном ручном наборе (123 признака); сравнить lambdarank и binary.",
                hypothesis="Бустинг заметно превзойдёт лучшую эвристику; lambdarank чуть лучше по NDCG, binary даёт вероятности (LogLoss).",
                config={"tag": YAMBDA_TAG, "train_days": FULL_TRAIN_DAYS, "neg_keep": NEG_KEEP, "n_features": len(names),
                        "params": T.DEFAULT_PARAMS}, data_files=[D.EVENTS]) as exp:
    with exp.stage("load_matrices"):
        store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
        train = T.build_mat(store, FULL_TRAIN_DAYS, names, neg_keep=NEG_KEEP, seed=0)
        val = T.build_mat(store, list(VAL), names)
        print("train", train.X.shape, "доля позитивов %.3f" % train.y.mean(), "| val", val.X.shape)
        exp.record("shapes", dict(train=list(train.X.shape), val=list(val.X.shape), train_positive_rate=float(train.y.mean())))
    results = {}
    for objective in ["lambdarank", "binary"]:
        with exp.stage(f"train_{objective}"):
            booster, seconds = T.train_lgb(train, val, objective)
        with exp.stage(f"evaluate_{objective}"):
            score = T.predict(booster, val)
            proba = None
            if objective == "binary":    # поправка вероятностей на отбрасывание негативов при обучении
                odds = score / (1 - score) * NEG_KEEP
                proba = odds / (1 + odds)
            res, *_ = T.eval_mat(val, score, proba)
            res.update(best_iteration=int(booster.best_iteration), train_seconds=seconds)
            gain = booster.feature_importance("gain")
            res["top_features_gain"] = [(names[i], float(gain[i] / gain.sum())) for i in np.argsort(-gain)[:25]]
            results[objective] = res
            exp.record("val_metrics", results)              # сохраняем сразу: сбой дальше не должен терять метрики
            # LightGBM не умеет писать в путь с кириллицей, поэтому сохраняем модель строкой через Python
            (exp.dir / f"model_{objective}.txt").write_text(booster.model_to_string(), encoding="utf-8")
            print(objective, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "top_features_gain"})
