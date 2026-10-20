"""Песочница для сравнения методов поиска признаков.

Все методы оценивают наборы признаков только через `Proxy.evaluate`: те же якоря, та же подвыборка
пользователей, те же гиперпараметры LightGBM и одна метрика (NDCG@10 на val). Так сравнение
методов не зависит от того, чем они отличаются в остальном. Test здесь не используется.
"""
import time
from dataclasses import dataclass, field

import lightgbm as lgb

from recsys.config import NUM_THREADS
from recsys import features as F
from recsys.models import gbdt as T
from recsys.protocol import VAL


def _default_params() -> dict:
    return dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63,
                num_threads=NUM_THREADS, verbose=-1, seed=0)


@dataclass
class Proxy:
    store: F.FeatureStore                    # хранилище признаков на подвыборке пользователей
    train_days: list[int]
    val_days: list[int] = field(default_factory=lambda: list(VAL))
    neg_keep: float = 0.3                    # доля негативов, оставляемая в обучении
    params: dict = field(default_factory=_default_params)
    rounds: int = 150
    # счётчики затрат
    n_fits: int = 0
    fit_seconds: float = 0.0
    load_seconds: float = 0.0
    log: list = field(default_factory=list)

    def mats(self, names: list[str]):
        t = time.time()
        tr = T.build_mat(self.store, self.train_days, names, neg_keep=self.neg_keep, seed=0, verbose=False)
        va = T.build_mat(self.store, self.val_days, names, verbose=False)
        self.load_seconds += time.time() - t
        return tr, va

    def evaluate(self, names: list[str], objective: str = "lambdarank", tag: str = ""):
        """Обучает модель на `names`, возвращает (запись с метриками, бустер)."""
        tr, va = self.mats(names)
        p = dict(self.params)
        feature_names = [f"f{i}" for i in range(len(names))]    # LightGBM не принимает спецсимволы в именах
        if objective == "lambdarank":
            p.update(objective="lambdarank", lambdarank_truncation_level=100, eval_at=[10])
            dtr = lgb.Dataset(tr.X, tr.y, group=tr.group_sizes, feature_name=feature_names)
        else:
            p.update(objective="binary")
            dtr = lgb.Dataset(tr.X, tr.y, feature_name=feature_names)
        t = time.time()
        booster = lgb.train(p, dtr, self.rounds)
        raw = booster.predict(va.X, num_threads=NUM_THREADS)
        seconds = time.time() - t
        res, *_ = T.eval_mat(va, raw)
        self.n_fits += 1
        self.fit_seconds += seconds
        rec = dict(tag=tag, n_features=len(names), ndcg10=res["ndcg@10"], ndcg100=res["ndcg@100"],
                   auc=res["auc"], seconds=seconds)
        self.log.append(rec)
        return rec, booster
