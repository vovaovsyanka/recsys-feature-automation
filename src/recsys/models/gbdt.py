"""Сборка матриц признаков и обучение LightGBM. Общий код всех экспериментов с бустингом."""
import json
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from recsys.config import EXPERIMENTS, NUM_THREADS
from recsys.evaluation import metrics as M
from recsys.features import FeatureStore, user_mask

TEST_LOG = EXPERIMENTS / "test_access_log.jsonl"    # журнал: какие модели и когда оценивались на test

DEFAULT_PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=127,
                      num_threads=NUM_THREADS, verbose=-1, seed=0)


@dataclass
class Mat:
    X: np.ndarray
    y: np.ndarray
    anchor: np.ndarray       # сутки-якорь строки
    uid: np.ndarray
    n_pos: np.ndarray        # всего позитивов в группе (для знаменателей метрик)
    names: list[str]
    group_sizes: np.ndarray


def _group_sizes(anchor: np.ndarray, uid: np.ndarray) -> np.ndarray:
    key = anchor.astype(np.int64) * (1 << 32) + uid.astype(np.int64)
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return np.diff(np.r_[start, len(key)])


def build_mat(store: FeatureStore, days: list[int], names: list[str], neg_keep: float = 1.0, seed: int = 0,
              user_frac: float = 1.0, verbose: bool = True) -> Mat:
    """Матрица признаков для набора якорей.

    neg_keep < 1 оставляет все позитивы и долю негативов (только для обучения);
    user_frac < 1 берёт подвыборку пользователей по хэшу id (одну и ту же для всех якорей).
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Xs, ys, anchors, uids, npos_all = [], [], [], [], []
    for i, day in enumerate(days, 1):
        rows = store.rows(day)
        groups = pl.read_parquet(store.rows_dir / "groups" / f"groups_{day:03d}.parquet").select("uid", "n_pos")
        X = store.get(day, names).to_numpy()
        uid = rows["uid"].to_numpy()
        y = rows["label"].to_numpy().astype(np.int8)
        n_pos = rows.select("uid").join(groups, on="uid", how="left")["n_pos"].to_numpy()
        keep = np.ones(len(y), dtype=bool)
        if user_frac < 1.0:
            keep &= user_mask(uid, user_frac)
        if neg_keep < 1.0:
            keep &= (y == 1) | (rng.random(len(y)) < neg_keep)
        Xs.append(X[keep])
        ys.append(y[keep])
        anchors.append(np.full(keep.sum(), day, dtype=np.int32))
        uids.append(uid[keep])
        npos_all.append(n_pos[keep])
        if verbose and i % 10 == 0:
            print(f"    матрица: {i}/{len(days)} якорей ({time.time() - t0:.0f} с)")
        if i % 5 == 0:
            store.drop_rows_cache()
    anchor, uid = np.concatenate(anchors), np.concatenate(uids)
    return Mat(np.concatenate(Xs), np.concatenate(ys), anchor, uid, np.concatenate(npos_all), names,
               _group_sizes(anchor, uid))


def _dataset(m: Mat, objective: str, reference=None):
    names = [f"f{i}" for i in range(len(m.names))]       # LightGBM не принимает спецсимволы в именах
    if objective == "lambdarank":
        return lgb.Dataset(m.X, m.y, group=m.group_sizes, feature_name=names, reference=reference, free_raw_data=False)
    return lgb.Dataset(m.X, m.y, feature_name=names, reference=reference, free_raw_data=False)


def train_lgb(train: Mat, val: Mat, objective: str = "lambdarank", params: dict | None = None, rounds: int = 1500,
              early: int = 100, log_every: int = 100):
    """Обучение с ранней остановкой по val. Возвращает (бустер, секунды обучения)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if objective == "lambdarank":
        p.update(objective="lambdarank", metric="ndcg", eval_at=[10], lambdarank_truncation_level=100)
    else:
        p.update(objective="binary", metric="binary_logloss")
    dtr = _dataset(train, objective)
    dva = _dataset(val, objective, reference=dtr)
    t = time.time()
    booster = lgb.train(p, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(log_every)])
    return booster, time.time() - t


def predict(booster, m: Mat) -> np.ndarray:
    return booster.predict(m.X, num_iteration=booster.best_iteration, num_threads=NUM_THREADS)


def eval_mat(m: Mat, score: np.ndarray, p: np.ndarray | None = None, boot: int = 0, seed: int = 0):
    return M.evaluate(m.anchor, m.uid, score, m.y, m.n_pos, p=p, seed=seed, boot=boot)


def log_test_access(model_id: str, note: str = "") -> None:
    """Каждое обращение к test пишется в журнал - чтобы по нему можно было проверить, что test не использовался для выбора."""
    TEST_LOG.parent.mkdir(exist_ok=True)
    with open(TEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_id, "note": note}) + "\n")
