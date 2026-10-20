"""Методы автоматического отбора признаков. Все оценивают наборы через общую песочницу `Proxy`.

A1  случайные признаки пула (контроль: что даёт простое добавление признаков);
A2  генерация всего пула + одна модель, отбор по важности (gain) - подход в духе Deep Feature Synthesis;
A3  OpenFE (Zhang et al., ICML 2023, arXiv:2211.12507, разд. 3.2): FeatureBoost + последовательное отсечение
    кандидатов по долям данных + совместная оценка оставшихся и отбор по важности;
A4  обучаемые окна: для каждого семейства (ключ, канал) полупериод затухающего счётчика выбирается по приросту FeatureBoost.

`Cost` накапливает число обучений и секунды, чтобы сравнивать методы по затратам.
"""
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from recsys.config import NUM_THREADS
from recsys import features as F
from recsys.features import grammar as G
from recsys.search.sandbox import Proxy


@dataclass
class Cost:
    fits: int = 0
    fit_seconds: float = 0.0
    boost_evals: int = 0
    boost_seconds: float = 0.0

    def as_dict(self):
        return dict(fits=self.fits, fit_seconds=round(self.fit_seconds, 1), boost_evals=self.boost_evals,
                    boost_seconds=round(self.boost_seconds, 1))


def logloss(y, raw):
    p = np.clip(1 / (1 + np.exp(-raw)), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _feature_names(n: int) -> list[str]:
    return [f"f{i}" for i in range(n)]     # LightGBM не принимает спецсимволы в именах


# ---------------------------------------------------------------------------------------------------- A1
def a1_random(pool: list[str], exclude: set[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    candidates = [n for n in pool if n not in exclude]
    return list(rng.choice(candidates, k, replace=False))


# ---------------------------------------------------------------------------------------------------- A2
def a2_gen_select(px: Proxy, pool: list[str], always: list[str], cost: Cost, rounds: int = 300,
                  feature_fraction: float = 0.3):
    """Одна модель на всём пуле, признаки ранжируются по доле gain. Возвращает список (имя, доля) по убыванию."""
    names = list(dict.fromkeys(always + [n for n in pool if n not in set(always)]))
    tr, _ = px.mats(names)
    params = dict(px.params)
    params.update(objective="lambdarank", lambdarank_truncation_level=100, feature_fraction=feature_fraction, max_bin=31)
    data = lgb.Dataset(tr.X, tr.y, group=tr.group_sizes, feature_name=_feature_names(len(names)))
    t = time.time()
    booster = lgb.train(params, data, rounds)
    cost.fits += 1
    cost.fit_seconds += time.time() - t
    gain = booster.feature_importance("gain")
    gain = gain / gain.sum()
    skip = set(always)
    return [(names[i], float(gain[i])) for i in np.argsort(-gain) if names[i] not in skip]


# ---------------------------------------------------------------------------------------------------- A3
class FeatureBoost:
    """Алгоритм 2 OpenFE: модель на новых признаках подгоняет остаток к предсказаниям базовой модели."""

    def __init__(self, y_tr, raw_tr, y_va, raw_va, rounds=10, lr=0.1, leaves=15, seed=0):
        self.y_tr, self.raw_tr, self.y_va, self.raw_va = y_tr, raw_tr, y_va, raw_va
        self.base_logloss = logloss(y_va, raw_va)
        self.params = dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=200, max_bin=63,
                           verbose=-1, num_threads=NUM_THREADS, seed=seed, feature_fraction=1.0)
        self.rounds = rounds

    def delta(self, X_tr, X_va, rows=None):
        """Прирост логлосса на валидации (больше - лучше); `rows` - подвыборка обучающих строк."""
        y, raw = self.y_tr, self.raw_tr
        if rows is not None:
            X_tr, y, raw = X_tr[rows], y[rows], raw[rows]
        booster = lgb.train(self.params, lgb.Dataset(X_tr, y, init_score=raw), self.rounds)
        raw_va = self.raw_va + booster.predict(X_va, raw_score=True, num_threads=NUM_THREADS)
        return self.base_logloss - logloss(self.y_va, raw_va)


def base_model_scores(px: Proxy, base: list[str], cost: Cost):
    """Базовая модель на `base` и её сырые предсказания на train/val - вход для FeatureBoost."""
    tr, va = px.mats(base)
    params = dict(px.params)
    params.update(objective="binary")
    t = time.time()
    booster = lgb.train(params, lgb.Dataset(tr.X, tr.y), px.rounds)
    cost.fits += 1
    cost.fit_seconds += time.time() - t
    return tr, va, booster.predict(tr.X, raw_score=True, num_threads=NUM_THREADS), \
        booster.predict(va.X, raw_score=True, num_threads=NUM_THREADS)


def a3_openfe(px: Proxy, base: list[str], pool: list[str], cost: Cost, q: int = 3, stage2_keep: int = 300,
              val_sub: float = 0.25, seed: int = 0, log=print):
    """Отбор в стиле OpenFE. `base` - уже имеющиеся признаки (в сценарии «вместо человека» только источники кандидатов).

    Этап I: данные делятся на 2^q блоков; на итерации i берутся 2^i блоков, каждый кандидат получает прирост FeatureBoost,
    остаётся лучшая половина. Этап II: оставшиеся кандидаты вместе с базой обучаются на остатке, отбор по gain.
    """
    candidates = [n for n in pool if n not in set(base)]
    tr0, va0, raw_tr, raw_va = base_model_scores(px, base, cost)
    rng = np.random.default_rng(seed)
    val_mask = F.user_mask(va0.uid, val_sub)          # Δ считается на подвыборке val пользователей - для скорости
    boost = FeatureBoost(tr0.y, raw_tr, va0.y[val_mask], raw_va[val_mask])
    block = rng.integers(0, 2 ** q, len(tr0.y))

    def load(names):
        tr, va = px.mats(names)
        return tr.X, va.X[val_mask]

    alive, scores = list(candidates), {}
    t0 = time.time()
    for i in range(q + 1):
        rows = np.flatnonzero(block < 2 ** i)
        log(f"  [A3] этап I, итерация {i}: {len(alive)} кандидатов, {len(rows)} обучающих строк")
        chunk = 150                                    # признаки читаются порциями, чтобы не держать весь пул в памяти
        for s in range(0, len(alive), chunk):
            part = alive[s:s + chunk]
            X_tr, X_va = load(part)
            for j, name in enumerate(part):
                t = time.time()
                scores[name] = boost.delta(X_tr[:, [j]], X_va[:, [j]], rows)
                cost.boost_evals += 1
                cost.boost_seconds += time.time() - t
        alive = sorted(alive, key=lambda n: -scores[n])
        if i < q:
            alive = alive[:max(1, len(alive) // 2)]
        else:
            alive = [n for n in alive if scores[n] > 0]
    survivors = alive[:stage2_keep]
    log(f"  [A3] после этапа I осталось {len(survivors)} признаков с положительным приростом ({time.time() - t0:.0f} с)")

    X_tr, _ = load(base + survivors)
    params = dict(boost.params)
    params.update(num_leaves=31, feature_fraction=0.5)
    data = lgb.Dataset(X_tr, tr0.y, init_score=raw_tr, feature_name=_feature_names(X_tr.shape[1]))
    t = time.time()
    booster = lgb.train(params, data, 100)
    cost.fits += 1
    cost.fit_seconds += time.time() - t
    gain = booster.feature_importance("gain")
    names = base + survivors
    skip = set(base)
    ranked = [(names[i], float(gain[i] / gain.sum())) for i in np.argsort(-gain) if names[i] not in skip]
    return ranked, scores


# ---------------------------------------------------------------------------------------------------- A4
def families(manual: list[str]) -> list[tuple[str, str]]:
    """Семейства (ключ, канал) счётчиков cnt в наборе, для которых можно подобрать затухание."""
    out = []
    for n in manual:
        if F.is_derived(n) or n in F.ROW_COLS:
            continue
        s = F.Spec.parse(n)
        if s.agg == "cnt" and s.chan in G.DECAY_CHANS:
            out.append((s.key, s.chan))
    return list(dict.fromkeys(out))


def a4_learned_windows(px: Proxy, base: list[str], cost: Cost, log=print):
    """Для каждого семейства выбирает полупериод затухающего счётчика с наибольшим приростом FeatureBoost поверх `base`."""
    tr0, va0, raw_tr, raw_va = base_model_scores(px, base, cost)
    boost = FeatureBoost(tr0.y, raw_tr, va0.y, raw_va, rounds=20)
    chosen, table = {}, {}
    for key, chan in families(base):
        names = [f"{key}.{chan}.decay.{h}" for h in G.HALFLIVES]
        tr, va = px.mats(names)
        row = {}
        for j, name in enumerate(names):
            t = time.time()
            row[name.split(".")[-1]] = boost.delta(tr.X[:, [j]], va.X[:, [j]])
            cost.boost_evals += 1
            cost.boost_seconds += time.time() - t
        best = max(row, key=row.get)
        chosen[f"{key}.{chan}"] = best
        table[f"{key}.{chan}"] = row
        log(f"  [A4] {key}.{chan}: лучший полупериод {best} (прирост {row[best]:.5f})")
    new = [f"{kc}.decay.{h}" for kc, h in chosen.items()]
    return new, chosen, table
