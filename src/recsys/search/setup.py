"""Общие настройки экспериментов по поиску признаков на Yambda: песочница, базовые наборы и размеры сценариев."""
import json
from pathlib import Path

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import grammar as G
from recsys.features import manual_yambda
from recsys.search.sandbox import Proxy

TAG = YAMBDA_TAG
USER_FRAC = 0.2                                   # доля пользователей в песочнице
PROXY_TRAIN = [163, 175, 187, 199, 211, 223, 235, 247, 259, 271]
SEEDS = [0, 1, 2, 3, 4]

CAND = sorted(F.ROW_COLS)                         # признаки источников кандидатов: одинаково доступны всем наборам
MAN = manual_yambda.manual_all()                  # ручной набор: 123 признака, из них 8 - источники кандидатов
MAN_NONCAND = [n for n in MAN if n not in F.ROW_COLS]

# Два сценария сравнения ручного и автоматического наборов:
K_AUGMENT = 60                                    # A: к ручному набору добавляется 60 автоматических признаков
K_REPLACE = len(MAN_NONCAND)                      # B: автоматические признаки целиком заменяют ручные (того же размера)


def setup():
    """Песочница, пул грамматики."""
    store = F.FeatureStore(TAG, F.Engine(D.load_events()), user_frac=USER_FRAC)
    return Proxy(store, PROXY_TRAIN), G.grammar_pool()


def save_sets(exp_dir: Path, sets: dict) -> None:
    (exp_dir / "selected_feature_sets.json").write_text(json.dumps(sets, indent=1, ensure_ascii=False), encoding="utf-8")


def fit_and_log(px: Proxy, exp, results: dict, tag: str, names: list[str], objective: str = "lambdarank") -> dict:
    """Обучает модель на `names` в песочнице, печатает и сохраняет метрики в results/эксперимент."""
    rec, _ = px.evaluate(names, tag=tag, objective=objective)
    print(f"{tag:52s} признаков {rec['n_features']:4d}  ndcg@10={rec['ndcg10']:.4f}  ndcg@100={rec['ndcg100']:.4f}  "
          f"auc={rec['auc']:.4f}  ({rec['seconds']:.0f} с)")
    results[tag] = rec
    exp.record("proxy_results", results)
    return rec


def proxy_cost(px: Proxy) -> dict:
    return dict(fits=px.n_fits, fit_seconds=round(px.fit_seconds, 1), load_seconds=round(px.load_seconds, 1))
