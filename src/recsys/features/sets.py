"""Реестр наборов признаков, которые доходят до финального сравнения.

Наборы, найденные автоматически, лежат в experiments/<эксперимент>/selected_feature_sets.json (их записали сами скрипты поиска;
выбор сделан только по train/val). Здесь они собираются в итоговые списки признаков.

Yambda:  manual - ручной набор (123);   A1_* - случайные признаки грамматики (контроль);   A2_* - генерация пула и отбор по важности;
         A3_* - OpenFE;   A4 - обучаемые окна;   A5_* - предложения ИИ-агента;   H_* - ручной набор + векторные признаки (ALS, LightGCN).
Суффикс _A: ручной набор плюс автоматические признаки; _B: автоматические признаки вместо ручных (того же размера).
KION:    ключи с префиксом kion: (kion:manual, kion:A1_B, kion:A2_A, kion:A2_B).
"""
import json

from recsys.config import EXPERIMENTS
from recsys.features import ROW_COLS
from recsys.features import manual_kion, manual_yambda

CAND = sorted(ROW_COLS)                         # признаки источников кандидатов: доступны всем наборам одинаково
CAND_KION = ["s1_rank", "s1_score", "s2_rank", "s2_score", "s3_rank", "s3_score"]

# где лежат найденные наборы
_SOURCES = {
    "A1": "E11_search_random_genselect", "A2": "E11_search_random_genselect", "A3": "E12_search_openfe",
    "A4": "E13_learned_windows", "A5": "E14_search_llm_agent", "A5s": "E14_search_llm_agent",
}
EMBEDDING_FEATURES = {"H_als": ["ext:als.ui"], "H_gcn": ["ext:gcn.ui", "ext:gcn.ua"],
                      "H_both": ["ext:als.ui", "ext:gcn.ui", "ext:gcn.ua"]}

YAMBDA_FINALISTS = ["manual", "A1_A", "A1_B", "A2_A", "A2_B", "A3_A", "A3_B", "A4", "A5_A", "A5_B", "H_als", "H_gcn", "H_both"]
KION_FINALISTS = ["kion:manual", "kion:A1_B", "kion:A2_A", "kion:A2_B"]


def _saved(experiment_dir: str) -> dict:
    return json.loads((EXPERIMENTS / experiment_dir / "selected_feature_sets.json").read_text(encoding="utf-8"))


def get_set(name: str) -> list[str]:
    if name.startswith("kion:"):
        return _get_kion(name[5:])
    manual = manual_yambda.manual_all()
    if name == "manual":
        return manual
    if name in EMBEDDING_FEATURES:
        return manual + EMBEDDING_FEATURES[name]
    method, scenario = (name.split("_") + [""])[:2]
    saved = _saved(_SOURCES[method])
    if method == "A1":                          # контроль: берём seed 0, зафиксированный заранее (не лучший из пяти)
        key = f"A1_{scenario}_seed0"
        return manual + saved[key] if scenario == "A" else CAND + saved[key]
    if method == "A4":
        return manual + saved["A4"]
    return manual + saved[name] if scenario == "A" else CAND + saved[name]


def _get_kion(name: str) -> list[str]:
    manual = manual_kion.manual_all()
    if name == "manual":
        return manual
    saved = _saved("E32_kion_search")
    method, scenario = name.split("_")
    key = f"A1_{scenario}_seed0" if method == "A1" else name
    return manual + saved[key] if scenario == "A" else CAND_KION + saved[key]
