"""E09. Аблация групп ручных признаков и чувствительность к выбору окон человеком.

Всё в песочнице (20% пользователей, 10 train-якорей, 7 val-якорей, lambdarank на 150 деревьев). Test не используется.
Запуск: python scripts/09_ablation_windows.py
"""
import re

import numpy as np

from recsys import features as F
from recsys.data import yambda as D
from recsys.features import manual_yambda as MF
from recsys.search import setup as C
from recsys.tracking import Experiment

GROUPS = MF.manual_groups()
MAN = C.MAN
WINDOW_TOKEN = re.compile(r"(?<=\.)(1|7|30)(?=[|)]|$)")      # окна 1, 7 и 30 в названии признака
ALL_WINDOWS = ["1", "2", "3", "5", "7", "10", "14", "21", "30", "45", "60", "90"]


def is_valid(name: str) -> bool:
    return all(b in F.ROW_COLS or F.Spec.parse(b).valid() for b in F.base_deps(name))


def remap_windows(names: list[str], mapping: dict[str, str]) -> list[str]:
    """Заменяет окна 1/7/30 ручного набора на другие; в производных признаках - независимо в каждом месте."""
    out = list(dict.fromkeys(WINDOW_TOKEN.sub(lambda m: mapping[m.group(1)], n) for n in names))
    assert all(is_valid(n) for n in out)
    return out


def expand_windows(names: list[str], grid: list[str]) -> list[str]:
    """К каждому ручному признаку с окном 1/7/30 добавляет версии со всеми окнами сетки (для признаков с одним окном)."""
    out = list(names)
    for n in names:
        tokens = WINDOW_TOKEN.findall(n)
        if tokens and len(set(tokens)) == 1:
            out += [WINDOW_TOKEN.sub(w, n) for w in grid]
    return [n for n in dict.fromkeys(out) if is_valid(n)]


with Experiment("E09", "ablation_windows", "Аблация групп признаков и чувствительность к выбору окон",
                goal="(а) оценить вклад групп ручных признаков; (б) измерить, насколько качество зависит от того, "
                     "какие окна выбрал человек (1/7/30 против других наборов).",
                hypothesis="(а) основной вклад даёт группа пользователь×трек (прямые связи), косвенные связи вторичны; "
                           "(б) разброс качества между наборами окон сопоставим с вкладом целых групп признаков.",
                config={"user_frac": C.USER_FRAC, "train": C.PROXY_TRAIN}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, _ = C.setup()
    out = {"ablation": {}, "windows": {}}

    def run(tag, names):
        rec, _ = px.evaluate(names, tag=tag)
        print(f"{tag:42s} признаков {rec['n_features']:4d}  ndcg@10={rec['ndcg10']:.4f}  ndcg@100={rec['ndcg100']:.4f}  "
              f"auc={rec['auc']:.4f}")
        exp.record("results", out)
        return rec

    with exp.stage("ablation"):
        out["ablation"]["полный набор"] = run("полный набор", MAN)
        for group in GROUPS:
            out["ablation"][f"без группы {group}"] = run(f"без группы {group}", [n for n in MAN if n not in GROUPS[group]])
        for group in ["user_item", "user", "item", "cand"]:
            names = GROUPS[group] + (GROUPS["cand"] if group != "cand" else [])
            out["ablation"][f"только {group} + cand"] = run(f"только {group} + cand", names)
    with exp.stage("windows"):
        variants = {
            "ручные (1, 7, 30)": MAN,
            "(2, 5, 14)": remap_windows(MAN, {"1": "2", "7": "5", "30": "14"}),
            "(3, 14, 90)": remap_windows(MAN, {"1": "3", "7": "14", "30": "90"}),
            "(1, 3, 10)": remap_windows(MAN, {"1": "1", "7": "3", "30": "10"}),
            "(5, 21, 60)": remap_windows(MAN, {"1": "5", "7": "21", "30": "60"}),
            "(7, 30, 90)": remap_windows(MAN, {"1": "7", "7": "30", "30": "90"}),
            "сетка из 12 окон": expand_windows(MAN, ALL_WINDOWS),
            "редкая сетка (1, 3, 7, 14, 30, 90)": expand_windows(MAN, ["1", "3", "7", "14", "30", "90"]),
        }
        rng = np.random.default_rng(0)
        for _ in range(6):                            # случайные тройки окон: как сильно можно ошибиться в выборе
            small, mid, large = sorted(rng.choice(ALL_WINDOWS, 3, replace=False), key=int)
            variants[f"случайные ({small}, {mid}, {large})"] = remap_windows(MAN, {"1": small, "7": mid, "30": large})
        for title, names in variants.items():
            out["windows"][title] = run(title, names)
    ndcg = [v["ndcg10"] for k, v in out["windows"].items() if "сетка" not in k]
    exp.record("window_triples_ndcg10", dict(min=min(ndcg), max=max(ndcg), spread=max(ndcg) - min(ndcg)))
    exp.record("proxy_cost", dict(n_fits=px.n_fits, fit_seconds=px.fit_seconds, load_seconds=px.load_seconds))
