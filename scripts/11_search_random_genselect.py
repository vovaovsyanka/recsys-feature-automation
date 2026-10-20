"""E11. Методы A1 (случайные признаки грамматики, контроль) и A2 (сгенерировать весь пул и отобрать по важности).

Два сценария: A - к ручному набору добавляется 60 автоматических признаков; B - 115 автоматических признаков вместо ручных
(плюс признаки источников кандидатов). Шум самой песочницы оценивается по пяти сидам LightGBM.
Запуск: python scripts/11_search_random_genselect.py
"""
import time

import numpy as np

from recsys.data import yambda as D
from recsys.search import methods as S
from recsys.search import setup as C
from recsys.tracking import Experiment

with Experiment("E11", "search_random_genselect", "Поиск признаков: случайный выбор (A1) и генерация пула с отбором (A2)",
                goal="Сравнить в одной песочнице случайное расширение грамматикой (контроль) и подход «сгенерировать весь пул "
                     "и оставить признаки с наибольшей важностью». Сценарии: +60 к ручному набору и 115 вместо ручных.",
                hypothesis="Отбор по важности лучше случайного; автоматический набор не хуже ручного набора того же размера.",
                config={"user_frac": C.USER_FRAC, "proxy_train": C.PROXY_TRAIN, "k_augment": C.K_AUGMENT,
                        "k_replace": C.K_REPLACE, "seeds": C.SEEDS}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, pool = C.setup()
    cost, results, sets = S.Cost(), {}, {}
    fit = lambda tag, names: C.fit_and_log(px, exp, results, tag, names)

    with exp.stage("references"):
        fit("только источники кандидатов (8)", C.CAND)
        fit("ручной набор (123)", C.MAN)
        for seed in range(1, 5):                      # шум песочницы: тот же набор, другой сид LightGBM
            px.params["seed"] = seed
            fit(f"ручной набор (123), сид LightGBM {seed}", C.MAN)
        px.params["seed"] = 0
        values = [r["ndcg10"] for t, r in results.items() if t.startswith("ручной набор")]
        exp.record("proxy_seed_noise_ndcg10", dict(values=values, mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
    with exp.stage("A1_random"):
        for seed in C.SEEDS:
            sets[f"A1_A_seed{seed}"] = S.a1_random(pool, set(C.MAN), C.K_AUGMENT, seed)
            fit(f"A1, сценарий A: ручной + {C.K_AUGMENT} случайных, сид {seed}", C.MAN + sets[f"A1_A_seed{seed}"])
            sets[f"A1_B_seed{seed}"] = S.a1_random(pool, set(C.CAND), C.K_REPLACE, 100 + seed)
            fit(f"A1, сценарий B: {C.K_REPLACE} случайных, сид {seed}", C.CAND + sets[f"A1_B_seed{seed}"])
    with exp.stage("A2_generate_and_select"):
        t = time.time()
        ranked = S.a2_gen_select(px, pool, C.CAND, cost)
        exp.record("A2_selection_seconds", time.time() - t)
        exp.record("A2_top50_by_gain", ranked[:50])
        sets["A2_A"] = [n for n, _ in ranked if n not in set(C.MAN)][:C.K_AUGMENT]
        sets["A2_B"] = [n for n, _ in ranked][:C.K_REPLACE]
        fit(f"A2, сценарий A: ручной + {C.K_AUGMENT} по важности", C.MAN + sets["A2_A"])
        fit(f"A2, сценарий B: {C.K_REPLACE} по важности", C.CAND + sets["A2_B"])
    C.save_sets(exp.dir, sets)
    exp.record("selection_cost", cost.as_dict())
    exp.record("proxy_cost", C.proxy_cost(px))
