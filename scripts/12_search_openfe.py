"""E12. Метод A3: отбор признаков в стиле OpenFE (FeatureBoost, последовательное отсечение, совместная оценка).

Сценарий A: база - ручной набор, добавляем 60 признаков. Сценарий B: база - только источники кандидатов, выбираем 115.
Самый долгий эксперимент поиска (порядка 1,5-2 часов): запускать на свободной машине.
Запуск: python scripts/12_search_openfe.py
"""
import time

from recsys.data import yambda as D
from recsys.search import methods as S
from recsys.search import setup as C
from recsys.tracking import Experiment

with Experiment("E12", "search_openfe", "Поиск признаков: OpenFE",
                goal="Реализовать OpenFE (Zhang et al., ICML 2023, разд. 3.2) на грамматике признаков рекомендаций и сравнить с "
                     "A1/A2 в той же песочнице, включая стоимость отбора.",
                hypothesis="OpenFE найдёт полезные расширения ручного набора, но в сценарии B даст избыточные наборы "
                           "(много окон одного и того же признака); отбор дороже A2 на порядок.",
                config={"user_frac": C.USER_FRAC, "q": 3, "stage2_keep": 300, "val_sub": 0.25, "k_augment": C.K_AUGMENT,
                        "k_replace": C.K_REPLACE}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, pool = C.setup()
    results, sets = {}, {}
    for scenario, base, k in [("A", C.MAN, C.K_AUGMENT), ("B", C.CAND, C.K_REPLACE)]:
        cost = S.Cost()
        with exp.stage(f"сценарий {scenario}: отбор"):
            t = time.time()
            ranked, scores = S.a3_openfe(px, base, pool, cost)
            exp.record(f"A3_{scenario}_selection_seconds", time.time() - t)
        exp.record(f"A3_{scenario}_cost", cost.as_dict())
        exp.record(f"A3_{scenario}_stage1_survivors", sum(1 for v in scores.values() if v > 0))
        exp.record(f"A3_{scenario}_top50", ranked[:50])
        sets[f"A3_{scenario}"] = [n for n, _ in ranked][:k]
        C.save_sets(exp.dir, sets)
        with exp.stage(f"сценарий {scenario}: обучение"):
            title = f"A3, сценарий A: ручной + {k} по OpenFE" if scenario == "A" else f"A3, сценарий B: {k} по OpenFE"
            C.fit_and_log(px, exp, results, title, base + sets[f"A3_{scenario}"])
    exp.record("proxy_cost", C.proxy_cost(px))
