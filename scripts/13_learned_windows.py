"""E13. Метод A4: обучаемые окна.

Для каждого семейства (ключ, канал) ручного набора полупериод затухающего счётчика выбирается по приросту FeatureBoost поверх
ручного набора. Затем выбранные признаки добавляются в набор и сравниваются с теми же семействами при фиксированных полупериодах.
Запуск: python scripts/13_learned_windows.py
"""
from recsys.data import yambda as D
from recsys.features import grammar as G
from recsys.search import methods as S
from recsys.search import setup as C
from recsys.tracking import Experiment

with Experiment("E13", "learned_windows", "Обучаемые окна: полупериоды затухающих счётчиков",
                goal="Ответить на вопрос «почему именно 7 суток?»: выбрать окно по данным. Для каждого семейства подобрать "
                     "полупериод и сравнить выигрыш с ручными окнами 1/7/30 и с фиксированными полупериодами.",
                hypothesis="Оптимальные полупериоды различаются по семействам; добавление затухающих счётчиков даёт прирост, "
                           "но меньший, чем отбор из всего пула.",
                config={"halflives": G.HALFLIVES, "boost_rounds": 20}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, pool = C.setup()
    cost, results = S.Cost(), {}
    with exp.stage("select_half_lives"):
        learned, chosen, deltas = S.a4_learned_windows(px, C.MAN, cost)
    exp.record("chosen_half_lives", chosen)
    exp.record("gain_table", deltas)
    exp.record("selection_cost", cost.as_dict())

    def with_halflife(h: str) -> list[str]:
        return [n.rsplit(".", 1)[0] + "." + h for n in learned]

    valid = set(pool) | set(C.CAND)
    with exp.stage("fits"):
        for title, names in [("ручной набор (123)", C.MAN),
                             (f"ручной + {len(learned)} затухающих с выбранным полупериодом", C.MAN + learned),
                             ("ручной + те же семейства с h=3", C.MAN + with_halflife("h3")),
                             ("ручной + те же семейства с h=7", C.MAN + with_halflife("h7")),
                             ("ручной + те же семейства с h=14", C.MAN + with_halflife("h14"))]:
            C.fit_and_log(px, exp, results, title, [n for n in names if n in valid])
    C.save_sets(exp.dir, {"A4": learned})
    exp.record("proxy_cost", C.proxy_cost(px))
