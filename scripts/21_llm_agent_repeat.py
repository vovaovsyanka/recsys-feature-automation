"""E21. Повтор поиска признаков ИИ-агентом на нескольких сидах.

В E14 агент запускался один раз, и ответ одношагового режима для сценария B был потерян из-за обрыва по лимиту токенов
(исправлено в llm_agent.parse_specs). Здесь режимы повторяются на сидах 0-2 (с обратной связью - на сидах 1-2, сид 0 уже есть
в E14), чтобы отделить разброс самого агента от разницы между методами. Оценка - в той же песочнице, что и в E11-E14.
Запуск: python scripts/21_llm_agent_repeat.py [--model=gpt-oss:120b]
"""
import json
import sys

from recsys.data import yambda as D
from recsys.search import llm_agent as A
from recsys.search import methods as S
from recsys.search import setup as C
from recsys.search.llm_client import LLMClient
from recsys.tracking import Experiment


def arg(name: str, default: str) -> str:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


MODEL = arg("model", "gpt-oss:120b")
SCENARIOS = {"A": dict(base=C.MAN, max_total=C.K_AUGMENT, rounds=6, per_round=20),
             "B": dict(base=C.CAND, max_total=C.K_REPLACE, rounds=8, per_round=25)}
RUNS = [(seed, mode) for seed in (0, 1, 2) for mode in ("oneshot", "iterative") if not (seed == 0 and mode == "iterative")]

with Experiment("E21", "llm_agent_repeat", "Поиск признаков ИИ-агентом: повторы на разных сидах",
                goal="Оценить разброс результата ИИ-агента между запусками и восстановить потерянный в E14 ответ без обратной связи "
                     "для сценария B.",
                hypothesis="Разброс между запусками агента сопоставим с разницей между агентом и случайным выбором, поэтому единичный "
                           "запуск не позволяет ранжировать агента относительно других методов.",
                config={"model": MODEL, "runs": RUNS, "scenarios": {k: {kk: vv for kk, vv in v.items() if kk != "base"}
                                                                     for k, v in SCENARIOS.items()}},
                data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, pool = C.setup()
    results, sets, transcripts, rounds_log = {}, {}, {}, {}
    cost = S.Cost()
    usage = {}
    C.fit_and_log(px, exp, results, "ручной набор (123)", C.MAN)
    for seed, mode in RUNS:
        client = LLMClient(model=MODEL, seed=seed)
        for scenario, cfg in SCENARIOS.items():
            key = f"{mode}_{scenario}_seed{seed}"
            with exp.stage(f"{key}: агент"):
                run = A.run_agent(px, cfg["base"], client, cost, mode=mode, rounds=cfg["rounds"], per_round=cfg["per_round"],
                                  max_total=cfg["max_total"])
            sets[key] = run.selected
            transcripts[key] = run.transcript
            rounds_log[key] = dict(rounds=run.rounds, invalid=run.invalid, duplicates=run.duplicates)
            C.save_sets(exp.dir, sets)
            (exp.dir / "transcripts.json").write_text(json.dumps(transcripts, ensure_ascii=False, indent=1), encoding="utf-8")
            exp.record("rounds", rounds_log)
            with exp.stage(f"{key}: итоговое обучение"):
                prefix = "ручной + " if scenario == "A" else ""
                C.fit_and_log(px, exp, results, f"{key}: {prefix}принято {len(run.selected)}", cfg["base"] + run.selected)
        usage[f"seed{seed}_{mode}"] = client.usage
        exp.record("llm_usage", usage)
    exp.record("selection_cost", cost.as_dict())
    exp.record("proxy_cost", C.proxy_cost(px))
