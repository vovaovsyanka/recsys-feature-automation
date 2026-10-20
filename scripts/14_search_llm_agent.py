"""E14. Метод A5: ИИ-агент предлагает признаки в пределах грамматики.

Модель по умолчанию - gpt-oss:120b в облаке Ollama (ключи OLLAMA_API_KEY и OLLAMA_API_KEY2, при исчерпании лимитов - локальный
сервер LM Studio, см. src/recsys/search/llm_client.py). Для каждого сценария запускаются два режима:
  iterative - несколько раундов с обратной связью о приросте качества каждого предложенного признака;
  oneshot   - один запрос без обратной связи.
Все промпты и ответы сохраняются в transcripts.json - прогон можно проверить и проанализировать.
Запуск: python scripts/14_search_llm_agent.py [--model=gpt-oss:120b] [--seed=0]
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


MODEL, SEED = arg("model", "gpt-oss:120b"), int(arg("seed", "0"))
SCENARIOS = {"A": dict(base=C.MAN, max_total=C.K_AUGMENT, rounds=6, per_round=20),
             "B": dict(base=C.CAND, max_total=C.K_REPLACE, rounds=8, per_round=25)}

with Experiment("E14", "search_llm_agent", "Поиск признаков: ИИ-агент (языковая модель)",
                goal="Проверить, может ли языковая модель, видящая только описание данных и грамматики, предлагать признаки не хуже "
                     "человека и не хуже алгоритмического отбора; оценить роль обратной связи.",
                hypothesis="Агент с обратной связью улучшит ручной набор, но не превзойдёт отбор из полного пула (A2); "
                           "без обратной связи прирост меньше.",
                config={"model": MODEL, "seed": SEED, "scenarios": {k: {kk: vv for kk, vv in v.items() if kk != "base"}
                                                                     for k, v in SCENARIOS.items()},
                        "user_frac": C.USER_FRAC}, data_files=[D.EVENTS]) as exp:
    with exp.stage("setup"):
        px, pool = C.setup()
        client = LLMClient(model=MODEL, seed=SEED)
    results, sets, transcripts, rounds_log = {}, {}, {}, {}
    cost = S.Cost()
    C.fit_and_log(px, exp, results, "ручной набор (123)", C.MAN)
    for scenario, cfg in SCENARIOS.items():
        for mode in ["iterative", "oneshot"]:
            key = f"A5_{scenario}" if mode == "iterative" else f"A5s_{scenario}"
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
                scenario_name = "ручной + принятые признаки" if scenario == "A" else "только принятые признаки"
                C.fit_and_log(px, exp, results, f"агент ({mode}), сценарий {scenario}: {scenario_name} ({len(run.selected)})",
                              cfg["base"] + run.selected)
    exp.record("llm_usage", client.usage)
    exp.record("selection_cost", cost.as_dict())
    exp.record("proxy_cost", C.proxy_cost(px))
