"""ИИ-агент как метод автоматического поиска признаков (A5).

Агент (языковая модель) предлагает признаки только в виде строк грамматики; система проверяет их допустимость, считает
прирост качества и возвращает агенту обратную связь. Исполнять код агента не нужно, а любой допустимый признак читает
события лишь до момента прогноза (это проверено scripts/05_leakage_tests.py), так что агент не может внести утечку.

Два режима: iterative - раунды с обратной связью (по аналогии с CAAFE, Hollmann et al. 2023, и LLM-FE, Abhyankar et al. 2025);
oneshot - один запрос без обратной связи (показывает, сколько даёт одна лишь «экспертиза» модели).
"""
import json
import re
import time
from dataclasses import dataclass, field

from recsys import features as F
from recsys.search import methods as S
from recsys.search.llm_client import LLMClient
from recsys.search.sandbox import Proxy

GRAMMAR_DOC = """
Признак - строка вида  <ключ>.<канал>.<агрегат>.<окно>  (производные признаки - ниже).
Ключи: U (пользователь), I (трек), UI (пара пользователь-трек), UA (пользователь×артист), UB (пользователь×альбом),
  A (артист глобально), B (альбом глобально).
Каналы событий: listen (любое прослушивание), plus (прослушано >= 50%), skip (< 20%), full (>= 90%), org (органическое),
  rec (из рекомендаций), plus_org, plus_rec, like, unlike, dislike, undislike.
Агрегаты:
  cnt    - число событий канала за окно (любой канал)
  mean   - средняя доля прослушивания (только каналы listen, org, rec)
  sec    - суммарно прослушано секунд (только канал listen)
  last   - суток с последнего события канала (каналы listen, plus, skip, org, rec, like, dislike)
  first  - суток с первого события канала (каналы listen, plus, like)
  decay  - сумма событий с экспоненциальным затуханием; окно записывается полупериодом в сутках: h3, h7, h14 (любой канал)
  nuniq  - число различных партнёров: для U, UA, UB - различных треков, для I, A, B - различных пользователей
Окно: число суток (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90 или любое другое) либо all - вся история до момента прогноза.
Для decay окно имеет вид h<число>.
Производные признаки:
  d:ratio(<A>|<B>) - частное двух признаков (пропуск, если знаменатель 0); A и B - любые признаки, в том числе производные;
  d:diff(<A>|<B>)  - разность;
  d:urank(<A>)     - процентильный ранг значения A среди кандидатов того же пользователя в тот же момент (от 0 до 1).
Примеры: UI.plus.cnt.7, U.listen.mean.30, UA.listen.last.all, I.plus.decay.h7, d:ratio(UI.listen.cnt.7|U.listen.cnt.7),
  d:urank(UI.plus.decay.h3).
Все признаки автоматически используют только события строго до момента прогноза.
"""

SYSTEM = ("Ты инженер по признакам в рекомендательной системе музыкального сервиса. Ты предлагаешь признаки для градиентного "
          "бустинга, который ранжирует кандидатов (треки) для пользователя. Код ты не пишешь: ты выдаёшь спецификации признаков "
          "в заданной грамматике, система их вычисляет, обучает модель и сообщает прирост качества. "
          "Отвечай только одним JSON-объектом без текста вне JSON.")

TASK = """Данные и задача.
- Журнал Яндекс Музыки: прослушивания (пользователь, трек, время, органическое или рекомендованное, доля прослушанного 0-100%,
  длина трека), лайки, дизлайки и их отмены. 10 тыс. пользователей, 300 суток, около 5 000 прослушиваний на пользователя.
- В начале суток для пары (пользователь, кандидат) нужно предсказать, прослушает ли пользователь трек (доля >= 50%) в течение
  следующих суток. Кандидаты (около 265 на пользователя) уже отобраны: любимые и недавние треки пользователя, треки его
  артистов, хиты недели.
- Известно: около 70% будущих прослушиваний - повторы треков из истории; глобальная популярность почти не помогает.
- Метрика: NDCG@10 на отложенных сутках; модель: LightGBM lambdarank.

Грамматика признаков.
""" + GRAMMAR_DOC


def parse_specs(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        payload = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        # ответ оборван по лимиту токенов (в E14 так потерялся весь ответ oneshot для сценария B): берём закрытые строки списка
        start = text.find('"features"')
        return re.findall(r'"([^"\n]+)"', text[start:])[1:] if start >= 0 else []
    out = []
    for item in payload.get("features") or payload.get("specs") or []:
        if isinstance(item, dict):
            item = item.get("spec") or item.get("name")
        if isinstance(item, str):
            out.append(item.strip())
    return out


def is_valid(spec: str) -> bool:
    """Строгая проверка допустимости: грамматика, сочетание канала и агрегата, рекурсивно для производных."""
    try:
        if F.is_derived(spec):
            return all(is_valid(a) for a in F.parse_derived(spec)[1])
        return spec in F.ROW_COLS or F.Spec.parse(spec).valid()
    except Exception:
        return False


def build_prompt(current: list[str], history: list[dict], n_new: int) -> str:
    groups: dict[str, list[str]] = {}
    for name in current:
        key = name.split("(")[0] if F.is_derived(name) else name.split(".")[0]
        groups.setdefault(key, []).append(name)
    listing = "\n".join(f"  [{k}] " + ", ".join(v) for k, v in groups.items()) or "  (пусто)"
    feedback = ""
    if history:
        lines = []
        for h in history[-3:]:
            good = ", ".join(f"{s} ({d * 1e4:+.2f})" for s, d in h["accepted"][:12]) or "-"
            bad = ", ".join(f"{s} ({d * 1e4:+.2f})" for s, d in h["rejected"][:12]) or "-"
            lines.append(f"Раунд {h['round']}: NDCG@10 набора = {h['ndcg']:.4f}. Приняты: {good}. Отвергнуты: {bad}.")
        feedback = ("\nОбратная связь (в скобках прирост логлосса ×1e4, больше - лучше; принимаются признаки с приростом > 0.1):\n"
                    + "\n".join(lines) + "\n")
    return (TASK + "\nТекущий набор признаков:\n" + listing + "\n" + feedback +
            f"\nПредложи ровно {n_new} НОВЫХ признаков, которых нет в текущем наборе. Используй разные ключи, каналы и окна, "
            'можно производные. Формат ответа:\n{"features": ["UI.plus.decay.h5", "d:ratio(...|...)"], '
            '"rationale": "1-3 предложения"}')


@dataclass
class AgentRun:
    selected: list[str] = field(default_factory=list)       # принятые признаки (без исходной базы)
    rounds: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    invalid: int = 0
    duplicates: int = 0


def run_agent(px: Proxy, base: list[str], client: LLMClient, cost: S.Cost, mode: str = "iterative", rounds: int = 6,
              per_round: int = 25, max_total: int = 60, min_gain: float = 1e-5, log=print) -> AgentRun:
    """Запускает агента. `base` - уже имеющиеся признаки, `max_total` - предельное число принимаемых признаков."""
    run = AgentRun()
    current = list(base)
    seen = set(base)
    n_rounds = 1 if mode == "oneshot" else rounds
    for r in range(1, n_rounds + 1):
        n_ask = max_total if mode == "oneshot" else min(per_round, max_total - len(run.selected))
        if n_ask <= 0:
            break
        prompt = build_prompt(current, run.rounds if mode == "iterative" else [], n_ask)
        reply = client.chat(SYSTEM, prompt, max_tokens=12000 if mode == "oneshot" else 4000)   # модель тратит токены на рассуждение
        proposed = parse_specs(reply.text)
        valid = []
        for spec in proposed:
            if not is_valid(spec):
                run.invalid += 1
            elif spec in seen or spec in valid:
                run.duplicates += 1
            else:
                valid.append(spec)
        run.transcript.append({"round": r, "prompt": prompt, "reply": reply.text, "source": reply.source, "model": reply.model,
                               "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out, "proposed": proposed, "valid_new": valid})
        log(f"  [A5/{mode}] раунд {r}: предложено {len(proposed)}, допустимых новых {len(valid)} ({reply.source}, {reply.seconds:.0f} с)")
        if not valid:
            continue
        seen.update(valid)
        if mode == "oneshot":
            accepted, rejected = valid[:max_total], []
            gains = {s: 0.0 for s in accepted}
        else:
            tr, va, raw_tr, raw_va = S.base_model_scores(px, current, cost)
            boost = S.FeatureBoost(tr.y, raw_tr, va.y, raw_va, rounds=20)
            tr_c, va_c = px.mats(valid)
            gains = {}
            for j, spec in enumerate(valid):
                t = time.time()
                gains[spec] = boost.delta(tr_c.X[:, [j]], va_c.X[:, [j]])
                cost.boost_evals += 1
                cost.boost_seconds += time.time() - t
            ordered = sorted(valid, key=lambda s: -gains[s])
            accepted = [s for s in ordered if gains[s] > min_gain][:max_total - len(run.selected)]
            rejected = [s for s in ordered if s not in accepted]
        run.selected += accepted
        current += accepted
        rec, _ = px.evaluate(current, tag=f"A5 {mode} раунд {r}")
        run.rounds.append({"round": r, "ndcg": rec["ndcg10"], "n_features": len(current), "n_proposed": len(proposed),
                           "n_valid": len(valid), "accepted": [(s, gains[s]) for s in accepted],
                           "rejected": [(s, gains[s]) for s in rejected]})
        log(f"  [A5/{mode}] раунд {r}: принято {len(accepted)}, всего {len(run.selected)}, NDCG@10 proxy {rec['ndcg10']:.4f}")
    return run
