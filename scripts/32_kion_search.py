"""E32. KION: автоматический поиск признаков (A1 случайные, A2 генерация пула и отбор) в песочнице. Test не используется.

Сценарии, как на Yambda: A - к ручному набору добавляется 40 признаков; B - автоматические признаки вместо ручных (того же размера).
Запуск: python scripts/32_kion_search.py
"""
import numpy as np

from recsys import features as F
from recsys.config import KION_TAG
from recsys.data import kion as K
from recsys.features import manual_kion
from recsys.search import methods as S
from recsys.search import setup as Y
from recsys.search import setup_kion as C
from recsys.search.sandbox import Proxy
from recsys.tracking import Experiment

pool = manual_kion.pool()

with Experiment("E32", "kion_search", "KION: поиск признаков A1 и A2 в песочнице",
                goal="Проверить на втором домене выводы Yambda: превосходит ли отбор из пула грамматики (A2) ручной набор и "
                     "случайный контроль (A1).",
                hypothesis="A2 не хуже ручного набора; случайные признаки грамматики не хуже ручного набора; эффект меньше, чем на Yambda.",
                config={"proxy_user_frac": C.PROXY_USER_FRAC, "proxy_train": C.PROXY_TRAIN, "k_augment": C.K_AUGMENT,
                        "k_replace": C.K_REPLACE, "pool_size": len(pool)}, data_files=[K.EVENTS]) as exp:
    with exp.stage("setup"):
        store = F.FeatureStore(KION_TAG, F.Engine(K.load_events()), user_frac=C.PROXY_USER_FRAC)
        px = Proxy(store, C.PROXY_TRAIN, val_days=[C.VAL_DAY], neg_keep=C.PROXY_NEG_KEEP)
    cost, results, sets = S.Cost(), {}, {}
    fit = lambda tag, names: Y.fit_and_log(px, exp, results, tag, names)

    with exp.stage("compute_pool"):
        for day in C.PROXY_TRAIN + [C.VAL_DAY]:
            store.ensure(day, pool + C.MAN)
            store.drop_rows_cache()
    with exp.stage("references"):
        fit("только источники кандидатов", C.CAND)
        fit("ручной набор", C.MAN)
        for seed in range(1, 4):
            px.params["seed"] = seed
            fit(f"ручной набор, сид LightGBM {seed}", C.MAN)
        px.params["seed"] = 0
        values = [r["ndcg10"] for t, r in results.items() if t.startswith("ручной набор")]
        exp.record("proxy_seed_noise_ndcg10", dict(values=values, mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
    with exp.stage("A1_random"):
        for seed in range(5):
            sets[f"A1_A_seed{seed}"] = S.a1_random(pool, set(C.MAN), C.K_AUGMENT, seed)
            fit(f"A1, сценарий A: ручной + {C.K_AUGMENT} случайных, сид {seed}", C.MAN + sets[f"A1_A_seed{seed}"])
            sets[f"A1_B_seed{seed}"] = S.a1_random(pool, set(C.CAND), C.K_REPLACE, 100 + seed)
            fit(f"A1, сценарий B: {C.K_REPLACE} случайных, сид {seed}", C.CAND + sets[f"A1_B_seed{seed}"])
    with exp.stage("A2_generate_and_select"):
        ranked = S.a2_gen_select(px, pool, C.CAND, cost, rounds=300)
        exp.record("A2_top40_by_gain", ranked[:40])
        sets["A2_A"] = [n for n, _ in ranked if n not in set(C.MAN)][:C.K_AUGMENT]
        sets["A2_B"] = [n for n, _ in ranked][:C.K_REPLACE]
        fit(f"A2, сценарий A: ручной + {C.K_AUGMENT} по важности", C.MAN + sets["A2_A"])
        fit(f"A2, сценарий B: {C.K_REPLACE} по важности", C.CAND + sets["A2_B"])
    Y.save_sets(exp.dir, sets)
    exp.record("selection_cost", cost.as_dict())
    exp.record("proxy_cost", Y.proxy_cost(px))
