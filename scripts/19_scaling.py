"""E19. Масштабирование: как качество и стоимость зависят от числа пользователей и числа обучающих якорей.

Для ручного набора и лучшего автоматического набора обучаются модели на подвыборках пользователей и на меньшем числе якорей;
оценка идёт на полном val. Нужно, чтобы понять, переносятся ли выводы 10 тысяч пользователей на выборку большего размера.
Запуск: python scripts/19_scaling.py
"""
import time

import psutil

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.features import sets
from recsys.models import gbdt as T
from recsys.protocol import FULL_TRAIN_DAYS, VAL
from recsys.tracking import Experiment

SETS = ["manual", "A2_B"]
USER_FRACTIONS = [0.1, 0.3]            # при 34 якорях
ANCHOR_COUNTS = [9, 17]                # при всех пользователях
NEG_KEEP = 0.3


def thin_out(days: list[int], n: int) -> list[int]:
    """Равномерная подвыборка n якорей из списка."""
    step = len(days) / n
    return [days[int(i * step)] for i in range(n)]


configs = [(f"{int(f * 1000) / 10}% пользователей, {len(FULL_TRAIN_DAYS)} якорей", f, FULL_TRAIN_DAYS) for f in USER_FRACTIONS]
configs += [(f"100% пользователей, {n} якорей", 1.0, thin_out(FULL_TRAIN_DAYS, n)) for n in ANCHOR_COUNTS]

with Experiment("E19", "scaling", "Зависимость от числа пользователей и якорей",
                goal="Оценить, как меняются качество и стоимость обучения при уменьшении числа пользователей и обучающих якорей, "
                     "и сохраняется ли преимущество автоматического набора над ручным.",
                hypothesis="Качество растёт с объёмом данных; автоматический набор, подобранный на всём объёме, не теряет преимущества "
                           "на меньших выборках.",
                config={"sets": SETS, "configs": [c[0] for c in configs], "neg_keep": NEG_KEEP}, data_files=[D.EVENTS]) as exp:
    store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
    results = {}
    for name in SETS:
        names = sets.get_set(name)
        for title, user_frac, days in configs:
            with exp.stage(f"{name}: {title}"):
                t = time.time()
                train = T.build_mat(store, days, names, neg_keep=NEG_KEEP, seed=0, user_frac=user_frac, verbose=False)
                val = T.build_mat(store, list(VAL), names, verbose=False)
                load_seconds = time.time() - t
                booster, seconds = T.train_lgb(train, val, "lambdarank")
                res, *_ = T.eval_mat(val, T.predict(booster, val))
                res.update(train_rows=int(len(train.y)), load_seconds=load_seconds, train_seconds=seconds,
                           best_iteration=int(booster.best_iteration), ram_gb=psutil.Process().memory_info().rss / 2**30)
                results[f"{name} | {title}"] = res
                exp.record("results", results)
                print(f"{name:7s} {title:34s} ndcg@10={res['ndcg@10']:.4f}  строк {res['train_rows']}  обучение {seconds:.0f} с")
                del train, val, booster
