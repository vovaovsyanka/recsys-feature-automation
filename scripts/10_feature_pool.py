"""E10. Расчёт всего пула грамматики (около 3,6 тыс. признаков) на подвыборке пользователей для якорей песочницы.

Пул нужен всем методам автоматического поиска (E11-E14). Расчёт тяжёлый и разовый, результат кэшируется.
Запуск: python scripts/10_feature_pool.py
"""
import time

from recsys import features as F
from recsys.data import yambda as D
from recsys.features import grammar as G
from recsys.features import manual_yambda as MF
from recsys.protocol import VAL, check_protocol, train_anchors
from recsys.search import setup as C
from recsys.tracking import Experiment

check_protocol(train_anchors())
pool = G.grammar_pool()

with Experiment("E10", "feature_pool", "Расчёт пула грамматики (около 3,6 тыс. признаков) на подвыборке",
                goal="Посчитать и сохранить все признаки пула на 20% пользователей для 10 обучающих и 7 val-якорей.",
                hypothesis="Стоимость расчёта растёт линейно с числом признаков; на один якорь уходит порядка минуты.",
                config={"tag": C.TAG, "user_frac": C.USER_FRAC, "train": C.PROXY_TRAIN, "val": VAL, "pool_size": len(pool)},
                data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        store = F.FeatureStore(C.TAG, F.Engine(D.load_events()), user_frac=C.USER_FRAC)
    seconds = {}
    with exp.stage("compute_pool"):
        for part, days in [("val", VAL), ("train", C.PROXY_TRAIN)]:
            for day in days:
                t = time.time()
                store.ensure(day, pool + MF.manual_all())
                seconds[day] = time.time() - t
                print(f"  {part}, сутки {day}: {seconds[day]:.0f} с, строк {store.rows(day).height}")
                store.drop_rows_cache()
    exp.record("seconds_per_anchor", seconds)
