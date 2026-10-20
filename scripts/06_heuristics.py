"""E06. Эвристические ранкеры (по одному признаку) на общих кандидатах, только val.

Нужны как нижняя граница качества и для проверки метрик: случайный порядок, глобальная популярность, личные счётчики, давность.
Запуск: python scripts/06_heuristics.py
"""
import numpy as np

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.models import gbdt as T
from recsys.protocol import VAL
from recsys.tracking import Experiment

# название -> (признак, знак); знак -1 значит «чем меньше, тем выше в списке»
HEURISTICS = {
    "случайный порядок": None,
    "популярность за 7 суток (I.plus.cnt.7)": ("I.plus.cnt.7", +1),
    "затухающая популярность h=7 (I.plus.decay.h7)": ("I.plus.decay.h7", +1),
    "затухающая популярность h=3 (I.plus.decay.h3)": ("I.plus.decay.h3", +1),
    "личный счётчик за всё время (UI.plus.cnt.all)": ("UI.plus.cnt.all", +1),
    "личный счётчик за 30 суток (UI.plus.cnt.30)": ("UI.plus.cnt.30", +1),
    "личный затухающий счётчик h=14 (UI.plus.decay.h14)": ("UI.plus.decay.h14", +1),
    "личный затухающий счётчик h=3 (UI.plus.decay.h3)": ("UI.plus.decay.h3", +1),
    "давность последнего прослушивания (UI.listen.last.all)": ("UI.listen.last.all", -1),
    "затухающий счётчик артиста h=14 (UA.plus.decay.h14)": ("UA.plus.decay.h14", +1),
}

with Experiment("E06", "heuristics", "Эвристические ранкеры на общих кандидатах (val)",
                goal="Получить нижнюю границу качества и проверить метрики: случайный порядок, популярность, личные счётчики, давность.",
                hypothesis="Личные счётчики (повторное потребление) намного сильнее глобальной популярности.",
                config={"tag": YAMBDA_TAG, "val_days": VAL, "heuristics": list(HEURISTICS)}, data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
    names = sorted({h[0] for h in HEURISTICS.values() if h})
    with exp.stage("features"):
        for day in VAL:
            store.ensure(day, names)
    with exp.stage("evaluate"):
        mat = T.build_mat(store, list(VAL), names, verbose=False)
        rng = np.random.default_rng(0)
        results = {}
        for title, spec in HEURISTICS.items():
            score = rng.random(len(mat.y)) if spec is None else np.nan_to_num(mat.X[:, mat.names.index(spec[0])] * spec[1], nan=-1e9)
            res, *_ = T.eval_mat(mat, score)
            results[title] = res
            print(f"{title:56s} ndcg@10={res['ndcg@10']:.4f} ndcg@100={res['ndcg@100']:.4f} "
                  f"recall@100={res['recall@100']:.4f} auc={res['auc']:.4f}")
        exp.record("val_metrics", results)
        # идеальный порядок внутри кандидатов: потолок, зависящий только от кандидатов
        oracle, *_ = T.eval_mat(mat, mat.y.astype(float) + rng.random(len(mat.y)) * 1e-3)
        exp.record("oracle_within_candidates", oracle)
        print("оракул:", oracle)
