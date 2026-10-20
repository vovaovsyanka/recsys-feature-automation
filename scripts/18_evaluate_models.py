"""E18. Оценка сохранённых моделей на val или test с доверительными интервалами.

Доверительные интервалы - парный бутстреп по пользователям (1000 повторов): пользователи пересэмплируются целиком, поэтому
учитывается, что один пользователь даёт несколько групп (якорей). Разности считаются относительно ручного базлайна.
Каждое обращение к test пишется в experiments/test_access_log.jsonl.

Запуск: python scripts/18_evaluate_models.py --split=val|test [--sets=manual,A2_B,A2_B@1] [--extra=имя=путь_к_per_group.npz]
Суффикс @N у набора - модель, обученная с сидом N (scripts/17_final_model.py --seed=N).
"""
import sys

import lightgbm as lgb
import numpy as np

from recsys import features as F
from recsys.config import EXPERIMENTS, NUM_THREADS, YAMBDA_TAG
from recsys.data import yambda as D
from recsys.evaluation import metrics as M
from recsys.features import sets
from recsys.models import gbdt as T
from recsys.protocol import TEST, VAL
from recsys.tracking import Experiment


def arg(name: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def model_path(name: str):
    base, _, seed = name.partition("@")
    if base == "manual" and not seed:
        return EXPERIMENTS / "E08_baseline_gbdt" / "model_lambdarank.txt"
    return EXPERIMENTS / f"E17_final_{base}{'_s' + seed if seed else ''}" / "model_lambdarank.txt"


SPLIT = arg("split", "val")
SETS = arg("sets", ",".join(sets.YAMBDA_FINALISTS)).split(",")
missing = [s for s in SETS if not model_path(s).exists()]
if missing:
    print("модели не найдены, пропускаю:", ", ".join(missing))
SETS = [s for s in SETS if s not in missing]
EXTRA = dict(e.split("=", 1) for e in (arg("extra", "") or "").split(",") if e)    # модели не из LightGBM (по группам)
N_BOOT = int(arg("boot", "1000"))
DAYS = list(VAL if SPLIT == "val" else TEST)

with Experiment(f"E18{SPLIT}", "evaluate_models", f"Оценка моделей на {SPLIT} с доверительными интервалами",
                goal=f"Единообразно оценить все финальные модели на {SPLIT} (7 суточных якорей) и сравнить с ручным базлайном "
                     "с учётом неопределённости выборки пользователей.",
                hypothesis="Автоматические наборы не хуже ручного, разности значимы при учёте бутстреп-интервалов.",
                config={"split": SPLIT, "days": DAYS, "sets": SETS, "extra": EXTRA, "n_boot": N_BOOT}, data_files=[D.EVENTS]) as exp:
    store = F.FeatureStore(YAMBDA_TAG, F.Engine(D.load_events()))
    metrics, packs = {}, {}
    for name in SETS:
        with exp.stage(f"оценка {name}"):
            names = sets.get_set(name.partition("@")[0])
            booster = lgb.Booster(model_str=model_path(name).read_text(encoding="utf-8"))
            if SPLIT == "test":
                T.log_test_access(name, "финальная оценка")
            mat = T.build_mat(store, DAYS, names, verbose=False)
            score = booster.predict(mat.X, num_iteration=booster.best_iteration, num_threads=NUM_THREADS)
            res, per_group, groups, pack = T.eval_mat(mat, score, boot=1)
            metrics[name], packs[name] = res, pack
            exp.record("metrics", metrics)
            print(f"{name:10s} ndcg@10={res['ndcg@10']:.4f}  ndcg@100={res['ndcg@100']:.4f}  recall@100={res['recall@100']:.4f}  "
                  f"auc={res['auc']:.4f}")
    for title, path in EXTRA.items():                  # например, модель над историей: per_group_<split>.npz из scripts/15
        z = np.load(path)
        per_group = {k: z[k] for k in z.files if k not in ("groups", "n_groups")}
        users = z["groups"] & ((1 << 32) - 1)
        packs[title] = M.user_level(z["groups"], per_group, users)
        metrics[title] = {k: float(v.mean()) for k, v in per_group.items()}
        exp.record("metrics", metrics)
    with exp.stage("bootstrap"):
        ci = {}
        reference = packs.get("manual")
        for name, (users, count, sums) in packs.items():
            ci[name] = {"abs": M.bootstrap_ci(count, sums, None, N_BOOT)}
            if reference is not None and name != "manual":
                assert np.array_equal(users, reference[0]), "множества пользователей не совпали"
                ci[name]["vs_manual"] = M.bootstrap_ci(count, sums, reference[2], N_BOOT)
                d = ci[name]["vs_manual"]["ndcg@10"]
                print(f"{name:10s} ΔNDCG@10 к ручному: {d['mean']:+.4f} [{d['lo']:+.4f}; {d['hi']:+.4f}]")
        exp.record("bootstrap_ci", ci)
