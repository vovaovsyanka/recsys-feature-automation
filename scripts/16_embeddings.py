"""E16. Векторные признаки: оценки ALS и LightGCN (граф «пользователь - трек - артист») как признаки для бустинга.

Для контрольных точек (каждые 7 суток) ALS и LightGCN обучаются только на событиях до точки; оценки пар (пользователь, кандидат)
для якорей, которым соответствует эта точка, сохраняются как внешние признаки ext:als.ui, ext:gcn.ui, ext:gcn.ua.
Число шагов LightGCN выбирается на val по качеству самой модели (стадия calibration). Затем признаки подключаются к бустингу
наборами H_als, H_gcn, H_both (src/recsys/features/sets.py) и сравниваются в E17 и E18.
Запуск: python scripts/16_embeddings.py
"""
import gc

import numpy as np
import polars as pl

from recsys.config import DATA, YAMBDA_TAG
from recsys.data import yambda as D
from recsys.evaluation import metrics as M
from recsys.models import embeddings as E
from recsys.protocol import FULL_TRAIN_DAYS, TEST, VAL
from recsys.tracking import Experiment

ANCHORS = FULL_TRAIN_DAYS + list(VAL) + list(TEST)
GCN_STEPS = [150, 300, 600]
OUT = DATA / "ext" / YAMBDA_TAG


class CheckpointModels:
    """ALS и LightGCN, обученные на событиях до контрольной точки."""

    def __init__(self, events: pl.DataFrame, checkpoint: int, gcn_steps: int):
        u, i, w, self.users, self.items, item_artist = E.edges_before(events, checkpoint)
        self.n_edges = len(w)
        self.als_users, self.als_items = E.train_als(u, i, w, len(self.users), len(self.items))
        self.gcn_users, self.gcn_items, self.gcn_artists, self.artist_row = E.train_lightgcn(
            u, i, w, len(self.users), len(self.items), item_artist, steps=gcn_steps)

    def als(self, rows: pl.DataFrame):
        return E.score_rows(rows, self.users, self.items, self.als_users, self.als_items)[0]

    def gcn(self, rows: pl.DataFrame):
        return E.score_rows(rows, self.users, self.items, self.gcn_users, self.gcn_items, self.gcn_artists, self.artist_row)


def load_rows(day: int) -> pl.DataFrame:
    return pl.read_parquet(DATA / "rows" / YAMBDA_TAG / f"rows_{day:03d}.parquet")


def standalone_quality(days: list[int], scorer) -> dict:
    """Качество одной лишь оценки (без бустинга) внутри кандидатов, среднее по якорям."""
    per_day = []
    for day in days:
        rows = load_rows(day)
        groups = pl.read_parquet(DATA / "rows" / YAMBDA_TAG / "groups" / f"groups_{day:03d}.parquet").select("uid", "n_pos")
        score = np.nan_to_num(scorer(rows), nan=-1e9)
        n_pos = rows.select("uid").join(groups, on="uid", how="left")["n_pos"].to_numpy()
        res, *_ = M.evaluate(np.full(len(score), day), rows["uid"].to_numpy(), score, rows["label"].to_numpy(), n_pos)
        per_day.append(res)
    return {k: float(np.mean([r[k] for r in per_day])) for k in ("ndcg@10", "ndcg@100", "auc")}


with Experiment("E16", "embeddings", "Векторные признаки: ALS и LightGCN по графу пользователь-трек-артист",
                goal="Проверить идею 2 (графовые модели): даёт ли обучаемое векторное представление сигнал, которого нет в ручных "
                     "агрегатах, и можно ли им заменить признаки пары.",
                hypothesis="Сами по себе оценки слабее признаков пары (повторы доминируют), но как дополнение к бустингу они помогают "
                           "на новых треках.",
                config={"checkpoints": E.CHECKPOINTS, "anchors": ANCHORS, "gcn_steps_grid": GCN_STEPS},
                data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        events = D.load_events()
    with exp.stage("calibration"):
        report, best_steps, best = {}, GCN_STEPS[0], -1.0
        for steps in GCN_STEPS:                       # контрольная точка 285 - последняя до val
            models = CheckpointModels(events, 285, steps)
            if steps == GCN_STEPS[0]:
                exp.record("graph_285", dict(users=len(models.users), items=len(models.items), edges=models.n_edges))
                report["als"] = standalone_quality(VAL, models.als)
                print("ALS", report["als"])
            report[f"gcn_steps{steps}"] = standalone_quality(VAL, lambda r: models.gcn(r)[0])
            print(f"LightGCN, шагов {steps}:", report[f"gcn_steps{steps}"])
            if report[f"gcn_steps{steps}"]["ndcg@10"] > best:
                best, best_steps = report[f"gcn_steps{steps}"]["ndcg@10"], steps
            models = None
            gc.collect()
        exp.record("calibration", report)
        exp.record("gcn_steps_selected", best_steps)
    with exp.stage("checkpoints"):
        for name in ("als", "gcn"):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        by_checkpoint: dict[int, list[int]] = {}
        for day in ANCHORS:
            by_checkpoint.setdefault(E.checkpoint_for(day), []).append(day)
        for checkpoint, days in sorted(by_checkpoint.items()):
            models = CheckpointModels(events, checkpoint, best_steps)
            for day in days:
                rows = load_rows(day)
                gcn_ui, gcn_ua = models.gcn(rows)
                np.save(OUT / "als" / f"ui_{day:03d}.npy", models.als(rows))
                np.save(OUT / "gcn" / f"ui_{day:03d}.npy", gcn_ui)
                np.save(OUT / "gcn" / f"ua_{day:03d}.npy", gcn_ua)
            print(f"  контрольная точка {checkpoint}: якорей {len(days)}, рёбер {models.n_edges}")
            models = None
            gc.collect()
