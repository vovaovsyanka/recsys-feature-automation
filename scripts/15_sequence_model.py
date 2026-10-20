"""E15. Модель над историей событий: внимание к истории, зависящее от кандидата, с обучаемыми временными ядрами.

Модель видит только сырые события пользователя и кандидата, без инженерных агрегатов; строки, кандидаты и метки те же, что у
бустинга. Варианты запускаются отдельными экспериментами (--variant), чтобы эффект каждого изменения был виден отдельно:
  base      L=256, потеря BCE                          listwise  потеря по группе (softmax) вместо BCE
  L512      длина истории 512                           extra     на вход добавлены признаки источников кандидатов
  all       listwise + L512 + extra
С флагом --test лучшая по val эпоха один раз оценивается на test.
Запуск: python scripts/15_sequence_model.py --variant=L512 --L=512 [--loss=listwise] [--extra=1] [--epochs=3] [--test]
"""
import sys

import numpy as np
import torch

from recsys import features as F
from recsys.config import YAMBDA_TAG
from recsys.data import yambda as D
from recsys.evaluation import metrics as M
from recsys.models import gbdt as T
from recsys.models import sequence as S
from recsys.protocol import FULL_TRAIN_DAYS, TEST, VAL
from recsys.tracking import Experiment


def arg(name: str, default, cast=str):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


VARIANT = arg("variant", "base")
DO_TEST = "--test" in sys.argv
CFG = dict(L=arg("L", 256, int), d=32, heads=4, n_kern=8, batch_groups=16, lr=2e-3, epochs=arg("epochs", 3, int),
           neg_keep=0.4, pos_weight=2.0, seed=0, loss=arg("loss", "bce"), extra=arg("extra", "0") == "1")
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


def evaluate(model, hist, data):
    """Метрики на группах `data` и метрики по группам (для бутстрепа)."""
    _, scores = S.run_epoch(model, hist, data, np.arange(len(data)), 24, CFG["L"], collect=True, loss_type=CFG["loss"])
    score = np.concatenate(scores)
    sizes = data.g1 - data.g0
    anchor, uid, n_pos = (np.repeat(a, sizes) for a in (data.day, data.uid, data.npos))
    res, per_group, groups, _ = M.evaluate(anchor, uid, score, data.y, n_pos, p=1 / (1 + np.exp(-score)))
    return res, per_group, groups


with Experiment("E15", f"sequence_{VARIANT}", f"Модель над историей событий: вариант {VARIANT}",
                goal="Проверить идею 3: модель, которая видит только сырую историю и сама подбирает окна (обучаемые ядра), "
                     "на тех же строках и кандидатах, что и бустинг.",
                hypothesis="Модель приблизится к бустингу на ручных признаках без единого инженерного агрегата; "
                           "более длинная история и признаки источников кандидатов улучшат её.",
                config={"tag": YAMBDA_TAG, "train_days": FULL_TRAIN_DAYS, "test": DO_TEST, **CFG}, data_files=[D.EVENTS]) as exp:
    with exp.stage("load_history"):
        events = D.load_events()
        hist = S.build_hist(events)
        store = F.FeatureStore(YAMBDA_TAG, F.Engine(events))
        del events
    with exp.stage("load_groups"):
        train = S.GroupData(store, FULL_TRAIN_DAYS, neg_keep=CFG["neg_keep"], seed=CFG["seed"], extra=CFG["extra"])
        val = S.GroupData(store, list(VAL), extra=CFG["extra"])
        print(f"групп: train {len(train)}, val {len(val)}")
    model = S.SeqRanker(hist.n_item, int(hist.art_vocab.max()) + 1, CFG["d"], CFG["heads"], CFG["n_kern"],
                        n_extra=train.n_extra).cuda()
    exp.record("parameters_millions", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-6)
    scaler = torch.amp.GradScaler()
    by_epoch, best = [], (-1.0, None)
    for epoch in range(CFG["epochs"]):
        with exp.stage(f"эпоха {epoch}"):
            order = np.random.default_rng(epoch).permutation(len(train))
            loss, _ = S.run_epoch(model, hist, train, order, CFG["batch_groups"], CFG["L"], opt=opt, scaler=scaler,
                                  pos_w=CFG["pos_weight"], loss_type=CFG["loss"])
        with exp.stage(f"val после эпохи {epoch}"):
            res, per_group, groups = evaluate(model, hist, val)
            res.update(epoch=epoch, train_loss=loss, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20)
            by_epoch.append(res)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
            if res["ndcg@10"] > best[0]:
                best = (res["ndcg@10"], {k: v.detach().clone() for k, v in model.state_dict().items()})
                np.savez_compressed(exp.dir / "per_group_val.npz", groups=groups, **per_group)
        exp.record("val_by_epoch", by_epoch)
        for g in opt.param_groups:
            g["lr"] *= 0.5                                  # затухание скорости обучения по эпохам
    model.load_state_dict(best[1])
    exp.record("val_metrics_best_epoch", max(by_epoch, key=lambda r: r["ndcg@10"]))
    if DO_TEST:
        with exp.stage("test"):
            T.log_test_access(f"sequence_{VARIANT}", "финальная оценка")
            test = S.GroupData(store, list(TEST), extra=CFG["extra"])
            res, per_group, groups = evaluate(model, hist, test)
            np.savez_compressed(exp.dir / "per_group_test.npz", groups=groups, **per_group)
            exp.record("test_metrics", res)
            print("test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
