"""E03. Подбор кандидатной стадии: как recall кандидатов зависит от их числа и состава.

Подбор идёт только по train- и val-якорям, test не используется. Выбранная конфигурация фиксируется в scripts/04_build_rows.py.
Запуск: python scripts/03_candidate_sweep.py
"""

from recsys.candidates import yambda as C
from recsys.data import yambda as D
from recsys.protocol import DAY
from recsys.tracking import Experiment

TRAIN_ANCHORS = [160, 200, 240, 280]
VAL_ANCHORS = [288]
Cfg = C.CandidateConfig
CONFIGS = {
    "base":                Cfg(s1_n=100, s2_n=50),
    "s1_200":              Cfg(s1_n=200, s2_n=50),
    "s1_300":              Cfg(s1_n=300, s2_n=50),
    "s1_200_hl7":          Cfg(s1_n=200, s2_n=50, s1_halflife_days=7),
    "s1_200_hl30":         Cfg(s1_n=200, s2_n=50, s1_halflife_days=30),
    "s1_200_s2_100":       Cfg(s1_n=200, s2_n=100),
    "s1_150_s2_100_s3big": Cfg(s1_n=150, s2_n=100, s3_artists=8, s3_per_artist=10),
    "no_s3":               Cfg(s1_n=200, s2_n=50, s3_artists=0),
    "no_s4":               Cfg(s1_n=200, s2_n=50, s4_n=0),
    "no_s2":               Cfg(s1_n=200, s2_n=0),
    "only_s1_200":         Cfg(s1_n=200, s2_n=0, s3_artists=0, s4_n=0),
}

with Experiment("E03", "candidate_sweep", "Подбор кандидатной стадии: recall против числа кандидатов",
                goal="Выбрать состав кандидатов с лучшим recall при бюджете 200-270 кандидатов на пару (якорь, пользователь). "
                     "Критерий - средний recall на train-якорях, val служит подтверждением.",
                hypothesis="Повторы доминируют (65-73% позитивов), поэтому основной вклад даёт затухающий личный счётчик; "
                           "артисты добавляют новые треки, хиты недели вносят мало.",
                config={"train_anchors": TRAIN_ANCHORS, "val_anchors": VAL_ANCHORS,
                        "configs": {k: vars(v) for k, v in CONFIGS.items()}},
                data_files=[D.EVENTS]) as exp:
    with exp.stage("load"):
        events, items = D.load_events(), D.load_items()
        ts = events["ts"].to_numpy()
    results = {}
    for name, cfg in CONFIGS.items():
        with exp.stage(f"config: {name}"):
            per_split = {"train": [], "val": []}
            for split, days in [("train", TRAIN_ANCHORS), ("val", VAL_ANCHORS)]:
                for day in days:
                    rows, groups = C.build_anchor(events, ts, items, day * DAY, cfg)
                    per_split[split].append(dict(users=groups.height, rows=rows.height, pos=int(groups["n_pos"].sum()),
                                                 pos_in=int(groups["n_pos_in_cand"].sum())))
            results[name] = {
                split: dict(recall=sum(r["pos_in"] for r in rs) / sum(r["pos"] for r in rs),
                            candidates_per_user=sum(r["rows"] for r in rs) / sum(r["users"] for r in rs))
                for split, rs in per_split.items()}
            print(name, results[name])
    exp.record("by_config", results)
