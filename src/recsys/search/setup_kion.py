"""Общие настройки экспериментов на KION."""
from recsys.candidates import kion as P
from recsys.features import manual_kion

ROWS_USER_FRAC = 0.10            # доля пользователей в наборе строк (вся их история сохранена)
TRAIN_DAYS = P.train_anchors(3)  # 28 обучающих якорей
VAL_DAY, TEST_DAY = P.A_VAL, P.A_TEST
NEG_KEEP = 0.2                   # доля негативов в обучении полных моделей

# песочница поиска признаков
PROXY_USER_FRAC = 0.3
PROXY_TRAIN = [60, 72, 84, 96, 108, 120, 132, 141]
PROXY_NEG_KEEP = 0.25
K_AUGMENT = 40
CAND = ["s1_rank", "s1_score", "s2_rank", "s2_score", "s3_rank", "s3_score"]
MAN = manual_kion.manual_all()
K_REPLACE = len([n for n in MAN if n not in CAND])
