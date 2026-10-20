"""Временной протокол Yambda: якоря, окна меток и проверка, что метки обучения не пересекаются с более поздними частями.

Время - секунды от начала данных (в Yambda-50M timestamp измеряется в секундах с шагом 5).
Якорь - начало суток; признаки используют только события строго до якоря, метка берётся из окна после якоря.
"""
DAY = 86_400
GAP = 1_800            # пауза между якорем и окном метки, как в Global Temporal Split из статьи Yambda
HORIZON = DAY          # длина окна метки
TS_MAX = 26_000_000    # последний timestamp в Yambda-50M
LISTEN_PLUS = 50       # доля прослушивания (%), с которой прослушивание считается положительным

TRAIN_FIRST, TRAIN_LAST = 150, 283
VAL = list(range(285, 292))
TEST = list(range(293, 300))


def label_window(anchor_ts: int) -> tuple[int, int]:
    """Окно метки [начало, конец) для якоря."""
    return anchor_ts + GAP, anchor_ts + GAP + HORIZON


def train_anchors(step: int = 1, first: int = TRAIN_FIRST, last: int = TRAIN_LAST) -> list[int]:
    return list(range(first, last + 1, step))


def check_protocol(train_days: list[int], val_days: list[int] = VAL, test_days: list[int] = TEST) -> None:
    """Метки train должны заканчиваться до первого якоря val, метки val - до первого якоря test, всё - внутри данных."""
    for name, days in [("train", train_days), ("val", val_days), ("test", test_days)]:
        assert days == sorted(days) and len(set(days)) == len(days), f"{name}: сутки не отсортированы или повторяются"
        assert label_window(days[-1] * DAY)[1] <= TS_MAX, f"{name}: окно метки выходит за пределы данных"
    assert label_window(train_days[-1] * DAY)[1] <= val_days[0] * DAY, "метки train пересекаются с якорем val"
    assert label_window(val_days[-1] * DAY)[1] <= test_days[0] * DAY, "метки val пересекаются с якорем test"

# якоря обучения полных моделей: каждые 4 суток (34 якоря); для сравнения методов поиска используется меньшая песочница
FULL_TRAIN_DAYS = list(range(151, 284, 4))
