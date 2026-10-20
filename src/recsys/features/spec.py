"""Грамматика признаков.

Признак задаётся строкой  <ключ>.<канал>.<агрегат>.<окно>, например  UI.plus.cnt.7,  U.listen.mean.30,  I.plus.decay.h14.

Ключ (по чему агрегируем): U - пользователь, I - трек, UI - пара, UA - пользователь×артист, UB - пользователь×альбом,
    A - артист, B - альбом.
Канал (какие события): listen - любое прослушивание, plus - доля >= 50%, skip - < 20%, full - >= 90%, org / rec - органическое /
    из рекомендаций, plus_org, plus_rec, like, unlike, dislike, undislike.
Агрегат: cnt - число событий; mean - средняя доля прослушивания; sec - прослушано секунд; last / first - суток с последнего /
    первого события; decay - сумма с экспоненциальным затуханием (окно записывается полупериодом hN); nuniq - число различных партнёров.
Окно: число суток или all. Окно считается целыми сутками до момента прогноза: [a - w, a).

Производные признаки собираются из других: d:ratio(A|B), d:diff(A|B), d:urank(A) - ранг A среди кандидатов того же пользователя.
Признаки источников кандидатов (ROW_COLS) и внешние признаки (ext:<модель>.<имя>) берутся не из движка, а из файлов.
"""
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

KEYS = {"U": ["uid"], "I": ["item_id"], "UI": ["uid", "item_id"], "UA": ["uid", "artist"], "UB": ["uid", "album"],
        "A": ["artist"], "B": ["album"]}
CHANS = ["listen", "plus", "skip", "full", "org", "rec", "plus_org", "plus_rec", "like", "unlike", "dislike", "undislike"]
SR_CH = ["listen", "org", "rec"]                      # каналы, для которых есть средняя доля прослушивания
TMAX_CH = ["listen", "plus", "skip", "org", "rec", "like", "dislike"]    # каналы с признаком «давность»
TMIN_CH = ["listen", "plus", "like"]                  # каналы с признаком «возраст»
AGGS = ["cnt", "mean", "sec", "last", "first", "decay", "nuniq"]
PARTNER = {"U": "item_id", "UA": "item_id", "UB": "item_id", "I": "uid", "A": "uid", "B": "uid"}   # что считает nuniq

ROW_COLS = {"s1_rank", "s1_score", "s2_rank", "s2_score", "s3_rank", "s3_score", "s4_rank", "s4_score"}


def chan_flags() -> dict[str, pl.Expr]:
    """Условия, определяющие канал событий (et: 0 прослушивание, 1 лайк, 2 дизлайк, 3 снятие лайка, 4 снятие дизлайка)."""
    listen, ratio, organic = pl.col("et") == 0, pl.col("ratio"), pl.col("org") == 1
    return {"listen": listen, "plus": listen & (ratio >= 50), "skip": listen & (ratio < 20), "full": listen & (ratio >= 90),
            "org": listen & organic, "rec": listen & ~organic,
            "plus_org": listen & (ratio >= 50) & organic, "plus_rec": listen & (ratio >= 50) & ~organic,
            "like": pl.col("et") == 1, "unlike": pl.col("et") == 3, "dislike": pl.col("et") == 2,
            "undislike": pl.col("et") == 4}


@dataclass(frozen=True)
class Spec:
    key: str
    chan: str
    agg: str
    win: str           # 'all', число суток или hN (для decay)

    @property
    def name(self) -> str:
        return f"{self.key}.{self.chan}.{self.agg}.{self.win}"

    @staticmethod
    def parse(name: str) -> "Spec":
        key, chan, agg, win = name.split(".", 3)
        return Spec(key, chan, agg, win)

    def valid(self) -> bool:
        if self.key not in KEYS or self.chan not in CHANS or self.agg not in AGGS:
            return False
        if self.agg == "mean" and self.chan not in SR_CH:
            return False
        if self.agg == "sec" and self.chan != "listen":
            return False
        if self.agg == "last" and self.chan not in TMAX_CH:
            return False
        if self.agg == "first" and self.chan not in TMIN_CH:
            return False
        if self.agg == "nuniq" and self.key not in PARTNER:
            return False
        if self.agg == "decay":
            return self.win.startswith("h")
        if self.win == "all":
            return True
        try:
            return float(self.win) > 0
        except ValueError:
            return False

    @property
    def window_days(self) -> float | None:
        """Длина окна в сутках; None - вся история."""
        if self.agg == "decay" or self.win == "all":
            return None
        return float(self.win)

    @property
    def fill(self) -> float:
        """Значение при отсутствии событий: пропуск для средних и давности, ноль для счётчиков."""
        return np.nan if self.agg in ("mean", "last", "first") else 0.0


def is_derived(name: str) -> bool:
    return name.startswith("d:")


def parse_derived(name: str):
    m = re.fullmatch(r"d:(ratio|diff)\((.+)\|(.+)\)", name)
    if m:
        return m.group(1), [m.group(2), m.group(3)]
    m = re.fullmatch(r"d:urank\((.+)\)", name)
    if m:
        return "urank", [m.group(1)]
    raise ValueError(f"не распознан производный признак: {name}")


def base_deps(name: str) -> list[str]:
    """Базовые признаки движка, от которых зависит `name` (рекурсивно)."""
    if name in ROW_COLS or name.startswith("ext:"):
        return []
    if not is_derived(name):
        return [name]
    out = []
    for arg in parse_derived(name)[1]:
        out += base_deps(arg)
    return out
