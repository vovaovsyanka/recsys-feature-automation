"""Вычисление признаков по посуточным агрегатам.

Для каждого ключа один раз строится таблица «ключ × сутки → счётчики по всем каналам». Признак на момент `a` читает
только строки с day < a (и события с ts < a), поэтому будущее в признаки попасть не может; это проверяет
scripts/05_leakage_tests.py (замена событий после `a` мусором не меняет признаки).
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import polars as pl

from recsys.config import DATA
from recsys.features.spec import (CHANS, KEYS, PARTNER, SR_CH, TMAX_CH, TMIN_CH, Spec, chan_flags, is_derived,
                                  parse_derived)
from recsys.protocol import DAY


def events_id(ev: pl.DataFrame) -> str:
    """Короткий отпечаток таблицы событий - им подписан каталог кэша посуточных таблиц."""
    key = str((ev.height, int(ev["ts"].sum()), int(ev["uid"].sum()), int(ev["item_id"].sum())))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def build_daily(ev: pl.DataFrame, key: str) -> pl.DataFrame:
    kc = KEYS[key]
    flags = chan_flags()
    d = ev.drop_nulls(kc).with_columns((pl.col("ts") // DAY).cast(pl.UInt16).alias("day"))
    d = d.with_columns([e.alias(f"f_{c}") for c, e in flags.items()])
    aggs = [pl.col(f"f_{c}").sum().cast(pl.UInt32).alias(f"cnt_{c}") for c in CHANS]
    aggs += [pl.col("ratio").filter(pl.col(f"f_{c}")).sum().alias(f"sr_{c}") for c in SR_CH]
    aggs += [(pl.col("ratio") * pl.col("tlen") / 100.0).filter(pl.col("f_listen")).sum().cast(pl.Float32).alias("sec_listen")]
    aggs += [pl.col("ts").filter(pl.col(f"f_{c}")).max().alias(f"tmax_{c}") for c in TMAX_CH]
    aggs += [pl.col("ts").filter(pl.col(f"f_{c}")).min().alias(f"tmin_{c}") for c in TMIN_CH]
    return d.group_by(kc + ["day"]).agg(aggs).sort("day")


class Engine:
    """Считает столбцы признаков для строк одного якоря."""

    def __init__(self, ev: pl.DataFrame, cache_dir: Path | None = None):
        self.ev = ev
        self.ts = ev["ts"].to_numpy()
        self.cache = (cache_dir or DATA / "daily") / events_id(ev)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._daily: dict[str, tuple[pl.DataFrame, np.ndarray]] = {}
        self._flags = chan_flags()

    def daily(self, key: str):
        if key not in self._daily:
            path = self.cache / f"{key}.parquet"
            if path.exists():
                df = pl.read_parquet(path)
            else:
                t = time.time()
                df = build_daily(self.ev, key)
                df.write_parquet(path, compression="zstd")
                print(f"  посуточная таблица {key}: {df.height} строк, {time.time() - t:.0f} с")
            self._daily[key] = (df, df["day"].to_numpy())
        return self._daily[key]

    def compute_base(self, specs: list[Spec], rows: pl.DataFrame, a_day: int) -> pl.DataFrame:
        """Столбцы базовых признаков в порядке строк `rows` для момента прогноза `a_day` (начало суток)."""
        out: dict[str, np.ndarray] = {}
        by_key: dict[str, list[Spec]] = {}
        for s in specs:
            assert s.valid(), f"невалидная спецификация {s.name}"
            by_key.setdefault(s.key, []).append(s)
        for key, key_specs in by_key.items():
            kc = KEYS[key]
            keys_df = rows.select(kc).drop_nulls().unique()
            from_daily = [s for s in key_specs if s.agg != "nuniq"]
            distinct = [s for s in key_specs if s.agg == "nuniq"]
            res = rows.select(kc)
            if from_daily:
                res = res.join(self._daily_aggregates(key, from_daily, keys_df, a_day), on=kc, how="left")
            if distinct:
                for s, g in self._distinct_counts(key, distinct, keys_df, a_day):
                    res = res.join(g, on=kc, how="left")
            assert res.height == rows.height, "соединение изменило число строк"
            for s in key_specs:
                col = res[s.name].to_numpy().astype(np.float32, copy=True)
                out[s.name] = np.nan_to_num(col, nan=0.0) if s.fill == 0.0 else col
        return pl.DataFrame(out)

    def _daily_aggregates(self, key: str, specs: list[Spec], keys_df: pl.DataFrame, a_day: int) -> pl.DataFrame:
        df, days = self.daily(key)
        windows = [s.window_days for s in specs]
        lo = 0 if any(w is None for w in windows) else max(0, int(a_day - max(windows)))
        i0, i1 = int(np.searchsorted(days, lo, "left")), int(np.searchsorted(days, a_day, "left"))   # только day < a_day
        part = df.slice(i0, i1 - i0).join(keys_df, on=KEYS[key], how="semi")
        exprs = []
        for s in specs:
            w = s.window_days
            in_window = (pl.col("day") >= max(0.0, a_day - w)) if w is not None else pl.lit(True)
            c = s.chan
            if s.agg == "cnt":
                e = pl.col(f"cnt_{c}").filter(in_window).sum()
            elif s.agg == "mean":
                e = pl.col(f"sr_{c}").filter(in_window).sum() / pl.col(f"cnt_{c}").filter(in_window).sum()
            elif s.agg == "sec":
                e = pl.col("sec_listen").filter(in_window).sum()
            elif s.agg == "last":
                e = (a_day * DAY - pl.col(f"tmax_{c}").filter(in_window).max().cast(pl.Float64)) / DAY
            elif s.agg == "first":
                e = (a_day * DAY - pl.col(f"tmin_{c}").filter(in_window).min().cast(pl.Float64)) / DAY
            else:    # decay: полупериод в сутках, вес сутки d - 0.5^((a - d - 0.5) / h)
                h = float(s.win[1:])
                weight = 0.5 ** ((a_day - pl.col("day").cast(pl.Float64) - 0.5) / h)
                e = (pl.col(f"cnt_{c}").cast(pl.Float64) * weight).sum()
            exprs.append(e.cast(pl.Float32).alias(s.name))
        return part.group_by(KEYS[key]).agg(exprs)

    def _distinct_counts(self, key: str, specs: list[Spec], keys_df: pl.DataFrame, a_day: int):
        """nuniq нельзя получить из посуточных счётчиков, поэтому читаем события напрямую (по-прежнему только ts < a)."""
        kc = KEYS[key]
        windows = [s.window_days for s in specs]
        lo_ts = 0 if any(w is None for w in windows) else int(max(0, a_day - max(windows)) * DAY)
        i0, i1 = int(np.searchsorted(self.ts, lo_ts, "left")), int(np.searchsorted(self.ts, a_day * DAY, "left"))
        events = self.ev.slice(i0, i1 - i0).join(keys_df, on=kc, how="semi")
        for s in specs:
            w = s.window_days
            cond = self._flags[s.chan]
            if w is not None:
                cond = cond & (pl.col("ts") >= int(max(0.0, a_day - w) * DAY))
            g = events.filter(cond).group_by(kc).agg(pl.col(PARTNER[key]).n_unique().cast(pl.Float32).alias(s.name))
            yield s, g


def apply_derived(name: str, cols: dict[str, np.ndarray], uid: np.ndarray) -> np.ndarray:
    op, args = parse_derived(name)
    if op == "urank":
        x = pl.DataFrame({"u": uid, "x": cols[args[0]]})
        rank = x.select(pl.col("x").rank("average").over("u") / pl.len().over("u")).to_series().to_numpy()
        return rank.astype(np.float32)
    a, b = cols[args[0]].astype(np.float64), cols[args[1]].astype(np.float64)
    if op == "diff":
        return (a - b).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan).astype(np.float32)


def derived_value(name: str, base_cols: dict[str, np.ndarray], cache: dict, uid: np.ndarray) -> np.ndarray:
    """Значение признака; производные могут ссылаться на другие производные."""
    if not is_derived(name):
        return base_cols[name]
    if name not in cache:
        args = {a: derived_value(a, base_cols, cache, uid) for a in parse_derived(name)[1]}
        cache[name] = apply_derived(name, args, uid)
    return cache[name]
