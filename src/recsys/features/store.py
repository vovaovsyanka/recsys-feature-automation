"""Кэш признаков по якорям.

Строки (якорь, пользователь, кандидат) фиксируются один раз (scripts/04_build_rows.py); хранилище считает недостающие
признаки пакетами и сохраняет их рядом, выровненными по строкам якоря. Любой набор признаков затем собирается из кэша.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from recsys.config import DATA
from recsys.features.engine import Engine, derived_value
from recsys.features.spec import ROW_COLS, Spec, base_deps, is_derived, parse_derived


def user_mask(uid: np.ndarray, frac: float) -> np.ndarray:
    """Детерминированная подвыборка пользователей по хэшу id: одна и та же для всех якорей и методов."""
    if frac >= 1.0:
        return np.ones(len(uid), dtype=bool)
    return ((uid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(10007)) < int(frac * 10007)


class FeatureStore:
    def __init__(self, cand_tag: str, engine: Engine, root: Path | None = None, user_frac: float = 1.0):
        self.engine = engine
        self.tag = cand_tag
        self.user_frac = user_frac
        suffix = "" if user_frac >= 1.0 else f"_u{int(user_frac * 100)}"
        self.root = (root or DATA / "fstore") / (cand_tag + suffix)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows_dir = DATA / "rows" / cand_tag
        self._rows: dict[int, pl.DataFrame] = {}

    def rows(self, day: int) -> pl.DataFrame:
        if day not in self._rows:
            r = pl.read_parquet(self.rows_dir / f"rows_{day:03d}.parquet")
            if self.user_frac < 1.0:
                r = r.filter(pl.Series(user_mask(r["uid"].to_numpy(), self.user_frac)))
            self._rows[day] = r
        return self._rows[day]

    def drop_rows_cache(self):
        self._rows.clear()

    def _index(self, day: int) -> dict[str, str]:
        p = self.root / f"d{day:03d}" / "index.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_index(self, day: int, index: dict):
        d = self.root / f"d{day:03d}"
        d.mkdir(exist_ok=True)
        (d / "index.json").write_text(json.dumps(index))

    def ensure(self, day: int, names: list[str], batch: int = 300) -> float:
        """Считает и кэширует недостающие базовые признаки. Возвращает затраченные секунды."""
        base = sorted({b for n in names for b in base_deps(n)})
        index = self._index(day)
        missing = [b for b in base if b not in index]
        if not missing:
            return 0.0
        t0 = time.time()
        rows = self.rows(day)
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            df = self.engine.compute_base([Spec.parse(b) for b in part], rows, day)
            fname = "b_" + hashlib.sha1("|".join(part).encode()).hexdigest()[:10] + ".parquet"
            (self.root / f"d{day:03d}").mkdir(exist_ok=True)
            df.write_parquet(self.root / f"d{day:03d}" / fname, compression="zstd")
            for b in part:
                index[b] = fname
            self._save_index(day, index)
        return time.time() - t0

    def _external(self, day: int, name: str, n_rows: int) -> np.ndarray:
        """Внешний признак ext:<модель>.<имя> - массив, выровненный со строками якоря (см. scripts/16_embeddings.py)."""
        model, col = name[4:].split(".", 1)
        arr = np.load(DATA / "ext" / self.tag / model / f"{col}_{day:03d}.npy").astype(np.float32)
        if self.user_frac < 1.0:       # массив хранится для всех строк якоря, поэтому применяем ту же маску пользователей
            full_uid = pl.read_parquet(self.rows_dir / f"rows_{day:03d}.parquet", columns=["uid"])["uid"].to_numpy()
            arr = arr[user_mask(full_uid, self.user_frac)]
        assert len(arr) == n_rows, "внешний признак не выровнен со строками якоря"
        return arr

    def get(self, day: int, names: list[str]) -> pl.DataFrame:
        """Признаки якоря (float32) в порядке `names`."""
        self.ensure(day, names)
        index = self._index(day)
        files: dict[str, list[str]] = {}
        for b in sorted({b for n in names for b in base_deps(n)}):
            files.setdefault(index[b], []).append(b)
        cols: dict[str, np.ndarray] = {}
        for fname, bs in files.items():
            part = pl.read_parquet(self.root / f"d{day:03d}" / fname, columns=bs)
            for b in bs:
                cols[b] = part[b].to_numpy()
        rows = self.rows(day)
        for rc in ROW_COLS:
            if rc in rows.columns:
                cols[rc] = rows[rc].to_numpy().astype(np.float32)
        for n in names:
            args = parse_derived(n)[1] if is_derived(n) else [n]
            for a in args:
                if a.startswith("ext:") and a not in cols:
                    cols[a] = self._external(day, a, rows.height)
        cache: dict = {}
        uid = rows["uid"].to_numpy()
        return pl.DataFrame({n: derived_value(n, cols, cache, uid) for n in names})
