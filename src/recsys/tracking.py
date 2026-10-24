"""Оформление экспериментов: каждый запуск пишет в experiments/<id>_<название>/ одинаковый набор файлов.

    config.json   параметры запуска
    log.txt       полный лог
    results.json  метрики, время по стадиям, CPU и память
    env.json      версии библиотек, железо, версия кода, отпечатки данных
    README.md     краткая сводка: цель, гипотеза, длительность, стадии

Время измеряется монотонными часами; стадии внутри эксперимента - `with exp.stage("имя"):`.
"""
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil

from recsys.config import EXPERIMENTS, ROOT


def _sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            if limit is not None and f.tell() >= limit:
                break
    return h.hexdigest()


def code_version() -> str:
    """Короткий хэш коммита (с пометкой +dirty при незакоммиченных правках); без git - хэш исходников."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        h = hashlib.sha256()
        for p in sorted((ROOT / "src").rglob("*.py")):
            h.update(p.read_bytes())
        return h.hexdigest()[:12]


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["polars", "numpy", "scipy", "sklearn", "lightgbm", "torch", "implicit"]:
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except ImportError:
            pass
    return out


def _hardware() -> dict:
    hw = {"cpu": platform.processor(), "cores_physical": psutil.cpu_count(logical=False),
          "cores_logical": psutil.cpu_count(logical=True), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    return hw


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class Experiment:
    def __init__(self, exp_id: str, name: str, title: str, goal: str, hypothesis: str, config: dict,
                 data_files: list[Path] | None = None):
        self.id, self.name, self.title = exp_id, name, title
        self.goal, self.hypothesis, self.config = goal, hypothesis, config
        self.data_files = data_files or []
        self.dir = EXPERIMENTS / f"{exp_id}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}
        self.stages: list[dict] = []
        self._proc = psutil.Process()
        self._peak_rss = 0

    def __enter__(self):
        self._log = open(self.dir / "log.txt", "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log)
        self.t0 = time.perf_counter()
        self.cpu0 = sum(self._proc.cpu_times()[:2])
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        print(f"[{self.id}] {self.title}\n  старт (UTC): {self.started}\n  версия кода: {code_version()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - self.t0
        cpu = sum(self._proc.cpu_times()[:2]) - self.cpu0
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        if exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)
        meta = {"id": self.id, "name": self.name, "title": self.title, "status": status,
                "started_utc": self.started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_seconds": round(wall, 2), "cpu_seconds_process": round(cpu, 2),
                "peak_rss_gb": round(max(self._peak_rss, self._proc.memory_info().rss) / 2**30, 2),
                "stages": self.stages}
        (self.dir / "results.json").write_text(
            json.dumps({"meta": meta, "results": self.results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        env = {"versions": _versions(), "hardware": _hardware(), "code_version": code_version(),
               "data_sha256_first64MB": {str(p.relative_to(ROOT)): _sha256(p, 64 << 20)
                                          for p in self.data_files if p.exists()}}
        (self.dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_readme(meta)
        print(f"[{self.id}] {status}; {wall:.1f} с, CPU {cpu:.1f} с, пик RSS {meta['peak_rss_gb']} ГБ")
        sys.stdout = self._stdout
        self._log.close()
        return False

    @contextlib.contextmanager
    def stage(self, name: str):
        t, c = time.perf_counter(), sum(self._proc.cpu_times()[:2])
        print(f"[{self.id}] -> {name}")
        try:
            yield
        finally:
            rss = self._proc.memory_info().rss
            self._peak_rss = max(self._peak_rss, rss)
            rec = {"stage": name, "wall_seconds": round(time.perf_counter() - t, 2),
                   "cpu_seconds": round(sum(self._proc.cpu_times()[:2]) - c, 2), "rss_gb_after": round(rss / 2**30, 2)}
            self.stages.append(rec)
            print(f"[{self.id}] <- {name}: {rec['wall_seconds']} с")

    def record(self, key: str, value):
        self.results[key] = value

    def _write_readme(self, meta: dict):
        lines = [f"# {self.id} - {self.title}", "",
                 f"**Статус:** {meta['status']}  ",
                 f"**Длительность:** {meta['wall_seconds']} с ({meta['wall_seconds'] / 60:.1f} мин); "
                 f"CPU процесса {meta['cpu_seconds_process']} с; пик RSS {meta['peak_rss_gb']} ГБ  ",
                 f"**Запуск:** {meta['started_utc']} - {meta['finished_utc']} (UTC)  ", "",
                 "## Цель", self.goal, "", "## Гипотеза", self.hypothesis, "",
                 "## Стадии", "", "| стадия | wall, с | CPU, с | RSS после, ГБ |", "|---|---|---|---|"]
        lines += [f"| {s['stage']} | {s['wall_seconds']} | {s['cpu_seconds']} | {s['rss_gb_after']} |" for s in meta["stages"]]
        lines += ["", "Метрики и параметры: `results.json`, `config.json`; окружение: `env.json`; интерпретация - в [README.md](../../README.md).", ""]
        (self.dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
