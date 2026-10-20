"""Собирает docs/EXPERIMENT_INDEX.md из experiments/*/results.json: статус, длительность, ресурсы каждого эксперимента.

Запуск: python scripts/99_make_index.py
"""
import json
from datetime import datetime

from recsys.config import EXPERIMENTS, ROOT


def fmt(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} ч"
    return f"{seconds / 60:.1f} мин" if seconds >= 60 else f"{seconds:.0f} с"


rows = []
for d in sorted(EXPERIMENTS.iterdir()):
    path = d / "results.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        rows.append((d.name, meta))

lines = ["# Индекс экспериментов", "",
         f"Сформирован {datetime.now():%Y-%m-%d %H:%M} скриптом `scripts/99_make_index.py` из `experiments/*/results.json`.",
         "Время - реальное (wall-clock); CPU суммируется по потокам. Тайминги осмысленны, только если эксперимент шёл без "
         "параллельной нагрузки.", "",
         "| эксперимент | название | статус | длительность | CPU, с | пик RAM, ГБ | запуск (UTC) |", "|---|---|---|---|---|---|---|"]
for name, m in rows:
    status = "ok" if m["status"] == "ok" else "ОШИБКА"
    lines.append(f"| [{name}](../experiments/{name}/README.md) | {m['title']} | {status} | {fmt(m['wall_seconds'])} | "
                 f"{m['cpu_seconds_process']:.0f} | {m['peak_rss_gb']} | {m['started_utc'].replace('T', ' ')} |")
lines += ["", f"Суммарная длительность: {fmt(sum(m['wall_seconds'] for _, m in rows))} ({len(rows)} экспериментов)."]
(ROOT / "docs" / "EXPERIMENT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
