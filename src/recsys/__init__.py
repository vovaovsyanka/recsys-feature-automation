"""Автоматизация построения признаков в рекомендательных системах (Yambda, KION)."""
import sys

# кириллица в консоли Windows и при перенаправлении вывода в файл
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
