"""Пути проекта и общие настройки вычислений."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"

# Число потоков для LightGBM/NumPy-расчётов. По умолчанию оставляем запас, чтобы компьютер не зависал.
NUM_THREADS = int(os.environ.get("RECSYS_THREADS", 8))

# Имена наборов строк (кандидатов) в data/rows: c268 - Yambda (около 265 кандидатов на пару), kion_c - KION
YAMBDA_TAG = "c268"
KION_TAG = "kion_c"
