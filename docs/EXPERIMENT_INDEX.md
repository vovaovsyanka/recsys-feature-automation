# Индекс экспериментов

Сформирован 2026-10-21 00:03 скриптом `scripts/99_make_index.py` из `experiments/*/results.json`.
Время - реальное (wall-clock); CPU суммируется по потокам. Тайминги осмысленны, только если эксперимент шёл без параллельной нагрузки.

| эксперимент | название | статус | длительность | CPU, с | пик RAM, ГБ | запуск (UTC) |
|---|---|---|---|---|---|---|
| [E01_audit_yambda](../experiments/E01_audit_yambda/README.md) | Аудит данных Yambda-50M | ok | 19 с | 59 | 5.72 | 2026-10-20 15:35:01+00:00 |
| [E02_audit_kion](../experiments/E02_audit_kion/README.md) | Аудит данных KION | ok | 1 с | 9 | 1.47 | 2026-10-20 15:37:31+00:00 |
| [E03_candidate_sweep](../experiments/E03_candidate_sweep/README.md) | Подбор кандидатной стадии (recall vs размер) | ok | 3.5 мин | 1484 | 7.82 | 2026-10-20 15:44:01+00:00 |
| [E04_build_rows](../experiments/E04_build_rows/README.md) | Построение кандидатных строк и меток для всех якорей | ok | 9.0 мин | 3960 | 5.12 | 2026-10-20 15:48:18+00:00 |
| [E05_leakage_tests](../experiments/E05_leakage_tests/README.md) | Проверка на утечки: замена будущего мусором и «канарейка» | ok | 32 с | 262 | 4.88 | 2026-10-20 20:44:15+00:00 |
| [E06_heuristics](../experiments/E06_heuristics/README.md) | Эвристические ранкеры на общих кандидатах (val) | ok | 1.2 мин | 77 | 3.0 | 2026-10-20 20:45:07+00:00 |
| [E07_manual_features](../experiments/E07_manual_features/README.md) | Расчёт ручных признаков (123) и их стоимость | ok | 8.5 мин | 3857 | 17.87 | 2026-10-20 16:01:27+00:00 |
| [E08_baseline_gbdt](../experiments/E08_baseline_gbdt/README.md) | LightGBM на ручных признаках (базлайн), val | ok | 41.9 мин | 25377 | 16.85 | 2026-10-20 16:32:04+00:00 |
| [E10_feature_pool](../experiments/E10_feature_pool/README.md) | Расчёт пула грамматики (3.6 тыс. признаков) на подвыборке | ok | 16.9 мин | 7452 | 19.55 | 2026-10-20 17:14:31+00:00 |
| [E15_sequence_base](../experiments/E15_sequence_base/README.md) | Target-attention модель над историей (val) | ok | 15.8 мин | 853 | 4.28 | 2026-10-20 17:50:56+00:00 |
| [E15_sequence_L512](../experiments/E15_sequence_L512/README.md) | Target-attention модель над историей (val) | ok | 57.4 мин | 2772 | 5.96 | 2026-10-20 18:25:34+00:00 |
| [E15_sequence_listwise](../experiments/E15_sequence_listwise/README.md) | Target-attention модель над историей (val) | ok | 16.4 мин | 876 | 5.33 | 2026-10-20 18:09:00+00:00 |
| [E30_kion_rows](../experiments/E30_kion_rows/README.md) | KION: кандидаты и метки (10% пользователей) | ok | 9.5 мин | 654 | 1.05 | 2026-10-20 18:15:01+00:00 |

Суммарная длительность: 3.02 ч (13 экспериментов).
