# Индекс экспериментов

Сформирован 2026-10-24 12:03 скриптом `scripts/99_make_index.py` из `experiments/*/results.json`.
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
| [E09_ablation_windows](../experiments/E09_ablation_windows/README.md) | Аблация групп признаков и чувствительность к выбору окон | ok | 9.2 мин | 3503 | 3.18 | 2026-10-23 00:42:16+00:00 |
| [E10_feature_pool](../experiments/E10_feature_pool/README.md) | Расчёт пула грамматики (3.6 тыс. признаков) на подвыборке | ok | 16.9 мин | 7452 | 19.55 | 2026-10-20 17:14:31+00:00 |
| [E11_search_random_genselect](../experiments/E11_search_random_genselect/README.md) | Поиск признаков: случайный выбор (A1) и генерация пула с отбором (A2) | ok | 16.4 мин | 5844 | 3.53 | 2026-10-23 00:51:36+00:00 |
| [E12_search_openfe](../experiments/E12_search_openfe/README.md) | Поиск признаков: OpenFE | ok | 1.04 ч | 25273 | 3.5 | 2026-10-23 06:59:16+00:00 |
| [E13_learned_windows](../experiments/E13_learned_windows/README.md) | Обучаемые окна: полупериоды затухающих счётчиков | ok | 4.5 мин | 1773 | 3.35 | 2026-10-23 01:08:08+00:00 |
| [E14_search_llm_agent](../experiments/E14_search_llm_agent/README.md) | Поиск признаков: ИИ-агент (языковая модель) | ok | 12.9 мин | 4278 | 8.32 | 2026-10-23 08:01:53+00:00 |
| [E15_sequence_base](../experiments/E15_sequence_base/README.md) | Target-attention модель над историей (val) | ok | 15.8 мин | 853 | 4.28 | 2026-10-20 17:50:56+00:00 |
| [E15_sequence_extra](../experiments/E15_sequence_extra/README.md) | Модель над историей событий: вариант extra | ok | 9.8 мин | 595 | 5.4 | 2026-10-23 08:14:59+00:00 |
| [E15_sequence_final](../experiments/E15_sequence_final/README.md) | Модель над историей событий: вариант final | ok | 11.0 мин | 675 | 5.64 | 2026-10-23 08:24:50+00:00 |
| [E15_sequence_L512](../experiments/E15_sequence_L512/README.md) | Target-attention модель над историей (val) | ok | 57.4 мин | 2772 | 5.96 | 2026-10-20 18:25:34+00:00 |
| [E15_sequence_listwise](../experiments/E15_sequence_listwise/README.md) | Target-attention модель над историей (val) | ok | 16.4 мин | 876 | 5.33 | 2026-10-20 18:09:00+00:00 |
| [E16_embeddings](../experiments/E16_embeddings/README.md) | Векторные признаки: ALS и LightGCN по графу пользователь-трек-артист | ok | 1.71 ч | 8907 | 4.41 | 2026-10-23 08:35:57+00:00 |
| [E17_final_A1_A](../experiments/E17_final_A1_A/README.md) | Полная модель LightGBM: набор A1_A (183 признаков), val | ok | 35.0 мин | 15911 | 35.21 | 2026-10-23 02:32:08+00:00 |
| [E17_final_A1_B](../experiments/E17_final_A1_B/README.md) | Полная модель LightGBM: набор A1_B (123 признаков), val | ok | 32.4 мин | 15109 | 31.47 | 2026-10-23 01:59:36+00:00 |
| [E17_final_A2_A](../experiments/E17_final_A2_A/README.md) | Полная модель LightGBM: набор A2_A (183 признаков), val | ok | 23.0 мин | 10551 | 19.11 | 2026-10-23 01:36:30+00:00 |
| [E17_final_A2_B](../experiments/E17_final_A2_B/README.md) | Полная модель LightGBM: набор A2_B (123 признаков), val | ok | 23.6 мин | 10886 | 28.24 | 2026-10-23 01:12:46+00:00 |
| [E17_final_A2_B_s1](../experiments/E17_final_A2_B_s1/README.md) | Полная модель LightGBM: набор A2_B (123 признаков), val | ok | 16.3 мин | 7240 | 14.04 | 2026-10-23 19:48:14+00:00 |
| [E17_final_A3_A](../experiments/E17_final_A3_A/README.md) | Полная модель LightGBM: набор A3_A (183 признаков), val | ok | 28.1 мин | 12870 | 26.86 | 2026-10-23 10:18:44+00:00 |
| [E17_final_A3_B](../experiments/E17_final_A3_B/README.md) | Полная модель LightGBM: набор A3_B (123 признаков), val | ok | 25.0 мин | 11631 | 28.88 | 2026-10-23 10:46:59+00:00 |
| [E17_final_A4](../experiments/E17_final_A4/README.md) | Полная модель LightGBM: набор A4 (147 признаков), val | ok | 29.3 мин | 13551 | 30.95 | 2026-10-23 03:07:14+00:00 |
| [E17_final_A5_A](../experiments/E17_final_A5_A/README.md) | Полная модель LightGBM: набор A5_A (129 признаков), val | ok | 22.2 мин | 10056 | 14.37 | 2026-10-23 17:30:59+00:00 |
| [E17_final_A5_B](../experiments/E17_final_A5_B/README.md) | Полная модель LightGBM: набор A5_B (50 признаков), val | ok | 18.5 мин | 8237 | 21.78 | 2026-10-23 17:53:23+00:00 |
| [E17_final_H_als](../experiments/E17_final_H_als/README.md) | Полная модель LightGBM: набор H_als (124 признаков), val | ok | 25.5 мин | 11635 | 12.8 | 2026-10-23 18:12:02+00:00 |
| [E17_final_H_both](../experiments/E17_final_H_both/README.md) | Полная модель LightGBM: набор H_both (126 признаков), val | ok | 21.2 мин | 9566 | 13.87 | 2026-10-23 19:04:09+00:00 |
| [E17_final_H_gcn](../experiments/E17_final_H_gcn/README.md) | Полная модель LightGBM: набор H_gcn (125 признаков), val | ok | 26.4 мин | 12046 | 13.29 | 2026-10-23 18:37:37+00:00 |
| [E17_final_manual_s1](../experiments/E17_final_manual_s1/README.md) | Полная модель LightGBM: набор manual (123 признаков), val | ok | 22.7 мин | 10215 | 13.48 | 2026-10-23 19:25:28+00:00 |
| [E18test_evaluate_models](../experiments/E18test_evaluate_models/README.md) | Оценка моделей на test с доверительными интервалами | ok | 32.3 мин | 13868 | 8.41 | 2026-10-24 08:31:01+00:00 |
| [E19_scaling](../experiments/E19_scaling/README.md) | Зависимость от числа пользователей и якорей | ok | 49.2 мин | 21338 | 2.79 | 2026-10-23 20:37:06+00:00 |
| [E20_feature_cost](../experiments/E20_feature_cost/README.md) | Стоимость расчёта признаков по ключу и агрегату | ok | 2.4 мин | 961 | 21.46 | 2026-10-23 21:26:29+00:00 |
| [E21_llm_agent_repeat](../experiments/E21_llm_agent_repeat/README.md) | Поиск признаков ИИ-агентом: повторы на разных сидах | ok | 30.3 мин | 9778 | 11.76 | 2026-10-24 08:00:37+00:00 |
| [E30_kion_rows](../experiments/E30_kion_rows/README.md) | KION: кандидаты и метки (10% пользователей) | ok | 9.5 мин | 654 | 1.05 | 2026-10-20 18:15:01+00:00 |
| [E31_kion_baseline](../experiments/E31_kion_baseline/README.md) | KION: эвристики и LightGBM на ручных признаках (val) | ok | 5.3 мин | 2292 | 7.9 | 2026-10-23 03:36:42+00:00 |
| [E32_kion_search](../experiments/E32_kion_search/README.md) | KION: поиск признаков A1 и A2 в песочнице | ok | 26.8 мин | 10535 | 1.46 | 2026-10-23 03:42:08+00:00 |
| [E33_kion_final](../experiments/E33_kion_final/README.md) | KION: финальные модели и оценка на test | ok | 29.0 мин | 12542 | 4.49 | 2026-10-23 04:09:08+00:00 |

Суммарная длительность: 15.57 ч (42 экспериментов).
