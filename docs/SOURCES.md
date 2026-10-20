# Реестр источников

Дата проверки: 2026-10-20. Каждый источник открыт по первичной ссылке (страница arXiv / HF / GitHub / Google Research),
а не взят из памяти. Это важно: при сверке выяснилось, что ряд идентификаторов, названных «по памяти», неверен (см. раздел «Исправленные ошибки»).

## Уровни верификации

| Код | Что означает |
|-----|--------------|
| **V1** | Открыта страница первоисточника (abstract / карточка датасета). Проверены название, авторы, дата, площадка. Утверждения ниже — **только на уровне аннотации**; полный текст не читался. |
| **V2** | Дополнительно проверено по полному тексту / коду / реальным данным (будет отмечено после чтения или после запуска на данных). |
| **S** | Только вторичные источники (блоги, пересказы). В отчёте используется как иллюстрация практики, не как доказательство. |

Утверждения, которые в отчёте опираются на источник уровня V1, формулируются как «авторы заявляют».

## A. Датасеты

| # | Источник | Что проверено | Уровень |
|---|----------|---------------|---------|
| A1 | Yambda, карточка HF: <https://huggingface.co/datasets/yandex/yambda> | Apache-2.0; 3 масштаба (50M/500M/5B); схемы `listens/likes/dislikes/unlikes/undislikes/multi_event`; `timestamp` — uint32, шаг 5 с; `is_organic`; `played_ratio_pct`; размеры файлов через HF API: `flat/50m/listens.parquet` 369.1 МБ, `likes.parquet` 7.2 МБ, `dislikes.parquet` 1.0 МБ; `artist_item_mapping.parquet`, `album_item_mapping.parquet` в корне. | V1 (V2 после загрузки) |
| A2 | Yambda, статья: Ploshkin et al., *Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval*, arXiv:2505.22238, 28.05.2025 (v2 01.06.2025). <https://arxiv.org/abs/2505.22238> | Авторы, дата, заявленный протокол Global Temporal Split, базлайны ItemKNN / iALS / SANSA / SASRec. Детали протокола в аннотации не раскрыты — нужно читать полный текст. | V1 |
| A3 | KION, репозиторий: <https://github.com/irsafilo/KION_DATASET> | Публичная часть: 5 476 251 взаимодействий, 962 179 пользователей, 15 706 объектов; период 13.03.2021–22.08.2022; к данным **добавлен случайный шум**; файлы `interactions.csv.zip` (18.7 МБ), `data_en.zip` (77.3 МБ), `data_original.zip` (78.8 МБ). **Лицензия в репозитории не указана** (GitHub API: `license = null`). | V1 |
| A4 | KION, статья: Petrov, Safilo, Tikhonovich, Ignatov, *MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation*, arXiv:2209.00325, 01.09.2022, RecSys CARS Workshop 2022. <https://arxiv.org/abs/2209.00325> | Название, авторы, площадка. | V1 |

## B. Автоматизация конструирования признаков

| # | Источник | Что заявляют авторы (по аннотации) | Уровень |
|---|----------|-------------------------------------|---------|
| B1 | Kanter & Veeramachaneni, *Deep Feature Synthesis*, IEEE DSAA 2015. Страница проекта: <https://people.csail.mit.edu/kalyan/dsm/index.html> | DFS обходит связи реляционной БД и последовательно применяет функции вдоль пути. Из 3 соревнований Kaggle в 2 обошли большинство участников. | V1 (через поиск; PDF не открывался) |
| B2 | Lam et al., *One Button Machine (OneBM)*, arXiv:1706.00327, 2017. <https://arxiv.org/abs/1706.00327> | Автоматические join-ы и трансформации; уровень «топ-16–24%» на 3 соревнованиях Kaggle. | V1 |
| B3 | Luo et al., *AutoCross*, arXiv:1904.12857, 2019. <https://arxiv.org/abs/1904.12857> | Beam search по дереву кросс-признаков; successive mini-batch GD. | V1 |
| B4 | Zhang et al., *OpenFE*, ICML 2023, arXiv:2211.12507. <https://arxiv.org/abs/2211.12507> | «Feature boosting» (оценка прироста кандидата), двухстадийная отсечка; на 2 соревнованиях Kaggle — лучше 99.3% и 99.6% команд (с простой моделью). | V1 |
| B5 | Hollmann, Müller, Hutter, *CAAFE*, arXiv:2305.03403 (v5, 28.09.2023). <https://arxiv.org/abs/2305.03403> | LLM итеративно генерирует код признаков по описанию датасета; ROC-AUC 0.798→0.822 на 14 датасетах (улучшение на 11). | V1 |
| B6 | Abhyankar, Shojaee, Reddy, *LLM-FE*, TMLR, arXiv:2503.14434. <https://arxiv.org/abs/2503.14434> | Feature engineering как поиск программ; LLM как эволюционный оптимизатор. | V1 |
| B7 | Han et al., *Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning* (FeatLLM), ICML 2024, arXiv:2404.09491. <https://proceedings.mlr.press/v235/han24f.html> | LLM предлагает правила → признаки для простой модели; важно: сценарий **few-shot**, не большой recsys. | V1 (через поиск) |
| B8 | Fey et al., *Relational Deep Learning*, arXiv:2312.04615 (07.12.2023). <https://arxiv.org/abs/2312.04615> | БД как временной гетерогенный граф + GNN вместо ручных признаков. | V1 |
| B9 | Robinson et al., *RelBench*, arXiv:2407.20060 (29.07.2024). <https://arxiv.org/abs/2407.20060> | User study: RDL лучше ручного FE и сокращает человеческий труд «более чем на порядок». **Конкретные часы и метрики в аннотации не приведены.** | V1 |
| B10 | Wang et al., *4DBInfer*, arXiv:2404.18209 (28.04.2024, «under review»). <https://arxiv.org/abs/2404.18209> | Наивные join-ы ограничены; важны все 4 измерения дизайн-пространства. | V1 |
| B11 | Kumo, *KumoRFM-2*, arXiv:2604.12596 (апрель 2026). <https://arxiv.org/pdf/2604.12596> (+ оригинальный KumoRFM — только PDF на kumo.ai) | Реляционная foundation-модель, in-context. Коммерческий источник, интерес конфликтен. | V1 (через поиск) |
| B12 | Рекомендательно-специфичный поиск (LLM-агенты + агрегаты по окнам): прямого аналога **не найдено**. Ближайшие: Information 2026 <https://doi.org/10.3390/ai7070245> (временные ряды, контроль утечки), RecSys Factory arXiv:2608.11241. | Не используются как доказательство, только как ориентир «пробел в литературе». | S |

## C. Модели над историей (идея 3)

| # | Источник | Заявление | Уровень |
|---|----------|-----------|---------|
| C1 | Zhou et al., *DIN*, KDD 2018, arXiv:1706.06978. <https://arxiv.org/abs/1706.06978> | Целевое внимание: представление пользователя зависит от кандидата. | V1 |
| C2 | Pi et al., *SIM*, arXiv:2006.05639. <https://arxiv.org/abs/2006.05639> | Двухступенчатый поиск (GSU → ESU) по истории до 54 000 событий; +7.1% CTR (заявлено). | V1 |
| C3 | Chang et al., *TWIN*, KDD 2023, arXiv:2302.02352. <https://arxiv.org/abs/2302.02352> | Согласованная метрика релевантности на обеих стадиях. | V1 |
| C4 | Chen et al., *ETA*, arXiv:2108.04468. <https://arxiv.org/abs/2108.04468> | LSH-внимание для длинных последовательностей, сквозное обучение. | V1 |
| C5 | Kang & McAuley, *SASRec*, ICDM 2018, arXiv:1808.09781. <https://arxiv.org/abs/1808.09781> | Self-attention для следующего элемента. | V1 |
| C6 | Zhai et al., *HSTU / Generative Recommenders*, arXiv:2402.17152 (ICML'24). <https://arxiv.org/abs/2402.17152> | Рекомендации как sequential transduction; до +65.8% NDCG на публичных данных (заявлено). | V1 |
| C7 | Pancha et al., *PinnerFormer*, arXiv:2205.04507. <https://arxiv.org/abs/2205.04507> | Предсказание долгосрочной активности по последовательности. | V1 |
| C8 | Khrylchenko et al., *ARGUS / Scaling Recommender Transformers to One Billion Parameters*, arXiv:2507.15994 (KDD'26). <https://arxiv.org/abs/2507.15994> | Yandex Music; +2.26% времени прослушивания, +6.37% лайков в A/B (заявлено). Прямо релевантно Yambda. | V1 |

## D. Графы

| # | Источник | Заявление | Уровень |
|---|----------|-----------|---------|
| D1 | He et al., *LightGCN*, SIGIR 2020, arXiv:2002.02126. <https://arxiv.org/abs/2002.02126> | Только линейная агрегация соседей по графу user–item. | V1 |
| D2 | Ying et al., *PinSage*, KDD 2018, arXiv:1806.01973. <https://arxiv.org/abs/1806.01973> | GCN + random walks на графе 3 млрд узлов. | V1 |
| D3 | Rossi et al., *TGN*, arXiv:2006.10637. <https://arxiv.org/abs/2006.10637> | Динамические графы как последовательности событий + память. | V1 |

## E. Бустинг vs нейросети, ранжирование

| # | Источник | Заявление | Уровень |
|---|----------|-----------|---------|
| E1 | Grinsztajn, Oyallon, Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?*, arXiv:2207.08815. <https://arxiv.org/abs/2207.08815> | Деревья лучше на данных среднего размера (~10K объектов). **Не распространяется автоматически на recsys-масштаб.** | V1 |
| E2 | Prokhorenkova et al., *CatBoost*, arXiv:1706.09516. <https://arxiv.org/abs/1706.09516> | Ordered boosting против prediction shift (target leakage внутри бустинга). | V1 |
| E3 | Jeunen et al., *On GBDT and Neural Rankers: ShareChat*, FIRE'23, arXiv:2312.01760. <https://arxiv.org/abs/2312.01760> | При больших данных и эмбеддингах нейросети точнее; GBDT дешевле. | V1 |
| E4 | Lutz, Wilm, Duwe, *Industry Insights ... GBDT ... E-Commerce LTR*, RecSys 2025, arXiv:2507.20753. <https://arxiv.org/abs/2507.20753> | Простая DNN обошла LambdaMART по кликам и выручке в 8-недельном A/B (OTTO). | V1 |
| E5 | Qin et al., *Are Neural Rankers still Outperformed by GBDT?*, ICLR 2021. <https://mlanthology.org/iclr/2021/qin2021iclr-neural> | Нейро-LTR сравнялся с лучшим деревом при правильных приёмах. | V1 (через поиск) |
| E6 | NVIDIA RecSys Challenge 2020 (Twitter): XGBoost + target encoding + GPU FE. Обзоры: <https://blogs.nvidia.com/blog/2020/09/10/ai-recommendation-systems-challenge> | Победа за счёт скорости итераций по признакам. | **S** |
| E7 | OTTO Kaggle: candidate generation (co-visitation) + LightGBM ranker. <https://developer.nvidia.com/blog/build-efficient-recommender-systems-with-co-visitation-matrices-and-rapids-cudf> | Типичная двухстадийная схема. Разбор победителя первичным источником не найден. | **S** |

## F. Методология оценки

| # | Источник | Заявление | Уровень |
|---|----------|-----------|---------|
| F1 | Ji, Sun, Zhang, Li, *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, TOIS 41(3), 2023, arXiv:2010.11060. <https://arxiv.org/abs/2010.11060> | Без глобального временного разбиения модели «видят будущее»; ранжирование моделей непредсказуемо. | V1 |
| F2 | Meng et al., *Exploring Data Splitting Strategies...*, arXiv:2007.13237. <https://arxiv.org/abs/2007.13237> | Способ разбиения — конфаундер при сравнении моделей. | V1 |
| F3 | Rendle, *Evaluation Metrics for Item Recommendation under Sampling*, arXiv:1912.02263. <https://arxiv.org/abs/1912.02263>; финальная версия — Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 (best paper): <https://research.google/pubs/on-sampled-metrics-for-item-recommendation/> | Sampled-метрики несогласованы с полными, при малой выборке все сводятся к AUC. **Следствие: оценивать по полному каталогу / общему фиксированному набору кандидатов.** | V1 |
| F4 | Ferrari Dacrema, Cremonesi, Jannach, *Are We Really Making Much Progress?*, RecSys 2019, arXiv:1907.06902. <https://arxiv.org/abs/1907.06902> | Воспроизведено 7 из 18 нейро-моделей, 6 из 7 проиграли простым методам. **Следствие: сильные простые базлайны обязательны.** | V1 |
| F5 | Rendle et al., *NCF vs. MF Revisited*, arXiv:2005.09683. <https://arxiv.org/abs/2005.09683> | Хорошо настроенное скалярное произведение сильнее MLP-сходства. | V1 (через поиск) |

## Повышение уровня верификации до V2 (чтение полного текста / сверка с данными)

| Источник | Что установлено | Как |
|---|---|---|
| Yambda (A1/A2) | Полный текст статьи прочитан (прочитан в полном объёме). ~11 месяцев наблюдений; **GTS: train 300 суток, пауза 30 мин, test 1 сутки**; Listen+ = доля ≥ 50%; метрики NDCG@{10,100}, Recall@{10,100}, Coverage@{10,100}; Table 6, Yambda-50M: ItemKNN NDCG@10 = 0.0781, SASRec 0.0748, DecayPop 0.0260, MostPop 0.0186, iALS 0.0407; медиана истории 3 076 прослушиваний (Table 5). **Единица `timestamp`: в статье формула `⌊(T−T_start)/5⌋·5` - т.е. СЕКУНДЫ с шагом 5, а не «5-секундные тики» как можно прочитать на карточке HF.** Подтверждено данными (E01: медиана gap/длина трека = 1.0, макс. = 300.9 сут). `is_organic=1` = органика подтверждено данными (E01). | PDF + E01 |
| KION (A3/A4) | HTML-версия статьи: период **13.03.2021–22.08.2021** (в README ошибочно «2022»); перед публикацией только анонимизация и «небольшой шум» (тип/величина не указаны); **при повторных просмотрах оставлена только последняя запись, длительность и процент суммируются** (поэтому пары уникальны, а колонка названа `last_watch_dt`); 46 участников / 844 решений; метрика MAP@10; популярность 0.0910, победитель 0.1221; **топ-решения - двухстадийные: кандидаты (BERT4Rec/SASRec/ItemKNN/…) → LightGBM/CatBoost-реранкер** с признаками (давность, жанры, демография, тренды популярности). | HTML + E02 |
| OpenFE (B4) | Полный текст (прочитан в полном объёме, §3.2, Alg. 2-4). **FeatureBoost**: модель обучается только на новых признаках T' для подгонки остатка относительно предсказаний ŷ базовой модели (минимизирует Σ l(y, ŷ + f'(x[T']))), прирост Δ = L(f) − L(f'). **Этап I (последовательное отсечение признаков)**: данные делятся на 2^q блоков; на итерации i берутся 2^i блоков; для каждого кандидата Δτ = FeatureBoost(D_i, {τ}, ŷ); остаётся верхняя половина; в конце оставляются Δ>0; дедупликация одинаковых признаков. **Этап II**: FeatureBoost на T + A'(T) (учёт взаимодействий) и отбор верхних по важности (MDI; сравниваются также PFI и SHAP). Реализация на LightGBM. Моя реализация A3 повторяет эту схему (ранжирование заменено на binary-logloss прокси). | PDF |
| RelBench (B9) | 15 задач; специалист в среднем **12.3 ч** (σ=1.6) на задачу против ~0.5 ч у RDL (-96%); код 878 vs 56 строк; признаки специалиста → LightGBM с подбором гиперпараметров; RDL «не хуже или лучше» в 11 из 15 задач. **Ограничения, названные авторами:** один специалист; RDL слабее на регрессии; время EDA ограничено 4 ч, идеи признаков 1 ч; **в одной задаче признаки специалиста протекли (утечка валидации), что было найдено и исправлено.** | HTML |

## Исправленные ошибки (иллюстрация, зачем нужна верификация)

1. **arXiv:2008.04323**, названный «по памяти» как *On Sampled Metrics for Item Recommendation*, на самом деле — *IceCube-Gen2* (астрофизика). Корректные ссылки — F3.
2. **arXiv для KION** — 2209.00325 (память подсказывала другой номер).
3. **Порядок авторов KION** различается: в README репозитория — Petrov, Safilo, Tikhonovich, Ignatov; в поисковой выдаче — Petrov, Tikhonovich, Safilo, Ignatov; на странице arXiv — Petrov, Safilo, Tikhonovich, Ignatov. В отчёте использован порядок со страницы arXiv.
4. **`is_organic` в Yambda**: на карточке HF таблица и FAQ противоречат друг другу в трактовке значений 0/1. **Не полагаться на карточку — определить по данным** (эксперимент E01) и по полному тексту статьи.
5. **Карточка HF для Yambda** трактует `timestamp` как «5-секундные единицы»; буквальное прочтение даёт 1 505 суток вместо 300.9 (E01). Верно прочтение статьи (секунды, шаг 5).
6. **Отдельный `interactions.csv.zip` в репозитории KION обрезан** (1 594 787 строк, ровно 50 МиБ); полная таблица - `data_en/interactions.csv` (5 476 251 строка) (E02).
7. В поисковой выдаче по KumoRFM, FeatLLM и OTTO часть пересказов — вторичная; из них взято только то, что подтверждается первичной страницей, остальное помечено S.

## Что ещё нужно проверить (бэклог)

- Полный текст OpenFE/CAAFE/LLM-FE (протоколы сравнения) - читать перед интерпретацией собственных результатов по автоматизации.
- Полные тексты DIN/SIM/TWIN/HSTU - перед реализацией C1 (реализация будет «в духе», не воспроизведением).
- LightGBM (Ke et al., NIPS 2017) и LambdaMART (Burges 2010) — первичные описания алгоритмов.
