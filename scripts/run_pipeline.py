"""Запускает эксперименты по очереди (по одному, чтобы замеры времени были честными) и пропускает уже выполненные.

Примеры:
  python scripts/run_pipeline.py --list                    показать этапы с оценкой времени
  python scripts/run_pipeline.py --group core              основные эксперименты
  python scripts/run_pipeline.py --only 08,09              отдельные этапы по номеру
  python scripts/run_pipeline.py --group all --force       пересчитать всё заново
Этап считается выполненным, если в experiments/<папка>/results.json статус ok. Остановка при первой ошибке (--keep-going - продолжать).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# номер, команда, папка результата (None - без папки), оценка времени в минутах, группа, комментарий
STAGES = [
    ("00", "00_prepare_data.py", None, 10, "core", "скачивание и подготовка данных"),
    ("01", "01_audit_yambda.py", "E01_audit_yambda", 1, "core", ""),
    ("02", "02_audit_kion.py", "E02_audit_kion", 1, "core", ""),
    ("03", "03_candidate_sweep.py", "E03_candidate_sweep", 5, "core", ""),
    ("04", "04_build_rows.py", "E04_build_rows", 10, "core", "кандидаты и метки Yambda"),
    ("05", "05_leakage_tests.py", "E05_leakage_tests", 2, "core", "проверка на утечки"),
    ("06", "06_heuristics.py", "E06_heuristics", 4, "core", ""),
    ("07", "07_manual_features.py", "E07_manual_features", 10, "core", ""),
    ("08", "08_baseline_gbdt.py", "E08_baseline_gbdt", 45, "core", "базлайн; время обучения - результат, нужна свободная машина"),
    ("09", "09_ablation_windows.py", "E09_ablation_windows", 12, "core", ""),
    ("10", "10_feature_pool.py", "E10_feature_pool", 20, "core", "пул грамматики на подвыборке"),
    ("11", "11_search_random_genselect.py", "E11_search_random_genselect", 20, "core", "A1, A2"),
    ("12", "12_search_openfe.py", "E12_search_openfe", 120, "extended", "A3 OpenFE: самый долгий поиск"),
    ("13", "13_learned_windows.py", "E13_learned_windows", 8, "core", "A4"),
    ("14", "14_search_llm_agent.py", "E14_search_llm_agent", 60, "extended", "A5: нужен OLLAMA_API_KEY"),
    ("15a", "15_sequence_model.py --variant=base", "E15_sequence_base", 20, "extended", "модель над историей"),
    ("15b", "15_sequence_model.py --variant=listwise --loss=listwise", "E15_sequence_listwise", 20, "extended", ""),
    ("15c", "15_sequence_model.py --variant=L512 --L=512", "E15_sequence_L512", 30, "extended", ""),
    ("15d", "15_sequence_model.py --variant=extra --extra=1", "E15_sequence_extra", 20, "extended", ""),
    ("15e", "15_sequence_model.py --variant=final --L=512 --extra=1 --test", "E15_sequence_final", 35, "extended",
     "заранее выбранная конфигурация; оценивается на test"),
    ("16", "16_embeddings.py", "E16_embeddings", 90, "extended", "ALS и LightGCN"),
    ("17-A2_B", "17_final_model.py --set=A2_B", "E17_final_A2_B", 25, "core", ""),
    ("17-A2_A", "17_final_model.py --set=A2_A", "E17_final_A2_A", 25, "core", ""),
    ("17-A1_B", "17_final_model.py --set=A1_B", "E17_final_A1_B", 25, "core", "контроль: случайные признаки"),
    ("17-A1_A", "17_final_model.py --set=A1_A", "E17_final_A1_A", 25, "core", "контроль: случайные признаки"),
    ("17-A4", "17_final_model.py --set=A4", "E17_final_A4", 25, "core", ""),
    ("17-A3_A", "17_final_model.py --set=A3_A", "E17_final_A3_A", 25, "extended", "после E12"),
    ("17-A3_B", "17_final_model.py --set=A3_B", "E17_final_A3_B", 25, "extended", "после E12"),
    ("17-A5_A", "17_final_model.py --set=A5_A", "E17_final_A5_A", 25, "extended", "после E14"),
    ("17-A5_B", "17_final_model.py --set=A5_B", "E17_final_A5_B", 25, "extended", "после E14"),
    ("17-H_als", "17_final_model.py --set=H_als", "E17_final_H_als", 25, "extended", "после E16"),
    ("17-H_gcn", "17_final_model.py --set=H_gcn", "E17_final_H_gcn", 25, "extended", "после E16"),
    ("17-H_both", "17_final_model.py --set=H_both", "E17_final_H_both", 25, "extended", "после E16"),
    ("17-seeds", "17_final_model.py --set=manual --seed=1", "E17_final_manual_s1", 25, "extended", "шум обучения на полных данных"),
    ("17-seeds2", "17_final_model.py --set=A2_B --seed=1", "E17_final_A2_B_s1", 25, "extended", "шум обучения на полных данных"),
    ("18", "18_evaluate_models.py --split=test --extra=SEQ=experiments/E15_sequence_final/per_group_test.npz",
     "E18test_evaluate_models", 60, "extended", "ЕДИНСТВЕННАЯ оценка на test; запускать, когда все модели готовы"),
    ("19", "19_scaling.py", "E19_scaling", 60, "extended", ""),
    ("20", "20_feature_cost.py", "E20_feature_cost", 6, "extended", "замер стоимости признаков"),
    ("30", "30_kion_rows.py", "E30_kion_rows", 10, "core", ""),
    ("31", "31_kion_baseline.py", "E31_kion_baseline", 15, "core", ""),
    ("32", "32_kion_search.py", "E32_kion_search", 40, "core", ""),
    ("33", "33_kion_final.py", "E33_kion_final", 90, "core", "включает оценку KION на test"),
    ("99", "99_make_index.py", None, 1, "core", "индекс экспериментов"),
]


def done(folder: str | None) -> bool:
    if folder is None:
        return False
    path = ROOT / "experiments" / folder / "results.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8"))["meta"]["status"] == "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["core", "extended", "all"], default="core")
    ap.add_argument("--only", help="номера этапов через запятую, например 08,17-A2_B")
    ap.add_argument("--force", action="store_true", help="выполнять даже готовые этапы")
    ap.add_argument("--keep-going", action="store_true", help="не останавливаться при ошибке")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    chosen = [s for s in STAGES if (s[0] in only if only else args.group == "all" or s[4] == args.group or s[0] == "99")]
    if args.list:
        total = 0
        for num, cmd, folder, minutes, group, note in chosen:
            state = "готово" if done(folder) else "—"
            total += 0 if done(folder) else minutes
            print(f"{num:10s} {group:9s} ~{minutes:4d} мин  {state:7s} {cmd}  {note}")
        print(f"осталось примерно {total / 60:.1f} ч")
        return
    for num, cmd, folder, minutes, group, note in chosen:
        if done(folder) and not args.force:
            print(f"[{num}] пропуск: уже выполнено")
            continue
        print(f"\n[{num}] {cmd}  (около {minutes} мин)", flush=True)
        t = time.time()
        code = subprocess.run([sys.executable, *(["scripts/" + cmd.split()[0]] + cmd.split()[1:])], cwd=ROOT).returncode
        print(f"[{num}] код возврата {code}, {(time.time() - t) / 60:.1f} мин")
        if code != 0 and not args.keep_going:
            sys.exit(f"этап {num} завершился с ошибкой - остановка")


if __name__ == "__main__":
    main()
