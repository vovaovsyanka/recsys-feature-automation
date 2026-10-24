"""Строит рисунки для статьи (docs/figures/*.png) по results.json экспериментов. Запуск: python scripts/98_make_figures.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP, OUT = ROOT / "experiments", ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GRAY, BLUE, ORANGE, INK, MUTED = "#8a8a85", "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a66"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(folder: str) -> dict:
    return json.loads((EXP / folder / "results.json").read_text(encoding="utf-8"))["results"]


def forest(ax, rows, xlabel):
    """rows: (подпись, среднее, нижняя, верхняя, цвет)"""
    for i, (label, m, lo, hi, color) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot(m, i, "o", color=color, ms=6, mec="white", mew=1.2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)


# 1. лестница качества на val: от случайного порядка до лучшей автоматической модели
def fig_ladder():
    heur, base = load("E06_heuristics"), load("E08_baseline_gbdt")
    final = {k: json.loads((EXP / f"E17_final_{k}" / "results.json").read_text(encoding="utf-8"))["results"]["val_metrics"]["ndcg@10"]
             for k in ("A2_A", "A2_B")}
    seq = {k: load(f"E15_sequence_{k}") for k in ("base", "L512", "final")}
    seq_val = {k: v["val_by_epoch"][-1]["ndcg@10"] for k, v in seq.items()}    # метрика после последней эпохи
    rand = heur["val_metrics"]["случайный порядок"]["ndcg@10"]
    best_heur = max(v["ndcg@10"] for k, v in heur["val_metrics"].items() if k != "случайный порядок")
    rows = [("случайный порядок", rand, GRAY), ("лучшая эвристика (личный затухающий счётчик)", best_heur, GRAY),
            ("модель над историей, L=256", seq_val["base"], ORANGE), ("модель над историей, L=512", seq_val["L512"], ORANGE),
            ("модель над историей, финальная", seq_val["final"], ORANGE),
            ("LightGBM, ручные признаки (базлайн)", base["val_metrics"]["lambdarank"]["ndcg@10"], INK),
            ("LightGBM, 115 признаков отбором из пула (A2_B)", final["A2_B"], BLUE),
            ("LightGBM, ручные + 60 отбором (A2_A)", final["A2_A"], BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (label, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.62)
        ax.text(v + 0.004, i, f"{v:.3f}".replace(".", ","), va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.36)
    ax.set_xlabel("NDCG@10 на val (потолок кандидатов: 0,728)")
    fig.savefig(OUT / "ladder_val.png")
    plt.close(fig)


# 2. итог на test: разность NDCG@10 с ручным базлайном и 95% доверительный интервал
def fig_test_forest():
    ci = load("E18test_evaluate_models")["bootstrap_ci"]
    names = {"A1_A": "A1 случайные, +60", "A1_B": "A1 случайные, вместо ручных", "A2_A": "A2 отбор из пула, +60",
             "A2_B": "A2 отбор из пула, вместо ручных", "A3_A": "A3 OpenFE, +60", "A3_B": "A3 OpenFE, вместо ручных",
             "A4": "A4 обучаемые окна, +24", "A5_A": "A5 ИИ-агент, +6", "A5_B": "A5 ИИ-агент, вместо ручных",
             "H_als": "ALS-признак", "H_gcn": "LightGCN-признак", "H_both": "ALS + LightGCN"}
    rows = []
    for k, label in names.items():
        d = ci[k]["vs_manual"]["ndcg@10"]
        color = BLUE if k.startswith("A") else ORANGE
        rows.append((label, d["mean"], d["lo"], d["hi"], color))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    forest(ax, rows, "ΔNDCG@10 к ручному базлайну на test (95% ДИ, бутстреп по пользователям)")
    fig.savefig(OUT / "test_delta_forest.png")
    plt.close(fig)


# 3. шум песочницы и разброс методов: каждая точка - один запуск
def fig_sandbox():
    pr = load("E11_search_random_genselect")["proxy_results"]
    a3 = load("E12_search_openfe")["proxy_results"]
    a4 = load("E13_learned_windows")["proxy_results"]
    a5 = load("E14_search_llm_agent")["proxy_results"]
    a5 = {k: v for k, v in a5.items() if "(0)" not in k}          # E14: ответ без обратной связи для сценария B потерян, см. E21
    a5.update(load("E21_llm_agent_repeat")["proxy_results"])
    only = lambda d, text: [v["ndcg10"] for k, v in d.items() if text in k]
    manual = only(pr, "ручной набор (123)")                      # основной запуск и четыре сида LightGBM
    rows_a = [("ручной набор, сиды LightGBM", manual, GRAY), ("A1 случайные +60, сиды", only(pr, "A1, сценарий A"), BLUE),
              ("A2 отбор +60", only(pr, "A2, сценарий A"), BLUE), ("A3 OpenFE +60", only(a3, "сценарий A"), BLUE),
              ("A4 обучаемые окна +24", only(a4, "выбранным полупериодом"), BLUE),
              ("A5 агент, оба режима (E14, E21)", only(a5, "сценарий A") + only(a5, "_A_seed"), BLUE)]
    rows_b = [("ручной набор, сиды LightGBM", manual, GRAY), ("A1 случайные 115, сиды", only(pr, "A1, сценарий B"), BLUE),
              ("A2 отбор 115", only(pr, "A2, сценарий B"), BLUE), ("A3 OpenFE 115", only(a3, "сценарий B"), BLUE),
              ("A5 агент с обратной связью", only(a5, "iterative), сценарий B") + only(a5, "iterative_B"), BLUE),
              ("A5 агент без обратной связи", only(a5, "oneshot_B"), BLUE)]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [len(rows_a), len(rows_b)]})
    for ax, rows, title in zip(axes, (rows_a, rows_b), ("A: ручной набор + автоматические", "B: автоматические вместо ручных")):
        ax.axvspan(min(manual), max(manual), color="#e5e4e0", alpha=0.8, lw=0)
        for i, (label, vals, color) in enumerate(rows):
            ax.plot(vals, [i] * len(vals), "o", color=color, ms=6, mec="white", mew=1.0, alpha=0.9)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="x", color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("NDCG@10 в песочнице (серая полоса — диапазон ручного набора по сидам LightGBM)")
    fig.tight_layout()
    fig.savefig(OUT / "sandbox_runs.png")
    plt.close(fig)


# 4. вклад групп признаков
def fig_ablation():
    ab = load("E09_ablation_windows")["results"]["ablation"]
    full = ab["полный набор"]["ndcg10"]
    rows = [(k.replace("без группы ", ""), v["ndcg10"] - full) for k, v in ab.items() if k.startswith("без группы")]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for i, (label, d) in enumerate(rows):
        ax.barh(i, d, color=BLUE if d < -0.002 else GRAY, height=0.6)
        ax.text(d - 0.0003, i, f"{d:+.4f}", va="center", ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.016, 0.003)
    ax.set_xlabel("ΔNDCG@10 при удалении группы (песочница, полный набор 0,2700)")
    fig.savefig(OUT / "ablation_groups.png")
    plt.close(fig)


# 5. масштабирование по объёму обучающих данных
def fig_scaling():
    sc = load("E19_scaling")["results"]
    full = {"manual": (10_800_000, json.loads((EXP / "E08_baseline_gbdt" / "results.json").read_text(encoding="utf-8"))["results"]["val_metrics"]["lambdarank"]["ndcg@10"]),
            "A2_B": (10_800_000, json.loads((EXP / "E17_final_A2_B" / "results.json").read_text(encoding="utf-8"))["results"]["val_metrics"]["ndcg@10"])}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, color, label in (("manual", GRAY, "ручной набор"), ("A2_B", BLUE, "A2_B (115 отбором из пула)")):
        pts = sorted((v["train_rows"], v["ndcg@10"]) for k, v in sc.items() if k.startswith(name))
        pts.append(full[name])
        pts.sort()
        ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "-o", color=color, lw=2, ms=6, mec="white", mew=1.0, label=label)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10], ["1", "3", "10"])
    ax.set_xlabel("строк в обучении, млн (логарифмическая шкала)")
    ax.set_ylabel("NDCG@10 на val")
    ax.grid(color="#e5e4e0", lw=0.6)
    ax.legend(frameon=False)
    fig.savefig(OUT / "scaling.png")
    plt.close(fig)


# 6. KION: test
def fig_kion():
    ci = load("E33_kion_final")["bootstrap_ci"]
    names = {"kion:A1_B": "A1 случайные, вместо ручных", "kion:A2_A": "A2 отбор, +40", "kion:A2_B": "A2 отбор, вместо ручных"}
    rows = [(label, ci[k]["vs_manual"]["ndcg@10"]["mean"], ci[k]["vs_manual"]["ndcg@10"]["lo"], ci[k]["vs_manual"]["ndcg@10"]["hi"], BLUE)
            for k, label in names.items()]
    fig, ax = plt.subplots(figsize=(7, 2.0))
    forest(ax, rows, "ΔNDCG@10 к ручному базлайну, KION, test (95% ДИ)")
    fig.savefig(OUT / "kion_test_forest.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_ladder, fig_test_forest, fig_sandbox, fig_ablation, fig_scaling, fig_kion):
        f()
        print("готово:", f.__name__)
