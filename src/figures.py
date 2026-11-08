"""Static figures for the report / presentation (reads results/results.json).

Run:  python -m src.figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import OUTCOME_LEVELS
from .train import RESULTS_DIR

TEAL, ROSA, INK, MUTED = "#0A8F9E", "#D6266F", "#2B1B17", "#6B5A52"
FS = {"demographic": ("Demographics Only", TEAL), "immigration": ("+ Immigration Experiences", ROSA)}
MODELS = ["Logistic regression", "Random forest", "Separate neural nets", "Multi-task neural net",
          "Multi-task ensemble"]
TASK_NAMES = {"depressive": "Depressive Symptoms", "anxiety": "Anxiety", "uncertainty": "Uncertainty Stress"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#BBB", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.dpi": 150})


def model_comparison(res):
    df = pd.DataFrame(res["overall"])
    order = (df[df.features == "immigration"].sort_values("macro_f1_mean", ascending=False).model.tolist())
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    x = np.arange(len(order))
    for k, (fs, (label, color)) in enumerate(FS.items()):
        sub = df[df.features == fs].set_index("model").loc[order]
        ax.bar(x + (k - 0.5) * 0.38, sub.macro_f1_mean, 0.36, yerr=sub.macro_f1_std, color=color,
               label=label, capsize=3, error_kw={"ecolor": INK, "lw": 1})
        for xi, v in zip(x + (k - 0.5) * 0.38, sub.macro_f1_mean):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [m.title().replace(" Neural", "\nNeural").replace(" Ensemble", "\nEnsemble") for m in order], fontsize=9)
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Macro F1 (3 Outcomes)")
    ax.set_ylim(0, max(df.macro_f1_mean) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Average Macro F1 By Model (Test Set, Mean ± SD Over 5 Seeds)", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_model_comparison.png")


def by_outcome(res):
    df = pd.DataFrame(res["by_outcome"])
    df = df[df.model == "Multi-task ensemble"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    tasks = list(TASK_NAMES)
    x = np.arange(len(tasks))
    for k, (fs, (label, color)) in enumerate(FS.items()):
        sub = df[df.features == fs].set_index("outcome").loc[tasks]
        ax.bar(x + (k - 0.5) * 0.38, sub.macro_f1_mean, 0.36, yerr=sub.macro_f1_std, color=color, label=label,
               capsize=3, error_kw={"ecolor": INK, "lw": 1})
    ax.set_xticks(x, [TASK_NAMES[t] for t in tasks])
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Macro F1")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Multi-Task Ensemble By Outcome", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_multitask_by_outcome.png")


def by_group(res):
    df = pd.DataFrame(res["by_group"])
    df = df[df.model == "Multi-task ensemble"].set_index("features")
    groups = sorted(["White", "Latino", "Black", "Asian"],
                    key=lambda g: -(df.loc["immigration", g] - df.loc["demographic", g]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(groups))
    for k, (fs, (label, color)) in enumerate(FS.items()):
        ax.bar(x + (k - 0.5) * 0.38, df.loc[fs, groups], 0.36, color=color, label=label)
    ax.set_xticks(x, groups)
    ax.set_xlabel("Race / Ethnicity (Ranked By Gain)")
    ax.set_ylabel("Average Macro F1")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Multi-Task Ensemble By Race / Ethnicity", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_multitask_by_group.png")


def confusion(res):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    short = ["Not At All", "Several\nDays", "Most\nDays", "Daily"]
    for ax, (t, name) in zip(axes, TASK_NAMES.items()):
        cm = np.array(res["confusion_deployed"]["immigration"][t], dtype=float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="RdPu", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), short, fontsize=8)
        ax.set_yticks(range(4), short, fontsize=8)
        ax.set_xlabel("Predicted Answer")
        ax.set_title(name, color=INK, fontsize=10)
    axes[0].set_ylabel("Actual Answer")
    fig.suptitle("Confusion Matrices · + Immigration Experiences Multi-Task Ensemble (Test Set)", color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_confusion_matrices.png")


def importance(res):
    imp = res["deployed"]["immigration"]["permutation_importance"]
    imm = {"Deportation worry", "Immigration-policy stress", "Discrimination by immigration officers"}
    items = sorted(imp.items(), key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([k.title() for k, _ in items], [v["mean"] for _, v in items], xerr=[v["std"] for _, v in items],
            color=[ROSA if k in imm else TEAL for k, _ in items], error_kw={"ecolor": INK, "lw": 1})
    ax.set_xlabel("Drop In Average Macro F1 When Shuffled")
    ax.set_ylabel("Input")
    ax.set_title("Which Inputs Matter Most (Permutation Importance)", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_permutation_importance.png")


def training_curves(res):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for fs, (label, color) in FS.items():
        for k, member in enumerate(res["deployed"][fs]["members"]):
            h = pd.DataFrame(member["history"])
            ax.plot(h.epoch, h.val_macro_f1, color=color, lw=1.4, alpha=0.8, label=label if k == 0 else None)
            ax.plot(member["best_epoch"], member["val_macro_f1"], "o", color=color, ms=4)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation Average Macro F1")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Validation Curves Of The 5 Ensemble Networks (Dot = Early-Stopping Epoch)", color=INK, loc="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_training_curves.png")


def main():
    res = json.loads((RESULTS_DIR / "results.json").read_text())
    model_comparison(res)
    by_outcome(res)
    by_group(res)
    confusion(res)
    importance(res)
    training_curves(res)
    print("figures written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
