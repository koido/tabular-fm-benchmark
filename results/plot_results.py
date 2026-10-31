#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7", "pandas==3.0.6", "numpy==2.4.6"]
# ///
"""Figures for the tabarena-small comparison: 16 real datasets x
5 train/test splits (seeds 42-46). TabFM and TabPFN-3 from the A100 runs
(tag tas-gpu-a100), XGBoost / LightGBM / CatBoost at defaults on the Mac
(tas-mac-cpu); the L4 runs (tas-gpu-l4) only for the GPU-agreement figure.

Reads results_long.csv next to this script, writes PNGs to figures/:
    uv run results/plot_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# Categorical slots 1-3 of the reference palette (validated all-pairs, light
# mode) for the two foundation models and the strongest GBDT; the other two
# GBDTs share a neutral ink and differ by marker, so identity never rests on
# color alone.
MODELS = ["tabfm", "tabpfn3", "catboost", "lightgbm", "xgboost"]
LABEL = {"tabfm": "TabFM", "tabpfn3": "TabPFN-3", "catboost": "CatBoost",
         "lightgbm": "LightGBM", "xgboost": "XGBoost"}
COLOR = {"tabfm": "#2a78d6", "tabpfn3": "#eb6834", "catboost": "#1baf7a",
         "lightgbm": "#8a8984", "xgboost": "#8a8984"}
MARKER = {"tabfm": "o", "tabpfn3": "s", "catboost": "D", "lightgbm": "^", "xgboost": "v"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# metric -> (label, +1 if higher is better else -1)
METRICS = {"roc_auc": ("ROC-AUC", 1), "log_loss": ("log loss", -1), "accuracy": ("accuracy", 1)}
SEEDS = [42, 43, 44, 45, 46]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def load():
    df = pd.read_csv(HERE / "results_long.csv")
    main = df[((df.run_tag == "tas-gpu-a100") & df.model.isin(["tabfm", "tabpfn3"]))
              | ((df.run_tag == "tas-mac-cpu") & df.model.isin(["catboost", "lightgbm", "xgboost"]))]
    # Fail fast: every model must have every (dataset, seed) exactly once.
    counts = main.groupby("model").size()
    assert (counts == 80).all() and set(counts.index) == set(MODELS), counts
    assert set(main.seed.unique()) == set(SEEDS)
    return df, main


def ranks_by_run(main, metric, sign):
    """Rank 1..5 of each model within each (dataset, seed); ties share the best rank."""
    piv = main.pivot_table(index=["dataset", "seed"], columns="model", values=metric)
    return (-sign * piv).rank(axis=1, method="min")[MODELS]


def fig_rank_distribution(main):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
    for ax, (metric, (label, sign)) in zip(axes, METRICS.items()):
        r = ranks_by_run(main, metric, sign)
        counts = np.array([[(r[m] == k).sum() for k in range(1, 6)] for m in MODELS])
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=80, aspect="auto")
        for i in range(5):
            for j in range(5):
                v = counts[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 45 else TEXT)
        ax.set_xticks(range(5), [str(k) for k in range(1, 6)])
        ax.set_yticks(range(5), [LABEL[m] for m in MODELS])
        ax.set_xlabel("rank within each dataset x split (1 = best)")
        ax.set_title(f"{label}\nmean rank (80 runs): " + " · ".join(
            f"{LABEL[m]} {r[m].mean():.2f}" for m in MODELS[:3]), fontsize=8, color=TEXT)
        ax.grid(False)
    fig.suptitle("How often each model ranks 1st..5th (80 = 16 datasets x 5 splits; ties share the better rank)",
                 fontsize=10)
    fig.savefig(OUT / "rank_distribution.png", dpi=160)
    plt.close(fig)


PAIRS = [("tabfm", "tabpfn3"), ("tabfm", "catboost"), ("tabpfn3", "catboost")]


def fig_paired_differences(main, metric):
    label, sign = METRICS[metric]
    piv = main.pivot_table(index=["dataset", "seed"], columns="model", values=metric)
    datasets = sorted(piv.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), sharey=True, constrained_layout=True)
    # One dataset order for all three panels: by the first pair's mean difference.
    first = sign * (piv[PAIRS[0][0]] - piv[PAIRS[0][1]])
    order = first.groupby(level=0).mean().sort_values().index.tolist()
    y = {d: i for i, d in enumerate(order)}
    for ax, (a, b) in zip(axes, PAIRS):
        diff = sign * (piv[a] - piv[b])
        wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
        for d in datasets:
            vals = diff.loc[d].values
            ax.scatter(vals, np.full(len(vals), y[d]), s=14, color=COLOR[a], alpha=0.55,
                       edgecolors="none", zorder=3)
            ax.scatter([vals.mean()], [y[d]], s=46, marker="|", color=TEXT, linewidths=2, zorder=4)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
        ax.set_title(f"{LABEL[a]} vs {LABEL[b]}\n{LABEL[a]} better {wins}, worse {losses}, tied {ties} (of 80)",
                     fontsize=9, color=TEXT)
        ax.set_xlabel(f"{label}: > 0 means {LABEL[a]} better")
    axes[0].set_yticks(range(len(order)), order)
    fig.suptitle(f"Paired {label} differences on the same split (dots = 5 splits, bar = their mean"
                 + ("; log loss sign flipped so > 0 = better)" if sign < 0 else ")"), fontsize=10)
    fig.savefig(OUT / f"paired_diff_{metric}.png", dpi=160)
    plt.close(fig)


def fig_per_dataset(main, metric):
    label, sign = METRICS[metric]
    datasets = sorted(main.dataset.unique())
    fig, axes = plt.subplots(4, 4, figsize=(11, 9.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, d in zip(axes.flat, datasets):
        sub = main[main.dataset == d]
        for i, m in enumerate(MODELS):
            v = sub[sub.model == m][metric].values
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, marker=MARKER[m],
                       color=COLOR[m], alpha=0.7, edgecolors="none", zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=TEXT, linewidth=1.5, zorder=4)
        ax.set_title(d if len(d) <= 30 else d[:28] + "…", fontsize=8, color=TEXT)
        ax.set_xticks(range(5), [LABEL[m] for m in MODELS], rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{label} per dataset ({'higher' if sign > 0 else 'lower'} is better); "
                 "dots = 5 splits, bar = mean. Each panel has its own y scale.", fontsize=10)
    fig.savefig(OUT / f"per_dataset_{metric}.png", dpi=160)
    plt.close(fig)


def fig_gpu_agreement(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax, m in zip(axes, ["tabfm", "tabpfn3"]):
        a = df[(df.run_tag == "tas-gpu-a100") & (df.model == m)].set_index(["dataset", "seed"])
        l4 = df[(df.run_tag == "tas-gpu-l4") & (df.model == m)].set_index(["dataset", "seed"])
        j = a[["log_loss", "accuracy", "n_test"]].join(l4[["log_loss", "accuracy"]], rsuffix="_l4")
        assert len(j) == 80, len(j)
        ax.scatter(j.log_loss, j.log_loss_l4, s=14, color=COLOR[m], marker=MARKER[m],
                   alpha=0.7, edgecolors="none", zorder=3)
        lim = [0, max(j.log_loss.max(), j.log_loss_l4.max()) * 1.05]
        ax.plot(lim, lim, color=MUTED, linewidth=1, zorder=2)
        rows = ((j.accuracy - j.accuracy_l4).abs() * j.n_test).round().astype(int)
        ax.set_title(f"{LABEL[m]}\nsame accuracy in {(rows == 0).sum()} of 80 runs\n"
                     f"at most {rows.max()} test rows apart", fontsize=9, color=TEXT)
        ax.set_xlabel("log loss on A100"); ax.set_ylabel("log loss on L4")
        ax.set_xlim(lim); ax.set_ylim(lim)
    fig.suptitle("Same runs on two GPUs (A100 vs L4)", fontsize=10)
    fig.savefig(OUT / "gpu_agreement.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df, main_rows = load()
    fig_rank_distribution(main_rows)
    for metric in ("roc_auc", "log_loss"):
        fig_paired_differences(main_rows, metric)
        fig_per_dataset(main_rows, metric)
    fig_gpu_agreement(df)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
