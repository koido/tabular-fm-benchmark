#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==3.0.6"]
# ///
"""Tables for the tabarena-small comparison, from the same rows as
plot_results.py: TabFM / TabPFN-3 from tas-gpu-a100, the GBDTs from
tas-mac-cpu.

Writes RESULTS.md (mean ranks, wins/losses/ties, the figures and the
per-dataset mean +- sd tables) and prints the first two to stdout, since
the README repeats the mean-rank table:
    uv run results/make_tables.py
Ranks use the same tie rule as the rank figure: ties share the better rank.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_MD = HERE.parent / "RESULTS.md"
MODELS = ["tabfm", "tabpfn3", "catboost", "lightgbm", "xgboost"]
LABEL = {"tabfm": "TabFM", "tabpfn3": "TabPFN-3", "catboost": "CatBoost",
         "lightgbm": "LightGBM", "xgboost": "XGBoost"}
# metric -> (label, +1 if higher is better else -1)
METRICS = {"roc_auc": ("ROC-AUC", 1), "log_loss": ("Log loss", -1), "accuracy": ("Accuracy", 1)}
PAIRS = [("tabfm", "tabpfn3"), ("tabfm", "catboost"), ("tabpfn3", "catboost")]

HEADER = """# Results

All models at default settings. TabFM and TabPFN-3 ran on an A100; CatBoost, LightGBM and XGBoost on a Mac CPU (Apple M3 Ultra). 16 datasets × 5 train/test splits (seeds 42–46) = 80 runs per model. Source: `results/results_long.csv`."""

FIGURES = """![Paired ROC-AUC differences per split](results/figures/paired_diff_roc_auc.png)

![Paired log loss differences per split](results/figures/paired_diff_log_loss.png)

![ROC-AUC per dataset](results/figures/per_dataset_roc_auc.png)

![Log loss per dataset](results/figures/per_dataset_log_loss.png)"""

PER_DATASET = """## Per-dataset tables

Mean ± standard deviation over the 5 splits. **Bold** marks the best unrounded mean per row, so two equal displayed values can differ in bold. Datasets are sorted by size."""


def load():
    df = pd.read_csv(HERE / "results_long.csv")
    main = df[((df.run_tag == "tas-gpu-a100") & df.model.isin(["tabfm", "tabpfn3"]))
              | ((df.run_tag == "tas-mac-cpu") & df.model.isin(["catboost", "lightgbm", "xgboost"]))]
    # Fail fast: every model must have every (dataset, seed) exactly once.
    counts = main.groupby("model").size()
    assert (counts == 80).all() and set(counts.index) == set(MODELS), counts
    assert not main.duplicated(["model", "dataset", "seed"]).any()
    return main


def results_md(main):
    size = main.groupby("dataset")[["n_train", "n_test"]].first().sum(axis=1)
    order = sorted(size.index, key=lambda d: (size[d], d))
    out = [HEADER, "\n## Ranks and head-to-head\n", readme_tables(main), "\n" + FIGURES + "\n", PER_DATASET]
    for metric, (label, sign) in METRICS.items():
        better = "higher" if sign > 0 else "lower"
        out += [f"\n### {label} ({better} is better)\n",
                "| Dataset | " + " | ".join(LABEL[m] for m in MODELS) + " |",
                "|---|" + "---:|" * len(MODELS)]
        g = main.groupby(["dataset", "model"])[metric]
        mean, sd = g.mean().unstack()[MODELS], g.std().unstack()[MODELS]
        for d in order:
            best = (sign * mean.loc[d]).max()
            cells = []
            for m in MODELS:
                c = f"{mean.loc[d, m]:.3f} ± {sd.loc[d, m]:.3f}"
                cells.append(f"**{c}**" if sign * mean.loc[d, m] == best else c)
            out.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n\n"


def readme_tables(main):
    lines = ["Mean rank of each model's per-dataset mean score (16 datasets; 1 = best; ties share the better rank):", "",
             "| Metric | " + " | ".join(LABEL[m] for m in MODELS) + " |",
             "|---|" + "---:|" * len(MODELS)]
    for metric, (label, sign) in METRICS.items():
        mean = main.groupby(["dataset", "model"])[metric].mean().unstack()[MODELS]
        r = (-sign * mean).rank(axis=1, method="min").mean()
        lines.append(f"| {label} | " + " | ".join(f"{r[m]:.2f}" for m in MODELS) + " |")
    lines += ["", "Wins / losses / ties of the first-named model over the 80 runs, paired by dataset and split:", "",
              "| Comparison | ROC-AUC | Log loss | Accuracy |", "|---|---|---|---|"]
    for a, b in PAIRS:
        cells = []
        for metric in METRICS:
            sign = METRICS[metric][1]
            piv = main.pivot_table(index=["dataset", "seed"], columns="model", values=metric)
            diff = sign * (piv[a] - piv[b])
            cells.append(f"{(diff > 0).sum()} / {(diff < 0).sum()} / {(diff == 0).sum()}")
        lines.append(f"| {LABEL[a]} vs {LABEL[b]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rows = load()
    RESULTS_MD.write_text(results_md(rows))
    print(readme_tables(rows))
    print(f"\nwrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
