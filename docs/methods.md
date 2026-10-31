# Methods

Neither foundation model is fine-tuned: both predict in context from the training
split. This is not a benchmark for procurement or vendor selection.

## Models

| | Package | Framework | Run on | Settings |
|---|---|---|---|---|
| TabFM | `tabfm[jax]==1.0.1` | JAX 0.10.2 | A100 (`jax[cuda13]`) | library defaults (32 ensemble members, each seeing up to 500 features), bfloat16 |
| TabPFN-3 | `tabpfn==8.1.0` | PyTorch 2.13.0 | A100 (CUDA) | `create_default_for_version(ModelVersion.V3)` (8 ensemble members, each seeing up to 200 features) |
| CatBoost | `catboost==1.2.10` | — | Mac CPU | defaults, `verbose=False` |
| LightGBM | `lightgbm==4.7.0` | — | Mac CPU | defaults, `random_state=0` |
| XGBoost | `xgboost==3.4.1` | — | Mac CPU | defaults, `random_state=0` |

Checkpoints:
- TabFM: `google/tabfm-1.0.0-jax`, HF revision `d5e74033fcf257699fab013e2cdd7edf424ff904`
  (pinned in `tabfm_bench.py`). Weights under the TabFM Non-Commercial License v1.0.
- TabPFN-3: `Prior-Labs/tabpfn_3`, HF revision `24a16a89d245878b846555110985634aa2e656d7`,
  `tabpfn-v3-classifier-v3_default.ckpt` (sha256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`; `tabpfn_bench.py`
  stops if the checkpoint differs). Weights under the TABPFN-3 Non-Commercial
  License v1.0.

Neither set of weights is redistributed here; download them yourself under their
licenses. Pin `tabpfn`: newer releases default to TabPFN-3.5.

## Software

| Runs | Python | Packages |
|---|---|---|
| CatBoost, LightGBM, XGBoost (Mac CPU) | 3.12 | catboost 1.2.10, lightgbm 4.7.0, xgboost 3.4.1, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3 |
| TabPFN-3 (every machine) | 3.12 | tabpfn 8.1.0, torch 2.13.0 (CUDA 13.0 build on the GPUs), scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.1 |
| TabFM on the GPUs (A100 and L4: all runs; T4) | 3.11 | tabfm 1.0.1, jax[cuda13] 0.10.2, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6 |
| TabFM on TPU v6e (July and October) | 3.11 | tabfm 1.0.1, jax[tpu] 0.10.2, libtpu 0.0.42.1, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.4.6 |
| TabFM on Mac CPU (September) | — | jax 0.10.2; other versions not recorded |
| Tables and figures (`results/`) | ≥3.11 | matplotlib 3.10.7, pandas 3.0.6, numpy 2.4.6 |

Dependencies not listed were not pinned.

## Settings that differ from the library defaults

Everything not listed here is the library default.

- LightGBM, XGBoost: `random_state=0` (also passed to CatBoost, whose default
  is already 0).
- CatBoost: `verbose=False` (logging only); categorical columns as strings
  through `cat_features`.
- XGBoost and LightGBM take categorical columns from the pandas `category` type.
  XGBoost also gets label-encoded targets, `enable_categorical=True` (the default
  in xgboost 3.4.1) and `tree_method="hist"` (the default, `auto`, also uses hist).
- TabPFN-3: built with `create_default_for_version(ModelVersion.V3)`, the same
  model and settings as the tabpfn 8.1.0 default but immune to an environment
  override, with `device` set explicitly (`cuda` on the GPUs, `mps` or `cpu` on
  the Mac).
- TabFM: weights loaded from the pinned revision. On the T4 only, `dtype=float32`
  (default bfloat16) and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (default 0.75).
- On the GPU VMs, `LD_LIBRARY_PATH` is unset for both foundation models.

No categorical column in these datasets has missing values; missing values occur
only in numeric columns (Fitness_Club, MIC, Marketing_Campaign), and every model
handles them with its own default.

## Splits and metrics

- Splits: stratified 80/20 `train_test_split` with `random_state` 42, 43, 44, 45, 46.
  Every model sees the same 5 splits.
- Metrics: accuracy and log loss as defined in scikit-learn; ROC-AUC one-vs-rest,
  macro-averaged, for multiclass datasets.

The datasets are described in [data.md](data.md).

## Caveats

1. **Small data, dependent splits.** 748–5,000 rows per dataset. The 5 splits of a
   dataset share data, so they are not independent, and no significance test is
   made.
2. **Defaults only.** No model is tuned. Tuned GBDTs, or larger foundation-model
   ensembles, could change the picture. "Default" also means different ensemble
   sizes for the two foundation models.
3. **Ties at the ceiling.** On anneal and splice every model is near the maximum,
   so their ranks carry little information.
4. **TabFM probabilities are renormalized.** TabFM returns bfloat16 probabilities
   whose rows do not sum exactly to 1. They are cast to float64 and divided by
   their row sum before computing log loss and ROC-AUC. Accuracy uses the raw
   argmax and is unaffected. Two side effects: bfloat16 makes many scores equal,
   which ROC-AUC counts as ties, and scikit-learn's clipping epsilon depends on the
   dtype, so TabFM's float64 probabilities are clipped less than the other
   models' float32 ones; this matters only for true-class probabilities near 0.
5. **Different hardware per model family.** The foundation models ran on an A100
   and the GBDTs on a Mac CPU. Speed is not compared between the foundation models
   and the GBDTs.
6. **Hardware changes a few predictions.** For the same model and split, accuracy
   differs by up to 2 test rows between A100 and L4.
7. **TPU numbers are limited.** TabFM ran on a TPU v6e only for one split (seed 42)
   of 4 datasets; TabPFN-3 was not run on a TPU.
8. **Memory counters differ by framework.** See the memory notes in [hardware.md](hardware.md).

## Reproducing

```bash
# Datasets only (no model); also checks every split. --suite tabarena-small
# below is the 16 datasets in data.md:
uv run bench_datasets.py

# GBDTs (CPU); results go to logs/bench_results/<tag>/:
for lib in catboost lightgbm xgboost; do
  uv run gbdt_bench.py --library $lib --suite tabarena-small --seeds 42 43 44 45 46 --tag my-cpu
done

# TabFM and TabPFN-3 on one NVIDIA GPU VM (Deep Learning VM image family
# common-cu129-ubuntu-2204-nvidia-580). Copy gpu_bench_run.sh, tabfm_bench.py,
# tabpfn_bench.py, bench_datasets.py, tabfm_gpu_setup.sh and sample_device.sh
# to $HOME on the VM (optionally also ~/scikit_learn_data, the OpenML cache from
# the first command; otherwise the VM downloads from OpenML), then run there:
export TABPFN_TOKEN=...   # TabPFN-3 license token (see below)
nohup bash gpu_bench_run.sh my-gpu --suite tabarena-small --seeds 42 43 44 45 46 > run.log 2>&1 &

# Tables and figures, from the published results/results_long.csv:
uv run results/make_tables.py && uv run results/plot_results.py
```

OpenML often fails transiently, sometimes with a misleading "Dataset … not found";
rerun, since finished downloads are cached in `~/scikit_learn_data`. The GBDT
script logs a failed dataset as `[FAILED]` and continues, so check that each
library wrote 80 files.

Each script writes one JSON file per model, dataset and split, to
`logs/bench_results/<tag>/` next to the script (on the GPU VM, `~/logs/bench_results/<tag>/`,
with per-step logs in `~/logs/`). Compare their `accuracy`, `log_loss` and `roc_auc`
with the rows of `results/results_long.csv` for the same model, `dataset` and
`seed`; the published tables use `run_tag` `tas-gpu-a100` for the foundation models
and `tas-mac-cpu` for the GBDTs. The script that turned the JSON files into the CSV
is not included.

TabPFN-3's first run needs either a browser, to accept its license with Prior Labs,
or the `TABPFN_TOKEN` environment variable set; an unattended GPU run needs the
token (see https://github.com/PriorLabs/TabPFN#installation--setup).

Each benchmark script pins the library versions its runs used in its header (for
TabFM on the GPU, in `tabfm_gpu_setup.sh`), so shared libraries such as pandas
differ slightly between scripts. `bench_datasets.py`'s own header, used only when
it runs on its own, matches `gbdt_bench.py`.

Only the A100, L4 and GBDT runs can be repeated with the files here. The setup for
the TPU, Mac and T4 runs (described in [hardware.md](hardware.md)) is not included. To reuse a
TabPFN-3 checkpoint you already have, set `TABPFN_MODEL_CACHE_DIR` to its directory.

## References

- TabFM: Kong, W., et al. (2026). *TabFM: A Zero-Shot Foundation Model for Tabular Data*. arXiv:2609.37959. https://arxiv.org/abs/2609.37959
- TabFM announcement: Kong, W., & Das, A. (2026, June 30). *Introducing TabFM: A zero-shot foundation model for tabular data*. Google Research Blog. https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
- TabPFN-3: Grinsztajn, L., et al. (2026). *TabPFN-3: Technical Report*. arXiv:2605.13986. https://arxiv.org/abs/2605.13986
- TabArena: Erickson, N., et al. (2025). *TabArena: A Living Benchmark for Machine Learning on Tabular Data*. arXiv:2506.16791. https://arxiv.org/abs/2506.16791 ; https://tabarena.ai (dataset selection source).
