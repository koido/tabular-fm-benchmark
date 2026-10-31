# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "xgboost==3.4.1",
#     "lightgbm==4.7.0",
#     "catboost==1.2.10",
#     "scikit-learn==1.9.1",
#     "pandas==3.0.6",
#     "numpy==2.5.3",
# ]
# ///
"""
GBDT baselines: runs one of XGBoost / LightGBM / CatBoost (--library)
over every dataset in the selected --suite (see
bench_datasets.SUITES), across one or more train/test splits (--seeds),
and writes results in the shared
TabFM/TabPFN comparison schema (model/dataset/n_*/accuracy/predict_seconds/
host_peak_rss_bytes/started_at/finished_at) plus log_loss, roc_auc,
fit_seconds, seed, library_version, device, to logs/bench_results/ next to
the script.

All three libraries run with default hyperparameters and a fixed
random_state=0, so the train/test split (--seeds) is the only thing that
varies across runs. n_jobs/thread_count are left at each library's own
default (not tuned). Documented deviations from "default hyperparameters
only":
- XGBoost needs integer class labels, so y is label-encoded for it only
  (classes_ is then the LabelEncoder's classes_, not XGBClassifier's own
  int classes_). enable_categorical=True and tree_method="hist" are passed
  explicitly but are already the defaults in xgboost 3.4.1; they let it
  read pandas "category" columns natively.
- LightGBM reads pandas "category" columns natively, no extra parameter.
- CatBoost does not read pandas dtype, so its categorical columns are passed
  explicitly via cat_features=; verbose=False (its default prints a
  per-iteration training log to stdout).

Unlike tabfm_bench.py / tabpfn_bench.py, this script has no "2026-07-27
root" default location to protect, so --tag is always required.

Run with:
    uv run gbdt_bench.py --library xgboost --tag mac-cpu
    uv run gbdt_bench.py --library lightgbm --seeds 42 43 --tag mac-cpu
    uv run gbdt_bench.py --library catboost --suite tabarena-small --tag mac-cpu-tas
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from bench_datasets import SUITES, classification_metrics, load_dataset, peak_rss_bytes, utc_now

RESULTS_DIR = Path(__file__).parent / "logs" / "bench_results"


def make_classifier(library):
    if library == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(enable_categorical=True, tree_method="hist", random_state=0)
    if library == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=0)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=0, verbose=False)


def bench_one(library, name, seed):
    started_at = utc_now()
    X_train, y_train, X_test, y_test = load_dataset(name, seed=seed)
    cat_cols = list(X_train.select_dtypes(include="category").columns)

    clf = make_classifier(library)
    t0 = time.perf_counter()
    if library == "xgboost":
        # XGBClassifier.classes_ would otherwise be the encoded ints, not
        # the original labels -- encode y ourselves and keep the encoder's
        # classes_ (sorted, like np.unique) as the canonical label order.
        le = LabelEncoder()
        clf.fit(X_train, le.fit_transform(y_train))
        classes_ = le.classes_
    elif library == "catboost":
        # CatBoost takes categorical columns as strings. Under pandas 3 a
        # missing value stays NaN here (CatBoost would reject it), but no
        # categorical column in the benchmark datasets has missing values.
        X_train = X_train.astype({c: str for c in cat_cols})
        X_test = X_test.astype({c: str for c in cat_cols})
        clf.fit(X_train, y_train, cat_features=cat_cols)
        classes_ = np.asarray(clf.classes_)
    else:
        clf.fit(X_train, y_train)
        classes_ = clf.classes_
    fit_seconds = time.perf_counter() - t0

    # warm-up: same call run twice, first discarded -- same protocol as
    # tabfm_bench.py / tabpfn_bench.py, kept for a consistent predict_seconds
    # definition even though GBDT predict has no JIT/compile cost to exclude.
    t0 = time.perf_counter()
    clf.predict_proba(X_test)
    warmup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = np.asarray(clf.predict_proba(X_test))
    predict_seconds = time.perf_counter() - t0

    preds = classes_[np.argmax(probs, axis=1)]
    accuracy = accuracy_score(y_test, preds)
    log_loss_, roc_auc = classification_metrics(y_test, probs, classes_)

    return {
        "model": library,
        "dataset": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(classes_),
        "accuracy": accuracy,
        "log_loss": log_loss_,
        "roc_auc": roc_auc,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "throughput_samples_per_sec": len(X_test) / predict_seconds,
        "warmup_seconds": warmup_seconds,
        "seed": seed,
        "library_version": importlib.metadata.version(library),
        "device": "cpu",
        "host_peak_rss_bytes": peak_rss_bytes(),
        "started_at": started_at,
        "finished_at": utc_now(),
    }


def main():
    # Line-buffer stdout so progress lines survive if the run is killed mid-way.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="train_test_split random_state values to run (default: 42)")
    parser.add_argument("--suite", choices=list(SUITES), default="default",
                        help="dataset suite from bench_datasets.SUITES (default: default)")
    parser.add_argument("--tag", required=True, help="write results to bench_results/<tag>/")
    args = parser.parse_args()
    results_dir = RESULTS_DIR / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    one_seed = args.seeds == [42]

    for name in SUITES[args.suite]:
        for seed in args.seeds:
            print(f"\n=== {args.library} / {name} / seed={seed} ===")
            try:
                result = bench_one(args.library, name, seed)
                suffix = "" if one_seed else f"_seed{seed}"
                out_path = results_dir / f"{args.library}_{name}{suffix}.json"
                out_path.write_text(json.dumps(result, indent=2))
                print(
                    f"accuracy={result['accuracy']:.4f}  log_loss={result['log_loss']:.4f}  "
                    f"roc_auc={result['roc_auc']:.4f}  fit_seconds={result['fit_seconds']:.3f}  "
                    f"predict_seconds={result['predict_seconds']:.3f}"
                )
                print(f"-> {out_path}")
            except Exception as e:
                print(f"[FAILED] {name} seed={seed}: {e}")
                continue

    print("\nDone.")


if __name__ == "__main__":
    main()
