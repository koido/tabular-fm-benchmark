"""
TabFM benchmark: runs Google's TabFM (TPU by default; GPU or CPU via
--backend) over every dataset in the selected --suite (see
bench_datasets.SUITES) and writes results in the shared TabFM/TabPFN
comparison schema to logs/bench_results/ next to the script.

Needs Python >=3.11 (tabfm) plus tabfm[jax] and a TPU-enabled jax -- install
tabfm[jax] together with jax[tpu] in the same step, or jax ends up CPU-only
(this script then stops, since the backend is checked); tabfm[jax] also pulls in scikit-learn/pandas/
numpy, which bench_datasets.py needs.

On an NVIDIA GPU VM, set up with tabfm_gpu_setup.sh instead and run with
`--backend gpu --tag <name>` (--tag keeps results out of the TPU run's files).
Multiple train/test splits: `--seeds 42 43 44 --tag <name>` (--tag required
whenever --seeds is anything but the default [42]). `--suite tabarena-small`
runs the 12 extra TabArena datasets plus the 4 real default-suite
ones (--tag required for any non-default --suite too).

Checkpoint (google/tabfm-1.0.0-jax) is not gated, so this downloads without
a Hugging Face token. tabfm's load() takes no revision, so the pinned HF
snapshot (CHECKPOINT_REVISION, the one every result so far used) is
downloaded here and passed in as a local checkpoint path.
Weights are under the non-commercial "TabFM Non-Commercial License v1.0"
-- fine for academic/research use, not for commercial use.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
from huggingface_hub import snapshot_download
from tabfm import TabFMClassifier, tabfm_v1_0_0_jax as tabfm_v1_0_0

from bench_datasets import (
    SUITES,
    classification_metrics,
    load_dataset,
    peak_rss_bytes,
    utc_now,
)

# Line-buffer stdout so progress lines survive if the run is killed mid-way.
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["tpu", "gpu", "cpu"], default="tpu",
                    help="jax backend this run must use (default: tpu)")
parser.add_argument("--tag", help="write results to bench_results/<tag>/")
parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="compute dtype passed to tabfm load() (its default is "
                         "bfloat16; T4 has no bf16 matmul, so it needs float32)")
parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="train_test_split random_state values to run (default: 42)")
parser.add_argument("--suite", choices=list(SUITES), default="default",
                    help="dataset suite from bench_datasets.SUITES (default: default)")
args = parser.parse_args()
ONE_SEED = args.seeds == [42]
# Only the single-seed default run (TPU, bfloat16) writes to bench_results/
# itself (the untagged location) -- every other run needs its own --tag
# directory, so runs never overwrite each other.
if (args.backend != "tpu" or args.dtype != "bfloat16" or not ONE_SEED
        or args.suite != "default") and not args.tag:
    parser.error("--tag is required with a non-default --backend/--dtype/--suite or "
                 "--seeds other than [42]")

RESULTS_DIR = Path(__file__).parent / "logs" / "bench_results"
if args.tag:
    RESULTS_DIR = RESULTS_DIR / args.tag

# --- Device check ---
print(f"[device check] jax version : {jax.__version__}")
print(f"[device check] backend     : {jax.default_backend()}")
print(f"[device check] devices     : {jax.devices()}")

# tabfm[jax] alone installs CPU-only jax; if jax[tpu] (libtpu) or
# jax[cuda13] wasn't installed alongside it, this would still "succeed" on
# CPU -- fail fast instead of burning billable VM time on a false pass.
SETUP_SCRIPT = {"tpu": "tabfm[jax] together with jax[tpu]",
                "gpu": "tabfm_gpu_setup.sh",
                "cpu": "(plain tabfm[jax])"}[args.backend]
assert jax.default_backend() == args.backend, (
    f"expected backend '{args.backend}', got '{jax.default_backend()}' -- "
    f"install deps via {SETUP_SCRIPT}"
)
DEVICE_KIND = jax.devices()[0].device_kind

# --- Load model once (bf16 by default), reused across datasets ---
print("\n[tabfm] loading google/tabfm-1.0.0-jax ...")
# Same download tabfm does itself (tabfm_v1_0_0.TabFM_HF._from_pretrained),
# plus the revision pin it does not expose.
CHECKPOINT_REVISION = "d5e74033fcf257699fab013e2cdd7edf424ff904"
snapshot = snapshot_download(repo_id=tabfm_v1_0_0.HF_REPO_ID, revision=CHECKPOINT_REVISION,
                             allow_patterns=["classification/**"])
model = tabfm_v1_0_0.load(model_type="classification",
                          checkpoint_path=os.path.join(snapshot, "classification"),
                          dtype=getattr(jax.numpy, args.dtype))


def device_memory():
    """jax's own counters for the first device (None on CPU). Default GPU
    preallocation (75% of memory) makes nvidia-smi useless for this, and
    peak_bytes_in_use cannot be reset, so it is the process-lifetime peak."""
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return {k: stats.get(k) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}


def bench_one(name, seed):
    started_at = utc_now()
    X_train, y_train, X_test, y_test = load_dataset(name, seed=seed)

    clf = TabFMClassifier(model=model)
    clf.fit(X_train, y_train)

    # warm-up: same call run twice, first discarded, to exclude JIT
    # compilation cost from the timed measurement. JAX dispatches
    # asynchronously, so np.asarray(...) inside the timed window forces the
    # result to materialize on host -- like calling .block_until_ready() --
    # or the timer would only catch dispatch, not compute.
    t0 = time.perf_counter()
    probs_warmup = np.asarray(clf.predict_proba(X_test))
    warmup_seconds = time.perf_counter() - t0
    assert np.all(np.isfinite(probs_warmup)), (
        f"predict_proba returned non-finite values: {probs_warmup}"
    )

    t0 = time.perf_counter()
    probs = np.asarray(clf.predict_proba(X_test))
    predict_seconds = time.perf_counter() - t0

    preds = clf.classes_[np.argmax(probs, axis=1)]
    accuracy = float(np.mean(preds == y_test))
    # sklearn's roc_auc_score rejects the ml_dtypes bfloat16 array
    # np.asarray() produces from a bfloat16 jax array -- cast to float64
    # (after timing, so this doesn't count against predict_seconds). bf16's
    # ~3-digit precision also means each row's probabilities routinely sum
    # to e.g. 0.997-1.003 rather than exactly 1 -- fine for log_loss/binary
    # roc_auc, but sklearn's multiclass roc_auc_score hard-rejects any row
    # outside its default np.allclose(1, ...) tolerance, which bf16 rounding
    # reliably fails -- renormalize rows to sum to 1 so it doesn't raise.
    probs64 = probs.astype(np.float64)
    probs64 = probs64 / probs64.sum(axis=1, keepdims=True)
    log_loss_, roc_auc = classification_metrics(y_test, probs64, clf.classes_)

    return {
        "model": "tabfm",
        "dataset": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(clf.classes_),
        "accuracy": accuracy,
        "log_loss": log_loss_,
        "roc_auc": roc_auc,
        "predict_seconds": predict_seconds,
        "throughput_samples_per_sec": len(X_test) / predict_seconds,
        "warmup_seconds": warmup_seconds,
        "seed": seed,
        "backend": jax.default_backend(),
        "device_kind": DEVICE_KIND,
        "jax_version": jax.__version__,
        "dtype": args.dtype,
        "device_memory": device_memory(),
        "host_peak_rss_bytes": peak_rss_bytes(),
        "started_at": started_at,
        "finished_at": utc_now(),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in SUITES[args.suite]:
        for seed in args.seeds:
            print(f"\n=== {name} / seed={seed} ===")
            try:
                result = bench_one(name, seed)
                suffix = "" if ONE_SEED else f"_seed{seed}"
                out_path = RESULTS_DIR / f"tabfm_{name}{suffix}.json"
                out_path.write_text(json.dumps(result, indent=2))
                print(
                    f"accuracy={result['accuracy']:.4f}  "
                    f"log_loss={result['log_loss']:.4f}  "
                    f"roc_auc={result['roc_auc']:.4f}  "
                    f"predict_seconds={result['predict_seconds']:.3f}  "
                    f"throughput={result['throughput_samples_per_sec']:.1f} samples/s  "
                    f"warmup_seconds={result['warmup_seconds']:.3f}"
                )
                print(f"-> {out_path}")
            except Exception as e:
                print(f"[FAILED] {name} seed={seed}: {e}")
                continue

    print("\nDone.")


if __name__ == "__main__":
    main()
