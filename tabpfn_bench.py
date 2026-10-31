# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "tabpfn==8.1.0",
#     "torch==2.13.0",
#     "scikit-learn==1.9.0",
#     "pandas==3.0.3",
#     "numpy==2.5.1",
# ]
# ///
"""
TabPFN-3 local benchmark: runs TabPFNClassifier over every dataset
in the selected --suite (see bench_datasets.SUITES) and writes results in
the shared TabFM/TabPFN comparison schema to logs/bench_results/ next to
the script.

By default (--device auto) tabpfn picks CUDA, then MPS, then CPU. The
2026-07-27 results ran on MPS (Apple M3 Ultra GPU), not
CPU -- confirmed on 2026-09-25 by re-running with device="mps" and "cpu":
only MPS reproduced all five accuracies.

Versions are pinned to what produced the 2026-07-27 results. The default
model depends on the tabpfn package version: tabpfn 8.1.0's default is
TabPFN-3, while the upstream README (as of 2026-09) says the default is now
TabPFN-3.5 -- an unpinned `tabpfn` would silently benchmark a different
model.

tabpfn's own download cannot be revision-pinned (it fetches the checkpoint
from the HF repo's main branch), so instead each run checks the checkpoint's
sha256 against CHECKPOINT_SHA256 and fails on a mismatch. The 2026-07-27
results used
Prior-Labs/tabpfn_3 revision 24a16a89d245878b846555110985634aa2e656d7,
tabpfn-v3-classifier-v3_default.ckpt sha256
d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988.

First run needs either a browser (to accept the TabPFN-3 license via
PriorLabs) or the TABPFN_TOKEN env var set -- see
https://github.com/PriorLabs/TabPFN#installation--setup.

Weights are under the "TABPFN-3 Non-Commercial License v1.0" (Prior Labs
GmbH, rev. 2026-03-24) -- non-commercial research/benchmarking (this
project's use) is explicitly permitted, including "internal benchmarking"
and "academic research" by name. Not for production/revenue-generating use
or hosting as an API/SaaS without a separate commercial license.

Run with:
    uv run tabpfn_bench.py
On a GPU VM, pass the device explicitly and a tag (non-default runs need
their own --tag directory, so runs never overwrite each other), e.g.:
    uv run tabpfn_bench.py --device cuda --tag gpu-l4
(export TABPFN_MODEL_CACHE_DIR, as an absolute path, to reuse a local
checkpoint instead of downloading it again).
Multiple train/test splits: `--seeds 42 43 44 --tag <name>` (--tag required
whenever --seeds is anything but the default [42]). `--suite tabarena-small`
runs the 12 extra TabArena datasets plus the 4 real default-suite
ones (--tag required for any non-default --suite too).
"""
import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from bench_datasets import (
    SUITES,
    classification_metrics,
    load_dataset,
    peak_rss_bytes,
    utc_now,
)

RESULTS_DIR = Path(__file__).parent / "logs" / "bench_results"
# tabpfn-v3-classifier-v3_default.ckpt that produced every result so far.
CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"


class CheckpointMismatch(Exception):
    """Raised past main()'s per-dataset handler: a wrong model must stop the run."""


@functools.cache
def verify_checkpoint(path):
    """Fail unless the checkpoint tabpfn loaded is the one the results used."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise CheckpointMismatch(
            f"{path}: sha256 {digest} != expected {CHECKPOINT_SHA256}"
        )


def device_memory(used):
    """Framework-reported memory on the first device (single-GPU VMs), peak
    over load + fit + warm-up + predict. nvidia-smi only shows what torch's
    caching allocator holds, not what the model needed."""
    d = used[0]
    if d.type == "cuda":
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(d),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(d),
        }
    if d.type == "mps":  # MPS exposes no peak counter; value right after predict
        return {"current_allocated_bytes": torch.mps.current_allocated_memory()}
    return None


def bench_one(name, device, seed):
    started_at = utc_now()
    # Drop the previous dataset's model first, or its cached blocks count
    # toward this dataset's peak.
    gc.collect()
    if device.startswith("cuda") or (device == "auto" and torch.cuda.is_available()):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
        torch.mps.empty_cache()
    X_train, y_train, X_test, y_test = load_dataset(name, seed=seed)

    # Explicit version: TabPFNClassifier()'s default can be overridden by the
    # TABPFN_MODEL_VERSION env var or a .env file, even with tabpfn pinned.
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device)
    clf.fit(X_train, y_train)
    verify_checkpoint(str(clf.model_path))  # fit() downloads it if missing
    # device="auto" picks CUDA (all visible GPUs), then MPS, then CPU --
    # record what actually ran.
    used = clf.devices_

    # warm-up: same call run twice, first discarded, to exclude first-call
    # model-loading overhead from the timed measurement.
    t0 = time.perf_counter()
    clf.predict_proba(X_test)
    warmup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = clf.predict_proba(X_test)
    predict_seconds = time.perf_counter() - t0

    preds = clf.classes_[np.argmax(probs, axis=1)]
    accuracy = accuracy_score(y_test, preds)
    log_loss_, roc_auc = classification_metrics(y_test, probs, clf.classes_)

    return {
        "model": "tabpfn3",
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
        "devices": [str(d) for d in used],
        "gpu_names": [torch.cuda.get_device_name(d) for d in used if d.type == "cuda"],
        "tabpfn_version": importlib.metadata.version("tabpfn"),
        "torch_version": torch.__version__,
        "device_memory": device_memory(used),
        "host_peak_rss_bytes": peak_rss_bytes(),
        "started_at": started_at,
        "finished_at": utc_now(),
    }


def main():
    # Line-buffer stdout so progress lines survive if the run is killed mid-way.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        help='torch device: "auto" (default), "cuda", "mps" or "cpu"')
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="train_test_split random_state values to run (default: 42)")
    parser.add_argument("--suite", choices=list(SUITES), default="default",
                        help="dataset suite from bench_datasets.SUITES (default: default)")
    parser.add_argument("--tag", help="write results to bench_results/<tag>/")
    args = parser.parse_args()
    one_seed = args.seeds == [42]
    # Only the single-seed default run writes to bench_results/ itself (the
    # untagged location) -- every other run needs its own --tag directory,
    # so runs never overwrite each other.
    if (args.device != "auto" or not one_seed or args.suite != "default") and not args.tag:
        parser.error("--tag is required with an explicit --device, --suite other than "
                     "default, or --seeds other than [42]")
    results_dir = RESULTS_DIR / args.tag if args.tag else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    # Each dataset's failure is only printed, so a broken CUDA install would
    # otherwise look like a finished run -- fail fast instead.
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "--device cuda but torch sees no CUDA GPU"

    for name in SUITES[args.suite]:
        for seed in args.seeds:
            print(f"\n=== {name} / seed={seed} ===")
            try:
                result = bench_one(name, args.device, seed)
                suffix = "" if one_seed else f"_seed{seed}"
                out_path = results_dir / f"tabpfn3_{name}{suffix}.json"
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
            except CheckpointMismatch:
                raise
            except Exception as e:
                print(f"[FAILED] {name} seed={seed}: {e}")
                continue

    print("\nDone.")


if __name__ == "__main__":
    main()
