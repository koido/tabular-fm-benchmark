# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn==1.9.1", "pandas==3.0.6", "numpy==2.5.3"]
# ///
"""
Dataset loaders for the TabFM vs. TabPFN tabular-classification comparison.

No jax/tabfm dependency here on purpose -- this module is imported by the
TabFM, TabPFN-3 and GBDT benchmark scripts, so it has to run standalone on
plain sklearn/pandas/numpy (pinned to the versions gbdt_bench.py uses). Deps are
declared inline (PEP 723), so `uv run <path>` installs them into an ephemeral
env automatically -- `uv run python3 <path>` does NOT read inline metadata.
Run/test it directly with:
    uv run bench_datasets.py

Dataset selection
------------------
TabFM (google-research/tabfm, released 2026-06-30) was evaluated zero-shot
against TabArena (OpenML Study 457, https://arxiv.org/abs/2506.16791), a
51-dataset benchmark (38 classification / 13 regression, 700-150,000 rows).
TabPFN's own benchmark practice draws its small-data test set from the
OpenML-CC18 suite (Study 99), filtered to <2000 rows -- TabPFN is also known
to need special handling (ensembling) above ~1000 training rows.

The four datasets below are the same underlying benchmark data TabArena uses
for its four smallest classification tasks -- same name, near-identical
row/feature counts -- but loaded via their original OpenML ids rather than
TabArena's own re-uploaded copies (which use different ids: 46913/46921/
46918/46941 respectively; NOT literal members of Study 457's id list).
blood-transfusion/diabetes/credit-g (ids 1464/37/31) *are* genuine OpenML-CC18
(Study 99) members, verified against its id list; maternal-health-risk (id
46600) is not (it post-dates CC18). Chosen for spread across domain, feature
mix, and class structure rather than near-duplicates, all under TabPFN's
~1000-row comfort zone once split:

- blood-transfusion (OpenML id 1464, 748 rows, 4 numeric features, binary):
  blood-donation history. Smallest task, purely numeric, in OpenML-CC18.
- diabetes (OpenML id 37, "Pima", 768 rows, 8 numeric features, binary):
  medical diagnostics. Purely numeric, in OpenML-CC18.
- credit-g (OpenML id 31, "German Credit", 1000 rows, 7 numeric + 13
  categorical features, binary): consumer credit risk. The one dataset here
  with a real numeric/categorical mix, in OpenML-CC18.
- maternal-health-risk (OpenML id 46600, 1014 rows, 6 numeric features,
  3-class): pregnancy-vitals risk triage. Only multiclass task; not in
  OpenML-CC18 (post-dates it). Its target embeds literal quote characters
  (e.g. "'high risk'") in this id's ARFF export -- stripped by this loader;
  TabArena's re-uploaded copy (id 46941) declares the target as a proper
  nominal attribute and may not have this artifact.

All four are classification only (no regression, per this round's scope) and
use a plain 80/20 stratified train/test split (via train_test_split) since
none of these tasks ship a canonical held-out split -- TabArena itself
resplits them into folds rather than using a fixed split.

tabarena-small suite
------------------
Adds 12 more TabArena classification datasets (<=5,000 rows), loaded via
their TabArena re-upload ids (OpenML Study 457) rather than any original
OpenML id -- unlike the four above, there is no earlier result on these
twelve to stay comparable with: anneal (46906), qsar-biodeg (46952),
website_phishing (46963), Fitness_Club (46927), MIC (46980),
Is-this-a-good-customer (46938), Marketing_Campaign (46940),
hazelnut-spread-contaminant-detection (46930), seismic-bumps (46956),
splice (46958), students_dropout_and_academic_success (46960), churn
(46915). Each uses its OpenML default target column. All twelve were
loaded end-to-end on 2026-10-23 (`uv run bench_datasets.py`):
898-5,000 rows, 6-111 features, 2-8 classes, labels without quote
artifacts, and every class present in train for seeds 42-46 (anneal's
smallest class has 8 rows). Missing values occur in Fitness_Club, MIC and
Marketing_Campaign.

These twelve plus the four real datasets above (blood-transfusion,
diabetes, credit-g, maternal-health-risk -- "synthetic" excluded) form the
"tabarena-small" suite registered in SUITES below, for runs across all 16
real datasets. The original five-dataset set stays registered as the
"default" suite, unchanged.

Offline fallback
------------------
"synthetic" (sklearn.datasets.make_classification, 900 rows, 10 numeric
features [6 informative, 2 redundant, 2 useless], binary) was added after
OpenML's API started returning persistent HTTP 504 Gateway Timeouts, blocking
all four datasets above end-to-end -- including on a TPU VM that had already
gotten as far as loading the TabFM checkpoint before failing on dataset
loading. It needs zero network access and is fully
deterministic (RANDOM_STATE), so the benchmark pipeline can be validated
end-to-end regardless of OpenML's availability. It's a pipeline-verification
fallback, not a replacement for the four real datasets above.
"""
import resource
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _load_openml(data_id, strip_target_quotes=False):
    """Fetch one OpenML dataset as (X: DataFrame, y: str ndarray)."""
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = data.data
    # Plain text columns (pandas "str"/object) break XGBoost and LightGBM;
    # treat them as categories, like the ARFF nominal columns already are.
    # The original 5 datasets have none, so their inputs are unchanged.
    text_cols = X.select_dtypes(exclude=["number", "category", "bool"]).columns
    # ISO date columns (Marketing_Campaign's Dt_Customer, 663 distinct
    # values) become days since 1970-01-01 instead: as a category, XGBoost
    # overfits on hundreds of levels.
    date_cols = [c for c in text_cols
                 if X[c].dropna().astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()]
    for c in date_cols:
        X[c] = (pd.to_datetime(X[c], format="%Y-%m-%d") - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    text_cols = text_cols.difference(date_cols)
    if len(text_cols):
        X = X.astype({c: "category" for c in text_cols})
    # Fail fast: astype(str) would turn a missing label into a class "nan".
    assert not data.target.isna().any(), f"OpenML {data_id}: target has missing values"
    y = data.target.astype(str)
    if strip_target_quotes:
        # maternal_health_risk's ARFF export embeds literal quotes in labels,
        # e.g. "'high risk'" (11 chars) instead of "high risk".
        y = y.str.strip("'")
    return X, y.to_numpy()


def _load_synthetic():
    """Generate the offline fallback dataset as (X: DataFrame, y: str ndarray)."""
    X, y = make_classification(
        n_samples=900,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = np.array([f"class_{label}" for label in y])
    return X, y


# name -> (openml data_id, strip_target_quotes); data_id=None means synthetic
# Kept as exactly these 5 entries (and this order) so any caller that still
# iterates `DATASETS` directly -- rather than through SUITES -- keeps
# running the original "default" suite unchanged, even if it's stale
# relative to this file (e.g. a benchmark script copied to a VM before a
# SUITES-aware version landed).
DATASETS = {
    "blood-transfusion": (1464, False),
    "diabetes": (37, False),
    "credit-g": (31, False),
    "maternal-health-risk": (46600, True),
    "synthetic": (None, False),
}

# TabArena re-uploads (OpenML Study 457), added for the "tabarena-small"
# suite -- see the tabarena-small suite docstring section above. strip_quotes
# is False for all twelve: their labels have no quote artifacts (checked
# 2026-10-23).
TABARENA_EXTRA = {
    "anneal": (46906, False),
    "qsar-biodeg": (46952, False),
    "website_phishing": (46963, False),
    "Fitness_Club": (46927, False),
    "MIC": (46980, False),
    "Is-this-a-good-customer": (46938, False),
    "Marketing_Campaign": (46940, False),
    "hazelnut-spread-contaminant-detection": (46930, False),
    "seismic-bumps": (46956, False),
    "splice": (46958, False),
    "students_dropout_and_academic_success": (46960, False),
    "churn": (46915, False),
}

# Full registry used by load_dataset() and this module's own self-test;
# benchmark scripts should iterate SUITES[name], not this dict, so growing
# it here never changes what an unmodified benchmark script runs.
ALL_DATASETS = {**DATASETS, **TABARENA_EXTRA}

SUITES = {
    "default": list(DATASETS),
    "tabarena-small": [name for name in DATASETS if name != "synthetic"] + list(TABARENA_EXTRA),
    # Re-run after the date-column fix in _load_openml (2026-10-28), the only
    # tabarena-small dataset whose inputs it changes.
    "dates-rerun": ["Marketing_Campaign"],
}


def peak_rss_bytes():
    """Peak resident memory of this process so far (it cannot be reset, so a
    later dataset's value includes everything before it)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # macOS: bytes, Linux: KiB


def utc_now():
    """ISO timestamp, for lining results up with the device-sampler logs."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_dataset(name, seed=RANDOM_STATE):
    """Load one registered dataset and return a stratified 80/20 split as
    (X_train, y_train, X_test, y_test): DataFrame, str ndarray, DataFrame,
    str ndarray -- the shapes TabFMClassifier.fit/predict_proba expect.
    `seed` only controls train_test_split's random_state -- the synthetic
    dataset's own make_classification call always uses RANDOM_STATE, so the
    dataset itself doesn't change across seeds, only the split."""
    data_id, strip_quotes = ALL_DATASETS[name]
    X, y = _load_synthetic() if data_id is None else _load_openml(data_id, strip_quotes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    return (
        X_train.reset_index(drop=True),
        y_train,
        X_test.reset_index(drop=True),
        y_test,
    )


def classification_metrics(y_test, probs, classes_):
    """log_loss and ROC-AUC for a predict_proba array, aligned to classes_
    (binary: AUC of the positive class classes_[1]; multiclass: one-vs-rest
    macro, sklearn's roc_auc_score(multi_class="ovr") default). Shared by
    tabfm_bench.py / tabpfn_bench.py / gbdt_bench.py so the three stay in
    sync."""
    classes_ = np.asarray(classes_)
    ll = log_loss(y_test, probs, labels=classes_)
    if len(classes_) == 2:
        auc = roc_auc_score(np.asarray(y_test) == classes_[1], probs[:, 1])
    else:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", labels=classes_)
    return ll, auc


if __name__ == "__main__":
    # seeds 42-46: the --seeds range the GBDT/TabFM/TabPFN benchmark scripts
    # use for their 5-split runs -- confirm the stratified split
    # works for every one of them, not just the loader's own default (42).
    SPLIT_SEEDS = range(42, 47)
    for name in ALL_DATASETS:
        X_train, y_train, X_test, y_test = load_dataset(name)
        n_num = X_train.select_dtypes(include="number").shape[1]
        n_cat = X_train.shape[1] - n_num
        n_missing = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
        class_counts = pd.Series(np.concatenate([y_train, y_test])).value_counts()

        print(f"\n=== {name} ===")
        print(f"X_train: {X_train.shape}  X_test: {X_test.shape}  "
              f"numeric_cols={n_num} cat_cols={n_cat}  missing_cells={n_missing}")
        print(f"train class balance: {dict(pd.Series(y_train).value_counts())}")
        print(f"test  class balance: {dict(pd.Series(y_test).value_counts())}")
        if class_counts.min() < 5:
            print(f"[WARN] minority class has <5 rows: "
                  f"{class_counts.idxmin()!r}={class_counts.min()}")

        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert set(np.unique(y_test)) <= set(np.unique(y_train)), (
            "test set has a class unseen in train"
        )
        for seed in SPLIT_SEEDS:
            X_tr, y_tr, X_te, y_te = load_dataset(name, seed=seed)
            assert set(np.unique(y_te)) <= set(np.unique(y_tr)), (
                f"{name} seed={seed}: test set has a class unseen in train"
            )
    print("\nDone.")
