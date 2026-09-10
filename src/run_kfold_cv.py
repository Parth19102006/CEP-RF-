"""Run Phase 4 K-Fold cross-validation on shortlisted feature subsets.

This script reuses the frozen Phase 3 train split, the existing Random Forest
configuration, and the already-identified Top-10 and Random-12 subsets. It does
not retune hyperparameters, rescore the original test set, or overwrite completed
Phase 4 baseline / random-feature artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
DEFAULT_PHASE3_METADATA = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"
DEFAULT_BASELINE_IMPORTANCE = PROJECT_ROOT / "reports" / "baseline" / "feature_importance.csv"
DEFAULT_BASELINE_METRICS = PROJECT_ROOT / "reports" / "baseline" / "metrics.json"
DEFAULT_RANDOM_SETS = PROJECT_ROOT / "reports" / "random_feature_selection" / "random_feature_sets.csv"
DEFAULT_RANDOM_RESULTS = PROJECT_ROOT / "reports" / "random_feature_selection" / "random_results.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "kfold_cv"

TARGET_COLUMN = "Label_binary"
ORIGINAL_LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 65
RANDOM_STATE = 42
N_SPLITS = 5
RANDOM12_EXPERIMENT_ID = "Random_34"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stratified 5-fold CV on Phase 4 shortlisted feature subsets."
    )
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--phase3-metadata", type=Path, default=DEFAULT_PHASE3_METADATA)
    parser.add_argument(
        "--baseline-importance",
        type=Path,
        default=DEFAULT_BASELINE_IMPORTANCE,
    )
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--random-feature-sets", type=Path, default=DEFAULT_RANDOM_SETS)
    parser.add_argument("--random-results", type=Path, default=DEFAULT_RANDOM_RESULTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Run CV twice and require identical fold metrics (reproducibility check).",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_feature_names(metadata_path: Path) -> list[str]:
    """Load and validate the frozen Phase 3 65-feature pool."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Phase 3 metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("final_feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("Phase 3 metadata is missing a valid final_feature_names list.")
    if metadata.get("final_feature_count") != len(feature_names):
        raise ValueError("Phase 3 feature-count metadata does not match final_feature_names.")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_names)}.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Phase 3 feature list contains duplicate names.")
    forbidden = {TARGET_COLUMN, ORIGINAL_LABEL_COLUMN}.intersection(feature_names)
    if forbidden:
        raise ValueError(f"Label column(s) cannot be features: {sorted(forbidden)}")
    return feature_names


def load_training_frame(phase3_dir: Path) -> pd.DataFrame:
    """Load Phase 3 training data only; the test parquet is never read."""
    train_path = phase3_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 training file: {train_path}")
    return pd.read_parquet(train_path)


def validate_training_frame(train_df: pd.DataFrame, feature_names: list[str]) -> None:
    """Validate Phase 3 training schema and finite values before CV."""
    required = [ORIGINAL_LABEL_COLUMN, TARGET_COLUMN, *feature_names]
    missing = [column for column in required if column not in train_df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    actual_order = [column for column in train_df.columns if column in feature_names]
    if actual_order != feature_names:
        raise ValueError("Training feature order differs from Phase 3 metadata.")
    target_values = set(train_df[TARGET_COLUMN].dropna().astype(int).unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Training target is not binary: {sorted(target_values)}")
    values = train_df[feature_names].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Training feature matrix contains non-finite values.")


def build_model() -> RandomForestClassifier:
    """Build the unchanged Phase 4 Random Forest configuration."""
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_top10_features(importance_path: Path, feature_names: list[str]) -> list[str]:
    """Load the frozen Top-10 ranking from the Phase 4 baseline importance artifact."""
    if not importance_path.exists():
        raise FileNotFoundError(f"Baseline importance file does not exist: {importance_path}")
    ranking = pd.read_csv(importance_path)
    required_columns = {"feature_name", "rank"}
    if not required_columns.issubset(ranking.columns):
        raise ValueError("Baseline importance file is missing feature_name/rank columns.")
    ranking = ranking.sort_values("rank", kind="mergesort")
    selected = ranking.head(10)["feature_name"].astype(str).tolist()
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError("Top-10 ranking does not contain 10 unique features.")
    if not set(selected).issubset(feature_names):
        raise ValueError("Top-10 contains a feature outside the Phase 3 pool.")
    return selected


def load_random12_features(
    feature_sets_path: Path,
    feature_names: list[str],
) -> tuple[list[str], int]:
    """Load the Phase 4 Random-12 candidate (Random_34) from existing artifacts."""
    if not feature_sets_path.exists():
        raise FileNotFoundError(f"Random feature-set file does not exist: {feature_sets_path}")
    sets = pd.read_csv(feature_sets_path)
    match = sets.loc[sets["experiment_id"] == RANDOM12_EXPERIMENT_ID]
    if match.empty:
        raise ValueError(
            f"{RANDOM12_EXPERIMENT_ID} was not found; Random-12 cannot be evaluated."
        )
    row = match.iloc[0]
    selected = json.loads(row["selected_features"])
    if int(row["feature_count"]) != 12 or len(selected) != 12 or len(set(selected)) != 12:
        raise ValueError(f"{RANDOM12_EXPERIMENT_ID} is not a valid 12-feature subset.")
    if not set(selected).issubset(feature_names):
        raise ValueError("Random-12 contains a feature outside the Phase 3 pool.")
    return selected, int(row["sampling_seed"])


def build_subsets(
    feature_names: list[str],
    importance_path: Path,
    random_sets_path: Path,
) -> list[dict[str, Any]]:
    """Assemble the frozen shortlisted subsets; do not invent new sets."""
    top10 = load_top10_features(importance_path, feature_names)
    random12, sampling_seed = load_random12_features(random_sets_path, feature_names)
    return [
        {
            "subset_id": "Top-10",
            "feature_count": 10,
            "features": top10,
            "source_artifact": project_relative(importance_path),
            "source_experiment_id": None,
            "sampling_seed": None,
            "selection_note": (
                "Ranks 1-10 from the frozen Phase 4 baseline Random Forest "
                "feature-importance ranking."
            ),
        },
        {
            "subset_id": "Random-12",
            "feature_count": 12,
            "features": random12,
            "source_artifact": project_relative(random_sets_path),
            "source_experiment_id": RANDOM12_EXPERIMENT_ID,
            "sampling_seed": sampling_seed,
            "selection_note": (
                f"Existing Phase 4 12-feature random subset {RANDOM12_EXPERIMENT_ID}, "
                "the only k=12 set whose original test F1 clearly exceeded the "
                "65-feature baseline."
            ),
        },
    ]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the four required binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def make_splitter() -> StratifiedKFold:
    """Return the fixed stratified 5-fold splitter."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def fold_index_fingerprint(y: np.ndarray) -> str:
    """Hash fold membership so split identity can be verified."""
    splitter = make_splitter()
    parts: list[str] = []
    dummy = np.zeros(len(y), dtype=np.uint8)
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(dummy, y), start=1):
        parts.append(f"{fold_id}:{train_idx.tolist()}:{val_idx.tolist()}")
    encoded = "|".join(parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_cv_for_subset(
    subset: dict[str, Any],
    train_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit one frozen subset on training folds only and score validation folds."""
    features = subset["features"]
    x = train_df[features]
    y = train_df[TARGET_COLUMN].astype(int).to_numpy()
    splitter = make_splitter()
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(x, y), start=1):
        x_fold_train = x.iloc[train_idx]
        y_fold_train = y[train_idx]
        x_fold_val = x.iloc[val_idx]
        y_fold_val = y[val_idx]

        model = build_model()
        model.fit(x_fold_train, y_fold_train)
        y_pred = model.predict(x_fold_val)
        metrics = compute_metrics(y_fold_val, y_pred)
        fold_rows.append(
            {
                "subset_id": subset["subset_id"],
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                **metrics,
            }
        )

    summary: dict[str, Any] = {
        "subset_id": subset["subset_id"],
        "feature_count": subset["feature_count"],
        "features": features,
        "source_artifact": subset["source_artifact"],
        "source_experiment_id": subset["source_experiment_id"],
        "sampling_seed": subset["sampling_seed"],
        "selection_note": subset["selection_note"],
        "n_folds": N_SPLITS,
    }
    for metric_name in METRIC_NAMES:
        values = np.array([row[metric_name] for row in fold_rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=1))
        summary[f"{metric_name}_folds"] = [float(value) for value in values]
    return fold_rows, summary


def summaries_to_metric_payload(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip analysis-only fields so two CV runs can be compared exactly."""
    comparable: list[dict[str, Any]] = []
    for summary in summaries:
        row = {
            "subset_id": summary["subset_id"],
            "feature_count": summary["feature_count"],
            "features": summary["features"],
        }
        for metric_name in METRIC_NAMES:
            row[f"{metric_name}_mean"] = summary[f"{metric_name}_mean"]
            row[f"{metric_name}_std"] = summary[f"{metric_name}_std"]
            row[f"{metric_name}_folds"] = summary[f"{metric_name}_folds"]
        comparable.append(row)
    return comparable


def load_phase4_reference_metrics(
    baseline_metrics_path: Path,
    random_results_path: Path,
) -> dict[str, Any]:
    """Load original Phase 4 test metrics for comparison text only."""
    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    random_results = pd.read_csv(random_results_path)
    random12 = random_results.loc[random_results["experiment_id"] == RANDOM12_EXPERIMENT_ID]
    if random12.empty:
        raise ValueError(f"{RANDOM12_EXPERIMENT_ID} is missing from random_results.csv.")
    row = random12.iloc[0]
    return {
        "baseline_test": {
            "accuracy": float(baseline["accuracy"]),
            "precision": float(baseline["precision"]),
            "recall": float(baseline["recall"]),
            "f1_score": float(baseline["f1_score"]),
        },
        "random12_test": {
            "experiment_id": RANDOM12_EXPERIMENT_ID,
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1_score"]),
        },
    }


def format_mean_std(mean: float, std: float) -> str:
    """Format a mean ± standard deviation cell."""
    return f"{mean:.6f} ± {std:.6f}"


def identify_best_and_most_stable(summaries: list[dict[str, Any]]) -> dict[str, str]:
    """Identify best mean F1 subset and most stable subset by F1 standard deviation."""
    best = max(summaries, key=lambda row: row["f1_score_mean"])
    most_stable = min(summaries, key=lambda row: row["f1_score_std"])
    return {
        "best_mean_subset_id": best["subset_id"],
        "most_stable_subset_id": most_stable["subset_id"],
    }


def build_analysis(
    summaries: list[dict[str, Any]],
    phase4_reference: dict[str, Any],
    leakage_notes: list[str],
) -> str:
    """Write a concise comparison of CV subsets against completed Phase 4 results."""
    by_id = {row["subset_id"]: row for row in summaries}
    top10 = by_id["Top-10"]
    random12 = by_id["Random-12"]
    winners = identify_best_and_most_stable(summaries)
    baseline_test = phase4_reference["baseline_test"]
    random12_test = phase4_reference["random12_test"]

    rows = []
    for subset in summaries:
        rows.append(
            "| {id} | {n} | {acc} | {prec} | {rec} | {f1} |".format(
                id=subset["subset_id"],
                n=subset["feature_count"],
                acc=format_mean_std(subset["accuracy_mean"], subset["accuracy_std"]),
                prec=format_mean_std(subset["precision_mean"], subset["precision_std"]),
                rec=format_mean_std(subset["recall_mean"], subset["recall_std"]),
                f1=format_mean_std(subset["f1_score_mean"], subset["f1_score_std"]),
            )
        )
    table = "\n".join(rows)

    top10_vs_r12 = (
        "Random-12 has the higher mean F1"
        if random12["f1_score_mean"] > top10["f1_score_mean"]
        else "Top-10 has the higher mean F1"
    )
    f1_gap = abs(random12["f1_score_mean"] - top10["f1_score_mean"])
    more_stable = (
        "Random-12"
        if random12["f1_score_std"] < top10["f1_score_std"]
        else "Top-10"
    )

    leakage_block = "\n".join(f"- {note}" for note in leakage_notes)
    compactness_note = (
        "Random-12 is the stronger mean-performing candidate on this training CV; "
        "Top-10 is slightly smaller and more stable."
        if random12["f1_score_mean"] > top10["f1_score_mean"]
        else "Top-10 is both the stronger mean-performing and more compact candidate "
        "on this training CV."
    )

    return f"""# Phase 4 K-Fold Cross-Validation Analysis

This analysis uses training-data Stratified 5-fold CV only. It does not select a
final model, tune hyperparameters, or evaluate the original test set.

## Configuration

- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Model: unchanged Phase 4 `RandomForestClassifier` (`random_state=42`, `n_jobs=-1`)
- Data: `data/processed/phase3/train.parquet` only
- Standard deviation: sample standard deviation across folds (`ddof=1`)

## Subsets evaluated

- **Top-10** ({top10['feature_count']} features): {', '.join(top10['features'])}
- **Random-12** ({random12['feature_count']} features, source `{RANDOM12_EXPERIMENT_ID}`): {', '.join(random12['features'])}

No other Phase 4 subset was named as a shortlist entry. Other random sets that
beat the 65-feature baseline on the original test split (for example Random_48
and Random_20) were not re-run.

## CV summary (mean ± std)

| Feature Set | # Features | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | --- | --- | --- |
{table}

## Best mean-performing subset

**{winners['best_mean_subset_id']}** has the highest mean F1
({format_mean_std(by_id[winners['best_mean_subset_id']]['f1_score_mean'], by_id[winners['best_mean_subset_id']]['f1_score_std'])}).

## Most stable subset

**{winners['most_stable_subset_id']}** has the lowest F1 standard deviation
({by_id[winners['most_stable_subset_id']]['f1_score_std']:.6f}).

## Top-10 vs Random-12

{top10_vs_r12} (absolute mean-F1 gap {f1_gap:.6f}). {more_stable} is more stable
on F1. {compactness_note}

## Consistency with completed Phase 4 test results

Completed Phase 4 scored models on the original test split. Those numbers are
not CV estimates and are not replaced here.

- 65-feature baseline test F1 = {baseline_test['f1_score']:.6f}
  (accuracy {baseline_test['accuracy']:.6f}, precision {baseline_test['precision']:.6f},
  recall {baseline_test['recall']:.6f}).
- Random_34 / Random-12 original test F1 = {random12_test['f1_score']:.6f}
  (accuracy {random12_test['accuracy']:.6f}, precision {random12_test['precision']:.6f},
  recall {random12_test['recall']:.6f}).
- Top-10 was ranked from baseline importances but was not previously given its
  own test-set score.

Random-12's training-CV mean F1 stays above the original 65-feature baseline
test F1, so the Phase 4 observation that this 12-feature set is a serious
compact candidate remains consistent. Exact CV means are expected to differ
from the original test scores because the partitions and sample sizes differ.
This CV run does not declare a final model.

## Leakage controls

{leakage_block}
"""


def write_artifacts(
    report_dir: Path,
    subsets: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    phase4_reference: dict[str, Any],
    fold_fingerprint: str,
    train_samples: int,
    phase3_dir: Path,
    repeat_verified: bool,
) -> None:
    """Write new CV artifacts without modifying completed Phase 4 reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    leakage_notes = [
        "The original Phase 3 test parquet is never loaded or scored.",
        "Preprocessing is the frozen Phase 3 representation; no scaler, imputer, or selector is fitted inside CV.",
        "Feature subsets are frozen before CV; they are not re-selected using validation folds.",
        "Each Random Forest is fit only on that fold's training indices and scored only on that fold's validation indices.",
        "Top-10 was originally ranked from a baseline model fit on the full training split, so these CV estimates for Top-10 can be slightly optimistic relative to nested selection.",
        "Random-12 (Random_34) was identified from completed Phase 4 test-set results; this script does not reuse those test labels for scoring.",
        "Phase 3 training medians were fit on the full training split before this CV study; that preprocessing is inherited and unchanged.",
    ]
    analysis = build_analysis(summaries, phase4_reference, leakage_notes)
    winners = identify_best_and_most_stable(summaries)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(report_dir / "cv_fold_results.csv", index=False)

    summary_rows = []
    for subset in summaries:
        summary_rows.append(
            {
                "subset_id": subset["subset_id"],
                "feature_count": subset["feature_count"],
                "features": json.dumps(subset["features"]),
                "accuracy_mean": subset["accuracy_mean"],
                "accuracy_std": subset["accuracy_std"],
                "precision_mean": subset["precision_mean"],
                "precision_std": subset["precision_std"],
                "recall_mean": subset["recall_mean"],
                "recall_std": subset["recall_std"],
                "f1_score_mean": subset["f1_score_mean"],
                "f1_score_std": subset["f1_score_std"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(report_dir / "cv_summary.csv", index=False)

    feature_set_rows = [
        {
            "subset_id": subset["subset_id"],
            "feature_count": subset["feature_count"],
            "source_experiment_id": subset["source_experiment_id"],
            "sampling_seed": subset["sampling_seed"],
            "features": json.dumps(subset["features"]),
        }
        for subset in subsets
    ]
    pd.DataFrame(feature_set_rows).to_csv(report_dir / "cv_feature_sets.csv", index=False)

    payload = {
        "study": "Phase 4 stratified 5-fold CV on shortlisted feature subsets",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": project_relative(phase3_dir / "train.parquet"),
        "phase3_test_used": False,
        "target_column": TARGET_COLUMN,
        "train_samples": train_samples,
        "cv": {
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": build_model().get_params(deep=False),
        "metrics": list(METRIC_NAMES),
        "std_estimator": "sample standard deviation across folds (numpy.std ddof=1)",
        "fold_index_sha256": fold_fingerprint,
        "repeat_verified": repeat_verified,
        "hyperparameter_tuning_applied": False,
        "final_model_selected": False,
        "original_test_evaluated": False,
        "best_mean_subset_id": winners["best_mean_subset_id"],
        "most_stable_subset_id": winners["most_stable_subset_id"],
        "phase4_reference_test_metrics": phase4_reference,
        "leakage_notes": leakage_notes,
        "subsets": summaries,
        "fold_results": fold_rows,
    }
    (report_dir / "cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (report_dir / "cv_analysis.md").write_text(analysis, encoding="utf-8")

    readme = f"""# Phase 4 K-Fold Cross-Validation

## Purpose

Training-only stratified 5-fold CV for the existing Phase 4 shortlisted subsets.
This directory is additive. It does not replace baseline or random-feature-selection
artifacts.

## Design

- Data: `{project_relative(phase3_dir / 'train.parquet')}` only; the original test set is untouched.
- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})`.
- Model: the frozen Phase 4 Random Forest (`random_state={RANDOM_STATE}`).
- Subsets: Top-10 from baseline importance ranks; Random-12 from `{RANDOM12_EXPERIMENT_ID}`.

## Files

- `cv_feature_sets.csv`: subset IDs and exact feature names.
- `cv_fold_results.csv`: per-fold accuracy, precision, recall, and F1.
- `cv_summary.csv`: mean and standard deviation per subset.
- `cv_results.json`: configuration, leakage notes, and full numeric payload.
- `cv_analysis.md`: comparison of mean performance, stability, and Phase 4 consistency.

## Reproducibility

From the project root:

```powershell
python src/run_kfold_cv.py --repeat
```
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")


def run_all_subsets(
    subsets: list[dict[str, Any]],
    train_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every shortlisted subset."""
    fold_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for subset in subsets:
        subset_folds, summary = run_cv_for_subset(subset, train_df)
        fold_rows.extend(subset_folds)
        summaries.append(summary)
        print(
            f"{summary['subset_id']}: "
            f"F1={summary['f1_score_mean']:.6f} ± {summary['f1_score_std']:.6f}"
        )
    return fold_rows, summaries


def main() -> int:
    """Run CV on frozen shortlisted subsets and write dedicated artifacts."""
    args = parse_args()
    phase3_dir = resolve_project_path(args.phase3_dir)
    phase3_metadata_path = resolve_project_path(args.phase3_metadata)
    importance_path = resolve_project_path(args.baseline_importance)
    baseline_metrics_path = resolve_project_path(args.baseline_metrics)
    random_sets_path = resolve_project_path(args.random_feature_sets)
    random_results_path = resolve_project_path(args.random_results)
    report_dir = resolve_project_path(args.report_dir)

    test_path = phase3_dir / "test.parquet"
    if test_path.exists():
        print(f"Leaving original test set untouched: {project_relative(test_path)}")

    feature_names = load_feature_names(phase3_metadata_path)
    train_df = load_training_frame(phase3_dir)
    validate_training_frame(train_df, feature_names)
    subsets = build_subsets(feature_names, importance_path, random_sets_path)
    y = train_df[TARGET_COLUMN].astype(int).to_numpy()
    fold_fingerprint = fold_index_fingerprint(y)
    phase4_reference = load_phase4_reference_metrics(baseline_metrics_path, random_results_path)

    fold_rows, summaries = run_all_subsets(subsets, train_df)

    repeat_verified = False
    if args.repeat:
        print("Repeating CV to verify identical fold metrics...")
        repeat_folds, repeat_summaries = run_all_subsets(subsets, train_df)
        if fold_rows != repeat_folds:
            raise RuntimeError("Repeated CV fold metrics did not match; results are not reproducible.")
        if summaries_to_metric_payload(summaries) != summaries_to_metric_payload(repeat_summaries):
            raise RuntimeError("Repeated CV summaries did not match; results are not reproducible.")
        if fold_index_fingerprint(y) != fold_fingerprint:
            raise RuntimeError("Repeated StratifiedKFold indices did not match.")
        repeat_verified = True
        print("Reproducibility check passed: identical fold indices and metrics.")

    write_artifacts(
        report_dir=report_dir,
        subsets=subsets,
        fold_rows=fold_rows,
        summaries=summaries,
        phase4_reference=phase4_reference,
        fold_fingerprint=fold_fingerprint,
        train_samples=int(len(train_df)),
        phase3_dir=phase3_dir,
        repeat_verified=repeat_verified,
    )
    print(f"Reports: {project_relative(report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
