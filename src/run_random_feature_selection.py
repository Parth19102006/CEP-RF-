"""Run a new, reproducible Phase 4 random-feature subset study.

This experiment is intentionally separate from the frozen baseline and Top-N
artifacts.  It creates exactly 70 uniformly random feature subsets from the
65 Phase 3 features and writes all outputs beneath reports/random_feature_selection.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
DEFAULT_PHASE3_METADATA = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "random_feature_selection"

TARGET_COLUMN = "Label_binary"
ORIGINAL_LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 65
MODEL_RANDOM_STATE = 42
MASTER_SAMPLING_SEED = 20_260_903
CLASS_LABELS = [0, 1]
SUBSET_ALLOCATION = {5: 10, 8: 10, 10: 10, 12: 10, 15: 10, 20: 10, 30: 10}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the reproducible Phase 4 random-feature subset study."
    )
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--phase3-metadata", type=Path, default=DEFAULT_PHASE3_METADATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
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


def load_phase3_frames(phase3_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the existing Phase 3 train/test data."""
    train_path = phase3_dir / "train.parquet"
    test_path = phase3_dir / "test.parquet"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3 file(s): " + ", ".join(missing))
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def validate_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Validate Phase 3 schema and finite values before fitting any model."""
    for split_name, frame in (("train", train_df), ("test", test_df)):
        required = [ORIGINAL_LABEL_COLUMN, TARGET_COLUMN, *feature_names]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{split_name} data is missing columns: {missing}")
        actual_order = [column for column in frame.columns if column in feature_names]
        if actual_order != feature_names:
            raise ValueError(f"{split_name} feature order differs from Phase 3 metadata.")
        target_values = set(frame[TARGET_COLUMN].dropna().astype(int).unique().tolist())
        if not target_values.issubset({0, 1}):
            raise ValueError(f"{split_name} target is not binary: {sorted(target_values)}")
        values = frame[feature_names].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{split_name} feature matrix contains non-finite values.")


def build_model() -> RandomForestClassifier:
    """Build the unchanged Phase 4 baseline/Top-N RF configuration."""
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )


def build_experiment_plan(feature_names: list[str]) -> list[dict[str, Any]]:
    """Generate the fixed 70-set plan using independent derived integer seeds."""
    plan: list[dict[str, Any]] = []
    experiment_number = 1
    seen_subsets: set[tuple[str, ...]] = set()
    for feature_count, repetitions in SUBSET_ALLOCATION.items():
        for _ in range(repetitions):
            sampling_seed = MASTER_SAMPLING_SEED + experiment_number - 1
            rng = np.random.default_rng(sampling_seed)
            selected = rng.choice(feature_names, size=feature_count, replace=False).tolist()
            selected_set = tuple(sorted(selected))
            if len(selected) != feature_count or len(set(selected)) != feature_count:
                raise ValueError(f"Random_{experiment_number:02d} has duplicate feature names.")
            if not set(selected).issubset(feature_names):
                raise ValueError(f"Random_{experiment_number:02d} selected an unknown feature.")
            if selected_set in seen_subsets:
                raise ValueError(f"Random_{experiment_number:02d} duplicated a prior subset.")
            seen_subsets.add(selected_set)
            plan.append(
                {
                    "experiment_id": f"Random_{experiment_number:02d}",
                    "feature_count": feature_count,
                    "sampling_seed": sampling_seed,
                    "selected_features": selected,
                }
            )
            experiment_number += 1
    expected_total = sum(SUBSET_ALLOCATION.values())
    if len(plan) != expected_total:
        raise ValueError(f"Expected {expected_total} experiments, generated {len(plan)}.")
    return plan


def evaluate_experiment(
    experiment: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit one planned subset and return its complete measured result."""
    features = experiment["selected_features"]
    x_train = train_df[features]
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    x_test = test_df[features]
    y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

    model = build_model()
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time_seconds = time.perf_counter() - train_start

    prediction_start = time.perf_counter()
    y_pred = model.predict(x_test)
    prediction_time_seconds = time.perf_counter() - prediction_start
    y_probability_attack = model.predict_proba(x_test)[:, 1]

    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).ravel()]
    return {
        **experiment,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability_attack)),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
    }


def validate_results(results: list[dict[str, Any]], feature_names: list[str]) -> None:
    """Validate complete, internally consistent output before writing artifacts."""
    expected_total = sum(SUBSET_ALLOCATION.values())
    if len(results) != expected_total:
        raise ValueError(f"Expected {expected_total} completed results, found {len(results)}.")
    counts = pd.Series([row["feature_count"] for row in results]).value_counts().to_dict()
    if counts != SUBSET_ALLOCATION:
        raise ValueError(f"Unexpected experiment allocation: {counts}")
    required_metrics = {
        "accuracy", "precision", "recall", "f1_score", "roc_auc", "true_positives",
        "true_negatives", "false_positives", "false_negatives", "training_time_seconds",
        "prediction_time_seconds",
    }
    for result in results:
        selected = result["selected_features"]
        if len(selected) != result["feature_count"] or len(set(selected)) != len(selected):
            raise ValueError(f"{result['experiment_id']} has an invalid feature subset.")
        if not set(selected).issubset(feature_names):
            raise ValueError(f"{result['experiment_id']} selected a feature outside the Phase 3 pool.")
        if not required_metrics.issubset(result):
            raise ValueError(f"{result['experiment_id']} is missing required metrics.")
        values = [float(result[name]) for name in required_metrics]
        if not np.isfinite(values).all():
            raise ValueError(f"{result['experiment_id']} has non-finite metrics.")


def write_artifacts(
    report_dir: Path,
    results: list[dict[str, Any]],
    feature_names: list[str],
    phase3_dir: Path,
    phase3_metadata_path: Path,
) -> None:
    """Write the complete reproducible study artifacts without touching prior reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_set_rows = [
        {
            "experiment_id": row["experiment_id"],
            "feature_count": row["feature_count"],
            "sampling_seed": row["sampling_seed"],
            "selected_features": json.dumps(row["selected_features"]),
        }
        for row in results
    ]
    result_columns = [
        "experiment_id", "feature_count", "sampling_seed", "accuracy", "precision", "recall",
        "f1_score", "roc_auc", "true_positives", "true_negatives", "false_positives",
        "false_negatives", "training_time_seconds", "prediction_time_seconds",
    ]
    pd.DataFrame(feature_set_rows).to_csv(report_dir / "random_feature_sets.csv", index=False)
    pd.DataFrame(results)[result_columns].to_csv(report_dir / "random_results.csv", index=False)

    model_config = build_model().get_params(deep=False)
    payload = {
        "study": "New controlled Phase 4 random-feature subset experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_random_experiment_artifacts_available": False,
        "reproduction_claim": "This is a newly defined study and does not reproduce an unrecoverable prior random experiment.",
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": project_relative(phase3_dir / "train.parquet"),
        "phase3_test_path": project_relative(phase3_dir / "test.parquet"),
        "phase3_metadata_path": project_relative(phase3_metadata_path),
        "target_column": TARGET_COLUMN,
        "feature_pool_count": len(feature_names),
        "feature_pool": feature_names,
        "subset_allocation": SUBSET_ALLOCATION,
        "experiment_count": len(results),
        "sampling_method": "Uniform random sampling without replacement from the frozen Phase 3 feature pool; no importance ranking is used.",
        "master_sampling_seed": MASTER_SAMPLING_SEED,
        "per_experiment_seed_rule": "sampling_seed = master_sampling_seed + experiment_number - 1",
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": model_config,
        "evaluation_metrics": [
            "accuracy", "precision", "recall", "f1_score", "roc_auc", "true_positives",
            "true_negatives", "false_positives", "false_negatives", "training_time_seconds",
            "prediction_time_seconds",
        ],
        "experiments": results,
    }
    (report_dir / "random_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    readme = f"""# Controlled Random Feature-Selection Study

## Purpose

This directory records a newly defined, reproducible 70-experiment Phase 4 study. The original random-feature-selection artifacts and methodology were unavailable, so this study must not be interpreted as a reproduction of an earlier run.

## Design

- Feature pool: the {len(feature_names)} final Phase 3 features in `{project_relative(phase3_metadata_path)}`.
- Allocation: 10 random subsets each with 5, 8, 10, 12, 15, 20, and 30 features (70 total).
- Sampling: uniform sampling without replacement; baseline feature importance/ranking is not used.
- Master sampling seed: `{MASTER_SAMPLING_SEED}`.
- Per-experiment seed: `master_sampling_seed + experiment_number - 1`; experiments are ordered `Random_01` through `Random_70`.
- Data: `{project_relative(phase3_dir / 'train.parquet')}` and `{project_relative(phase3_dir / 'test.parquet')}` with target `{TARGET_COLUMN}`.
- Preprocessing: the existing frozen Phase 3 representation is reused without scaling, PCA, sampling, or correlation filtering.

## Model

`RandomForestClassifier(n_estimators=100, criterion=\"gini\", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=\"sqrt\", bootstrap=True, class_weight=None, random_state=42, n_jobs=-1)`

## Evaluation

Each subset is trained on the existing Phase 3 training data and evaluated on the same Phase 3 test data. Saved metrics are accuracy, precision, recall, F1, ROC-AUC, TP, TN, FP, FN, training time, and prediction time.

## Files

- `random_feature_sets.csv`: exact selected feature names and sampling seed for each experiment.
- `random_results.csv`: one metric row per experiment.
- `random_results.json`: complete metadata, exact feature lists, configuration, and results.

## Reproducibility

From the project root, run:

```powershell
python src/run_random_feature_selection.py
```

The fixed feature pool, master seed, seed derivation rule, allocation, and RF configuration make the generated subset plan deterministic. Runtime measurements can vary by machine and concurrent system load.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    """Run all planned experiments and persist only the dedicated study outputs."""
    args = parse_args()
    phase3_dir = resolve_project_path(args.phase3_dir)
    phase3_metadata_path = resolve_project_path(args.phase3_metadata)
    report_dir = resolve_project_path(args.report_dir)
    feature_names = load_feature_names(phase3_metadata_path)
    train_df, test_df = load_phase3_frames(phase3_dir)
    validate_frames(train_df, test_df, feature_names)
    plan = build_experiment_plan(feature_names)

    results: list[dict[str, Any]] = []
    for experiment in plan:
        result = evaluate_experiment(experiment, train_df, test_df)
        results.append(result)
        print(
            f"{result['experiment_id']}: features={result['feature_count']}, "
            f"F1={result['f1_score']:.6f}, ROC-AUC={result['roc_auc']:.6f}"
        )

    validate_results(results, feature_names)
    write_artifacts(report_dir, results, feature_names, phase3_dir, phase3_metadata_path)
    print(f"Completed {len(results)} random-feature experiments.")
    print(f"Reports: {project_relative(report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
