"""Run Phase 4 Random Forest selected-feature experiments.

This script uses the already-generated baseline Random Forest impurity-based
feature-importance ranking to construct reproducible reduced feature subsets.
It preserves the Phase 3 train/test split and does not tune hyperparameters or
overwrite baseline artifacts.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
PHASE3_METADATA_PATH = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"
BASELINE_REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"
BASELINE_FEATURE_IMPORTANCE_PATH = BASELINE_REPORT_DIR / "feature_importance.csv"
BASELINE_METRICS_PATH = BASELINE_REPORT_DIR / "metrics.json"
BASELINE_METADATA_PATH = BASELINE_REPORT_DIR / "baseline_metadata.json"
HIGH_CORRELATION_PAIRS_PATH = PROJECT_ROOT / "reports" / "dataset_audit" / "high_correlation_pairs.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "feature_selection"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Label_binary"
ORIGINAL_LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 65
RANDOM_STATE = 42
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Benign", "DDoS/Attack"]
SUBSET_SIZES = [50, 40, 30, 20, 10]
REFERENCE_SUBSET_SIZE = 65


def project_relative(path: Path) -> str:
    """Return a project-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_phase3_feature_names(metadata_path: Path) -> list[str]:
    """Load the authoritative Phase 3 feature list."""
    metadata = read_json(metadata_path)
    feature_names = metadata.get("final_feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("Phase 3 metadata is missing a valid final_feature_names list.")
    if metadata.get("final_feature_count") != len(feature_names):
        raise ValueError("Phase 3 final_feature_count does not match final_feature_names length.")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} Phase 3 features, found {len(feature_names)}."
        )
    forbidden = {TARGET_COLUMN, ORIGINAL_LABEL_COLUMN}.intersection(feature_names)
    if forbidden:
        raise ValueError(f"Target/label column(s) must not appear as features: {sorted(forbidden)}")
    return feature_names


def load_phase3_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 3 train/test data after checking file existence."""
    train_path = PHASE3_DIR / "train.parquet"
    test_path = PHASE3_DIR / "test.parquet"
    missing = [project_relative(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3 train/test file(s): " + ", ".join(missing))
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def validate_frame_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Validate target columns, feature schema, and finite feature values."""
    for split_name, df in (("train", train_df), ("test", test_df)):
        for column in (ORIGINAL_LABEL_COLUMN, TARGET_COLUMN):
            if column not in df.columns:
                raise ValueError(f"{column!r} is missing from the {split_name} dataframe.")

        missing_features = [column for column in feature_names if column not in df.columns]
        if missing_features:
            raise ValueError(
                f"{split_name} dataframe is missing feature column(s): "
                + ", ".join(missing_features)
            )

        actual_feature_order = [column for column in df.columns if column in feature_names]
        if actual_feature_order != feature_names:
            raise ValueError(f"{split_name} feature columns do not match Phase 3 metadata order.")

        target_values = set(df[TARGET_COLUMN].dropna().astype(int).unique().tolist())
        if not target_values.issubset({0, 1}):
            raise ValueError(
                f"{split_name} target contains values outside {{0, 1}}: {sorted(target_values)}"
            )

        feature_values = df[feature_names].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(feature_values).all():
            raise ValueError(
                f"{split_name} feature matrix contains NaN, +Inf, or -Inf values."
            )

    if list(train_df[feature_names].columns) != list(test_df[feature_names].columns):
        raise ValueError("Train/test feature schemas differ.")


def load_baseline_feature_importance(feature_names: list[str]) -> pd.DataFrame:
    """Load and validate the frozen baseline feature-importance artifact."""
    if not BASELINE_FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline feature importance artifact does not exist: {BASELINE_FEATURE_IMPORTANCE_PATH}"
        )
    importance_df = pd.read_csv(BASELINE_FEATURE_IMPORTANCE_PATH)
    expected_columns = {"feature_name", "importance", "rank"}
    if not expected_columns.issubset(importance_df.columns):
        raise ValueError(
            "Baseline feature importance must contain columns: "
            + ", ".join(sorted(expected_columns))
        )
    if len(importance_df) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} baseline features, found {len(importance_df)}."
        )
    if importance_df["feature_name"].duplicated().any():
        duplicates = importance_df.loc[
            importance_df["feature_name"].duplicated(), "feature_name"
        ].tolist()
        raise ValueError(f"Duplicate baseline feature importance rows: {duplicates}")

    importance_df = importance_df.copy()
    importance_df["rank"] = importance_df["rank"].astype(int)
    importance_df["importance"] = importance_df["importance"].astype(float)
    importance_df = importance_df.sort_values("rank", kind="mergesort").reset_index(drop=True)

    expected_ranks = list(range(1, EXPECTED_FEATURE_COUNT + 1))
    if importance_df["rank"].tolist() != expected_ranks:
        raise ValueError("Baseline feature importance ranks must be contiguous from 1 to 65.")
    if set(importance_df["feature_name"]) != set(feature_names):
        raise ValueError("Baseline feature importance features do not match Phase 3 features.")
    if not np.isfinite(importance_df["importance"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Baseline feature importance contains non-finite values.")
    return importance_df[["feature_name", "importance", "rank"]]


def load_high_correlation_summary() -> dict[str, Any]:
    """Load Phase 2 high-correlation metadata for documentation only."""
    if not HIGH_CORRELATION_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Phase 2 high-correlation artifact does not exist: {HIGH_CORRELATION_PAIRS_PATH}"
        )
    pairs = pd.read_csv(HIGH_CORRELATION_PAIRS_PATH)
    return {
        "source": project_relative(HIGH_CORRELATION_PAIRS_PATH),
        "pair_count": int(len(pairs)),
        "used_for_filtering": False,
        "note": (
            "High-correlation pairs were inspected as Phase 2 context only. "
            "No correlation-based filtering was applied in this experiment."
        ),
    }


def build_candidate_subsets(importance_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create full and reduced subsets from baseline importance ranks."""
    subsets = {f"Baseline_{REFERENCE_SUBSET_SIZE}": importance_df.head(REFERENCE_SUBSET_SIZE)}
    for size in SUBSET_SIZES:
        subsets[f"Top_{size}"] = importance_df.head(size)

    for name, subset in subsets.items():
        expected_size = int(name.rsplit("_", 1)[1])
        if len(subset) != expected_size:
            raise ValueError(f"{name} contains {len(subset)} features, expected {expected_size}.")
    return subsets


def write_selected_features(subsets: dict[str, pd.DataFrame]) -> None:
    """Write the exact selected feature names and ranks for every subset."""
    rows: list[dict[str, Any]] = []
    for subset_name, subset_df in subsets.items():
        for row in subset_df.itertuples(index=False):
            rows.append(
                {
                    "method": "baseline_random_forest_mdi_rank_top_k",
                    "subset_name": subset_name,
                    "feature_rank": int(row.rank),
                    "feature_name": str(row.feature_name),
                    "baseline_importance": float(row.importance),
                }
            )
    pd.DataFrame(rows).to_csv(REPORT_DIR / "selected_features.csv", index=False)


def build_model() -> RandomForestClassifier:
    """Create the fixed Random Forest configuration used by the baseline."""
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability_attack: np.ndarray,
) -> dict[str, Any]:
    """Calculate binary and per-class metrics."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability_attack)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "class_names": CLASS_NAMES,
            "matrix": matrix.tolist(),
        },
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_evaluate_subset(
    subset_name: str,
    feature_names: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train a fresh selected-feature RF and evaluate it on the Phase 3 test split."""
    x_train = train_df[feature_names]
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    x_test = test_df[feature_names]
    y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

    model = build_model()
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(x_test)
    test_prediction_time_seconds = time.perf_counter() - predict_start

    proba_start = time.perf_counter()
    y_probability_attack = model.predict_proba(x_test)[:, 1]
    test_predict_proba_time_seconds = time.perf_counter() - proba_start

    metrics = evaluate_predictions(y_test, y_pred, y_probability_attack)
    model_path = MODEL_DIR / f"rf_top{len(feature_names)}.joblib"
    joblib.dump(model, model_path)

    return {
        "model": subset_name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_path": project_relative(model_path),
        "training_time_seconds": float(training_time_seconds),
        "test_prediction_time_seconds": float(test_prediction_time_seconds),
        "test_predict_proba_time_seconds": float(test_predict_proba_time_seconds),
        **metrics,
    }


def baseline_comparison_row(baseline_metrics: dict[str, Any], baseline_metadata: dict[str, Any]) -> dict[str, Any]:
    """Create the comparison row for the existing frozen baseline."""
    matrix = baseline_metrics.get("confusion_matrix", {}).get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 2:
        raise ValueError("Baseline metrics are missing a valid 2x2 confusion matrix.")
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    timings = baseline_metadata.get("timings", {})
    return {
        "model": "Baseline_65",
        "feature_count": int(baseline_metadata.get("feature_count", EXPECTED_FEATURE_COUNT)),
        "feature_names": baseline_metadata.get("feature_names", []),
        "model_path": baseline_metadata.get("artifacts", {}).get("model", "models/rf_baseline.joblib"),
        "accuracy": float(baseline_metrics["accuracy"]),
        "precision": float(baseline_metrics["precision"]),
        "recall": float(baseline_metrics["recall"]),
        "f1_score": float(baseline_metrics["f1_score"]),
        "roc_auc": float(baseline_metrics["roc_auc"]),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_time_seconds": float(timings["training_time_seconds"]),
        "test_prediction_time_seconds": float(timings["test_prediction_time_seconds"]),
        "test_predict_proba_time_seconds": timings.get("test_predict_proba_time_seconds"),
        "confusion_matrix": baseline_metrics.get("confusion_matrix"),
        "per_class": baseline_metrics.get("per_class"),
    }


def build_comparison_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create the flat comparison table with deltas from baseline."""
    baseline = rows[0]
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        feature_reduction_pct = (
            (REFERENCE_SUBSET_SIZE - row["feature_count"]) / REFERENCE_SUBSET_SIZE * 100.0
        )
        flat_rows.append(
            {
                "model": row["model"],
                "feature_count": row["feature_count"],
                "feature_reduction_pct": float(feature_reduction_pct),
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
                "roc_auc": row["roc_auc"],
                "true_negatives": row["true_negatives"],
                "false_positives": row["false_positives"],
                "false_negatives": row["false_negatives"],
                "true_positives": row["true_positives"],
                "training_time_seconds": row["training_time_seconds"],
                "test_prediction_time_seconds": row["test_prediction_time_seconds"],
                "accuracy_delta_vs_baseline": row["accuracy"] - baseline["accuracy"],
                "precision_delta_vs_baseline": row["precision"] - baseline["precision"],
                "recall_delta_vs_baseline": row["recall"] - baseline["recall"],
                "f1_delta_vs_baseline": row["f1_score"] - baseline["f1_score"],
                "roc_auc_delta_vs_baseline": row["roc_auc"] - baseline["roc_auc"],
                "false_positive_delta_vs_baseline": row["false_positives"] - baseline["false_positives"],
                "false_negative_delta_vs_baseline": row["false_negatives"] - baseline["false_negatives"],
                "training_time_delta_pct_vs_baseline": (
                    (row["training_time_seconds"] - baseline["training_time_seconds"])
                    / baseline["training_time_seconds"]
                    * 100.0
                ),
                "prediction_time_delta_pct_vs_baseline": (
                    (row["test_prediction_time_seconds"] - baseline["test_prediction_time_seconds"])
                    / baseline["test_prediction_time_seconds"]
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(flat_rows)


def choose_tradeoff_model(comparison_df: pd.DataFrame) -> pd.Series:
    """Choose a balanced candidate using detection quality and cost, not accuracy alone."""
    candidates = comparison_df[comparison_df["model"] != "Baseline_65"].copy()
    tolerated_f1_drop = 0.005
    baseline_f1 = float(comparison_df.loc[comparison_df["model"] == "Baseline_65", "f1_score"].iloc[0])
    baseline_recall = float(comparison_df.loc[comparison_df["model"] == "Baseline_65", "recall"].iloc[0])
    maintained = candidates[
        (candidates["f1_score"] >= baseline_f1 - tolerated_f1_drop)
        & (candidates["recall"] >= baseline_recall - tolerated_f1_drop)
    ]
    pool = maintained if not maintained.empty else candidates
    pool = pool.sort_values(
        by=[
            "feature_reduction_pct",
            "f1_score",
            "recall",
            "roc_auc",
            "false_negatives",
            "training_time_seconds",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    return pool.iloc[0]


def write_detailed_evaluation(rows: list[dict[str, Any]]) -> None:
    """Write nested per-model metrics, including confusion matrices and per-class metrics."""
    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                "model": row["model"],
                "feature_count": row["feature_count"],
                "feature_names": row["feature_names"],
                "model_path": row["model_path"],
                "training_time_seconds": row["training_time_seconds"],
                "test_prediction_time_seconds": row["test_prediction_time_seconds"],
                "test_predict_proba_time_seconds": row["test_predict_proba_time_seconds"],
                "metrics": {
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "roc_auc": row["roc_auc"],
                    "true_negatives": row["true_negatives"],
                    "false_positives": row["false_positives"],
                    "false_negatives": row["false_negatives"],
                    "true_positives": row["true_positives"],
                    "confusion_matrix": row["confusion_matrix"],
                    "per_class": row["per_class"],
                },
            }
        )
    (REPORT_DIR / "model_evaluation_details.json").write_text(
        json.dumps(serializable_rows, indent=2),
        encoding="utf-8",
    )


def write_metadata(
    subsets: dict[str, pd.DataFrame],
    feature_names: list[str],
    correlation_summary: dict[str, Any],
    tradeoff_model: str,
) -> None:
    """Write machine-readable feature-selection metadata."""
    selected_feature_lists = {
        subset_name: subset_df["feature_name"].tolist()
        for subset_name, subset_df in subsets.items()
    }
    metadata = {
        "phase": "Phase 4 selected-feature Random Forest experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "source_baseline_artifact": project_relative(BASELINE_FEATURE_IMPORTANCE_PATH),
        "phase3_metadata_path": project_relative(PHASE3_METADATA_PATH),
        "phase3_train_path": project_relative(PHASE3_DIR / "train.parquet"),
        "phase3_test_path": project_relative(PHASE3_DIR / "test.parquet"),
        "original_feature_count": len(feature_names),
        "candidate_subset_sizes": [REFERENCE_SUBSET_SIZE, *SUBSET_SIZES],
        "selection_method": (
            "Candidate top-k subsets are constructed from the saved baseline "
            "RandomForestClassifier feature_importances_ ranking. These are "
            "impurity-based feature importances, also known as mean decrease "
            "in impurity."
        ),
        "feature_importance_ranking_vs_feature_selection": (
            "The baseline ranking already existed before this script ran. This "
            "script uses that ranking to select candidate reduced feature sets "
            "and empirically evaluate new models."
        ),
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": build_model().get_params(deep=False),
        "random_seed": RANDOM_STATE,
        "correlation_filtering_applied": False,
        "correlation_context": correlation_summary,
        "test_labels_used_for_feature_selection": False,
        "test_set_used_during_selection": False,
        "train_test_split_preserved": True,
        "hyperparameter_tuning_applied": False,
        "oversampling_applied": False,
        "undersampling_applied": False,
        "scaling_applied": False,
        "pca_applied": False,
        "target_column": TARGET_COLUMN,
        "forbidden_feature_columns": [ORIGINAL_LABEL_COLUMN, TARGET_COLUMN],
        "selected_feature_lists": selected_feature_lists,
        "selected_tradeoff_model": tradeoff_model,
        "artifacts": {
            "selected_features": project_relative(REPORT_DIR / "selected_features.csv"),
            "selection_metadata": project_relative(REPORT_DIR / "selection_metadata.json"),
            "model_comparison_csv": project_relative(REPORT_DIR / "model_comparison.csv"),
            "model_comparison_json": project_relative(REPORT_DIR / "model_comparison.json"),
            "model_comparison_md": project_relative(REPORT_DIR / "model_comparison.md"),
            "model_evaluation_details": project_relative(REPORT_DIR / "model_evaluation_details.json"),
        },
    }
    (REPORT_DIR / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_comparison_report(comparison_df: pd.DataFrame, tradeoff_row: pd.Series) -> None:
    """Write a concise research-oriented Markdown comparison report."""
    baseline = comparison_df[comparison_df["model"] == "Baseline_65"].iloc[0]
    lines = [
        "# Phase 4 Selected-Feature Random Forest Comparison",
        "",
        "## Method",
        "",
        (
            "Feature selection used the saved baseline RandomForestClassifier "
            "feature_importances_ ranking from reports/baseline/feature_importance.csv. "
            "These values are impurity-based feature importances, also known as "
            "mean decrease in impurity. They are not information gain."
        ),
        "",
        (
            "The Phase 3 train/test split was preserved. Candidate subsets were "
            "constructed before evaluating selected-feature models on the test set. "
            "No test labels or test-set performance metrics were used for feature selection. "
            "The Phase 2 high-correlation pairs were inspected as context only; no "
            "correlation filtering was applied."
        ),
        "",
        "## Baseline",
        "",
        (
            f"Baseline_65 used {int(baseline['feature_count'])} features and achieved "
            f"accuracy {baseline['accuracy']:.6f}, precision {baseline['precision']:.6f}, "
            f"recall {baseline['recall']:.6f}, F1 {baseline['f1_score']:.6f}, and "
            f"ROC-AUC {baseline['roc_auc']:.6f}. It produced "
            f"{int(baseline['false_positives'])} false positives and "
            f"{int(baseline['false_negatives'])} false negatives."
        ),
        "",
        "## Candidate Results",
        "",
        "| Model | Features | Reduction % | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Train s | Predict s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.feature_count} | {row.feature_reduction_pct:.2f} | "
            f"{row.accuracy:.6f} | {row.precision:.6f} | {row.recall:.6f} | "
            f"{row.f1_score:.6f} | {row.roc_auc:.6f} | {row.false_positives} | "
            f"{row.false_negatives} | {row.training_time_seconds:.6f} | "
            f"{row.test_prediction_time_seconds:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Changes Relative To Baseline",
            "",
            "| Model | Accuracy delta | Recall delta | F1 delta | ROC-AUC delta | FP delta | FN delta | Train time delta % | Predict time delta % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.accuracy_delta_vs_baseline:+.6f} | "
            f"{row.recall_delta_vs_baseline:+.6f} | {row.f1_delta_vs_baseline:+.6f} | "
            f"{row.roc_auc_delta_vs_baseline:+.6f} | "
            f"{row.false_positive_delta_vs_baseline:+d} | "
            f"{row.false_negative_delta_vs_baseline:+d} | "
            f"{row.training_time_delta_pct_vs_baseline:+.2f} | "
            f"{row.prediction_time_delta_pct_vs_baseline:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Trade-Off Assessment",
            "",
            (
                f"The selected trade-off candidate is {tradeoff_row['model']}. It removes "
                f"{tradeoff_row['feature_reduction_pct']:.2f}% of features while achieving "
                f"F1 {tradeoff_row['f1_score']:.6f}, recall {tradeoff_row['recall']:.6f}, "
                f"ROC-AUC {tradeoff_row['roc_auc']:.6f}, "
                f"{int(tradeoff_row['false_positives'])} false positives, and "
                f"{int(tradeoff_row['false_negatives'])} false negatives."
            ),
            "",
            (
                "This recommendation considers F1, recall, false negatives, ROC-AUC, "
                "feature reduction, and runtime. It is not based on accuracy alone."
            ),
        ]
    )

    (REPORT_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Print a concise comparison table to the terminal."""
    display_columns = [
        "model",
        "feature_count",
        "feature_reduction_pct",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "false_positives",
        "false_negatives",
        "training_time_seconds",
        "test_prediction_time_seconds",
    ]
    printable = comparison_df[display_columns].copy()
    for column in [
        "feature_reduction_pct",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
        "test_prediction_time_seconds",
    ]:
        printable[column] = printable[column].map(lambda value: f"{value:.6f}")
    print("=" * 120)
    print("PHASE 4 SELECTED-FEATURE RANDOM FOREST COMPARISON")
    print("=" * 120)
    print(printable.to_string(index=False))
    print("=" * 120)


def main() -> int:
    """Run the selected-feature experiment."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_phase3_feature_names(PHASE3_METADATA_PATH)
    train_df, test_df = load_phase3_frames()
    validate_frame_schema(train_df, test_df, feature_names)

    importance_df = load_baseline_feature_importance(feature_names)
    correlation_summary = load_high_correlation_summary()
    if correlation_summary["pair_count"] != 33:
        raise ValueError(
            "Expected 33 Phase 2 high-correlation pairs, found "
            f"{correlation_summary['pair_count']}."
        )

    subsets = build_candidate_subsets(importance_df)
    baseline_metrics = read_json(BASELINE_METRICS_PATH)
    baseline_metadata = read_json(BASELINE_METADATA_PATH)

    comparison_rows = [baseline_comparison_row(baseline_metrics, baseline_metadata)]
    for subset_size in SUBSET_SIZES:
        subset_name = f"Top_{subset_size}"
        subset_features = subsets[subset_name]["feature_name"].tolist()
        if not set(subset_features).issubset(set(feature_names)):
            raise ValueError(f"{subset_name} contains features outside the Phase 3 feature list.")
        comparison_rows.append(
            train_and_evaluate_subset(subset_name, subset_features, train_df, test_df)
        )

    write_selected_features(subsets)
    comparison_df = build_comparison_dataframe(comparison_rows)
    comparison_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_detailed_evaluation(comparison_rows)
    tradeoff_row = choose_tradeoff_model(comparison_df)
    write_metadata(subsets, feature_names, correlation_summary, str(tradeoff_row["model"]))
    write_comparison_report(comparison_df, tradeoff_row)

    print_comparison_table(comparison_df)
    print(f"Selected trade-off candidate: {tradeoff_row['model']}")
    print(f"Reports written to: {project_relative(REPORT_DIR)}")
    print("Baseline artifacts were read as frozen inputs and were not overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
