"""Train the Phase 4 all-valid-features Random Forest baseline.

This script consumes the Phase 3 processed train/test Parquet files exactly as
written by src/preprocess.py. It does not modify preprocessing artifacts, create
a new split, tune hyperparameters, or perform feature selection.
"""

from __future__ import annotations

import argparse
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
DEFAULT_PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
DEFAULT_PHASE3_METADATA = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rf_baseline.joblib"

TARGET_COLUMN = "Label_binary"
ORIGINAL_LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 65
RANDOM_STATE = 42
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Benign", "DDoS/Attack"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Phase 4 all-valid-features Random Forest baseline."
    )
    parser.add_argument(
        "--phase3-dir",
        type=Path,
        default=DEFAULT_PHASE3_DIR,
        help="Directory containing Phase 3 train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--phase3-metadata",
        type=Path,
        default=DEFAULT_PHASE3_METADATA,
        help="Path to Phase 3 preprocessing metadata JSON.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where baseline reports will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained Random Forest model will be saved.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Return a project-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_phase3_feature_names(metadata_path: Path) -> list[str]:
    """Load the exact Phase 3 feature order from preprocessing metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Phase 3 metadata file does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("final_feature_names")
    if not isinstance(feature_names, list) or not all(
        isinstance(column, str) for column in feature_names
    ):
        raise ValueError("Phase 3 metadata is missing a valid final_feature_names list.")

    metadata_count = metadata.get("final_feature_count")
    if metadata_count != len(feature_names):
        raise ValueError(
            "Phase 3 metadata final_feature_count does not match final_feature_names length: "
            f"{metadata_count} != {len(feature_names)}"
        )
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} Phase 3 features, found {len(feature_names)}."
        )
    if ORIGINAL_LABEL_COLUMN in feature_names:
        raise ValueError(f"{ORIGINAL_LABEL_COLUMN!r} must not appear in the feature list.")
    if TARGET_COLUMN in feature_names:
        raise ValueError(f"{TARGET_COLUMN!r} must not appear in the feature list.")
    return feature_names


def load_phase3_frames(phase3_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 3 train/test Parquet files after checking they exist."""
    train_path = phase3_dir / "train.parquet"
    test_path = phase3_dir / "test.parquet"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3 processed file(s): " + ", ".join(missing))
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def validate_frame_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Validate target columns, feature count, feature order, and finite values."""
    for split_name, df in (("train", train_df), ("test", test_df)):
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{TARGET_COLUMN!r} is missing from the {split_name} dataframe.")
        if ORIGINAL_LABEL_COLUMN not in df.columns:
            raise ValueError(f"{ORIGINAL_LABEL_COLUMN!r} is missing from the {split_name} dataframe.")

        missing_features = [column for column in feature_names if column not in df.columns]
        if missing_features:
            raise ValueError(
                f"{split_name} dataframe is missing feature column(s): "
                + ", ".join(missing_features)
            )

        actual_feature_order = [column for column in df.columns if column in feature_names]
        if actual_feature_order != feature_names:
            raise ValueError(f"{split_name} feature columns do not match Phase 3 metadata order.")
        if ORIGINAL_LABEL_COLUMN in feature_names:
            raise ValueError(f"{ORIGINAL_LABEL_COLUMN!r} accidentally appears in the feature matrix.")
        if len(feature_names) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_names)}."
            )

        y_values = set(df[TARGET_COLUMN].dropna().astype(int).unique().tolist())
        if not y_values.issubset({0, 1}):
            raise ValueError(
                f"{split_name} target contains values outside binary encoding {{0, 1}}: "
                f"{sorted(y_values)}"
            )

        feature_values = df[feature_names].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(feature_values).all():
            nan_count = int(np.isnan(feature_values).sum())
            pos_inf_count = int(np.isposinf(feature_values).sum())
            neg_inf_count = int(np.isneginf(feature_values).sum())
            raise ValueError(
                f"{split_name} feature matrix contains non-finite values: "
                f"nan={nan_count}, pos_inf={pos_inf_count}, neg_inf={neg_inf_count}"
            )

    if list(train_df[feature_names].columns) != list(test_df[feature_names].columns):
        raise ValueError("Train/test feature schemas differ.")


def build_model() -> RandomForestClassifier:
    """Create a simple, reproducible, untuned Random Forest baseline."""
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


def write_classification_report_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Write the full per-class classification report."""
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(path, index_label="class")


def write_confusion_matrix_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Write a labeled 2x2 confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in CLASS_NAMES],
        columns=[f"predicted_{name}" for name in CLASS_NAMES],
    ).to_csv(path, index_label="actual_class")


def write_feature_importance_csv(
    path: Path,
    feature_names: list[str],
    importances: np.ndarray,
) -> pd.DataFrame:
    """Write feature importances sorted descending and return the dataframe."""
    rows = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    rows["rank"] = np.arange(1, len(rows) + 1)
    rows = rows[["feature_name", "importance", "rank"]]
    rows.to_csv(path, index=False)
    return rows


def build_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability_attack: np.ndarray | None,
) -> dict[str, Any]:
    """Build baseline evaluation metrics for JSON output."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_class": {"label": 1, "name": "DDoS/Attack"},
        "negative_class": {"label": 0, "name": "Benign"},
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "class_names": CLASS_NAMES,
            "matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
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
    if y_probability_attack is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability_attack))
    else:
        metrics["roc_auc"] = None
    return metrics


def run_baseline(
    *,
    phase3_dir: Path,
    phase3_metadata_path: Path,
    report_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Train, evaluate, and save the Phase 4 baseline artifacts."""
    feature_names = load_phase3_feature_names(phase3_metadata_path)
    train_df, test_df = load_phase3_frames(phase3_dir)
    validate_frame_schema(train_df, test_df, feature_names)

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

    probability_time_seconds = None
    y_probability_attack = None
    if hasattr(model, "predict_proba"):
        probability_start = time.perf_counter()
        y_probability_attack = model.predict_proba(x_test)[:, 1]
        probability_time_seconds = time.perf_counter() - probability_start

    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics(y_test, y_pred, y_probability_attack)
    timings = {
        "training_time_seconds": training_time_seconds,
        "test_prediction_time_seconds": test_prediction_time_seconds,
        "test_predict_proba_time_seconds": probability_time_seconds,
    }
    model_config = model.get_params(deep=False)
    metadata: dict[str, Any] = {
        "phase": "Phase 4 Random Forest baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "baseline_description": "All-valid-features Random Forest baseline using Phase 3 outputs.",
        "phase3_train_path": project_relative(phase3_dir / "train.parquet"),
        "phase3_test_path": project_relative(phase3_dir / "test.parquet"),
        "phase3_metadata_path": project_relative(phase3_metadata_path),
        "target_column": TARGET_COLUMN,
        "original_label_column_preserved": ORIGINAL_LABEL_COLUMN,
        "feature_source": "Phase 3 metadata final_feature_names",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": model_config,
        "random_state": RANDOM_STATE,
        "train_test_split_preserved": True,
        "feature_selection_applied": False,
        "hyperparameter_tuning_applied": False,
        "oversampling_applied": False,
        "scaling_applied": False,
        "pca_applied": False,
        "timings": timings,
        "artifacts": {
            "metrics": project_relative(report_dir / "metrics.json"),
            "classification_report": project_relative(report_dir / "classification_report.csv"),
            "confusion_matrix": project_relative(report_dir / "confusion_matrix.csv"),
            "feature_importance": project_relative(report_dir / "feature_importance.csv"),
            "metadata": project_relative(report_dir / "baseline_metadata.json"),
            "model": project_relative(model_path),
        },
    }

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_classification_report_csv(report_dir / "classification_report.csv", y_test, y_pred)
    write_confusion_matrix_csv(report_dir / "confusion_matrix.csv", y_test, y_pred)
    feature_importance = write_feature_importance_csv(
        report_dir / "feature_importance.csv",
        feature_names,
        model.feature_importances_,
    )
    joblib.dump(model, model_path)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "top_features": feature_importance.head(20).to_dict(orient="records"),
    }


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    result = run_baseline(
        phase3_dir=resolve_project_path(args.phase3_dir),
        phase3_metadata_path=resolve_project_path(args.phase3_metadata),
        report_dir=resolve_project_path(args.report_dir),
        model_path=resolve_project_path(args.model_path),
    )
    metrics = result["metrics"]
    metadata = result["metadata"]

    print("=" * 80)
    print("CIC-DDOS2019 PHASE 4 RANDOM FOREST BASELINE")
    print("=" * 80)
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Test samples: {metadata['test_samples']}")
    print(f"Features: {metadata['feature_count']}")
    print(f"Trees: {metadata['model_config']['n_estimators']}")
    print(f"Training time: {metadata['timings']['training_time_seconds']:.6f} seconds")
    print(
        "Test prediction time: "
        f"{metadata['timings']['test_prediction_time_seconds']:.6f} seconds"
    )
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1-score: {metrics['f1_score']:.6f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Reports: {metadata['artifacts']['metadata']}")
    print(f"Model: {metadata['artifacts']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
