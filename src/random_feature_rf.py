"""Run exploratory random-feature Random Forest experiments.

This script generates reproducible random combinations from the 65 Phase 3
features, trains one fixed-configuration Random Forest per combination, and
compares the results against the frozen baseline and Top-K feature-selection
experiments. The highest random test scores are exploratory candidates, not
unbiased final performance estimates.
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
PHASE3_METADATA_PATH = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"
FEATURE_SELECTION_COMPARISON_PATH = (
    PROJECT_ROOT / "reports" / "feature_selection" / "model_comparison.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "random_feature_selection"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Label_binary"
ORIGINAL_LABEL_COLUMN = "Label"
EXPECTED_FEATURE_COUNT = 65
RANDOM_STATE = 42
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Benign", "DDoS/Attack"]
RANDOM_SUBSET_SIZES = [5, 8, 10, 12, 15, 20, 30]
COMBINATIONS_PER_SIZE = 10
TOP_MODELS_TO_SAVE = 5
REFERENCE_MODELS = ["Baseline_65", "Top_50", "Top_40", "Top_30", "Top_20", "Top_10"]


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


def load_phase3_feature_names() -> list[str]:
    """Load the authoritative 65 Phase 3 features."""
    metadata = read_json(PHASE3_METADATA_PATH)
    feature_names = metadata.get("final_feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("Phase 3 metadata is missing a valid final_feature_names list.")
    if metadata.get("final_feature_count") != len(feature_names):
        raise ValueError("Phase 3 final_feature_count does not match final_feature_names length.")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} valid Phase 3 features, found {len(feature_names)}."
        )
    forbidden = {ORIGINAL_LABEL_COLUMN, TARGET_COLUMN}.intersection(feature_names)
    if forbidden:
        raise ValueError(f"Target/label column(s) must not appear as features: {sorted(forbidden)}")
    return feature_names


def load_phase3_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preserved Phase 3 train/test files."""
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
    """Validate schemas, target, and finite feature values."""
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


def build_model() -> RandomForestClassifier:
    """Create the fixed Random Forest configuration used by prior Phase 4 RF runs."""
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


def generate_random_subsets(feature_names: list[str]) -> list[dict[str, Any]]:
    """Generate reproducible unique random feature subsets."""
    rng = np.random.default_rng(RANDOM_STATE)
    all_subset_keys: set[tuple[str, ...]] = set()
    experiments: list[dict[str, Any]] = []

    for size in RANDOM_SUBSET_SIZES:
        generated_for_size = 0
        attempts = 0
        while generated_for_size < COMBINATIONS_PER_SIZE:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(f"Could not generate enough unique random subsets for size {size}.")
            selected = rng.choice(feature_names, size=size, replace=False).tolist()
            subset_key = tuple(sorted(selected))
            if subset_key in all_subset_keys:
                continue
            all_subset_keys.add(subset_key)
            generated_for_size += 1
            experiments.append(
                {
                    "experiment": f"Random_{size}_{generated_for_size:02d}",
                    "feature_count": size,
                    "features": selected,
                }
            )

    expected_total = len(RANDOM_SUBSET_SIZES) * COMBINATIONS_PER_SIZE
    if len(experiments) != expected_total:
        raise ValueError(f"Expected {expected_total} random experiments, found {len(experiments)}.")
    validate_random_subsets(experiments, set(feature_names))
    return experiments


def validate_random_subsets(experiments: list[dict[str, Any]], valid_features: set[str]) -> None:
    """Validate random subset uniqueness and feature membership."""
    seen: set[tuple[str, ...]] = set()
    for experiment in experiments:
        name = str(experiment["experiment"])
        feature_count = int(experiment["feature_count"])
        features = list(experiment["features"])
        if feature_count not in RANDOM_SUBSET_SIZES:
            raise ValueError(f"{name} has unexpected feature count {feature_count}.")
        if len(features) != feature_count:
            raise ValueError(f"{name} has {len(features)} features, expected {feature_count}.")
        if len(set(features)) != len(features):
            raise ValueError(f"{name} contains duplicate features.")
        if not set(features).issubset(valid_features):
            raise ValueError(f"{name} contains features outside the Phase 3 feature list.")
        forbidden = {ORIGINAL_LABEL_COLUMN, TARGET_COLUMN}.intersection(features)
        if forbidden:
            raise ValueError(f"{name} contains forbidden label column(s): {sorted(forbidden)}")
        subset_key = tuple(sorted(features))
        if subset_key in seen:
            raise ValueError(f"{name} duplicates a previously generated random subset.")
        seen.add(subset_key)

    counts = pd.Series([experiment["feature_count"] for experiment in experiments]).value_counts()
    for size in RANDOM_SUBSET_SIZES:
        observed = int(counts.get(size, 0))
        if observed != COMBINATIONS_PER_SIZE:
            raise ValueError(
                f"Expected {COMBINATIONS_PER_SIZE} subsets of size {size}, found {observed}."
            )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability_attack: np.ndarray,
) -> dict[str, Any]:
    """Calculate test metrics for the positive attack class."""
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
    }


def train_and_evaluate(
    experiment: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], RandomForestClassifier]:
    """Train and evaluate one random-feature model."""
    features = list(experiment["features"])
    x_train = train_df[features]
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    x_test = test_df[features]
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
    result = {
        "experiment": str(experiment["experiment"]),
        "feature_count": int(experiment["feature_count"]),
        "feature_reduction_pct": (
            (EXPECTED_FEATURE_COUNT - int(experiment["feature_count"]))
            / EXPECTED_FEATURE_COUNT
            * 100.0
        ),
        **metrics,
        "training_time_seconds": float(training_time_seconds),
        "test_prediction_time_seconds": float(test_prediction_time_seconds),
        "test_predict_proba_time_seconds": float(test_predict_proba_time_seconds),
        "features": features,
    }
    return result, model


def update_top_models(
    top_models: list[tuple[dict[str, Any], RandomForestClassifier]],
    result: dict[str, Any],
    model: RandomForestClassifier,
) -> None:
    """Keep only the top F1 models in memory for final persistence."""
    top_models.append((result, model))
    top_models.sort(
        key=lambda item: (
            item[0]["f1_score"],
            item[0]["recall"],
            item[0]["roc_auc"],
            -item[0]["false_negatives"],
        ),
        reverse=True,
    )
    del top_models[TOP_MODELS_TO_SAVE:]


def save_top_models(
    top_models: list[tuple[dict[str, Any], RandomForestClassifier]]
) -> list[dict[str, Any]]:
    """Save only the top five random-feature models by F1."""
    saved_rows: list[dict[str, Any]] = []
    for index, (result, model) in enumerate(top_models, start=1):
        model_path = MODEL_DIR / f"random_best_{index:02d}.joblib"
        joblib.dump(model, model_path)
        saved_rows.append(
            {
                "rank_by_f1": index,
                "experiment": result["experiment"],
                "feature_count": result["feature_count"],
                "f1_score": result["f1_score"],
                "recall": result["recall"],
                "roc_auc": result["roc_auc"],
                "model_path": project_relative(model_path),
            }
        )
    pd.DataFrame(saved_rows).to_csv(REPORT_DIR / "saved_top_models.csv", index=False)
    return saved_rows


def load_existing_comparison() -> pd.DataFrame:
    """Load frozen baseline and Top-K results for comparison."""
    if not FEATURE_SELECTION_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Feature-selection comparison artifact does not exist: "
            f"{FEATURE_SELECTION_COMPARISON_PATH}"
        )
    comparison = pd.read_csv(FEATURE_SELECTION_COMPARISON_PATH)
    missing_models = [model for model in REFERENCE_MODELS if model not in set(comparison["model"])]
    if missing_models:
        raise ValueError(
            "Feature-selection comparison is missing reference model(s): "
            + ", ".join(missing_models)
        )
    return comparison[comparison["model"].isin(REFERENCE_MODELS)].copy()


def write_random_feature_sets(experiments: list[dict[str, Any]]) -> None:
    """Write every random experiment's exact feature list."""
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for feature in experiment["features"]:
            rows.append(
                {
                    "experiment": experiment["experiment"],
                    "feature_count": experiment["feature_count"],
                    "feature_name": feature,
                }
            )
    pd.DataFrame(rows).to_csv(REPORT_DIR / "random_feature_sets.csv", index=False)


def write_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Write flat CSV and detailed JSON result artifacts."""
    result_columns = [
        "experiment",
        "feature_count",
        "feature_reduction_pct",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "training_time_seconds",
        "test_prediction_time_seconds",
    ]
    results_df = pd.DataFrame(results)
    results_df[result_columns].to_csv(REPORT_DIR / "random_results.csv", index=False)
    (REPORT_DIR / "random_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results_df


def build_combined_comparison(
    reference_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine existing reference models with all random experiments."""
    common_columns = [
        "model",
        "feature_count",
        "feature_reduction_pct",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "training_time_seconds",
        "test_prediction_time_seconds",
    ]
    reference = reference_df.copy()
    reference["experiment_type"] = "existing_reference"
    reference = reference.rename(columns={"model": "model"})
    random_part = random_df.rename(columns={"experiment": "model"}).copy()
    random_part["experiment_type"] = "random_feature_search"

    combined = pd.concat(
        [reference[common_columns + ["experiment_type"]], random_part[common_columns + ["experiment_type"]]],
        ignore_index=True,
    )
    baseline_f1 = float(combined.loc[combined["model"] == "Baseline_65", "f1_score"].iloc[0])
    top10_f1 = float(combined.loc[combined["model"] == "Top_10", "f1_score"].iloc[0])
    combined["f1_delta_vs_baseline_65"] = combined["f1_score"] - baseline_f1
    combined["f1_delta_vs_top_10"] = combined["f1_score"] - top10_f1
    combined.to_csv(REPORT_DIR / "comparison_with_existing.csv", index=False)
    (REPORT_DIR / "comparison_with_existing.json").write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return combined


def best_random_rows(random_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Identify best random experiments by key metrics."""
    return {
        "f1": random_df.sort_values(
            ["f1_score", "recall", "roc_auc", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0],
        "recall": random_df.sort_values(
            ["recall", "f1_score", "roc_auc", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0],
        "roc_auc": random_df.sort_values(
            ["roc_auc", "f1_score", "recall", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0],
        "accuracy": random_df.sort_values(
            ["accuracy", "f1_score", "recall", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0],
    }


def write_metadata(
    experiments: list[dict[str, Any]],
    saved_models: list[dict[str, Any]],
    top10_f1: float,
    best_by: dict[str, pd.Series],
) -> None:
    """Write machine-readable metadata for the exploratory experiment."""
    metadata = {
        "phase": "Phase 4 exploratory random feature-combination Random Forest experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_metadata_path": project_relative(PHASE3_METADATA_PATH),
        "phase3_train_path": project_relative(PHASE3_DIR / "train.parquet"),
        "phase3_test_path": project_relative(PHASE3_DIR / "test.parquet"),
        "feature_source": "Phase 3 preprocessing metadata final_feature_names",
        "valid_feature_count": EXPECTED_FEATURE_COUNT,
        "random_state": RANDOM_STATE,
        "subset_sizes": RANDOM_SUBSET_SIZES,
        "combinations_per_size": COMBINATIONS_PER_SIZE,
        "total_experiments": len(experiments),
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": build_model().get_params(deep=False),
        "train_test_split_preserved": True,
        "hyperparameter_tuning_applied": False,
        "oversampling_applied": False,
        "undersampling_applied": False,
        "scaling_applied": False,
        "pca_applied": False,
        "target_column": TARGET_COLUMN,
        "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "forbidden_feature_columns": [ORIGINAL_LABEL_COLUMN, TARGET_COLUMN],
        "methodological_limitation": (
            "This is an exploratory random-search experiment. Because the same "
            "test set is used to compare many random combinations, the best test "
            "score among these experiments is not an unbiased final performance estimate."
        ),
        "top_10_reference_f1": top10_f1,
        "best_random_by": {
            metric: {
                "experiment": row["experiment"],
                "feature_count": int(row["feature_count"]),
                "f1_score": float(row["f1_score"]),
                "recall": float(row["recall"]),
                "roc_auc": float(row["roc_auc"]),
                "accuracy": float(row["accuracy"]),
            }
            for metric, row in best_by.items()
        },
        "saved_top_f1_models": saved_models,
        "artifacts": {
            "random_results_csv": project_relative(REPORT_DIR / "random_results.csv"),
            "random_results_json": project_relative(REPORT_DIR / "random_results.json"),
            "random_feature_sets": project_relative(REPORT_DIR / "random_feature_sets.csv"),
            "comparison_with_existing_csv": project_relative(
                REPORT_DIR / "comparison_with_existing.csv"
            ),
            "comparison_with_existing_json": project_relative(
                REPORT_DIR / "comparison_with_existing.json"
            ),
            "saved_top_models": project_relative(REPORT_DIR / "saved_top_models.csv"),
            "report": project_relative(REPORT_DIR / "random_feature_selection.md"),
        },
    }
    (REPORT_DIR / "random_feature_selection_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def format_features(features: list[str]) -> str:
    """Format a feature list for Markdown."""
    return ", ".join(f"`{feature}`" for feature in features)


def write_markdown_report(
    random_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    best_by: dict[str, pd.Series],
    saved_models: list[dict[str, Any]],
) -> None:
    """Write the concise research report."""
    baseline = combined_df[combined_df["model"] == "Baseline_65"].iloc[0]
    top10 = combined_df[combined_df["model"] == "Top_10"].iloc[0]
    best_f1 = best_by["f1"]
    exceeded_top10 = bool(best_f1["f1_score"] > top10["f1_score"])
    grouped = random_df.groupby("feature_count").agg(
        experiments=("experiment", "count"),
        best_f1=("f1_score", "max"),
        mean_f1=("f1_score", "mean"),
        best_recall=("recall", "max"),
        best_roc_auc=("roc_auc", "max"),
        best_accuracy=("accuracy", "max"),
    )

    feature_lookup = {
        row["experiment"]: row["features"]
        for row in random_df[["experiment", "features"]].to_dict(orient="records")
    }

    lines = [
        "# Phase 4 Exploratory Random Feature-Combination RF",
        "",
        "## Method",
        "",
        (
            "This experiment generated 70 reproducible random feature combinations "
            "from the 65 Phase 3 features using random_state=42. For each subset, "
            "a fresh RandomForestClassifier was trained with the same fixed "
            "configuration used by the baseline and Top-K experiments."
        ),
        "",
        (
            "The Phase 3 train/test split was preserved. No new split, tuning, "
            "sampling, scaling, or PCA was applied. `Label` and `Label_binary` "
            "were excluded from every input feature set."
        ),
        "",
        "## Methodological Limitation",
        "",
        (
            "This is an exploratory random-search experiment. Because the same "
            "test set is used to compare many random combinations, the highest "
            "test score from these 70 experiments should be treated as an "
            "exploratory candidate, not as an unbiased final performance estimate."
        ),
        "",
        "## Reference Results",
        "",
        (
            f"Baseline_65 F1: {baseline['f1_score']:.6f}. "
            f"Top_10 F1: {top10['f1_score']:.6f}."
        ),
        "",
        "## Random Results By Feature Count",
        "",
        "| Features | Experiments | Best F1 | Mean F1 | Best Recall | Best ROC-AUC | Best Accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped.reset_index().itertuples(index=False):
        lines.append(
            f"| {row.feature_count} | {row.experiments} | {row.best_f1:.6f} | "
            f"{row.mean_f1:.6f} | {row.best_recall:.6f} | "
            f"{row.best_roc_auc:.6f} | {row.best_accuracy:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Random Combinations",
            "",
            "| Criterion | Experiment | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for criterion, row in best_by.items():
        lines.append(
            f"| {criterion} | {row['experiment']} | {int(row['feature_count'])} | "
            f"{row['accuracy']:.6f} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['f1_score']:.6f} | "
            f"{row['roc_auc']:.6f} | {int(row['false_positives'])} | "
            f"{int(row['false_negatives'])} |"
        )

    lines.extend(
        [
            "",
            "## Best F1 Feature Set",
            "",
            f"Experiment: {best_f1['experiment']}",
            "",
            f"Features: {format_features(feature_lookup[str(best_f1['experiment'])])}",
            "",
            (
                f"F1 difference from Baseline_65: "
                f"{best_f1['f1_score'] - baseline['f1_score']:+.6f}."
            ),
            (
                f"F1 difference from Top_10: "
                f"{best_f1['f1_score'] - top10['f1_score']:+.6f}."
            ),
            (
                "A random combination exceeded the current Top_10 F1."
                if exceeded_top10
                else "No random combination exceeded the current Top_10 F1."
            ),
            "",
            "## Saved Models",
            "",
            "Only the top five random combinations by F1 were saved.",
            "",
            "| Rank | Experiment | Features | F1 | Recall | ROC-AUC | Model |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in saved_models:
        lines.append(
            f"| {row['rank_by_f1']} | {row['experiment']} | {row['feature_count']} | "
            f"{row['f1_score']:.6f} | {row['recall']:.6f} | "
            f"{row['roc_auc']:.6f} | `{row['model_path']}` |"
        )

    (REPORT_DIR / "random_feature_selection.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_summary(
    random_df: pd.DataFrame,
    best_by: dict[str, pd.Series],
    combined_df: pd.DataFrame,
) -> None:
    """Print the required terminal summary."""
    baseline = combined_df[combined_df["model"] == "Baseline_65"].iloc[0]
    top10 = combined_df[combined_df["model"] == "Top_10"].iloc[0]
    best_f1 = best_by["f1"]
    grouped = random_df.groupby("feature_count").agg(
        experiments=("experiment", "count"),
        best_f1=("f1_score", "max"),
        mean_f1=("f1_score", "mean"),
        best_recall=("recall", "max"),
        best_roc_auc=("roc_auc", "max"),
        best_accuracy=("accuracy", "max"),
    )
    feature_lookup = {
        row["experiment"]: row["features"]
        for row in random_df[["experiment", "features"]].to_dict(orient="records")
    }
    best_features = feature_lookup[str(best_f1["experiment"])]

    print("=" * 120)
    print("PHASE 4 EXPLORATORY RANDOM FEATURE-COMBINATION RF")
    print("=" * 120)
    print(f"Total experiments: {len(random_df)}")
    print()
    print("Results grouped by feature count:")
    print(grouped.to_string(float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Best random combination by F1: {best_f1['experiment']}")
    print(f"Selected features: {', '.join(best_features)}")
    print(f"F1: {best_f1['f1_score']:.6f}")
    print(f"Recall: {best_f1['recall']:.6f}")
    print(f"ROC-AUC: {best_f1['roc_auc']:.6f}")
    print(f"F1 difference from Baseline_65: {best_f1['f1_score'] - baseline['f1_score']:+.6f}")
    print(f"F1 difference from Top_10: {best_f1['f1_score'] - top10['f1_score']:+.6f}")
    if best_f1["f1_score"] > top10["f1_score"]:
        print("At least one random combination exceeded the current Top_10 F1.")
    else:
        print("No random combination exceeded the current Top_10 F1.")
    print()
    for metric, row in best_by.items():
        print(
            f"Best by {metric}: {row['experiment']} "
            f"(features={int(row['feature_count'])}, accuracy={row['accuracy']:.6f}, "
            f"recall={row['recall']:.6f}, F1={row['f1_score']:.6f}, "
            f"ROC-AUC={row['roc_auc']:.6f})"
        )
    print(f"Reports written to: {project_relative(REPORT_DIR)}")
    print("Existing baseline and Top-K artifacts/models were read as references only.")
    print("=" * 120)


def main() -> int:
    """Run the exploratory random feature-combination experiment."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_phase3_feature_names()
    train_df, test_df = load_phase3_frames()
    validate_frame_schema(train_df, test_df, feature_names)

    reference_df = load_existing_comparison()
    experiments = generate_random_subsets(feature_names)
    write_random_feature_sets(experiments)

    results: list[dict[str, Any]] = []
    top_models: list[tuple[dict[str, Any], RandomForestClassifier]] = []
    for experiment in experiments:
        result, model = train_and_evaluate(experiment, train_df, test_df)
        results.append(result)
        update_top_models(top_models, result, model)
        print(
            f"Completed {result['experiment']} "
            f"({result['feature_count']} features): F1={result['f1_score']:.6f}"
        )

    random_df = write_results(results)
    combined_df = build_combined_comparison(reference_df, random_df)
    best_by = best_random_rows(random_df)
    saved_models = save_top_models(top_models)
    write_metadata(experiments, saved_models, float(combined_df.loc[combined_df["model"] == "Top_10", "f1_score"].iloc[0]), best_by)
    write_markdown_report(random_df, combined_df, best_by, saved_models)
    print_summary(random_df, best_by, combined_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
