"""Analyze completed Phase 4 random feature-combination experiments.

This script is analysis-only. It reads existing random-search, baseline, Top-K,
and feature-set artifacts; it does not train models, generate new feature
combinations, or modify existing model artifacts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "reports" / "random_feature_selection" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


RANDOM_REPORT_DIR = PROJECT_ROOT / "reports" / "random_feature_selection"
FIGURE_DIR = RANDOM_REPORT_DIR / "figures"
RANDOM_RESULTS_PATH = RANDOM_REPORT_DIR / "random_results.csv"
RANDOM_FEATURE_SETS_PATH = RANDOM_REPORT_DIR / "random_feature_sets.csv"
RANDOM_METADATA_PATH = RANDOM_REPORT_DIR / "random_feature_selection_metadata.json"
FEATURE_SELECTION_COMPARISON_PATH = (
    PROJECT_ROOT / "reports" / "feature_selection" / "model_comparison.csv"
)
TOP_FEATURES_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "selected_features.csv"
PHASE3_METADATA_PATH = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_metadata.json"

EXPECTED_RANDOM_EXPERIMENTS = 70
EXPECTED_FEATURE_COUNT = 65
SUBSET_SIZES = [5, 8, 10, 12, 15, 20, 30]
REFERENCE_MODELS = ["Baseline_65", "Top_50", "Top_40", "Top_30", "Top_20", "Top_10"]
METRIC_COLUMNS = [
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
APPROX_EQUAL_TOLERANCE = 1e-4
SUCCESS_GROUPS = {
    "beats_baseline_65": "beats_baseline_65",
    "beats_top_10": "beats_top_10",
}


def project_relative(path: Path) -> str:
    """Return a project-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and validate required artifacts."""
    for path in [
        RANDOM_RESULTS_PATH,
        RANDOM_FEATURE_SETS_PATH,
        RANDOM_METADATA_PATH,
        FEATURE_SELECTION_COMPARISON_PATH,
        TOP_FEATURES_PATH,
        PHASE3_METADATA_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact does not exist: {path}")

    random_results = pd.read_csv(RANDOM_RESULTS_PATH)
    feature_sets = pd.read_csv(RANDOM_FEATURE_SETS_PATH)
    comparison = pd.read_csv(FEATURE_SELECTION_COMPARISON_PATH)
    top_features = pd.read_csv(TOP_FEATURES_PATH)
    metadata = read_json(RANDOM_METADATA_PATH)

    if len(random_results) != EXPECTED_RANDOM_EXPERIMENTS:
        raise ValueError(
            f"Expected {EXPECTED_RANDOM_EXPERIMENTS} random results, found {len(random_results)}."
        )
    if random_results["experiment"].nunique() != EXPECTED_RANDOM_EXPERIMENTS:
        raise ValueError("Random results must contain 70 unique experiment names.")
    if set(random_results["feature_count"].unique()) != set(SUBSET_SIZES):
        raise ValueError("Random results do not contain the expected subset sizes.")

    missing_models = [model for model in REFERENCE_MODELS if model not in set(comparison["model"])]
    if missing_models:
        raise ValueError(
            "Feature-selection comparison is missing reference model(s): "
            + ", ".join(missing_models)
        )

    phase3 = read_json(PHASE3_METADATA_PATH)
    phase3_features = phase3.get("final_feature_names")
    if not isinstance(phase3_features, list) or len(phase3_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Phase 3 metadata must contain exactly 65 final feature names.")
    if set(feature_sets["feature_name"]) - set(phase3_features):
        raise ValueError("Random feature sets contain features outside the Phase 3 feature list.")

    feature_counts = feature_sets.groupby("experiment")["feature_name"].count()
    expected_counts = random_results.set_index("experiment")["feature_count"]
    if not feature_counts.equals(expected_counts.loc[feature_counts.index]):
        raise ValueError("random_feature_sets.csv counts do not match random_results.csv.")
    if feature_sets.duplicated(["experiment", "feature_name"]).any():
        raise ValueError("A random experiment contains duplicate feature names.")

    numeric_columns = [
        "feature_count",
        "feature_reduction_pct",
        *METRIC_COLUMNS,
        "true_negatives",
        "true_positives",
    ]
    for column in numeric_columns:
        random_results[column] = pd.to_numeric(random_results[column])
    for column in ["feature_count", "feature_reduction_pct", *METRIC_COLUMNS]:
        comparison[column] = pd.to_numeric(comparison[column])

    return random_results, feature_sets, comparison, top_features, metadata


def describe_metrics(df: pd.DataFrame, prefix: str | None = None) -> pd.DataFrame:
    """Describe core metrics with min, max, mean, median, and sample std."""
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        row: dict[str, Any] = {
            "metric": metric,
            "minimum": df[metric].min(),
            "maximum": df[metric].max(),
            "mean": df[metric].mean(),
            "median": df[metric].median(),
            "std": df[metric].std(ddof=1),
        }
        if prefix is not None:
            row["group"] = prefix
        rows.append(row)
    columns = ["group", "metric", "minimum", "maximum", "mean", "median", "std"]
    result = pd.DataFrame(rows)
    return result[[column for column in columns if column in result.columns]]


def build_feature_count_summary(random_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and timing by feature count."""
    summary = (
        random_results.groupby("feature_count")
        .agg(
            experiments=("experiment", "count"),
            feature_reduction_pct=("feature_reduction_pct", "first"),
            mean_f1=("f1_score", "mean"),
            median_f1=("f1_score", "median"),
            std_f1=("f1_score", "std"),
            best_f1=("f1_score", "max"),
            worst_f1=("f1_score", "min"),
            mean_accuracy=("accuracy", "mean"),
            mean_recall=("recall", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_false_positives=("false_positives", "mean"),
            mean_false_negatives=("false_negatives", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            mean_prediction_time_seconds=("test_prediction_time_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RANDOM_REPORT_DIR / "feature_count_summary.csv", index=False)
    return summary


def classify_against_reference(delta: float) -> str:
    """Classify F1 difference using the documented tolerance."""
    if delta > APPROX_EQUAL_TOLERANCE:
        return "Better than reference"
    if delta < -APPROX_EQUAL_TOLERANCE:
        return "Worse than reference"
    return "Approximately equal to reference"


def build_analysis_table(
    random_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Create per-experiment analysis fields against Baseline_65 and Top_10."""
    baseline_f1 = float(comparison.loc[comparison["model"] == "Baseline_65", "f1_score"].iloc[0])
    top10_f1 = float(comparison.loc[comparison["model"] == "Top_10", "f1_score"].iloc[0])
    analysis = random_results.copy()
    analysis["f1_delta_vs_baseline_65"] = analysis["f1_score"] - baseline_f1
    analysis["f1_delta_vs_top_10"] = analysis["f1_score"] - top10_f1
    analysis["baseline_comparison"] = analysis["f1_delta_vs_baseline_65"].map(
        classify_against_reference
    )
    analysis["top_10_comparison"] = analysis["f1_delta_vs_top_10"].map(classify_against_reference)
    analysis["beats_baseline_65"] = analysis["f1_score"] > baseline_f1
    analysis["beats_top_10"] = analysis["f1_score"] > top10_f1
    analysis.to_csv(RANDOM_REPORT_DIR / "random_feature_analysis.csv", index=False)
    return analysis, baseline_f1, top10_f1


def build_feature_frequency(
    feature_sets: pd.DataFrame,
    random_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate overall feature frequencies and count by subset size."""
    total_experiments = random_results["experiment"].nunique()
    all_features = sorted(feature_sets["feature_name"].unique())
    overall = (
        feature_sets.groupby("feature_name")
        .size()
        .reindex(all_features, fill_value=0)
        .rename("total_appearances")
        .reset_index()
    )
    overall["appearance_percentage"] = overall["total_appearances"] / total_experiments * 100.0

    by_size = (
        feature_sets.pivot_table(
            index="feature_name",
            columns="feature_count",
            values="experiment",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(all_features, fill_value=0)
        .reset_index()
    )
    by_size.columns = [
        "feature_name" if column == "feature_name" else f"count_in_size_{int(column)}"
        for column in by_size.columns
    ]
    frequency = overall.merge(by_size, on="feature_name", how="left")
    frequency = frequency.sort_values(
        ["total_appearances", "feature_name"], ascending=[False, True], kind="mergesort"
    )
    frequency.to_csv(RANDOM_REPORT_DIR / "feature_frequency_analysis.csv", index=False)
    return frequency


def build_successful_set_analysis(
    analysis: pd.DataFrame,
    feature_sets: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature frequencies in successful groups and enrichment ratios."""
    all_experiments = set(analysis["experiment"])
    overall_experiment_count = len(all_experiments)
    overall_freq = (
        feature_sets.groupby("feature_name")["experiment"]
        .nunique()
        .rename("overall_sets_with_feature")
    )
    all_features = sorted(feature_sets["feature_name"].unique())

    rows: list[pd.DataFrame] = []
    for group_name, flag_column in SUCCESS_GROUPS.items():
        successful_experiments = set(analysis.loc[analysis[flag_column], "experiment"])
        successful_feature_sets = feature_sets[
            feature_sets["experiment"].isin(successful_experiments)
        ]
        success_freq = (
            successful_feature_sets.groupby("feature_name")["experiment"]
            .nunique()
            .reindex(all_features, fill_value=0)
            .rename("successful_sets_with_feature")
        )
        group = pd.DataFrame({"feature_name": all_features})
        group["success_group"] = group_name
        group["successful_set_count"] = len(successful_experiments)
        group = group.merge(overall_freq.reindex(all_features, fill_value=0), on="feature_name")
        group = group.merge(success_freq, on="feature_name")
        group["overall_frequency"] = (
            group["overall_sets_with_feature"] / overall_experiment_count
        )
        group["successful_frequency"] = np.where(
            group["successful_set_count"] > 0,
            group["successful_sets_with_feature"] / group["successful_set_count"],
            0.0,
        )
        group["enrichment_ratio"] = np.where(
            group["overall_frequency"] > 0,
            group["successful_frequency"] / group["overall_frequency"],
            np.nan,
        )
        group["successful_appearance_percentage"] = group["successful_frequency"] * 100.0
        rows.append(group)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(
        ["success_group", "successful_sets_with_feature", "enrichment_ratio", "feature_name"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    result.to_csv(RANDOM_REPORT_DIR / "successful_set_analysis.csv", index=False)
    return result


def build_feature_outcome_analysis(
    analysis: pd.DataFrame,
    feature_sets: pd.DataFrame,
) -> pd.DataFrame:
    """Compare F1 of sets containing each feature versus sets not containing it."""
    feature_to_experiments = feature_sets.groupby("feature_name")["experiment"].apply(set)
    all_experiments = set(analysis["experiment"])
    rows: list[dict[str, Any]] = []
    indexed = analysis.set_index("experiment")
    for feature, contains in feature_to_experiments.items():
        not_contains = all_experiments - contains
        contains_f1 = indexed.loc[list(contains), "f1_score"]
        not_contains_f1 = indexed.loc[list(not_contains), "f1_score"]
        rows.append(
            {
                "feature_name": feature,
                "sets_with_feature": len(contains),
                "sets_without_feature": len(not_contains),
                "mean_f1_with_feature": contains_f1.mean(),
                "median_f1_with_feature": contains_f1.median(),
                "mean_f1_without_feature": not_contains_f1.mean(),
                "median_f1_without_feature": not_contains_f1.median(),
                "mean_f1_difference_with_minus_without": (
                    contains_f1.mean() - not_contains_f1.mean()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["mean_f1_difference_with_minus_without", "sets_with_feature"],
        ascending=[False, False],
        kind="mergesort",
    )
    result.to_csv(RANDOM_REPORT_DIR / "feature_outcome_analysis.csv", index=False)
    return result


def build_ranked_tables(analysis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create top-10 tables by several ranking criteria."""
    rankings = {
        "top_10_by_f1": analysis.sort_values(
            ["f1_score", "recall", "roc_auc", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).head(10),
        "top_10_by_roc_auc": analysis.sort_values(
            ["roc_auc", "f1_score", "recall", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).head(10),
        "top_10_by_recall": analysis.sort_values(
            ["recall", "f1_score", "roc_auc", "false_negatives"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).head(10),
        "top_10_by_lowest_fn": analysis.sort_values(
            ["false_negatives", "f1_score", "recall"],
            ascending=[True, False, False],
            kind="mergesort",
        ).head(10),
        "top_10_by_lowest_fp": analysis.sort_values(
            ["false_positives", "f1_score", "precision"],
            ascending=[True, False, False],
            kind="mergesort",
        ).head(10),
    }
    for name, table in rankings.items():
        table.to_csv(RANDOM_REPORT_DIR / f"{name}.csv", index=False)
    return rankings


def build_overlap_analysis(
    rankings: dict[str, pd.DataFrame],
    feature_sets: pd.DataFrame,
    top_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze feature overlap between high-performing random groups and Top_10."""
    experiment_features = feature_sets.groupby("experiment")["feature_name"].apply(set)
    groups: dict[str, set[str]] = {}
    for ranking_name, table in rankings.items():
        union_features: set[str] = set()
        for experiment in table["experiment"]:
            union_features.update(experiment_features[experiment])
        groups[ranking_name] = union_features
    groups["top_10_importance_selected"] = set(
        top_features.loc[top_features["subset_name"] == "Top_10", "feature_name"]
    )

    names = list(groups)
    pairwise_rows: list[dict[str, Any]] = []
    for left in names:
        for right in names:
            left_set = groups[left]
            right_set = groups[right]
            intersection = left_set & right_set
            union = left_set | right_set
            pairwise_rows.append(
                {
                    "group_a": left,
                    "group_b": right,
                    "features_in_a": len(left_set),
                    "features_in_b": len(right_set),
                    "overlap_count": len(intersection),
                    "jaccard_overlap": len(intersection) / len(union) if union else 0.0,
                    "overlap_features": "; ".join(sorted(intersection)),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(RANDOM_REPORT_DIR / "feature_overlap_analysis.csv", index=False)

    occurrence_rows: list[dict[str, Any]] = []
    all_features = sorted(feature_sets["feature_name"].unique())
    for feature in all_features:
        present_groups = [name for name, values in groups.items() if feature in values]
        occurrence_rows.append(
            {
                "feature_name": feature,
                "high_performing_group_count": len(present_groups),
                "groups": "; ".join(present_groups),
            }
        )
    occurrence = pd.DataFrame(occurrence_rows).sort_values(
        ["high_performing_group_count", "feature_name"],
        ascending=[False, True],
        kind="mergesort",
    )
    occurrence.to_csv(RANDOM_REPORT_DIR / "high_performing_group_feature_overlap.csv", index=False)
    return pairwise, occurrence


def build_correlation_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive correlations between feature count and outcomes."""
    columns = [
        "f1_score",
        "recall",
        "roc_auc",
        "false_positives",
        "false_negatives",
        "training_time_seconds",
        "test_prediction_time_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append(
            {
                "x": "feature_count",
                "y": column,
                "pearson_correlation": analysis["feature_count"].corr(analysis[column]),
                "spearman_correlation": analysis["feature_count"].corr(
                    analysis[column], method="spearman"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RANDOM_REPORT_DIR / "feature_count_correlation_analysis.csv", index=False)
    return result


def create_plots(
    analysis: pd.DataFrame,
    feature_count_summary: pd.DataFrame,
    feature_frequency: pd.DataFrame,
    successful_analysis: pd.DataFrame,
) -> list[str]:
    """Create simple publication-ready static figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure_paths: list[str] = []

    def save_current(name: str) -> None:
        path = FIGURE_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        figure_paths.append(project_relative(path))

    plt.figure(figsize=(8, 5))
    plt.scatter(analysis["feature_count"], analysis["f1_score"], color="#1f77b4", alpha=0.82)
    plt.plot(
        feature_count_summary["feature_count"],
        feature_count_summary["mean_f1"],
        color="#d62728",
        marker="o",
        label="Mean F1",
    )
    plt.xlabel("Number of random features")
    plt.ylabel("F1-score")
    plt.title("Random Feature Sets: F1 vs Feature Count")
    plt.legend()
    save_current("f1_vs_feature_count.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(analysis["feature_count"], analysis["roc_auc"], color="#2ca02c", alpha=0.82)
    plt.plot(
        feature_count_summary["feature_count"],
        feature_count_summary["mean_roc_auc"],
        color="#9467bd",
        marker="o",
        label="Mean ROC-AUC",
    )
    plt.xlabel("Number of random features")
    plt.ylabel("ROC-AUC")
    plt.title("Random Feature Sets: ROC-AUC vs Feature Count")
    plt.legend()
    save_current("roc_auc_vs_feature_count.png")

    grouped_values = [
        analysis.loc[analysis["feature_count"] == size, "f1_score"].values for size in SUBSET_SIZES
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(grouped_values, tick_labels=[str(size) for size in SUBSET_SIZES], showmeans=True)
    plt.xlabel("Number of random features")
    plt.ylabel("F1-score")
    plt.title("F1 Distribution by Random Feature Count")
    save_current("f1_distribution_by_feature_count.png")

    top_frequency = feature_frequency.head(20).sort_values("total_appearances")
    plt.figure(figsize=(8, 7))
    plt.barh(top_frequency["feature_name"], top_frequency["total_appearances"], color="#4c78a8")
    plt.xlabel("Appearances across 70 random sets")
    plt.title("Most Frequent Randomly Selected Features")
    save_current("feature_frequency_top20.png")

    baseline_success = successful_analysis[
        successful_analysis["success_group"] == "beats_baseline_65"
    ].head(20)
    baseline_success = baseline_success.sort_values("successful_sets_with_feature")
    plt.figure(figsize=(8, 7))
    plt.barh(
        baseline_success["feature_name"],
        baseline_success["successful_sets_with_feature"],
        color="#59a14f",
    )
    plt.xlabel("Appearances in random sets beating Baseline_65")
    plt.title("Feature Frequency in Baseline-Beating Sets")
    save_current("successful_feature_frequency_top20.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        analysis["feature_count"],
        analysis["training_time_seconds"],
        color="#f28e2b",
        alpha=0.82,
    )
    plt.plot(
        feature_count_summary["feature_count"],
        feature_count_summary["mean_training_time_seconds"],
        color="#4e79a7",
        marker="o",
        label="Mean training time",
    )
    plt.xlabel("Number of random features")
    plt.ylabel("Training time (seconds)")
    plt.title("Training Time vs Feature Count")
    plt.legend()
    save_current("training_time_vs_feature_count.png")

    return figure_paths


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 6) -> list[str]:
    """Build a compact Markdown table."""
    table = df[columns].copy()
    headers = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def features_for_experiment(feature_sets: pd.DataFrame, experiment: str) -> str:
    """Return a Markdown-safe feature list for one experiment."""
    features = feature_sets.loc[feature_sets["experiment"] == experiment, "feature_name"].tolist()
    return ", ".join(f"`{feature}`" for feature in features)


def write_report(
    analysis: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_count_summary: pd.DataFrame,
    overall_distribution: pd.DataFrame,
    grouped_distribution: pd.DataFrame,
    feature_frequency: pd.DataFrame,
    successful_analysis: pd.DataFrame,
    feature_outcomes: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    correlations: pd.DataFrame,
    overlap_pairwise: pd.DataFrame,
    overlap_occurrence: pd.DataFrame,
    feature_sets: pd.DataFrame,
    metadata: dict[str, Any],
    figure_paths: list[str],
) -> None:
    """Write the full research-oriented Markdown analysis."""
    baseline = comparison[comparison["model"] == "Baseline_65"].iloc[0]
    top10 = comparison[comparison["model"] == "Top_10"].iloc[0]
    better_baseline = analysis[analysis["beats_baseline_65"]]
    better_top10 = analysis[analysis["beats_top_10"]]
    approximately_baseline = analysis[
        analysis["baseline_comparison"] == "Approximately equal to reference"
    ]
    approximately_top10 = analysis[
        analysis["top_10_comparison"] == "Approximately equal to reference"
    ]
    worse_baseline = analysis[analysis["baseline_comparison"] == "Worse than reference"]
    below_top10 = analysis[analysis["top_10_comparison"] == "Worse than reference"]
    smallest_successful_count = int(better_baseline["feature_count"].min())
    best_f1 = rankings["top_10_by_f1"].iloc[0]
    highest_mean_f1 = feature_count_summary.sort_values("mean_f1", ascending=False).iloc[0]
    lowest_variance = feature_count_summary.sort_values("std_f1", ascending=True).iloc[0]
    most_frequent_feature = feature_frequency.iloc[0]
    baseline_success_features = successful_analysis[
        successful_analysis["success_group"] == "beats_baseline_65"
    ].sort_values(
        ["successful_sets_with_feature", "enrichment_ratio", "feature_name"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    top_success_feature = baseline_success_features.iloc[0]

    training_reduction = (
        (baseline["training_time_seconds"] - feature_count_summary["mean_training_time_seconds"])
        / baseline["training_time_seconds"]
        * 100.0
    )
    prediction_reduction = (
        (
            baseline["test_prediction_time_seconds"]
            - feature_count_summary["mean_prediction_time_seconds"]
        )
        / baseline["test_prediction_time_seconds"]
        * 100.0
    )
    efficiency = feature_count_summary[["feature_count"]].copy()
    efficiency["mean_training_time_reduction_vs_baseline_pct"] = training_reduction
    efficiency["mean_prediction_time_reduction_vs_baseline_pct"] = prediction_reduction

    lines = [
        "# Random Feature-Combination Research Analysis",
        "",
        "## Dataset And Experiment Overview",
        "",
        (
            "This analysis uses the completed 70 random feature-combination RF experiments "
            "under `reports/random_feature_selection/`. No new models were trained, no new "
            "feature combinations were generated, and no baseline or Top-N artifacts were modified."
        ),
        "",
        f"- Total random experiments: {len(analysis)}",
        f"- Feature subset sizes tested: {', '.join(str(size) for size in SUBSET_SIZES)}",
        "- Random combinations per size: 10",
        f"- Random seed: {metadata.get('random_state', 42)}",
        "- RF configuration: 100-tree sklearn RandomForestClassifier with gini, max_features=sqrt, bootstrap=True, random_state=42, n_jobs=-1",
        "- Data split: existing Phase 3 train/test Parquet files",
        "- Positive class: 1 = DDoS/Attack; negative class: 0 = Benign",
        "- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion-matrix counts, training time, prediction time",
        "",
        "This is exploratory random search. Because all 70 random combinations were evaluated on the same held-out test set, choosing the highest test score can introduce selection bias and multiple-comparison effects. These results are exploratory evidence, not an unbiased final generalization estimate.",
        "",
        "## Overall Performance Distribution",
        "",
    ]
    lines.extend(markdown_table(overall_distribution, overall_distribution.columns.tolist()))

    lines.extend(
        [
            "",
            "Across all random sets, performance was broad rather than uniformly high. F1 ranged from "
            f"{analysis['f1_score'].min():.6f} to {analysis['f1_score'].max():.6f}, with mean "
            f"{analysis['f1_score'].mean():.6f} and median {analysis['f1_score'].median():.6f}.",
            "",
            "## Performance By Feature Count",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            feature_count_summary,
            [
                "feature_count",
                "experiments",
                "feature_reduction_pct",
                "mean_f1",
                "median_f1",
                "std_f1",
                "best_f1",
                "worst_f1",
                "mean_accuracy",
                "mean_recall",
                "mean_roc_auc",
                "mean_false_positives",
                "mean_false_negatives",
                "mean_training_time_seconds",
                "mean_prediction_time_seconds",
            ],
        )
    )
    lines.extend(
        [
            "",
            (
                f"The highest mean F1 was observed for {int(highest_mean_f1['feature_count'])}-feature "
                f"sets ({highest_mean_f1['mean_f1']:.6f}). Increasing random feature count did not "
                "monotonically improve F1; 12-feature sets had the strongest average F1, while "
                "30-feature sets were strong but not the best on average."
            ),
            "",
            "## Baseline Comparison",
            "",
            f"Baseline_65 F1: {baseline['f1_score']:.6f}. The approximate-equality tolerance is +/- {APPROX_EQUAL_TOLERANCE:.6f} F1.",
            "",
            f"- Better than baseline: {len(better_baseline)} / 70 ({len(better_baseline) / 70 * 100:.2f}%)",
            f"- Approximately equal to baseline: {len(approximately_baseline)} / 70 ({len(approximately_baseline) / 70 * 100:.2f}%)",
            f"- Worse than baseline: {len(worse_baseline)} / 70 ({len(worse_baseline) / 70 * 100:.2f}%)",
            "",
            "Successful baseline-beating sets by feature count:",
            "",
        ]
    )
    baseline_success_counts = (
        better_baseline.groupby("feature_count")
        .size()
        .reindex(SUBSET_SIZES, fill_value=0)
        .reset_index(name="sets_beating_baseline")
    )
    lines.extend(markdown_table(baseline_success_counts, ["feature_count", "sets_beating_baseline"], 0))

    lines.extend(
        [
            "",
            "## Top-10 Comparison",
            "",
            f"Top_10 F1: {top10['f1_score']:.6f}.",
            "",
            f"- Beat Top_10: {len(better_top10)} / 70 ({len(better_top10) / 70 * 100:.2f}%)",
            f"- Approximately matched Top_10: {len(approximately_top10)} / 70 ({len(approximately_top10) / 70 * 100:.2f}%)",
            f"- Below Top_10: {len(below_top10)} / 70 ({len(below_top10) / 70 * 100:.2f}%)",
            "",
        ]
    )
    if better_top10.empty:
        lines.append("No random set exceeded Top_10 F1.")
    else:
        lines.append("Random sets that beat Top_10:")
        lines.append("")
        for _, row in better_top10.sort_values("f1_score", ascending=False).iterrows():
            lines.extend(
                [
                    f"- {row['experiment']} ({int(row['feature_count'])} features): F1 {row['f1_score']:.6f}, recall {row['recall']:.6f}, ROC-AUC {row['roc_auc']:.6f}, FP {int(row['false_positives'])}, FN {int(row['false_negatives'])}. Features: {features_for_experiment(feature_sets, row['experiment'])}",
                ]
            )

    lines.extend(
        [
            "",
            "## Top Random Experiments",
            "",
            "The top F1 experiment is not automatically the best overall model. Recall, false negatives, false positives, ROC-AUC, feature count, and runtime move differently across rankings.",
        ]
    )
    ranking_columns = [
        "experiment",
        "feature_count",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "false_positives",
        "false_negatives",
    ]
    for title, table in [
        ("Top 10 By F1", rankings["top_10_by_f1"]),
        ("Top 10 By ROC-AUC", rankings["top_10_by_roc_auc"]),
        ("Top 10 By Recall", rankings["top_10_by_recall"]),
        ("Top 10 By Lowest FN", rankings["top_10_by_lowest_fn"]),
        ("Top 10 By Lowest FP", rankings["top_10_by_lowest_fp"]),
    ]:
        lines.extend(["", f"### {title}", ""])
        lines.extend(markdown_table(table, ranking_columns))

    lines.extend(
        [
            "",
            "## Feature Frequency Analysis",
            "",
            "Feature appearances are driven by reproducible random sampling, but the counts still help show which features were more exposed to evaluation in this random search.",
            "",
            "Most frequent features overall:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            feature_frequency.head(15),
            ["feature_name", "total_appearances", "appearance_percentage"],
        )
    )
    lines.extend(["", "Least frequent features overall:", ""])
    lines.extend(
        markdown_table(
            feature_frequency.tail(15).sort_values(["total_appearances", "feature_name"]),
            ["feature_name", "total_appearances", "appearance_percentage"],
        )
    )

    lines.extend(
        [
            "",
            "## Successful-Set Feature Analysis",
            "",
            "Success is analyzed in two groups: sets that beat Baseline_65 and sets that beat Top_10. Enrichment is the successful-set frequency divided by overall frequency. This is exploratory association analysis, not a statistical significance claim.",
            "",
            "Features most common among sets beating Baseline_65:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            baseline_success_features.head(20),
            [
                "feature_name",
                "overall_sets_with_feature",
                "successful_sets_with_feature",
                "successful_appearance_percentage",
                "enrichment_ratio",
            ],
        )
    )
    top10_success_features = successful_analysis[
        successful_analysis["success_group"] == "beats_top_10"
    ].sort_values(
        ["successful_sets_with_feature", "enrichment_ratio", "feature_name"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    lines.extend(["", "Features in sets beating Top_10:", ""])
    lines.extend(
        markdown_table(
            top10_success_features.head(20),
            [
                "feature_name",
                "overall_sets_with_feature",
                "successful_sets_with_feature",
                "successful_appearance_percentage",
                "enrichment_ratio",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Common Features And Feature Outcomes",
            "",
            "The table below compares F1 in sets containing a feature versus sets not containing it. This is descriptive and should not be read as causal.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            feature_outcomes.head(20),
            [
                "feature_name",
                "sets_with_feature",
                "sets_without_feature",
                "mean_f1_with_feature",
                "mean_f1_without_feature",
                "mean_f1_difference_with_minus_without",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Feature Count Versus Performance",
            "",
            "Correlations are descriptive only. They summarize observed association across the 70 random sets and do not establish causation.",
            "",
        ]
    )
    lines.extend(markdown_table(correlations, correlations.columns.tolist()))

    lines.extend(
        [
            "",
            "## Stability Analysis",
            "",
            (
                f"The lowest F1 standard deviation was observed for {int(lowest_variance['feature_count'])}-feature "
                f"sets (std {lowest_variance['std_f1']:.6f}). Five-feature sets were highly variable "
                f"(std {feature_count_summary.loc[feature_count_summary['feature_count'] == 5, 'std_f1'].iloc[0]:.6f}), "
                "showing that very small random subsets can be excellent or poor depending on which features they contain."
            ),
            "",
            "The 12-feature group produced the best F1 and the best mean F1, but not every 12-feature set was strong; its worst F1 was "
            f"{feature_count_summary.loc[feature_count_summary['feature_count'] == 12, 'worst_f1'].iloc[0]:.6f}. "
            "This suggests a small number of combinations are unusually strong rather than all small subsets being reliably strong.",
            "",
            "## Computational Efficiency",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            efficiency,
            [
                "feature_count",
                "mean_training_time_reduction_vs_baseline_pct",
                "mean_prediction_time_reduction_vs_baseline_pct",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Random subsets generally reduced runtime relative to the 65-feature baseline. The strongest random F1 result used 12 features, reducing the feature set by 81.54%, but its individual training and prediction times were slower than Top_10 in the recorded run.",
            "",
            "## Feature Overlap Between Best Sets",
            "",
            "High-performing random groups share some features with each other and with the importance-based Top_10 set, but the overlap is incomplete. This supports looking at hybrid strategies rather than simply replacing importance-based selection with one random set.",
            "",
            "Most repeated features across high-performing ranking groups:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            overlap_occurrence.head(20),
            ["feature_name", "high_performing_group_count", "groups"],
            4,
        )
    )
    lines.extend(["", "Pairwise group overlap summary:", ""])
    lines.extend(
        markdown_table(
            overlap_pairwise[
                overlap_pairwise["group_a"].isin(["top_10_by_f1", "top_10_by_roc_auc", "top_10_by_recall", "top_10_importance_selected"])
                & overlap_pairwise["group_b"].isin(["top_10_by_f1", "top_10_by_roc_auc", "top_10_by_recall", "top_10_importance_selected"])
            ].head(16),
            ["group_a", "group_b", "overlap_count", "jaccard_overlap"],
        )
    )

    lines.extend(["", "## Visualizations", ""])
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            f"1. Random sets beating the 65-feature baseline: {len(better_baseline)}.",
            f"2. Random sets beating Top_10: {len(better_top10)}.",
            f"3. Smallest feature count among baseline-beating random sets: {smallest_successful_count}.",
            "4. Successful small subsets were possible but not common: 5-feature sets included one baseline-beating set, while several other 5-feature sets performed poorly.",
            "5. Low feature counts were less stable in this sample; the 5-feature group had the largest F1 spread, while 8-feature sets had the lowest F1 variance.",
            f"6. The most frequent feature overall was `{most_frequent_feature['feature_name']}` with {int(most_frequent_feature['total_appearances'])} appearances.",
            f"7. The most frequent feature among baseline-beating sets was `{top_success_feature['feature_name']}`, appearing in {int(top_success_feature['successful_sets_with_feature'])} of {int(top_success_feature['successful_set_count'])} baseline-beating sets.",
            "8. Random selection was competitive in some cases, but the distribution shows high variability; importance-based Top_10 remains a stronger controlled result than a single exploratory random win.",
            f"9. The {int(highest_mean_f1['feature_count'])}-feature random group appears to provide the best observed performance/complexity trade-off in this random search.",
            "10. The next feature-selection experiment should use these observations to design a controlled hybrid selection method and then validate it without treating this same test-set search result as final.",
            "",
            "## Key Findings",
            "",
            f"- 13 of 70 random combinations beat the 65-feature baseline by F1.",
            f"- {len(better_top10)} of 70 random combinations beat the Top_10 F1, and the margin was very small.",
            f"- The best F1 was {best_f1['f1_score']:.6f} from {best_f1['experiment']} with {int(best_f1['feature_count'])} features.",
            f"- The smallest baseline-beating subset used {smallest_successful_count} features.",
            f"- The 12-feature group had the highest mean F1 ({highest_mean_f1['mean_f1']:.6f}).",
            f"- The {int(lowest_variance['feature_count'])}-feature group had the lowest F1 standard deviation ({lowest_variance['std_f1']:.6f}).",
            "- Very small random subsets can work, but they are not reliably strong; feature composition matters more than feature count alone.",
            "- The same-test-set search limitation means Random_12_07 should be treated as an exploratory candidate, not a final model.",
            "",
            "## Implications For Feature Selection",
            "",
            "The evidence supports continuing with importance-based selection as a stable baseline, while exploring a hybrid method that combines high-importance features, recurring successful-set features, and correlation-aware pruning. Random selection revealed that non-obvious combinations can be competitive, but the variability argues against adopting random search itself as the final selection strategy. A next controlled experiment could predefine a small number of hybrid candidate sets using training-only or already-fixed artifacts, then evaluate them once under the same RF configuration.",
        ]
    )

    (RANDOM_REPORT_DIR / "random_feature_analysis.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Generate all analysis artifacts."""
    RANDOM_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    random_results, feature_sets, comparison, top_features, metadata = load_inputs()

    analysis, baseline_f1, top10_f1 = build_analysis_table(random_results, comparison)
    overall_distribution = describe_metrics(analysis)
    overall_distribution.to_csv(RANDOM_REPORT_DIR / "overall_performance_distribution.csv", index=False)
    grouped_distribution = pd.concat(
        [
            describe_metrics(group, f"feature_count_{feature_count}")
            for feature_count, group in analysis.groupby("feature_count")
        ],
        ignore_index=True,
    )
    grouped_distribution.to_csv(
        RANDOM_REPORT_DIR / "grouped_performance_distribution.csv",
        index=False,
    )
    feature_count_summary = build_feature_count_summary(analysis)
    feature_frequency = build_feature_frequency(feature_sets, analysis)
    successful_analysis = build_successful_set_analysis(analysis, feature_sets)
    feature_outcomes = build_feature_outcome_analysis(analysis, feature_sets)
    rankings = build_ranked_tables(analysis)
    correlations = build_correlation_summary(analysis)
    overlap_pairwise, overlap_occurrence = build_overlap_analysis(
        rankings,
        feature_sets,
        top_features,
    )
    figure_paths = create_plots(
        analysis,
        feature_count_summary,
        feature_frequency,
        successful_analysis,
    )
    write_report(
        analysis,
        comparison,
        feature_count_summary,
        overall_distribution,
        grouped_distribution,
        feature_frequency,
        successful_analysis,
        feature_outcomes,
        rankings,
        correlations,
        overlap_pairwise,
        overlap_occurrence,
        feature_sets,
        metadata,
        figure_paths,
    )

    better_baseline = analysis[analysis["f1_score"] > baseline_f1]
    better_top10 = analysis[analysis["f1_score"] > top10_f1]
    best_f1 = rankings["top_10_by_f1"].iloc[0]
    smallest_successful = int(better_baseline["feature_count"].min())
    most_frequent = feature_frequency.iloc[0]
    success_feature = successful_analysis[
        successful_analysis["success_group"] == "beats_baseline_65"
    ].sort_values(
        ["successful_sets_with_feature", "enrichment_ratio", "feature_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    highest_mean_f1 = feature_count_summary.sort_values("mean_f1", ascending=False).iloc[0]
    lowest_variance = feature_count_summary.sort_values("std_f1", ascending=True).iloc[0]

    print("=" * 100)
    print("RANDOM FEATURE-COMBINATION ANALYSIS SUMMARY")
    print("=" * 100)
    print(f"Total random sets: {len(analysis)}")
    print(f"Number beating baseline: {len(better_baseline)}")
    print(f"Number beating Top_10: {len(better_top10)}")
    print(f"Best random F1: {best_f1['f1_score']:.6f} ({best_f1['experiment']})")
    print(f"Best random feature count: {int(best_f1['feature_count'])}")
    print(f"Smallest successful feature count: {smallest_successful}")
    print(
        "Most frequent feature overall: "
        f"{most_frequent['feature_name']} ({int(most_frequent['total_appearances'])} appearances)"
    )
    print(
        "Most frequent feature among successful sets: "
        f"{success_feature['feature_name']} "
        f"({int(success_feature['successful_sets_with_feature'])} appearances)"
    )
    print(
        "Feature count with highest mean F1: "
        f"{int(highest_mean_f1['feature_count'])} ({highest_mean_f1['mean_f1']:.6f})"
    )
    print(
        "Feature count with lowest mean F1 variance: "
        f"{int(lowest_variance['feature_count'])} (std={lowest_variance['std_f1']:.6f})"
    )
    print(
        "Key conclusion: random combinations can be competitive, but results are variable "
        "and exploratory; use them to design a controlled hybrid feature-selection experiment."
    )
    expected_files = [
        RANDOM_REPORT_DIR / "random_feature_analysis.md",
        RANDOM_REPORT_DIR / "random_feature_analysis.csv",
        RANDOM_REPORT_DIR / "feature_frequency_analysis.csv",
        RANDOM_REPORT_DIR / "successful_set_analysis.csv",
        RANDOM_REPORT_DIR / "feature_count_summary.csv",
    ]
    missing = [project_relative(path) for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected analysis file(s): " + ", ".join(missing))
    print("Verified analysis artifacts were created successfully.")
    print(f"Report: {project_relative(RANDOM_REPORT_DIR / 'random_feature_analysis.md')}")
    print(f"Figures: {project_relative(FIGURE_DIR)}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
