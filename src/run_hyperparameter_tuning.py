"""Phase 4 hyperparameter tuning on training data only.

Tunes candidate feature sets (Top-10, Random_12_07, and Baseline-65) with
RandomizedSearchCV and the project's Stratified 5-fold splitter.
This script uses training data only and never loads the test set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV

import run_kfold_cv as cv


PROJECT_ROOT = cv.PROJECT_ROOT
DEFAULT_PHASE3_DIR = cv.DEFAULT_PHASE3_DIR
DEFAULT_PHASE3_METADATA = cv.DEFAULT_PHASE3_METADATA
DEFAULT_BASELINE_IMPORTANCE = cv.DEFAULT_BASELINE_IMPORTANCE
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "hyperparameter_tuning"

RANDOM_STATE = cv.RANDOM_STATE
N_SPLITS = cv.N_SPLITS
N_ITER = 20
TOP_CONFIGS = 10
EXPECTED_FOLD_FINGERPRINT = "8a2c36f102175dacfaee0995b24b6d7e0f712e5167130bbe13711f49ed904655"

PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
}

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

PROTECTED_PATHS = (
    PROJECT_ROOT / "reports" / "kfold_cv" / "cv_fold_results.csv",
    PROJECT_ROOT / "reports" / "kfold_cv" / "cv_summary.csv",
    PROJECT_ROOT / "reports" / "kfold_cv" / "cv_results.json",
    PROJECT_ROOT / "reports" / "kfold_cv" / "cv_analysis.md",
    PROJECT_ROOT / "reports" / "baseline" / "metrics.json",
    PROJECT_ROOT / "reports" / "random_feature_selection" / "random_results.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune Random Forest hyperparameters on training data only."
    )
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--phase3-metadata", type=Path, default=DEFAULT_PHASE3_METADATA)
    parser.add_argument("--baseline-importance", type=Path, default=DEFAULT_BASELINE_IMPORTANCE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    """Convert numpy scalars so JSON/CSV output is plain Python."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def json_ready_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize sklearn parameter dicts for artifacts."""
    ready: dict[str, Any] = {}
    for key, value in params.items():
        ready[key] = to_builtin(value)
    return ready


def sampled_param_grid(n_iter: int) -> list[dict[str, Any]]:
    """Draw the exact RandomizedSearchCV candidate list with the project seed."""
    sampler = ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )
    return [json_ready_params(dict(params)) for params in sampler]


def build_search_estimator() -> RandomForestClassifier:
    """RF estimator for search: same seed/family as Phase 4, n_jobs=1 to avoid nested parallelism."""
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
        n_jobs=1,
    )


def fold_metric_stats(cv_results: pd.DataFrame, row_index: int, metric: str) -> dict[str, float]:
    """Mean and sample SD (ddof=1) from the five validation-fold scores."""
    values = np.array(
        [float(cv_results.loc[row_index, f"split{fold}_test_{metric}"]) for fold in range(N_SPLITS)],
        dtype=np.float64,
    )
    return {
        "mean": float(values.mean()),
        "std_sample": float(values.std(ddof=1)),
        "std_sklearn": float(cv_results.loc[row_index, f"std_test_{metric}"]),
        "folds": [float(value) for value in values],
    }


def ranking_rows(cv_results: pd.DataFrame, n_top: int) -> list[dict[str, Any]]:
    """Return the top configurations ranked by mean CV F1."""
    ordered = cv_results.sort_values(
        ["mean_test_f1", "std_test_f1", "rank_test_f1"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    rows: list[dict[str, Any]] = []
    for rank, (index, record) in enumerate(ordered.head(n_top).iterrows(), start=1):
        params = json_ready_params(record["params"])
        f1 = fold_metric_stats(cv_results, index, "f1")
        accuracy = fold_metric_stats(cv_results, index, "accuracy")
        precision = fold_metric_stats(cv_results, index, "precision")
        recall = fold_metric_stats(cv_results, index, "recall")
        rows.append(
            {
                "rank": rank,
                "candidate_index": int(index),
                "params": params,
                "mean_cv_accuracy": accuracy["mean"],
                "std_cv_accuracy": accuracy["std_sample"],
                "mean_cv_precision": precision["mean"],
                "std_cv_precision": precision["std_sample"],
                "mean_cv_recall": recall["mean"],
                "std_cv_recall": recall["std_sample"],
                "mean_cv_f1": f1["mean"],
                "std_cv_f1": f1["std_sample"],
                "sklearn_std_cv_f1": f1["std_sklearn"],
            }
        )
    return rows


def load_random_12_07_features(feature_names: list[str]) -> list[str]:
    """Reproducibly extract the exact 12 features of Random_12_07."""
    rng = np.random.default_rng(RANDOM_STATE)
    all_subset_keys: set[tuple[str, ...]] = set()
    experiments: dict[str, list[str]] = {}

    for size in [5, 8, 10, 12, 15, 20, 30]:
        generated_for_size = 0
        attempts = 0
        while generated_for_size < 10:
            attempts += 1
            selected = rng.choice(feature_names, size=size, replace=False).tolist()
            subset_key = tuple(sorted(selected))
            if subset_key in all_subset_keys:
                continue
            all_subset_keys.add(subset_key)
            generated_for_size += 1
            name = f"Random_{size}_{generated_for_size:02d}"
            experiments[name] = selected

    return experiments["Random_12_07"]


def run_search(
    subset_id: str,
    features: list[str],
    train_df: pd.DataFrame,
    n_iter: int,
) -> dict[str, Any]:
    """Run RandomizedSearchCV on one frozen feature subset using training rows only."""
    x = train_df[features]
    y = train_df[cv.TARGET_COLUMN].astype(int).to_numpy()
    search = RandomizedSearchCV(
        estimator=build_search_estimator(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=SCORING,
        refit="f1",
        cv=cv.make_splitter(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x, y)
    cv_results = pd.DataFrame(search.cv_results_)
    best_index = int(search.best_index_)
    ranking = ranking_rows(cv_results, TOP_CONFIGS)
    sampled = sampled_param_grid(n_iter)
    realized = [json_ready_params(params) for params in cv_results["params"].tolist()]
    if realized != sampled:
        raise RuntimeError(f"{subset_id}: sampled parameter grid did not match RandomizedSearchCV.")
    best = ranking[0]
    n_candidates = int(len(cv_results))
    return {
        "subset_id": subset_id,
        "feature_count": len(features),
        "features": features,
        "n_iter": n_iter,
        "n_splits": N_SPLITS,
        "n_candidates_evaluated": n_candidates,
        "n_cv_fits": n_candidates * N_SPLITS,
        "best_hyperparameters": json_ready_params(search.best_params_),
        "best_mean_cv_f1": best["mean_cv_f1"],
        "best_std_cv_f1": best["std_cv_f1"],
        "best_cv_metrics": {
            "accuracy": {
                "mean": best["mean_cv_accuracy"],
                "std": best["std_cv_accuracy"],
            },
            "precision": {
                "mean": best["mean_cv_precision"],
                "std": best["std_cv_precision"],
            },
            "recall": {
                "mean": best["mean_cv_recall"],
                "std": best["std_cv_recall"],
            },
            "f1": {
                "mean": best["mean_cv_f1"],
                "std": best["std_cv_f1"],
            },
        },
        "top_configurations": ranking,
        "cv_results_frame": cv_results,
        "sampled_parameter_grid": sampled,
        "best_index": best_index,
        "reproducible_parameter_sample": True,
    }


def write_cv_results_csv(path: Path, cv_results: pd.DataFrame) -> None:
    """Write sklearn cv_results_ with JSON-encoded params."""
    frame = cv_results.copy()
    frame["params"] = frame["params"].map(lambda params: json.dumps(json_ready_params(params)))
    frame.to_csv(path, index=False)


def write_ranking_csv(path: Path, ranking: list[dict[str, Any]]) -> None:
    """Write the ranked candidate table."""
    rows = []
    for item in ranking:
        row = {key: value for key, value in item.items() if key != "params"}
        row["params"] = json.dumps(item["params"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def format_mean_std(mean: float, std: float) -> str:
    """Format mean ± sample SD."""
    return f"{mean:.6f} ± {std:.6f}"


def write_artifacts(
    report_dir: Path,
    results: list[dict[str, Any]],
    fold_fingerprint: str,
    train_samples: int,
    n_iter: int,
    phase3_dir: Path,
) -> None:
    """Write hyperparameter tuning artifacts."""
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    payload_results = []
    for result in results:
        subset_key = result["subset_id"].lower().replace("-", "").replace("_", "")
        write_cv_results_csv(report_dir / f"tuning_cv_results_{subset_key}.csv", result["cv_results_frame"])
        write_ranking_csv(report_dir / f"tuning_top_configurations_{subset_key}.csv", result["top_configurations"])
        metrics = result["best_cv_metrics"]
        summary_rows.append(
            {
                "subset_id": result["subset_id"],
                "feature_count": result["feature_count"],
                "n_iter": result["n_iter"],
                "n_cv_fits": result["n_cv_fits"],
                "best_hyperparameters": json.dumps(result["best_hyperparameters"]),
                "accuracy_mean": metrics["accuracy"]["mean"],
                "accuracy_std": metrics["accuracy"]["std"],
                "precision_mean": metrics["precision"]["mean"],
                "precision_std": metrics["precision"]["std"],
                "recall_mean": metrics["recall"]["mean"],
                "recall_std": metrics["recall"]["std"],
                "f1_mean": metrics["f1"]["mean"],
                "f1_std": metrics["f1"]["std"],
                "features": json.dumps(result["features"]),
            }
        )
        payload_results.append({key: value for key, value in result.items() if key != "cv_results_frame"})

    pd.DataFrame(summary_rows).to_csv(report_dir / "tuning_summary.csv", index=False)
    (report_dir / "tuning_best_params.json").write_text(
        json.dumps(
            {
                result["subset_id"]: result["best_hyperparameters"]
                for result in results
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Phase 4 Hyperparameter Tuning Analysis",
        "",
        "Training-only `RandomizedSearchCV` for candidate feature sets: **Top-10** and **Random_12_07** (along with Baseline-65 for reference).",
        "The test set was strictly kept untouched during this search.",
        "",
        "## Search Space & Methodology",
        "",
        f"- **Cross-Validation Splitter:** `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})`",
        f"- **Random State / Seed:** `{RANDOM_STATE}`",
        f"- **Evaluated Iterations (`n_iter`):** `{n_iter}` configurations per candidate subset",
        f"- **Total CV Fits per Subset:** `{n_iter * N_SPLITS}` fits",
        "- **Primary Refit Metric:** Binary classification F1-Score (`refit='f1'`)",
        "- **Monitored Metrics:** F1, Precision, Recall, Accuracy",
        "- **Standard Deviation Reporting:** Sample standard deviation (`ddof=1`) across 5 validation folds",
        "",
        "### Hyperparameter Search Space",
        "",
        "```json",
        json.dumps(PARAM_DISTRIBUTIONS, indent=2),
        "```",
        "",
        "## Best Cross-Validation Performance (Mean ± Sample Std)",
        "",
        "| Candidate Subset | Feature Count | Accuracy | Precision | Recall | F1-Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for result in results:
        metrics = result["best_cv_metrics"]
        lines.append(
            "| **{id}** | {n} | {acc} | {prec} | {rec} | **{f1}** |".format(
                id=result["subset_id"],
                n=result["feature_count"],
                acc=format_mean_std(metrics["accuracy"]["mean"], metrics["accuracy"]["std"]),
                prec=format_mean_std(metrics["precision"]["mean"], metrics["precision"]["std"]),
                rec=format_mean_std(metrics["recall"]["mean"], metrics["recall"]["std"]),
                f1=format_mean_std(metrics["f1"]["mean"], metrics["f1"]["std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Best Hyperparameter Configurations",
            "",
        ]
    )
    for result in results:
        lines.append(f"### {result['subset_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result["best_hyperparameters"], indent=2))
        lines.append("```")
        lines.append("")
        lines.append("Top 5 configurations ranked by mean validation F1:")
        lines.append("")
        for item in result["top_configurations"][:5]:
            lines.append(
                f"- **Rank {item['rank']}:** F1 = `{item['mean_cv_f1']:.6f} ± {item['std_cv_f1']:.6f}`; "
                f"Params: `{json.dumps(item['params'])}`"
            )
        lines.append("")

    (report_dir / "tuning_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "study": "Phase 4 RandomizedSearchCV Hyperparameter Tuning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": cv.project_relative(phase3_dir / "train.parquet"),
        "phase3_test_used": False,
        "target_column": cv.TARGET_COLUMN,
        "train_samples": train_samples,
        "cv": {
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "search": {
            "method": "sklearn.model_selection.RandomizedSearchCV",
            "n_iter": n_iter,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "refit": "f1",
            "random_state": RANDOM_STATE,
            "param_distributions": PARAM_DISTRIBUTIONS,
            "n_jobs_search": -1,
            "n_jobs_estimator": 1,
        },
        "std_estimator": "sample standard deviation across the 5 validation folds (numpy.std ddof=1)",
        "fold_index_sha256": fold_fingerprint,
        "fold_index_matches_original_cv": fold_fingerprint == EXPECTED_FOLD_FINGERPRINT,
        "parameter_sample_reproducible": all(
            result["reproducible_parameter_sample"] for result in results
        ),
        "hyperparameter_tuning_applied": True,
        "subsets": payload_results,
    }
    (report_dir / "tuning_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    readme = f"""# Phase 4 Hyperparameter Tuning

## Purpose

Training-only RandomizedSearchCV for candidate feature sets: **Top-10**, **Random_12_07**, and **Baseline-65** Random Forest models.
This tuning was performed strictly on the training set (`train.parquet`) using Stratified 5-Fold Cross-Validation.
The test set remained completely untouched throughout.

## Methodology

- **Training Data:** `{cv.project_relative(phase3_dir / "train.parquet")}` ({train_samples:,} rows).
- **Splitter:** `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state=42)`.
- **Search:** `RandomizedSearchCV(n_iter={n_iter}, scoring=f1, random_state=42)`.
- **Candidates Tuned:** Top-10 (10 features), Random_12_07 (12 features), Baseline-65 (65 features).

## Generated Artifacts

- `tuning_summary.csv`: Summary of best hyperparameters and CV metrics per candidate.
- `tuning_best_params.json`: Best parameter dictionary per candidate.
- `tuning_cv_results_*.csv`: Full sklearn `cv_results_` tables for all evaluated configurations.
- `tuning_top_configurations_*.csv`: Top 10 ranked parameter configurations per candidate.
- `tuning_results.json`: Complete serialized metadata.
- `tuning_analysis.md`: Detailed analysis and comparison markdown.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    """Tune Top-10, Random_12_07, and Baseline-65 on training folds only."""
    args = parse_args()
    phase3_dir = cv.resolve_project_path(args.phase3_dir)
    phase3_metadata_path = cv.resolve_project_path(args.phase3_metadata)
    importance_path = cv.resolve_project_path(args.baseline_importance)
    report_dir = cv.resolve_project_path(args.report_dir)
    n_iter = int(args.n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")

    print("=== STEP 2: HYPERPARAMETER TUNING (TRAINING DATA ONLY) ===")
    print("NOTE: The test set will NOT be loaded or scored during hyperparameter tuning.")

    feature_names = cv.load_feature_names(phase3_metadata_path)
    train_df = cv.load_training_frame(phase3_dir)
    cv.validate_training_frame(train_df, feature_names)
    
    top10_features = cv.load_top10_features(importance_path, feature_names)
    random1207_features = load_random_12_07_features(feature_names)
    
    y = train_df[cv.TARGET_COLUMN].astype(int).to_numpy()
    fold_fingerprint = cv.fold_index_fingerprint(y)
    if fold_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError("StratifiedKFold indices do not match the expected CV fingerprint.")

    subsets = [
        ("Top-10", top10_features),
        ("Random_12_07", random1207_features),
        ("Baseline-65", feature_names),
    ]
    results: list[dict[str, Any]] = []
    for subset_id, features in subsets:
        print(f"\n--- Tuning {subset_id} ({len(features)} features) | n_iter={n_iter} ({n_iter * N_SPLITS} fits) ---")
        result = run_search(subset_id, features, train_df, n_iter)
        metrics = result["best_cv_metrics"]["f1"]
        print(
            f"-> {subset_id} Best CV F1 = {metrics['mean']:.6f} ± {metrics['std']:.6f}\n"
            f"   Best Params = {json.dumps(result['best_hyperparameters'])}"
        )
        results.append(result)

    write_artifacts(
        report_dir=report_dir,
        results=results,
        fold_fingerprint=fold_fingerprint,
        train_samples=int(len(train_df)),
        n_iter=n_iter,
        phase3_dir=phase3_dir,
    )
    print(f"\nHyperparameter tuning artifacts saved to: {cv.project_relative(report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
