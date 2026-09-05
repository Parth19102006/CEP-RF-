"""Phase 4 hyperparameter tuning on training data only.

Tunes frozen Baseline-65 and Top-10 feature sets with RandomizedSearchCV and
the existing Stratified 5-fold splitter. Random-12 / Random_34 is excluded
from selection because it was promoted using original test-set performance.
This script never loads the original test split and does not choose a final model.
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
    """Write new tuning artifacts without touching Phase 4 or K-Fold reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    payload_results = []
    for result in results:
        subset_key = result["subset_id"].lower().replace("-", "")
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
        "# Phase 4 Hyperparameter Tuning",
        "",
        "Training-only `RandomizedSearchCV` for frozen Baseline-65 and Top-10.",
        "Random-12 / `Random_34` was not tuned for selection.",
        "No final model was selected. The original test set was not loaded or scored.",
        "",
        "## Search",
        "",
        f"- Splitter: `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})`",
        f"- Search seed: `{RANDOM_STATE}`",
        f"- `n_iter`: `{n_iter}`",
        f"- CV fits per subset: `{n_iter * N_SPLITS}`",
        "- Scoring: binary F1 (`refit='f1'`), plus accuracy, precision, and recall",
        "- Fold std in summaries: sample SD (`ddof=1`) from the five validation folds",
        "- Estimator `n_jobs=1`; search `n_jobs=-1` (n_jobs is not a tuned parameter)",
        "",
        "## Parameter distributions",
        "",
        "```json",
        json.dumps(PARAM_DISTRIBUTIONS, indent=2),
        "```",
        "",
        "## Best CV metrics (mean ± sample SD)",
        "",
        "| Feature Set | # Features | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for result in results:
        metrics = result["best_cv_metrics"]
        lines.append(
            "| {id} | {n} | {acc} | {prec} | {rec} | {f1} |".format(
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
            "## Best hyperparameters",
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
        lines.append("Top configurations:")
        lines.append("")
        for item in result["top_configurations"][:5]:
            lines.append(
                f"- Rank {item['rank']}: F1 {item['mean_cv_f1']:.6f} ± {item['std_cv_f1']:.6f}; "
                f"`{json.dumps(item['params'])}`"
            )
        lines.append("")
    lines.extend(
        [
            "Final model selection and final test evaluation have not been performed.",
            "",
        ]
    )
    (report_dir / "tuning_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "study": "Phase 4 RandomizedSearchCV hyperparameter tuning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": cv.project_relative(phase3_dir / "train.parquet"),
        "phase3_test_used": False,
        "random12_tuned": False,
        "random12_exclusion_reason": (
            "Random_34 / Random-12 was previously promoted using original test-set "
            "performance and is excluded from confirmatory selection."
        ),
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
        "final_model_selected": False,
        "original_test_evaluated": False,
        "subsets": payload_results,
    }
    (report_dir / "tuning_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    readme = f"""# Phase 4 Hyperparameter Tuning

## Purpose

Train-only RandomizedSearchCV for frozen Baseline-65 and Top-10 Random Forest
models. This directory is additive and does not replace Phase 4 or K-Fold artifacts.

## Design

- Data: `{cv.project_relative(phase3_dir / "train.parquet")}` only.
- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Search: `RandomizedSearchCV(n_iter={n_iter}, scoring=f1, random_state=42)`.
- Excluded from selection: Random-12 / `Random_34`.

## Files

- `tuning_summary.csv`: best hyperparameters and CV metrics per subset.
- `tuning_best_params.json`: best parameter dicts.
- `tuning_cv_results_baseline65.csv` / `tuning_cv_results_top10.csv`: full sklearn `cv_results_`.
- `tuning_top_configurations_*.csv`: ranked candidate configurations.
- `tuning_results.json`: complete metadata.
- `tuning_analysis.md`: comparison text without final model selection.

## Reproducibility

```powershell
python src/run_hyperparameter_tuning.py
```
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")


def assert_protected_artifacts_untouched(before_hashes: dict[Path, str]) -> None:
    """Fail if a completed Phase 4 / K-Fold artifact changed during this run."""
    for path, expected in before_hashes.items():
        if not path.exists():
            continue
        actual = path.read_bytes()
        digest = __import__("hashlib").sha256(actual).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Protected artifact was modified: {cv.project_relative(path)}")


def snapshot_protected_artifacts() -> dict[Path, str]:
    """Hash existing Phase 4 / K-Fold files so they can be proven unchanged."""
    hashes: dict[Path, str] = {}
    for path in PROTECTED_PATHS:
        if path.exists():
            hashes[path] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    return hashes


def main() -> int:
    """Tune Baseline-65 and Top-10 on training folds only."""
    args = parse_args()
    phase3_dir = cv.resolve_project_path(args.phase3_dir)
    phase3_metadata_path = cv.resolve_project_path(args.phase3_metadata)
    importance_path = cv.resolve_project_path(args.baseline_importance)
    report_dir = cv.resolve_project_path(args.report_dir)
    n_iter = int(args.n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")

    protected = snapshot_protected_artifacts()
    print("Original test set will not be loaded or scored.")

    feature_names = cv.load_feature_names(phase3_metadata_path)
    train_df = cv.load_training_frame(phase3_dir)
    cv.validate_training_frame(train_df, feature_names)
    top10 = cv.load_top10_features(importance_path, feature_names)
    y = train_df[cv.TARGET_COLUMN].astype(int).to_numpy()
    fold_fingerprint = cv.fold_index_fingerprint(y)
    if fold_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError("StratifiedKFold indices do not match the existing CV fingerprint.")

    subsets = [
        ("Baseline-65", feature_names),
        ("Top-10", top10),
    ]
    results: list[dict[str, Any]] = []
    for subset_id, features in subsets:
        print(f"Tuning {subset_id} ({len(features)} features), n_iter={n_iter}...")
        result = run_search(subset_id, features, train_df, n_iter)
        metrics = result["best_cv_metrics"]["f1"]
        print(
            f"{subset_id} best CV F1={metrics['mean']:.6f} ± {metrics['std']:.6f}; "
            f"params={result['best_hyperparameters']}"
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
    assert_protected_artifacts_untouched(protected)
    print(f"Reports: {cv.project_relative(report_dir)}")
    print("Final model selection and final test evaluation were not performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
