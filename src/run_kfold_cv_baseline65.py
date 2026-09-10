"""Add the 65-feature baseline to the existing train-only Stratified 5-fold CV.

This script does not modify src/run_kfold_cv.py or completed Phase 4 / original
K-Fold artifacts. It reuses that CV and Random Forest configuration, the same
five folds, and writes additive reports only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_kfold_cv as cv


PROJECT_ROOT = cv.PROJECT_ROOT
DEFAULT_EXISTING_CV_DIR = cv.DEFAULT_REPORT_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "kfold_cv" / "baseline65"
SUBSET_ID = "Baseline-65"
METRIC_NAMES = cv.METRIC_NAMES
EXPECTED_FOLD_FINGERPRINT = "8a2c36f102175dacfaee0995b24b6d7e0f712e5167130bbe13711f49ed904655"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stratified 5-fold CV for the frozen 65-feature baseline."
    )
    parser.add_argument("--phase3-dir", type=Path, default=cv.DEFAULT_PHASE3_DIR)
    parser.add_argument("--phase3-metadata", type=Path, default=cv.DEFAULT_PHASE3_METADATA)
    parser.add_argument("--existing-cv-dir", type=Path, default=DEFAULT_EXISTING_CV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_preserved_cv_tables(existing_cv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the frozen Top-10 / Random-12 CV artifacts without rewriting them."""
    fold_path = existing_cv_dir / "cv_fold_results.csv"
    summary_path = existing_cv_dir / "cv_summary.csv"
    json_path = existing_cv_dir / "cv_results.json"
    missing = [str(path) for path in (fold_path, summary_path, json_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing existing CV artifact(s): " + ", ".join(missing))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    recorded_fingerprint = payload.get("fold_index_sha256")
    if recorded_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise ValueError(
            "Existing CV fold fingerprint does not match the expected seed/split hash: "
            f"{recorded_fingerprint}"
        )
    return pd.read_csv(fold_path), pd.read_csv(summary_path), payload


def format_row(subset_id: str, feature_count: int, means_stds: dict[str, float]) -> str:
    """Format one markdown summary row."""
    return (
        f"| {subset_id} | {feature_count} | "
        f"{cv.format_mean_std(means_stds['accuracy_mean'], means_stds['accuracy_std'])} | "
        f"{cv.format_mean_std(means_stds['precision_mean'], means_stds['precision_std'])} | "
        f"{cv.format_mean_std(means_stds['recall_mean'], means_stds['recall_std'])} | "
        f"{cv.format_mean_std(means_stds['f1_score_mean'], means_stds['f1_score_std'])} |"
    )


def summary_from_csv_row(row: pd.Series) -> dict[str, float]:
    """Extract mean/std fields from a preserved summary CSV row."""
    return {
        metric: float(row[metric])
        for metric in (
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_score_mean",
            "f1_score_std",
        )
    }


def write_comparison(
    output_dir: Path,
    preserved_summary: pd.DataFrame,
    preserved_folds: pd.DataFrame,
    baseline_folds: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    fold_fingerprint: str,
    train_samples: int,
    phase3_dir: Path,
    existing_cv_dir: Path,
) -> None:
    """Write additive CV artifacts; never overwrite the original K-Fold files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() == existing_cv_dir.resolve():
        raise ValueError("Refusing to write into the original K-Fold report directory.")

    top10 = preserved_summary.loc[preserved_summary["subset_id"] == "Top-10"]
    random12 = preserved_summary.loc[preserved_summary["subset_id"] == "Random-12"]
    if top10.empty or random12.empty:
        raise ValueError("Preserved CV summary is missing Top-10 or Random-12.")
    top10_stats = summary_from_csv_row(top10.iloc[0])
    random12_stats = summary_from_csv_row(random12.iloc[0])
    baseline_stats = {
        "accuracy_mean": baseline_summary["accuracy_mean"],
        "accuracy_std": baseline_summary["accuracy_std"],
        "precision_mean": baseline_summary["precision_mean"],
        "precision_std": baseline_summary["precision_std"],
        "recall_mean": baseline_summary["recall_mean"],
        "recall_std": baseline_summary["recall_std"],
        "f1_score_mean": baseline_summary["f1_score_mean"],
        "f1_score_std": baseline_summary["f1_score_std"],
    }

    f1_gap = baseline_stats["f1_score_mean"] - top10_stats["f1_score_mean"]
    if f1_gap > 0:
        f1_winner = "Baseline-65 has the higher mean F1"
    elif f1_gap < 0:
        f1_winner = "Top-10 has the higher mean F1"
    else:
        f1_winner = "Top-10 and Baseline-65 have the same mean F1"
    more_stable = (
        "Top-10"
        if top10_stats["f1_score_std"] < baseline_stats["f1_score_std"]
        else "Baseline-65"
    )

    combined_summary = pd.concat(
        [
            preserved_summary.copy(),
            pd.DataFrame(
                [
                    {
                        "subset_id": SUBSET_ID,
                        "feature_count": baseline_summary["feature_count"],
                        "features": json.dumps(baseline_summary["features"]),
                        **baseline_stats,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    combined_folds = pd.concat(
        [preserved_folds.copy(), pd.DataFrame(baseline_folds)],
        ignore_index=True,
    )
    combined_folds.to_csv(output_dir / "cv_fold_results.csv", index=False)
    combined_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    pd.DataFrame(baseline_folds).to_csv(output_dir / "cv_baseline65_fold_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "subset_id": SUBSET_ID,
                "feature_count": baseline_summary["feature_count"],
                "features": json.dumps(baseline_summary["features"]),
                **baseline_stats,
            }
        ]
    ).to_csv(output_dir / "cv_baseline65_summary.csv", index=False)

    table = "\n".join(
        [
            format_row("Top-10", int(top10.iloc[0]["feature_count"]), top10_stats),
            format_row("Random-12", int(random12.iloc[0]["feature_count"]), random12_stats),
            format_row(SUBSET_ID, int(baseline_summary["feature_count"]), baseline_stats),
        ]
    )
    analysis = f"""# Train-only CV: Top-10 vs 65-feature baseline

This additive study reuses the frozen Stratified 5-fold split and Phase 4
Random Forest configuration. Original K-Fold files under
`{cv.project_relative(existing_cv_dir)}` were read, not overwritten.
Random-12 is copied for completeness and is not used as a train-only
selection candidate.

## Configuration

- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Same fold-index SHA-256 as the original CV: `{fold_fingerprint}`
- Model: unchanged Phase 4 `RandomForestClassifier` (`random_state=42`, `n_jobs=-1`)
- Data: `{cv.project_relative(phase3_dir / "train.parquet")}` only
- Standard deviation: sample standard deviation across folds (`ddof=1`)
- Test parquet: never loaded or scored

## CV summary (mean ± std)

| Feature Set | # Features | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | --- | --- | --- |
{table}

## Top-10 vs Baseline-65

{f1_winner} (mean-F1 difference Baseline-65 − Top-10 = {f1_gap:+.6f}).
{more_stable} is more stable on F1.

This comparison is train-only. It does not select a final model and does not
tune hyperparameters.

## Preserved original CV

Top-10 and Random-12 fold metrics were copied from
`{cv.project_relative(existing_cv_dir / "cv_fold_results.csv")}` without rerunning
those subsets.
"""
    (output_dir / "cv_analysis.md").write_text(analysis, encoding="utf-8")

    payload = {
        "study": "Additive train-only Stratified 5-fold CV for the 65-feature baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": cv.project_relative(phase3_dir / "train.parquet"),
        "phase3_test_used": False,
        "original_kfold_dir": cv.project_relative(existing_cv_dir),
        "original_kfold_overwritten": False,
        "target_column": cv.TARGET_COLUMN,
        "train_samples": train_samples,
        "cv": {
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "n_splits": cv.N_SPLITS,
            "shuffle": True,
            "random_state": cv.RANDOM_STATE,
        },
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_config": cv.build_model().get_params(deep=False),
        "metrics": list(METRIC_NAMES),
        "std_estimator": "sample standard deviation across folds (numpy.std ddof=1)",
        "fold_index_sha256": fold_fingerprint,
        "fold_index_matches_original_cv": fold_fingerprint == EXPECTED_FOLD_FINGERPRINT,
        "hyperparameter_tuning_applied": False,
        "final_model_selected": False,
        "original_test_evaluated": False,
        "preserved_subsets": ["Top-10", "Random-12"],
        "new_subset": SUBSET_ID,
        "baseline_65": baseline_summary,
        "fold_results": baseline_folds,
        "preserved_top10": top10_stats,
        "preserved_random12": random12_stats,
    }
    (output_dir / "cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    readme = f"""# Additive 65-feature baseline K-Fold CV

## Purpose

Train-only Stratified 5-fold CV for the frozen 65-feature Phase 4 baseline,
compared with the already completed Top-10 CV. Original K-Fold artifacts are
preserved.

## Design

- Same splitter, seed, and RF configuration as `src/run_kfold_cv.py`.
- Same five folds (fold-index SHA-256 `{EXPECTED_FOLD_FINGERPRINT}`).
- Training data only.

## Files

- `cv_baseline65_fold_results.csv`: new Baseline-65 fold metrics.
- `cv_baseline65_summary.csv`: Baseline-65 mean ± sample SD.
- `cv_fold_results.csv`: preserved Top-10 / Random-12 folds plus Baseline-65.
- `cv_summary.csv`: preserved summaries plus Baseline-65.
- `cv_results.json`: configuration and numeric payload.
- `cv_analysis.md`: Top-10 vs Baseline-65 comparison.

## Reproducibility

```powershell
python src/run_kfold_cv_baseline65.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    """Run 65-feature CV on the frozen training folds and write additive reports."""
    args = parse_args()
    phase3_dir = cv.resolve_project_path(args.phase3_dir)
    phase3_metadata_path = cv.resolve_project_path(args.phase3_metadata)
    existing_cv_dir = cv.resolve_project_path(args.existing_cv_dir)
    output_dir = cv.resolve_project_path(args.output_dir)

    test_path = phase3_dir / "test.parquet"
    if test_path.exists():
        print(f"Leaving original test set untouched: {cv.project_relative(test_path)}")

    preserved_folds, preserved_summary, _payload = load_preserved_cv_tables(existing_cv_dir)
    feature_names = cv.load_feature_names(phase3_metadata_path)
    train_df = cv.load_training_frame(phase3_dir)
    cv.validate_training_frame(train_df, feature_names)
    y = train_df[cv.TARGET_COLUMN].astype(int).to_numpy()
    fold_fingerprint = cv.fold_index_fingerprint(y)
    if fold_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError(
            "Computed StratifiedKFold indices do not match the original CV fingerprint."
        )

    subset = {
        "subset_id": SUBSET_ID,
        "feature_count": len(feature_names),
        "features": feature_names,
        "source_artifact": cv.project_relative(phase3_metadata_path),
        "source_experiment_id": None,
        "sampling_seed": None,
        "selection_note": (
            "Frozen Phase 3 / Phase 4 all-valid-features baseline set "
            "(65 features). Defined independently of the test set."
        ),
    }
    fold_rows, summary = cv.run_cv_for_subset(subset, train_df)
    print(
        f"{summary['subset_id']}: "
        f"F1={summary['f1_score_mean']:.6f} ± {summary['f1_score_std']:.6f}"
    )

    write_comparison(
        output_dir=output_dir,
        preserved_summary=preserved_summary,
        preserved_folds=preserved_folds,
        baseline_folds=fold_rows,
        baseline_summary=summary,
        fold_fingerprint=fold_fingerprint,
        train_samples=int(len(train_df)),
        phase3_dir=phase3_dir,
        existing_cv_dir=existing_cv_dir,
    )
    print(f"Reports: {cv.project_relative(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
