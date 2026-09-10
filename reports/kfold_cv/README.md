# Phase 4 K-Fold Cross-Validation

## ⚠️ Historical Artifact Notice

> **Note:** The K-Fold artifact in this directory corresponds to an **earlier
> Random-12 feature configuration** (experiment ID `Random_34`). Its 12-feature
> list is **NOT** the final Phase 4 random candidate.
> The final Phase 4 random candidate is **Random_12_07**, whose authoritative
> validation, hyperparameter-tuning, and final test results are documented under
> `reports/hyperparameter_tuning/` and `reports/final_evaluation/`.

## Purpose

Training-only stratified 5-fold CV for the existing Phase 4 shortlisted subsets.
This directory is additive. It does not replace baseline or random-feature-selection
artifacts.

## Design

- Data: `data\processed\phase3\train.parquet` only; the original test set is untouched.
- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Model: the frozen Phase 4 Random Forest (`random_state=42`).
- Subsets: Top-10 from baseline importance ranks; Random-12 (earlier configuration,
  source experiment `Random_34`) — **not** the final Random_12_07 feature set.

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
