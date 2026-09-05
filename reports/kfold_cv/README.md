# Phase 4 K-Fold Cross-Validation

## Purpose

Training-only stratified 5-fold CV for the existing Phase 4 shortlisted subsets.
This directory is additive. It does not replace baseline or random-feature-selection
artifacts.

## Design

- Data: `data\processed\phase3\train.parquet` only; the original test set is untouched.
- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Model: the frozen Phase 4 Random Forest (`random_state=42`).
- Subsets: Top-10 from baseline importance ranks; Random-12 from `Random_34`.

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
