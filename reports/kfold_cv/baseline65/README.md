# Additive 65-feature baseline K-Fold CV

## Purpose

Train-only Stratified 5-fold CV for the frozen 65-feature Phase 4 baseline,
compared with the already completed Top-10 CV. Original K-Fold artifacts are
preserved.

## Design

- Same splitter, seed, and RF configuration as `src/run_kfold_cv.py`.
- Same five folds (fold-index SHA-256 `8a2c36f102175dacfaee0995b24b6d7e0f712e5167130bbe13711f49ed904655`).
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
