# Phase 4 Hyperparameter Tuning

## Purpose

Training-only RandomizedSearchCV for candidate feature sets: **Top-10**, **Random_12_07**, and **Baseline-65** Random Forest models.
This tuning was performed strictly on the training set (`train.parquet`) using Stratified 5-Fold Cross-Validation.
The test set remained completely untouched throughout.

## Methodology

- **Training Data:** `data\processed\phase3\train.parquet` (125,170 rows).
- **Splitter:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Search:** `RandomizedSearchCV(n_iter=20, scoring=f1, random_state=42)`.
- **Candidates Tuned:** Top-10 (10 features), Random_12_07 (12 features), Baseline-65 (65 features).

## Generated Artifacts

- `tuning_summary.csv`: Summary of best hyperparameters and CV metrics per candidate.
- `tuning_best_params.json`: Best parameter dictionary per candidate.
- `tuning_cv_results_*.csv`: Full sklearn `cv_results_` tables for all evaluated configurations.
- `tuning_top_configurations_*.csv`: Top 10 ranked parameter configurations per candidate.
- `tuning_results.json`: Complete serialized metadata.
- `tuning_analysis.md`: Detailed analysis and comparison markdown.
