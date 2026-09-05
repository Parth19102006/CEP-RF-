# Controlled Random Feature-Selection Study

## Purpose

This directory records a newly defined, reproducible 70-experiment Phase 4 study. The original random-feature-selection artifacts and methodology were unavailable, so this study must not be interpreted as a reproduction of an earlier run.

## Design

- Feature pool: the 65 final Phase 3 features in `reports\preprocessing\preprocessing_metadata.json`.
- Allocation: 10 random subsets each with 5, 8, 10, 12, 15, 20, and 30 features (70 total).
- Sampling: uniform sampling without replacement; baseline feature importance/ranking is not used.
- Master sampling seed: `20260903`.
- Per-experiment seed: `master_sampling_seed + experiment_number - 1`; experiments are ordered `Random_01` through `Random_70`.
- Data: `data\processed\phase3\train.parquet` and `data\processed\phase3\test.parquet` with target `Label_binary`.
- Preprocessing: the existing frozen Phase 3 representation is reused without scaling, PCA, sampling, or correlation filtering.

## Model

`RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True, class_weight=None, random_state=42, n_jobs=-1)`

## Evaluation

Each subset is trained on the existing Phase 3 training data and evaluated on the same Phase 3 test data. Saved metrics are accuracy, precision, recall, F1, ROC-AUC, TP, TN, FP, FN, training time, and prediction time.

## Files

- `random_feature_sets.csv`: exact selected feature names and sampling seed for each experiment.
- `random_results.csv`: one metric row per experiment.
- `random_results.json`: complete metadata, exact feature lists, configuration, and results.

## Reproducibility

From the project root, run:

```powershell
python src/run_random_feature_selection.py
```

The fixed feature pool, master seed, seed derivation rule, allocation, and RF configuration make the generated subset plan deterministic. Runtime measurements can vary by machine and concurrent system load.
