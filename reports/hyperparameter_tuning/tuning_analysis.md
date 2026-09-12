# Phase 4 Hyperparameter Tuning Analysis

Training-only `RandomizedSearchCV` for candidate feature sets: **Top-10** and **Random_12_07** (along with Baseline-65 for reference).
The test set was strictly kept untouched during this search.

## Search Space & Methodology

- **Cross-Validation Splitter:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Random State / Seed:** `42`
- **Evaluated Iterations (`n_iter`):** `20` configurations per candidate subset
- **Total CV Fits per Subset:** `100` fits
- **Primary Refit Metric:** Binary classification F1-Score (`refit='f1'`)
- **Monitored Metrics:** F1, Precision, Recall, Accuracy
- **Standard Deviation Reporting:** Sample standard deviation (`ddof=1`) across 5 validation folds

### Hyperparameter Search Space

```json
{
  "n_estimators": [
    50,
    100,
    200,
    300
  ],
  "max_depth": [
    null,
    10,
    20,
    30,
    40
  ],
  "min_samples_split": [
    2,
    5,
    10
  ],
  "min_samples_leaf": [
    1,
    2,
    4
  ],
  "max_features": [
    "sqrt",
    "log2",
    null
  ],
  "criterion": [
    "gini",
    "entropy"
  ],
  "bootstrap": [
    true,
    false
  ]
}
```

## Best Cross-Validation Performance (Mean ± Sample Std)

| Candidate Subset | Feature Count | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | 0.999097 ± 0.000108 | 0.999467 ± 0.000264 | 0.999098 ± 0.000334 | **0.999282 ± 0.000086** |
| **Random_12_07** | 12 | 0.999409 ± 0.000264 | 0.999797 ± 0.000053 | 0.999263 ± 0.000383 | **0.999530 ± 0.000210** |
| **Baseline-65** | 65 | 0.999505 ± 0.000154 | 0.999835 ± 0.000096 | 0.999378 ± 0.000252 | **0.999606 ± 0.000122** |

## Best Hyperparameter Configurations

### Top-10

```json
{
  "n_estimators": 300,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "max_depth": 40,
  "criterion": "gini",
  "bootstrap": false
}
```

Top 5 configurations ranked by mean validation F1:

- **Rank 1:** F1 = `0.999282 ± 0.000086`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 40, "criterion": "gini", "bootstrap": false}`
- **Rank 2:** F1 = `0.999276 ± 0.000104`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 30, "criterion": "gini", "bootstrap": true}`
- **Rank 3:** F1 = `0.999270 ± 0.000133`; Params: `{"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "log2", "max_depth": null, "criterion": "gini", "bootstrap": false}`
- **Rank 4:** F1 = `0.999270 ± 0.000133`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "gini", "bootstrap": false}`
- **Rank 5:** F1 = `0.999270 ± 0.000112`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "entropy", "bootstrap": false}`

### Random_12_07

```json
{
  "n_estimators": 300,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "max_depth": 40,
  "criterion": "gini",
  "bootstrap": false
}
```

Top 5 configurations ranked by mean validation F1:

- **Rank 1:** F1 = `0.999530 ± 0.000210`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 40, "criterion": "gini", "bootstrap": false}`
- **Rank 2:** F1 = `0.999524 ± 0.000201`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 30, "criterion": "gini", "bootstrap": true}`
- **Rank 3:** F1 = `0.999511 ± 0.000230`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "entropy", "bootstrap": false}`
- **Rank 4:** F1 = `0.999511 ± 0.000223`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "gini", "bootstrap": false}`
- **Rank 5:** F1 = `0.999511 ± 0.000245`; Params: `{"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "log2", "max_depth": null, "criterion": "gini", "bootstrap": false}`

### Baseline-65

```json
{
  "n_estimators": 300,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "max_depth": 40,
  "criterion": "gini",
  "bootstrap": false
}
```

Top 5 configurations ranked by mean validation F1:

- **Rank 1:** F1 = `0.999606 ± 0.000122`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 40, "criterion": "gini", "bootstrap": false}`
- **Rank 2:** F1 = `0.999587 ± 0.000167`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "entropy", "bootstrap": false}`
- **Rank 3:** F1 = `0.999562 ± 0.000195`; Params: `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": null, "criterion": "gini", "bootstrap": false}`
- **Rank 4:** F1 = `0.999562 ± 0.000201`; Params: `{"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 30, "criterion": "gini", "bootstrap": true}`
- **Rank 5:** F1 = `0.999549 ± 0.000214`; Params: `{"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "log2", "max_depth": null, "criterion": "gini", "bootstrap": false}`
