# Phase 4 Selected-Feature Random Forest Comparison

## Method

Feature selection used the saved baseline RandomForestClassifier feature_importances_ ranking from reports/baseline/feature_importance.csv. These values are impurity-based feature importances, also known as mean decrease in impurity. They are not information gain.

The Phase 3 train/test split was preserved. Candidate subsets were constructed before evaluating selected-feature models on the test set. No test labels or test-set performance metrics were used for feature selection. The Phase 2 high-correlation pairs were inspected as context only; no correlation filtering was applied.

## Baseline

Baseline_65 used 65 features and achieved accuracy 0.972845, precision 0.999728, recall 0.967629, F1 0.983417, and ROC-AUC 0.994541. It produced 67 false positives and 8248 false negatives.

## Candidate Results

| Model | Features | Reduction % | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline_65 | 65 | 0.00 | 0.972845 | 0.999728 | 0.967629 | 0.983417 | 0.994541 | 67 | 8248 | 4.525479 | 0.458071 |
| Top_50 | 50 | 23.08 | 0.972077 | 0.998676 | 0.967727 | 0.982958 | 0.992815 | 327 | 8223 | 2.873430 | 0.295719 |
| Top_40 | 40 | 38.46 | 0.972574 | 0.999744 | 0.967288 | 0.983248 | 0.995860 | 63 | 8335 | 2.408430 | 0.289813 |
| Top_30 | 30 | 53.85 | 0.972841 | 0.999777 | 0.967578 | 0.983414 | 0.994821 | 55 | 8261 | 2.022625 | 0.264274 |
| Top_20 | 20 | 69.23 | 0.971068 | 0.998787 | 0.966405 | 0.982329 | 0.991954 | 299 | 8560 | 1.339528 | 0.255510 |
| Top_10 | 10 | 84.62 | 0.977031 | 0.999734 | 0.972657 | 0.986009 | 0.995142 | 66 | 6967 | 0.882415 | 0.249609 |

## Changes Relative To Baseline

| Model | Accuracy delta | Recall delta | F1 delta | ROC-AUC delta | FP delta | FN delta | Train time delta % | Predict time delta % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline_65 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0 | +0 | +0.00 | +0.00 |
| Top_50 | -0.000767 | +0.000098 | -0.000459 | -0.001726 | +260 | -25 | -36.51 | -35.44 |
| Top_40 | -0.000271 | -0.000341 | -0.000169 | +0.001320 | -4 | +87 | -46.78 | -36.73 |
| Top_30 | -0.000003 | -0.000051 | -0.000003 | +0.000280 | -12 | +13 | -55.31 | -42.31 |
| Top_20 | -0.001777 | -0.001225 | -0.001088 | -0.002587 | +232 | +312 | -70.40 | -44.22 |
| Top_10 | +0.004187 | +0.005028 | +0.002592 | +0.000601 | -1 | -1281 | -80.50 | -45.51 |

## Trade-Off Assessment

The selected trade-off candidate is Top_10. It removes 84.62% of features while achieving F1 0.986009, recall 0.972657, ROC-AUC 0.995142, 66 false positives, and 6967 false negatives.

This recommendation considers F1, recall, false negatives, ROC-AUC, feature reduction, and runtime. It is not based on accuracy alone.
