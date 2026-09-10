# Phase 4 Exploratory Random Feature-Combination RF

## Method

This experiment generated 70 reproducible random feature combinations from the 65 Phase 3 features using random_state=42. For each subset, a fresh RandomForestClassifier was trained with the same fixed configuration used by the baseline and Top-K experiments.

The Phase 3 train/test split was preserved. No new split, tuning, sampling, scaling, or PCA was applied. `Label` and `Label_binary` were excluded from every input feature set.

## Methodological Limitation

This is an exploratory random-search experiment. Because the same test set is used to compare many random combinations, the highest test score from these 70 experiments should be treated as an exploratory candidate, not as an unbiased final performance estimate.

## Reference Results

Baseline_65 F1: 0.983417. Top_10 F1: 0.986009.

## Random Results By Feature Count

| Features | Experiments | Best F1 | Mean F1 | Best Recall | Best ROC-AUC | Best Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 0.983145 | 0.908312 | 0.974725 | 0.994174 | 0.972378 |
| 8 | 10 | 0.984553 | 0.976840 | 0.969972 | 0.997720 | 0.974673 |
| 10 | 10 | 0.983873 | 0.947742 | 0.972904 | 0.997417 | 0.973570 |
| 12 | 10 | 0.986014 | 0.977472 | 0.972598 | 0.999189 | 0.977041 |
| 15 | 10 | 0.982191 | 0.966175 | 0.967307 | 0.998336 | 0.970810 |
| 20 | 10 | 0.983810 | 0.979787 | 0.968383 | 0.999485 | 0.973478 |
| 30 | 10 | 0.983865 | 0.977916 | 0.969179 | 0.998497 | 0.973570 |

## Best Random Combinations

| Criterion | Experiment | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 | Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| recall | Random_5_06 | 5 | 0.932841 | 0.946190 | 0.974725 | 0.960246 | 0.943997 | 14124 | 6440 |
| roc_auc | Random_20_04 | 20 | 0.973119 | 0.999797 | 0.967892 | 0.983586 | 0.999485 | 50 | 8181 |
| accuracy | Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |

## Best F1 Feature Set

Experiment: Random_12_07

Features: `Total Fwd Packets`, `Idle Max`, `Init Fwd Win Bytes`, `Fwd IAT Std`, `Active Mean`, `Flow IAT Mean`, `Protocol`, `Fwd IAT Max`, `Bwd Packets/s`, `Fwd Packet Length Max`, `Bwd Header Length`, `Fwd Header Length`

F1 difference from Baseline_65: +0.002598.
F1 difference from Top_10: +0.000005.
A random combination exceeded the current Top_10 F1.

## Saved Models

Only the top five random combinations by F1 were saved.

| Rank | Experiment | Features | F1 | Recall | ROC-AUC | Model |
|---:|---|---:|---:|---:|---:|---|
| 1 | Random_12_07 | 12 | 0.986014 | 0.972598 | 0.999189 | `models\random_best_01.joblib` |
| 2 | Random_12_02 | 12 | 0.985001 | 0.972182 | 0.996263 | `models\random_best_02.joblib` |
| 3 | Random_12_10 | 12 | 0.984718 | 0.970094 | 0.998909 | `models\random_best_03.joblib` |
| 4 | Random_8_09 | 8 | 0.984553 | 0.969972 | 0.997039 | `models\random_best_04.joblib` |
| 5 | Random_12_05 | 12 | 0.983892 | 0.968496 | 0.999020 | `models\random_best_05.joblib` |
