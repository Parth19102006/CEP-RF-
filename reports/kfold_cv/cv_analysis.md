# Phase 4 K-Fold Cross-Validation Analysis

> **⚠️ Historical Artifact Notice:** The "Random-12" subset evaluated here
> corresponds to an **earlier random 12-feature configuration** (source experiment
> `Random_34`) and is **NOT** the final Phase 4 random candidate.
> The final Phase 4 random candidate is **Random_12_07**, whose authoritative
> validation, hyperparameter-tuning, and final test results are documented under
> `reports/hyperparameter_tuning/` and `reports/final_evaluation/`.

This analysis uses training-data Stratified 5-fold CV only. It does not select a
final model, tune hyperparameters, or evaluate the original test set.

## Configuration

- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Model: unchanged Phase 4 `RandomForestClassifier` (`random_state=42`, `n_jobs=-1`)
- Data: `data/processed/phase3/train.parquet` only
- Standard deviation: sample standard deviation across folds (`ddof=1`)

## Subsets evaluated

- **Top-10** (10 features): URG Flag Count, Bwd Packets/s, Packet Length Min, Bwd Packet Length Max, Bwd Packet Length Mean, Avg Bwd Segment Size, Bwd Packets Length Total, Init Bwd Win Bytes, Fwd Packet Length Min, Subflow Bwd Bytes
- **Random-12** (12 features, source `Random_34`): Bwd Packet Length Min, Subflow Bwd Packets, Total Backward Packets, Active Std, Fwd PSH Flags, Packet Length Max, Idle Std, Fwd Seg Size Min, Bwd IAT Min, Down/Up Ratio, Init Fwd Win Bytes, Fwd Packets/s

No other Phase 4 subset was named as a shortlist entry. Other random sets that
beat the 65-feature baseline on the original test split (for example Random_48
and Random_20) were not re-run.

## CV summary (mean ± std)

| Feature Set | # Features | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | --- | --- | --- |
| Top-10 | 10 | 0.999081 ± 0.000129 | 0.999479 ± 0.000263 | 0.999060 ± 0.000312 | 0.999270 ± 0.000103 |
| Random-12 | 12 | 0.999353 ± 0.000161 | 0.999759 ± 0.000053 | 0.999213 ± 0.000248 | 0.999486 ± 0.000128 |

## Best mean-performing subset

**Random-12** has the highest mean F1
(0.999486 ± 0.000128).

## Most stable subset

**Top-10** has the lowest F1 standard deviation
(0.000103).

## Top-10 vs Random-12

Random-12 has the higher mean F1 (absolute mean-F1 gap 0.000216). Top-10 is more stable
on F1. Random-12 is the stronger mean-performing candidate on this training CV;
Top-10 is slightly smaller and more stable.

## Consistency with completed Phase 4 test results

Completed Phase 4 scored models on the original test split. Those numbers are
not CV estimates and are not replaced here.

- 65-feature baseline test F1 = 0.983417
  (accuracy 0.972845, precision 0.999728,
  recall 0.967629).
- Random_34 / Random-12 original test F1 = 0.998019
  (accuracy 0.996708, precision 0.999685,
  recall 0.996358).
- Top-10 was ranked from baseline importances but was not previously given its
  own test-set score.

Random-12's training-CV mean F1 stays above the original 65-feature baseline
test F1, so the Phase 4 observation that this 12-feature set is a serious
compact candidate remains consistent. Exact CV means are expected to differ
from the original test scores because the partitions and sample sizes differ.
This CV run does not declare a final model.

## Leakage controls

- The original Phase 3 test parquet is never loaded or scored.
- Preprocessing is the frozen Phase 3 representation; no scaler, imputer, or selector is fitted inside CV.
- Feature subsets are frozen before CV; they are not re-selected using validation folds.
- Each Random Forest is fit only on that fold's training indices and scored only on that fold's validation indices.
- Top-10 was originally ranked from a baseline model fit on the full training split, so these CV estimates for Top-10 can be slightly optimistic relative to nested selection.
- Random-12 (Random_34) was identified from completed Phase 4 test-set results; this script does not reuse those test labels for scoring.
- Phase 3 training medians were fit on the full training split before this CV study; that preprocessing is inherited and unchanged.
