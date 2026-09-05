# Train-only CV: Top-10 vs 65-feature baseline

This additive study reuses the frozen Stratified 5-fold split and Phase 4
Random Forest configuration. Original K-Fold files under
`reports\kfold_cv` were read, not overwritten.
Random-12 is copied for completeness and is not used as a train-only
selection candidate.

## Configuration

- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Same fold-index SHA-256 as the original CV: `8a2c36f102175dacfaee0995b24b6d7e0f712e5167130bbe13711f49ed904655`
- Model: unchanged Phase 4 `RandomForestClassifier` (`random_state=42`, `n_jobs=-1`)
- Data: `data\processed\phase3\train.parquet` only
- Standard deviation: sample standard deviation across folds (`ddof=1`)
- Test parquet: never loaded or scored

## CV summary (mean ± std)

| Feature Set | # Features | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | --- | --- | --- |
| Top-10 | 10 | 0.999081 ± 0.000129 | 0.999479 ± 0.000263 | 0.999060 ± 0.000312 | 0.999270 ± 0.000103 |
| Random-12 | 12 | 0.999353 ± 0.000161 | 0.999759 ± 0.000053 | 0.999213 ± 0.000248 | 0.999486 ± 0.000128 |
| Baseline-65 | 65 | 0.999441 ± 0.000206 | 0.999809 ± 0.000127 | 0.999302 ± 0.000308 | 0.999555 ± 0.000164 |

## Top-10 vs Baseline-65

Baseline-65 has the higher mean F1 (mean-F1 difference Baseline-65 − Top-10 = +0.000286).
Top-10 is more stable on F1.

This comparison is train-only. It does not select a final model and does not
tune hyperparameters.

## Preserved original CV

Top-10 and Random-12 fold metrics were copied from
`reports\kfold_cv\cv_fold_results.csv` without rerunning
those subsets.
