# Phase 4.3 — Baseline Error Analysis

## Artifact source and verification

The baseline artifacts were verified in the `origin/P4` baseline revision (`a427a1f`):

- `reports/baseline/confusion_matrix.csv`
- `reports/baseline/classification_report.csv`
- `reports/baseline/metrics.json`
- `reports/baseline/baseline_metadata.json`

The current checkout is `main` at the Phase 3 revision, so these baseline inputs were read from `origin/P4`; they were not changed.

Target mapping: `0 = Benign`; `1 = DDoS/Attack`.

## Test samples and confusion matrix

- Total test samples: 306,201
- Actual benign samples: 51,404
- Actual DDoS/Attack samples: 254,797

| Actual class | Predicted Benign | Predicted DDoS/Attack |
| --- | ---: | ---: |
| Benign | 51,337 (TN) | 67 (FP) |
| DDoS/Attack | 8,248 (FN) | 246,549 (TP) |

## Metrics calculated from the matrix

- False positive rate: `67 / (51,337 + 67) = 0.0013034005` (0.1303%)
- False negative rate: `8,248 / (8,248 + 246,549) = 0.0323708678` (3.2371%)
- Accuracy: 97.2845%
- Precision (DDoS/Attack): 99.9728%
- Recall (DDoS/Attack): 96.7629%
- F1-score (DDoS/Attack): 98.3417%

## Interpretation

67 benign samples were incorrectly classified as attacks, while 8,248 attack samples were incorrectly classified as benign. False negatives are the larger error type. The current baseline therefore produces very few false alerts on benign traffic, but it misses more actual attack traffic than it falsely flags benign traffic.
