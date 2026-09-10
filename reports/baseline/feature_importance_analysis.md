# Phase 4.4 — Random Forest Feature-Importance Analysis

## Scope and source

This analysis reuses the feature importances produced by the unchanged 100-tree, 65-feature baseline Random Forest. The source artifact is `reports/baseline/feature_importance.csv` from the baseline revision `origin/P4` (`a427a1f`). The baseline training script writes these values from `model.feature_importances_`; no model was retrained and no feature was selected or removed.

The baseline metadata declares 65 features, and the source ranking contains exactly 65 rows. Their importances sum to 1.0000000000 (within floating-point precision).

## Top 20 features

| Rank | Feature | Importance | Cumulative importance |
| ---: | --- | ---: | ---: |
| 1 | URG Flag Count | 0.099318 | 9.9318% |
| 2 | Bwd Packets/s | 0.095233 | 19.4550% |
| 3 | Packet Length Min | 0.057691 | 25.2241% |
| 4 | Bwd Packet Length Max | 0.055038 | 30.7279% |
| 5 | Bwd Packet Length Mean | 0.049478 | 35.6757% |
| 6 | Avg Bwd Segment Size | 0.047947 | 40.4704% |
| 7 | Bwd Packets Length Total | 0.041330 | 44.6034% |
| 8 | Init Bwd Win Bytes | 0.039915 | 48.5949% |
| 9 | Fwd Packet Length Min | 0.039215 | 52.5163% |
| 10 | Subflow Bwd Bytes | 0.031946 | 55.7109% |
| 11 | Down/Up Ratio | 0.029878 | 58.6987% |
| 12 | Subflow Fwd Bytes | 0.027627 | 61.4614% |
| 13 | Fwd Act Data Packets | 0.027001 | 64.1615% |
| 14 | Fwd Packet Length Mean | 0.025211 | 66.6826% |
| 15 | Fwd PSH Flags | 0.024751 | 69.1577% |
| 16 | Init Fwd Win Bytes | 0.022463 | 71.4041% |
| 17 | Avg Packet Size | 0.020887 | 73.4928% |
| 18 | RST Flag Count | 0.018504 | 75.3432% |
| 19 | Flow Bytes/s | 0.017649 | 77.1081% |
| 20 | Bwd Packet Length Min | 0.015955 | 78.7036% |

## Bottom 10 features

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 56 | Idle Mean | 0.000363072 |
| 57 | Bwd IAT Total | 0.000331235 |
| 58 | SYN Flag Count | 0.000290764 |
| 59 | Active Mean | 0.000284466 |
| 60 | Bwd IAT Std | 0.000278994 |
| 61 | Active Max | 0.000197332 |
| 62 | Idle Max | 0.000158614 |
| 63 | Active Std | 0.000052508 |
| 64 | Bwd IAT Min | 0.000047995 |
| 65 | Idle Min | 0.000037034 |

## Cumulative importance

| Top ranked features included | Cumulative importance |
| ---: | ---: |
| 1 | 9.9318% |
| 5 | 35.6757% |
| 10 | 55.7109% |
| 20 | 78.7036% |
| 30 | 90.6668% |
| 40 | 97.0790% |
| 50 | 99.4796% |
| 60 | 99.9507% |
| 65 | 100.0000% |

## Interpretation

Importance is concentrated near the top of the baseline ranking: the first 10 features account for 55.7109% of total importance and the first 20 account for 78.7036%. The lowest-ranked 10 features together account for 0.2042%. These values describe how the existing baseline Random Forest distributes its built-in feature importance across all 65 retained Phase 3 features; this report does not select, remove, or otherwise alter any feature.
