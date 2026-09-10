# Random Feature-Combination Research Analysis

## Dataset And Experiment Overview

This analysis uses the completed 70 random feature-combination RF experiments under `reports/random_feature_selection/`. No new models were trained, no new feature combinations were generated, and no baseline or Top-N artifacts were modified.

- Total random experiments: 70
- Feature subset sizes tested: 5, 8, 10, 12, 15, 20, 30
- Random combinations per size: 10
- Random seed: 42
- RF configuration: 100-tree sklearn RandomForestClassifier with gini, max_features=sqrt, bootstrap=True, random_state=42, n_jobs=-1
- Data split: existing Phase 3 train/test Parquet files
- Positive class: 1 = DDoS/Attack; negative class: 0 = Benign
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion-matrix counts, training time, prediction time

This is exploratory random search. Because all 70 random combinations were evaluated on the same held-out test set, choosing the highest test score can introduce selection bias and multiple-comparison effects. These results are exploratory evidence, not an unbiased final generalization estimate.

## Overall Performance Distribution

| metric | minimum | maximum | mean | median | std |
|---|---|---|---|---|---|
| accuracy | 0.689318 | 0.977041 | 0.941351 | 0.967730 | 0.060990 |
| precision | 0.946190 | 0.999914 | 0.997115 | 0.999548 | 0.007492 |
| recall | 0.639666 | 0.974725 | 0.932072 | 0.962364 | 0.071257 |
| f1_score | 0.774090 | 0.986014 | 0.962035 | 0.980242 | 0.043854 |
| roc_auc | 0.831663 | 0.999485 | 0.985670 | 0.994652 | 0.027335 |
| false_positives | 21.000000 | 14124.000000 | 650.628571 | 111.000000 | 1829.126735 |
| false_negatives | 6440.000000 | 91812.000000 | 17307.771429 | 9589.500000 | 18156.075183 |
| training_time_seconds | 0.861735 | 2.771237 | 1.556229 | 1.511418 | 0.401981 |
| test_prediction_time_seconds | 0.276719 | 0.398381 | 0.341564 | 0.344730 | 0.027057 |

Across all random sets, performance was broad rather than uniformly high. F1 ranged from 0.774090 to 0.986014, with mean 0.962035 and median 0.980242.

## Performance By Feature Count

| feature count | experiments | feature reduction pct | mean f1 | median f1 | std f1 | best f1 | worst f1 | mean accuracy | mean recall | mean roc auc | mean false positives | mean false negatives | mean training time seconds | mean prediction time seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.000000 | 10.000000 | 92.307692 | 0.908312 | 0.942039 | 0.084427 | 0.983145 | 0.774090 | 0.866635 | 0.850804 | 0.942553 | 2821.700000 | 38014.700000 | 1.240888 | 0.324636 |
| 8.000000 | 10.000000 | 87.692308 | 0.976840 | 0.982871 | 0.010342 | 0.984553 | 0.959578 | 0.962449 | 0.955566 | 0.993765 | 176.500000 | 11321.600000 | 1.290602 | 0.317427 |
| 10.000000 | 10.000000 | 84.615385 | 0.947742 | 0.959948 | 0.051085 | 0.983873 | 0.813874 | 0.920183 | 0.906233 | 0.983277 | 548.400000 | 23891.500000 | 1.600867 | 0.359389 |
| 12.000000 | 10.000000 | 81.538462 | 0.977472 | 0.983337 | 0.010425 | 0.986014 | 0.958203 | 0.963440 | 0.956936 | 0.994472 | 221.900000 | 10972.700000 | 1.559124 | 0.339781 |
| 15.000000 | 10.000000 | 76.923077 | 0.966175 | 0.962414 | 0.010864 | 0.982191 | 0.952025 | 0.945624 | 0.936172 | 0.992306 | 386.600000 | 16263.300000 | 1.290042 | 0.343983 |
| 20.000000 | 10.000000 | 69.230769 | 0.979787 | 0.983213 | 0.005399 | 0.983810 | 0.968768 | 0.967043 | 0.961093 | 0.997696 | 177.900000 | 9913.500000 | 1.695013 | 0.339802 |
| 30.000000 | 10.000000 | 53.846154 | 0.977916 | 0.982365 | 0.007210 | 0.983865 | 0.962831 | 0.964081 | 0.957703 | 0.995621 | 221.400000 | 10777.100000 | 2.217066 | 0.365931 |

The highest mean F1 was observed for 20-feature sets (0.979787). Increasing random feature count did not monotonically improve F1; 12-feature sets had the strongest average F1, while 30-feature sets were strong but not the best on average.

## Baseline Comparison

Baseline_65 F1: 0.983417. The approximate-equality tolerance is +/- 0.000100 F1.

- Better than baseline: 13 / 70 (18.57%)
- Approximately equal to baseline: 7 / 70 (10.00%)
- Worse than baseline: 53 / 70 (75.71%)

Successful baseline-beating sets by feature count:

| feature count | sets beating baseline |
|---|---|
| 5 | 0 |
| 8 | 2 |
| 10 | 2 |
| 12 | 4 |
| 15 | 0 |
| 20 | 4 |
| 30 | 1 |

## Top-10 Comparison

Top_10 F1: 0.986009.

- Beat Top_10: 1 / 70 (1.43%)
- Approximately matched Top_10: 1 / 70 (1.43%)
- Below Top_10: 69 / 70 (98.57%)

Random sets that beat Top_10:

- Random_12_07 (12 features): F1 0.986014, recall 0.972598, ROC-AUC 0.999189, FP 48, FN 6982. Features: `Total Fwd Packets`, `Idle Max`, `Init Fwd Win Bytes`, `Fwd IAT Std`, `Active Mean`, `Flow IAT Mean`, `Protocol`, `Fwd IAT Max`, `Bwd Packets/s`, `Fwd Packet Length Max`, `Bwd Header Length`, `Fwd Header Length`

## Top Random Experiments

The top F1 experiment is not automatically the best overall model. Recall, false negatives, false positives, ROC-AUC, feature count, and runtime move differently across rankings.

### Top 10 By F1

| experiment | feature count | accuracy | precision | recall | f1 score | roc auc | false positives | false negatives |
|---|---|---|---|---|---|---|---|---|
| Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| Random_12_02 | 12 | 0.975363 | 0.998163 | 0.972182 | 0.985001 | 0.996263 | 456 | 7088 |
| Random_12_10 | 12 | 0.974945 | 0.999790 | 0.970094 | 0.984718 | 0.998909 | 52 | 7620 |
| Random_8_09 | 8 | 0.974673 | 0.999579 | 0.969972 | 0.984553 | 0.997039 | 104 | 7651 |
| Random_12_05 | 12 | 0.973612 | 0.999785 | 0.968496 | 0.983892 | 0.999020 | 53 | 8027 |
| Random_10_03 | 10 | 0.973570 | 0.999324 | 0.968893 | 0.983873 | 0.987784 | 167 | 7926 |
| Random_30_03 | 30 | 0.973570 | 0.999870 | 0.968363 | 0.983865 | 0.993069 | 32 | 8061 |
| Random_20_10 | 20 | 0.973478 | 0.999737 | 0.968383 | 0.983810 | 0.998944 | 65 | 8056 |
| Random_8_06 | 8 | 0.973331 | 0.999469 | 0.968465 | 0.983723 | 0.994518 | 131 | 8035 |
| Random_20_04 | 20 | 0.973119 | 0.999797 | 0.967892 | 0.983586 | 0.999485 | 50 | 8181 |

### Top 10 By ROC-AUC

| experiment | feature count | accuracy | precision | recall | f1 score | roc auc | false positives | false negatives |
|---|---|---|---|---|---|---|---|---|
| Random_20_04 | 20 | 0.973119 | 0.999797 | 0.967892 | 0.983586 | 0.999485 | 50 | 8181 |
| Random_20_05 | 20 | 0.966836 | 0.999771 | 0.960365 | 0.979672 | 0.999209 | 56 | 10099 |
| Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| Random_20_06 | 20 | 0.949582 | 0.999699 | 0.939693 | 0.968768 | 0.999116 | 72 | 15366 |
| Random_12_05 | 12 | 0.973612 | 0.999785 | 0.968496 | 0.983892 | 0.999020 | 53 | 8027 |
| Random_20_10 | 20 | 0.973478 | 0.999737 | 0.968383 | 0.983810 | 0.998944 | 65 | 8056 |
| Random_12_10 | 12 | 0.974945 | 0.999790 | 0.970094 | 0.984718 | 0.998909 | 52 | 7620 |
| Random_20_09 | 20 | 0.972920 | 0.999607 | 0.967837 | 0.983465 | 0.998686 | 97 | 8195 |
| Random_20_02 | 20 | 0.958707 | 0.999690 | 0.950671 | 0.974564 | 0.998661 | 75 | 12569 |
| Random_20_08 | 20 | 0.972946 | 0.999745 | 0.967735 | 0.983479 | 0.998562 | 63 | 8221 |

### Top 10 By Recall

| experiment | feature count | accuracy | precision | recall | f1 score | roc auc | false positives | false negatives |
|---|---|---|---|---|---|---|---|---|
| Random_5_06 | 5 | 0.932841 | 0.946190 | 0.974725 | 0.960246 | 0.943997 | 14124 | 6440 |
| Random_10_07 | 10 | 0.969834 | 0.990676 | 0.972904 | 0.981710 | 0.969320 | 2333 | 6904 |
| Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| Random_12_02 | 12 | 0.975363 | 0.998163 | 0.972182 | 0.985001 | 0.996263 | 456 | 7088 |
| Random_12_10 | 12 | 0.974945 | 0.999790 | 0.970094 | 0.984718 | 0.998909 | 52 | 7620 |
| Random_8_09 | 8 | 0.974673 | 0.999579 | 0.969972 | 0.984553 | 0.997039 | 104 | 7651 |
| Random_30_09 | 30 | 0.971835 | 0.996888 | 0.969179 | 0.982838 | 0.994735 | 771 | 7853 |
| Random_10_03 | 10 | 0.973570 | 0.999324 | 0.968893 | 0.983873 | 0.987784 | 167 | 7926 |
| Random_12_05 | 12 | 0.973612 | 0.999785 | 0.968496 | 0.983892 | 0.999020 | 53 | 8027 |
| Random_8_06 | 8 | 0.973331 | 0.999469 | 0.968465 | 0.983723 | 0.994518 | 131 | 8035 |

### Top 10 By Lowest FN

| experiment | feature count | accuracy | precision | recall | f1 score | roc auc | false positives | false negatives |
|---|---|---|---|---|---|---|---|---|
| Random_5_06 | 5 | 0.932841 | 0.946190 | 0.974725 | 0.960246 | 0.943997 | 14124 | 6440 |
| Random_10_07 | 10 | 0.969834 | 0.990676 | 0.972904 | 0.981710 | 0.969320 | 2333 | 6904 |
| Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| Random_12_02 | 12 | 0.975363 | 0.998163 | 0.972182 | 0.985001 | 0.996263 | 456 | 7088 |
| Random_12_10 | 12 | 0.974945 | 0.999790 | 0.970094 | 0.984718 | 0.998909 | 52 | 7620 |
| Random_8_09 | 8 | 0.974673 | 0.999579 | 0.969972 | 0.984553 | 0.997039 | 104 | 7651 |
| Random_30_09 | 30 | 0.971835 | 0.996888 | 0.969179 | 0.982838 | 0.994735 | 771 | 7853 |
| Random_10_03 | 10 | 0.973570 | 0.999324 | 0.968893 | 0.983873 | 0.987784 | 167 | 7926 |
| Random_12_05 | 12 | 0.973612 | 0.999785 | 0.968496 | 0.983892 | 0.999020 | 53 | 8027 |
| Random_8_06 | 8 | 0.973331 | 0.999469 | 0.968465 | 0.983723 | 0.994518 | 131 | 8035 |

### Top 10 By Lowest FP

| experiment | feature count | accuracy | precision | recall | f1 score | roc auc | false positives | false negatives |
|---|---|---|---|---|---|---|---|---|
| Random_15_06 | 15 | 0.968919 | 0.999914 | 0.962731 | 0.980971 | 0.996628 | 21 | 9496 |
| Random_15_02 | 15 | 0.923805 | 0.999892 | 0.908531 | 0.952025 | 0.987887 | 25 | 23306 |
| Random_30_03 | 30 | 0.973570 | 0.999870 | 0.968363 | 0.983865 | 0.993069 | 32 | 8061 |
| Random_12_04 | 12 | 0.972815 | 0.999858 | 0.967468 | 0.983396 | 0.988768 | 35 | 8289 |
| Random_15_09 | 15 | 0.940875 | 0.999840 | 0.929096 | 0.963171 | 0.996284 | 38 | 18066 |
| Random_12_08 | 12 | 0.949860 | 0.999825 | 0.939909 | 0.968941 | 0.997421 | 42 | 15311 |
| Random_8_08 | 8 | 0.941829 | 0.999823 | 0.930258 | 0.963787 | 0.997720 | 42 | 17770 |
| Random_12_07 | 12 | 0.977041 | 0.999806 | 0.972598 | 0.986014 | 0.999189 | 48 | 6982 |
| Random_20_04 | 20 | 0.973119 | 0.999797 | 0.967892 | 0.983586 | 0.999485 | 50 | 8181 |
| Random_15_04 | 15 | 0.943557 | 0.999790 | 0.932366 | 0.964901 | 0.997261 | 50 | 17233 |

## Feature Frequency Analysis

Feature appearances are driven by reproducible random sampling, but the counts still help show which features were more exposed to evaluation in this random search.

Most frequent features overall:

| feature name | total appearances | appearance percentage |
|---|---|---|
| Subflow Fwd Bytes | 24 | 34.285714 |
| Init Bwd Win Bytes | 22 | 31.428571 |
| Fwd Packet Length Min | 21 | 30.000000 |
| Bwd Header Length | 20 | 28.571429 |
| Bwd IAT Max | 20 | 28.571429 |
| Bwd Packet Length Max | 20 | 28.571429 |
| Bwd Packets Length Total | 20 | 28.571429 |
| Bwd Packets/s | 20 | 28.571429 |
| Fwd Packet Length Max | 20 | 28.571429 |
| Active Max | 19 | 27.142857 |
| Flow IAT Std | 19 | 27.142857 |
| Fwd Packet Length Mean | 19 | 27.142857 |
| Protocol | 19 | 27.142857 |
| Subflow Fwd Packets | 19 | 27.142857 |
| Total Fwd Packets | 19 | 27.142857 |

Least frequent features overall:

| feature name | total appearances | appearance percentage |
|---|---|---|
| Bwd Packet Length Std | 10 | 14.285714 |
| Fwd Seg Size Min | 10 | 14.285714 |
| Total Backward Packets | 10 | 14.285714 |
| Active Min | 11 | 15.714286 |
| Bwd IAT Total | 11 | 15.714286 |
| Fwd IAT Min | 11 | 15.714286 |
| Fwd IAT Std | 11 | 15.714286 |
| Fwd PSH Flags | 11 | 15.714286 |
| Fwd Packets Length Total | 11 | 15.714286 |
| Subflow Bwd Packets | 11 | 15.714286 |
| Bwd Packet Length Mean | 12 | 17.142857 |
| Fwd Header Length | 12 | 17.142857 |
| Fwd IAT Total | 12 | 17.142857 |
| Packet Length Max | 13 | 18.571429 |
| Packet Length Min | 13 | 18.571429 |

## Successful-Set Feature Analysis

Success is analyzed in two groups: sets that beat Baseline_65 and sets that beat Top_10. Enrichment is the successful-set frequency divided by overall frequency. This is exploratory association analysis, not a statistical significance claim.

Features most common among sets beating Baseline_65:

| feature name | overall sets with feature | successful sets with feature | successful appearance percentage | enrichment ratio |
|---|---|---|---|---|
| Init Fwd Win Bytes | 17 | 7 | 53.846154 | 2.217195 |
| Protocol | 19 | 7 | 53.846154 | 1.983806 |
| Fwd Packet Length Max | 20 | 6 | 46.153846 | 1.615385 |
| Flow Duration | 14 | 5 | 38.461538 | 1.923077 |
| Flow IAT Min | 17 | 5 | 38.461538 | 1.583710 |
| Fwd Packet Length Mean | 19 | 5 | 38.461538 | 1.417004 |
| Subflow Fwd Packets | 19 | 5 | 38.461538 | 1.417004 |
| Bwd Packet Length Max | 20 | 5 | 38.461538 | 1.346154 |
| Bwd Packets/s | 20 | 5 | 38.461538 | 1.346154 |
| Fwd IAT Total | 12 | 4 | 30.769231 | 1.794872 |
| Bwd IAT Std | 13 | 4 | 30.769231 | 1.656805 |
| Avg Packet Size | 14 | 4 | 30.769231 | 1.538462 |
| Flow IAT Max | 14 | 4 | 30.769231 | 1.538462 |
| Idle Max | 14 | 4 | 30.769231 | 1.538462 |
| URG Flag Count | 14 | 4 | 30.769231 | 1.538462 |
| Avg Fwd Segment Size | 16 | 4 | 30.769231 | 1.346154 |
| Bwd IAT Min | 16 | 4 | 30.769231 | 1.346154 |
| Flow IAT Mean | 17 | 4 | 30.769231 | 1.266968 |
| Idle Std | 17 | 4 | 30.769231 | 1.266968 |
| Active Max | 19 | 4 | 30.769231 | 1.133603 |

Features in sets beating Top_10:

| feature name | overall sets with feature | successful sets with feature | successful appearance percentage | enrichment ratio |
|---|---|---|---|---|
| Fwd IAT Std | 11 | 1 | 100.000000 | 6.363636 |
| Fwd Header Length | 12 | 1 | 100.000000 | 5.833333 |
| Active Mean | 13 | 1 | 100.000000 | 5.384615 |
| Idle Max | 14 | 1 | 100.000000 | 5.000000 |
| Flow IAT Mean | 17 | 1 | 100.000000 | 4.117647 |
| Init Fwd Win Bytes | 17 | 1 | 100.000000 | 4.117647 |
| Fwd IAT Max | 18 | 1 | 100.000000 | 3.888889 |
| Protocol | 19 | 1 | 100.000000 | 3.684211 |
| Total Fwd Packets | 19 | 1 | 100.000000 | 3.684211 |
| Bwd Header Length | 20 | 1 | 100.000000 | 3.500000 |
| Bwd Packets/s | 20 | 1 | 100.000000 | 3.500000 |
| Fwd Packet Length Max | 20 | 1 | 100.000000 | 3.500000 |
| ACK Flag Count | 17 | 0 | 0.000000 | 0.000000 |
| Active Max | 19 | 0 | 0.000000 | 0.000000 |
| Active Min | 11 | 0 | 0.000000 | 0.000000 |
| Active Std | 13 | 0 | 0.000000 | 0.000000 |
| Avg Bwd Segment Size | 13 | 0 | 0.000000 | 0.000000 |
| Avg Fwd Segment Size | 16 | 0 | 0.000000 | 0.000000 |
| Avg Packet Size | 14 | 0 | 0.000000 | 0.000000 |
| Bwd IAT Max | 20 | 0 | 0.000000 | 0.000000 |

## Common Features And Feature Outcomes

The table below compares F1 in sets containing a feature versus sets not containing it. This is descriptive and should not be read as causal.

| feature name | sets with feature | sets without feature | mean f1 with feature | mean f1 without feature | mean f1 difference with minus without |
|---|---|---|---|---|---|
| Avg Fwd Segment Size | 16 | 54 | 0.980531 | 0.956554 | 0.023976 |
| Fwd Packet Length Mean | 19 | 51 | 0.979052 | 0.955695 | 0.023357 |
| Subflow Fwd Packets | 19 | 51 | 0.978048 | 0.956069 | 0.021979 |
| Protocol | 19 | 51 | 0.977904 | 0.956123 | 0.021781 |
| Init Fwd Win Bytes | 17 | 53 | 0.977789 | 0.956982 | 0.020807 |
| Idle Max | 14 | 56 | 0.978462 | 0.957928 | 0.020534 |
| Fwd Packet Length Max | 20 | 50 | 0.976492 | 0.956252 | 0.020240 |
| Flow IAT Mean | 17 | 53 | 0.977301 | 0.957138 | 0.020163 |
| Bwd IAT Min | 16 | 54 | 0.977348 | 0.957497 | 0.019851 |
| Fwd IAT Mean | 13 | 57 | 0.978003 | 0.958393 | 0.019610 |
| URG Flag Count | 14 | 56 | 0.977683 | 0.958123 | 0.019560 |
| Subflow Fwd Bytes | 24 | 46 | 0.974320 | 0.955625 | 0.018695 |
| Flow IAT Min | 17 | 53 | 0.976122 | 0.957516 | 0.018605 |
| Flow Packets/s | 14 | 56 | 0.976886 | 0.958322 | 0.018563 |
| Packet Length Mean | 14 | 56 | 0.976775 | 0.958350 | 0.018425 |
| Packet Length Max | 13 | 57 | 0.976711 | 0.958688 | 0.018024 |
| Fwd Packets/s | 15 | 55 | 0.975917 | 0.958249 | 0.017668 |
| Bwd Packet Length Mean | 12 | 58 | 0.976552 | 0.959031 | 0.017521 |
| Idle Std | 17 | 53 | 0.975283 | 0.957785 | 0.017498 |
| Avg Packet Size | 14 | 56 | 0.975663 | 0.958628 | 0.017035 |

## Feature Count Versus Performance

Correlations are descriptive only. They summarize observed association across the 70 random sets and do not establish causation.

| x | y | pearson correlation | spearman correlation |
|---|---|---|---|
| feature_count | f1_score | 0.329154 | 0.209988 |
| feature_count | recall | 0.317726 | 0.127619 |
| feature_count | roc_auc | 0.382711 | 0.521082 |
| feature_count | false_positives | -0.255071 | -0.363790 |
| feature_count | false_negatives | -0.317726 | -0.127619 |
| feature_count | training_time_seconds | 0.703218 | 0.594260 |
| feature_count | test_prediction_time_seconds | 0.414948 | 0.333012 |

## Stability Analysis

The lowest F1 standard deviation was observed for 20-feature sets (std 0.005399). Five-feature sets were highly variable (std 0.084427), showing that very small random subsets can be excellent or poor depending on which features they contain.

The 12-feature group produced the best F1 and the best mean F1, but not every 12-feature set was strong; its worst F1 was 0.958203. This suggests a small number of combinations are unusually strong rather than all small subsets being reliably strong.

## Computational Efficiency

| feature count | mean training time reduction vs baseline pct | mean prediction time reduction vs baseline pct |
|---|---|---|
| 5.000000 | 72.579960 | 29.129762 |
| 8.000000 | 71.481424 | 30.703635 |
| 10.000000 | 64.625477 | 21.542930 |
| 12.000000 | 65.547879 | 25.823562 |
| 15.000000 | 71.493796 | 24.906270 |
| 20.000000 | 62.545116 | 25.819052 |
| 30.000000 | 51.009263 | 20.114858 |

Random subsets generally reduced runtime relative to the 65-feature baseline. The strongest random F1 result used 12 features, reducing the feature set by 81.54%, but its individual training and prediction times were slower than Top_10 in the recorded run.

## Feature Overlap Between Best Sets

High-performing random groups share some features with each other and with the importance-based Top_10 set, but the overlap is incomplete. This supports looking at hybrid strategies rather than simply replacing importance-based selection with one random set.

Most repeated features across high-performing ranking groups:

| feature name | high performing group count | groups |
|---|---|---|
| Bwd Packet Length Max | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| Bwd Packet Length Mean | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| Bwd Packets Length Total | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| Bwd Packets/s | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| Init Bwd Win Bytes | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| Subflow Bwd Bytes | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| URG Flag Count | 6 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp; top_10_importance_selected |
| ACK Flag Count | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Active Max | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Active Mean | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Active Min | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Active Std | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Avg Packet Size | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Bwd Header Length | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Bwd IAT Max | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Bwd IAT Mean | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Bwd IAT Min | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Bwd IAT Std | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| CWE Flag Count | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |
| Down/Up Ratio | 5 | top_10_by_f1; top_10_by_roc_auc; top_10_by_recall; top_10_by_lowest_fn; top_10_by_lowest_fp |

Pairwise group overlap summary:

| group a | group b | overlap count | jaccard overlap |
|---|---|---|---|
| top_10_by_f1 | top_10_by_f1 | 61 | 1.000000 |
| top_10_by_f1 | top_10_by_roc_auc | 57 | 0.890625 |
| top_10_by_f1 | top_10_by_recall | 51 | 0.796875 |
| top_10_by_f1 | top_10_importance_selected | 10 | 0.163934 |
| top_10_by_roc_auc | top_10_by_f1 | 57 | 0.890625 |
| top_10_by_roc_auc | top_10_by_roc_auc | 60 | 1.000000 |
| top_10_by_roc_auc | top_10_by_recall | 49 | 0.753846 |
| top_10_by_roc_auc | top_10_importance_selected | 10 | 0.166667 |
| top_10_by_recall | top_10_by_f1 | 51 | 0.796875 |
| top_10_by_recall | top_10_by_roc_auc | 49 | 0.753846 |
| top_10_by_recall | top_10_by_recall | 54 | 1.000000 |
| top_10_by_recall | top_10_importance_selected | 7 | 0.122807 |
| top_10_importance_selected | top_10_by_f1 | 10 | 0.163934 |
| top_10_importance_selected | top_10_by_roc_auc | 10 | 0.166667 |
| top_10_importance_selected | top_10_by_recall | 7 | 0.122807 |
| top_10_importance_selected | top_10_importance_selected | 10 | 1.000000 |

## Visualizations

- `reports\random_feature_selection\figures\f1_vs_feature_count.png`
- `reports\random_feature_selection\figures\roc_auc_vs_feature_count.png`
- `reports\random_feature_selection\figures\f1_distribution_by_feature_count.png`
- `reports\random_feature_selection\figures\feature_frequency_top20.png`
- `reports\random_feature_selection\figures\successful_feature_frequency_top20.png`
- `reports\random_feature_selection\figures\training_time_vs_feature_count.png`

## Research Interpretation

1. Random sets beating the 65-feature baseline: 13.
2. Random sets beating Top_10: 1.
3. Smallest feature count among baseline-beating random sets: 8.
4. Successful small subsets were possible but not common: 5-feature sets included one baseline-beating set, while several other 5-feature sets performed poorly.
5. Low feature counts were less stable in this sample; the 5-feature group had the largest F1 spread, while 8-feature sets had the lowest F1 variance.
6. The most frequent feature overall was `Subflow Fwd Bytes` with 24 appearances.
7. The most frequent feature among baseline-beating sets was `Init Fwd Win Bytes`, appearing in 7 of 13 baseline-beating sets.
8. Random selection was competitive in some cases, but the distribution shows high variability; importance-based Top_10 remains a stronger controlled result than a single exploratory random win.
9. The 20-feature random group appears to provide the best observed performance/complexity trade-off in this random search.
10. The next feature-selection experiment should use these observations to design a controlled hybrid selection method and then validate it without treating this same test-set search result as final.

## Key Findings

- 13 of 70 random combinations beat the 65-feature baseline by F1.
- 1 of 70 random combinations beat the Top_10 F1, and the margin was very small.
- The best F1 was 0.986014 from Random_12_07 with 12 features.
- The smallest baseline-beating subset used 8 features.
- The 12-feature group had the highest mean F1 (0.979787).
- The 20-feature group had the lowest F1 standard deviation (0.005399).
- Very small random subsets can work, but they are not reliably strong; feature composition matters more than feature count alone.
- The same-test-set search limitation means Random_12_07 should be treated as an exploratory candidate, not a final model.

## Implications For Feature Selection

The evidence supports continuing with importance-based selection as a stable baseline, while exploring a hybrid method that combines high-importance features, recurring successful-set features, and correlation-aware pruning. Random selection revealed that non-obvious combinations can be competitive, but the variability argues against adopting random search itself as the final selection strategy. A next controlled experiment could predefine a small number of hybrid candidate sets using training-only or already-fixed artifacts, then evaluate them once under the same RF configuration.
