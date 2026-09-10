# Phase 4 (P4) Final Comprehensive Report: DDoS Detection with Random Forest

## 1. Executive Summary & Overview

Phase 4 completes the development, optimization, and final untouched test evaluation of lightweight Machine Learning (Random Forest) models for real-time DDoS attack detection on network flow data (CIC-DDoS2019 dataset).

The primary objective of this phase was to identify and optimize ultra-lightweight feature subsets capable of achieving high detection accuracy and low false alarm rates while maintaining millisecond inference latency for edge deployment.

### Key Methodology Pipeline
$$\text{Phase 3 Preprocessing} \longrightarrow \text{Feature Importance / Ranking} \longrightarrow \text{Top-N \& Random Exploration} \longrightarrow \text{K-Fold CV \& Hyperparameter Tuning (Train Only)} \longrightarrow \text{Training-Only Comparison} \longrightarrow \text{Final Model Training} \longrightarrow \text{One-Time Untouched Test Evaluation}$$

---

## 2. Dataset Partition & Preprocessing

- **Dataset Source:** CIC-DDoS2019 flow telemetry.
- **Preprocessing:** Identifiers, timestamps, and IP addresses dropped (zero leakage); infinite and missing values imputed/cleaned; constant and zero-variance features eliminated (yielding 65 clean features in Phase 3).
- **Partitioning:**
  - **Training Set (`train.parquet`):** $125,170$ samples
    - Attack / DDoS (`Label_binary = 1`): $78,743$ ($62.91\%$)
    - Benign (`Label_binary = 0`): $46,427$ ($37.09\%$)
  - **Test Set (`test.parquet`):** $306,201$ samples
    - Attack / DDoS (`Label_binary = 1`): $254,797$ ($83.21\%$)
    - Benign (`Label_binary = 0`): $51,404$ ($16.79\%$)
- **Data Integrity Protocol:** The test dataset remained strictly **untouched** during all feature selection, cross-validation, hyperparameter tuning, and candidate selection stages.

---

## 3. Final Candidate Feature Sets

Through feature importance ranking and extensive random feature exploration ($70$ randomized subset trials), two feature subsets achieved top performance:

### Candidate A: Top-10 (10 Features)
Derived from the top 10 ranked features using Random Forest Gini feature importance on the training dataset:
1. `URG Flag Count` (Importance: 0.099318, Rank: 1)
2. `Bwd Packets/s` (Importance: 0.095233, Rank: 2)
3. `Packet Length Min` (Importance: 0.057691, Rank: 3)
4. `Bwd Packet Length Max` (Importance: 0.055038, Rank: 4)
5. `Bwd Packet Length Mean` (Importance: 0.049478, Rank: 5)
6. `Avg Bwd Segment Size` (Importance: 0.047947, Rank: 6)
7. `Bwd Packets Length Total` (Importance: 0.041330, Rank: 7)
8. `Init Bwd Win Bytes` (Importance: 0.039915, Rank: 8)
9. `Fwd Packet Length Min` (Importance: 0.039215, Rank: 9)
10. `Subflow Bwd Bytes` (Importance: 0.031946, Rank: 10)

### Candidate B: Random_12_07 (12 Features)
The top-performing feature subset (#1 rank by F1-score across 70 random feature experiments):
1. `Total Fwd Packets`
2. `Idle Max`
3. `Init Fwd Win Bytes`
4. `Fwd IAT Std`
5. `Active Mean`
6. `Flow IAT Mean`
7. `Protocol`
8. `Fwd IAT Max`
9. `Bwd Packets/s`
10. `Fwd Packet Length Max`
11. `Bwd Header Length`
12. `Fwd Header Length`

---

## 4. Hyperparameter Tuning (Training Data Only)

Hyperparameter tuning was executed independently on `Top-10`, `Random_12_07`, and `Baseline-65` using `RandomizedSearchCV` on the training dataset only (`train.parquet`).

### Cross-Validation & Search Space
- **Splitter:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Iterations (`n_iter`):** 20 configurations per candidate ($100$ total fits per candidate)
- **Primary Optimization Metric:** Binary F1-Score (`refit='f1'`)
- **Estimator:** `RandomForestClassifier(random_state=42)`

```json
{
  "n_estimators": [50, 100, 200, 300],
  "max_depth": [null, 10, 20, 30, 40],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4],
  "max_features": ["sqrt", "log2", null],
  "criterion": ["gini", "entropy"],
  "bootstrap": [true, false]
}
```

### Best Hyperparameter Configurations Found

| Parameter | Top-10 Optimal | Random_12_07 Optimal | Baseline-65 Optimal |
| :--- | :---: | :---: | :---: |
| `n_estimators` | **300** | **300** | **300** |
| `max_depth` | **40** | **40** | **40** |
| `min_samples_split` | **2** | **2** | **2** |
| `min_samples_leaf` | **1** | **1** | **1** |
| `max_features` | **"sqrt"** | **"sqrt"** | **"sqrt"** |
| `criterion` | **"gini"** | **"gini"** | **"gini"** |
| `bootstrap` | **false** | **false** | **false** |

### Cross-Validation Performance (Training-Only Validation Mean ± Sample Std)

| Candidate | Features | CV F1-Score | CV Precision | CV Recall | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | **0.999282 ± 0.000086** | 0.999467 ± 0.000264 | 0.999098 ± 0.000334 | 0.999097 ± 0.000108 |
| **Random_12_07** | 12 | **0.999530 ± 0.000210** | 0.999797 ± 0.000053 | 0.999263 ± 0.000383 | 0.999409 ± 0.000264 |
| **Baseline-65** | 65 | **0.999606 ± 0.000122** | 0.999835 ± 0.000096 | 0.999378 ± 0.000252 | 0.999505 ± 0.000154 |

---

## 5. Candidate Comparison & Selection Rationale

- **Comparison Basis:** Validation metrics from training-only 5-fold cross-validation.
- **Observed Difference:**
  - $\Delta \text{F1} = |0.999530 - 0.999282| = 0.000248$ ($0.0248\%$)
  - Both candidates exhibit near-perfect detection on training folds with overlapping confidence intervals.
- **Decision:** **TIE (Both Candidates Retained)**
- **Selection Rationale:**
  - `Top-10` offers maximum compression ($84.6\%$ feature reduction) with $10$ features.
  - `Random_12_07` offers high robustness ($81.5\%$ feature reduction) with $12$ features.
  - Because the performance difference in training validation is negligible ($\Delta < 0.001$), both models were retained and trained for final evaluation without manufacturing an arbitrary winner.

---

## 6. Final Model Training

Both final models were trained on the entire training set ($125,170$ samples) using their optimal hyperparameters:

1. **`models/rf_final_top_10.joblib`:**
   - Features: $10$ Top-10 features
   - Parameters: `n_estimators=300, max_depth=40, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', criterion='gini', bootstrap=False, random_state=42`
   - Training Fit Time: $2.73$ seconds
2. **`models/rf_final_random_12_07.joblib`:**
   - Features: $12$ Random_12_07 features
   - Parameters: `n_estimators=300, max_depth=40, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', criterion='gini', bootstrap=False, random_state=42`
   - Training Fit Time: $6.49$ seconds

---

## 7. FINAL UNTOUCHED TEST EVALUATION

> [!IMPORTANT]
> The test dataset (`test.parquet`, $306,201$ samples: $254,797$ Attack, $51,404$ Benign) was loaded and evaluated **one time only** on the frozen final models.

### Final Test Metrics

| Model | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Time | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | **0.977048** | **0.999661** | **0.972747** | **0.986020** | **0.995224** | 1.255s | 243,943 samples/s |
| **Random_12_07** | 12 | **0.977107** | **0.999738** | **0.972743** | **0.986056** | **0.998620** | 1.185s | 258,348 samples/s |

### Final Confusion Matrices

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Test Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | **51,320** | **84** | **6,944** | **247,853** | 306,201 |
| **Random_12_07** | **51,339** | **65** | **6,945** | **247,852** | 306,201 |

#### Analysis of Test Results:
- **Low False Alarm Rate:** Both models achieve false positive rates under $0.17\%$ (84 FP for Top-10 and only 65 FP for Random_12_07 out of 51,404 benign flows), which is critical for real-world IDS deployments.
- **High Attack Detection:** Both models correctly flag $>247,850$ out of $254,797$ attack instances ($>97.27\%$ recall) with near-zero false alarms.
- **Ultra-Fast Throughput:** Both models achieve throughput $>240,000$ flow records per second on test data (~$1.2$ seconds for $306,201$ records).

---

## 8. Reproducibility & Verification Audit

- **No Test Data Leakage:** Confirmed. Feature selection, K-Fold CV, and hyperparameter tuning used training rows only.
- **Deterministic Random Seed:** All steps used `random_state=42`.
- **Exact Matches:** All numbers reported above match the underlying CSV, JSON, and joblib artifacts exactly.
