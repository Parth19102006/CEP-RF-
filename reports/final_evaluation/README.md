# Phase 4: Final Candidate Selection, Model Training, and Untouched Test Evaluation

## Executive Summary

In Phase 4, two candidate lightweight feature subsets emerged with the highest F1-scores:
1. **Top-10:** 10 features selected via Random Forest Gini feature importance ranking.
2. **Random_12_07:** 12 features discovered during random feature subset exploration (#1 rank across 70 trials).

Both subsets underwent Stratified 5-Fold Cross-Validation hyperparameter tuning on the **training set only** (`train.parquet`, 125,170 rows).
Following hyperparameter tuning, model selection decisions were made strictly on training validation metrics.
Final models were trained on the complete training set and evaluated **one time** on the previously untouched test set (`test.parquet`, 306,201 rows).

---

## 1. Candidate Feature Sets

### A. Top-10 Feature List (10 Features)

Selected from the frozen baseline Random Forest importance ranking:
1. `URG Flag Count`
2. `Bwd Packets/s`
3. `Packet Length Min`
4. `Bwd Packet Length Max`
5. `Bwd Packet Length Mean`
6. `Avg Bwd Segment Size`
7. `Bwd Packets Length Total`
8. `Init Bwd Win Bytes`
9. `Fwd Packet Length Min`
10. `Subflow Bwd Bytes`

### B. Random_12_07 Feature List (12 Features)

Deterministic 7th 12-feature subset from random exploration:
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

## 2. Training-Only Cross-Validation & Hyperparameter Tuning Results

- **Cross-Validation Scheme:** Stratified 5-Fold Cross-Validation (`random_state=42`)
- **Training Samples:** 125,170 (Attack: 78,743 | Benign: 46,427)
- **Search Method:** `RandomizedSearchCV` (20 iterations, 100 fits per candidate)
- **Optimization Criterion:** Binary F1-Score (`refit='f1'`)

| Candidate | Feature Count | Best HT Configuration | CV F1 (Mean ± Std) | CV Precision | CV Recall | CV Accuracy |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 40, "criterion": "gini", "bootstrap": false}` | **0.999282 ± 0.000086** | 0.999467 | 0.999098 | 0.999097 |
| **Random_12_07** | 12 | `{"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 40, "criterion": "gini", "bootstrap": false}` | **0.999530 ± 0.000210** | 0.999797 | 0.999263 | 0.999409 |

### Selection Decision & Rationale

- **Decision:** **TIE (Both Retained)**
- **Rationale:** Top-10 (F1 = 0.999282 ± 0.000086) and Random_12_07 (F1 = 0.999530 ± 0.000210) have nearly identical CV F1 scores (delta = 0.000248 < 0.001). Per evaluation protocol, both candidates are retained as final candidate models without manufacturing an arbitrary winner.

---

## 3. FINAL UNTOUCHED TEST EVALUATION

> [!IMPORTANT]
> The test set (`data/processed/phase3/test.parquet`, 306,201 samples: 254,797 Attack, 51,404 Benign) was strictly kept untouched until all feature selection, hyperparameter tuning, and candidate decisions were finalized and frozen.

### Final Performance Metrics

| Model | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Time (s) | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 10 | **0.977048** | **0.999661** | **0.972747** | **0.986020** | **0.995224** | 1.255s | 243,943 |
| **Random_12_07** | 12 | **0.977107** | **0.999738** | **0.972743** | **0.986056** | **0.998620** | 1.185s | 258,348 |

### Final Confusion Matrices

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10** | 51,320 | 84 | 6,944 | 247,853 | 306,201 |
| **Random_12_07** | 51,339 | 65 | 6,945 | 247,852 | 306,201 |

---

## 4. Reproducibility & Integrity Statement

1. **No Test Leakage:** Neither the feature selection, nor the K-Fold CV, nor the hyperparameter tuning accessed the test set.
2. **Deterministic Execution:** All random states (`random_state=42`) and fold splits are fully documented and reproducible.
3. **Saved Models:** Final trained models are saved in the `models/` directory for deployment and inspection.