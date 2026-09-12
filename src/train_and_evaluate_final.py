"""Phase 4: Final candidate comparison, model training, and one-time untouched test evaluation.

This script executes:
1. Training-only comparison of Top-10 and Random_12_07 based on CV hyperparameter tuning.
2. Final model training on full training data (125,170 samples).
3. ONE-TIME final untouched test evaluation on test.parquet (306,201 samples).
4. Generation of final evaluation artifacts and comprehensive reporting.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import run_kfold_cv as cv


PROJECT_ROOT = cv.PROJECT_ROOT
DEFAULT_PHASE3_DIR = cv.DEFAULT_PHASE3_DIR
DEFAULT_PHASE3_METADATA = cv.DEFAULT_PHASE3_METADATA
DEFAULT_BASELINE_IMPORTANCE = cv.DEFAULT_BASELINE_IMPORTANCE
DEFAULT_HT_REPORT_DIR = PROJECT_ROOT / "reports" / "hyperparameter_tuning"
DEFAULT_EVAL_REPORT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = cv.RANDOM_STATE
TARGET_COLUMN = cv.TARGET_COLUMN
ORIGINAL_LABEL_COLUMN = cv.ORIGINAL_LABEL_COLUMN
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Benign (0)", "DDoS/Attack (1)"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Final training and one-time untouched test evaluation."
    )
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--phase3-metadata", type=Path, default=DEFAULT_PHASE3_METADATA)
    parser.add_argument("--baseline-importance", type=Path, default=DEFAULT_BASELINE_IMPORTANCE)
    parser.add_argument("--ht-report-dir", type=Path, default=DEFAULT_HT_REPORT_DIR)
    parser.add_argument("--eval-report-dir", type=Path, default=DEFAULT_EVAL_REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def load_random_12_07_features(feature_names: list[str]) -> list[str]:
    """Reproducibly extract the exact 12 features of Random_12_07."""
    rng = np.random.default_rng(RANDOM_STATE)
    all_subset_keys: set[tuple[str, ...]] = set()
    experiments: dict[str, list[str]] = {}

    for size in [5, 8, 10, 12, 15, 20, 30]:
        generated_for_size = 0
        attempts = 0
        while generated_for_size < 10:
            attempts += 1
            selected = rng.choice(feature_names, size=size, replace=False).tolist()
            subset_key = tuple(sorted(selected))
            if subset_key in all_subset_keys:
                continue
            all_subset_keys.add(subset_key)
            generated_for_size += 1
            name = f"Random_{size}_{generated_for_size:02d}"
            experiments[name] = selected

    return experiments["Random_12_07"]


def load_best_params(ht_report_dir: Path) -> dict[str, dict[str, Any]]:
    """Load tuned hyperparameters from hyperparameter tuning artifacts."""
    params_file = ht_report_dir / "tuning_best_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Tuning best params file not found: {params_file}")
    return json.loads(params_file.read_text(encoding="utf-8"))


def load_ht_summary(ht_report_dir: Path) -> pd.DataFrame:
    """Load hyperparameter tuning summary table."""
    summary_file = ht_report_dir / "tuning_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Tuning summary file not found: {summary_file}")
    return pd.read_csv(summary_file)


def evaluate_test_set(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Perform one-time final test set evaluation."""
    start_time = time.perf_counter()
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    inference_time = time.perf_counter() - start_time

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=6
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "classification_report": report_dict,
        "test_inference_time_seconds": inference_time,
        "test_samples_per_second": float(len(y_test) / inference_time) if inference_time > 0 else 0.0,
    }


def main() -> int:
    """Run candidate comparison, model training, and final test evaluation."""
    args = parse_args()
    phase3_dir = cv.resolve_project_path(args.phase3_dir)
    phase3_metadata_path = cv.resolve_project_path(args.phase3_metadata)
    importance_path = cv.resolve_project_path(args.baseline_importance)
    ht_report_dir = cv.resolve_project_path(args.ht_report_dir)
    eval_report_dir = cv.resolve_project_path(args.eval_report_dir)
    model_dir = cv.resolve_project_path(args.model_dir)

    eval_report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  PHASE 4: CANDIDATE COMPARISON, FINAL TRAINING & TEST EVALUATION")
    print("=================================================================\n")

    # 1. Load training dataset and metadata
    feature_names = cv.load_feature_names(phase3_metadata_path)
    train_df = cv.load_training_frame(phase3_dir)
    cv.validate_training_frame(train_df, feature_names)

    top10_features = cv.load_top10_features(importance_path, feature_names)
    random1207_features = load_random_12_07_features(feature_names)

    # 2. STEP 3: Compare Final Candidates (Training-Only CV)
    ht_summary = load_ht_summary(ht_report_dir)
    best_params_all = load_best_params(ht_report_dir)

    print("--- STEP 3: TRAINING-ONLY CANDIDATE COMPARISON ---")
    print(ht_summary[["subset_id", "feature_count", "f1_mean", "f1_std", "precision_mean", "recall_mean", "accuracy_mean"]])

    top10_cv_row = ht_summary[ht_summary["subset_id"] == "Top-10"].iloc[0]
    rand1207_cv_row = ht_summary[ht_summary["subset_id"] == "Random_12_07"].iloc[0]

    top10_f1_mean = float(top10_cv_row["f1_mean"])
    top10_f1_std = float(top10_cv_row["f1_std"])
    rand1207_f1_mean = float(rand1207_cv_row["f1_mean"])
    rand1207_f1_std = float(rand1207_cv_row["f1_std"])

    f1_diff = abs(top10_f1_mean - rand1207_f1_mean)
    print(f"\nTop-10 CV F1:       {top10_f1_mean:.6f} ± {top10_f1_std:.6f}")
    print(f"Random_12_07 CV F1: {rand1207_f1_mean:.6f} ± {rand1207_f1_std:.6f}")
    print(f"Absolute F1 difference: {f1_diff:.6f}")

    # Decision rationale:
    # Check if statistically/meaningfully distinguishable or practically tied.
    # We follow the protocol: If distinguishable, select best; if effectively tied / both valid, retain both.
    tie_threshold = 0.001  # 0.1% margin
    is_tie = f1_diff < tie_threshold

    if not is_tie:
        if top10_f1_mean > rand1207_f1_mean:
            selection_decision = "Top-10"
            selection_rationale = (
                f"Top-10 achieved higher mean CV F1 ({top10_f1_mean:.6f} vs {rand1207_f1_mean:.6f}) "
                f"with fewer features (10 vs 12) during training-only 5-fold CV."
            )
        else:
            selection_decision = "Random_12_07"
            selection_rationale = (
                f"Random_12_07 achieved higher mean CV F1 ({rand1207_f1_mean:.6f} vs {top10_f1_mean:.6f}) "
                f"during training-only 5-fold CV."
            )
        retained_candidates = [selection_decision]
    else:
        selection_decision = "TIE (Both Retained)"
        selection_rationale = (
            f"Top-10 (F1 = {top10_f1_mean:.6f} ± {top10_f1_std:.6f}) and Random_12_07 "
            f"(F1 = {rand1207_f1_mean:.6f} ± {rand1207_f1_std:.6f}) have nearly identical CV F1 scores "
            f"(delta = {f1_diff:.6f} < {tie_threshold}). Per evaluation protocol, both candidates "
            f"are retained as final candidate models without manufacturing an arbitrary winner."
        )
        retained_candidates = ["Top-10", "Random_12_07"]

    print(f"\nSelection Decision: {selection_decision}")
    print(f"Rationale: {selection_rationale}")

    # 3. STEP 4: Final Model Training on Full Training Set
    print("\n--- STEP 4: FINAL MODEL TRAINING ---")
    trained_models: dict[str, RandomForestClassifier] = {}
    training_metadata: dict[str, Any] = {}

    candidates_to_train = [
        ("Top-10", top10_features, best_params_all["Top-10"]),
        ("Random_12_07", random1207_features, best_params_all["Random_12_07"]),
    ]

    x_train_full = train_df
    y_train_full = train_df[TARGET_COLUMN].astype(int).to_numpy()
    train_sample_count = len(train_df)
    train_class_counts = train_df[TARGET_COLUMN].value_counts().to_dict()

    for name, features, params in candidates_to_train:
        print(f"\nTraining final model for {name} ({len(features)} features)...")
        print(f"Hyperparameters: {json.dumps(params)}")
        rf = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        rf.fit(x_train_full[features], y_train_full)
        train_time = time.perf_counter() - t0
        print(f"-> Fitted in {train_time:.2f} seconds.")

        model_filename = f"rf_final_{name.lower().replace('-', '_')}.joblib"
        model_path = model_dir / model_filename
        joblib.dump(rf, model_path)
        print(f"-> Saved model to: {cv.project_relative(model_path)}")

        trained_models[name] = rf
        training_metadata[name] = {
            "candidate_name": name,
            "feature_count": len(features),
            "features": features,
            "hyperparameters": params,
            "random_state": RANDOM_STATE,
            "train_sample_count": train_sample_count,
            "train_class_counts": {
                "0_Benign": int(train_class_counts.get(0, 0)),
                "1_Attack": int(train_class_counts.get(1, 0)),
            },
            "training_time_seconds": train_time,
            "model_path": cv.project_relative(model_path),
        }

    # 4. STEP 5: ONE-TIME FINAL UNTOUCHED TEST EVALUATION
    print("\n--- STEP 5: ONE-TIME FINAL UNTOUCHED TEST EVALUATION ---")
    test_path = phase3_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    test_df = pd.read_parquet(test_path)
    test_sample_count = len(test_df)
    test_class_counts = test_df[TARGET_COLUMN].value_counts().to_dict()
    y_test_full = test_df[TARGET_COLUMN].astype(int).to_numpy()

    print(f"Test Set Loaded: {test_sample_count:,} samples ({test_class_counts.get(1, 0):,} Attack, {test_class_counts.get(0, 0):,} Benign)")

    final_test_results: dict[str, Any] = {}
    test_summary_rows: list[dict[str, Any]] = []
    cm_summary_rows: list[dict[str, Any]] = []

    for name, features, _ in candidates_to_train:
        print(f"\nEvaluating final {name} model on untouched test set...")
        rf = trained_models[name]
        eval_metrics = evaluate_test_set(rf, test_df[features], y_test_full)
        final_test_results[name] = eval_metrics

        print(f"-> Accuracy:  {eval_metrics['accuracy']:.6f}")
        print(f"-> Precision: {eval_metrics['precision']:.6f}")
        print(f"-> Recall:    {eval_metrics['recall']:.6f}")
        print(f"-> F1-Score:  {eval_metrics['f1_score']:.6f}")
        print(f"-> ROC-AUC:   {eval_metrics['roc_auc']:.6f}")
        cm = eval_metrics["confusion_matrix"]
        print(f"-> Confusion Matrix: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

        test_summary_rows.append(
            {
                "model": name,
                "feature_count": len(features),
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1_score": eval_metrics["f1_score"],
                "roc_auc": eval_metrics["roc_auc"],
                "inference_time_seconds": eval_metrics["test_inference_time_seconds"],
                "samples_per_second": eval_metrics["test_samples_per_second"],
                "is_retained_candidate": name in retained_candidates,
            }
        )

        cm_summary_rows.append(
            {
                "model": name,
                "true_negatives": cm["tn"],
                "false_positives": cm["fp"],
                "false_negatives": cm["fn"],
                "true_positives": cm["tp"],
                "total_test_samples": test_sample_count,
            }
        )

    test_metrics_df = pd.DataFrame(test_summary_rows)
    cm_df = pd.DataFrame(cm_summary_rows)

    test_metrics_df.to_csv(eval_report_dir / "final_test_metrics.csv", index=False)
    cm_df.to_csv(eval_report_dir / "final_confusion_matrices.csv", index=False)

    # 5. STEP 6: Write Final Evaluation Artifacts & Markdown Report
    full_eval_payload = {
        "title": "Phase 4 Final Candidate Selection, Training and Untouched Test Evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "CIC-DDoS2019",
        "phase3_train_path": cv.project_relative(phase3_dir / "train.parquet"),
        "phase3_test_path": cv.project_relative(test_path),
        "train_sample_count": train_sample_count,
        "train_class_counts": {
            "0_Benign": int(train_class_counts.get(0, 0)),
            "1_Attack": int(train_class_counts.get(1, 0)),
        },
        "test_sample_count": test_sample_count,
        "test_class_counts": {
            "0_Benign": int(test_class_counts.get(0, 0)),
            "1_Attack": int(test_class_counts.get(1, 0)),
        },
        "random_state": RANDOM_STATE,
        "selection_decision": selection_decision,
        "selection_rationale": selection_rationale,
        "retained_candidates": retained_candidates,
        "training_metadata": training_metadata,
        "final_untouched_test_evaluation": final_test_results,
    }

    (eval_report_dir / "final_evaluation_results.json").write_text(
        json.dumps(full_eval_payload, indent=2), encoding="utf-8"
    )

    # Markdown Report
    md_lines = [
        "# Phase 4: Final Candidate Selection, Model Training, and Untouched Test Evaluation",
        "",
        "## Executive Summary",
        "",
        "In Phase 4, two candidate lightweight feature subsets emerged with the highest F1-scores:",
        "1. **Top-10:** 10 features selected via Random Forest Gini feature importance ranking.",
        "2. **Random_12_07:** 12 features discovered during random feature subset exploration (#1 rank across 70 trials).",
        "",
        "Both subsets underwent Stratified 5-Fold Cross-Validation hyperparameter tuning on the **training set only** (`train.parquet`, 125,170 rows).",
        "Following hyperparameter tuning, model selection decisions were made strictly on training validation metrics.",
        "Final models were trained on the complete training set and evaluated **one time** on the previously untouched test set (`test.parquet`, 306,201 rows).",
        "",
        "---",
        "",
        "## 1. Candidate Feature Sets",
        "",
        "### A. Top-10 Feature List (10 Features)",
        "",
        "Selected from the frozen baseline Random Forest importance ranking:",
    ]
    for idx, f in enumerate(top10_features, 1):
        md_lines.append(f"{idx}. `{f}`")

    md_lines.extend(
        [
            "",
            "### B. Random_12_07 Feature List (12 Features)",
            "",
            "Deterministic 7th 12-feature subset from random exploration:",
        ]
    )
    for idx, f in enumerate(random1207_features, 1):
        md_lines.append(f"{idx}. `{f}`")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Training-Only Cross-Validation & Hyperparameter Tuning Results",
            "",
            "- **Cross-Validation Scheme:** Stratified 5-Fold Cross-Validation (`random_state=42`)",
            "- **Training Samples:** 125,170 (Attack: 78,743 | Benign: 46,427)",
            "- **Search Method:** `RandomizedSearchCV` (20 iterations, 100 fits per candidate)",
            "- **Optimization Criterion:** Binary F1-Score (`refit='f1'`)",
            "",
            "| Candidate | Feature Count | Best HT Configuration | CV F1 (Mean ± Std) | CV Precision | CV Recall | CV Accuracy |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
            f"| **Top-10** | 10 | `{json.dumps(best_params_all['Top-10'])}` | **{top10_f1_mean:.6f} ± {top10_f1_std:.6f}** | {float(top10_cv_row['precision_mean']):.6f} | {float(top10_cv_row['recall_mean']):.6f} | {float(top10_cv_row['accuracy_mean']):.6f} |",
            f"| **Random_12_07** | 12 | `{json.dumps(best_params_all['Random_12_07'])}` | **{rand1207_f1_mean:.6f} ± {rand1207_f1_std:.6f}** | {float(rand1207_cv_row['precision_mean']):.6f} | {float(rand1207_cv_row['recall_mean']):.6f} | {float(rand1207_cv_row['accuracy_mean']):.6f} |",
            "",
            "### Selection Decision & Rationale",
            "",
            f"- **Decision:** **{selection_decision}**",
            f"- **Rationale:** {selection_rationale}",
            "",
            "---",
            "",
            "## 3. FINAL UNTOUCHED TEST EVALUATION",
            "",
            "> [!IMPORTANT]",
            "> The test set (`data/processed/phase3/test.parquet`, 306,201 samples: 254,797 Attack, 51,404 Benign) was strictly kept untouched until all feature selection, hyperparameter tuning, and candidate decisions were finalized and frozen.",
            "",
            "### Final Performance Metrics",
            "",
            "| Model | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Time (s) | Throughput (samples/s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in test_summary_rows:
        md_lines.append(
            f"| **{row['model']}** | {row['feature_count']} | **{row['accuracy']:.6f}** | **{row['precision']:.6f}** | **{row['recall']:.6f}** | **{row['f1_score']:.6f}** | **{row['roc_auc']:.6f}** | {row['inference_time_seconds']:.3f}s | {row['samples_per_second']:,.0f} |"
        )

    md_lines.extend(
        [
            "",
            "### Final Confusion Matrices",
            "",
            "| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Samples |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in cm_summary_rows:
        md_lines.append(
            f"| **{row['model']}** | {row['true_negatives']:,} | {row['false_positives']:,} | {row['false_negatives']:,} | {row['true_positives']:,} | {row['total_test_samples']:,} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Reproducibility & Integrity Statement",
            "",
            "1. **No Test Leakage:** Neither the feature selection, nor the K-Fold CV, nor the hyperparameter tuning accessed the test set.",
            "2. **Deterministic Execution:** All random states (`random_state=42`) and fold splits are fully documented and reproducible.",
            "3. **Saved Models:** Final trained models are saved in the `models/` directory for deployment and inspection.",
        ]
    )

    (eval_report_dir / "final_evaluation_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    (eval_report_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nFinal evaluation reports written to: {cv.project_relative(eval_report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
