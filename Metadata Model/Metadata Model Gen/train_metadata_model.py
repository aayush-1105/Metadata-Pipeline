"""
Module I (Metadata Predictor) - Training, Calibration, and Evaluation Script
==============================================================================
AI-Powered CI/CD Predictor System

This script performs:
  1. Ingestion of the zipped 38-feature metadata CSV dataset.
  2. DEDUPLICATION (Leakage Prevention): Collapses matrix job duplicates (~4x) 
     into single unique build events to prevent exact-row leakage between splits.
  3. Stratified train/validation/test splitting (70/15/15) on target failure,
     preserving continuous feature spaces across all splits.
  4. Training of an unweighted LightGBM classifier optimized via early stopping
     on AUC.
  5. Dual probability calibration (Platt/sigmoid and isotonic) on validation data.
  6. Evaluation using unexploitable metrics: Matthews Correlation Coefficient
     (MCC) and the Composite Fitness Score [MCC@0.50 + (1 - Brier Score)].
  7. SHAP TreeExplainer construction for Module III Root Cause Analysis.
  8. Artifact persistence and detailed JSON/Markdown scorecard generation.

Author: Senior MLOps Engineer
"""

import json
import logging
import os
import sys
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

DATASET_ZIP_PATH = r"F:\Capstone\Dataset\Pre-Processed Metadata Dataset.zip"
DATASET_CSV_INTERNAL_NAME = "Pre-Processed Metadata Dataset.csv"
SCRIPT_ROOT = r"F:\Capstone\Metadata Model\Metadata Model Gen"
MODELS_ROOT = r"F:\Capstone\Metadata Model\Models"

RAW_TARGET_COL = "target_binary"  # 1 = PASS, 0 = FAIL
ENGINEERED_TARGET_COL = "y_fail"  # 1 = FAIL, 0 = PASS (model target)

# Strict 38-feature schema
FEATURE_COLUMNS = [
    "gh_is_pr",
    "gh_lang_java",
    "gh_lang_python",
    "gh_lang_ruby",
    "git_prev_commit_resolution_status_merge_found",
    "git_prev_commit_resolution_status_no_previous_build",
    "gh_num_issue_comments",
    "gh_num_commit_comments",
    "gh_num_pr_comments",
    "log_team_size",
    "log_built_commits",
    "gh_repo_age",
    "gh_repo_num_commits",
    "tr_build_number",
    "gh_sloc",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_num_commits_on_files_touched",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member",
    "total_files_changed",
    "code_churn_density",
    "test_to_src_churn_ratio",
    "has_previous_build",
    "gh_pr_created_at_year",
    "gh_pr_created_at_month",
    "pr_dayofweek",
    "git_merged_with_present",
]

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 12,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 5,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "metric": "auc",
}
EARLY_STOPPING_ROUNDS = 50

DIAGNOSTIC_THRESHOLDS = {
    "low_risk_25pct": 0.25,
    "standard_50pct": 0.50,
    "high_risk_canary_65pct": 0.65,
}

TOP_N_FEATURES = 15
ECE_NUM_BINS = 10

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("metadata_predictor")

# ==============================================================================
# 3. DATA INGESTION & TARGET ENGINEERING
# ==============================================================================

def load_dataset(zip_path: str) -> pd.DataFrame:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset archive not found at: {zip_path}")
    logger.info("Loading dataset directly from zip archive: %s", zip_path)
    df = pd.read_csv(zip_path, compression="zip")
    logger.info("Loaded dataframe with shape: %s", df.shape)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = set(FEATURE_COLUMNS + [RAW_TARGET_COL]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required schema columns: {sorted(missing_columns)}")
    logger.info("Schema validation successful. All %d features and target present.", len(FEATURE_COLUMNS))


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[ENGINEERED_TARGET_COL] = (1 - df[RAW_TARGET_COL]).astype(np.int32)
    logger.info("Target engineered: y_fail = 1 - target_binary (1 = FAIL, 0 = PASS).")
    return df


# ==============================================================================
# 4. LEAKAGE PREVENTION (BUILD DEDUPLICATION)
# ==============================================================================

def deduplicate_builds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses matrix job duplicates (~4 rows per commit) into single build events.
    Prevents exact-row twins from straddling Train and Test splits.
    """
    logger.info("Initiating deduplication to prevent cross-commit data leakage...")
    initial_rows = len(df)
    
    # Sort by y_fail descending (1s first, then 0s). 
    # This ensures that if ANY matrix job in a build failed, the collapsed build event counts as a FAIL.
    df_sorted = df.sort_values(by=ENGINEERED_TARGET_COL, ascending=False)
    
    # Drop duplicates across the exact 38-feature signature
    df_dedup = df_sorted.drop_duplicates(subset=FEATURE_COLUMNS, keep="first").reset_index(drop=True)
    
    final_rows = len(df_dedup)
    dupes_removed = initial_rows - final_rows
    logger.info(f"Deduplication complete. Removed {dupes_removed:,} duplicate matrix rows.")
    logger.info(f"True distinct build events remaining for modeling: {final_rows:,}")
    
    return df_dedup


# ==============================================================================
# 5. STRATIFIED SPLITTING
# ==============================================================================

def stratified_split(df: pd.DataFrame):
    """
    Executes a stratified 70/15/15 train/validation/test split on unique build events.
    Maintains equivalent target ratios and continuous metric bounds across splits.
    """
    logger.info("Performing stratified split (Train: 70%%, Val: 15%%, Test: 15%%)...")
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_FRACTION + TEST_FRACTION),
        random_state=42,
        stratify=df[ENGINEERED_TARGET_COL]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[ENGINEERED_TARGET_COL]
    )

    for name, split in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        logger.info(
            "%-5s -> %d unique builds | FAIL%%=%.2f | PR%%=%.2f",
            name,
            len(split),
            100.0 * split[ENGINEERED_TARGET_COL].mean(),
            100.0 * split["gh_is_pr"].mean(),
        )

    return train_df, val_df, test_df

# ==============================================================================
# 6. MODEL TRAINING & CALIBRATION
# ==============================================================================

def train_base_model(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(
        **LGBM_PARAMS,
        objective="binary",
        verbosity=-1,
    )

    logger.info(
        "Fitting LightGBM (num_leaves=%d, max_depth=%d, n_estimators=%d, lr=%.3f) with AUC early stopping...",
        LGBM_PARAMS["num_leaves"], LGBM_PARAMS["max_depth"],
        LGBM_PARAMS["n_estimators"], LGBM_PARAMS["learning_rate"]
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        ],
    )

    logger.info("Base tree model trained successfully. Best iteration: %s", model.best_iteration_)
    return model


def calibrate_models(base_model, X_val, y_val):
    def _make(method):
        if _HAS_FROZEN_ESTIMATOR:
            return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
        return CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")

    logger.info("Calibrating model using Sigmoid (Platt scaling) on validation split...")
    sigmoid_model = _make("sigmoid").fit(X_val, y_val)

    logger.info("Calibrating model using Isotonic Regression on validation split...")
    isotonic_model = _make("isotonic").fit(X_val, y_val)

    return sigmoid_model, isotonic_model

# ==============================================================================
# 7. EVALUATION METRICS & DIAGNOSTICS
# ==============================================================================

def compute_expected_calibration_error(y_true, y_prob, n_bins=ECE_NUM_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    total_samples = len(y_true)
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = int(np.sum(mask))
        if bin_count == 0:
            continue
        bin_confidence = np.mean(y_prob[mask])
        bin_accuracy = np.mean(y_true[mask])
        ece += (bin_count / total_samples) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def compute_threshold_diagnostics(y_true, y_prob, thresholds):
    diagnostics = {}
    for label, threshold in thresholds.items():
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        diagnostics[label] = {
            "threshold": threshold,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        }
    return diagnostics


def compute_class_distribution(df: pd.DataFrame) -> dict:
    y = df[ENGINEERED_TARGET_COL]
    total = len(y)
    fail_count = int((y == 1).sum())
    pass_count = int((y == 0).sum())
    return {
        "total_rows": total,
        "fail_count": fail_count,
        "pass_count": pass_count,
        "fail_percentage": round(100.0 * fail_count / total, 4) if total else 0.0,
        "pass_percentage": round(100.0 * pass_count / total, 4) if total else 0.0,
        "pr_percentage": round(100.0 * float(df["gh_is_pr"].mean()), 4) if total else 0.0,
    }


def get_top_n_feature_importance(base_model, feature_names, n=TOP_N_FEATURES):
    gains = base_model.booster_.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_names, gains), key=lambda p: p[1], reverse=True)
    return {name: round(float(gain), 4) for name, gain in pairs[:n]}


def summarize_probabilities(y_true, y_prob):
    y_pred_50 = (y_prob >= 0.50).astype(int)
    mcc_50 = float(matthews_corrcoef(y_true, y_pred_50))
    brier = float(brier_score_loss(y_true, y_prob))
    composite_fitness = mcc_50 + (1.0 - brier)

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "brier_score": round(brier, 6),
        "ece": round(compute_expected_calibration_error(y_true, y_prob), 6),
        "mcc_at_50": round(mcc_50, 6),
        "composite_fitness_score": round(composite_fitness, 6),
        "prob_min": round(float(np.min(y_prob)), 6),
        "prob_max": round(float(np.max(y_prob)), 6),
        "prob_mean": round(float(np.mean(y_prob)), 6),
        "prob_std": round(float(np.std(y_prob)), 6),
    }

# ==============================================================================
# 8. SCORECARD BUILDING & SERIALIZATION
# ==============================================================================

def build_scorecard(train_df, val_df, test_df, y_test, prob_map,
                    base_model, best_iteration, run_timestamp):
    primary = "calibrated_sigmoid"
    per_scorer = {name: summarize_probabilities(y_test, p) for name, p in prob_map.items()}
    threshold_diag = {
        name: compute_threshold_diagnostics(y_test, p, DIAGNOSTIC_THRESHOLDS)
        for name, p in prob_map.items()
    }

    return {
        "metadata": {
            "run_timestamp": run_timestamp,
            "module_name": "Module I - Metadata Predictor",
            "target_definition": "y_fail = 1 - target_binary (1 = FAIL, 0 = PASS)",
            "feature_count": len(FEATURE_COLUMNS),
            "primary_calibrator": "sigmoid (Platt)",
            "split_strategy": "deduplicated_stratified_70_15_15",
            "best_iteration": int(best_iteration) if best_iteration else None,
        },
        "hyperparameters": {
            **LGBM_PARAMS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },
        "dataset_demographics": {
            "train": compute_class_distribution(train_df),
            "validation": compute_class_distribution(val_df),
            "test": compute_class_distribution(test_df),
            "total_build_events_all_splits": len(train_df) + len(val_df) + len(test_df),
        },
        "global_metrics": {
            "composite_fitness_score": per_scorer[primary]["composite_fitness_score"],
            "mcc_at_50": per_scorer[primary]["mcc_at_50"],
            "roc_auc": per_scorer[primary]["roc_auc"],
            "pr_auc": per_scorer[primary]["pr_auc"],
            "note": "ROC-AUC and PR-AUC are invariant under monotone calibration.",
        },
        "calibration_metrics": {
            "per_scorer": per_scorer,
            "brier_improvement_sigmoid_vs_raw": round(
                per_scorer["raw_lightgbm"]["brier_score"] - per_scorer["calibrated_sigmoid"]["brier_score"], 6
            ),
            "brier_improvement_isotonic_vs_raw": round(
                per_scorer["raw_lightgbm"]["brier_score"] - per_scorer["calibrated_isotonic"]["brier_score"], 6
            ),
            "calibration_bins_used": ECE_NUM_BINS,
        },
        "threshold_diagnostics": threshold_diag,
        "feature_importance_top_15_by_gain": get_top_n_feature_importance(base_model, FEATURE_COLUMNS),
    }


def write_scorecard_markdown(scorecard, output_path):
    md = []
    meta = scorecard["metadata"]
    gm = scorecard["global_metrics"]

    md.append("# Module I (Metadata Predictor) - Evaluation Scorecard")
    md.append("")
    md.append(f"**Run Timestamp:** `{meta['run_timestamp']}`  ")
    md.append(f"**Target Definition:** {meta['target_definition']}  ")
    md.append(f"**Feature Count:** {meta['feature_count']}  ")
    md.append(f"**Split Strategy:** {meta['split_strategy']}  ")
    md.append(f"**Primary Calibrator:** {meta['primary_calibrator']}  ")
    md.append(f"**Best Iteration:** {meta['best_iteration']}")
    md.append("")

    md.append("## Unexploitable Fitness Metrics (Test Set)")
    md.append("")
    md.append(f"- **Composite Fitness Score [MCC@0.50 + (1 - Brier)]:** **{gm['composite_fitness_score']}**")
    md.append(f"- **Matthews Correlation Coefficient (MCC @ 0.50):** {gm['mcc_at_50']}")
    md.append(f"- **ROC-AUC:** {gm['roc_auc']}")
    md.append(f"- **PR-AUC:** {gm['pr_auc']}")
    md.append("")

    md.append("## Dataset Demographics (Unique Build Events)")
    md.append("")
    md.append("| Split | Total Builds | Fail Count | Fail % | Pass Count | Pass % | PR % |")
    md.append("|---|---|---|---|---|---|---|")
    for split in ["train", "validation", "test"]:
        d = scorecard["dataset_demographics"][split]
        md.append(
            f"| {split.capitalize()} | {d['total_rows']} | {d['fail_count']} | "
            f"{d['fail_percentage']}% | {d['pass_count']} | {d['pass_percentage']}% | "
            f"{d['pr_percentage']}% |"
        )
    md.append("")
    md.append(f"**Total distinct build events across all splits:** {scorecard['dataset_demographics']['total_build_events_all_splits']}")
    md.append("")

    md.append("## Calibration Metrics (raw vs calibrated)")
    md.append("")
    md.append("| Scorer | Brier | ECE | MCC@0.50 | Fitness | Prob Min | Prob Max | Prob Mean | Prob Std |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for name, s in scorecard["calibration_metrics"]["per_scorer"].items():
        md.append(
            f"| {name} | {s['brier_score']} | {s['ece']} | {s['mcc_at_50']} | "
            f"{s['composite_fitness_score']} | {s['prob_min']} | {s['prob_max']} | "
            f"{s['prob_mean']} | {s['prob_std']} |"
        )
    md.append("")
    cm = scorecard["calibration_metrics"]
    md.append(f"- **Brier improvement (sigmoid vs raw):** {cm['brier_improvement_sigmoid_vs_raw']}")
    md.append(f"- **Brier improvement (isotonic vs raw):** {cm['brier_improvement_isotonic_vs_raw']}")
    md.append("")

    md.append("## Threshold Diagnostics (Confusion Matrices & Unexploitable MCC)")
    md.append("")
    for scorer, diag_set in scorecard["threshold_diagnostics"].items():
        md.append(f"### {scorer}")
        md.append("")
        md.append("| Threshold | TN | FP | FN | TP | FPR | Precision | Recall | F1 | MCC |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, diag in diag_set.items():
            c = diag["confusion_matrix"]
            md.append(
                f"| {diag['threshold']} | {c['true_negatives']} | {c['false_positives']} | "
                f"{c['false_negatives']} | {c['true_positives']} | {diag['false_positive_rate']} | "
                f"{diag['precision']} | {diag['recall']} | {diag['f1_score']} | {diag['mcc']} |"
            )
        md.append("")

    md.append("## Top 15 Features by LightGBM Split Gain")
    md.append("")
    md.append("| Rank | Feature | Gain |")
    md.append("|---|---|---|")
    for rank, (feat, gain) in enumerate(scorecard["feature_importance_top_15_by_gain"].items(), start=1):
        md.append(f"| {rank} | {feat} | {gain} |")
    md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("Wrote Markdown scorecard to: %s", output_path)

# ==============================================================================
# 9. MAIN EXECUTION PIPELINE
# ==============================================================================

def main() -> None:
    logger.info("=" * 80)
    logger.info("Starting Module I (Metadata Predictor) Training Pipeline")
    logger.info("=" * 80)

    os.makedirs(SCRIPT_ROOT, exist_ok=True)
    os.makedirs(MODELS_ROOT, exist_ok=True)

    # 1. Ingestion & Target Engineering
    df_raw = load_dataset(DATASET_ZIP_PATH)
    validate_schema(df_raw)
    df = engineer_target(df_raw)

    # 2. Leakage Prevention (Deduplication)
    df = deduplicate_builds(df)

    # 3. Stratified Splitting
    train_df, val_df, test_df = stratified_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[ENGINEERED_TARGET_COL]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[ENGINEERED_TARGET_COL]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[ENGINEERED_TARGET_COL].to_numpy()

    # 4. Base LightGBM Training
    base_model = train_base_model(X_train, y_train, X_val, y_val)

    # 5. Calibration
    sigmoid_model, isotonic_model = calibrate_models(base_model, X_val, y_val)

    # 6. Test Inference
    logger.info("Evaluating predictions on held-out test split...")
    prob_map = {
        "raw_lightgbm": base_model.predict_proba(X_test)[:, 1],
        "calibrated_sigmoid": sigmoid_model.predict_proba(X_test)[:, 1],
        "calibrated_isotonic": isotonic_model.predict_proba(X_test)[:, 1],
    }

    # 7. SHAP TreeExplainer
    logger.info("Building SHAP TreeExplainer for Module III integration...")
    explainer = shap.TreeExplainer(base_model)

    # 8. Persistence
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MODELS_ROOT, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    artifacts = {
        "calibrated_metadata_model.joblib": sigmoid_model,
        "calibrated_metadata_model_isotonic.joblib": isotonic_model,
        "base_lightgbm_model.joblib": base_model,
        "shap_explainer.joblib": explainer,
    }
    for filename, obj in artifacts.items():
        path = os.path.join(run_dir, filename)
        joblib.dump(obj, path)
        logger.info("Saved %s", path)

    # 9. Scorecards
    scorecard = build_scorecard(
        train_df, val_df, test_df, y_test, prob_map,
        base_model, base_model.best_iteration_, timestamp
    )
    with open(os.path.join(run_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    write_scorecard_markdown(scorecard, os.path.join(run_dir, "scorecard.md"))

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully. Artifacts saved to: %s", run_dir)
    logger.info("Fitness Score: %s | MCC @ 0.50: %s | Test ROC-AUC: %s",
                scorecard["global_metrics"]["composite_fitness_score"],
                scorecard["global_metrics"]["mcc_at_50"],
                scorecard["global_metrics"]["roc_auc"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()