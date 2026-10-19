"""
Module I (Metadata Predictor) - Dedicated TreeSHAP Attribution Pipeline
=======================================================================
AI-Powered CI/CD Predictor System

This script produces the exact global + local SHAP attribution artifacts that
feed Page 4 of the Module I Executive Report:

  1. Loads the trained base LightGBM model.
  2. Reproduces the deterministic held-out TEST split bit-for-bit (identical
     dedup + seed + stratification as train_metadata_model.py).
  3. Draws a stratified sample of 5,000 build events that preserves the true
     test failure rate (~25.1%) across the strict 38-feature contract.
  4. Loads the persisted shap_explainer.joblib if present, otherwise builds a
     fresh shap.TreeExplainer on the base model.
  5. Computes exact SHAP values for the 5,000 samples and the Mean Absolute
     SHAP (mean(|SHAP|)) ranking for all 38 features.
  6. Exports shap_scorecard.json and shap_scorecard.md.
  7. Renders high-resolution (DPI=300) beeswarm + high/low-risk waterfall PNGs.

Run:
    python "F:\\Capstone\\Metadata Model\\Metadata Model Gen\\run_shap_analysis.py"
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (headless-safe)
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
RUN_DIR = r"F:\Capstone\Metadata Model\Models\run_20260926_013515"
EXPORTS_DIR = os.path.join(RUN_DIR, "exports")
BASE_MODEL_PATH = os.path.join(RUN_DIR, "base_lightgbm_model.joblib")
EXPLAINER_PATH = os.path.join(RUN_DIR, "shap_explainer.joblib")
DATASET_ZIP = r"F:\Capstone\Dataset\Pre-Processed Metadata Dataset.zip"

SHAP_SCORECARD_JSON = os.path.join(RUN_DIR, "shap_scorecard.json")
SHAP_SCORECARD_MD = os.path.join(RUN_DIR, "shap_scorecard.md")
BEESWARM_PNG = os.path.join(EXPORTS_DIR, "shap_beeswarm.png")
WATERFALL_HIGH_PNG = os.path.join(EXPORTS_DIR, "shap_waterfall_high_risk.png")
WATERFALL_LOW_PNG = os.path.join(EXPORTS_DIR, "shap_waterfall_low_risk.png")

# Strict 38-feature schema (must match train_metadata_model.py exactly so the
# held-out test split is reproduced bit-for-bit).
FEATURE_COLUMNS = [
    "gh_is_pr", "gh_lang_java", "gh_lang_python", "gh_lang_ruby",
    "git_prev_commit_resolution_status_merge_found",
    "git_prev_commit_resolution_status_no_previous_build",
    "gh_num_issue_comments", "gh_num_commit_comments", "gh_num_pr_comments",
    "log_team_size", "log_built_commits", "gh_repo_age", "gh_repo_num_commits",
    "tr_build_number", "gh_sloc", "git_diff_src_churn", "git_diff_test_churn",
    "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified",
    "gh_diff_tests_added", "gh_diff_tests_deleted", "gh_diff_src_files",
    "gh_diff_doc_files", "gh_diff_other_files", "gh_num_commits_on_files_touched",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member", "total_files_changed", "code_churn_density",
    "test_to_src_churn_ratio", "has_previous_build", "gh_pr_created_at_year",
    "gh_pr_created_at_month", "pr_dayofweek", "git_merged_with_present",
]
RAW_TARGET_COL = "target_binary"   # 1 = PASS, 0 = FAIL
ENGINEERED_TARGET_COL = "y_fail"   # 1 = FAIL, 0 = PASS

SAMPLE_SIZE = 5000
HIGH_RISK_THRESHOLD = 0.65   # Strict Pre-Merge / Canary Block
LOW_RISK_THRESHOLD = 0.25    # Fast Track
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("shap_analysis")


# ----------------------------------------------------------------------------
# Data: reproduce the exact held-out test split, then stratified-sample it
# ----------------------------------------------------------------------------
def reproduce_test_split():
    """Rebuild the deterministic 15% test split (same dedup + seed as training)."""
    logger.info("Loading dataset from zip archive: %s", DATASET_ZIP)
    df = pd.read_csv(DATASET_ZIP, compression="zip")
    df[ENGINEERED_TARGET_COL] = (1 - df[RAW_TARGET_COL]).astype(np.int32)

    # Collapse matrix-job duplicates on the 38-feature signature (leakage guard),
    # sorting FAIL-first so any failed job marks the collapsed build as a failure.
    df = (df.sort_values(by=ENGINEERED_TARGET_COL, ascending=False)
            .drop_duplicates(subset=FEATURE_COLUMNS, keep="first")
            .reset_index(drop=True))

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE,
        stratify=df[ENGINEERED_TARGET_COL])
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE,
        stratify=temp_df[ENGINEERED_TARGET_COL])

    logger.info("Reproduced held-out test split: %d unique builds | FAIL%%=%.4f",
                len(test_df), 100.0 * test_df[ENGINEERED_TARGET_COL].mean())
    return test_df


def stratified_sample(test_df, n=SAMPLE_SIZE):
    """Draw an n-row sample stratified on y_fail so the true test failure rate
    (~25.1%) is preserved across the 38-feature contract."""
    n = min(n, len(test_df))
    sample_df, _ = train_test_split(
        test_df, train_size=n, random_state=RANDOM_STATE,
        stratify=test_df[ENGINEERED_TARGET_COL])
    sample_df = sample_df.reset_index(drop=True)
    logger.info("Stratified sample drawn: %d rows | FAIL%%=%.4f",
                len(sample_df), 100.0 * sample_df[ENGINEERED_TARGET_COL].mean())
    return sample_df


# ----------------------------------------------------------------------------
# SHAP explainer + value computation
# ----------------------------------------------------------------------------
def load_or_build_explainer(base_model):
    if os.path.exists(EXPLAINER_PATH):
        logger.info("Loading persisted SHAP explainer: %s", EXPLAINER_PATH)
        return joblib.load(EXPLAINER_PATH)
    logger.info("No persisted explainer found; building shap.TreeExplainer(base_model).")
    return shap.TreeExplainer(base_model)


def _positive_class_2d(values):
    """Collapse SHAP value output to a 2-D (n_samples, n_features) array for the
    positive (FAIL) class, regardless of legacy/new API return shape."""
    arr = np.asarray(values)
    if arr.ndim == 3:            # (n, features, outputs) -> positive class
        arr = arr[..., -1]
    return arr


def _positive_class_base(base, n_rows):
    """Normalise base/expected values to a 1-D (n_samples,) array."""
    arr = np.asarray(base, dtype=float)
    if arr.ndim == 2:            # (n, outputs) -> positive class
        arr = arr[:, -1]
    arr = np.ravel(arr)
    if arr.size == 1:
        arr = np.full(n_rows, float(arr[0]))
    return arr


def compute_shap(explainer, X_sample):
    """Return an (explanation, values_2d, base_1d, baseline_scalar) bundle with
    exact SHAP values for the positive (FAIL) class."""
    logger.info("Computing exact SHAP values for %d samples across %d features...",
                len(X_sample), X_sample.shape[1])
    explanation = explainer(X_sample, check_additivity=False)
    values_2d = _positive_class_2d(explanation.values)
    base_1d = _positive_class_base(explanation.base_values, len(X_sample))

    # Baseline expected value (log-odds space, positive class) from the explainer.
    baseline_scalar = float(np.ravel(np.asarray(explainer.expected_value))[-1])

    # Rebuild a clean single-output Explanation so downstream plots always carry
    # the correct feature names and positive-class attributions.
    explanation_pos = shap.Explanation(
        values=values_2d,
        base_values=base_1d,
        data=np.asarray(X_sample.values),
        feature_names=list(X_sample.columns),
    )
    return explanation_pos, values_2d, base_1d, baseline_scalar


def rank_features(values_2d, feature_names):
    """Mean Absolute SHAP per feature, sorted descending."""
    mean_abs = np.abs(values_2d).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"rank": int(rank),
         "feature_name": feature_names[idx],
         "mean_abs_shap": round(float(mean_abs[idx]), 8)}
        for rank, idx in enumerate(order, start=1)
    ]


# ----------------------------------------------------------------------------
# Scorecard serialization
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def write_scorecard(baseline_scalar, ranked_features):
    baseline_prob = float(sigmoid(baseline_scalar))
    scorecard = {
        "metadata": {
            "module_name": "Module I - Metadata Predictor",
            "artifact_run": os.path.basename(RUN_DIR),
            "attribution_method": "TreeSHAP (shap.TreeExplainer)",
            "shap_output_space": "log-odds (raw margin, positive/FAIL class)",
            "feature_count": len(FEATURE_COLUMNS),
        },
        "baseline_expected_value": round(baseline_scalar, 8),
        "baseline_expected_value_probability": round(baseline_prob, 8),
        "sample_size": SAMPLE_SIZE,
        "ranked_features": ranked_features,
    }
    with open(SHAP_SCORECARD_JSON, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)
    logger.info("Wrote SHAP scorecard JSON -> %s", SHAP_SCORECARD_JSON)

    md = []
    md.append("# Module I (Metadata Predictor) - TreeSHAP Attribution Scorecard")
    md.append("")
    md.append(f"**Artifact Run:** `{os.path.basename(RUN_DIR)}`  ")
    md.append(f"**Attribution Method:** TreeSHAP (`shap.TreeExplainer`)  ")
    md.append(f"**SHAP Output Space:** log-odds (raw margin, positive/FAIL class)  ")
    md.append(f"**Sample Size:** {SAMPLE_SIZE:,} stratified test build events  ")
    md.append(f"**Feature Count:** {len(FEATURE_COLUMNS)}")
    md.append("")
    md.append(f"**Baseline Expected Value E[f(x)]:** `{baseline_scalar:.6f}` "
              f"(log-odds)  ->  `{baseline_prob*100:.2f}%` baseline failure probability")
    md.append("")
    md.append("## Global Feature Attribution - Mean Absolute SHAP (descending)")
    md.append("")
    md.append("| Rank | Feature | Mean \\|SHAP\\| |")
    md.append("|---|---|---|")
    for row in ranked_features:
        md.append(f"| {row['rank']} | {row['feature_name']} | {row['mean_abs_shap']:.6f} |")
    md.append("")
    with open(SHAP_SCORECARD_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    logger.info("Wrote SHAP scorecard Markdown -> %s", SHAP_SCORECARD_MD)


# ----------------------------------------------------------------------------
# High-resolution plots (DPI=300)
# ----------------------------------------------------------------------------
def save_beeswarm(explanation_pos):
    # Natural 4:3 aspect with fewer rows so feature labels stay large and clear.
    shap.plots.beeswarm(explanation_pos, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.0, 4.8)
    fig.savefig(BEESWARM_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved beeswarm PNG -> %s", BEESWARM_PNG)


def save_waterfall(explanation_pos, idx, out_path, title):
    plt.figure()
    shap.plots.waterfall(explanation_pos[idx], max_display=10, show=False)
    fig = plt.gcf()
    # Natural 4:3 aspect so bar deltas and numbers stay crisp and legible.
    fig.set_size_inches(7.0, 4.8)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved waterfall PNG -> %s", out_path)


def pick_risk_indices(risk):
    """Return (high_idx, low_idx) for representative canary-block / fast-track
    commits, falling back to the extremes if no sample clears a threshold."""
    high_candidates = np.where(risk > HIGH_RISK_THRESHOLD)[0]
    low_candidates = np.where(risk < LOW_RISK_THRESHOLD)[0]
    # Most representative: the strongest clear example in each band.
    high_idx = int(high_candidates[np.argmax(risk[high_candidates])]) \
        if high_candidates.size else int(np.argmax(risk))
    low_idx = int(low_candidates[np.argmin(risk[low_candidates])]) \
        if low_candidates.size else int(np.argmin(risk))
    return high_idx, low_idx


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    logger.info("=" * 78)
    logger.info("Module I - Dedicated TreeSHAP Attribution Pipeline")
    logger.info("=" * 78)
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    base_model = joblib.load(BASE_MODEL_PATH)
    logger.info("Loaded base LightGBM model: %s", type(base_model).__name__)

    test_df = reproduce_test_split()
    sample_df = stratified_sample(test_df, SAMPLE_SIZE)
    X_sample = sample_df[FEATURE_COLUMNS]

    risk = base_model.predict_proba(X_sample)[:, 1]
    logger.info("Sample predicted risk: min=%.4f max=%.4f mean=%.4f",
                risk.min(), risk.max(), risk.mean())

    explainer = load_or_build_explainer(base_model)
    explanation_pos, values_2d, base_1d, baseline_scalar = compute_shap(
        explainer, X_sample)
    logger.info("Baseline expected value E[f(x)] = %.6f log-odds (%.2f%% prob)",
                baseline_scalar, 100.0 * sigmoid(baseline_scalar))

    ranked_features = rank_features(values_2d, FEATURE_COLUMNS)
    write_scorecard(baseline_scalar, ranked_features)

    high_idx, low_idx = pick_risk_indices(risk)
    logger.info("High-risk exemplar: row %d (risk=%.4f) | Low-risk exemplar: row %d (risk=%.4f)",
                high_idx, risk[high_idx], low_idx, risk[low_idx])

    save_beeswarm(explanation_pos)
    save_waterfall(explanation_pos, high_idx, WATERFALL_HIGH_PNG,
                   f"High-Risk Commit (Canary Block, predicted risk {risk[high_idx]*100:.1f}%)")
    save_waterfall(explanation_pos, low_idx, WATERFALL_LOW_PNG,
                   f"Low-Risk Commit (Fast Track, predicted risk {risk[low_idx]*100:.1f}%)")

    logger.info("=" * 78)
    logger.info("SHAP attribution pipeline complete.")
    logger.info("Top 5 features by Mean |SHAP|:")
    for row in ranked_features[:5]:
        logger.info("  %d. %-45s %.6f", row["rank"], row["feature_name"],
                    row["mean_abs_shap"])
    logger.info("=" * 78)


if __name__ == "__main__":
    main()


