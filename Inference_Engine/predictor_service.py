"""
Predictor Service - Module I (Pre-Build Commit Metadata Risk Predictor)
======================================================================
Loads calibrated LightGBM model and SHAP explainer at startup.
Provides in-memory prediction with 38-feature schema validation,
latency measurement, and top-3 SHAP feature attribution.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

# Strict 38-feature schema (must match train_metadata_model.py exactly)
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

# Production gating thresholds
THRESHOLDS = {
    "FAST_TRACK": 0.25,       # score < 0.25 -> LOW RISK
    "STANDARD_GATE": 0.50,    # 0.25 <= score < 0.50 -> MODERATE RISK
    "ELEVATED_RISK": 0.65,    # 0.50 <= score <= 0.65 -> ELEVATED RISK
    # score > 0.65 -> HIGH RISK (CANARY ISOLATION / PRE-MERGE BLOCK)
}


class PredictorService:
    """In-memory predictor service for Module I metadata risk prediction."""

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize predictor service by loading model artifacts.

        Args:
            model_dir: Path to directory containing model artifacts.
                       Defaults to Metadata Model/Models/run_20260926_013515/
        """
        if model_dir is None:
            # Default to the run directory relative to this file
            base = Path(__file__).parent.parent
            model_dir = base / "Metadata Model" / "Models" / "run_20260926_013515"

        self.model_dir = Path(model_dir)
        self.calibrated_model = None
        self.base_model = None
        self.shap_explainer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all model artifacts into RAM."""
        cal_path = self.model_dir / "calibrated_metadata_model.joblib"
        base_path = self.model_dir / "base_lightgbm_model.joblib"
        shap_path = self.model_dir / "shap_explainer.joblib"

        print(f"[MODEL] Loading calibrated model from {cal_path}")
        self.calibrated_model = joblib.load(cal_path)

        print(f"[MODEL] Loading base LightGBM model from {base_path}")
        self.base_model = joblib.load(base_path)

        print(f"[MODEL] Loading SHAP explainer from {shap_path}")
        self.shap_explainer = joblib.load(shap_path)

        # Verify feature order matches
        booster_names = list(self.base_model.booster_.feature_name())
        if booster_names != FEATURE_COLUMNS:
            raise ValueError(
                f"Model feature order mismatch!\n"
                f"Expected: {FEATURE_COLUMNS}\n"
                f"Got: {booster_names}"
            )

        print(f"[MODEL] All artifacts loaded successfully. Feature schema validated (38 features).")

    def _validate_features(self, feature_df: pd.DataFrame) -> None:
        """Validate that DataFrame has exactly the 38 required columns in correct order."""
        if list(feature_df.columns) != FEATURE_COLUMNS:
            raise ValueError(
                f"Feature columns mismatch!\n"
                f"Expected: {FEATURE_COLUMNS}\n"
                f"Got: {list(feature_df.columns)}"
            )
        if feature_df.shape[1] != 38:
            raise ValueError(f"Expected 38 features, got {feature_df.shape[1]}")
        if feature_df.shape[0] != 1:
            raise ValueError(f"Expected single-row DataFrame, got {feature_df.shape[0]} rows")

    def _get_gate_verdict(self, score: float) -> Tuple[str, str]:
        """
        Determine gating verdict based on calibrated failure probability.

        Returns:
            Tuple of (verdict_label, verdict_description)
        """
        if score < THRESHOLDS["FAST_TRACK"]:
            return ("FAST-TRACK", "LOW RISK — Proceed with standard/accelerated build")
        elif score < THRESHOLDS["STANDARD_GATE"]:
            return ("STANDARD GATE", "MODERATE RISK — Proceed with monitoring")
        elif score <= THRESHOLDS["ELEVATED_RISK"]:
            return ("ALERT", "ELEVATED RISK — Flag PR/commit for review")
        else:
            return ("PRE-MERGE BLOCK", "HIGH RISK — Canary isolation / Pre-merge block (Halt expensive runner)")

    def _get_shap_top_drivers(self, feature_df: pd.DataFrame, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute top-k positive and negative SHAP feature contributions.

        Returns:
            Dict with 'positive' and 'negative' lists of (feature_name, shap_value) tuples
        """
        # Get SHAP values for class 1 (FAIL)
        sv = self.shap_explainer.shap_values(feature_df)

        # Handle different SHAP output formats
        if isinstance(sv, list):
            arr = sv[1] if len(sv) > 1 else sv[0]
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]

        # Single row -> shape (1, 38), get first row
        shap_row = np.asarray(arr)[0]

        # Pair feature names with SHAP values
        feature_shap = list(zip(FEATURE_COLUMNS, shap_row))

        # Sort by SHAP value (positive pushes toward failure, negative pushes toward pass)
        positive = sorted(
            [(name, val) for name, val in feature_shap if val > 0],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        negative = sorted(
            [(name, val) for name, val in feature_shap if val < 0],
            key=lambda x: x[1]
        )[:top_k]

        return {
            "positive": positive,
            "negative": negative
        }

    def predict_commit_risk(self, feature_df: pd.DataFrame) -> Dict:
        """
        Run full prediction pipeline on a single commit's feature vector.

        Args:
            feature_df: Single-row DataFrame with 38 features in exact FEATURE_COLUMNS order

        Returns:
            Dict with prediction results, latency, SHAP drivers, and gate verdict
        """
        # Validate input
        self._validate_features(feature_df)

        # Measure pure model inference latency
        t0 = time.perf_counter()
        proba = self.calibrated_model.predict_proba(feature_df)[:, 1][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Calibrated failure probability
        failure_prob = float(proba)

        # Get SHAP top drivers
        shap_drivers = self._get_shap_top_drivers(feature_df)

        # Get gate verdict
        verdict_label, verdict_desc = self._get_gate_verdict(failure_prob)

        return {
            "failure_probability": failure_prob,
            "failure_probability_pct": round(failure_prob * 100, 2),
            "raw_score": failure_prob,  # For calibrated model, this IS the calibrated probability
            "latency_ms": round(latency_ms, 3),
            "shap_drivers": shap_drivers,
            "verdict": verdict_label,
            "verdict_description": verdict_desc,
            "thresholds": THRESHOLDS,
        }

    def get_model_info(self) -> Dict:
        """Return model metadata for telemetry."""
        return {
            "model_dir": str(self.model_dir),
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "calibrated_model_type": type(self.calibrated_model).__name__,
            "base_model_type": type(self.base_model).__name__,
        }


def create_feature_dataframe(features_dict: dict) -> pd.DataFrame:
    """
    Create a properly ordered single-row DataFrame from a feature dictionary.

    Args:
        features_dict: Dictionary with feature names as keys

    Returns:
        Single-row DataFrame with columns in FEATURE_COLUMNS order
    """
    row = {col: features_dict.get(col, 0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)