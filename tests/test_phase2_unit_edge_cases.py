#!/usr/bin/env python3
"""
Phase 2: Unit & Edge-Case Torture Tests for Module I Metadata Pipeline
=======================================================================
Tests PredictorService and normalize_features against extreme edge cases.
"""

import sys
import os
import time
import math
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, r"F:\Capstone\Metadata Pipeline")

from Inference_Engine.predictor_service import PredictorService, create_feature_dataframe, FEATURE_COLUMNS
from Metadata_Extractor.schema.normalizer import normalize_features
from Metadata_Extractor.schema.columns import COLUMNS


def test_schema_and_order_enforcement():
    """Test 1: Schema & Order Enforcement - Verify all 38 FEATURE_COLUMNS in exact order"""
    print("\n" + "="*70)
    print("TEST 1: Schema & Order Enforcement")
    print("="*70)

    # Verify FEATURE_COLUMNS matches expected 38 features (excluding target_binary)
    assert len(FEATURE_COLUMNS) == 38, f"Expected 38 features, got {len(FEATURE_COLUMNS)}"
    print(f"[OK] FEATURE_COLUMNS has exactly 38 features")

    # Verify COLUMNS has 39 features (38 + target_binary)
    assert len(COLUMNS) == 39, f"Expected 39 columns (with target_binary), got {len(COLUMNS)}"
    print(f"[OK] COLUMNS has exactly 39 features (including target_binary)")

    # Verify FEATURE_COLUMNS == COLUMNS[:-1] (target_binary is last)
    assert FEATURE_COLUMNS == COLUMNS[:-1], "FEATURE_COLUMNS should match COLUMNS without target_binary"
    print(f"[OK] FEATURE_COLUMNS matches COLUMNS[:-1] (target_binary excluded)")

    # Test that create_feature_dataframe filters out target_binary when passed 39 columns
    test_dict_39 = {col: 1.0 for col in COLUMNS}  # All 39 columns
    df = create_feature_dataframe(test_dict_39)
    assert df.shape == (1, 38), f"Expected (1, 38), got {df.shape}"
    assert list(df.columns) == FEATURE_COLUMNS, "Columns order mismatch after filtering"
    print(f"[OK] create_feature_dataframe correctly filters 39->38 columns and preserves order")

    # Test PredictorService validation rejects wrong column count/order
    predictor = PredictorService()
    wrong_order_df = pd.DataFrame([[1]*38], columns=reversed(FEATURE_COLUMNS))
    try:
        predictor.predict_commit_risk(wrong_order_df)
        assert False, "Should have raised ValueError for wrong column order"
    except ValueError as e:
        assert "Feature columns mismatch" in str(e)
        print(f"[OK] PredictorService rejects wrong column order")

    wrong_count_df = pd.DataFrame([[1]*37], columns=FEATURE_COLUMNS[:-1])
    try:
        predictor.predict_commit_risk(wrong_count_df)
        assert False, "Should have raised ValueError for wrong column count"
    except ValueError as e:
        assert "Feature columns mismatch" in str(e)
        print(f"[OK] PredictorService rejects wrong column count")

    multirow_df = pd.DataFrame([[1]*38, [2]*38], columns=FEATURE_COLUMNS)
    try:
        predictor.predict_commit_risk(multirow_df)
        assert False, "Should have raised ValueError for multi-row"
    except ValueError as e:
        assert "Expected single-row" in str(e)
        print(f"[OK] PredictorService rejects multi-row DataFrame")

    print("[OK] TEST 1 PASSED: Schema & Order Enforcement")


def test_mathematical_boundary_conditions():
    """Test 2: Mathematical Boundary Conditions"""
    print("\n" + "="*70)
    print("TEST 2: Mathematical Boundary Conditions")
    print("="*70)

    predictor = PredictorService()

    # 2a: Zero-division safety - total_files_changed = 0
    print("\n2a: Zero total_files_changed")
    features = {col: 0 for col in FEATURE_COLUMNS}
    features.update({
        "total_files_changed": 0,
        "git_diff_src_churn": 0,
        "git_diff_test_churn": 0,
        "code_churn_density": 0.0,
        "test_to_src_churn_ratio": 0.0,
        "gh_repo_num_commits": 1,
        "gh_repo_age": 0,
        "gh_sloc": 100,
        "log_team_size": 0.0,
        "log_built_commits": 0.0,
        "tr_build_number": 1,
    })
    df = create_feature_dataframe(features)
    result = predictor.predict_commit_risk(df)
    assert not math.isnan(result["failure_probability"]), "NaN probability with zero files"
    assert 0 <= result["failure_probability"] <= 1, "Probability out of bounds"
    print(f"  [OK] Zero files handled: prob={result['failure_probability']:.4f}, latency={result['latency_ms']:.3f}ms")

    # 2b: Brand-new repo fallback - has_previous_build = 0, missing CI history
    print("\n2b: Brand-new repo (no previous build, no CI history)")
    features = {col: 0 for col in FEATURE_COLUMNS}
    features.update({
        "has_previous_build": 0,
        "git_prev_commit_resolution_status_no_previous_build": 1,
        "git_prev_commit_resolution_status_merge_found": 0,
        "tr_build_number": 1,
        "gh_repo_num_commits": 1,
        "gh_repo_age": 0,
        "log_built_commits": 0.0,
        "gh_sloc": 500,
        "total_files_changed": 1,
        "git_diff_src_churn": 50,
        "log_team_size": 0.0,
    })
    df = create_feature_dataframe(features)
    result = predictor.predict_commit_risk(df)
    assert not math.isnan(result["failure_probability"]), "NaN probability for new repo"
    print(f"  [OK] Brand-new repo handled: prob={result['failure_probability']:.4f}")

    # 2c: Massive monolithic PR - 500+ files, 50,000 lines churn
    print("\n2c: Massive monolithic PR (500 files, 50k lines)")
    features = {col: 0 for col in FEATURE_COLUMNS}
    features.update({
        "total_files_changed": 523,
        "git_diff_src_churn": 35000,
        "git_diff_test_churn": 15000,
        "code_churn_density": 95.6,
        "test_to_src_churn_ratio": 0.43,
        "gh_diff_files_added": 50,
        "gh_diff_files_deleted": 12,
        "gh_diff_files_modified": 461,
        "gh_diff_src_files": 400,
        "gh_diff_tests_added": 8000,
        "gh_diff_tests_deleted": 2000,
        "gh_diff_doc_files": 20,
        "gh_diff_other_files": 41,
        "gh_num_commits_on_files_touched": 1200,
        "gh_repo_num_commits": 5000,
        "gh_repo_age": 1000,
        "gh_sloc": 500000,
        "log_built_commits": math.log1p(5000),
        "log_team_size": math.log1p(25),
        "tr_build_number": 1500,
        "has_previous_build": 1,
        "git_prev_commit_resolution_status_merge_found": 1,
        "git_prev_commit_resolution_status_no_previous_build": 0,
    })
    df = create_feature_dataframe(features)
    result = predictor.predict_commit_risk(df)
    assert not math.isnan(result["failure_probability"]), "NaN probability for massive PR"
    assert 0 <= result["failure_probability"] <= 1, "Probability out of bounds for massive PR"
    print(f"  [OK] Massive PR handled: prob={result['failure_probability']:.4f}, latency={result['latency_ms']:.3f}ms")

    # 2d: Extreme feature values
    print("\n2d: Extreme feature values (max realistic bounds)")
    features = {col: 0 for col in FEATURE_COLUMNS}
    features.update({
        "gh_repo_num_commits": 100000,
        "gh_repo_age": 5000,
        "gh_sloc": 10000000,
        "tr_build_number": 100000,
        "total_files_changed": 1000,
        "git_diff_src_churn": 100000,
        "git_diff_test_churn": 50000,
        "gh_num_commits_on_files_touched": 50000,
        "log_built_commits": math.log1p(100000),
        "log_team_size": math.log1p(100),
    })
    df = create_feature_dataframe(features)
    result = predictor.predict_commit_risk(df)
    assert not math.isnan(result["failure_probability"]), "NaN probability for extreme values"
    print(f"  [OK] Extreme values handled: prob={result['failure_probability']:.4f}")

    print("\n[OK] TEST 2 PASSED: Mathematical Boundary Conditions")


def test_gating_tiers_and_calibration():
    """Test 3: Gating Tiers & Calibration"""
    print("\n" + "="*70)
    print("TEST 3: Gating Tiers & Calibration")
    print("="*70)

    predictor = PredictorService()
    thresholds = {
        "FAST_TRACK": 0.25,
        "STANDARD_GATE": 0.50,
        "ELEVATED_RISK": 0.65,
    }

    # We need to find feature vectors that produce scores in each tier
    # Since we can't easily control the model output, we'll test the threshold logic directly
    from Inference_Engine.predictor_service import THRESHOLDS

    # Test the _get_gate_verdict method directly
    test_cases = [
        (0.10, "FAST-TRACK"),
        (0.24, "FAST-TRACK"),
        (0.25, "STANDARD GATE"),
        (0.35, "STANDARD GATE"),
        (0.49, "STANDARD GATE"),
        (0.50, "ALERT"),
        (0.55, "ALERT"),
        (0.65, "ALERT"),
        (0.66, "PRE-MERGE BLOCK"),
        (0.90, "PRE-MERGE BLOCK"),
        (0.99, "PRE-MERGE BLOCK"),
    ]

    for score, expected_verdict in test_cases:
        verdict, desc = predictor._get_gate_verdict(score)
        assert verdict == expected_verdict, f"Score {score}: expected {expected_verdict}, got {verdict}"
        print(f"  [OK] Score {score:.2f} -> {verdict}")

    # Verify threshold constants match
    assert THRESHOLDS == thresholds, "Threshold constants mismatch"
    print(f"  [OK] Threshold constants verified")

    # Test that actual predictions fall into valid tiers
    print("\n  Testing actual predictions fall into valid tiers...")
    for _ in range(10):
        features = {col: np.random.uniform(0, 100) for col in FEATURE_COLUMNS}
        features["gh_is_pr"] = np.random.randint(0, 2)
        features["gh_by_core_team_member"] = np.random.randint(0, 2)
        features["has_previous_build"] = np.random.randint(0, 2)
        features["git_merged_with_present"] = np.random.randint(0, 2)
        df = create_feature_dataframe(features)
        result = predictor.predict_commit_risk(df)
        prob = result["failure_probability"]
        verdict = result["verdict"]
        assert verdict in ["FAST-TRACK", "STANDARD GATE", "ALERT", "PRE-MERGE BLOCK"], f"Invalid verdict: {verdict}"
        assert 0 <= prob <= 1, f"Probability out of range: {prob}"

    print(f"  [OK] 10 random predictions all produced valid tiers and probabilities")

    print("\n[OK] TEST 3 PASSED: Gating Tiers & Calibration")


def test_latency_and_shap():
    """Test 4: Latency & SHAP Validation"""
    print("\n" + "="*70)
    print("TEST 4: Latency & SHAP Validation")
    print("="*70)

    predictor = PredictorService()

    # Warm-up run
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    features.update({
        "gh_repo_num_commits": 100,
        "gh_repo_age": 365,
        "gh_sloc": 10000,
        "total_files_changed": 10,
        "git_diff_src_churn": 500,
        "git_diff_test_churn": 200,
        "log_built_commits": math.log1p(100),
        "log_team_size": math.log1p(5),
        "tr_build_number": 50,
    })
    df = create_feature_dataframe(features)
    _ = predictor.predict_commit_risk(df)  # Warm up

    # Measure latency over multiple runs
    latencies = []
    for i in range(20):
        df = create_feature_dataframe(features)
        result = predictor.predict_commit_risk(df)
        latencies.append(result["latency_ms"])

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    max_latency = np.max(latencies)

    print(f"  Latencies over 20 runs: avg={avg_latency:.3f}ms, p95={p95_latency:.3f}ms, max={max_latency:.3f}ms")
    assert avg_latency < 15, f"Average latency {avg_latency:.3f}ms exceeds 15ms threshold"
    assert p95_latency < 15, f"P95 latency {p95_latency:.3f}ms exceeds 15ms threshold"
    print(f"  [OK] Latency under 15ms (avg={avg_latency:.3f}ms, p95={p95_latency:.3f}ms)")

    # Test SHAP drivers
    print("\n  Testing SHAP top drivers...")
    df = create_feature_dataframe(features)
    result = predictor.predict_commit_risk(df)
    shap = result["shap_drivers"]

    # Verify structure
    assert "positive" in shap and "negative" in shap, "SHAP drivers missing keys"
    assert isinstance(shap["positive"], list), "Positive drivers not a list"
    assert isinstance(shap["negative"], list), "Negative drivers not a list"

    # Verify no NaN values
    for name, val in shap["positive"] + shap["negative"]:
        assert not math.isnan(val), f"NaN SHAP value for {name}"
        assert not math.isinf(val), f"Infinite SHAP value for {name}"
        assert name in FEATURE_COLUMNS, f"SHAP feature {name} not in FEATURE_COLUMNS"

    # Verify top-k limit (3)
    assert len(shap["positive"]) <= 3, f"Too many positive drivers: {len(shap['positive'])}"
    assert len(shap["negative"]) <= 3, f"Too many negative drivers: {len(shap['negative'])}"

    # Verify ordering (descending for positive, ascending for negative)
    pos_vals = [v for _, v in shap["positive"]]
    neg_vals = [v for _, v in shap["negative"]]
    assert pos_vals == sorted(pos_vals, reverse=True), "Positive drivers not sorted descending"
    assert neg_vals == sorted(neg_vals), "Negative drivers not sorted ascending"

    print(f"  [OK] SHAP drivers: {len(shap['positive'])} positive, {len(shap['negative'])} negative, all valid, correctly ordered")
    print(f"    Positive: {[(n, f'{v:+.4f}') for n, v in shap['positive']]}")
    print(f"    Negative: {[(n, f'{v:+.4f}') for n, v in shap['negative']]}")

    print("\n[OK] TEST 4 PASSED: Latency & SHAP Validation")


def test_normalize_features_edge_cases():
    """Test 5: normalize_features edge cases"""
    print("\n" + "="*70)
    print("TEST 5: normalize_features Edge Cases")
    print("="*70)

    # Test with empty dict
    df = normalize_features({})
    assert df.shape == (1, 39), f"Empty dict: expected (1, 39), got {df.shape}"
    assert list(df.columns) == COLUMNS, "Columns mismatch with empty dict"
    print("  [OK] Empty dict handled correctly")

    # Test with None values
    test_dict = {col: None for col in COLUMNS}
    df = normalize_features(test_dict)
    assert df.shape == (1, 39), f"None values: expected (1, 39), got {df.shape}"
    # Binary cols should be 0, numeric should be 0, string should be ""
    for col in COLUMNS:
        val = df[col].iloc[0]
        if col in {"gh_is_pr", "gh_by_core_team_member", "target_binary", "git_merged_with_present", "has_previous_build"}:
            assert val == 0, f"Binary col {col} should be 0, got {val}"
        elif any(tok in col for tok in ("num", "churn", "files", "sloc", "age", "duration", "year", "month", "density", "kloc", "log", "ratio", "dayofweek", "build", "commits")):
            assert val == 0, f"Numeric col {col} should be 0, got {val}"
        elif col.startswith("gh_lang_") or col.startswith("git_prev_commit_resolution_status_"):
            assert val == 0, f"Language/prev_build col {col} should be 0, got {val}"
        else:
            assert val == "", f"String col {col} should be '', got '{val}'"
    print("  [OK] None values correctly defaulted by type")

    # Test with partial dict (only some keys provided)
    partial = {
        "gh_is_pr": 1,
        "gh_lang_python": 1,
        "gh_num_issue_comments": 5,
        "total_files_changed": 10,
        "git_diff_src_churn": 100,
        "has_previous_build": 1,
    }
    df = normalize_features(partial)
    assert df.shape == (1, 39), f"Partial dict: expected (1, 39), got {df.shape}"
    assert df["gh_is_pr"].iloc[0] == 1, "Provided value not preserved"
    assert df["gh_lang_python"].iloc[0] == 1, "Provided value not preserved"
    assert df["gh_lang_java"].iloc[0] == 0, "Unprovided lang should be 0"
    print("  [OK] Partial dict preserves provided values, defaults others")

    # Test that target_binary is included in output
    assert "target_binary" in df.columns, "target_binary missing from output"
    assert df["target_binary"].iloc[0] == 0, "target_binary should default to 0"
    print("  [OK] target_binary included and defaults to 0")

    print("\n[OK] TEST 5 PASSED: normalize_features Edge Cases")


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "#"*70)
    print("# PHASE 2: UNIT & EDGE-CASE TORTURE TESTS")
    print("#"*70)

    test_schema_and_order_enforcement()
    test_mathematical_boundary_conditions()
    test_gating_tiers_and_calibration()
    test_latency_and_shap()
    test_normalize_features_edge_cases()

    print("\n" + "#"*70)
    print("# ALL PHASE 2 TESTS PASSED [OK]")
    print("#"*70)


if __name__ == "__main__":
    run_all_tests()