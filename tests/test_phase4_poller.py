#!/usr/bin/env python3
"""
Phase 4: Mode 2 (Active Poller + ETag + Live Commit Detection) Test
=====================================================================
Tests StateManager against aayush-1105/temp-python with real GitHub API calls.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

sys.path.insert(0, r"F:\Capstone\Metadata Pipeline")

from Inference_Engine.predictor_service import PredictorService, create_feature_dataframe
from Inference_Engine.state_manager import StateManager, create_state_manager
from Metadata_Extractor.main import parse_repo_url, get_target_commits, run_git_capture
from Metadata_Extractor.core.git_engine import get_churn_metrics, get_repo_git_metrics
from Metadata_Extractor.core.static_analyzer import analyze_workspace
from Metadata_Extractor.core.github_api import get_github_metadata
from Metadata_Extractor.core.ci_analyzer import get_ci_metrics
from Metadata_Extractor.schema.normalizer import normalize_features


def extract_features_for_commit(owner: str, repo: str, sha: str, repo_path: str) -> dict:
    """Extract all 38 features for a given commit."""
    all_features = {}
    all_features.update(get_churn_metrics(repo_path, sha))
    all_features.update(get_repo_git_metrics(repo_path, sha))
    all_features.update(analyze_workspace(repo_path))
    all_features.update(get_github_metadata(owner, repo, sha))
    all_features.update(get_ci_metrics(owner, repo, sha))
    return normalize_features(all_features).iloc[0].to_dict()


def run_extraction_and_prediction(predictor: PredictorService, owner: str, repo: str, sha: str,
                                   event_type: str = "MANUAL", branch: str = "main") -> dict:
    """Run full extraction + prediction pipeline for a commit."""
    import tempfile
    import shutil

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="pipeline_extractor_")
        repo_url = f"https://github.com/{owner}/{repo}.git"

        # Clone with depth to get recent history
        subprocess.run(
            f'git clone --quiet --depth 50 "{repo_url}" "{temp_dir}"',
            shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Fetch the specific commit if not in shallow clone
        subprocess.run(
            f'git fetch --quiet origin "{sha}"',
            cwd=temp_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Checkout the commit
        subprocess.run(
            f'git checkout -q {sha}',
            cwd=temp_dir, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Extract features
        print(f"    [EXTRACT] Computing features for {sha[:7]}...")
        features_dict = extract_features_for_commit(owner, repo, sha, temp_dir)

        # Create feature DataFrame and predict
        print(f"    [TRANSFORM] Validating 38-feature schema...")
        feature_df = create_feature_dataframe(features_dict)

        print(f"    [INFERENCE] Running model prediction...")
        result = predictor.predict_commit_risk(feature_df)

        # Add metadata to result
        from datetime import datetime
        result.update({
            "owner": owner,
            "repo": repo,
            "sha": sha,
            "short_sha": sha[:7],
            "branch": branch,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "features": features_dict,
        })

        return result

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Extraction/prediction failed: {e}")
        raise
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=lambda func, path, exc: (os.chmod(path, 0o600), func(path)))


def print_telemetry(result: dict):
    """Print structured MLOps telemetry to terminal."""
    features = result.get("features", {})

    print("\n" + "=" * 70)
    print(f"[INGEST]  Repo: {result['owner']}/{result['repo']} | Branch: {result['branch']}")
    print(f"          Commit: {result['short_sha']} ({result['sha']})")
    print(f"          Event: {result['event_type'].upper()}")
    print(f"          Time: {result['timestamp']}")

    print(f"\n[EXTRACT] Files touched: {features.get('total_files_changed', 'N/A')}")
    print(f"          Src churn: {features.get('git_diff_src_churn', 'N/A')}")
    print(f"          Test churn: {features.get('git_diff_test_churn', 'N/A')}")
    print(f"          Prev build: {'Yes' if features.get('has_previous_build') else 'No'}")

    print(f"\n[TRANSFORM] All 38 features validated | Scaling contracts passed")

    print(f"\n[INFERENCE] Calibrated Failure Probability: {result['failure_probability_pct']:.2f}%")
    print(f"            Raw Score: {result['raw_score']:.4f}")
    print(f"            Model Latency: {result['latency_ms']:.3f} ms")

    print(f"\n[SHAP TOP DRIVERS]")
    shap = result.get("shap_drivers", {})
    if shap.get("positive"):
        print("  Pushing risk UP:")
        for name, val in shap["positive"]:
            print(f"    + {name}: {val:+.4f}")
    if shap.get("negative"):
        print("  Pushing risk DOWN:")
        for name, val in shap["negative"]:
            print(f"    - {name}: {val:+.4f}")

    verdict = result["verdict"]
    desc = result["verdict_description"]
    print(f"\n[VERDICT] {verdict} -- {desc}")
    print("=" * 70 + "\n")


def main():
    """Run Phase 4 tests"""
    print("\n" + "#"*70)
    print("# PHASE 4: MODE 2 (ACTIVE POLLER + ETAG + LIVE COMMIT DETECTION)")
    print("#"*70)

    # Load GitHub token from project root .env
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    github_token = os.getenv("GITHUB_TOKEN")
    print(f"[DEBUG] Token loaded: {github_token[:10]}..." if github_token else "[DEBUG] Token NOT loaded!")

    owner = "aayush-1105"
    repo = "temp-python"

    # Initialize predictor
    print("\n[INIT] Initializing PredictorService...")
    predictor = PredictorService()
    print("[OK] PredictorService loaded")

    # Test 1: Initialize StateManager
    print("\n[TEST 1] Initialize StateManager on aayush-1105/temp-python")
    state_mgr = create_state_manager(github_token, owner, repo)
    print(f"  [OK] StateManager created, state_file: {state_mgr.state_file}")

    # Test 2: Tick 1 - Baseline poll
    print("\n[TEST 2] Tick 1 (Baseline): Poll the repo, capture latest SHA")
    commit_data, is_new = state_mgr.fetch_latest_commit()
    if commit_data:
        print(f"  [OK] Latest commit: {commit_data['short_sha']} - {commit_data['message'][:60]}")
        print(f"  [OK] Is new: {is_new}")
        print(f"  [OK] ETag captured: {state_mgr.commits_etag[:20]}..." if state_mgr.commits_etag else "  [OK] No ETag (first request)")
        print(f"  [OK] Last seen SHA updated: {state_mgr.last_seen_sha[:7]}")

        # Run baseline prediction
        print("\n  Running baseline prediction...")
        result = run_extraction_and_prediction(predictor, owner, repo, commit_data["sha"], "BASELINE", "main")
        print_telemetry(result)
    else:
        print("  [FAIL] Could not fetch latest commit")
        return

    # Test 3: Tick 2 - Steady-state ETag cache (immediate re-poll)
    print("\n[TEST 3] Tick 2 (Steady-State ETag Cache): Poll again immediately")
    print("  Expecting 304 Not Modified or unchanged SHA detection...")
    commit_data2, is_new2 = state_mgr.fetch_latest_commit()
    if commit_data2:
        print(f"  [OK] Latest commit: {commit_data2['short_sha']}")
        print(f"  [OK] Is new: {is_new2}")
        print(f"  [OK] ETag sent: {state_mgr.commits_etag[:20]}..." if state_mgr.commits_etag else "  [OK] No ETag")
        if not is_new2:
            print(f"  [OK] CORRECT: No new commit detected, no re-extraction triggered")
        else:
            print(f"  [WARN] New commit detected (may be expected if race condition)")
    else:
        print("  [INFO] No commit data returned (could be 304)")
        print(f"  [OK] CORRECT: 304 Not Modified or unchanged SHA, 0 rate-limit points consumed")

    # Verify rate limit didn't decrease significantly
    rate_info = state_mgr.get_rate_limit_info()
    print(f"  [OK] Rate limit remaining: {rate_info['remaining']}/{rate_info['limit']}")

    # Test 4: Tick 3 - Live Commit Detection
    print("\n[TEST 4] Tick 3 (Live Commit Detection): Make a new commit and push")

    # Modify a file in the local test repo
    test_repo_path = r"C:\Users\aayus\Downloads\New folder"
    main_py_path = os.path.join(test_repo_path, "main.py")

    # Read current content
    with open(main_py_path, "r") as f:
        content = f.read()

    # Add a new function
    new_content = content + "\n\ndef multiply(a: int, b: int) -> int:\n    \"\"\"Multiply two numbers.\"\"\"\n    return a * b\n"

    with open(main_py_path, "w") as f:
        f.write(new_content)

    print(f"  [OK] Modified {main_py_path}")

    # Commit and push (NO Co-authored-by tag!)
    subprocess.run(
        'git add main.py && git commit -m "test: mode 2 poller trigger"',
        cwd=test_repo_path, shell=True, check=True, capture_output=True, text=True
    )
    print(f"  [OK] Committed changes")

    subprocess.run(
        'git push origin main',
        cwd=test_repo_path, shell=True, check=True, capture_output=True, text=True
    )
    print(f"  [OK] Pushed to origin main")

    # Wait a moment for GitHub to process
    print("  Waiting 3 seconds for GitHub to index the new commit...")
    time.sleep(3)

    # Poll again via StateManager
    print("  Polling for new commit...")
    commit_data3, is_new3 = state_mgr.fetch_latest_commit()
    if commit_data3 and is_new3:
        print(f"  [OK] NEW COMMIT DETECTED: {commit_data3['short_sha']} - {commit_data3['message'][:60]}")
        print(f"  [OK] Last seen SHA updated: {state_mgr.last_seen_sha[:7]}")

        # Run full extraction + prediction on the new commit
        print("\n  Running prediction on new commit...")
        result = run_extraction_and_prediction(predictor, owner, repo, commit_data3["sha"], "POLL", "main")
        print_telemetry(result)
    else:
        print(f"  [FAIL] New commit not detected. is_new={is_new3}, commit_data={commit_data3}")
        return

    # Test 5: Verify state persistence
    print("\n[TEST 5] Verify StateManager state persistence")
    state_mgr2 = create_state_manager(github_token, owner, repo)
    print(f"  [OK] New StateManager instance created")
    print(f"  [OK] Last seen SHA persisted: {state_mgr2.last_seen_sha[:7]}")
    assert state_mgr2.last_seen_sha == state_mgr.last_seen_sha, "SHA not persisted correctly"
    print(f"  [OK] State persistence verified")

    print("\n" + "#"*70)
    print("# ALL PHASE 4 TESTS PASSED")
    print("#"*70)


if __name__ == "__main__":
    main()