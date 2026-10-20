#!/usr/bin/env python3
"""
Run Pipeline - Module I (Pre-Build Commit Metadata Risk Predictor)
===================================================================
Main entrypoint with interactive CLI for both operation modes:
1. Webhook Listener Mode (FastAPI server with automated ngrok + GitHub webhook registration)
2. Passive Poller Mode (polling loop with ETag caching for external/3rd-party repos)
"""

import os
import sys
import time
import signal
import argparse
import tempfile
import subprocess
import shutil
import stat
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime


def _handle_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree to handle Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from Inference_Engine.predictor_service import PredictorService, create_feature_dataframe
from Inference_Engine.state_manager import StateManager
import Inference_Engine.cli_server as cli_server_module

# Import Metadata_Extractor components
from Metadata_Extractor.main import (
    parse_repo_url,
    get_target_commits,
    run_git_capture,
    is_dirty,
    current_ref,
)
from Metadata_Extractor.core.git_engine import get_churn_metrics, get_repo_git_metrics
from Metadata_Extractor.core.static_analyzer import analyze_workspace
from Metadata_Extractor.core.github_api import get_github_metadata
from Metadata_Extractor.core.ci_analyzer import get_ci_metrics
from Metadata_Extractor.schema.normalizer import normalize_features


# Global state for graceful shutdown
shutdown_requested = False
poller_stats = {"checks": 0, "new_commits": 0, "predictions": 0, "start_time": None}
_shutdown_event = None  # Will be initialized in main()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n\n[SHUTDOWN] Ctrl+C received. Shutting down gracefully...")
    shutdown_requested = True
    if _shutdown_event:
        _shutdown_event.set()


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CONFIG] Loaded environment from {env_path}")
    else:
        print(f"[CONFIG] No .env file found at {env_path}, using system environment")

    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not github_token:
        print("[ERROR] GITHUB_TOKEN not found in environment!")
        print("        Please create a .env file with GITHUB_TOKEN=your_pat")
        print("        Or export GITHUB_TOKEN in your shell")
        sys.exit(1)

    # Also load ngrok auth token
    ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_auth_token:
        print(f"[CONFIG] Ngrok auth token loaded")
    else:
        print(f"[CONFIG] No NGROK_AUTHTOKEN found - webhook automation will use unauthenticated tunnel")

    return github_token, ngrok_auth_token


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("  MODULE I - PRE-BUILD COMMIT METADATA RISK PREDICTOR")
    print("  AI-Powered CI/CD Failure Predictor | Production Pipeline")
    print("=" * 70)


def print_model_load_info(predictor: PredictorService):
    """Print model loading confirmation."""
    info = predictor.get_model_info()
    print(f"\n[MODEL] Calibrated model loaded: {info['calibrated_model_type']}")
    print(f"[MODEL] Base LightGBM loaded: {info['base_model_type']}")
    print(f"[MODEL] SHAP explainer loaded: TreeExplainer")
    print(f"[MODEL] Feature schema validated: {info['feature_count']} features in exact order")
    print(f"[MODEL] Ready for inference (target latency: ~3.4ms)")


def prompt_repo_url() -> tuple:
    """Prompt for GitHub repository URL and parse owner/repo."""
    while True:
        url = input("\nEnter GitHub Repository URL (e.g., https://github.com/owner/repo or owner/repo): ").strip()
        if not url:
            print("  [ERROR] Repository URL cannot be empty. Please try again.")
            continue
        try:
            owner, repo = parse_repo_url(url)
            print(f"  [OK] Parsed: owner={owner}, repo={repo}")
            return owner, repo
        except Exception as e:
            print(f"  [ERROR] Invalid URL format: {e}. Please try again.")


def prompt_operation_mode() -> int:
    """Prompt for operation mode selection."""
    print("\nSelect Operation Mode:")
    print("  [1] Webhook Listener Mode (For Owned Repos - Automated ngrok + Webhook Registration)")
    print("  [2] Passive Poller Mode (For External/3rd-Party Repos - ETag-Cached Radar Loop)")

    while True:
        choice = input("Enter choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("  [ERROR] Invalid choice. Please enter 1 or 2.")


def prompt_webhook_port() -> int:
    """Prompt for webhook server port."""
    while True:
        port_str = input("\nEnter local port for webhook server (default 5000): ").strip()
        if not port_str:
            return 5000
        try:
            port = int(port_str)
            if 1 <= port <= 65535:
                return port
            print("  [ERROR] Port must be between 1 and 65535.")
        except ValueError:
            print("  [ERROR] Invalid port number.")


def prompt_polling_interval() -> int:
    """Prompt for polling interval in seconds."""
    while True:
        interval_str = input("\nEnter polling interval in seconds (default 30, minimum 5): ").strip()
        if not interval_str:
            return 30
        try:
            interval = int(interval_str)
            if interval >= 5:
                return interval
            print("  [ERROR] Minimum polling interval is 5 seconds.")
        except ValueError:
            print("  [ERROR] Invalid number.")


def extract_features_for_commit(owner: str, repo: str, sha: str, repo_path: str) -> dict:
    """Extract all 38 features for a given commit using the Metadata_Extractor pipeline."""
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
    print(f"\n[EXTRACT] Cloning {owner}/{repo} @ {sha[:7]}...")

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
        print("[EXTRACT] Computing git churn metrics...")
        features_dict = extract_features_for_commit(owner, repo, sha, temp_dir)

        # Create feature DataFrame and predict
        print("[TRANSFORM] Validating 38-feature schema and scaling contracts...")
        feature_df = create_feature_dataframe(features_dict)

        print("[INFERENCE] Running model prediction...")
        result = predictor.predict_commit_risk(feature_df)

        # Add metadata to result
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
            shutil.rmtree(temp_dir, onerror=_handle_remove_readonly)


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

    print(f"\n[TRANSFORM] All 38 features validated ✓ | Scaling contracts passed ✓")

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

    # Color coding for verdict
    if verdict == "FAST-TRACK":
        color = "\033[92m"  # Green
    elif verdict == "STANDARD GATE":
        color = "\033[93m"  # Yellow
    elif verdict == "ALERT":
        color = "\033[91m"  # Red
    else:
        color = "\033[95m"  # Magenta
    reset = "\033[0m"

    print(f"\n[VERDICT] {color}{verdict}{reset} — {desc}")
    print("=" * 70 + "\n")


def print_heartbeat(owner: str, repo: str, short_sha: str):
    """Print silent heartbeat (just a dot) for poller mode."""
    # Silent heartbeat - just a single dot on the same line
    print(".", end="", flush=True)


def print_shutdown_summary():
    """Print shutdown summary for poller mode."""
    if poller_stats["start_time"]:
        elapsed = time.time() - poller_stats["start_time"]
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        elapsed_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
    else:
        elapsed_str = "unknown"

    print("\n" + "=" * 70)
    print("  POLLER SHUTDOWN SUMMARY")
    print("=" * 70)
    print(f"  Total runtime: {elapsed_str}")
    print(f"  Total checks: {poller_stats['checks']}")
    print(f"  New commits detected: {poller_stats['new_commits']}")
    print(f"  Predictions made: {poller_stats['predictions']}")
    print("=" * 70 + "\n")


def run_webhook_mode(predictor: PredictorService, github_token: str, owner: str, repo: str, port: int, ngrok_auth_token: Optional[str]):
    """Run Mode 1: Webhook Listener Mode with automated ngrok and webhook registration."""

    # Configure server
    cli_server_module.config.update({
        "owner": owner,
        "repo": repo,
        "github_token": github_token,
        "ngrok_auth_token": ngrok_auth_token,
    })

    # Initialize state manager for webhook mode too
    cli_server_module.state_manager = StateManager(github_token, owner, repo)
    cli_server_module.predictor = predictor

    # Ask if user wants baseline prediction on latest commit
    print("\n" + "-" * 50)
    baseline = input("Run baseline prediction on latest commit now? [y/N]: ").strip().lower()
    if baseline in ("y", "yes"):
        print("[BASELINE] Fetching latest commit...")
        state_mgr = StateManager(github_token, owner, repo)
        commit_data, _ = state_mgr.fetch_latest_commit()
        if commit_data:
            print(f"[BASELINE] Found latest commit: {commit_data['short_sha']} - {commit_data['message'][:60]}")
            result = run_extraction_and_prediction(predictor, owner, repo, commit_data["sha"], "BASELINE", "main")
            print_telemetry(result)
        else:
            print("[BASELINE] Could not fetch latest commit.")

    print("\n[MODE 1] Starting automated webhook server with ngrok tunneling...")
    print("[MODE 1] This will automatically:")
    print("         1. Authenticate ngrok (if token provided)")
    print("         2. Create a secure public tunnel")
    print("         3. Register webhook with GitHub (if token has admin rights)")
    print("         4. Start the FastAPI server")
    print("         5. Clean up everything on Ctrl+C\n")

    # Run the automated server (blocks until shutdown)
    cli_server_module.run_server_with_automation(host="0.0.0.0", port=port)


def run_poller_mode(predictor: PredictorService, github_token: str, owner: str, repo: str, interval: int):
    """Run Mode 2: Passive Poller Mode with ETag caching."""
    global shutdown_requested, poller_stats, _shutdown_event

    print("\n" + "=" * 70)
    print("  PASSIVE POLLER MODE STARTING")
    print("=" * 70)
    print(f"  Target: {owner}/{repo}")
    print(f"  Polling interval: {interval} seconds")
    print(f"  Rate-limit protection: ETag/If-None-Match caching enabled")
    print(f"  Press Ctrl+C to stop gracefully")
    print("=" * 70 + "\n")

    # Initialize state manager
    state_mgr = StateManager(github_token, owner, repo)

    # Initial baseline fetch
    print("[INIT] Fetching initial baseline commit...")
    commit_data, is_new = state_mgr.fetch_latest_commit()

    if commit_data:
        state_mgr.set_last_seen_sha(commit_data["sha"])
        print(f"[INIT] Baseline commit: {commit_data['short_sha']} - {commit_data['message'][:60]}")
        print(f"[INIT] Running initial prediction...")
        result = run_extraction_and_prediction(predictor, owner, repo, commit_data["sha"], "BASELINE", "main")
        print_telemetry(result)
        poller_stats["predictions"] += 1
    else:
        print("[INIT] WARNING: Could not fetch initial commit. Will retry on first poll.")

    poller_stats["start_time"] = time.time()

    # Main polling loop - wrapped in try/except for KeyboardInterrupt safety
    first_heartbeat = True
    try:
        while not shutdown_requested:
            poller_stats["checks"] += 1

            commit_data, is_new = state_mgr.fetch_latest_commit()

            if commit_data:
                if is_new:
                    poller_stats["new_commits"] += 1
                    print(f"\n[NEW COMMIT] Detected: {commit_data['short_sha']} - {commit_data['message'][:60]}")
                    result = run_extraction_and_prediction(
                        predictor, owner, repo, commit_data["sha"],
                        "POLL", "main"
                    )
                    print_telemetry(result)
                    poller_stats["predictions"] += 1
                    first_heartbeat = True  # Reset for next heartbeat series
                else:
                    # Silent heartbeat - just a dot
                    if first_heartbeat:
                        print(f"[POLLER] Monitoring {owner}/{repo} (HEAD: {commit_data['short_sha']}) ", end="", flush=True)
                        first_heartbeat = False
                    print_heartbeat(owner, repo, commit_data["short_sha"])
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                if first_heartbeat:
                    print(f"[POLLER] Monitoring {owner}/{repo} (no commits found) ", end="", flush=True)
                    first_heartbeat = False
                print_heartbeat(owner, repo, "unknown")

            # Sleep with periodic shutdown check using threading.Event for responsiveness
            # This allows immediate wake-up when signal is received
            if _shutdown_event is not None:
                # Wait for interval seconds or until shutdown event is set
                if _shutdown_event.wait(timeout=interval):
                    break  # Shutdown requested
            else:
                # Fallback to original sleep logic
                slept = 0
                while slept < interval and not shutdown_requested:
                    time.sleep(min(1, interval - slept))
                    slept += 1

    except KeyboardInterrupt:
        # This should not normally be reached since we have signal handler,
        # but catch it just in case for absolute safety
        print("\n\n[SHUTDOWN] KeyboardInterrupt caught. Shutting down gracefully...")
        shutdown_requested = True

    finally:
        # Print newline after heartbeat dots
        if not first_heartbeat:
            print()  # New line after dots

        print_shutdown_summary()

        # Ensure clean exit
        sys.exit(0)


def main():
    """Main entrypoint."""
    # Initialize shutdown event for responsive polling loop interruption
    global _shutdown_event
    _shutdown_event = threading.Event()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_banner()

    # Load environment
    github_token, ngrok_auth_token = load_environment()

    # Initialize predictor service
    print("\n[INIT] Initializing PredictorService...")
    predictor = PredictorService()
    print_model_load_info(predictor)

    # Interactive setup
    owner, repo = prompt_repo_url()
    mode = prompt_operation_mode()

    if mode == 1:
        # Webhook Listener Mode
        port = prompt_webhook_port()
        run_webhook_mode(predictor, github_token, owner, repo, port, ngrok_auth_token)
    else:
        # Passive Poller Mode
        interval = prompt_polling_interval()
        run_poller_mode(predictor, github_token, owner, repo, interval)


if __name__ == "__main__":
    main()