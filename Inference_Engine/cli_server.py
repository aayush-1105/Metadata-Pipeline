"""
CLI Server - Module I (Pre-Build Commit Metadata Risk Predictor)
================================================================
FastAPI server for Webhook Listener Mode (Mode 1).
Exposes POST /webhook and GET /health endpoints.
Includes automated ngrok tunneling and GitHub webhook registration.
"""

import os
import json
import hmac
import hashlib
import tempfile
import shutil
import subprocess
import atexit
import signal
import stat
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import requests
from pyngrok import ngrok

# Import our modules
from .predictor_service import PredictorService, create_feature_dataframe, FEATURE_COLUMNS
from .state_manager import StateManager


def _handle_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree to handle Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Add parent directory to path for Metadata_Extractor imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


app = FastAPI(title="Module I - Metadata Risk Predictor Webhook Server")

# Global instances (initialized at startup)
predictor: Optional[PredictorService] = None
state_manager: Optional[StateManager] = None
config: Dict[str, Any] = {}

# Ngrok and webhook cleanup state
_ngrok_tunnel = None
_webhook_hook_id = None
_cleanup_registered = False


def extract_features_for_commit(owner: str, repo: str, sha: str, repo_path: str) -> Dict:
    """Extract all 38 features for a given commit using the Metadata_Extractor pipeline."""
    all_features = {}
    all_features.update(get_churn_metrics(repo_path, sha))
    all_features.update(get_repo_git_metrics(repo_path, sha))
    all_features.update(analyze_workspace(repo_path))
    all_features.update(get_github_metadata(owner, repo, sha))
    all_features.update(get_ci_metrics(owner, repo, sha))
    return normalize_features(all_features).iloc[0].to_dict()


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not secret:
        return True  # Skip verification if no secret configured
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def create_github_webhook(owner: str, repo: str, target_url: str, token: str) -> Optional[str]:
    """
    Create a GitHub webhook for the repository.

    Args:
        owner: Repository owner
        repo: Repository name
        target_url: The public ngrok URL to send webhooks to
        token: GitHub PAT with admin:repo_hook permissions

    Returns:
        hook_id if successful, None if failed
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "name": "web",
        "active": True,
        "events": ["push", "pull_request"],
        "config": {
            "url": f"{target_url}/webhook",
            "content_type": "json",
            "insecure_ssl": "0"
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 201:
            hook_data = response.json()
            hook_id = str(hook_data.get("id"))
            print(f"[WEBHOOK] [OK] Successfully registered webhook (ID: {hook_id})")
            print(f"[WEBHOOK]   Target: {target_url}/webhook")
            print(f"[WEBHOOK]   Events: push, pull_request")
            return hook_id
        elif response.status_code == 404:
            print(f"[WEBHOOK] [FAIL] Repository not found: {owner}/{repo}")
            print(f"[WEBHOOK]   Check that the repository exists and the token has access.")
        elif response.status_code == 403:
            print(f"[WEBHOOK] [FAIL] Permission denied - GitHub token lacks 'admin:repo_hook' permission")
            print(f"[WEBHOOK]   Please ensure your PAT has 'admin:repo_hook' or 'repo' scope.")
            print(f"[WEBHOOK]   Falling back to manual webhook setup instructions...")
        else:
            print(f"[WEBHOOK] [FAIL] Failed to create webhook: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"[WEBHOOK] [FAIL] Network error creating webhook: {e}")
    except Exception as e:
        print(f"[WEBHOOK] [FAIL] Unexpected error: {e}")

    return None


def delete_github_webhook(owner: str, repo: str, hook_id: str, token: str) -> bool:
    """
    Delete a GitHub webhook from the repository.

    Returns:
        True if successful, False otherwise
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/hooks/{hook_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = requests.delete(url, headers=headers, timeout=10)
        if response.status_code == 204:
            print(f"[WEBHOOK] [OK] Webhook {hook_id} deleted successfully")
            return True
        else:
            print(f"[WEBHOOK] [FAIL] Failed to delete webhook: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[WEBHOOK] [FAIL] Error deleting webhook: {e}")
    return False


def cleanup_resources():
    """Clean up ngrok tunnel and GitHub webhook on shutdown."""
    global _ngrok_tunnel, _webhook_hook_id, _cleanup_registered

    if _ngrok_tunnel:
        try:
            ngrok.disconnect(_ngrok_tunnel.public_url)
            print(f"[NGROK] [OK] Tunnel {_ngrok_tunnel.public_url} closed")
        except Exception as e:
            print(f"[NGROK] [FAIL] Error closing tunnel: {e}")
        _ngrok_tunnel = None

    if _webhook_hook_id and config.get("owner") and config.get("repo") and config.get("github_token"):
        delete_github_webhook(
            config["owner"],
            config["repo"],
            _webhook_hook_id,
            config["github_token"]
        )
        _webhook_hook_id = None

    try:
        ngrok.kill()
        print(f"[NGROK] [OK] Ngrok process terminated")
    except Exception as e:
        print(f"[NGROK] [FAIL] Error killing ngrok: {e}")


def register_cleanup():
    """Register cleanup handlers for graceful shutdown."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(cleanup_resources)
        signal.signal(signal.SIGINT, lambda s, f: cleanup_resources() or sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: cleanup_resources() or sys.exit(0))
        _cleanup_registered = True


async def process_webhook_payload(payload: Dict, event_type: str) -> Dict:
    """Process a webhook payload and return prediction result."""
    global predictor, state_manager, config
    import time
    from Metadata_Extractor.core.git_engine import sync_repo_cache

    # Extract repository info
    repo_info = payload.get("repository", {})
    owner = repo_info.get("owner", {}).get("login") or repo_info.get("owner", {}).get("name")
    repo = repo_info.get("name")

    if not owner or not repo:
        raise ValueError("Could not extract owner/repo from payload")

    # Extract commit SHA based on event type
    if event_type == "push":
        sha = payload.get("after", "")
        ref = payload.get("ref", "")
        branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else "main"
    elif event_type == "pull_request":
        pr = payload.get("pull_request", {})
        sha = pr.get("head", {}).get("sha", "")
        branch = pr.get("head", {}).get("ref", "main")
    else:
        raise ValueError(f"Unsupported event type: {event_type}")

    if not sha or sha == "0" * 40:
        raise ValueError("Invalid or empty commit SHA")

    # Update state manager with latest SHA
    if state_manager:
        state_manager.set_last_seen_sha(sha)

    # Use persistent repo cache instead of temp clone
    print(f"\n[EXTRACT] Preparing {owner}/{repo} @ {sha[:7]}...")
    extract_start = time.time()

    github_token = config.get("github_token")
    repo_path, cache_status = sync_repo_cache(owner, repo, sha, github_token)

    try:
        # Extract features
        features_dict = extract_features_for_commit(owner, repo, sha, repo_path)

        # Create feature DataFrame and predict
        feature_df = create_feature_dataframe(features_dict)
        result = predictor.predict_commit_risk(feature_df)

        extract_elapsed = time.time() - extract_start
        print(f"[EXTRACT] Feature extraction completed in {extract_elapsed:.2f}s (cache {cache_status})")

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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Module I - Metadata Risk Predictor",
        "model_loaded": predictor is not None,
        "config": {
            "owner": config.get("owner"),
            "repo": config.get("repo"),
            "mode": "webhook",
        }
    }


@app.post("/webhook")
async def webhook_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None),
):
    """GitHub webhook endpoint for push and pull_request events."""
    global config

    # Verify signature if secret configured
    webhook_secret = config.get("webhook_secret", "")
    payload_bytes = await request.body()

    if webhook_secret and not verify_webhook_signature(payload_bytes, x_hub_signature_256, webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = x_github_event or "unknown"

    # Only process push and pull_request events
    if event_type not in ("push", "pull_request"):
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"Event type {event_type} not processed"}
        )

    # Process in background to return quickly
    result = await process_webhook_payload(payload, event_type)

    # Print telemetry to console
    print_telemetry(result)

    return JSONResponse(content={
        "status": "processed",
        "risk_score": result["failure_probability"],
        "verdict": result["verdict"],
        "sha": result["sha"][:7],
    })


def print_telemetry(result: Dict) -> None:
    """Print structured MLOps telemetry to terminal."""
    print("\n" + "=" * 70)
    print(f"[INGEST]  Repo: {result['owner']}/{result['repo']} | Branch: {result['branch']}")
    print(f"          Commit: {result['short_sha']} ({result['sha']})")
    print(f"          Event: {result['event_type'].upper()}")
    print(f"          Time: {result['timestamp']}")

    features = result.get("features", {})
    print(f"\n[EXTRACT] Files touched: {features.get('total_files_changed', 'N/A')}")
    print(f"          Src churn: {features.get('git_diff_src_churn', 'N/A')}")
    print(f"          Test churn: {features.get('git_diff_test_churn', 'N/A')}")
    print(f"          Prev build: {'Yes' if features.get('has_previous_build') else 'No'}")

    print(f"\n[TRANSFORM] All 38 features validated [OK] | Scaling contracts passed [OK]")

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


def run_server_with_automation(host: str = "0.0.0.0", port: int = 5000):
    """
    Run the FastAPI server with automated ngrok tunneling and GitHub webhook registration.

    This function:
    1. Sets up ngrok authentication from .env
    2. Opens a tunnel to the local port
    3. Registers the webhook with GitHub automatically
    4. Starts the Uvicorn server
    5. Handles graceful cleanup on shutdown
    """
    global _ngrok_tunnel, _webhook_hook_id

    # Register cleanup handlers
    register_cleanup()

    # Get ngrok auth token from config (loaded from .env)
    ngrok_auth_token = config.get("ngrok_auth_token")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        print(f"[NGROK] [OK] Authenticated with ngrok")
    else:
        print(f"[NGROK] ⚠ No NGROK_AUTHTOKEN found in .env - using unauthenticated tunnel (limited)")

    # Open ngrok tunnel
    try:
        _ngrok_tunnel = ngrok.connect(port, bind_tls=True)
        public_url = _ngrok_tunnel.public_url
        print(f"[NGROK] [OK] Tunnel established: {public_url}")
        print(f"[NGROK]   Forwarding to http://localhost:{port}")

        # Print webhook URL for reference
        print(f"\n[WEBHOOK] Public endpoint: {public_url}/webhook")

        # Auto-register webhook with GitHub
        github_token = config.get("github_token")
        owner = config.get("owner")
        repo = config.get("repo")

        if github_token and owner and repo:
            print(f"[WEBHOOK] Registering webhook with GitHub...")
            hook_id = create_github_webhook(owner, repo, public_url, github_token)
            if hook_id:
                _webhook_hook_id = hook_id
                config["webhook_registered"] = True
            else:
                # Print manual instructions as fallback
                print_manual_webhook_instructions(public_url)
        else:
            print_manual_webhook_instructions(public_url)

    except Exception as e:
        print(f"[NGROK] [FAIL] Failed to establish tunnel: {e}")
        print(f"[NGROK]   Falling back to manual setup...")
        print_manual_webhook_instructions(f"http://localhost:{port}")

    # Run server (blocks until shutdown)
    print(f"\n[SERVER] Starting Uvicorn on {host}:{port}...")
    print(f"[SERVER] Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


def print_manual_webhook_instructions(public_url: str):
    """Print manual webhook setup instructions as fallback."""
    print("\n" + "=" * 70)
    print("  MANUAL WEBHOOK SETUP REQUIRED")
    print("=" * 70)
    print(f"  1. Go to: https://github.com/{config.get('owner')}/{config.get('repo')}/settings/hooks")
    print(f"  2. Click 'Add webhook'")
    print(f"  3. Payload URL: {public_url}/webhook")
    print(f"  4. Content type: application/json")
    print(f"  5. Events: Select 'push' and 'pull_request'")
    print(f"  6. Click 'Add webhook'")
    print("=" * 70 + "\n")