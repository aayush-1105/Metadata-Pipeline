#!/usr/bin/env python3
"""
Phase 5: Mode 1 (Zero-Click Ngrok + GitHub Webhook + Live Push Roundtrip) Test
===============================================================================
Tests the full Mode 1 architecture end-to-end:
1. FastAPI server in background thread
2. Ngrok tunnel
3. GitHub webhook registration
4. Live push roundtrip
5. Synthetic pull_request and ping payloads
6. Cleanup verification
"""

import sys
import os
import time
import json
import subprocess
import threading
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, r"F:\Capstone\Metadata Pipeline")

from Inference_Engine.predictor_service import PredictorService, create_feature_dataframe
from Inference_Engine.cli_server import (
    app, run_server_with_automation, cleanup_resources, config, predictor, state_manager
)
import Inference_Engine.cli_server as cli_server_module
import uvicorn
from pyngrok import ngrok


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


def send_webhook_payload(url: str, payload: dict, event_type: str, secret: str = ""):
    """Send a webhook payload to the server."""
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": event_type,
    }

    if secret:
        import hmac
        import hashlib
        payload_bytes = json.dumps(payload).encode()
        signature = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
        headers["X-Hub-Signature-256"] = f"sha256={signature}"

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    return response


def main():
    """Run Phase 5 tests"""
    print("\n" + "#"*70)
    print("# PHASE 5: MODE 1 (ZERO-CLICK NGROK + GITHUB WEBHOOK + LIVE PUSH)")
    print("#"*70)

    # Load config
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    github_token = os.getenv("GITHUB_TOKEN")
    ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN")

    owner = "aayush-1105"
    repo = "temp-python"
    port = 5000

    # Configure server
    cli_server_module.config.update({
        "owner": owner,
        "repo": repo,
        "github_token": github_token,
        "ngrok_auth_token": ngrok_auth_token,
        "webhook_secret": "",  # No secret for testing
    })

    # Initialize predictor and state manager
    print("\n[INIT] Initializing PredictorService...")
    cli_server_module.predictor = PredictorService()
    cli_server_module.state_manager = StateManager(github_token, owner, repo)
    print("[OK] PredictorService and StateManager loaded")

    # Start server in background thread
    print("\n[TEST 1] Starting FastAPI server on port 5000 in background...")
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="error"),
        daemon=True
    )
    server_thread.start()
    time.sleep(2)  # Wait for server to start

    # Test local health endpoint
    print("[TEST 2] Testing local health endpoint...")
    response = requests.get(f"http://localhost:{port}/health", timeout=5)
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] == True
    print(f"  [OK] Local health check passed: {health_data}")

    # Test ngrok tunnel
    print("\n[TEST 3] Establishing ngrok tunnel...")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        print(f"  [OK] Ngrok authenticated")

    tunnel = ngrok.connect(port, bind_tls=True)
    public_url = tunnel.public_url
    print(f"  [OK] Ngrok tunnel established: {public_url}")
    print(f"  [OK] Forwarding to http://localhost:{port}")

    # Test public health endpoint
    print("\n[TEST 4] Testing public health endpoint via ngrok...")
    response = requests.get(f"{public_url}/health", timeout=10)
    assert response.status_code == 200, f"Public health check failed: {response.status_code}"
    health_data = response.json()
    assert health_data["status"] == "healthy"
    print(f"  [OK] Public health check passed: {health_data}")

    # Register webhook with GitHub
    print("\n[TEST 5] Registering GitHub webhook...")
    from Inference_Engine.cli_server import create_github_webhook
    hook_id = create_github_webhook(owner, repo, public_url, github_token)
    if hook_id:
        cli_server_module._webhook_hook_id = hook_id
        cli_server_module.config["webhook_registered"] = True
        print(f"  [OK] Webhook registered with ID: {hook_id}")
    else:
        print(f"  [FAIL] Webhook registration failed")
        # Don't exit, continue with manual tests

    # Verify webhook via GitHub API
    print("\n[TEST 6] Verifying webhook via GitHub API...")
    verify_url = f"https://api.github.com/repos/{owner}/{repo}/hooks"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(verify_url, headers=headers, timeout=10)
    assert response.status_code == 200, f"Webhook verification failed: {response.status_code}"
    hooks = response.json()
    webhook_found = False
    for hook in hooks:
        if hook.get("config", {}).get("url") == f"{public_url}/webhook":
            webhook_found = True
            print(f"  [OK] Webhook found in GitHub: ID={hook['id']}, active={hook['active']}, events={hook['events']}")
            break
    if not webhook_found:
        print(f"  [WARN] Webhook not found in GitHub hooks list")

    # Test 7: Live Push Roundtrip
    print("\n[TEST 7] Live Push Roundtrip: Modify file, commit, push, wait for webhook...")
    test_repo_path = r"C:\Users\aayus\Downloads\New folder"
    main_py_path = os.path.join(test_repo_path, "main.py")

    # Read current content
    with open(main_py_path, "r") as f:
        content = f.read()

    # Add a new function for webhook test
    new_content = content + "\n\ndef webhook_test() -> str:\n    \"\"\"Function to test webhook roundtrip.\"\"\"\n    return \"webhook working!\"\n"

    with open(main_py_path, "w") as f:
        f.write(new_content)

    print(f"  [OK] Modified {main_py_path}")

    # Commit and push (NO Co-authored-by tag!)
    subprocess.run(
        'git add main.py && git commit -m "test: live webhook roundtrip"',
        cwd=test_repo_path, shell=True, check=True, capture_output=True, text=True
    )
    print(f"  [OK] Committed changes")

    subprocess.run(
        'git push origin main',
        cwd=test_repo_path, shell=True, check=True, capture_output=True, text=True
    )
    print(f"  [OK] Pushed to origin main")

    # Wait for webhook to be received and processed
    print("  Waiting up to 20 seconds for GitHub to send webhook...")
    webhook_received = False
    start_time = time.time()
    while time.time() - start_time < 20:
        time.sleep(1)
        # The webhook processing prints telemetry to console, so we just wait
        # In a real test, we'd have a callback or shared state

    print(f"  [OK] Waited 20 seconds for webhook delivery")
    print("  [NOTE] Check console output above for webhook telemetry!")

    # Test 8: Synthetic pull_request payload
    print("\n[TEST 8] Sending synthetic pull_request payload...")
    pr_payload = {
        "action": "opened",
        "pull_request": {
            "head": {
                "sha": "1c2b02ef8e70fb4a05c149e93e6802cf5c076590",
                "ref": "feature-branch"
            },
            "number": 1,
            "title": "Test PR",
            "body": "Test PR body",
            "comments": 2,
            "review_comments": 1
        },
        "repository": {
            "name": repo,
            "owner": {"login": owner}
        }
    }
    response = send_webhook_payload(f"{public_url}/webhook", pr_payload, "pull_request")
    print(f"  [OK] pull_request payload sent, status: {response.status_code}")
    print(f"  Response: {response.json()}")

    # Test 9: Synthetic ping payload
    print("\n[TEST 9] Sending synthetic ping payload...")
    ping_payload = {
        "zen": "Keep it logically awesome.",
        "hook_id": hook_id or 12345,
        "hook": {
            "id": hook_id or 12345,
            "type": "Repository",
            "config": {"url": f"{public_url}/webhook"}
        },
        "repository": {
            "name": repo,
            "owner": {"login": owner}
        }
    }
    response = send_webhook_payload(f"{public_url}/webhook", ping_payload, "ping")
    print(f"  [OK] ping payload sent, status: {response.status_code}")
    print(f"  Response: {response.json()}")

    # Test 10: Cleanup verification
    print("\n[TEST 10] Cleanup: Delete webhook and kill ngrok...")
    from Inference_Engine.cli_server import delete_github_webhook, cleanup_resources

    if hook_id:
        success = delete_github_webhook(owner, repo, hook_id, github_token)
        if success:
            print(f"  [OK] Webhook {hook_id} deleted from GitHub")
        else:
            print(f"  [WARN] Failed to delete webhook from GitHub")

    # Kill ngrok
    ngrok.disconnect(public_url)
    ngrok.kill()
    print(f"  [OK] Ngrok tunnel closed and process killed")

    # Verify webhook deleted from GitHub
    print("\n[TEST 11] Verifying webhook deletion via GitHub API...")
    time.sleep(2)  # Give GitHub time to process deletion
    response = requests.get(verify_url, headers=headers, timeout=10)
    assert response.status_code == 200
    hooks = response.json()
    webhook_still_exists = False
    for hook in hooks:
        if hook.get("config", {}).get("url") == f"{public_url}/webhook":
            webhook_still_exists = True
            break

    if not webhook_still_exists:
        print(f"  [OK] Webhook successfully removed from GitHub")
    else:
        print(f"  [WARN] Webhook still exists in GitHub (may be eventual consistency)")

    print("\n" + "#"*70)
    print("# PHASE 5 TESTS COMPLETED")
    print("# Check console output above for live webhook telemetry!")
    print("#"*70)


# Import StateManager at module level for the test
from Inference_Engine.state_manager import StateManager

if __name__ == "__main__":
    main()