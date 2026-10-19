"""
State Manager - Module I (Pre-Build Commit Metadata Risk Predictor)
===================================================================
Manages commit tracking for Mode 2 (Active Poller).
Implements HTTP ETag / If-None-Match caching alongside last_seen_sha tracking
to minimize GitHub API rate-limit consumption.
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

import requests


class StateManager:
    """Manages commit state and GitHub API polling with ETag caching."""

    def __init__(self, github_token: str, owner: str, repo: str, state_file: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            github_token: GitHub PAT for API authentication
            owner: Repository owner
            repo: Repository name
            state_file: Optional path to persist state (defaults to .pipeline_state.json in cwd)
        """
        self.github_token = github_token
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"

        # Headers for GitHub API
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # State persistence
        if state_file is None:
            state_file = ".pipeline_state.json"
        self.state_file = Path(state_file)
        self.state = self._load_state()

        # ETag cache for commits endpoint
        self.commits_etag: Optional[str] = self.state.get("commits_etag")
        self.last_seen_sha: Optional[str] = self.state.get("last_seen_sha")

    def _load_state(self) -> Dict:
        """Load persisted state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state["commits_etag"] = self.commits_etag
        self.state["last_seen_sha"] = self.last_seen_sha
        self.state["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass  # Best effort

    def fetch_latest_commit(self) -> Tuple[Optional[Dict], bool]:
        """
        Fetch the latest commit from GitHub with ETag caching.

        Returns:
            Tuple of (commit_data_dict, is_new_commit)
            - commit_data_dict: Contains sha, author, message, timestamp, branch info
            - is_new_commit: True if this is a new commit not seen before
        """
        url = f"{self.base_url}/commits"
        params = {"per_page": 1, "sha": "main"}  # default to main branch

        # Add ETag header if we have one
        headers = self.headers.copy()
        if self.commits_etag:
            headers["If-None-Match"] = self.commits_etag

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 304:
                # Not modified - no new commits
                return None, False

            if response.status_code != 200:
                print(f"[STATE] GitHub API error: {response.status_code} - {response.text}")
                return None, False

            # Update ETag from response
            new_etag = response.headers.get("ETag")
            if new_etag:
                self.commits_etag = new_etag

            commits = response.json()
            if not commits:
                return None, False

            latest = commits[0]
            latest_sha = latest.get("sha", "")

            # Check if this is a new commit
            is_new = latest_sha != self.last_seen_sha

            if is_new:
                self.last_seen_sha = latest_sha
                self._save_state()

            # Extract relevant commit metadata
            commit_data = {
                "sha": latest_sha,
                "short_sha": latest_sha[:7],
                "author": latest.get("commit", {}).get("author", {}).get("name", "unknown"),
                "author_email": latest.get("commit", {}).get("author", {}).get("email", ""),
                "message": latest.get("commit", {}).get("message", "").strip().split("\n")[0],
                "timestamp": latest.get("commit", {}).get("author", {}).get("date", ""),
                "url": latest.get("html_url", ""),
            }

            return commit_data, is_new

        except requests.RequestException as e:
            print(f"[STATE] Network error fetching commits: {e}")
            return None, False
        except Exception as e:
            print(f"[STATE] Unexpected error fetching commits: {e}")
            return None, False

    def get_last_seen_sha(self) -> Optional[str]:
        """Get the last seen commit SHA."""
        return self.last_seen_sha

    def set_last_seen_sha(self, sha: str) -> None:
        """Manually set the last seen SHA (e.g., for initialization)."""
        self.last_seen_sha = sha
        self._save_state()

    def get_rate_limit_info(self) -> Dict:
        """Get current GitHub API rate limit status."""
        try:
            response = requests.get(
                "https://api.github.com/rate_limit",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                core = data.get("resources", {}).get("core", {})
                return {
                    "limit": core.get("limit", 5000),
                    "remaining": core.get("remaining", 5000),
                    "reset_timestamp": core.get("reset", 0),
                    "reset_time": datetime.fromtimestamp(core.get("reset", 0)).strftime("%H:%M:%S") if core.get("reset") else "unknown"
                }
        except Exception:
            pass
        return {"limit": 5000, "remaining": 5000, "reset_time": "unknown"}


def create_state_manager(github_token: str, owner: str, repo: str) -> StateManager:
    """Factory function to create a StateManager instance."""
    return StateManager(github_token, owner, repo)