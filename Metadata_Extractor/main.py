import os
import argparse
import subprocess
import shutil
import tempfile
import stat
import time

import pandas as pd

from .core.git_engine import get_churn_metrics, get_repo_git_metrics, sync_repo_cache
from .core.static_analyzer import analyze_workspace
from .core.github_api import get_github_metadata
from .core.ci_analyzer import get_ci_metrics
from .schema.normalizer import normalize_features


def parse_repo_url(url: str):
    clean = url.rstrip("/").replace(".git", "")
    parts = clean.split("/")
    return parts[-2], parts[-1]


def run_git_capture(cmd: str, cwd: str):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=True)


def resolve_ref(repo_path: str, ref: str):
    """Return the first ref name that exists (local branch, then origin/<branch>)."""
    for candidate in [ref, f"origin/{ref}"]:
        r = run_git_capture(f'git rev-parse --verify --quiet "{candidate}"', repo_path)
        if r.returncode == 0 and r.stdout.strip():
            return candidate
    return None


def get_target_commits(repo_path: str, branch: str, start_commit: str, num_commits: int):
    """Newest-first list of up to num_commits SHAs ending at the branch tip (or start_commit)."""
    if start_commit:
        tip = start_commit
    else:
        ref = resolve_ref(repo_path, branch)
        if ref is None:
            print(f"[!] Branch '{branch}' not found; falling back to current HEAD.")
            tip = "HEAD"
        else:
            tip = ref
    r = run_git_capture(f"git rev-list -n {num_commits} {tip}", repo_path)
    return [s for s in r.stdout.strip().splitlines() if s]


def is_dirty(repo_path: str) -> bool:
    return bool(run_git_capture("git status --porcelain", repo_path).stdout.strip())


def current_ref(repo_path: str) -> str:
    r = run_git_capture("git symbolic-ref --quiet --short HEAD", repo_path)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return run_git_capture("git rev-parse HEAD", repo_path).stdout.strip()


def write_output(df: pd.DataFrame, path: str, write_mode: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if write_mode == "overwrite":
        df.to_csv(path, index=False, mode="w")
    else:  # append: keep existing rows, only write the header for a new/empty file
        file_has_rows = os.path.exists(path) and os.path.getsize(path) > 0
        df.to_csv(path, index=False, mode="a", header=not file_has_rows)


def _handle_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree to handle Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    parser = argparse.ArgumentParser(description="Extract 39 TravisTorrent Metadata Features from a Repository")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository URL or local path")
    parser.add_argument("--commit", type=str, default=None, help="Tip commit SHA to start from (defaults to branch tip)")
    parser.add_argument("--branch", type=str, default="main", help="Branch to read commits from (default: main)")
    parser.add_argument("-n", "--num-commits", type=int, default=1, help="Number of most-recent commits to extract (default: 1)")
    parser.add_argument("--write-mode", choices=["append", "overwrite"], default="append",
                        help="Append to the output CSV (default) or overwrite it")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV path")
    parser.add_argument("--token", type=str, default=None, help="GitHub PAT for private repos (optional)")
    args = parser.parse_args()

    # Use persistent repo cache for HTTP/HTTPS repos
    repo_path = None
    owner = None
    repo = None

    if args.repo.startswith("http://") or args.repo.startswith("https://"):
        owner, repo = parse_repo_url(args.repo)
        print(f"[EXTRACT] Syncing persistent cache for {owner}/{repo}...")
        repo_path, cache_status = sync_repo_cache(owner, repo, None, args.token)
        print(f"[EXTRACT] Using cached repository at {repo_path} (cache {cache_status})")
    else:
        repo_path = os.path.abspath(args.repo)
        owner, repo = "local", os.path.basename(repo_path)

    commits = get_target_commits(repo_path, args.branch, args.commit, max(1, args.num_commits))
    if not commits:
        print("[!] No commits found for the given branch/commit.")
        return

    # We check out each commit so the static-analysis features reflect that
    # commit's tree. Safe with persistent cache since we own it.
    can_checkout = True
    original_ref = current_ref(repo_path)
    if is_dirty(repo_path):
        can_checkout = False
        print("[!] Working tree has uncommitted changes; static features will reflect the current tree for every commit.")

    print(f"[*] Extracting {len(commits)} commit(s) for Repo: {owner}/{repo} | Branch: {args.branch}")

    rows = []
    for sha in commits:
        if can_checkout:
            run_git_capture(f"git checkout -q {sha}", repo_path)
        print(f"    - {sha}")
        all_features = {}
        all_features.update(get_churn_metrics(repo_path, sha))
        all_features.update(get_repo_git_metrics(repo_path, sha))
        all_features.update(analyze_workspace(repo_path))
        all_features.update(get_github_metadata(owner, repo, sha))
        all_features.update(get_ci_metrics(owner, repo, sha))
        rows.append(normalize_features(all_features))

    if can_checkout and original_ref:
        run_git_capture(f"git checkout -q {original_ref}", repo_path)

    final_df = pd.concat(rows, ignore_index=True)
    output_path = os.path.abspath(args.output)
    write_output(final_df, output_path, args.write_mode)

    action = "Appended" if args.write_mode == "append" else "Wrote"
    print(f"[+] {action} {final_df.shape[0]} row(s) x {final_df.shape[1]} features to {output_path}")
    print(final_df.T)
    # NOTE: Do NOT delete repo_path - persistent cache is kept for future runs


if __name__ == "__main__":
    main()
