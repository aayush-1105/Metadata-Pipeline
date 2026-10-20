import subprocess
import os
import re
import time
import shutil
import stat
from datetime import datetime
from pathlib import Path

SRC_EXTS = {'.py', '.java', '.rb', '.c', '.cpp', '.js', '.ts', '.go', '.rs'}
DOC_EXTS = {'.md', '.rst', '.txt', '.pdf', '.docx', '.html'}

# Root cache directory
REPO_CACHE_ROOT = Path(__file__).parent.parent.parent / ".repo_cache"
REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _handle_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree to handle Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def sync_repo_cache(owner: str, repo: str, commit_sha: str = None, token: str = None) -> tuple[str, str]:
    """
    Synchronize a persistent local Git cache for the given repository.

    Returns:
        tuple: (cache_dir_path, cache_status) where cache_status is "hit" or "miss"
    """
    cache_dir = REPO_CACHE_ROOT / f"{owner}__{repo}"
    cache_git_dir = cache_dir / ".git"

    # Build authenticated clone URL if token provided
    repo_url = f"https://github.com/{owner}/{repo}.git"
    if token:
        # Insert token into URL for authenticated access
        repo_url = f"https://{token}@github.com/{owner}/{repo}.git"

    start_time = time.time()

    if not cache_git_dir.exists():
        # Path A — Cache Miss (First Run for a Repo)
        print(f"[EXTRACT] Cache miss — Cloning {owner}/{repo} into persistent cache...")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, onerror=_handle_remove_readonly)

        subprocess.run(
            f'git clone --quiet "{repo_url}" "{cache_dir}"',
            shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cache_status = "miss"
    else:
        # Path B — Cache Hit (Subsequent Runs / Commits)
        print(f"[EXTRACT] Cache hit — Fetching latest delta for {owner}/{repo}...")
        cache_status = "hit"

        try:
            # Fetch all refs and prune deleted ones
            subprocess.run(
                'git fetch --quiet --all --prune',
                cwd=cache_dir, shell=True, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            if commit_sha:
                # Checkout specific commit
                subprocess.run(
                    f'git checkout --quiet --force "{commit_sha}"',
                    cwd=cache_dir, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                # Clean any untracked files
                subprocess.run(
                    'git clean --quiet -fd',
                    cwd=cache_dir, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                # No specific commit - checkout remote default branch HEAD
                # Determine default branch
                default_branch_output = subprocess.run(
                    'git symbolic-ref refs/remotes/origin/HEAD',
                    cwd=cache_dir, shell=True, capture_output=True, text=True
                )
                if default_branch_output.returncode == 0:
                    default_branch = default_branch_output.stdout.strip().split('/')[-1]
                else:
                    # Fallback: try to get from remote show
                    remote_show = subprocess.run(
                        'git remote show origin',
                        cwd=cache_dir, shell=True, capture_output=True, text=True
                    )
                    default_branch = "main"
                    for line in remote_show.stdout.splitlines():
                        if "HEAD branch" in line:
                            default_branch = line.split(":")[-1].strip()
                            break

                subprocess.run(
                    f'git checkout --quiet --force origin/{default_branch}',
                    cwd=cache_dir, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    'git reset --quiet --hard',
                    cwd=cache_dir, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    'git clean --quiet -fd',
                    cwd=cache_dir, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

        except subprocess.CalledProcessError as e:
            # Path C — Self-Healing Fallback
            print(f"[EXTRACT] WARNING: Cache sync failed ({e}), wiping cache and falling back to fresh clone...")
            try:
                shutil.rmtree(cache_dir, onerror=_handle_remove_readonly)
            except Exception:
                pass
            # Recursively call ourselves to do a fresh clone (Path A)
            return sync_repo_cache(owner, repo, commit_sha, token)

    elapsed = time.time() - start_time
    print(f"[EXTRACT] Git sync completed in {elapsed:.2f}s (cache {cache_status})")

    return str(cache_dir), cache_status

def run_git(cmd, cwd):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=True)
    return res.stdout.strip()

def get_churn_metrics(repo_path: str, commit_sha: str) -> dict:
    metrics = {
        "total_files_changed": 0,
        "code_churn_density": 0.0,
        "git_diff_src_churn": 0,
        "git_diff_test_churn": 0,
        "test_to_src_churn_ratio": 0.0,
        "gh_diff_files_added": 0,
        "gh_diff_files_deleted": 0,
        "gh_diff_files_modified": 0,
        "gh_diff_tests_added": 0,
        "gh_diff_tests_deleted": 0,
        "gh_diff_src_files": 0,
        "gh_diff_doc_files": 0,
        "gh_diff_other_files": 0,
        "gh_num_commits_on_files_touched": 0
    }
    
    # NOTE: use "~1" rather than "^" — under shell=True on Windows the caret is
    # cmd.exe's escape char and gets stripped, so "sha^" resolves back to "sha".
    parent = run_git(f"git rev-parse {commit_sha}~1", repo_path)
    if parent and "fatal" not in parent and parent != commit_sha:
        diff_target = f"{parent} {commit_sha}"
    else:
        # Root commit: diff against the empty tree so the initial import counts as churn.
        empty_tree = run_git("git hash-object -t tree /dev/null", repo_path)
        diff_target = f"{empty_tree} {commit_sha}" if empty_tree else commit_sha
    
    numstat = run_git(f"git diff --numstat {diff_target}", repo_path)
    namestat = run_git(f"git diff --name-status {diff_target}", repo_path)
    
    total_added = 0
    total_deleted = 0
    files_touched = []
    
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            add_str, del_str, fpath = parts[0], parts[1], parts[2]
            added = int(add_str) if add_str.isdigit() else 0
            deleted = int(del_str) if del_str.isdigit() else 0
            total_added += added
            total_deleted += deleted
            files_touched.append(fpath)
            
            ext = os.path.splitext(fpath)[1].lower()
            is_test = "test" in fpath.lower()
            
            if is_test:
                metrics["gh_diff_tests_added"] += added
                metrics["gh_diff_tests_deleted"] += deleted
                metrics["git_diff_test_churn"] += (added + deleted)
            elif ext in SRC_EXTS:
                metrics["git_diff_src_churn"] += (added + deleted)
                metrics["gh_diff_src_files"] += 1
            elif ext in DOC_EXTS:
                metrics["gh_diff_doc_files"] += 1
            else:
                metrics["gh_diff_other_files"] += 1

    for line in namestat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status = parts[0][0]
            if status == 'A':
                metrics["gh_diff_files_added"] += 1
            elif status == 'D':
                metrics["gh_diff_files_deleted"] += 1
            elif status == 'M':
                metrics["gh_diff_files_modified"] += 1

    metrics["total_files_changed"] = len(files_touched)
    if metrics["total_files_changed"] > 0:
        metrics["code_churn_density"] = round((total_added + total_deleted) / metrics["total_files_changed"], 2)

    if metrics["git_diff_src_churn"] > 0:
        metrics["test_to_src_churn_ratio"] = round(
            metrics["git_diff_test_churn"] / metrics["git_diff_src_churn"], 2
        )
    
    commits_touched = 0
    for f in files_touched[:10]:
        c_count = run_git(f"git rev-list --count {commit_sha} -- \"{f}\"", repo_path)
        if c_count.isdigit():
            commits_touched += int(c_count)
    metrics["gh_num_commits_on_files_touched"] = commits_touched
    
    return metrics

def get_repo_git_metrics(repo_path: str, commit_sha: str) -> dict:
    metrics = {}
    total_commits = run_git(f"git rev-list --count {commit_sha}", repo_path)
    metrics["gh_repo_num_commits"] = int(total_commits) if total_commits.isdigit() else 1
    
    branch = run_git("git rev-parse --abbrev-ref HEAD", repo_path)
    metrics["git_branch"] = branch if branch else "main"
    
    parents = run_git(f"git log --pretty=%P -n 1 {commit_sha}", repo_path).split()
    metrics["git_merged_with"] = parents[1] if len(parents) > 1 else ""
    metrics["git_merged_with_present"] = 1 if len(parents) > 1 else 0
    metrics["gh_num_commits_in_push"] = 1
    metrics["gh_commits_in_push"] = commit_sha
    
    first_commit_date = run_git("git log --reverse --format=%ad --date=iso", repo_path).splitlines()
    target_commit_date = run_git(f"git log -1 --format=%ad --date=iso {commit_sha}", repo_path)
    
    if first_commit_date and target_commit_date:
        try:
            d_first = datetime.fromisoformat(first_commit_date[0].split("+")[0].strip())
            d_target = datetime.fromisoformat(target_commit_date.split("+")[0].strip())
            metrics["gh_first_commit_created_at_year"] = d_first.year
            metrics["gh_first_commit_created_at_month"] = d_first.month
            metrics["gh_repo_age"] = (d_target - d_first).days
        except Exception:
            metrics["gh_first_commit_created_at_year"] = 2026
            metrics["gh_first_commit_created_at_month"] = 1
            metrics["gh_repo_age"] = 0
    else:
        metrics["gh_first_commit_created_at_year"] = 2026
        metrics["gh_first_commit_created_at_month"] = 1
        metrics["gh_repo_age"] = 0
        
    return metrics
