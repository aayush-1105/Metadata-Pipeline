import subprocess
import json
import math

def get_ci_metrics(owner: str, repo: str, commit_sha: str) -> dict:
    metrics = {
        "tr_build_id": 0,
        "tr_job_id": 0,
        "tr_build_number": 0,
        "tr_original_commit": commit_sha,
        "tr_duration": 0,
        "tr_jobs": 1,
        "gh_build_started_at": "",
        "git_prev_built_commit": "",
        "tr_prev_build": 0,
        "git_all_built_commits": commit_sha,
        "git_num_all_built_commits": 1,
        "log_built_commits": 0.0,
        "has_previous_build": 0,
        "target_binary": 0,
        "git_prev_commit_resolution_status_merge_found": 0,
        "git_prev_commit_resolution_status_no_previous_build": 1
    }

    cmd = f"gh run list --repo {owner}/{repo} --limit 10 --json databaseId,number,headSha,conclusion,status,createdAt,updatedAt"
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if res.returncode == 0:
        try:
            runs = json.loads(res.stdout)
            current_run = next((r for r in runs if r.get("headSha") == commit_sha), None)
            if current_run:
                metrics["tr_build_id"] = current_run.get("databaseId", 0)
                metrics["tr_job_id"] = current_run.get("databaseId", 0)
                metrics["tr_build_number"] = current_run.get("number", 0)
                metrics["gh_build_started_at"] = current_run.get("createdAt", "")
                metrics["target_binary"] = 1 if current_run.get("conclusion") == "failure" else 0

            if len(runs) > 1:
                metrics["git_prev_commit_resolution_status_no_previous_build"] = 0
                metrics["git_prev_commit_resolution_status_merge_found"] = 1
                metrics["has_previous_build"] = 1
                prev_run = runs[1]
                metrics["tr_prev_build"] = prev_run.get("databaseId", 0)
                metrics["git_prev_built_commit"] = prev_run.get("headSha", "")
                metrics["git_num_all_built_commits"] = len(runs)
        except Exception:
            pass

    metrics["log_built_commits"] = round(math.log1p(metrics["git_num_all_built_commits"]), 4)

    return metrics
