import pandas as pd
from .columns import COLUMNS

def normalize_features(data_dict: dict) -> pd.DataFrame:
    row = {}
    numeric_tokens = ("num", "churn", "files", "sloc", "age", "duration", "year",
                      "month", "density", "kloc", "log", "ratio", "dayofweek",
                      "build", "commits")
    binary_cols = {"gh_is_pr", "gh_by_core_team_member", "target_binary",
                   "git_merged_with_present", "has_previous_build"}
    for col in COLUMNS:
        val = data_dict.get(col, None)
        if val is None:
            if col.startswith("gh_lang_") or col.startswith("git_prev_commit_resolution_status_") or col in binary_cols:
                row[col] = 0
            elif any(tok in col for tok in numeric_tokens):
                row[col] = 0
            else:
                row[col] = ""
        else:
            row[col] = val
    df = pd.DataFrame([row], columns=COLUMNS)
    return df
