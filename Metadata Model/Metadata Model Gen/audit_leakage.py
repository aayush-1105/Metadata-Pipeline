"""
ML FORENSIC AUDIT - Data Leakage Diagnostic for Module I (Metadata Predictor)
=============================================================================
Independent audit of run_20260926_010053 (stratified-random split, ROC-AUC 0.9002).

Hypotheses under test:
  H1 (Repository Fingerprinting):   the model memorizes per-repo / per-build-event
                                     baselines because the SAME builds appear in both
                                     train and test after a random split.
  H2 (Cross-Commit Temporal Autocorr): adjacent commits from one repo are scattered
                                     across train and test, so "future" leaks into "past".

This script is READ-ONLY with respect to the shipped model/artifacts. It trains its
own throwaway LightGBM models on a 10% stratified subsample to compare split regimes.

Outputs: prints a blunt verdict and writes audit_results.json next to this file.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ZIP_PATH = r"F:\Capstone\Dataset\Pre-Processed Metadata Dataset.zip"
SCRIPT_ROOT = r"F:\Capstone\Metadata Model\Metadata Model Gen"
RESULTS_PATH = os.path.join(SCRIPT_ROOT, "audit_results.json")

RAW_TARGET_COL = "target_binary"       # 1 = PASS, 0 = FAIL
TARGET = "y_fail"                       # 1 = FAIL, 0 = PASS (model target)

FEATURE_COLUMNS = [
    "gh_is_pr", "gh_lang_java", "gh_lang_python", "gh_lang_ruby",
    "git_prev_commit_resolution_status_merge_found",
    "git_prev_commit_resolution_status_no_previous_build",
    "gh_num_issue_comments", "gh_num_commit_comments", "gh_num_pr_comments",
    "log_team_size", "log_built_commits", "gh_repo_age", "gh_repo_num_commits",
    "tr_build_number", "gh_sloc", "git_diff_src_churn", "git_diff_test_churn",
    "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified",
    "gh_diff_tests_added", "gh_diff_tests_deleted", "gh_diff_src_files",
    "gh_diff_doc_files", "gh_diff_other_files", "gh_num_commits_on_files_touched",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member", "total_files_changed", "code_churn_density",
    "test_to_src_churn_ratio", "has_previous_build", "gh_pr_created_at_year",
    "gh_pr_created_at_month", "pr_dayofweek", "git_merged_with_present",
]

# The 4 macro-repo descriptors under suspicion (top-4 by gain in the shipped model)
MACRO4 = ["gh_repo_age", "tr_build_number", "gh_repo_num_commits", "gh_sloc"]

# Signature that identifies a single physical build event (its rows are duplicated ~4x).
# lang flags added to disambiguate rare macro collisions across repos.
BUILD_EVENT_KEYS = [
    "gh_repo_age", "gh_sloc", "gh_repo_num_commits", "tr_build_number",
    "gh_lang_java", "gh_lang_python", "gh_lang_ruby",
]

# Lifecycle composite that interleaves PRs and pushes (per memory note); used as the
# global chronological ordering proxy since no calendar timestamp spans all repos.
LIFECYCLE_ORDER = ["tr_build_number", "gh_repo_age", "gh_repo_num_commits"]

SUBSAMPLE_FRAC = 0.10
SEED = 42

# Audit LGBM params: same shape as production, n_estimators capped for turnaround.
LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=127, max_depth=12,
    min_child_samples=100, subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, objective="binary", metric="auc", verbosity=-1,
)
EARLY_STOP = 50

RESULTS = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------------
def train_eval(df_tr, df_va, df_te, feats, label=""):
    """Fit a throwaway LightGBM on `feats`, early-stop on val AUC, score on test."""
    Xtr, ytr = df_tr[feats], df_tr[TARGET]
    Xva, yva = df_va[feats], df_va[TARGET]
    Xte, yte = df_te[feats], df_te[TARGET].to_numpy()

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    p = model.predict_proba(Xte)[:, 1]
    yhat = (p >= 0.5).astype(int)
    res = dict(
        roc_auc=round(float(roc_auc_score(yte, p)), 4),
        pr_auc=round(float(average_precision_score(yte, p)), 4),
        mcc_at_50=round(float(matthews_corrcoef(yte, yhat)), 4),
        best_iteration=int(model.best_iteration_ or LGBM_PARAMS["n_estimators"]),
        n_train=len(df_tr), n_test=len(df_te), n_features=len(feats),
    )
    if label:
        log(f"  {label:<46s} ROC-AUC={res['roc_auc']:.4f}  "
            f"PR-AUC={res['pr_auc']:.4f}  MCC@0.5={res['mcc_at_50']:.4f}  "
            f"(best_iter={res['best_iteration']})")
    return res


def stratified_71515(df, seed=SEED):
    """Reproduce the production split logic: stratified random 70/15/15 on y_fail."""
    tr, tmp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[TARGET])
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp[TARGET])
    return tr, va, te


def group_split_71515(df, group_col, seed=SEED):
    """70/15/15 split where every value of `group_col` lands entirely in ONE split.
    Prevents a group's rows from straddling train and test."""
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    tr_g = set(uniq[: int(0.70 * n)])
    va_g = set(uniq[int(0.70 * n): int(0.85 * n)])
    in_tr = df[group_col].isin(tr_g)
    in_va = df[group_col].isin(va_g)
    tr = df[in_tr]
    va = df[in_va]
    te = df[~(in_tr | in_va)]
    return tr, va, te


def temporal_71515(df, order_cols):
    """Global chronological split: sort by lifecycle order, first 70% -> train,
    middle 15% -> val, last 15% -> test. Past predicts future."""
    d = df.sort_values(order_cols, kind="mergesort").reset_index(drop=True)
    n = len(d)
    a, b = int(0.70 * n), int(0.85 * n)
    return d.iloc[:a], d.iloc[a:b], d.iloc[b:]


# ---------------------------------------------------------------------------
# Repository reconstruction (no repo-ID column exists in the dataset)
# ---------------------------------------------------------------------------
def reconstruct_repos(df):
    """
    There is NO repo/project/commit identifier column. We reconstruct a surrogate
    repository id by linking build events into monotone lifecycle trajectories.

    Within one repo, as builds progress: tr_build_number strictly increases and
    gh_repo_age / gh_repo_num_commits are non-decreasing (time and history only grow).
    Across repos, these jump. Language one-hots are repo-invariant and partition the
    space cheaply. Greedy nearest-predecessor tracking per language group:

      process build events in age (calendar-proxy) order; assign each to the open
      track whose last build_number is the closest one still below the current build
      and whose (age, commits) are consistent; else open a new repo track.

    Returns a Series (repo_id per row) or None if reconstruction fails validation.
    """
    lang = (df["gh_lang_java"].astype(int).astype(str)
            + df["gh_lang_python"].astype(int).astype(str)
            + df["gh_lang_ruby"].astype(int).astype(str))
    # Unique build events only (rows are ~4x duplicated).
    ev = (df.assign(_lang=lang)[BUILD_EVENT_KEYS + ["_lang"]]
          .drop_duplicates()
          .reset_index(drop=True))
    ev = ev.sort_values(["_lang", "gh_repo_age", "tr_build_number"], kind="mergesort")

    AGE_GAP_MAX = 400.0     # days: same repo builds are not centuries apart
    COMMIT_GAP_MAX = 20000  # commits added between two observed builds
    next_id = 0
    repo_of_event = np.empty(len(ev), dtype=np.int64)

    build_arr = ev["tr_build_number"].to_numpy()
    age_arr = ev["gh_repo_age"].to_numpy()
    commit_arr = ev["gh_repo_num_commits"].to_numpy()
    lang_arr = ev["_lang"].to_numpy()

    # Process each language block independently (langs are repo-invariant). Within a
    # block events are already age-sorted, so we keep only "active" tracks touched
    # within AGE_GAP_MAX and prune the rest -> bounded scan per event.
    i = 0
    N = len(ev)
    while i < N:
        j = i
        lg = lang_arr[i]
        while j < N and lang_arr[j] == lg:
            j += 1
        active = []  # each: [last_build, last_age, last_commits, repo_id]
        for k in range(i, j):
            b = build_arr[k]; ag = age_arr[k]; cm = commit_arr[k]
            if active and len(active) > 8:
                cutoff = ag - AGE_GAP_MAX
                active = [t for t in active if t[1] >= cutoff]
            best_t, best_score = None, None
            for t in active:
                if t[0] < b and t[1] <= ag + 1e-9 and t[2] <= cm \
                        and (ag - t[1]) <= AGE_GAP_MAX and (cm - t[2]) <= COMMIT_GAP_MAX:
                    score = ag - t[1]  # temporally closest predecessor wins
                    if best_score is None or score < best_score:
                        best_score, best_t = score, t
            if best_t is not None:
                best_t[0], best_t[1], best_t[2] = b, ag, cm
                repo_of_event[k] = best_t[3]
            else:
                repo_of_event[k] = next_id
                active.append([b, ag, cm, next_id])
                next_id += 1
        i = j
    ev["_repo"] = repo_of_event


    # ---- validation: sane repo count + within-repo build monotonicity ----
    n_repos = ev["_repo"].nunique()
    grp = ev.sort_values(["_repo", "tr_build_number"])
    mono = grp.groupby("_repo")["gh_repo_num_commits"].apply(
        lambda s: bool((s.diff().fillna(0) >= 0).all()))
    frac_mono = float(mono.mean())
    RESULTS.setdefault("repo_reconstruction", {})
    RESULTS["repo_reconstruction"].update(
        n_repos=int(n_repos), frac_commit_monotone=round(frac_mono, 4),
        n_build_events=int(len(ev)))
    log(f"  reconstructed repos={n_repos}  commit-monotone repos={frac_mono:.3f}")
    if not (150 <= n_repos <= 20000 and frac_mono >= 0.90):
        log("  [!] reconstruction failed validation; within-repo test will be skipped")
        return None

    # map event repo id back to every duplicated row
    key = df.assign(_lang=lang).set_index(BUILD_EVENT_KEYS + ["_lang"]).index
    ev_map = ev.set_index(BUILD_EVENT_KEYS + ["_lang"])["_repo"]
    return pd.Series(ev_map.reindex(key).to_numpy(), index=df.index, name="repo_id")


def within_repo_temporal_split(df, repo_col, order_cols):
    """Per Task 3: inside EACH repo, first 70% chronologically -> train,
    middle 15% -> val, last 15% -> test. No future commit of a repo is in its train."""
    d = df.sort_values([repo_col] + order_cols, kind="mergesort").reset_index(drop=True)
    rank = d.groupby(repo_col).cumcount()
    size = d.groupby(repo_col)[repo_col].transform("size")
    frac = rank / size
    tr = d[frac < 0.70]
    va = d[(frac >= 0.70) & (frac < 0.85)]
    te = d[frac >= 0.85]
    return tr, va, te


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    hr("MODULE I - DATA LEAKAGE FORENSIC AUDIT")
    log(f"Loading dataset: {DATASET_ZIP_PATH}")
    df = pd.read_csv(DATASET_ZIP_PATH, compression="zip")
    df[TARGET] = (1 - df[RAW_TARGET_COL]).astype(np.int32)
    log(f"Loaded {len(df):,} rows x {df.shape[1]} cols")

    # ================================================================
    # TASK 1: identifier / group columns + train-test overlap of the SHIPPED split
    # ================================================================
    hr("TASK 1 - IDENTIFIER COLUMNS & TRAIN/TEST OVERLAP (shipped random split)")
    non_feature = [c for c in df.columns if c not in FEATURE_COLUMNS + [RAW_TARGET_COL, TARGET]]
    id_like = [c for c in df.columns
               if any(k in c.lower() for k in ("name", "repo_id", "project", "slug",
                                               "commit_sha", "sha", "url", "id"))]
    log(f"Columns beyond the 38 features + target: {non_feature if non_feature else 'NONE'}")
    log(f"Explicit project/repo/commit identifier columns: "
        f"{id_like if id_like else 'NONE FOUND'}")
    RESULTS["task1_identifiers"] = dict(
        total_columns=df.shape[1],
        non_feature_columns=non_feature,
        explicit_identifier_columns=id_like,
        verdict=("No project/repo/commit ID column exists. Grouping must use a "
                 "reconstructed surrogate key."),
    )

    # Build-event signature (its ~4x duplicate rows should never straddle splits)
    log("Computing build-event signatures (full data)...")
    sig = df.groupby(BUILD_EVENT_KEYS, sort=False).ngroup()
    df["_sig"] = sig.to_numpy()
    n_events = int(df["_sig"].nunique())
    dup_per_event = round(len(df) / n_events, 3)
    log(f"Distinct build events: {n_events:,} | mean duplicate rows/event: {dup_per_event}")

    # Reproduce the EXACT production split on full data and measure overlap
    tr, va, te = stratified_71515(df, seed=SEED)
    tr_sig = set(tr["_sig"].unique())
    te_sig_arr = te["_sig"].to_numpy()
    te_sig_unique = set(np.unique(te_sig_arr))

    row_overlap = float(np.isin(te_sig_arr, list(tr_sig)).mean())      # test ROWS whose event is in train
    event_overlap = len(te_sig_unique & tr_sig) / len(te_sig_unique)   # test EVENTS also in train
    RESULTS["task1_overlap"] = dict(
        n_build_events=n_events, mean_dup_rows_per_event=dup_per_event,
        pct_test_rows_whose_build_event_also_in_train=round(100 * row_overlap, 3),
        pct_test_build_events_also_in_train=round(100 * event_overlap, 3),
    )
    log(f"OVERLAP: {100*row_overlap:.2f}% of TEST rows share a build-event with TRAIN")
    log(f"OVERLAP: {100*event_overlap:.2f}% of distinct TEST build-events also appear in TRAIN")

    # Reconstruct surrogate repos and measure repo-level overlap
    log("Reconstructing surrogate repositories (this may take a minute)...")
    repo_id = reconstruct_repos(df)
    if repo_id is not None:
        df["_repo"] = repo_id.to_numpy()
        tr_r = set(df.loc[tr.index, "_repo"].unique())
        te_r = df.loc[te.index, "_repo"].to_numpy()
        te_r_unique = set(np.unique(te_r))
        repo_row_overlap = float(np.isin(te_r, list(tr_r)).mean())
        repo_overlap = len(te_r_unique & tr_r) / len(te_r_unique)
        RESULTS["task1_overlap"].update(
            pct_test_rows_whose_repo_also_in_train=round(100 * repo_row_overlap, 3),
            pct_test_repos_also_in_train=round(100 * repo_overlap, 3),
        )
        log(f"OVERLAP: {100*repo_overlap:.2f}% of reconstructed TEST repos also in TRAIN")
        log(f"OVERLAP: {100*repo_row_overlap:.2f}% of TEST rows belong to a repo seen in TRAIN")

    # ================================================================
    # Build the 10% stratified subsample used for all modeling experiments
    # ================================================================
    hr("BUILDING 10% STRATIFIED SUBSAMPLE FOR MODELING EXPERIMENTS")
    sub, _ = train_test_split(df, train_size=SUBSAMPLE_FRAC, random_state=SEED,
                              stratify=df[TARGET])
    sub = sub.reset_index(drop=True)
    log(f"Subsample: {len(sub):,} rows | FAIL%={100*sub[TARGET].mean():.2f}")
    RESULTS["subsample_rows"] = len(sub)

    # ================================================================
    # TASK 2: feature ablation / fingerprint test (random split, apples-to-apples)
    # ================================================================
    hr("TASK 2 - FEATURE ABLATION (random stratified split on subsample)")
    a_tr, a_va, a_te = stratified_71515(sub, seed=SEED)
    non_macro = [f for f in FEATURE_COLUMNS if f not in MACRO4]
    RESULTS["task2_ablation"] = dict(
        all_38_features=train_eval(a_tr, a_va, a_te, FEATURE_COLUMNS, "ALL 38 features"),
        only_4_macro=train_eval(a_tr, a_va, a_te, MACRO4, "ONLY 4 macro features"),
        exclude_4_macro=train_eval(a_tr, a_va, a_te, non_macro, "EXCLUDING 4 macro features"),
    )

    # ================================================================
    # TASK 3: honest splits - group-by-build-event, global temporal, within-repo temporal
    # ================================================================
    hr("TASK 3 - LEAK-FREE SPLITS (the ground-truth tests)")

    log("[A] GROUP split by build-event (duplicate rows cannot straddle train/test):")
    g_tr, g_va, g_te = group_split_71515(sub, "_sig", seed=SEED)
    RESULTS["task3_group_by_build_event"] = train_eval(
        g_tr, g_va, g_te, FEATURE_COLUMNS, "group-by-build-event (all 38)")

    log("[B] GLOBAL TEMPORAL split (sort by lifecycle composite, past->future):")
    t_tr, t_va, t_te = temporal_71515(sub, LIFECYCLE_ORDER)
    log(f"    PR%% train/val/test = {100*t_tr['gh_is_pr'].mean():.1f}/"
        f"{100*t_va['gh_is_pr'].mean():.1f}/{100*t_te['gh_is_pr'].mean():.1f} | "
        f"FAIL%% = {100*t_tr[TARGET].mean():.1f}/{100*t_va[TARGET].mean():.1f}/"
        f"{100*t_te[TARGET].mean():.1f}")
    RESULTS["task3_global_temporal"] = train_eval(
        t_tr, t_va, t_te, FEATURE_COLUMNS, "global temporal (all 38)")

    if "_repo" in sub.columns:
        log("[C] WITHIN-REPO TEMPORAL split (per repo: first70/mid15/last15 by lifecycle):")
        w_tr, w_va, w_te = within_repo_temporal_split(sub, "_repo", LIFECYCLE_ORDER)
        log(f"    rows train/val/test = {len(w_tr):,}/{len(w_va):,}/{len(w_te):,}")
        RESULTS["task3_within_repo_temporal"] = train_eval(
            w_tr, w_va, w_te, FEATURE_COLUMNS, "within-repo temporal (all 38)")
        # combine the leak-free regime with macro ablation: honest signal w/o fingerprints
        RESULTS["task3_within_repo_temporal_no_macro"] = train_eval(
            w_tr, w_va, w_te, non_macro, "within-repo temporal, NO macro-4")
    else:
        log("[C] within-repo temporal split skipped (reconstruction unavailable)")

    # ================================================================
    # VERDICT
    # ================================================================
    print_verdict()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)
    log(f"Full numeric results written to: {RESULTS_PATH}")
    log(f"Total audit runtime: {time.time()-t0:.0f}s")


def print_verdict():
    hr("BLUNT VERDICT")
    shipped = 0.9002
    ab = RESULTS["task2_ablation"]
    grp = RESULTS.get("task3_group_by_build_event", {})
    tmp = RESULTS.get("task3_global_temporal", {})
    wr = RESULTS.get("task3_within_repo_temporal", {})
    ov = RESULTS["task1_overlap"]

    honest = None
    for cand in (wr, tmp):
        if cand.get("roc_auc") is not None:
            honest = cand["roc_auc"]; break

    print(f"Shipped random-split ROC-AUC (run_20260926_010053) : {shipped:.4f}")
    print(f"  Test rows sharing a build-event with train        : "
          f"{ov['pct_test_rows_whose_build_event_also_in_train']:.2f}%")
    if "pct_test_repos_also_in_train" in ov:
        print(f"  Reconstructed test repos also seen in train        : "
              f"{ov['pct_test_repos_also_in_train']:.2f}%")
    print(f"  Ablation - ONLY 4 macro features (random split)    : "
          f"{ab['only_4_macro']['roc_auc']:.4f}")
    print(f"  Ablation - EXCLUDING 4 macro features (random)     : "
          f"{ab['exclude_4_macro']['roc_auc']:.4f}")
    if grp:
        print(f"  Group-by-build-event split (all 38 feats)          : "
              f"{grp['roc_auc']:.4f}  (MCC {grp['mcc_at_50']:.4f})")
    if tmp:
        print(f"  GLOBAL temporal split (all 38 feats)               : "
              f"{tmp['roc_auc']:.4f}  (MCC {tmp['mcc_at_50']:.4f})")
    if wr:
        print(f"  WITHIN-REPO temporal split (all 38 feats)          : "
              f"{wr['roc_auc']:.4f}  (MCC {wr['mcc_at_50']:.4f})")

    if honest is not None:
        leaked = shipped - honest
        pct = 100 * leaked / (shipped - 0.5) if shipped > 0.5 else 0
        print()
        print(f">>> Estimated honest ROC-AUC baseline               : ~{honest:.3f}")
        print(f">>> AUC lost to leakage (skill above 0.5 random)    : "
              f"{leaked:+.3f} of {shipped-0.5:.3f}  (~{pct:.0f}% of the model's "
              f"apparent skill was leakage)")
    RESULTS["verdict_summary"] = dict(
        shipped_roc_auc=shipped, honest_roc_auc=honest,
        auc_lost_to_leakage=(round(shipped - honest, 4) if honest else None),
    )


if __name__ == "__main__":
    main()



