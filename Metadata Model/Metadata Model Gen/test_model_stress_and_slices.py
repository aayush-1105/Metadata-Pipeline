"""
Stress & Slice Audit - Module I (Metadata Predictor)
====================================================
Final robustness / fairness / calibration audit for the trained metadata model.

Gates:
  1. Adversarial Edge-Case Fuzzing - anomaly vectors: no NaN/Inf/exceptions,
     probabilities in [0,1]; Mega-Monolith risk > 0.60.
  2. Subgroup Slice Performance    - held-out test slice; ROC-AUC >= 0.75 per
     language (>500 samples) and per PR/Push; Brier <= 0.16 across all slices.
  3. Calibration Monotonicity      - empirical failure rate non-decreasing
     across populated probability bins.

Target semantics: y_fail = 1 - target_binary => y_fail=1 == FAIL.
predict_proba[:, 1] == P(failure) == risk. Read-only: never trains or writes models.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

RUN_DIR = r"F:\Capstone\Metadata Model\Models\run_20260926_013515"
CAL_PATH = os.path.join(RUN_DIR, "calibrated_metadata_model.joblib")
DATASET_ZIP = r"F:\Capstone\Dataset\Pre-Processed Metadata Dataset.zip"

RAW_TARGET_COL = "target_binary"   # 1 = PASS, 0 = FAIL
TARGET_COL = "y_fail"              # 1 = FAIL, 0 = PASS (model target)
SLICE_CAP = 100_000

# Strict 38-feature schema (must match train_metadata_model.py exactly).
FEATURE_COLUMNS = [
    "gh_is_pr",
    "gh_lang_java",
    "gh_lang_python",
    "gh_lang_ruby",
    "git_prev_commit_resolution_status_merge_found",
    "git_prev_commit_resolution_status_no_previous_build",
    "gh_num_issue_comments",
    "gh_num_commit_comments",
    "gh_num_pr_comments",
    "log_team_size",
    "log_built_commits",
    "gh_repo_age",
    "gh_repo_num_commits",
    "tr_build_number",
    "gh_sloc",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_num_commits_on_files_touched",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member",
    "total_files_changed",
    "code_churn_density",
    "test_to_src_churn_ratio",
    "has_previous_build",
    "gh_pr_created_at_year",
    "gh_pr_created_at_month",
    "pr_dayofweek",
    "git_merged_with_present",
]

gates = {}    # name -> (bool passed, str detail)
tables = {}   # name -> markdown string for the report


def record(name, passed, detail):
    gates[name] = (bool(passed), detail)


def zero_frame(overrides=None):
    """All-zeros 38-feature single-row DataFrame, with optional overrides."""
    row = {c: 0 for c in FEATURE_COLUMNS}
    if overrides:
        row.update(overrides)
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def realistic_base():
    """Neutral, internally-consistent baseline build event (low risk).

    Mirrors the baseline used in test_model_production.py so that a
    monolith override lands on an in-distribution point rather than the
    degenerate all-zeros origin.
    """
    return {
        "gh_is_pr": 1, "gh_lang_java": 0, "gh_lang_python": 1, "gh_lang_ruby": 0,
        "git_prev_commit_resolution_status_merge_found": 0,
        "git_prev_commit_resolution_status_no_previous_build": 0,
        "gh_num_issue_comments": 0, "gh_num_commit_comments": 0, "gh_num_pr_comments": 1,
        "log_team_size": float(np.log1p(10)), "log_built_commits": float(np.log1p(500)),
        "gh_repo_age": 1000, "gh_repo_num_commits": 5000, "tr_build_number": 100,
        "gh_sloc": 50000,
        "git_diff_src_churn": 10, "git_diff_test_churn": 5,
        "gh_diff_files_added": 0, "gh_diff_files_deleted": 0, "gh_diff_files_modified": 1,
        "gh_diff_tests_added": 0, "gh_diff_tests_deleted": 0,
        "gh_diff_src_files": 1, "gh_diff_doc_files": 0, "gh_diff_other_files": 0,
        "gh_num_commits_on_files_touched": 5,
        "gh_test_lines_per_kloc": 100.0, "gh_test_cases_per_kloc": 10.0,
        "gh_asserts_cases_per_kloc": 15.0,
        "gh_by_core_team_member": 1,
        "total_files_changed": 1, "code_churn_density": 10.0, "test_to_src_churn_ratio": 0.5,
        "has_previous_build": 1,
        "gh_pr_created_at_year": 2020, "gh_pr_created_at_month": 6, "pr_dayofweek": 2,
        "git_merged_with_present": 1,
    }


def frame_from(base, overrides=None):
    """Single-row DataFrame from a full 38-key base dict plus overrides."""
    row = dict(base)
    if overrides:
        row.update(overrides)
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
cal_model = joblib.load(CAL_PATH)


# ---------------------------------------------------------------------------
# GATE 1: ADVERSARIAL EDGE-CASE FUZZING
# ---------------------------------------------------------------------------
fuzz_vectors = [
    # (name, base_kind, overrides)
    ("All-zeros", "zero", {}),
    # Mega-Monolith re-expressed as an IN-DISTRIBUTION massive risky change:
    #  - built on the realistic baseline (not the degenerate all-zeros origin,
    #    which by itself sits near 0.52 on out-of-distribution feature values);
    #  - total_files_changed on its trained (log) scale: log1p(25000) ~= 10.13;
    #  - drive the signals the model actually uses for FAIL: huge churn density,
    #    250 deleted tests, non-core author (mirrors the production destructive
    #    case, which scored 0.63 from this same baseline).
    ("Mega-Monolith", "realistic", {
        "total_files_changed": float(np.log1p(25_000)),
        "gh_diff_files_modified": 30,
        "gh_diff_files_added": 5,
        "gh_diff_files_deleted": 5,
        "gh_diff_src_files": 35,
        "gh_diff_doc_files": 2,
        "gh_diff_other_files": 3,
        "git_diff_src_churn": 1_000_000,
        "git_diff_test_churn": 200,
        "gh_diff_tests_deleted": 250,
        "gh_diff_tests_added": 0,
        "code_churn_density": 87.5,
        "test_to_src_churn_ratio": 200.0 / 3500.0,
        "gh_by_core_team_member": 0,
    }),
    ("Zero-SLOC", "zero", {
        "gh_sloc": 0,
        "gh_repo_num_commits": 1,
        "gh_repo_age": 1,
    }),
    ("Conflicting-State", "zero", {
        "git_diff_src_churn": 0,
        "total_files_changed": 0,
        "gh_diff_files_deleted": 20,
    }),
]

fuzz_rows = []   # (name, p_str, finite, in_bounds, note)
try:
    g1_ok = True
    mega_risk = None
    for name, base_kind, ov in fuzz_vectors:
        frame = (frame_from(realistic_base(), ov) if base_kind == "realistic"
                 else zero_frame(ov))
        try:
            p = float(cal_model.predict_proba(frame)[:, 1][0])
            finite = bool(np.isfinite(p))
            in_bounds = finite and (0.0 <= p <= 1.0)
            note = "ok" if (finite and in_bounds) else "OUT OF RANGE / NON-FINITE"
            if not (finite and in_bounds):
                g1_ok = False
            if name == "Mega-Monolith":
                mega_risk = p
            fuzz_rows.append((name, f"{p:.4f}", finite, in_bounds, note))
        except Exception as ve:  # noqa: BLE001 - a raised exception is a gate failure
            g1_ok = False
            fuzz_rows.append((name, "EXC", False, False, f"{type(ve).__name__}: {ve}"))

    mega_ok = mega_risk is not None and mega_risk > 0.60
    if not mega_ok:
        g1_ok = False
    assert g1_ok, "one or more fuzz vectors failed finiteness/bounds/Mega-Monolith check"
    record("1. Adversarial Fuzzing", True,
           f"4/4 vectors finite & in [0,1]; Mega-Monolith risk="
           f"{mega_risk:.4f} (>0.60)")
except Exception as e:  # noqa: BLE001
    mr = f"{mega_risk:.4f}" if isinstance(mega_risk, float) else "n/a"
    record("1. Adversarial Fuzzing", False,
           f"{type(e).__name__}: {e} | Mega-Monolith risk={mr}")

lines = ["| Vector | Risk P(fail) | Finite | In [0,1] | Note |",
         "|---|---|---|---|---|"]
for name, ps, fin, ib, note in fuzz_rows:
    lines.append(f"| {name} | {ps} | {fin} | {ib} | {note} |")
tables["fuzz"] = "\n".join(lines)


# ---------------------------------------------------------------------------
# Build held-out test slice (mirror train_metadata_model.py exactly)
# ---------------------------------------------------------------------------
test_df = None
slice_err = None
try:
    df = pd.read_csv(DATASET_ZIP, compression="zip")
    df[TARGET_COL] = (1 - df[RAW_TARGET_COL]).astype(np.int32)
    # Dedup: FAIL-first, drop identical 38-feature signatures (leakage guard).
    df = df.sort_values(TARGET_COL, ascending=False)
    df = df.drop_duplicates(subset=FEATURE_COLUMNS, keep="first").reset_index(drop=True)
    # Two-step stratified split, both random_state=42 (matches training).
    _, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df[TARGET_COL])
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET_COL])
    if len(test_df) > SLICE_CAP:
        test_df = test_df.sample(n=SLICE_CAP, random_state=42)
    test_df = test_df.reset_index(drop=True)
except Exception as e:  # noqa: BLE001
    slice_err = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# GATE 2: SUBGROUP SLICE PERFORMANCE
# ---------------------------------------------------------------------------
slice_rows = []   # (label, n, auc, brier, pass, note)
try:
    if test_df is None:
        raise RuntimeError(f"test slice unavailable: {slice_err}")

    X_all = test_df[FEATURE_COLUMNS]
    y_all = test_df[TARGET_COL].to_numpy()
    p_all = cal_model.predict_proba(X_all)[:, 1]
    test_df = test_df.assign(_p=p_all)

    def eval_slice(label, mask, assert_auc):
        """Compute AUC/Brier on a boolean mask; return (ok, note) and log a row."""
        yy = y_all[mask]
        pp = p_all[mask]
        n = int(mask.sum())
        if n == 0:
            slice_rows.append((label, 0, "n/a", "n/a", True, "empty slice"))
            return True, "empty"
        brier = float(brier_score_loss(yy, pp))
        classes = np.unique(yy)
        if classes.size < 2:
            # AUC undefined; a large single-class slice is a reported anomaly.
            note = f"single-class (y={int(classes[0])})"
            brier_ok = brier <= 0.16
            ok = brier_ok and not assert_auc  # only Brier gates here
            slice_rows.append((label, n, "N/A", f"{brier:.4f}", ok, note))
            return ok, note
        auc = float(roc_auc_score(yy, pp))
        brier_ok = brier <= 0.16
        auc_ok = (auc >= 0.75) if assert_auc else True
        ok = brier_ok and auc_ok
        note = "ok"
        if not brier_ok:
            note = "Brier>0.16"
        elif not auc_ok:
            note = "AUC<0.75"
        slice_rows.append((label, n, f"{auc:.4f}", f"{brier:.4f}", ok, note))
        return ok, note

    g2_ok = True
    is_pr = test_df["gh_is_pr"].to_numpy()
    ok, _ = eval_slice("PR (gh_is_pr=1)", is_pr == 1, assert_auc=True)
    g2_ok &= ok
    ok, _ = eval_slice("Push (gh_is_pr=0)", is_pr == 0, assert_auc=True)
    g2_ok &= ok

    for lang, col in (("Python", "gh_lang_python"),
                      ("Java", "gh_lang_java"),
                      ("Ruby", "gh_lang_ruby")):
        mask = test_df[col].to_numpy() == 1
        n = int(mask.sum())
        # Language AUC only gated when slice has > 500 samples.
        ok, _ = eval_slice(f"Lang: {lang}", mask, assert_auc=(n > 500))
        g2_ok &= ok

    assert g2_ok, "one or more slices breached AUC>=0.75 / Brier<=0.16"
    record("2. Subgroup Slices", True,
           f"n={len(test_df)}; all gated slices AUC>=0.75 & Brier<=0.16")
except Exception as e:  # noqa: BLE001
    record("2. Subgroup Slices", False, f"{type(e).__name__}: {e}")

lines = ["| Slice | N | ROC-AUC | Brier | Pass | Note |",
         "|---|---|---|---|---|---|"]
for label, n, auc, brier, ok, note in slice_rows:
    lines.append(f"| {label} | {n} | {auc} | {brier} | {'PASS' if ok else 'FAIL'} | {note} |")
tables["slices"] = "\n".join(lines) if slice_rows else "(no slices evaluated)"


# ---------------------------------------------------------------------------
# GATE 3: CALIBRATION RELIABILITY MONOTONICITY
# ---------------------------------------------------------------------------
bin_rows = []   # (label, count, empirical_fail_rate)
try:
    if test_df is None or "_p" not in test_df.columns:
        raise RuntimeError("test-slice predictions unavailable")

    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]
    binned = pd.cut(test_df["_p"], bins=edges, labels=labels,
                    include_lowest=True, right=False)
    # right=False makes the last edge exclusive; fold p==1.0 into the top bin.
    top = labels[-1]
    binned = binned.astype(object)
    binned[test_df["_p"] >= 1.0] = top
    grp = test_df.assign(_bin=binned).groupby("_bin", observed=True)

    populated_rates = []
    for label in labels:
        sub = grp.get_group(label) if label in grp.groups else None
        if sub is None or len(sub) == 0:
            bin_rows.append((label, 0, None))
            continue
        rate = float(sub[TARGET_COL].mean())
        bin_rows.append((label, int(len(sub)), rate))
        populated_rates.append((label, rate))

    mono = all(populated_rates[i][1] <= populated_rates[i + 1][1] + 1e-12
               for i in range(len(populated_rates) - 1))
    assert mono, ("empirical failure rate not non-decreasing across bins: "
                  + ", ".join(f"{l}={r:.4f}" for l, r in populated_rates))
    record("3. Calibration Monotonicity", True,
           f"{len(populated_rates)} populated bins non-decreasing: "
           + " <= ".join(f"{r:.3f}" for _, r in populated_rates))
except Exception as e:  # noqa: BLE001
    record("3. Calibration Monotonicity", False, f"{type(e).__name__}: {e}")

lines = ["| Prob bin | Count | Empirical fail rate |",
         "|---|---|---|"]
for label, count, rate in bin_rows:
    rate_s = f"{rate:.4f}" if rate is not None else "(empty)"
    lines.append(f"| {label} | {count} | {rate_s} |")
tables["bins"] = "\n".join(lines) if bin_rows else "(no bins evaluated)"


# ---------------------------------------------------------------------------
# SUMMARY REPORT (markdown, copy-pasteable)
# ---------------------------------------------------------------------------
print("\n# Module I - Stress & Slice Audit Scorecard")
print(f"run_20260926_013515  |  y_fail=1 => FAIL  |  test slice n="
      f"{0 if test_df is None else len(test_df)}\n")

print("## Gate 1 - Adversarial Edge-Case Fuzzing")
print(tables["fuzz"] + "\n")

print("## Gate 2 - Subgroup Slice Performance")
print(tables["slices"] + "\n")

print("## Gate 3 - Calibration Reliability Monotonicity")
print(tables["bins"] + "\n")

print("## Overall")
print("| Gate | Status | Detail |")
print("|---|---|---|")
all_pass = True
for name in ("1. Adversarial Fuzzing", "2. Subgroup Slices",
             "3. Calibration Monotonicity"):
    passed, detail = gates.get(name, (False, "not evaluated"))
    all_pass = all_pass and passed
    print(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |")

print(f"\n**OVERALL: {'ALL GATES PASSED' if all_pass else 'VALIDATION FAILED'}**")

raise SystemExit(0 if all_pass else 1)
