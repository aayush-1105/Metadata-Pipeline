# Repository Metadata Feature Extractor

A command-line tool that extracts **39 TravisTorrent-aligned metadata features**
from any Git repository commit and writes them as rows in a CSV. The output is
designed as a ready-to-train feature matrix for **machine-learning models that
predict CI build failures** (`target_binary` is the label; the other 38 columns
are predictors).

> **For an AI/agent reader:** this document is the source of truth for the
> project's intent, architecture, feature semantics, and gotchas. You should be
> able to understand and safely modify the project from this README alone. The
> canonical output contract is the ordered list in `schema/columns.py`; every
> extractor returns a superset of those keys and the normalizer selects/orders
> them.

---

## 1. Why this exists

Predicting whether a commit will break the build is a classic software-analytics
task, popularized by the **TravisTorrent** dataset. Training such a model needs a
consistent feature vector per commit: churn, test activity, project/social
signals, and CI history. This tool reconstructs those features directly from a
repository + the GitHub API, so you can build a labeled dataset from *any* repo
without depending on the original TravisTorrent dumps.

Each row = one commit. `target_binary = 1` means that commit's CI build
concluded in **failure**; `0` means success (or unknown/no run).

---

## 2. How it works (pipeline)

```
repo URL or local path
        │
        ├─ clone to temp dir (URL)  ── or ── use path as-is (local)
        │
        ├─ resolve branch tip (default: main) and take N most-recent commits
        │
        └─ for each commit:  (checked out so file-tree metrics match the commit)
               ├─ core/git_engine.py     → diff churn + repo-history metrics
               ├─ core/static_analyzer.py→ SLOC, test density, language mix
               ├─ core/github_api.py     → PR / comment / team signals (gh api)
               ├─ core/ci_analyzer.py    → CI build history + target label (gh)
               └─ schema/normalizer.py   → select & order the 39 columns → row
        │
        └─ write rows to CSV (append by default, or overwrite)
```

---

## 3. Project layout

```
Metadata_Extractor/
├── main.py                 # CLI entry point + per-commit orchestration
├── core/
│   ├── git_engine.py       # git-diff churn + repository history metrics
│   ├── static_analyzer.py  # filesystem walk: SLOC, tests, language mix
│   ├── github_api.py       # GitHub REST via `gh api`: PR/comment/team signals
│   └── ci_analyzer.py      # GitHub Actions runs via `gh run list`: CI history + label
├── schema/
│   ├── columns.py          # THE 39-column output contract (order matters)
│   └── normalizer.py       # dict -> single-row DataFrame, typed defaults
├── requirements.txt        # pandas, requests
└── README.md
```

Extractors return plain dicts and may include extra intermediate keys (e.g.
`gh_team_size`, `git_merged_with`); `normalizer.normalize_features` keeps only
the keys listed in `columns.COLUMNS`, in that exact order, filling missing ones
with typed defaults (0 for numeric/binary, "" otherwise).

---

## 4. Prerequisites

- **Python 3.9+** with deps: `pip install -r requirements.txt` (pandas, requests).
- **Git** on `PATH` (all churn/history features shell out to `git`).
- **GitHub CLI (`gh`) installed and authenticated**: `gh auth login`.
  The GitHub- and CI-derived features call `gh api` and `gh run list`. Without
  auth (or for a local repo whose commits aren't on a GitHub remote) those
  fields gracefully fall back to `0`/empty instead of failing.

---

## 5. Usage

```bash
python main.py --repo https://github.com/pallets/flask --output features.csv
```

By default this extracts the **latest commit on `main`** and **appends** one row
to the CSV.

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | (required) | GitHub URL (cloned to a temp dir) or local path |
| `--branch` | `main` | Branch to read commits from; falls back to HEAD if absent |
| `-n`, `--num-commits` | `1` | Number of most-recent commits to extract (one row each) |
| `--commit` | branch tip | Start from a specific commit SHA instead of the branch tip |
| `--write-mode` | `append` | `append` (header written only for a new/empty file) or `overwrite` |
| `--output` | `features.csv` | Output CSV path |

Examples:
```bash
# last 10 commits on main, appended to the dataset
python main.py --repo https://github.com/pallets/flask -n 10

# last 5 commits on another branch, replacing the file
python main.py --repo https://github.com/pallets/flask --branch develop -n 5 --write-mode overwrite

# a single pinned commit from a local checkout
python main.py --repo . --commit 3a46438 --write-mode overwrite
```

---

## 6. The 39 features

Output order is fixed by `schema/columns.py`. `type`: **bin** = 0/1, **int** =
count, **float** = ratio/derived. "Source" is the module that produces it.

### Diff / churn — `core/git_engine.py` (`get_churn_metrics`)
Computed from `git diff` between the commit and its parent (`sha~1`; the
empty-tree for a root commit). A file is a *test* file if its path contains
`test`/`spec`; source vs doc vs other is decided by extension.

| Feature | Type | Meaning |
|---------|------|---------|
| `git_diff_src_churn` | int | added+deleted lines across source files |
| `git_diff_test_churn` | int | added+deleted lines across test files |
| `test_to_src_churn_ratio` | float | `test_churn / src_churn` (0 if no src churn) |
| `gh_diff_files_added` | int | files with status `A` |
| `gh_diff_files_deleted` | int | files with status `D` |
| `gh_diff_files_modified` | int | files with status `M` |
| `gh_diff_tests_added` | int | lines added in test files |
| `gh_diff_tests_deleted` | int | lines deleted in test files |
| `gh_diff_src_files` | int | count of changed source files |
| `gh_diff_doc_files` | int | count of changed doc files (`.md/.rst/.txt/...`) |
| `gh_diff_other_files` | int | count of other changed files |
| `total_files_changed` | int | total files in the diff |
| `code_churn_density` | float | `(added+deleted) / total_files_changed` |
| `gh_num_commits_on_files_touched` | int | commit count reachable for the touched files (first 10) |

### Repository history — `core/git_engine.py` (`get_repo_git_metrics`)
| Feature | Type | Meaning |
|---------|------|---------|
| `gh_repo_num_commits` | int | commits reachable from this commit |
| `gh_repo_age` | int | days between first commit and this commit |
| `git_merged_with_present` | bin | 1 if the commit is a merge (has a 2nd parent) |

### Static analysis — `core/static_analyzer.py`
Walks the checked-out tree (ignoring `.git`, `node_modules`, venvs, build dirs),
counting non-comment lines in `.py/.java/.rb` files. Test files are those whose
path contains `test`/`spec`; KLOC = SLOC/1000.

| Feature | Type | Meaning |
|---------|------|---------|
| `gh_sloc` | int | non-comment source lines of code |
| `gh_test_lines_per_kloc` | float | test-file lines per KLOC |
| `gh_test_cases_per_kloc` | float | test cases per KLOC (`def test`, `@Test`, `public void test`, `it "..."`) |
| `gh_asserts_cases_per_kloc` | float | lines containing `assert` per KLOC |
| `gh_lang_java` | bin | Java is the dominant language |
| `gh_lang_python` | bin | Python is the dominant language |
| `gh_lang_ruby` | bin | Ruby is the dominant language |

### GitHub / social — `core/github_api.py` (`gh api`)
Looks up the pull request associated with the commit, the commit object, and the
contributor list.

| Feature | Type | Meaning |
|---------|------|---------|
| `gh_is_pr` | bin | 1 if the commit belongs to a pull request |
| `gh_pr_created_at_year` | int | year the PR was opened (0 if not a PR) |
| `gh_pr_created_at_month` | int | month the PR was opened (0 if not a PR) |
| `pr_dayofweek` | int | weekday (Mon=0..Sun=6) of the PR date; falls back to commit date |
| `gh_num_issue_comments` | int | issue-conversation comments on the PR |
| `gh_num_pr_comments` | int | issue comments + code-review comments |
| `gh_num_commit_comments` | int | comments on the commit object |
| `log_team_size` | float | `log1p(number of contributors)` |
| `gh_by_core_team_member` | bin | placeholder: assumed core author (defaults 1) |

### CI history + label — `core/ci_analyzer.py` (`gh run list`)
Reads recent GitHub Actions runs and matches the commit's run.

| Feature | Type | Meaning |
|---------|------|---------|
| `tr_build_number` | int | run number of the commit's CI build |
| `has_previous_build` | bin | 1 if an earlier CI run exists |
| `log_built_commits` | float | `log1p(number of built commits seen)` |
| `git_prev_commit_resolution_status_merge_found` | bin | 1 if a previous build was resolved |
| `git_prev_commit_resolution_status_no_previous_build` | bin | 1 if no previous build exists |
| **`target_binary`** | bin | **LABEL:** 1 if the build concluded in failure, else 0 |

---

## 7. Output & write modes

- One CSV row per commit, 39 columns in `columns.py` order.
- `--write-mode append` (default): rows are appended; the header line is written
  only when the file is new/empty, so repeated runs **accumulate a dataset**.
- `--write-mode overwrite`: the file is truncated and rewritten each run.
- Feed the resulting CSV straight into pandas/scikit-learn; split off
  `target_binary` as `y` and use the remaining 38 columns as `X`.

---

## 8. Caveats & design notes

- **Windows/cmd caret bug (fixed):** parent lookup uses `sha~1`, not `sha^` —
  under `shell=True` on Windows the caret is cmd.exe's escape char and was
  silently stripped, making every diff compare a commit to itself (all-zero
  churn). Keep using `~1` in any new git calls.
- **Static features require a checkout:** each target commit is checked out so
  SLOC/test/language metrics match that commit's tree, then the original ref is
  restored. For a **local repo with uncommitted changes** the checkout is
  skipped (with a warning) and those metrics reflect the current working tree.
- **GitHub features need the commit on a GitHub remote + `gh` auth.** Running
  against a purely local repo leaves the `gh_*`/`tr_*`/`target_binary` fields at
  their `0` defaults — expected, not an error.
- `gh_by_core_team_member` is currently a constant placeholder (1); refine it
  against the contributor list if you need a real signal.

---

## 9. Extending the feature set

1. Compute the value inside the relevant `core/*.py` extractor and include it in
   the returned dict.
2. Add the key to `schema/columns.py` in the position you want in the CSV.
3. If it can be missing, confirm `normalizer.normalize_features` defaults it
   sensibly (numeric/binary → 0). Extend `numeric_tokens`/`binary_cols` there if
   the name doesn't already match.

That's the whole contract: **extractors fill a dict, `columns.py` defines the
output shape, `normalizer.py` enforces it.**



