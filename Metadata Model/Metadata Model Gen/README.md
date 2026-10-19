# Module I — Metadata Predictor (Model Generation)

This folder is the **model-generation half** of an AI-powered CI/CD build-failure
predictor. It trains a LightGBM classifier on 38 build/repo metadata features,
calibrates its probabilities, evaluates it with leakage-resistant metrics, runs a
forensic data-leakage audit, validates the trained artifacts, and renders an
executive report (PDF + Word) with SHAP explanations.

The model predicts `y_fail` — the probability that a CI build **fails**
(`y_fail = 1 - target_binary`, so `1 = FAIL`, `0 = PASS`).
`predict_proba(...)[:, 1]` is the failure risk.

---

## ⚠️ Before you run anything

1. **You need the dataset — it is NOT in this zip.**
   Every script expects a zipped CSV at:
   ```
   F:\Capstone\Dataset\Pre-Processed Metadata Dataset.zip
   ```
   containing `Pre-Processed Metadata Dataset.csv` with the 38 feature columns
   plus `target_binary`. Supply your own copy and update the path (see below).

2. **Paths are hardcoded to `F:\Capstone\...`.**
   Open each script and edit the constants near the top before running:
   - `DATASET_ZIP_PATH` / `DATASET_ZIP` — where the dataset lives.
   - `SCRIPT_ROOT` / `MODELS_ROOT` / `RUN_DIR` — where scripts and model
     artifacts are read from and written to.
   The test/report scripts point at a specific run folder
   (`RUN_DIR = ...\Models\run_20260926_013515`); repoint this to a run you
   generate with `train_metadata_model.py`.

3. **Model artifacts are NOT included.** Only the code ships here. Run the
   training script first to produce the `.joblib` models the other scripts load.

---

## Requirements

- Python 3.11+ (developed on 3.13)
- Packages:
  ```
  pip install lightgbm scikit-learn pandas numpy joblib shap matplotlib python-docx
  ```
  `python-docx` is only needed for `generate_word_report.py`.

---

## What each file does

### Core pipeline
- **`train_metadata_model.py`** — the main entry point. Loads the dataset,
  deduplicates matrix-job duplicates to prevent row leakage, does a stratified
  70/15/15 split, trains LightGBM with AUC early stopping, calibrates with both
  Platt (sigmoid) and isotonic regression, evaluates on the held-out test set
  (MCC, Brier, ROC/PR-AUC, composite fitness, threshold diagnostics), builds a
  SHAP TreeExplainer, and writes all artifacts + `scorecard.json` / `scorecard.md`
  into a new timestamped `Models\run_<timestamp>\` folder.

### Auditing & validation (read-only — never modify model files)
- **`audit_leakage.py`** — independent forensic audit for data leakage. Reproduces
  the production split, measures train/test build-event and reconstructed-repo
  overlap, runs feature ablations, and compares honest splits (group-by-build-event,
  global-temporal, within-repo-temporal). Prints a verdict and writes
  `audit_results.json`.
- **`test_model_production.py`** — production QA gate. Checks artifact integrity
  and 38-feature schema order, behavioral invariants (destructive changes should
  score riskier than pristine ones), SHAP output shape, and single-row inference
  latency (mean < 15 ms).
- **`test_model_stress_and_slices.py`** — robustness/fairness/calibration audit.
  Adversarial edge-case fuzzing (no NaN/Inf, probs in [0,1]), per-language and
  PR/Push subgroup performance (ROC-AUC / Brier thresholds), and calibration
  monotonicity across probability bins.

### Explainability & reporting
- **`run_shap_analysis.py`** — dedicated TreeSHAP pipeline. Reproduces the test
  split bit-for-bit, samples 5,000 build events, computes SHAP values, writes
  `shap_scorecard.json` / `.md`, and renders beeswarm + high/low-risk waterfall
  PNGs (DPI 300) into the run's `exports\` folder.
- **`generate_pdf_report.py`** — builds a multi-page executive PDF from the run
  artifacts (`scorecard.json`, threshold cache, SHAP scorecard + figures), in
  plain-English terminology for a non-specialist reader.
- **`generate_word_report.py`** — same report as an editable Microsoft Word
  `.docx`. Text/tables are native Word objects; only charts are embedded images.
  Reads the same artifacts so the Word and PDF stay numerically identical.

### Misc
- **`test.py`** — empty placeholder / scratch file.
- **`run.log`, `audit_run.log`** — sample run logs from previous executions
  (reference only).
- **`__pycache__/`** — Python bytecode cache (safe to ignore/delete; excluded
  from this zip).

---

## How to run (typical order)

```bash
# 1. Train, calibrate, evaluate — creates Models\run_<timestamp>\ with artifacts
python train_metadata_model.py

# 2. (optional) Confirm the split isn't leaking signal
python audit_leakage.py

# --- point RUN_DIR in the scripts below at the run folder from step 1 ---

# 3. Validate the trained artifacts
python test_model_production.py
python test_model_stress_and_slices.py

# 4. Generate SHAP artifacts (needed before the reports)
python run_shap_analysis.py

# 5. Build the executive reports
python generate_pdf_report.py
python generate_word_report.py
```

---

## Feature schema (38 features)

The exact ordered feature list lives at the top of `train_metadata_model.py`
(`FEATURE_COLUMNS`) and is duplicated in every other script — they must match
exactly, or the test split won't reproduce. If you change features, change them
everywhere.
