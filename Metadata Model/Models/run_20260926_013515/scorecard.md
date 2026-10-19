# Module I (Metadata Predictor) - Evaluation Scorecard

**Run Timestamp:** `20260926_013515`  
**Target Definition:** y_fail = 1 - target_binary (1 = FAIL, 0 = PASS)  
**Feature Count:** 38  
**Split Strategy:** deduplicated_stratified_70_15_15  
**Primary Calibrator:** sigmoid (Platt)  
**Best Iteration:** 1000

## Unexploitable Fitness Metrics (Test Set)

- **Composite Fitness Score [MCC@0.50 + (1 - Brier)]:** **1.426552**
- **Matthews Correlation Coefficient (MCC @ 0.50):** 0.542534
- **ROC-AUC:** 0.857884
- **PR-AUC:** 0.744879

## Dataset Demographics (Unique Build Events)

| Split | Total Builds | Fail Count | Fail % | Pass Count | Pass % | PR % |
|---|---|---|---|---|---|---|
| Train | 643883 | 161631 | 25.1025% | 482252 | 74.8975% | 18.0328% |
| Validation | 137975 | 34635 | 25.1024% | 103340 | 74.8976% | 17.9815% |
| Test | 137975 | 34635 | 25.1024% | 103340 | 74.8976% | 17.9764% |

**Total distinct build events across all splits:** 919833

## Calibration Metrics (raw vs calibrated)

| Scorer | Brier | ECE | MCC@0.50 | Fitness | Prob Min | Prob Max | Prob Mean | Prob Std |
|---|---|---|---|---|---|---|---|---|
| raw_lightgbm | 0.117027 | 0.023816 | 0.536076 | 1.419049 | 0.00027 | 0.999785 | 0.250205 | 0.240742 |
| calibrated_sigmoid | 0.115982 | 0.005171 | 0.542534 | 1.426552 | 4.3e-05 | 0.99998 | 0.249996 | 0.267727 |
| calibrated_isotonic | 0.115971 | 0.002268 | 0.540479 | 1.424508 | 0.0 | 1.0 | 0.24995 | 0.268897 |

- **Brier improvement (sigmoid vs raw):** 0.001045
- **Brier improvement (isotonic vs raw):** 0.001056

## Threshold Diagnostics (Confusion Matrices & Unexploitable MCC)

### raw_lightgbm

| Threshold | TN | FP | FN | TP | FPR | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 83109 | 20231 | 8965 | 25670 | 0.195771 | 0.559247 | 0.741158 | 0.637479 | 0.501899 |
| 0.5 | 99793 | 3547 | 18458 | 16177 | 0.034324 | 0.820168 | 0.467071 | 0.595191 | 0.536076 |
| 0.65 | 102337 | 1003 | 22857 | 11778 | 0.009706 | 0.921524 | 0.340061 | 0.496794 | 0.494081 |

### calibrated_sigmoid

| Threshold | TN | FP | FN | TP | FPR | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 83787 | 19553 | 9188 | 25447 | 0.18921 | 0.565489 | 0.734719 | 0.639091 | 0.50455 |
| 0.5 | 98373 | 4967 | 16904 | 17731 | 0.048065 | 0.78117 | 0.511939 | 0.618527 | 0.542534 |
| 0.65 | 101348 | 1992 | 20674 | 13961 | 0.019276 | 0.875133 | 0.403089 | 0.551949 | 0.520441 |

### calibrated_isotonic

| Threshold | TN | FP | FN | TP | FPR | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 81954 | 21386 | 8616 | 26019 | 0.206948 | 0.548866 | 0.751234 | 0.6343 | 0.496953 |
| 0.5 | 98964 | 4376 | 17510 | 17125 | 0.042346 | 0.796475 | 0.494442 | 0.610125 | 0.540479 |
| 0.65 | 101653 | 1687 | 21265 | 13370 | 0.016325 | 0.887959 | 0.386026 | 0.538115 | 0.514121 |

## Top 15 Features by LightGBM Split Gain

| Rank | Feature | Gain |
|---|---|---|
| 1 | gh_sloc | 321754.661 |
| 2 | gh_repo_age | 289318.3984 |
| 3 | tr_build_number | 270384.9075 |
| 4 | gh_repo_num_commits | 265633.4356 |
| 5 | log_team_size | 264031.8089 |
| 6 | gh_test_lines_per_kloc | 236331.0357 |
| 7 | gh_test_cases_per_kloc | 226694.6104 |
| 8 | gh_asserts_cases_per_kloc | 220563.6346 |
| 9 | git_prev_commit_resolution_status_merge_found | 58204.9243 |
| 10 | gh_num_commits_on_files_touched | 56986.135 |
| 11 | log_built_commits | 27001.1273 |
| 12 | gh_lang_ruby | 26165.4797 |
| 13 | total_files_changed | 19474.6391 |
| 14 | gh_lang_java | 19021.9916 |
| 15 | gh_lang_python | 18242.4616 |
