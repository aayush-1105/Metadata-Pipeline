# Module I (Metadata Predictor) - TreeSHAP Attribution Scorecard

**Artifact Run:** `run_20260926_013515`  
**Attribution Method:** TreeSHAP (`shap.TreeExplainer`)  
**SHAP Output Space:** log-odds (raw margin, positive/FAIL class)  
**Sample Size:** 5,000 stratified test build events  
**Feature Count:** 38

**Baseline Expected Value E[f(x)]:** `-1.406981` (log-odds)  ->  `19.67%` baseline failure probability

## Global Feature Attribution - Mean Absolute SHAP (descending)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | log_team_size | 0.223177 |
| 2 | gh_repo_num_commits | 0.206223 |
| 3 | gh_sloc | 0.201125 |
| 4 | gh_test_lines_per_kloc | 0.193324 |
| 5 | gh_test_cases_per_kloc | 0.175443 |
| 6 | git_prev_commit_resolution_status_merge_found | 0.165006 |
| 7 | gh_asserts_cases_per_kloc | 0.163087 |
| 8 | tr_build_number | 0.159799 |
| 9 | gh_repo_age | 0.158400 |
| 10 | gh_lang_ruby | 0.094200 |
| 11 | log_built_commits | 0.054185 |
| 12 | gh_num_commits_on_files_touched | 0.049629 |
| 13 | gh_pr_created_at_year | 0.041151 |
| 14 | has_previous_build | 0.039901 |
| 15 | total_files_changed | 0.038996 |
| 16 | gh_diff_files_added | 0.035784 |
| 17 | gh_diff_other_files | 0.032085 |
| 18 | gh_lang_python | 0.026914 |
| 19 | gh_lang_java | 0.020992 |
| 20 | gh_pr_created_at_month | 0.020383 |
| 21 | gh_diff_files_modified | 0.013637 |
| 22 | gh_is_pr | 0.012564 |
| 23 | gh_by_core_team_member | 0.010702 |
| 24 | pr_dayofweek | 0.010283 |
| 25 | gh_diff_tests_deleted | 0.007930 |
| 26 | gh_diff_files_deleted | 0.006304 |
| 27 | gh_num_issue_comments | 0.006088 |
| 28 | code_churn_density | 0.005466 |
| 29 | gh_num_pr_comments | 0.003378 |
| 30 | git_diff_src_churn | 0.003097 |
| 31 | git_prev_commit_resolution_status_no_previous_build | 0.003014 |
| 32 | gh_diff_src_files | 0.002644 |
| 33 | gh_num_commit_comments | 0.002354 |
| 34 | gh_diff_tests_added | 0.001025 |
| 35 | gh_diff_doc_files | 0.000012 |
| 36 | git_merged_with_present | 0.000000 |
| 37 | test_to_src_churn_ratio | 0.000000 |
| 38 | git_diff_test_churn | 0.000000 |
