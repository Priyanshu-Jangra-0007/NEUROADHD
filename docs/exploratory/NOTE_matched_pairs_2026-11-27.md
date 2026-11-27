# Note on Realized vs. Pre-Declared Motion-Matched Cohort Counts (2026-11-27)

## Summary

- **Pre-declared counts (`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`, Section 3.3):** `215` post-exclusion pairs (`N = 430`: Peking `73`, KKI `19`, NeuroIMAGE `16`, OHSU `21`, NYU `86`) and `226` pre-exclusion pairs (`N = 452`: Peking `73`, KKI `20`, NeuroIMAGE `16`, OHSU `28`, NYU `89`).
- **Realized counts (`scripts/run_exploratory.py`, reported in `docs/exploratory/RESULTS_SUMMARY.md`):** `207` post-exclusion pairs (`N = 414`: Peking `71`, KKI `19`, NeuroIMAGE `16`, OHSU `21`, NYU `80`) and `219` pre-exclusion pairs (`N = 438`: Peking `71`, KKI `20`, NeuroIMAGE `16`, OHSU `28`, NYU `84`).
- **Scope and Primary Analysis Status:** Both `matched_215` (`N = 414`, `207` pairs) and `matched_226` (`N = 438`, `219` pairs) are **sensitivity cohorts only**. The **primary exploratory analysis (`N = 564` clean LOIO cohort)**—as well as the `N = 620` all-scans sensitivity cohort and `CFG_13`–`CFG_16` multi-run sub-analysis—is **completely unaffected**.

## Root Cause (Code Inspection, No Pool or Exclusion Difference)

1. **Cohort pools, Section 5.3 exclusions, and FD values did not change:**
   - `load_label_free_inputs()` in `scripts/run_exploratory.py` reproduces the exact planned cohorts per institution:
     - Full scanned pool (`N = 620`): KKI `83` (`61` TDC, `22` ADHD), NYU `216` (`98` TDC, `118` ADHD), NeuroIMAGE `48` (`23` TDC, `25` ADHD), OHSU `79` (`42` TDC, `37` ADHD), Peking `194` (`116` TDC, `78` ADHD).
     - Clean primary pool after Section 5.3 exclusion (`N = 564`, `56` excluded): KKI `79` (`60` TDC, `19` ADHD), NYU `196` (`91` TDC, `105` ADHD), NeuroIMAGE `42` (`23` TDC, `19` ADHD), OHSU `58` (`33` TDC, `25` ADHD), Peking `189` (`114` TDC, `75` ADHD).
   - `dev_niak_fd_metrics.csv` `mean_fd_mm` values match `inputs.motion_covariates[:, 0]` exactly (`max diff = 0.0`), and NYU `rest_1` FD uses the same Study 2 Power FD computation from `intermediate_nyu.tar` (`DUMMY_FRAMES_DROPPED = 3`).
2. **All differences come from Peking (`-2` pairs) and NYU (`-6` clean / `-5` pre-exclusion pairs):**
   - KKI (`19`/`20`), NeuroIMAGE (`16`/`16`), and OHSU (`21`/`28`) matched counts are identical to the plan.
3. **Greedy `(mean_fd_mm ascending, normalized_subject_id ascending)` ordering in commit `8b5f1d2`:**
   - Originally (`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`, Section 3.3), the `215` and `226` counts were noted under input-order greedy matching (cf. Section 3.2, comparing unsorted CSV vs. `adhd200_preprocessed_phenotypics.tsv` row order).
   - Commit `8b5f1d2` updated `match_subjects_on_motion_within_site()` in `scripts/run_exploratory.py` to remove row-order sensitivity by sorting ADHD rows by `(mean_fd_mm ascending, normalized_subject_id ascending)` before matching greedily without replacement to the nearest available TDC within `caliper_mm = 0.05` (with deterministic tie-breaking).
   - Under the repository's label-blind pre-freeze guard, the real `DX` labels were not opened to recompute Section 3.3's illustrative counts after adding the `(mean_fd_mm ascending, normalized_subject_id ascending)` sort prior to execution under `exploratory-frozen`.
   - Algorithmically, ascending-FD greedy nearest-neighbor matching is suboptimal for pair cardinality: processing lower-FD ADHD subjects first and claiming the nearest TDC within `±0.05 mm` leaves lower-FD TDCs unmatched while exhausting mid-FD TDCs needed by later higher-FD ADHD subjects (`3` unmatched ADHD subjects in Peking and `25` clean / `29` all unmatched ADHD subjects in NYU originally had eligible TDCs within `0.05 mm` prior to greedy claiming; maximum-cardinality bipartite matching on the exact same pools and FD values yields `224` clean and `236` pre-exclusion pairs).
