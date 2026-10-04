# CC200 split validation report

Strategy: `fixed_group_holdout_by_site`; partitions are assigned by whole acquisition site.

| Partition | Subjects | ADHD (1) | Control (0) | Sites |
|---|---:|---:|---:|---|
| Train | 114 | 30 | 84 | KKI, NYU, OHSU, Peking_1, Peking_3, Pittsburgh |
| Validation | 19 | 10 | 9 | NeuroIMAGE |
| Test | 20 | 10 | 10 | Peking_2 |

## Leakage checks

- Every subject assigned exactly once: **True**.
- Pairwise subject overlaps: `{"train_test": [], "train_validation": [], "validation_test": []}`.
- Sites are disjoint across train, validation, and test.
- Preprocessing pipeline is defined but unfitted; no test information was used to fit transforms.
- No model was trained and no accuracy was calculated.

## Preprocessing plan

Within each training fold: median imputation (if needed), remove zero-variance features, standardize, then select the top 100 features by ANOVA F-statistic. These learned steps must stay inside a complete estimator Pipeline during future cross-validation/tuning. No preprocessing was fit while creating these split manifests.

## Site-holdout rationale

NeuroIMAGE is the validation site (19 subjects; both classes) and Peking_2 is the final test site (20 subjects; balanced classes). The other six sites form training data. This provides a site-disjoint final evaluation while keeping both classes represented in validation and test. OHSU and Pittsburgh are control-only in this cohort and remain in training.
