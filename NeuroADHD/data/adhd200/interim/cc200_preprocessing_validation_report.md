# CC200 preprocessing validation report

- Input matrix: 153 × 19900 ROI-pair features; metadata and target are separate.
- Input X shape by partition: train 114 × 19,900; validation 19 × 19,900; test 20 × 19,900.
- Output shape by partition: train 114 × 100; validation 19 × 100; test 20 × 100.
- Sequence: median imputation → zero-variance filtering → standard scaling → ANOVA F-statistic SelectKBest(k=100).
- Every learned step is fitted only on X_train/y_train; validation and test are transform-only.
- Subject IDs preserved; no subject or site crosses partitions; all transformed values are finite.
- k=100 retained as a reasonable initial p≫n reduction; later k comparisons belong inside training-only CV.
- No classifier trained; no accuracy calculated.
- Synthetic/unit tests: 2 passed (training-fit invariance under extreme validation/test changes; overlapping subject IDs rejected).
