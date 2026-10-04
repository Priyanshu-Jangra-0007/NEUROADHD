# ADHD-200 dataset inspection and label validation

Inspection date (UTC): 2026-09-29

## CC200 schema

- Subjects: 161
- ROI rows per file: 200 (header excluded); ROI IDs 1–200 are identical across files.
- Timepoints per subject: 77–261; 13 distinct lengths.
- Missing cells: 0; nonnumeric cells: 0.
- CSV orientation: rows are ROIs; columns after `node/volume` are volume/timepoint indices.

## Label validation

- Matched: 161/161; unmatched: 0.
- Missing/pending DX: 7; conflicting mapped labels: 0.
- Site available: 161/161.
- Usable labeled subjects: 154; ADHD: 50; Control: 104.
- Binary encoding: Control=0, ADHD=1. Diagnosis subtype codes 1/2/3 are grouped into ADHD per the official ADHD-200 phenotypic key.
- The seven Brown subjects with DX `pending` (and blank `dx` in Brown participants.tsv) are excluded from usable labels. See the CSV mapping for IDs.

Raw source files were read only. No features, splits, or models were generated.
