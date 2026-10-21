# ADHD-200 raw-fMRI availability and feasibility audit

Audit date: 2026-10-05. Candidate denominator reconstructed from the official phenotype/CC200 match manifest: **781 diagnosis-valid subjects** lacking a compatible CC200 derivative (780 without derivative plus 8191384 without a compatible variant).

## Findings

- S3 subject-prefix listings completed: **781/781**; listing failures: **0**.
- Resting-state BOLD listed: **{summary["rest_bold_available"]}**.
- Anatomical T1w/MPRAGE listed: **{summary["anatomical_available"]}**.
- Both listed: **{summary["both_available"]}**.
- Potentially processable (rest BOLD + anatomy + applicable sidecars/TR + site-level BIDS metadata): **{len(ready)}**. This is an availability classification, not a QC pass.
- Potentially processable diagnosis groups: **304 ADHD** and **471 Typically Developing Controls**.
- Sites represented among potentially processable subjects: KKI (59), NYU (251), NeuroIMAGE (54), OHSU (99), Peking_1 (98), Peking_2 (47), Peking_3 (31), Pittsburgh (77), WashU (59).
- Potential final cohort if every candidate subsequently passes preprocessing and CC200 QC: **928** (153 existing + 775 potential additions). Do not call this a usable cohort yet.

## Site summary

| Site | Candidates | RS-fMRI | Anatomy | Both | Ready | Missing/incomplete | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| KKI | 59 | 59 | 59 | 59 | 59 | 0 | 0 |
| NYU | 257 | 251 | 257 | 251 | 251 | 6 | 0 |
| NeuroIMAGE | 54 | 54 | 54 | 54 | 54 | 0 | 0 |
| OHSU | 99 | 99 | 99 | 99 | 99 | 0 | 0 |
| Peking_1 | 98 | 98 | 98 | 98 | 98 | 0 | 0 |
| Peking_2 | 47 | 47 | 47 | 47 | 47 | 0 | 0 |
| Peking_3 | 31 | 31 | 31 | 31 | 31 | 0 | 0 |
| Pittsburgh | 77 | 77 | 77 | 77 | 77 | 0 | 0 |
| WashU | 59 | 59 | 59 | 59 | 59 | 0 | 0 |

## Preprocessing compatibility

Target requested: `pipeline_adhd200-benchmark__freq-filter`, 0.01–0.1 Hz, global signal regression OFF, CC200. The raw BIDS objects do not have prior C-PAC processing status, so all candidates require preprocessing; no raw image can establish the eventual derivative configuration match until processed.

Among the 775 potentially processable subjects, inherited BIDS metadata provided a positive rest task label and numeric TR for each listed rest BOLD. Observed TR values (subject counts): 1.5 s: 69, 1.96 s: 54, 2.0 s: 427, 2.5 s: 217, 3 s: 1, 3.0 s: 7. Acquisition/TR variation is present across sites and needs to be represented correctly in the C-PAC inputs/configuration.

No NIfTI data or headers were fetched. Therefore image dimensions, affine/orientation, payload integrity, missing volumes, and anatomical/BOLD alignment remain **unknown**. These are explicit preprocessing compatibility checks, not inferred from file presence. No event/task files were listed in the subject inventories; all BOLD matches are rest task. Six NYU subjects (five ADHD, one control) have T1w but no rest BOLD listed and are excluded from potentially processable count.

## Exact listed raw download size for the potentially processable set

- Functional rest BOLD: 70,133,625,307 bytes (65.32 GiB)
- Anatomical T1w/MPRAGE: 7,201,739,516 bytes (6.71 GiB)
- Subject-level metadata: 285 bytes (0.00 GiB)
- Deduplicated shared site metadata and inherited JSON sidecars: 163,238 bytes (0.00 GiB) across 18 objects
- **Total: 77,335,528,346 bytes (72.02 GiB)**

All values are exact S3 object `Size` metadata for the candidate subjects classified READY, plus the deduplicated site-root files required to interpret them. This covers 775 subjects only; it excludes the six without rest BOLD and does not download any object. The full subject-prefix inventory across all 781 sums to 77,373,970,470 bytes (72.06 GiB) (includes the six subjects’ available files).

## Feasibility

The 775-subject workload is a substantial multi-site C-PAC run. At audit time the machine had 8 GiB RAM, 8 CPU cores, and about 341 GiB free disk. The listed raw set fits within available disk by byte count, but that does not account for temporary processing data, work directories, or derivative outputs. Expected C-PAC output size is unknown from the available evidence. **Runtime not yet benchmarked.** The machine’s 8 GiB RAM makes running all candidates concurrently imprudent; processing feasibility should be checked on a small representative pilot after a cohort is selected.

## Priority and limitations

The file `metadata/raw_processing_priority.csv` ranks every subject. It puts READY subjects first, then uses deterministic greedy balancing across ADHD/control and site representation. It does not use model performance. This is a planning list, not a decision that all 775 should be processed.

The status READY_FOR_PREPROCESSING means S3 lists rest BOLD and T1w objects, their applicable JSON sidecars and valid TR/task metadata, and site participants/dataset metadata. It does not certify NIfTI readability, image dimensions, motion/QC, successful preprocessing, or CC200 quality. No raw MRI GET, preprocessing, feature extraction, or model training was performed. The original NeuroADHD project and 153-subject feature table were not modified.
