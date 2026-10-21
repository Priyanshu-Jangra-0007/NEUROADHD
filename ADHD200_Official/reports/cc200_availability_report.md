# Official ADHD-200 CC200 derivative availability and phenotype match

Audit date: 2026-10-05. No imaging object was downloaded or opened. S3 inventory is from anonymous `ListObjectsV2` metadata requests only. Existing NeuroADHD workspace files were read only.

## Results

- **Unique official CC200 derivative subjects:** 162 (at least one `roi_CC200.csv` in either audited C-PAC benchmark output).
- **CC200 ROI objects:** 647 across two pipelines; subject count is not object count because pipeline/configuration variants exist.
- **Subjects with valid diagnosis:** 155: **50 ADHD** (original subtype codes/labels pooled only for this count) and **105 typically developing controls**. Original phenotype labels remain unchanged in the CSV.
- **CC200 subjects without valid diagnosis:** 7; all 7 have phenotype IDs but Brown diagnosis is `pending`/blank.
- **CC200 subjects with no phenotype ID match:** 0.
- **Official phenotype IDs without CC200 in the two audited pipelines:** 780 with valid diagnosis, among 935 valid-diagnosis participant IDs in the local official BIDS phenotype tables.
- **Sites represented:** 10 site labels (eight official institutions, with Peking split as Peking_1/2/3 in the tables).
- **Potential cohort after applying the already documented exclusion:** 154 (includes the new candidate `8191384`; this is not final QC clearance).
- **Current NeuroADHD feature cohort:** 153 subjects. **New candidate absent from that feature table:** `8191384`.

## Subjects by site label

| Site label | CC200 subjects | Valid diagnosis | ADHD | Control | Missing/pending Dx | Already in current 153 | Potentially usable after known exclusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brown | 7 | 0 | 0 | 0 | 7 | 0 | 0 |
| KKI | 24 | 24 | 8 | 16 | 0 | 24 | 24 |
| NYU | 6 | 6 | 4 | 2 | 0 | 6 | 6 |
| NeuroIMAGE | 19 | 19 | 10 | 9 | 0 | 19 | 19 |
| OHSU | 14 | 14 | 0 | 14 | 0 | 14 | 14 |
| Peking_1 | 38 | 38 | 14 | 24 | 0 | 38 | 38 |
| Peking_2 | 20 | 20 | 10 | 10 | 0 | 20 | 20 |
| Peking_3 | 12 | 12 | 4 | 8 | 0 | 11 | 12 |
| Pittsburgh | 21 | 21 | 0 | 21 | 0 | 21 | 21 |
| WashU | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| TOTAL | 162 | 155 | 50 | 105 | 7 | 153 | 154 |

## Pipeline/object coverage

| C-PAC pipeline | CC200 ROI objects | Unique subjects |
|---|---:|---:|
| `pipeline_adhd200-benchmark` | 324 | 162 |
| `pipeline_adhd200-benchmark__freq-filter` | 323 | 162 |

The standard `pipeline_adhd200-benchmark` has two CC200 nuisance-regression variants (global signal off/on) for each of 162 subjects. The `pipeline_adhd200-benchmark__freq-filter` has 323 CC200 objects across 162 subjects: two variants for most subjects and a single variant for `8191384`. The exact object variant and path are recorded in `official_cc200_inventory.csv`.

Motion files (`FD.1D`, `motion_parameters.txt`) were found for all 162 subjects in both pipeline trees (324 files of each type across both trees). MRIQC report objects (`.json` and `.html`) were listed under `Outputs/mriqc/mriqc/`; the complete metadata listing contained 4,714 objects, with 644 report objects matching 161 of the 162 CC200 subjects (T1w and resting BOLD reports). `0026053` had no matched MRIQC report object in this listing. The local official per-site participant tables contain `qc_rest_1` values for 161 of 162 CC200 subjects: Pass=146, Questionable=12, -999=3, blank=1. `qc_anatomical_1` values are Pass=154, Questionable=4, -999=3, blank=1. The phenotype-match CSV preserves these fields. MRIQC reports were not downloaded or parsed, so no report-derived pass/fail conclusions are made.

## Phenotype match and interpretation

The match uses official per-site BIDS `participants.tsv` files already present read-only in `/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/data/adhd200/raw/phenotypes/`; Brown's `DX=pending` values are filled from the existing official Brown test-release phenotype CSV because Brown `participants.tsv` has no diagnosis column. The normalized numeric identifier was used for joining padded C-PAC IDs to phenotype IDs, with site labels retained. All 162 CC200 derivative subjects matched a phenotype participant ID, including `8191384` (Peking_3, typically developing). Seven Brown CC200 subjects have `pending` diagnosis; no binary label is assigned to them.

The audited local official phenotype tables contain 961 unique site/participant rows, of which 935 have a recognized diagnosis (359 ADHD subtypes combined for summary count; 576 controls), and 26 Brown rows are pending. There are 780 valid-diagnosis rows without a CC200 object in these two benchmark pipeline trees. This is a precise comparison to the local official BIDS phenotype table set inspected, not a claim that those tables represent every historical ADHD-200 competition roster or every possible derivative elsewhere in S3.

## Comparison with current NeuroADHD cohort

The existing feature table contains 153 subjects. Intersection with the S3 CC200 inventory is 153 subjects. The local prior mapping has 161 CC200 subjects; the official S3 inventory adds `8191384`, which has a Peking_3 phenotype record and a control diagnosis. It is absent from the current 153-row feature table and is the sole **new labeled CC200 candidate** in this match.

Previously excluded: `0015038` (WashU, control), skipped because **83 ROI time series were constant; Pearson correlations were undefined** according to `cc200_extraction_summary.json`. Seven Brown subjects (`0026022`, `0026024`, `0026052`, `0026053`, `0026054`, `0026055`, `0026057`) have CC200 outputs but pending diagnoses and were not in the current labeled feature cohort. The other 153 valid diagnosed CC200 subjects are already in the current feature table.

`8191384` is only potentially usable. Its CC200 numeric content was not inspected, and its available output variants do not fully match the current frequency-filtered pipeline variant: only global-signal-on was listed in the frequency-filtered tree. The standard benchmark tree has global-signal-off and global-signal-on objects. This configuration gap must be resolved before appending it to the existing cohort.

## Recommended CC200-only download list for the new candidate

The only new valid-diagnosis candidate absent from current features is `8191384`. Available CC200 objects are:

- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark__freq-filter/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global1.motion1.quadratic1.gm0.compcor1.csf0/_bandpass_freqs_0.01.0.1/_mask_CC200/roi_CC200.csv` — 472127 bytes (global1)
- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf0/_mask_CC200/roi_CC200.csv` — 478550 bytes (global0)
- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global1.motion1.quadratic1.gm0.compcor1.csf0/_mask_CC200/roi_CC200.csv` — 476668 bytes (global1)

This is a recommendation list of derivative objects only. It excludes raw BIDS and includes alternatives that should be reviewed for configuration compatibility. For strict continuation of the current feature pipeline, the frequency-filtered object is the closest named pipeline but is global-signal-on; do not treat it as configuration-identical to existing features without checking their exact provenance.

## Exact cohort counts and limits

- 162 CC200 subjects found; 155 diagnosis-valid (50 ADHD, 105 controls); 7 pending diagnosis.
- 154 remain after excluding the one previously documented constant-signal subject. Of these, 153 are already represented in current features and one is a new candidate.
- **154 is the exact inventory/label-based potential cohort count before download. It is not the final QC-cleared count.** The new candidate's ROI values, time-series properties, and MRIQC report contents remain unread; its usability cannot be confirmed until a later authorized derivative/QC step.
- No raw BIDS, CC200 ROI, MRIQC report, phenotype file, or other object was downloaded in this audit. No preprocessing or modeling was performed. The existing NeuroADHD project was not modified.

## Source references

- Official project: https://fcon_1000.projects.nitrc.org/indi/adhd200/
- Official S3 instructions and outputs structure: https://fcon_1000.projects.nitrc.org/indi/s3/index.html
- Official download instructions: https://fcon_1000.projects.nitrc.org/indi/adhd200/download_scripts/Download_Instructions_ADHD200.pdf
- Official phenotype key: https://fcon_1000.projects.nitrc.org/indi/adhd200/general/ADHD-200_PhenotypicKey.pdf
- Audited roots: `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/`, `.../pipeline_adhd200-benchmark__freq-filter/`, and `s3://fcp-indi/data/Projects/ADHD200/Outputs/mriqc/mriqc/`.
