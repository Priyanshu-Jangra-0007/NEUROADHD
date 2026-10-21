# 8191384 CC200 compatibility review

**Decision: STOP. Do not add 8191384 to the current feature cohort. No variant is exactly compatible.** No CC200 object was downloaded, no preprocessing or feature extraction was run, and no file in the NeuroADHD project was changed.

## Existing 153-subject cohort configuration

Source of truth inspected read-only: the NeuroADHD `manifest.csv`, local CC200 inputs and schema/QC summaries, feature metadata, and `src/features/extract_cc200_connectivity.py`. The manifest records official S3 source keys using `pipeline_adhd200-benchmark__freq-filter`, `global0`, `_bandpass_freqs_0.01.0.1`, and `_mask_CC200/roi_CC200.csv`. The local cohort’s 153 extracted inputs plus the one signal-quality exclusion (154 labeled inputs total) were matched against their recorded S3 global0 frequency-filtered variants: **154/154 local files matched S3 ETag-derived MD5 and Content-Length**. This establishes the object configuration for the current labeled cohort.

- C-PAC output pipeline identifier: `pipeline_adhd200-benchmark__freq-filter`. A C-PAC software version/release is **not recorded** in the inspected project files; none is inferred.
- Frequency filtering: **0.01–0.1 Hz**, encoded in the S3 path.
- Global signal regression: **off** (`global0`).
- Other path-encoded settings: CSF threshold 0.96; GM 0.7; WM 0.96; CompCor 5 components; selector pc10; `linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf0`.
- ROI extraction: C-PAC `roi_timeseries`, CC200 mask, CSV (`roi_CC200.csv`).
- Input shape: 200 ROI rows/nodes, ROI IDs 1–200 in order; first column is the node/volume identifier and following columns are timepoints. Existing extracted cohort timepoints vary from **77 to 261**; they are not a fixed row count. Existing local inspection reported no missing or nonnumeric input values.
- Feature method: Pearson correlation across timepoints, symmetrized correlation matrix, diagonal removed, upper triangle in row-major `(i,j)` order, then Fisher z `arctanh(r)` with epsilon 1e-7 clipping. Exactly 19,900 features; existing order begins `ROI_001__ROI_002` and ends `ROI_199__ROI_200`.
- Signal-quality rule: reject a subject if any ROI time series has population standard deviation exactly zero; also reject non-finite/missing input or non-finite correlation/Fisher-z values.
- File dimensions/QC of existing cohort: 153 extracted subjects; 200 ROI rows each; observed timepoints 77–261. One labeled input, `0015038`, failed because 83 ROI time series were constant.

## 8191384 variants

Only S3 listing/HEAD metadata and path-encoded settings were inspected. The object contents were not fetched because **no variant matches the current cohort configuration**. Therefore CSV internals, exact node count, timepoints, numeric validity, constant ROI count, and missing/non-numeric cell counts are **UNKNOWN**, not assumed. Object size and filename/path are known from S3 metadata.

| # | Pipeline | GSR | Bandpass | S3 size (bytes) | Compatibility | Reason |
|---:|---|---|---|---:|---|---|
| 1 | `pipeline_adhd200-benchmark` | OFF | Not encoded | 478550 | **INCOMPATIBLE** | Global signal setting matches, but this pipeline has no 0.01–0.1 Hz bandpass output. |
| 2 | `pipeline_adhd200-benchmark` | ON | Not encoded | 476668 | **INCOMPATIBLE** | Global signal regression is ON and the 0.01–0.1 Hz bandpass output is absent. |
| 3 | `pipeline_adhd200-benchmark__freq-filter` | ON | 0.01–0.1 Hz | 472127 | **INCOMPATIBLE** | Same frequency-filtered pipeline, but global signal regression is ON; cohort has it OFF. |

Exact S3 paths:

- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf0/_mask_CC200/roi_CC200.csv` (478550 bytes) — INCOMPATIBLE: Global signal setting matches, but this pipeline has no 0.01–0.1 Hz bandpass output.
- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global1.motion1.quadratic1.gm0.compcor1.csf0/_mask_CC200/roi_CC200.csv` (476668 bytes) — INCOMPATIBLE: Global signal regression is ON and the 0.01–0.1 Hz bandpass output is absent.
- `s3://fcp-indi/data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark__freq-filter/8191384_session_1/roi_timeseries/_scan_rest_1/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global1.motion1.quadratic1.gm0.compcor1.csf0/_bandpass_freqs_0.01.0.1/_mask_CC200/roi_CC200.csv` (472127 bytes) — INCOMPATIBLE: Same frequency-filtered pipeline, but global signal regression is ON; cohort has it OFF.

## Compatibility decision

All variants are **INCOMPATIBLE** with the exact existing inputs:

1. The frequency-filtered `global1` object has the matching 0.01–0.1 Hz bandpass but turns global signal regression on; the cohort’s files use `global0`.
2. The standard benchmark `global0` object matches global signal regression but has no bandpass path component, so it does not match the 0.01–0.1 Hz filtering.
3. The standard benchmark `global1` object has neither the cohort’s global signal setting nor its bandpass filtering.

There is no frequency-filtered `global0` object listed for 8191384. Selecting a nearby variant would silently change preprocessing. Consequently, no candidate file was downloaded; the 153-subject feature table remains unchanged and no feature file for 8191384 was created.

## Cohort impact

- Original usable cohort: **153**.
- Candidate: **8191384**.
- Selected variant: **None**.
- Compatibility: **No compatible variant**.
- Signal-quality result: **UNKNOWN** (no candidate values read).
- New potential cohort: **153**.
- Adding this one candidate alone would not constitute a meaningful cohort expansion, even if a compatible derivative were later produced.

## Next-step analysis: labeled subjects without compatible CC200 derivatives

Within the 935 valid-diagnosis participant IDs in the inspected local official per-site `participants.tsv` set, **781** do not have a compatible CC200 derivative under the current cohort configuration: **780** have no CC200 object in the two audited benchmark output trees, and **one** (`8191384`) has CC200 objects only in incompatible configurations. These subjects would require an exact-compatible derivative to be produced or raw-fMRI preprocessing under the chosen pipeline. One additional labeled subject (`0015038`) has the matching derivative configuration but failed the current signal-quality rule; it is separate from the 781 and is not included in the current 153.

This count is scoped to the inspected local official participant tables and the two S3 trees `pipeline_adhd200-benchmark/` and `pipeline_adhd200-benchmark__freq-filter/`. It does not assert absence of other derivatives outside those audited paths.

## Integrity / actions

- S3 HEAD sizes: 472,127; 478,550; 476,668 bytes for the three variants, respectively. No GET request was made.
- Downloaded candidate objects: **0**.
- Candidate numeric validation / signal QC / connectivity extraction: **not run**.
- Existing NeuroADHD feature table: **not modified**.
- Model training: **not performed**.
