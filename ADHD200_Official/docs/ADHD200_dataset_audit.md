# ADHD-200 official dataset audit

Audit date: 2026-10-05. Scope: official FCP-INDI sources only. No imaging or phenotype objects were downloaded.

## Executive summary

ADHD-200 is a multi-site de-identified imaging dataset contributed by the ADHD-200 Consortium and published through INDI. The official retrospective-sample summary reports **598 participants**, including **383 ADHD participants**, ages **7–21**; it identifies resting-state fMRI (R-fMRI) and MPRAGE. These published cohort figures are distinct from extant S3 object counts and from any future usable CC200 cohort. The official S3 documentation says both raw and preprocessed data are available.

- Project: https://fcon_1000.projects.nitrc.org/indi/adhd200/
- S3 root: `s3://fcp-indi/data/Projects/ADHD200/`
- Raw BIDS: `s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS/`
- Official reported total / ADHD: 598 / 383.
- Modalities: resting-state fMRI and structural MPRAGE.
- Raw BIDS, derivatives and phenotype availability: yes at dataset level.
- CC200: yes, evidence from prior local manifest records verified official S3 `Outputs/cpac/.../roi_CC200.csv` objects. Full current count/coverage is unverified.
- Exact S3 inventory and sizes: **Not determinable from the accessible official inventory**.

## Evidence and inventory method

The official anonymous S3 root listing returned `Outputs/`, `RawData/`, `RawDataBIDS/`, `Resources/`, and `surfaces/`. A read-only anonymous ListObjectsV2 pagination attempt did not return an object page before the network read timed out. Therefore this audit records a partial directory-level inventory only; the CSV has headers and no fabricated object rows. The API was used only to inspect metadata; no object GET/download was made.

`RawDataBIDS` is documented by FCP-INDI as BIDS-organized raw data. `RawData` is a top-level directory observed in S3 but its exact ADHD-200 contents/purpose were not enumerated. `Outputs` conventionally holds derivatives including FreeSurfer, fieldmap-corrected data and C-PAC outputs per FCP-INDI S3 documentation. `Resources` and `surfaces` were observed; their exact contents and sizes are unverified. The ADHD-200 download PDF describes per-site lists and the raw BIDS path. `RawDataTars` and a distinct `PhenotypicData` directory were not present at the observed root; this does not rule out phenotype files elsewhere.

## Cohort and sites

Official INDI retrospective page reports 598 participants (383 ADHD) and ages 7–21. The old ADHD-200 competition page describes the consortium as almost 800, so public historical sources use different scopes/rounding. For this audit, the explicit retrospective table figure 598 is reported as the dataset overview number and the near-800 description is not substituted. Neither figure is an S3 file count or an eligible CC200 count.

The official phenotype key and download instructions agree on eight sites: Peking University; Bradley Hospital/Brown University; Kennedy Krieger Institute; NeuroIMAGE; NYU Child Study Center; OHSU; University of Pittsburgh; Washington University in St. Louis. Site-level participant/ADHD/control counts were not available from sources retrieved here. See `manifests/site_summary.csv`.

## Phenotype coding

The official key documents site, gender, handedness, four diagnosis codes, rating-scale instrument, IQ instrument, medication status and QC. Full encoding appears in `manifests/phenotype_dictionary.csv`. Preserve diagnosis 0/1/2/3; do not collapse subtypes to binary at this stage. The key does not specify a universal subject-ID column spelling or all score-column details; inspect delivered phenotype tables at the matching phase. ADHD measures: ADHD-RS IV, CPRS-LV, Conners Rating Scale 3. IQ: WISC-IV, WASI, WISCC-R and named short-form variants. Medication status is naïve vs not naïve, not detailed medication/dose. QC is questionable/pass.

## Available data and size

| Major path | Purpose / evidence | Research relevance | Size/count |
|---|---|---|---|
| `RawDataBIDS/` | Official BIDS raw data directory | Primary source for future processing | Not determinable from the accessible official inventory |
| `RawData/` | Directory observed; contents not enumerated | Potential raw imaging | Not determinable from the accessible official inventory |
| `Outputs/` | Official docs describe derivatives, including C-PAC | Useful; local manifest has verified CC200 ROI CSV examples | Not determinable from the accessible official inventory |
| `Resources/` | Directory observed; contents unknown | Possible project resources | Not determinable from the accessible official inventory |
| `surfaces/` | Directory observed; contents unknown | Likely surface products; unverified | Not determinable from the accessible official inventory |
| `RawDataTars/` | Not present in observed root listing | None established | Not observed |
| `PhenotypicData/` | Not present in observed root listing; phenotype is reported by official cohort page and may be elsewhere | Essential for labels/covariates | Not determinable from the accessible official inventory |

Raw BIDS total, anatomical MRI total, resting-state fMRI total, phenotype total, preprocessed derivative total, CC200 derivative total, and overall dataset total are all **Not determinable from the accessible official inventory**.

## How this dataset can be used for NeuroADHD

Official ADHD-200 → raw resting-state fMRI → preprocessing → CC200 parcellation → 200 ROI time series → functional connectivity → 19,900 pairwise edges → feature selection → ML classifier. The official FCP-INDI source says preprocessed data are available, and local read-only manifest evidence shows CC200 ROI time series at official C-PAC output paths. Verify the exact pipeline configuration and subject coverage before deciding whether to use those derivatives or process raw scans. This audit makes no strategy decision.

## Comparison with current NeuroADHD

Read-only inspection of `/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/` found its local extraction summary: 153 extracted subjects, 50 ADHD and 103 controls; 154 labeled input subjects were requested, one was skipped because 83 ROI signals were constant. Its report says the scanned local phenotype files had 161 mapped IDs, seven Brown labels pending, leaving 154 labeled subjects (50 ADHD/104 controls). Existing usable sites: Peking_1 38, KKI 24, Pittsburgh 21, Peking_2 20, NeuroIMAGE 19, OHSU 14, Peking_3 11, NYU 6. Current table contains 19,900 Pearson-correlation upper-triangle edges per person from CC200 ROI time series. Its README says an earlier requested NITRC AAL time-course file redirected to login and remaining derivatives were not attempted; however existing local CC200 files and manifest entries document outputs available locally.

Thus the 153 is a pipeline-specific usable feature cohort, not the official dataset size. It reflects 154 locally label-matched subjects minus one unusable ROI time series. Official total 598 minus 153 is **not** a valid “additional available” count: S3 inventory completeness, phenotype correspondence, imaging availability, QC, usable runs, and project inclusion/exclusion policy remain to be checked. Existing current project has site data and 3-class phenotype source collapsed into Control/ADHD for its current table; this new audit preserves official four codes. Current project documents a Peking_2 held-out split; that is historical context only. Peking_2 must not enter any new experiment.

| Dimension | Current NeuroADHD | Official ADHD-200 |
|---|---|---|
| Subjects | 153 extracted; 154 labeled candidates | 598 officially reported |
| Sites | 8 site labels represented in current feature table | 8 official sites |
| Phenotype | Local matched tables; 7 Brown pending in inspection report | Official cohort page/key affirm phenotype and clinical variables; actual full files not inventoried |
| Raw fMRI | Current table based on ROI time series; no full raw inventory verified here | Raw and BIDS available per official S3 docs |
| Preprocessing | Existing CC200 series; extraction summary describes Pearson edge features | Preprocessed outputs available; C-PAC/CC200 verified examples, complete coverage unknown |
| CC200 | Feature table with 19,900 edges | Official outputs contain CC200 examples by locally recorded verified object references |
| Labels | Present table 0/1; current local summary shows 103/50 | Original 0,1,2,3 phenotype key; preserve |
| Provenance | Workspace artifacts include local source manifest; see README and manifest | FCP-INDI official S3 |

## Preliminary roadmap (not executed)

1. Dataset audit (this phase)
2. Download
3. Integrity/QC
4. Phenotype matching
5. Raw-fMRI preprocessing or verified official derivatives
6. CC200 extraction
7. Connectivity extraction
8. Site-disjoint splitting
9. Feature selection
10. Model benchmarking
11. Repeated validation
12. Locked external-site test
13. Explainability

## Constraints and open items

No data download, preprocessing, split creation, feature generation, model training or tuning was performed. No existing NeuroADHD project file was modified. No experiment used Peking_2. Site counts, detailed age distribution, complete phenotype field inventory, QC/motion field presence, file counts and size breakdowns require a functioning complete official S3 object listing and/or official site CSV link lists. Do not treat the 598 cohort figure or CC200 examples as assurance every person has usable fMRI or CC200.
