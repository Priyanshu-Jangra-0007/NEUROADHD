# Study 2: Open Science Framework (OSF) Preregistration Record

**Project:** NeuroADHD — Study 2 Multi-Site Resting-State fMRI Generalization  
**Date Registered:** November 17, 2026  
**Status:** Publicly Registered & Timestamped  
**Integrity Seal:** Protocol locked and registered prior to downloading, inspecting, or analyzing any Athena derivative dataset.

---

## 1. Verified OSF Registrations

Two public registrations were recorded on OSF Registries on 2026-11-17 to provide comprehensive open-science tracking:

### Primary Registration (Complete Research Plan)
* **URL:** [https://osf.io/grbt2](https://osf.io/grbt2)
* **DOI:** [10.17605/OSF.IO/GRBT2](https://doi.org/10.17605/OSF.IO/GRBT2)
* **Registration Type:** OSF Preregistration
* **Scope:** Contains the complete, binding research plan: formal hypotheses, sampling strategy, four-step preprocessing pipeline, locked Random Forest estimator (`50` trees, depth `3`, `balanced`), sealed-institution rule, within-institution permutation testing ($B = 1,000$), secondary $\Delta\text{AUC}$ endpoint, statistical power caveat ($18\%$ at $N = 404$, $26\%$ at $N = 620$ for true $\Delta\text{AUC} = +0.04$), and pre-defined interpretive outcome zones.
* **Status:** Public
* **Date Registered:** 2026-11-17 14:58 UTC (shown as 8:28 PM IST / UTC+05:30 local time on OSF; registered 2 minutes after `db7nv`)
* **License:** CC0 1.0 Universal

### Secondary ("Duplicate") Registration (Registered First — Supplementary Hashes & Descriptor)
* **URL:** [https://osf.io/db7nv](https://osf.io/db7nv)
* **DOI:** [10.17605/OSF.IO/DB7NV](https://doi.org/10.17605/OSF.IO/DB7NV)
* **Registration Type:** Open-Ended Registration
* **Scope:** Supplementary registration recording repository description, cryptographic hashes, and provenance metadata. The substantive research protocol is housed in the primary registration (`grbt2`).
* **Status:** Public
* **Date Registered:** 2026-11-17 14:56 UTC (registered first, 2 minutes before `grbt2`)
* **License:** CC0 1.0 Universal

---

## 2. Protocol Integrity & Version Control Traceability

* **Protocol File:** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)
* **Source Git Commit:** [`2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e`](https://github.com/Priyanshu-Jangra-0007/NEUROADHD/commit/2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e)
* **Cryptographic Checksum (SHA-256):**
  ```text
  e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec
  ```
* **OSF Upload Timestamp:** Logged by the OSF platform as `PREREGISTRATION.md` at 14:52 UTC on 2026-11-17, strictly preceding the final registrations.

---

## 3. Disclosures of Prior Dataset Exposure

To uphold strict scientific transparency, the following prior exposures to ADHD-200 metadata and derivatives were formally declared on OSF:
1. **Phenotype Tables:** Per-site diagnostic counts (TDC vs. ADHD) from public ADHD-200 phenotype summary tables were reviewed during study design to verify institutional class balance and establish the pre-registered exclusion rule for single-class sites (e.g., WashU and Pittsburgh).
2. **Historical C-PAC Exploration (Study 1 — 126 Development Subjects + 6 NYU Subjects):** **126 development subjects** (KKI 24, NeuroIMAGE 19, Peking 69, 14 OHSU controls) plus 6 subjects from the candidate sealed site (NYU) were included in the exploratory 153-subject C-PAC cohort during Study 1, where the locked pipeline's hyperparameters were chosen.
3. **Foreknowledge Declaration (Clarification Added 2026-11-18):** The OSF declaration option was recorded as:
   > *"Authors' limited observation of the data could not influence their analysis decisions."*
   - **Clarification (see [`OSF_RECORD.md`](OSF_RECORD.md)):** That prior-observation option **understates** the overlap: 126 development subjects (KKI 24, NeuroIMAGE 19, Peking 69, 14 OHSU controls) were also in Study 1, where the locked pipeline's hyperparameters were chosen.

---

## 4. Confirmation of Zero Athena Data Exposure

* **Confirmation:** At the exact moment of registration (and commit `2d7ed3b`), **no Athena time-series files (`.1D` or `.csv`) or motion parameter files had been downloaded, unpacked, inspected, or analyzed**.
* The directory `NeuroADHD/data/athena/` is untracked and staged in `.gitignore`.
* All empirical evaluations remain strictly prospective.

---

## 5. Next Milestone

* **Milestone:** Athena Dataset Acquisition & Automated Inspection.
* **Procedure:** Manual download of pre-computed Athena derivatives from NITRC followed by running `scripts/athena_inspect.py` (read-only, count-only audit) to mechanically verify exact on-disk sample counts and confirm the sealed institution before initiating model fitting.
