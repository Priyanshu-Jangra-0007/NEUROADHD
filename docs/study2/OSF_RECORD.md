# Study 2 OSF Registration Record & Prior-Observation Clarification

**Date Updated:** 2026-11-18  
**Project:** NeuroADHD — Study 2 Multi-Site Resting-State fMRI Generalization  
**Locked Protocol File (Unmodified):** [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md)  
**Source Commit:** `2d7ed3b3be3780c0b8a5b3de7d3c786bb77df19e`  
**Locked Protocol SHA-256 (verified against OSF & commit `2d7ed3b`):**  
`e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec`

---

## 1. Chronological OSF Timestamps (All Times in UTC)

Both public registrations on OSF Registries were created on **2026-11-17** prior to downloading, inspecting, or analyzing any Athena derivative data. Listing all timestamps in **UTC** clarifies the exact chronological order (the Open-Ended "duplicate" registration was registered **first**):

| Order | Event / Registration | OSF Link & DOI | Type | Timestamp (UTC) |
| :-: | :--- | :--- | :--- | :---: |
| 1 | `PREREGISTRATION.md` uploaded to OSF | — | Protocol file upload | **2026-11-17 14:52 UTC** |
| 2 | **Open-Ended ("Duplicate") Registration** *(registered first)* | [https://osf.io/db7nv](https://osf.io/db7nv) / `10.17605/OSF.IO/DB7NV` | Open-Ended Registration (metadata & hashes) | **2026-11-17 14:56 UTC** |
| 3 | **Primary Registration** *(registered second)* | [https://osf.io/grbt2](https://osf.io/grbt2) / `10.17605/OSF.IO/GRBT2` | OSF Preregistration (complete binding plan) | **2026-11-17 14:58 UTC** *(displayed as 20:28 IST / UTC+05:30 locally)* |

- **Status:** Both registrations are **Public** under **CC0 1.0 Universal**.
- **Substantive Protocol:** The complete binding research plan resides in `grbt2`; `db7nv` is the earlier-timestamped Open-Ended Registration recording repository metadata and hashes.

---

## 2. Prior-Observation & Study 1 Subject Overlap Disclosure

To correct and make fully explicit the prior data exposure disclosed in the OSF registration form:

1. **Study 1 Development-Subject Overlap (126 Development Subjects):**
   - **126 development subjects** in Study 2 were also part of the Study 1 exploratory C-PAC cohort ($N = 153$), where the locked pipeline's architecture and hyperparameters were selected:
     - **KKI:** 24 subjects
     - **NeuroIMAGE:** 19 subjects
     - **Peking (1–3):** 69 subjects
     - **OHSU:** 14 controls
     - **Total development overlap:** $24 + 19 + 69 + 14 = \mathbf{126}$ development subjects.
   - In addition, **6 NYU subjects** (candidate sealed external institution) were present in Study 1's C-PAC cohort (as previously noted in the registration).
2. **Correction to OSF Foreknowledge Declaration:**
   - The OSF registration form recorded the prior-observation option:
     > *"Authors' limited observation of the data could not influence their analysis decisions."*
   - **Disclosure:** That option **understates** the actual prior exposure. Because 126 Study 2 development subjects (KKI 24, NeuroIMAGE 19, Peking 69, 14 OHSU controls) were in Study 1 where the locked pipeline's hyperparameters were chosen (albeit under the C-PAC pipeline rather than Athena), Study 2's development evaluation is **not** strictly naive to those 126 individuals.
   - No Athena derivatives (`.1D` BOLD time series, motion traces, or connectivity matrices) had been downloaded, unpacked, or analyzed at the time of either 2026-11-17 registration.

---

## 3. Cross-References

- [`docs/study2/PREREGISTRATION.md`](PREREGISTRATION.md) — Locked protocol (never modified).
- [`docs/study2/OSF_REGISTRATION.md`](OSF_REGISTRATION.md) & [`docs/study2/REGISTRATION_RECORD.md`](REGISTRATION_RECORD.md) — Initial registration records, updated for UTC consistency and overlap transparency.
- [`docs/study2/DEVIATIONS.md`](DEVIATIONS.md) — Pre-inspection protocol deviation and fallback rules.
