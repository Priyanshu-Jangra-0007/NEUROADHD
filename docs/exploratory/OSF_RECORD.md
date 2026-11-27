# OSF Pre-Registration & Freeze Record

- **OSF Registration URL:** https://osf.io/eqcsb
- **Registered Timestamp:** 2026-11-27 10:26 AM
- **Date Recorded:** 2026-11-27
- **Branch:** `exploratory/accuracy-study`
- **Annotated Freeze Tag:** `exploratory-frozen` (`aed0f8fa55c797d09b0a8977116786c78f89468b`)
- **Freeze Target Commit:** `393ed6d5ddd8a69f57468e4eb35fc3498aed9530`

## Frozen Artifact Checksums (SHA-256 from `exploratory-frozen`)

| Artifact | Path | SHA-256 |
| :--- | :--- | :--- |
| Protocol Plan | `docs/exploratory/EXPLORATORY_PLAN_DRAFT.md` | `eb1e11770010fffb059a9889c9d920db3d2d1ea738153f429e3099ff9077072a` |
| Analysis Runner | `scripts/run_exploratory.py` | `174dfc56030f454bd77f3b95db31c6cf29a7799f8fb56293a69887e9647c6764` |
| Synthetic Test Suite | `tests/test_run_exploratory.py` | `7b6b3e69d21af16b8be2860085e695f8f3f609a0b3e231ffca65407ffcc801eb` |

## Public-Safety Scan Disposition

Public-safety scan found one 5-character synthetic placeholder in a test mock (two literals) that coincides with a phenotype ID. It was not copied from the data and has no label or other subject data linked to it. The frozen test file was not edited because that would change its registered hash.
