# Exploratory Analysis Code Freeze Proposal

> [!IMPORTANT]
> **PRE-EXECUTION INTEGRITY LOCK**  
> In accordance with the Study 2 reproducibility architecture, all exploratory code, test suites, and protocol documentation must be frozen via an annotated Git tag prior to accessing or joining any diagnostic labels.  
> **Tag `exploratory-frozen` has NOT been created.**  
> Real label files remain completely unread and untouched.

---

## 1. Frozen Artifact Checksums (SHA-256)

The following SHA-256 hashes uniquely identify the protocol specification, execution runner, and synthetic test suite:

| Component | Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Protocol Specification** | [`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`](EXPLORATORY_PLAN_DRAFT.md) | `eb1e11770010fffb059a9889c9d920db3d2d1ea738153f429e3099ff9077072a` |
| **Analysis Runner** | [`scripts/run_exploratory.py`](../../scripts/run_exploratory.py) | `174dfc56030f454bd77f3b95db31c6cf29a7799f8fb56293a69887e9647c6764` |
| **Synthetic Test Suite** | [`tests/test_run_exploratory.py`](../../tests/test_run_exploratory.py) | `7b6b3e69d21af16b8be2860085e695f8f3f609a0b3e231ffca65407ffcc801eb` |

---

## 2. Hard Guards Verification

`scripts/run_exploratory.py` enforces two strict guards prior to reading any phenotype or label data:
1. **Command-Line Intent Guard:** Refuses execution unless `--execute-real` is passed on the command line.
2. **Git Tag Ancestry Guard:** Verifies via `git merge-base --is-ancestor exploratory-frozen HEAD` that the annotated tag `exploratory-frozen` exists and is an ancestor of the current commit.
3. **Privacy Invariant:** Subject identifiers are never printed to console, logs, or reports.

---

## 3. Freeze & Execution Commands

### Step A: Create the Freeze Tag (To Be Executed ONLY Upon Approval)
Do NOT run this command until the user explicitly replies **`APPROVE EXPLORATORY RUN`**:

```bash
git tag -a exploratory-frozen -m "freeze: lock exploratory analysis runner, plan, and synthetic tests prior to label unblinding"
```

### Step B: Execute the Exploratory Analysis Run
Once the tag exists and approval is granted, execute with `caffeinate -dimsu` and null-only checkpointing using the dedicated project virtual environment:

```bash
caffeinate -dimsu ./NeuroADHD/.venv/bin/python scripts/run_exploratory.py --execute-real --n-permutations 200 --checkpoint NeuroADHD/data/athena/results/exploratory_perm_checkpoint.json
```

---

## 4. Benchmark & Runtime Profile for Fused Design (Estimated, `C = 1.0`, `n_workers = 2`)
*(All timings labeled as **estimates**, fitted from `--rehearsal` runs at `n_permutations = 2` and `4` on the real label-free CC200 arrays with synthetic 45%-positive stratified labels, shared subject covariances, cached tangent spaces, `SharedMemory` workers capped at 2 for `< 6 GB` peak RSS, `peak_rss_mb = 3014.97`, `child_max_rss_mb = 2026.05`)*:

- **Observed Grids + Covariances (`primary_564`, `all_620`, `matched_215`, `matched_226`, `CFG_13–16`):** ~718–727 s at `n_boot = 50`; **~880 s (~14.7 min)** extrapolated to `--execute-real` (`n_boot = 2000`).
- **Primary Test 1 (`12` configs, 5 outer LOIO folds):** fixed `3.34 s` + **`4.91 s/perm`** → **~985 s (~16.4 min)** estimated for 200 permutations.
- **Primary Test 2 (`12` configs, 15 unique LOIO splits + late fusion + per-perm OLS motion residualization):** fixed `3.87 s` + **`13.99 s/perm`** → **~2,801 s (~46.7 min)** estimated for 200 permutations.
- **Separate Multi-Run Family (`CFG_13–16`, 4 configs on NYU + OHSU):** fixed `5.09 s` + **`1.19 s/perm`** combined → **~242 s (~4.0 min)** estimated for 200 permutations.
- **Estimated Total Runtime (`B = 200`, `n_boot = 2000`):** **~82 min (~1.36 h)** point estimate, **~70–105 min (~1.2–1.75 h)** range — comfortably below the 8-hour ceiling.
