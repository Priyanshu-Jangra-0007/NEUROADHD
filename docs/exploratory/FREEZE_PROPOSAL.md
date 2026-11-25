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
| **Protocol Specification** | [`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`](EXPLORATORY_PLAN_DRAFT.md) | `585f44e3e8059f02a90d165a3d9eb63d23a4ca6e10cb3e3628c142a60e6d8afb` |
| **Analysis Runner** | [`scripts/run_exploratory.py`](../../scripts/run_exploratory.py) | `80a48f36b9f11d407f9f72cdf72f46b21c357468bf3fecd28602f106da447b0d` |
| **Synthetic Test Suite** | [`tests/test_run_exploratory.py`](../../tests/test_run_exploratory.py) | `369cc6f8342740a41e7dd623ce194460c2145753287e5c80ec0bf67b380b0b4b` |

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
Once the tag exists and approval is granted, execute using the dedicated project virtual environment:

```bash
./NeuroADHD/.venv/bin/python scripts/run_exploratory.py --execute-real --n-permutations 200
```

---

## 4. Benchmark & Runtime Profile (Estimated from Exact Real Site Sizes, `C = 1.0`)
*(All timings labeled as estimates, measured on synthetic cohorts at exact real site sizes: Primary $N = 564$ [NYU 196, Peking 189, NeuroIMAGE 42, KKI 79, OHSU 58] and 215-Pair Matched $N = 430$, CC200 (190 populated ROIs), $T \in [78, 257]$, logistic regression `C = 1.0`)*:

- **Primary Cohort ($N = 564$):**
  - Tangent Feature Construction: $5.71\text{ s}$ per fold ($28.56\text{ s}$ across 5 folds).
  - Motion Residualization (OLS): $0.107\text{ s}$ per fold ($0.54\text{ s}$ across 5 folds).
  - Test 1 (Label Permutation, Cached Features): $5.16\text{ s}$ per iteration ($17.20\text{ min}$ for 200 permutations).
  - Test 2 (Incremental Contrast, Recomputing Tangent + Residualization): $35.28\text{ s}$ per iteration ($117.59\text{ min}$ for 200 permutations).
  - **Both Tests Together (200 Permutations):** **$\approx 134.79\text{ minutes}$** ($2.25\text{ hours}$).
- **215-Pair Matched Subset ($N = 430$):**
  - Test 1 (200 Perms): $10.65\text{ min}$; Test 2 (200 Perms): $75.46\text{ min}$; **Both Tests Together:** **$\approx 86.10\text{ minutes}$** ($1.44\text{ hours}$).
Both cohorts complete comfortably below the 3-hour budget threshold.
