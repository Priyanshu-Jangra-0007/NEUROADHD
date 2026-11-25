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
| **Protocol Specification** | [`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`](EXPLORATORY_PLAN_DRAFT.md) | `a608993dbf3dc9c378cd214b7e24180e098ee6eed4a7af232f18891868a8d777` |
| **Analysis Runner** | [`scripts/run_exploratory.py`](../../scripts/run_exploratory.py) | `0f171d26d42f8d97f9b7e0602c57029d20a252a195d1d46b2ac12004b078a83b` |
| **Synthetic Test Suite** | [`tests/test_run_exploratory.py`](../../tests/test_run_exploratory.py) | `0b23809188d8e1146647ce651a2041358eb3b5eea3a1f4bee3ef480f9a577e0f` |

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

## 4. Benchmark & Runtime Profile for Fused Design (Estimated, `C = 1.0`)
*(All timings labeled as estimates, measured on synthetic cohorts at exact real site sizes: Primary $N = 564$ [NYU 196, Peking 189, NeuroIMAGE 42, KKI 79, OHSU 58] and 215-Pair Matched $N = 430$, CC200 (190 populated ROIs), $T \in [78, 257]$, logistic regression `C = 1.0`, late fusion)*:

- **Primary Cohort ($N = 564$):**
  - Test 1 (Label Permutation, Cached Features): Setup $29.5\text{ s}$, $1.53\text{ s}$/perm $\rightarrow$ $5.6\text{ min}$ for 200 permutations.
  - Test 2 (Late Fusion Incremental Contrast, 15-Split Tangent Setup): Setup $63.6\text{ s}$, $26.19\text{ s}$/perm $\rightarrow$ $88.4\text{ min}$ for 200 permutations.
  - **Both Tests Together (200 Permutations):** **$\approx 93.9\text{ minutes}$** ($1.57\text{ hours}$).
- **215-Pair Matched Subset ($N = 430$):**
  - Test 1 (200 Perms): $8.6\text{ min}$; Test 2 (Late Fusion, 200 Perms): $60.0\text{ min}$; **Both Tests Together:** **$\approx 68.6\text{ minutes}$** ($1.14\text{ hours}$).
Both cohorts complete comfortably below the 3-hour budget threshold.
