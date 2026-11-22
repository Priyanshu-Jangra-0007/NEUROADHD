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
| **Protocol Specification** | [`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`](EXPLORATORY_PLAN_DRAFT.md) | `3dd24c122dd6dcdf70bf29a67aedd7a5211d0a686bfc8ee98dbfbef356f740f3` |
| **Analysis Runner** | [`scripts/run_exploratory.py`](../../scripts/run_exploratory.py) | `302fb58b52ecf9bd95826eacd0fbefdc26e8361870edd4a323781efa599a9c1a` |
| **Synthetic Test Suite** | [`tests/test_run_exploratory.py`](../../tests/test_run_exploratory.py) | `cca7025e216d4fa9b76ef8ffbf9971abbf6de7d77cf5fc34252fb41941801b25` |

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
Once the tag exists and approval is granted:

```bash
python3 scripts/run_exploratory.py --execute-real --n-permutations 200
```

---

## 4. Benchmark & Runtime Profile (Estimated from Synthetic Folds)

- **Single LOIO 5-Fold Evaluation (All 12 Configurations):** $2.70\text{ seconds}$
- **Primary Permutation Test (200 Within-Site Iterations):** $\approx 9.0\text{ minutes}$ (recommended)
- **Extended Permutation Test (1,000 Within-Site Iterations):** $\approx 45.0\text{ minutes}$
- **Bootstrap Confidence Intervals (2,000 Resamples):** $\approx 15\text{ seconds}$
- **Total Expected Run Time (with $B = 200$ permutations):** $\mathbf{\approx 9.5\text{ minutes}}$
