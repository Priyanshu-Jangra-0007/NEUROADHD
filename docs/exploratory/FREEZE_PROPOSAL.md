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
| **Protocol Specification** | [`docs/exploratory/EXPLORATORY_PLAN_DRAFT.md`](EXPLORATORY_PLAN_DRAFT.md) | `bbe8d27784d75f485b4b858290b0d73df5767b1d1714017442eea1dab0ed0f73` |
| **Analysis Runner** | [`scripts/run_exploratory.py`](../../scripts/run_exploratory.py) | `a1af1c1029c676a3fda871e89db777dcf439f382a3c774c72df456e48e429b3b` |
| **Synthetic Test Suite** | [`tests/test_run_exploratory.py`](../../tests/test_run_exploratory.py) | `0e9d7ed415c9a716562c66035df85e7289aa64533d3f1f8bb128914027161a4c` |

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

## 4. Benchmark & Runtime Profile (Estimated from Exact Real Shapes)
*(All timings labeled as estimates, measured on synthetic cohort at exact primary dimensions: $N = 564, K = 190$, exact scan lengths $T \in [78, 257]$)*:

- **Tangent Feature Construction:** $3.75\text{ s}$ per fold ($18.75\text{ s}$ across all 5 folds).
- **Motion Residualization (OLS):** $0.055\text{ s}$ per fold.
- **Single Fold Model Fitting & Scoring (All 12 Configurations):** $0.861\text{ s}$ per fold.
- **Single Complete 5-Fold LOIO Evaluation:** $4.30\text{ s}$.
- **Permutation Cost with Cached Features:** **$3.58\text{ s}$ per iteration**.
- **200 Permutations Total:** $\approx \mathbf{11.9\text{ minutes}}$ ($\approx 716\text{ s}$).
- **1,000 Permutations Total:** $\approx \mathbf{59.7\text{ minutes}}$ ($\approx 3,581\text{ s}$).
- **Bootstrap Confidence Intervals (2,000 Resamples):** $\approx 15\text{ s}$.
- **Total Expected Run Time ($B = 200$ permutations):** $\mathbf{\approx 12.5\text{ minutes}}$.
Both 200 and 1,000 permutations fall comfortably below the 3-hour budget threshold.
