# [SUPERSEDED] Historical Audit Report

> [!WARNING]
> **SUPERSEDED / HISTORICAL AUDIT DOCUMENTATION**
> This report has been archived to `archive/audit_reports/` for historical reference and audit trail provenance.
> Key corrections to superseded claims in this document:
> 1. **Audit score superseded**: The preliminary rating of 81.8/100 is superseded by the follow-up site confound and generalization audit.
> 2. **"Evaluated strictly once" superseded**: The Peking_2 test set was scored 10 times in total across preliminary runs, locking iterations, and baseline comparisons.
> 3. **Probability baseline corrected**: The claimed 0.35–0.48 baseline probability range was incorrect; the empirical prediction probability range is 0.05–0.67.
> 4. **Batch holdout scope**: Peking_2 is a same-institution, same-scanner batch holdout (Peking_1 and Peking_3 are in the training split).
> 5. **Site confound collapse**: On the full 153-subject cohort under Leave-One-Institution-Out (LOIO) cross-validation, connectivity drops to near-chance (AUC 0.5342, p = 0.0859, not statistically significant).
> 6. **Exploratory subset**: The 118-subject clean subset (excluding zero-ADHD sites OHSU and Pittsburgh) was chosen after the full cohort showed no signal (exploratory). Demographic covariates (age and sex alone) achieve LOIO AUC 0.5976 [0.492, 0.702], while connectivity achieves 0.6221 [0.515, 0.724] (p = 0.021); the difference was not statistically tested.
> 
> Note: Superseded Study 1 audit note — the referenced standalone corrections file was not kept; see [`docs/study2/STUDY2_REPORT.md`](../../docs/study2/STUDY2_REPORT.md) and [`docs/PRESENTATION_OUTLINE.md`](../../docs/PRESENTATION_OUTLINE.md) for current corrected findings and defense notes.

---

# NEUROADHD: EXPLAINABILITY AUDIT & PRIORITIZED OPTIMIZATION ROADMAP

**Document ID:** `NEUROADHD-AUDIT-005`  
**Date of Audit:** 2026-11-15  
**Audit Standard:** Computational Neuroscience Review, Algorithmic Feasibility Analysis, and Systems Engineering  
**Primary Sources:**
- `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/`
- `NeuroADHD_ModelBenchmark/runs/run_20261004_185219_atlas_mapping/atlas_mapping/`
- `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`
- `ADHD200_Official/metadata/official_cc200_summary.json`

---

## PART 9 — SHAP & BRAIN NETWORK EXPLAINABILITY AUDIT

Explainability analyses were conducted during `run_20261004_184658_explainability` and `run_20261004_185219_atlas_mapping` to determine which functional connections and neural hubs drive the Balanced Random Forest model's predictions.

### 9.1 Explainability Methodology: TreeSHAP
- **Algorithm:** TreeSHAP (Lundberg et al., 2020, *Nature Machine Intelligence*). TreeSHAP computes exact Shapley values in polynomial time $\mathcal{O}(T L D^2)$ by tracking decision paths through all 50 constituent decision trees in the ensemble.
- **Mathematical Principle:** For subject $x$, the Shapley value $\phi_i(x)$ represents the additive contribution of functional edge $i$ toward pushing the model's prediction away from the baseline base value $\mathbb{E}[f(X)]$ to the subject prediction $f(x)$:
$$f(x) = \mathbb{E}[f(X)] + \sum_{i=1}^{100} \phi_i(x)$$
- **Global Importance:** Evaluated as the mean absolute Shapley value across subjects:
$$I_i = \frac{1}{N} \sum_{s=1}^N |\phi_i(x_s)|$$

---

### 9.2 Top 10 Predictive Functional Connectivity Edges

Verified from `NeuroADHD_ModelBenchmark/runs/run_20261004_184658_explainability/explainability/top_connectivity_edges.csv`:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                      TOP 10 FUNCTIONAL CONNECTIVITY EDGES BY TREESHAP                        │
├──────┬───────────────────┬─────────┬─────────┬─────────────┬─────────────────────────────────┤
│ Rank │ Edge Feature Name │ Node A  │ Node B  │ Mean |SHAP| │ Model Association Direction     │
├──────┼───────────────────┼─────────┼─────────┼─────────────┼─────────────────────────────────┤
│ 1    │ ROI_023__ROI_189  │ Node 23 │ Node 189│ 0.01676     │ Higher FC increases ADHD risk   │
│ 2    │ ROI_100__ROI_153  │ Node 100│ Node 153│ 0.01617     │ Higher FC increases ADHD risk   │
│ 3    │ ROI_034__ROI_041  │ Node 34 │ Node 41 │ 0.01550     │ Higher FC increases ADHD risk   │
│ 4    │ ROI_143__ROI_153  │ Node 143│ Node 153│ 0.01426     │ Higher FC increases ADHD risk   │
│ 5    │ ROI_023__ROI_103  │ Node 23 │ Node 103│ 0.01414     │ Higher FC increases ADHD risk   │
│ 6    │ ROI_003__ROI_006  │ Node 3  │ Node 6  │ 0.01386     │ Higher FC increases ADHD risk   │
│ 7    │ ROI_056__ROI_138  │ Node 56 │ Node 138│ 0.01322     │ Higher FC increases ADHD risk   │
│ 8    │ ROI_026__ROI_153  │ Node 26 │ Node 153│ 0.01271     │ Higher FC increases ADHD risk   │
│ 9    │ ROI_034__ROI_103  │ Node 34 │ Node 103│ 0.01257     │ Higher FC increases ADHD risk   │
│ 10   │ ROI_006__ROI_127  │ Node 6  │ Node 127│ 0.01215     │ Higher FC increases ADHD risk   │
└──────┴───────────────────┴─────────┴─────────┴─────────────┴─────────────────────────────────┤
```

---

### 9.3 Top Brain Network Hub ROIs

Identified by aggregating the frequency of ROI appearances across the top 50 predictive edges (`mapped_hub_rois.csv`):

1. **CC200 Node 3 (`ROI_003`):**  
   - Appearances in top-50 edges: **8 edges** (16 edges across all 100 features).
   - Mean edge SHAP: $0.0050$.
   - Primary role: Major functional hub connecting anterior frontal and striatal clusters.
2. **CC200 Node 34 (`ROI_034`):**  
   - Appearances in top-50 edges: **6 edges** (9 edges across all 100 features).
   - Mean edge SHAP: $0.0065$.
   - Primary role: Prefrontal attentional control hub.
3. **CC200 Node 174 (`ROI_174`):**  
   - Appearances in top-50 edges: **6 edges** (9 edges across all 100 features).
   - Mean edge SHAP: $0.0049$.
4. **CC200 Node 189 (`ROI_189`):**  
   - Appearances in top-50 edges: **6 edges** (7 edges across all 100 features).
   - Mean edge SHAP: $0.0066$.
5. **CC200 Node 6 (`ROI_006`):**  
   - Appearances in top-50 edges: **4 edges** (6 edges across all 100 features).
6. **CC200 Node 153 (`ROI_153`):**  
   - Appearances in top-50 edges: **3 edges** (4 edges across all 100 features).
   - Mean edge SHAP: **$0.0108$** (Highest average per-edge contribution).

---

### 9.4 Neuroscience Context & The Triple Network Model of ADHD

In cognitive neuroscience, resting-state fMRI investigations of ADHD converge on the **Triple Network Model** (Menon, 2011, *Trends in Cognitive Sciences*), which posits that psychopathology arises from dysregulated interactions among three core neurocognitive networks:

1. **The Default Mode Network (DMN):**
   - *Anatomy:* Precuneus, Posterior Cingulate Cortex (PCC), Medial Prefrontal Cortex (mPFC), inferior parietal lobule.
   - *Function:* Mediates internally focused cognition, autobiographical memory, and mind-wandering.
   - *ADHD Pathology:* In neurotypical individuals, the DMN is anti-correlated with task-positive attentional networks. In ADHD, failure to suppress the DMN during attentional demand leads to lapses of attention and intrusive task-unrelated thoughts.
2. **The Executive Control Network (ECN / Frontoparietal):**
   - *Anatomy:* Dorsolateral Prefrontal Cortex (dlPFC), Posterior Parietal Cortex (PPC).
   - *Function:* Sustained attention, goal-directed behavioral maintenance, working memory, and cognitive flexibility.
   - *ADHD Pathology:* Hypo-connectivity within the frontoparietal circuit impairs executive control and sustained task focus.
3. **The Salience Network (SN):**
   - *Anatomy:* Anterior Insula (AI), Dorsal Anterior Cingulate Cortex (dACC).
   - *Function:* Acts as a dynamic physiological switch mediating transition between DMN and ECN in response to salient external stimuli.
   - *ADHD Pathology:* Impaired salience switching triggers inappropriate persistence of DMN activity.

### 9.5 Methodological Scope & Non-Causal Declaration
> **METHODOLOGICAL WARNING:**  
> The feature attributions identified by TreeSHAP reflect **statistical predictive associations** within a trained Random Forest model. They **do not prove biological causality** or clinical pathology. Because the CC200 atlas is a data-driven clustering scheme and no anatomical lookup table was bundled locally, the model associations should be interpreted as candidates for hypothesis generation, not validated medical biomarkers.

---

## PART 13 — PRIORITIZED RESEARCH & ENGINEERING ROADMAP

Below is a prioritized, step-by-step engineering and neuroimaging improvement roadmap designed to elevate NeuroADHD from a high-quality academic prototype to a publication-ready neuroimaging platform.

---

### Phase 1: High Priority (Critical Methodological Enhancements)

#### Recommendation 1: Scanner Harmonization via ComBat / neuroCombat
- **Current Problem:** Model performance drops from validation ROC-AUC ($0.7000$) to test ROC-AUC ($0.5600$) due to non-biological scanner batch effects (different scanner models, gradients, head coils, and TRs between training sites and Peking_2).
- **Proposed Solution:** Apply **ComBat (Location and Scale empirical Bayes harmonization)** or **neuroCombat** to the $19,900$ Fisher-$z$ connectivity edges to remove site variance while preserving biological variance (diagnosis, age, sex).
- **Exact File to Modify:** `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`
- **Implementation Detail:** Fit ComBat strictly on training sites; apply the empirical Bayes location/scale adjustments to held-out validation and test sites.
- **Expected Benefit:** Reduces cross-site domain shift; expected to raise out-of-site generalization and test ROC-AUC significantly.
- **Risk & Mitigation:** Risk of leaking target labels if diagnosis is included improperly as a covariate during test transformation. *Mitigation:* Ensure harmonization transform parameters are computed strictly on training data without test labels.
- **Verification:** Run a $t$-SNE or PCA plot of subjects colored by site before and after ComBat; site clusters must dissolve while diagnosis separation is preserved.

#### Recommendation 2: Leave-One-Site-Out (LOSO) Cross-Validation
- **Current Problem:** The current protocol relies on a single fixed holdout site for validation (`NeuroIMAGE`, $N=19$) and one for test (`Peking_2`, $N=20$). This makes performance estimates sensitive to the idiosyncratic demographic quirks of those two specific sites.
- **Proposed Solution:** Implement an automated **Leave-One-Site-Out (LOSO) Cross-Validation** harness across all 8 sites in the 153-subject cohort ($8$ distinct folds).
- **Exact File to Modify:** `NeuroADHD_ModelBenchmark/scripts/run_all_models.py` (add `--loso` flag).
- **Expected Benefit:** Provides true unbiased multi-site generalizability distributions across 8 distinct clinical settings.
- **Risk & Mitigation:** Higher compute time (8 full training runs). *Mitigation:* Pipeline execution is fast ($\approx 5\text{ seconds}$ per run).
- **Verification:** Generate a boxplot of test ROC-AUC across all 8 held-out sites.

---

### Phase 2: Medium Priority (Algorithm & Feature Enhancements)

#### Recommendation 3: Multivariate Regularized Feature Selection
- **Current Problem:** Univariate ANOVA $F$-test (`SelectKBest`) evaluates each of the $19,900$ edges in complete isolation, ignoring collinearity and multi-edge network topology.
- **Proposed Solution:** Replace or augment ANOVA with multivariate sparse penalization:
  - **L1-Penalized Logistic Regression (LASSO)** or **ElasticNet** inside the training pipeline.
  - **Recursive Feature Elimination (RFE)** using tree feature importance.
  - **Mutual Information Feature Selection** (`mutual_info_classif`) to capture non-linear edge relationships.
- **Exact File to Modify:** `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py` (`make_preprocessing_pipeline`).
- **Expected Benefit:** Selects compact, non-redundant feature subsets with higher biological stability across folds.
- **Risk & Mitigation:** Increased hyperparameter tuning complexity. *Mitigation:* Tune sparsity parameter $\alpha$ inside inner 5-fold CV.
- **Verification:** Calculate Jaccard similarity index of selected features across cross-validation folds.

#### Recommendation 4: Connectome Graph Theory & Topological Metrics
- **Current Problem:** The model treats functional connectivity edges as independent tabular features, discarding the intrinsic graph topology of the brain.
- **Proposed Solution:** Use the **Brain Connectivity Toolbox (BCT)** or `networkx` to compute graph theoretical metrics per subject:
  - Global efficiency and characteristic path length.
  - Node degree, betweenness centrality, and eigenvector centrality.
  - Modularity index ($Q$) and participation coefficient.
- **Exact File to Modify:** `NeuroADHD/src/features/extract_cc200_connectivity.py`
- **Expected Benefit:** Adds macroscale organizational features that correlate strongly with executive dysfunction in ADHD.
- **Risk & Mitigation:** Computational overhead during feature extraction. *Mitigation:* Compute graph metrics offline during feature generation.
- **Verification:** Verify whether adding 200 node-level graph metrics improves classifier balanced accuracy.

#### Recommendation 5: Probability Calibration via Isotonic Regression
- **Current Problem:** Random Forest predicted probabilities tend to push toward intermediate values ($[0.35, 0.65]$) due to tree averaging, leading to uncalibrated risk scores.
- **Proposed Solution:** Wrap the final classifier in `sklearn.calibration.CalibratedClassifierCV(method='isotonic', cv='prefit')` using out-of-fold validation predictions.
- **Exact File to Modify:** `NeuroADHD_ModelBenchmark/scripts/run_final_test_evaluation.py`
- **Expected Benefit:** Produces true posterior probabilities $P(\text{ADHD} | X)$ suitable for clinical decision support.
- **Risk & Mitigation:** Isotonic regression can overfit on small datasets. *Mitigation:* Use Platt scaling (logistic sigmoid) if validation sample size $N < 50$.
- **Verification:** Plot Brier score and reliability calibration curve before and after calibration.

---

### Phase 3: Infrastructure & Production Scaling

#### Recommendation 6: Containerized Raw MRI Pipeline (Docker / fMRIPrep / C-PAC)
- **Current Problem:** Raw MRI preprocessing is currently 0/18 executed locally due to Docker memory limits ($3.8\text{ GiB}$) and lack of an exact historical C-PAC recipe.
- **Proposed Solution:** Build a dedicated, self-contained Docker/Singularity image bundling:
  - **fMRIPrep 23.2.0** or **C-PAC 1.8.6**.
  - Pinned CC200 NIfTI atlas in MNI152NLin2009cAsym space.
  - Automated Nilearn extraction script to output `roi_CC200.csv`.
- **Exact File to Create:** `NeuroADHD_RawMRI/Dockerfile` and `scripts/run_raw_pipeline.sh`.
- **Expected Benefit:** Unlocks true end-to-end processing: raw 4D BOLD NIfTI $\to$ diagnosis prediction.
- **Risk & Mitigation:** High resource requirement ($16\text{ GB}$ host RAM, $4\text{ hours}$ per subject). *Mitigation:* Execute on a cloud compute instance (AWS EC2 `c5.4xlarge` or GCP Compute Engine).
- **Verification:** Process one raw subject from `ADHD200_Official/experiments/raw_pilot_18/` and verify that the output matches the shape and format required by `predict_subject.py`.

#### Recommendation 7: Cohort Expansion from ADHD-200 S3 Bucket
- **Current Problem:** Current training is restricted to $N=153$ subjects due to derivative availability in historical folders.
- **Proposed Solution:** Leverage the AWS S3 inventory (`ADHD200_Official/metadata/official_s3_inventory.csv`) to download precomputed CC200 derivatives or raw scans for the remaining **780 labeled subjects** in the ADHD-200 Consortium.
- **Exact File to Modify:** `work_download_adhd200.py`
- **Expected Benefit:** Expands dataset from $N=153$ to $N=900+$, increasing statistical power and model generalization.
- **Verification:** Confirm that the expanded feature matrix contains $>800$ subjects with zero missing labels.
