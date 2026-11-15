# NEUROADHD: MACHINE LEARNING PIPELINE & MODEL TRAINING AUDIT

**Document ID:** `NEUROADHD-AUDIT-002`  
**Date of Audit:** 2026-11-15  
**Audit Standard:** Forensic Code Inspection, Serialized Artifact Analysis, and Execution Provenance  
**Files Audited:**
- `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`
- `NeuroADHD/src/preprocessing/validate_cc200_preprocessing.py`
- `NeuroADHD/tests/test_cc200_preprocessing_leakage.py`
- `NeuroADHD_ModelBenchmark/scripts/train_model.py`
- `NeuroADHD_ModelBenchmark/scripts/run_all_models.py`
- `NeuroADHD_ModelBenchmark/scripts/run_final_test_evaluation.py`
- `NeuroADHD_ModelBenchmark/results/final_locked/final_model.joblib`
- `NeuroADHD_ModelBenchmark/results/final_locked/final_preprocessing_pipeline.joblib`
- `NeuroADHD_ModelBenchmark/results/final_locked/final_model_metadata.json`

---

## PART 4 — COMPLETE PREPROCESSING PIPELINE AUDIT

The preprocessing subsystem is implemented in `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`. It is engineered to transform raw, high-dimensional pairwise functional connectivity matrices into normalized, feature-selected representations without inducing **data leakage**.

### 4.1 Step-by-Step Transformation Sequence

The pipeline is formally defined via `make_preprocessing_pipeline(k=100)` (lines 74–91):

```python
Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("variance", VarianceThreshold(threshold=0.0)),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=100)),
])
```

#### Step 1: Missing Value Imputation (`SimpleImputer`)
- **Class:** `sklearn.impute.SimpleImputer`
- **Configuration:** `strategy="median"`, `keep_empty_features=True`
- **Scientific Rationale:** In resting-state fMRI correlation matrices, missing values ($NaN$) can arise if an ROI has zero temporal variance (e.g., poor spatial coverage or edge artifact) or dropout. While the extracted 153-subject matrix contains zero missing cells across all 19,900 edges, the imputer provides fold-local defense against runtime evaluation errors.
- **Fitting Scope:** Medians are calculated **strictly on the training partition** ($N=114$ or $N=133$).

#### Step 2: Zero-Variance Feature Removal (`VarianceThreshold`)
- **Class:** `sklearn.feature_selection.VarianceThreshold`
- **Configuration:** `threshold=0.0`
- **Scientific Rationale:** Removes features that exhibit zero variance across all training subjects. Constant features possess zero mutual information with the diagnostic target and cause division-by-zero errors in standardization and ANOVA $F$-tests.
- **Fitting Scope:** Variances are calculated strictly across training rows. In the 153-subject cohort, all 19,900 edges exhibited non-zero variance; thus, all 19,900 passed to Step 3.

#### Step 3: Feature Standardization (`StandardScaler`)
- **Class:** `sklearn.preprocessing.StandardScaler`
- **Configuration:** `with_mean=True`, `with_std=True`
- **Scientific Rationale:** Centers each Fisher-$z$ connectivity feature to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$). This prevents features with larger baseline correlation magnitudes from dominating distance-based (SVM) or gradient-based (Logistic Regression, XGBoost) optimizers.
- **Fitting Scope:** The empirical mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are computed strictly on training data. Validation and test vectors are transformed using these frozen training statistics:
$$z_{\text{norm}} = \frac{z - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

#### Step 4: Univariate Feature Selection (`SelectKBest` with ANOVA $F$-test)
- **Class:** `sklearn.feature_selection.SelectKBest`
- **Score Function:** `sklearn.feature_selection.f_classif` (one-way analysis of variance)
- **Configuration:** $K = 100$
- **Scientific Rationale:** Given the severe curse of dimensionality ($P = 19,900 \gg N = 114$), feature selection reduces noise and prevents catastrophic model overfitting. The one-way ANOVA $F$-statistic tests the hypothesis that the mean functional connectivity of edge $e$ differs between the ADHD group and the Control group:
$$F = \frac{\text{Between-Group Mean Square}}{\text{Within-Group Mean Square}} = \frac{\sum_{k \in \{0, 1\}} n_k (\bar{z}_k - \bar{z})^2 / (K - 1)}{\sum_{k \in \{0, 1\}} \sum_{i=1}^{n_k} (z_{ki} - \bar{z}_k)^2 / (N - K)}$$
The top $100$ edges with the highest $F$-ratios are selected.
- **Fitting Scope:** Computed strictly on training labels $y_{\text{train}}$. Validation and test labels are **never** passed to this step.

---

### 4.2 Rigorous Leakage Safety Verification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA LEAKAGE AUDIT CHECKLIST                       │
├─────────────────────────────────────────────────┬──────────┬────────────┤
│ Audit Item                                      │ Status   │ Proof      │
├─────────────────────────────────────────────────┼──────────┼────────────┤
│ Are subject IDs excluded from feature matrix X? │ VERIFIED │ Lines 49-66│
│ Are site IDs excluded from feature matrix X?    │ VERIFIED │ Lines 49-66│
│ Is preprocessing encapsulated in an Sklearn Pipe│ VERIFIED │ Line 74-91 │
│ Is pipeline.fit() called ONLY on training data? │ VERIFIED │ Line 165   │
│ Are validation/test sets transform-only?        │ VERIFIED │ Lines 167-8│
│ Are split partitions site-disjoint?             │ VERIFIED │ Manifest   │
│ Do unit tests confirm zero leakage?             │ VERIFIED │ 5 Tests OK │
└─────────────────────────────────────────────────┴──────────┴────────────┘
```

#### Evidence from Source Code:
In `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`:
- Lines 49–57: Features are strictly filtered using `c.startswith("ROI_")`. Non-feature columns (`subject_id`, `site`, `diagnosis`) are rejected if they appear in $X$.
- Lines 64–70: $X$ receives only the 19,900 `ROI_*` columns. Target $y$ receives only `diagnosis`. Metadata is returned in a separate DataFrame.
- Lines 164–168:
  ```python
  pipeline = make_preprocessing_pipeline(k=k)
  # The only fit call: neither validation nor test features/labels enter it.
  pipeline.fit(X_train, y)
  train_array = pipeline.transform(X_train)
  validation_array = pipeline.transform(X_validation)
  test_array = pipeline.transform(X_test)
  ```

#### Synthetic Unit Test Evidence:
In `NeuroADHD/tests/test_cc200_preprocessing_leakage.py`:
- Test `test_validation_and_test_cannot_change_training_fit_or_selected_features`:
  Replaces validation and test matrices with extreme values ($10^{12}$ and $-10^{12}$). The test asserts that `imputer.statistics_`, `scaler.mean_`, `select.get_support()`, and `selected_feature_names` remain identical to machine precision ($100\%$ passed).
- Test `test_subject_overlap_is_rejected`:
  Injects a duplicate subject ID across partitions. The test confirms that `fit_transform_splits` raises a `ValueError` ($100\%$ passed).

---

### 4.3 Data Quality & Methodological Integrity Checks

1. **Duplicate Subjects:** Checked across `subject_split_manifest.csv`. Exactly 153 unique subjects. Zero duplicate IDs.
2. **Duplicate Features:** Exactly 19,900 unique column names (`ROI_001__ROI_002` through `ROI_199__ROI_200`). Zero duplicate features.
3. **Missing Values:** Zero missing values ($NaN$) in `cc200_connectivity_features.csv` (153 rows $\times$ 19,903 columns).
4. **Class Imbalance:**
   - Entire Cohort: 103 Controls ($67.3\%$), 50 ADHD ($32.7\%$) $\implies$ Ratio $\approx 2.06:1$.
   - Training Partition ($N=114$): 84 Controls ($73.7\%$), 30 ADHD ($26.3\%$) $\implies$ Ratio $= 2.80:1$.
   - Validation Partition ($N=19$): 9 Controls ($47.4\%$), 10 ADHD ($52.6\%$) $\implies$ Balanced.
   - Test Partition ($N=20$): 10 Controls ($50.0\%$), 10 ADHD ($50.0\%$) $\implies$ Perfectly Balanced.
5. **Site Confounding & Domain Shift:**
   Because acquisition sites are held out whole, site differences (scanner vendor: Siemens vs. Philips vs. GE, magnetic field strength: 3.0T, repetition time TR: 1.5s to 2.5s) act as strong domain shifts. The training partition contains 6 sites, while validation is exclusively `NeuroIMAGE` and test is exclusively `Peking_2`. No ComBat harmonization was applied, which explains why cross-site generalization is lower than intra-site cross-validation.
6. **Feature Selection Instability:**
   Univariate ANOVA $F$-selection with $K=100$ selects features based on the training fold. In multi-seed repeated cross-validation (`run_20261004_183929_3more_rf_repeated`), feature subsets exhibit moderate variance across folds, which is typical for small-sample neuroimaging regimes ($N \ll P$).

---

## PART 5 — ALL MACHINE LEARNING MODELS AUDIT

A total of **7 distinct model families** plus **3 ensemble paradigms** were implemented, tuned, and evaluated across 29 experimental runs.

### 5.1 Classifier Inventory & Execution Summary

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFIER BENCHMARK SPECIFICATIONS                             │
├──────────────────────┬──────────────────────┬─────────────┬─────────────┬──────────────┤
│ Model Name           │ Python Class         │ Config File │ Val ROC-AUC │ Test ROC-AUC │
├──────────────────────┼──────────────────────┼─────────────┼─────────────┼──────────────┤
│ Logistic Regression  │ LogisticRegression   │ JSON Config │ 0.6889      │ Not selected │
│ Linear SVM           │ SVC(kernel='linear') │ JSON Config │ 0.6889      │ Not selected │
│ RBF SVM              │ SVC(kernel='rbf')    │ JSON Config │ 0.6889      │ Not selected │
│ Random Forest (Base) │ RandomForestClassif. │ JSON Config │ 0.7333      │ 0.6300       │
│ Balanced RF (Final)  │ RandomForestClassif. │ JSON Config │ 0.7000      │ 0.5600       │
│ Extra Trees          │ ExtraTreesClassifier │ JSON Config │ 0.6889      │ Not selected │
│ XGBoost              │ XGBClassifier        │ JSON Config │ 0.7111      │ Not selected │
│ Soft Voting Ensemble │ VotingClassifier     │ Script Init │ 0.6889      │ Not selected │
│ Hard Voting Ensemble │ VotingClassifier     │ Script Init │ 0.5000      │ Not selected │
│ Stacking Ensemble    │ StackingClassifier   │ Script Init │ 0.6889      │ Not selected │
└──────────────────────┴──────────────────────┴─────────────┴─────────────┴──────────────┘
```

---

### 5.2 Detailed Architectural Specifications per Classifier

#### 1. Logistic Regression
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/logistic_regression.json`
- **Constructor:** `sklearn.linear_model.LogisticRegression`
- **Hyperparameter Grid Tested:**
  - `penalty`: `["l1", "l2"]`
  - `C`: `[0.001, 0.01, 0.1, 1.0, 10.0]`
  - `solver`: `["liblinear", "lbfgs", "saga"]`
  - `class_weight`: `[None, "balanced"]`
- **Best Baseline Parameters:** `C=0.01`, `penalty='l2'`, `solver='lbfgs'`, `class_weight=None`
- **Validation Metrics (Baseline):** Accuracy: $0.4211$, Balanced Accuracy: $0.4444$, ROC-AUC: $0.6889$, Sensitivity: $0.0000$, Specificity: $0.8889$, F1: $0.0000$ (Failed to predict ADHD cases due to unweighted majority bias).
- **Validation Metrics (Opt-1 Balanced, $K=100$):** Accuracy: $0.4737$, Balanced Accuracy: $0.4889$, ROC-AUC: $0.6778$, Sensitivity: $0.2000$, Specificity: $0.7778$, F1: $0.2857$.

#### 2. Linear Support Vector Machine (Linear SVM)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/linear_svm.json`
- **Constructor:** `sklearn.svm.SVC(kernel='linear')`
- **Hyperparameter Grid Tested:**
  - `C`: `[0.001, 0.01, 0.1, 1.0, 10.0]`
  - `class_weight`: `[None, "balanced"]`
  - `probability`: `True`
- **Best Baseline Parameters:** `C=0.01`, `kernel='linear'`, `class_weight=None`
- **Validation Metrics (Baseline):** Accuracy: $0.3684$, Balanced Accuracy: $0.3889$, ROC-AUC: $0.6889$, Sensitivity: $0.0000$, Specificity: $0.7778$, F1: $0.0000$.

#### 3. Radial Basis Function Support Vector Machine (RBF SVM)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/rbf_svm.json`
- **Constructor:** `sklearn.svm.SVC(kernel='rbf')`
- **Hyperparameter Grid Tested:**
  - `C`: `[0.1, 1.0, 10.0, 100.0]`
  - `gamma`: `["scale", "auto", 0.001, 0.01, 0.1]`
  - `class_weight`: `[None, "balanced"]`
- **Best Baseline Parameters:** `C=1.0`, `gamma='scale'`, `class_weight=None`
- **Validation Metrics (Baseline):** Accuracy: $0.4737$, Balanced Accuracy: $0.5000$, ROC-AUC: $0.6889$, Sensitivity: $0.0000$, Specificity: $1.0000$, F1: $0.0000$ (Predicted all subjects as Controls).

#### 4. Random Forest Classifier (Baseline Unweighted)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/random_forest.json`
- **Constructor:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameter Grid Tested:**
  - `n_estimators`: `[50, 100, 200]`
  - `max_depth`: `[None, 3, 5, 10]`
  - `min_samples_split`: `[2, 5]`
  - `min_samples_leaf`: `[1, 2]`
  - `max_features`: `["sqrt", "log2"]`
- **Best Parameters:** `n_estimators=100`, `max_depth=None`, `max_features='sqrt'`, `class_weight=None`
- **Validation Metrics:** Accuracy: $0.4211$, Balanced Accuracy: $0.4389$, ROC-AUC: **$0.7333$**, Sensitivity: $0.1000$, Specificity: $0.7778$, F1: $0.1538$.
- **Test Metrics (Peking_2):** Accuracy: $0.5500$, Balanced Accuracy: $0.5500$, ROC-AUC: $0.6300$, Sensitivity: $0.3000$ (Missed 7 out of 10 ADHD patients), Specificity: $0.8000$, F1: $0.4000$.

#### 5. Balanced Random Forest Classifier (Final Selected Model)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/balanced_random_forest.json`
- **Constructor:** `sklearn.ensemble.RandomForestClassifier(class_weight='balanced')`
- **Scientific Mechanism:** Weights each sample inversely proportional to class frequencies:
$$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$
For training ($N=114$, $N_0=84$, $N_1=30$):
$$w_0 = \frac{114}{2 \times 84} \approx 0.678, \quad w_1 = \frac{114}{2 \times 30} = 1.900$$
This forces split criteria (Gini impurity) to heavily penalize misclassifying the minority ADHD class.
- **Selected Hyperparameters:**
  - `n_estimators`: $50$
  - `max_depth`: $3$ (Shallow depth prevents tree memorization of noise)
  - `min_samples_split`: $2$
  - `min_samples_leaf`: $1$
  - `max_features`: `"sqrt"` ($\sqrt{100} = 10$)
  - `class_weight`: `"balanced"`
  - `random_state`: $42$
- **Validation Metrics:** Accuracy: **$0.6842$**, Balanced Accuracy: **$0.6833$**, ROC-AUC: **$0.7000$**, Sensitivity: **$0.7000$** ($7/10$ ADHD detected), Specificity: **$0.6667$**, F1: **$0.7000$**.
- **Test Metrics (Locked Peking_2, Refit on 133 subjects):**
  - **Accuracy:** **$0.6000$** ($12/20$)
  - **Balanced Accuracy:** **$0.6000$**
  - **Sensitivity (Recall):** **$0.6000$** ($6/10$ ADHD detected, $+30\%$ improvement over baseline)
  - **Specificity:** **$0.6000$** ($6/10$ Controls detected)
  - **Precision:** **$0.6000$**
  - **F1-Score:** **$0.6000$**
  - **Test ROC-AUC:** **$0.5600$**
  - **Test PR-AUC:** **$0.6341$**

#### 6. Extremely Randomized Trees (Extra Trees)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/extra_trees.json`
- **Constructor:** `sklearn.ensemble.ExtraTreesClassifier`
- **Best Baseline Parameters:** `n_estimators=100`, `max_depth=5`, `max_features='sqrt'`
- **Validation Metrics (Baseline):** Accuracy: $0.4737$, Balanced Accuracy: $0.4944$, ROC-AUC: $0.6889$, Sensitivity: $0.1000$, Specificity: $0.8889$, F1: $0.1667$.

#### 7. Extreme Gradient Boosting (XGBoost)
- **Source File:** `NeuroADHD_ModelBenchmark/configs/models/xgboost.json`
- **Constructor:** `xgboost.XGBClassifier`
- **Hyperparameter Grid Tested:**
  - `n_estimators`: `[50, 100, 200]`
  - `max_depth`: `[2, 3, 5]`
  - `learning_rate`: `[0.01, 0.05, 0.1]`
  - `scale_pos_weight`: `[1.0, 2.8]` (Ratio of negative to positive training cases)
- **Validation Metrics (Baseline unweighted):** Accuracy: $0.4737$, Balanced Accuracy: $0.5000$, ROC-AUC: $0.7111$, Sensitivity: $0.0000$, Specificity: $1.0000$, F1: $0.0000$.
- **Validation Metrics (Opt-2, `scale_pos_weight=2.8`, `max_depth=2`):** Accuracy: $0.6316$, Balanced Accuracy: $0.6389$, ROC-AUC: $0.6778$, Sensitivity: $0.5000$, Specificity: $0.7778$, F1: $0.5882$.

#### 8. Ensembles: Voting & Stacking
Implemented in `NeuroADHD_ModelBenchmark/scripts/run_ensemble_experiment.py` (36 candidate configurations evaluated in `run_20261004_182834_ensemble`):
- **Soft Voting (Weighted RF + XGBoost):** $0.6 \times P_{\text{RF}} + 0.4 \times P_{\text{XGB}}$. Achieved Validation Accuracy: **$0.6842$**, Balanced Accuracy: **$0.6889$**, ROC-AUC: **$0.6889$**, Sensitivity: **$0.6000$**, Specificity: **$0.7778$**, F1: **$0.6667$**.
- **Hard Voting Ensemble:** Majority rule across all 6 base classifiers. Achieved Validation Accuracy: $0.6842$, Balanced Accuracy: $0.6889$, ROC-AUC: $0.6778$, Sensitivity: $0.6000$, Specificity: $0.7778$.
- **Stacking Classifier:** Base estimators fed out-of-fold predictions into a Logistic Regression meta-learner (`C=1.0`, `penalty='l2'`). Achieved Validation Accuracy: $0.6842$, Balanced Accuracy: $0.6778$, ROC-AUC: $0.6778$, Sensitivity: $0.8000$, Specificity: $0.5556$, F1: $0.7273$.
- **Selection Decision:** While Soft Voting was highly competitive, Balanced Random Forest was selected as the final production model due to its superior seed stability in repeated CV, lower inference latency, and direct TreeSHAP interpretability.

---

## PART 6 — FINAL MODEL DEEP INSPECTION

The authoritative final model artifacts were generated during `run_20261004_182318_final_locked` on 2026-10-04 at 18:23:20 UTC and are stored under:
`/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/`

### 6.1 Artifact Inventory & Physical File Properties

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FINAL PRODUCTION MODEL ARTIFACTS                                │
├─────────────────────────────────────┬──────────────┬─────────────┬─────────────────────┤
│ Filename                            │ Size (Bytes) │ Size (Disk) │ MD5 / Status        │
├─────────────────────────────────────┼──────────────┼─────────────┼─────────────────────┤
│ final_model.joblib                  │ 74,409 bytes │ ~74.4 KB    │ VERIFIED (Safe)     │
│ final_preprocessing_pipeline.joblib │ 1,115,825 B  │ ~1.12 MB    │ VERIFIED (Safe)     │
│ final_test_results.json             │ 481 bytes    │ ~1 KB       │ VERIFIED (Metrics)  │
│ final_test_predictions.csv         │ 628 bytes    │ ~1 KB       │ VERIFIED (20 rows)  │
└─────────────────────────────────────┴──────────────┴─────────────┴─────────────────────┘
```

### 6.2 Serialized Artifact Attributes (Direct Python Verification)

Direct inspection via `joblib.load()` within an isolated Python 3.11 environment confirms the following exact internal attributes:

#### Preprocessing Pipeline (`final_preprocessing_pipeline.joblib`):
- `named_steps['imputer']`: `SimpleImputer(strategy='median', keep_empty_features=True)`
  - `n_features_in_`: $19,900$
  - `statistics_.shape`: $(19900,)$
- `named_steps['variance']`: `VarianceThreshold(threshold=0.0)`
  - `n_features_in_`: $19,900$
  - `get_support().sum()`: $19,900$ (zero features removed)
- `named_steps['scaler']`: `StandardScaler()`
  - `n_features_in_`: $19,900$
  - `mean_.shape`: $(19900,)$
  - `var_.shape`: $(19900,)$
- `named_steps['select']`: `SelectKBest(score_func=f_classif, k=100)`
  - `n_features_in_`: $19,900$
  - `k`: $100$
  - `scores_.shape`: $(19900,)$
  - `pvalues_.shape`: $(19900,)$
  - `get_support().sum()`: Exactly $100$ features selected

#### Model Checkpoint (`final_model.joblib`):
- **Class:** `sklearn.ensemble.RandomForestClassifier`
- **Parameters:**
  - `n_estimators`: $50$
  - `max_depth`: $3$
  - `min_samples_split`: $2$
  - `min_samples_leaf`: $1$
  - `max_features`: `"sqrt"` ($\sqrt{100} = 10$)
  - `class_weight`: `"balanced"`
  - `criterion`: `"gini"`
  - `bootstrap`: `True`
  - `random_state`: $42$
- **Fitted State:**
  - `n_features_in_`: $100$
  - `n_classes_`: $2$
  - `classes_`: `array([0, 1])`
  - `n_outputs_`: $1$
  - `len(estimators_)`: Exactly $50$ decision trees

---

### 6.3 End-to-End Inference Trace for a New Subject

When a new subject is evaluated via `NeuroADHD/src/inference/predict_subject.py` or the web dashboard (`POST /api/predict/subject`), the data flows through the following exact mathematical sequence:

```
[Subject File: roi_CC200.csv (200 rows x T timepoints)]
                         │
                         ▼
1. Validation: Verify 200 ROI rows, numeric finite values, non-zero std.
                         │
                         ▼
2. Transposition: Shape becomes (T, 200).
                         │
                         ▼
3. Correlation: R = corrcoef(T, rowvar=False). Shape: (200, 200).
   Symmetrize: R = (R + R.T) / 2.0; np.fill_diagonal(R, 1.0).
                         │
                         ▼
4. Upper Triangle: tri_i, tri_j = triu_indices(200, k=1).
   edge_corr = R[tri_i, tri_j]. Shape: (19900,).
                         │
                         ▼
5. Fisher Transform: z = arctanh(clip(edge_corr, -1 + 1e-7, 1 - 1e-7)).
                         │
                         ▼
6. Pipeline Transform: X_selected = final_preprocessing_pipeline.transform(z.reshape(1, -1))
   - Imputer applies stored medians: (1, 19900)
   - Variance filter passes: (1, 19900)
   - Scaler applies stored (z - mu_train) / sigma_train: (1, 19900)
   - SelectKBest filters to pre-fit top 100 indices: (1, 100)
                         │
                         ▼
7. Tree Ensemble Prediction:
   probabilities = final_model.predict_proba(X_selected)[0]
   prob_control = probabilities[0]
   prob_adhd = probabilities[1]
   predicted_class = int(prob_adhd >= 0.50)
                         │
                         ▼
[Output: {"predicted_class": 1, "class_label": "ADHD Pattern",
          "adhd_probability": 0.6224, "control_probability": 0.3776}]
```
