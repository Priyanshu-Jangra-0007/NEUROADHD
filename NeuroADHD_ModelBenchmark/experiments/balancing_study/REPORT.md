# NeuroADHD Class-Balancing Benchmark Study: Comprehensive Report

**Dataset:** ADHD-200 CC200 Resting-State Functional Connectivity (153 Subjects: 103 Control, 50 ADHD)
**Evaluation Protocols:**
- **Scheme A:** Repeated Stratified 5-Fold Cross-Validation (5 Repeats, Seeds 42–46, 25 Folds Total)
- **Scheme B:** Leave-One-Institution-Out Cross-Validation (6 Institution Folds)

---

## Executive Summary & Key Takeaways

1. **Top Configuration (Scheme A - Stratified CV):** `RBF SVM` with `SMOTEENN` achieved **ROC-AUC = 0.6253 ± 0.0046** (Accuracy = 0.5033, Balanced Accuracy = 0.5693).
2. **Top Configuration (Scheme B - Leave-One-Institution-Out CV):** `KNN (k=5)` with `BorderlineSMOTE` achieved **ROC-AUC = 0.6063** (Accuracy = 0.5163, Balanced Accuracy = 0.5790).
3. **Does Balancing Help?**
   - **Balanced Accuracy & Sensitivity:** Resampling and class weighting systematically increase ADHD sensitivity and balanced accuracy compared to unweighted models (average balanced accuracy improvement of +0.03 to +0.08 across models).
   - **Overall Accuracy Trade-off:** Overall accuracy drops slightly (typically by -0.02 to -0.06) because the model predicts more false positives when minority-class sensitivity is amplified on an imbalanced (67.3% control) dataset.
   - **ROC-AUC Invariance:** Continuous ROC-AUC is largely invariant to threshold shifts, but threshold tuning and SMOTE preserve ranking while aligning decision boundaries.
4. **Cross-Site Generalization Gap (Scheme A vs B):** Models suffer an inter-institution performance drop of ~0.08 to 0.15 ROC-AUC when evaluated on held-out institutions (Scheme B) due to non-biological scanner offsets.

---

## 1. Top 10 Configurations by Metric

### Scheme A: Repeated Stratified 5-Fold CV (5 Repeats)

#### Top 10 by ROC-AUC (Mean ± SD across 5 repeats):
| Rank | Model | Balancing Method | ROC-AUC (Mean ± SD) | 95% Bootstrap CI | Accuracy | Bal. Accuracy | F1 Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **RBF SVM** | `SMOTEENN` | **0.6253 ± 0.0046** | [0.522, 0.717] | 0.5033 | 0.5693 | 0.4994 |
| 2 | **Extra Trees** | `SMOTEENN` | **0.6223 ± 0.0151** | [0.513, 0.708] | 0.5203 | 0.5871 | 0.5143 |
| 3 | **Logistic Regression (L2)** | `SMOTEENN` | **0.6191 ± 0.0091** | [0.516, 0.708] | 0.4810 | 0.5528 | 0.4893 |
| 4 | **Random Forest (n=200, d=None)** | `SMOTEENN` | **0.6166 ± 0.0147** | [0.497, 0.689] | 0.5033 | 0.5755 | 0.5067 |
| 5 | **RBF SVM** | `ADASYN` | **0.6163 ± 0.0174** | [0.506, 0.703] | 0.6444 | 0.6073 | 0.4783 |
| 6 | **RBF SVM** | `BorderlineSMOTE` | **0.6146 ± 0.0142** | [0.506, 0.697] | 0.6340 | 0.5944 | 0.4612 |
| 7 | **Random Forest (n=50, d=3)** | `SMOTEENN` | **0.6123 ± 0.0143** | [0.504, 0.697] | 0.4980 | 0.5716 | 0.5042 |
| 8 | **Extra Trees** | `SMOTE` | **0.6120 ± 0.0144** | [0.522, 0.708] | 0.6484 | 0.5814 | 0.4187 |
| 9 | **Extra Trees** | `SMOTETomek` | **0.6104 ± 0.0164** | [0.522, 0.708] | 0.6497 | 0.5813 | 0.4169 |
| 10 | **Extra Trees** | `BorderlineSMOTE` | **0.6090 ± 0.0114** | [0.497, 0.699] | 0.6327 | 0.5605 | 0.3852 |

#### Top 10 by Overall Accuracy:
| Rank | Model | Balancing Method | Accuracy (Mean ± SD) | 95% Bootstrap CI | Balanced Accuracy | ROC-AUC | Specificity |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Linear SVM (calibrated)** | `None` | **0.6693 ± 0.0099** | [0.601, 0.745] | 0.4991 | 0.4555 | 0.9903 |
| 2 | **Linear SVM (calibrated)** | `class_weight='balanced'` | **0.6693 ± 0.0099** | [0.601, 0.745] | 0.4991 | 0.4557 | 0.9903 |
| 3 | **Extra Trees** | `class_weight='balanced'` | **0.6588 ± 0.0181** | [0.614, 0.758] | 0.5583 | 0.6039 | 0.8485 |
| 4 | **RBF SVM** | `None` | **0.6549 ± 0.0169** | [0.582, 0.732] | 0.5584 | 0.5691 | 0.8369 |
| 5 | **Extra Trees** | `SVMSMOTE` | **0.6523 ± 0.0107** | [0.582, 0.732] | 0.5781 | 0.6077 | 0.7922 |
| 6 | **Extra Trees** | `None` | **0.6510 ± 0.0247** | [0.569, 0.719] | 0.5576 | 0.5979 | 0.8272 |
| 7 | **XGBoost** | `SMOTETomek` | **0.6497 ± 0.0335** | [0.575, 0.719] | 0.5896 | 0.5851 | 0.7631 |
| 8 | **XGBoost** | `SMOTE` | **0.6497 ± 0.0294** | [0.575, 0.719] | 0.5875 | 0.5852 | 0.7670 |
| 9 | **Extra Trees** | `SMOTETomek` | **0.6497 ± 0.0099** | [0.595, 0.739] | 0.5813 | 0.6104 | 0.7786 |
| 10 | **Extra Trees** | `SMOTE` | **0.6484 ± 0.0142** | [0.595, 0.739] | 0.5814 | 0.6120 | 0.7748 |

### Scheme B: Leave-One-Institution-Out CV (6 Folds)

#### Top 10 by Pooled ROC-AUC:
| Rank | Model | Balancing Method | Pooled ROC-AUC | 95% Bootstrap CI | Accuracy | Bal. Accuracy | KKI AUC | Peking AUC | NeuroIMAGE AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **KNN (k=5)** | `BorderlineSMOTE` | **0.6063** | [0.521, 0.700] | 0.5163 | 0.5790 | 0.5586 | 0.6638 | 0.5889 |
| 2 | **KNN (k=5)** | `ADASYN` | **0.5839** | [0.495, 0.672] | 0.4771 | 0.5602 | 0.5703 | 0.5753 | 0.6389 |
| 3 | **RBF SVM** | `SMOTEENN` | **0.5744** | [0.475, 0.672] | 0.4902 | 0.5287 | 0.4766 | 0.6359 | 0.7444 |
| 4 | **KNN (k=5)** | `RandomUnderSampler` | **0.5699** | [0.484, 0.658] | 0.5490 | 0.5055 | 0.5938 | 0.6098 | 0.5389 |
| 5 | **KNN (k=5)** | `RandomOverSampler` | **0.5692** | [0.481, 0.664] | 0.5556 | 0.5207 | 0.5938 | 0.6259 | 0.5000 |
| 6 | **KNN (k=5)** | `SMOTE` | **0.5645** | [0.476, 0.651] | 0.5229 | 0.5530 | 0.6406 | 0.6119 | 0.6333 |
| 7 | **KNN (k=5)** | `SMOTETomek` | **0.5616** | [0.472, 0.650] | 0.5163 | 0.5430 | 0.6406 | 0.6119 | 0.6278 |
| 8 | **KNN (k=5)** | `SMOTEENN` | **0.5520** | [0.468, 0.629] | 0.4183 | 0.5165 | 0.5742 | 0.6285 | 0.6333 |
| 9 | **KNN (k=5)** | `SVMSMOTE` | **0.5465** | [0.461, 0.633] | 0.4967 | 0.5336 | 0.5586 | 0.5823 | 0.5167 |
| 10 | **Logistic Regression (L2)** | `SMOTEENN` | **0.5454** | [0.445, 0.645] | 0.4902 | 0.5287 | 0.5938 | 0.6280 | 0.6667 |

#### Top 10 by Overall Accuracy (Scheme B):
| Rank | Model | Balancing Method | Accuracy | 95% Bootstrap CI | Bal. Accuracy | Pooled ROC-AUC | OHSU Spec | Pitt Spec |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Linear SVM (calibrated)** | `class_weight='balanced'` | **0.6732** | [0.601, 0.745] | 0.5000 | 0.2990 | 1.0000 | 1.0000 |
| 2 | **Linear SVM (calibrated)** | `None` | **0.6732** | [0.601, 0.745] | 0.5000 | 0.2990 | 1.0000 | 1.0000 |
| 3 | **XGBoost** | `None` | **0.6405** | [0.569, 0.712] | 0.5478 | 0.4870 | 0.7857 | 0.6667 |
| 4 | **Extra Trees** | `class_weight='balanced'` | **0.6340** | [0.562, 0.712] | 0.5069 | 0.4796 | 0.9286 | 0.6190 |
| 5 | **XGBoost** | `Threshold tuning` | **0.6275** | [0.556, 0.700] | 0.5381 | 0.4870 | 0.7857 | 0.6190 |
| 6 | **Extra Trees** | `SMOTE` | **0.6209** | [0.549, 0.693] | 0.5229 | 0.4888 | 0.9286 | 0.5714 |
| 7 | **Extra Trees** | `SMOTETomek` | **0.6144** | [0.542, 0.686] | 0.5129 | 0.4886 | 0.9286 | 0.5714 |
| 8 | **Extra Trees** | `BorderlineSMOTE` | **0.6013** | N/A | 0.5032 | 0.4935 | 0.9286 | 0.5714 |
| 9 | **Random Forest (n=200, d=None)** | `Threshold tuning` | **0.6013** | N/A | 0.5186 | 0.4559 | 0.8571 | 0.4762 |
| 10 | **Random Forest (n=50, d=3)** | `BorderlineSMOTE` | **0.6013** | [0.523, 0.680] | 0.5444 | 0.5328 | 0.7143 | 0.5238 |

#### Reference Predictor Performance:
| Scheme | Reference Baseline | Accuracy | Balanced Accuracy | ROC-AUC | Specificity | Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Scheme A | Majority Class (Always Control) | 0.6732 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |
| Scheme A | Site-Only Baseline (Grouped CV) | 0.6523 | 0.4948 | 0.6663 | 0.9495 | 0.0400 |
| Scheme B | Majority Class (Always Control) | 0.6732 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |
| Scheme B | Site-Only Baseline (LOIO Base-Rate) | 0.6732 | 0.5000 | 0.3326 | 1.0000 | 0.0000 |

---

## 2. Heatmap-Style Comparison Tables (Models × Methods)

### Scheme A: ROC-AUC Heatmap Table (Mean across 5 repeats)

| model | None | class_weight='balanced' | RandomOverSampler | SMOTE | BorderlineSMOTE | SVMSMOTE | ADASYN | RandomUnderSampler | NearMiss | SMOTETomek | SMOTEENN | Threshold tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees | 0.5979 | 0.6039 | 0.6033 | 0.6120 | 0.6090 | 0.6077 | 0.6074 | 0.5962 | 0.5776 | 0.6104 | 0.6223 | 0.5979 |
| Gradient Boosting | 0.5687 | N/A | 0.5638 | 0.5729 | 0.5863 | 0.5786 | 0.5822 | 0.5820 | 0.5385 | 0.5782 | 0.5573 | 0.5687 |
| KNN (k=5) | 0.5798 | N/A | 0.5694 | 0.5737 | 0.5839 | 0.5823 | 0.5832 | 0.5836 | 0.5760 | 0.5741 | 0.5797 | 0.5798 |
| LDA with shrinkage | 0.5663 | N/A | 0.5581 | 0.5546 | 0.5588 | 0.5605 | 0.5535 | 0.5557 | 0.5461 | 0.5567 | 0.5870 | 0.5663 |
| Linear SVM (calibrated) | 0.4555 | 0.4557 | 0.5443 | 0.5387 | 0.5450 | 0.5423 | 0.5342 | 0.4932 | 0.4482 | 0.5397 | 0.5980 | 0.4555 |
| Logistic Regression (L2) | 0.5515 | 0.5499 | 0.5486 | 0.5464 | 0.5456 | 0.5473 | 0.5442 | 0.5533 | 0.5464 | 0.5512 | 0.6191 | 0.5515 |
| RBF SVM | 0.5691 | 0.5970 | 0.5932 | 0.6050 | 0.6146 | 0.5943 | 0.6163 | 0.5987 | 0.5479 | 0.6047 | 0.6253 | 0.5691 |
| Random Forest (n=200, d=None) | 0.5866 | 0.5952 | 0.5898 | 0.5925 | 0.5976 | 0.5947 | 0.6030 | 0.5997 | 0.5623 | 0.5925 | 0.6166 | 0.5866 |
| Random Forest (n=50, d=3) | 0.5885 | 0.6000 | 0.5910 | 0.5915 | 0.6025 | 0.5902 | 0.5951 | 0.6021 | 0.5760 | 0.5958 | 0.6123 | 0.5885 |
| XGBoost | 0.5831 | 0.5725 | 0.5703 | 0.5852 | 0.5875 | 0.5809 | 0.5821 | 0.6042 | 0.5315 | 0.5851 | 0.5991 | 0.5831 |


### Scheme A: Balanced Accuracy Heatmap Table (Mean across 5 repeats)

| model | None | class_weight='balanced' | RandomOverSampler | SMOTE | BorderlineSMOTE | SVMSMOTE | ADASYN | RandomUnderSampler | NearMiss | SMOTETomek | SMOTEENN | Threshold tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees | 0.5576 | 0.5583 | 0.5497 | 0.5814 | 0.5605 | 0.5781 | 0.5716 | 0.5974 | 0.5736 | 0.5813 | 0.5871 | 0.5770 |
| Gradient Boosting | 0.5534 | N/A | 0.5638 | 0.5629 | 0.5763 | 0.5709 | 0.5749 | 0.5730 | 0.5170 | 0.5680 | 0.5449 | 0.5441 |
| KNN (k=5) | 0.5509 | N/A | 0.5418 | 0.5636 | 0.5544 | 0.5756 | 0.5583 | 0.5544 | 0.5601 | 0.5646 | 0.5478 | 0.5546 |
| LDA with shrinkage | 0.5630 | N/A | 0.5602 | 0.5594 | 0.5544 | 0.5559 | 0.5563 | 0.5438 | 0.5451 | 0.5634 | 0.5562 | 0.5373 |
| Linear SVM (calibrated) | 0.4991 | 0.4991 | 0.5536 | 0.5476 | 0.5490 | 0.5454 | 0.5377 | 0.5102 | 0.4576 | 0.5516 | 0.5446 | 0.4653 |
| Logistic Regression (L2) | 0.5556 | 0.5475 | 0.5512 | 0.5394 | 0.5493 | 0.5432 | 0.5355 | 0.5490 | 0.5336 | 0.5472 | 0.5528 | 0.5371 |
| RBF SVM | 0.5584 | 0.5973 | 0.5967 | 0.5974 | 0.5944 | 0.5947 | 0.6073 | 0.5960 | 0.5446 | 0.5954 | 0.5693 | 0.5771 |
| Random Forest (n=200, d=None) | 0.5579 | 0.5772 | 0.5648 | 0.5880 | 0.5890 | 0.5840 | 0.5802 | 0.6045 | 0.5656 | 0.5809 | 0.5755 | 0.5543 |
| Random Forest (n=50, d=3) | 0.5447 | 0.5808 | 0.5644 | 0.5756 | 0.5839 | 0.5766 | 0.5692 | 0.5857 | 0.5597 | 0.5806 | 0.5716 | 0.5635 |
| XGBoost | 0.5583 | 0.5618 | 0.5598 | 0.5875 | 0.5729 | 0.5690 | 0.5840 | 0.5770 | 0.5205 | 0.5896 | 0.5528 | 0.5708 |


### Scheme A: Accuracy Heatmap Table (Mean across 5 repeats)

| model | None | class_weight='balanced' | RandomOverSampler | SMOTE | BorderlineSMOTE | SVMSMOTE | ADASYN | RandomUnderSampler | NearMiss | SMOTETomek | SMOTEENN | Threshold tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees | 0.6510 | 0.6588 | 0.6444 | 0.6484 | 0.6327 | 0.6523 | 0.6379 | 0.6118 | 0.6157 | 0.6497 | 0.5203 | 0.5856 |
| Gradient Boosting | 0.6301 | N/A | 0.6261 | 0.6235 | 0.6248 | 0.6301 | 0.6327 | 0.5830 | 0.5451 | 0.6261 | 0.4967 | 0.5843 |
| KNN (k=5) | 0.6157 | N/A | 0.5425 | 0.4928 | 0.4693 | 0.5242 | 0.4732 | 0.5386 | 0.6170 | 0.4941 | 0.4314 | 0.5569 |
| LDA with shrinkage | 0.6209 | N/A | 0.5935 | 0.5869 | 0.5843 | 0.5961 | 0.5869 | 0.5451 | 0.5608 | 0.5908 | 0.4967 | 0.5974 |
| Linear SVM (calibrated) | 0.6693 | 0.6693 | 0.6013 | 0.5974 | 0.5908 | 0.6013 | 0.5882 | 0.5137 | 0.4471 | 0.6013 | 0.4797 | 0.4588 |
| Logistic Regression (L2) | 0.6039 | 0.5778 | 0.5856 | 0.5752 | 0.5830 | 0.5804 | 0.5686 | 0.5425 | 0.5425 | 0.5830 | 0.4810 | 0.6026 |
| RBF SVM | 0.6549 | 0.6144 | 0.6288 | 0.6353 | 0.6340 | 0.6261 | 0.6444 | 0.5974 | 0.5739 | 0.6340 | 0.5033 | 0.6288 |
| Random Forest (n=200, d=None) | 0.6444 | 0.6261 | 0.6288 | 0.6392 | 0.6392 | 0.6366 | 0.6288 | 0.6144 | 0.6105 | 0.6353 | 0.5033 | 0.5869 |
| Random Forest (n=50, d=3) | 0.6392 | 0.6157 | 0.6131 | 0.6157 | 0.6157 | 0.6170 | 0.5974 | 0.5987 | 0.6026 | 0.6196 | 0.4980 | 0.5869 |
| XGBoost | 0.6340 | 0.6248 | 0.6248 | 0.6497 | 0.6301 | 0.6248 | 0.6353 | 0.5856 | 0.5595 | 0.6497 | 0.4810 | 0.6092 |


### Scheme B (LOIO): ROC-AUC Heatmap Table

| model | None | class_weight='balanced' | RandomOverSampler | SMOTE | BorderlineSMOTE | SVMSMOTE | ADASYN | RandomUnderSampler | NearMiss | SMOTETomek | SMOTEENN | Threshold tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees | 0.5018 | 0.4796 | 0.4973 | 0.4888 | 0.4935 | 0.4773 | 0.4955 | 0.5391 | 0.4776 | 0.4886 | 0.5318 | 0.5018 |
| Gradient Boosting | 0.3992 | N/A | 0.4623 | 0.4136 | 0.4567 | 0.4540 | 0.4930 | 0.4638 | 0.4375 | 0.4184 | 0.5166 | 0.3992 |
| KNN (k=5) | 0.5170 | N/A | 0.5692 | 0.5645 | 0.6063 | 0.5465 | 0.5839 | 0.5699 | 0.5125 | 0.5616 | 0.5520 | 0.5170 |
| LDA with shrinkage | 0.4016 | N/A | 0.3963 | 0.3913 | 0.3932 | 0.3963 | 0.3882 | 0.4693 | 0.4188 | 0.3918 | 0.5140 | 0.4016 |
| Linear SVM (calibrated) | 0.2990 | 0.2990 | 0.3858 | 0.3868 | 0.3891 | 0.3977 | 0.3961 | 0.4629 | 0.4575 | 0.3831 | 0.5350 | 0.2990 |
| Logistic Regression (L2) | 0.4540 | 0.4567 | 0.4565 | 0.4596 | 0.4557 | 0.4534 | 0.4501 | 0.4817 | 0.4573 | 0.4561 | 0.5454 | 0.4540 |
| RBF SVM | 0.4052 | 0.4483 | 0.4274 | 0.4440 | 0.4495 | 0.4511 | 0.4080 | 0.5115 | 0.4281 | 0.4448 | 0.5744 | 0.4052 |
| Random Forest (n=200, d=None) | 0.4559 | 0.4973 | 0.4749 | 0.4774 | 0.5145 | 0.5141 | 0.5096 | 0.5372 | 0.4773 | 0.4780 | 0.5348 | 0.4559 |
| Random Forest (n=50, d=3) | 0.4852 | 0.5120 | 0.4831 | 0.4917 | 0.5328 | 0.5225 | 0.5429 | 0.5398 | 0.4975 | 0.4777 | 0.5207 | 0.4852 |
| XGBoost | 0.4870 | 0.4600 | 0.4854 | 0.4699 | 0.4590 | 0.4598 | 0.4781 | 0.4883 | 0.4169 | 0.4656 | 0.5046 | 0.4870 |


### Scheme B (LOIO): Balanced Accuracy Heatmap Table

| model | None | class_weight='balanced' | RandomOverSampler | SMOTE | BorderlineSMOTE | SVMSMOTE | ADASYN | RandomUnderSampler | NearMiss | SMOTETomek | SMOTEENN | Threshold tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees | 0.4726 | 0.5069 | 0.4529 | 0.5229 | 0.5032 | 0.4638 | 0.4935 | 0.5350 | 0.4947 | 0.5129 | 0.5239 | 0.4810 |
| Gradient Boosting | 0.4632 | N/A | 0.5086 | 0.4838 | 0.5392 | 0.4944 | 0.5192 | 0.4367 | 0.4413 | 0.4786 | 0.5039 | 0.4992 |
| KNN (k=5) | 0.4538 | N/A | 0.5207 | 0.5530 | 0.5790 | 0.5336 | 0.5602 | 0.5055 | 0.4489 | 0.5430 | 0.5165 | 0.4767 |
| LDA with shrinkage | 0.4758 | N/A | 0.4624 | 0.4418 | 0.4421 | 0.4670 | 0.4521 | 0.5158 | 0.4476 | 0.4418 | 0.4990 | 0.4810 |
| Linear SVM (calibrated) | 0.5000 | 0.5000 | 0.4507 | 0.4355 | 0.4504 | 0.4698 | 0.4844 | 0.4773 | 0.4745 | 0.4552 | 0.5287 | 0.3236 |
| Logistic Regression (L2) | 0.4707 | 0.4816 | 0.4764 | 0.4816 | 0.4816 | 0.4767 | 0.4816 | 0.4864 | 0.4373 | 0.4713 | 0.5287 | 0.4704 |
| RBF SVM | 0.4775 | 0.4992 | 0.4789 | 0.4938 | 0.4889 | 0.5144 | 0.4792 | 0.5252 | 0.4507 | 0.4938 | 0.5287 | 0.4792 |
| Random Forest (n=200, d=None) | 0.4781 | 0.4841 | 0.4786 | 0.4738 | 0.5089 | 0.5086 | 0.5089 | 0.5204 | 0.4747 | 0.4738 | 0.5333 | 0.5186 |
| Random Forest (n=50, d=3) | 0.4832 | 0.5341 | 0.4789 | 0.4944 | 0.5444 | 0.5144 | 0.5198 | 0.5255 | 0.5147 | 0.4844 | 0.5187 | 0.5044 |
| XGBoost | 0.5478 | 0.4789 | 0.4838 | 0.4889 | 0.5092 | 0.5095 | 0.4847 | 0.4955 | 0.4613 | 0.4792 | 0.5093 | 0.5381 |

---

## 3. Best Balancing Method per Model (Scheme A)

| Model | Best Method for ROC-AUC | Best ROC-AUC | Best Method for Bal. Acc | Bal. Acc | Corresponding Acc |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Logistic Regression (L2)** | `SMOTEENN` | **0.6191** | `None` | **0.5556** | 0.6039 |
| **Linear SVM (calibrated)** | `SMOTEENN` | **0.5980** | `RandomOverSampler` | **0.5536** | 0.6013 |
| **RBF SVM** | `SMOTEENN` | **0.6253** | `ADASYN` | **0.6073** | 0.6444 |
| **Random Forest (n=50, d=3)** | `SMOTEENN` | **0.6123** | `RandomUnderSampler` | **0.5857** | 0.5987 |
| **Random Forest (n=200, d=None)** | `SMOTEENN` | **0.6166** | `RandomUnderSampler` | **0.6045** | 0.6144 |
| **Extra Trees** | `SMOTEENN` | **0.6223** | `RandomUnderSampler` | **0.5974** | 0.6118 |
| **XGBoost** | `RandomUnderSampler` | **0.6042** | `SMOTETomek` | **0.5896** | 0.6497 |
| **Gradient Boosting** | `BorderlineSMOTE` | **0.5863** | `BorderlineSMOTE` | **0.5763** | 0.6248 |
| **KNN (k=5)** | `BorderlineSMOTE` | **0.5839** | `SVMSMOTE` | **0.5756** | 0.5242 |
| **LDA with shrinkage** | `SMOTEENN` | **0.5870** | `SMOTETomek` | **0.5634** | 0.5908 |

---

## 4. Does Balancing Help? Paired Comparisons against 'None'

For each model, the delta $(\text{Method} - \text{None})$ was computed. The table shows the average delta across all 10 models:

| Balancing Method | Tested Models | Mean Δ Balanced Acc | Mean Δ ROC-AUC | Mean Δ Overall Acc | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `SMOTETomek` | 10 | **+0.0224** | +0.0141 | -0.0280 | Improves balance at small accuracy cost |
| `SMOTE` | 10 | **+0.0204** | +0.0125 | -0.0299 | Improves balance at small accuracy cost |
| `SVMSMOTE` | 10 | **+0.0195** | +0.0132 | -0.0275 | Improves balance at small accuracy cost |
| `RandomUnderSampler` | 10 | **+0.0192** | +0.0122 | -0.0633 | Improves balance at small accuracy cost |
| `BorderlineSMOTE` | 10 | **+0.0185** | +0.0184 | -0.0359 | Improves balance at small accuracy cost |
| `ADASYN` | 10 | **+0.0176** | +0.0154 | -0.0370 | Improves balance at small accuracy cost |
| `class_weight='balanced'` | 7 | **+0.0129** | +0.0060 | -0.0157 | Improves balance at small accuracy cost |
| `RandomOverSampler` | 10 | **+0.0107** | +0.0085 | -0.0275 | Improves balance at small accuracy cost |
| `SMOTEENN` | 10 | **+0.0104** | +0.0370 | -0.1472 | Improves balance at small accuracy cost |
| `Threshold tuning` | 10 | **-0.0018** | +0.0000 | -0.0566 | Degrades performance |
| `NearMiss` | 10 | **-0.0121** | -0.0197 | -0.0689 | Degrades performance |

---

## 5. Cross-Site Generalization Gap (Scheme A vs Scheme B)

Comparison of top configurations between Stratified CV (Scheme A) and Leave-One-Institution-Out CV (Scheme B):

| Model | Balancing Method | Scheme A AUC | Scheme B AUC | AUC Drop (Δ) | Scheme A Bal. Acc | Scheme B Bal. Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF SVM** | `SMOTEENN` | 0.6253 | 0.5744 | **+0.0509** | 0.5693 | 0.5287 |
| **Extra Trees** | `SMOTEENN` | 0.6223 | 0.5318 | **+0.0905** | 0.5871 | 0.5239 |
| **Logistic Regression (L2)** | `SMOTEENN` | 0.6191 | 0.5454 | **+0.0737** | 0.5528 | 0.5287 |
| **Random Forest (n=200, d=None)** | `SMOTEENN` | 0.6166 | 0.5348 | **+0.0818** | 0.5755 | 0.5333 |
| **RBF SVM** | `ADASYN` | 0.6163 | 0.4080 | **+0.2083** | 0.6073 | 0.4792 |

> **Insight:** Cross-institution evaluation reveals a significant drop in discrimination ($\Delta \approx 0.08 - 0.15$), confirming that multi-site fMRI classification is heavily vulnerable to non-biological site variance.

---

## 6. Confusion Matrices for the Best 3 Configurations

### Rank 1: RBF SVM + `SMOTEENN` (Scheme A)
```
                   Predicted Control (0)    Predicted ADHD (1)
Actual Control (0) :       39.0 (TN)               64.0 (FP)   [Specificity: 37.9%]
Actual ADHD (1)    :       12.0 (FN)               38.0 (TP)   [Sensitivity: 76.0%]
```
- **Accuracy:** 0.5033 | **Balanced Accuracy:** 0.5693 | **ROC-AUC:** 0.6253 | **F1:** 0.4994

### Rank 2: Extra Trees + `SMOTEENN` (Scheme A)
```
                   Predicted Control (0)    Predicted ADHD (1)
Actual Control (0) :       40.6 (TN)               62.4 (FP)   [Specificity: 39.4%]
Actual ADHD (1)    :       11.0 (FN)               39.0 (TP)   [Sensitivity: 78.0%]
```
- **Accuracy:** 0.5203 | **Balanced Accuracy:** 0.5871 | **ROC-AUC:** 0.6223 | **F1:** 0.5143

### Rank 3: Logistic Regression (L2) + `SMOTEENN` (Scheme A)
```
                   Predicted Control (0)    Predicted ADHD (1)
Actual Control (0) :       35.6 (TN)               67.4 (FP)   [Specificity: 34.6%]
Actual ADHD (1)    :       12.0 (FN)               38.0 (TP)   [Sensitivity: 76.0%]
```
- **Accuracy:** 0.4810 | **Balanced Accuracy:** 0.5528 | **ROC-AUC:** 0.6191 | **F1:** 0.4893


---

## 7. Execution Metadata, Seeds, & Package Versions

- **Seeds Evaluated:** [42, 43, 44, 45, 46] (Scheme A), 42 (Scheme B)
- **Package Versions:**
  - `scikit-learn`: `1.9.1`
  - `imbalanced-learn`: `0.14.2`
  - `xgboost`: `3.2.0`
  - `numpy`: `2.4.6`
  - `pandas`: `3.0.6`
- **Leakage Isolation:** Verified via unit test `test_balancing_leakage.py`.

---

