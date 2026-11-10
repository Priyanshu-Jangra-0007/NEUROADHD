# NeuroADHD Experiment Run Report — `full_benchmark_01`
**Date (UTC):** 2026-10-21T04:22:22.697993+00:00  
**Status:** `completed`  
**Duration:** `51.91 seconds`  

## 1. Executive Summary
This experiment evaluated 6 individual classifiers and 3 ensemble models on the ADHD-200 CC200 dataset.
The best performing model on the Validation set was **Random Forest** (ROC-AUC = 0.7333).

## 2. Dataset Metadata
- Path: `NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv`
- SHA-256 Checksum: `4750beb6523ce534012b26c1020782731eae0a05ca8658f57bad577649733b65`
- Subjects: 153 total (50 ADHD, 103 Control)
- Features: 19,900 ROI-pair edges → 100 selected

## 3. Preprocessing & Leakage Boundary
- Pipeline: `SimpleImputer(median)` → `VarianceThreshold` → `StandardScaler` → `SelectKBest(f_classif, k=100)`
- Scope: `.fit()` executed strictly on Training partition; Validation and Test transform-only.

## 4. Validation Model Leaderboard
| Model Name | Accuracy | Bal. Acc | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (L1 & L2) | 0.4211 | 0.4444 | 0.0000 | **0.6889** |
| Linear SVM | 0.3684 | 0.3889 | 0.0000 | **0.6889** |
| RBF SVM | 0.4737 | 0.5000 | 0.0000 | **0.6889** |
| Random Forest | 0.4211 | 0.4389 | 0.1538 | **0.7333** |
| Extra Trees | 0.4737 | 0.4944 | 0.1667 | **0.6889** |
| XGBoost Classifier | 0.4737 | 0.5000 | 0.0000 | **0.7111** |
| Soft Voting Ensemble | 0.4737 | 0.5000 | 0.0000 | **0.7111** |
| Hard Voting Ensemble | 0.4737 | 0.5000 | 0.0000 | **0.5000** |
| Stacking Meta-Classifier | 0.4737 | 0.5000 | 0.0000 | **0.6889** |
