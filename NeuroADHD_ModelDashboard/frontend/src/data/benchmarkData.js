export const BENCHMARK_DATA = {
  "cohort_breakdown": {
    "total_subjects": 153,
    "train_subjects": 114,
    "train_adhd": 30,
    "train_control": 84,
    "validation_subjects": 19,
    "validation_adhd": 10,
    "validation_control": 9,
    "validation_site": "NeuroIMAGE",
    "test_subjects": 20,
    "test_adhd": 10,
    "test_control": 10,
    "test_site": "Peking_2 (Locked)",
    "sites_count": 8,
    "sites_list": [
      "KKI",
      "NYU",
      "NeuroIMAGE",
      "OHSU",
      "Peking_1",
      "Peking_2",
      "Peking_3",
      "Pittsburgh"
    ]
  },
  "final_model_metadata": {
    "run_id": "run_20261004_182318_final_locked",
    "status": "FINAL_LOCKED_TEST_EVALUATION",
    "model_architecture": "RandomForestClassifier",
    "feature_selection": "SelectKBest(f_classif, k=100)",
    "hyperparameters": {
      "n_estimators": 50,
      "max_depth": 3,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "max_features": "sqrt",
      "class_weight": "balanced",
      "random_state": 42
    },
    "refit_dataset_subjects": 133,
    "test_dataset_subjects": 20,
    "test_site": "Peking_2",
    "final_test_metrics": {
      "accuracy": 0.6,
      "balanced_accuracy": 0.6,
      "precision": 0.6,
      "recall_sensitivity": 0.6,
      "specificity": 0.6,
      "f1_score": 0.6,
      "roc_auc": 0.56,
      "pr_auc": 0.634112926547137,
      "confusion_matrix": {
        "tp": 6,
        "tn": 6,
        "fp": 4,
        "fn": 4
      }
    },
    "baseline_comparison_deltas": {
      "accuracy_delta": 0.05,
      "balanced_accuracy_delta": 0.05,
      "sensitivity_delta": 0.3,
      "specificity_delta": -0.2,
      "f1_score_delta": 0.2,
      "roc_auc_delta": -0.07
    }
  },
  "final_test_results": {
    "run_id": "run_20261004_182318_final_locked",
    "test_site": "Peking_2",
    "test_subject_count": 20,
    "metrics": {
      "accuracy": 0.6,
      "balanced_accuracy": 0.6,
      "precision": 0.6,
      "recall_sensitivity": 0.6,
      "specificity": 0.6,
      "f1_score": 0.6,
      "roc_auc": 0.56,
      "pr_auc": 0.634112926547137,
      "confusion_matrix": {
        "tp": 6,
        "tn": 6,
        "fp": 4,
        "fn": 4
      }
    },
    "evaluation_timestamp": "2026-10-04T18:23:20.274625+00:00"
  },
  "final_confusion_matrix": {
    "tp": 6,
    "tn": 6,
    "fp": 4,
    "fn": 4
  },
  "rf_robustness": {
    "summary": {
      "new_repetitions_seeds_47_48_49": {
        "roc_auc": {
          "mean": 0.6022401433691756,
          "std": 0.008502163071322214,
          "min": 0.592741935483871,
          "max": 0.6091397849462366
        },
        "pr_auc": {
          "mean": 0.3814645643113215,
          "std": 0.019973124749447815,
          "min": 0.3584378752160872,
          "max": 0.39409791075204875
        },
        "accuracy": {
          "mean": 0.6441102756892231,
          "std": 0.015651624056136324,
          "min": 0.631578947368421,
          "max": 0.6616541353383458
        },
        "balanced_accuracy": {
          "mean": 0.6077956989247312,
          "std": 0.014352799782707835,
          "min": 0.5940860215053764,
          "max": 0.6227150537634409
        },
        "recall_sensitivity": {
          "mean": 0.5166666666666666,
          "std": 0.014433756729740658,
          "min": 0.5,
          "max": 0.525
        },
        "specificity": {
          "mean": 0.6989247311827956,
          "std": 0.018624202231923393,
          "min": 0.6881720430107527,
          "max": 0.7204301075268817
        },
        "f1_score": {
          "mean": 0.4662878298678376,
          "std": 0.01666343929941647,
          "min": 0.449438202247191,
          "max": 0.4827586206896552
        }
      },
      "existing_repetitions_seeds_42_46": {
        "roc_auc": {
          "mean": 0.6058064516129031,
          "std": 0.02053735292268957,
          "min": 0.589247311827957,
          "max": 0.6403225806451612
        },
        "pr_auc": {
          "mean": 0.3815052195436308,
          "std": 0.026012621367683907,
          "min": 0.35218140680353816,
          "max": 0.4186966530073861
        },
        "accuracy": {
          "mean": 0.6451127819548872,
          "std": 0.014462694783211548,
          "min": 0.631578947368421,
          "max": 0.6691729323308271
        },
        "balanced_accuracy": {
          "mean": 0.599489247311828,
          "std": 0.018085175638358472,
          "min": 0.5780913978494624,
          "max": 0.6280913978494624
        },
        "recall_sensitivity": {
          "mean": 0.48500000000000004,
          "std": 0.03791437722025776,
          "min": 0.425,
          "max": 0.525
        },
        "specificity": {
          "mean": 0.713978494623656,
          "std": 0.01953322809095154,
          "min": 0.6881720430107527,
          "max": 0.7311827956989247
        },
        "f1_score": {
          "mean": 0.4508097439373553,
          "std": 0.026217885090778208,
          "min": 0.4146341463414634,
          "max": 0.4883720930232558
        }
      },
      "all_8_repetitions": {
        "roc_auc": {
          "mean": 0.6044690860215054,
          "std": 0.016281243826363652,
          "min": 0.589247311827957,
          "max": 0.6403225806451612
        },
        "pr_auc": {
          "mean": 0.3814899738315148,
          "std": 0.02237498732729981,
          "min": 0.35218140680353816,
          "max": 0.4186966530073861
        },
        "accuracy": {
          "mean": 0.644736842105263,
          "std": 0.013776321523502322,
          "min": 0.631578947368421,
          "max": 0.6691729323308271
        },
        "balanced_accuracy": {
          "mean": 0.6026041666666666,
          "std": 0.016255415225972344,
          "min": 0.5780913978494624,
          "max": 0.6280913978494624
        },
        "recall_sensitivity": {
          "mean": 0.49687499999999996,
          "std": 0.03390506704651345,
          "min": 0.425,
          "max": 0.525
        },
        "specificity": {
          "mean": 0.7083333333333334,
          "std": 0.01943786595214094,
          "min": 0.6881720430107527,
          "max": 0.7311827956989247
        },
        "f1_score": {
          "mean": 0.4566140261612862,
          "std": 0.023157986305985676,
          "min": 0.4146341463414634,
          "max": 0.4883720930232558
        }
      }
    },
    "repetitions": [
      {
        "repetition": 1,
        "seed": 42,
        "group": "Existing (42-46)",
        "accuracy": 0.6390977443609023,
        "balanced_accuracy": 0.5994623655913979,
        "precision": 0.4166666666666667,
        "recall_sensitivity": 0.5,
        "specificity": 0.6989247311827957,
        "f1_score": 0.4545454545454545,
        "roc_auc": 0.5919354838709677,
        "pr_auc": 0.3521814068035381,
        "tp": 20,
        "tn": 65,
        "fp": 28,
        "fn": 20
      },
      {
        "repetition": 2,
        "seed": 43,
        "group": "Existing (42-46)",
        "accuracy": 0.631578947368421,
        "balanced_accuracy": 0.5940860215053764,
        "precision": 0.4081632653061224,
        "recall_sensitivity": 0.5,
        "specificity": 0.6881720430107527,
        "f1_score": 0.449438202247191,
        "roc_auc": 0.6069892473118279,
        "pr_auc": 0.3960659370565496,
        "tp": 20,
        "tn": 64,
        "fp": 29,
        "fn": 20
      },
      {
        "repetition": 3,
        "seed": 44,
        "group": "Existing (42-46)",
        "accuracy": 0.6390977443609023,
        "balanced_accuracy": 0.5780913978494624,
        "precision": 0.4047619047619047,
        "recall_sensitivity": 0.425,
        "specificity": 0.7311827956989247,
        "f1_score": 0.4146341463414634,
        "roc_auc": 0.6005376344086021,
        "pr_auc": 0.3703522568413084,
        "tp": 17,
        "tn": 68,
        "fp": 25,
        "fn": 23
      },
      {
        "repetition": 4,
        "seed": 45,
        "group": "Existing (42-46)",
        "accuracy": 0.6691729323308271,
        "balanced_accuracy": 0.6280913978494624,
        "precision": 0.4565217391304347,
        "recall_sensitivity": 0.525,
        "specificity": 0.7311827956989247,
        "f1_score": 0.4883720930232558,
        "roc_auc": 0.589247311827957,
        "pr_auc": 0.3702298440093716,
        "tp": 21,
        "tn": 68,
        "fp": 25,
        "fn": 19
      },
      {
        "repetition": 5,
        "seed": 46,
        "group": "Existing (42-46)",
        "accuracy": 0.6466165413533834,
        "balanced_accuracy": 0.5977150537634408,
        "precision": 0.4222222222222222,
        "recall_sensitivity": 0.475,
        "specificity": 0.7204301075268817,
        "f1_score": 0.4470588235294118,
        "roc_auc": 0.6403225806451612,
        "pr_auc": 0.4186966530073861,
        "tp": 19,
        "tn": 67,
        "fp": 26,
        "fn": 21
      },
      {
        "repetition": 6,
        "seed": 47,
        "group": "New (47-49)",
        "accuracy": 0.6390977443609023,
        "balanced_accuracy": 0.6065860215053764,
        "precision": 0.42,
        "recall_sensitivity": 0.525,
        "specificity": 0.6881720430107527,
        "f1_score": 0.4666666666666667,
        "roc_auc": 0.6091397849462366,
        "pr_auc": 0.3918579069658284,
        "tp": 21,
        "tn": 64,
        "fp": 29,
        "fn": 19
      },
      {
        "repetition": 7,
        "seed": 48,
        "group": "New (47-49)",
        "accuracy": 0.631578947368421,
        "balanced_accuracy": 0.5940860215053764,
        "precision": 0.4081632653061224,
        "recall_sensitivity": 0.5,
        "specificity": 0.6881720430107527,
        "f1_score": 0.449438202247191,
        "roc_auc": 0.592741935483871,
        "pr_auc": 0.3940979107520487,
        "tp": 20,
        "tn": 64,
        "fp": 29,
        "fn": 20
      },
      {
        "repetition": 8,
        "seed": 49,
        "group": "New (47-49)",
        "accuracy": 0.6616541353383458,
        "balanced_accuracy": 0.6227150537634409,
        "precision": 0.4468085106382978,
        "recall_sensitivity": 0.525,
        "specificity": 0.7204301075268817,
        "f1_score": 0.4827586206896552,
        "roc_auc": 0.6048387096774194,
        "pr_auc": 0.3584378752160872,
        "tp": 21,
        "tn": 67,
        "fp": 26,
        "fn": 19
      }
    ]
  },
  "hybrid_comparison": {
    "summary": {
      "hard_voting": {
        "roc_auc": {
          "mean": 0.5925806451612903,
          "std": 0.022915842857201132,
          "min": 0.5588709677419356,
          "max": 0.6303763440860215
        },
        "pr_auc": {
          "mean": 0.3682162265971508,
          "std": 0.017806884725009218,
          "min": 0.33889145464630255,
          "max": 0.39226526194688405
        },
        "accuracy": {
          "mean": 0.631578947368421,
          "std": 0.019606623775045542,
          "min": 0.5939849624060151,
          "max": 0.6466165413533834
        },
        "balanced_accuracy": {
          "mean": 0.5798387096774194,
          "std": 0.022638973564642565,
          "min": 0.5387096774193548,
          "max": 0.5994623655913979
        },
        "recall_sensitivity": {
          "mean": 0.45,
          "std": 0.03535533905932737,
          "min": 0.4,
          "max": 0.5
        },
        "specificity": {
          "mean": 0.7096774193548387,
          "std": 0.019234993354836923,
          "min": 0.6774193548387096,
          "max": 0.7311827956989247
        },
        "f1_score": {
          "mean": 0.42341415906131674,
          "std": 0.029860919274989465,
          "min": 0.37209302325581395,
          "max": 0.45454545454545453
        }
      },
      "soft_voting": {
        "roc_auc": {
          "mean": 0.5925806451612903,
          "std": 0.022915842857201132,
          "min": 0.5588709677419356,
          "max": 0.6303763440860215
        },
        "pr_auc": {
          "mean": 0.3682162265971508,
          "std": 0.017806884725009218,
          "min": 0.33889145464630255,
          "max": 0.39226526194688405
        },
        "accuracy": {
          "mean": 0.631578947368421,
          "std": 0.019606623775045542,
          "min": 0.5939849624060151,
          "max": 0.6466165413533834
        },
        "balanced_accuracy": {
          "mean": 0.5798387096774194,
          "std": 0.022638973564642565,
          "min": 0.5387096774193548,
          "max": 0.5994623655913979
        },
        "recall_sensitivity": {
          "mean": 0.45,
          "std": 0.03535533905932737,
          "min": 0.4,
          "max": 0.5
        },
        "specificity": {
          "mean": 0.7096774193548387,
          "std": 0.019234993354836923,
          "min": 0.6774193548387096,
          "max": 0.7311827956989247
        },
        "f1_score": {
          "mean": 0.42341415906131674,
          "std": 0.029860919274989465,
          "min": 0.37209302325581395,
          "max": 0.45454545454545453
        }
      },
      "stacking": {
        "roc_auc": {
          "mean": 0.5555376344086022,
          "std": 0.0279547145476589,
          "min": 0.5247311827956989,
          "max": 0.6010752688172043
        },
        "pr_auc": {
          "mean": 0.3380848036263645,
          "std": 0.017681783784232116,
          "min": 0.3176325122723813,
          "max": 0.36701675867554806
        },
        "accuracy": {
          "mean": 0.5578947368421053,
          "std": 0.012935827469237049,
          "min": 0.5413533834586466,
          "max": 0.5789473684210527
        },
        "balanced_accuracy": {
          "mean": 0.5428225806451613,
          "std": 0.019110729123356738,
          "min": 0.5153225806451613,
          "max": 0.5706989247311829
        },
        "recall_sensitivity": {
          "mean": 0.505,
          "std": 0.04847679857416329,
          "min": 0.45,
          "max": 0.575
        },
        "specificity": {
          "mean": 0.5806451612903225,
          "std": 0.022555028992906505,
          "min": 0.5376344086021505,
          "max": 0.6021505376344086
        },
        "f1_score": {
          "mean": 0.4065364390844842,
          "std": 0.026299947012733665,
          "min": 0.3711340206185567,
          "max": 0.44
        }
      }
    },
    "rankings": [
      {
        "rank": 1,
        "hybrid_type": "hard_voting",
        "display_name": "Hard Voting (RF + XGBoost)",
        "mean_roc_auc": 0.5925806451612903,
        "std_roc_auc": 0.022915842857201132,
        "mean_balanced_acc": 0.5798387096774194,
        "mean_sensitivity": 0.45,
        "mean_f1": 0.42341415906131674
      },
      {
        "rank": 2,
        "hybrid_type": "soft_voting",
        "display_name": "Soft Voting (0.6 RF / 0.4 XGBoost)",
        "mean_roc_auc": 0.5925806451612903,
        "std_roc_auc": 0.022915842857201132,
        "mean_balanced_acc": 0.5798387096774194,
        "mean_sensitivity": 0.45,
        "mean_f1": 0.42341415906131674
      },
      {
        "rank": 3,
        "hybrid_type": "stacking",
        "display_name": "Stacking (Balanced LogReg Meta)",
        "mean_roc_auc": 0.5555376344086022,
        "std_roc_auc": 0.0279547145476589,
        "mean_balanced_acc": 0.5428225806451613,
        "mean_sensitivity": 0.505,
        "mean_f1": 0.4065364390844842
      }
    ]
  },
  "explainability": {
    "top_edges": [
      {
        "feature_name": "ROI_023__ROI_189",
        "roi_a_id": 23,
        "roi_a_name": "ROI_023",
        "roi_a_region": "[CC200 Functional Cluster Node 23]",
        "roi_a_network": "[CC200 Cluster 23]",
        "roi_b_id": 189,
        "roi_b_name": "ROI_189",
        "roi_b_region": "[CC200 Functional Cluster Node 189]",
        "roi_b_network": "[CC200 Cluster 189]",
        "mean_abs_shap": 0.0167622092884908,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0046774193548387,
        "permutation_importance_std": 0.0018080829807312,
        "shap_rank": 1,
        "perm_rank": 9
      },
      {
        "feature_name": "ROI_100__ROI_153",
        "roi_a_id": 100,
        "roi_a_name": "ROI_100",
        "roi_a_region": "[CC200 Functional Cluster Node 100]",
        "roi_a_network": "[CC200 Cluster 100]",
        "roi_b_id": 153,
        "roi_b_name": "ROI_153",
        "roi_b_region": "[CC200 Functional Cluster Node 153]",
        "roi_b_network": "[CC200 Cluster 153]",
        "mean_abs_shap": 0.0161654820749022,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0083870967741936,
        "permutation_importance_std": 0.0024836149381615,
        "shap_rank": 2,
        "perm_rank": 3
      },
      {
        "feature_name": "ROI_034__ROI_041",
        "roi_a_id": 34,
        "roi_a_name": "ROI_034",
        "roi_a_region": "[CC200 Functional Cluster Node 34]",
        "roi_a_network": "[CC200 Cluster 34]",
        "roi_b_id": 41,
        "roi_b_name": "ROI_041",
        "roi_b_region": "[CC200 Functional Cluster Node 41]",
        "roi_b_network": "[CC200 Cluster 41]",
        "mean_abs_shap": 0.015495812483631,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0086021505376344,
        "permutation_importance_std": 0.0055078014651944,
        "shap_rank": 3,
        "perm_rank": 2
      },
      {
        "feature_name": "ROI_143__ROI_153",
        "roi_a_id": 143,
        "roi_a_name": "ROI_143",
        "roi_a_region": "[CC200 Functional Cluster Node 143]",
        "roi_a_network": "[CC200 Cluster 143]",
        "roi_b_id": 153,
        "roi_b_name": "ROI_153",
        "roi_b_region": "[CC200 Functional Cluster Node 153]",
        "roi_b_network": "[CC200 Cluster 153]",
        "mean_abs_shap": 0.0142647387869038,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.006747311827957,
        "permutation_importance_std": 0.0009651248962911,
        "shap_rank": 4,
        "perm_rank": 5
      },
      {
        "feature_name": "ROI_023__ROI_103",
        "roi_a_id": 23,
        "roi_a_name": "ROI_023",
        "roi_a_region": "[CC200 Functional Cluster Node 23]",
        "roi_a_network": "[CC200 Cluster 23]",
        "roi_b_id": 103,
        "roi_b_name": "ROI_103",
        "roi_b_region": "[CC200 Functional Cluster Node 103]",
        "roi_b_network": "[CC200 Cluster 103]",
        "mean_abs_shap": 0.0141425940534273,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0091129032258065,
        "permutation_importance_std": 0.0019364170390808,
        "shap_rank": 5,
        "perm_rank": 1
      },
      {
        "feature_name": "ROI_003__ROI_006",
        "roi_a_id": 3,
        "roi_a_name": "ROI_003",
        "roi_a_region": "[CC200 Functional Cluster Node 3]",
        "roi_a_network": "[CC200 Cluster 3]",
        "roi_b_id": 6,
        "roi_b_name": "ROI_006",
        "roi_b_region": "[CC200 Functional Cluster Node 6]",
        "roi_b_network": "[CC200 Cluster 6]",
        "mean_abs_shap": 0.0138578317042686,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0044086021505377,
        "permutation_importance_std": 0.0042251360654599,
        "shap_rank": 6,
        "perm_rank": 10
      },
      {
        "feature_name": "ROI_056__ROI_138",
        "roi_a_id": 56,
        "roi_a_name": "ROI_056",
        "roi_a_region": "[CC200 Functional Cluster Node 56]",
        "roi_a_network": "[CC200 Cluster 56]",
        "roi_b_id": 138,
        "roi_b_name": "ROI_138",
        "roi_b_region": "[CC200 Functional Cluster Node 138]",
        "roi_b_network": "[CC200 Cluster 138]",
        "mean_abs_shap": 0.0132217333124712,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0034408602150537,
        "permutation_importance_std": 0.0028494623655913,
        "shap_rank": 7,
        "perm_rank": 16
      },
      {
        "feature_name": "ROI_026__ROI_153",
        "roi_a_id": 26,
        "roi_a_name": "ROI_026",
        "roi_a_region": "[CC200 Functional Cluster Node 26]",
        "roi_a_network": "[CC200 Cluster 26]",
        "roi_b_id": 153,
        "roi_b_name": "ROI_153",
        "roi_b_region": "[CC200 Functional Cluster Node 153]",
        "roi_b_network": "[CC200 Cluster 153]",
        "mean_abs_shap": 0.0127070672935729,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0054569892473118,
        "permutation_importance_std": 0.0016529525956605,
        "shap_rank": 8,
        "perm_rank": 7
      },
      {
        "feature_name": "ROI_034__ROI_103",
        "roi_a_id": 34,
        "roi_a_name": "ROI_034",
        "roi_a_region": "[CC200 Functional Cluster Node 34]",
        "roi_a_network": "[CC200 Cluster 34]",
        "roi_b_id": 103,
        "roi_b_name": "ROI_103",
        "roi_b_region": "[CC200 Functional Cluster Node 103]",
        "roi_b_network": "[CC200 Cluster 103]",
        "mean_abs_shap": 0.0125732956825545,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0036290322580645,
        "permutation_importance_std": 0.0024021190109693,
        "shap_rank": 9,
        "perm_rank": 14
      },
      {
        "feature_name": "ROI_006__ROI_127",
        "roi_a_id": 6,
        "roi_a_name": "ROI_006",
        "roi_a_region": "[CC200 Functional Cluster Node 6]",
        "roi_a_network": "[CC200 Cluster 6]",
        "roi_b_id": 127,
        "roi_b_name": "ROI_127",
        "roi_b_region": "[CC200 Functional Cluster Node 127]",
        "roi_b_network": "[CC200 Cluster 127]",
        "mean_abs_shap": 0.0121536972416562,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0018817204301075,
        "permutation_importance_std": 0.0025727927126893,
        "shap_rank": 10,
        "perm_rank": 32
      },
      {
        "feature_name": "ROI_003__ROI_073",
        "roi_a_id": 3,
        "roi_a_name": "ROI_003",
        "roi_a_region": "[CC200 Functional Cluster Node 3]",
        "roi_a_network": "[CC200 Cluster 3]",
        "roi_b_id": 73,
        "roi_b_name": "ROI_073",
        "roi_b_region": "[CC200 Functional Cluster Node 73]",
        "roi_b_network": "[CC200 Cluster 73]",
        "mean_abs_shap": 0.0117395984547504,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0039784946236559,
        "permutation_importance_std": 0.0014009477656769,
        "shap_rank": 11,
        "perm_rank": 12
      },
      {
        "feature_name": "ROI_016__ROI_174",
        "roi_a_id": 16,
        "roi_a_name": "ROI_016",
        "roi_a_region": "[CC200 Functional Cluster Node 16]",
        "roi_a_network": "[CC200 Cluster 16]",
        "roi_b_id": 174,
        "roi_b_name": "ROI_174",
        "roi_b_region": "[CC200 Functional Cluster Node 174]",
        "roi_b_network": "[CC200 Cluster 174]",
        "mean_abs_shap": 0.0103403517273842,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0028763440860214,
        "permutation_importance_std": 0.0015256412498562,
        "shap_rank": 12,
        "perm_rank": 21
      },
      {
        "feature_name": "ROI_003__ROI_017",
        "roi_a_id": 3,
        "roi_a_name": "ROI_003",
        "roi_a_region": "[CC200 Functional Cluster Node 3]",
        "roi_a_network": "[CC200 Cluster 3]",
        "roi_b_id": 17,
        "roi_b_name": "ROI_017",
        "roi_b_region": "[CC200 Functional Cluster Node 17]",
        "roi_b_network": "[CC200 Cluster 17]",
        "mean_abs_shap": 0.0098575549052058,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0013172043010752,
        "permutation_importance_std": 0.0010090496978023,
        "shap_rank": 13,
        "perm_rank": 49
      },
      {
        "feature_name": "ROI_058__ROI_151",
        "roi_a_id": 58,
        "roi_a_name": "ROI_058",
        "roi_a_region": "[CC200 Functional Cluster Node 58]",
        "roi_a_network": "[CC200 Cluster 58]",
        "roi_b_id": 151,
        "roi_b_name": "ROI_151",
        "roi_b_region": "[CC200 Functional Cluster Node 151]",
        "roi_b_network": "[CC200 Cluster 151]",
        "mean_abs_shap": 0.0091493196680119,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0076344086021505,
        "permutation_importance_std": 0.0014436259765696,
        "shap_rank": 14,
        "perm_rank": 4
      },
      {
        "feature_name": "ROI_003__ROI_034",
        "roi_a_id": 3,
        "roi_a_name": "ROI_003",
        "roi_a_region": "[CC200 Functional Cluster Node 3]",
        "roi_a_network": "[CC200 Cluster 3]",
        "roi_b_id": 34,
        "roi_b_name": "ROI_034",
        "roi_b_region": "[CC200 Functional Cluster Node 34]",
        "roi_b_network": "[CC200 Cluster 34]",
        "mean_abs_shap": 0.0085502744836434,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0019354838709677,
        "permutation_importance_std": 0.0013158320699484,
        "shap_rank": 15,
        "perm_rank": 31
      },
      {
        "feature_name": "ROI_070__ROI_149",
        "roi_a_id": 70,
        "roi_a_name": "ROI_070",
        "roi_a_region": "[CC200 Functional Cluster Node 70]",
        "roi_a_network": "[CC200 Cluster 70]",
        "roi_b_id": 149,
        "roi_b_name": "ROI_149",
        "roi_b_region": "[CC200 Functional Cluster Node 149]",
        "roi_b_network": "[CC200 Cluster 149]",
        "mean_abs_shap": 0.0079712375695787,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0061290322580645,
        "permutation_importance_std": 0.0019189858384489,
        "shap_rank": 16,
        "perm_rank": 6
      },
      {
        "feature_name": "ROI_049__ROI_100",
        "roi_a_id": 49,
        "roi_a_name": "ROI_049",
        "roi_a_region": "[CC200 Functional Cluster Node 49]",
        "roi_a_network": "[CC200 Cluster 49]",
        "roi_b_id": 100,
        "roi_b_name": "ROI_100",
        "roi_b_region": "[CC200 Functional Cluster Node 100]",
        "roi_b_network": "[CC200 Cluster 100]",
        "mean_abs_shap": 0.0078678881732019,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0015860215053763,
        "permutation_importance_std": 0.0019697166776045,
        "shap_rank": 17,
        "perm_rank": 40
      },
      {
        "feature_name": "ROI_038__ROI_189",
        "roi_a_id": 38,
        "roi_a_name": "ROI_038",
        "roi_a_region": "[CC200 Functional Cluster Node 38]",
        "roi_a_network": "[CC200 Cluster 38]",
        "roi_b_id": 189,
        "roi_b_name": "ROI_189",
        "roi_b_region": "[CC200 Functional Cluster Node 189]",
        "roi_b_network": "[CC200 Cluster 189]",
        "mean_abs_shap": 0.0076205970454943,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0014247311827957,
        "permutation_importance_std": 0.0010756047862227,
        "shap_rank": 18,
        "perm_rank": 45
      },
      {
        "feature_name": "ROI_006__ROI_115",
        "roi_a_id": 6,
        "roi_a_name": "ROI_006",
        "roi_a_region": "[CC200 Functional Cluster Node 6]",
        "roi_a_network": "[CC200 Cluster 6]",
        "roi_b_id": 115,
        "roi_b_name": "ROI_115",
        "roi_b_region": "[CC200 Functional Cluster Node 115]",
        "roi_b_network": "[CC200 Cluster 115]",
        "mean_abs_shap": 0.0074754311083288,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0035483870967741,
        "permutation_importance_std": 0.0013158320699484,
        "shap_rank": 19,
        "perm_rank": 15
      },
      {
        "feature_name": "ROI_003__ROI_151",
        "roi_a_id": 3,
        "roi_a_name": "ROI_003",
        "roi_a_region": "[CC200 Functional Cluster Node 3]",
        "roi_a_network": "[CC200 Cluster 3]",
        "roi_b_id": 151,
        "roi_b_name": "ROI_151",
        "roi_b_region": "[CC200 Functional Cluster Node 151]",
        "roi_b_network": "[CC200 Cluster 151]",
        "mean_abs_shap": 0.0072742855481976,
        "direction": "Higher connectivity increases ADHD probability",
        "permutation_importance_mean": 0.0049193548387097,
        "permutation_importance_std": 0.00097703128201,
        "shap_rank": 20,
        "perm_rank": 8
      }
    ],
    "hub_rois": [
      {
        "hub_rank": 1,
        "roi_id": 3,
        "roi_name": "ROI_003",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 3 (Craddock 200 Cluster)]",
        "top_50_edge_count": 8,
        "top_100_edge_count": 16,
        "mean_edge_shap_importance": 0.0050253808208711
      },
      {
        "hub_rank": 2,
        "roi_id": 34,
        "roi_name": "ROI_034",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 34 (Craddock 200 Cluster)]",
        "top_50_edge_count": 6,
        "top_100_edge_count": 9,
        "mean_edge_shap_importance": 0.0064995513533908
      },
      {
        "hub_rank": 3,
        "roi_id": 174,
        "roi_name": "ROI_174",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 174 (Craddock 200 Cluster)]",
        "top_50_edge_count": 6,
        "top_100_edge_count": 9,
        "mean_edge_shap_importance": 0.0048576758726145
      },
      {
        "hub_rank": 4,
        "roi_id": 189,
        "roi_name": "ROI_189",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 189 (Craddock 200 Cluster)]",
        "top_50_edge_count": 6,
        "top_100_edge_count": 7,
        "mean_edge_shap_importance": 0.0065759451816573
      },
      {
        "hub_rank": 5,
        "roi_id": 6,
        "roi_name": "ROI_006",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 6 (Craddock 200 Cluster)]",
        "top_50_edge_count": 4,
        "top_100_edge_count": 6,
        "mean_edge_shap_importance": 0.00691435081426
      },
      {
        "hub_rank": 6,
        "roi_id": 151,
        "roi_name": "ROI_151",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 151 (Craddock 200 Cluster)]",
        "top_50_edge_count": 4,
        "top_100_edge_count": 5,
        "mean_edge_shap_importance": 0.0056743075204724
      },
      {
        "hub_rank": 7,
        "roi_id": 23,
        "roi_name": "ROI_023",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 23 (Craddock 200 Cluster)]",
        "top_50_edge_count": 3,
        "top_100_edge_count": 6,
        "mean_edge_shap_importance": 0.0067967309665191
      },
      {
        "hub_rank": 8,
        "roi_id": 153,
        "roi_name": "ROI_153",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 153 (Craddock 200 Cluster)]",
        "top_50_edge_count": 3,
        "top_100_edge_count": 4,
        "mean_edge_shap_importance": 0.0107843220388447
      },
      {
        "hub_rank": 9,
        "roi_id": 100,
        "roi_name": "ROI_100",
        "atlas": "CC200 (Craddock 200)",
        "anatomical_label": "[CC200 Node 100 (Craddock 200 Cluster)]",
        "top_50_edge_count": 3,
        "top_100_edge_count": 4,
        "mean_edge_shap_importance": 0.0072674651605834
      }
    ],
    "roi_summary": [
      {
        "roi_name": "ROI_003",
        "roi_id": 3,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 5,
        "appearances_in_top_50_edges": 8,
        "appearances_in_all_100_edges": 16
      },
      {
        "roi_name": "ROI_034",
        "roi_id": 34,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 3,
        "appearances_in_top_50_edges": 6,
        "appearances_in_all_100_edges": 9
      },
      {
        "roi_name": "ROI_006",
        "roi_id": 6,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 3,
        "appearances_in_top_50_edges": 4,
        "appearances_in_all_100_edges": 6
      },
      {
        "roi_name": "ROI_153",
        "roi_id": 153,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 3,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 4
      },
      {
        "roi_name": "ROI_189",
        "roi_id": 189,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 2,
        "appearances_in_top_50_edges": 6,
        "appearances_in_all_100_edges": 7
      },
      {
        "roi_name": "ROI_151",
        "roi_id": 151,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 2,
        "appearances_in_top_50_edges": 4,
        "appearances_in_all_100_edges": 5
      },
      {
        "roi_name": "ROI_023",
        "roi_id": 23,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 2,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 6
      },
      {
        "roi_name": "ROI_100",
        "roi_id": 100,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 2,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 4
      },
      {
        "roi_name": "ROI_103",
        "roi_id": 103,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 2,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 2
      },
      {
        "roi_name": "ROI_174",
        "roi_id": 174,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 6,
        "appearances_in_all_100_edges": 9
      },
      {
        "roi_name": "ROI_149",
        "roi_id": 149,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 5
      },
      {
        "roi_name": "ROI_138",
        "roi_id": 138,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 4
      },
      {
        "roi_name": "ROI_026",
        "roi_id": 26,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_115",
        "roi_id": 115,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_056",
        "roi_id": 56,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 5
      },
      {
        "roi_name": "ROI_038",
        "roi_id": 38,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 4
      },
      {
        "roi_name": "ROI_058",
        "roi_id": 58,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_073",
        "roi_id": 73,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 2
      },
      {
        "roi_name": "ROI_017",
        "roi_id": 17,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_143",
        "roi_id": 143,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_070",
        "roi_id": 70,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 2
      },
      {
        "roi_name": "ROI_049",
        "roi_id": 49,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 2
      },
      {
        "roi_name": "ROI_127",
        "roi_id": 127,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 1
      },
      {
        "roi_name": "ROI_041",
        "roi_id": 41,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 1
      },
      {
        "roi_name": "ROI_016",
        "roi_id": 16,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 1,
        "appearances_in_top_50_edges": 1,
        "appearances_in_all_100_edges": 1
      },
      {
        "roi_name": "ROI_166",
        "roi_id": 166,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 0,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 9
      },
      {
        "roi_name": "ROI_105",
        "roi_id": 105,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 0,
        "appearances_in_top_50_edges": 3,
        "appearances_in_all_100_edges": 7
      },
      {
        "roi_name": "ROI_050",
        "roi_id": 50,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 0,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_061",
        "roi_id": 61,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 0,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 3
      },
      {
        "roi_name": "ROI_072",
        "roi_id": 72,
        "atlas": "CC200 (Craddock 200)",
        "appearances_in_top_20_edges": 0,
        "appearances_in_top_50_edges": 2,
        "appearances_in_all_100_edges": 2
      }
    ],
    "subject_explanations": [
      {
        "category": "True Positive (High-Confidence ADHD)",
        "cohort": "Development",
        "subject_id": 3310328,
        "site": "KKI",
        "actual_diagnosis": "ADHD",
        "predicted_class": "ADHD",
        "prediction_probability_adhd": 0.8833859088398489,
        "top_positive_shap_contributors": "ROI_023__ROI_103 (SHAP=+0.0187, z=1.35) | ROI_003__ROI_006 (SHAP=+0.0163, z=1.66) | ROI_034__ROI_041 (SHAP=+0.0155, z=2.04) | ROI_115__ROI_174 (SHAP=+0.0142, z=1.59) | ROI_026__ROI_153 (SHAP=+0.0128, z=0.77)",
        "top_negative_shap_contributors": "ROI_100__ROI_153 (SHAP=-0.0034, z=0.17) | ROI_034__ROI_081 (SHAP=-0.0015, z=2.27) | ROI_011__ROI_189 (SHAP=-0.0012, z=2.35) | ROI_019__ROI_093 (SHAP=-0.0010, z=2.10) | ROI_070__ROI_166 (SHAP=-0.0009, z=1.56)"
      },
      {
        "category": "Lowest-Margin ADHD (Near-Boundary Prediction)",
        "cohort": "Development",
        "subject_id": 27003,
        "site": "NeuroIMAGE",
        "actual_diagnosis": "ADHD",
        "predicted_class": "ADHD",
        "prediction_probability_adhd": 0.5232177474081865,
        "top_positive_shap_contributors": "ROI_023__ROI_103 (SHAP=+0.0269, z=1.09) | ROI_023__ROI_189 (SHAP=+0.0232, z=0.98) | ROI_026__ROI_153 (SHAP=+0.0217, z=0.85) | ROI_034__ROI_103 (SHAP=+0.0161, z=0.53) | ROI_100__ROI_153 (SHAP=+0.0151, z=0.60)",
        "top_negative_shap_contributors": "ROI_143__ROI_153 (SHAP=-0.0156, z=-0.40) | ROI_073__ROI_174 (SHAP=-0.0134, z=-1.35) | ROI_016__ROI_174 (SHAP=-0.0122, z=-2.02) | ROI_056__ROI_138 (SHAP=-0.0098, z=-0.50) | ROI_149__ROI_179 (SHAP=-0.0081, z=-0.66)"
      },
      {
        "category": "True Negative (High-Confidence Control)",
        "cohort": "Development",
        "subject_id": 2081754,
        "site": "Peking_1",
        "actual_diagnosis": "Control",
        "predicted_class": "Control",
        "prediction_probability_adhd": 0.0656929354380125,
        "top_positive_shap_contributors": "ROI_058__ROI_151 (SHAP=+0.0080, z=0.48) | ROI_003__ROI_013 (SHAP=+0.0052, z=0.11) | ROI_003__ROI_151 (SHAP=+0.0035, z=0.85) | ROI_149__ROI_179 (SHAP=+0.0019, z=-0.06) | ROI_031__ROI_144 (SHAP=+0.0019, z=-1.95)",
        "top_negative_shap_contributors": "ROI_034__ROI_103 (SHAP=-0.0289, z=-1.80) | ROI_056__ROI_138 (SHAP=-0.0268, z=-1.08) | ROI_003__ROI_006 (SHAP=-0.0254, z=-0.90) | ROI_038__ROI_189 (SHAP=-0.0240, z=-1.11) | ROI_143__ROI_153 (SHAP=-0.0209, z=-0.41)"
      },
      {
        "category": "False Positive (Misclassified Control)",
        "cohort": "Development",
        "subject_id": 16034,
        "site": "Pittsburgh",
        "actual_diagnosis": "Control",
        "predicted_class": "ADHD",
        "prediction_probability_adhd": 0.7542721343355021,
        "top_positive_shap_contributors": "ROI_034__ROI_041 (SHAP=+0.0159, z=0.70) | ROI_100__ROI_153 (SHAP=+0.0138, z=0.92) | ROI_023__ROI_103 (SHAP=+0.0129, z=1.81) | ROI_058__ROI_151 (SHAP=+0.0128, z=-0.04) | ROI_003__ROI_115 (SHAP=+0.0122, z=1.08)",
        "top_negative_shap_contributors": "ROI_026__ROI_153 (SHAP=-0.0151, z=-0.32) | ROI_070__ROI_149 (SHAP=-0.0151, z=-0.04) | ROI_003__ROI_073 (SHAP=-0.0081, z=-0.01) | ROI_002__ROI_178 (SHAP=-0.0043, z=0.85) | ROI_115__ROI_174 (SHAP=-0.0043, z=0.89)"
      },
      {
        "category": "Peking_2 Test: True Positive",
        "cohort": "Peking_2 Test",
        "subject_id": 3827352,
        "site": "Peking_2",
        "actual_diagnosis": "ADHD",
        "predicted_class": "ADHD",
        "prediction_probability_adhd": 0.6223824416562561,
        "top_positive_shap_contributors": "ROI_023__ROI_103 (SHAP=+0.0189, z=1.08) | ROI_026__ROI_153 (SHAP=+0.0179, z=1.78) | ROI_023__ROI_189 (SHAP=+0.0160, z=0.19) | ROI_003__ROI_034 (SHAP=+0.0153, z=0.26) | ROI_016__ROI_174 (SHAP=+0.0142, z=1.99)",
        "top_negative_shap_contributors": "ROI_056__ROI_138 (SHAP=-0.0218, z=-1.17) | ROI_070__ROI_149 (SHAP=-0.0119, z=0.00) | ROI_034__ROI_041 (SHAP=-0.0119, z=-0.13) | ROI_166__ROI_189 (SHAP=-0.0109, z=-0.51) | ROI_039__ROI_116 (SHAP=-0.0098, z=-0.41)"
      },
      {
        "category": "Peking_2 Test: False Negative",
        "cohort": "Peking_2 Test",
        "subject_id": 1809715,
        "site": "Peking_2",
        "actual_diagnosis": "ADHD",
        "predicted_class": "Control",
        "prediction_probability_adhd": 0.171229305757413,
        "top_positive_shap_contributors": "ROI_033__ROI_093 (SHAP=+0.0204, z=1.21) | ROI_006__ROI_064 (SHAP=+0.0121, z=0.39) | ROI_058__ROI_151 (SHAP=+0.0092, z=0.21) | ROI_006__ROI_127 (SHAP=+0.0087, z=0.56) | ROI_081__ROI_156 (SHAP=+0.0077, z=-0.06)",
        "top_negative_shap_contributors": "ROI_023__ROI_189 (SHAP=-0.0244, z=-1.73) | ROI_056__ROI_138 (SHAP=-0.0230, z=-0.80) | ROI_003__ROI_034 (SHAP=-0.0179, z=-1.10) | ROI_023__ROI_103 (SHAP=-0.0174, z=-0.62) | ROI_034__ROI_041 (SHAP=-0.0173, z=-0.37)"
      },
      {
        "category": "Peking_2 Test: True Negative",
        "cohort": "Peking_2 Test",
        "subject_id": 1068505,
        "site": "Peking_2",
        "actual_diagnosis": "Control",
        "predicted_class": "Control",
        "prediction_probability_adhd": 0.3495232542814459,
        "top_positive_shap_contributors": "ROI_003__ROI_017 (SHAP=+0.0136, z=0.39) | ROI_006__ROI_127 (SHAP=+0.0112, z=0.93) | ROI_034__ROI_041 (SHAP=+0.0102, z=0.21) | ROI_058__ROI_151 (SHAP=+0.0100, z=0.36) | ROI_003__ROI_073 (SHAP=+0.0098, z=0.97)",
        "top_negative_shap_contributors": "ROI_023__ROI_189 (SHAP=-0.0291, z=-0.50) | ROI_100__ROI_153 (SHAP=-0.0201, z=-0.54) | ROI_056__ROI_138 (SHAP=-0.0194, z=-0.61) | ROI_143__ROI_153 (SHAP=-0.0180, z=-0.15) | ROI_016__ROI_174 (SHAP=-0.0143, z=-1.02)"
      },
      {
        "category": "Peking_2 Test: False Positive",
        "cohort": "Peking_2 Test",
        "subject_id": 2296326,
        "site": "Peking_2",
        "actual_diagnosis": "Control",
        "predicted_class": "ADHD",
        "prediction_probability_adhd": 0.6099021752025165,
        "top_positive_shap_contributors": "ROI_115__ROI_174 (SHAP=+0.0257, z=2.40) | ROI_006__ROI_127 (SHAP=+0.0176, z=1.56) | ROI_143__ROI_153 (SHAP=+0.0175, z=0.62) | ROI_023__ROI_103 (SHAP=+0.0146, z=0.85) | ROI_003__ROI_017 (SHAP=+0.0126, z=1.34)",
        "top_negative_shap_contributors": "ROI_034__ROI_041 (SHAP=-0.0487, z=-1.41) | ROI_034__ROI_162 (SHAP=-0.0219, z=-1.89) | ROI_100__ROI_153 (SHAP=-0.0174, z=0.10) | ROI_026__ROI_153 (SHAP=-0.0155, z=-0.37) | ROI_058__ROI_151 (SHAP=-0.0150, z=-1.38)"
      }
    ],
    "network_summary": [
      {
        "network_group": "CC200 Data-Driven Parcellation Subnetwork",
        "total_selected_edges": 100,
        "unique_rois_represented": 66,
        "within_cluster_edges": 0,
        "inter_cluster_edges": 100,
        "top_hub_roi_1": "ROI_003 (8 top-50 edges)",
        "top_hub_roi_2": "ROI_034 (6 top-50 edges)",
        "top_hub_roi_3": "ROI_174 (6 top-50 edges)",
        "mapping_status": "Verified data-driven CC200 node indices (Anatomical labels unavailable locally)"
      }
    ]
  }
};
