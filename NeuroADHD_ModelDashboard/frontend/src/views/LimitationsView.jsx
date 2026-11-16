import React from "react";
import { AlertTriangle, ShieldAlert, Cpu, Scale, HelpCircle, FileWarning } from "lucide-react";

export function LimitationsView() {
  const limitations = [
    {
      title: "Site Confounding & Post-Hoc Clean Cohort Selection",
      desc: "In multi-site resting-state fMRI, random k-fold CV can inflate performance estimates by allowing models to exploit site-correlated demographic and scanner distributions. While Leave-One-Institution-Out (LOIO) cross-validation avoids train-test site overlap, LOIO still lets site affect scores through differences in demographic profiles, acquisition protocols, and scanner baselines. What the data show: in the full 153-subject cohort (Config 1), OHSU (N=14) and Pittsburgh (N=21) contain exclusively typically developing controls with zero ADHD cases, enabling site identity alone to reach an in-sample ROC-AUC of 0.7221. Under LOIO CV, functional connectivity drops to near-chance (LOIO AUC 0.5342 [0.439, 0.633], permutation p=0.0859). The 118-subject subset (Config 2) was chosen after the full cohort showed no signal (exploratory). Within this clean subset, demographic covariates (age and sex alone) reach LOIO AUC 0.5976 [0.492, 0.702], while connectivity achieves 0.6221 [0.515, 0.724], p=0.021.",
      icon: ShieldAlert,
      color: "var(--accent-danger)"
    },
    {
      title: "Same-Scanner Batch Holdout (Peking_2)",
      desc: "Peking_2 was collected on the same scanner hardware using identical acquisition parameters as Peking_1 and Peking_3, both of which are present in the training partition. Therefore, evaluating on Peking_2 measures temporal batch stability within the same institution rather than generalization to an unseen clinical scanner environment.",
      icon: FileWarning,
      color: "var(--accent-warning)"
    },
    {
      title: "Small Test Sample Size & Statistical Power (N=20)",
      desc: "The held-out Peking_2 test set consists of only 20 subjects (10 Control, 10 ADHD). At N=20, each subject shifts accuracy by 5.0 percentage points. The Wilson 95% confidence interval for 12/20 accuracy spans [38.7%, 78.1%], and the stratified bootstrap 95% CI for test ROC-AUC spans [0.290, 0.830]. Because 50% chance lies comfortably within both intervals, the test accuracy of 60.0% is not statistically distinguishable from chance (binomial p ≈ 0.25).",
      icon: Scale,
      color: "var(--accent-warning)"
    },
    {
      title: "CC200 Data-Driven Parcellation Representation",
      desc: "The Craddock 200 atlas is derived from spatial spectral clustering of functional time courses. Node numbers (ROI_001 to ROI_200) represent data-driven functional clusters without a 1-to-1 anatomical correspondence in local C-PAC release files, necessitating coordinate-based centroid approximations for neuroanatomical localization.",
      icon: Cpu,
      color: "var(--brand-blue)"
    },
    {
      title: "Non-Causal Statistical Association",
      desc: "All TreeSHAP feature attributions and univariate ANOVA F-scores represent mathematical associations within the decision trees of the Random Forest. They reflect statistical correlation with class labels in this cohort and do not imply causal neurobiological etiology or pathology.",
      icon: HelpCircle,
      color: "var(--text-muted)"
    },
    {
      title: "Academic Research Prototype — Not for Clinical Use",
      desc: "NeuroADHD is strictly an academic methodology benchmark designed to explore machine learning robustness, site confounding, and feature attribution in resting-state fMRI. It is NOT cleared by any medical regulatory authority (such as the US FDA or EMA) and must never be used for clinical psychiatric diagnosis, triage, or treatment planning.",
      icon: AlertTriangle,
      color: "var(--accent-danger)"
    }
  ];

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.35rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Study Limitations &amp; Ethical Boundaries
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", lineHeight: 1.5, marginTop: "4px" }}>
          Explicit documentation of methodological constraints, statistical power boundaries, site confounding, and non-clinical status.
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "14px", marginBottom: "24px" }}>
        {limitations.map((lim, i) => {
          const Icon = lim.icon;
          return (
            <div 
              key={i} 
              className="card" 
              style={{ 
                display: "flex", 
                alignItems: "flex-start", 
                gap: "14px", 
                padding: "18px 20px" 
              }}
            >
              <div 
                style={{ 
                  background: "var(--bg-subtle)", 
                  padding: "8px", 
                  borderRadius: "var(--radius-md)", 
                  color: lim.color,
                  flexShrink: 0,
                  marginTop: "2px"
                }}
              >
                <Icon size={18} />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)", marginBottom: "4px" }}>
                  {i + 1}. {lim.title}
                </h3>
                <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                  {lim.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
