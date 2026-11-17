import React from "react";
import { AlertTriangle, ShieldAlert, Cpu, Scale, HelpCircle, FileWarning } from "lucide-react";
import siteConfoundData from "../data/siteConfoundData.json";

export function LimitationsView({ data }) {
  const c1 = siteConfoundData?.config1_153;
  const c2 = siteConfoundData?.config2_118;
  const finalTest = data?.final_test_results;
  const testTotal = finalTest?.test_subject_count ?? null;
  const totalStr = testTotal !== null ? String(testTotal) : "N/A";
  const cm = finalTest?.metrics?.confusion_matrix;
  const correctCount = (cm && typeof cm.tp === "number" && typeof cm.tn === "number") ? (cm.tp + cm.tn) : null;
  const countStr = (correctCount !== null && testTotal !== null) ? `${correctCount}/${testTotal}` : "N/A";
  const testAcc = finalTest?.metrics?.accuracy != null ? `${(finalTest.metrics.accuracy * 100).toFixed(1)}%` : "N/A";
  const testAccCi = finalTest?.metrics?.accuracy_ci_95 ?? finalTest?.metrics?.accuracy_wilson_ci_95;
  const testAccCiStr = testAccCi ? `[${(testAccCi[0] * 100).toFixed(1)}%, ${(testAccCi[1] * 100).toFixed(1)}%]` : "N/A";
  const testAucCi = finalTest?.metrics?.roc_auc_ci_95 ?? finalTest?.metrics?.auc_bootstrap_ci_95;
  const testAucCiStr = testAucCi ? `[${testAucCi[0].toFixed(3)}, ${testAucCi[1].toFixed(3)}]` : "N/A";
  const singleSubjectEffect = (typeof testTotal === "number" && testTotal > 0) ? `${(100 / testTotal).toFixed(1)} percentage points` : "N/A";

  const c1N = c1?.n_subjects != null ? String(c1.n_subjects) : "N/A";
  const c1SiteAuc = c1?.site_only_insample_auc != null ? c1.site_only_insample_auc.toFixed(4) : "N/A";
  const c1ConnAuc = c1?.loio_connectivity_only?.roc_auc != null ? c1.loio_connectivity_only.roc_auc.toFixed(4) : "N/A";
  const c1ConnCi = c1?.loio_connectivity_only?.roc_auc_ci ? `[${c1.loio_connectivity_only.roc_auc_ci.join(", ")}]` : "N/A";
  const c1PVal = c1?.loio_connectivity_only?.p_value != null ? `p = ${c1.loio_connectivity_only.p_value.toFixed(4)}` : "N/A";

  const c2N = c2?.n_subjects != null ? String(c2.n_subjects) : "N/A";
  const c2CovAuc = c2?.covariates_only_baseline?.loio_auc != null ? c2.covariates_only_baseline.loio_auc.toFixed(4) : "N/A";
  const c2CovCi = c2?.covariates_only_baseline?.loio_auc_ci ? `[${c2.covariates_only_baseline.loio_auc_ci.join(", ")}]` : "N/A";
  const c2ConnAuc = c2?.loio_connectivity_only?.roc_auc != null ? c2.loio_connectivity_only.roc_auc.toFixed(4) : "N/A";
  const c2ConnCi = c2?.loio_connectivity_only?.roc_auc_ci ? `[${c2.loio_connectivity_only.roc_auc_ci.join(", ")}]` : "N/A";
  const c2PVal = c2?.loio_connectivity_only?.p_value != null ? `p = ${c2.loio_connectivity_only.p_value.toFixed(3)}` : "N/A";

  const siteDesc = `In multi-site resting-state fMRI, random k-fold CV can inflate performance estimates by allowing models to exploit site-correlated demographic and scanner distributions. While Leave-One-Institution-Out (LOIO) cross-validation avoids train-test site overlap, LOIO still lets site affect scores through differences in demographic profiles, acquisition protocols, and scanner baselines. What the data show: in the full ${c1N}-subject cohort (Config 1), OHSU and Pittsburgh contain exclusively typically developing controls with zero ADHD cases, enabling site identity alone to reach an in-sample ROC-AUC of ${c1SiteAuc}. Under LOIO CV, functional connectivity drops to near-chance (LOIO AUC ${c1ConnAuc} ${c1ConnCi}, permutation ${c1PVal}). The ${c2N}-subject subset (Config 2) was chosen after the full cohort showed no signal (exploratory). Within this clean subset, demographic covariates (age and sex alone) reach LOIO AUC ${c2CovAuc} ${c2CovCi}, while connectivity achieves ${c2ConnAuc} ${c2ConnCi}, ${c2PVal}. The difference between connectivity and age/sex alone was not statistically tested.`;

  const p2Desc = `The held-out Peking_2 test set consists of only ${totalStr} subjects. At N=${totalStr}, each subject shifts accuracy by ${singleSubjectEffect}. The Wilson 95% confidence interval for ${countStr} accuracy spans ${testAccCiStr}, and the stratified bootstrap 95% CI for test ROC-AUC spans ${testAucCiStr}. Because 50% chance lies comfortably within both intervals, the test accuracy of ${testAcc} is not statistically distinguishable from chance (binomial p ≈ 0.25).`;

  const limitations = [
    {
      title: "Site Confounding & Post-Hoc Clean Cohort Selection",
      desc: siteDesc,
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
      title: "Small Test Sample Size & Statistical Power",
      desc: p2Desc,
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
          Rigorous methodological caveats, site-confounding realities, sample size constraints, and clinical disclaimers.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "16px" }}>
        {limitations.map((lim, idx) => {
          const Icon = lim.icon;
          return (
            <div key={idx} className="card" style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                <div style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--bg-subtle)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0
                }}>
                  <Icon size={18} color={lim.color} />
                </div>
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)" }}>
                  {lim.title}
                </h3>
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.6, flexGrow: 1 }}>
                {lim.desc}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
