import React from "react";
import { AlertTriangle, ShieldAlert, Cpu, Scale, HelpCircle, FileWarning, ClipboardCheck, Lock } from "lucide-react";
import siteConfoundData from "../data/siteConfoundData.json";
import study2Data from "../data/study2Data.json";

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

  const s2DevPrim = study2Data.development_primary;
  const s2DevSens = study2Data.development_sensitivity;
  const s2SealedPrim = study2Data.sealed_nyu_primary;
  const s2SealedSens = study2Data.sealed_nyu_sensitivity;

  const siteDesc = `In multi-site resting-state fMRI, random k-fold CV can inflate performance estimates by allowing models to exploit site-correlated demographic and scanner distributions. While Leave-One-Institution-Out (LOIO) cross-validation avoids train-test site overlap, LOIO still lets site affect scores through differences in demographic profiles, acquisition protocols, and scanner baselines. What the data show (Study 1): in the full ${c1N}-subject cohort (Config 1), OHSU and Pittsburgh contain exclusively typically developing controls with zero ADHD cases, enabling site identity alone to reach an in-sample ROC-AUC of ${c1SiteAuc}. Under LOIO CV, functional connectivity drops to near-chance (LOIO AUC ${c1ConnAuc} ${c1ConnCi}, permutation ${c1PVal}). The ${c2N}-subject subset (Config 2) was chosen post-hoc after the full cohort showed no signal (exploratory). Within this exploratory subset, demographic covariates (age and sex alone) reach LOIO AUC ${c2CovAuc} ${c2CovCi}, while connectivity reaches ${c2ConnAuc} ${c2ConnCi}, ${c2PVal}; that exploratory Study 1 LOIO result did not hold up in Study 2 (a pre-registered follow-up on a different preprocessing pipeline, atlas [190 ROIs], cohort, and sites).`;

  const study2Desc = `In Study 2 — a pre-registered follow-up on a different preprocessing pipeline (Athena CC200, 190 populated ROIs, different cohort and sites; ${study2Data.preregistration.osf_primary_url}, commit ${study2Data.preregistration.commit_short}) — development Section 8.1 criteria were not met in either the primary motion-filtered analysis (N=${s2DevPrim.n_scans}: fMRI LOIO AUC ${s2DevPrim.fmri_auc_display} ${s2DevPrim.fmri_ci_display}, p_perm = ${s2DevPrim.p_perm_display}; motion baseline ${s2DevPrim.motion_baseline_auc_display}; age/sex baseline ${s2DevPrim.age_sex_baseline_auc_display}) or the registered Section 5.4 no-exclusion sensitivity analysis (N=${s2DevSens.n_scans}: fMRI LOIO AUC ${s2DevSens.fmri_auc_display} ${s2DevSens.fmri_ci_display}, p_perm = ${s2DevSens.p_perm_display}; motion baseline ${s2DevSens.motion_baseline_auc_display}; age/sex baseline ${s2DevSens.age_sex_baseline_auc_display}). On the sealed NYU site, both the governing primary evaluation (N=${s2SealedPrim.n_scans}: ${s2SealedPrim.fmri_auc_display} ${s2SealedPrim.fmri_ci_95_exact_display}) and the sensitivity evaluation (N=${s2SealedSens.n_scans}: ${s2SealedSens.fmri_auc_display} ${s2SealedSens.fmri_ci_95_exact_display}) fell in the registered ${s2SealedPrim.outcome_zone_label}. Motion-only and age/sex baselines scored as high or higher than the fMRI pipeline across all four regimes. Study 1's exploratory 0.62 LOIO AUC did not hold up in this follow-up (not a direct test of the identical C-PAC protocol), and this conclusion applies to this pipeline rather than to connectivity in general.`;

  const sealedUsedUpDesc = `${study2Data.preregistration.sealed_site_status} In addition, ${study2Data.prior_exposure.dev_overlap_subjects} Study 2 development subjects and ${study2Data.prior_exposure.nyu_overlap_subjects} NYU subjects overlapped with Study 1's C-PAC cohort where the locked pipeline's hyperparameters were chosen, so development LOIO estimates may be optimistic for those overlapping subjects and only the sealed NYU evaluation was close to independent. Only the OSF registrations are externally timestamped; the four freeze/results tags were pushed to origin after the sealed run.`;

  const computeExactBinomialP = (k, n) => {
    if (typeof k !== "number" || typeof n !== "number" || n <= 0 || k < 0 || k > n) return null;
    const logFact = [0];
    for (let i = 1; i <= n; i++) logFact.push(logFact[i - 1] + Math.log(i));
    let oneSided = 0;
    for (let j = k; j <= n; j++) {
      oneSided += Math.exp(logFact[n] - logFact[j] - logFact[n - j] - n * Math.log(2));
    }
    return { oneSided, twoSided: Math.min(1, 2 * oneSided) };
  };
  const p2Binom = computeExactBinomialP(correctCount, testTotal);
  const p2BinomStr = p2Binom
    ? `one-sided exact binomial p = ${p2Binom.oneSided.toFixed(4)}; two-sided ${p2Binom.twoSided.toFixed(4)} (about 0.50)`
    : "N/A";

  const p2Desc = `The held-out Peking_2 test set consists of only ${totalStr} subjects and was scored 10 times in total (9 baseline runs, 1 final). At N=${totalStr}, each subject shifts accuracy by ${singleSubjectEffect}. The Wilson 95% confidence interval for ${countStr} accuracy spans ${testAccCiStr}, and the stratified bootstrap 95% CI for test ROC-AUC spans ${testAucCiStr}. Because chance lies within both intervals, the test accuracy of ${testAcc} is not statistically distinguishable from chance (${p2BinomStr}).`;

  const limitations = [
    {
      title: "Study 2 Pre-Registered Follow-Up Outcome (Section 8.1 Not Met; Sealed NYU Inconclusive)",
      desc: study2Desc,
      icon: ClipboardCheck,
      color: "var(--accent-danger)"
    },
    {
      title: "Sealed Set Used Up & Prior Study 1 Subject Overlap",
      desc: sealedUsedUpDesc,
      icon: Lock,
      color: "var(--accent-warning)"
    },
    {
      title: "Site Confounding & Exploratory Study 1 Clean Cohort (Did Not Hold Up in Study 2)",
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
      desc: "NeuroADHD is strictly an academic methodology benchmark designed to explore machine learning robustness, site confounding, and feature attribution in resting-state fMRI. Research use only: not a medical device or diagnostic tool; model scores are not calibrated probabilities. It must never be used for clinical psychiatric diagnosis, triage, or treatment planning.",
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
