import React from "react";
import { Database, Network, Users, Cpu, Lock, CheckCircle2, ShieldAlert, Info, Sparkles, Scale, BarChart3, ClipboardCheck } from "lucide-react";
import siteConfoundData from "../data/siteConfoundData.json";
import finalCandidateData from "../data/finalCandidateData.json";
import modelComparisonData from "../data/modelComparisonData.json";
import study2Data from "../data/study2Data.json";

export function OverviewView({ data }) {
  const cohort = data?.cohort_breakdown;
  const rfStats = data?.rf_robustness?.summary?.all_8_repetitions;
  const finalTest = data?.final_test_results;

  const cm = finalTest?.metrics?.confusion_matrix;
  const correctCount = (cm && typeof cm.tp === "number" && typeof cm.tn === "number") ? (cm.tp + cm.tn) : null;
  const totalCount = finalTest?.test_subject_count ?? cohort?.test_subjects ?? null;
  const countStr = (correctCount !== null && totalCount !== null) ? `${correctCount}/${totalCount}` : "N/A";

  const testAcc = finalTest?.metrics?.accuracy ?? null;
  const testAccCi = finalTest?.metrics?.accuracy_ci_95 ?? finalTest?.metrics?.accuracy_wilson_ci_95 ?? null;
  const testAuc = finalTest?.metrics?.roc_auc ?? null;
  const testAucCi = finalTest?.metrics?.roc_auc_ci_95 ?? finalTest?.metrics?.auc_bootstrap_ci_95 ?? null;

  const testAccDisplay = testAcc !== null ? `${(testAcc * 100).toFixed(1)}%` : "N/A";
  const testAccCardDisplay = testAcc !== null ? `${(testAcc * 100).toFixed(1)}% Acc (${countStr})` : "N/A";
  const testAccCiDisplay = (testAccCi && Array.isArray(testAccCi) && testAccCi.length === 2 && testAccCi[0] !== null && testAccCi[1] !== null)
    ? `[${(testAccCi[0] * 100).toFixed(1)}%, ${(testAccCi[1] * 100).toFixed(1)}%]`
    : "N/A";

  const testAucDisplay = testAuc !== null ? testAuc.toFixed(4) : "N/A";
  const testAucCiDisplay = (testAucCi && Array.isArray(testAucCi) && testAucCi.length === 2 && testAucCi[0] !== null && testAucCi[1] !== null)
    ? `[${testAucCi[0].toFixed(3)}, ${testAucCi[1].toFixed(3)}]`
    : "N/A";

  const testSubDisplay = (testAuc !== null && testAucCiDisplay !== "N/A")
    ? `AUC: ${testAucDisplay} ${testAucCiDisplay}`
    : (testAuc !== null ? `AUC: ${testAucDisplay}` : "N/A");

  const kMatch = data?.final_model_metadata?.feature_selection?.match(/k=(\d+)/);
  const kVal = kMatch ? kMatch[1] : null;

  const statCards = [
    { 
      title: "Dataset & Atlas", 
      value: cohort?.total_subjects != null ? `${cohort.total_subjects} Subjects` : "N/A", 
      sub: "ADHD-200 · CC200 Atlas (200 ROIs)", 
      icon: Database, 
      color: "var(--brand-blue)" 
    },
    { 
      title: "Selected Subnetwork", 
      value: kVal != null ? `${kVal} Connections` : "N/A", 
      sub: "From 19,900 Fisher-z edges (ANOVA)", 
      icon: Cpu, 
      color: "var(--accent-success)" 
    },
    { 
      title: "Random-Fold CV (8 Seeds)", 
      value: rfStats?.roc_auc?.mean != null ? `${rfStats.roc_auc.mean.toFixed(4)} AUC` : "N/A", 
      sub: rfStats?.roc_auc?.std != null ? `± ${rfStats.roc_auc.std.toFixed(4)} seed variability (N=${cohort?.train_subjects != null ? cohort.train_subjects : "133"})` : "N/A", 
      icon: CheckCircle2, 
      color: "var(--brand-blue)" 
    },
    { 
      title: "Peking_2 Batch Holdout", 
      value: testAccCardDisplay, 
      sub: testSubDisplay, 
      icon: Lock, 
      color: "var(--accent-warning)" 
    }
  ];

  return (
    <div className="view-container">
      {/* Lead Scientific Headline */}
      <div style={{ marginBottom: "24px" }}>
        <h2 style={{ fontSize: "1.35rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Resting-State fMRI Connectivity Benchmark for ADHD Classification
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", lineHeight: 1.6, marginTop: "6px", maxWidth: "980px" }}>
          This platform documents Study 1 (exploratory C-PAC CC200 benchmark) and Study 2 (preregistered multi-site replication attempt on Athena CC200 derivatives).
          <strong> What the model does:</strong> Identifies statistical functional edge correlations and discriminative connectivity weights across subjects.
          <strong> What the model cannot do:</strong> Research use only — not a medical device or diagnostic tool; model scores are not calibrated probabilities. Study 1&apos;s exploratory LOIO result was not replicated in Study 2, where motion-only and age/sex baselines scored as high or higher than the locked fMRI pipeline and the sealed NYU evaluation fell in the registered Inconclusive Zone.
        </p>
      </div>

      {/* Study 2: Preregistered Multi-Site Replication Attempt (Read from study2Data.json — no typed numbers) */}
      <div className="card" style={{ marginBottom: "24px", borderColor: "rgba(217, 119, 6, 0.35)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <ClipboardCheck size={18} color="var(--accent-warning)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Study 2: Preregistered Multi-Site Replication Attempt (Athena CC200)
            </h3>
          </div>
          <span style={{ fontSize: "0.74rem", background: "var(--accent-warning-subtle)", padding: "4px 8px", borderRadius: "4px", color: "var(--accent-warning)", fontWeight: "600" }}>
            OSF Preregistered ({study2Data.preregistration.commit_short}) · Section 8.1 Not Met · Sealed Site Used Up
          </span>
        </div>

        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: "14px" }}>
          Preregistered test of the locked Study 1 pipeline on ADHD-200 Athena CC200 derivatives (
          <a href={study2Data.preregistration.osf_primary_url} target="_blank" rel="noreferrer" style={{ color: "var(--brand-blue)" }}>
            {study2Data.preregistration.osf_primary_url}
          </a>
          {" "}and{" "}
          <a href={study2Data.preregistration.osf_open_ended_url} target="_blank" rel="noreferrer" style={{ color: "var(--brand-blue)" }}>
            {study2Data.preregistration.osf_open_ended_url}
          </a>
          ). All figures below are compiled directly from tracked Study 2 summary records.
        </p>

        <div className="table-container" style={{ marginBottom: "12px" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Study 2 Regime</th>
                <th>fMRI ROC-AUC (95% CI)</th>
                <th>Motion-Only Baseline</th>
                <th>Age &amp; Sex Baseline</th>
                <th>Registered Criterion / Outcome</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <strong>Development Primary LOIO (N={study2Data.development_primary.n_scans})</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Section 5.3 primary FD exclusion (KKI, NeuroIMAGE, OHSU, Peking)</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {study2Data.development_primary.fmri_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.development_primary.fmri_ci_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.development_primary.motion_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Δ={study2Data.development_primary.delta_vs_motion_display} {study2Data.development_primary.delta_vs_motion_ci_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.development_primary.age_sex_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Δ={study2Data.development_primary.delta_vs_age_sex_display} {study2Data.development_primary.delta_vs_age_sex_ci_display}</span>
                </td>
                <td className="tabular-nums" style={{ color: "var(--accent-danger)", fontWeight: "700" }}>
                  Section 8.1: {study2Data.development_primary.meets_section_8_1 ? "MET" : "NOT MET"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>p_perm = {study2Data.development_primary.p_perm_display}</span>
                </td>
              </tr>
              <tr>
                <td>
                  <strong>Development Sensitivity LOIO (N={study2Data.development_sensitivity.n_scans})</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Registered Section 5.4 no-exclusion sensitivity analysis</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {study2Data.development_sensitivity.fmri_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.development_sensitivity.fmri_ci_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.development_sensitivity.motion_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Δ={study2Data.development_sensitivity.delta_vs_motion_display} {study2Data.development_sensitivity.delta_vs_motion_ci_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.development_sensitivity.age_sex_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Δ={study2Data.development_sensitivity.delta_vs_age_sex_display} {study2Data.development_sensitivity.delta_vs_age_sex_ci_display}</span>
                </td>
                <td className="tabular-nums" style={{ color: "var(--accent-danger)", fontWeight: "700" }}>
                  Section 8.1: {study2Data.development_sensitivity.meets_section_8_1 ? "MET" : "NOT MET"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>p_perm = {study2Data.development_sensitivity.p_perm_display} (motion &gt; fMRI)</span>
                </td>
              </tr>
              <tr style={{ background: "var(--accent-warning-subtle)" }}>
                <td>
                  <strong>Sealed NYU Primary (N={study2Data.sealed_nyu_primary.n_scans}, Governing)</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>One-shot external test (Section 5.3 FD exclusion; {study2Data.sealed_nyu_primary.n_tdc} TDC / {study2Data.sealed_nyu_primary.n_adhd} ADHD)</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {study2Data.sealed_nyu_primary.fmri_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_primary.fmri_ci_95_exact_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.sealed_nyu_primary.motion_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_primary.motion_baseline_ci_95_exact_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.sealed_nyu_primary.age_sex_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_primary.age_sex_baseline_ci_95_exact_display}</span>
                </td>
                <td style={{ color: "var(--accent-warning)", fontWeight: "700" }}>
                  {study2Data.sealed_nyu_primary.outcome_zone_label}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Section 8.2 (governs)</span>
                </td>
              </tr>
              <tr>
                <td>
                  <strong>Sealed NYU Sensitivity (N={study2Data.sealed_nyu_sensitivity.n_scans})</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Registered Section 5.4 no-exclusion ({study2Data.sealed_nyu_sensitivity.n_tdc} TDC / {study2Data.sealed_nyu_sensitivity.n_adhd} ADHD)</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {study2Data.sealed_nyu_sensitivity.fmri_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_sensitivity.fmri_ci_95_exact_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.sealed_nyu_sensitivity.motion_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_sensitivity.motion_baseline_ci_95_exact_display}</span>
                </td>
                <td className="tabular-nums">
                  {study2Data.sealed_nyu_sensitivity.age_sex_baseline_auc_display}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{study2Data.sealed_nyu_sensitivity.age_sex_baseline_ci_95_exact_display}</span>
                </td>
                <td style={{ color: "var(--accent-warning)", fontWeight: "700" }}>
                  {study2Data.sealed_nyu_sensitivity.outcome_zone_label}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>CI spans chance</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", fontSize: "0.78rem", color: "var(--text-muted)" }}>
          <Info size={16} color="var(--accent-warning)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            <strong>Registered Interpretation:</strong> {study2Data.section_8_2_inconclusive_zone.conclusion_plain}{" "}
            Motion-only and age/sex baselines scored as high or higher than the fMRI pipeline in every regime.{" "}
            <strong>Sealed-site status:</strong> {study2Data.preregistration.sealed_site_status}{" "}
            Prior Study 1 overlap: {study2Data.prior_exposure.dev_overlap_subjects} development subjects and {study2Data.prior_exposure.nyu_overlap_subjects} NYU subjects.
          </span>
        </div>
      </div>

      {/* Primary KPI Grid (Balanced 4-Column) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px", marginBottom: "24px" }}>
        {statCards.map((c, i) => {
          const Icon = c.icon;
          return (
            <div key={i} className="card" style={{ padding: "18px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span className="card-title">{c.title}</span>
                <Icon size={16} color={c.color} />
              </div>
              <div className="card-value tabular-nums" style={{ fontSize: "1.25rem" }}>{c.value}</div>
              <div className="card-subtext">{c.sub}</div>
            </div>
          );
        })}
      </div>

      {/* Dual Evaluation Architecture & Peking_2 Intervals */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginBottom: "24px" }}>
        {/* Development CV Robustness Card */}
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px" }}>
            <CheckCircle2 size={18} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Development Partition: Random-Fold CV (N={cohort?.train_subjects != null ? cohort.train_subjects : "133"})
            </h3>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "14px", borderRadius: "var(--radius-md)", marginBottom: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Mean ROC-AUC across 8 random seeds</span>
              <span className="tabular-nums" style={{ fontSize: "1.2rem", fontWeight: "800", color: "var(--brand-blue)" }}>
                {rfStats?.roc_auc?.mean != null ? rfStats.roc_auc.mean.toFixed(4) : "N/A"}
              </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
              Seed variability: ± {rfStats?.roc_auc?.std != null ? rfStats.roc_auc.std.toFixed(4) : "N/A"} (range: {rfStats?.roc_auc?.min != null ? rfStats.roc_auc.min.toFixed(4) : "N/A"} – {rfStats?.roc_auc?.max != null ? rfStats.roc_auc.max.toFixed(4) : "N/A"})
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "6px" }}>
              Balanced Acc: <strong className="tabular-nums">{rfStats?.balanced_accuracy?.mean != null ? rfStats.balanced_accuracy.mean.toFixed(4) : "N/A"}</strong> · Sensitivity: <strong className="tabular-nums">{rfStats?.recall_sensitivity?.mean != null ? rfStats.recall_sensitivity.mean.toFixed(4) : "N/A"}</strong>
            </div>
          </div>

          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
            *Note: Random-fold CV mixes subjects from the same acquisition site across training and validation folds; random k-fold CV can inflate performance estimates by allowing models to exploit site-correlated demographic and scanner distributions.
          </p>
        </div>

        {/* Peking_2 Batch Holdout Card */}
        <div className="card" style={{ borderColor: "rgba(217, 119, 6, 0.3)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px" }}>
            <Lock size={18} color="var(--accent-warning)" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-warning)" }}>
              Held-out Peking_2 Evaluation (N={totalCount ?? "N/A"})
            </h3>
          </div>

          <div style={{ background: "var(--accent-warning-subtle)", padding: "14px", borderRadius: "var(--radius-md)", marginBottom: "12px", border: "1px solid rgba(217, 119, 6, 0.2)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--accent-warning)" }}>Test Accuracy ({countStr} correct)</span>
              <span className="tabular-nums" style={{ fontSize: "1.2rem", fontWeight: "800", color: "var(--accent-warning)" }}>
                {testAccDisplay}
              </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--accent-warning)", marginTop: "4px" }}>
              Wilson 95% Confidence Interval: {testAccCiDisplay}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--accent-warning)", marginTop: "6px" }}>
              Test ROC-AUC: <strong className="tabular-nums">{testAucDisplay}</strong> {testAucCiDisplay !== "N/A" ? `[95% Bootstrap CI: ${testAucCiDisplay}]` : ""}
            </div>
          </div>

          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
            Scored once for this model (baseline previously scored). Peking_2 is a same-institution, same-scanner batch holdout (Peking_1/3 are in training). Wide confidence intervals reflect the small sample size ($N={totalCount ?? "N/A"}$); {testAccDisplay !== "N/A" ? testAccDisplay : "60%"} accuracy is not distinguishable from chance ($p \approx 0.25$).
          </p>
        </div>
      </div>

      {/* Site Confound & Generalization Panel (Follow-up Study Results) */}
      <div className="card" style={{ marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <ShieldAlert size={18} color="var(--accent-warning)" />
          <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
            Site Confound &amp; Generalization Audit (Leave-One-Institution-Out CV)
          </h3>
        </div>

        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: "16px" }}>
          Rigorous Leave-One-Institution-Out (LOIO) cross-validation evaluating scanner generalization, baseline demographic confound predictability, and non-parametric within-institution permutation significance.
        </p>

        <div className="table-container" style={{ marginBottom: "14px" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Cohort / Protocol</th>
                <th>Site Confound In-Sample AUC</th>
                <th>Demographics Alone (Age+Sex LOIO AUC)</th>
                <th>Connectivity LOIO ROC-AUC (95% CI)</th>
                <th>LOIO Accuracy (95% CI)</th>
                <th>Permutation Significance (LOIO)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <strong>{siteConfoundData.config1_153.title}</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.note}</span>
                </td>
                <td className="tabular-nums" style={{ color: "var(--accent-danger)", fontWeight: "700" }}>
                  {siteConfoundData.config1_153.site_only_insample_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.site_only_stratified_auc.toFixed(4)} (CV)</span>
                </td>
                <td className="tabular-nums">
                  {siteConfoundData.config1_153.covariates_only_baseline?.loio_auc != null ? siteConfoundData.config1_153.covariates_only_baseline.loio_auc.toFixed(4) : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.covariates_only_baseline?.loio_auc_ci ? `[${siteConfoundData.config1_153.covariates_only_baseline.loio_auc_ci.join(", ")}]` : "N/A"}</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600" }}>
                  {siteConfoundData.config1_153.loio_connectivity_only?.roc_auc != null ? siteConfoundData.config1_153.loio_connectivity_only.roc_auc.toFixed(4) : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.loio_connectivity_only?.roc_auc_ci ? `[${siteConfoundData.config1_153.loio_connectivity_only.roc_auc_ci.join(", ")}]` : "N/A"}</span>
                </td>
                <td className="tabular-nums">
                  {siteConfoundData.config1_153.loio_connectivity_only?.accuracy != null ? `${(siteConfoundData.config1_153.loio_connectivity_only.accuracy * 100).toFixed(1)}%` : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.loio_connectivity_only?.accuracy_ci ? `[${siteConfoundData.config1_153.loio_connectivity_only.accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]` : "N/A"}</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600", color: "var(--text-muted)" }}>
                  {siteConfoundData.config1_153.loio_connectivity_only?.p_value != null ? `p = ${siteConfoundData.config1_153.loio_connectivity_only.p_value.toFixed(4)}` : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--accent-warning)" }}>Not significant</span>
                </td>
              </tr>
              <tr style={{ background: "var(--accent-green-bg)" }}>
                <td>
                  <strong>{siteConfoundData.config2_118.title}</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.note}</span>
                </td>
                <td className="tabular-nums" style={{ color: "var(--accent-success)", fontWeight: "600" }}>
                  {siteConfoundData.config2_118.site_only_insample_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.site_only_stratified_auc.toFixed(4)} (CV)</span>
                </td>
                <td className="tabular-nums">
                  <div style={{ fontSize: "0.76rem" }}>
                    <span style={{ fontWeight: "600", color: "var(--brand-blue)" }}>followup_study:</span> nested model selection on age+sex, LOIO AUC <strong>{siteConfoundData.config2_118.covariates_only_baseline?.loio_auc != null ? siteConfoundData.config2_118.covariates_only_baseline.loio_auc.toFixed(4) : "0.5976"}</strong><br />
                    <span style={{ fontSize: "0.70rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.covariates_only_baseline?.loio_auc_ci ? `[${siteConfoundData.config2_118.covariates_only_baseline.loio_auc_ci.join(", ")}]` : "N/A"}</span>
                  </div>
                  <div style={{ fontSize: "0.76rem", marginTop: "4px", paddingTop: "4px", borderTop: "1px dashed var(--border-color)" }}>
                    <span style={{ fontWeight: "600", color: "var(--brand-blue)" }}>final_candidate:</span> fixed logistic regression, LOIO AUC <strong>{finalCandidateData?.covariates_only_model?.loio_auc != null ? finalCandidateData.covariates_only_model.loio_auc.toFixed(4) : "0.5857"}</strong><br />
                    <span style={{ fontSize: "0.70rem", color: "var(--text-muted)" }}>{finalCandidateData?.covariates_only_model?.loio_auc_ci ? `[${finalCandidateData.covariates_only_model.loio_auc_ci[0].toFixed(3)}, ${finalCandidateData.covariates_only_model.loio_auc_ci[1].toFixed(3)}]` : "N/A"}</span>
                  </div>
                  <div style={{ fontSize: "0.68rem", color: "var(--accent-warning)", marginTop: "4px", fontStyle: "italic" }}>
                    Age/sex alone carries information about the label in this cohort; this is not an ADHD-specific neural signal.
                  </div>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--brand-blue)" }}>
                  {siteConfoundData.config2_118.loio_connectivity_only?.roc_auc != null ? siteConfoundData.config2_118.loio_connectivity_only.roc_auc.toFixed(4) : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.loio_connectivity_only?.roc_auc_ci ? `[${siteConfoundData.config2_118.loio_connectivity_only.roc_auc_ci.join(", ")}]` : "N/A"}</span>
                </td>
                <td className="tabular-nums">
                  {siteConfoundData.config2_118.loio_connectivity_only?.accuracy != null ? `${(siteConfoundData.config2_118.loio_connectivity_only.accuracy * 100).toFixed(1)}%` : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.loio_connectivity_only?.accuracy_ci ? `[${siteConfoundData.config2_118.loio_connectivity_only.accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]` : "N/A"}</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--accent-warning)" }}>
                  {siteConfoundData.config2_118.loio_connectivity_only?.p_value != null ? `p = ${siteConfoundData.config2_118.loio_connectivity_only.p_value.toFixed(4)}` : "N/A"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--accent-warning)" }}>Exploratory (not replicated in Study 2)</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", fontSize: "0.78rem", color: "var(--text-muted)" }}>
          <Info size={16} color="var(--brand-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            <strong>What the data show (Study 1 — Exploratory, Not Replicated):</strong> While Leave-One-Institution-Out (LOIO) cross-validation prevents train-test site overlap, LOIO still lets site affect scores because demographic distributions, base rates, and scanner characteristics vary across institutions. In Config 1 ({siteConfoundData.config1_153.n_subjects} subjects), site identity alone predicts diagnosis with an in-sample ROC-AUC of {siteConfoundData.config1_153.site_only_insample_auc.toFixed(4)} because OHSU and Pittsburgh contain zero ADHD cases in this release. Under LOIO evaluation, functional connectivity drops to near-chance (AUC {siteConfoundData.config1_153.loio_connectivity_only.roc_auc.toFixed(4)} [{siteConfoundData.config1_153.loio_connectivity_only.roc_auc_ci.join(", ")}]), which non-parametric permutation testing confirms is not statistically significant (p = {siteConfoundData.config1_153.loio_connectivity_only.p_value.toFixed(4)}). The {siteConfoundData.config2_118.n_subjects}-subject subset (Config 2, excluding zero-ADHD sites) was chosen post-hoc after the full cohort showed no signal. In this exploratory subset, demographic covariates reach LOIO AUC {siteConfoundData.config2_118.covariates_only_baseline.loio_auc.toFixed(4)} (followup_study) and {finalCandidateData?.covariates_only_model?.loio_auc != null ? finalCandidateData.covariates_only_model.loio_auc.toFixed(4) : "N/A"} (final_candidate), while connectivity reaches LOIO AUC {siteConfoundData.config2_118.loio_connectivity_only.roc_auc.toFixed(4)} [{siteConfoundData.config2_118.loio_connectivity_only.roc_auc_ci.join(", ")}] (p = {siteConfoundData.config2_118.loio_connectivity_only.p_value.toFixed(3)}). That exploratory Study 1 LOIO AUC was not replicated in Study 2 (primary LOIO AUC {study2Data.development_primary.fmri_auc_display}, p_perm = {study2Data.development_primary.p_perm_display}; sealed NYU {study2Data.sealed_nyu_primary.fmri_auc_display} {study2Data.sealed_nyu_primary.fmri_ci_95_exact_display}).
          </span>
        </div>
      </div>

      {/* Apples-to-Apples Model Comparison Table Card */}
      <div className="card" style={{ marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px", flexWrap: "wrap", gap: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <BarChart3 size={18} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Study 1 Exploratory Model Comparison (118-Subject Clean Cohort — Not Replicated in Study 2)
            </h3>
          </div>
          <span style={{ fontSize: "0.75rem", background: "var(--bg-subtle)", padding: "4px 8px", borderRadius: "4px", color: "var(--text-muted)", fontWeight: "500" }}>
            Identical 4-Fold LOIO CV · 2,000 Bootstrap CIs · 200 Permutations
          </span>
        </div>

        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: "14px" }}>
          Side-by-side empirical benchmarking evaluating the locked production Random Forest against the demographic covariates baselines, the naive majority class classifier, and the 126-configuration nested candidate ensemble.
        </p>

        <div className="table-container" style={{ marginBottom: "12px" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model / Pipeline Specification</th>
                <th>LOIO ROC-AUC (95% CI)</th>
                <th>LOIO Accuracy (95% CI)</th>
                <th>Balanced Accuracy (95% CI)</th>
                <th>Permutation Significance</th>
              </tr>
            </thead>
            <tbody>
              {/* Majority Baseline */}
              <tr>
                <td>
                  <strong>Majority Class Baseline</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Always predict typically developing Control (68/118)</span>
                </td>
                <td className="tabular-nums">0.5000</td>
                <td className="tabular-nums">{(modelComparisonData.models[0].loio_accuracy * 100).toFixed(2)}%</td>
                <td className="tabular-nums">50.00%</td>
                <td className="tabular-nums" style={{ color: "var(--text-muted)" }}>p = 1.0000</td>
              </tr>

              {/* Demographics: followup_study */}
              <tr style={{ background: "var(--bg-subtle)" }}>
                <td>
                  <strong>Demographics Alone (followup_study)</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--brand-blue)", fontFamily: "var(--font-mono)" }}>
                    followup_study: nested model selection on age+sex, LOIO AUC 0.5976
                  </span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600" }}>
                  {modelComparisonData.models[1].loio_roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[1].loio_roc_auc_ci.map(v => v.toFixed(3)).join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[1].loio_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[1].loio_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[1].loio_balanced_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[1].loio_balanced_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ color: "var(--text-muted)" }}>—</td>
              </tr>

              {/* Demographics: final_candidate */}
              <tr style={{ background: "var(--bg-subtle)" }}>
                <td>
                  <strong>Demographics Alone (final_candidate)</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--brand-blue)", fontFamily: "var(--font-mono)" }}>
                    final_candidate: fixed logistic regression, LOIO AUC 0.5857
                  </span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600" }}>
                  {modelComparisonData.models[2].loio_roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[2].loio_roc_auc_ci.map(v => v.toFixed(3)).join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[2].loio_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[2].loio_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[2].loio_balanced_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[2].loio_balanced_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600", color: "var(--accent-success)" }}>
                  p = {modelComparisonData.models[2].permutation_p.toFixed(4)}
                </td>
              </tr>

              {/* Note under the age/sex rows */}
              <tr style={{ background: "var(--bg-subtle)" }}>
                <td colSpan={5} style={{ padding: "6px 12px", fontSize: "0.72rem", color: "var(--accent-warning)", fontStyle: "italic", borderTop: "1px dashed var(--border-color)" }}>
                  Age/sex alone carries information about the label in this cohort; this is not an ADHD-specific neural signal.
                </td>
              </tr>

              {/* Locked Production Random Forest */}
              <tr style={{ background: "var(--accent-green-bg)" }}>
                <td>
                  <strong>Locked Production Pipeline (Random Forest)</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>SelectKBest(k=100) + Balanced RF(50 trees, depth 3)</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--brand-blue)" }}>
                  {modelComparisonData.models[3].loio_roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[3].loio_roc_auc_ci.map(v => v.toFixed(3)).join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {(modelComparisonData.models[3].loio_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[3].loio_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600" }}>
                  {(modelComparisonData.models[3].loio_balanced_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[3].loio_balanced_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--accent-success)" }}>
                  p = {modelComparisonData.models[3].permutation_p.toFixed(4)}
                </td>
              </tr>

              {/* Final Candidate Ensemble */}
              <tr>
                <td>
                  <strong>Final Candidate Pipeline (Nested Ensemble)</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>126-Config Nested Grid (Pearson/Partial/Tangent + K + Classifiers)</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--brand-blue)" }}>
                  {modelComparisonData.models[4].loio_roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[4].loio_roc_auc_ci.map(v => v.toFixed(3)).join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[4].loio_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[4].loio_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(modelComparisonData.models[4].loio_balanced_accuracy * 100).toFixed(2)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{modelComparisonData.models[4].loio_balanced_accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--accent-success)" }}>
                  p = {modelComparisonData.models[4].permutation_p.toFixed(4)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Methodological Caveat & Non-Significance Callout */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", fontSize: "0.78rem", color: "var(--text-muted)" }}>
          <Info size={16} color="var(--brand-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            <strong>Key Finding:</strong> Connectivity did not significantly outperform age/sex alone (Locked RF vs Demographics: ΔAUC = +0.0397 [-0.1061, +0.1695], p = 0.2670; Candidate Ensemble vs Demographics: ΔAUC = +0.0400 [-0.1038, +0.1714], p = 0.2685). Both pipelines were tuned with knowledge of these subjects (the 118-subject clean subset and feature spaces were selected after observing full cohort results). Furthermore, the complex 126-configuration nested ensemble search did not outperform the locked Random Forest (ΔAUC = +0.0003 [-0.0633, +0.0639], p = 0.4960).
          </span>
        </div>
      </div>

      {/* Final Candidate Benchmark Panel (126-Configuration Space Exploration) */}
      <div className="card" style={{ marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px", flexWrap: "wrap", gap: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles size={18} color="var(--brand-blue)" />
            <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
              Final Candidate Benchmark: 126-Configuration Space Exploration (N={finalCandidateData?._metadata?.n_subjects ?? "118"})
            </h3>
          </div>
          <span style={{ fontSize: "0.75rem", background: "var(--bg-subtle)", padding: "4px 8px", borderRadius: "4px", color: "var(--text-muted)", fontWeight: "500" }}>
            Nested CV · 4 Institutional Folds (KKI, NYU, NeuroIMAGE, Peking)
          </span>
        </div>

        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: "16px" }}>
          To rigorously test whether more complex architectures can overcome cross-site variability, an exhaustive benchmark of 126 configurations was executed across Pearson, Partial, and Riemannian Tangent-space embeddings, feature cutoffs (k=50, 100, 200), and 4 classifier families (L2-Logistic Regression, Linear SVM, RBF SVM, and Random Forests) with and without demographic covariates and multi-model ensembling.
        </p>

        {/* Metrics Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "16px" }}>
          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "var(--radius-md)" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>LOIO Generalization AUC</span>
            <div className="tabular-nums" style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--brand-blue)", marginTop: "2px" }}>
              {finalCandidateData?.loio?.roc_auc != null ? finalCandidateData.loio.roc_auc.toFixed(4) : "N/A"}
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              95% CI: {finalCandidateData?.loio?.roc_auc_ci ? `[${finalCandidateData.loio.roc_auc_ci[0].toFixed(3)}, ${finalCandidateData.loio.roc_auc_ci[1].toFixed(3)}]` : "N/A"}
            </div>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "var(--radius-md)" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>LOIO Accuracy vs Majority</span>
            <div className="tabular-nums" style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--text-primary)", marginTop: "2px" }}>
              {finalCandidateData?.loio?.accuracy != null ? `${(finalCandidateData.loio.accuracy * 100).toFixed(2)}%` : "N/A"}
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Majority Baseline: {finalCandidateData?.majority_accuracy != null ? `${(finalCandidateData.majority_accuracy * 100).toFixed(2)}%` : "57.63%"} {finalCandidateData?.loio?.accuracy_ci ? `[${(finalCandidateData.loio.accuracy_ci[0] * 100).toFixed(1)}%, ${(finalCandidateData.loio.accuracy_ci[1] * 100).toFixed(1)}%]` : ""}
            </div>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "var(--radius-md)" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Balanced Acc / Sens / Spec</span>
            <div className="tabular-nums" style={{ fontSize: "1.1rem", fontWeight: "700", color: "var(--text-primary)", marginTop: "4px" }}>
              {finalCandidateData?.loio?.balanced_accuracy != null ? `${(finalCandidateData.loio.balanced_accuracy * 100).toFixed(1)}%` : "N/A"} <span style={{ fontSize: "0.75rem", fontWeight: "400", color: "var(--text-muted)" }}>BalAcc</span>
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Sens: {finalCandidateData?.loio?.sensitivity != null ? `${(finalCandidateData.loio.sensitivity * 100).toFixed(1)}%` : "N/A"} · Spec: {finalCandidateData?.loio?.specificity != null ? `${(finalCandidateData.loio.specificity * 100).toFixed(1)}%` : "N/A"}
            </div>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "var(--radius-md)" }}>
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Permutation Significance</span>
            <div className="tabular-nums" style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--accent-success)", marginTop: "2px" }}>
              {finalCandidateData?.permutation_test?.p_value_display ?? "p = 0.0050"}
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Null mean: {finalCandidateData?.permutation_test?.null_mean != null ? `${finalCandidateData.permutation_test.null_mean.toFixed(4)} ± ${finalCandidateData.permutation_test.null_std.toFixed(4)}` : "N/A"} (200 permutations; min possible p=0.0050)
            </div>
          </div>
        </div>

        {/* Statistical Contrast vs Demographics */}
        <div style={{ background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", marginBottom: "16px", fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: "700", color: "var(--text-primary)", marginBottom: "4px" }}>
            <Scale size={16} color="var(--brand-blue)" />
            <span>Paired Statistical Test vs. Age &amp; Sex Demographics Baseline</span>
          </div>
          <div>
            Age and sex demographic covariates alone reach a LOIO AUC of <strong>{finalCandidateData?.covariates_only_model?.loio_auc != null ? finalCandidateData.covariates_only_model.loio_auc.toFixed(4) : "0.5857"}</strong> [95% CI: {finalCandidateData?.covariates_only_model?.loio_auc_ci ? `${finalCandidateData.covariates_only_model.loio_auc_ci[0].toFixed(3)}, ${finalCandidateData.covariates_only_model.loio_auc_ci[1].toFixed(3)}` : "0.489, 0.685"}] (permutation p = {finalCandidateData?.covariates_only_model?.p_value != null ? finalCandidateData.covariates_only_model.p_value.toFixed(4) : "0.0498"}).
            The paired subject-level bootstrap test between the final candidate and demographics yields:
            <br />
            <strong>ΔAUC = {finalCandidateData?.paired_bootstrap_vs_covars?.delta_mean != null ? `${finalCandidateData.paired_bootstrap_vs_covars.delta_mean >= 0 ? "+" : ""}${finalCandidateData.paired_bootstrap_vs_covars.delta_mean.toFixed(4)}` : "+0.0400"}</strong> [95% Bootstrap CI: {finalCandidateData?.paired_bootstrap_vs_covars?.delta_ci_95 ? `${finalCandidateData.paired_bootstrap_vs_covars.delta_ci_95[0].toFixed(4)}, ${finalCandidateData.paired_bootstrap_vs_covars.delta_ci_95[1].toFixed(4)}` : "-0.1038, +0.1714"}], <strong>p = {finalCandidateData?.paired_bootstrap_vs_covars?.p_value != null ? finalCandidateData.paired_bootstrap_vs_covars.p_value.toFixed(4) : "0.2685"}</strong>.
            Because the 95% confidence interval spans zero, the difference between functional connectivity and demographic covariates alone is not statistically significant.
          </div>
        </div>

        {/* Fold Configurations & Limitations */}
        <div style={{ marginBottom: "14px" }}>
          <div style={{ fontSize: "0.8rem", fontWeight: "700", color: "var(--text-primary)", marginBottom: "6px" }}>
            Fold-Selected Configurations (Selected Configuration Differed by Fold):
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "8px", fontSize: "0.75rem" }}>
            {finalCandidateData?.selected_configurations?.folds?.map((f, idx) => (
              <div key={idx} style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", padding: "8px 10px", borderRadius: "6px" }}>
                <strong>Fold {idx + 1} (Held-out: {f.held_out}):</strong>
                <div style={{ color: "var(--brand-blue)", fontFamily: "var(--font-mono)", marginTop: "2px" }}>{f.winner}</div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>Inner CV AUC: {f.inner_auc.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>

        {/* 3 Explicit Methodological Limitations */}
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.75rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
          <div style={{ fontWeight: "700", color: "var(--text-primary)" }}>Critical Benchmark Limitations (from REPORT.md):</div>
          {finalCandidateData?.limitations?.map((lim, idx) => (
            <div key={idx} style={{ display: "flex", alignItems: "flex-start", gap: "6px" }}>
              <span style={{ color: "var(--accent-warning)", fontWeight: "700" }}>•</span>
              <span><strong>{lim.title}:</strong> {lim.detail}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Cohort Stratification Table */}
      <div className="card">
        <h3 style={{ fontSize: "0.95rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>
          Cohort Partitioning &amp; Site Stratification Summary
        </h3>

        <div className="table-container" style={{ marginBottom: 0 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Partition</th>
                <th>Total Subjects</th>
                <th>Typically Developing (Control)</th>
                <th>ADHD Diagnosis</th>
                <th>Included Sites</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Development Train</strong></td>
                <td className="tabular-nums">{cohort?.train_subjects ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.train_control ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.train_adhd ?? "N/A"}</td>
                <td>6 Sites (KKI, NYU, OHSU, Peking_1, Peking_3, Pittsburgh)</td>
              </tr>
              <tr>
                <td><strong>Development Validation</strong></td>
                <td className="tabular-nums">{cohort?.validation_subjects ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.validation_control ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.validation_adhd ?? "N/A"}</td>
                <td>NeuroIMAGE (Single External Site)</td>
              </tr>
              <tr style={{ background: "var(--accent-warning-subtle)" }}>
                <td><strong>Held-out Batch Partition</strong></td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>{cohort?.test_subjects ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.test_control ?? "N/A"}</td>
                <td className="tabular-nums">{cohort?.test_adhd ?? "N/A"}</td>
                <td><strong>Peking_2 (Batch holdout; same institution as Peking_1/3)</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
