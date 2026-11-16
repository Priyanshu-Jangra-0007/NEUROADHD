import React from "react";
import { Database, Network, Users, Cpu, Lock, CheckCircle2, ShieldAlert, Info } from "lucide-react";
import siteConfoundData from "../data/siteConfoundData.json";

export function OverviewView({ data }) {
  const cohort = data.cohort_breakdown;
  const rfStats = data.rf_robustness.summary.all_8_repetitions;
  const finalTest = data.final_test_results;

  const testAcc = finalTest?.metrics?.accuracy ?? 0.6;
  const testAccCi = finalTest?.metrics?.accuracy_ci_95 ?? [0.387, 0.781];
  const testAuc = finalTest?.metrics?.roc_auc ?? 0.56;
  const testAucCi = finalTest?.metrics?.roc_auc_ci_95 ?? [0.290, 0.830];

  const statCards = [
    { 
      title: "Dataset & Atlas", 
      value: "153 Subjects", 
      sub: "ADHD-200 · CC200 Atlas (200 ROIs)", 
      icon: Database, 
      color: "var(--brand-blue)" 
    },
    { 
      title: "Selected Subnetwork", 
      value: "100 Connections", 
      sub: "From 19,900 Fisher-z edges (ANOVA)", 
      icon: Cpu, 
      color: "var(--accent-success)" 
    },
    { 
      title: "Random-Fold CV (8 Seeds)", 
      value: `${rfStats.roc_auc.mean.toFixed(4)} AUC`, 
      sub: `± ${rfStats.roc_auc.std.toFixed(4)} seed variability (N=133)`, 
      icon: CheckCircle2, 
      color: "var(--brand-blue)" 
    },
    { 
      title: "Peking_2 Batch Holdout", 
      value: `${(testAcc * 100).toFixed(1)}% Acc (12/20)`, 
      sub: `AUC: ${testAuc.toFixed(4)} [${testAucCi[0].toFixed(3)}, ${testAucCi[1].toFixed(3)}]`, 
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
          This platform benchmarks machine learning algorithms classifying Attention-Deficit/Hyperactivity Disorder from pairwise resting-state functional connectivity (Craddock 200 parcellation). 
          <strong> What the model does:</strong> Identifies statistical cross-network correlations and discriminative functional edge weights across subjects. 
          <strong> What the model cannot do:</strong> It cannot make medical or psychiatric diagnoses, guarantee individual neurobiological biomarkers, or generalize reliably across external scanners without site-aware harmonization.
        </p>
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
              Development Partition: Random-Fold CV (N=133)
            </h3>
          </div>

          <div style={{ background: "var(--bg-subtle)", padding: "14px", borderRadius: "var(--radius-md)", marginBottom: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Mean ROC-AUC across 8 random seeds</span>
              <span className="tabular-nums" style={{ fontSize: "1.2rem", fontWeight: "800", color: "var(--brand-blue)" }}>
                {rfStats.roc_auc.mean.toFixed(4)}
              </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
              Seed variability: ± {rfStats.roc_auc.std.toFixed(4)} (range: {rfStats.roc_auc.min.toFixed(4)} – {rfStats.roc_auc.max.toFixed(4)})
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "6px" }}>
              Balanced Acc: <strong className="tabular-nums">{rfStats.balanced_accuracy.mean.toFixed(4)}</strong> · Sensitivity: <strong className="tabular-nums">{rfStats.recall_sensitivity.mean.toFixed(4)}</strong>
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
              Held-out Peking_2 Evaluation (N=20)
            </h3>
          </div>

          <div style={{ background: "var(--accent-warning-subtle)", padding: "14px", borderRadius: "var(--radius-md)", marginBottom: "12px", border: "1px solid rgba(217, 119, 6, 0.2)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--accent-warning)" }}>Test Accuracy (12/20 correct)</span>
              <span className="tabular-nums" style={{ fontSize: "1.2rem", fontWeight: "800", color: "var(--accent-warning)" }}>
                {(testAcc * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--accent-warning)", marginTop: "4px" }}>
              Wilson 95% Confidence Interval: [{(testAccCi[0] * 100).toFixed(1)}%, {(testAccCi[1] * 100).toFixed(1)}%]
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--accent-warning)", marginTop: "6px" }}>
              Test ROC-AUC: <strong className="tabular-nums">{testAuc.toFixed(4)}</strong> [95% Bootstrap CI: {testAucCi[0].toFixed(3)}, {testAucCi[1].toFixed(3)}]
            </div>
          </div>

          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
            Scored once for this model (baseline previously scored). Peking_2 is a same-institution, same-scanner batch holdout (Peking_1/3 are in training). Wide confidence intervals reflect the small sample size ($N=20$); 60% accuracy is not distinguishable from chance ($p \approx 0.25$).
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

        <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: "16px" }}>
          In multi-site resting-state fMRI, random k-fold CV can inflate performance estimates by allowing models to exploit site-correlated demographic and scanner distributions.
          Below are the empirical Leave-One-Institution-Out (LOIO) cross-validation results from the follow-up study (nested CV, inner 3-fold selection) evaluating generalization to unseen clinical sites alongside explicit majority-class, site-only, and age/sex-only baselines:
        </p>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Cohort Configuration</th>
                <th>Majority Class Baseline</th>
                <th>Site-Only Baseline AUC</th>
                <th>Age/Sex Covariates LOIO AUC</th>
                <th>LOIO Connectivity AUC [95% CI]</th>
                <th>LOIO Accuracy [95% CI]</th>
                <th>Permutation Test p-value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <strong>Config 1: All 153 Subjects</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Includes OHSU (N=14) &amp; Pitt (N=21) with 0 ADHD cases</span>
                </td>
                <td className="tabular-nums">{(siteConfoundData.config1_153.majority_accuracy * 100).toFixed(1)}%</td>
                <td className="tabular-nums" style={{ color: "var(--accent-danger)", fontWeight: "700" }}>
                  {siteConfoundData.config1_153.site_only_insample_auc.toFixed(4)} (Fit)<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config1_153.site_only_stratified_auc.toFixed(4)} (CV)</span>
                </td>
                <td className="tabular-nums">
                  {siteConfoundData.config1_153.covariates_only_baseline?.loio_auc?.toFixed(4) ?? "0.4457"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config1_153.covariates_only_baseline?.loio_auc_ci?.join(", ") ?? "0.359, 0.547"}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>
                  {siteConfoundData.config1_153.loio_connectivity_only.roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config1_153.loio_connectivity_only.roc_auc_ci.join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(siteConfoundData.config1_153.loio_connectivity_only.accuracy * 100).toFixed(1)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config1_153.loio_connectivity_only.accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "600", color: "var(--text-muted)" }}>
                  p = {siteConfoundData.config1_153.loio_connectivity_only?.p_value?.toFixed(4) ?? "0.0859"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--accent-warning)" }}>Not significant</span>
                </td>
              </tr>
              <tr style={{ background: "var(--bg-subtle)" }}>
                <td>
                  <strong>Config 2: 118 Clean Subjects</strong><br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Excludes zero-ADHD sites; KKI, NYU, NeuroIMAGE, Peking</span>
                </td>
                <td className="tabular-nums">{(siteConfoundData.config2_118.majority_accuracy * 100).toFixed(1)}%</td>
                <td className="tabular-nums" style={{ fontWeight: "600" }}>
                  {siteConfoundData.config2_118.site_only_insample_auc.toFixed(4)} (Fit)<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{siteConfoundData.config2_118.site_only_stratified_auc.toFixed(4)} (CV)</span>
                </td>
                <td className="tabular-nums">
                  {siteConfoundData.config2_118.covariates_only_baseline?.loio_auc?.toFixed(4) ?? "0.5976"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config2_118.covariates_only_baseline?.loio_auc_ci?.join(", ") ?? "0.492, 0.702"}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--brand-blue)" }}>
                  {siteConfoundData.config2_118.loio_connectivity_only.roc_auc.toFixed(4)}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config2_118.loio_connectivity_only.roc_auc_ci.join(", ")}]</span>
                </td>
                <td className="tabular-nums">
                  {(siteConfoundData.config2_118.loio_connectivity_only.accuracy * 100).toFixed(1)}%<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>[{siteConfoundData.config2_118.loio_connectivity_only.accuracy_ci.map(v => (v * 100).toFixed(1) + "%").join(", ")}]</span>
                </td>
                <td className="tabular-nums" style={{ fontWeight: "700", color: "var(--accent-success)" }}>
                  p = {siteConfoundData.config2_118.loio_connectivity_only?.p_value?.toFixed(4) ?? "0.0210"}<br />
                  <span style={{ fontSize: "0.72rem", color: "var(--accent-success)" }}>Significant (α=0.05)</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", background: "var(--bg-subtle)", padding: "12px 14px", borderRadius: "var(--radius-md)", fontSize: "0.78rem", color: "var(--text-muted)" }}>
          <Info size={16} color="var(--brand-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            <strong>What the data show:</strong> In Config 1 (153 subjects), site identity alone predicts diagnosis with an in-sample ROC-AUC of 0.7221 because OHSU and Pittsburgh contain zero ADHD cases in this release. Under Leave-One-Institution-Out evaluation where site shortcuts cannot be exploited, functional connectivity drops to near-chance (LOIO AUC 0.5342 [0.439, 0.633]), and non-parametric within-institution label permutation testing confirms this is not statistically significant (p = 0.0859). When zero-ADHD sites are excluded (Config 2, 118 subjects), cross-site generalization achieves AUC 0.6221 [0.515, 0.724] and reaches statistical significance (p = 0.0210).
          </span>
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
                <td className="tabular-nums">{cohort.train_subjects}</td>
                <td className="tabular-nums">{cohort.train_control}</td>
                <td className="tabular-nums">{cohort.train_adhd}</td>
                <td>6 Sites (KKI, NYU, OHSU, Peking_1, Peking_3, Pittsburgh)</td>
              </tr>
              <tr>
                <td><strong>Development Validation</strong></td>
                <td className="tabular-nums">{cohort.validation_subjects}</td>
                <td className="tabular-nums">{cohort.validation_control}</td>
                <td className="tabular-nums">{cohort.validation_adhd}</td>
                <td>NeuroIMAGE (Single External Site)</td>
              </tr>
              <tr style={{ background: "var(--accent-warning-subtle)" }}>
                <td><strong>Held-out Batch Partition</strong></td>
                <td className="tabular-nums" style={{ fontWeight: "700" }}>{cohort.test_subjects}</td>
                <td className="tabular-nums">{cohort.test_control}</td>
                <td className="tabular-nums">{cohort.test_adhd}</td>
                <td><strong>Peking_2 (Batch holdout; same institution as Peking_1/3)</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
