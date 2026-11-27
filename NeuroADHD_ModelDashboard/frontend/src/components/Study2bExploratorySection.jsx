import React from "react";
import { FlaskConical, Info, ExternalLink } from "lucide-react";
import study2bData from "../data/study2bData.json";

function displayField(value) {
  return value !== undefined && value !== null && value !== "" ? value : "N/A";
}

export function Study2bExploratorySection() {
  const osf = study2bData?.osf ?? {};
  const primary = study2bData?.primary ?? {};
  const baselines = primary.baselines ?? {};
  const motion = baselines.motion ?? {};
  const ageSex = baselines.age_sex ?? {};
  const combined = baselines.combined ?? {};
  const power = study2bData?.power ?? {};
  const matched = study2bData?.matched_cohorts ?? {};
  const links = study2bData?.links ?? {};

  const bannerText = "Exploratory — NYU was scored in Study 2 and is no longer sealed; not confirmatory.";

  return (
    <section
      data-testid="study2b-exploratory-section"
      aria-label="Study 2b exploratory follow-up"
      className="card"
      style={{
        marginBottom: "24px",
        borderStyle: "dashed",
        borderWidth: "1.5px",
        borderColor: "var(--brand-blue)",
        background: "var(--bg-subtle)",
      }}
    >
      {/* Always-visible Exploratory Banner */}
      <div
        data-testid="study2b-banner"
        style={{
          background: "var(--brand-blue-bg)",
          color: "var(--brand-blue)",
          border: "1px solid var(--brand-blue)",
          borderRadius: "var(--radius-sm)",
          padding: "8px 12px",
          marginBottom: "14px",
          fontSize: "0.8rem",
          fontWeight: "700",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <span>{bannerText}</span>
        <span style={{ fontSize: "0.72rem", fontWeight: "600" }}>
          Separate from Study 2 confirmatory evaluation
        </span>
      </div>

      {/* Section Title */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <FlaskConical size={18} color="var(--brand-blue)" />
          <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
            Study 2b (exploratory follow-up)
          </h3>
        </div>
        <span style={{ fontSize: "0.74rem", color: "var(--text-muted)", fontWeight: "600" }}>
          Primary N = {displayField(primary.n_subjects)}
        </span>
      </div>

      {/* Pre-declared Outcome Sentence (Verbatim) */}
      <div
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-md)",
          padding: "10px 14px",
          marginBottom: "14px",
          fontSize: "0.84rem",
          color: "var(--text-primary)",
        }}
      >
        <strong>Pre-declared outcome:</strong>{" "}
        <span>&ldquo;{displayField(primary.pre_declared_outcome)}.&rdquo;</span>
      </div>

      {/* Key Aggregate Summary (Read strictly from study2bData.json; no per-configuration table) */}
      <div
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-md)",
          padding: "12px 14px",
          marginBottom: "12px",
          fontSize: "0.8rem",
          lineHeight: 1.65,
          color: "var(--text-secondary)",
        }}
      >
        <div>
          <strong>Key aggregates (primary N={displayField(primary.n_subjects)}):</strong>{" "}
          best connectivity-only AUC{" "}
          <span className="tabular-nums">{displayField(primary.best_conn_auc_display)}</span> (permutation{" "}
          <span className="tabular-nums">p = {displayField(primary.best_conn_perm_p_display)}</span>);{" "}
          chosen configuration{" "}
          <code style={{ fontFamily: "var(--font-mono)" }}>{displayField(primary.chosen_config)}</code>{" "}
          delta-AUC vs motion+age+sex ={" "}
          <span className="tabular-nums">
            {displayField(primary.delta_auc_display)} {displayField(primary.delta_ci_display)}
          </span>
          , permutation <span className="tabular-nums">p = {displayField(primary.delta_perm_p_display)}</span>.
        </div>
        <div>
          <strong>Baselines:</strong> motion-only{" "}
          <span className="tabular-nums">{displayField(motion.auc_display)}</span>, age/sex{" "}
          <span className="tabular-nums">{displayField(ageSex.auc_display)}</span>, motion+age+sex{" "}
          <span className="tabular-nums">{displayField(combined.auc_display)}</span>.{" "}
          {displayField(primary.motion_beats_best_conn_sentence)}{" "}
          None of the three &ldquo;interesting&rdquo; conditions met ({displayField(primary.sites_positive_display)} sites with positive delta-AUC).
        </div>
      </div>

      {/* Power Caveat & Realized Matched Cohort Note */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "8px",
          background: "var(--bg-surface)",
          border: "1px solid var(--border-color)",
          padding: "10px 14px",
          borderRadius: "var(--radius-md)",
          fontSize: "0.78rem",
          lineHeight: 1.55,
          color: "var(--text-muted)",
          marginBottom: "12px",
        }}
      >
        <Info size={16} color="var(--brand-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div>
          <div>
            <strong>Power caveat:</strong> simulated power of the full decision rule about{" "}
            {displayField(power.weak1_rule_power_display)} at connectivity-only AUC{" "}
            {displayField(power.weak1_conn_auc_display)} and about{" "}
            {displayField(power.weak2_rule_power_display)} at{" "}
            {displayField(power.weak2_conn_auc_display)}, so a negative result cannot rule out small
            effects. Any positive result would have been a hypothesis for a new cohort.
          </div>
          <div style={{ marginTop: "4px" }}>
            <strong>Note:</strong> realized matched cohorts had {displayField(matched.realized_pairs_display)} pairs
            (plan quoted {displayField(matched.pre_declared_pairs_display)}). See{" "}
            <a
              href={links.matched_pairs_note || "#"}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--brand-blue)", fontWeight: "600" }}
            >
              {displayField(matched.note_filename)}
            </a>{" "}
            for why the realized pair counts differ (sensitivity cohorts only; primary analysis unaffected).
          </div>
        </div>
      </div>

      {/* Provenance & Full Table Links */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "14px",
          alignItems: "center",
          fontSize: "0.78rem",
          color: "var(--text-secondary)",
        }}
      >
        <a
          href={links.osf_doi || "#"}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          OSF plan registration ({displayField(osf.doi)}) <ExternalLink size={12} />
        </a>
        <a
          href={links.frozen_tag || "#"}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          GitHub tag: {displayField(osf.frozen_tag)} <ExternalLink size={12} />
        </a>
        <a
          href={links.results_tag || "#"}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          GitHub tag: {displayField(osf.results_tag)} <ExternalLink size={12} />
        </a>
        <a
          href={links.results_summary || "#"}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          Full configuration summary (RESULTS_SUMMARY.md) <ExternalLink size={12} />
        </a>
      </div>
    </section>
  );
}
