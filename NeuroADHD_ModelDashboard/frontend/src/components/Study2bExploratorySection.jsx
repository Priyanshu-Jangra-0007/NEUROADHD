import React from "react";
import { FlaskConical, Info, ExternalLink } from "lucide-react";

const EXPLORATORY_STUDY2B = {
  banner: "Exploratory, unblinded NYU, not confirmatory.",
  outcomeSentence: "no evidence of incremental connectivity signal in this design.",
  primaryN: 564,
  bestConnAuc: "0.551",
  bestConnPermP: "0.44",
  chosenConfig: "CFG_09",
  deltaAuc: "-0.0024",
  deltaCi: "[-0.0144, +0.0096]",
  deltaPermP: "0.93",
  baselines: {
    motion: "0.611",
    ageSex: "0.570",
    combined: "0.649",
  },
  sitesPositive: "1 of 5",
  powerAt055: "8%",
  powerAt060: "22%",
  realizedMatchedPairs: "207 and 219",
  quotedMatchedPairs: "215 and 226",
  links: {
    osfDoi: "https://doi.org/10.17605/OSF.IO/EQCSB",
    frozenTag: "https://github.com/Priyanshu-Jangra-0007/NEUROADHD/releases/tag/exploratory-frozen",
    resultsTag: "https://github.com/Priyanshu-Jangra-0007/NEUROADHD/releases/tag/exploratory-results",
    resultsSummary: "https://github.com/Priyanshu-Jangra-0007/NEUROADHD/blob/exploratory-results/docs/exploratory/RESULTS_SUMMARY.md",
  },
};

export function Study2bExploratorySection() {
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
        <span>{EXPLORATORY_STUDY2B.banner}</span>
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
          Primary N = {EXPLORATORY_STUDY2B.primaryN}
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
        <span>&ldquo;{EXPLORATORY_STUDY2B.outcomeSentence}&rdquo;</span>
      </div>

      {/* Key Aggregate Summary (No leaderboard or per-configuration table) */}
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
          <strong>Key aggregates (primary N={EXPLORATORY_STUDY2B.primaryN}):</strong>{" "}
          best connectivity-only AUC{" "}
          <span className="tabular-nums">{EXPLORATORY_STUDY2B.bestConnAuc}</span> (permutation{" "}
          <span className="tabular-nums">p = {EXPLORATORY_STUDY2B.bestConnPermP}</span>);{" "}
          chosen configuration{" "}
          <code style={{ fontFamily: "var(--font-mono)" }}>{EXPLORATORY_STUDY2B.chosenConfig}</code>{" "}
          delta-AUC vs motion+age+sex ={" "}
          <span className="tabular-nums">
            {EXPLORATORY_STUDY2B.deltaAuc} {EXPLORATORY_STUDY2B.deltaCi}
          </span>
          , permutation <span className="tabular-nums">p = {EXPLORATORY_STUDY2B.deltaPermP}</span>.
        </div>
        <div>
          <strong>Baselines:</strong> motion-only{" "}
          <span className="tabular-nums">{EXPLORATORY_STUDY2B.baselines.motion}</span>, age/sex{" "}
          <span className="tabular-nums">{EXPLORATORY_STUDY2B.baselines.ageSex}</span>, motion+age+sex{" "}
          <span className="tabular-nums">{EXPLORATORY_STUDY2B.baselines.combined}</span>.{" "}
          None of the three &ldquo;interesting&rdquo; conditions met ({EXPLORATORY_STUDY2B.sitesPositive} sites with positive delta-AUC).
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
            {EXPLORATORY_STUDY2B.powerAt055} at connectivity-only AUC 0.55 and about{" "}
            {EXPLORATORY_STUDY2B.powerAt060} at 0.60, so a negative result cannot rule out small
            effects. Any positive result would have been a hypothesis for a new cohort.
          </div>
          <div style={{ marginTop: "4px" }}>
            <strong>Note:</strong> realized matched cohorts had {EXPLORATORY_STUDY2B.realizedMatchedPairs} pairs
            (plan quoted {EXPLORATORY_STUDY2B.quotedMatchedPairs}).
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
          href={EXPLORATORY_STUDY2B.links.osfDoi}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          OSF plan registration (10.17605/OSF.IO/EQCSB) <ExternalLink size={12} />
        </a>
        <a
          href={EXPLORATORY_STUDY2B.links.frozenTag}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          GitHub tag: exploratory-frozen <ExternalLink size={12} />
        </a>
        <a
          href={EXPLORATORY_STUDY2B.links.resultsTag}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--brand-blue)", display: "inline-flex", alignItems: "center", gap: "4px", fontWeight: "600" }}
        >
          GitHub tag: exploratory-results <ExternalLink size={12} />
        </a>
        <a
          href={EXPLORATORY_STUDY2B.links.resultsSummary}
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
