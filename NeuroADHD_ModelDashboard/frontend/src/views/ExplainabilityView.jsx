import React, { useState } from "react";
import { Search, ArrowUpDown, Info } from "lucide-react";

export function ExplainabilityView({ data }) {
  const topEdges = data.explainability.top_edges;
  const hubRois = data.explainability.hub_rois;
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState("mean_abs_shap");
  const [sortAsc, setSortAsc] = useState(false);

  const filteredEdges = topEdges
    .filter(e => 
      e.feature_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      `node ${e.roi_a_id}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
      `node ${e.roi_b_id}`.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      return sortAsc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
    });

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>TreeSHAP Feature Attribution &amp; Subnetwork Ranking</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Global SHAP feature importance and permutation scoring for the selected CC200 functional connectivity subnetwork.
        </p>
      </div>

      {/* Hub ROIs Summary Cards */}
      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>
          Top Predictive CC200 Hub ROIs
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px" }}>
          {hubRois.slice(0, 6).map((h, i) => (
            <div key={i} style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "8px", padding: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.75rem", fontWeight: "700", color: "var(--brand-blue)" }}>CC200 Node {h.roi_id}</span>
                <span style={{ fontSize: "0.7rem", background: "var(--bg-subtle)", padding: "2px 6px", borderRadius: "4px" }}>#{h.hub_rank}</span>
              </div>
              <div style={{ fontSize: "1.1rem", fontWeight: "800", marginTop: "4px" }}>{h.roi_name}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                {h.top_50_edge_count} Top-50 Edges ({h.top_100_edge_count} All-100)
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top 20 Connectivity Table */}
      <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }}>
            Top Connectivity Edges (SHAP &amp; Permutation Ranked)
          </h3>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "var(--bg-subtle)", padding: "6px 12px", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
            <Search size={14} color="var(--text-muted)" />
            <input
              type="text"
              placeholder="Search edge or node ID..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{ border: "none", background: "transparent", outline: "none", fontSize: "0.8rem", color: "var(--text-primary)" }}
            />
          </div>
        </div>

        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-muted)" }}>
              <th style={{ padding: "8px", cursor: "pointer" }} onClick={() => handleSort("shap_rank")}>
                Rank <ArrowUpDown size={12} style={{ display: "inline" }} />
              </th>
              <th style={{ padding: "8px" }}>Edge Name</th>
              <th style={{ padding: "8px" }}>ROI A (Cluster)</th>
              <th style={{ padding: "8px" }}>ROI B (Cluster)</th>
              <th style={{ padding: "8px", cursor: "pointer" }} onClick={() => handleSort("mean_abs_shap")}>
                Mean |SHAP| <ArrowUpDown size={12} style={{ display: "inline" }} />
              </th>
              <th style={{ padding: "8px", cursor: "pointer" }} onClick={() => handleSort("permutation_importance_mean")}>
                Perm. ΔAUC <ArrowUpDown size={12} style={{ display: "inline" }} />
              </th>
              <th style={{ padding: "8px" }}>Predictive Association Direction</th>
            </tr>
          </thead>
          <tbody>
            {filteredEdges.map((e, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--border-color)" }}>
                <td style={{ padding: "8px", fontWeight: "700" }}>{e.shap_rank}</td>
                <td style={{ padding: "8px" }}><code>{e.feature_name}</code></td>
                <td style={{ padding: "8px" }}>CC200 Node {e.roi_a_id}</td>
                <td style={{ padding: "8px" }}>CC200 Node {e.roi_b_id}</td>
                <td style={{ padding: "8px", fontWeight: "700", color: "var(--brand-blue)" }}>{e.mean_abs_shap.toFixed(4)}</td>
                <td style={{ padding: "8px" }}>+{e.permutation_importance_mean.toFixed(4)}</td>
                <td style={{ padding: "8px", fontSize: "0.8rem", color: e.direction.includes("ADHD") ? "var(--accent-red)" : "var(--brand-blue)" }}>
                  {e.direction}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ marginTop: "12px", display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <Info size={14} />
          <span>Note: CC200 ROIs correspond to spatial spectral clusters from Craddock et al. (2012). Anatomical labels are left uninvented to maintain scientific validity.</span>
        </div>
      </div>
    </div>
  );
}
