import React, { useState } from "react";
import { Network, Info } from "lucide-react";

export function NetworkGraphView({ data }) {
  const edges = data.explainability.top_edges;
  const hubs = data.explainability.hub_rois;

  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);

  // Position nodes in a circular layout for clean scientific visualization
  const nodeMap = {};
  const activeRoiIds = new Set();
  edges.slice(0, 15).forEach(e => {
    activeRoiIds.add(e.roi_a_id);
    activeRoiIds.add(e.roi_b_id);
  });

  const uniqueRoiList = Array.from(activeRoiIds);
  const totalNodes = uniqueRoiList.length;
  const radius = 170;
  const centerX = 260;
  const centerY = 210;

  uniqueRoiList.forEach((roiId, idx) => {
    const angle = (idx / totalNodes) * 2 * Math.PI - Math.PI / 2;
    const isHub = hubs.some(h => h.roi_id === roiId && h.hub_rank <= 5);
    const hubData = hubs.find(h => h.roi_id === roiId);

    nodeMap[roiId] = {
      id: roiId,
      name: `ROI_${String(roiId).padStart(3, "0")}`,
      label: `Node ${roiId}`,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      isHub: isHub,
      hubData: hubData,
      size: isHub ? 16 : 10
    };
  });

  return (
    <div className="view-container">
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: "800", color: "var(--text-primary)" }}>Model-Derived Predictive Connectivity Network</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Interactive topological graph showing model-important CC200 functional connectivity edges and hub nodes.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: "20px", marginBottom: "20px" }}>
        {/* SVG Network Graph */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "12px", textAlign: "left" }}>
            Topology Graph: Top 15 Predictive Functional Edges
          </div>

          <svg width="520" height="420" viewBox="0 0 520 420" style={{ maxWidth: "100%", height: "auto", background: "var(--bg-subtle)", borderRadius: "8px" }}>
            {/* Draw Edges */}
            {edges.slice(0, 15).map((e, idx) => {
              const nodeA = nodeMap[e.roi_a_id];
              const nodeB = nodeMap[e.roi_b_id];
              if (!nodeA || !nodeB) return null;

              const isSelected = selectedEdge === e.feature_name || selectedNode === e.roi_a_id || selectedNode === e.roi_b_id;
              const strokeColor = e.direction.includes("ADHD") ? "#ef4444" : "#3b82f6";
              const strokeWidth = Math.max(1.5, e.mean_abs_shap * 160);

              return (
                <line
                  key={idx}
                  x1={nodeA.x}
                  y1={nodeA.y}
                  x2={nodeB.x}
                  y2={nodeB.y}
                  stroke={strokeColor}
                  strokeWidth={isSelected ? strokeWidth + 2 : strokeWidth}
                  strokeOpacity={isSelected ? 0.95 : 0.6}
                  style={{ cursor: "pointer", transition: "all 0.2s" }}
                  onClick={() => { setSelectedEdge(e.feature_name); setSelectedNode(null); }}
                />
              );
            })}

            {/* Draw Nodes */}
            {Object.values(nodeMap).map((n, idx) => {
              const isSelected = selectedNode === n.id;
              const nodeColor = n.isHub ? "#8b5cf6" : "#0284c7";

              return (
                <g key={idx} style={{ cursor: "pointer" }} onClick={() => { setSelectedNode(n.id); setSelectedEdge(null); }}>
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={n.size}
                    fill={nodeColor}
                    stroke="#ffffff"
                    strokeWidth={isSelected ? 3 : 1.5}
                    style={{ transition: "all 0.2s" }}
                  />
                  <text
                    x={n.x}
                    y={n.y + n.size + 12}
                    textAnchor="middle"
                    fontSize="9px"
                    fill="var(--text-primary)"
                    fontWeight={n.isHub ? "700" : "500"}
                  >
                    N{n.id}
                  </text>
                </g>
              );
            })}
          </svg>

          <div style={{ display: "flex", justifyContent: "center", gap: "16px", marginTop: "12px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#8b5cf6", display: "inline-block" }}></span>
              <span>Hub Node (Top-5 Centrality)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#0284c7", display: "inline-block" }}></span>
              <span>Peripheral Cluster Node</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: "16px", height: "3px", background: "#ef4444", display: "inline-block" }}></span>
              <span>+ADHD Weight</span>
            </div>
          </div>
        </div>

        {/* Detail Panel */}
        <div className="card" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: "700", marginBottom: "12px", color: "var(--text-primary)" }}>
            Subnetwork Element Details
          </h3>

          {selectedNode ? (
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Selected Node</div>
              <div style={{ fontSize: "1.3rem", fontWeight: "800", color: "var(--brand-blue)", marginTop: "2px" }}>
                CC200 Node {selectedNode}
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "8px" }}>
                Atlas Identifier: <strong>ROI_{String(selectedNode).padStart(3, "0")}</strong>
              </div>
              {nodeMap[selectedNode]?.hubData ? (
                <div style={{ marginTop: "12px", background: "var(--bg-subtle)", padding: "12px", borderRadius: "6px", fontSize: "0.8rem" }}>
                  <div>Hub Centrality Rank: <strong>#{nodeMap[selectedNode].hubData.hub_rank}</strong></div>
                  <div style={{ marginTop: "4px" }}>Top-50 Edges Involving Node: <strong>{nodeMap[selectedNode].hubData.top_50_edge_count}</strong></div>
                  <div style={{ marginTop: "4px" }}>Mean Edge |SHAP|: <strong>{nodeMap[selectedNode].hubData.mean_edge_shap_importance.toFixed(4)}</strong></div>
                </div>
              ) : (
                <div style={{ marginTop: "12px", fontSize: "0.8rem", color: "var(--text-muted)" }}>Peripheral connectivity node in model-selected feature set.</div>
              )}
            </div>
          ) : selectedEdge ? (
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Selected Edge</div>
              <div style={{ fontSize: "1.1rem", fontWeight: "800", color: "var(--text-primary)", marginTop: "2px" }}>
                <code>{selectedEdge}</code>
              </div>
              {(() => {
                const edgeObj = edges.find(e => e.feature_name === selectedEdge);
                if (!edgeObj) return null;
                return (
                  <div style={{ marginTop: "12px", background: "var(--bg-subtle)", padding: "12px", borderRadius: "6px", fontSize: "0.8rem" }}>
                    <div>Node Pair: <strong>CC200 Node {edgeObj.roi_a_id} — Node {edgeObj.roi_b_id}</strong></div>
                    <div style={{ marginTop: "4px" }}>Mean |SHAP|: <strong>{edgeObj.mean_abs_shap.toFixed(4)}</strong></div>
                    <div style={{ marginTop: "4px" }}>Permutation ΔAUC: <strong>+{edgeObj.permutation_importance_mean.toFixed(4)}</strong></div>
                    <div style={{ marginTop: "6px", color: edgeObj.direction.includes("ADHD") ? "var(--accent-red)" : "var(--brand-blue)" }}>
                      Direction: {edgeObj.direction}
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", padding: "20px 0", textAlign: "center" }}>
              Click any node or connectivity edge in the graph to inspect detailed attribution weights and hub centrality metrics.
            </div>
          )}

          <div style={{ marginTop: "24px", padding: "12px", background: "var(--bg-subtle)", borderRadius: "6px", display: "flex", alignItems: "flex-start", gap: "8px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            <Info size={16} style={{ flexShrink: 0, marginTop: "2px" }} />
            <span><strong>Model-derived predictive connectivity network:</strong> This diagram reflects features utilized by the Random Forest classifier, NOT a complete biological connectome.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
