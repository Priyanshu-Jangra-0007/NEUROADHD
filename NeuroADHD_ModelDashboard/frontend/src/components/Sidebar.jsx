import React from "react";
import { 
  LayoutDashboard, 
  BrainCircuit, 
  Activity, 
  Layers, 
  Sparkles, 
  Network, 
  UserCheck, 
  GitBranch, 
  AlertTriangle 
} from "lucide-react";

export function Sidebar({ currentTab, setTab }) {
  const menuItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "model", label: "Model & Pipeline", icon: BrainCircuit },
    { id: "performance", label: "Robustness & Test", icon: Activity },
    { id: "hybrids", label: "Hybrid Comparison", icon: Layers },
    { id: "explainability", label: "SHAP Explainability", icon: Sparkles },
    { id: "network", label: "Connectivity Network", icon: Network },
    { id: "subjects", label: "Subject Decompositions", icon: UserCheck },
    { id: "methodology", label: "Methodology & Splits", icon: GitBranch },
    { id: "limitations", label: "Limitations", icon: AlertTriangle }
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-badge">RESEARCH</div>
        <h2 style={{ fontSize: "1.15rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.02em" }}>NeuroADHD</h2>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Benchmark Dashboard v2.2</span>
      </div>

      <nav className="sidebar-nav" style={{ marginTop: "16px" }}>
        {menuItems.map(item => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setTab(item.id)}
              className={`nav-item ${isActive ? "active" : ""}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                width: "100%",
                padding: "10px 18px",
                border: "none",
                background: isActive ? "var(--bg-subtle)" : "transparent",
                color: isActive ? "var(--brand-blue)" : "var(--text-secondary)",
                fontWeight: isActive ? "600" : "500",
                fontSize: "0.875rem",
                textAlign: "left",
                cursor: "pointer",
                borderLeft: isActive ? "3px solid var(--brand-blue)" : "3px solid transparent",
                transition: "all 0.15s ease"
              }}
            >
              <Icon size={17} color={isActive ? "var(--brand-blue)" : "var(--text-muted)"} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div style={{ marginTop: "auto", padding: "16px 18px", borderTop: "1px solid var(--border-color)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
        <div>Cohort: <strong>153 Subjects</strong></div>
        <div style={{ marginTop: "4px" }}>Site: <strong>7 Dev / 1 Locked Test</strong></div>
      </div>
    </aside>
  );
}
