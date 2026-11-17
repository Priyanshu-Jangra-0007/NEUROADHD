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
  AlertTriangle,
  UploadCloud,
  X
} from "lucide-react";

export function Sidebar({ currentTab, setTab, isOpen, onClose }) {
  const menuItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "model", label: "Model & Pipeline", icon: BrainCircuit },
    { id: "predict", label: "Single-Subject Prediction", icon: UploadCloud },
    { id: "performance", label: "Held-out Peking_2 test (same institution as two training sites)", icon: Activity },
    { id: "hybrids", label: "Hybrid Comparison", icon: Layers },
    { id: "explainability", label: "SHAP Explainability", icon: Sparkles },
    { id: "network", label: "Connectivity Network", icon: Network },
    { id: "subjects", label: "Subject Decompositions", icon: UserCheck },
    { id: "methodology", label: "Methodology & Splits", icon: GitBranch },
    { id: "limitations", label: "Limitations", icon: AlertTriangle }
  ];

  const handleSelectTab = (id) => {
    setTab(id);
    if (onClose) {
      onClose();
    }
  };

  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose?.();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      <div 
        className={`sidebar-backdrop ${isOpen ? "open" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`sidebar ${isOpen ? "open" : ""}`} aria-label="Main Navigation">
        <div className="sidebar-brand">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div className="brand-badge">RESEARCH PLATFORM</div>
              <h2 style={{ fontSize: "1.15rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.02em" }}>NeuroADHD</h2>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Benchmark Dashboard v2.2</span>
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="mobile-menu-btn"
                aria-label="Close navigation sidebar"
                style={{ marginLeft: "auto" }}
              >
                <X size={18} />
              </button>
            )}
          </div>
        </div>

        <nav className="sidebar-nav">
          {menuItems.map(item => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleSelectTab(item.id)}
                className={`nav-item ${isActive ? "active" : ""}`}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon size={17} style={{ flexShrink: 0 }} />
                <span style={{ lineHeight: 1.35 }}>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div style={{ marginTop: "auto", padding: "16px 18px", borderTop: "1px solid var(--border-color)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <div>Cohort: <strong style={{ color: "var(--text-primary)" }}>153 subjects (6 institutions)</strong></div>
          <div style={{ marginTop: "4px" }}>Design: <span style={{ color: "var(--brand-blue)", fontWeight: "600" }}>Research Benchmark</span></div>
        </div>
      </aside>
    </>
  );
}
