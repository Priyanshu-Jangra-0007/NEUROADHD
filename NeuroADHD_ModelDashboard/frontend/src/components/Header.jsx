import React from "react";
import { Lock, Cpu } from "lucide-react";

export function Header() {
  return (
    <header className="header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 28px", background: "var(--bg-surface)", borderBottom: "1px solid var(--border-color)" }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <h1 style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--text-primary)" }}>NeuroADHD Research Platform</h1>
          <span className="badge" style={{ background: "var(--accent-amber-bg)", color: "var(--accent-amber)", border: "1px solid rgba(217, 119, 6, 0.3)", padding: "2px 8px", borderRadius: "9999px", fontSize: "0.7rem", fontWeight: "700" }}>
            RESEARCH ONLY
          </span>
        </div>
        <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "2px" }}>
          ADHD Classification from Resting-State fMRI Functional Connectivity (Craddock 200 Parcellation)
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", background: "var(--brand-blue-bg)", color: "var(--brand-blue)", padding: "6px 12px", borderRadius: "6px", fontSize: "0.75rem", fontWeight: "600" }}>
          <Cpu size={14} />
          <span>Balanced Random Forest (K=100)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", background: "#fef3c7", color: "#b45309", padding: "6px 12px", borderRadius: "6px", fontSize: "0.75rem", fontWeight: "600" }}>
          <Lock size={14} />
          <span>Peking_2 Locked Test (N=20)</span>
        </div>
      </div>
    </header>
  );
}
