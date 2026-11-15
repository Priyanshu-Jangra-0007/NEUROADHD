import React from "react";
import { Lock, Cpu, Menu } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

export function Header({ onOpenMobileMenu }) {
  return (
    <header className="header">
      <div className="header-left">
        <button
          onClick={onOpenMobileMenu}
          className="mobile-menu-btn"
          aria-label="Open navigation sidebar"
        >
          <Menu size={20} />
        </button>

        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <h1 style={{ fontSize: "1.15rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
              NeuroADHD Research Platform
            </h1>
            <span style={{ 
              background: "var(--accent-amber-bg)", 
              color: "var(--accent-amber)", 
              border: "1px solid var(--accent-amber)", 
              padding: "1px 8px", 
              borderRadius: "var(--radius-full)", 
              fontSize: "0.68rem", 
              fontWeight: "700",
              letterSpacing: "0.04em"
            }}>
              RESEARCH ONLY
            </span>
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "1px", display: "none" }} className="header-subtitle-desktop">
            fMRI Functional Connectivity Machine Learning Benchmark (Craddock 200)
          </p>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: "6px", 
          background: "var(--brand-blue-bg)", 
          color: "var(--brand-blue)", 
          padding: "5px 10px", 
          borderRadius: "var(--radius-md)", 
          fontSize: "0.72rem", 
          fontWeight: "600" 
        }} className="header-badge-desktop">
          <Cpu size={13} />
          <span>Balanced RF (K=100)</span>
        </div>

        <div style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: "6px", 
          background: "var(--accent-amber-bg)", 
          color: "var(--accent-amber)", 
          padding: "5px 10px", 
          borderRadius: "var(--radius-md)", 
          fontSize: "0.72rem", 
          fontWeight: "600" 
        }} className="header-badge-desktop">
          <Lock size={13} />
          <span>Peking_2 Batch Holdout (N=20)</span>
        </div>

        <ThemeToggle />
      </div>
    </header>
  );
}
