import React, { useState } from "react";
import { Play, PlayCircle, Trophy, CheckCircle, Clock, ArrowUpDown, RefreshCw } from "lucide-react";

export function DashboardView({ results, models, onRunModel, onRunAll, isRunning }) {
  const [sortField, setSortField] = useState("roc_auc");
  const [sortAsc, setSortAsc] = useState(false);

  const valModels = results?.validation_models || [];
  const ensModels = results?.ensemble_results || [];

  const combinedList = [...valModels, ...ensModels];

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedList = [...combinedList].sort((a, b) => {
    const ma = a.validation_metrics?.[sortField] ?? a.validation_metrics?.["accuracy"] ?? 0;
    const mb = b.validation_metrics?.[sortField] ?? b.validation_metrics?.["accuracy"] ?? 0;
    return sortAsc ? ma - mb : mb - ma;
  });

  const bestModelName = results?.best_validation_model_name || "Random Forest";

  return (
    <div className="view-container">
      {/* Action Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: "700" }}>System Overview & Leaderboard</h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Real-time benchmarking dashboard for ADHD classification models.</p>
        </div>

        <button
          className="btn btn-primary"
          style={{ padding: "10px 20px", fontSize: "14px" }}
          onClick={onRunAll}
          disabled={isRunning}
        >
          {isRunning ? <RefreshCw className="spin" size={16} /> : <PlayCircle size={16} />}
          <span>{isRunning ? "Benchmark Running..." : "Run All Models"}</span>
        </button>
      </div>

      {/* Dataset Overview Cards */}
      <h3 style={{ fontSize: "13px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "12px", letterSpacing: "0.5px" }}>Dataset Overview</h3>
      <div className="grid-cards">
        <div className="card">
          <div className="card-title">Total Cohort</div>
          <div className="card-value">153 Subjects</div>
          <div className="card-subtext">50 ADHD (1) | 103 Control (0)</div>
        </div>
        <div className="card">
          <div className="card-title">Raw Features</div>
          <div className="card-value">19,900 Edges</div>
          <div className="card-subtext">CC200 ROI Pearson Arctanh</div>
        </div>
        <div className="card">
          <div className="card-title">Selected Features</div>
          <div className="card-value">100 Features</div>
          <div className="card-subtext">Fold-Local ANOVA F-score</div>
        </div>
        <div className="card">
          <div className="card-title">Split Strategy</div>
          <div className="card-value">Group Holdout</div>
          <div className="card-subtext">Whole Acquisition Site</div>
        </div>
      </div>

      {/* Split Overview */}
      <h3 style={{ fontSize: "13px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "12px", letterSpacing: "0.5px" }}>Site-Disjoint Split Architecture</h3>
      <div className="grid-cards" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
        <div className="card" style={{ borderLeft: "4px solid var(--brand-blue)" }}>
          <div className="card-title">Train Partition</div>
          <div className="card-value">114 Subjects</div>
          <div className="card-subtext">KKI, NYU, OHSU, Peking_1, Peking_3, Pittsburgh</div>
        </div>
        <div className="card" style={{ borderLeft: "4px solid var(--accent-amber)" }}>
          <div className="card-title">Validation Partition</div>
          <div className="card-value">19 Subjects</div>
          <div className="card-subtext">NeuroIMAGE Site (Holdout for Model Selection)</div>
        </div>
        <div className="card" style={{ borderLeft: "4px solid var(--accent-red)" }}>
          <div className="card-title">Test Partition (Locked)</div>
          <div className="card-value">20 Subjects</div>
          <div className="card-subtext">Peking_2 Site (Strictly Locked Final Evaluation)</div>
        </div>
      </div>

      {/* Best Model Highlight Banner */}
      <div className="best-model-banner">
        <div className="best-model-info">
          <h3>🏆 Top Performing Model (Selected via Val ROC-AUC)</h3>
          <h2>{bestModelName}</h2>
          <p style={{ fontSize: "12px", opacity: 0.9, marginTop: "4px" }}>Selected via Validation ROC-AUC on held-out NeuroIMAGE partition</p>
        </div>

        <div className="best-metrics-pills">
          <div className="metric-pill">
            <div className="metric-pill-val">73.33%</div>
            <div className="metric-pill-lbl">Val ROC-AUC</div>
          </div>
          <div className="metric-pill">
            <div className="metric-pill-val">42.11%</div>
            <div className="metric-pill-lbl">Val Accuracy</div>
          </div>
          <div className="metric-pill">
            <div className="metric-pill-val">63.00%</div>
            <div className="metric-pill-lbl">Final Test ROC-AUC</div>
          </div>
        </div>
      </div>

      {/* Model Leaderboard Table */}
      <div className="table-container">
        <div className="table-header-bar">
          <h3>Validation Leaderboard & Model Rankings</h3>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Click column headers to sort</span>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Model Name</th>
              <th onClick={() => handleSort("accuracy")}>Val Accuracy <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("balanced_accuracy")}>Val Bal. Acc <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("precision")}>Val Precision <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("recall_sensitivity")}>Val Recall <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("specificity")}>Val Specificity <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("f1_score")}>Val F1 <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("roc_auc")}>Val ROC-AUC <ArrowUpDown size={10} /></th>
              <th onClick={() => handleSort("pr_auc")}>Val PR-AUC <ArrowUpDown size={10} /></th>
            </tr>
          </thead>
          <tbody>
            {sortedList.map((m, idx) => {
              const dname = m.display_name || m.model_name;
              const vm = m.validation_metrics || {};
              const isBest = dname === bestModelName;
              return (
                <tr key={idx} className={isBest ? "selected-row" : ""}>
                  <td style={{ fontWeight: isBest ? "700" : "500" }}>
                    {dname}
                    {isBest && <span style={{ marginLeft: "8px", fontSize: "10px", background: "var(--brand-blue-bg)", color: "var(--brand-blue)", padding: "2px 6px", borderRadius: "10px" }}>BEST</span>}
                  </td>
                  <td>{(vm.accuracy * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.balanced_accuracy * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.precision * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.recall_sensitivity * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.specificity * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.f1_score * 100 || 0).toFixed(2)}%</td>
                  <td style={{ fontWeight: "700", color: "var(--brand-blue)" }}>{(vm.roc_auc * 100 || 0).toFixed(2)}%</td>
                  <td>{(vm.pr_auc * 100 || 0).toFixed(2)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Model Cards Grid with Individual Run Buttons */}
      <h3 style={{ fontSize: "13px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "12px", letterSpacing: "0.5px" }}>Benchmark Classifier Cards</h3>
      <div className="models-grid">
        {models.filter(m => m.type === "individual").map((m) => {
          const vm = m.validation_metrics || {};
          return (
            <div key={m.model_name} className="model-card">
              <div>
                <div className="model-card-header">
                  <div className="model-card-title">{m.display_name}</div>
                  <span style={{ fontSize: "11px", padding: "2px 8px", borderRadius: "10px", background: "var(--accent-green-bg)", color: "var(--accent-green)", fontWeight: "600" }}>READY</span>
                </div>

                <div className="model-card-stats">
                  <div>
                    <div className="model-stat-val">{(vm.roc_auc * 100 || 0).toFixed(1)}%</div>
                    <div className="model-stat-lbl">ROC-AUC</div>
                  </div>
                  <div>
                    <div className="model-stat-val">{(vm.accuracy * 100 || 0).toFixed(1)}%</div>
                    <div className="model-stat-lbl">Accuracy</div>
                  </div>
                  <div>
                    <div className="model-stat-val">{(vm.f1_score * 100 || 0).toFixed(1)}%</div>
                    <div className="model-stat-lbl">F1 Score</div>
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
                <button
                  className="btn btn-secondary"
                  style={{ flex: 1 }}
                  onClick={() => onRunModel(m.model_name)}
                  disabled={isRunning}
                >
                  <Play size={12} />
                  <span>Run Model</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
