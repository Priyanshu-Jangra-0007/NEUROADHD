import React, { useState, useEffect } from "react";
import { X, CheckCircle2, AlertCircle, FileText, Database, Shield, Cpu, Terminal, Download, Sparkles } from "lucide-react";

export function RunDetailModal({ runId, onClose }) {
  const [report, setReport] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("leaderboard");
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDetail() {
      try {
        setLoading(true);
        const [repRes, aiRes] = await Promise.all([
          fetch(`http://localhost:8000/api/runs/${runId}`),
          fetch(`http://localhost:8000/api/runs/${runId}/ai-summary`).catch(() => null)
        ]);
        if (!repRes.ok) throw new Error(`Failed to load run report for ${runId}`);
        const repData = await repRes.json();
        setReport(repData);
        if (aiRes && aiRes.ok) {
          const aiData = await aiRes.json();
          setAiSummary(aiData);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchDetail();
  }, [runId]);

  if (loading) {
    return (
      <div className="modal-backdrop">
        <div className="modal-card" style={{ maxWidth: "600px", padding: "32px", textAlign: "center" }}>
          <div className="loading-spinner" style={{ margin: "0 auto 16px" }} />
          <p style={{ color: "var(--text-secondary)" }}>Loading complete run report artifact for <code>{runId}</code>...</p>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="modal-backdrop">
        <div className="modal-card" style={{ maxWidth: "500px", padding: "24px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <h3 style={{ color: "var(--accent-red)" }}>Failed to Load Run</h3>
            <button onClick={onClose} className="btn-icon"><X size={18} /></button>
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>{error || "Report unavailable."}</p>
        </div>
      </div>
    );
  }

  const dataset = report.dataset || {};
  const splits = report.splits || {};
  const preproc = report.preprocessing || {};
  const models = report.models || {};
  const ensembles = report.ensembles || {};
  const selection = report.selection || {};
  const finalTest = report.final_test || {};
  const env = report.environment || {};

  const allModelsList = [...Object.values(models), ...Object.values(ensembles)];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card modal-large" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <h2 style={{ fontSize: "18px", margin: 0 }}>Run Detail: <code>{runId}</code></h2>
              <span className={`badge ${report.status === "failed" ? "badge-danger" : "badge-success"}`}>
                {report.status}
              </span>
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
              Created: {new Date(report.created_at || Date.now()).toLocaleString()} | Duration: {report.duration_seconds}s | Purpose: {report.purpose || "Benchmark"}
            </p>
          </div>
          <button onClick={onClose} className="btn-icon" title="Close Modal"><X size={20} /></button>
        </div>

        {/* Tab Navigation */}
        <div className="modal-tabs">
          <button className={`tab-btn ${activeTab === "leaderboard" ? "active" : ""}`} onClick={() => setActiveTab("leaderboard")}>Leaderboard</button>
          <button className={`tab-btn ${activeTab === "dataset" ? "active" : ""}`} onClick={() => setActiveTab("dataset")}>Dataset & Splits</button>
          <button className={`tab-btn ${activeTab === "preproc" ? "active" : ""}`} onClick={() => setActiveTab("preproc")}>Preprocessing</button>
          <button className={`tab-btn ${activeTab === "models" ? "active" : ""}`} onClick={() => setActiveTab("models")}>Models ({allModelsList.length})</button>
          <button className={`tab-btn ${activeTab === "test" ? "active" : ""}`} onClick={() => setActiveTab("test")}>Final Test</button>
          <button className={`tab-btn ${activeTab === "env" ? "active" : ""}`} onClick={() => setActiveTab("env")}>Reproducibility</button>
          <button className={`tab-btn ${activeTab === "ai" ? "active" : ""}`} onClick={() => setActiveTab("ai")}>AI Summary</button>
        </div>

        {/* Tab Body */}
        <div className="modal-body" style={{ maxHeight: "70vh", overflowY: "auto", padding: "20px" }}>
          {activeTab === "leaderboard" && (
            <div>
              <div className="metric-cards-grid" style={{ marginBottom: "20px" }}>
                <div className="metric-card">
                  <span className="label">Selected Best Model</span>
                  <span className="value" style={{ color: "var(--accent-yellow)", fontSize: "18px" }}>{selection.selected_model || "N/A"}</span>
                  <span className="subtext">Selection Metric: {selection.selection_metric || "val_roc_auc"}</span>
                </div>
                <div className="metric-card">
                  <span className="label">Validation ROC-AUC</span>
                  <span className="value" style={{ color: "var(--accent-green)" }}>{selection.selected_validation_roc_auc ? selection.selected_validation_roc_auc.toFixed(4) : "N/A"}</span>
                  <span className="subtext">Evaluated on NeuroIMAGE</span>
                </div>
                <div className="metric-card">
                  <span className="label">Final Test Status</span>
                  <span className="value">{finalTest.final_test_executed ? "Evaluated" : "Locked"}</span>
                  <span className="subtext">Peking_2 Test Site</span>
                </div>
              </div>

              <h4 style={{ fontSize: "14px", marginBottom: "10px" }}>Validation Model Performance Leaderboard</h4>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Accuracy</th>
                    <th>Bal Acc</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Specificity</th>
                    <th>F1 Score</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {allModelsList.map((m, idx) => {
                    const vm = m.validation_metrics || {};
                    const isBest = (m.display_name === selection.selected_model || m.model_name === selection.selected_model);
                    return (
                      <tr key={idx} className={isBest ? "selected-row" : ""}>
                        <td><strong>{m.display_name || m.model_name}</strong> {isBest && <span className="badge badge-success" style={{ marginLeft: "6px" }}>SELECTED</span>}</td>
                        <td>{vm.accuracy ? vm.accuracy.toFixed(4) : "N/A"}</td>
                        <td>{vm.balanced_accuracy ? vm.balanced_accuracy.toFixed(4) : "N/A"}</td>
                        <td>{vm.precision ? vm.precision.toFixed(4) : "N/A"}</td>
                        <td>{vm.recall_sensitivity ? vm.recall_sensitivity.toFixed(4) : "N/A"}</td>
                        <td>{vm.specificity ? vm.specificity.toFixed(4) : "N/A"}</td>
                        <td>{vm.f1_score ? vm.f1_score.toFixed(4) : "N/A"}</td>
                        <td><strong style={{ color: "var(--accent-green)" }}>{vm.roc_auc ? vm.roc_auc.toFixed(4) : "N/A"}</strong></td>
                        <td>{vm.pr_auc ? vm.pr_auc.toFixed(4) : "N/A"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "dataset" && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "20px" }}>
                <div className="info-card">
                  <h4>Primary ML Dataset Metadata</h4>
                  <div className="info-list">
                    <div><span>Path:</span> <code>{dataset.dataset_path || "N/A"}</code></div>
                    <div><span>SHA-256 Checksum:</span> <code style={{ fontSize: "11px" }}>{dataset.sha256 || "N/A"}</code></div>
                    <div><span>Subjects:</span> <strong>{dataset.subject_count} (50 ADHD / 103 Control)</strong></div>
                    <div><span>Original Features:</span> {dataset.original_features} ROI connectivity edges</div>
                    <div><span>Target Column:</span> <code>{dataset.target_column}</code></div>
                  </div>
                </div>

                <div className="info-card">
                  <h4>Site-Group Holdout Partition</h4>
                  <div className="info-list">
                    <div><span>Strategy:</span> {splits.split_strategy}</div>
                    <div><span>Train:</span> {splits.train?.subject_count} Subjects ({splits.train?.sites?.join(", ")})</div>
                    <div><span>Validation:</span> {splits.validation?.subject_count} Subjects ({splits.validation?.sites?.join(", ")})</div>
                    <div><span>Test:</span> {splits.test?.subject_count} Subjects ({splits.test?.sites?.join(", ")}) — <strong>{splits.test?.test_locked ? "LOCKED" : "UNLOCKED"}</strong></div>
                    <div><span>Manifest SHA-256:</span> <code style={{ fontSize: "11px" }}>{splits.manifest_sha256 || "N/A"}</code></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "preproc" && (
            <div>
              <div className="info-card" style={{ marginBottom: "16px" }}>
                <h4>Feature Preprocessing & Leakage Protection Boundary</h4>
                <div className="info-list">
                  <div><span>Fitting Scope:</span> <span className="badge badge-primary">{preproc.fit_scope || "train_only"}</span></div>
                  <div><span>Imputer:</span> <code>{preproc.imputation_method || "SimpleImputer(median)"}</code></div>
                  <div><span>Scaler:</span> <code>{preproc.scaler || "StandardScaler"}</code></div>
                  <div><span>Feature Selector:</span> <code>{preproc.feature_selector || "SelectKBest(f_classif, k=100)"}</code></div>
                  <div><span>Input &rarr; Selected:</span> 19,900 ROI edges &rarr; <strong>100 Selected Features</strong></div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "models" && (
            <div>
              {allModelsList.map((m, idx) => (
                <div className="info-card" style={{ marginBottom: "12px" }} key={idx}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h4 style={{ margin: 0, color: "var(--accent-blue)" }}>{m.display_name || m.model_name}</h4>
                    <span className="badge badge-success">Val ROC-AUC: {m.validation_metrics?.roc_auc?.toFixed(4) || "N/A"}</span>
                  </div>
                  <div style={{ fontSize: "12px", marginTop: "8px", color: "var(--text-secondary)" }}>
                    <strong>Hyperparameters:</strong> <code>{JSON.stringify(m.best_hyperparameters || m.params || {})}</code>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "test" && (
            <div>
              {finalTest.final_test_executed ? (
                <div className="info-card" style={{ border: "1px solid var(--accent-green)" }}>
                  <h4 style={{ color: "var(--accent-green)" }}>Final Test Evaluation Results (Peking_2)</h4>
                  <div className="metric-cards-grid" style={{ marginTop: "12px" }}>
                    <div className="metric-card">
                      <span className="label">Test ROC-AUC</span>
                      <span className="value" style={{ color: "var(--accent-yellow)" }}>{finalTest.metrics?.roc_auc?.toFixed(4)}</span>
                    </div>
                    <div className="metric-card">
                      <span className="label">Test Accuracy</span>
                      <span className="value">{finalTest.metrics?.accuracy?.toFixed(4)}</span>
                    </div>
                    <div className="metric-card">
                      <span className="label">Test Balanced Acc</span>
                      <span className="value">{finalTest.metrics?.balanced_accuracy?.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="info-card" style={{ padding: "24px", textAlign: "center" }}>
                  <Shield size={32} style={{ color: "var(--accent-yellow)", margin: "0 auto 8px" }} />
                  <h4>Test Partition Remained Locked</h4>
                  <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                    The Peking_2 test set (20 subjects) was kept strictly locked during this run to prevent dataset overfitting.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === "env" && (
            <div className="info-card">
              <h4>Reproducibility & Execution Environment</h4>
              <div className="info-list">
                <div><span>OS:</span> {env.os || "N/A"}</div>
                <div><span>Python Version:</span> {env.python_version || "N/A"}</div>
                <div><span>Architecture:</span> {env.architecture || "N/A"}</div>
                <div><span>scikit-learn:</span> {env.packages?.["scikit-learn"] || "N/A"}</div>
                <div><span>xgboost:</span> {env.packages?.["xgboost"] || "N/A"}</div>
                <div><span>numpy / pandas:</span> {env.packages?.["numpy"]} / {env.packages?.["pandas"]}</div>
              </div>
            </div>
          )}

          {activeTab === "ai" && (
            <div className="info-card" style={{ background: "rgba(56, 189, 248, 0.05)", border: "1px solid var(--accent-blue)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                <Sparkles size={18} style={{ color: "var(--accent-blue)" }} />
                <h4 style={{ margin: 0, color: "var(--accent-blue)" }}>Compact Machine AI Summary</h4>
              </div>
              <pre style={{ background: "#0b0f19", padding: "14px", borderRadius: "6px", fontSize: "12px", color: "#e2e8f0", overflowX: "auto" }}>
                {JSON.stringify(aiSummary || report, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer" style={{ display: "flex", justifyContent: "space-between" }}>
          <a
            href={`http://localhost:8000/api/runs/${runId}/html`}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
            style={{ textDecoration: "none" }}
          >
            Open Standalone HTML Report &rarr;
          </a>
          <button onClick={onClose} className="btn btn-primary">Close</button>
        </div>
      </div>
    </div>
  );
}
