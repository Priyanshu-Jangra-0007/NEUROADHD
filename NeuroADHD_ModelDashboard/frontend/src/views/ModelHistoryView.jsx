import React, { useState, useEffect } from "react";
import { TrendingUp, LineChart, Cpu, Calendar, CheckCircle2 } from "lucide-react";

export function ModelHistoryView() {
  const [historyData, setHistoryData] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState("all");

  useEffect(() => {
    async function fetchHistory() {
      try {
        setLoading(true);
        const res = await fetch("http://localhost:8000/api/models/history");
        if (res.ok) {
          const data = await res.json();
          setHistoryData(data);
        }
      } catch (err) {
        console.error("Failed to fetch model performance history:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, []);

  if (loading) {
    return (
      <div className="view-container" style={{ textAlign: "center", padding: "40px" }}>
        <div className="loading-spinner" style={{ margin: "0 auto 16px" }} />
        <p style={{ color: "var(--text-secondary)" }}>Loading model performance trajectories across historical runs...</p>
      </div>
    );
  }

  const modelKeys = Object.keys(historyData);

  return (
    <div className="view-container">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "4px" }}>Model Performance Progression History</h2>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Track validation ROC-AUC, accuracy, and F1 trajectories across historical benchmark runs to verify model improvements over time.
          </p>
        </div>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{
            background: "var(--card-bg)",
            border: "1px solid var(--border-color)",
            color: "var(--text-primary)",
            padding: "8px 14px",
            borderRadius: "6px",
            fontSize: "13px"
          }}
        >
          <option value="all">Show All Model Architectures</option>
          {modelKeys.map((key) => (
            <option key={key} value={key}>
              {historyData[key]?.[0]?.display_name || key}
            </option>
          ))}
        </select>
      </div>

      {modelKeys.length === 0 ? (
        <div className="info-card" style={{ padding: "32px", textAlign: "center" }}>
          <TrendingUp size={32} style={{ color: "var(--accent-blue)", margin: "0 auto 8px" }} />
          <h4>No Historical Progression Data Yet</h4>
          <p style={{ color: "var(--text-secondary)", fontSize: "13px" }}>
            Run more benchmark experiments to track algorithm improvement trajectories over time.
          </p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {modelKeys
            .filter((mKey) => selectedModel === "all" || selectedModel === mKey)
            .map((mKey) => {
              const entries = historyData[mKey] || [];
              const displayName = entries[0]?.display_name || mKey;
              const firstScore = entries[0]?.roc_auc || 0;
              const latestScore = entries[entries.length - 1]?.roc_auc || 0;
              const delta = latestScore - firstScore;

              return (
                <div key={mKey} className="info-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Cpu size={18} style={{ color: "var(--accent-blue)" }} />
                      <h3 style={{ margin: 0, fontSize: "16px", color: "var(--text-primary)" }}>{displayName}</h3>
                      <span className="badge badge-primary">{entries.length} Run{entries.length > 1 ? "s" : ""}</span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                        Current ROC-AUC: <strong style={{ color: "var(--accent-green)" }}>{latestScore.toFixed(4)}</strong>
                      </span>
                      {entries.length > 1 && (
                        <span
                          style={{
                            fontSize: "12px",
                            fontWeight: "bold",
                            color: delta >= 0 ? "var(--accent-green)" : "var(--accent-red)"
                          }}
                        >
                          {delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>

                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Run Order</th>
                        <th>Run ID</th>
                        <th>Val ROC-AUC</th>
                        <th>Val Accuracy</th>
                        <th>Val F1 Score</th>
                        <th>Val Bal Acc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entries.map((item, idx) => (
                        <tr key={item.run_id || idx}>
                          <td><strong>Run #{idx + 1}</strong></td>
                          <td><code>{item.run_id}</code></td>
                          <td>
                            <strong style={{ color: "var(--accent-green)" }}>{item.roc_auc ? item.roc_auc.toFixed(4) : "N/A"}</strong>
                          </td>
                          <td>{item.accuracy ? item.accuracy.toFixed(4) : "N/A"}</td>
                          <td>{item.f1_score ? item.f1_score.toFixed(4) : "N/A"}</td>
                          <td>{item.balanced_accuracy ? item.balanced_accuracy.toFixed(4) : "N/A"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}
