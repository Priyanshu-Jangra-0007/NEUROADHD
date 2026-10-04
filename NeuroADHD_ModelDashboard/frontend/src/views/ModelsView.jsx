import React, { useState } from "react";
import { Play, CheckCircle, Info } from "lucide-react";

export function ModelsView({ models, onRunModel, isRunning }) {
  const [selectedModel, setSelectedModel] = useState(models[0]?.model_name || "random_forest");

  const activeModel = models.find(m => m.model_name === selectedModel) || models[0];
  const vm = activeModel?.validation_metrics || {};

  return (
    <div className="view-container">
      <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "6px" }}>Classification Models & Configurations</h2>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "24px" }}>Inspect model hyperparameter grids, isolated virtual environments, and evaluation metrics.</p>

      {/* Model Selector Tabs */}
      <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid var(--border-color)", marginBottom: "24px" }}>
        {models.map(m => (
          <div
            key={m.model_name}
            style={{
              padding: "10px 16px",
              cursor: "pointer",
              fontWeight: selectedModel === m.model_name ? "600" : "400",
              color: selectedModel === m.model_name ? "var(--brand-blue)" : "var(--text-secondary)",
              borderBottom: selectedModel === m.model_name ? "2px solid var(--brand-blue)" : "none"
            }}
            onClick={() => setSelectedModel(m.model_name)}
          >
            {m.display_name}
          </div>
        ))}
      </div>

      {activeModel && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
            <div>
              <h3 style={{ fontSize: "18px", fontWeight: "700" }}>{activeModel.display_name}</h3>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Environment: <code>{activeModel.environment}</code></span>
            </div>

            {activeModel.type === "individual" && (
              <button
                className="btn btn-primary"
                onClick={() => onRunModel(activeModel.model_name)}
                disabled={isRunning}
              >
                <Play size={14} />
                <span>Run {activeModel.display_name}</span>
              </button>
            )}
          </div>

          {/* Metrics Grid */}
          <div className="grid-cards">
            <div className="card">
              <div className="card-title">Val Accuracy</div>
              <div className="card-value">{(vm.accuracy * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val Bal. Accuracy</div>
              <div className="card-value">{(vm.balanced_accuracy * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val Precision</div>
              <div className="card-value">{(vm.precision * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val Recall (ADHD)</div>
              <div className="card-value">{(vm.recall_sensitivity * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val Specificity (TDC)</div>
              <div className="card-value">{(vm.specificity * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val F1 Score</div>
              <div className="card-value">{(vm.f1_score * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val ROC-AUC</div>
              <div className="card-value" style={{ color: "var(--brand-blue)" }}>{(vm.roc_auc * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Val PR-AUC</div>
              <div className="card-value">{(vm.pr_auc * 100 || 0).toFixed(2)}%</div>
            </div>
          </div>

          {/* Hyperparameters & Environment Card */}
          <div className="card" style={{ marginBottom: "24px" }}>
            <div className="card-title">Best Hyperparameters & Preprocessing Configuration</div>
            <pre style={{ background: "var(--bg-subtle)", padding: "12px", borderRadius: "6px", fontSize: "12px", overflowX: "auto" }}>
              {JSON.stringify({
                model_name: activeModel.model_name,
                best_hyperparameters: activeModel.best_hyperparameters || "Default parameters fitted",
                preprocessing_pipeline: [
                  "SimpleImputer(strategy='median')",
                  "VarianceThreshold(threshold=0.0)",
                  "StandardScaler()",
                  "SelectKBest(score_func=f_classif, k=100)"
                ],
                fitted_partition: "TRAIN (114 subjects strictly)"
              }, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
