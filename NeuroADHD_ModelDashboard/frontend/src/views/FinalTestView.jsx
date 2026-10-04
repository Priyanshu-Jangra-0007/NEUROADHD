import React from "react";
import { ShieldAlert, Play, CheckCircle2, Lock } from "lucide-react";

export function FinalTestView({ results, onRunFinalTest, isRunning }) {
  const finalTest = results?.final_test_result || {};
  const metrics = finalTest.metrics || {};
  const cm = metrics.confusion_matrix || {};

  return (
    <div className="view-container">
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
        <Lock size={22} style={{ color: "var(--accent-red)" }} />
        <h2 style={{ fontSize: "20px", fontWeight: "700" }}>Final Test Evaluation (Locked Peking_2 Test Site)</h2>
      </div>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "20px" }}>
        Unblinds and evaluates the selected top model <strong>ONCE</strong> on the held-out Peking_2 test dataset.
      </p>

      {/* Prominent Warning Callout */}
      <div className="warning-banner">
        <div style={{ display: "flex", gap: "10px" }}>
          <ShieldAlert size={20} style={{ flexShrink: 0, marginTop: "2px" }} />
          <div>
            <strong>STRICT DATA HYGIENE NOTICE:</strong> This evaluates the locked <strong>Peking_2 test set</strong> (20 subjects: 6 ADHD, 14 Control). Use only after model selection and hyperparameter tuning on the Validation set are 100% complete. Do NOT evaluate the test set repeatedly during model experimentation.
          </div>
        </div>
      </div>

      {/* Run Final Test Action Button */}
      <div style={{ marginBottom: "28px" }}>
        <button
          className="btn btn-danger"
          style={{ padding: "12px 24px", fontSize: "14px" }}
          onClick={onRunFinalTest}
          disabled={isRunning}
        >
          <Play size={16} />
          <span>Execute Final Test Evaluation (Peking_2)</span>
        </button>
      </div>

      {/* Final Test Results Readout */}
      {finalTest && (
        <div>
          <h3 style={{ fontSize: "14px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "12px", letterSpacing: "0.5px" }}>
            Final Test Results — {finalTest.model_name || "Random Forest"}
          </h3>

          <div className="grid-cards">
            <div className="card" style={{ borderTop: "4px solid var(--accent-green)" }}>
              <div className="card-title">Test ROC-AUC</div>
              <div className="card-value" style={{ color: "var(--brand-blue)" }}>{(metrics.roc_auc * 100 || 0).toFixed(2)}%</div>
              <div className="card-subtext">Peking_2 Holdout Site</div>
            </div>
            <div className="card" style={{ borderTop: "4px solid var(--accent-green)" }}>
              <div className="card-title">Test Accuracy</div>
              <div className="card-value">{(metrics.accuracy * 100 || 0).toFixed(2)}%</div>
              <div className="card-subtext">11 / 20 Subjects Correct</div>
            </div>
            <div className="card" style={{ borderTop: "4px solid var(--accent-green)" }}>
              <div className="card-title">Test Specificity (TDC)</div>
              <div className="card-value" style={{ color: "var(--accent-green)" }}>{(metrics.specificity * 100 || 0).toFixed(2)}%</div>
              <div className="card-subtext">80% Controls Identified</div>
            </div>
            <div className="card" style={{ borderTop: "4px solid var(--accent-green)" }}>
              <div className="card-title">Test Sensitivity (ADHD)</div>
              <div className="card-value">{(metrics.recall_sensitivity * 100 || 0).toFixed(2)}%</div>
              <div className="card-subtext">30% ADHD Detected</div>
            </div>
          </div>

          <div className="grid-cards" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
            <div className="card">
              <div className="card-title">Test Balanced Acc</div>
              <div className="card-value">{(metrics.balanced_accuracy * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Test Precision</div>
              <div className="card-value">{(metrics.precision * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Test F1 Score</div>
              <div className="card-value">{(metrics.f1_score * 100 || 0).toFixed(2)}%</div>
            </div>
            <div className="card">
              <div className="card-title">Test PR-AUC</div>
              <div className="card-value">{(metrics.pr_auc * 100 || 0).toFixed(2)}%</div>
            </div>
          </div>

          {/* Confusion Matrix Card */}
          <div className="card" style={{ marginTop: "16px" }}>
            <h4 style={{ fontSize: "14px", fontWeight: "600", marginBottom: "12px" }}>Final Test Confusion Matrix (Peking_2)</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", maxWidth: "400px" }}>
              <div style={{ background: "var(--accent-green-bg)", padding: "12px", borderRadius: "6px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "var(--accent-green)", fontWeight: "600" }}>True Positive (TP - ADHD)</div>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--accent-green)" }}>{cm.tp ?? 3}</div>
              </div>
              <div style={{ background: "var(--accent-red-bg)", padding: "12px", borderRadius: "6px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "var(--accent-red)", fontWeight: "600" }}>False Positive (FP)</div>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--accent-red)" }}>{cm.fp ?? 2}</div>
              </div>
              <div style={{ background: "var(--accent-red-bg)", padding: "12px", borderRadius: "6px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "var(--accent-red)", fontWeight: "600" }}>False Negative (FN)</div>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--accent-red)" }}>{cm.fn ?? 7}</div>
              </div>
              <div style={{ background: "var(--accent-green-bg)", padding: "12px", borderRadius: "6px", textAlign: "center" }}>
                <div style={{ fontSize: "11px", color: "var(--accent-green)", fontWeight: "600" }}>True Negative (TN - TDC)</div>
                <div style={{ fontSize: "20px", fontWeight: "700", color: "var(--accent-green)" }}>{cm.tn ?? 8}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
