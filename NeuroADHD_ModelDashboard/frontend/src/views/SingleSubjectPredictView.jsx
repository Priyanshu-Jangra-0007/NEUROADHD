import React, { useState } from "react";
import {
  UploadCloud,
  AlertCircle,
  Brain,
  AlertTriangle,
  RotateCcw,
  Activity,
  Info,
  Download,
  FileCheck,
  CheckCircle2
} from "lucide-react";

export function SingleSubjectPredictView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const fetchWithFallback = async (endpoint, options = {}) => {
    // Candidates in priority order: Vite proxy relative URL, 127.0.0.1 direct, localhost direct
    const candidates = [
      endpoint,
      `http://127.0.0.1:8000${endpoint}`,
      `http://localhost:8000${endpoint}`
    ];

    for (const url of candidates) {
      try {
        const res = await fetch(url, options);
        if (res.ok) {
          return res;
        }
        // If HTTP status error from server (400, 413, 500, etc.)
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error (${res.status})`);
      } catch (err) {
        // If it's a specific HTTP error from the server (e.g. invalid CSV), don't try other hosts
        if (err.message && !err.message.includes("Failed to fetch") && !err.message.includes("NetworkError")) {
          throw err;
        }
      }
    }

    throw new Error(
      "Cannot connect to NeuroADHD backend on port 8000. Please ensure the backend is running with: " +
      "cd NeuroADHD_ModelDashboard/backend && ../../NeuroADHD/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000"
    );
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file) => {
    setError(null);
    setResult(null);

    const nameLower = file.name.toLowerCase();
    if (!nameLower.endsWith(".csv") && !nameLower.endsWith(".txt")) {
      setError("Invalid file type. Please upload a .csv file containing CC200 ROI time series.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("File size exceeds maximum allowed limit (10 MB).");
      return;
    }

    setSelectedFile(file);
  };

  const handleLoadSample = async (e) => {
    if (e) e.stopPropagation();
    setSampleLoading(true);
    setError(null);
    try {
      const response = await fetchWithFallback("/api/predict/sample");
      const blob = await response.blob();
      const file = new File([blob], "1017176_session_1_roi_CC200.csv", { type: "text/csv" });
      setSelectedFile(file);
    } catch (err) {
      setError(err.message || "Failed to load sample scan from backend.");
    } finally {
      setSampleLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetchWithFallback("/api/predict/subject", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Inference failed. Please verify that the backend is active on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setError(null);
    setResult(null);
  };


  return (
    <div className="view-container" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* View Header */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
          <span className="brand-badge" style={{ background: "var(--brand-blue-bg)", color: "var(--brand-blue)" }}>
            INFERENCE ENGINE
          </span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>CC200 Functional Connectome</span>
        </div>
        <h2 style={{ fontSize: "1.5rem", fontWeight: "800", letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
          Single-Subject fMRI Prediction
        </h2>
        <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginTop: "4px", maxWidth: "800px" }}>
          Upload a pre-extracted CC200 resting-state ROI time course CSV to compute 19,900 pairwise Fisher-z
          correlations and generate an exploratory machine-learning research prediction using the frozen benchmark model.
        </p>
      </div>

      {/* Input Requirements & Domain Scope Callout */}
      <div
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-lg)",
          padding: "18px 22px",
          display: "flex",
          gap: "16px",
          alignItems: "flex-start"
        }}
      >
        <div
          style={{
            background: "var(--brand-blue-bg)",
            color: "var(--brand-blue)",
            padding: "8px",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          <Info size={20} />
        </div>
        <div style={{ flex: 1, fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          <div style={{ fontWeight: "700", color: "var(--text-primary)", marginBottom: "4px" }}>
            Input Compatibility & Preprocessing Requirements
          </div>
          <ul style={{ paddingLeft: "18px", margin: "4px 0" }}>
            <li>
              <strong>File Format:</strong> A CSV file (e.g. <code>roi_CC200.csv</code>) containing <strong>exactly 200 ROI rows</strong> (labeled 1 to 200) across resting-state BOLD timepoints.
            </li>
            <li>
              <strong>Preprocessing:</strong> Data must be registered to MNI152 space, nuisance-regressed (motion + WM/CSF), and bandpass filtered (0.01–0.1 Hz).
            </li>
            <li>
              <strong>Scope Constraint:</strong> This endpoint accepts <strong>only pre-extracted CC200 time-series CSVs</strong>. Raw 4D NIfTI (<code>.nii.gz</code>) files, structural T1w scans, and questionnaire reports are not accepted.
            </li>
          </ul>
        </div>
      </div>

      {/* Upload Box (Visible when no result is loaded) */}
      {!result && (
        <div
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-lg)",
            padding: "28px",
            boxShadow: "var(--shadow-sm)"
          }}
        >
          <form onSubmit={handleSubmit}>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              style={{
                border: isDragging ? "2px dashed var(--brand-blue)" : "2px dashed var(--border-color)",
                background: isDragging ? "var(--brand-blue-bg)" : "var(--bg-app)",
                borderRadius: "var(--radius-lg)",
                padding: "40px 20px",
                textAlign: "center",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
              onClick={() => document.getElementById("roi-file-input")?.click()}
            >
              <input
                id="roi-file-input"
                type="file"
                accept=".csv,.txt"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
              <div
                style={{
                  width: "56px",
                  height: "56px",
                  margin: "0 auto 14px",
                  borderRadius: "50%",
                  background: "var(--brand-blue-bg)",
                  color: "var(--brand-blue)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}
              >
                <UploadCloud size={28} />
              </div>

              {selectedFile ? (
                <div>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: "8px", fontWeight: "700", color: "var(--text-primary)", fontSize: "1rem" }}>
                    <CheckCircle2 size={18} color="var(--accent-green)" />
                    <span>{selectedFile.name}</span>
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: "4px" }}>
                    {(selectedFile.size / 1024).toFixed(1)} KB — Ready for inference
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                    }}
                    style={{
                      marginTop: "10px",
                      background: "transparent",
                      border: "1px solid var(--border-color)",
                      borderRadius: "var(--radius-sm)",
                      padding: "4px 10px",
                      fontSize: "0.75rem",
                      color: "var(--text-muted)",
                      cursor: "pointer"
                    }}
                  >
                    Remove / Choose another file
                  </button>
                </div>
              ) : (
                <div>
                  <div style={{ fontWeight: "600", color: "var(--text-primary)", fontSize: "0.95rem" }}>
                    Drag & drop subject <code>roi_CC200.csv</code> here, or click to browse
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: "6px" }}>
                    Accepts CC200 ROI time-series tables (.csv, up to 10 MB)
                  </div>
                  <div style={{ marginTop: "14px" }}>
                    <button
                      type="button"
                      disabled={sampleLoading}
                      onClick={handleLoadSample}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "6px",
                        background: "var(--brand-blue-bg)",
                        color: "var(--brand-blue)",
                        border: "1px solid var(--brand-blue)",
                        padding: "6px 14px",
                        borderRadius: "var(--radius-sm)",
                        fontWeight: "600",
                        fontSize: "0.8rem",
                        cursor: sampleLoading ? "wait" : "pointer"
                      }}
                    >
                      {sampleLoading ? (
                        <>
                          <Activity size={13} className="animate-spin" />
                          <span>Loading Sample...</span>
                        </>
                      ) : (
                        <>
                          <FileCheck size={14} />
                          <span>1-Click: Load Sample Subject 1017176</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Error Banner */}
            {error && (
              <div
                style={{
                  marginTop: "16px",
                  padding: "12px 16px",
                  background: "var(--accent-red-bg)",
                  border: "1px solid var(--accent-red)",
                  borderRadius: "var(--radius-md)",
                  color: "var(--accent-red)",
                  fontSize: "0.85rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px"
                }}
              >
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            {/* Action Bar */}
            <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                <span>Sample available: <strong>Subject 1017176</strong> (261 timepoints)</span>
                <a
                  href="/api/predict/sample"
                  download="1017176_roi_CC200.csv"
                  style={{ display: "inline-flex", alignItems: "center", gap: "4px", color: "var(--brand-blue)", textDecoration: "none" }}
                >
                  <Download size={13} />
                  <span>Download Sample CSV</span>
                </a>
              </div>
              <button
                type="submit"
                disabled={!selectedFile || loading}
                className="btn-primary"
                style={{
                  padding: "10px 24px",
                  borderRadius: "var(--radius-md)",
                  border: "none",
                  background: !selectedFile || loading ? "var(--border-color)" : "var(--brand-blue)",
                  color: "#ffffff",
                  fontWeight: "600",
                  fontSize: "0.875rem",
                  cursor: !selectedFile || loading ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  transition: "background 0.15s ease"
                }}
              >
                {loading ? (
                  <>
                    <Activity size={16} className="animate-spin" />
                    <span>Processing Inference...</span>
                  </>
                ) : (
                  <>
                    <Brain size={16} />
                    <span>Run Research Prediction</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Main Prediction Score Card */}
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-color)",
              borderRadius: "var(--radius-lg)",
              padding: "28px",
              boxShadow: "var(--shadow-sm)"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
              <div>
                <span style={{ fontSize: "0.75rem", fontWeight: "700", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em" }}>
                  CLASSIFICATION OUTCOME
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "4px" }}>
                  <h3 style={{ fontSize: "1.75rem", fontWeight: "800", color: "var(--text-primary)" }}>
                    {result.prediction.class_label.toUpperCase()}
                  </h3>
                  <span
                    style={{
                      background: result.prediction.predicted_class === 1 ? "var(--accent-amber-bg)" : "var(--brand-blue-bg)",
                      color: result.prediction.predicted_class === 1 ? "var(--accent-amber)" : "var(--brand-blue)",
                      fontWeight: "700",
                      fontSize: "0.75rem",
                      padding: "4px 10px",
                      borderRadius: "12px"
                    }}
                  >
                    Class {result.prediction.predicted_class}
                  </span>
                </div>
              </div>

              <button
                onClick={handleReset}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "8px 14px",
                  border: "1px solid var(--border-color)",
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-surface)",
                  color: "var(--text-secondary)",
                  fontSize: "0.8rem",
                  cursor: "pointer",
                  fontWeight: "500"
                }}
              >
                <RotateCcw size={14} />
                <span>Analyze Another Scan</span>
              </button>
            </div>

            {/* Probability Gauge Bar */}
            <div style={{ marginBottom: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", fontWeight: "600", marginBottom: "8px" }}>
                <span style={{ color: "var(--brand-blue)" }}>
                  Control Resemblance: {(result.prediction.control_probability * 100).toFixed(1)}%
                </span>
                <span style={{ color: "var(--accent-amber)" }}>
                  ADHD Pattern Resemblance: {(result.prediction.adhd_probability * 100).toFixed(1)}%
                </span>
              </div>

              <div
                style={{
                  height: "16px",
                  width: "100%",
                  background: "var(--bg-subtle)",
                  borderRadius: "8px",
                  overflow: "hidden",
                  display: "flex",
                  position: "relative"
                }}
              >
                <div
                  style={{
                    width: `${result.prediction.control_probability * 100}%`,
                    background: "var(--brand-blue)",
                    transition: "width 0.5s ease"
                  }}
                />
                <div
                  style={{
                    width: `${result.prediction.adhd_probability * 100}%`,
                    background: "var(--accent-amber)",
                    transition: "width 0.5s ease"
                  }}
                />
                {/* 50% Threshold Marker */}
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    bottom: 0,
                    width: "2px",
                    background: "#0f172a",
                    transform: "translateX(-50%)"
                  }}
                />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "6px" }}>
                <span>0.0 (Pure Typical Control)</span>
                <span>Decision Cut-off: 0.50</span>
                <span>1.0 (Pure ADHD Pattern)</span>
              </div>
            </div>

            {/* Metrics Breakdown Grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "12px",
                borderTop: "1px solid var(--border-color)",
                paddingTop: "20px"
              }}
            >
              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Source Filename</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.filename}
                </div>
              </div>

              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Assessed ROIs</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.input_validation.roi_count} CC200 Regions
                </div>
              </div>

              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Processed Timepoints</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.input_validation.timepoints} BOLD Frames
                </div>
              </div>

              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Evaluated Edges</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.input_validation.features_extracted.toLocaleString()} Functional Pairs
                </div>
              </div>

              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Biomarkers Fed to Model</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.input_validation.features_selected} Selected Features
                </div>
              </div>

              <div style={{ background: "var(--bg-app)", padding: "12px", borderRadius: "var(--radius-md)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Architecture</div>
                <div style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "0.9rem", marginTop: "2px" }}>
                  {result.model_architecture}
                </div>
              </div>
            </div>
          </div>

          {/* Prominent Research Notice & Clinical Disclaimer */}
          <div
            style={{
              background: "var(--accent-amber-bg)",
              border: "1px solid var(--accent-amber)",
              borderRadius: "var(--radius-lg)",
              padding: "20px 24px",
              display: "flex",
              gap: "16px",
              alignItems: "flex-start"
            }}
          >
            <div style={{ color: "var(--accent-amber)", marginTop: "2px" }}>
              <AlertTriangle size={24} />
            </div>
            <div style={{ fontSize: "0.85rem", color: "#78350f", lineHeight: 1.6 }}>
              <div style={{ fontWeight: "800", fontSize: "0.95rem", marginBottom: "4px" }}>
                RESEARCH PREDICTION NOTICE — NOT A CLINICAL DIAGNOSIS
              </div>
              <p>
                The probability displayed above represents the mathematical resemblance of this resting-state fMRI
                connectome to the patterns observed in the ADHD-200 Consortium training cohort. It is <strong>NOT a clinically
                calibrated medical risk score, biomarker guarantee, or DSM-5/ICD-11 diagnostic assessment</strong>.
              </p>
              <p style={{ marginTop: "6px" }}>
                ADHD is a complex neurodevelopmental condition diagnosed exclusively by certified clinicians through
                structured psychiatric evaluations, behavioral questionnaires, and developmental history. NeuroADHD is intended
                strictly for neuroscience methodology research and model benchmarking.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
