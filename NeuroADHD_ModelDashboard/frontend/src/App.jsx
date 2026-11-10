import React, { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { OverviewView } from "./views/OverviewView";
import { ModelView } from "./views/ModelView";
import { PerformanceView } from "./views/PerformanceView";
import { HybridComparisonView } from "./views/HybridComparisonView";
import { ExplainabilityView } from "./views/ExplainabilityView";
import { NetworkGraphView } from "./views/NetworkGraphView";
import { IndividualSubjectView } from "./views/IndividualSubjectView";
import { SingleSubjectPredictView } from "./views/SingleSubjectPredictView";
import { MethodologyView } from "./views/MethodologyView";
import { LimitationsView } from "./views/LimitationsView";

import { BENCHMARK_DATA } from "./data/benchmarkData";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "32px", background: "var(--accent-red-bg)", border: "1px solid var(--accent-red)", borderRadius: "var(--radius-lg)", margin: "20px 0" }}>
          <h3 style={{ color: "var(--accent-red)", fontWeight: "800", marginBottom: "8px" }}>View Render Error</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
            An unexpected error occurred while rendering this tab:
          </p>
          <pre style={{ background: "#ffffff", padding: "12px", borderRadius: "6px", marginTop: "12px", fontSize: "0.8rem", color: "#991b1b", overflowX: "auto" }}>
            {this.state.error?.toString()}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [currentTab, setCurrentTab] = useState("overview");

  return (
    <div className="app-container">
      <Sidebar currentTab={currentTab} setTab={setCurrentTab} />

      <div className="main-wrapper" style={{ marginLeft: "240px", flex: 1, display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        <Header />

        <main className="content-area" style={{ padding: "24px 32px", maxWidth: "1300px", width: "100%", margin: "0 auto" }}>
          <DisclaimerBanner />

          <ErrorBoundary key={currentTab}>
            {currentTab === "overview" && <OverviewView data={BENCHMARK_DATA} />}
            {currentTab === "model" && <ModelView data={BENCHMARK_DATA} />}
            {currentTab === "predict" && <SingleSubjectPredictView />}
            {currentTab === "performance" && <PerformanceView data={BENCHMARK_DATA} />}
            {currentTab === "hybrids" && <HybridComparisonView data={BENCHMARK_DATA} />}
            {currentTab === "explainability" && <ExplainabilityView data={BENCHMARK_DATA} />}
            {currentTab === "network" && <NetworkGraphView data={BENCHMARK_DATA} />}
            {currentTab === "subjects" && <IndividualSubjectView data={BENCHMARK_DATA} />}
            {currentTab === "methodology" && <MethodologyView data={BENCHMARK_DATA} />}
            {currentTab === "limitations" && <LimitationsView />}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
