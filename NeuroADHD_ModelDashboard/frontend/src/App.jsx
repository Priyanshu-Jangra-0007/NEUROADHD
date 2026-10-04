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
import { MethodologyView } from "./views/MethodologyView";
import { LimitationsView } from "./views/LimitationsView";

import { BENCHMARK_DATA } from "./data/benchmarkData";

export default function App() {
  const [currentTab, setCurrentTab] = useState("overview");

  return (
    <div className="app-container">
      <Sidebar currentTab={currentTab} setTab={setCurrentTab} />

      <div className="main-wrapper" style={{ marginLeft: "240px", flex: 1, display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        <Header />

        <main className="content-area" style={{ padding: "24px 32px", maxWidth: "1300px", width: "100%", margin: "0 auto" }}>
          <DisclaimerBanner />

          {currentTab === "overview" && <OverviewView data={BENCHMARK_DATA} />}
          {currentTab === "model" && <ModelView data={BENCHMARK_DATA} />}
          {currentTab === "performance" && <PerformanceView data={BENCHMARK_DATA} />}
          {currentTab === "hybrids" && <HybridComparisonView data={BENCHMARK_DATA} />}
          {currentTab === "explainability" && <ExplainabilityView data={BENCHMARK_DATA} />}
          {currentTab === "network" && <NetworkGraphView data={BENCHMARK_DATA} />}
          {currentTab === "subjects" && <IndividualSubjectView data={BENCHMARK_DATA} />}
          {currentTab === "methodology" && <MethodologyView data={BENCHMARK_DATA} />}
          {currentTab === "limitations" && <LimitationsView />}
        </main>
      </div>
    </div>
  );
}
