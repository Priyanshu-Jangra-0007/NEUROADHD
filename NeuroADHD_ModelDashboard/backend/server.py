"""FastAPI backend server for the NeuroADHD Model Dashboard."""
from __future__ import annotations

import json
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from services.benchmark_service import (
    get_dataset_info,
    get_latest_run_id,
    load_latest_results,
    get_all_models_summary,
    get_all_runs_index,
    load_run_report,
    compare_runs,
    get_model_performance_history,
    run_single_model_async,
    run_all_models_async,
    CURRENT_JOB,
    BENCHMARK_ROOT,
    RUNS_DIR
)

app = FastAPI(
    title="NeuroADHD Model Benchmark API",
    description="Backend API serving model benchmark leaderboards, experiment history, comparisons, and execution triggers.",
    version="2.0.0"
)

# Enable CORS for Vite dev server frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_system_status():
    return {
        "status": "online",
        "dataset": get_dataset_info(),
        "active_run_id": get_latest_run_id(),
        "current_job": CURRENT_JOB,
        "disclaimer": "NeuroADHD is an academic research system for ADHD classification experiments. It is not a clinically validated diagnostic tool."
    }


@app.get("/api/models")
def get_models_list():
    return get_all_models_summary()


@app.get("/api/results")
def get_benchmark_results():
    return load_latest_results()


@app.get("/api/results/{model_name}")
def get_model_detail(model_name: str):
    run_id = get_latest_run_id()
    if not run_id:
        raise HTTPException(status_code=404, detail="No benchmark runs found")
        
    metrics_path = RUNS_DIR / run_id / "metrics" / f"{model_name}.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
        
    results = load_latest_results()
    for m in results.get("validation_models", []) + results.get("ensemble_results", []):
        if m["model_name"] == model_name:
            return m
            
    raise HTTPException(status_code=404, detail=f"No results found for model {model_name}")


@app.get("/api/runs")
def list_runs():
    return get_all_runs_index()


@app.get("/api/runs/compare")
def compare_selected_runs(run_ids: str = Query(..., description="Comma separated run_ids")):
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="No run_ids provided")
    return compare_runs(ids)


@app.get("/api/models/history")
def get_models_history():
    return get_model_performance_history()


@app.get("/api/runs/{run_id}")
def get_run_report_detail(run_id: str):
    try:
        return load_run_report(run_id)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.get("/api/runs/{run_id}/html")
def get_run_html_report(run_id: str):
    html_path = RUNS_DIR / run_id / "run_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"HTML report for run {run_id} not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/ai-summary")
def get_run_ai_summary(run_id: str):
    ai_json = RUNS_DIR / run_id / "ai_summary.json"
    if not ai_json.exists():
        raise HTTPException(status_code=404, detail=f"AI summary for run {run_id} not found.")
    return json.loads(ai_json.read_text(encoding="utf-8"))


@app.post("/api/models/{model_name}/run")
def trigger_single_model(model_name: str):
    valid_models = ["logistic_regression", "linear_svm", "rbf_svm", "random_forest", "extra_trees", "xgboost"]
    if model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Must be one of {valid_models}")
        
    started = run_single_model_async(model_name)
    if not started:
        raise HTTPException(status_code=409, detail="A benchmark job is already running.")
        
    return {"status": "started", "model_name": model_name}


@app.post("/api/run-all")
def trigger_run_all():
    started = run_all_models_async()
    if not started:
        raise HTTPException(status_code=409, detail="A benchmark job is already running.")
        
    return {"status": "started", "job": "run_all_models"}


@app.get("/api/reports/html")
def get_latest_html_report():
    run_id = get_latest_run_id()
    if not run_id:
        raise HTTPException(status_code=404, detail="No benchmark runs found.")
    return get_run_html_report(run_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
