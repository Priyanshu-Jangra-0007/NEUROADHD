"""FastAPI backend server for the NeuroADHD Model Dashboard."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Dynamic path resolution to import inference logic from NeuroADHD
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEUROADHD_ROOT = WORKSPACE_ROOT / "NeuroADHD"
if str(NEUROADHD_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROADHD_ROOT))

try:
    from src.inference.predict_subject import predict_single_subject
except ImportError:
    predict_single_subject = None


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


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB upload limit


@app.post("/api/predict/subject")
async def predict_subject(file: UploadFile = File(...)):
    """Accept an uploaded roi_CC200.csv file and return an exploratory ADHD research prediction.

    Accepts: multipart/form-data with a file field containing roi_CC200.csv
    Returns: JSON containing validation status, prediction probabilities, and research disclaimer.
    """
    if predict_single_subject is None:
        raise HTTPException(
            status_code=500,
            detail="Inference module is not available on the server."
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Uploaded file must be a CSV file (.csv) containing CC200 ROI time series."
        )

    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE:
                temp_file.close()
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds maximum allowed limit of 10 MB."
                )
            temp_file.write(chunk)
        temp_file.close()

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            raw_result = predict_single_subject(temp_path)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Model artifact could not be located on the server."
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Inference processing failed: {str(exc)}"
            )

        # Return sanitized response without exposing internal server filesystem paths
        return {
            "status": "success",
            "filename": file.filename,
            "input_validation": raw_result["input_validation"],
            "model_architecture": raw_result["model_architecture"],
            "prediction": raw_result["prediction"],
            "disclaimer": raw_result["disclaimer"]
        }

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.get("/api/predict/sample")
def get_sample_subject():
    """Return a verified sample CC200 ROI time-series CSV for testing single-subject inference."""
    sample_path = NEUROADHD_ROOT / "data" / "adhd200" / "raw" / "cc200" / "1017176_session_1" / "roi_CC200.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample subject file not found on disk.")
    return FileResponse(
        path=str(sample_path),
        filename="1017176_session_1_roi_CC200.csv",
        media_type="text/csv"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


