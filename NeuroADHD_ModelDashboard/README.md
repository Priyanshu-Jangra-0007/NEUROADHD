# NeuroADHD Model Dashboard

A professional, research-oriented web dashboard for managing, running, comparing, and monitoring machine learning classification models trained on the ADHD-200 CC200 dataset.

## Architecture

```
down/
├── NeuroADHD/                  ← READ-ONLY Original Research Project
├── NeuroADHD_ModelBenchmark/   ← READ-ONLY Benchmark Suite & Data Lake
└── NeuroADHD_ModelDashboard/   ← APPLICATION SUITE (Frontend + Backend)
    ├── frontend/               ← React + Vite Web UI
    └── backend/                ← FastAPI Server API
```

## Quick Start Instructions

### 1. Launch FastAPI Backend Server

```bash
cd NeuroADHD_ModelDashboard/backend
../../NeuroADHD/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Launch Vite React Frontend

```bash
cd NeuroADHD_ModelDashboard/frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

## Features & Capabilities

- **System Status & Dataset Specs**: Live display of ADHD-200 CC200 cohort parameters (153 subjects, 50 ADHD, 103 Control, 19,900 Fisher-z edge features).
- **Validation Leaderboard**: Real-time ranking of classifiers (Random Forest, XGBoost, SVMs, Logistic Regression, Voting Ensembles, Stacking Meta-Classifier).
- **Model Execution Triggers**: Trigger single models or master benchmark runs (`run_all_models.py`) in isolated model virtual environments (`model_envs/`).
- **Locked Peking_2 Final Test Control**: Visually isolated Peking_2 final test evaluation trigger with data hygiene warning callouts.
- **Interactive Reports**: Built-in HTML report dashboard viewer (`model_comparison.html`).
- **Academic Research Focus**: Non-clinical framing with explicit academic disclaimers.
