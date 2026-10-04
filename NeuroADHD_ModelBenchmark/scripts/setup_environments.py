"""Setup isolated Python virtual environments for each benchmarked model."""
import os
import subprocess
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ENVS_DIR = BENCHMARK_ROOT / "model_envs"
PYTHON_BIN = sys.executable

MODELS = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "extra_trees",
    "xgboost",
    "ensemble"
]

BASE_REQUIREMENTS = [
    "numpy==2.4.6",
    "pandas==3.0.6",
    "scikit-learn==1.9.1",
    "scipy==1.17.1",
    "joblib==1.6.0"
]

MODEL_SPECIFIC_REQUIREMENTS = {
    "logistic_regression": [],
    "linear_svm": [],
    "rbf_svm": [],
    "random_forest": [],
    "extra_trees": [],
    "xgboost": ["xgboost==3.2.0"],
    "ensemble": ["xgboost==3.2.0"]
}


def setup_env(model_name: str) -> Path:
    env_dir = ENVS_DIR / model_name
    req_file = env_dir / "requirements.txt"
    py_executable = env_dir / "bin" / "python"
    
    print(f"Creating isolated environment for [{model_name}] at: {env_dir}")
    if not env_dir.exists():
        subprocess.run([PYTHON_BIN, "-m", "venv", str(env_dir)], check=True)
    
    reqs = BASE_REQUIREMENTS + MODEL_SPECIFIC_REQUIREMENTS.get(model_name, [])
    req_file.write_text("\n".join(reqs) + "\n", encoding="utf-8")
    
    print(f"Installing dependencies into [{model_name}] environment...")
    subprocess.run([str(py_executable), "-m", "pip", "install", "--quiet", "-r", str(req_file)], check=True)
    
    # Save lockfile with exact installed versions
    pip_freeze = subprocess.run([str(py_executable), "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (env_dir / "environment_lock.txt").write_text(pip_freeze.stdout, encoding="utf-8")
    
    print(f"Environment for [{model_name}] ready.\n")
    return py_executable


def main():
    ENVS_DIR.mkdir(parents=True, exist_ok=True)
    print("==================================================")
    print("SETTING UP MODEL-SPECIFIC ISOLATED ENVIRONMENTS")
    print("==================================================")
    for model_name in MODELS:
        setup_env(model_name)
    print("All model environments successfully created and isolated.")


if __name__ == "__main__":
    main()
