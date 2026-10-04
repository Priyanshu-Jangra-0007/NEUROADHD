# NeuroADHD — Phase 1

Research environment for initial ADHD-200 experiments. Python dependencies and the Jupyter kernel are installed in the project-local `.venv/`.

## Environment

From this directory:

```bash
source .venv/bin/activate
jupyter lab
```

Select **Python (NeuroADHD .venv)** in Jupyter.

## Data

Put unchanged ADHD-200 files in the matching folders under `data/adhd200/raw/`. Raw data, processed data, features, models, and results are excluded from Git. Do not edit downloaded source files.

The first requested NITRC file, **AAL Time Courses Corrected Filtering**, redirected to the NITRC login page. Per the download instructions, the remaining derivatives were not attempted. See `data/adhd200/manifest.csv` for the blocked request. No ADHD-200 data was downloaded or modified.
