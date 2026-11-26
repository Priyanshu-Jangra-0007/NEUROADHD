#!/usr/bin/env python3
"""
Opt-in real-shape synthetic end-to-end timing benchmark for scripts/run_exploratory.py.

Runs the full exploratory pipeline on synthetic data at exact real Athena shapes
(N=620 scanned -> N=564 primary clean, K=190 CC200 ROIs, exact per-site scan lengths)
with an explicit --timeout guard. Uses ZERO real data and ZERO real labels.
"""

from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.run_exploratory import (  # noqa: E402
    LabelFreeInputs,
    build_cohorts,
    run_exploratory_synthetic_e2e,
    timeseries_to_fisher_z,
)

REAL_SITE_SPECS = (
    ("NYU", 196, 20, 172),
    ("Peking", 189, 5, 233),
    ("NeuroIMAGE", 42, 6, 257),
    ("KKI", 79, 4, 149),
    ("OHSU", 58, 21, 78),
)


def build_real_shape_synthetic_inputs(n_rois: int = 190, seed: int = 2026) -> tuple:
    rng = np.random.default_rng(seed)
    sids, insts, ts_list, z_list, mot_list, as_list, excl_list, tlen_list, y_list = (
        [], [], [], [], [], [], [], [], []
    )
    mr_sids, mr_insts, mr_z, mr_mot, mr_as, mr_cnts, mr_y = [], [], [], [], [], [], []
    counter = 0
    for site, n_clean, n_excl, t_len in REAL_SITE_SPECS:
        for i in range(n_clean + n_excl):
            counter += 1
            sid = f"SYN_{counter:05d}"
            is_excl = 1 if i >= n_clean else 0
            y = i % 2
            fd = (0.55 + 0.05 * rng.random()) if is_excl else (0.15 + 0.04 * (i % 5) + 0.01 * rng.random())
            ts = rng.normal(size=(t_len, n_rois))
            z = timeseries_to_fisher_z(ts)
            sids.append(sid)
            insts.append(site)
            ts_list.append(ts)
            z_list.append(z)
            mot_list.append([fd, fd * 2.5])
            as_list.append([10.0 + 2.0 * rng.normal(), float(i % 2)])
            excl_list.append(is_excl)
            tlen_list.append(t_len)
            y_list.append(y)
            if site in {"NYU", "OHSU"}:
                mr_sids.append(sid)
                mr_insts.append(site)
                mr_z.append(z)
                mr_mot.append([fd, fd * 2.5])
                mr_as.append([10.0 + 2.0 * rng.normal(), float(i % 2)])
                mr_cnts.append(2)
                mr_y.append(y)

    inputs = LabelFreeInputs(
        subject_ids=np.array(sids, dtype="<U16"),
        institutions=np.array(insts, dtype="<U16"),
        timeseries_list=ts_list,
        pearson_z_edges=np.vstack(z_list),
        motion_covariates=np.array(mot_list, dtype=np.float64),
        age_sex_covariates=np.array(as_list, dtype=np.float64),
        exclude_primary_050=np.array(excl_list, dtype=np.int64),
        t_lengths=np.array(tlen_list, dtype=np.int64),
        multirun_subject_ids=np.array(mr_sids, dtype="<U16"),
        multirun_institutions=np.array(mr_insts, dtype="<U16"),
        multirun_pearson_z_avg=np.vstack(mr_z),
        multirun_motion_covariates=np.array(mr_mot, dtype=np.float64),
        multirun_age_sex_covariates=np.array(mr_as, dtype=np.float64),
        multirun_run_counts=np.array(mr_cnts, dtype=np.int64),
    )
    return inputs, np.array(y_list, dtype=np.int64), np.array(mr_y, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in real-shape synthetic E2E timing benchmark.")
    parser.add_argument("--n-permutations", type=int, default=2, help="Permutation count (default: 2).")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds (default: 1200).")
    args = parser.parse_args()

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Real-shape E2E benchmark exceeded --timeout={args.timeout}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)
    try:
        t0 = time.perf_counter()
        inputs, y_r1, y_mr = build_real_shape_synthetic_inputs(n_rois=190)
        cohorts = build_cohorts(inputs, y_r1, y_mr)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "real_shape_e2e.json"
            ckpt_p = Path(tmpdir) / "real_shape_ckpt.json"
            run_exploratory_synthetic_e2e(
                cohorts=cohorts,
                output_path=out_p,
                checkpoint_path=ckpt_p,
                n_permutations=args.n_permutations,
                n_boot=20,
                seed=42,
            )
        elapsed = time.perf_counter() - t0
        print(f"Real-shape synthetic E2E (n_permutations={args.n_permutations}) completed in {elapsed:.2f}s")
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
