#!/usr/bin/env python3
"""
Label-Free Validation of NIAK-Derived Motion Parameters Against Athena AFNI Extrema.

Compares volume-centered, volume-0-re-referenced NIAK max translation (mm) and
max rotation (deg) against the official Athena site *_motion.csv extrema across
all 404 development-site rest_1 scans.

Strict Guardrails:
- Zero diagnosis labels loaded or joined.
- Zero NYU data accessed.
- Never prints individual subject IDs.
"""

from __future__ import annotations

import csv
import re
import tarfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from scripts.compute_athena_fd import (
    DEV_NIAK_ARCHIVES,
    DEV_SITES_CC200,
    DUMMY_FRAMES_DROPPED,
    center_shift_transf,
    extract_6dof_params,
    load_dev_cc200_frame_counts,
    parse_octave_transf_text,
    read_nifti_volume_center_world,
)


def compute_stats(niak_vals: np.ndarray, athena_vals: np.ndarray) -> Dict[str, object]:
    res = spearmanr(niak_vals, athena_vals)
    rho = float(res.statistic)
    pval = float(res.pvalue)
    ratio = niak_vals / athena_vals
    med_ratio = float(np.median(ratio))
    iqr_ratio = (float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75)))
    diff = niak_vals - athena_vals
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd
    flags: List[str] = []
    if rho < 0.80:
        flags.append("rho<0.80")
    if abs(med_ratio - 1.0) > 0.25:
        flags.append("scale>25%")
    return {
        "n": int(len(niak_vals)),
        "rho": rho,
        "p": pval,
        "med_ratio": med_ratio,
        "iqr_ratio": iqr_ratio,
        "ba_bias": bias,
        "ba_lo": loa_lo,
        "ba_hi": loa_hi,
        "flags": flags,
    }


def run_validation(staging_dir: Path) -> Dict[str, object]:
    cc200_dev = load_dev_cc200_frame_counts(staging_dir / "ADHD200_CC200_TCs_filtfix.tar.gz")

    athena: Dict[tuple[str, str], Dict[str, object]] = {}
    csv_headers: Dict[str, List[str]] = {}
    for site in DEV_SITES_CC200:
        inst = "Peking" if site.startswith("Peking") else site
        with open(staging_dir / f"{site}_motion.csv", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            csv_headers[site] = [c.strip() for c in next(reader)]
            for r in reader:
                if not r or "rest_1" not in r[0]:
                    continue
                m = re.search(r"rp_(\d+)_session_1_rest_1", r[0])
                if not m:
                    continue
                sub_norm = m.group(1).lstrip("0") or "0"
                athena[(inst, sub_norm)] = {
                    "site": site,
                    "max_trans": float(r[1]),
                    "max_rot": float(r[3]),
                }

    records: List[Dict[str, object]] = []
    for inst, arc_name in DEV_NIAK_ARCHIVES.items():
        with tarfile.open(staging_dir / arc_name, "r") as tf:
            members = tf.getmembers()
            targets = {}
            for m in members:
                if m.isfile() and "motion_target" in m.name and m.name.endswith(".nii.gz"):
                    hit = re.search(r"X_(\d+)_session_1_run1", m.name)
                    if hit:
                        targets[hit.group(1).lstrip("0") or "0"] = m

            for m in members:
                if not m.isfile() or "motion_parameters" not in m.name or not m.name.endswith(".mat"):
                    continue
                hit = re.search(r"X_(\d+)_session_1_run1", m.name)
                if not hit:
                    continue
                sub = hit.group(1).lstrip("0") or "0"
                key = (inst, sub)
                if key not in cc200_dev or key not in athena:
                    continue

                cw = read_nifti_volume_center_world(tf.extractfile(targets[sub]).read())
                transf = parse_octave_transf_text(
                    tf.extractfile(m).read().decode("utf-8", errors="replace")
                )
                tr_c = center_shift_transf(transf[:, :, DUMMY_FRAMES_DROPPED:], cw)
                # Re-reference to frame 0 of retained scan (matching Athena 3dvolreg -base rm_da[0])
                m0_inv = np.linalg.inv(tr_c[:, :, 0])
                tr_c0 = np.empty_like(tr_c)
                for t in range(tr_c.shape[2]):
                    tr_c0[:, :, t] = m0_inv @ tr_c[:, :, t]
                p = extract_6dof_params(tr_c0)

                records.append(
                    {
                        "site": athena[key]["site"],
                        "inst": inst,
                        "ath_t": athena[key]["max_trans"],
                        "ath_r": athena[key]["max_rot"],
                        "niak_t_l2": float(np.max(np.linalg.norm(p[:, 3:6], axis=1))),
                        "niak_r_l2": float(np.max(np.linalg.norm(p[:, :3], axis=1))),
                        "niak_r_linf": float(np.max(np.abs(p[:, :3]))),
                    }
                )

    groups = [*DEV_SITES_CC200, "Peking_ALL", "POOLED"]
    summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    for grp in groups:
        if grp == "POOLED":
            sub = records
        elif grp == "Peking_ALL":
            sub = [r for r in records if r["inst"] == "Peking"]
        else:
            sub = [r for r in records if r["site"] == grp]
        summary[grp] = {
            "trans_l2": compute_stats(
                np.array([x["niak_t_l2"] for x in sub]),
                np.array([x["ath_t"] for x in sub]),
            ),
            "rot_l2": compute_stats(
                np.array([x["niak_r_l2"] for x in sub]),
                np.array([x["ath_r"] for x in sub]),
            ),
            "rot_linf": compute_stats(
                np.array([x["niak_r_linf"] for x in sub]),
                np.array([x["ath_r"] for x in sub]),
            ),
        }

    return {"n_matched": len(records), "headers": csv_headers, "summary": summary}


def main() -> None:
    out = run_validation(Path("NeuroADHD/data/athena/staging"))
    print(f"Matched development scans (label-free, zero NYU): {out['n_matched']} / 404\n")
    print("=== MAX TRANSLATION (mm): NIAK Volume-Centered L2 vs Athena Max Motion (mm) ===")
    for grp, metrics in out["summary"].items():
        s = metrics["trans_l2"]
        fl = ",".join(s["flags"]) if s["flags"] else "OK"
        print(
            f"{grp:11s} (N={s['n']:3d}) | rho={s['rho']:.4f} (p={s['p']:.2e}) | "
            f"med_ratio={s['med_ratio']:.4f} | BA={s['ba_bias']:+.3f} [{s['ba_lo']:+.3f}, {s['ba_hi']:+.3f}] mm | {fl}"
        )

    print("\n=== MAX ROTATION (deg): NIAK L2 vs Athena Max Rotation (degree) ===")
    for grp, metrics in out["summary"].items():
        s = metrics["rot_l2"]
        fl = ",".join(s["flags"]) if s["flags"] else "OK"
        print(
            f"{grp:11s} (N={s['n']:3d}) | rho={s['rho']:.4f} (p={s['p']:.2e}) | "
            f"med_ratio={s['med_ratio']:.4f} | BA={s['ba_bias']:+.3f} [{s['ba_lo']:+.3f}, {s['ba_hi']:+.3f}] deg | {fl}"
        )

    print("\n=== MAX ROTATION (deg): NIAK L_inf (max single axis) vs Athena Max Rotation (degree) ===")
    for grp, metrics in out["summary"].items():
        s = metrics["rot_linf"]
        fl = ",".join(s["flags"]) if s["flags"] else "OK"
        print(
            f"{grp:11s} (N={s['n']:3d}) | rho={s['rho']:.4f} (p={s['p']:.2e}) | "
            f"med_ratio={s['med_ratio']:.4f} | BA={s['ba_bias']:+.3f} [{s['ba_lo']:+.3f}, {s['ba_hi']:+.3f}] deg | {fl}"
        )


if __name__ == "__main__":
    main()
