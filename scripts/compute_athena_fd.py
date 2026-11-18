#!/usr/bin/env python3
"""
Study 2 Development-Site Framewise Displacement (FD) Calculator (Protocol Deviation Branch b).

Computes Power et al. (2012) Framewise Displacement (50 mm sphere radius) for the 404
development-site rest_1 scans (KKI, NeuroIMAGE, OHSU, Peking 1-3) from the staged
Neuro Bureau NIAK intermediate 4x4xT rigid-body transformation matrices after
removing the first 3 frames (per docs/study2/DEVIATIONS.md Branch b).

Strict Guardrails:
- Reads ONLY development-site members (KKI, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3).
- Strictly skips NYU, WashU, and Pittsburgh (both on disk and in memory).
- Zero diagnosis labels loaded or joined.
- Never prints individual subject IDs to stdout or logs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import struct
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEV_SITES_CC200 = ("KKI", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3")

DEV_NIAK_ARCHIVES = {
    "KKI": "intermediate_kki.tar",
    "NeuroIMAGE": "intermediate_neuroimage.tar",
    "OHSU": "intermediate_ohsu.tar",
    "Peking": "intermediate_peking.tar",
}

HEAD_RADIUS_MM = 50.0
DUMMY_FRAMES_DROPPED = 3


def read_nifti_volume_center_world(nii_gz_bytes: bytes) -> np.ndarray:
    """
    Read a NIfTI-1 .nii.gz header and return the 3D world coordinate (mm) of the
    volume geometric center [(nx-1)/2, (ny-1)/2, (nz-1)/2] via the sform affine.
    """
    raw = gzip.decompress(nii_gz_bytes)
    if len(raw) < 348:
        raise ValueError(f"Invalid NIfTI-1 header length: {len(raw)}")
    dim = struct.unpack("<8h", raw[40:56])
    nx, ny, nz = dim[1], dim[2], dim[3]
    sform_code = struct.unpack("<h", raw[254:256])[0]
    if sform_code <= 0:
        raise ValueError(f"Expected positive sform_code in NIfTI header, got {sform_code}")
    srow_x = struct.unpack("<4f", raw[280:296])
    srow_y = struct.unpack("<4f", raw[296:312])
    srow_z = struct.unpack("<4f", raw[312:328])
    sform = np.array([srow_x, srow_y, srow_z, [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    center_vox = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0, 1.0], dtype=np.float64)
    return (sform @ center_vox)[:3]


def center_shift_transf(transf: np.ndarray, center_mm: np.ndarray) -> np.ndarray:
    """
    Conjugate a (4, 4, T) world-space affine series to the volume center c:
        M_centered(t) = T(-c) @ M(t) @ T(+c)
    This leaves rotation R(t) unchanged and extracts physical brain-center translation
    t_center(t) = M(t) c - c, removing spurious (I - R) c lever-arm coupling when
    the NIfTI world origin (0,0,0) does not coincide with the volume center.
    """
    if transf.ndim != 3 or transf.shape[:2] != (4, 4):
        raise ValueError(f"Expected shape (4, 4, T), got {transf.shape}")
    c = np.asarray(center_mm, dtype=np.float64).reshape(3)
    t_pos = np.eye(4, dtype=np.float64)
    t_pos[:3, 3] = c
    t_neg = np.eye(4, dtype=np.float64)
    t_neg[:3, 3] = -c
    out = np.empty_like(transf, dtype=np.float64)
    for t in range(transf.shape[2]):
        out[:, :, t] = t_neg @ transf[:, :, t] @ t_pos
    return out


def parse_octave_transf_text(text: str) -> np.ndarray:
    """Parse an Octave 3D 4x4xT matrix file ('transf') into a numpy array of shape (4, 4, T)."""
    lines = text.splitlines()
    dim_line = None
    data_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("4 4 "):
            dim_line = stripped
            data_start = idx + 1
            break
    if dim_line is None or data_start is None:
        raise ValueError("Could not locate '4 4 <T>' dimension header in Octave matrix file.")

    dims = [int(x) for x in dim_line.split()]
    if len(dims) != 3 or dims[0] != 4 or dims[1] != 4 or dims[2] <= 0:
        raise ValueError(f"Unexpected matrix dimensions: {dims}")

    raw_vals = [float(x.strip()) for x in lines[data_start:] if x.strip()]
    expected_n = 4 * 4 * dims[2]
    if len(raw_vals) != expected_n:
        raise ValueError(f"Expected {expected_n} matrix values, found {len(raw_vals)}")

    # Octave stores multi-dimensional arrays in column-major (Fortran) order.
    return np.array(raw_vals, dtype=np.float64).reshape(tuple(dims), order="F")


def extract_6dof_params(transf: np.ndarray) -> np.ndarray:
    """
    Decompose a (4, 4, T) array of rigid-body SE(3) transformation matrices into (T, 6):
    columns = [roll_deg, pitch_deg, yaw_deg, tx_mm, ty_mm, tz_mm].
    Validates rigid-body properties (det(R) ~ 1, R^T R ~ I) for every frame.
    """
    if transf.ndim != 3 or transf.shape[:2] != (4, 4):
        raise ValueError(f"Expected shape (4, 4, T), got {transf.shape}")

    n_frames = transf.shape[2]
    params = np.zeros((n_frames, 6), dtype=np.float64)

    for t in range(n_frames):
        mat = transf[:, :, t]
        rot = mat[:3, :3]
        det = float(np.linalg.det(rot))
        if abs(det - 1.0) > 1e-4:
            raise ValueError(f"Non-rigid rotation matrix at frame {t}: det={det:.6f}")
        ortho_err = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
        if ortho_err > 1e-4:
            raise ValueError(f"Non-orthogonal rotation matrix at frame {t}: err={ortho_err:.2e}")

        # Standard Tait-Bryan Z-Y-X decomposition (radians -> degrees)
        pitch = math.asin(float(np.clip(-rot[2, 0], -1.0, 1.0)))
        if math.cos(pitch) > 1e-6:
            roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
            yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        else:
            roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
            yaw = 0.0

        params[t, 0] = math.degrees(roll)
        params[t, 1] = math.degrees(pitch)
        params[t, 2] = math.degrees(yaw)
        params[t, 3] = mat[0, 3]
        params[t, 4] = mat[1, 3]
        params[t, 5] = mat[2, 3]

    return params


def compute_power_fd(params_6dof: np.ndarray, radius_mm: float = HEAD_RADIUS_MM) -> np.ndarray:
    """
    Compute Power et al. (2012) Framewise Displacement from (T, 6) parameters:
    columns = [roll_deg, pitch_deg, yaw_deg, tx_mm, ty_mm, tz_mm].
    Sets FD[0] = 0.0 per PREREGISTRATION.md Section 5.3.
    """
    if params_6dof.ndim != 2 or params_6dof.shape[1] != 6:
        raise ValueError(f"Expected shape (T, 6), got {params_6dof.shape}")
    n_frames = params_6dof.shape[0]
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames to compute FD, got {n_frames}")

    deg_to_mm = (math.pi / 180.0) * radius_mm
    rot_mm = params_6dof[:, :3] * deg_to_mm
    trans_mm = params_6dof[:, 3:6]

    diffs_rot = np.abs(np.diff(rot_mm, axis=0))
    diffs_trans = np.abs(np.diff(trans_mm, axis=0))

    fd = np.zeros(n_frames, dtype=np.float64)
    fd[1:] = np.sum(diffs_trans, axis=1) + np.sum(diffs_rot, axis=1)
    return fd


def load_dev_cc200_frame_counts(cc200_archive: Path) -> Dict[Tuple[str, str], Tuple[str, int]]:
    """
    Read ONLY development-site filtered rest_1 CC200 members from the archive.
    Strictly skips NYU, WashU, Pittsburgh, and all non-development members.
    Returns mapping: (institution_group, normalized_sub_id) -> (exact_site_name, n_frames).
    """
    dev_map: Dict[Tuple[str, str], Tuple[str, int]] = {}
    with tarfile.open(cc200_archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            site_dir = parts[0]
            # Strict development-site filter: skip NYU, WashU, Pittsburgh completely
            if site_dir not in DEV_SITES_CC200:
                continue
            filename = parts[-1]
            if not (filename.startswith("sfnwmrda") and "rest_1" in filename and filename.endswith(".1D")):
                continue

            sub_norm = parts[1].lstrip("0") or "0"
            inst_group = "Peking" if site_dir.startswith("Peking") else site_dir

            stream = tf.extractfile(member)
            if stream is None:
                continue
            n_frames = sum(1 for line in stream if line.strip())
            dev_map[(inst_group, sub_norm)] = (site_dir, n_frames)

    return dev_map


def compute_development_fd(
    staging_dir: Path,
    output_csv: Path | None = None,
    drop_frames: int = DUMMY_FRAMES_DROPPED,
) -> Dict[str, object]:
    """
    Compute NIAK-derived Power et al. (2012) FD (Branch b fallback, first 3 frames removed)
    for all 404 development-site rest_1 scans without joining any diagnosis labels.
    """
    cc200_archive = staging_dir / "ADHD200_CC200_TCs_filtfix.tar.gz"
    if not cc200_archive.is_file():
        raise FileNotFoundError(f"CC200 archive not found: {cc200_archive}")

    cc200_dev = load_dev_cc200_frame_counts(cc200_archive)

    rows: List[Dict[str, object]] = []
    frame_mismatches = 0

    for inst_group, arc_name in DEV_NIAK_ARCHIVES.items():
        arc_path = staging_dir / arc_name
        if not arc_path.is_file():
            raise FileNotFoundError(f"NIAK intermediate archive not found: {arc_path}")

        with tarfile.open(arc_path, "r") as tf:
            members = tf.getmembers()
            target_members: Dict[str, tarfile.TarInfo] = {}
            for member in members:
                if (
                    member.isfile()
                    and "motion_target" in member.name
                    and member.name.endswith(".nii.gz")
                ):
                    t_match = re.search(r"X_(\d+)_session_1_run1", member.name)
                    if t_match:
                        target_members[t_match.group(1).lstrip("0") or "0"] = member

            for member in members:
                if not member.isfile():
                    continue
                if "motion_parameters" not in member.name or not member.name.endswith(".mat"):
                    continue
                match = re.search(r"X_(\d+)_session_(\d+)_run(\d+)", member.name)
                if not match:
                    continue
                raw_id, session_num, run_num = match.group(1), match.group(2), match.group(3)
                if session_num != "1" or run_num != "1":
                    continue

                sub_norm = raw_id.lstrip("0") or "0"
                key = (inst_group, sub_norm)
                if key not in cc200_dev:
                    continue
                if sub_norm not in target_members:
                    raise FileNotFoundError(
                        f"Missing motion_target NIfTI for {inst_group} session_1_run1"
                    )

                exact_site, cc200_frames = cc200_dev[key]
                t_stream = tf.extractfile(target_members[sub_norm])
                stream = tf.extractfile(member)
                if stream is None or t_stream is None:
                    continue
                center_world = read_nifti_volume_center_world(t_stream.read())
                transf = parse_octave_transf_text(stream.read().decode("utf-8", errors="replace"))
                n_raw = transf.shape[2]

                if n_raw <= drop_frames:
                    raise ValueError(f"Too few frames ({n_raw}) to drop {drop_frames}")

                transf_centered = center_shift_transf(transf[:, :, drop_frames:], center_world)
                n_retained = transf_centered.shape[2]
                if n_retained != cc200_frames:
                    frame_mismatches += 1

                params = extract_6dof_params(transf_centered)
                fd = compute_power_fd(params, radius_mm=HEAD_RADIUS_MM)

                mean_fd = float(np.mean(fd))
                max_fd = float(np.max(fd))
                # Relative translation from first retained frame (Euclidean mm)
                rel_trans = params[:, 3:6] - params[0, 3:6]
                max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))

                pct_gt_050 = float(100.0 * np.mean(fd > 0.50))
                pct_gt_020 = float(100.0 * np.mean(fd > 0.20))

                # Also check denominator T-1 (excluding FD_0 = 0) for sensitivity transparency
                mean_fd_tm1 = float(np.mean(fd[1:]))
                pct_gt_050_tm1 = float(100.0 * np.mean(fd[1:] > 0.50))
                pct_gt_020_tm1 = float(100.0 * np.mean(fd[1:] > 0.20))

                ex_050_mean = mean_fd > 0.50
                ex_050_pct = pct_gt_050 > 20.0
                ex_050_any = ex_050_mean or ex_050_pct

                ex_020_mean = mean_fd > 0.20
                ex_020_pct = pct_gt_020 > 10.0
                ex_020_any = ex_020_mean or ex_020_pct

                ex_050_any_tm1 = (mean_fd_tm1 > 0.50) or (pct_gt_050_tm1 > 20.0)
                ex_020_any_tm1 = (mean_fd_tm1 > 0.20) or (pct_gt_020_tm1 > 10.0)

                rows.append(
                    {
                        "subject_id_norm": sub_norm,
                        "institution": inst_group,
                        "site": exact_site,
                        "session": f"session_{session_num}",
                        "run": f"rest_{run_num}",
                        "motion_source": "NIAK_intermediate_drop3_deviation",
                        "frames_niak_raw": n_raw,
                        "frames_dropped": drop_frames,
                        "frames_retained": n_retained,
                        "frames_cc200": cc200_frames,
                        "mean_fd_mm": round(mean_fd, 6),
                        "max_fd_mm": round(max_fd, 6),
                        "max_disp_mm": round(max_disp_mm, 6),
                        "pct_frames_gt_050": round(pct_gt_050, 4),
                        "pct_frames_gt_020": round(pct_gt_020, 4),
                        "exceed_mean_050": int(ex_050_mean),
                        "exceed_pct20_050": int(ex_050_pct),
                        "exclude_primary_050": int(ex_050_any),
                        "exceed_mean_020": int(ex_020_mean),
                        "exceed_pct10_020": int(ex_020_pct),
                        "exclude_strict_020": int(ex_020_any),
                        "exclude_primary_050_tm1": int(ex_050_any_tm1),
                        "exclude_strict_020_tm1": int(ex_020_any_tm1),
                    }
                )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Build aggregate-only summary (NO subject IDs, NO diagnosis labels)
    by_site: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "n_scans": 0,
            "exceed_mean_050": 0,
            "exceed_pct20_050": 0,
            "exclude_primary_050": 0,
            "retained_primary_050": 0,
            "exceed_mean_020": 0,
            "exceed_pct10_020": 0,
            "exclude_strict_020": 0,
            "retained_strict_020": 0,
            "exclude_primary_050_tm1": 0,
            "exclude_strict_020_tm1": 0,
        }
    )

    for r in rows:
        for bucket in (str(r["site"]), "TOTAL"):
            s = by_site[bucket]
            s["n_scans"] += 1
            s["exceed_mean_050"] += int(r["exceed_mean_050"])
            s["exceed_pct20_050"] += int(r["exceed_pct20_050"])
            s["exclude_primary_050"] += int(r["exclude_primary_050"])
            s["retained_primary_050"] += 1 - int(r["exclude_primary_050"])
            s["exceed_mean_020"] += int(r["exceed_mean_020"])
            s["exceed_pct10_020"] += int(r["exceed_pct10_020"])
            s["exclude_strict_020"] += int(r["exclude_strict_020"])
            s["retained_strict_020"] += 1 - int(r["exclude_strict_020"])
            s["exclude_primary_050_tm1"] += int(r["exclude_primary_050_tm1"])
            s["exclude_strict_020_tm1"] += int(r["exclude_strict_020_tm1"])

    return {
        "total_cc200_dev_scans": len(cc200_dev),
        "total_fd_computed": len(rows),
        "frame_count_mismatches": frame_mismatches,
        "by_site": dict(by_site),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Power et al. (2012) FD on development scans (no diagnosis labels, no NYU)."
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("NeuroADHD/data/athena/staging"),
        help="Path to Athena staging directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("NeuroADHD/data/athena/motion/dev_niak_fd_metrics.csv"),
        help="Output CSV path (Git-ignored under **/athena/).",
    )
    args = parser.parse_args()

    summary = compute_development_fd(args.staging_dir, args.output_csv)
    print(f"Development CC200 scans matched: {summary['total_fd_computed']} / {summary['total_cc200_dev_scans']}")
    print(f"Frame-count mismatches after dropping {DUMMY_FRAMES_DROPPED} frames: {summary['frame_count_mismatches']}")
    print("Source: NIAK intermediate 4x4xT matrices with first 3 frames removed (DEVIATIONS.md Branch b)")
    print("Diagnosis labels joined: NONE | NYU scans touched: 0\n")

    header = (
        f"{'Site':<12} | {'N':>4} | {'>0.50m':>6} | {'>20%':>5} | {'PrimEx':>6} | {'PrimRet':>7} | "
        f"{'>0.20m':>6} | {'>10%':>5} | {'StrEx':>5} | {'StrRet':>6}"
    )
    print(header)
    print("-" * len(header))
    for site in (*DEV_SITES_CC200, "TOTAL"):
        s = summary["by_site"][site]
        print(
            f"{site:<12} | {s['n_scans']:>4} | {s['exceed_mean_050']:>6} | {s['exceed_pct20_050']:>5} | "
            f"{s['exclude_primary_050']:>6} | {s['retained_primary_050']:>7} | "
            f"{s['exceed_mean_020']:>6} | {s['exceed_pct10_020']:>5} | "
            f"{s['exclude_strict_020']:>5} | {s['retained_strict_020']:>6}"
        )

    tot = summary["by_site"]["TOTAL"]
    print(
        f"\nDenominator sanity check (T vs T-1 excluding FD_0=0): "
        f"Primary exclusions T={tot['exclude_primary_050']} vs T-1={tot['exclude_primary_050_tm1']} | "
        f"Strict exclusions T={tot['exclude_strict_020']} vs T-1={tot['exclude_strict_020_tm1']}"
    )


if __name__ == "__main__":
    main()
