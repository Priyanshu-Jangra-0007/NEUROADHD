#!/usr/bin/env python3
"""
Athena Derivative Inspection Tool (Read-Only)

Inspects a directory of downloaded neuroimaging derivatives and reports:
- File extensions actually present and their frequencies
- Discovered file name and directory path patterns (with numeric IDs masked)
- Total unique subjects discovered
- Matrix dimension distributions (timepoints x ROI columns)
- Motion parameter file pairing rate
- Diagnostic and institutional breakdown after joining with phenotypic records

STRICT PRIVACY GUARDRAIL:
This tool reports aggregate counts and masked templates only.
It NEVER prints individual subject identifiers.
"""

import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def mask_numeric_identifiers(text: str) -> str:
    """Mask any sequence of 4 to 8 consecutive digits with '<ID>' (handles underscores, hyphens, etc.)."""
    return re.sub(r"(?<!\d)\d{4,8}(?!\d)", "<ID>", text)


def extract_numeric_id(text: str) -> Optional[str]:
    """Extract candidate subject numeric identifier for internal dictionary pairing only."""
    m = re.search(r"(\d{4,8})", text)
    return m.group(1).lstrip("0") if m else None


def get_file_extension(p: Path) -> str:
    """Extract full file extension, accounting for compound extensions like .tar.gz."""
    name_lower = p.name.lower()
    for compound in [".tar.gz", ".tar.bz2", ".nii.gz", ".1d.gz"]:
        if name_lower.endswith(compound):
            return compound
    return p.suffix.lower() if p.suffix else "[no extension]"


def read_matrix_shape(file_path: Path) -> Optional[Tuple[int, int]]:
    """Determine number of rows and columns in a numeric data file without loading entire array into memory."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            if not lines:
                return None
            num_rows = len(lines)
            # Probe first non-empty line for columns
            first_line = lines[0]
            delims = [",", "\t", " "]
            cols = None
            for d in delims:
                parts = [p for p in first_line.split(d) if p]
                if len(parts) > 1:
                    cols = len(parts)
                    break
            if cols is None:
                cols = 1
            return (num_rows, cols)
    except Exception:
        return None


def load_phenotypes(pheno_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load phenotype tables mapping normalized subject ID -> {site, dx, dx_label}."""
    pheno_map = {}
    if not pheno_dir.is_dir():
        return pheno_map

    for p_file in pheno_dir.glob("*.csv"):
        try:
            with open(p_file, mode="r", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                id_col = None
                for candidate in ["ScanDir ID", "ScanDirID", "SubjectID", "ID", "subject_id"]:
                    if reader.fieldnames and candidate in reader.fieldnames:
                        id_col = candidate
                        break
                if not id_col:
                    continue

                dx_col = "DX" if reader.fieldnames and "DX" in reader.fieldnames else None
                site_col = "Site" if reader.fieldnames and "Site" in reader.fieldnames else None
                site_from_name = p_file.stem.split("_")[0]

                for row in reader:
                    raw_id = row.get(id_col, "").strip()
                    if not raw_id:
                        continue
                    norm_id = raw_id.lstrip("0")

                    dx_val = row.get(dx_col, "").strip() if dx_col else ""
                    site_val = row.get(site_col, "").strip() if site_col else site_from_name

                    if dx_val == "0":
                        dx_label = "TDC (Control)"
                    elif dx_val in ["1", "2", "3"]:
                        dx_label = "ADHD"
                    else:
                        dx_label = "Unreleased / Unknown"

                    pheno_map[norm_id] = {
                        "site": site_val or site_from_name,
                        "dx": dx_val,
                        "dx_label": dx_label,
                    }
        except Exception:
            continue

    return pheno_map


def inspect_athena_directory(
    data_dir: Path,
    pheno_dir: Path,
    output_stream: Optional[io.TextIOBase] = None,
) -> Dict[str, Any]:
    """
    Inspect the directory and return structured metrics dictionary.
    Prints an aggregate-only summary report to output_stream.
    """
    out = output_stream or sys.stdout

    if not data_dir.is_dir():
        out.write(f"\n[INFO] Data directory not found: {data_dir}\n")
        out.write("Inspection tool is ready for manual download folder.\n\n")
        return {"status": "DIRECTORY_NOT_FOUND", "data_dir": str(data_dir)}

    out.write("======================================================================\n")
    out.write("ATHENA DERIVATIVES INSPECTION REPORT (COUNTS & PATTERNS ONLY)\n")
    out.write("======================================================================\n")
    out.write(f"Target Directory : {data_dir.resolve()}\n")
    out.write("Privacy Protocol : Aggregate counts only (Zero Subject IDs printed)\n")
    out.write("----------------------------------------------------------------------\n")

    # 1. Collect all files and count file extensions actually present
    all_files = [p for p in data_dir.rglob("*") if p.is_file()]
    extension_counts = Counter()
    filename_patterns = Counter()
    relative_path_patterns = Counter()

    for p in all_files:
        ext = get_file_extension(p)
        extension_counts[ext] += 1
        masked_fname = mask_numeric_identifiers(p.name)
        filename_patterns[masked_fname] += 1
        try:
            rel_path = p.relative_to(data_dir)
            masked_rel = mask_numeric_identifiers(str(rel_path))
            relative_path_patterns[masked_rel] += 1
        except ValueError:
            pass

    out.write(f"\n[1] File Extensions Actually Discovered ({len(all_files)} total files):\n")
    for ext, cnt in sorted(extension_counts.items(), key=lambda x: -x[1]):
        out.write(f"    • {ext:<15} : {cnt:>6} files\n")

    out.write(f"\n[2] Discovered File Name Pattern Shapes (IDs Masked):\n")
    for pat, cnt in filename_patterns.most_common(8):
        out.write(f"    • Pattern: {pat:<45} Count: {cnt:>5}\n")

    # 2. Discern Time-Series Files vs Motion Files by Structure & Content
    subject_ts_files = defaultdict(list)
    subject_motion_files = defaultdict(list)
    time_series_shapes = Counter()

    for p in all_files:
        ext = get_file_extension(p)
        # Check text/matrix formats
        if ext in [".1d", ".csv", ".tsv", ".txt"]:
            shape = read_matrix_shape(p)
            if not shape:
                continue
            rows, cols = shape
            sid = extract_numeric_id(p.name) or extract_numeric_id(p.parent.name)
            if not sid:
                continue

            name_lower = p.name.lower()
            # Motion files typically have 6 columns or explicit 'motion'/'fd' in name
            if cols == 6 or "motion" in name_lower or "fd" in name_lower:
                subject_motion_files[sid].append((p, shape))
            elif cols >= 10 and rows >= 10:
                subject_ts_files[sid].append((p, shape))
                time_series_shapes[shape] += 1

    unique_ts_subjects = len(subject_ts_files)
    unique_motion_subjects = len(subject_motion_files)
    paired_subjects = len(set(subject_ts_files.keys()) & set(subject_motion_files.keys()))

    out.write(f"\n[3] Subject & File Discovery Counts:\n")
    out.write(f"    Total Unique Subjects with Time Series : {unique_ts_subjects}\n")
    out.write(f"    Total Unique Subjects with Motion Data : {unique_motion_subjects}\n")
    out.write(f"    Paired Subjects (Time Series + Motion) : {paired_subjects}\n")
    if unique_ts_subjects > 0:
        pairing_pct = (paired_subjects / unique_ts_subjects) * 100
        out.write(f"    Motion File Pairing Rate               : {pairing_pct:.1f}%\n")

    out.write(f"\n[4] Matrix Dimensions Across Discovered Time-Series Files:\n")
    if time_series_shapes:
        out.write(f"    Total Time Series Files Measured : {sum(time_series_shapes.values())}\n")
        out.write(f"    Distinct Matrix Dimensions (Timepoints x ROIs):\n")
        for (r, c), cnt in sorted(time_series_shapes.items(), key=lambda x: -x[1]):
            out.write(f"      • {r:>4} timepoints  x  {c:>4} ROIs  : {cnt:>4} files\n")
    else:
        out.write("    [NOTICE] No multi-column ROI time-series files found.\n")

    # 3. Phenotypic Matching & Site Breakdown
    out.write(f"\n[5] Phenotype Record Cross-Tabulation:\n")
    pheno_map = load_phenotypes(pheno_dir)
    site_dx_breakdown = defaultdict(lambda: Counter())

    matched_subjects = 0
    unmatched_subjects = 0

    for sid in subject_ts_files.keys():
        if sid in pheno_map:
            info = pheno_map[sid]
            site_dx_breakdown[info["site"]][info["dx_label"]] += 1
            matched_subjects += 1
        else:
            site_dx_breakdown["Unmatched / Unknown"]["Unmatched"] += 1
            unmatched_subjects += 1

    out.write(f"    Phenotype Match Rate : {matched_subjects}/{unique_ts_subjects} "
              f"({(matched_subjects / unique_ts_subjects * 100) if unique_ts_subjects else 0:.1f}%)\n")
    if unmatched_subjects > 0:
        out.write(f"    Unmatched Subjects   : {unmatched_subjects}\n")

    out.write("\n    Site-by-Site Diagnosis Distribution:\n")
    out.write(f"    {'Site / Institution':<22} | {'TDC (Control)':<14} | {'ADHD':<8} | {'Other/Unk':<10} | {'Total':<6}\n")
    out.write("    " + "-" * 70 + "\n")

    total_tdc = 0
    total_adhd = 0
    total_other = 0

    for site, counts in sorted(site_dx_breakdown.items()):
        tdc = counts["TDC (Control)"]
        adhd = counts["ADHD"]
        other = sum(v for k, v in counts.items() if k not in ["TDC (Control)", "ADHD"])
        tot = tdc + adhd + other

        total_tdc += tdc
        total_adhd += adhd
        total_other += other
        out.write(f"    {site:<22} | {tdc:<14} | {adhd:<8} | {other:<10} | {tot:<6}\n")

    out.write("    " + "-" * 70 + "\n")
    out.write(f"    {'TOTAL':<22} | {total_tdc:<14} | {total_adhd:<8} | {total_other:<10} | {total_tdc + total_adhd + total_other:<6}\n")
    out.write("\n======================================================================\n")
    out.write("INSPECTION COMPLETE (ALL OUTPUTS COMPLIANT WITH ZERO-ID POLICY)\n")
    out.write("======================================================================\n\n")

    return {
        "status": "SUCCESS",
        "total_files": len(all_files),
        "extension_counts": dict(extension_counts),
        "filename_patterns": dict(filename_patterns.most_common(10)),
        "unique_ts_subjects": unique_ts_subjects,
        "unique_motion_subjects": unique_motion_subjects,
        "paired_subjects": paired_subjects,
        "time_series_shapes": {f"{r}x{c}": cnt for (r, c), cnt in time_series_shapes.items()},
        "site_dx_breakdown": {s: dict(c) for s, c in site_dx_breakdown.items()},
        "total_tdc": total_tdc,
        "total_adhd": total_adhd,
        "total_other": total_other,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Athena Derivatives Inspection Tool (Read-Only, Counts Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("NeuroADHD/data/athena"),
        help="Path to folder containing downloaded Athena files (default: NeuroADHD/data/athena).",
    )
    parser.add_argument(
        "--pheno-dir",
        "-p",
        type=Path,
        default=Path("NeuroADHD/data/adhd200/raw/phenotypes"),
        help="Path to ADHD-200 phenotypic CSV directory.",
    )
    args = parser.parse_args()

    results = inspect_athena_directory(args.data_dir, args.pheno_dir)
    sys.exit(0 if results.get("status") == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
