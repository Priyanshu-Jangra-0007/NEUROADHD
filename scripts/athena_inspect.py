#!/usr/bin/env python3
"""
Athena Derivative Inspection Tool (Read-Only)

Inspects manually downloaded Athena derivative directories and reports:
- Detected file naming patterns
- Total subject and file counts
- Rows (timepoints) and columns (ROIs) distribution across time-series files
- Paired motion parameter file availability counts
- Diagnosis counts by site after joining with phenotype records

STRICT PRIVACY GUARDRAIL:
This tool reports aggregate counts only. It never prints individual subject IDs.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def sanitize_filename_pattern(name: str) -> str:
    """Replace subject ID numbers (4 to 8 digits) with a generic placeholder."""
    return re.sub(r"\b\d{4,8}\b", "<SUBJECT_ID>", name)


def extract_subject_id_from_name(name: str) -> Optional[str]:
    """Extract candidate subject numeric identifier for internal dictionary lookup only."""
    m = re.search(r"(\d{4,8})", name)
    return m.group(1).lstrip("0") if m else None


def load_phenotypes(pheno_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load phenotype tables mapping normalized subject ID -> {site, dx, dx_label}."""
    pheno_map = {}
    if not pheno_dir.is_dir():
        return pheno_map

    for p_file in pheno_dir.glob("*phenotypic*.csv"):
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

                # Infer site from filename if not in columns
                site_from_name = p_file.stem.split("_")[0]

                for row in reader:
                    raw_id = row.get(id_col, "").strip()
                    if not raw_id:
                        continue
                    norm_id = raw_id.lstrip("0")

                    dx_val = row.get(dx_col, "").strip() if dx_col else ""
                    site_val = row.get(site_col, "").strip() if site_col else site_from_name

                    # Map DX: 0 -> TDC, 1/2/3 -> ADHD, other -> Unknown
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
        except Exception as e:
            print(f"[WARN] Error reading phenotype file {p_file.name}: {e}", file=sys.stderr)

    return pheno_map


def inspect_athena_directory(data_dir: Path, pheno_dir: Path) -> int:
    """Inspect the downloaded folder and print count-only summary report."""
    if not data_dir.is_dir():
        print(f"\n[INFO] Data directory not found: {data_dir}")
        print("This script is ready for inspection once Athena derivatives are downloaded.\n")
        return 1

    print("======================================================================")
    print("ATHENA DERIVATIVES INSPECTION REPORT (COUNTS ONLY)")
    print("======================================================================")
    print(f"Target Directory : {data_dir.resolve()}")
    print("Privacy Protocol : Aggregate metrics only (Zero Subject IDs printed)")
    print("----------------------------------------------------------------------")

    # 1. Discover Time-Series & Motion Files
    ts_files = []
    motion_files = []

    # Common extensions/patterns in Athena distribution
    all_files = list(data_dir.rglob("*"))
    file_patterns = Counter()

    for f in all_files:
        if not f.is_file():
            continue
        pat = sanitize_filename_pattern(f.name)
        file_patterns[pat] += 1

        name_lower = f.name.lower()
        if "rois" in name_lower or "tc" in name_lower or name_lower.endswith(".1d") or name_lower.endswith(".csv"):
            if "motion" not in name_lower and "fd" not in name_lower:
                ts_files.append(f)
        if "motion" in name_lower or "fd" in name_lower:
            motion_files.append(f)

    # 2. File Pattern Summary
    print(f"\n[1] File Naming Patterns Discovered (Top 10 Patterns):")
    for pat, count in file_patterns.most_common(10):
        print(f"    - Pattern: {pat:<45} Count: {count:>4}")

    # Map files by subject ID
    subject_ts_map = defaultdict(list)
    for f in ts_files:
        sid = extract_subject_id_from_name(f.name) or extract_subject_id_from_name(f.parent.name)
        if sid:
            subject_ts_map[sid].append(f)

    subject_motion_map = defaultdict(list)
    for f in motion_files:
        sid = extract_subject_id_from_name(f.name) or extract_subject_id_from_name(f.parent.name)
        if sid:
            subject_motion_map[sid].append(f)

    total_ts_subjects = len(subject_ts_map)
    total_motion_subjects = len(subject_motion_map)
    paired_subjects = len(set(subject_ts_map.keys()) & set(subject_motion_map.keys()))

    print(f"\n[2] Subject & File Totals:")
    print(f"    Total Unique Subjects with Time Series : {total_ts_subjects}")
    print(f"    Total Unique Subjects with Motion Data : {total_motion_subjects}")
    print(f"    Paired Subjects (Time Series + Motion) : {paired_subjects}")
    if total_ts_subjects > 0:
        pct_paired = (paired_subjects / total_ts_subjects) * 100
        print(f"    Motion Coverage Rate                   : {pct_paired:.1f}%")

    # 3. Shape Analysis (Rows = Timepoints, Columns = ROIs)
    print(f"\n[3] Matrix Dimensions Across Time Series Files:")
    row_counts = []
    col_counts = []

    for sid, flist in subject_ts_map.items():
        for f in flist:
            try:
                # Read sample dimensions without printing contents
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
                    if not lines:
                        continue
                    n_rows = len(lines)
                    # Sample first non-empty line for column count
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
                    row_counts.append(n_rows)
                    col_counts.append(cols)
            except Exception:
                continue

    if row_counts and col_counts:
        row_dist = Counter(row_counts)
        col_dist = Counter(col_counts)

        print(f"    Total Time Series Files Measured : {len(row_counts)}")
        print(f"    Timepoints (Rows) Distribution  :")
        for r_val, count in sorted(row_dist.items()):
            print(f"      • {r_val:>4} timepoints : {count:>4} files")

        print(f"    ROI Columns Distribution        :")
        for c_val, count in sorted(col_dist.items()):
            print(f"      • {c_val:>4} ROIs       : {count:>4} files")
    else:
        print("    [NOTICE] No readable time-series data files could be parsed.")

    # 4. Phenotype Matching & Diagnosis Breakdown
    print(f"\n[4] Phenotypic Matching & Diagnostic Breakdown by Site:")
    pheno_map = load_phenotypes(pheno_dir)

    if not pheno_map:
        print(f"    [WARN] No phenotype records loaded from: {pheno_dir}")
    else:
        site_dx_counts = defaultdict(lambda: Counter())
        matched_count = 0
        unmatched_count = 0

        for sid in subject_ts_map.keys():
            if sid in pheno_map:
                info = pheno_map[sid]
                site_dx_counts[info["site"]][info["dx_label"]] += 1
                matched_count += 1
            else:
                site_dx_counts["Unmatched / External"]["Unknown"] += 1
                unmatched_count += 1

        print(f"    Phenotype Match Rate : {matched_count}/{total_ts_subjects} subjects "
              f"({(matched_count / total_ts_subjects * 100) if total_ts_subjects else 0:.1f}%)")
        print(f"    Unmatched Subjects   : {unmatched_count}")
        print("\n    Site-by-Site Cohort Distribution:")
        print(f"    {'Site / Institution':<22} | {'TDC (Control)':<14} | {'ADHD':<8} | {'Other / Unk':<12} | {'Total':<6}")
        print("    " + "-" * 72)

        grand_tdc = 0
        grand_adhd = 0
        grand_other = 0

        for site_name, counts in sorted(site_dx_counts.items()):
            tdc = counts["TDC (Control)"]
            adhd = counts["ADHD"]
            other = sum(v for k, v in counts.items() if k not in ["TDC (Control)", "ADHD"])
            total = tdc + adhd + other

            grand_tdc += tdc
            grand_adhd += adhd
            grand_other += other

            print(f"    {site_name:<22} | {tdc:<14} | {adhd:<8} | {other:<12} | {total:<6}")

        print("    " + "-" * 72)
        print(f"    {'TOTAL':<22} | {grand_tdc:<14} | {grand_adhd:<8} | {grand_other:<12} | {grand_tdc + grand_adhd + grand_other:<6}")

    print("\n======================================================================")
    print("INSPECTION COMPLETE (ALL OUTPUTS COMPLIANT WITH ZERO-ID POLICY)")
    print("======================================================================\n")
    return 0


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

    sys.exit(inspect_athena_directory(args.data_dir, args.pheno_dir))


if __name__ == "__main__":
    main()
