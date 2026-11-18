#!/usr/bin/env python3
"""
Study 2 Cohort Freeze & Attrition Audit (Counts Only — No Modeling, No Subject IDs).

Joins development-site blinded FD metrics (404 scans) with phenotype diagnosis labels
strictly for aggregate site x diagnosis cohort counts and Fisher's exact attrition tests.
NYU counts are taken strictly from the phenotype TSV without opening any NYU imaging.

Strict Guardrails:
- Never prints individual subject IDs.
- Zero NYU imaging or NYU motion data accessed.
- Zero modeling performed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import fisher_exact

DEV_SITES = ("KKI", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3")
LOIO_INSTITUTIONS = ("KKI", "NeuroIMAGE", "OHSU", "Peking")


def load_phenotype_dx(pheno_tsv: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Load binary diagnosis mapping (normalized ScanDir ID -> 'TDC' | 'ADHD' | 'MISSING')
    and aggregate NYU counts directly from the phenotype TSV only.
    """
    dx_map: Dict[str, str] = {}
    nyu_counts = {"TDC": 0, "ADHD": 0, "MISSING": 0, "TOTAL": 0}
    with open(pheno_tsv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sub_norm = row["ScanDir ID"].strip().lstrip("0") or "0"
            dx_raw = row["DX"].strip()
            if dx_raw == "0":
                label = "TDC"
            elif dx_raw in ("1", "2", "3"):
                label = "ADHD"
            else:
                label = "MISSING"
            dx_map[sub_norm] = label

            if row["Site"].strip() == "5":  # NYU
                nyu_counts[label] += 1
                nyu_counts["TOTAL"] += 1
    return dx_map, nyu_counts


def compute_cohort_tables(fd_csv: Path, pheno_tsv: Path) -> Dict[str, object]:
    dx_map, nyu_pheno = load_phenotype_dx(pheno_tsv)

    records: List[Dict[str, object]] = []
    missing_dx = 0
    with open(fd_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sub_norm = row["subject_id_norm"].strip()
            label = dx_map.get(sub_norm, "MISSING")
            if label == "MISSING":
                missing_dx += 1
            records.append(
                {
                    "site": row["site"],
                    "institution": row["institution"],
                    "dx": label,
                    "ex_primary": int(row["exclude_primary_050"]),
                    "ex_strict": int(row["exclude_strict_020"]),
                }
            )

    def summarize_group(sub: List[Dict[str, object]]) -> Dict[str, object]:
        def _counts(flag_key: str | None) -> Dict[str, int]:
            tdc_all = sum(1 for r in sub if r["dx"] == "TDC")
            adhd_all = sum(1 for r in sub if r["dx"] == "ADHD")
            if flag_key is None:
                return {"TDC": tdc_all, "ADHD": adhd_all, "Total": tdc_all + adhd_all}
            tdc_ex = sum(1 for r in sub if r["dx"] == "TDC" and r[flag_key] == 1)
            adhd_ex = sum(1 for r in sub if r["dx"] == "ADHD" and r[flag_key] == 1)
            tdc_ret = tdc_all - tdc_ex
            adhd_ret = adhd_all - adhd_ex
            odds, pval = fisher_exact([[tdc_ex, tdc_ret], [adhd_ex, adhd_ret]], alternative="two-sided")
            return {
                "TDC_ex": tdc_ex,
                "TDC_ret": tdc_ret,
                "ADHD_ex": adhd_ex,
                "ADHD_ret": adhd_ret,
                "Total_ex": tdc_ex + adhd_ex,
                "Total_ret": tdc_ret + adhd_ret,
                "TDC_ex_pct": round(100.0 * tdc_ex / tdc_all, 1) if tdc_all else 0.0,
                "ADHD_ex_pct": round(100.0 * adhd_ex / adhd_all, 1) if adhd_all else 0.0,
                "fisher_or": float(odds),
                "fisher_p": float(pval),
                "low_class_flag": (tdc_ret < 15) or (adhd_ret < 15),
            }

        return {
            "all": _counts(None),
            "primary": _counts("ex_primary"),
            "strict": _counts("ex_strict"),
        }

    groups = [*DEV_SITES, "Peking", "POOLED"]
    tables: Dict[str, Dict[str, object]] = {}
    for g in groups:
        if g == "POOLED":
            sub = records
        elif g == "Peking":
            sub = [r for r in records if r["institution"] == "Peking"]
        else:
            sub = [r for r in records if r["site"] == g]
        tables[g] = summarize_group(sub)

    return {
        "n_dev_total": len(records),
        "missing_dx": missing_dx,
        "nyu_pheno_only": nyu_pheno,
        "tables": tables,
    }


def main() -> None:
    res = compute_cohort_tables(
        Path("NeuroADHD/data/athena/motion/dev_niak_fd_metrics.csv"),
        Path("NeuroADHD/data/athena/phenotypes/adhd200_preprocessed_phenotypics.tsv"),
    )
    print(f"Development scans joined: {res['n_dev_total']} (missing DX: {res['missing_dx']})")
    nyu = res["nyu_pheno_only"]
    print(
        f"NYU (phenotype TSV only, NO imaging opened): Total={nyu['TOTAL']} "
        f"(TDC={nyu['TDC']}, ADHD={nyu['ADHD']}, {100.0*nyu['ADHD']/nyu['TOTAL']:.1f}% ADHD)\n"
    )

    print("=== ALL DEV / NO-EXCLUSION (CO-PRIMARY) ===")
    for g, t in res["tables"].items():
        a = t["all"]
        flag = "FLAG(<15)" if (a["TDC"] < 15 or a["ADHD"] < 15) else "OK"
        print(f"{g:11s} | Total={a['Total']:3d} | TDC={a['TDC']:3d} | ADHD={a['ADHD']:3d} | {flag}")

    for mode, label in [("primary", "PRIMARY EXCLUSION (>0.50mm OR >20%)"), ("strict", "STRICT EXCLUSION (>0.20mm OR >10%)")]:
        print(f"\n=== {label} ===")
        for g, t in res["tables"].items():
            m = t[mode]
            flag = "FLAG(<15 in class)" if m["low_class_flag"] else "OK"
            print(
                f"{g:11s} | Ret={m['Total_ret']:3d} (TDC={m['TDC_ret']:3d}, ADHD={m['ADHD_ret']:3d}) | "
                f"Ex={m['Total_ex']:3d} (TDC={m['TDC_ex']:3d} [{m['TDC_ex_pct']:4.1f}%], ADHD={m['ADHD_ex']:3d} [{m['ADHD_ex_pct']:4.1f}%]) | "
                f"Fisher p={m['fisher_p']:.4f} | {flag}"
            )


if __name__ == "__main__":
    main()
