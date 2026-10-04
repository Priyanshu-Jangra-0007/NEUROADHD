"""Create deterministic subject-disjoint, whole-site CC200 split manifests."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/splits/cc200_site_holdout.json"
FEATURE_TABLE = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
SPLIT_DIR = PROJECT_ROOT / "data/adhd200/processed/splits"
REPORT_JSON = PROJECT_ROOT / "data/adhd200/interim/cc200_split_validation_report.json"
REPORT_MD = PROJECT_ROOT / "data/adhd200/interim/cc200_split_validation_report.md"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    table = pd.read_csv(FEATURE_TABLE, dtype={"subject_id": str, "site": str})
    if table["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs in feature matrix")
    if len(table) != config["expected_subjects"]:
        raise ValueError(f"Expected {config['expected_subjects']} rows, found {len(table)}")
    if table["diagnosis"].value_counts().to_dict() != {0: 103, 1: 50}:
        raise ValueError("Feature matrix diagnosis counts differ from the validated cohort")

    assignments = {}
    site_partition = {}
    for partition, sites in config["partitions"].items():
        for site in sites:
            if site in site_partition:
                raise ValueError(f"Site {site} assigned to both {site_partition[site]} and {partition}")
            site_partition[site] = partition
    observed_sites = set(table["site"].unique())
    if observed_sites != set(site_partition):
        raise ValueError(f"Site assignment mismatch; observed={sorted(observed_sites)}, configured={sorted(site_partition)}")
    for row in table.itertuples(index=False):
        assignments[row.subject_id] = site_partition[row.site]
    if len(assignments) != len(table):
        raise ValueError("Not every subject has exactly one partition assignment")

    manifest = table[["subject_id", "site", "diagnosis"]].copy()
    manifest["partition"] = manifest["subject_id"].map(assignments)
    order = {"train": 0, "validation": 1, "test": 2}
    manifest["_order"] = manifest["partition"].map(order)
    manifest = manifest.sort_values(["_order", "site", "subject_id"]).drop(columns="_order")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(SPLIT_DIR / "subject_split_manifest.csv", index=False)
    for partition in ("train", "validation", "test"):
        manifest.loc[manifest["partition"].eq(partition), ["subject_id"]].to_csv(
            SPLIT_DIR / f"{partition}_subjects.csv", index=False
        )

    ids = {p: set(manifest.loc[manifest.partition.eq(p), "subject_id"]) for p in order}
    overlaps = {
        "train_validation": sorted(ids["train"] & ids["validation"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "validation_test": sorted(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Subject overlap detected: {overlaps}")
    if set.union(*ids.values()) != set(table["subject_id"]):
        raise AssertionError("Some subjects were omitted from the split manifests")

    distributions = {}
    for partition in ("train", "validation", "test"):
        subset = manifest.loc[manifest.partition.eq(partition)]
        distributions[partition] = {
            "subjects": int(len(subset)),
            "diagnosis_counts": {
                "Control_0": int((subset.diagnosis == 0).sum()),
                "ADHD_1": int((subset.diagnosis == 1).sum()),
            },
            "site_counts": {k: int(v) for k, v in sorted(Counter(subset.site).items())},
            "site_list": sorted(subset.site.unique().tolist()),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": config["strategy"],
        "config": str(CONFIG.relative_to(PROJECT_ROOT)),
        "manifest": str((SPLIT_DIR / "subject_split_manifest.csv").relative_to(PROJECT_ROOT)),
        "subjects_total": int(len(manifest)),
        "class_counts_total": {"Control_0": int((manifest.diagnosis == 0).sum()), "ADHD_1": int((manifest.diagnosis == 1).sum())},
        "partitions": distributions,
        "subject_overlap": overlaps,
        "all_subjects_assigned_once": len(set.union(*ids.values())) == len(manifest) and not any(overlaps.values()),
        "site_overlap": {"train_validation": [], "train_test": [], "validation_test": []},
        "preprocessing_fitted": False,
        "test_data_used_to_fit_or_select_preprocessing": False,
        "model_training_performed": False,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        "# CC200 split validation report", "",
        f"Strategy: `{config['strategy']}`; partitions are assigned by whole acquisition site.",
        "", "| Partition | Subjects | ADHD (1) | Control (0) | Sites |", "|---|---:|---:|---:|---|",
    ]
    for partition in ("train", "validation", "test"):
        d = distributions[partition]
        md.append(f"| {partition.title()} | {d['subjects']} | {d['diagnosis_counts']['ADHD_1']} | {d['diagnosis_counts']['Control_0']} | {', '.join(d['site_list'])} |")
    md += [
        "", "## Leakage checks", "",
        f"- Every subject assigned exactly once: **{report['all_subjects_assigned_once']}**.",
        f"- Pairwise subject overlaps: `{json.dumps(overlaps, sort_keys=True)}`.",
        "- Sites are disjoint across train, validation, and test.",
        "- Preprocessing pipeline is defined but unfitted; no test information was used to fit transforms.",
        "- No model was trained and no accuracy was calculated.", "",
        "## Preprocessing plan", "",
        "Within each training fold: median imputation (if needed), remove zero-variance features, standardize, then select the top 100 features by ANOVA F-statistic. These learned steps must stay inside a complete estimator Pipeline during future cross-validation/tuning. No preprocessing was fit while creating these split manifests.", "",
        "## Site-holdout rationale", "",
        "NeuroIMAGE is the validation site (19 subjects; both classes) and Peking_2 is the final test site (20 subjects; balanced classes). The other six sites form training data. This provides a site-disjoint final evaluation while keeping both classes represented in validation and test. OHSU and Pittsburgh are control-only in this cohort and remain in training.", ""
    ]
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
