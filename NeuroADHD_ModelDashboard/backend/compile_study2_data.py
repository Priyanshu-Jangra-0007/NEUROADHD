#!/usr/bin/env python3
"""
Compile Study 2 aggregate results into frontend JSON strictly by parsing the tracked
Study 2 result summaries in docs/study2/ (no typed numbers, no raw data access,
zero subject IDs).

Sources parsed (read-only):
- docs/study2/DEV_RESULTS_2026-11-18.md
- docs/study2/SEALED_RESULTS_2026-11-18.md
- docs/study2/OSF_RECORD.md
- docs/study2/PREREGISTRATION.md (Section 8.2 Inconclusive Zone text)

Output:
- NeuroADHD_ModelDashboard/frontend/src/data/study2Data.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DOCS_STUDY2_DIR = WORKSPACE_ROOT / "docs" / "study2"
FRONTEND_DATA_DIR = SCRIPT_DIR.parent / "frontend" / "src" / "data"


def _require_match(pattern: str, text: str, label: str, flags: int = 0) -> re.Match[str]:
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"Failed to parse {label} with pattern: {pattern}")
    return m


def _parse_ci_pair(ci_str: str) -> Tuple[float, float]:
    m = _require_match(r"\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]", ci_str, "CI pair")
    return float(m.group(1)), float(m.group(2))


def _parse_dev_section(section_text: str, label: str) -> Dict[str, Any]:
    n_match = _require_match(r"Cohort:\s*(\d+)\s*scans", section_text, f"{label} cohort N")
    h1 = _require_match(
        r"\*\*H1: fMRI Pooled LOIO ROC-AUC\*\*\s*\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*\*\*`(\[[^\]]+\])`\*\*\s*\|\s*\*\*`p_perm = (\d+\.\d+)`\*\*",
        section_text,
        f"{label} H1",
    )
    motion = _require_match(
        r"\*\*Motion-Only Baseline Pooled LOIO AUC\*\*\s*\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*`(\[[^\]]+\])`",
        section_text,
        f"{label} motion baseline",
    )
    agesex = _require_match(
        r"\*\*Age & Sex Baseline Pooled LOIO AUC\*\*\s*\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*`(\[[^\]]+\])`",
        section_text,
        f"{label} age/sex baseline",
    )
    h2 = _require_match(
        r"\*\*H2:[^\n|]+\|\s*\*\*([+-]?\d+\.\d+)\*\*\s*\(`([+-]?\d+\.\d+)`\)\s*\|\s*\*\*`(\[[^\]]+\])`\*\*",
        section_text,
        f"{label} H2 delta_motion",
    )
    h3 = _require_match(
        r"\*\*H3:[^\n|]+\|\s*\*\*([+-]?\d+\.\d+)\*\*\s*\(`([+-]?\d+\.\d+)`\)\s*\|\s*\*\*`(\[[^\]]+\])`\*\*",
        section_text,
        f"{label} H3 delta_age_sex",
    )
    sec81 = _require_match(
        r"meets_dev_success_criteria\s*=\s*(True|False)",
        section_text,
        f"{label} Section 8.1 status",
    )

    h1_ci = _parse_ci_pair(h1.group(3))
    motion_ci = _parse_ci_pair(motion.group(3))
    agesex_ci = _parse_ci_pair(agesex.group(3))
    h2_ci = _parse_ci_pair(h2.group(3))
    h3_ci = _parse_ci_pair(h3.group(3))

    return {
        "n_scans": int(n_match.group(1)),
        "fmri_auc": float(h1.group(2)),
        "fmri_auc_display": h1.group(1),
        "fmri_ci_95": [h1_ci[0], h1_ci[1]],
        "fmri_ci_display": h1.group(3),
        "p_perm": float(h1.group(4)),
        "p_perm_display": h1.group(4),
        "motion_baseline_auc": float(motion.group(2)),
        "motion_baseline_auc_display": motion.group(1),
        "motion_baseline_ci_95": [motion_ci[0], motion_ci[1]],
        "motion_baseline_ci_display": motion.group(3),
        "age_sex_baseline_auc": float(agesex.group(2)),
        "age_sex_baseline_auc_display": agesex.group(1),
        "age_sex_baseline_ci_95": [agesex_ci[0], agesex_ci[1]],
        "age_sex_baseline_ci_display": agesex.group(3),
        "delta_vs_motion": float(h2.group(2)),
        "delta_vs_motion_display": h2.group(1),
        "delta_vs_motion_ci_95": [h2_ci[0], h2_ci[1]],
        "delta_vs_motion_ci_display": h2.group(3),
        "delta_vs_age_sex": float(h3.group(2)),
        "delta_vs_age_sex_display": h3.group(1),
        "delta_vs_age_sex_ci_95": [h3_ci[0], h3_ci[1]],
        "delta_vs_age_sex_ci_display": h3.group(3),
        "meets_section_8_1": sec81.group(1) == "True",
    }


def _parse_sealed_section(section_text: str, label: str) -> Dict[str, Any]:
    n_match = _require_match(
        r"NYU[^(\n]*\(\$N\s*=\s*(\d+)\$:\s*(\d+)\s*TDC\s*/\s*(\d+)\s*ADHD\)",
        section_text,
        f"{label} NYU N",
    )
    fmri = _require_match(
        r"\*\*Locked fMRI Pipeline[^\n|]+\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*\*\*`\[[^\]]+\]`\*\*\s*\(`(\[[^\]]+\])`\)\s*\|\s*\*\*`([^`]+)`\*\*",
        section_text,
        f"{label} fMRI",
    )
    motion = _require_match(
        r"\*\*Motion-Only Baseline[^\n|]+\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*`\[[^\]]+\]`\s*\(`(\[[^\]]+\])`\)",
        section_text,
        f"{label} motion baseline",
    )
    agesex = _require_match(
        r"\*\*Age & Sex Baseline[^\n|]+\|\s*\*\*(\d+\.\d+)\*\*\s*\(`(\d+\.\d+)`\)\s*\|\s*`\[[^\]]+\]`\s*\(`(\[[^\]]+\])`\)",
        section_text,
        f"{label} age/sex baseline",
    )

    fmri_ci = _parse_ci_pair(fmri.group(3))
    motion_ci = _parse_ci_pair(motion.group(3))
    agesex_ci = _parse_ci_pair(agesex.group(3))

    return {
        "n_scans": int(n_match.group(1)),
        "n_tdc": int(n_match.group(2)),
        "n_adhd": int(n_match.group(3)),
        "fmri_auc": float(fmri.group(2)),
        "fmri_auc_display": fmri.group(1),
        "fmri_ci_95_exact": [fmri_ci[0], fmri_ci[1]],
        "fmri_ci_95_exact_display": fmri.group(3),
        "outcome_zone": fmri.group(4),
        "outcome_zone_label": "Inconclusive Zone" if fmri.group(4) == "inconclusive" else fmri.group(4),
        "motion_baseline_auc": float(motion.group(2)),
        "motion_baseline_auc_display": motion.group(1),
        "motion_baseline_ci_95_exact": [motion_ci[0], motion_ci[1]],
        "motion_baseline_ci_95_exact_display": motion.group(3),
        "age_sex_baseline_auc": float(agesex.group(2)),
        "age_sex_baseline_auc_display": agesex.group(1),
        "age_sex_baseline_ci_95_exact": [agesex_ci[0], agesex_ci[1]],
        "age_sex_baseline_ci_95_exact_display": agesex.group(3),
    }


def compile_study2_data(docs_dir: Path = DOCS_STUDY2_DIR) -> Dict[str, Any]:
    dev_md = (docs_dir / "DEV_RESULTS_2026-11-18.md").read_text(encoding="utf-8")
    sealed_md = (docs_dir / "SEALED_RESULTS_2026-11-18.md").read_text(encoding="utf-8")
    cohort_md = (docs_dir / "COHORT_COUNTS_2026-11-18.md").read_text(encoding="utf-8")
    osf_md = (docs_dir / "OSF_RECORD.md").read_text(encoding="utf-8")
    prereg_md = (docs_dir / "PREREGISTRATION.md").read_text(encoding="utf-8")

    # Split DEV_RESULTS into Section 2 (primary_368) and Section 3 (no_exclusion_404)
    sec2_start = dev_md.index("## 2. Primary Analysis")
    sec3_start = dev_md.index("## 3. Registered Sensitivity Analysis")
    sec4_start = dev_md.index("## 4. Reading Under the Registered Rules")
    dev_primary = _parse_dev_section(dev_md[sec2_start:sec3_start], "dev primary_368")
    dev_sensitivity = _parse_dev_section(dev_md[sec3_start:sec4_start], "dev no_exclusion_404")

    # Split SEALED_RESULTS into Section 3 (primary_sealed) and Section 4 (sensitivity_no_exclusion)
    s_sec3 = sealed_md.index("## 3. Primary Sealed Analysis")
    s_sec4 = sealed_md.index("## 4. Sensitivity Sealed Analysis")
    sealed_primary = _parse_sealed_section(sealed_md[s_sec3:s_sec4], "sealed primary_196")
    sealed_sensitivity = _parse_sealed_section(sealed_md[s_sec4:], "sealed sensitivity_216")

    # Parse primary development differential motion exclusion rates from COHORT_COUNTS_2026-11-18.md
    sec32_start = cohort_md.index("### 3.2. Primary Motion Exclusion")
    sec33_start = cohort_md.index("### 3.3. Strict Motion Exclusion")
    dev_excl = _require_match(
        r"\*\*POOLED DEV\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\s*\(([\d.]+%)\)\*\*\s*\|\s*\*\*(\d+)\s*\(([\d.]+%)\)\*\*\s*\|\s*\*\*`(\d+\.\d+)`\*\*",
        cohort_md[sec32_start:sec33_start],
        "Development primary attrition",
    )

    # Parse NYU attrition counts & rates from Section 2 of SEALED_RESULTS
    nyu_excl = _require_match(
        r"\*\*Excluded by Section 5\.3 Primary FD Rule\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\(([\d.]+%)\)\s*\|\s*\*\*(\d+)\*\*\s*\(([\d.]+%)\)\s*\|\s*Two-sided Fisher exact \$p = (\d+\.\d+)\$",
        sealed_md,
        "NYU attrition",
    )

    # Parse OSF links, commit, SHA-256, and Study 1 overlap counts from OSF_RECORD.md
    osf_primary_url = _require_match(r"(https://osf\.io/grbt2)", osf_md, "OSF grbt2 URL").group(1)
    osf_open_url = _require_match(r"(https://osf\.io/db7nv)", osf_md, "OSF db7nv URL").group(1)
    commit_hash = _require_match(r"\*\*Source Commit:\*\*\s*`([0-9a-f]{40})`", osf_md, "Source commit").group(1)
    protocol_sha = _require_match(
        r"\*\*Locked Protocol SHA-256[^\n]*\n`([0-9a-f]{64})`",
        osf_md,
        "Protocol SHA-256",
    ).group(1)
    dev_overlap = int(_require_match(r"\*\*(\d+)\s+development subjects\*\*", osf_md, "Dev overlap").group(1))
    nyu_overlap = int(_require_match(r"\*\*(\d+)\s+NYU subjects\*\*", osf_md, "NYU overlap").group(1))

    # Extract verbatim Section 8.2 Inconclusive Zone scientific conclusion from PREREGISTRATION.md
    inc_row = _require_match(
        r"\|\s*\*\*Inconclusive Zone\*\*\s*\|\s*([^|]+)\|\s*([^|]+)\|",
        prereg_md,
        "Section 8.2 Inconclusive Zone",
    )
    inc_conclusion_md = inc_row.group(2).strip()
    inc_conclusion_plain = (
        inc_conclusion_md.replace("**", "")
        .replace("$\\pm 0.07$", "±0.07")
        .replace("$N \\approx 216$", "N ≈ 216")
        .replace("$0.50$", "0.50")
    )

    return {
        "source_files": [
            "docs/study2/DEV_RESULTS_2026-11-18.md",
            "docs/study2/SEALED_RESULTS_2026-11-18.md",
            "docs/study2/COHORT_COUNTS_2026-11-18.md",
            "docs/study2/OSF_RECORD.md",
            "docs/study2/PREREGISTRATION.md",
        ],
        "preregistration": {
            "osf_primary_url": osf_primary_url,
            "osf_open_ended_url": osf_open_url,
            "commit_sha": commit_hash,
            "commit_short": commit_hash[:7],
            "protocol_sha256": protocol_sha,
            "sealed_site_status": "Sealed NYU site has been scored once and is used up; any further modeling is exploratory.",
        },
        "prior_exposure": {
            "dev_overlap_subjects": dev_overlap,
            "nyu_overlap_subjects": nyu_overlap,
        },
        "development_primary": dev_primary,
        "development_primary_attrition": {
            "retained_total": int(dev_excl.group(1)),
            "retained_tdc": int(dev_excl.group(2)),
            "retained_adhd": int(dev_excl.group(3)),
            "excluded_total": int(dev_excl.group(4)),
            "excluded_tdc": int(dev_excl.group(5)),
            "excluded_tdc_pct_display": dev_excl.group(6),
            "excluded_adhd": int(dev_excl.group(7)),
            "excluded_adhd_pct_display": dev_excl.group(8),
            "fisher_p": float(dev_excl.group(9)),
            "fisher_p_display": dev_excl.group(9),
        },
        "development_sensitivity": dev_sensitivity,
        "sealed_nyu_attrition": {
            "excluded_total": int(nyu_excl.group(1)),
            "excluded_tdc": int(nyu_excl.group(2)),
            "excluded_tdc_pct_display": nyu_excl.group(3),
            "excluded_adhd": int(nyu_excl.group(4)),
            "excluded_adhd_pct_display": nyu_excl.group(5),
            "fisher_p": float(nyu_excl.group(6)),
            "fisher_p_display": nyu_excl.group(6),
        },
        "sealed_nyu_primary": sealed_primary,
        "sealed_nyu_sensitivity": sealed_sensitivity,
        "section_8_2_inconclusive_zone": {
            "condition": inc_row.group(1).strip(),
            "conclusion": inc_conclusion_md,
            "conclusion_plain": inc_conclusion_plain,
        },
    }


def main() -> None:
    payload = compile_study2_data(DOCS_STUDY2_DIR)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FRONTEND_DATA_DIR / "study2Data.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(WORKSPACE_ROOT)} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
