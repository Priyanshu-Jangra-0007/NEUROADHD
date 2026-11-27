#!/usr/bin/env python3
"""
Compile Study 2b (exploratory follow-up) aggregate metrics into frontend JSON
strictly by parsing tracked summary documents (no typed numbers, no raw data
access, zero subject IDs).

Sources parsed (read-only):
- docs/exploratory/RESULTS_SUMMARY.md
- docs/exploratory/NOTE_matched_pairs_2026-11-27.md
- docs/exploratory/OSF_RECORD.md (OSF DOI and GitHub tag provenance)

Output:
- NeuroADHD_ModelDashboard/frontend/src/data/study2bData.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DOCS_EXPLORATORY_DIR = WORKSPACE_ROOT / "docs" / "exploratory"
FRONTEND_DATA_DIR = (
    WORKSPACE_ROOT / "NeuroADHD_ModelDashboard" / "frontend" / "src" / "data"
)
MATCHED_PAIRS_NOTE_NAME = "NOTE_matched_pairs_2026-11-27.md"
GITHUB_REPO_BASE = "https://github.com/Priyanshu-Jangra-0007/NEUROADHD"


def _require_match(
    pattern: str, text: str, label: str, flags: int = 0
) -> re.Match[str]:
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"Failed to parse {label} with pattern: {pattern}")
    return m


def _parse_ci_pair(ci_str: str) -> Tuple[float, float]:
    m = _require_match(
        r"\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]", ci_str, "CI pair"
    )
    return float(m.group(1)), float(m.group(2))


def _parse_baseline_line(section_text: str, name: str) -> Dict[str, Any]:
    m = _require_match(
        rf"-\s*\*\*{re.escape(name)}:\*\*\s*`(\d+\.\d+)`\s*\(`95%\s*CI\s*(\[[^\]]+\])`\)",
        section_text,
        f"Baseline {name}",
    )
    ci_pair = _parse_ci_pair(m.group(2))
    return {
        "auc": float(m.group(1)),
        "auc_display": m.group(1),
        "ci_95": [ci_pair[0], ci_pair[1]],
        "ci_display": m.group(2),
    }


def _parse_matched_section(section_text: str, label: str) -> Dict[str, Any]:
    hdr = _require_match(
        r"Realized \$N = (\d+)\$ / (\d+) Pairs\)",
        section_text,
        f"{label} header",
    )
    chosen = _require_match(
        r"\*\*Chosen by Max[^\n]*:\*\*\s*`(CFG_\d+)`\s*\(`([+-]?\d+\.\d+)`\);\s*\*\*Max Conn-Only AUC:\*\*\s*`(\d+\.\d+)`\s*\(`(CFG_\d+)`\)",
        section_text,
        f"{label} chosen/max",
    )
    return {
        "realized_n": int(hdr.group(1)),
        "realized_pairs": int(hdr.group(2)),
        "chosen_config": chosen.group(1),
        "max_delta_auc": float(chosen.group(2)),
        "max_delta_auc_display": chosen.group(2),
        "max_conn_auc": float(chosen.group(3)),
        "max_conn_auc_display": chosen.group(3),
        "max_conn_config": chosen.group(4),
    }


def compile_study2b_data(docs_dir: Path = DOCS_EXPLORATORY_DIR) -> Dict[str, Any]:
    results_md = (docs_dir / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    note_md = (docs_dir / MATCHED_PAIRS_NOTE_NAME).read_text(encoding="utf-8")
    osf_md = (docs_dir / "OSF_RECORD.md").read_text(encoding="utf-8")

    # Header / provenance fields
    banner_match = _require_match(
        r"^>\s*\*\*([^\n*]+)\*\*\s*$",
        results_md,
        "Results summary banner",
        flags=re.MULTILINE,
    )
    protocol_version = _require_match(
        r"-\s*\*\*Protocol Version:\*\*\s*`([^`]+)`",
        results_md,
        "Protocol version",
    ).group(1)
    osf_url = _require_match(
        r"(https://osf\.io/[a-z0-9]+)",
        results_md,
        "OSF URL",
    ).group(1)
    osf_doi = _require_match(
        r"\*\*OSF Registration DOI:\*\*\s*`(10\.\d+/OSF\.IO/[A-Z0-9]+)`\s*\((https://doi\.org/[^)]+)\)",
        osf_md,
        "OSF DOI",
    )
    frozen_tag = _require_match(
        r"freeze tag `(exploratory-frozen)` at `([0-9a-f]{40})`",
        results_md,
        "Freeze tag",
    )
    results_tag = _require_match(
        r"Tag `(exploratory-results)` \(tag object `([0-9a-f]{40})`\), commit `([0-9a-f]{40})`",
        osf_md,
        "Results tag",
    )
    resampling = _require_match(
        r"`n_permutations = (\d+)`, `n_boot = (\d+)`, `seed = (\d+)`, `n_workers = (\d+)`",
        results_md,
        "Resampling parameters",
    )

    # Section 1: Outcome & Power Table
    sec1_start = results_md.index("## 1. Pre-Declared Outcome & Power Context")
    sec2_start = results_md.index("## 2. Primary Analysis")
    sec1_text = results_md[sec1_start:sec2_start]

    outcome_sentence = _require_match(
        r'-\s*\*\*Pre-Declared Outcome Sentence:\*\*\s*\*\*`"([^"]+)"`\*\*',
        sec1_text,
        "Pre-declared outcome sentence",
    ).group(1)
    chosen_cfg = _require_match(
        r"-\s*\*\*Chosen Configuration[^:]*:\*\*\s*`(CFG_\d+)`",
        sec1_text,
        "Chosen configuration",
    ).group(1)

    weak1_pow = _require_match(
        r"\|\s*\*\*Weak 1 \(\$\\\\?text\{conn\}\s*\\approx\s*(\d+\.\d+)\$\)\*\*[^\n]*\|\s*\*\*`\d+/\d+`\s*\(`(\d+\.\d+%)`",
        sec1_text,
        "Power Weak 1 row",
    )
    weak2_pow = _require_match(
        r"\|\s*\*\*Weak 2 \(\$\\\\?text\{conn\}\s*\\approx\s*(\d+\.\d+)\$\)\*\*[^\n]*\|\s*\*\*`\d+/\d+`\s*\(`(\d+\.\d+%)`",
        sec1_text,
        "Power Weak 2 row",
    )

    # Section 2: Primary Analysis (N = 564)
    sec3_start = results_md.index("## 3. Sensitivity Analyses")
    sec2_text = results_md[sec2_start:sec3_start]

    primary_n = int(
        _require_match(
            r"## 2\. Primary Analysis \(\$N = (\d+)\$",
            sec2_text,
            "Primary analysis N",
        ).group(1)
    )
    baseline_motion = _parse_baseline_line(sec2_text, "Motion Only")
    baseline_age_sex = _parse_baseline_line(sec2_text, "Age + Sex")
    baseline_combined = _parse_baseline_line(sec2_text, "Motion + Age + Sex")

    test1 = _require_match(
        r"Test 1[^\n]+?:\*\*\s*Observed max connectivity-only AUC = `(\d+\.\d+)` \(`(CFG_\d+)`\),\s*\*\*\$p = (\d+\.\d+)\$\*\*\s*\(`(\d+\.\d+)`\)",
        sec2_text,
        "Test 1 label permutation",
    )
    test2 = _require_match(
        r"Test 2[^\n]+?:\*\*\s*Observed max[^\n]*= `([+-]?\d+\.\d+)` \(`(CFG_\d+)`\),\s*\*\*\$p = (\d+\.\d+)\$\*\*\s*\(`(\d+\.\d+)`\)",
        sec2_text,
        "Test 2 connectivity permutation",
    )
    conj_verdict = _require_match(
        r'`verdict = "([^"]+)"` \(`is_interesting = (true|false)`\)',
        sec2_text,
        "Conjunction verdict",
    )
    cond_b = _require_match(
        r"\(b\)[^\n]+?:\*\*\s*`(True|False)`[^\n]*=\s*([+-]?\d+\.\d+)\s*(\[[^\]]+\])\$,\s*\$p = (\d+\.\d+)\$",
        sec2_text,
        "Condition b delta CI",
    )
    cond_c = _require_match(
        r"\(c\)[^\n]+?:\*\*\s*`(True|False)`\s*\(`n_sites_positive = (\d+)/(\d+)`",
        sec2_text,
        "Condition c sites positive",
    )
    delta_ci_pair = _parse_ci_pair(cond_b.group(3))

    # Section 3.1-3.3: Sensitivity cohorts & NOTE_matched_pairs_2026-11-27.md
    sec31_start = results_md.index("### 3.1 All-Scans Cohort")
    sec32_start = results_md.index("### 3.2 Clean Deterministic Motion-Matched")
    sec33_start = results_md.index("### 3.3 All-Scans Deterministic Motion-Matched")
    sec4_start = results_md.index("## 4. Separate Multi-Run Family")

    all_620_n = int(
        _require_match(
            r"### 3\.1 All-Scans Cohort \(`all_620`, \$N = (\d+)\$\)",
            results_md[sec31_start:sec32_start],
            "all_620 N",
        ).group(1)
    )
    matched_clean_res = _parse_matched_section(
        results_md[sec32_start:sec33_start], "matched_215"
    )
    matched_all_res = _parse_matched_section(
        results_md[sec33_start:sec4_start], "matched_226"
    )

    pre_decl_note = _require_match(
        r"\*\*Pre-declared counts[^:]*:\*\*\s*`(\d+)`\s*post-exclusion pairs\s*\(`N = (\d+)`[^\n]*\)\s*and\s*`(\d+)`\s*pre-exclusion pairs\s*\(`N = (\d+)`",
        note_md,
        "Pre-declared matched counts in note",
    )
    realized_note = _require_match(
        r"\*\*Realized counts[^:]*:\*\*\s*`(\d+)`\s*post-exclusion pairs\s*\(`N = (\d+)`[^\n]*\)\s*and\s*`(\d+)`\s*pre-exclusion pairs\s*\(`N = (\d+)`",
        note_md,
        "Realized matched counts in note",
    )
    if (
        int(realized_note.group(1)) != matched_clean_res["realized_pairs"]
        or int(realized_note.group(3)) != matched_all_res["realized_pairs"]
    ):
        raise ValueError(
            "Mismatch between RESULTS_SUMMARY.md and matched-pairs note counts"
        )

    # Section 4: Multi-run family
    multirun_n = int(
        _require_match(
            r"## 4\. Separate Multi-Run Family \([^)]*\$N = (\d+)\$",
            results_md[sec4_start:],
            "Multi-run N",
        ).group(1)
    )

    best_conn_auc = float(test1.group(1))
    motion_auc = baseline_motion["auc"]
    motion_beats = motion_auc > best_conn_auc
    comparison_word = "higher" if motion_beats else "not higher"
    motion_vs_conn_sentence = (
        f"The motion-only baseline AUC ({baseline_motion['auc_display']}) is "
        f"{comparison_word} than the best connectivity-only AUC ({test1.group(1)})."
    )

    return {
        "source_files": [
            "docs/exploratory/RESULTS_SUMMARY.md",
            f"docs/exploratory/{MATCHED_PAIRS_NOTE_NAME}",
            "docs/exploratory/OSF_RECORD.md",
        ],
        "protocol_version": protocol_version,
        "banner_from_summary": banner_match.group(1).strip(),
        "resampling": {
            "n_permutations": int(resampling.group(1)),
            "n_boot": int(resampling.group(2)),
            "seed": int(resampling.group(3)),
            "n_workers": int(resampling.group(4)),
        },
        "osf": {
            "url": osf_url,
            "doi": osf_doi.group(1),
            "doi_url": osf_doi.group(2),
            "frozen_tag": frozen_tag.group(1),
            "frozen_commit": frozen_tag.group(2),
            "results_tag": results_tag.group(1),
            "results_commit": results_tag.group(3),
        },
        "primary": {
            "n_subjects": primary_n,
            "pre_declared_outcome": outcome_sentence,
            "chosen_config": chosen_cfg,
            "best_conn_config": test1.group(2),
            "best_conn_auc": best_conn_auc,
            "best_conn_auc_display": test1.group(1),
            "best_conn_perm_p": float(test1.group(3)),
            "best_conn_perm_p_display": test1.group(3),
            "best_conn_perm_p_exact": float(test1.group(4)),
            "delta_auc": float(cond_b.group(2)),
            "delta_auc_display": cond_b.group(2),
            "delta_ci_95": [delta_ci_pair[0], delta_ci_pair[1]],
            "delta_ci_display": cond_b.group(3),
            "delta_perm_p": float(test2.group(3)),
            "delta_perm_p_display": test2.group(3),
            "delta_perm_p_exact": float(test2.group(4)),
            "verdict": conj_verdict.group(1),
            "is_interesting": conj_verdict.group(2) == "true",
            "n_sites_positive": int(cond_c.group(2)),
            "n_sites_total": int(cond_c.group(3)),
            "sites_positive_display": f"{cond_c.group(2)} of {cond_c.group(3)}",
            "motion_beats_best_conn": motion_beats,
            "motion_beats_best_conn_sentence": motion_vs_conn_sentence,
            "baselines": {
                "motion": baseline_motion,
                "age_sex": baseline_age_sex,
                "combined": baseline_combined,
            },
        },
        "power": {
            "weak1_conn_auc_display": weak1_pow.group(1),
            "weak1_rule_power_display": weak1_pow.group(2),
            "weak2_conn_auc_display": weak2_pow.group(1),
            "weak2_rule_power_display": weak2_pow.group(2),
        },
        "matched_cohorts": {
            "realized_clean_pairs": matched_clean_res["realized_pairs"],
            "realized_clean_n": matched_clean_res["realized_n"],
            "realized_all_pairs": matched_all_res["realized_pairs"],
            "realized_all_n": matched_all_res["realized_n"],
            "realized_pairs_display": f"{matched_clean_res['realized_pairs']} and {matched_all_res['realized_pairs']}",
            "pre_declared_clean_pairs": int(pre_decl_note.group(1)),
            "pre_declared_clean_n": int(pre_decl_note.group(2)),
            "pre_declared_all_pairs": int(pre_decl_note.group(3)),
            "pre_declared_all_n": int(pre_decl_note.group(4)),
            "pre_declared_pairs_display": f"{pre_decl_note.group(1)} and {pre_decl_note.group(3)}",
            "note_filename": MATCHED_PAIRS_NOTE_NAME,
            "note_path": f"docs/exploratory/{MATCHED_PAIRS_NOTE_NAME}",
        },
        "sensitivity": {
            "all_620_n": all_620_n,
            "multirun_n": multirun_n,
        },
        "links": {
            "osf_doi": osf_doi.group(2),
            "osf_url": osf_url,
            "frozen_tag": f"{GITHUB_REPO_BASE}/releases/tag/{frozen_tag.group(1)}",
            "results_tag": f"{GITHUB_REPO_BASE}/releases/tag/{results_tag.group(1)}",
            "results_summary": f"{GITHUB_REPO_BASE}/blob/{results_tag.group(1)}/docs/exploratory/RESULTS_SUMMARY.md",
            "matched_pairs_note": f"{GITHUB_REPO_BASE}/blob/note/matched-pairs/docs/exploratory/{MATCHED_PAIRS_NOTE_NAME}",
        },
    }


def main() -> None:
    payload = compile_study2b_data(DOCS_EXPLORATORY_DIR)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FRONTEND_DATA_DIR / "study2bData.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out_path.relative_to(WORKSPACE_ROOT)} ({out_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
