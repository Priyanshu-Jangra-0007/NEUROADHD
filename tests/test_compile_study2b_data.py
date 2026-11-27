"""Tests for scripts/compile_study2b_data.py and Study2bExploratorySection.jsx."""

import json
import re
import unittest
from pathlib import Path

from scripts.compile_study2b_data import (
    DOCS_EXPLORATORY_DIR,
    FRONTEND_DATA_DIR,
    compile_study2b_data,
)

STUDY2B_JSX_PATH = (
    FRONTEND_DATA_DIR.parent / "components" / "Study2bExploratorySection.jsx"
)


class TestCompileStudy2bData(unittest.TestCase):
    def test_compiled_json_matches_tracked_artifact(self) -> None:
        fresh = compile_study2b_data(DOCS_EXPLORATORY_DIR)
        disk_path = FRONTEND_DATA_DIR / "study2bData.json"
        self.assertTrue(disk_path.is_file(), "study2bData.json must exist")
        on_disk = json.loads(disk_path.read_text(encoding="utf-8"))
        self.assertEqual(fresh, on_disk)

    def test_json_values_match_results_summary_and_note(self) -> None:
        results_md = (DOCS_EXPLORATORY_DIR / "RESULTS_SUMMARY.md").read_text(
            encoding="utf-8"
        )
        note_md = (
            DOCS_EXPLORATORY_DIR / "NOTE_matched_pairs_2026-11-27.md"
        ).read_text(encoding="utf-8")
        data = json.loads(
            (FRONTEND_DATA_DIR / "study2bData.json").read_text(encoding="utf-8")
        )

        primary = data["primary"]
        self.assertEqual(primary["n_subjects"], 564)
        self.assertIn("N = 564", results_md)
        self.assertEqual(
            primary["pre_declared_outcome"],
            "no evidence of incremental connectivity signal in this design",
        )
        self.assertIn(primary["pre_declared_outcome"], results_md)
        self.assertEqual(primary["best_conn_config"], "CFG_01")
        self.assertEqual(primary["best_conn_auc_display"], "0.5511")
        self.assertIn(f"`{primary['best_conn_auc_display']}`", results_md)
        self.assertEqual(primary["best_conn_perm_p_display"], "0.4378")
        self.assertIn(f"$p = {primary['best_conn_perm_p_display']}$", results_md)
        self.assertEqual(primary["chosen_config"], "CFG_09")
        self.assertEqual(primary["delta_auc_display"], "-0.0024")
        self.assertEqual(primary["delta_ci_display"], "[-0.0144, +0.0096]")
        self.assertEqual(primary["delta_perm_p_display"], "0.9303")
        self.assertIn(
            f"{primary['delta_auc_display']} {primary['delta_ci_display']}",
            results_md,
        )
        self.assertFalse(primary["is_interesting"])
        self.assertEqual(primary["n_sites_positive"], 1)
        self.assertEqual(primary["n_sites_total"], 5)
        self.assertTrue(primary["motion_beats_best_conn"])

        baselines = primary["baselines"]
        self.assertEqual(baselines["motion"]["auc_display"], "0.6110")
        self.assertIn(f"`{baselines['motion']['auc_display']}`", results_md)
        self.assertEqual(baselines["age_sex"]["auc_display"], "0.5697")
        self.assertIn(f"`{baselines['age_sex']['auc_display']}`", results_md)
        self.assertEqual(baselines["combined"]["auc_display"], "0.6487")
        self.assertIn(f"`{baselines['combined']['auc_display']}`", results_md)

        power = data["power"]
        self.assertEqual(power["weak1_conn_auc_display"], "0.55")
        self.assertEqual(power["weak1_rule_power_display"], "8.0%")
        self.assertEqual(power["weak2_conn_auc_display"], "0.60")
        self.assertEqual(power["weak2_rule_power_display"], "22.0%")

        matched = data["matched_cohorts"]
        self.assertEqual(matched["realized_clean_pairs"], 207)
        self.assertEqual(matched["realized_clean_n"], 414)
        self.assertEqual(matched["realized_all_pairs"], 219)
        self.assertEqual(matched["realized_all_n"], 438)
        self.assertEqual(matched["pre_declared_clean_pairs"], 215)
        self.assertEqual(matched["pre_declared_all_pairs"], 226)
        self.assertIn("207", note_md)
        self.assertIn("219", note_md)

    def test_jsx_reads_json_has_na_fallback_and_no_metric_numeric_literals(self) -> None:
        jsx = STUDY2B_JSX_PATH.read_text(encoding="utf-8")
        self.assertNotIn("EXPLORATORY_STUDY2B", jsx)
        self.assertNotIn("unblinded NYU", jsx)
        self.assertIn("NYU was scored in Study 2 and is no longer sealed", jsx)
        self.assertIn("motion_beats_best_conn_sentence", jsx)
        self.assertIn("note_filename", jsx)
        self.assertIn('from "../data/study2bData.json"', jsx)
        self.assertIn('"N/A"', jsx)

        # Extract return (...) JSX markup and inspect rendered text nodes outside tags/styles
        ret_start = jsx.index("return (")
        jsx_body = jsx[ret_start:]
        jsx_body = re.sub(r"\{/\*.*?\*/\}", "", jsx_body, flags=re.S)
        text_nodes = re.findall(r"[>}]([^<>{}]+)[<{]", jsx_body)
        # Strip benign study-version labels ("Study 2", "Study 2b") before scanning for numbers
        metric_text_nodes = [
            re.sub(r"\bStudy 2b?\b", "", node).strip() for node in text_nodes
        ]
        for node in metric_text_nodes:
            self.assertIsNone(
                re.search(r"\d", node),
                f"Unexpected numeric literal in JSX metric text node: {node!r}",
            )


if __name__ == "__main__":
    unittest.main()
