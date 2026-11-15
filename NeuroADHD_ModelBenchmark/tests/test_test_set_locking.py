#!/usr/bin/env python3
"""
Unit tests verifying strict locking of the Peking_2 test set in run_all_models.py.
Ensures the test set cannot be evaluated by default or with a single flag.
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BENCHMARK_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_all_models


class TestTestSetLocking(unittest.TestCase):
    """Test suite ensuring test set evaluation is locked and double-gated."""

    def test_run_benchmark_default_argument_is_false(self):
        """Verify run_benchmark() signature defaults run_test to False."""
        sig = inspect.signature(run_all_models.run_benchmark)
        self.assertIn("run_test", sig.parameters)
        default_val = sig.parameters["run_test"].default
        self.assertIs(default_val, False, "run_benchmark default run_test MUST be False")

    @patch("run_all_models.run_benchmark")
    def test_cli_default_does_not_evaluate_test(self, mock_run):
        """Invoking run_all_models with default args must pass run_test=False."""
        with patch.object(sys, "argv", ["run_all_models.py"]):
            run_all_models.main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertFalse(kwargs.get("run_test"), "Default execution must not evaluate test set")

    @patch("run_all_models.run_benchmark")
    def test_single_evaluate_flag_does_not_unlock(self, mock_run):
        """Passing only --evaluate-test without --confirm-test-eval must NOT unlock test set."""
        with patch.object(sys, "argv", ["run_all_models.py", "--evaluate-test"]):
            run_all_models.main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertFalse(kwargs.get("run_test"), "Single --evaluate-test flag must NOT unlock test set")

    @patch("run_all_models.run_benchmark")
    def test_single_confirm_flag_does_not_unlock(self, mock_run):
        """Passing only --confirm-test-eval without --evaluate-test must NOT unlock test set."""
        with patch.object(sys, "argv", ["run_all_models.py", "--confirm-test-eval"]):
            run_all_models.main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertFalse(kwargs.get("run_test"), "Single --confirm-test-eval flag must NOT unlock test set")

    @patch("run_all_models.run_benchmark")
    def test_both_flags_required_to_unlock(self, mock_run):
        """Only passing BOTH --evaluate-test AND --confirm-test-eval unlocks test set."""
        with patch.object(sys, "argv", ["run_all_models.py", "--evaluate-test", "--confirm-test-eval"]):
            run_all_models.main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertTrue(kwargs.get("run_test"), "Both flags together must unlock test set")


if __name__ == "__main__":
    unittest.main()
