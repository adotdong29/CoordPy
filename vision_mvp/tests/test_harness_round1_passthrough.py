"""Verify that passing round1= to harness run() skips run_round1().

These tests mock out the LLM and scoring so they run without Ollama.
The only property being asserted: when round1 is pre-supplied, the harness
must use it and must NOT call run_round1() for a fresh LLM generation.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import vision_mvp.experiments.phase14_benchmark as ph14
import vision_mvp.experiments.phase17_generality as ph17
from vision_mvp.tasks.protocol_codesign import SPEC_ORDER as PK_SPEC_ORDER
from vision_mvp.tasks.numeric_ledger import SPEC_ORDER as NL_SPEC_ORDER


# ---- Dummy data helpers ----------------------------------------------------

_DUMMY_SCORE = {
    "weighted_score": 0.5,
    "n_passed": 10,
    "n_total": 25,
    "passed_tests": [],
    "failed_tests": [],
    "per_test": {},
    "syntax_error": None,
    "timed_out": False,
    "stderr": "",
    "module_src": "",
}


def _dummy_r1(specs: list[str]) -> dict:
    return {
        "drafts": {s: f"def f_{s}(): pass" for s in specs},
        "tokens": {s: {"prompt": 100, "completion": 50} for s in specs},
        "acceptance": {s: {"accepted": True, "attempts": 1} for s in specs},
    }


def _dummy_r2(specs: list[str]) -> dict:
    return {
        "drafts": {s: f"def f_{s}(): pass" for s in specs},
        "tokens": {s: {"prompt": 0, "completion": 0} for s in specs},
        "routing_stats": {s: {
            "delivered": 0, "dropped": 0,
            "delivered_tokens": 0, "dropped_tokens": 0,
        } for s in specs},
        "trigger_info": {s: {
            "tier": 0, "refined": False, "reason": "tier0_frozen",
        } for s in specs},
        "acceptance": {s: {"accepted": True, "attempts": 1} for s in specs},
    }


# ---- Phase-14 (ProtocolKit) ------------------------------------------------

class TestPhase14Round1Passthrough(unittest.TestCase):
    def test_skips_run_round1_when_provided(self):
        pre_built = _dummy_r1(list(PK_SPEC_ORDER))

        with patch.object(ph14, "run_round1") as mock_r1, \
             patch.object(ph14, "score_drafts", return_value=_DUMMY_SCORE), \
             patch.object(ph14, "run_round2_topological",
                          return_value=_dummy_r2(list(PK_SPEC_ORDER))), \
             patch("vision_mvp.experiments.phase14_benchmark.LLMClient"):
            ph14.run(
                model="test-model",
                out_path=None,
                max_retries=0,
                ablation_seed=42,
                event_threshold=0.34,
                round1=pre_built,
            )

        mock_r1.assert_not_called()

    def test_calls_run_round1_when_not_provided(self):
        with patch.object(ph14, "run_round1",
                          return_value=_dummy_r1(list(PK_SPEC_ORDER))) as mock_r1, \
             patch.object(ph14, "score_drafts", return_value=_DUMMY_SCORE), \
             patch.object(ph14, "run_round2_topological",
                          return_value=_dummy_r2(list(PK_SPEC_ORDER))), \
             patch("vision_mvp.experiments.phase14_benchmark.LLMClient"):
            ph14.run(
                model="test-model",
                out_path=None,
                max_retries=0,
                ablation_seed=42,
                event_threshold=0.34,
            )

        mock_r1.assert_called_once()


# ---- Phase-17 (NumericLedger) ----------------------------------------------

class TestPhase17Round1Passthrough(unittest.TestCase):
    def test_skips_run_round1_when_provided(self):
        pre_built = _dummy_r1(list(NL_SPEC_ORDER))

        with patch.object(ph17, "run_round1") as mock_r1, \
             patch.object(ph17, "score_drafts", return_value=_DUMMY_SCORE), \
             patch.object(ph17, "run_round2_topological",
                          return_value=_dummy_r2(list(NL_SPEC_ORDER))), \
             patch("vision_mvp.experiments.phase17_generality.LLMClient"):
            ph17.run(
                model="test-model",
                out_path=None,
                max_retries=0,
                ablation_seed=42,
                event_threshold=0.34,
                round1=pre_built,
            )

        mock_r1.assert_not_called()

    def test_calls_run_round1_when_not_provided(self):
        with patch.object(ph17, "run_round1",
                          return_value=_dummy_r1(list(NL_SPEC_ORDER))) as mock_r1, \
             patch.object(ph17, "score_drafts", return_value=_DUMMY_SCORE), \
             patch.object(ph17, "run_round2_topological",
                          return_value=_dummy_r2(list(NL_SPEC_ORDER))), \
             patch("vision_mvp.experiments.phase17_generality.LLMClient"):
            ph17.run(
                model="test-model",
                out_path=None,
                max_retries=0,
                ablation_seed=42,
                event_threshold=0.34,
            )

        mock_r1.assert_called_once()


if __name__ == "__main__":
    unittest.main()
