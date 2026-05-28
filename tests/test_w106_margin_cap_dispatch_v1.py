"""W106 — margin-cap dispatch decision-rule tests.

Covers GATE 1 routing (all four failure-signature rows), GATE 2
sub-conditions (2a fair battlefield / 2b no-authoritative-fair-
result / 2c fixable-confound), the end-to-end NO-GO on the
realized W105 Verdict-C/C1 case, a counterfactual GO path, the
refuse-to-run paths (schema / class-count / no-retired-class),
decision-CID determinism, and the real W105 consolidated verdict
json (NO-GO).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.margin_cap_dispatch_v1 import (  # noqa: E402
    ENTITLE_APPS_PREFLIGHT,
    ENTITLE_CHEAP_CONFIRMATION,
    ENTITLE_CROSS_SCALE_COLLAPSE_AUDIT,
    ENTITLE_LIVECODEBENCH_PREFLIGHT,
    SLICE_FAIR_BROAD,
    SLICE_RESCUE_CONCENTRATED,
    MarginCapDispatchError,
    build_margin_cap_dispatch_decision_v1,
    classify_entitlement_v1,
    evaluate_gate2_v1,
    format_margin_cap_dispatch_markdown_v1,
)
from coordpy.phase3_retirement_evaluator_v1 import (  # noqa: E402
    W105_PHASE3_RETIREMENT_EVALUATOR_V1_SCHEMA_VERSION,
)

_REAL_VERDICT = (
    ROOT / "results" / "w105"
    / "humaneval_plus_phase3_retirement_bench"
    / "w105_phase3_FINAL_consolidated"
    / "phase3_retirement_verdict.json")


def _mk_class(
        *, class_id, label, margin, mlb2, a1, per_seed_maj=3,
        n_seeds=3, executor_clean=True, audit=True, bars=None):
    """Build a synthetic per-class verdict dict matching the
    phase3_retirement_evaluator_v1 output shape."""
    if bars is None:
        bars = 6 if label == "RETIRED" else 5
    return {
        "model_class_id": class_id,
        "verdict_label": label,
        "bar1_margin_mean_b_minus_a1_pp": float(margin),
        "mean_mlb2_rescue_rate": float(mlb2),
        "bar2_per_seed_majority_count": int(per_seed_maj),
        "bar5_audit_chain_passes": bool(audit),
        "bar6_executor_clean_passes": bool(executor_clean),
        "n_bars_passed_of_6": int(bars),
        "cells": [{"a1_pct": float(a1)} for _ in range(n_seeds)],
    }


def _mk_verdict(per_class, schema=None):
    return {
        "schema": (
            schema
            or W105_PHASE3_RETIREMENT_EVALUATOR_V1_SCHEMA_VERSION),
        "slice_pack_cid": "deadbeef",
        "corpus_sha256": "cafe",
        "per_class": per_class,
    }


# Signatures reused below.
def _sig(*, margin, mlb2, a1, per_seed_maj=3, n_seeds=3,
         executor_clean=True, audit=True):
    from coordpy.margin_cap_dispatch_v1 import FailedClassSignatureV1
    return FailedClassSignatureV1(
        model_class_id="x", verdict_label="FAIL_MARGIN",
        margin_pp=float(margin), mlb2_rescue_rate=float(mlb2),
        a1_max_pct=float(a1), per_seed_majority_count=int(per_seed_maj),
        n_seeds=int(n_seeds), executor_clean=bool(executor_clean),
        audit_chain_passes=bool(audit), n_bars_passed_of_6=5)


class TestGate1Entitlement(unittest.TestCase):

    def test_cheap_confirmation_row(self):
        step, entitles, ceil = classify_entitlement_v1(
            sig=_sig(margin=2.33, mlb2=0.5054, a1=87.0))
        self.assertEqual(step, ENTITLE_CHEAP_CONFIRMATION)
        self.assertTrue(entitles)
        self.assertEqual(ceil, 990)

    def test_livecodebench_row_margin_neg_mlb2_weak(self):
        step, entitles, ceil = classify_entitlement_v1(
            sig=_sig(margin=-3.0, mlb2=0.11, a1=80.0))
        self.assertEqual(step, ENTITLE_LIVECODEBENCH_PREFLIGHT)
        self.assertFalse(entitles)
        self.assertEqual(ceil, 0)

    def test_apps_row_a1_saturated(self):
        step, entitles, _ = classify_entitlement_v1(
            sig=_sig(margin=2.0, mlb2=0.5, a1=91.0))
        self.assertEqual(step, ENTITLE_APPS_PREFLIGHT)
        self.assertFalse(entitles)

    def test_cross_scale_collapse_row(self):
        step, entitles, ceil = classify_entitlement_v1(
            sig=_sig(margin=-2.0, mlb2=0.5, a1=80.0))
        self.assertEqual(step, ENTITLE_CROSS_SCALE_COLLAPSE_AUDIT)
        self.assertFalse(entitles)
        self.assertEqual(ceil, 3300)


class TestGate2(unittest.TestCase):

    def test_2a_rescue_concentrated_fails(self):
        g2a, _, _, _ = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5, a1=87.0),
            proposed_confirmation_slice_type=(
                SLICE_RESCUE_CONCENTRATED),
            fair_broad_phase3_result_exists=False)
        self.assertFalse(g2a)

    def test_2a_fair_broad_passes(self):
        g2a, _, _, _ = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5, a1=87.0),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=False)
        self.assertTrue(g2a)

    def test_2b_fails_when_fair_result_exists(self):
        _, g2b, _, _ = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5, a1=87.0),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=True)
        self.assertFalse(g2b)

    def test_2c_clean_magnitude_miss_fails(self):
        # executor clean + audit + per-seed majority + MLB-2 healthy
        # = no confound.
        _, _, g2c, _ = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5054, a1=87.0,
                     per_seed_maj=3, executor_clean=True, audit=True),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=False)
        self.assertFalse(g2c)

    def test_2c_confound_present_passes(self):
        # executor NOT clean ⇒ a fixable confound ⇒ 2c PASS.
        _, _, g2c, _ = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5054, a1=87.0,
                     executor_clean=False),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=False)
        self.assertTrue(g2c)

    def test_gate2_all_three_required(self):
        # fair slice + no fair result + confound ⇒ GATE 2 overall PASS
        _, _, _, g2 = evaluate_gate2_v1(
            sig=_sig(margin=2.33, mlb2=0.5054, a1=87.0,
                     executor_clean=False),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=False)
        self.assertTrue(g2)


class TestEndToEnd(unittest.TestCase):

    def _w105_like_verdict(self):
        return _mk_verdict([
            _mk_class(class_id="meta/llama-3.1-70b-instruct",
                      label="FAIL_MARGIN", margin=2.33, mlb2=0.5054,
                      a1=87.0, per_seed_maj=3, bars=5),
            _mk_class(class_id="meta/llama-3.3-70b-instruct",
                      label="RETIRED", margin=7.0, mlb2=0.5562,
                      a1=84.0, per_seed_maj=3, bars=6),
        ])

    def test_realized_w105_case_is_no_go(self):
        d = build_margin_cap_dispatch_decision_v1(
            phase3_verdict=self._w105_like_verdict())
        self.assertEqual(d.decision, "NO_GO")
        self.assertTrue(d.gate1_entitles_cheap_confirmation)
        self.assertFalse(d.gate2_verdict_changing_power)
        self.assertEqual(
            d.failed_class.model_class_id,
            "meta/llama-3.1-70b-instruct")
        self.assertEqual(
            d.retired_class_id, "meta/llama-3.3-70b-instruct")
        self.assertEqual(
            d.carry_forward_id,
            "W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-"
            "CHEAP-CONFIRMATION-NOT-EARNED-CAP")

    def test_no_go_holds_even_if_fair_slice_proposed(self):
        # Fair slice fixes 2a, but 2b (fair result exists) + 2c
        # (clean miss) still fail ⇒ NO-GO.
        d = build_margin_cap_dispatch_decision_v1(
            phase3_verdict=self._w105_like_verdict(),
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=True)
        self.assertEqual(d.decision, "NO_GO")

    def test_counterfactual_go_path(self):
        # Entitled + fair slice + no fair result yet + confound
        # present ⇒ GO.
        v = _mk_verdict([
            _mk_class(class_id="c-fail", label="FAIL_MARGIN",
                      margin=2.0, mlb2=0.5, a1=80.0,
                      executor_clean=False),
            _mk_class(class_id="c-ok", label="RETIRED", margin=7.0,
                      mlb2=0.56, a1=84.0),
        ])
        d = build_margin_cap_dispatch_decision_v1(
            phase3_verdict=v,
            proposed_confirmation_slice_type=SLICE_FAIR_BROAD,
            fair_broad_phase3_result_exists=False)
        self.assertEqual(d.decision, "GO")
        self.assertTrue(d.gate2_verdict_changing_power)
        self.assertEqual(d.carry_forward_id, "")

    def test_decision_cid_deterministic(self):
        v = self._w105_like_verdict()
        d1 = build_margin_cap_dispatch_decision_v1(phase3_verdict=v)
        d2 = build_margin_cap_dispatch_decision_v1(phase3_verdict=v)
        self.assertEqual(d1.cid(), d2.cid())

    def test_markdown_renders_decision(self):
        d = build_margin_cap_dispatch_decision_v1(
            phase3_verdict=self._w105_like_verdict())
        md = format_margin_cap_dispatch_markdown_v1(decision=d)
        self.assertIn("NO_GO", md)
        self.assertIn("GATE 1", md)
        self.assertIn("GATE 2", md)


class TestRefuseToRun(unittest.TestCase):

    def test_schema_mismatch_refuses(self):
        v = _mk_verdict(
            [_mk_class(class_id="a", label="FAIL_MARGIN",
                       margin=2.0, mlb2=0.5, a1=80.0),
             _mk_class(class_id="b", label="RETIRED", margin=7.0,
                       mlb2=0.5, a1=84.0)],
            schema="coordpy.some_other_schema.v9")
        with self.assertRaises(MarginCapDispatchError):
            build_margin_cap_dispatch_decision_v1(phase3_verdict=v)

    def test_two_fail_classes_refuses(self):
        v = _mk_verdict([
            _mk_class(class_id="a", label="FAIL_MARGIN", margin=2.0,
                      mlb2=0.5, a1=80.0),
            _mk_class(class_id="b", label="FAIL_PER_SEED_MAJORITY",
                      margin=-1.0, mlb2=0.5, a1=80.0),
        ])
        with self.assertRaises(MarginCapDispatchError):
            build_margin_cap_dispatch_decision_v1(phase3_verdict=v)

    def test_no_retired_class_refuses(self):
        v = _mk_verdict([
            _mk_class(class_id="a", label="FAIL_MARGIN", margin=2.0,
                      mlb2=0.5, a1=80.0),
            _mk_class(class_id="b",
                      label="RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING",
                      margin=7.0, mlb2=0.2, a1=84.0, bars=6),
        ])
        with self.assertRaises(MarginCapDispatchError):
            build_margin_cap_dispatch_decision_v1(phase3_verdict=v)

    def test_empty_per_class_refuses(self):
        with self.assertRaises(MarginCapDispatchError):
            build_margin_cap_dispatch_decision_v1(
                phase3_verdict=_mk_verdict([]))


class TestRealW105Verdict(unittest.TestCase):

    def test_real_verdict_json_is_no_go(self):
        if not _REAL_VERDICT.exists():
            self.skipTest(f"real verdict not present: {_REAL_VERDICT}")
        with open(_REAL_VERDICT, encoding="utf-8") as fh:
            verdict = json.load(fh)
        d = build_margin_cap_dispatch_decision_v1(
            phase3_verdict=verdict)
        self.assertEqual(d.decision, "NO_GO")
        self.assertEqual(
            d.failed_class.model_class_id,
            "meta/llama-3.1-70b-instruct")
        self.assertEqual(
            d.retired_class_id, "meta/llama-3.3-70b-instruct")
        self.assertTrue(d.gate1_entitles_cheap_confirmation)
        self.assertAlmostEqual(d.failed_class.margin_pp, 2.3333,
                               places=3)


if __name__ == "__main__":
    unittest.main()
