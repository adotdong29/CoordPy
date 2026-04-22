"""Tests for Phase-18 repeat/aggregate logic.

Tests `_aggregate_surface_repeats` in isolation — no LLM or Ollama needed.
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase18_general_trigger import (
    _aggregate_surface_repeats,
)


# ---- Fixture builders -------------------------------------------------------

def _make_summary(
    gap_general: float,
    gap_specific: float,
    casr_general: float = 0.8,
    casr_specific: float = 0.8,
    abl_general: float | None = None,
    abl_specific: float | None = None,
    tok_specific_casr: int = 4000,
    tok_general_casr: int = 4200,
    fire_general: int = 4,
    fire_specific: int = 3,
    n_higher: int = 7,
    p18a: bool = True,
    p18b: bool = True,
    p18c: bool = True,
) -> dict:
    """Build a minimal _surface_summary()-shaped dict for testing."""
    if abl_general is None:
        abl_general = casr_general - gap_general
    if abl_specific is None:
        abl_specific = casr_specific - gap_specific
    return {
        "casr_vs_ablation_gap": {
            "general": gap_general,
            "specific": gap_specific,
            "ratio": (gap_general / gap_specific) if gap_specific != 0 else None,
        },
        "scores": {
            "general":  {"round1": 0.5, "full": casr_general,
                         "casr": casr_general, "ablation": abl_general},
            "specific": {"round1": 0.5, "full": casr_specific,
                         "casr": casr_specific, "ablation": abl_specific},
        },
        "tokens": {
            "specific_full": tok_specific_casr + 500,
            "specific_casr": tok_specific_casr,
            "specific_ablation": tok_specific_casr - 200,
            "general_full": tok_general_casr + 600,
            "general_casr": tok_general_casr,
            "general_ablation": tok_general_casr - 100,
        },
        "fire_rates_casr_leg": {
            "general_refined": fire_general,
            "specific_refined": fire_specific,
            "general_skipped": n_higher - fire_general,
            "specific_skipped": n_higher - fire_specific,
            "n_higher_tier_agents": n_higher,
        },
        "claims": {
            "P18A_gap_preserved": p18a,
            "P18B_token_cost_bounded": p18b,
            "P18C_fire_rate_sensible": p18c,
        },
        "surface": "test",
        "label": "Test Surface",
    }


# ---- Tests ------------------------------------------------------------------

class TestAggregateSingleRepeat(unittest.TestCase):
    def setUp(self):
        self.summ = _make_summary(gap_general=0.120, gap_specific=0.0)
        self.agg = _aggregate_surface_repeats("test", [self.summ])

    def test_n_repeats(self):
        self.assertEqual(self.agg["n_repeats"], 1)

    def test_mean_equals_value(self):
        self.assertAlmostEqual(self.agg["gap_general"]["mean"], 0.120, places=4)
        self.assertAlmostEqual(self.agg["gap_specific"]["mean"], 0.0,   places=4)

    def test_min_max_equal_mean(self):
        for key in ("gap_general", "gap_specific"):
            self.assertEqual(self.agg[key]["min"], self.agg[key]["mean"])
            self.assertEqual(self.agg[key]["max"], self.agg[key]["mean"])

    def test_no_stddev_when_n_equals_1(self):
        self.assertNotIn("stddev", self.agg["gap_general"])

    def test_token_ratio(self):
        self.assertAlmostEqual(
            self.agg["token_ratio_casr"]["mean"], 4200 / 4000, places=3
        )

    def test_claim_pass_rate_all_pass(self):
        cr = self.agg["claim_pass_rates"]
        self.assertAlmostEqual(cr["P18A"], 1.0)
        self.assertAlmostEqual(cr["P18B"], 1.0)
        self.assertAlmostEqual(cr["P18C"], 1.0)

    def test_general_beats_specific(self):
        self.assertEqual(self.agg["general_beats_specific_gap"]["count"], 1)
        self.assertAlmostEqual(self.agg["general_beats_specific_gap"]["rate"], 1.0)


class TestAggregateMultipleRepeats(unittest.TestCase):
    def setUp(self):
        # Three repeats with varying gaps
        self.summaries = [
            _make_summary(gap_general=0.10, gap_specific=0.00),
            _make_summary(gap_general=0.15, gap_specific=0.05),
            _make_summary(gap_general=0.05, gap_specific=0.00),
        ]
        self.agg = _aggregate_surface_repeats("test", self.summaries)

    def test_n_repeats(self):
        self.assertEqual(self.agg["n_repeats"], 3)

    def test_mean_gap_general(self):
        expected = round((0.10 + 0.15 + 0.05) / 3, 4)
        self.assertAlmostEqual(self.agg["gap_general"]["mean"], expected, places=4)

    def test_min_max_gap_general(self):
        self.assertAlmostEqual(self.agg["gap_general"]["min"], 0.05, places=4)
        self.assertAlmostEqual(self.agg["gap_general"]["max"], 0.15, places=4)

    def test_stddev_present_when_n_gt_1(self):
        self.assertIn("stddev", self.agg["gap_general"])
        self.assertGreater(self.agg["gap_general"]["stddev"], 0)

    def test_general_beats_specific_all_three(self):
        # All three have gap_general > gap_specific
        self.assertEqual(self.agg["general_beats_specific_gap"]["count"], 3)
        self.assertAlmostEqual(self.agg["general_beats_specific_gap"]["rate"], 1.0)


class TestAggregateClaimPassRates(unittest.TestCase):
    def test_mixed_p18a(self):
        summaries = [
            _make_summary(0.12, 0.0, p18a=True),
            _make_summary(0.0,  0.0, p18a=False),  # gap=0, claim fails
            _make_summary(0.08, 0.0, p18a=True),
        ]
        agg = _aggregate_surface_repeats("test", summaries)
        self.assertAlmostEqual(agg["claim_pass_rates"]["P18A"], 2 / 3, places=4)

    def test_all_p18b_fail(self):
        summaries = [_make_summary(0.1, 0.0, p18b=False) for _ in range(3)]
        agg = _aggregate_surface_repeats("test", summaries)
        self.assertAlmostEqual(agg["claim_pass_rates"]["P18B"], 0.0)

    def test_general_does_not_beat_specific(self):
        # gap_general < gap_specific in all repeats
        summaries = [
            _make_summary(gap_general=0.10, gap_specific=0.20),
            _make_summary(gap_general=0.05, gap_specific=0.30),
        ]
        agg = _aggregate_surface_repeats("test", summaries)
        self.assertEqual(agg["general_beats_specific_gap"]["count"], 0)
        self.assertAlmostEqual(agg["general_beats_specific_gap"]["rate"], 0.0)


class TestAggregateTokenRatio(unittest.TestCase):
    def test_ratio_averaged_correctly(self):
        summaries = [
            _make_summary(0.1, 0.0, tok_specific_casr=4000, tok_general_casr=4200),
            _make_summary(0.1, 0.0, tok_specific_casr=5000, tok_general_casr=5500),
        ]
        agg = _aggregate_surface_repeats("test", summaries)
        expected_mean = round((4200 / 4000 + 5500 / 5000) / 2, 4)
        self.assertAlmostEqual(
            agg["token_ratio_casr"]["mean"], expected_mean, places=3
        )

    def test_zero_specific_tokens_excluded(self):
        # specific_casr=0 should not produce inf
        summaries = [
            _make_summary(0.1, 0.0, tok_specific_casr=0, tok_general_casr=100),
            _make_summary(0.1, 0.0, tok_specific_casr=4000, tok_general_casr=4200),
        ]
        agg = _aggregate_surface_repeats("test", summaries)
        # Only one valid ratio (4200/4000)
        self.assertAlmostEqual(agg["token_ratio_casr"]["mean"], 4200 / 4000, places=3)


class TestFuzzGridExtension(unittest.TestCase):
    """Verify that the extended pairwise fuzz grid includes signed-encoding
    boundary cases that distinguish two's-complement from sign-magnitude."""

    def test_negative_8bit_cases_present(self):
        from vision_mvp.core.general_trigger import _FUZZ_INPUTS_PAIRWISE
        pairs = set(_FUZZ_INPUTS_PAIRWISE)
        self.assertIn((-1, 8),   pairs, "(-1, 8) must be in fuzz grid")
        self.assertIn((-128, 8), pairs, "(-128, 8) must be in fuzz grid")
        self.assertIn((-5, 8),   pairs, "(-5, 8) must be in fuzz grid")

    def test_hybrid_fires_on_signed_encoding_drift(self):
        from vision_mvp.core.general_trigger import HybridStructuralTrigger

        twos_comp = """
def encode_signed(value, bits):
    if value < 0:
        return (1 << bits) + value
    return value
"""
        sign_mag = """
def encode_signed(value, bits):
    if value < 0:
        return (1 << (bits - 1)) | (-value)
    return value
"""
        t = HybridStructuralTrigger()
        d = t.should_refine(twos_comp, [sign_mag], threshold=0.34)
        self.assertTrue(
            d.refine,
            f"Expected trigger to fire on signed-encoding drift; "
            f"components={d.info.get('components')}",
        )


if __name__ == "__main__":
    unittest.main()
