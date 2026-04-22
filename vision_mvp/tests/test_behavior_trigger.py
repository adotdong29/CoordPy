"""Tests for the behavior-fingerprint event trigger."""

from __future__ import annotations

import unittest

from vision_mvp.core.behavior_trigger import (
    pair_disagreement, should_refine, _function_names_defined,
)


# --- reference (shared half-up / cents / skip / saturate / two's-comp) ---

_REF_ROUND_HALFUP = """
def round_amount(value, decimals):
    import math
    mult = 10 ** decimals
    x = value * mult
    if x >= 0:
        r = math.floor(x + 0.5)
    else:
        r = -math.floor(-x + 0.5)
    return r / mult
"""

_REF_CHECK_HALFUP = """
def check_rounded(value, rounded, decimals):
    import math
    mult = 10 ** decimals
    x = value * mult
    if x >= 0:
        r = math.floor(x + 0.5)
    else:
        r = -math.floor(-x + 0.5)
    return abs(rounded - r / mult) < 1e-9
"""

# Banker's-rounding variant (disagrees with half-up on 0.5)
_BANKERS_ROUND = """
def round_amount(value, decimals):
    return round(value, decimals)
"""

_BANKERS_CHECK = """
def check_rounded(value, rounded, decimals):
    return abs(rounded - round(value, decimals)) < 1e-9
"""


class TestFunctionNames(unittest.TestCase):
    def test_parse_def_names(self):
        names = _function_names_defined(_REF_ROUND_HALFUP)
        self.assertEqual(names, {"round_amount"})

    def test_syntax_error_returns_empty(self):
        self.assertEqual(
            _function_names_defined("def broken(:"), set(),
        )


class TestPairDisagreement(unittest.TestCase):
    def test_matching_halfup_halfup(self):
        d = pair_disagreement(_REF_ROUND_HALFUP, _REF_CHECK_HALFUP)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_matching_bankers_bankers(self):
        d = pair_disagreement(_BANKERS_ROUND, _BANKERS_CHECK)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_halfup_rounder_vs_bankers_checker(self):
        # Producer half-up gives 1.0 for 0.5; consumer bankers expects 0.0.
        # Most probes disagree — score should be high.
        d = pair_disagreement(_REF_ROUND_HALFUP, _BANKERS_CHECK)
        self.assertIsNotNone(d)
        self.assertGreater(d, 0.25)  # at least 1 of 4 probes flips

    def test_nonpair_sources_return_none(self):
        src_a = "def make_foo(x): return {'a': x}"
        src_b = "def make_bar(y): return y * 2"
        self.assertIsNone(pair_disagreement(src_a, src_b))

    def test_scale_cents_vs_mils_disagree(self):
        cents_prod = "def to_ledger(d): return int(d * 100)"
        cents_cons = "def from_ledger(u): return u / 100.0"
        mils_cons  = "def from_ledger(u): return u / 1000.0"
        # Agree within pair
        self.assertAlmostEqual(pair_disagreement(cents_prod, cents_cons), 0.0)
        # Disagree across convention
        d = pair_disagreement(cents_prod, mils_cons)
        self.assertGreater(d, 0.5)


class TestShouldRefine(unittest.TestCase):
    def test_matching_pair_no_refine(self):
        d = should_refine(_REF_CHECK_HALFUP, [_REF_ROUND_HALFUP], threshold=0.34)
        self.assertFalse(d.refine)
        self.assertEqual(d.score, 0.0)

    def test_mismatched_pair_triggers_refine(self):
        d = should_refine(_BANKERS_CHECK, [_REF_ROUND_HALFUP], threshold=0.34)
        self.assertTrue(d.refine)
        self.assertGreater(d.score, 0.34)

    def test_no_signal_no_refine(self):
        # Unrelated sources — no pair match at all.
        d = should_refine(
            "def unrelated_a(x): return x",
            ["def unrelated_b(y): return y + 1"],
            threshold=0.34,
        )
        self.assertFalse(d.refine)
        self.assertEqual(d.score, 0.0)


if __name__ == "__main__":
    unittest.main()
