"""Unit tests for the Phase-18 general trigger (LLM-judge + hybrid fallback).

These tests are intentionally task-agnostic: they exercise the trigger on
both ProtocolKit-style (dict-key) and NumericLedger-style (numerical-
convention) drafts, and they never spin up a real Ollama instance — the
LLM client is replaced with an in-memory fake."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from vision_mvp.core.general_trigger import (
    HybridStructuralTrigger, LLMJudgeTrigger, GeneralTrigger,
    _parse_score,
)
from vision_mvp.core.trigger import TriggerDecision, Trigger


# ---------- Test doubles for the LLMClient ---------------------------------

@dataclass
class _FakeStats:
    n_generate_calls: int = 0


@dataclass
class _ScriptedLLM:
    """Returns the next response from `responses` on every generate() call.
    If `raise_on_call`, raises the supplied exception instead."""
    responses: list[str] = field(default_factory=list)
    raise_on_call: Exception | None = None
    calls: list[dict] = field(default_factory=list)
    stats: _FakeStats = field(default_factory=_FakeStats)

    def generate(self, prompt, max_tokens=16, temperature=0.0):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                            "temperature": temperature})
        self.stats.n_generate_calls += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if not self.responses:
            return ""
        return self.responses.pop(0)


# ---------- Score parsing ---------------------------------------------------

class TestParseScore(unittest.TestCase):
    def test_plain_number(self):
        self.assertAlmostEqual(_parse_score("0.7"), 0.7)

    def test_with_text(self):
        self.assertAlmostEqual(_parse_score("Score: 0.42 (high)"), 0.42)

    def test_percentage_form(self):
        # "75" -> 0.75
        self.assertAlmostEqual(_parse_score("75"), 0.75)

    def test_no_number(self):
        self.assertIsNone(_parse_score("no number here"))

    def test_empty(self):
        self.assertIsNone(_parse_score(""))

    def test_out_of_range(self):
        self.assertIsNone(_parse_score("9999"))


# ---------- Hybrid structural trigger --------------------------------------

class TestHybridStructural(unittest.TestCase):
    def setUp(self):
        self.t = HybridStructuralTrigger()

    def test_satisfies_protocol(self):
        self.assertIsInstance(self.t, Trigger)

    def test_no_input_no_refine(self):
        d = self.t.should_refine("", [])
        self.assertFalse(d.refine)

    # ProtocolKit-style: dict-key drift
    def test_dict_key_drift_fires(self):
        own = 'def make_range(s, e): return {"start": s, "end": e}'
        bul = ['def make_range(s, e): return {"lo": s, "hi": e}']
        d = self.t.should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)
        self.assertGreater(d.info["components"]["dict_keys"], 0.34)

    def test_dict_key_match_skips(self):
        own = 'def f(): return {"start": 0, "end": 10}'
        bul = ['def g(r): return r["start"] + r["end"]']
        d = self.t.should_refine(own, bul, threshold=0.34)
        self.assertFalse(d.refine)

    # NumericLedger-style: convention drift via fuzz behavior
    def test_rounding_convention_drift_fires(self):
        # Half-up vs floor behave differently on 0.5
        halfup = """
import math
def round_amount(value, decimals):
    mult = 10 ** decimals
    x = value * mult
    if x >= 0:
        return math.floor(x + 0.5) / mult
    return -math.floor(-x + 0.5) / mult
"""
        floor = """
import math
def round_amount(value, decimals):
    mult = 10 ** decimals
    return math.floor(value * mult) / mult
"""
        d = self.t.should_refine(halfup, [floor], threshold=0.34)
        # Either fuzz_behavior or number_buckets should rise; combined max
        # must exceed threshold.
        self.assertTrue(
            d.refine,
            f"expected refine; components={d.info['components']}",
        )

    def test_scale_convention_drift_fires(self):
        cents = "def to_ledger(d): return int(d * 100)"
        mils = "def to_ledger(d): return int(d * 1000)"
        d = self.t.should_refine(cents, [mils], threshold=0.34)
        # number_buckets jaccard should fire (different OOM bucket).
        self.assertTrue(d.refine)

    def test_unrelated_functions_no_signal(self):
        own = 'def alpha(x): return x'
        bul = ['def beta(y): return y']
        d = self.t.should_refine(own, bul, threshold=0.34)
        # No co-defined function, no shared keys/strings/numbers -> low score
        self.assertFalse(d.refine)


# ---------- LLM-judge trigger ----------------------------------------------

class TestLLMJudge(unittest.TestCase):
    def test_high_score_refines(self):
        client = _ScriptedLLM(responses=["0.85"])
        t = LLMJudgeTrigger(client=client)
        d = t.should_refine("def f(): return 1", ["def g(): return 2"],
                             threshold=0.34)
        self.assertTrue(d.refine)
        self.assertAlmostEqual(d.score, 0.85)
        self.assertEqual(client.stats.n_generate_calls, 1)

    def test_low_score_skips(self):
        client = _ScriptedLLM(responses=["0.10"])
        t = LLMJudgeTrigger(client=client)
        d = t.should_refine("def f(): return 1", ["def g(): return 2"],
                             threshold=0.34)
        self.assertFalse(d.refine)
        self.assertAlmostEqual(d.score, 0.10)

    def test_unparseable_falls_back(self):
        fallback = HybridStructuralTrigger()
        client = _ScriptedLLM(responses=["I cannot judge."])
        t = LLMJudgeTrigger(client=client, fallback=fallback)
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        # Disjoint dict keys -> hybrid fires
        self.assertTrue(d.refine)
        self.assertEqual(d.info.get("reason"), "llm_parse_failed_fellback")

    def test_transport_error_falls_back(self):
        fallback = HybridStructuralTrigger()
        client = _ScriptedLLM(raise_on_call=ConnectionError("boom"))
        t = LLMJudgeTrigger(client=client, fallback=fallback)
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)
        self.assertIn("error", d.info)

    def test_unparseable_no_fallback_refines(self):
        # Safety bias: refine on uncertainty when there's no fallback.
        client = _ScriptedLLM(responses=["?"])
        t = LLMJudgeTrigger(client=client, fallback=None)
        d = t.should_refine("def f(): pass", ["def g(): pass"], threshold=0.34)
        self.assertTrue(d.refine)
        self.assertEqual(d.score, 0.5)

    def test_no_input_no_call(self):
        client = _ScriptedLLM(responses=["0.99"])
        t = LLMJudgeTrigger(client=client)
        d = t.should_refine("", [], threshold=0.34)
        self.assertFalse(d.refine)
        self.assertEqual(client.stats.n_generate_calls, 0)


# ---------- GeneralTrigger composition --------------------------------------

class TestGeneralTrigger(unittest.TestCase):
    def test_no_client_uses_hybrid_only(self):
        t = GeneralTrigger(client=None)
        self.assertEqual(t.name, "general")
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        # Hybrid fires on disjoint keys
        self.assertTrue(d.refine)

    def test_with_client_uses_judge(self):
        client = _ScriptedLLM(responses=["0.05"])
        t = GeneralTrigger(client=client)
        own = 'def f(): return {"start": 0}'
        # Hybrid would fire (disjoint keys), but judge says "agree"
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertFalse(d.refine)
        self.assertEqual(client.stats.n_generate_calls, 1)

    def test_judge_failure_falls_through_to_hybrid(self):
        client = _ScriptedLLM(raise_on_call=RuntimeError("ollama down"))
        t = GeneralTrigger(client=client)
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)


if __name__ == "__main__":
    unittest.main()
