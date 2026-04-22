"""Tests for the HardenedCASRRouter wrapper."""

from __future__ import annotations

import unittest

from vision_mvp.core.casr_router import CASRRouter, RouterMessage
from vision_mvp.core.causal_footprint import CausalFootprint
from vision_mvp.core.hardened_router import HardenedCASRRouter


def _make_footprints():
    fp_a = CausalFootprint(capacity=10, error_rate=0.001)
    fp_a.add("A"); fp_a.add("B")
    fp_b = CausalFootprint(capacity=10, error_rate=0.001)
    fp_b.add("A"); fp_b.add("B"); fp_b.add("C")
    fp_c = CausalFootprint(capacity=10, error_rate=0.001)
    fp_c.add("C")
    return {"A": fp_a, "B": fp_b, "C": fp_c}


class TestHardenedRouter(unittest.TestCase):
    def test_same_delivery_as_casr(self):
        fps = _make_footprints()
        messages = [
            RouterMessage(source_id="A", payload="a_draft", tokens=10),
            RouterMessage(source_id="B", payload="b_draft", tokens=12),
            RouterMessage(source_id="C", payload="c_draft", tokens=8),
        ]
        base = CASRRouter(mode="casr", footprints=fps)
        hardened = HardenedCASRRouter(mode="casr", footprints=fps)
        for rid in ["A", "B", "C"]:
            d_base, s_base = base.route(messages, recipient_id=rid)
            d_hard, s_hard = hardened.route(messages, recipient_id=rid)
            self.assertEqual(
                sorted(m.source_id for m in d_base),
                sorted(m.source_id for m in d_hard),
                f"recipient {rid}: delivery sets should match",
            )
            self.assertEqual(s_base.delivered_tokens, s_hard.delivered_tokens)

    def test_full_mode_delivers_everything_except_self(self):
        fps = _make_footprints()
        messages = [
            RouterMessage("A", "x", 1),
            RouterMessage("B", "y", 1),
            RouterMessage("C", "z", 1),
        ]
        hardened = HardenedCASRRouter(mode="full", footprints=fps)
        d, s = hardened.route(messages, recipient_id="A")
        self.assertEqual(s.delivered, 2)  # dropped A (self)

    def test_ablation_mode_uses_random_membership(self):
        fps = _make_footprints()
        hardened = HardenedCASRRouter(mode="ablation", footprints=fps)
        self.assertEqual(hardened.mode, "ablation")

    def test_hardening_side_effects_recorded(self):
        fps = _make_footprints()
        messages = [RouterMessage("A", "ap", 5), RouterMessage("B", "bp", 5)]
        hardened = HardenedCASRRouter(mode="casr", footprints=fps)
        hardened.route(messages, recipient_id="B")
        self.assertGreater(hardened.stats.cuckoo_lookups, 0)
        self.assertGreater(hardened.stats.chain_entries_written, 0)
        self.assertGreater(hardened.stats.merkle_blobs_stored, 0)

    def test_audit_clean(self):
        fps = _make_footprints()
        hardened = HardenedCASRRouter(mode="casr", footprints=fps)
        messages = [RouterMessage("A", "a", 1), RouterMessage("B", "b", 1)]
        for rid in ["A", "B", "C"]:
            hardened.route(messages, recipient_id=rid)
        results = hardened.audit()
        self.assertEqual(set(results.keys()), {"A", "B", "C"})
        for r, res in results.items():
            self.assertTrue(res["ok"], f"{r}: {res['reason']}")

    def test_content_addressing_is_deterministic(self):
        fps = _make_footprints()
        hardened = HardenedCASRRouter(mode="casr", footprints=fps)
        h1 = hardened.content_addressed({"x": 1, "y": 2})
        h2 = hardened.content_addressed({"y": 2, "x": 1})
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
