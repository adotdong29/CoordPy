"""Tests for core.bus — the communication-accounting primitive."""
from __future__ import annotations
import unittest
from vision_mvp.core.bus import Bus, Message


class TestBus(unittest.TestCase):
    def test_empty_bus(self):
        b = Bus()
        self.assertEqual(b.total_tokens(), 0)
        self.assertEqual(b.n_messages(), 0)
        self.assertEqual(b.peak_per_agent_context(), 0)
        self.assertEqual(b.mean_per_agent_context(), 0.0)

    def test_send_accumulates_tokens(self):
        b = Bus()
        b.send(0, 1, payload_size=5, kind="x", round_idx=0)
        b.send(1, 2, payload_size=7, kind="y", round_idx=0)
        self.assertEqual(b.total_tokens(), 12)
        self.assertEqual(b.n_messages(), 2)

    def test_broadcast_receiver_none(self):
        b = Bus()
        b.send(0, None, payload_size=3, kind="bcast", round_idx=0)
        self.assertEqual(b.messages[0].receiver, None)

    def test_note_context_tracks_peak(self):
        b = Bus()
        b.note_context(0, 10)
        b.note_context(0, 5)   # should not lower peak
        b.note_context(0, 20)  # should raise peak
        self.assertEqual(b.peak_per_agent_context(), 20)

    def test_peak_across_agents(self):
        b = Bus()
        b.note_context(0, 3)
        b.note_context(1, 10)
        b.note_context(2, 5)
        self.assertEqual(b.peak_per_agent_context(), 10)

    def test_mean_context(self):
        b = Bus()
        b.note_context(0, 3)
        b.note_context(1, 9)
        self.assertAlmostEqual(b.mean_per_agent_context(), 6.0)

    def test_summary_keys(self):
        b = Bus()
        b.send(0, 1, 5, "x", 0)
        b.note_context(0, 5)
        s = b.summary()
        for k in ("total_tokens", "n_messages",
                  "peak_agent_context", "mean_agent_context"):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
