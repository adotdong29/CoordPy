"""Unit tests for Phase-12 CASR components: causal footprint, router, token meter."""
from __future__ import annotations
import unittest

from vision_mvp.core.causal_footprint import (
    CausalFootprint,
    footprint_from_call_graph,
    random_footprint,
)
from vision_mvp.core.casr_router import (
    CASRRouter,
    RouterMessage,
    RoutingStats,
)
from vision_mvp.core.token_meter import (
    count_tokens,
    count_prompt,
    count_completion,
)


# -------------------- CausalFootprint

class TestCausalFootprint(unittest.TestCase):
    def test_item_added_is_present(self):
        fp = CausalFootprint(capacity=20, error_rate=0.01)
        fp.add("agent_3")
        fp.add("agent_7")
        self.assertIn("agent_3", fp)
        self.assertIn("agent_7", fp)
        self.assertEqual(len(fp), 2)

    def test_item_not_added_is_probably_absent(self):
        fp = CausalFootprint(capacity=100, error_rate=0.01)
        for i in range(10):
            fp.add(f"added_{i}")
        false_positives = 0
        trials = 100
        for i in range(trials):
            # Queries that were NOT added. Namespace is distinct from added set.
            if f"query_{i}" in fp:
                false_positives += 1
        # Accept <2% false positive rate.
        self.assertLess(false_positives / trials, 0.02)

    def test_footprint_from_call_graph_includes_forward_edges(self):
        # A calls B and C
        call_graph = {"A": ["B", "C"], "B": [], "C": [], "D": []}
        fp = footprint_from_call_graph(call_graph, "A")
        self.assertIn("B", fp)
        self.assertIn("C", fp)

    def test_footprint_from_call_graph_includes_reverse_edges(self):
        # X and Y call A; A calls nothing.
        call_graph = {"A": [], "X": ["A"], "Y": ["A", "Z"], "Z": []}
        fp = footprint_from_call_graph(call_graph, "A")
        self.assertIn("X", fp)
        self.assertIn("Y", fp)

    def test_footprint_from_call_graph_includes_self(self):
        call_graph = {"A": ["B"], "B": []}
        fp = footprint_from_call_graph(call_graph, "A")
        self.assertIn("A", fp)

    def test_random_footprint_has_right_cardinality(self):
        all_agents = [f"a{i}" for i in range(20)]
        fp = random_footprint(all_agents, size=5, seed=42, self_agent="a0")
        # Exactly `size` distinct members when the pool is large enough.
        self.assertEqual(len(fp), 5)
        # Self is always included.
        self.assertIn("a0", fp)

    def test_random_footprint_is_deterministic_with_seed(self):
        all_agents = [f"a{i}" for i in range(20)]
        fp1 = random_footprint(all_agents, size=6, seed=123, self_agent="a3")
        fp2 = random_footprint(all_agents, size=6, seed=123, self_agent="a3")
        self.assertEqual(fp1.members(), fp2.members())
        # Different seed should (with very high probability) differ.
        fp3 = random_footprint(all_agents, size=6, seed=999, self_agent="a3")
        self.assertNotEqual(fp1.members(), fp3.members())


# -------------------- CASRRouter

def _make_messages(sources: list[str], tokens: int = 10) -> list[RouterMessage]:
    return [RouterMessage(source_id=s, payload=f"from {s}", tokens=tokens)
            for s in sources]


class TestCASRRouter(unittest.TestCase):
    def setUp(self):
        # Build footprints: A's footprint = {A, B}; B's footprint = {A, B, C};
        # C's footprint = {B, C}; D's footprint = {D}.
        self.footprints = {}
        call_graph = {"A": ["B"], "B": ["C"], "C": [], "D": []}
        for aid in ["A", "B", "C", "D"]:
            self.footprints[aid] = footprint_from_call_graph(call_graph, aid)

    def test_full_delivers_everything(self):
        router = CASRRouter("full", self.footprints)
        msgs = _make_messages(["A", "B", "C", "D"])
        delivered, stats = router.route(msgs, recipient_id="A")
        # All non-self messages delivered.
        self.assertEqual(len(delivered), 3)  # B, C, D (not A)
        self.assertEqual(stats.delivered, 3)
        self.assertEqual(stats.dropped, 1)  # self-message dropped
        self.assertEqual(stats.delivered_tokens, 30)
        self.assertEqual(stats.dropped_tokens, 10)

    def test_casr_drops_out_of_footprint(self):
        router = CASRRouter("casr", self.footprints)
        # A's footprint is {A, B}; D is out of footprint.
        msgs = _make_messages(["D"])
        delivered, stats = router.route(msgs, recipient_id="A")
        self.assertEqual(len(delivered), 0)
        self.assertEqual(stats.delivered, 0)
        self.assertEqual(stats.dropped, 1)

    def test_casr_keeps_in_footprint(self):
        router = CASRRouter("casr", self.footprints)
        # A's footprint = {A, B}; B is in, D is out.
        msgs = _make_messages(["B", "D"])
        delivered, stats = router.route(msgs, recipient_id="A")
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].source_id, "B")
        self.assertEqual(stats.delivered, 1)
        self.assertEqual(stats.dropped, 1)

    def test_no_self_delivery(self):
        # Self-messages are dropped in all modes.
        msgs = _make_messages(["A", "A", "B"])
        for mode in ("full", "casr", "ablation"):
            if mode == "ablation":
                # Need an ablation footprint that happens to include B.
                all_agents = ["A", "B", "C", "D"]
                fps = {
                    "A": random_footprint(all_agents, size=3, seed=0,
                                          self_agent="A"),
                }
                # Force B in by rebuilding until present, or just add.
                fps["A"].add("B")
            else:
                fps = self.footprints
            router = CASRRouter(mode, fps)
            delivered, stats = router.route(msgs, recipient_id="A")
            for d in delivered:
                self.assertNotEqual(d.source_id, "A")

    def test_ablation_has_same_cardinality_as_casr_but_different_membership(self):
        call_graph = {f"a{i}": [f"a{(i+1) % 12}"] for i in range(12)}
        all_agents = list(call_graph.keys())
        # CASR footprint for a0: forward={a1}, reverse={a11}, self={a0} -> 3 items.
        casr_fp = footprint_from_call_graph(call_graph, "a0")
        casr_members = casr_fp.members()
        self.assertEqual(len(casr_members), 3)

        # Over many seeds, the random footprint should have the same cardinality
        # but differ in membership for most seeds.
        differing = 0
        total = 30
        for seed in range(total):
            rfp = random_footprint(all_agents, size=len(casr_fp),
                                   seed=seed, self_agent="a0")
            self.assertEqual(len(rfp), len(casr_fp))
            if rfp.members() != casr_members:
                differing += 1
        # The vast majority of random draws should NOT coincide with the
        # exact causal footprint (there are C(11,2) = 55 possible 2-subsets
        # of the non-self agents, only one of which matches).
        self.assertGreater(differing, total * 0.8)


# -------------------- Token meter

class TestTokenMeter(unittest.TestCase):
    def test_count_tokens_nonzero_for_nonempty(self):
        self.assertGreater(count_tokens("hello world"), 0)
        self.assertGreater(count_prompt("a prompt string"), 0)
        self.assertGreater(count_completion("a completion"), 0)

    def test_count_tokens_zero_for_empty(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_prompt(""), 0)
        self.assertEqual(count_completion(""), 0)

    def test_longer_string_more_tokens(self):
        short = "hi"
        long = "this is a considerably longer string with many more words " * 5
        self.assertGreater(count_tokens(long), count_tokens(short))


if __name__ == "__main__":
    unittest.main()
