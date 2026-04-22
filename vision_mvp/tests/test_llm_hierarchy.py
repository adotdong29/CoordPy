"""Tests for LLMHierarchy with mock client."""
from __future__ import annotations
import unittest
import numpy as np
from dataclasses import dataclass, field

from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.core.llm_hierarchy import LLMHierarchy
from vision_mvp.core.llm_client import LLMStats


@dataclass
class FakeClient:
    model: str = "fake"
    stats: LLMStats = field(default_factory=LLMStats)
    _embed_dim: int = 16

    def generate(self, prompt, max_tokens=60, temperature=0.2):
        self.stats.n_generate_calls += 1
        self.stats.prompt_tokens += max(1, len(prompt) // 4)
        self.stats.output_tokens += 10
        return f"verdict-{hash(prompt) % 100}"

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        self.stats.n_embed_calls += 1
        self.stats.embed_tokens += sum(len(t) // 4 for t in texts)
        out = []
        for t in texts:
            r = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(r.standard_normal(self._embed_dim).tolist())
        return out


def _make_team(n, personas_base, question):
    personas = [f"{personas_base}-persona-{i%3}" for i in range(n)]
    return LLMTeam(n_agents=n, personas=personas, question=question,
                   client=FakeClient())


class TestLLMHierarchy(unittest.TestCase):
    def test_construct_mismatch(self):
        t1 = _make_team(10, "s", "Q?")
        with self.assertRaises(ValueError):
            LLMHierarchy(sub_teams=[t1], sub_team_names=["security", "perf"],
                         orchestrator_client=FakeClient())

    def test_initialize_and_step(self):
        q = "What is the main issue?"
        sub_teams = [
            _make_team(20, "security", q),
            _make_team(20, "perf", q),
            _make_team(20, "correctness", q),
        ]
        h = LLMHierarchy(
            sub_teams=sub_teams,
            sub_team_names=["security", "perf", "correctness"],
            orchestrator_client=FakeClient(),
        )
        h.initialize()
        info = h.step()
        self.assertEqual(len(info["sub_team_reports"]), 3)
        self.assertIn("orchestrator_verdict", info)

    def test_stats(self):
        q = "Q?"
        subs = [_make_team(10, "a", q), _make_team(10, "b", q)]
        h = LLMHierarchy(sub_teams=subs, sub_team_names=["a", "b"],
                        orchestrator_client=FakeClient())
        h.initialize()
        for _ in range(2):
            h.step()
        s = h.stats()
        self.assertEqual(s["n_sub_teams"], 2)
        self.assertEqual(s["total_agents"], 20)
        self.assertEqual(s["rounds"], 2)
        # Orchestrator calls = rounds
        self.assertEqual(s["orchestrator_llm_generate_calls"], 2)


if __name__ == "__main__":
    unittest.main()
