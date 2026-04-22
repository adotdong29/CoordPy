"""Tests for LLMTeam using a mock LLM client.

Don't require a running Ollama — use a fake client that produces deterministic
text and embeddings.
"""
from __future__ import annotations
import unittest
import numpy as np
from dataclasses import dataclass, field

from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.core.llm_client import LLMStats


@dataclass
class FakeClient:
    """Mock LLMClient returning deterministic outputs."""
    model: str = "fake"
    stats: LLMStats = field(default_factory=LLMStats)
    _embed_dim: int = 32
    _rng: np.random.Generator = None  # type: ignore

    def __post_init__(self):
        self._rng = np.random.default_rng(0)

    def generate(self, prompt: str, max_tokens: int = 60,
                 temperature: float = 0.2) -> str:
        self.stats.n_generate_calls += 1
        self.stats.prompt_tokens += max(1, len(prompt) // 4)
        self.stats.output_tokens += 10
        # Deterministic: first few chars of prompt as answer
        return f"ans-{hash(prompt) % 100}"

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.stats.n_embed_calls += 1
        self.stats.embed_tokens += sum(len(t) // 4 for t in texts)
        out = []
        for t in texts:
            # Use hash as seed for reproducible embeddings
            seeded = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(seeded.standard_normal(self._embed_dim).tolist())
        return out


class TestLLMTeam(unittest.TestCase):
    def test_initialize(self):
        personas = ["persona A", "persona B"] * 10  # 20 agents, 2 archetypes
        team = LLMTeam(n_agents=20, personas=personas, question="Is X > Y?",
                       client=FakeClient())
        team.initialize()
        # Archetype calls = #unique personas
        self.assertEqual(team.client.stats.n_generate_calls, 2)
        # One batch embed call
        self.assertEqual(team.client.stats.n_embed_calls, 1)
        self.assertEqual(team.embeddings.shape, (20, 32))

    def test_step_llm_calls_bounded_by_workspace(self):
        personas = [f"persona {i%3}" for i in range(30)]
        team = LLMTeam(n_agents=30, personas=personas, question="Q?",
                       client=FakeClient(), surprise_tau=-1.0)
        team.initialize()
        calls_before = team.client.stats.n_generate_calls
        team.step()
        delta_gen = team.client.stats.n_generate_calls - calls_before
        # Per round we expect at most workspace_size LLM generate calls.
        self.assertLessEqual(delta_gen, team.workspace.capacity())

    def test_non_admitted_embeddings_drift_toward_centroid(self):
        personas = [f"p{i%4}" for i in range(50)]
        team = LLMTeam(n_agents=50, personas=personas, question="Q?",
                       client=FakeClient(), blend_alpha=0.5)
        team.initialize()
        # Run one step
        pre = team.embeddings.copy()
        team.step()
        # At least some embeddings should have changed due to drift
        changed = np.any(~np.isclose(team.embeddings, pre), axis=1)
        self.assertGreater(int(changed.sum()), 0)

    def test_workspace_equals_log2_n(self):
        import math
        for n in (50, 200, 1000):
            personas = [f"p{i%5}" for i in range(n)]
            team = LLMTeam(n_agents=n, personas=personas, question="Q?",
                           client=FakeClient())
            self.assertEqual(team.workspace.capacity(),
                             max(1, math.ceil(math.log2(n))))

    def test_stats_after_round(self):
        personas = [f"p{i%3}" for i in range(20)]
        team = LLMTeam(n_agents=20, personas=personas, question="Q?",
                       client=FakeClient())
        team.initialize()
        for _ in range(3):
            team.step()
        s = team.stats()
        self.assertEqual(s["rounds"], 3)
        self.assertGreater(s["llm_generate_calls"], 0)

    def test_nearest_neighbor_texts(self):
        personas = [f"p{i%3}" for i in range(30)]
        team = LLMTeam(n_agents=30, personas=personas, question="Q?",
                       client=FakeClient())
        team.initialize()
        team.step()
        texts = team.nearest_neighbor_texts()
        self.assertEqual(len(texts), 30)

    def test_synthesize_produces_text(self):
        personas = [f"p{i%3}" for i in range(20)]
        team = LLMTeam(n_agents=20, personas=personas, question="Q?",
                       client=FakeClient())
        team.initialize()
        team.step()
        team.step()
        framing = "Combine these reviews into one:"
        result = team.synthesize(framing, max_tokens=50)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_per_agent_context_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            LLMTeam(n_agents=5, personas=["p"] * 5, question="Q?",
                    client=FakeClient(), per_agent_context=["c"] * 3)

    def test_per_agent_context_treats_each_as_archetype(self):
        # All personas identical, but chunks differ -> every agent has its own archetype
        personas = ["same"] * 8
        chunks = [f"chunk {i}" for i in range(8)]
        team = LLMTeam(n_agents=8, personas=personas, question="Q?",
                       client=FakeClient(), per_agent_context=chunks)
        team.initialize()
        # 8 distinct archetype LLM calls
        self.assertEqual(team.client.stats.n_generate_calls, 8)

    def test_synthesize_before_steps_no_crash(self):
        # Before any step() is called, per_round_admitted is empty —
        # synthesize should gracefully return empty string.
        personas = [f"p{i%2}" for i in range(10)]
        team = LLMTeam(n_agents=10, personas=personas, question="Q?",
                       client=FakeClient())
        team.initialize()
        result = team.synthesize("combine", max_tokens=30)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
