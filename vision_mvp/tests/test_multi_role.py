"""Tests for MultiRoleTeam and QuantTask — mock-client where possible."""
from __future__ import annotations
import unittest
from dataclasses import dataclass, field

from vision_mvp.core.multi_role_team import MultiRoleTeam, Role
from vision_mvp.core.llm_client import LLMStats
from vision_mvp.tasks.quant_strategy import QuantTask


@dataclass
class FakeClient:
    model: str = "fake"
    stats: LLMStats = field(default_factory=LLMStats)

    def generate(self, prompt, max_tokens=60, temperature=0.2):
        self.stats.n_generate_calls += 1
        self.stats.prompt_tokens += len(prompt) // 4
        self.stats.output_tokens += 10
        # Deterministic fake response
        return f"reply-{hash(prompt) % 1000}"


class TestMultiRoleTeam(unittest.TestCase):
    def _simple_team(self, client):
        return MultiRoleTeam(
            roles=[
                Role(name="research", persona="researcher",
                     prompt_template=lambda ins, p: f"{p}: analyze {ins[0]}",
                     n_agents=3, reads_from="raw"),
                Role(name="pm", persona="pm",
                     prompt_template=lambda ins, p: f"{p}: decide from {ins[0]}",
                     n_agents=1, reads_from="research"),
            ],
            raw_inputs_by_role={"research": ["input1", "input2", "input3"]},
            client=client,
        )

    def test_pipeline_runs(self):
        c = FakeClient()
        team = self._simple_team(c)
        out = team.run()
        self.assertEqual(len(out["research"]), 3)
        self.assertEqual(len(out["pm"]), 1)
        self.assertEqual(c.stats.n_generate_calls, 4)

    def test_invalid_reads_from(self):
        with self.assertRaises(ValueError):
            MultiRoleTeam(
                roles=[Role(name="a", persona="x",
                            prompt_template=lambda ins, p: "",
                            n_agents=1, reads_from="unknown_role")],
                raw_inputs_by_role={},
                client=FakeClient(),
            )

    def test_raw_inputs_length_mismatch(self):
        with self.assertRaises(ValueError):
            MultiRoleTeam(
                roles=[Role(name="a", persona="x",
                            prompt_template=lambda ins, p: "",
                            n_agents=5, reads_from="raw")],
                raw_inputs_by_role={"a": ["one", "two"]},   # 2 not 5
                client=FakeClient(),
            )

    def test_stats_after_run(self):
        c = FakeClient()
        team = self._simple_team(c)
        team.run()
        s = team.stats
        self.assertEqual(s["total_llm_generate_calls"], 4)
        self.assertEqual(s["per_role_output_count"]["research"], 3)
        self.assertEqual(s["per_role_output_count"]["pm"], 1)


class TestQuantTask(unittest.TestCase):
    def test_generate_shape(self):
        t = QuantTask(n_assets=20, n_research_notes=30, seed=1)
        t.generate()
        self.assertEqual(len(t.assets), 20)
        self.assertEqual(len(t.research_notes), 30)

    def test_each_asset_has_10_returns(self):
        t = QuantTask(n_assets=12, seed=2)
        t.generate()
        for a in t.assets:
            self.assertEqual(len(a.past_returns), 10)

    def test_regimes_cover_all_four(self):
        t = QuantTask(n_assets=16, seed=3)
        t.generate()
        regimes = {a.regime for a in t.assets}
        self.assertEqual(regimes, {"momentum", "mean_reversion", "event_pos", "event_neg"})

    def test_optimal_beats_random_baseline(self):
        t = QuantTask(n_assets=20, seed=5)
        t.generate()
        opt = t.optimal_score()
        rnd = t.random_baseline_score(seed=7)
        self.assertGreater(opt["hit_rate"], 0.95)   # optimal is perfect (1.0)
        self.assertGreater(opt["gross_return_pct"], rnd["gross_return_pct"])

    def test_score_portfolio_shape(self):
        t = QuantTask(n_assets=5, seed=1)
        t.generate()
        dirs = {a.ticker: "long" for a in t.assets}
        s = t.score_portfolio(dirs)
        for key in ("hit_rate", "n_correct", "n_wrong", "gross_return_pct",
                    "sharpe_proxy"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
