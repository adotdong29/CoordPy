"""Tests for HierarchicalRouter — the two-level CASR composition."""
from __future__ import annotations
import math
import unittest
import numpy as np

from vision_mvp import CASRRouter
from vision_mvp.core.hierarchical_router import HierarchicalRouter


def _make_router(n_workers: int = 5, worker_size: int = 100,
                 state_dim: int = 32) -> HierarchicalRouter:
    workers = [CASRRouter(n_agents=worker_size, state_dim=state_dim,
                          task_rank=6, seed=i)
               for i in range(n_workers)]
    orchestrator = CASRRouter(n_agents=n_workers, state_dim=state_dim,
                              task_rank=2, seed=99)
    return HierarchicalRouter(worker_teams=workers, orchestrator=orchestrator)


class TestHierarchicalRouter(unittest.TestCase):
    def test_step_shape(self):
        r = _make_router(n_workers=4, worker_size=50, state_dim=16)
        obs = [np.random.default_rng(i).standard_normal((50, 16)) for i in range(4)]
        out = r.step(obs)
        self.assertEqual(out.shape, (4, 16))

    def test_step_wrong_count_raises(self):
        r = _make_router(n_workers=3, worker_size=20, state_dim=8)
        with self.assertRaises(ValueError):
            r.step([np.zeros((20, 8))])   # only 1 team's worth of obs

    def test_construct_state_dim_mismatch(self):
        w = [CASRRouter(n_agents=20, state_dim=16) for _ in range(3)]
        o = CASRRouter(n_agents=3, state_dim=8)
        with self.assertRaises(ValueError):
            HierarchicalRouter(worker_teams=w, orchestrator=o)

    def test_orchestrator_size_mismatch(self):
        w = [CASRRouter(n_agents=20, state_dim=16) for _ in range(3)]
        o = CASRRouter(n_agents=5, state_dim=16)  # should be 3
        with self.assertRaises(ValueError):
            HierarchicalRouter(worker_teams=w, orchestrator=o)

    def test_peak_context_is_log_of_largest_level(self):
        # Peak is capped by the smaller of task_rank and ⌈log₂ N⌉.
        # Workers: task_rank=6, N=100 → manifold_dim=6.
        # Orchestrator: task_rank=2, N=5 → manifold_dim=2.
        # So expected peak is 6 (the worker-level value).
        r = _make_router(n_workers=5, worker_size=100, state_dim=16)
        obs = [np.random.default_rng(i).standard_normal((100, 16))
               for i in range(5)]
        for _ in range(2):
            r.step(obs)
        peak = r.stats["peak_context_per_agent"]
        worker_expected = min(6, math.ceil(math.log2(100)))  # task_rank vs log N
        orch_expected = min(2, math.ceil(math.log2(5)))
        self.assertEqual(peak, max(worker_expected, orch_expected))

    def test_inter_level_tokens_scale_with_n_workers(self):
        r4 = _make_router(n_workers=4, worker_size=20, state_dim=8)
        r8 = _make_router(n_workers=8, worker_size=20, state_dim=8)
        for r, nw in [(r4, 4), (r8, 8)]:
            obs = [np.random.default_rng(i).standard_normal((20, 8))
                   for i in range(nw)]
            r.step(obs)
        self.assertGreater(r8.stats["inter_level_tokens"],
                           r4.stats["inter_level_tokens"])

    def test_deterministic_given_seed(self):
        r1 = _make_router(n_workers=3, worker_size=30, state_dim=8)
        r2 = _make_router(n_workers=3, worker_size=30, state_dim=8)
        obs = [np.random.default_rng(i).standard_normal((30, 8)) for i in range(3)]
        out1 = r1.step(obs)
        out2 = r2.step(obs)
        np.testing.assert_allclose(out1, out2)

    def test_stats_keys(self):
        r = _make_router(n_workers=3, worker_size=20, state_dim=8)
        obs = [np.random.default_rng(i).standard_normal((20, 8)) for i in range(3)]
        r.step(obs)
        s = r.stats
        for k in ("peak_context_per_agent", "total_tokens", "n_worker_teams",
                  "worker_workspace", "orchestrator_workspace",
                  "rounds_executed", "inter_level_tokens"):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
