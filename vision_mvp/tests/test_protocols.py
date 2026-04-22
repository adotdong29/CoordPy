"""Integration tests — each protocol runs end-to-end correctly."""
from __future__ import annotations
import math
import unittest
import numpy as np

from vision_mvp.tasks.consensus import ConsensusTask
from vision_mvp.tasks.drifting_consensus import DriftingConsensus
from vision_mvp.protocols.naive import run_naive
from vision_mvp.protocols.gossip import run_gossip
from vision_mvp.protocols.manifold_only import run_manifold_only
from vision_mvp.protocols.full_stack import run_full
from vision_mvp.protocols.adaptive import run_adaptive, run_naive_drift
from vision_mvp.protocols.hierarchical import run_hierarchical
from vision_mvp.protocols.holographic import run_holographic


def make_static_task(n=10, d=16, rank=None, seed=0):
    rank = rank or max(2, math.ceil(math.log2(max(n, 2))))
    t = ConsensusTask(n_agents=n, dim=d, noise=1.0, seed=seed, intrinsic_rank=rank)
    t.generate()
    return t


def make_drift_task(n=10, d=16, rank=None, n_steps=20, seed=0):
    rank = rank or max(2, math.ceil(math.log2(max(n, 2))))
    t = DriftingConsensus(n_agents=n, dim=d, intrinsic_rank=rank,
                          n_steps=n_steps, noise=1.0, drift_sigma=0.05, seed=seed)
    t.generate()
    return t


class TestStaticProtocols(unittest.TestCase):
    """All four static-task protocols run and produce finite metrics."""

    def test_naive_runs(self):
        t = make_static_task()
        r = run_naive(t)
        self.assertGreater(r.bus_summary["total_tokens"], 0)
        self.assertTrue(np.isfinite(r.task_metrics["mean_accuracy_error"]))

    def test_gossip_runs(self):
        t = make_static_task()
        r = run_gossip(t)
        self.assertGreater(r.bus_summary["total_tokens"], 0)
        self.assertTrue(np.isfinite(r.task_metrics["mean_accuracy_error"]))

    def test_manifold_only_runs(self):
        t = make_static_task()
        r = run_manifold_only(t)
        self.assertGreater(r.bus_summary["total_tokens"], 0)
        self.assertTrue(np.isfinite(r.task_metrics["mean_accuracy_error"]))

    def test_full_stack_runs(self):
        t = make_static_task()
        r = run_full(t)
        self.assertGreater(r.bus_summary["total_tokens"], 0)
        self.assertTrue(np.isfinite(r.task_metrics["mean_accuracy_error"]))


class TestScalingLaws(unittest.TestCase):
    """The core scientific claims — must hold on every run, not on average."""

    def test_full_stack_peak_context_is_log_n(self):
        # Phase 1 claim: peak per-agent context = O(log N) for full stack.
        for n in (50, 200, 1000):
            t = make_static_task(n=n)
            r = run_full(t)
            peak = r.bus_summary["peak_agent_context"]
            expected = max(2, math.ceil(math.log2(n)))
            # Peak should be exactly the manifold dimension (m)
            self.assertEqual(
                peak, expected,
                f"N={n}: peak {peak} != expected log2(N)={expected}")

    def test_naive_peak_context_is_linear_n(self):
        # Naive: peak = (N-1) * (d+1). Should grow linearly.
        t1 = make_static_task(n=10, d=16)
        t2 = make_static_task(n=40, d=16)
        r1 = run_naive(t1)
        r2 = run_naive(t2)
        # Ratio should be ~4x (linear scaling)
        ratio = r2.bus_summary["peak_agent_context"] / r1.bus_summary["peak_agent_context"]
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 5.0)

    def test_full_stack_total_tokens_less_than_naive(self):
        # Full stack beats naive by a wide margin at moderate N.
        for n in (50, 200):
            t = make_static_task(n=n)
            r_naive = run_naive(t)
            r_full = run_full(t)
            self.assertLess(r_full.bus_summary["total_tokens"],
                            r_naive.bus_summary["total_tokens"] * 0.5,
                            f"N={n}: full stack not < half of naive tokens")

    def test_full_stack_at_least_matches_naive_accuracy(self):
        # On low-rank tasks, full stack should have ≤ error than naive oracle.
        for n in (50, 200):
            t = make_static_task(n=n)
            r_full = run_full(t)
            # Full stack error ≤ oracle (loose bound)
            self.assertLessEqual(
                r_full.task_metrics["mean_accuracy_error"],
                r_full.task_metrics["oracle_error"] + 0.01,
                f"N={n}: full stack worse than oracle by >0.01")


class TestDriftingProtocols(unittest.TestCase):
    def test_adaptive_runs(self):
        t = make_drift_task(n=20, d=16, n_steps=15)
        r = run_adaptive(t, surprise_tau=0.03, decay=0.7, pca_lr=0.1)
        self.assertTrue(np.isfinite(r.task_metrics["mean_tracking_error"]))
        self.assertEqual(len(r.writes_per_round), t.n_steps)

    def test_naive_drift_runs(self):
        t = make_drift_task(n=10, d=16, n_steps=10)
        r = run_naive_drift(t)
        self.assertTrue(np.isfinite(r.task_metrics["mean_tracking_error"]))

    def test_hierarchical_runs(self):
        t = make_drift_task(n=20, d=16, n_steps=15)
        r = run_hierarchical(t, surprise_tau=0.5, decay=0.8, pca_lr=0.1,
                              pred_hidden=4, seed=0)
        self.assertTrue(np.isfinite(r.task_metrics["mean_tracking_error"]))
        self.assertGreater(r.workspace_size, 0)

    def test_hierarchical_writes_bounded_by_workspace(self):
        # Claim: writes/round never exceeds workspace capacity.
        t = make_drift_task(n=50, d=16, n_steps=20)
        r = run_hierarchical(t, surprise_tau=0.5, seed=0)
        for w in r.writes_per_round:
            self.assertLessEqual(w, r.workspace_size)

    def test_holographic_runs_and_respects_boundary(self):
        t = make_drift_task(n=100, d=16, n_steps=15)
        r = run_holographic(t, surprise_tau=0.5, seed=0)
        self.assertGreater(r.boundary_size, 0)
        self.assertLess(r.boundary_size, t.n_agents)
        # Writes come only from boundary, so bounded by workspace.
        for w in r.writes_per_round:
            self.assertLessEqual(w, r.workspace_size)


class TestDeterminismUnderSeed(unittest.TestCase):
    """Fixed-seed regression tests — protocols must be reproducible."""

    def test_full_stack_deterministic(self):
        t1 = make_static_task(seed=42)
        t2 = make_static_task(seed=42)
        r1 = run_full(t1)
        r2 = run_full(t2)
        self.assertEqual(r1.bus_summary["total_tokens"],
                         r2.bus_summary["total_tokens"])

    def test_hierarchical_deterministic(self):
        t1 = make_drift_task(seed=7, n_steps=12)
        t2 = make_drift_task(seed=7, n_steps=12)
        r1 = run_hierarchical(t1, seed=0)
        r2 = run_hierarchical(t2, seed=0)
        self.assertEqual(r1.bus_summary["total_tokens"],
                         r2.bus_summary["total_tokens"])


if __name__ == "__main__":
    unittest.main()
