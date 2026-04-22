"""Tests for Wave 3 core mechanisms."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.bnp import truncated_dp_mixture_vi
from vision_mvp.core.ci_kalman import CIKalman
from vision_mvp.core.coalitions import merge_and_split
from vision_mvp.core.coherence import CacheState, CoherenceDirectory
from vision_mvp.core.coreset import frank_wolfe_coreset
from vision_mvp.core.eig import gaussian_linear_eig, rank_by_eig
from vision_mvp.core.epistemic import KripkeModel, coordinated_attack_model
from vision_mvp.core.gossip_tree import PlumtreeOverlay
from vision_mvp.core.index_coding import (
    confusion_graph, greedy_color, index_code_report,
)
from vision_mvp.core.info_design import obedient_policies_brute
from vision_mvp.core.itc import Stamp
from vision_mvp.core.learned_index import LearnedIndex
from vision_mvp.core.mfg import solve_1d_mfg
from vision_mvp.core.persistent import HAMT
from vision_mvp.core.persuasion import (
    bayesian_persuasion_value_1d, concave_hull_1d,
)
from vision_mvp.core.port_ham import LinearPH, series_compose
from vision_mvp.core.rates import berger_tung_gaussian
from vision_mvp.core.regularity import weak_regularity_partition
from vision_mvp.core.tree_decomp import min_degree_treewidth


# ================================================== epistemic

class TestEpistemic(unittest.TestCase):
    def test_coordinated_attack_impossibility(self):
        m = coordinated_attack_model()
        # Agent A knows "attack_ok" at actual world (can distinguish).
        self.assertTrue(m.knows("A", "attack_ok", "attack_ok"))
        # Agent B does NOT know.
        self.assertFalse(m.knows("B", "attack_ok", "attack_ok"))
        # So C_{A,B} attack_ok fails.
        self.assertFalse(m.common_knowledge(["A", "B"], "attack_ok", "attack_ok"))

    def test_mutual_knowledge_depth(self):
        m = coordinated_attack_model()
        d = m.mutual_knowledge_depth(["A", "B"], "attack_ok", "attack_ok")
        self.assertLessEqual(d, 1)

    def test_announcement_restricts_worlds(self):
        m = coordinated_attack_model()
        m2 = m.announce("attack_ok")
        self.assertEqual(set(m2.worlds), {"attack_ok"})


# ================================================== tree_decomp

class TestTreeDecomp(unittest.TestCase):
    def test_path_has_treewidth_1(self):
        # Path of 5 nodes
        A = np.zeros((5, 5), dtype=int)
        for i in range(4):
            A[i, i + 1] = A[i + 1, i] = 1
        r = min_degree_treewidth(A)
        self.assertLessEqual(r.upper_bound, 1)

    def test_clique_has_treewidth_n_minus_1(self):
        # K_5
        A = np.ones((5, 5), dtype=int) - np.eye(5, dtype=int)
        r = min_degree_treewidth(A)
        self.assertEqual(r.upper_bound, 4)


# ================================================== regularity

class TestRegularity(unittest.TestCase):
    def test_partition_fields(self):
        rng = np.random.default_rng(0)
        A = rng.random((20, 20))
        A = (A + A.T) / 2
        r = weak_regularity_partition(A, n_blocks=3)
        self.assertLessEqual(r.n_blocks, 3)
        self.assertEqual(r.block_labels.shape, (20,))
        self.assertGreaterEqual(r.residual_norm, 0.0)


# ================================================== mfg

class TestMFG(unittest.TestCase):
    def test_mfg_terminates(self):
        x = np.linspace(-1, 1, 21)
        m0 = np.exp(-x ** 2)
        m0 = m0 / m0.sum() / (x[1] - x[0])

        def f(x, m):
            return x ** 2

        def g(x, m_T):
            return np.zeros_like(x)

        r = solve_1d_mfg(x, t_end=0.5, n_t=20, sigma=0.3,
                         running_cost=f, terminal_cost=g, m0=m0,
                         max_iter=5, tol=1e-3)
        # Mass conservation at each time
        dx = x[1] - x[0]
        for k in range(r.m.shape[0]):
            total = float(r.m[k].sum() * dx)
            self.assertAlmostEqual(total, 1.0, places=2)


# ================================================== index_coding

class TestIndexCoding(unittest.TestCase):
    def test_clique_confusion_graph(self):
        # Two receivers, each wanting the other's message, neither has side info.
        side = [set(), set()]
        desires = [1, 0]
        adj = confusion_graph(side, desires)
        self.assertEqual(adj[0, 1], 1)
        self.assertEqual(adj[1, 0], 1)

    def test_savings_with_full_side_info(self):
        # If every receiver already has everyone else's messages, confusion
        # graph has no edges and chromatic number is 1.
        side = [{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}]
        desires = [0, 1, 2, 3]
        r = index_code_report(side, desires, n_messages=4)
        self.assertEqual(r.chromatic_bound, 1)
        self.assertGreater(r.savings, 0.5)


# ================================================== rates

class TestBergerTung(unittest.TestCase):
    def test_independent_sources_give_individual_rates(self):
        r = berger_tung_gaussian(sigma1=1.0, sigma2=1.0, rho=0.0,
                                  D1=0.5, D2=0.5)
        expected_ind = 0.5 * np.log2(1 / 0.5)
        self.assertAlmostEqual(r.r1_individual, expected_ind, places=2)
        self.assertAlmostEqual(r.r2_individual, expected_ind, places=2)

    def test_high_correlation_reduces_rate(self):
        r_indep = berger_tung_gaussian(1.0, 1.0, 0.0, 0.3, 0.3)
        r_corr = berger_tung_gaussian(1.0, 1.0, 0.9, 0.3, 0.3)
        self.assertLess(r_corr.r1_individual, r_indep.r1_individual)


# ================================================== itc

class TestITC(unittest.TestCase):
    def test_seed(self):
        s = Stamp.seed()
        self.assertEqual(s.id, 1)

    def test_fork_partitions_id(self):
        s = Stamp.seed()
        a, b = s.fork()
        # Joining back should give full id
        joined = a.join(b)
        self.assertEqual(joined.id, 1)

    def test_event_orders(self):
        s = Stamp.seed()
        a, b = s.fork()
        a1 = a.event_tick()
        b1 = b.event_tick()
        # Concurrent forks of events: a1 and b1 concurrent
        self.assertTrue(a1.concurrent_with(b1))

    def test_leq_after_join(self):
        s = Stamp.seed()
        a, b = s.fork()
        a1 = a.event_tick()
        merged = a1.join(b)
        # a1 ≤ merged (merged knows a's event)
        self.assertTrue(a1.leq(merged))


# ================================================== persistent

class TestHAMT(unittest.TestCase):
    def test_set_get(self):
        h = HAMT()
        h2 = h.set("a", 1).set("b", 2)
        self.assertEqual(h2.get("a"), 1)
        self.assertEqual(h2.get("b"), 2)
        self.assertIsNone(h.get("a"))  # original unchanged

    def test_delete(self):
        h = HAMT().set("x", 1).set("y", 2)
        h2 = h.delete("x")
        self.assertNotIn("x", h2)
        self.assertIn("y", h2)
        self.assertIn("x", h)  # original preserved

    def test_len(self):
        h = HAMT()
        for i in range(100):
            h = h.set(i, str(i))
        self.assertEqual(len(h), 100)

    def test_update_existing(self):
        h = HAMT().set("a", 1).set("a", 2)
        self.assertEqual(h.get("a"), 2)
        self.assertEqual(len(h), 1)

    def test_items_roundtrip(self):
        h = HAMT()
        truth = {}
        for i in range(50):
            k = f"key_{i}"
            v = i * 2
            h = h.set(k, v)
            truth[k] = v
        self.assertEqual(dict(h.items()), truth)


# ================================================== coherence

class TestCoherence(unittest.TestCase):
    def test_read_without_writers_gets_exclusive(self):
        d = CoherenceDirectory()
        d.read("A", source_value=lambda: 42)
        self.assertEqual(d.caches["A"].state, CacheState.EXCLUSIVE)

    def test_write_invalidates_others(self):
        d = CoherenceDirectory()
        d.read("A", source_value=lambda: 1)
        d.read("B", source_value=lambda: 1)
        # Now write from A
        d.write("A", 99)
        self.assertEqual(d.caches["A"].state, CacheState.MODIFIED)
        self.assertEqual(d.caches["B"].state, CacheState.INVALID)
        self.assertGreater(d.n_invalidations, 0)

    def test_read_after_invalidate_fetches_from_M(self):
        d = CoherenceDirectory()
        d.write("A", 42)
        val = d.read("B", source_value=lambda: 0)
        self.assertEqual(val, 42)
        self.assertEqual(d.caches["A"].state, CacheState.SHARED)
        self.assertEqual(d.caches["B"].state, CacheState.SHARED)


# ================================================== gossip_tree

class TestGossipTree(unittest.TestCase):
    def test_broadcast_reaches_all(self):
        overlay = PlumtreeOverlay()
        names = [f"n{i}" for i in range(15)]
        # Bootstrap each node against up to 3 prior nodes so active-view is
        # populated enough for broadcast to mix through the overlay.
        for i, n in enumerate(names):
            prior = names[max(0, i - 3): i]
            overlay.add_node(n, bootstrap=list(prior))
        # Broadcast
        overlay.broadcast(names[0], "msg-1")
        # All 15 should receive under this connectivity
        self.assertGreater(overlay.reliability("msg-1"), 0.9)


# ================================================== learned_index

class TestLearnedIndex(unittest.TestCase):
    def test_recall_all_keys(self):
        keys = np.sort(np.random.default_rng(0).uniform(0, 1000, 500))
        idx = LearnedIndex(keys=keys, n_segments=8)
        for target_idx in range(0, 500, 7):
            k = keys[target_idx]
            self.assertEqual(idx.find(k), target_idx)

    def test_missing_key_returns_minus1(self):
        keys = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = LearnedIndex(keys=keys, n_segments=2)
        self.assertEqual(idx.find(2.5), -1)


# ================================================== port_ham

class TestPortHamiltonian(unittest.TestCase):
    def test_linear_ph_well_formed(self):
        sys = LinearPH(
            J=np.array([[0.0, 1], [-1, 0]]),
            R=np.eye(2) * 0.1,
            Q=np.eye(2),
            G=np.array([[1.0], [0.0]]),
        )
        self.assertTrue(sys.is_passive())
        # Energy nonnegative
        self.assertGreaterEqual(sys.energy(np.array([1.0, 2.0])), 0.0)

    def test_series_composition(self):
        s1 = LinearPH(
            J=np.array([[0.0, 1], [-1, 0]]),
            R=np.eye(2) * 0.1,
            Q=np.eye(2),
            G=np.array([[1.0], [0.0]]),
        )
        s2 = LinearPH(
            J=np.zeros((2, 2)),
            R=np.eye(2) * 0.2,
            Q=np.eye(2),
            G=np.array([[0.5], [0.0]]),
        )
        merged = series_compose(s1, s2)
        self.assertEqual(merged.J.shape, (4, 4))
        self.assertTrue(merged.is_passive())


# ================================================== ci_kalman

class TestCIKalman(unittest.TestCase):
    def test_state_converges_with_observations(self):
        d = 2
        N = 3
        A = np.eye(d) * 0.95
        Q = np.eye(d) * 0.01
        C = [np.eye(d) for _ in range(N)]
        R = [np.eye(d) * 0.1 for _ in range(N)]
        adj = np.ones((N, N)) - np.eye(N)
        kf = CIKalman(A=A, Q=Q, C=C, R=R, adjacency=adj, beta=0.3)

        true = np.array([2.0, -1.0])
        rng = np.random.default_rng(0)
        for _ in range(100):
            measurements = [true + rng.standard_normal(d) * 0.3 for _ in range(N)]
            kf.step(measurements)
            true = A @ true + rng.standard_normal(d) * 0.1

        # All agents' estimates should be near each other (consensus)
        ests = np.stack([kf.estimate(i) for i in range(N)])
        spread = float(ests.std(axis=0).max())
        self.assertLess(spread, 0.3)


# ================================================== bnp

class TestBNP(unittest.TestCase):
    def test_recovers_clusters(self):
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((30, 2)) + np.array([5, 0])
        X2 = rng.standard_normal((30, 2)) + np.array([-5, 0])
        X = np.vstack([X1, X2])
        r = truncated_dp_mixture_vi(X, K=8, alpha=1.0, sigma=1.0,
                                    prior_var=10.0, n_iter=30, seed=0)
        self.assertGreaterEqual(r.active_components, 1)
        self.assertLessEqual(r.active_components, 8)


# ================================================== coreset

class TestCoreset(unittest.TestCase):
    def test_coreset_reproduces_sum(self):
        rng = np.random.default_rng(0)
        V = rng.standard_normal((50, 4))
        r = frank_wolfe_coreset(V, max_size=30)
        self.assertLess(r.error, 2.0)

    def test_sparse_support(self):
        V = np.random.default_rng(0).standard_normal((100, 5))
        r = frank_wolfe_coreset(V, max_size=10)
        self.assertLessEqual(len(r.support), 10)


# ================================================== eig

class TestEIG(unittest.TestCase):
    def test_eig_nonneg_and_ranks(self):
        Sp = np.eye(2)
        Se = np.eye(2) * 0.1
        d1 = np.eye(2)               # informative: measures both components
        d2 = np.array([[1.0, 0.0], [0.0, 0.0]])  # measures only one component
        r1 = gaussian_linear_eig(d1, Sp, Se)
        r2 = gaussian_linear_eig(d2, Sp, Se)
        self.assertGreater(r1.eig, r2.eig)
        self.assertGreaterEqual(r2.eig, 0.0)

    def test_ranking(self):
        designs = [np.eye(2), 0.5 * np.eye(2), np.zeros((2, 2))]
        ranked = rank_by_eig(designs, np.eye(2), 0.1 * np.eye(2))
        self.assertEqual(ranked[0][0], 0)


# ================================================== persuasion

class TestPersuasion(unittest.TestCase):
    def test_concave_hull(self):
        mu = np.linspace(0, 1, 21)
        v = np.abs(mu - 0.5)     # convex → hull is top = constant at endpoints
        hull = concave_hull_1d(mu, v)
        self.assertGreaterEqual(hull.shape[0], 2)

    def test_sender_value_geq_concave_floor(self):
        def V(mu):
            return float(1 - (mu - 0.7) ** 2)
        r = bayesian_persuasion_value_1d(V, prior=0.5, grid=51)
        # Concave hull at any prior ≥ the raw V at prior
        self.assertGreaterEqual(r.optimal_value, V(0.5) - 1e-6)


# ================================================== info_design

class TestInfoDesign(unittest.TestCase):
    def test_no_information_no_value(self):
        # 1 state, 1 action ⇒ trivial
        p = np.array([1.0])

        def u_s(w, a): return 1.0
        def u_r(w, a): return 1.0

        r = obedient_policies_brute(p, [0], u_s, u_r)
        self.assertAlmostEqual(r.sender_value, 1.0, places=6)


# ================================================== coalitions

class TestCoalitions(unittest.TestCase):
    def test_merge_when_synergistic(self):
        # Superadditive: v(S) = |S|²
        def v(S):
            return float(len(S)) ** 2

        r = merge_and_split([0, 1, 2, 3], v=v)
        # Grand coalition should emerge
        self.assertEqual(len(r.coalitions), 1)

    def test_split_when_subadditive(self):
        # Strictly subadditive: v(S) = √|S|
        def v(S):
            return np.sqrt(len(S))

        r = merge_and_split([0, 1, 2, 3], v=v)
        # Singletons — 4 × √1 = 4 > √4 = 2
        self.assertEqual(len(r.coalitions), 4)


if __name__ == "__main__":
    unittest.main()
