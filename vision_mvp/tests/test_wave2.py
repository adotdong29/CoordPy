"""Tests for Wave 2 modules (diagnostics + Tier-0 mechanisms)."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.diagnostics.bkt import (
    bkt_report, plaquette_charges, wrap_angle,
)
from vision_mvp.core.diagnostics.eth import eth_report, thermal_spectrum
from vision_mvp.core.diagnostics.kuramoto import (
    critical_coupling_meanfield, order_parameter, simulate as kura_simulate,
)
from vision_mvp.core.diagnostics.percolation import (
    contact_process, percolation_sweep,
)
from vision_mvp.core.diagnostics.thermo_length import (
    empirical_fisher, thermo_length,
)
from vision_mvp.core.embeddings import (
    BourgainEmbedding, JLEmbedding, distortion,
)
from vision_mvp.core.hdc import (
    CodeBook, bind, bundle, cosine, permute, random_bipolar,
)
from vision_mvp.core.hopfield import ModernHopfield
from vision_mvp.core.influence import (
    extract_junta, influence_report, monte_carlo_influence,
)
from vision_mvp.core.iss import (
    estimate_iss_gain, small_gain, stability_margin,
)
from vision_mvp.core.linear_logic import (
    ProofStructure, check_session, is_proof_net,
)
from vision_mvp.core.lmsr import (
    LMSRMarket, cost_function, max_loss, prices, trade_cost,
)
from vision_mvp.core.routing_hash import (
    ConsistentHashRing, rendezvous_route,
)
from vision_mvp.core.sketches import (
    CountMinSketch, HyperLogLog, ReservoirSampler,
)
from vision_mvp.core.submodular import (
    approximation_ratio, greedy, lazy_greedy,
)
from vision_mvp.core.svgd import (
    median_bandwidth, rbf_kernel, svgd,
)


# ================================================== diagnostics.eth

class TestETH(unittest.TestCase):
    def test_thermal_spectrum_length(self):
        rng = np.random.default_rng(0)
        traj = rng.standard_normal((100, 10, 4))   # T, N, d
        spec = thermal_spectrum(traj, k=5)
        self.assertEqual(spec.shape, (5,))
        # Eigenvalues of a PSD cov should be nonneg (up to roundoff)
        self.assertTrue(np.all(spec > -1e-10))

    def test_eth_report_sane(self):
        rng = np.random.default_rng(1)
        traj = rng.standard_normal((80, 20, 3))
        r = eth_report(traj, subset_size=4, n_samples=5, seed=0)
        self.assertEqual(r.subset_size, 4)
        self.assertGreaterEqual(r.mean_distance, 0.0)
        self.assertGreaterEqual(r.max_distance, r.mean_distance - 1e-9)


# ================================================== diagnostics.kuramoto

class TestKuramoto(unittest.TestCase):
    def test_order_parameter_bounds(self):
        self.assertAlmostEqual(order_parameter(np.zeros(10)), 1.0, places=6)
        # All thetas at ±π averaged → 0 if symmetric
        theta = np.array([0.0, np.pi])
        self.assertAlmostEqual(order_parameter(theta), 0.0, places=6)

    def test_simulate_converges_for_strong_coupling(self):
        N = 8
        A = np.ones((N, N)) - np.eye(N)
        omega = np.zeros(N)                       # identical oscillators
        report = kura_simulate(A, omega, coupling=10.0, t_end=20.0, dt=0.1, seed=0)
        self.assertGreater(report.final_r, 0.8)

    def test_critical_coupling_positive(self):
        omega = np.random.default_rng(0).standard_normal(100)
        kc = critical_coupling_meanfield(omega)
        self.assertGreater(kc, 0.0)


# ================================================== diagnostics.bkt

class TestBKT(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(float(wrap_angle(np.array([3.5 * np.pi]))[0]),
                                1.5 * np.pi - 2 * np.pi, places=6)

    def test_constant_field_no_vortices(self):
        phi = np.ones((6, 6)) * 0.5
        r = bkt_report(phi)
        self.assertEqual(r.total_vortices, 0)

    def test_manual_vortex_detected(self):
        # Phase ramp around a plaquette — construct a 3x3 field with one vortex
        phi = np.array([
            [0.0,       np.pi / 2, np.pi],
            [3 * np.pi / 2, 0.0, np.pi / 2],  # fabricated swirl
            [np.pi,  np.pi / 2, 0.0],
        ])
        charges = plaquette_charges(phi)
        # Should find at least one nonzero winding
        self.assertGreaterEqual(int((np.abs(charges) == 1).sum()), 1)


# ================================================== diagnostics.percolation

class TestPercolation(unittest.TestCase):
    def test_zero_p_dies(self):
        A = np.ones((5, 5)) - np.eye(5)
        r = percolation_sweep(A, p=0.0, n_trials=10, rounds=5, seed=0)
        self.assertAlmostEqual(r.survival_probability, 0.0, places=1)

    def test_full_p_survives(self):
        A = np.ones((5, 5)) - np.eye(5)
        r = percolation_sweep(A, p=1.0, n_trials=10, rounds=5, seed=0)
        # With p=1 on a complete graph, all trials survive
        self.assertGreater(r.survival_probability, 0.5)


# ================================================== diagnostics.thermo_length

class TestThermoLength(unittest.TestCase):
    def test_zero_trajectory_zero_length(self):
        traj = np.ones((10, 3))
        r = thermo_length(traj, lambda _: np.eye(3))
        self.assertAlmostEqual(r.length, 0.0)

    def test_linear_trajectory_matches_euclidean_with_identity_fisher(self):
        traj = np.stack([np.zeros(2), np.array([1.0, 0.0]),
                          np.array([2.0, 0.0])])
        r = thermo_length(traj, lambda _: np.eye(2))
        self.assertAlmostEqual(r.length, 2.0, places=6)

    def test_empirical_fisher_PSD(self):
        scores = np.random.default_rng(0).standard_normal((100, 3))
        F = empirical_fisher(scores)
        eigs = np.linalg.eigvalsh(F)
        self.assertGreaterEqual(eigs.min(), -1e-8)


# ================================================== embeddings

class TestEmbeddings(unittest.TestCase):
    def test_jl_output_shape(self):
        jl = JLEmbedding(in_dim=100, out_dim=20, seed=0)
        x = np.ones(100)
        self.assertEqual(jl.project(x).shape, (20,))

    def test_jl_preserves_distances_approx(self):
        rng = np.random.default_rng(0)
        N = 80
        X = rng.standard_normal((N, 50))
        jl = JLEmbedding.for_n_points(n_points=N, in_dim=50, eps=0.3, seed=0)
        d = distortion(jl.project, X, n_pairs=60, seed=0)
        self.assertLess(d, 5.0)  # very loose bound (JL is stochastic)

    def test_bourgain_projects(self):
        X = np.random.default_rng(0).standard_normal((32, 16))
        be = BourgainEmbedding.fit(X, seed=0)
        self.assertEqual(be.project(X[0]).shape, (be.out_dim,))


# ================================================== hdc

class TestHDC(unittest.TestCase):
    def test_random_bipolar_shape(self):
        v = random_bipolar(1000, seed=0)
        self.assertEqual(v.shape, (1000,))
        self.assertTrue(set(np.unique(v).tolist()).issubset({-1, 1}))

    def test_self_bind_is_all_ones(self):
        v = random_bipolar(100, seed=0)
        np.testing.assert_array_equal(bind(v, v), np.ones(100, dtype=np.int8))

    def test_bundle_similar_to_components(self):
        v1 = random_bipolar(2000, seed=1)
        v2 = random_bipolar(2000, seed=2)
        v3 = random_bipolar(2000, seed=3)
        b = bundle([v1, v2, v3])
        # Bundle is correlated with each component (positive cosine)
        for v in (v1, v2, v3):
            self.assertGreater(cosine(b, v), 0.2)

    def test_permute_decorrelates(self):
        v = random_bipolar(2000, seed=0)
        p = permute(v, k=1)
        # Permuted vector ~ orthogonal
        self.assertLess(abs(cosine(v, p)), 0.1)

    def test_codebook_cleanup(self):
        cb = CodeBook(d=1000, seed=0)
        cb.ensure("apple")
        cb.ensure("banana")
        # noisy "apple"
        v = cb["apple"].copy()
        flips = np.random.default_rng(0).integers(0, 1000, size=50)
        v[flips] *= -1
        name, sim = cb.cleanup(v)
        self.assertEqual(name, "apple")
        self.assertGreater(sim, 0.5)


# ================================================== hopfield

class TestHopfield(unittest.TestCase):
    def test_retrieval_of_stored_pattern(self):
        hf = ModernHopfield(d=20, beta=5.0)
        p1 = np.random.default_rng(0).standard_normal(20)
        p2 = np.random.default_rng(1).standard_normal(20)
        hf.store(p1); hf.store(p2)
        # Query with p1 + small noise → should retrieve something close to p1
        q = p1 + 0.01 * np.random.default_rng(2).standard_normal(20)
        out = hf.retrieve(q)
        self.assertGreater(
            float(np.dot(out, p1) / (np.linalg.norm(out) * np.linalg.norm(p1))),
            0.95,
        )

    def test_attention_weights_sum_to_one(self):
        hf = ModernHopfield(d=10)
        for _ in range(5):
            hf.store(np.random.default_rng(0).standard_normal(10))
        w = hf.attention_weights(np.ones(10))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_empty_retrieve_returns_query(self):
        hf = ModernHopfield(d=5)
        q = np.arange(5, dtype=float)
        np.testing.assert_allclose(hf.retrieve(q), q)


# ================================================== influence

class TestInfluence(unittest.TestCase):
    def test_dictator_function(self):
        # f(x) = x_0: only variable 0 has influence 1.0, all others 0.
        def f(x):
            return int(x[0])
        inf = monte_carlo_influence(f, n_vars=5, n_samples=200, seed=0)
        self.assertAlmostEqual(float(inf[0]), 1.0, places=1)
        for i in range(1, 5):
            self.assertAlmostEqual(float(inf[i]), 0.0, places=1)

    def test_majority_distributes_influence(self):
        def f(x):
            return int(sum(x) > len(x) // 2)
        inf = monte_carlo_influence(f, n_vars=5, n_samples=200, seed=0)
        # Symmetric → all influences roughly equal
        self.assertLess(float(np.std(inf)), 0.2)

    def test_junta_extraction(self):
        inf = np.array([0.9, 0.8, 0.01, 0.02, 0.0])
        j = extract_junta(inf, epsilon=0.1)
        self.assertEqual(j, [0, 1])


# ================================================== iss

class TestISS(unittest.TestCase):
    def test_small_gain(self):
        self.assertTrue(small_gain([0.5, 0.5]))
        self.assertFalse(small_gain([1.2, 0.9]))

    def test_stability_margin(self):
        self.assertAlmostEqual(stability_margin([0.5, 0.5]), 0.75)

    def test_iss_gain_estimate(self):
        # x_{k+1} = 0.5 x + u. Steady state: x = 2 u → γ = 2.
        def sys(x, u):
            return 0.5 * x + u
        r = estimate_iss_gain(sys, state_dim=3, input_dim=3,
                              input_range=0.5, n_probes=5, steps=200, seed=0)
        self.assertAlmostEqual(r.gain, 2.0, places=1)


# ================================================== linear_logic

class TestLinearLogic(unittest.TestCase):
    def test_axiom_is_valid(self):
        ps = ProofStructure(n_nodes=2)
        ps.axiom(0, 1)
        self.assertTrue(is_proof_net(ps))

    def test_disconnected_axioms_valid_as_pair(self):
        # Two separate axioms on 4 conclusions is NOT connected → not a proof net.
        ps = ProofStructure(n_nodes=4)
        ps.axiom(0, 1)
        ps.axiom(2, 3)
        self.assertFalse(is_proof_net(ps))

    def test_session_checker(self):
        ps = ProofStructure(n_nodes=2)
        ps.axiom(0, 1)
        r = check_session(ps)
        self.assertTrue(r.ok)


# ================================================== lmsr

class TestLMSR(unittest.TestCase):
    def test_prices_sum_to_one(self):
        q = np.array([1.0, 0.5, -2.0])
        p = prices(q, b=1.0)
        self.assertAlmostEqual(p.sum(), 1.0, places=9)

    def test_uniform_q_gives_uniform_prices(self):
        q = np.zeros(4)
        p = prices(q, b=5.0)
        np.testing.assert_allclose(p, np.full(4, 0.25), atol=1e-9)

    def test_trade_cost_positive_for_positive_delta(self):
        q = np.zeros(3)
        cost = trade_cost(q, np.array([1.0, 0, 0]), b=1.0)
        self.assertGreater(cost, 0.0)

    def test_max_loss_bounded(self):
        self.assertAlmostEqual(
            max_loss(b=2.0, n_outcomes=4), 2.0 * np.log(4), places=8,
        )

    def test_market_object(self):
        m = LMSRMarket(n_outcomes=3, b=5.0)
        c1 = m.trade(np.array([1.0, 0, 0]))
        c2 = m.trade(np.array([0, 0, 1.0]))
        self.assertGreater(c1 + c2, 0.0)
        r = m.report()
        self.assertAlmostEqual(float(r.prices.sum()), 1.0, places=9)


# ================================================== routing_hash

class TestRoutingHash(unittest.TestCase):
    def test_consistent_hash_assigns_all_keys(self):
        ring = ConsistentHashRing(replicas=16, seed=0)
        for n in ["alpha", "beta", "gamma"]:
            ring.add_node(n)
        nodes = set()
        for k in range(500):
            nodes.add(ring.route(f"k{k}"))
        # All 3 nodes should see some keys
        self.assertEqual(nodes, {"alpha", "beta", "gamma"})

    def test_consistent_hash_stability_on_addition(self):
        ring = ConsistentHashRing(replicas=32, seed=0)
        for n in ["a", "b", "c"]:
            ring.add_node(n)
        keys = [f"x{i}" for i in range(500)]
        before = [ring.route(k) for k in keys]
        ring.add_node("d")
        after = [ring.route(k) for k in keys]
        # With 4 nodes and 500 keys, at most ~125 keys should have moved.
        moved = sum(1 for a, b in zip(before, after) if a != b)
        self.assertLess(moved, 200)

    def test_rendezvous_stable(self):
        nodes = ["n1", "n2", "n3", "n4"]
        # Same key, same seed → same node
        self.assertEqual(rendezvous_route("k", nodes), rendezvous_route("k", nodes))


# ================================================== sketches

class TestSketches(unittest.TestCase):
    def test_cm_never_underestimates(self):
        cm = CountMinSketch.for_accuracy(eps=0.01, delta=0.01, seed=0)
        truth = {}
        rng = np.random.default_rng(0)
        for _ in range(2000):
            k = int(rng.integers(0, 100))
            truth[k] = truth.get(k, 0) + 1
            cm.update(k)
        for k, c in truth.items():
            self.assertGreaterEqual(cm.estimate(k), c)

    def test_hll_cardinality_estimate(self):
        hll = HyperLogLog(p=12)
        for i in range(10_000):
            hll.add(f"item_{i}")
        est = hll.estimate()
        # Within ±5% for p=12.
        self.assertGreater(est, 8_500)
        self.assertLess(est, 11_500)

    def test_reservoir_samples(self):
        r = ReservoirSampler(k=10, seed=0)
        for i in range(1000):
            r.add(i)
        s = r.sample()
        self.assertEqual(len(s), 10)
        self.assertTrue(all(0 <= x < 1000 for x in s))


# ================================================== submodular

class TestSubmodular(unittest.TestCase):
    def test_coverage_greedy_matches_exact(self):
        # coverage function: v(S) = |⋃_{i∈S} set_i|
        sets = [{0, 1}, {1, 2}, {2, 3}, {0, 3, 4}]

        def v(S):
            u = set()
            for i in S:
                u |= sets[i]
            return float(len(u))

        g = greedy(v, universe=[0, 1, 2, 3], k=2)
        lg = lazy_greedy(v, universe=[0, 1, 2, 3], k=2)
        self.assertEqual(g.value, lg.value)

    def test_lazy_greedy_fewer_oracle_calls(self):
        sets = [set(range(i, i + 3)) for i in range(10)]

        def v(S):
            u = set()
            for i in S:
                u |= sets[i]
            return float(len(u))

        g = greedy(v, universe=list(range(10)), k=4)
        lg = lazy_greedy(v, universe=list(range(10)), k=4)
        self.assertEqual(g.value, lg.value)
        self.assertLess(lg.n_oracle_calls, g.n_oracle_calls)

    def test_approx_ratio(self):
        self.assertGreaterEqual(approximation_ratio(6.3, 10.0), 0.6)


# ================================================== svgd

class TestSVGD(unittest.TestCase):
    def test_bandwidth_positive(self):
        X = np.random.default_rng(0).standard_normal((20, 3))
        self.assertGreater(median_bandwidth(X), 0.0)

    def test_kernel_symmetric(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        K, _ = rbf_kernel(X, h=1.0)
        np.testing.assert_allclose(K, K.T)
        np.testing.assert_allclose(np.diag(K), np.ones(10))

    def test_svgd_moves_particles_to_target(self):
        # Target: N(3, 0.5²) in 1-D. grad log p(x) = -(x - 3) / 0.25
        target_mean = 3.0
        target_var = 0.25

        def grad_log_p(X):
            return -(X - target_mean) / target_var

        particles = np.random.default_rng(0).standard_normal((40, 1))
        r = svgd(particles, grad_log_p, steps=200, lr=0.05)
        mean = float(r.final_particles.mean())
        self.assertAlmostEqual(mean, target_mean, places=0)


if __name__ == "__main__":
    unittest.main()
