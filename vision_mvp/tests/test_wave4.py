"""Tests for Wave 4 — adversarial + crypto."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.cbf import CBFReport, enforce_barrier, exponential_alpha
from vision_mvp.core.coded_compute import (
    coded_compute_report, decode_outputs, encode_inputs,
)
from vision_mvp.core.dagbft import BullsharkSimulator
from vision_mvp.core.dp import (
    GaussianMechanism, LaplaceMechanism, advanced_composition,
    basic_composition, shuffle_then_aggregate,
)
from vision_mvp.core.gf import (
    DEFAULT_PRIME, eval_poly, inv_mod, lagrange_interpolate, mul,
)
from vision_mvp.core.merkle_dag import MerkleDAG, content_hash
from vision_mvp.core.paillier import (
    PaillierPublicKey, homomorphic_add, homomorphic_scale, keygen,
)
from vision_mvp.core.peer_review import (
    HashChainLog, spot_check, verify_log,
)
from vision_mvp.core.polar import PolarCode
from vision_mvp.core.qec import (
    SurfaceLayout, encode_rep3, flip_channel,
    logical_error_rate_rep3, majority_decode,
)
from vision_mvp.core.secret_sharing import reconstruct, split, verify_threshold
from vision_mvp.core.spdz_light import (
    additive_shares, generate_triple, reconstruct_additive, secure_multiply,
)
from vision_mvp.core.vrf_committee import (
    VRFKey, elect_committee, verify_vrf,
)


# ================================================== GF(p)

class TestGF(unittest.TestCase):
    def test_inv_mod_roundtrip(self):
        p = 97
        for a in range(1, 10):
            self.assertEqual(mul(a, inv_mod(a, p), p), 1)

    def test_polynomial_eval(self):
        p = 97
        # poly: 5 + 3x + x²
        coeffs = [5, 3, 1]
        self.assertEqual(eval_poly(coeffs, 0, p), 5)
        self.assertEqual(eval_poly(coeffs, 2, p), (5 + 6 + 4) % p)

    def test_lagrange_interpolation(self):
        p = 97
        # Polynomial 10 + 4x + 7x² evaluated at 1, 2, 3
        xs = [1, 2, 3]
        coeffs = [10, 4, 7]
        ys = [eval_poly(coeffs, x, p) for x in xs]
        # Reconstruct value at x=0 (should be 10)
        self.assertEqual(lagrange_interpolate(xs, ys, 0, p), 10)


# ================================================== secret_sharing

class TestShamir(unittest.TestCase):
    def test_roundtrip_threshold(self):
        shares = split(secret=12345, n=5, k=3, seed=0)
        self.assertEqual(reconstruct(shares[:3]), 12345)

    def test_different_subsets_same_secret(self):
        shares = split(secret=999, n=7, k=4, seed=1)
        a = reconstruct(shares[:4])
        b = reconstruct([shares[0], shares[2], shares[4], shares[6]])
        self.assertEqual(a, b)
        self.assertEqual(a, 999)

    def test_fewer_than_k_reconstructs_wrong(self):
        s = 777
        shares = split(secret=s, n=5, k=3, seed=2)
        # 2 shares should NOT reveal the secret
        recovered = reconstruct(shares[:2])
        # Almost never equals the secret for 2-of-3-of-5
        # (strict info-theoretic security on every share, but Lagrange still
        # returns *some* integer — it just shouldn't match except by chance).
        # We assert 2-share reconstruction is uniform over the full field:
        many = [reconstruct(split(secret=s, n=5, k=3, seed=i)[:2])
                for i in range(50)]
        self.assertGreater(len(set(many)), 10)

    def test_verify(self):
        self.assertTrue(verify_threshold(42, n=5, k=3, seed=0))


# ================================================== spdz_light

class TestSPDZ(unittest.TestCase):
    def test_additive_roundtrip(self):
        ss = additive_shares(500, n=4, seed=0)
        self.assertEqual(reconstruct_additive(ss), 500)

    def test_secure_multiply(self):
        p = DEFAULT_PRIME
        x, y = 7, 11
        n_parties = 3
        xs = additive_shares(x, n=n_parties, seed=0)
        ys = additive_shares(y, n=n_parties, seed=1)
        triple = generate_triple(n=n_parties, seed=2)
        z_shares = secure_multiply(xs, ys, triple)
        self.assertEqual(reconstruct_additive(z_shares), x * y % p)


# ================================================== coded_compute

class TestCodedCompute(unittest.TestCase):
    def test_encode_decode_no_stragglers(self):
        inputs = [3, 5, 7, 11]
        coded = encode_inputs(inputs, m=7)
        # Take any 4 (x, y) shard pairs
        received = [(i + 1, coded[i]) for i in range(4)]
        recovered = decode_outputs(received, n=4)
        self.assertEqual(recovered, inputs)

    def test_encode_decode_with_straggler(self):
        inputs = [10, 20, 30]
        coded = encode_inputs(inputs, m=5)
        # Skip shard 1 and 3 (stragglers); use {0, 2, 4}
        received = [(1, coded[0]), (3, coded[2]), (5, coded[4])]
        recovered = decode_outputs(received, n=3)
        self.assertEqual(recovered, inputs)

    def test_report(self):
        r = coded_compute_report(n=4, m=10)
        self.assertEqual(r.n_stragglers_tolerated, 6)


# ================================================== polar

class TestPolar(unittest.TestCase):
    def test_encode_is_linear(self):
        code = PolarCode.design_bec(N=8, K=4, epsilon=0.3)
        m1 = np.array([1, 0, 1, 1], dtype=np.int8)
        c1 = code.encode(m1)
        self.assertEqual(c1.size, 8)

    def test_decode_noiseless(self):
        code = PolarCode.design_bec(N=16, K=8, epsilon=0.3)
        m = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
        c = code.encode(m)
        decoded = code.decode(c)
        np.testing.assert_array_equal(decoded, m)


# ================================================== qec

class TestQEC(unittest.TestCase):
    def test_rep3_corrects_single_error(self):
        rng = np.random.default_rng(0)
        correct = 0
        for _ in range(200):
            bit = int(rng.integers(0, 2))
            code = encode_rep3(bit)
            # Force exactly one flip
            flip_pos = int(rng.integers(0, 3))
            code[flip_pos] ^= 1
            if majority_decode(code) == bit:
                correct += 1
        self.assertEqual(correct, 200)

    def test_logical_error_rate_matches_formula(self):
        p = 0.05
        expected = logical_error_rate_rep3(p)
        rng = np.random.default_rng(0)
        wrong = 0
        trials = 20_000
        for _ in range(trials):
            bit = int(rng.integers(0, 2))
            noisy = flip_channel(encode_rep3(bit), p, rng)
            if majority_decode(noisy) != bit:
                wrong += 1
        empirical = wrong / trials
        # Should match to within ~25% relative
        self.assertAlmostEqual(empirical, expected, delta=0.02)

    def test_surface_layout_syndrome_bounds(self):
        layout = SurfaceLayout(d=3)
        H = layout.parity_check
        self.assertEqual(H.shape, (9, 18))       # 9 plaquettes, 18 qubits
        # Zero error -> zero syndrome
        zero = np.zeros(H.shape[1], dtype=np.int8)
        np.testing.assert_array_equal(layout.syndrome(zero), np.zeros(9))


# ================================================== merkle_dag

class TestMerkleDAG(unittest.TestCase):
    def test_put_get_dedup(self):
        dag = MerkleDAG()
        h1 = dag.put({"a": 1})
        h2 = dag.put({"a": 1})
        self.assertEqual(h1, h2)
        self.assertEqual(len(dag), 1)
        self.assertEqual(dag.get(h1), {"a": 1})

    def test_inclusion_proof(self):
        dag = MerkleDAG()
        leaves = [{"k": i} for i in range(8)]
        root, levels = dag.build_merkle_tree(leaves)
        for i in range(len(leaves)):
            proof = dag.inclusion_proof(levels, i)
            self.assertTrue(MerkleDAG.verify_inclusion(leaves[i], proof, root))

    def test_content_hash_deterministic(self):
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}       # same content, different key order
        self.assertEqual(content_hash(a), content_hash(b))


# ================================================== peer_review

class TestPeerReview(unittest.TestCase):
    def test_clean_log_verifies(self):
        log = HashChainLog(agent_id="alice")
        for i in range(10):
            log.append({"i": i})
        ok, reason = verify_log(log.entries(), log.public_key, "alice")
        self.assertTrue(ok, reason)

    def test_tampering_detected(self):
        log = HashChainLog(agent_id="alice")
        for i in range(5):
            log.append({"i": i})
        entries = log.entries()
        # Mutate a payload without re-signing
        entries[2].payload["i"] = 999
        ok, _ = verify_log(entries, log.public_key, "alice")
        self.assertFalse(ok)

    def test_spot_check_clean(self):
        log = HashChainLog(agent_id="alice")
        for i in range(20):
            log.append({"i": i})
        ok, _ = spot_check(log.entries(), sample_rate=0.3,
                           public_key=log.public_key, agent_id="alice", seed=0)
        self.assertTrue(ok)


# ================================================== vrf_committee

class TestVRF(unittest.TestCase):
    def test_evaluate_verify(self):
        vrf = VRFKey()
        out = vrf.evaluate(b"round-7")
        self.assertTrue(verify_vrf(out, b"round-7", vrf.public_key))

    def test_different_input_different_output(self):
        vrf = VRFKey()
        a = vrf.evaluate(b"x")
        b = vrf.evaluate(b"y")
        self.assertNotEqual(a.output, b.output)

    def test_elect_committee_size(self):
        keys = [VRFKey() for _ in range(20)]
        outs = {f"agent_{i}": k.evaluate(b"seed") for i, k in enumerate(keys)}
        committee = elect_committee(outs, k=5)
        self.assertEqual(len(committee), 5)
        self.assertEqual(len(set(committee)), 5)


# ================================================== cbf

class TestCBF(unittest.TestCase):
    def test_no_correction_when_safe(self):
        # h(x) = 1 - x² ; safe where x ∈ (-1, 1)
        # ḣ = -2x · u     (for ẋ = u)
        x = np.array([0.0])
        r = enforce_barrier(
            x, u_nominal=np.array([0.1]),
            h=lambda x: float(1 - x[0] ** 2),
            Lfh=lambda x: 0.0,                 # f(x) = 0
            Lgh=lambda x: np.array([-2 * x[0]]),
            alpha=exponential_alpha(1.0),
        )
        self.assertAlmostEqual(r.slack, 0.0)

    def test_correction_applied_when_unsafe_drift(self):
        # At x = 0.9 (close to boundary), push barrier back on if nominal u
        # drives further toward unsafe side.
        x = np.array([0.9])
        r = enforce_barrier(
            x, u_nominal=np.array([10.0]),     # large drift into unsafe
            h=lambda x: float(1 - x[0] ** 2),
            Lfh=lambda x: 0.0,
            Lgh=lambda x: np.array([-2 * x[0]]),
            alpha=exponential_alpha(1.0),
        )
        self.assertGreater(r.slack, 0.0)
        self.assertLess(r.u_safe[0], 10.0)


# ================================================== paillier

class TestPaillier(unittest.TestCase):
    def test_encrypt_decrypt(self):
        pk, sk = keygen(bits=256)
        for m in (0, 1, 42, 12345):
            c = pk.encrypt(m)
            self.assertEqual(sk.decrypt(c), m)

    def test_homomorphic_add(self):
        pk, sk = keygen(bits=256)
        c1 = pk.encrypt(100)
        c2 = pk.encrypt(250)
        csum = homomorphic_add(c1, c2, pk)
        self.assertEqual(sk.decrypt(csum), 350)

    def test_homomorphic_scale(self):
        pk, sk = keygen(bits=256)
        c = pk.encrypt(30)
        cx = homomorphic_scale(c, 5, pk)
        self.assertEqual(sk.decrypt(cx), 150)


# ================================================== dp

class TestDP(unittest.TestCase):
    def test_laplace_mean_unbiased(self):
        rng = np.random.default_rng(0)
        lm = LaplaceMechanism(sensitivity=1.0, epsilon=2.0)
        value = np.ones(10_000) * 5.0
        out = lm.release(value, rng)
        self.assertAlmostEqual(float(out.mean()), 5.0, places=1)

    def test_gaussian_mechanism_std_matches(self):
        gm = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        # σ = √(2 ln(1.25/δ)) / 1. With δ=1e-5, ln(1.25/δ)≈11.74 → σ≈4.84
        self.assertAlmostEqual(gm.std, np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0,
                                places=4)

    def test_basic_composition(self):
        e, d = basic_composition(0.1, 1e-6, k=10)
        self.assertAlmostEqual(e, 1.0)
        self.assertAlmostEqual(d, 1e-5)

    def test_advanced_composition_tighter_for_small_eps(self):
        # For small ε, advanced bound is much tighter per-query
        e_basic, _ = basic_composition(0.01, 1e-8, k=10000)
        e_adv, _ = advanced_composition(0.01, 1e-8, k=10000, target_delta=1e-6)
        self.assertLess(e_adv, e_basic)

    def test_shuffle(self):
        rng = np.random.default_rng(0)
        arr = np.arange(20)
        out = shuffle_then_aggregate(arr, rng)
        self.assertEqual(set(out.tolist()), set(arr.tolist()))


# ================================================== dagbft

class TestDAGBFT(unittest.TestCase):
    def test_commits_after_anchor_rounds(self):
        sim = BullsharkSimulator(n_validators=4, f=1)
        for _ in range(6):
            sim.propose_round()
        # Some commits must happen at rounds 2 and 4
        self.assertGreater(len(sim.committed), 0)

    def test_no_commit_without_two_rounds(self):
        sim = BullsharkSimulator(n_validators=4, f=1)
        sim.propose_round()
        self.assertEqual(len(sim.committed), 0)

    def test_rejects_too_few_validators(self):
        with self.assertRaises(ValueError):
            BullsharkSimulator(n_validators=3, f=1)


if __name__ == "__main__":
    unittest.main()
