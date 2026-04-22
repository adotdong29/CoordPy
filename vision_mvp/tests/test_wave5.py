"""Tests for Wave 5 — LLM-native numpy subset."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.data_migration import (
    Instance, Schema, SchemaMorphism, delta, pi, sigma,
)
from vision_mvp.core.deep_sets import DeepSetsSynth, mean_synthesizer
from vision_mvp.core.deq_numpy import anderson_iterate, picard_iterate
from vision_mvp.core.exec_plan import PlanError, PlanInterpreter
from vision_mvp.core.retrieval_store import BruteForceVectorStore
from vision_mvp.core.speculative import (
    expected_acceptance_rate, rejection_sample, speculative_multi_token,
)
from vision_mvp.core.spn import LeafCat, ProdNode, SumNode, SumProductNetwork


# ================================================== deq_numpy

class TestDEQ(unittest.TestCase):
    def test_picard_converges_for_contraction(self):
        # f(z) = 0.5 z + 1 has fixed point z* = 2
        r = picard_iterate(lambda z: 0.5 * z + 1, np.array([0.0]))
        self.assertTrue(r.converged)
        self.assertAlmostEqual(float(r.fixed_point[0]), 2.0, places=4)

    def test_anderson_converges_to_same_fixed_point(self):
        # Anderson's λ regularisation may add iterations on easy linear
        # problems; the key claim is convergence to the same fixed point.
        p = picard_iterate(lambda z: 0.5 * z + 1, np.array([0.0]), tol=1e-8)
        a = anderson_iterate(lambda z: 0.5 * z + 1, np.array([0.0]), tol=1e-8)
        self.assertTrue(p.converged and a.converged)
        self.assertAlmostEqual(float(a.fixed_point[0]), float(p.fixed_point[0]),
                                places=6)

    def test_anderson_converges_on_nonlinear(self):
        # f(z) = sin(z) + 0.5;  contraction for |z| small
        r = anderson_iterate(lambda z: np.sin(z) + 0.5, np.array([0.0]))
        self.assertTrue(r.converged)


# ================================================== data_migration

class TestDataMigration(unittest.TestCase):
    def _build_source_schema(self) -> Schema:
        S = Schema()
        S.add_object("Employee")
        S.add_object("Department")
        S.add_morphism("Employee", "Department", "dept")
        return S

    def _build_target_schema(self) -> Schema:
        T = Schema()
        T.add_object("Person")
        T.add_object("Group")
        T.add_morphism("Person", "Group", "group")
        return T

    def test_delta_pulls_back_instance(self):
        S = self._build_source_schema()
        T = self._build_target_schema()
        I_T = Instance(T)
        I_T.set_object("Person", {"alice", "bob"})
        I_T.set_object("Group", {"eng", "sales"})
        I_T.set_morphism("Person", "Group", "group",
                         {"alice": "eng", "bob": "sales"})
        F = SchemaMorphism(
            source=S, target=T,
            obj_map={"Employee": "Person", "Department": "Group"},
            mor_map={("Employee", "Department", "dept"): ("Person", "Group", "group")},
        )
        I_S = delta(F, I_T)
        self.assertEqual(I_S.data["Employee"], {"alice", "bob"})
        self.assertEqual(I_S.data["Department"], {"eng", "sales"})

    def test_sigma_unions_fibers(self):
        S = Schema(); S.add_object("A")
        T = Schema(); T.add_object("X")
        F = SchemaMorphism(S, T, obj_map={"A": "X"}, mor_map={})
        I_S = Instance(S); I_S.set_object("A", {1, 2, 3})
        I_T = sigma(F, I_S)
        self.assertEqual(
            I_T.data["X"], {("A", 1), ("A", 2), ("A", 3)},
        )


# ================================================== speculative

class TestSpeculative(unittest.TestCase):
    def test_unbiased_sampling(self):
        # If draft = target, every draft is accepted; output distribution = target
        rng = np.random.default_rng(0)
        p = np.array([0.1, 0.3, 0.6])
        counts = np.zeros(3)
        for _ in range(20000):
            r = rejection_sample(p, p, rng)
            counts[r.token] += 1
        emp = counts / counts.sum()
        np.testing.assert_allclose(emp, p, atol=0.05)

    def test_expected_rate_bounds(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        rate = expected_acceptance_rate(q, p)
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_multi_token(self):
        def draft_fn(hist):
            return np.array([0.5, 0.5])

        def target_fn(hist):
            return np.array([0.3, 0.7])

        rng = np.random.default_rng(0)
        reports = speculative_multi_token(draft_fn, target_fn, 10, rng)
        self.assertEqual(len(reports), 10)


# ================================================== retrieval_store

class TestRetrievalStore(unittest.TestCase):
    def test_knn_returns_similar(self):
        store = BruteForceVectorStore(dim=3)
        store.add(np.array([1.0, 0.0, 0.0]), "x-axis")
        store.add(np.array([0.0, 1.0, 0.0]), "y-axis")
        store.add(np.array([0.0, 0.0, 1.0]), "z-axis")
        results = store.knn(np.array([0.9, 0.1, 0.0]), k=1)
        self.assertEqual(results[0][2].payload, "x-axis")

    def test_empty_store_knn(self):
        store = BruteForceVectorStore(dim=4)
        self.assertEqual(store.knn(np.zeros(4)), [])


# ================================================== exec_plan

class TestExecPlan(unittest.TestCase):
    def test_basic_arithmetic(self):
        itp = PlanInterpreter()
        env = itp.run("a = 2 + 3 * 4\nb = a - 1", {})
        self.assertEqual(env["a"], 14)
        self.assertEqual(env["b"], 13)

    def test_if_branch(self):
        itp = PlanInterpreter()
        env = itp.run("""
if x > 10:
    y = 1
else:
    y = 2
""", {"x": 5})
        self.assertEqual(env["y"], 2)

    def test_for_loop(self):
        itp = PlanInterpreter()
        env = itp.run("""
total = 0
for i in values:
    total = total + i
""", {"values": [1, 2, 3, 4]})
        self.assertEqual(env["total"], 10)

    def test_forbidden_import(self):
        itp = PlanInterpreter()
        with self.assertRaises(PlanError):
            itp.run("import os", {})

    def test_forbidden_call(self):
        itp = PlanInterpreter()
        with self.assertRaises(PlanError):
            itp.run("x = open('/etc/passwd')", {})

    def test_allowed_fn(self):
        itp = PlanInterpreter(allowed_fns={"square": lambda x: x * x})
        env = itp.run("y = square(3)", {})
        self.assertEqual(env["y"], 9)

    def test_step_limit(self):
        itp = PlanInterpreter(max_steps=100)
        with self.assertRaises(PlanError):
            itp.run("while True:\n    x = 1", {})


# ================================================== spn

class TestSPN(unittest.TestCase):
    def test_leaf_log_likelihood(self):
        leaf = LeafCat(var=0, probs=np.array([0.3, 0.7]))
        self.assertAlmostEqual(leaf.log_value({0: 1}), np.log(0.7), places=6)

    def test_product_factorized(self):
        l0 = LeafCat(var=0, probs=np.array([0.5, 0.5]))
        l1 = LeafCat(var=1, probs=np.array([0.4, 0.6]))
        prod = ProdNode(children=[l0, l1])
        spn = SumProductNetwork(root=prod)
        # P(X0=1, X1=1) = 0.5 * 0.6 = 0.3
        self.assertAlmostEqual(spn.likelihood({0: 1, 1: 1}), 0.3, places=6)

    def test_sum_mixture(self):
        l0 = LeafCat(var=0, probs=np.array([0.1, 0.9]))
        l1 = LeafCat(var=0, probs=np.array([0.9, 0.1]))
        s = SumNode(children=[l0, l1], weights=np.array([0.5, 0.5]))
        spn = SumProductNetwork(root=s)
        # P(X0=1) = 0.5 * 0.9 + 0.5 * 0.1 = 0.5
        self.assertAlmostEqual(spn.likelihood({0: 1}), 0.5, places=6)

    def test_marginalization(self):
        leaf = LeafCat(var=0, probs=np.array([0.3, 0.7]))
        spn = SumProductNetwork(root=leaf)
        # Empty evidence → P = 1
        self.assertAlmostEqual(spn.likelihood({}), 1.0, places=6)


# ================================================== deep_sets

class TestDeepSets(unittest.TestCase):
    def test_permutation_invariance(self):
        synth = mean_synthesizer(dim=3)
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        out1 = synth.forward(X)
        out2 = synth.forward(X[::-1])
        np.testing.assert_allclose(out1, out2)

    def test_mean_output(self):
        synth = mean_synthesizer(dim=2)
        X = np.array([[1.0, 1.0], [3.0, 5.0]])
        out = synth.forward(X)
        np.testing.assert_allclose(out, np.array([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
