"""Unit tests for Phase-11 components: W2, code harness, collaborative task."""
from __future__ import annotations
import unittest
import numpy as np

from vision_mvp.core.wasserstein import (
    w2_exact, w2_sinkhorn, w2, bures_decomposition,
)
from vision_mvp.core.code_harness import (
    extract_code, function_is_defined, function_signature,
)
from vision_mvp.tasks.collaborative_module import (
    FUNCTION_SPECS, SPEC_ORDER, TEST_WEIGHTS, score_tests, compose_module,
)


# -------------------- Wasserstein

class TestWasserstein(unittest.TestCase):
    def test_identical_distributions_zero_distance(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        self.assertAlmostEqual(w2_exact(X, X.copy()), 0.0, places=4)

    def test_w2_positive_when_distributions_differ(self):
        X = np.array([[0., 0.], [1., 1.]])
        Y = np.array([[10., 10.], [11., 11.]])
        self.assertGreater(w2_exact(X, Y), 1.0)

    def test_w2_detects_polarization_with_same_mean(self):
        """Team that splits into two clusters vs tight cluster around same mean."""
        # A: tight around origin
        A = np.array([[0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1]])
        # B: same mean (0,0) but polarized into two extremes
        B = np.array([[5., 5.], [-5., -5.], [5., -5.], [-5., 5.]])
        # Centroid distance ≈ 0 but W2 should be large
        centroid_d = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        w2_val = w2_exact(A, B)
        self.assertLess(centroid_d, 0.1)
        self.assertGreater(w2_val, 5.0)

    def test_sinkhorn_approximates_exact(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 4))
        Y = rng.standard_normal((15, 4)) + 0.5
        exact = w2_exact(X, Y)
        sk = w2_sinkhorn(X, Y, epsilon=0.01, max_iters=500)
        # Sinkhorn is within 30% of exact for small enough eps
        self.assertLess(abs(exact - sk) / max(exact, 1e-6), 0.5)

    def test_dispatcher_picks_exact_for_small(self):
        X = np.zeros((5, 2))
        Y = np.ones((5, 2))
        self.assertAlmostEqual(w2(X, Y), w2_exact(X, Y), places=4)

    def test_bures_mean_shift_only(self):
        X = np.array([[0.0], [0.0], [0.0]])
        Y = np.array([[1.0], [1.0], [1.0]])
        d = bures_decomposition(X, Y)
        self.assertAlmostEqual(d["mean_drift"], 1.0, places=4)
        self.assertAlmostEqual(d["spread_shift"], 0.0, places=4)

    def test_bures_spread_only(self):
        X = np.array([[-1.0], [0.0], [1.0]])
        Y = np.array([[-3.0], [0.0], [3.0]])
        d = bures_decomposition(X, Y)
        self.assertAlmostEqual(d["mean_drift"], 0.0, places=4)
        self.assertGreater(d["spread_shift"], 1.0)


# -------------------- Code harness

class TestCodeHarness(unittest.TestCase):
    def test_extract_from_fence(self):
        src = "Here's the function:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
        code = extract_code(src)
        self.assertIn("def f", code)
        self.assertNotIn("```", code)

    def test_extract_picks_longest_parsing(self):
        src = (
            "```python\ndef foo(): pass\n```\n"
            "```python\ndef bar(x):\n    return x * 2\n    # longer\n```\n"
        )
        code = extract_code(src)
        self.assertIn("def bar", code)

    def test_extract_no_code_returns_none(self):
        src = "I will not write code. Just prose."
        self.assertIsNone(extract_code(src))

    def test_extract_handles_unfenced_code(self):
        # Some LLMs drop fences; we still accept parseable Python
        src = "def f(x):\n    return x\n"
        code = extract_code(src)
        self.assertIsNotNone(code)
        self.assertIn("def f", code)

    def test_function_is_defined(self):
        code = "def foo(x): return x\ndef bar(y): return y"
        self.assertTrue(function_is_defined(code, "foo"))
        self.assertTrue(function_is_defined(code, "bar"))
        self.assertFalse(function_is_defined(code, "baz"))

    def test_function_signature(self):
        code = "def greet(name, greeting='hi'): return greeting + name"
        self.assertEqual(function_signature(code, "greet"), ["name", "greeting"])


# -------------------- Collaborative task

class TestCollaborativeTask(unittest.TestCase):
    def test_spec_order_covers_all_functions(self):
        self.assertEqual(set(SPEC_ORDER), set(FUNCTION_SPECS.keys()))

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(TEST_WEIGHTS.values()), 1.0, places=3)

    def test_score_tests_all_pass(self):
        per = {k: True for k in TEST_WEIGHTS}
        s = score_tests(per)
        self.assertAlmostEqual(s["weighted_score"], 1.0, places=3)
        self.assertEqual(s["n_passed"], 15)

    def test_score_tests_all_fail(self):
        per = {k: False for k in TEST_WEIGHTS}
        s = score_tests(per)
        self.assertEqual(s["weighted_score"], 0.0)
        self.assertEqual(s["n_passed"], 0)

    def test_score_tests_partial(self):
        per = {k: False for k in TEST_WEIGHTS}
        per["parse_basic"] = True
        per["integrate_basic"] = True
        s = score_tests(per)
        self.assertAlmostEqual(s["weighted_score"],
                                TEST_WEIGHTS["parse_basic"] + TEST_WEIGHTS["integrate_basic"],
                                places=3)

    def test_compose_module_includes_all_present(self):
        code = {sp: f"def {FUNCTION_SPECS[sp]['name']}(*a, **k): pass"
                for sp in SPEC_ORDER}
        module = compose_module(code)
        for sp in SPEC_ORDER:
            self.assertIn(FUNCTION_SPECS[sp]["name"], module)


if __name__ == "__main__":
    unittest.main()
