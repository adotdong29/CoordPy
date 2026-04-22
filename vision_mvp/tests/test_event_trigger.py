"""Unit tests for event_trigger.py."""
from __future__ import annotations
import unittest

from vision_mvp.core.event_trigger import (
    extract_dict_keys, disagreement_score, should_refine,
)


class TestExtractKeys(unittest.TestCase):
    def test_dict_literal(self):
        code = 'def f(): return {"a": 1, "b": 2}'
        self.assertEqual(extract_dict_keys(code), {"a", "b"})

    def test_subscript(self):
        code = 'def f(d): return d["foo"] + d["bar"]'
        self.assertEqual(extract_dict_keys(code), {"foo", "bar"})

    def test_get_call(self):
        code = 'def f(d): return d.get("x", 0)'
        self.assertIn("x", extract_dict_keys(code))

    def test_mix(self):
        code = '''
def f(d):
    x = {"key1": d["key2"]}
    return d.get("key3")
'''
        self.assertEqual(
            extract_dict_keys(code),
            {"key1", "key2", "key3"},
        )

    def test_invalid_code_returns_empty(self):
        self.assertEqual(extract_dict_keys("this is not python :::"), set())

    def test_empty(self):
        self.assertEqual(extract_dict_keys(""), set())


class TestDisagreement(unittest.TestCase):
    def test_identical_keys_zero_distance(self):
        a = 'def f(): return {"k": 1}'
        b = 'def g(d): return d["k"]'
        self.assertAlmostEqual(disagreement_score(a, [b]), 0.0, places=3)

    def test_disjoint_keys_max_distance(self):
        a = 'def f(): return {"foo": 1}'
        b = 'def g(d): return d["bar"]'
        self.assertAlmostEqual(disagreement_score(a, [b]), 1.0, places=3)

    def test_partial_overlap(self):
        a = 'def f(): return {"x": 1, "y": 2}'
        b = 'def g(d): return d["x"]'  # overlap of 1, union of 2
        self.assertAlmostEqual(disagreement_score(a, [b]), 0.5, places=3)

    def test_empty_either_side_returns_zero(self):
        # No signal on either side → don't trigger refinement
        self.assertEqual(disagreement_score("def f(): pass", ['def g(): pass']), 0.0)

    def test_union_across_bulletin(self):
        a = 'def f(): return {"x": 1}'
        b1 = 'def g(d): return d["x"]'
        b2 = 'def h(d): return d["y"]'
        # Own = {x}; bulletin union = {x, y}; inter=1, union=2 → 0.5
        self.assertAlmostEqual(disagreement_score(a, [b1, b2]), 0.5, places=3)


class TestTrigger(unittest.TestCase):
    def test_no_refine_when_aligned(self):
        own = 'def f(): return {"start": 0, "end": 10}'
        bul = ['def g(r): return r["start"] + r["end"]']
        d = should_refine(own, bul, threshold=0.34)
        self.assertFalse(d.refine)
        self.assertEqual(d.score, 0.0)

    def test_refine_when_disjoint(self):
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)

    def test_no_refine_small_perturbation(self):
        # 3 keys in common + 1 extra in bulletin → Jaccard = 1 - 3/4 = 0.25 < 0.34
        own = 'def f(): return {"a": 1, "b": 2, "c": 3}'
        bul = ['def g(d): return d["a"] + d["b"] + d["c"] + d["x"]']
        d = should_refine(own, bul, threshold=0.34)
        self.assertFalse(d.refine)


if __name__ == "__main__":
    unittest.main()
