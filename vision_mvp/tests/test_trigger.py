"""Unit tests for the trigger abstraction (core/trigger.py)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vision_mvp.core.trigger import (
    CallableTrigger, Trigger, TriggerDecision,
    schema_key_trigger, behavior_probe_trigger,
    register_trigger, get_trigger, list_triggers,
    get_default_trigger,
)


@dataclass
class _LegacyDecision:
    """Mimics the per-task TriggerDecision dataclasses to exercise the
    coercion path in `_coerce_decision`."""
    refine: bool
    score: float
    threshold: float
    own_keys: set


class TestCallableTriggerCoercion(unittest.TestCase):
    def test_legacy_decision_is_normalized(self):
        def fake_fn(own, bul, threshold=0.34):
            return _LegacyDecision(
                refine=True, score=0.7, threshold=threshold,
                own_keys={"a", "b"},
            )
        t = CallableTrigger("fake", fake_fn)
        d = t.should_refine("x", ["y"], threshold=0.5)
        self.assertIsInstance(d, TriggerDecision)
        self.assertTrue(d.refine)
        self.assertAlmostEqual(d.score, 0.7)
        self.assertAlmostEqual(d.threshold, 0.5)
        # Extra fields surface under info
        self.assertEqual(d.info.get("own_keys"), {"a", "b"})

    def test_canonical_decision_passes_through(self):
        def fake_fn(own, bul, threshold=0.34):
            return TriggerDecision(refine=False, score=0.1, threshold=threshold,
                                    info={"reason": "test"})
        t = CallableTrigger("fake", fake_fn)
        d = t.should_refine("x", ["y"], threshold=0.4)
        self.assertFalse(d.refine)
        self.assertEqual(d.info, {"reason": "test"})


class TestRegistry(unittest.TestCase):
    def test_builtin_triggers_present(self):
        names = list_triggers()
        self.assertIn("schema-key", names)
        self.assertIn("behavior-probe", names)
        self.assertIn("hybrid-structural", names)  # registered by general_trigger
        self.assertIn("general-heuristic", names)

    def test_get_trigger_returns_named_instance(self):
        t = get_trigger("schema-key")
        self.assertEqual(t.name, "schema-key-jaccard")
        t2 = get_trigger("behavior-probe")
        self.assertEqual(t2.name, "behavior-probe")

    def test_get_trigger_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_trigger("does-not-exist")

    def test_register_custom_trigger(self):
        class Custom:
            name = "custom-test"
            def should_refine(self, own, bul, threshold=0.34):
                return TriggerDecision(refine=False, score=0.0,
                                        threshold=threshold)
        register_trigger("custom-test", lambda: Custom())
        try:
            t = get_trigger("custom-test")
            self.assertEqual(t.name, "custom-test")
        finally:
            # Don't pollute other tests' registry view.
            from vision_mvp.core.trigger import _REGISTRY
            _REGISTRY.pop("custom-test", None)


class TestExistingTriggersAdaptedSemantics(unittest.TestCase):
    """The adapter must preserve the semantics of the wrapped functions."""

    def test_schema_key_skips_when_aligned(self):
        t = schema_key_trigger()
        own = 'def f(): return {"start": 0, "end": 10}'
        bul = ['def g(r): return r["start"] + r["end"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertFalse(d.refine)

    def test_schema_key_refines_when_disjoint(self):
        t = schema_key_trigger()
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)

    def test_behavior_probe_skips_when_no_pair_match(self):
        t = behavior_probe_trigger()
        d = t.should_refine(
            "def unrelated_a(x): return x",
            ["def unrelated_b(y): return y + 1"],
            threshold=0.34,
        )
        self.assertFalse(d.refine)
        self.assertEqual(d.score, 0.0)


class TestProtocolConformance(unittest.TestCase):
    def test_runtime_protocol_check(self):
        # `Trigger` is `runtime_checkable`; both adapted instances satisfy it.
        self.assertIsInstance(schema_key_trigger(), Trigger)
        self.assertIsInstance(behavior_probe_trigger(), Trigger)


class TestDefaultTrigger(unittest.TestCase):
    def test_returns_trigger_instance(self):
        t = get_default_trigger()
        self.assertIsInstance(t, Trigger)

    def test_is_hybrid_structural(self):
        t = get_default_trigger()
        self.assertEqual(t.name, "hybrid-structural")

    def test_fires_on_dict_key_drift(self):
        t = get_default_trigger()
        own = 'def f(): return {"start": 0}'
        bul = ['def g(r): return r["lo"]']
        d = t.should_refine(own, bul, threshold=0.34)
        self.assertTrue(d.refine)

    def test_skips_on_identical_code(self):
        t = get_default_trigger()
        src = 'def f(x): return x + 1'
        d = t.should_refine(src, [src], threshold=0.34)
        self.assertFalse(d.refine)


if __name__ == "__main__":
    unittest.main()
