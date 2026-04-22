"""Phase 29 — safe method-instance auto-construction tests.

Covers:

  1. ``analyze_class_construction`` returns True for classes with no
     explicit ``__init__``, classes whose ``__init__`` takes only
     self + defaulted positional params, and dataclasses with all
     fields defaulted.
  2. ``analyze_class_construction`` returns False for classes whose
     ``__init__`` has required positional params, required kw-only
     params, or *args / **kwargs, and for dataclasses with at least
     one required field, and for subclasses of ``Exception`` /
     ``BaseException``.
  3. ``classify_function_candidate`` promotes methods on constructable
     classes to ``ready_method`` when the method's own positional
     params are either empty or fully typed.
  4. ``classify_function_candidate`` keeps methods on non-constructable
     classes as ``unsupported_method`` with a reason that surfaces the
     init-shape rejection.
  5. ``_try_construct_instance`` constructs simple classes under the
     sandbox; returns ``(None, tag)`` with the right error tag for
     classes whose ``__init__`` raises, hangs, or tries to escape the
     sandbox.
  6. ``probe_corpus_function`` runs the probe against a bound method,
     reports entered+applicable correctly, and the observation is
     honoured by ``summarise_corpus_calibration``.
  7. Coverage accounting: ``ready_method`` is its own bucket and
     counts toward ``ready_fraction``.

These tests are deterministic, self-contained, and do not require the
multi-corpus benchmark machinery or any third-party imports.
"""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
import textwrap
import unittest

from vision_mvp.core.code_corpus_runtime import (
    CorpusFunctionCandidate,
    SafeRecipeRegistry,
    _try_construct_instance,
    analyze_class_construction,
    calibrate_corpus,
    classify_function_candidate,
    discover_candidates,
    probe_corpus_function,
    summarise_corpus_calibration,
)


def _parse_class(src: str) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node
    raise AssertionError("no class in source")


def _parse_method(src: str, method: str = "m") -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return node
    raise AssertionError(f"method {method} not found")


class TestAnalyzeClassConstruction(unittest.TestCase):

    def test_plain_class_no_init_is_constructable(self) -> None:
        cls = _parse_class("class C:\n    x = 1\n")
        ok, strat = analyze_class_construction(cls)
        self.assertTrue(ok)
        self.assertEqual(strat, "inherited_object_init")

    def test_class_with_methods_no_init_is_constructable(self) -> None:
        cls = _parse_class("class C:\n    def m(self): return 1\n")
        ok, _ = analyze_class_construction(cls)
        self.assertTrue(ok)

    def test_zero_arg_init_is_constructable(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertTrue(ok)
        self.assertEqual(strat, "explicit_init_all_defaults")

    def test_init_with_all_defaults_is_constructable(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, a=1, b='x'):\n"
            "        self.a = a\n"
        )
        ok, _ = analyze_class_construction(cls)
        self.assertTrue(ok)

    def test_init_with_required_positional_is_not(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, a, b=1):\n"
            "        self.a = a\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "init_required_positional")

    def test_init_with_varargs_is_not(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, *a):\n"
            "        self.a = a\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "varargs_init")

    def test_init_with_kwargs_is_not(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, **kw):\n"
            "        self.kw = kw\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "varargs_init")

    def test_init_with_required_kwonly_is_not(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, *, k):\n"
            "        self.k = k\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "init_required_kwonly")

    def test_async_init_is_not(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    async def __init__(self):\n"
            "        self.x = 1\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "async_init")

    def test_dataclass_all_defaults_is_constructable(self) -> None:
        cls = _parse_class(
            "@dataclass\n"
            "class C:\n"
            "    x: int = 0\n"
            "    y: str = 'a'\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertTrue(ok)
        self.assertEqual(strat, "dataclass_all_defaults")

    def test_dataclass_with_required_field_is_not(self) -> None:
        cls = _parse_class(
            "@dataclass\n"
            "class C:\n"
            "    x: int\n"
            "    y: str = 'a'\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "dataclass_required_field")

    def test_dataclass_classvar_does_not_require_default(self) -> None:
        # ClassVar annotations don't participate in the generated
        # __init__, so they can be declared without a default without
        # blocking construction.
        cls = _parse_class(
            "@dataclass\n"
            "class C:\n"
            "    VERSION: ClassVar[int]\n"
            "    x: int = 0\n"
        )
        ok, _ = analyze_class_construction(cls)
        self.assertTrue(ok)

    def test_dataclass_frozen_form_is_constructable(self) -> None:
        cls = _parse_class(
            "@dataclass(frozen=True)\n"
            "class C:\n"
            "    x: int = 0\n"
        )
        ok, _ = analyze_class_construction(cls)
        self.assertTrue(ok)

    def test_dataclasses_dataclass_attribute_form_is_recognised(self) -> None:
        cls = _parse_class(
            "@dataclasses.dataclass\n"
            "class C:\n"
            "    x: int = 0\n"
        )
        ok, _ = analyze_class_construction(cls)
        self.assertTrue(ok)

    def test_exception_subclass_is_not(self) -> None:
        cls = _parse_class(
            "class C(ValueError):\n"
            "    pass\n"
        )
        ok, strat = analyze_class_construction(cls)
        self.assertFalse(ok)
        self.assertEqual(strat, "exception_subclass")


class TestClassifyPromotesMethods(unittest.TestCase):

    def test_zero_arg_method_on_plain_class_is_ready_method(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def m(self): return 1\n"
        )
        method = cls.body[0]
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True,
            class_node=cls,
        )
        self.assertEqual(cand.callable_status, "ready_method")

    def test_typed_method_on_plain_class_is_ready_method(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def m(self, x: int, y: str): return y\n"
        )
        method = cls.body[0]
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True,
            class_node=cls,
        )
        self.assertEqual(cand.callable_status, "ready_method")

    def test_untyped_method_on_constructable_class_is_unsupported_untyped(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def m(self, x): return x\n"
        )
        method = cls.body[0]
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True,
            class_node=cls,
        )
        self.assertEqual(cand.callable_status, "unsupported_untyped")

    def test_method_on_nonconstructable_class_is_unsupported_method(self) -> None:
        cls = _parse_class(
            "class C:\n"
            "    def __init__(self, required):\n"
            "        self.r = required\n"
            "    def m(self): return 1\n"
        )
        method = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "m"][0]
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True,
            class_node=cls,
        )
        self.assertEqual(cand.callable_status, "unsupported_method")
        self.assertIn("init_required_positional", cand.reason)

    def test_classifier_without_class_node_keeps_unsupported_method(self) -> None:
        method = _parse_method("def m(self, x: int): return x\n")
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True)
        self.assertEqual(cand.callable_status, "unsupported_method")

    def test_method_on_dataclass_all_defaults_is_ready_method(self) -> None:
        cls = _parse_class(
            "@dataclass\n"
            "class C:\n"
            "    x: int = 0\n"
            "    def m(self): return self.x\n"
        )
        method = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "m"][0]
        cand = classify_function_candidate(
            "m", "m.py", "C.m", method, is_method=True,
            class_node=cls,
        )
        self.assertEqual(cand.callable_status, "ready_method")


class TestTryConstructInstance(unittest.TestCase):

    def test_simple_class_constructs(self) -> None:
        class Plain:
            def __init__(self) -> None:
                self.x = 1
            def m(self) -> int:
                return self.x
        inst, err = _try_construct_instance(Plain, budget_s=0.1)
        self.assertIsNotNone(inst)
        self.assertEqual(err, "")
        self.assertEqual(inst.m(), 1)

    def test_plain_class_no_init_constructs(self) -> None:
        class Plain:
            x = 1
        inst, err = _try_construct_instance(Plain, budget_s=0.1)
        self.assertIsNotNone(inst)
        self.assertEqual(err, "")

    def test_init_raises_returns_error_tag(self) -> None:
        class Raises:
            def __init__(self) -> None:
                raise RuntimeError("nope")
        inst, err = _try_construct_instance(Raises, budget_s=0.1)
        self.assertIsNone(inst)
        self.assertTrue(err.startswith("construct_exc:"))
        self.assertIn("RuntimeError", err)

    def test_non_class_returns_not_callable(self) -> None:
        inst, err = _try_construct_instance(42, budget_s=0.1)  # type: ignore[arg-type]
        self.assertIsNone(inst)
        self.assertEqual(err, "construct_not_callable")

    def test_init_hangs_times_out(self) -> None:
        class Hangs:
            def __init__(self) -> None:
                # A pure-Python loop so sys.settrace can fire on a
                # line event past the deadline.
                i = 0
                while True:
                    i += 1
                    if i < 0:  # pragma: no cover — guards against no-op
                        break
        inst, err = _try_construct_instance(Hangs, budget_s=0.02)
        self.assertIsNone(inst)
        self.assertEqual(err, "construct_budget")


class TestEndToEndMethodProbe(unittest.TestCase):
    """Calibrate a synthetic corpus whose methods exercise the new
    ``ready_method`` path end-to-end.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p29_method_")
        src = textwrap.dedent("""
            # Three classes — one plain, one dataclass with defaults, one
            # with a required init arg (should stay unsupported).

            from dataclasses import dataclass

            class Plain:
                def helper(self, x: int) -> int:
                    return x + 1
                def raiser(self) -> None:
                    raise ValueError('boom')

            @dataclass
            class WithDefaults:
                val: int = 7
                def value(self) -> int:
                    return self.val

            class Required:
                def __init__(self, r: int) -> None:
                    self.r = r
                def m(self) -> int:
                    return self.r
        """).strip() + "\n"
        with open(os.path.join(self.tmp, "m.py"), "w") as f:
            f.write(src)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_coverage_breakdown(self) -> None:
        rows, cov = calibrate_corpus(
            "p29_method", self.tmp, corpus_package=None,
            seeds=(0, 1), budget_s=0.1,
        )
        d = cov.as_dict()
        # 5 total: Plain.helper, Plain.raiser, WithDefaults.value,
        # Required.__init__, Required.m.
        self.assertEqual(d["n_total"], 5)
        # Plain.helper (typed), Plain.raiser (zero-arg after self),
        # WithDefaults.value (zero-arg), Required.__init__ (has typed
        # `r`, but __init__ on a class without a further class_node
        # is classified as a method on a class with required init —
        # `unsupported_method`).
        self.assertGreaterEqual(d["ready_method"], 3)
        self.assertGreaterEqual(d["unsupported_method"], 1)
        # Phase 29 ready_fraction must lift above Phase-27/28 which
        # would have counted all four methods as unsupported_method.
        self.assertGreater(d["ready_fraction"], 0.5)

    def test_method_probe_observes_may_raise(self) -> None:
        rows, _ = calibrate_corpus(
            "p29_method", self.tmp, corpus_package=None,
            seeds=(0, 1), budget_s=0.1,
        )
        # Find the Plain.raiser row.
        raiser = None
        for r in rows:
            if r.candidate.qname == "Plain.raiser":
                raiser = r
                break
        self.assertIsNotNone(raiser)
        self.assertEqual(raiser.candidate.callable_status, "ready_method")
        obs = raiser.observations["may_raise"]
        self.assertTrue(obs.applicable,
                         msg=f"expected applicable, notes={obs.notes}")
        self.assertTrue(obs.entered)
        self.assertTrue(obs.runtime_flag)
        self.assertEqual(obs.recipe_kind, "method")

    def test_method_probe_respects_budget_on_hanging_init(self) -> None:
        # Write a second corpus with a constructable class but whose
        # __init__ would hang; the classifier accepts it (since the AST
        # cannot tell), and the probe should return applicable=False
        # with notes="construct_budget".
        tmp2 = tempfile.mkdtemp(prefix="p29_hang_")
        try:
            src = textwrap.dedent("""
                class Hangs:
                    def __init__(self) -> None:
                        i = 0
                        while True:
                            i += 1
                            if i < 0:
                                break
                    def m(self) -> int:
                        return 1
            """).strip() + "\n"
            with open(os.path.join(tmp2, "m.py"), "w") as f:
                f.write(src)
            rows, cov = calibrate_corpus(
                "p29_hang", tmp2, corpus_package=None,
                seeds=(0,), budget_s=0.02,
            )
            hangs_m = [r for r in rows
                         if r.candidate.qname == "Hangs.m"][0]
            self.assertEqual(hangs_m.candidate.callable_status,
                              "ready_method")
            for obs in hangs_m.observations.values():
                self.assertFalse(obs.applicable)
                self.assertTrue(obs.notes.startswith("construct_"))
            self.assertGreaterEqual(cov.n_construct_failed, 1)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_summarise_includes_method_slice(self) -> None:
        rows, _ = calibrate_corpus(
            "p29_method", self.tmp, corpus_package=None,
            seeds=(0, 1), budget_s=0.1,
        )
        metrics = summarise_corpus_calibration(rows)
        mr = metrics["may_raise"]
        # At least Plain.raiser should enter (runtime True) and
        # analyzer should agree.
        self.assertGreaterEqual(mr.n_entered, 3)
        self.assertEqual(mr.n_false_negatives, 0)


if __name__ == "__main__":
    unittest.main()
