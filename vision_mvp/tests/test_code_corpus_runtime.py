"""Tests for corpus-scale runtime calibration — Phase 27.

These tests cover:

  1. **Candidate classification.** ``classify_function_candidate`` maps
     AST function nodes to the correct callability status — zero-arg,
     typed, untyped, varargs, async, generator, method.
  2. **Recipe integrity.** ``no_args_recipe`` / ``typed_recipe`` /
     ``curated_recipe`` produce deterministic argument sequences under
     the same seed; ``typed_recipe`` refuses unsupported annotations.
  3. **Sandbox restoration.** The entry+budget tracer installs and
     restores ``sys.settrace`` cleanly; the sandbox recorders from
     Phase 26 are reused intact.
  4. **Entry detection.** When the recipe produces args that never
     reach the target body (e.g. immediate TypeError on arg unpack),
     the observation correctly reports ``entered=False`` and is
     excluded from FP/FN.
  5. **Timeout.** A deliberately slow target triggers the budget
     tracer and lands in the ``timeout`` bucket without blocking the
     benchmark.
  6. **Calibration aggregation.** ``summarise_corpus_calibration``
     respects ``entered=True`` — non-entering observations do not
     contribute to per-predicate FP/FN.
  7. **End-to-end.** Running ``calibrate_corpus`` on a tiny synthetic
     corpus produces the expected coverage distribution.
"""

from __future__ import annotations

import ast
import os
import random
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from dataclasses import replace

from vision_mvp.core.code_corpus_runtime import (
    CorpusCalibrationRow, CorpusFunctionCandidate, CorpusObservation,
    CoverageAccount, SafeRecipeRegistry,
    _classify_annotation, _entry_and_budget_tracer,
    _function_is_generator,
    build_corpus_static_flags, calibrate_corpus, classify_function_candidate,
    collect_divergences, curated_recipe, discover_candidates,
    no_args_recipe, probe_corpus_function,
    summarise_corpus_calibration, typed_recipe,
)


# =============================================================================
# Helpers
# =============================================================================


def _parse_func(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return n
    raise AssertionError("no function in source")


def _make_candidate(name: str, status: str = "ready_no_args",
                    n_params: int = 0, **kw) -> CorpusFunctionCandidate:
    defaults = dict(
        module_name="m", module_path="m.py", qname=name,
        n_params=n_params, param_annotations=("",) * n_params,
        has_star_args=False, has_kwargs=False,
        is_generator=False, is_async=False, is_method=False,
        callable_status=status, reason="",
    )
    defaults.update(kw)
    return CorpusFunctionCandidate(**defaults)


# =============================================================================
# Annotation classifier
# =============================================================================


class TestClassifyAnnotation(unittest.TestCase):

    def test_whitelisted_types(self):
        for t in (int, str, bool, list, dict, tuple, bytes, float, type(None)):
            self.assertIs(_classify_annotation(t), t)

    def test_whitelisted_strings(self):
        self.assertIs(_classify_annotation("int"), int)
        self.assertIs(_classify_annotation("str"), str)
        self.assertIs(_classify_annotation("list[int]"), list)
        self.assertIs(_classify_annotation("None"), type(None))

    def test_non_whitelisted_returns_none(self):
        class _Foo: pass
        self.assertIsNone(_classify_annotation(_Foo))
        self.assertIsNone(_classify_annotation("SomeUnknownType"))

    def test_empty_annotation(self):
        import inspect
        self.assertIsNone(_classify_annotation(inspect.Parameter.empty))


# =============================================================================
# Function-level generator detection
# =============================================================================


class TestFunctionIsGenerator(unittest.TestCase):

    def test_pure_function_not_generator(self):
        fn = _parse_func("def f(): return 1\n")
        self.assertFalse(_function_is_generator(fn))

    def test_yield_is_generator(self):
        fn = _parse_func("def g():\n    yield 1\n")
        self.assertTrue(_function_is_generator(fn))

    def test_yield_from_is_generator(self):
        fn = _parse_func("def g():\n    yield from [1, 2]\n")
        self.assertTrue(_function_is_generator(fn))


# =============================================================================
# Candidate classification
# =============================================================================


class TestClassifyFunctionCandidate(unittest.TestCase):

    def test_zero_arg(self):
        fn = _parse_func("def f(): return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "ready_no_args")

    def test_typed_simple_params(self):
        fn = _parse_func("def f(a: int, b: str) -> int: return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "ready_typed")
        self.assertEqual(c.param_annotations, ("int", "str"))

    def test_untyped_param_rejected(self):
        fn = _parse_func("def f(a, b): return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_untyped")

    def test_varargs_rejected(self):
        fn = _parse_func("def f(*args): return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_varargs")

    def test_kwargs_rejected(self):
        fn = _parse_func("def f(**kw): return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_varargs")

    def test_async_rejected(self):
        fn = _parse_func("async def f(): return 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_async")

    def test_generator_rejected(self):
        fn = _parse_func("def f():\n    yield 1\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_generator")

    def test_method_rejected(self):
        fn = _parse_func("def m(self, x: int): return x\n")
        c = classify_function_candidate("m", "m.py", "Cls.m", fn, True)
        self.assertEqual(c.callable_status, "unsupported_method")

    def test_curated_wins_over_heuristics(self):
        reg = SafeRecipeRegistry()
        reg.register("m", "f", no_args_recipe())
        fn = _parse_func("def f(x):\n    return x\n")  # untyped
        c = classify_function_candidate("m", "m.py", "f", fn, False, reg)
        self.assertEqual(c.callable_status, "ready_curated")

    def test_kwonly_without_default_rejected(self):
        fn = _parse_func("def f(*, x): return x\n")
        c = classify_function_candidate("m", "m.py", "f", fn, False)
        self.assertEqual(c.callable_status, "unsupported_varargs")


# =============================================================================
# Recipe determinism + refusal
# =============================================================================


class TestRecipes(unittest.TestCase):

    def test_no_args_recipe_produces_empty_tuples(self):
        r = no_args_recipe(n_calls=5)
        out = r.build(random.Random(0),
                      _make_candidate("f", "ready_no_args"),
                      lambda: None)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(a == () for a in out))

    def test_typed_recipe_deterministic(self):
        def f(a: int, b: str) -> None: ...
        cand = _make_candidate("f", "ready_typed", n_params=2,
                               param_annotations=("int", "str"))
        r = typed_recipe(n_calls=4)
        out1 = r.build(random.Random(42), cand, f)
        out2 = r.build(random.Random(42), cand, f)
        self.assertEqual(out1, out2)
        self.assertEqual(len(out1), 4)
        for args in out1:
            self.assertEqual(len(args), 2)
            self.assertIsInstance(args[0], int)
            self.assertIsInstance(args[1], str)

    def test_typed_recipe_refuses_untyped(self):
        def f(a): ...
        cand = _make_candidate("f", "ready_typed", n_params=1)
        r = typed_recipe()
        self.assertEqual(r.build(random.Random(0), cand, f), [])

    def test_typed_recipe_refuses_unsupported_type(self):
        class Foo: pass
        def f(a: Foo): ...
        cand = _make_candidate("f", "ready_typed", n_params=1,
                               param_annotations=("Foo",))
        r = typed_recipe()
        self.assertEqual(r.build(random.Random(0), cand, f), [])

    def test_curated_recipe_normalises_tuples(self):
        def builder(rng):
            return [1, (2,), (3, 4)]
        r = curated_recipe(builder)
        out = r.build(random.Random(0),
                      _make_candidate("f", "ready_curated"), lambda: None)
        self.assertEqual(out, [(1,), (2,), (3, 4)])

    def test_curated_recipe_swallows_builder_errors(self):
        def bad(rng):
            raise RuntimeError("boom")
        r = curated_recipe(bad)
        out = r.build(random.Random(0),
                      _make_candidate("f", "ready_curated"), lambda: None)
        self.assertEqual(out, [])


# =============================================================================
# Entry + budget tracer
# =============================================================================


class TestEntryAndBudgetTracer(unittest.TestCase):

    def test_restores_previous_tracer(self):
        prev = sys.gettrace()
        def f(): return 1
        with _entry_and_budget_tracer(f, budget_s=0.1):
            self.assertIsNotNone(sys.gettrace())
        self.assertIs(sys.gettrace(), prev)

    def test_counts_target_entry(self):
        def f(): return 42
        with _entry_and_budget_tracer(f, budget_s=0.1) as st:
            self.assertEqual(f(), 42)
        self.assertGreaterEqual(st["enter_count"], 1)

    def test_ignores_non_target_calls(self):
        def f(): return 1
        def g(): return 2
        with _entry_and_budget_tracer(f, budget_s=0.1) as st:
            g()
        self.assertEqual(st["enter_count"], 0)

    def test_budget_exceeded_on_slow_body(self):
        def slow():
            # A Python-level loop with enough line events to let the tracer
            # fire after the budget expires.
            import time
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                x = 1 + 1       # pragma: no cover
        # Probe body calls slow(); our tracer should raise _BudgetExceeded
        # inside, and the probe body catches _ProbeSentinel → no
        # ``may_raise`` flag, but ``exceeded`` is True.
        from vision_mvp.core.code_runtime_calibration import _ProbeSentinel
        with _entry_and_budget_tracer(slow, budget_s=0.05) as st:
            try:
                slow()
            except _ProbeSentinel:
                pass
        self.assertTrue(st["exceeded"])


# =============================================================================
# probe_corpus_function
# =============================================================================


class TestProbeCorpusFunction(unittest.TestCase):

    def _load_source(self, src: str) -> types.ModuleType:
        mod = types.ModuleType("_test_mod")
        code = compile(textwrap.dedent(src), "<test>", "exec")
        exec(code, mod.__dict__)
        return mod

    def test_no_args_pure_function_all_false(self):
        mod = self._load_source("def f(): return 1\n")
        cand = _make_candidate("f", "ready_no_args")
        obs = probe_corpus_function(
            cand, mod.f, mod, predicate="may_raise", seeds=(0,))
        self.assertTrue(obs.applicable)
        self.assertTrue(obs.entered)
        self.assertFalse(obs.runtime_flag)

    def test_no_args_raising_function_true(self):
        mod = self._load_source(
            "def f():\n    raise ValueError('x')\n")
        cand = _make_candidate("f", "ready_no_args")
        obs = probe_corpus_function(
            cand, mod.f, mod, predicate="may_raise", seeds=(0,))
        self.assertTrue(obs.runtime_flag)
        self.assertTrue(obs.entered)

    def test_unsupported_status_is_applicable_false(self):
        mod = self._load_source("def f(): return 1\n")
        cand = _make_candidate("f", "unsupported_untyped")
        obs = probe_corpus_function(
            cand, mod.f, mod, predicate="may_raise", seeds=(0,))
        self.assertFalse(obs.applicable)

    def test_typed_recipe_with_unsupported_annot_is_inapplicable(self):
        class _Foo: pass
        def f(a: _Foo): return a
        mod = types.ModuleType("_m")
        mod.f = f
        cand = _make_candidate("f", "ready_typed", n_params=1,
                               param_annotations=("_Foo",))
        obs = probe_corpus_function(
            cand, f, mod, predicate="may_raise", seeds=(0,))
        # typed_recipe returns [] for unsupported types → inapplicable
        self.assertFalse(obs.applicable)

    def test_timeout_marks_observation_timeout(self):
        mod = self._load_source(
            "def f():\n"
            "    import time\n"
            "    end = time.monotonic() + 0.5\n"
            "    while time.monotonic() < end:\n"
            "        x = 1 + 1\n"
        )
        cand = _make_candidate("f", "ready_no_args")
        obs = probe_corpus_function(
            cand, mod.f, mod, predicate="may_raise",
            seeds=(0,), budget_s=0.03)
        self.assertTrue(obs.timeout)


# =============================================================================
# Module discovery
# =============================================================================


class TestDiscoverCandidates(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p27_test_")
        self._write("a.py",
                    "def zero(): return 1\n"
                    "def typed(x: int, y: str) -> str: return y\n"
                    "def untyped(x): return x\n"
                    "def varargs(*xs): return xs\n"
                    "async def afn(): return 1\n"
                    "def gen():\n    yield 1\n"
                    "class C:\n"
                    "    def method(self, x: int): return x\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel: str, src: str) -> None:
        full = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(full) or self.tmp, exist_ok=True)
        with open(full, "w") as f:
            f.write(src)

    def test_disc_classifies_everything(self):
        disc = discover_candidates(self.tmp, corpus_package=None)
        statuses = {d.candidate.qname: d.candidate.callable_status
                    for d in disc}
        self.assertEqual(statuses["zero"], "ready_no_args")
        self.assertEqual(statuses["typed"], "ready_typed")
        self.assertEqual(statuses["untyped"], "unsupported_untyped")
        self.assertEqual(statuses["varargs"], "unsupported_varargs")
        self.assertEqual(statuses["afn"], "unsupported_async")
        self.assertEqual(statuses["gen"], "unsupported_generator")
        # Phase 29 — ``class C: def method(self, x: int): ...`` now
        # promotes to ``ready_method`` because ``C`` has no custom
        # ``__init__`` (inherits ``object.__init__``) and the method's
        # positional params after self are fully typed.
        self.assertEqual(statuses["C.method"], "ready_method")

    def test_disc_skips_files_by_suffix(self):
        self._write("__main__.py", "def x(): pass\n")
        disc = discover_candidates(self.tmp, corpus_package=None,
                                    skip_files=("__main__.py",))
        qnames = {d.candidate.qname for d in disc}
        self.assertNotIn("x", qnames)


# =============================================================================
# Static flag bridge
# =============================================================================


class TestBuildCorpusStaticFlags(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p27_static_")
        src = textwrap.dedent("""
            import subprocess
            def helper():
                subprocess.run(["/bin/true"])
            def wrapper():
                helper()
            def pure(): return 1
        """).strip() + "\n"
        with open(os.path.join(self.tmp, "a.py"), "w") as f:
            f.write(src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_interproc_flags_propagate(self):
        flags = build_corpus_static_flags(self.tmp, corpus_package=None)
        # `wrapper` calls `helper` which calls subprocess.run — trans should fire
        wrapper_qname = "a.wrapper"
        helper_qname = "a.helper"
        pure_qname = "a.pure"
        self.assertIn(wrapper_qname, flags)
        self.assertIn(helper_qname, flags)
        self.assertIn(pure_qname, flags)
        self.assertTrue(flags[helper_qname]["calls_subprocess"])
        self.assertTrue(flags[wrapper_qname]["calls_subprocess"])
        self.assertFalse(flags[pure_qname]["calls_subprocess"])


# =============================================================================
# End-to-end calibrate_corpus
# =============================================================================


class TestCalibrateCorpusEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p27_cal_")
        src = textwrap.dedent("""
            def pure_zero(): return 0
            def raising_zero():
                raise ValueError('x')
            def typed_pure(a: int, b: str) -> str: return b
            def typed_raise(a: int):
                raise RuntimeError('r')
            def gen_func():
                yield 1
            def varargs_func(*xs): return xs
            class C:
                def meth(self, x: int): return x
        """).strip() + "\n"
        with open(os.path.join(self.tmp, "m.py"), "w") as f:
            f.write(src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_coverage_and_calibration(self):
        rows, cov = calibrate_corpus(
            "synthetic", self.tmp, corpus_package=None,
            seeds=(0, 1), budget_s=0.1,
        )
        d = cov.as_dict()
        self.assertEqual(d["n_total"], 7)          # 6 top-level + 1 method
        self.assertEqual(d["ready_no_args"], 2)    # pure_zero, raising_zero
        self.assertEqual(d["ready_typed"], 2)      # typed_pure, typed_raise
        self.assertEqual(d["unsupported_generator"], 1)
        self.assertEqual(d["unsupported_varargs"], 1)
        # Phase 29 — ``C.meth`` now promotes to ``ready_method`` and
        # the old ``unsupported_method`` bucket drops to zero.
        self.assertEqual(d["unsupported_method"], 0)
        self.assertEqual(d["ready_method"], 1)
        self.assertGreaterEqual(d["n_entered"], 4)

        metrics = summarise_corpus_calibration(rows)
        mr = metrics["may_raise"]
        # Phase 29 — applicable slice now includes ``C.meth`` alongside
        # pure_zero / raising_zero / typed_pure / typed_raise = 5.
        # Runtime True on raising_zero + typed_raise; analyzer agrees.
        self.assertEqual(mr.n_applicable, 5)
        self.assertGreaterEqual(mr.n_entered, 5)
        self.assertEqual(mr.n_false_negatives, 0)

    def test_row_serialisation_roundtrips_keys(self):
        from vision_mvp.core.code_corpus_runtime import rows_as_dict_list
        rows, _ = calibrate_corpus(
            "synthetic", self.tmp, corpus_package=None,
            seeds=(0,), budget_s=0.1,
        )
        out = rows_as_dict_list(rows)
        self.assertIsInstance(out, list)
        for r in out:
            self.assertIn("candidate", r)
            self.assertIn("observations", r)


# =============================================================================
# Divergence detection
# =============================================================================


class TestCollectDivergences(unittest.TestCase):

    def test_fp_when_static_true_runtime_false(self):
        cand = _make_candidate("f", "ready_no_args")
        row = CorpusCalibrationRow(
            candidate=cand,
            static_flags={"may_raise": True},
            observations={
                "may_raise": CorpusObservation(
                    qname="f", predicate="may_raise",
                    runtime_flag=False, n_runs=2, n_triggered=0,
                    n_entered=2, n_timeout=0,
                    applicable=True, entered=True,
                    decidable=True,
                )
            },
        )
        divs = collect_divergences([row], predicates=["may_raise"])
        self.assertEqual(len(divs), 1)
        self.assertEqual(divs[0].kind, "false_positive")

    def test_no_div_when_not_entered(self):
        cand = _make_candidate("f", "ready_no_args")
        row = CorpusCalibrationRow(
            candidate=cand,
            static_flags={"may_raise": True},
            observations={
                "may_raise": CorpusObservation(
                    qname="f", predicate="may_raise",
                    runtime_flag=False, n_runs=2, n_triggered=0,
                    n_entered=0, n_timeout=0,
                    applicable=True, entered=False,
                    decidable=True,
                )
            },
        )
        divs = collect_divergences([row], predicates=["may_raise"])
        self.assertEqual(divs, [])


# =============================================================================
# Phase-26 reuse — make sure we didn't break snippet calibration
# =============================================================================


class TestPhase26RegressionGuard(unittest.TestCase):
    """Phase 27 reuses Phase-26 context managers. Re-verify that
    snippet-level calibration still works after Phase-27 import —
    regression guard against symbol collisions."""

    def test_phase26_snippet_registry_still_runs(self):
        from vision_mvp.core.code_runtime_calibration import calibrate_snippet
        from vision_mvp.tasks.executable_snippets import snippet_by_name
        spec = snippet_by_name("direct_raise")
        res = calibrate_snippet(spec, predicates=["may_raise"], seeds=(0,))
        self.assertTrue(res.runtime_observations["may_raise"].runtime_flag)


if __name__ == "__main__":
    unittest.main()
