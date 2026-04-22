"""Phase 28 tests — explicit vs implicit raise analyzer and runtime.

Covers:

  1. ``code_semantics._analyze_may_raise_implicit`` flags the six
     Phase-28 implicit-raise-risk patterns and respects the catch-all
     try/except escape hatch.
  2. ``code_semantics.analyze_function`` exposes ``may_raise_implicit``
     on the returned ``FunctionSemantics`` record, independently of
     ``may_raise``.
  3. ``code_interproc.analyze_interproc`` propagates
     ``trans_may_raise_implicit`` over the resolved call graph.
  4. ``code_runtime_calibration`` exception origin classifier returns
     ``"explicit"`` for a traceback terminating on a ``raise``
     statement line inside the target and ``"implicit"`` otherwise.
  5. ``probe_may_raise_split`` partitions observations by origin and
     reports triggers in the correct bucket.
  6. ``probe_predicate`` dispatches the new
     ``may_raise_explicit`` / ``may_raise_implicit`` labels.
  7. ``compute_static_flags_from_source`` includes the new flags.

These tests are deterministic and do not depend on Phase-26's snippet
corpus or Phase-27's real-corpus calibration machinery.
"""

from __future__ import annotations

import ast
import textwrap
import unittest

from vision_mvp.core.code_semantics import (
    FunctionSemantics,
    _analyze_may_raise_implicit,
    _contains_implicit_raise_pattern,
    analyze_function,
    analyze_module,
)
from vision_mvp.core.code_interproc import (
    analyze_interproc, build_module_context,
)
from vision_mvp.core.code_runtime_calibration import (
    RUNTIME_DECIDABLE_PREDICATES,
    _classify_exception_origin,
    _raise_line_numbers,
    compute_static_flags_from_source,
    probe_may_raise_explicit,
    probe_may_raise_implicit,
    probe_may_raise_split,
    probe_predicate,
    load_snippet_module,
)


def _parse_function(src: str, name: str = "f") -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in source")


class TestImplicitRaisePatternDetection(unittest.TestCase):
    """Each risky pattern produces a True flag in isolation; benign
    functions produce False."""

    def test_div_flags(self) -> None:
        f = _parse_function("def f(x, y):\n    return x / y\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_floordiv_flags(self) -> None:
        f = _parse_function("def f(x, y):\n    return x // y\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_mod_flags(self) -> None:
        f = _parse_function("def f(x, y):\n    return x % y\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_pow_flags(self) -> None:
        f = _parse_function("def f(x, y):\n    return x ** y\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_subscript_flags(self) -> None:
        f = _parse_function("def f(xs):\n    return xs[0]\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_risky_builtin_flags(self) -> None:
        for call in ("int(x)", "float(x)", "len(x)", "iter(x)",
                     "next(x)", "divmod(x, 2)", "pow(x, 2)"):
            f = _parse_function(f"def f(x):\n    return {call}\n")
            self.assertTrue(_contains_implicit_raise_pattern(f),
                             f"failed for {call}")

    def test_attribute_on_parameter_flags(self) -> None:
        f = _parse_function("def f(x):\n    return x.attr\n")
        self.assertTrue(_contains_implicit_raise_pattern(f))

    def test_benign_returns_none_does_not_flag(self) -> None:
        f = _parse_function("def f():\n    return None\n")
        self.assertFalse(_contains_implicit_raise_pattern(f))

    def test_string_concat_does_not_flag(self) -> None:
        f = _parse_function(
            "def f(a, b):\n    return str(a) + str(b)\n"  # str is not in
        )
        # `str(a)` uses the `str` builtin — NOT in the risky list
        # because `str()` on most Python values is total. Confirm we
        # aren't overzealous. But + is not in the risky list either.
        self.assertFalse(_contains_implicit_raise_pattern(f))

    def test_attribute_on_self_does_not_flag(self) -> None:
        src = (
            "class C:\n"
            "    def f(self):\n"
            "        return self.x\n"
        )
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "f":
                self.assertFalse(_contains_implicit_raise_pattern(node))
                return
        self.fail("method f not found")


class TestMayRaiseImplicitAnalyzer(unittest.TestCase):
    """End-to-end analyzer test — ensure `may_raise_implicit`
    composes with `may_raise` the way the Phase-28 design says."""

    def test_division_implicit_not_explicit(self) -> None:
        src = "def f(x, y):\n    return x / y\n"
        sem = analyze_function(_parse_function(src))
        self.assertTrue(sem.may_raise_implicit)
        self.assertFalse(sem.may_raise)

    def test_explicit_and_implicit_both_flagged(self) -> None:
        src = (
            "def f(x, y):\n"
            "    if y == 0:\n"
            "        raise ValueError('bad')\n"
            "    return x / y\n"
        )
        sem = analyze_function(_parse_function(src))
        self.assertTrue(sem.may_raise)
        self.assertTrue(sem.may_raise_implicit)

    def test_catchall_suppresses_implicit(self) -> None:
        src = (
            "def f(x, y):\n"
            "    try:\n"
            "        return x / y\n"
            "    except Exception:\n"
            "        return None\n"
        )
        sem = analyze_function(_parse_function(src))
        self.assertFalse(sem.may_raise_implicit)

    def test_narrow_except_does_not_suppress_implicit(self) -> None:
        src = (
            "def f(x, y):\n"
            "    try:\n"
            "        return x / y\n"
            "    except ZeroDivisionError:\n"
            "        return 0\n"
        )
        sem = analyze_function(_parse_function(src))
        # Narrow except does not catch TypeError etc. → still flagged.
        self.assertTrue(sem.may_raise_implicit)

    def test_bare_except_suppresses_implicit(self) -> None:
        src = (
            "def f(x, y):\n"
            "    try:\n"
            "        return x / y\n"
            "    except:\n"
            "        return None\n"
        )
        sem = analyze_function(_parse_function(src))
        self.assertFalse(sem.may_raise_implicit)

    def test_benign_both_false(self) -> None:
        src = "def f():\n    return 0\n"
        sem = analyze_function(_parse_function(src))
        self.assertFalse(sem.may_raise)
        self.assertFalse(sem.may_raise_implicit)

    def test_as_tuple_length(self) -> None:
        sem = FunctionSemantics(
            may_raise=False, is_recursive=False,
            may_write_global=False, calls_subprocess=False,
            calls_filesystem=False, calls_network=False,
            may_raise_implicit=True,
        )
        # Phase 28 adds a 7th field at the end.
        self.assertEqual(len(sem.as_tuple()), 7)
        self.assertEqual(sem.as_tuple()[-1], True)


class TestInterprocPropagation(unittest.TestCase):
    """`trans_may_raise_implicit` propagates over the resolved call
    graph in the same way as the other trans-* flags."""

    def test_caller_inherits_implicit_from_callee(self) -> None:
        src = (
            "def helper(x, y):\n"
            "    return x / y\n"
            "def wrapper(a, b):\n"
            "    return helper(a, b)\n"
        )
        tree = ast.parse(textwrap.dedent(src))
        ctx, intra = build_module_context("m", tree)
        interproc, _cg = analyze_interproc([ctx], intra)
        self.assertTrue(interproc["m.helper"].trans_may_raise_implicit)
        self.assertTrue(interproc["m.wrapper"].trans_may_raise_implicit)

    def test_trans_may_raise_unchanged(self) -> None:
        """Phase 24's `may_raise` semantics are NOT altered by Phase 28."""
        src = (
            "def explicit():\n"
            "    raise ValueError('x')\n"
            "def implicit(x, y):\n"
            "    return x / y\n"
        )
        tree = ast.parse(textwrap.dedent(src))
        ctx, intra = build_module_context("m", tree)
        interproc, _cg = analyze_interproc([ctx], intra)
        self.assertTrue(interproc["m.explicit"].trans_may_raise)
        self.assertFalse(interproc["m.implicit"].trans_may_raise)
        self.assertFalse(interproc["m.explicit"].trans_may_raise_implicit)
        self.assertTrue(interproc["m.implicit"].trans_may_raise_implicit)


class TestExceptionOriginClassifier(unittest.TestCase):
    """Runtime classifier distinguishes explicit from implicit using
    the exception's innermost traceback frame + source line."""

    def test_explicit_classified(self) -> None:
        def target():
            raise ValueError("explicit")
        try:
            target()
        except BaseException as e:
            origin = _classify_exception_origin(target, e)
            self.assertEqual(origin, "explicit")
        else:
            self.fail("target did not raise")

    def test_implicit_arithmetic_classified(self) -> None:
        def target():
            return 1 / 0
        try:
            target()
        except BaseException as e:
            origin = _classify_exception_origin(target, e)
            self.assertEqual(origin, "implicit")
        else:
            self.fail("target did not raise")

    def test_implicit_from_called_helper_classified(self) -> None:
        def helper():
            raise RuntimeError("inside helper")
        def target():
            helper()
        try:
            target()
        except BaseException as e:
            origin = _classify_exception_origin(target, e)
            # The innermost frame is `helper`'s, not `target`'s —
            # classify as implicit with respect to the target.
            self.assertEqual(origin, "implicit")
        else:
            self.fail("target did not raise")

    def test_raise_lines_from_simple_function(self) -> None:
        def target():
            raise ValueError("hi")
        lines = _raise_line_numbers(target)
        # The function contains exactly one `raise` line. We don't
        # pin on an exact file line — just that the set is non-empty.
        self.assertEqual(len(lines), 1)


class TestProbeMayRaiseSplit(unittest.TestCase):
    """The split probe partitions observations into the two buckets."""

    def test_explicit_only(self) -> None:
        def target(x):
            raise ValueError("explicit")
        exp, imp = probe_may_raise_split(
            target, invocations=[(0,), (1,)])
        self.assertTrue(exp.runtime_flag)
        self.assertFalse(imp.runtime_flag)
        self.assertEqual(exp.n_runs, 2)
        self.assertEqual(exp.n_triggered, 2)
        self.assertEqual(imp.n_triggered, 0)

    def test_implicit_only(self) -> None:
        def target(x):
            return 1 / x  # ZeroDivisionError when x == 0
        exp, imp = probe_may_raise_split(
            target, invocations=[(0,)])
        self.assertFalse(exp.runtime_flag)
        self.assertTrue(imp.runtime_flag)
        self.assertEqual(imp.n_triggered, 1)
        self.assertEqual(exp.n_triggered, 0)

    def test_mixed_inputs(self) -> None:
        def target(mode):
            if mode == "explicit":
                raise RuntimeError("explicit path")
            return [][0]  # IndexError — implicit
        exp, imp = probe_may_raise_split(
            target, invocations=[("explicit",), ("implicit",)])
        self.assertTrue(exp.runtime_flag)
        self.assertTrue(imp.runtime_flag)
        self.assertEqual(exp.n_triggered, 1)
        self.assertEqual(imp.n_triggered, 1)

    def test_no_raise_no_trigger(self) -> None:
        def target(x):
            return x + 1
        exp, imp = probe_may_raise_split(
            target, invocations=[(0,), (1,)])
        self.assertFalse(exp.runtime_flag)
        self.assertFalse(imp.runtime_flag)


class TestProbePredicateDispatchPhase28(unittest.TestCase):
    """`probe_predicate` recognises the Phase-28 predicate labels."""

    def test_dispatch_explicit(self) -> None:
        def target():
            raise ValueError("x")
        obs = probe_predicate(
            "may_raise_explicit", target, module=None,  # type: ignore
            invocations=[()],
        )
        self.assertTrue(obs.runtime_flag)

    def test_dispatch_implicit(self) -> None:
        def target():
            return 1 / 0
        obs = probe_predicate(
            "may_raise_implicit", target, module=None,  # type: ignore
            invocations=[()],
        )
        self.assertTrue(obs.runtime_flag)

    def test_predicates_in_runtime_decidable_set(self) -> None:
        self.assertIn("may_raise_explicit", RUNTIME_DECIDABLE_PREDICATES)
        self.assertIn("may_raise_implicit", RUNTIME_DECIDABLE_PREDICATES)


class TestComputeStaticFlagsPhase28(unittest.TestCase):
    """`compute_static_flags_from_source` surfaces the new flags."""

    def test_implicit_only_function(self) -> None:
        src = "def f(x, y):\n    return x / y\n"
        flags = compute_static_flags_from_source(src, "f")
        self.assertIn("may_raise_explicit", flags)
        self.assertIn("may_raise_implicit", flags)
        self.assertFalse(flags["may_raise_explicit"])
        self.assertTrue(flags["may_raise_implicit"])

    def test_explicit_only_function(self) -> None:
        src = "def f(x):\n    raise ValueError('x')\n"
        flags = compute_static_flags_from_source(src, "f")
        self.assertTrue(flags["may_raise_explicit"])
        # Explicit-only function has no risky arithmetic / subscript /
        # attribute on params → implicit = False.
        self.assertFalse(flags["may_raise_implicit"])


if __name__ == "__main__":
    unittest.main()
