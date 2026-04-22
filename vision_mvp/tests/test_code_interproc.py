"""Tests for the Phase-25 interprocedural semantic analyser.

The analyser builds a local call graph across parsed modules, then
propagates each Phase-24 conservative predicate transitively along
resolved call edges until a least fixed point is reached. Separately,
it detects strongly-connected components (including self-loops) to
flag `participates_in_cycle`, and records calls to names that could
not be resolved via `has_unresolved_callees`.

Each test below pins one of:

  * monotonicity (trans ⊇ intra for every predicate),
  * correctness on a hand-built chain (A → B → C → effect),
  * mutual-recursion detection,
  * conservative handling of unresolved callees (no propagation,
    honest flag on the caller),
  * resolution cases (bare, self., Class., module alias,
    `from mod import name`),
  * SCC detection (iterative Tarjan handles deep graphs),
  * determinism (two runs on the same input produce the same output).
"""

from __future__ import annotations

import ast
import textwrap
import unittest

from vision_mvp.core.code_interproc import (
    CallGraph, InterprocSemantics, ModuleContext,
    analyze_interproc, build_call_graph, build_module_context,
    propagate_effect, strongly_connected_components,
)
from vision_mvp.core.code_semantics import FunctionSemantics


def _mod(name: str, src: str) -> tuple[ModuleContext, dict]:
    tree = ast.parse(textwrap.dedent(src))
    return build_module_context(name, tree)


def _analyze(modules: list[tuple[str, str]]) -> tuple[
        dict[str, InterprocSemantics], CallGraph]:
    ctxs = []
    intra_all: dict[str, FunctionSemantics] = {}
    for name, src in modules:
        ctx, intra = _mod(name, src)
        ctxs.append(ctx)
        intra_all.update(intra)
    return analyze_interproc(ctxs, intra_all)


# =============================================================================
# Monotonicity & basic propagation
# =============================================================================


class TestBasicPropagation(unittest.TestCase):
    def test_trans_is_superset_of_intra(self) -> None:
        """For every predicate, trans(f) must be True whenever
        intra(f) is True (the 0-length path). Test on a two-function
        module where helper directly raises and wrapper doesn't."""
        result, _ = _analyze([("m", """
            def helper():
                raise ValueError('x')

            def wrapper():
                return helper()
        """)])
        self.assertTrue(result["m.helper"].trans_may_raise)
        self.assertTrue(result["m.wrapper"].trans_may_raise,
                        "trans_may_raise should propagate wrapper→helper")

    def test_direct_call_propagates_one_hop(self) -> None:
        """Wrapper directly calling a subprocess-using helper."""
        result, _ = _analyze([("m", """
            import subprocess
            def run_cmd(c):
                return subprocess.run(c, shell=True)
            def wrapper(c):
                return run_cmd(c)
        """)])
        self.assertTrue(result["m.run_cmd"].trans_calls_subprocess)
        self.assertTrue(result["m.wrapper"].trans_calls_subprocess,
                        "wrapper→run_cmd→subprocess.run must propagate")

    def test_chain_of_three_hops_propagates(self) -> None:
        result, _ = _analyze([("m", """
            import subprocess
            def leaf(c):
                subprocess.run(c, shell=True)
            def a(c):
                leaf(c)
            def b(c):
                a(c)
            def c(cmd):
                b(cmd)
        """)])
        for name in ("leaf", "a", "b", "c"):
            self.assertTrue(result[f"m.{name}"].trans_calls_subprocess,
                            f"{name} should transitively call subprocess")

    def test_non_propagating_branches_stay_false(self) -> None:
        """A function that doesn't reach an effect stays False."""
        result, _ = _analyze([("m", """
            def pure(x):
                return x + 1
            def also_pure():
                return pure(42)
        """)])
        self.assertFalse(result["m.pure"].trans_may_raise)
        self.assertFalse(result["m.also_pure"].trans_may_raise)
        self.assertFalse(result["m.pure"].trans_calls_subprocess)
        self.assertFalse(result["m.also_pure"].trans_calls_subprocess)


class TestCrossModulePropagation(unittest.TestCase):
    def test_cross_module_via_from_import(self) -> None:
        """`from mod import helper` in one file; wrapper in another."""
        result, _ = _analyze([
            ("helpers", """
                import subprocess
                def run_cmd(c):
                    return subprocess.run(c, shell=True)
            """),
            ("caller", """
                from helpers import run_cmd
                def wrapper(c):
                    return run_cmd(c)
            """),
        ])
        self.assertTrue(result["caller.wrapper"].trans_calls_subprocess,
                        "cross-module from-import chain must propagate")

    def test_cross_module_via_module_alias(self) -> None:
        """`import helpers as h; h.run_cmd()` — alias-module form."""
        result, _ = _analyze([
            ("helpers", """
                import subprocess
                def run_cmd(c):
                    return subprocess.run(c, shell=True)
            """),
            ("caller", """
                import helpers as h
                def wrapper(c):
                    return h.run_cmd(c)
            """),
        ])
        self.assertTrue(result["caller.wrapper"].trans_calls_subprocess)


class TestMethodResolution(unittest.TestCase):
    def test_self_dot_method_resolves(self) -> None:
        """`self.do()` inside a class resolves to `Class.do`."""
        result, _ = _analyze([("m", """
            import subprocess
            class Runner:
                def do(self, c):
                    subprocess.run(c, shell=True)
                def wrapper(self, c):
                    return self.do(c)
        """)])
        self.assertTrue(result["m.Runner.do"].trans_calls_subprocess)
        self.assertTrue(result["m.Runner.wrapper"].trans_calls_subprocess)

    def test_class_qualified_method_resolves(self) -> None:
        """`Cls.method(instance, ...)` resolves when Cls is local."""
        result, _ = _analyze([("m", """
            import subprocess
            class Runner:
                def do(self, c):
                    subprocess.run(c, shell=True)
            def outer(instance, c):
                return Runner.do(instance, c)
        """)])
        self.assertTrue(result["m.outer"].trans_calls_subprocess)


# =============================================================================
# Unresolved callees
# =============================================================================


class TestUnresolvedCallees(unittest.TestCase):
    def test_unresolved_call_flags_caller(self) -> None:
        """A call to a name we can't resolve in the corpus and that
        isn't a known API surface must flag `has_unresolved_callees`."""
        result, _ = _analyze([("m", """
            def caller(x):
                return external_thing.do_it(x)
        """)])
        self.assertTrue(result["m.caller"].has_unresolved_callees)

    def test_known_api_call_is_NOT_unresolved(self) -> None:
        """`subprocess.run` is a known API — the intraprocedural
        analyzer captures it. It must NOT be tallied as unresolved."""
        result, _ = _analyze([("m", """
            import subprocess
            def caller(c):
                return subprocess.run(c, shell=True)
        """)])
        self.assertFalse(result["m.caller"].has_unresolved_callees)
        # But intra-subprocess IS captured — the interprocedural
        # trans-flag still fires via the direct intra flag (length-0
        # resolved chain is trivial).
        self.assertTrue(result["m.caller"].trans_calls_subprocess)

    def test_unresolved_call_does_NOT_propagate_effects(self) -> None:
        """Resolved-only stance: an unresolved call contributes
        nothing to trans-flags. The caller's `trans_*` stays False
        unless some OTHER resolved path hits an effect."""
        result, _ = _analyze([("m", """
            def caller(x):
                # external lib could do anything, but we don't know.
                return external_lib.query(x)
        """)])
        self.assertFalse(result["m.caller"].trans_calls_subprocess)
        self.assertFalse(result["m.caller"].trans_calls_network)
        self.assertTrue(result["m.caller"].has_unresolved_callees,
                        "unresolved callees flag must be honest")


# =============================================================================
# Cycle detection (SCC)
# =============================================================================


class TestCycleDetection(unittest.TestCase):
    def test_self_recursive_is_in_cycle(self) -> None:
        result, _ = _analyze([("m", """
            def fac(n):
                if n <= 1:
                    return 1
                return n * fac(n - 1)
        """)])
        self.assertTrue(result["m.fac"].participates_in_cycle)

    def test_mutual_recursion_flags_both(self) -> None:
        """f → g → f. Both members of the SCC must be flagged."""
        result, _ = _analyze([("m", """
            def f(x):
                return g(x)
            def g(x):
                return f(x)
        """)])
        self.assertTrue(result["m.f"].participates_in_cycle)
        self.assertTrue(result["m.g"].participates_in_cycle)

    def test_three_cycle_flags_all(self) -> None:
        """a → b → c → a. All three must be in cycle."""
        result, _ = _analyze([("m", """
            def a():
                b()
            def b():
                c()
            def c():
                a()
        """)])
        for name in ("a", "b", "c"):
            self.assertTrue(result[f"m.{name}"].participates_in_cycle)

    def test_non_cycle_chain_is_not_flagged(self) -> None:
        """a → b → c (no back edge). None in cycle."""
        result, _ = _analyze([("m", """
            def a():
                b()
            def b():
                c()
            def c():
                pass
        """)])
        for name in ("a", "b", "c"):
            self.assertFalse(result[f"m.{name}"].participates_in_cycle)

    def test_cross_module_mutual_recursion(self) -> None:
        """Mutual recursion across two files: f in A, g in B."""
        result, _ = _analyze([
            ("A", """
                from B import g
                def f(x):
                    return g(x)
            """),
            ("B", """
                from A import f
                def g(x):
                    return f(x)
            """),
        ])
        self.assertTrue(result["A.f"].participates_in_cycle)
        self.assertTrue(result["B.g"].participates_in_cycle)


# =============================================================================
# Tarjan correctness
# =============================================================================


class TestTarjan(unittest.TestCase):
    def test_single_node_no_self_loop_is_not_an_scc_by_our_rule(self) -> None:
        """A node with no self-edge is its own SCC of size 1 — NOT
        flagged as in_cycle by our rule (we only flag SCC≥2 or
        explicit self-loops)."""
        sccs = strongly_connected_components(
            ["a", "b"], {"a": {"b"}, "b": set()},
        )
        # Expect two SCCs, each of size 1.
        sizes = sorted(len(s) for s in sccs)
        self.assertEqual(sizes, [1, 1])

    def test_iterative_tarjan_deep_chain(self) -> None:
        """1000-node chain — ensure no recursion-depth crash. Each
        node is its own SCC."""
        n = 1000
        nodes = [str(i) for i in range(n)]
        edges = {str(i): {str(i + 1)} for i in range(n - 1)}
        edges[str(n - 1)] = set()
        sccs = strongly_connected_components(nodes, edges)
        self.assertEqual(len(sccs), n, "every node is its own SCC")

    def test_tarjan_finds_strongly_connected_cluster(self) -> None:
        # Graph: a ↔ b ↔ c all connected, d isolated.
        sccs = strongly_connected_components(
            ["a", "b", "c", "d"],
            {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()},
        )
        sizes = sorted(len(s) for s in sccs)
        self.assertEqual(sizes, [1, 3])


# =============================================================================
# Fixed-point propagation properties
# =============================================================================


class TestFixedPointProperties(unittest.TestCase):
    def test_propagation_is_monotone_in_intra(self) -> None:
        """Adding any True intra flag to the input can only add Trues
        to the propagated map."""
        cg = CallGraph(
            nodes={"a", "b", "c"},
            edges={"a": {"b"}, "b": {"c"}, "c": set()},
            rev_edges={"b": {"a"}, "c": {"b"}},
        )
        all_false = {"a": False, "b": False, "c": False}
        with_c = {"a": False, "b": False, "c": True}
        r_false = propagate_effect(cg, all_false)
        r_c = propagate_effect(cg, with_c)
        for n in ("a", "b", "c"):
            self.assertLessEqual(int(r_false[n]), int(r_c[n]),
                                 f"monotonicity violated at {n}")

    def test_propagation_is_idempotent(self) -> None:
        """Running propagation twice on the same input must produce
        the same output."""
        result1, _ = _analyze([("m", """
            def a():
                b()
            def b():
                raise ValueError()
        """)])
        result2, _ = _analyze([("m", """
            def a():
                b()
            def b():
                raise ValueError()
        """)])
        self.assertEqual(result1, result2)

    def test_propagation_reaches_least_fixed_point_not_top(self) -> None:
        """Unrelated functions must NOT become True just because some
        other function is True (we compute the least, not greatest,
        fixed point)."""
        result, _ = _analyze([("m", """
            def unrelated(x):
                return x
            def bad():
                raise RuntimeError()
        """)])
        self.assertFalse(result["m.unrelated"].trans_may_raise,
                         "unrelated must stay False")
        self.assertTrue(result["m.bad"].trans_may_raise)


# =============================================================================
# Integration with extract_metadata / CodeIndexer
# =============================================================================


class TestIndexerIntegration(unittest.TestCase):
    def test_indexer_populates_interproc_tuples(self) -> None:
        """Full pipeline: CodeIndexer walks a mini corpus and ends up
        with parallel interproc tuples on each handle."""
        import os
        import tempfile
        from vision_mvp.core.code_index import CodeIndexer
        from vision_mvp.core.context_ledger import ContextLedger, hash_embedding

        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "helpers.py"), "w") as f:
                f.write(textwrap.dedent("""
                    import subprocess
                    def run_cmd(c):
                        subprocess.run(c, shell=True)
                """))
            with open(os.path.join(root, "callers.py"), "w") as f:
                f.write(textwrap.dedent("""
                    from helpers import run_cmd
                    def wrapper(c):
                        return run_cmd(c)
                    def f(x):
                        return g(x)
                    def g(x):
                        return f(x)
                """))
            indexer = CodeIndexer(root=root)
            ledger = ContextLedger(
                embed_dim=32,
                embed_fn=lambda t: hash_embedding(t, dim=32))
            handles = indexer.index_into(ledger)
            by_name = {
                h.metadata_dict()["module_name"]: h
                for h in handles
            }
            # `callers.wrapper` must be flagged as
            # trans_calls_subprocess through the resolved
            # `from helpers import run_cmd` chain.
            cm = by_name["callers"].metadata_dict()
            names = cm["semantic_function_names"]
            trans_sp = cm["function_trans_calls_subprocess"]
            trans_cy = cm["function_participates_in_cycle"]
            idx = {n: i for i, n in enumerate(names)}
            self.assertTrue(trans_sp[idx["wrapper"]])
            self.assertTrue(trans_cy[idx["f"]])
            self.assertTrue(trans_cy[idx["g"]])
            self.assertEqual(cm["n_functions_trans_calls_subprocess"], 1)
            self.assertEqual(cm["n_functions_participates_in_cycle"], 2)


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism(unittest.TestCase):
    def test_two_runs_same_output(self) -> None:
        """Same input → same interproc map (order, values)."""
        r1, _ = _analyze([("m", """
            import subprocess
            def h(c): subprocess.run(c, shell=True)
            def w(c): h(c)
        """)])
        r2, _ = _analyze([("m", """
            import subprocess
            def h(c): subprocess.run(c, shell=True)
            def w(c): h(c)
        """)])
        self.assertEqual(r1, r2)


# =============================================================================
# Soundness corner cases
# =============================================================================


class TestSoundnessCornerCases(unittest.TestCase):
    def test_reflection_opaque_call_does_not_propagate(self) -> None:
        """`getattr(mod, 'x')()` is opaque — no effect propagation.
        (Also: `_call_name` returns None for reflection, so no
        unresolved-callee flag either — this is a known boundary
        condition, documented in the module.)"""
        result, _ = _analyze([("m", """
            def caller(name):
                fn = getattr(mod, name)
                return fn()
        """)])
        self.assertFalse(result["m.caller"].trans_calls_subprocess)

    def test_no_call_sites_means_everything_false(self) -> None:
        """A function with zero calls has no outgoing edges; only its
        own intra flags matter."""
        result, _ = _analyze([("m", """
            def pure(x):
                return x + 1
        """)])
        sem = result["m.pure"]
        for flag in (sem.trans_may_raise, sem.trans_may_write_global,
                     sem.trans_calls_subprocess, sem.trans_calls_filesystem,
                     sem.trans_calls_network, sem.participates_in_cycle,
                     sem.has_unresolved_callees):
            self.assertFalse(flag)

    def test_try_except_suppresses_may_raise_in_trans(self) -> None:
        """If the helper's raise is caught inside itself, wrapper's
        trans_may_raise must be False — the helper's intra is False."""
        result, _ = _analyze([("m", """
            def helper():
                try:
                    raise ValueError('caught')
                except BaseException:
                    return
            def wrapper():
                return helper()
        """)])
        self.assertFalse(result["m.helper"].trans_may_raise)
        self.assertFalse(result["m.wrapper"].trans_may_raise)

    def test_mutual_recursion_without_effects_is_in_cycle_but_effects_false(
            self) -> None:
        """SCC participation is independent of effect propagation."""
        result, _ = _analyze([("m", """
            def f(x):
                return g(x)
            def g(x):
                return f(x)
        """)])
        self.assertTrue(result["m.f"].participates_in_cycle)
        self.assertFalse(result["m.f"].trans_calls_subprocess)
        self.assertFalse(result["m.f"].trans_may_raise)


if __name__ == "__main__":
    unittest.main()
