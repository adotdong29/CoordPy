"""Tests for the conservative static semantic analyser — Phase 24.

The analyser is **conservative**: soundness is the first requirement.
Every test below pins one of:

  * a true-positive (`may_raise = True` when the code really can raise),
  * a true-negative (`may_raise = False` when the code provably cannot
    raise under the analysis's scope),
  * a known-boundary false-positive (e.g. "if False: raise X" still
    flags True because we don't do dead-code analysis).

The test cases are tiny AST snippets so the soundness claims are
easy to read and review. Each test documents *which* analysis
property it is pinning and, where relevant, what the conservative
fallback behaviour should be.
"""

from __future__ import annotations

import ast
import unittest

from vision_mvp.core.code_semantics import (
    FunctionSemantics, analyze_function, analyze_module,
    _collect_import_hints, _module_level_names,
)


def _first_func(src: str) -> tuple[ast.FunctionDef, ast.Module]:
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node, tree
    raise AssertionError("no top-level function found in snippet")


def _sem(src: str, *, enclosing_class: str | None = None) -> FunctionSemantics:
    """Analyse the first top-level function in `src`."""
    fn, tree = _first_func(src)
    return analyze_function(fn, tree=tree, enclosing_class=enclosing_class)


# =============================================================================
# may_raise
# =============================================================================


class TestMayRaise(unittest.TestCase):
    def test_explicit_raise_is_detected(self) -> None:
        sem = _sem("def f():\n    raise ValueError('bad')\n")
        self.assertTrue(sem.may_raise)

    def test_no_raise_returns_false(self) -> None:
        sem = _sem("def f(x):\n    return x + 1\n")
        self.assertFalse(sem.may_raise)

    def test_conditional_raise_is_detected(self) -> None:
        sem = _sem(
            "def f(x):\n"
            "    if x < 0:\n"
            "        raise ValueError('negative')\n"
            "    return x\n"
        )
        self.assertTrue(sem.may_raise,
                        "even a conditional raise must flag may_raise=True")

    def test_raise_in_nested_loop(self) -> None:
        sem = _sem(
            "def f(xs):\n"
            "    for x in xs:\n"
            "        if x is None:\n"
            "            raise ValueError('null')\n"
        )
        self.assertTrue(sem.may_raise)

    def test_try_except_catches_everything_is_not_flagged(self) -> None:
        # Bare `except:` catches every exception, so the raise
        # cannot escape the function.
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        raise ValueError('caught')\n"
            "    except:\n"
            "        return\n"
        )
        self.assertFalse(sem.may_raise,
                         "bare except suppresses the raise; must be False")

    def test_try_except_base_exception_is_not_flagged(self) -> None:
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        raise RuntimeError('boom')\n"
            "    except BaseException:\n"
            "        return\n"
        )
        self.assertFalse(sem.may_raise)

    def test_narrow_except_still_flags_may_raise(self) -> None:
        # `except ValueError` doesn't catch e.g. RuntimeError; we must
        # conservatively flag the function as may_raise.
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        raise RuntimeError('boom')\n"
            "    except ValueError:\n"
            "        return\n"
        )
        self.assertTrue(sem.may_raise,
                        "narrow except leaves other exceptions uncaught")

    def test_raise_in_except_handler_is_flagged(self) -> None:
        # A raise that lives inside an EXCEPT handler is NOT lexically in
        # the try body, so it is not "caught" by its own try. It
        # propagates out → may_raise=True.
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        raise RuntimeError('rethrown')\n"
        )
        self.assertTrue(sem.may_raise)

    def test_raise_in_else_handler_is_flagged(self) -> None:
        # Even more subtle: try/except/else — a raise in `else` is NOT
        # covered by the handlers of that try.
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        return\n"
            "    else:\n"
            "        raise ValueError('in else')\n"
        )
        self.assertTrue(sem.may_raise)

    def test_tuple_except_with_catch_all_suppresses(self) -> None:
        # `except (TypeError, Exception):` contains Exception → catches
        # everything user-raisable.
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        raise RuntimeError('x')\n"
            "    except (TypeError, Exception):\n"
            "        return\n"
        )
        self.assertFalse(sem.may_raise)

    def test_nested_try_inner_catches(self) -> None:
        sem = _sem(
            "def f():\n"
            "    try:\n"
            "        try:\n"
            "            raise ValueError('inner')\n"
            "        except BaseException:\n"
            "            return\n"
            "    except Exception:\n"
            "        return\n"
        )
        self.assertFalse(sem.may_raise,
                         "inner BaseException handler catches the raise")


# =============================================================================
# is_recursive
# =============================================================================


class TestIsRecursive(unittest.TestCase):
    def test_simple_self_call(self) -> None:
        sem = _sem(
            "def fact(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
        )
        self.assertTrue(sem.is_recursive)

    def test_no_self_call_is_false(self) -> None:
        sem = _sem("def f(xs):\n    return sum(xs)\n")
        self.assertFalse(sem.is_recursive)

    def test_calls_other_function_is_false(self) -> None:
        sem = _sem(
            "def f(xs):\n"
            "    return other(xs) + 1\n"
        )
        self.assertFalse(sem.is_recursive,
                         "calling some other name must not flag is_recursive")

    def test_self_method_call(self) -> None:
        # Method inside class calling self.name → recursive
        src = (
            "class C:\n"
            "    def walk(self, n):\n"
            "        if n <= 0:\n"
            "            return\n"
            "        self.walk(n - 1)\n"
        )
        tree = ast.parse(src)
        cls = [n for n in ast.iter_child_nodes(tree)
               if isinstance(n, ast.ClassDef)][0]
        fn = [n for n in ast.iter_child_nodes(cls)
              if isinstance(n, ast.FunctionDef)][0]
        sem = analyze_function(fn, tree=tree, enclosing_class="C")
        self.assertTrue(sem.is_recursive)

    def test_class_qualified_self_call(self) -> None:
        src = (
            "class C:\n"
            "    def walk(self, n):\n"
            "        if n <= 0:\n"
            "            return\n"
            "        C.walk(self, n - 1)\n"
        )
        tree = ast.parse(src)
        cls = [n for n in ast.iter_child_nodes(tree)
               if isinstance(n, ast.ClassDef)][0]
        fn = [n for n in ast.iter_child_nodes(cls)
              if isinstance(n, ast.FunctionDef)][0]
        sem = analyze_function(fn, tree=tree, enclosing_class="C")
        self.assertTrue(sem.is_recursive)

    def test_mutual_recursion_is_not_flagged(self) -> None:
        # Known conservative limitation: we don't track cross-function
        # cycles. `a` calls `b` but never itself → is_recursive=False.
        sem = _sem(
            "def a(x):\n"
            "    return b(x - 1)\n"
        )
        self.assertFalse(sem.is_recursive,
                         "mutual recursion explicitly out of scope")


# =============================================================================
# may_write_global
# =============================================================================


class TestMayWriteGlobal(unittest.TestCase):
    def test_global_statement_plus_assign(self) -> None:
        src = (
            "X = 0\n"
            "def f():\n"
            "    global X\n"
            "    X = 42\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.may_write_global)

    def test_global_statement_without_assign_is_false(self) -> None:
        # `global X` alone does not write anything. Must be False.
        src = (
            "X = 0\n"
            "def f():\n"
            "    global X\n"
            "    return X\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertFalse(sem.may_write_global,
                         "`global` declaration alone is not a write")

    def test_local_assignment_only_is_false(self) -> None:
        src = (
            "X = 0\n"
            "def f():\n"
            "    X = 42\n"      # no `global` → shadows, doesn't write
            "    return X\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertFalse(sem.may_write_global)

    def test_mutation_of_module_level_list_via_append(self) -> None:
        src = (
            "REG = []\n"
            "def f():\n"
            "    REG.append(1)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.may_write_global,
                        "mutating a module-level list must flag may_write_global")

    def test_attribute_assignment_to_module_level_name(self) -> None:
        src = (
            "class Cfg:\n"
            "    pass\n"
            "CFG = Cfg()\n"
            "def f():\n"
            "    CFG.flag = True\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.may_write_global,
                        "module-level attribute assignment must flag may_write_global")

    def test_nonlocal_is_not_global(self) -> None:
        # `nonlocal` refers to an enclosing function's scope, not
        # module-level. Must NOT be flagged.
        src = (
            "def outer():\n"
            "    x = 0\n"
            "    def inner():\n"
            "        nonlocal x\n"
            "        x = 1\n"
            "    return inner\n"
        )
        # Analyse the inner function specifically.
        tree = ast.parse(src)
        outer = [n for n in ast.iter_child_nodes(tree)
                 if isinstance(n, ast.FunctionDef)][0]
        inner = [n for n in ast.iter_child_nodes(outer)
                 if isinstance(n, ast.FunctionDef)][0]
        sem = analyze_function(inner, tree=tree)
        self.assertFalse(sem.may_write_global,
                         "`nonlocal` is not global — must not flag")


# =============================================================================
# IO calls (subprocess / filesystem / network)
# =============================================================================


class TestIOCalls(unittest.TestCase):
    def test_subprocess_run_via_dotted_call(self) -> None:
        src = (
            "import subprocess\n"
            "def f(cmd):\n"
            "    subprocess.run(cmd)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_subprocess)

    def test_subprocess_run_via_import_alias(self) -> None:
        # `from subprocess import run; run(cmd)`
        src = (
            "from subprocess import run\n"
            "def f(cmd):\n"
            "    run(cmd)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_subprocess,
                        "named-import call into subprocess must be detected")

    def test_os_system_as_subprocess(self) -> None:
        src = (
            "import os\n"
            "def f(cmd):\n"
            "    os.system(cmd)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_subprocess)
        self.assertFalse(sem.calls_filesystem)

    def test_open_is_filesystem_without_import(self) -> None:
        # `open` is a builtin; must match without any import.
        src = (
            "def f(path):\n"
            "    with open(path) as fh:\n"
            "        return fh.read()\n"
        )
        sem = _sem(src)
        self.assertTrue(sem.calls_filesystem)
        self.assertFalse(sem.calls_network)

    def test_os_remove_is_filesystem(self) -> None:
        src = (
            "import os\n"
            "def f(p):\n"
            "    os.remove(p)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_filesystem)

    def test_shutil_rmtree_is_filesystem(self) -> None:
        src = (
            "import shutil\n"
            "def f(p):\n"
            "    shutil.rmtree(p)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_filesystem)

    def test_pathlib_read_text_requires_import_hint(self) -> None:
        # `p.read_text()` flagged only when pathlib is imported.
        src_with = (
            "import pathlib\n"
            "def f(p):\n"
            "    return p.read_text()\n"
        )
        fn, tree = _first_func(src_with)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_filesystem)

        src_without = (
            "def f(p):\n"
            "    return p.read_text()\n"
        )
        fn2, tree2 = _first_func(src_without)
        sem2 = analyze_function(fn2, tree=tree2)
        self.assertFalse(
            sem2.calls_filesystem,
            "without the pathlib import hint, read_text is ambiguous — "
            "must not flag (false negative is the conservative choice here "
            "given how many read_text-style methods exist on non-Path objects)",
        )

    def test_requests_get_is_network(self) -> None:
        src = (
            "import requests\n"
            "def f(url):\n"
            "    return requests.get(url)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_network)

    def test_urllib_urlopen_via_named_import(self) -> None:
        src = (
            "from urllib.request import urlopen\n"
            "def f(url):\n"
            "    return urlopen(url)\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_network)

    def test_socket_is_network(self) -> None:
        src = (
            "import socket\n"
            "def f():\n"
            "    return socket.socket()\n"
        )
        fn, tree = _first_func(src)
        sem = analyze_function(fn, tree=tree)
        self.assertTrue(sem.calls_network)

    def test_pure_function_flags_nothing(self) -> None:
        sem = _sem("def add(a, b):\n    return a + b\n")
        self.assertFalse(sem.may_raise)
        self.assertFalse(sem.is_recursive)
        self.assertFalse(sem.may_write_global)
        self.assertFalse(sem.calls_subprocess)
        self.assertFalse(sem.calls_filesystem)
        self.assertFalse(sem.calls_network)
        self.assertFalse(sem.calls_external_io)

    def test_reflection_is_not_flagged(self) -> None:
        # `getattr(module, name)(...)` cannot be resolved statically →
        # conservative FALSE (known limitation, documented).
        src = (
            "def f(mod, name):\n"
            "    return getattr(mod, name)()\n"
        )
        sem = _sem(src)
        self.assertFalse(sem.calls_filesystem)
        self.assertFalse(sem.calls_network)
        self.assertFalse(sem.calls_subprocess)


# =============================================================================
# calls_external_io convenience
# =============================================================================


class TestCallsExternalIO(unittest.TestCase):
    def test_subprocess_triggers_external_io(self) -> None:
        sem = _sem(
            "import subprocess\n"
            "def f():\n"
            "    subprocess.run(['ls'])\n"
        )
        self.assertTrue(sem.calls_external_io)

    def test_filesystem_triggers_external_io(self) -> None:
        sem = _sem("def f(p):\n    return open(p).read()\n")
        self.assertTrue(sem.calls_external_io)

    def test_network_triggers_external_io(self) -> None:
        sem = _sem(
            "import requests\n"
            "def f(u):\n"
            "    return requests.get(u)\n"
        )
        self.assertTrue(sem.calls_external_io)

    def test_pure_fn_has_no_external_io(self) -> None:
        sem = _sem("def f(x):\n    return x * 2\n")
        self.assertFalse(sem.calls_external_io)


# =============================================================================
# analyze_module aggregation
# =============================================================================


class TestAnalyzeModule(unittest.TestCase):
    def test_module_with_top_level_and_methods(self) -> None:
        src = (
            "import subprocess\n"
            "def top_fn():\n"
            "    subprocess.run(['a'])\n"
            "class Thing:\n"
            "    def method(self):\n"
            "        raise ValueError\n"
            "    def pure(self, x):\n"
            "        return x + 1\n"
        )
        results = analyze_module(ast.parse(src))
        names = [n for n, _s in results]
        self.assertEqual(names, ["top_fn", "Thing.method", "Thing.pure"])
        by_name = dict(results)
        self.assertTrue(by_name["top_fn"].calls_subprocess)
        self.assertTrue(by_name["Thing.method"].may_raise)
        self.assertFalse(by_name["Thing.pure"].may_raise)
        self.assertFalse(by_name["Thing.pure"].calls_external_io)


# =============================================================================
# Helpers
# =============================================================================


class TestHelpers(unittest.TestCase):
    def test_collect_import_hints(self) -> None:
        src = (
            "import os\n"
            "import os.path\n"
            "from subprocess import run\n"
        )
        hints = _collect_import_hints(ast.parse(src))
        self.assertIn("os", hints)
        self.assertIn("os.path", hints)
        self.assertIn("subprocess", hints)
        self.assertIn("run", hints)
        self.assertIn("subprocess.run", hints)

    def test_collect_module_level_names(self) -> None:
        src = (
            "X = 1\n"
            "Y: int = 2\n"
            "def f(): pass\n"
            "class C: pass\n"
            "import os\n"
            "from pathlib import Path\n"
        )
        names = _module_level_names(ast.parse(src))
        self.assertIn("X", names)
        self.assertIn("Y", names)
        self.assertIn("f", names)
        self.assertIn("C", names)
        self.assertIn("os", names)
        self.assertIn("Path", names)


if __name__ == "__main__":
    unittest.main()
