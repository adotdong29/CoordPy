"""Curated ``SafeRecipeRegistry`` for Phase-27 corpus-scale runtime
calibration.

The Phase-27 harness auto-derives recipes for zero-arg functions
(``ready_no_args``) and for functions whose positional parameters are
all annotated with types in a small whitelist (``ready_typed``).
Anything outside that slice is classified ``unsupported_untyped`` and
skipped.

A handful of corpus functions are worth probing even though they lie
outside the whitelist — for example ``code_semantics._call_name`` takes
an ``ast.AST`` node, which is never synthesisable from the default
fuzz pool. For those we ship a small, hand-authored, code-reviewed
``InvocationRecipe`` here.

Design guardrails:

  * **Pure functions only.** Every curated entry points at a function
    that, to our best analysis, has no side effect beyond what the
    Phase-26 sandbox already neuters. This is the trust boundary
    documented in ``code_corpus_runtime.py`` §"Safety stance".
  * **Small, local arguments.** We synthesise tiny ast nodes, tiny
    frozensets, tiny mappings — not real corpus-derived objects.
  * **No I/O at argument-construction time.** The ``build`` callable
    must not read the filesystem, open sockets, or spawn subprocesses.
  * **Deterministic given ``seed``.** The same seed produces the same
    argument sequence — a Phase-26 invariant the probe depends on.

Adding a curated recipe requires three things:

  1. The function is reachable from the corpus the benchmark indexes.
  2. The function's signature is stable enough that a single-shape
     recipe works. Otherwise register it as a Phase-27 coverage
     exclusion instead.
  3. A one-line justification in the docstring.
"""

from __future__ import annotations

import ast
import random
from typing import Any, Callable

from ..core.code_corpus_runtime import (
    InvocationRecipe, SafeRecipeRegistry, curated_recipe,
)


# =============================================================================
# ast-node arg factories
# =============================================================================


def _fresh_name_node(ident: str = "x") -> ast.AST:
    """A simple ``ast.Name`` node. ``_call_name`` resolves these
    directly."""
    return ast.Name(id=ident, ctx=ast.Load())


def _fresh_call_node(name: str = "foo") -> ast.AST:
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[], keywords=[],
    )


def _fresh_attr_call_node(base: str = "subprocess",
                           attr: str = "run") -> ast.AST:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=base, ctx=ast.Load()),
            attr=attr, ctx=ast.Load(),
        ),
        args=[], keywords=[],
    )


def _fresh_module_with_raise() -> ast.Module:
    return ast.parse("def f():\n    raise ValueError('x')\n")


def _fresh_function_def_raise() -> ast.FunctionDef:
    mod = _fresh_module_with_raise()
    return [n for n in mod.body if isinstance(n, ast.FunctionDef)][0]


def _fresh_function_def_pure() -> ast.FunctionDef:
    mod = ast.parse("def g():\n    return 1\n")
    return [n for n in mod.body if isinstance(n, ast.FunctionDef)][0]


def _fresh_module_with_try_catch_all() -> ast.Module:
    return ast.parse(
        "def h():\n"
        "    try:\n"
        "        raise RuntimeError('x')\n"
        "    except Exception:\n"
        "        return 0\n"
    )


# =============================================================================
# Recipe builders
# =============================================================================


def _recipe_call_name(rng: random.Random) -> list[tuple[Any, ...]]:
    """Arguments for ``code_semantics._call_name``: an AST node (Name,
    Call, or Attribute)."""
    return [
        (_fresh_name_node("foo"),),
        (_fresh_call_node("bar"),),
        (_fresh_attr_call_node("subprocess", "run"),),
    ]


def _recipe_call_matches(rng: random.Random) -> list[tuple[Any, ...]]:
    """``code_semantics._call_matches``: (dotted_name, table)."""
    from ..core.code_semantics import (
        _FILESYSTEM_APIS, _NETWORK_APIS, _SUBPROCESS_APIS,
    )
    choices: list[tuple[str, frozenset[str]]] = [
        ("subprocess.run", _SUBPROCESS_APIS),
        ("open", _FILESYSTEM_APIS),
        ("socket.socket", _NETWORK_APIS),
        ("nonexistent.func", _SUBPROCESS_APIS),
    ]
    rng.shuffle(choices)
    return [(name, table) for name, table in choices[:3]]


def _recipe_collect_store_names(rng: random.Random) -> list[tuple[Any, ...]]:
    """``code_semantics._collect_store_names``: an AST expression node
    usable on the LHS of an assignment."""
    return [
        (ast.Name(id="a", ctx=ast.Store()),),
        (ast.Tuple(elts=[
            ast.Name(id="x", ctx=ast.Store()),
            ast.Name(id="y", ctx=ast.Store()),
        ], ctx=ast.Store()),),
        (ast.Subscript(
            value=ast.Name(id="d", ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Store(),
        ),),
    ]


def _recipe_analyze_may_raise(rng: random.Random) -> list[tuple[Any, ...]]:
    """``code_semantics._analyze_may_raise``: a FunctionDef node."""
    return [
        (_fresh_function_def_raise(),),
        (_fresh_function_def_pure(),),
    ]


def _recipe_enclosing_try_nodes(rng: random.Random) -> list[tuple[Any, ...]]:
    return [
        (_fresh_function_def_pure(),),
        (_fresh_function_def_raise(),),
    ]


def _recipe_raise_is_caught(rng: random.Random) -> list[tuple[Any, ...]]:
    mod = _fresh_module_with_try_catch_all()
    raise_nodes = [n for n in ast.walk(mod) if isinstance(n, ast.Raise)]
    try_nodes = [n for n in ast.walk(mod) if isinstance(n, ast.Try)]
    if not raise_nodes or not try_nodes:
        return []
    return [(raise_nodes[0], try_nodes)]


def _recipe_node_contains(rng: random.Random) -> list[tuple[Any, ...]]:
    mod = _fresh_module_with_raise()
    return [
        (mod, mod.body[0]),
        (mod, ast.Name(id="not_here", ctx=ast.Load())),
    ]


def _recipe_analyze_is_recursive(rng: random.Random) -> list[tuple[Any, ...]]:
    fn = _fresh_function_def_pure()
    return [(fn, None)]


def _recipe_analyze_may_write_global(rng: random.Random) -> list[tuple[Any, ...]]:
    fn = _fresh_function_def_pure()
    return [(fn, frozenset({"X", "Y"}))]


def _recipe_module_level_names(rng: random.Random) -> list[tuple[Any, ...]]:
    return [(_fresh_module_with_raise(),)]


def _recipe_collect_import_hints(rng: random.Random) -> list[tuple[Any, ...]]:
    tree = ast.parse("import os\nfrom subprocess import run\n")
    return [(tree,)]


def _recipe_analyze_io_calls(rng: random.Random) -> list[tuple[Any, ...]]:
    fn = _fresh_function_def_pure()
    return [(fn, frozenset({"os", "subprocess"}))]


def _recipe_analyze_function(rng: random.Random) -> list[tuple[Any, ...]]:
    fn = _fresh_function_def_raise()
    tree = _fresh_module_with_raise()
    return [(fn,)]


def _recipe_analyze_module(rng: random.Random) -> list[tuple[Any, ...]]:
    tree = _fresh_module_with_raise()
    return [(tree,)]


def _recipe_function_is_generator(rng: random.Random) -> list[tuple[Any, ...]]:
    fn_pure = _fresh_function_def_pure()
    gen_src = ast.parse("def g():\n    yield 1\n").body[0]
    return [(fn_pure,), (gen_src,)]


def _recipe_bare_method_matches(rng: random.Random) -> list[tuple[Any, ...]]:
    call = _fresh_attr_call_node("p", "read_text")
    return [(call, frozenset({"read_text", "write_text"}))]


# =============================================================================
# The default registry
# =============================================================================


def build_default_recipe_registry() -> SafeRecipeRegistry:
    """Return the Phase-27 default registry.

    Every entry maps ``(module_name, qname) → InvocationRecipe``. The
    module names are the importable dotted paths used by
    ``build_corpus_static_flags`` / ``discover_candidates`` when run on
    ``vision_mvp/core``. Adding a new corpus means adding a new block
    of entries — entries are keyed by the exact dotted module name so
    multi-corpus registries compose cleanly.
    """
    reg = SafeRecipeRegistry()

    # vision_mvp.core.code_semantics — pure AST helpers
    cs = "vision_mvp.core.code_semantics"
    reg.register(cs, "_call_name", curated_recipe(_recipe_call_name))
    reg.register(cs, "_call_matches", curated_recipe(_recipe_call_matches))
    reg.register(cs, "_collect_store_names",
                 curated_recipe(_recipe_collect_store_names))
    reg.register(cs, "_analyze_may_raise",
                 curated_recipe(_recipe_analyze_may_raise))
    reg.register(cs, "_enclosing_try_nodes",
                 curated_recipe(_recipe_enclosing_try_nodes))
    reg.register(cs, "_raise_is_caught",
                 curated_recipe(_recipe_raise_is_caught))
    reg.register(cs, "_node_contains",
                 curated_recipe(_recipe_node_contains))
    reg.register(cs, "_analyze_is_recursive",
                 curated_recipe(_recipe_analyze_is_recursive))
    reg.register(cs, "_analyze_may_write_global",
                 curated_recipe(_recipe_analyze_may_write_global))
    reg.register(cs, "_module_level_names",
                 curated_recipe(_recipe_module_level_names))
    reg.register(cs, "_collect_import_hints",
                 curated_recipe(_recipe_collect_import_hints))
    reg.register(cs, "_analyze_io_calls",
                 curated_recipe(_recipe_analyze_io_calls))
    reg.register(cs, "_bare_method_matches",
                 curated_recipe(_recipe_bare_method_matches))
    reg.register(cs, "analyze_function",
                 curated_recipe(_recipe_analyze_function))
    reg.register(cs, "analyze_module",
                 curated_recipe(_recipe_analyze_module))

    # vision_mvp.core.code_corpus_runtime — generator classifier
    ccr = "vision_mvp.core.code_corpus_runtime"
    reg.register(ccr, "_function_is_generator",
                 curated_recipe(_recipe_function_is_generator))

    return reg


# =============================================================================
# Module exclusions — files whose import-time side effects we'd rather avoid
# =============================================================================


# Files known to do heavy work or import third-party libs at import time;
# Phase-27 skips them. Suffix-match on file path.
#
# Two classes of skips:
#
#  * **Import-heavy**: modules that import optional third-party deps
#    (``ingest_click.py``, ``ingest_json.py``) — would spuriously fail
#    on machines without those deps.
#
#  * **Fuzz-hostile**: modules whose top-level functions are dominated by
#    heavy numerical or cryptographic work that is correctly classified
#    as ``ready_typed`` but whose runtime cost under typed fuzzing is
#    minutes rather than sub-second. Probing them under the default
#    budget produces no calibration signal (they hit ``timeout``
#    uniformly) and dominates the benchmark wall time. They are
#    HONESTLY reported as out-of-scope for the default Phase-27
#    strategy and are candidates for OQ-27b / OQ-27c (custom recipes
#    with domain-appropriate inputs) in a follow-up phase.
DEFAULT_PHASE27_SKIP_FILES: tuple[str, ...] = (
    "__main__.py",
    # Import-heavy
    "ingest_click.py",
    "ingest_json.py",
    # Fuzz-hostile (crypto / heavy numerical)
    "paillier.py",             # primality tests → multi-second fuzz calls
    "polar.py",                # polar-code encoder uses recursive decoding
    "persuasion.py",            # concave-hull + bayesian persuasion loops
    "mfg.py",                  # mean-field-game PDE solvers
    "pac_bayes.py",            # PAC-Bayes bounds compute log(...) on 0
    "meta_learn.py",           # gradient descent loops — long typed-fuzz runtime
    "dp.py",                   # DP budget amplification — log() on 0 under fuzz
    "ctw_predictor.py",        # context-tree weighting — log(0) under fuzz
    "port_ham.py",              # port-Hamiltonian ODE stepping
    "persistent.py",            # hash-table that can pick pathological sizes
)

