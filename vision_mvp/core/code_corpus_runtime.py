"""Corpus-scale runtime calibration of the conservative analyzer — Phase 27 / 29.

Phase 26 measured analyzer-vs-runtime agreement on a 21-snippet curated
corpus: each snippet was hand-authored, ground-truthed per predicate,
and ships with an `invoke` driver that produces recipe-compatible
arguments. That proved the framework works; the next research gap is
**corpus scale**. Can the same probes be driven against real functions
drawn from the repository (rather than author-declared snippets)?

Phase 29 (this module's extension) adds ONE coverage-lifting recipe —
**safe zero-arg / all-defaults method-instance auto-construction** —
that materially widens the ready slice on method-heavy corpora
(`vision-core` 58 % methods, `vision-tests` 97 % methods) without
sacrificing conservatism. The construction is sandboxed with the same
subprocess / filesystem / network neutralisation used by the probes
themselves and is bounded by the per-call wall-clock tracer, so an
`__init__` that hangs or attempts side effects is handled the same
way an unsafe probe body is. See Theorem P29-1 in
``vision_mvp/RESULTS_PHASE29.md``.

The Phase-27 answer is: *partially, and the partiality is the
research finding*. Not every function admits a safe, terminating,
recipe-compatible invocation from an auto-derived input set:

  - A function with `*args, **kwargs` has no canonical recipe.
  - A function whose body needs an ``ast.Module`` instance cannot be
    fuzzed from integers and strings.
  - A method requires an instance of its enclosing class; the
    auto-constructor does not exist in the general case.
  - Async functions need an event loop; our sequential probe doesn't
    host one.
  - Generators return a generator object without executing their
    body; a `next()` driver is a separate recipe.
  - C-extension callables cannot be traced by ``sys.settrace``.

For the remainder — functions with zero args, or all-annotated args
whose types lie in a small fuzz-pool whitelist, or functions for which
a human-curated ``SafeRecipeRegistry`` entry exists — the Phase-26
probes work unchanged.

Phase 27 therefore measures **two axes at corpus scale**:

  1. **Runtime-calibration coverage** — of ``N`` functions declared
     in the corpus, how many are classifiable as *ready* for runtime
     probing under the current recipe derivation strategy. This is a
     witness-availability bound (Theorem P27-2), not an analyzer or
     planner bound.
  2. **Per-predicate analyzer/runtime agreement on the ready slice**
     — same FP/FN/agreement metrics as Phase 26, restricted to
     functions for which the runtime probe actually exercised the
     target frame (`sys.settrace` entry count ≥ 1 within the timeout
     budget).

Architecture
------------

The module is **additive** — it does not replace any Phase-24/25/26
primitive. It reuses:

  - ``probe_predicate`` from ``code_runtime_calibration`` for the
    per-predicate observation mechanics.
  - Analyzer static flags via ``analyze_interproc`` /
    ``build_module_context`` (corpus-wide so transitive flags match
    what the planner surfaces).
  - The same sentinel sandbox (subprocess / filesystem / network
    neutered) so effects never escape.

It adds four new pieces:

  - ``CorpusFunctionCandidate`` — AST-derived + signature-derived
    classification of one function's callability status.
  - ``InvocationRecipe`` — builds a list of positional-arg tuples for
    one candidate under one seed.
  - ``SafeRecipeRegistry`` — curated per-(module, qname) overrides
    for functions that need hand-crafted arguments (e.g. ones that
    take an AST node).
  - ``probe_corpus_function`` — runs the Phase-26 probe with
    ``sys.settrace``-based **entry detection** (so an observation is
    only counted when the target frame was actually reached) and
    a tracer-based **wall-clock budget** (so a runaway recipe can't
    stall the whole benchmark).

The probe outputs a ``CorpusObservation`` record that distinguishes
four states:

  - ``applicable=True, entered=True``  — real observation, feeds FP/FN
  - ``applicable=True, entered=False`` — recipe never reached target;
      excluded from calibration but counted in coverage.
  - ``applicable=False``               — recipe derivation returned
      nothing usable (e.g. typed-recipe saw an unsupported annotation
      at signature inspection time).
  - ``timeout=True``                    — target frame entered but the
      call did not return within the budget. Counted as an honest
      "untestable under current budget" bucket.

Safety stance
-------------

Real corpus functions may do unpredictable things — mutate globals,
spawn threads, hammer a package-level cache, etc. Phase 27 mitigates
but does not eliminate this:

  - The same sandbox as Phase 26 neuters subprocess / filesystem /
    network egress.
  - A per-call wall-time budget (default 0.25 s) capped by the trace
    tracer raises a sentinel on budget expiry.
  - Non-whitelisted annotations are refused during recipe derivation,
    so we never synthesise an argument of an unexpected type.
  - ``ready_curated`` entries are code-reviewed; ``ready_no_args`` and
    ``ready_typed`` are the only auto-derived slices, and both are
    bounded by the fuzz-pool and type whitelist.

This harness is designed for honest research corpora, not for
adversarial code. The trust boundary is identical to Phase 26.

See ``vision_mvp/RESULTS_PHASE27.md`` for the research framing, the
four-theorem set, the per-corpus calibration tables, and the
coverage/boundary attribution.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import os
import random
import sys
import textwrap
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .code_interproc import analyze_interproc, build_module_context
from .code_runtime_calibration import (
    RUNTIME_DECIDABLE_PREDICATES,
    RuntimeObservation,
    _NetworkAttempted,
    _ProbeSentinel,
    _SubprocessAttempted,
    _classify_exception_origin,
    _globals_diff,
    _observe_globals,
    _raise_line_numbers,
    _record_filesystem,
    _record_network,
    _record_subprocess,
    _track_reentry,
)


# =============================================================================
# Type whitelist + fuzz pool for auto-derivation
# =============================================================================


# The set of runtime type objects we are willing to synthesise values
# for. Deliberately small: auto-derivation is a *conservative* recipe
# strategy. Untyped or exotic-typed parameters fall through to
# ``unsupported_untyped`` and are recorded in coverage.
_SUPPORTED_RUNTIME_TYPES: tuple[type, ...] = (
    int, str, bool, float, list, dict, tuple, bytes, type(None),
)


# Per-type fuzz pools. Kept compact — the point is to get *any* valid
# value in the right shape, not to cover the type's state space.
_POOL_BY_TYPE: dict[type, tuple[Any, ...]] = {
    # Small pools chosen for fast termination: avoid very large integers
    # (which can drive numpy routines into multi-second computations) and
    # exotic floats (nan / inf) that some code paths handle via slow
    # fallbacks. The pools are intentionally conservative — aggressive
    # fuzzing is OQ-27a.
    int:         (0, 1, -1, 2, 3),
    str:         ("", "a", "hello"),
    bool:        (True, False),
    float:       (0.0, 1.0, -1.0),
    list:        ([], [0], [1, 2, 3]),
    dict:        ({}, {"k": "v"}),
    tuple:       ((), (1,), (1, 2)),
    bytes:       (b"", b"a"),
    type(None):  (None,),
}


# Recognised annotation-string spellings (covers ``from __future__ import
# annotations`` where ``inspect.signature`` returns strings). Any other
# annotation disqualifies the parameter from auto-derivation.
_STR_ANNOT_TO_TYPE: dict[str, type] = {
    "int": int, "str": str, "bool": bool, "float": float,
    "list": list, "dict": dict, "tuple": tuple, "bytes": bytes,
    "None": type(None), "NoneType": type(None),
}


def _classify_annotation(ann: Any) -> type | None:
    """Return a supported Python type for ``ann``, else None.

    Handles three annotation encodings:
      - Runtime type (``int``, ``str``, ...): checked against the
        whitelist tuple.
      - String (from ``from __future__ import annotations``): looked
        up in a small spelling map. Generic forms like ``list[int]``
        collapse to ``list`` because the probe only synthesises the
        outer container.
      - Everything else (e.g. forward refs, type-alias strings we
        don't recognise, ``Optional[...]``, user classes): returns
        None; the function is classified ``unsupported_untyped``.
    """
    if ann is inspect.Parameter.empty:
        return None
    if isinstance(ann, type) and ann in _SUPPORTED_RUNTIME_TYPES:
        return ann
    if isinstance(ann, str):
        # Strip a possible subscript so "list[int]" -> "list".
        base = ann.split("[", 1)[0].strip()
        return _STR_ANNOT_TO_TYPE.get(base)
    return None


# =============================================================================
# Candidate description + recipes
# =============================================================================


@dataclass(frozen=True)
class CorpusFunctionCandidate:
    """Static description of one function's callability status.

    Produced once per function during corpus discovery. Does NOT hold
    the function object itself (which is not pickleable in general) —
    callers pair the candidate with a resolved callable at probe
    time.

    ``callable_status`` values:
      * ``ready_no_args``     — 0 positional params; invoke with ().
      * ``ready_typed``       — all positional params annotated with a
                                supported type.
      * ``ready_curated``     — ``SafeRecipeRegistry`` supplies a
                                recipe for (module, qname).
      * ``unsupported_varargs``    — ``*args`` or ``**kwargs``.
      * ``unsupported_untyped``    — at least one positional param
                                      without a recipe-compatible
                                      annotation.
      * ``unsupported_async``      — async def.
      * ``unsupported_generator``  — body contains ``yield``.
      * ``unsupported_method``     — method on a class (no auto
                                      instance-construction).
      * ``unsupported_import``     — parent module failed to import.
      * ``unsupported_missing``    — qname not resolvable on the
                                      imported module object.
    """

    module_name: str
    module_path: str
    qname: str
    n_params: int
    param_annotations: tuple[str, ...]
    has_star_args: bool
    has_kwargs: bool
    is_generator: bool
    is_async: bool
    is_method: bool
    callable_status: str
    reason: str = ""

    @property
    def is_ready(self) -> bool:
        return self.callable_status.startswith("ready_")


@dataclass(frozen=True)
class InvocationRecipe:
    """Produces a deterministic sequence of positional-arg tuples."""

    kind: str  # "no_args" | "typed" | "curated"
    build: Callable[[random.Random, CorpusFunctionCandidate, Callable], list[tuple[Any, ...]]]


@dataclass
class SafeRecipeRegistry:
    """(module_name, qname) → curated InvocationRecipe.

    Curated entries are the escape hatch for functions whose arguments
    cannot be synthesised from the type whitelist — e.g. functions
    that take an ``ast.Module`` or a ``frozenset[str]``. Keep entries
    small, pure, and code-reviewed.
    """

    entries: dict[tuple[str, str], InvocationRecipe] = field(default_factory=dict)

    def register(self, module_name: str, qname: str,
                 recipe: InvocationRecipe) -> None:
        self.entries[(module_name, qname)] = recipe

    def lookup(self, module_name: str, qname: str) -> InvocationRecipe | None:
        return self.entries.get((module_name, qname))

    def __len__(self) -> int:
        return len(self.entries)


# =============================================================================
# Built-in recipe factories
# =============================================================================


def no_args_recipe(n_calls: int = 3) -> InvocationRecipe:
    """``n_calls`` copies of the empty tuple. For functions with 0
    positional parameters."""
    def _build(_rng: random.Random, _cand: CorpusFunctionCandidate,
               _func: Callable) -> list[tuple[Any, ...]]:
        return [() for _ in range(n_calls)]
    return InvocationRecipe(kind="no_args", build=_build)


def typed_recipe(n_calls: int = 2) -> InvocationRecipe:
    """Synthesise ``n_calls`` arg tuples by sampling from the per-type
    fuzz pool. Introspects ``inspect.signature(func)`` so that
    annotations populated at import time (including stringified
    annotations under ``from __future__ import annotations``) are
    honoured.

    If any parameter's annotation is not whitelisted, returns an empty
    list, which the caller maps to ``applicable=False``.
    """
    def _build(rng: random.Random, cand: CorpusFunctionCandidate,
               func: Callable) -> list[tuple[Any, ...]]:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return []
        types_in_order: list[type] = []
        for name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                return []
            if cand.is_method and name in ("self", "cls") and not types_in_order:
                # First parameter on an unbound method descriptor — skip
                # but only once and only when we were told the candidate
                # is a method.
                continue
            t = _classify_annotation(param.annotation)
            if t is None:
                return []
            types_in_order.append(t)
        out: list[tuple[Any, ...]] = []
        for _ in range(max(1, n_calls)):
            args: list[Any] = []
            for t in types_in_order:
                pool = _POOL_BY_TYPE.get(t, ())
                if not pool:
                    return []
                args.append(rng.choice(pool))
            out.append(tuple(args))
        return out
    return InvocationRecipe(kind="typed", build=_build)


def curated_recipe(arg_builder: Callable[[random.Random], list[tuple[Any, ...]]],
                   ) -> InvocationRecipe:
    """Wrap a human-authored argument factory so it participates in
    the same deterministic probe pipeline as auto-derived recipes.

    ``arg_builder(rng)`` returns a list of positional-arg tuples. The
    caller receives the seeded RNG — deterministic given the seed.
    """
    def _build(rng: random.Random, _cand: CorpusFunctionCandidate,
               _func: Callable) -> list[tuple[Any, ...]]:
        try:
            out = list(arg_builder(rng))
        except Exception:
            return []
        return [a if isinstance(a, tuple) else (a,) for a in out]
    return InvocationRecipe(kind="curated", build=_build)


# =============================================================================
# AST classification
# =============================================================================


def _function_is_generator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _is_dataclass_decorator(dec: ast.expr) -> bool:
    """Return True if ``dec`` looks like a ``@dataclass`` / ``@dataclass(...)``
    decorator by syntactic shape. Does not resolve aliases.
    """
    if isinstance(dec, ast.Name) and dec.id == "dataclass":
        return True
    if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
        return True
    if isinstance(dec, ast.Call):
        return _is_dataclass_decorator(dec.func)
    return False


def analyze_class_construction(class_node: ast.ClassDef
                                ) -> tuple[bool, str]:
    """Decide whether ``cls()`` is a safe, zero-arg construction strategy.

    Looks at:
      * Whether the class declares ``__init__`` in its AST body. If it
        does, every positional parameter after ``self`` must have a
        default and there must be no required kw-only parameter.
      * Whether the class is decorated with ``@dataclass`` (or
        ``@dataclasses.dataclass`` / ``@dataclass(frozen=True)`` etc).
        For dataclasses, every ``AnnAssign`` field at the class level
        must either have a value (e.g. ``x: int = 0``) or not exist as
        a required field. ``ClassVar[...]`` annotations and plain
        methods do not count.
      * Whether the class is a subclass of one of a small set of
        types known to require constructor arguments (``BaseException``,
        ``TypeError``, …). Those are classified as non-constructable
        to avoid raising during probe setup.

    Returns ``(is_constructable, strategy)``. ``strategy`` is a short
    human-readable tag used in the reason field and in coverage
    reports.

    This is a *static* decision. The probe additionally guards the
    actual ``cls()`` call with the sandbox + per-call wall-time budget
    tracer, so even if the AST heuristic says True the construction
    can still fail at probe time — in which case the observation is
    classified as ``applicable=False`` with ``notes="construct_failed:..."``.
    """
    # Fast rejection: explicit exception bases. Constructing these
    # always either raises or produces an object with its own surprising
    # state; not worth the yield.
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id in (
            "Exception", "BaseException", "TypeError", "ValueError",
            "RuntimeError", "OSError", "ArithmeticError",
            "LookupError", "KeyError", "IndexError", "AttributeError",
            "ImportError", "NameError", "NotImplementedError",
            "StopIteration", "StopAsyncIteration",
        ):
            return False, "exception_subclass"

    # Check decorators first so we can spot dataclass + ABCMeta cases.
    is_dataclass = any(_is_dataclass_decorator(d)
                         for d in class_node.decorator_list)

    # Find an explicit __init__ in the class body (does not traverse
    # nested classes — only direct children).
    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for body_node in class_node.body:
        if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if body_node.name == "__init__":
                init_node = body_node
                break

    if init_node is not None:
        if isinstance(init_node, ast.AsyncFunctionDef):
            return False, "async_init"
        args = init_node.args
        if args.vararg is not None or args.kwarg is not None:
            return False, "varargs_init"
        positional = list(args.args)
        if positional and positional[0].arg in ("self", "cls"):
            positional = positional[1:]
        # A positional arg at index i has a default iff
        # i >= len(positional) - len(args.defaults).
        defaults_count = len(args.defaults)
        missing_defaults = len(positional) - defaults_count
        if missing_defaults > 0:
            return False, "init_required_positional"
        # kw-only params: each entry in args.kwonlyargs has a parallel
        # entry in args.kw_defaults that is None iff there's no default.
        kw_missing = any(d is None for d in args.kw_defaults)
        if kw_missing:
            return False, "init_required_kwonly"
        return True, "explicit_init_all_defaults"

    if is_dataclass:
        # Every AnnAssign at the class level without a value is a
        # required field. ClassVar[...] annotations are exempt.
        for body_node in class_node.body:
            if isinstance(body_node, ast.AnnAssign) and body_node.value is None:
                ann = body_node.annotation
                if isinstance(ann, ast.Subscript):
                    base = ann.value
                    base_id = (base.id if isinstance(base, ast.Name)
                                 else (base.attr
                                         if isinstance(base, ast.Attribute)
                                         else ""))
                    if base_id == "ClassVar":
                        continue
                return False, "dataclass_required_field"
        return True, "dataclass_all_defaults"

    # Plain class, no __init__ declared in the AST body. This is the
    # dominant case (`class Foo: ...` bodies, helper classes, nested
    # subclasses inheriting from `object` or from a constructable
    # parent in the corpus we cannot statically resolve). Inherited
    # ``object.__init__`` takes no args. Subclasses of constructable
    # classes also work. We accept and let the probe-time construction
    # attempt catch the rare failure.
    return True, "inherited_object_init"


def classify_function_candidate(
    module_name: str, module_path: str, qname: str,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_method: bool,
    recipe_registry: SafeRecipeRegistry | None = None,
    class_node: ast.ClassDef | None = None,
) -> CorpusFunctionCandidate:
    """Assign a ``callable_status`` based purely on AST + curated
    registry. No module import, no code execution.

    ``class_node`` (Phase 29) — when ``is_method=True``, pass the
    enclosing ``ast.ClassDef`` so the classifier can promote methods
    to ``ready_method`` when the class is safely zero-arg-constructable.
    """
    registry = recipe_registry or SafeRecipeRegistry()
    is_async = isinstance(func_node, ast.AsyncFunctionDef)
    is_generator = _function_is_generator(func_node)
    args_obj = func_node.args
    has_star = args_obj.vararg is not None
    has_kwargs = args_obj.kwarg is not None

    positional = list(args_obj.args)
    if is_method and positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    # Keyword-only params count towards n_params for auto-typed recipes
    # only when they have defaults; otherwise the function can't be
    # invoked without supplying them. To keep things simple we classify
    # any function with kw-only params (no default) as unsupported.
    kwonly_without_default = any(
        p is not None and d is None
        for p, d in zip(args_obj.kwonlyargs, args_obj.kw_defaults)
    )

    param_ann_strs: list[str] = []
    for p in positional:
        ann = p.annotation
        if ann is None:
            param_ann_strs.append("")
        else:
            try:
                param_ann_strs.append(ast.unparse(ann))
            except Exception:
                param_ann_strs.append("<unparseable>")

    cand_meta = dict(
        module_name=module_name, module_path=module_path, qname=qname,
        n_params=len(positional),
        param_annotations=tuple(param_ann_strs),
        has_star_args=has_star, has_kwargs=has_kwargs,
        is_generator=is_generator, is_async=is_async,
        is_method=is_method,
    )

    if registry.lookup(module_name, qname) is not None:
        return CorpusFunctionCandidate(
            **cand_meta,
            callable_status="ready_curated",
            reason="curated SafeRecipeRegistry entry",
        )
    if is_async:
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="unsupported_async",
            reason="async def; probe harness is sequential",
        )
    if is_generator:
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="unsupported_generator",
            reason="generator body does not execute without a drain recipe",
        )
    if has_star or has_kwargs:
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="unsupported_varargs",
            reason="variadic (*args / **kwargs); ambiguous recipe",
        )
    if kwonly_without_default:
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="unsupported_varargs",
            reason="keyword-only parameters without defaults",
        )
    if is_method:
        # Phase 29 — try to promote methods on safely constructable
        # classes. The class must be AST-level "zero-arg-constructable"
        # (no custom __init__, or __init__ with only self+defaults, or
        # a dataclass with all fields defaulted), and the method's own
        # positional params (after self) must be either empty or fully
        # annotated with types in the Phase-27 whitelist. The probe
        # then wraps ``cls()`` in the Phase-26/27 sandbox and handles
        # any construction failure as ``applicable=False``.
        if class_node is not None:
            can_construct, strategy = analyze_class_construction(class_node)
            if can_construct:
                if len(positional) == 0:
                    return CorpusFunctionCandidate(
                        **cand_meta, callable_status="ready_method",
                        reason=(f"constructable class ({strategy}); "
                                f"zero-arg method"),
                    )
                if all(s for s in param_ann_strs):
                    return CorpusFunctionCandidate(
                        **cand_meta, callable_status="ready_method",
                        reason=(f"constructable class ({strategy}); "
                                f"typed method"),
                    )
                # Method on a constructable class but its positional
                # params are not typed — still unsupported for probing,
                # but tag the reason so coverage reports see the class
                # construction was not the blocker.
                return CorpusFunctionCandidate(
                    **cand_meta, callable_status="unsupported_untyped",
                    reason=(f"constructable class ({strategy}) but "
                            f"method positional params lack annotations"),
                )
            # Class is not constructable — fall through to the classic
            # ``unsupported_method`` classification below, but surface
            # the strategy rejection reason so the divergence report can
            # attribute coverage gaps to init-shape rather than fold
            # them into an opaque "method" bucket.
            return CorpusFunctionCandidate(
                **cand_meta, callable_status="unsupported_method",
                reason=(f"method requires an instance; class not "
                        f"zero-arg-constructable ({strategy})"),
            )
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="unsupported_method",
            reason="method requires an instance; no auto-construction",
        )
    if len(positional) == 0:
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="ready_no_args",
            reason="zero-arg callable",
        )
    # All positional params present — the AST-level check for annotations
    # is a fast rejection. Runtime annotations (via inspect.signature)
    # are a tighter check and will be applied in typed_recipe.build().
    # Here we accept if the AST shows non-empty annotations on every
    # positional parameter.
    if all(s for s in param_ann_strs):
        return CorpusFunctionCandidate(
            **cand_meta, callable_status="ready_typed",
            reason="all params annotated; typed recipe eligible",
        )
    return CorpusFunctionCandidate(
        **cand_meta, callable_status="unsupported_untyped",
        reason="positional params lack annotations",
    )


# =============================================================================
# Module import + qname resolution
# =============================================================================


@dataclass
class _ImportResult:
    module: types.ModuleType | None
    error: str = ""


def _import_module_from_path(module_name: str, module_path: str
                               ) -> _ImportResult:
    """Import a single .py file as ``module_name``.

    If the file is part of a regular package (has an ``__init__.py``
    walking up), we fall back to ``importlib.import_module`` which
    goes through the normal import machinery. Otherwise we use
    ``importlib.util.spec_from_file_location``.

    Import errors are captured and returned in ``.error`` (string) so
    the caller can tally them in coverage without crashing the
    benchmark.
    """
    try:
        # First preference: module_name is importable via normal path.
        if module_name and not module_name.startswith("_"):
            try:
                mod = importlib.import_module(module_name)
                return _ImportResult(module=mod)
            except ImportError:
                pass  # fall through to file-based import
        spec = importlib.util.spec_from_file_location(
            module_name or os.path.basename(module_path), module_path)
        if spec is None or spec.loader is None:
            return _ImportResult(module=None,
                                 error=f"no spec for {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return _ImportResult(module=mod)
    except BaseException as e:
        return _ImportResult(module=None,
                             error=f"{type(e).__name__}: {e}")


def _resolve_qname(module: types.ModuleType, qname: str) -> Callable | None:
    """Walk a dotted qname on the module. For a top-level function
    the qname is a single identifier; for a method it is
    ``ClassName.method_name``.
    """
    parts = qname.split(".")
    obj: Any = module
    for p in parts:
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj if callable(obj) else None


# =============================================================================
# Entry-detection + timeout tracer
# =============================================================================


class _BudgetExceeded(_ProbeSentinel):
    """Raised by the budget tracer when the per-call wall-time budget
    expires inside the target frame. Inherits ``_ProbeSentinel`` so
    the Phase-26 ``_call_safely`` wrapper swallows it without
    registering a ``may_raise`` observation."""


@contextmanager
def _entry_and_budget_tracer(target: Callable,
                               budget_s: float,
                               ) -> Iterator[dict[str, Any]]:
    """Install a single ``sys.settrace`` tracer that does two things:

      * Counts every ``call`` event whose frame code object is the
        target. The running count is exposed as ``state["enter_count"]``
        — callers check ``>= 1`` to decide whether the target was ever
        entered.
      * Checks ``time.monotonic()`` on every ``line`` and ``call`` event.
        If the deadline has passed, raises ``_BudgetExceeded`` *inside*
        the traced frame, which the probe's ``_call_safely`` will catch.

    Restores any previous tracer on exit; composes cleanly with the
    Phase-26 ``_track_reentry`` when a caller wants cycle detection on
    the same invocation (they would layer both via the
    ``probe_participates_in_cycle`` path, not this tracer).

    Note: Python's trace events are Python-level. A function executing
    entirely inside a C extension will not be interruptible by this
    tracer. That is honest; we report such functions as either
    ``unsupported_method`` (if they're attribute-bound) or ``timeout``
    (if they hang). We do NOT use ``signal.setitimer`` because it is
    delivered to the process and interacts poorly with test harness
    signal handlers.
    """
    state: dict[str, Any] = {
        "enter_count": 0, "deadline": time.monotonic() + max(0.001, budget_s),
        "exceeded": False,
    }
    code_obj = getattr(target, "__code__", None)
    prev = sys.gettrace()

    def tracer(frame, event, arg):
        # Budget check first — fires on every event.
        if time.monotonic() > state["deadline"]:
            state["exceeded"] = True
            # Only raise from a line event; raising on call/return can
            # trigger interpreter-level bailouts.
            if event == "line":
                raise _BudgetExceeded(
                    f"probe budget {budget_s}s exceeded for "
                    f"{getattr(code_obj, 'co_name', '?')}")
        if event == "call" and code_obj is not None \
                and frame.f_code is code_obj:
            state["enter_count"] += 1
        return tracer

    sys.settrace(tracer)
    try:
        yield state
    finally:
        sys.settrace(prev)


# =============================================================================
# Corpus observation record
# =============================================================================


@dataclass(frozen=True)
class CorpusObservation:
    """Per-(function, predicate) runtime observation at corpus scale.

    Extends Phase-26's ``RuntimeObservation`` with two fields that
    only make sense at corpus scale:

      - ``entered`` — at least one invocation reached the target frame
        (settrace ``call`` event fired on the target's code object).
      - ``timeout`` — at least one invocation was aborted by the
        per-call budget tracer.

    An observation with ``applicable=False`` (recipe returned nothing
    usable) or ``entered=False`` does NOT contribute to the
    calibration matrix — it lives in the coverage accounting instead.
    """

    qname: str
    predicate: str
    runtime_flag: bool
    n_runs: int
    n_triggered: int
    n_entered: int
    n_timeout: int
    witnesses: tuple[str, ...] = ()
    decidable: bool = True
    applicable: bool = True
    entered: bool = False
    timeout: bool = False
    recipe_kind: str = ""
    notes: str = ""

    @property
    def trigger_rate(self) -> float:
        if self.n_runs <= 0:
            return 0.0
        return self.n_triggered / self.n_runs


# =============================================================================
# The corpus probe
# =============================================================================


def _probe_body(predicate: str, func: Callable, module: types.ModuleType,
                args: tuple[Any, ...], budget_s: float,
                ) -> tuple[bool, str | None, bool, bool]:
    """Run a single-invocation observation for ``predicate`` on ``func``
    with ``args``. Returns ``(triggered, witness, entered, timeout)``.

    This is the atomic unit of corpus-scale observation. It layers the
    four sandbox recorders (subprocess / filesystem / network / globals
    diff) around the entry-and-budget tracer so that an observation
    both (a) confirms the target frame was actually reached and (b)
    cannot exceed the wall-clock budget even if the body loops.
    """
    with _entry_and_budget_tracer(func, budget_s) as ebt:
        triggered = False
        witness: str | None = None
        if predicate == "may_raise":
            # may_raise wraps all three sandbox recorders to neuter
            # side-effects but records NO witness from them.
            with _record_subprocess(), _record_filesystem(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException as e:
                    triggered = True
                    witness = type(e).__name__
        elif predicate in ("may_raise_explicit", "may_raise_implicit"):
            # Phase 28 — classify exception origin using the
            # traceback-line analysis. Only the bucket matching this
            # predicate's label contributes to `triggered`.
            want_explicit = predicate == "may_raise_explicit"
            raise_lines = _raise_line_numbers(func)
            with _record_subprocess(), _record_filesystem(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException as e:
                    origin = _classify_exception_origin(
                        func, e, raise_lines=raise_lines)
                    if (origin == "explicit") == want_explicit:
                        triggered = True
                        witness = f"{origin}:{type(e).__name__}"
        elif predicate == "may_write_global":
            with _observe_globals(module) as before, \
                    _record_subprocess(), _record_filesystem(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException:
                    pass
            diff = _globals_diff(module, before)
            if diff:
                triggered = True
                witness = ",".join(diff[:3])
        elif predicate == "calls_subprocess":
            with _record_subprocess() as sub_hits, \
                    _record_filesystem(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException:
                    pass
            if sub_hits:
                triggered = True
                witness = sub_hits[0]
        elif predicate == "calls_filesystem":
            with _record_filesystem() as fs_hits, \
                    _record_subprocess(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException:
                    pass
            if fs_hits:
                triggered = True
                witness = fs_hits[0]
        elif predicate == "calls_network":
            with _record_network() as net_hits, \
                    _record_subprocess(), _record_filesystem():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException:
                    pass
            if net_hits:
                triggered = True
                witness = net_hits[0]
        elif predicate == "participates_in_cycle":
            # Cannot nest ``_track_reentry`` inside ``_entry_and_budget_tracer``
            # because ``sys.settrace`` only supports one tracer at a time —
            # the inner installer would silence the outer budget + entry
            # counter. Instead, we read the outer tracer's ``enter_count``
            # AFTER the call: enter_count >= 2 ⇒ target re-entered ⇒ cycle
            # observed. This is semantically equivalent to the Phase-26
            # ``_track_reentry`` probe, but compatible with the budget tracer.
            with _record_subprocess(), _record_filesystem(), _record_network():
                try:
                    func(*args)
                except _ProbeSentinel:
                    pass
                except BaseException:
                    pass
            # Note: enter_count is evaluated after the exit of
            # _entry_and_budget_tracer below, but the state dict is mutable
            # and we are still INSIDE the with-block — cf ``ebt["enter_count"]``
            # below. Only declare ``triggered`` when the counter jumped above 1.
        else:
            return False, None, False, False
    # Post-exit reads: ebt is a plain dict captured by the tracer; the
    # tracer has fired every call/line event up to this point.
    entered = ebt["enter_count"] >= 1
    timeout = bool(ebt["exceeded"])
    if predicate == "participates_in_cycle" and ebt["enter_count"] >= 2:
        triggered = True
        witness = f"reentries={ebt['enter_count']}"
    return triggered, witness, entered, timeout


def _try_construct_instance(cls: type, budget_s: float,
                              ) -> tuple[Any | None, str]:
    """Attempt ``cls()`` under the Phase-26 sandbox + Phase-27 budget
    tracer. Phase 29.

    Returns ``(instance, error_tag)`` where ``instance`` is the
    constructed object on success or ``None`` on failure, and
    ``error_tag`` is a short string describing the failure class:

      * ``""``                      — success
      * ``"construct_exc:<Name>"``  — ``__init__`` raised
      * ``"construct_budget"``      — per-call wall-clock exceeded
      * ``"construct_not_callable"``— ``cls`` not callable (e.g. class
                                       body defined ``__init__`` as
                                       something that is not actually
                                       a function at runtime)

    Safety stance is identical to the probe body:

      * subprocess / filesystem / network sentinels are active, so a
        constructor that opens a file or spawns a process is neutered.
      * The budget tracer raises ``_BudgetExceeded`` (a sentinel) on
        the first line event past the deadline, so a constructor with
        a tight loop is terminable.
      * Any other exception — including ``_ProbeSentinel`` variants
        raised from neutered APIs — is swallowed and reported as a
        construction failure. The probe then classifies the whole
        observation as ``applicable=False`` so it contributes to
        coverage but not to FP/FN.
    """
    if not callable(cls):
        return None, "construct_not_callable"
    deadline = time.monotonic() + max(0.001, budget_s)
    prev = sys.gettrace()
    state = {"exceeded": False}

    def _budget_tracer(frame, event, arg):
        if time.monotonic() > deadline:
            state["exceeded"] = True
            if event == "line":
                raise _BudgetExceeded(
                    f"construct budget {budget_s}s exceeded for "
                    f"{getattr(cls, '__name__', '?')}")
        return _budget_tracer

    try:
        with _record_subprocess(), _record_filesystem(), _record_network():
            sys.settrace(_budget_tracer)
            try:
                instance = cls()
            finally:
                sys.settrace(prev)
    except _BudgetExceeded:
        return None, "construct_budget"
    except _ProbeSentinel as e:
        return None, f"construct_sandbox:{type(e).__name__}"
    except BaseException as e:
        return None, f"construct_exc:{type(e).__name__}"
    return instance, ""


def probe_corpus_function(
    candidate: CorpusFunctionCandidate,
    func: Callable,
    module: types.ModuleType,
    *,
    predicate: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    budget_s: float = 0.25,
    recipe_registry: SafeRecipeRegistry | None = None,
) -> CorpusObservation:
    """Run ``predicate`` probes against ``func`` using the recipe
    implied by ``candidate.callable_status``, aggregating over seeds.

    Returns a single ``CorpusObservation``. On a non-ready candidate
    the observation has ``applicable=False``.
    """
    if not candidate.is_ready:
        return CorpusObservation(
            qname=candidate.qname, predicate=predicate,
            runtime_flag=False, n_runs=0, n_triggered=0,
            n_entered=0, n_timeout=0, applicable=False,
            decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
            notes=f"status={candidate.callable_status}",
        )
    registry = recipe_registry or SafeRecipeRegistry()
    # Phase 29 — for ready_method, we construct the enclosing class
    # instance ONCE at the top of the probe and bind the method so the
    # rest of the call site is exactly the same as for free functions.
    # The bound method's signature naturally excludes ``self``, so the
    # typed_recipe / no_args_recipe flow works unchanged.
    probe_func: Callable = func
    if candidate.callable_status == "ready_curated":
        recipe = registry.lookup(candidate.module_name, candidate.qname)
        if recipe is None:
            return CorpusObservation(
                qname=candidate.qname, predicate=predicate,
                runtime_flag=False, n_runs=0, n_triggered=0,
                n_entered=0, n_timeout=0, applicable=False,
                decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
                notes="curated recipe missing at probe time",
                recipe_kind="curated",
            )
    elif candidate.callable_status == "ready_no_args":
        recipe = no_args_recipe()
    elif candidate.callable_status == "ready_typed":
        recipe = typed_recipe()
    elif candidate.callable_status == "ready_method":
        parts = candidate.qname.rsplit(".", 1)
        if len(parts) != 2:
            return CorpusObservation(
                qname=candidate.qname, predicate=predicate,
                runtime_flag=False, n_runs=0, n_triggered=0,
                n_entered=0, n_timeout=0, applicable=False,
                decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
                notes="ready_method qname not class.method",
                recipe_kind="method",
            )
        cls_name, method_name = parts[0], parts[1]
        cls = getattr(module, cls_name, None)
        if not isinstance(cls, type):
            return CorpusObservation(
                qname=candidate.qname, predicate=predicate,
                runtime_flag=False, n_runs=0, n_triggered=0,
                n_entered=0, n_timeout=0, applicable=False,
                decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
                notes=f"class {cls_name} not found on module",
                recipe_kind="method",
            )
        instance, err_tag = _try_construct_instance(cls, budget_s=budget_s)
        if instance is None:
            return CorpusObservation(
                qname=candidate.qname, predicate=predicate,
                runtime_flag=False, n_runs=0, n_triggered=0,
                n_entered=0, n_timeout=0, applicable=False,
                decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
                notes=err_tag,
                recipe_kind="method",
            )
        bound = getattr(instance, method_name, None)
        if not callable(bound):
            return CorpusObservation(
                qname=candidate.qname, predicate=predicate,
                runtime_flag=False, n_runs=0, n_triggered=0,
                n_entered=0, n_timeout=0, applicable=False,
                decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
                notes=f"method {method_name} not bound on instance",
                recipe_kind="method",
            )
        probe_func = bound
        # Reuse the existing no_args / typed recipes on the bound
        # method — ``inspect.signature(bound)`` naturally skips self,
        # so the parameter list matches ``candidate.n_params``. The
        # recipe kind is tagged with a ``method:`` prefix so corpus
        # reports can separate method-bucket probes from free-function
        # probes.
        if candidate.n_params == 0:
            _inner = no_args_recipe()
        else:
            _inner = typed_recipe()
        recipe = InvocationRecipe(kind="method", build=_inner.build)
    else:
        return CorpusObservation(
            qname=candidate.qname, predicate=predicate,
            runtime_flag=False, n_runs=0, n_triggered=0,
            n_entered=0, n_timeout=0, applicable=False,
            decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
            notes=f"unhandled ready status: {candidate.callable_status}",
        )

    n_runs = 0
    n_triggered = 0
    n_entered = 0
    n_timeout = 0
    witnesses: list[str] = []
    any_applicable = False
    # Per-candidate wall-clock ceiling: even with per-call budgets the
    # total probe cost for a pathological function (e.g. pure-C numpy
    # code the tracer cannot interrupt) can accumulate. Abort the seed
    # loop early once the candidate has consumed ``max(5*budget_s,
    # 1.0)`` seconds. The candidate's remaining predicates are still
    # probed, but this one returns what it has — applicable/entered
    # flags are honest about partial data.
    candidate_deadline = time.monotonic() + max(5 * budget_s, 1.0)
    for seed in seeds:
        if time.monotonic() > candidate_deadline:
            break
        rng = random.Random(seed)
        arg_list = recipe.build(rng, candidate, probe_func)
        if not arg_list:
            continue
        any_applicable = True
        for args in arg_list:
            if time.monotonic() > candidate_deadline:
                n_timeout += 1
                break
            n_runs += 1
            try:
                triggered, witness, entered, timeout = _probe_body(
                    predicate, probe_func, module, args, budget_s)
            except BaseException as e:
                # Probe itself failed structurally (e.g. settrace clash
                # outside the tracer). Count the run but record no witness.
                triggered = False
                witness = None
                entered = False
                timeout = False
                witnesses.append(f"probe_error:{type(e).__name__}")
            if triggered:
                n_triggered += 1
                if witness is not None and len(witnesses) < 5:
                    witnesses.append(witness)
            if entered:
                n_entered += 1
            if timeout:
                n_timeout += 1
    if not any_applicable:
        return CorpusObservation(
            qname=candidate.qname, predicate=predicate,
            runtime_flag=False, n_runs=0, n_triggered=0,
            n_entered=0, n_timeout=0, applicable=False,
            decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
            notes="recipe produced no invocations",
            recipe_kind=recipe.kind,
        )
    return CorpusObservation(
        qname=candidate.qname, predicate=predicate,
        runtime_flag=n_triggered > 0 and n_entered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        n_entered=n_entered, n_timeout=n_timeout,
        witnesses=tuple(witnesses), decidable=True, applicable=True,
        entered=n_entered > 0, timeout=n_timeout > 0,
        recipe_kind=recipe.kind,
    )


# =============================================================================
# Corpus discovery
# =============================================================================


@dataclass(frozen=True)
class DiscoveredCandidate:
    """Pair of (static candidate, resolved callable, enclosing module).

    The callable and module references are Python objects and cannot
    outlive the benchmark process. The candidate itself is frozen and
    serialisable.
    """

    candidate: CorpusFunctionCandidate
    func: Callable | None
    module: types.ModuleType | None


def _walk_python_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in ("__pycache__", ".git", ".venv", ".mypy_cache")
        )
        for f in sorted(filenames):
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f))
    return out


def _module_name_from_path(file_path: str, root: str) -> str:
    rel = os.path.relpath(file_path, root)
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith(os.sep + "__init__"):
        rel = rel[: -len(os.sep + "__init__")]
    return rel.replace(os.sep, ".")


def _full_module_name(file_path: str, corpus_root: str,
                       corpus_package: str | None) -> str:
    """Compute the module's *importable* name.

    If ``corpus_package`` is provided (e.g. ``"vision_mvp.core"``),
    every file gets that as a prefix so ``importlib.import_module``
    finds it via the installed package rather than the raw file. If
    None, we fall back to the relative dotted name.
    """
    rel = _module_name_from_path(file_path, corpus_root)
    if corpus_package:
        if rel:
            return f"{corpus_package}.{rel}"
        return corpus_package
    return rel


def discover_candidates(
    corpus_root: str,
    *,
    corpus_package: str | None = None,
    recipe_registry: SafeRecipeRegistry | None = None,
    max_files: int | None = None,
    skip_files: tuple[str, ...] = (),
) -> list[DiscoveredCandidate]:
    """Walk ``corpus_root``, parse every .py file, import the module,
    and emit a ``DiscoveredCandidate`` for every function.

    Import failures are recorded — the per-function candidates for
    that file land as ``unsupported_import`` with the error in
    ``reason``.

    ``skip_files`` is a tuple of *suffixes* (e.g. ``("__main__.py",)``)
    that cause the file to be skipped entirely — useful for files
    that are known to run benchmarks at import time.
    """
    registry = recipe_registry or SafeRecipeRegistry()
    files = _walk_python_files(corpus_root)
    if max_files is not None:
        files = files[:max_files]
    out: list[DiscoveredCandidate] = []
    for fp in files:
        if any(fp.endswith(s) for s in skip_files):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=fp)
        except SyntaxError:
            continue
        module_name = _full_module_name(fp, corpus_root, corpus_package)
        import_result = _import_module_from_path(module_name, fp)
        module = import_result.module

        # Enumerate functions and methods lexically — same order the
        # analyzer uses so qnames line up.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = node.name
                cand = classify_function_candidate(
                    module_name, fp, qname, node,
                    is_method=False, recipe_registry=registry)
                if module is None:
                    cand = dataclasses.replace(
                        cand, callable_status="unsupported_import",
                        reason=f"import failed: {import_result.error}")
                    out.append(DiscoveredCandidate(cand, None, None))
                    continue
                func = _resolve_qname(module, qname)
                if func is None:
                    cand = dataclasses.replace(
                        cand, callable_status="unsupported_missing",
                        reason="qname not on imported module")
                    out.append(DiscoveredCandidate(cand, None, module))
                    continue
                out.append(DiscoveredCandidate(cand, func, module))
            elif isinstance(node, ast.ClassDef):
                for sub in ast.iter_child_nodes(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qname = f"{node.name}.{sub.name}"
                        cand = classify_function_candidate(
                            module_name, fp, qname, sub,
                            is_method=True, recipe_registry=registry,
                            class_node=node)
                        if module is None:
                            cand = dataclasses.replace(
                                cand, callable_status="unsupported_import",
                                reason=f"import failed: {import_result.error}")
                            out.append(DiscoveredCandidate(cand, None, None))
                            continue
                        func = _resolve_qname(module, qname)
                        if func is None:
                            cand = dataclasses.replace(
                                cand, callable_status="unsupported_missing",
                                reason="qname not on imported module")
                            out.append(DiscoveredCandidate(
                                cand, None, module))
                            continue
                        out.append(DiscoveredCandidate(cand, func, module))
    return out


# =============================================================================
# Static flags for corpus functions (Phase 24 + 25)
# =============================================================================


def build_corpus_static_flags(
    corpus_root: str,
    *,
    corpus_package: str | None = None,
    max_files: int | None = None,
) -> dict[str, dict[str, bool]]:
    """Run Phase-24/25 analysis over the corpus and return a map from
    module-qualified qname → {predicate → flag}.

    The qname keys match ``{full_module_name}.{short_qname}`` where
    ``short_qname`` is either a bare function name or ``ClassName.method``.
    The analyzer stores the same shape via ``build_module_context``; we
    simply bridge the keys so they equal the ``DiscoveredCandidate.qname``
    once prefixed with ``full_module_name``.
    """
    files = _walk_python_files(corpus_root)
    if max_files is not None:
        files = files[:max_files]
    modules: list = []
    intra_all: dict = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=fp)
        except SyntaxError:
            continue
        module_name = _full_module_name(fp, corpus_root, corpus_package)
        ctx, intra = build_module_context(module_name, tree)
        modules.append(ctx)
        intra_all.update(intra)
    if not modules:
        return {}
    interproc_map, _cg = analyze_interproc(modules, intra_all)
    out: dict[str, dict[str, bool]] = {}
    for full_qname, sem in interproc_map.items():
        out[full_qname] = {
            "may_raise": sem.trans_may_raise,
            "may_write_global": sem.trans_may_write_global,
            "calls_subprocess": sem.trans_calls_subprocess,
            "calls_filesystem": sem.trans_calls_filesystem,
            "calls_network": sem.trans_calls_network,
            "participates_in_cycle": sem.participates_in_cycle,
            # Phase 28 — `may_raise_explicit` mirrors `may_raise` (the
            # Phase-24 contract); `may_raise_implicit` is the new
            # separate analyzer axis propagated by Phase 25/28 over the
            # resolved call graph.
            "may_raise_explicit": sem.trans_may_raise,
            "may_raise_implicit": getattr(
                sem, "trans_may_raise_implicit", False),
        }
    return out


# =============================================================================
# Aggregation + coverage accounting
# =============================================================================


@dataclass(frozen=True)
class CorpusCalibrationRow:
    """One row in the corpus calibration table — per (candidate,
    predicate)."""

    candidate: CorpusFunctionCandidate
    static_flags: dict[str, bool]
    observations: dict[str, CorpusObservation]


@dataclass
class CoverageAccount:
    """Per-corpus coverage breakdown — the Phase-27 research headline.

    The ``status_*`` fields sum to ``n_total`` and tell the reader why
    a function was or wasn't probed. The calibration-participating
    fraction is ``n_calibrated / n_total``.
    """

    corpus_name: str = ""
    n_total: int = 0
    status_ready_no_args: int = 0
    status_ready_typed: int = 0
    status_ready_curated: int = 0
    status_ready_method: int = 0                   # Phase 29
    status_unsupported_varargs: int = 0
    status_unsupported_untyped: int = 0
    status_unsupported_async: int = 0
    status_unsupported_generator: int = 0
    status_unsupported_method: int = 0
    status_unsupported_import: int = 0
    status_unsupported_missing: int = 0
    n_probed: int = 0                 # ready status & at least one predicate produced an observation
    n_entered: int = 0                # at least one probe entered the target frame
    n_timeout: int = 0                # at least one probe hit the budget
    n_calibrated: int = 0             # contributed to per-predicate FP/FN (entered + applicable)
    n_construct_failed: int = 0       # Phase 29 — instance construction attempted and failed

    def as_dict(self) -> dict:
        return {
            "corpus_name": self.corpus_name,
            "n_total": self.n_total,
            "ready_no_args": self.status_ready_no_args,
            "ready_typed": self.status_ready_typed,
            "ready_curated": self.status_ready_curated,
            "ready_method": self.status_ready_method,
            "unsupported_varargs": self.status_unsupported_varargs,
            "unsupported_untyped": self.status_unsupported_untyped,
            "unsupported_async": self.status_unsupported_async,
            "unsupported_generator": self.status_unsupported_generator,
            "unsupported_method": self.status_unsupported_method,
            "unsupported_import": self.status_unsupported_import,
            "unsupported_missing": self.status_unsupported_missing,
            "n_probed": self.n_probed,
            "n_entered": self.n_entered,
            "n_timeout": self.n_timeout,
            "n_calibrated": self.n_calibrated,
            "n_construct_failed": self.n_construct_failed,
            "ready_fraction": round(
                (self.status_ready_no_args + self.status_ready_typed
                 + self.status_ready_curated
                 + self.status_ready_method) / max(1, self.n_total), 4),
            "calibrated_fraction": round(
                self.n_calibrated / max(1, self.n_total), 4),
        }


def _tally_status(cov: CoverageAccount, status: str) -> None:
    field_name = f"status_{status}"
    if hasattr(cov, field_name):
        setattr(cov, field_name, getattr(cov, field_name) + 1)


def calibrate_corpus(
    corpus_name: str,
    corpus_root: str,
    *,
    corpus_package: str | None = None,
    recipe_registry: SafeRecipeRegistry | None = None,
    predicates: tuple[str, ...] = tuple(sorted(RUNTIME_DECIDABLE_PREDICATES)),
    seeds: tuple[int, ...] = (0, 1, 2),
    budget_s: float = 0.25,
    max_files: int | None = None,
    skip_files: tuple[str, ...] = (),
    progress: Callable[[str], None] | None = None,
) -> tuple[list[CorpusCalibrationRow], CoverageAccount]:
    """End-to-end Phase-27 pipeline for one corpus.

    Steps:
      1. Static pass: build Phase-24/25 analyzer flags for every
         function in the corpus.
      2. Discovery: walk the corpus, classify every function, resolve
         callables.
      3. Probe: for each *ready* candidate × predicate, run the
         Phase-26 probe with entry+budget tracer, aggregating over
         seeds.
      4. Aggregate: return per-function rows + coverage summary.

    Never raises on a single-function failure; structural probe errors
    are captured in ``CorpusObservation.witnesses`` and tallied in
    coverage.
    """
    registry = recipe_registry or SafeRecipeRegistry()
    if progress:
        progress(f"[{corpus_name}] running Phase-24/25 static analysis...")
    static_by_qname = build_corpus_static_flags(
        corpus_root, corpus_package=corpus_package, max_files=max_files)

    if progress:
        progress(f"[{corpus_name}] discovering candidates "
                 f"(max_files={max_files})...")
    discovered = discover_candidates(
        corpus_root, corpus_package=corpus_package,
        recipe_registry=registry, max_files=max_files,
        skip_files=skip_files,
    )

    cov = CoverageAccount(corpus_name=corpus_name,
                          n_total=len(discovered))
    rows: list[CorpusCalibrationRow] = []

    for i, disc in enumerate(discovered):
        cand = disc.candidate
        _tally_status(cov, cand.callable_status)
        full_qname = f"{cand.module_name}.{cand.qname}"
        static = static_by_qname.get(full_qname, {})
        obs_map: dict[str, CorpusObservation] = {}
        if cand.is_ready and disc.func is not None and disc.module is not None:
            any_probed = False
            any_entered = False
            any_timeout = False
            any_construct_fail = False
            for pred in predicates:
                obs = probe_corpus_function(
                    cand, disc.func, disc.module,
                    predicate=pred, seeds=seeds, budget_s=budget_s,
                    recipe_registry=registry)
                obs_map[pred] = obs
                if obs.applicable:
                    any_probed = True
                if obs.entered:
                    any_entered = True
                if obs.timeout:
                    any_timeout = True
                # Phase 29 — surface failed method construction as a
                # first-class coverage bucket so the writeup can
                # separate "method was probed and entered" from
                # "method's class could not be constructed".
                if (cand.callable_status == "ready_method"
                        and not obs.applicable
                        and obs.notes.startswith("construct_")):
                    any_construct_fail = True
            if any_probed:
                cov.n_probed += 1
            if any_entered:
                cov.n_entered += 1
                cov.n_calibrated += 1
            if any_timeout:
                cov.n_timeout += 1
            if any_construct_fail:
                cov.n_construct_failed += 1
        else:
            for pred in predicates:
                obs_map[pred] = CorpusObservation(
                    qname=cand.qname, predicate=pred,
                    runtime_flag=False, n_runs=0, n_triggered=0,
                    n_entered=0, n_timeout=0, applicable=False,
                    decidable=pred in RUNTIME_DECIDABLE_PREDICATES,
                    notes=f"status={cand.callable_status}",
                )
        rows.append(CorpusCalibrationRow(
            candidate=cand, static_flags=static, observations=obs_map))
        if progress and (i + 1) % max(1, len(discovered) // 10) == 0:
            progress(f"[{corpus_name}] {i+1}/{len(discovered)} "
                     f"(probed={cov.n_probed}, entered={cov.n_entered})")
    return rows, cov


# =============================================================================
# Per-predicate summarisation
# =============================================================================


@dataclass(frozen=True)
class CorpusPredicateMetrics:
    predicate: str
    n_applicable: int
    n_entered: int
    n_static_true: int
    n_runtime_true: int
    n_agree: int
    n_false_positives: int
    n_false_negatives: int
    fp_rate: float | None
    fn_rate: float | None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def summarise_corpus_calibration(
    rows: list[CorpusCalibrationRow],
    *, predicates: Iterable[str] | None = None,
) -> dict[str, CorpusPredicateMetrics]:
    """Produce per-predicate agreement metrics, restricted to rows
    where the probe entered the target frame.

    A row with ``applicable=True`` but ``entered=False`` contributes
    only to coverage accounting — the entry-failure bucket — not to
    FP/FN, because an observation that never reached the target body
    is by definition silent on the target's semantic behaviour.
    """
    preds = list(predicates) if predicates is not None \
        else sorted(RUNTIME_DECIDABLE_PREDICATES)
    out: dict[str, CorpusPredicateMetrics] = {}
    for pred in preds:
        n_applicable = 0
        n_entered = 0
        n_static_true = 0
        n_runtime_true = 0
        n_agree = 0
        n_fp = 0
        n_fn = 0
        for row in rows:
            obs = row.observations.get(pred)
            if obs is None or not obs.decidable:
                continue
            if not obs.applicable:
                continue
            n_applicable += 1
            if not obs.entered:
                continue
            n_entered += 1
            sf = bool(row.static_flags.get(pred, False))
            rf = bool(obs.runtime_flag)
            if sf: n_static_true += 1
            if rf: n_runtime_true += 1
            if sf == rf:
                n_agree += 1
            elif sf and not rf:
                n_fp += 1
            else:
                n_fn += 1
        fp_rate = (n_fp / n_static_true) if n_static_true else None
        fn_rate = (n_fn / n_runtime_true) if n_runtime_true else None
        out[pred] = CorpusPredicateMetrics(
            predicate=pred, n_applicable=n_applicable,
            n_entered=n_entered,
            n_static_true=n_static_true, n_runtime_true=n_runtime_true,
            n_agree=n_agree, n_false_positives=n_fp, n_false_negatives=n_fn,
            fp_rate=round(fp_rate, 4) if fp_rate is not None else None,
            fn_rate=round(fn_rate, 4) if fn_rate is not None else None,
        )
    return out


def rows_as_dict_list(rows: list[CorpusCalibrationRow]) -> list[dict]:
    """Serialise rows for JSON output — drops module-object references."""
    out: list[dict] = []
    for r in rows:
        c = r.candidate
        out.append({
            "candidate": {
                "module_name": c.module_name, "qname": c.qname,
                "n_params": c.n_params,
                "callable_status": c.callable_status,
                "reason": c.reason,
                "param_annotations": list(c.param_annotations),
                "is_method": c.is_method, "is_async": c.is_async,
                "is_generator": c.is_generator,
            },
            "static_flags": dict(r.static_flags),
            "observations": {
                p: {
                    "runtime_flag": o.runtime_flag,
                    "n_runs": o.n_runs, "n_triggered": o.n_triggered,
                    "n_entered": o.n_entered, "n_timeout": o.n_timeout,
                    "trigger_rate": round(o.trigger_rate, 4),
                    "witnesses": list(o.witnesses),
                    "applicable": o.applicable, "entered": o.entered,
                    "timeout": o.timeout, "recipe_kind": o.recipe_kind,
                    "notes": o.notes,
                }
                for p, o in r.observations.items()
            },
        })
    return out


# =============================================================================
# Divergence attribution
# =============================================================================


@dataclass(frozen=True)
class CorpusDivergence:
    """One analyzer-vs-runtime disagreement on a corpus function."""

    qname: str
    predicate: str
    kind: str  # "false_positive" | "false_negative"
    static_flag: bool
    runtime_flag: bool
    witnesses: tuple[str, ...]
    callable_status: str
    module_name: str


def collect_divergences(rows: list[CorpusCalibrationRow],
                          predicates: Iterable[str] | None = None,
                          ) -> list[CorpusDivergence]:
    """Surface every row where the analyzer flag and runtime flag
    disagree on a predicate that was actually observed (entered + applicable).

    The result is what gets attributed by the Phase-27 writeup: each
    entry is either (a) a documented boundary case (eval, reflection,
    dead code), (b) a new boundary class the snippet corpus did not
    exercise, or (c) a recipe artifact that the harness did not
    otherwise reject. Case (b) is the highest-value research signal.
    """
    preds = list(predicates) if predicates is not None \
        else sorted(RUNTIME_DECIDABLE_PREDICATES)
    out: list[CorpusDivergence] = []
    for row in rows:
        for p in preds:
            obs = row.observations.get(p)
            if obs is None or not obs.applicable or not obs.entered:
                continue
            sf = bool(row.static_flags.get(p, False))
            rf = bool(obs.runtime_flag)
            if sf == rf:
                continue
            kind = "false_positive" if sf else "false_negative"
            out.append(CorpusDivergence(
                qname=f"{row.candidate.module_name}.{row.candidate.qname}",
                predicate=p, kind=kind, static_flag=sf, runtime_flag=rf,
                witnesses=obs.witnesses,
                callable_status=row.candidate.callable_status,
                module_name=row.candidate.module_name,
            ))
    return out
