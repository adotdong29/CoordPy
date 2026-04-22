"""Interprocedural conservative semantic analysis — Phase 25.

Phase 24 shipped *intraprocedural* predicates computed against each
function's own AST body. A wrapper that calls a helper which itself
invokes `subprocess.run` was NOT flagged as `calls_subprocess` because
the call went through a user function. Phase 25 closes that gap by
building a **local call graph** across the functions defined in a
corpus, then **propagating each conservative semantic predicate
transitively** across the graph until a least fixed point is reached.

The module exposes two public entry points:

  * `build_call_graph(modules)` — takes a list of pre-indexed
    `ModuleContext` records and returns a `CallGraph` with resolved
    edges and a separate set of *unresolved* call targets per caller.

  * `analyze_interproc(modules, intra_semantics)` — runs the
    propagation over the call graph and returns a dict mapping each
    qualified function name to an `InterprocSemantics` record with:

        trans_may_raise            f may raise OR any resolved
                                   callee transitively may
        trans_may_write_global     likewise for global writes
        trans_calls_subprocess     likewise for subprocess APIs
        trans_calls_filesystem     likewise for filesystem APIs
        trans_calls_network        likewise for network APIs
        participates_in_cycle      f belongs to a non-trivial SCC in
                                   the resolved call graph, OR f is
                                   self-recursive. This is a proper
                                   superset of Phase-24
                                   `is_recursive`: mutual recursion
                                   (f→g→f) flags both f and g.
        has_unresolved_callees     f statically calls at least one
                                   target that could NOT be resolved
                                   to a function in the corpus AND
                                   that is not a known-API surface
                                   (subprocess.*, open, socket.*, …).

Soundness stance

The analysis over-approximates the *existence* of a resolved effect
path. It does NOT over-approximate through unresolved calls — an
unresolved call target contributes nothing to any `trans_*` predicate.
This is the "resolved-only" convention:

  - True  ⇒ there IS a resolved call chain from f to a function in
           the corpus that is flagged intraprocedurally. Sound.
  - False ⇒ no such resolved chain exists. An unresolved callee COULD
           still do the effect (we can't see inside it), but the
           analyzer prefers honesty over an opaque True. The
           `has_unresolved_callees` flag is surfaced separately so
           that callers who want the "maybe-does-X" semantics can
           widen trans-flags by OR-ing with that bit.

This stance is the same one underlying Phase 24's `calls_*` predicates,
extended across the corpus-wide call graph.

Name resolution

Call targets are resolved in this order (each attempt is strict; a
failure falls through to the next):

  1. Bare name `foo` → `{caller_module}.foo` if `foo` is a top-level
     function defined in the caller's module.
  2. `self.name` within a method of class C → `{caller_module}.C.name`
     if `C.name` is defined.
  3. `C.name` where C is a class in the caller's module →
     `{caller_module}.C.name`.
  4. `mod.name` where `mod` was `import`ed (possibly as an alias) and
     `<full-qualified-module>.name` is defined in the corpus.
  5. Bare name `name` where `from src_mod import name [as alias]` was
     recorded in the caller's module and `src_mod.name` is in the
     corpus.

Anything else (including relative-import resolution, attribute
chains through instance variables, and reflection) is treated as
unresolved — the call edge is recorded in `CallGraph.unresolved[f]`
but does NOT propagate effects. Unresolved targets that match a
Phase-24 known-API surface (e.g. `subprocess.run`) are NOT recorded
as unresolved either — the intra-procedural analyzer already
captures them.

Complexity

All three operations are linear in the graph size:

  - Graph construction: O(sum of AST nodes across modules).
  - SCC (Tarjan): O(|V| + |E|).
  - Fixed-point propagation: O(|V| + |E|) per predicate via a
    worklist seeded by intra-procedurally True functions.

With six propagated predicates this is 6·(|V|+|E|) work, entirely
negligible against embedding cost in real ingestion runs.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from .code_semantics import (
    FunctionSemantics,
    _FILESYSTEM_APIS, _NETWORK_APIS, _SUBPROCESS_APIS,
    _call_name,
)


# =============================================================================
# Public dataclasses
# =============================================================================


@dataclass(frozen=True)
class InterprocSemantics:
    """Per-function interprocedural semantic summary.

    Every boolean is a conservative over-approximation of an
    existence-of-resolved-path claim. See the module docstring for the
    full soundness argument.

    `trans_may_raise_implicit` (Phase 28) is propagated in the same
    way as the other trans-* flags, over the same resolved call
    graph. See `RESULTS_PHASE28.md` § "Explicit vs implicit raise
    semantics" for why this is a separate axis from `trans_may_raise`.
    """

    trans_may_raise: bool
    trans_may_write_global: bool
    trans_calls_subprocess: bool
    trans_calls_filesystem: bool
    trans_calls_network: bool
    participates_in_cycle: bool
    has_unresolved_callees: bool
    # Phase 28: appended so positional consumers keep the Phase-25
    # field order.
    trans_may_raise_implicit: bool = False

    @property
    def trans_calls_external_io(self) -> bool:
        """Convenience disjunction — matches Phase-24 naming."""
        return (self.trans_calls_subprocess
                or self.trans_calls_filesystem
                or self.trans_calls_network)

    def as_tuple(self) -> tuple[bool, ...]:
        return (
            self.trans_may_raise,
            self.trans_may_write_global,
            self.trans_calls_subprocess,
            self.trans_calls_filesystem,
            self.trans_calls_network,
            self.participates_in_cycle,
            self.has_unresolved_callees,
            self.trans_may_raise_implicit,
        )


@dataclass
class ModuleContext:
    """Pre-indexed data for one module used by the call-graph builder.

    The CodeIndexer builds one `ModuleContext` per file during the
    first pass (parse + intra-procedural analysis). The call-graph
    pass then consumes all of them together.

    Attributes:
      module_name: the dotted module name (same as `CodeMetadata.module_name`).
      functions_by_qname: map from a *fully-qualified* function name
          (e.g. `"pkg.mod.foo"` for a top-level `foo` in module
          `pkg.mod`, or `"pkg.mod.Bar.baz"` for a method `baz` on
          class `Bar`) to its AST node.
      enclosing_class: qname → class name (or None for top-level).
      import_hints: frozenset of strings naming imported modules /
          members. Used to recognise named-import API aliases.
      module_aliases: dict mapping a local name bound by `import` to
          a full module name (e.g. `"np" → "numpy"` from
          `import numpy as np`).
      name_imports: dict mapping a local name bound by `from src
          import name [as alias]` to the *source* `src.name` (e.g.
          `"run" → "subprocess.run"` from `from subprocess import
          run`). The value is a dotted string; resolution queries the
          corpus qnames map.
    """

    module_name: str
    functions_by_qname: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    enclosing_class: dict[str, str | None]
    import_hints: frozenset[str]
    module_aliases: dict[str, str]
    name_imports: dict[str, str]


@dataclass
class CallGraph:
    """Directed graph of intra-corpus function calls.

    - `nodes`: every function qname in the corpus.
    - `edges[f]`: resolved callees of f (every element is in nodes).
    - `unresolved[f]`: dotted call targets that could not be resolved
      to a node AND are not a known-API surface.
    - `self_recursive`: qnames with at least one self-loop edge.
    - `rev_edges[g]`: callers of g (reverse of `edges`). Used by the
      worklist propagator to push parents when a child becomes True.
    """

    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    unresolved: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    self_recursive: set[str] = field(default_factory=set)
    rev_edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def summary(self) -> dict:
        """Compact description — useful in tests and result JSON."""
        return {
            "n_nodes": len(self.nodes),
            "n_edges": sum(len(s) for s in self.edges.values()),
            "n_self_recursive": len(self.self_recursive),
            "n_callers_with_unresolved": sum(
                1 for s in self.unresolved.values() if s),
            "n_unresolved_call_sites": sum(
                len(s) for s in self.unresolved.values()),
        }


# =============================================================================
# Call-name resolution
# =============================================================================


def _is_known_api_surface(dotted: str,
                           import_hints: frozenset[str]) -> bool:
    """True iff `dotted` names a call target already captured by the
    intraprocedural IO predicates (Phase 24). Such calls must NOT be
    tallied as 'unresolved' — they ARE fully resolved, just to an
    external API the analyzer knows about."""
    for table in (_SUBPROCESS_APIS, _FILESYSTEM_APIS, _NETWORK_APIS):
        if dotted in table:
            return True
    if "." not in dotted:
        for table in (_SUBPROCESS_APIS, _FILESYSTEM_APIS, _NETWORK_APIS):
            for entry in table:
                parts = entry.rsplit(".", 1)
                if (len(parts) == 2 and parts[1] == dotted
                        and parts[0] in import_hints):
                    return True
    return False


def _resolve_call(
    dotted: str,
    *,
    caller_module: str,
    enclosing_class: str | None,
    module_local_top_level: dict[str, str],
    module_aliases: dict[str, str],
    name_imports: dict[str, str],
    all_module_top_level_qnames: dict[str, dict[str, str]],
    all_method_qnames: frozenset[str],
) -> str | None:
    """Resolve a dotted call target to a qname in the corpus.

    Returns the qname when the target is a function defined somewhere
    in the corpus; returns None when it cannot be resolved. See the
    module docstring for the resolution rule order.
    """
    if not dotted:
        return None

    # Case 1 — bare name, local top-level.
    if "." not in dotted:
        if dotted in module_local_top_level:
            return module_local_top_level[dotted]
        # Case 5 — bare name imported from another corpus module.
        if dotted in name_imports:
            source = name_imports[dotted]
            # source is a dotted "module.name" (or deeper). Try the
            # deepest matching prefix that maps to a corpus module.
            parts = source.rsplit(".", 1)
            if len(parts) == 2:
                src_mod, src_bare = parts
                qmap = all_module_top_level_qnames.get(src_mod)
                if qmap is not None and src_bare in qmap:
                    return qmap[src_bare]
        return None

    parts = dotted.split(".")

    # Case 2 — `self.name[...]` inside a method.
    if parts[0] == "self" and len(parts) >= 2 and enclosing_class is not None:
        method = parts[1]
        candidate = f"{caller_module}.{enclosing_class}.{method}"
        if candidate in all_method_qnames:
            return candidate
        return None

    # Case 3 — `LocalClass.method` (also catches deeper chains by
    # trying just the first two segments).
    if len(parts) >= 2:
        head, second = parts[0], parts[1]
        candidate_method = f"{caller_module}.{head}.{second}"
        if candidate_method in all_method_qnames:
            return candidate_method

    # Case 4 — `module_alias.name`. The head is a local name bound to
    # a full module by `import <full> [as head]`.
    if parts[0] in module_aliases and len(parts) == 2:
        src_mod = module_aliases[parts[0]]
        qmap = all_module_top_level_qnames.get(src_mod)
        if qmap is not None and parts[1] in qmap:
            return qmap[parts[1]]

    # Case 4b — `module_alias.Class.method` or deeper. Only the
    # shallowest form (alias.func) is handled above; for `alias.Cls.m`
    # we look up the method qname directly.
    if parts[0] in module_aliases and len(parts) >= 3:
        src_mod = module_aliases[parts[0]]
        # Build candidate method qname.
        cand = f"{src_mod}." + ".".join(parts[1:])
        if cand in all_method_qnames:
            return cand
        # Fall through — too ambiguous for deeper chains.

    return None


# =============================================================================
# Call-graph construction
# =============================================================================


def build_call_graph(modules: list[ModuleContext]) -> CallGraph:
    """Construct the corpus-wide call graph by walking every
    function's body and resolving each `ast.Call` target.

    Idempotent and deterministic given `modules` (order of `modules`
    affects only `edges` iteration order, not the graph itself).
    """
    cg = CallGraph()

    # Global qname tables — precomputed once so resolution is O(1).
    all_top_level: dict[str, dict[str, str]] = {}
    for m in modules:
        local: dict[str, str] = {}
        for qname in m.functions_by_qname:
            if m.enclosing_class.get(qname) is None:
                # Top-level function in this module — qname = `module.bare`.
                bare = qname[len(m.module_name) + 1:] if qname.startswith(
                    m.module_name + ".") else qname
                local[bare] = qname
        all_top_level[m.module_name] = local

    all_method_qnames = frozenset(
        q for m in modules for q in m.functions_by_qname
        if m.enclosing_class.get(q) is not None
    )

    # Register every qname as a node.
    for m in modules:
        for q in m.functions_by_qname:
            cg.nodes.add(q)

    # Walk each function and populate edges.
    for m in modules:
        top_level_here = all_top_level[m.module_name]
        for caller_qname, node in m.functions_by_qname.items():
            enclosing_class = m.enclosing_class.get(caller_qname)
            callees: set[str] = set()
            unresolved: set[str] = set()
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                dotted = _call_name(sub.func)
                if dotted is None:
                    continue
                resolved = _resolve_call(
                    dotted,
                    caller_module=m.module_name,
                    enclosing_class=enclosing_class,
                    module_local_top_level=top_level_here,
                    module_aliases=m.module_aliases,
                    name_imports=m.name_imports,
                    all_module_top_level_qnames=all_top_level,
                    all_method_qnames=all_method_qnames,
                )
                if resolved is not None:
                    callees.add(resolved)
                    if resolved == caller_qname:
                        cg.self_recursive.add(caller_qname)
                    continue
                if _is_known_api_surface(dotted, m.import_hints):
                    # Known external API — intra-procedural analyzer
                    # owns this. Don't classify as unresolved.
                    continue
                unresolved.add(dotted)
            cg.edges[caller_qname] = callees
            cg.unresolved[caller_qname] = unresolved
            for c in callees:
                cg.rev_edges[c].add(caller_qname)

    return cg


# =============================================================================
# SCC (Tarjan, iterative) for recursion cycles
# =============================================================================


def strongly_connected_components(
    nodes: Iterable[str],
    edges: dict[str, set[str]],
) -> list[list[str]]:
    """Return the list of SCCs of the directed graph (nodes, edges).

    Iterative Tarjan implementation — Python recursion would blow up
    on large corpora. Output is a list of lists of node names; each
    inner list is one SCC in arbitrary order.
    """
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    dfs_stack: list[str] = []
    work_stack: list[tuple[str, list[str]]] = []
    sccs: list[list[str]] = []
    next_index = [0]

    def iter_children(v: str) -> list[str]:
        return sorted(edges.get(v, ()))

    for start in sorted(nodes):
        if start in index_of:
            continue
        # Iterative DFS with explicit work stack.
        work_stack.append((start, iter_children(start)))
        index_of[start] = next_index[0]
        lowlink[start] = next_index[0]
        next_index[0] += 1
        dfs_stack.append(start)
        on_stack.add(start)
        while work_stack:
            v, children = work_stack[-1]
            advanced = False
            while children:
                w = children.pop(0)
                if w not in index_of:
                    index_of[w] = next_index[0]
                    lowlink[w] = next_index[0]
                    next_index[0] += 1
                    dfs_stack.append(w)
                    on_stack.add(w)
                    work_stack.append((w, iter_children(w)))
                    advanced = True
                    break
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_of[w])
            if advanced:
                continue
            # All children exhausted for v — post-order.
            if lowlink[v] == index_of[v]:
                scc: list[str] = []
                while True:
                    w = dfs_stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
            work_stack.pop()
            if work_stack:
                parent_v, _ = work_stack[-1]
                lowlink[parent_v] = min(lowlink[parent_v], lowlink[v])

    return sccs


# =============================================================================
# Least-fixed-point propagation
# =============================================================================


def propagate_effect(call_graph: CallGraph,
                     intra_flags: dict[str, bool]) -> dict[str, bool]:
    """Compute the least fixed point of

        trans[f] = intra[f]  ∨  OR over g in callees(f) of trans[g]

    using a monotone worklist. `intra_flags` must be defined for every
    node in the call graph; missing keys are treated as False.

    Returns a fresh dict (intra_flags is not mutated).
    """
    trans: dict[str, bool] = {q: bool(intra_flags.get(q, False))
                              for q in call_graph.nodes}
    work: deque[str] = deque(q for q, v in trans.items() if v)
    while work:
        f = work.popleft()
        for caller in call_graph.rev_edges.get(f, ()):
            if not trans.get(caller, False):
                trans[caller] = True
                work.append(caller)
    return trans


# =============================================================================
# Public top-level entry
# =============================================================================


def analyze_interproc(
    modules: list[ModuleContext],
    intra_semantics: dict[str, FunctionSemantics],
) -> tuple[dict[str, InterprocSemantics], CallGraph]:
    """Run the full interprocedural pass.

    Args:
      modules: one `ModuleContext` per corpus file (same order as
        ingestion — determinism matters for downstream provenance).
      intra_semantics: qname → `FunctionSemantics` from the Phase-24
        per-function analyzer. Every key appearing in `modules` must
        have an entry here.

    Returns:
      - dict qname → `InterprocSemantics`.
      - the call graph used (exposed so benchmarks can log
        graph-level aggregates — n_edges, n_unresolved_callsites).
    """
    cg = build_call_graph(modules)

    def _intra_map(getter) -> dict[str, bool]:
        return {q: getter(intra_semantics[q])
                for q in cg.nodes if q in intra_semantics}

    trans_mr  = propagate_effect(cg, _intra_map(lambda s: s.may_raise))
    trans_mwg = propagate_effect(cg, _intra_map(lambda s: s.may_write_global))
    trans_cs  = propagate_effect(cg, _intra_map(lambda s: s.calls_subprocess))
    trans_cfs = propagate_effect(cg, _intra_map(lambda s: s.calls_filesystem))
    trans_cn  = propagate_effect(cg, _intra_map(lambda s: s.calls_network))
    # Phase 28 — implicit-raise propagation over the same resolved
    # call graph. Kept strictly orthogonal to `trans_may_raise`.
    trans_mri = propagate_effect(
        cg, _intra_map(lambda s: getattr(s, "may_raise_implicit", False))
    )

    sccs = strongly_connected_components(cg.nodes, cg.edges)
    in_cycle: set[str] = set(cg.self_recursive)
    for scc in sccs:
        if len(scc) >= 2:
            in_cycle.update(scc)

    out: dict[str, InterprocSemantics] = {}
    for q in cg.nodes:
        out[q] = InterprocSemantics(
            trans_may_raise=trans_mr.get(q, False),
            trans_may_write_global=trans_mwg.get(q, False),
            trans_calls_subprocess=trans_cs.get(q, False),
            trans_calls_filesystem=trans_cfs.get(q, False),
            trans_calls_network=trans_cn.get(q, False),
            participates_in_cycle=q in in_cycle,
            has_unresolved_callees=bool(cg.unresolved.get(q)),
            trans_may_raise_implicit=trans_mri.get(q, False),
        )
    return out, cg


# =============================================================================
# Convenience: build ModuleContext from a parsed tree
# =============================================================================


def build_module_context(
    module_name: str,
    tree: ast.Module,
) -> tuple[ModuleContext, dict[str, FunctionSemantics]]:
    """Produce a `ModuleContext` and a parallel intra-semantics map
    directly from a parsed AST. Used by tests and by
    `CodeIndexer.index_into`'s first pass when a caller already has
    the tree in hand.

    The intra-semantics map is keyed by qualified name (same form as
    `ModuleContext.functions_by_qname`).
    """
    # Deferred import to avoid a hard cycle.
    from .code_semantics import (
        _collect_import_hints, _module_level_names, analyze_function,
    )

    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    enc: dict[str, str | None] = {}
    module_aliases: dict[str, str] = {}
    name_imports: dict[str, str] = {}
    import_hints = _collect_import_hints(tree)
    module_level_names = _module_level_names(tree)

    # Collect import-binding tables.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                local = n.asname or n.name.split(".")[0]
                module_aliases[local] = n.name
        elif isinstance(node, ast.ImportFrom):
            src = node.module or ""
            # Relative imports: level>0. We cannot resolve these
            # generally without the package path; the best we can do
            # is treat `src` as-is, which WILL miss corpus matches for
            # relative imports. That's sound — unresolved calls don't
            # propagate. Callers who care can lift this later.
            for n in node.names:
                local = n.asname or n.name
                if src:
                    name_imports[local] = f"{src}.{n.name}"

    intra_map: dict[str, FunctionSemantics] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qname = f"{module_name}.{node.name}"
            funcs[qname] = node
            enc[qname] = None
            intra_map[qname] = analyze_function(
                node, tree=tree, enclosing_class=None,
                import_hints=import_hints,
                module_level_names=module_level_names,
            )
        elif isinstance(node, ast.ClassDef):
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qname = f"{module_name}.{node.name}.{sub.name}"
                    funcs[qname] = sub
                    enc[qname] = node.name
                    intra_map[qname] = analyze_function(
                        sub, tree=tree, enclosing_class=node.name,
                        import_hints=import_hints,
                        module_level_names=module_level_names,
                    )

    ctx = ModuleContext(
        module_name=module_name,
        functions_by_qname=funcs,
        enclosing_class=enc,
        import_hints=import_hints,
        module_aliases=module_aliases,
        name_imports=name_imports,
    )
    return ctx, intra_map
