"""Code-aware ingestion for the lossless context substrate — Phase 22
(extended in Phase 23 with ingestion-coverage accounting, in
Phase 24 with conservative intraprocedural semantic metadata, and
in Phase 25 with conservative interprocedural semantic metadata
computed over a local call graph).

Walks a directory of `.py` files, parses each one with the stdlib `ast`
module, and ingests it into a `ContextLedger` as:

  - `body`     = the file's exact source bytes (lossless).
  - `metadata` = typed structural fields derived from the AST:
       file_path, module_name, line_count, n_functions, n_classes,
       n_methods, imports (tuple), function_names (tuple),
       class_names (tuple), function_returns_none (tuple of bools),
       function_n_args (tuple), function_is_async (tuple),
       function_is_test (tuple), is_test_file, has_docstring,
       n_docstring_chars,

    plus Phase-24 conservative semantic fields (parallel to
    `function_names`, in per-file order):
       function_may_raise, function_is_recursive,
       function_may_write_global, function_calls_subprocess,
       function_calls_filesystem, function_calls_network,

    plus file-level aggregate counts derived from the semantics:
       n_functions_may_raise, n_functions_is_recursive,
       n_functions_may_write_global, n_functions_calls_subprocess,
       n_functions_calls_filesystem, n_functions_calls_network,
       n_functions_calls_external_io.

  Semantic analysis is *conservative*: soundness takes precedence over
  precision. A `True` flag means "the analyzer could not rule it out";
  a `False` flag means "provably not, under this analysis's
  assumptions". See `core.code_semantics` for the exact predicate
  semantics and known boundary conditions.

Why a separate module instead of overloading `context_ledger.put`?

  * Code metadata extraction is non-trivial (AST walk, return-type
    inference) and changes frequently as the corpus changes. Keeping
    it out of the substrate keeps the substrate generic.
  * Code questions need *typed-collection* fields (lists of function
    names, lists of imports) that the operator layer's `Unnest`
    operator (Phase 22) will flatten. Other corpora can stay flat.
  * The same `ContextLedger` is used; nothing about Phase 19/20/21
    behaviour changes. Phase 22 just adds a richer ingestion path.

Deterministic + idempotent: re-ingesting the same file with the same
content produces the same CID (the metadata hash includes only the
extracted fields, not the file's mtime).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Iterator

from .code_semantics import FunctionSemantics, analyze_function
from .context_ledger import ContextLedger, Handle


# =============================================================================
# AST extraction
# =============================================================================


@dataclass
class FunctionInfo:
    """Per-function metadata. Tuple-friendly so it can live in handle metadata."""
    name: str
    n_args: int
    is_async: bool
    returns_none: bool         # True iff every code path provably returns None
    is_test: bool              # name starts with "test_" or class is Test*
    has_docstring: bool
    line_no: int


@dataclass
class CodeMetadata:
    """File-level structural metadata. The substrate stores this in the
    Handle's metadata dict — every field becomes a typed field that
    operators can filter / extract / count over.

    Structural fields (Phase 22): file_path, module_name, line_count,
    n_functions, n_classes, n_methods, imports, function_names,
    class_names, function_returns_none, function_n_args,
    function_is_async, function_is_test, is_test_file, has_docstring,
    n_docstring_chars.

    Conservative semantic fields (Phase 24): parallel boolean tuples
    aligned with `function_names` recording per-function semantic
    facts derived by `core.code_semantics.analyze_function`. Plus
    file-level aggregate counts (`n_functions_may_raise`, etc.).
    """

    file_path: str             # absolute or relative; stable across runs
    module_name: str           # dotted name relative to the corpus root
    line_count: int
    n_functions: int
    n_classes: int
    n_methods: int
    imports: tuple[str, ...]
    function_names: tuple[str, ...]
    class_names: tuple[str, ...]
    function_returns_none: tuple[bool, ...]
    function_n_args: tuple[int, ...]
    function_is_async: tuple[bool, ...]
    function_is_test: tuple[bool, ...]
    is_test_file: bool
    has_docstring: bool
    n_docstring_chars: int

    # --- Phase 24 conservative-semantic fields ----------------------------
    # `semantic_function_names` contains qualified names for BOTH top-level
    # functions AND methods (e.g. "foo" for a top-level `foo`, and
    # "Bar.baz" for a method `baz` on class `Bar`). The semantic boolean
    # tuples below are parallel to this list, in deterministic
    # declaration order. The separate name list keeps Phase-22's
    # `function_names` unchanged (top-level only) so `count_functions_total`
    # and `count_methods_total` stay byte-stable.
    semantic_function_names: tuple[str, ...] = ()
    function_may_raise: tuple[bool, ...] = ()
    function_is_recursive: tuple[bool, ...] = ()
    function_may_write_global: tuple[bool, ...] = ()
    function_calls_subprocess: tuple[bool, ...] = ()
    function_calls_filesystem: tuple[bool, ...] = ()
    function_calls_network: tuple[bool, ...] = ()

    n_functions_may_raise: int = 0
    n_functions_is_recursive: int = 0
    n_functions_may_write_global: int = 0
    n_functions_calls_subprocess: int = 0
    n_functions_calls_filesystem: int = 0
    n_functions_calls_network: int = 0
    n_functions_calls_external_io: int = 0

    # --- Phase 25 interprocedural-semantic fields -------------------------
    # Parallel to `semantic_function_names`, in the same per-file order.
    # Every value is a sound over-approximation of an existence-of-
    # resolved-path claim; see `core.code_interproc` for the exact
    # propagation rules.
    #
    # These tuples are empty until the corpus-wide interprocedural pass
    # runs in `CodeIndexer.index_into`. Per-file `extract_metadata` alone
    # cannot compute them (the call graph spans files), so the fallback
    # — calling `extract_metadata` on a single file — produces zeros.
    function_trans_may_raise: tuple[bool, ...] = ()
    function_trans_may_write_global: tuple[bool, ...] = ()
    function_trans_calls_subprocess: tuple[bool, ...] = ()
    function_trans_calls_filesystem: tuple[bool, ...] = ()
    function_trans_calls_network: tuple[bool, ...] = ()
    function_participates_in_cycle: tuple[bool, ...] = ()
    function_has_unresolved_callees: tuple[bool, ...] = ()

    n_functions_trans_may_raise: int = 0
    n_functions_trans_may_write_global: int = 0
    n_functions_trans_calls_subprocess: int = 0
    n_functions_trans_calls_filesystem: int = 0
    n_functions_trans_calls_network: int = 0
    n_functions_trans_calls_external_io: int = 0
    n_functions_participates_in_cycle: int = 0
    n_functions_has_unresolved_callees: int = 0

    def as_dict(self) -> dict:
        # Plain types only so `Handle.metadata` (which is sorted-tuple of
        # (k, v) pairs) stays hashable.
        return {
            "file_path": self.file_path,
            "module_name": self.module_name,
            "line_count": self.line_count,
            "n_functions": self.n_functions,
            "n_classes": self.n_classes,
            "n_methods": self.n_methods,
            "imports": self.imports,
            "function_names": self.function_names,
            "class_names": self.class_names,
            "function_returns_none": self.function_returns_none,
            "function_n_args": self.function_n_args,
            "function_is_async": self.function_is_async,
            "function_is_test": self.function_is_test,
            "is_test_file": self.is_test_file,
            "has_docstring": self.has_docstring,
            "n_docstring_chars": self.n_docstring_chars,
            # --- Phase 24 ---
            "semantic_function_names": self.semantic_function_names,
            "function_may_raise": self.function_may_raise,
            "function_is_recursive": self.function_is_recursive,
            "function_may_write_global": self.function_may_write_global,
            "function_calls_subprocess": self.function_calls_subprocess,
            "function_calls_filesystem": self.function_calls_filesystem,
            "function_calls_network": self.function_calls_network,
            "n_functions_may_raise": self.n_functions_may_raise,
            "n_functions_is_recursive": self.n_functions_is_recursive,
            "n_functions_may_write_global": self.n_functions_may_write_global,
            "n_functions_calls_subprocess": self.n_functions_calls_subprocess,
            "n_functions_calls_filesystem": self.n_functions_calls_filesystem,
            "n_functions_calls_network": self.n_functions_calls_network,
            "n_functions_calls_external_io": self.n_functions_calls_external_io,
            # --- Phase 25 ---
            "function_trans_may_raise": self.function_trans_may_raise,
            "function_trans_may_write_global": self.function_trans_may_write_global,
            "function_trans_calls_subprocess": self.function_trans_calls_subprocess,
            "function_trans_calls_filesystem": self.function_trans_calls_filesystem,
            "function_trans_calls_network": self.function_trans_calls_network,
            "function_participates_in_cycle": self.function_participates_in_cycle,
            "function_has_unresolved_callees": self.function_has_unresolved_callees,
            "n_functions_trans_may_raise": self.n_functions_trans_may_raise,
            "n_functions_trans_may_write_global": self.n_functions_trans_may_write_global,
            "n_functions_trans_calls_subprocess": self.n_functions_trans_calls_subprocess,
            "n_functions_trans_calls_filesystem": self.n_functions_trans_calls_filesystem,
            "n_functions_trans_calls_network": self.n_functions_trans_calls_network,
            "n_functions_trans_calls_external_io": self.n_functions_trans_calls_external_io,
            "n_functions_participates_in_cycle": self.n_functions_participates_in_cycle,
            "n_functions_has_unresolved_callees": self.n_functions_has_unresolved_callees,
        }


def _is_test_name(name: str) -> bool:
    return name.startswith("test_") or name.startswith("Test")


def _function_returns_none(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Conservative static analysis: True iff EVERY explicit `return` has
    no value (or there is no `return` at all) AND the function is not a
    generator. Defensive about exits.

    We don't try to follow type annotations or external calls. The
    decidable signals are:
      - Annotated return type is `None`
      - No `return X` with a value
      - No `yield` / `yield from` (generator functions return generators,
        not None)

    Functions that propagate by raising are conservatively classified
    by the same rule (no explicit value return → True).
    """
    # Annotated-as-None?
    if (node.returns is not None
            and isinstance(node.returns, ast.Constant)
            and node.returns.value is None):
        return True
    if (node.returns is not None
            and isinstance(node.returns, ast.Name)
            and node.returns.id == "None"):
        return True
    has_value_return = False
    is_generator = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            has_value_return = True
        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
            is_generator = True
    if is_generator:
        return False
    return not has_value_return


def _module_name_from_path(file_path: str, root: str) -> str:
    """Convert a file path to a dotted module name, relative to `root`."""
    rel = os.path.relpath(file_path, root)
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith(os.sep + "__init__"):
        rel = rel[: -len(os.sep + "__init__")]
    return rel.replace(os.sep, ".")


def _extract_metadata_with_tree(
    file_path: str, source: str, *, root: str | None = None,
) -> tuple[CodeMetadata, "ast.Module | None"]:
    """Same as `extract_metadata` but also returns the parsed `ast.Module`
    (or `None` on SyntaxError). Used by `CodeIndexer.index_into` so the
    interprocedural post-pass can reuse the tree without re-parsing.
    """
    md = extract_metadata(file_path, source, root=root)
    # If the metadata is the SyntaxError fallback record, we know
    # parsing failed; no tree available. Otherwise re-parse cheaply
    # (ast.parse is fast; avoiding the re-parse by returning the tree
    # from inside extract_metadata would require refactoring its
    # signature in a way that ripples through Phase-22 callers).
    if (md.n_functions == 0 and md.n_classes == 0
            and not md.imports and not md.has_docstring):
        # Could be syntax-error OR genuinely trivial; re-parse to tell.
        try:
            tree = ast.parse(source, filename=file_path)
            return md, tree
        except SyntaxError:
            return md, None
    try:
        return md, ast.parse(source, filename=file_path)
    except SyntaxError:
        return md, None


def extract_metadata(file_path: str, source: str, *,
                     root: str | None = None) -> CodeMetadata:
    """Parse `source` and produce typed metadata. Never raises on
    SyntaxError — returns a minimal record marked `n_functions=0` so
    the file still appears in the index but isn't analysed.

    Files whose metadata was produced via the SyntaxError fallback are
    identifiable externally by `n_functions == 0 and n_classes == 0 and
    imports == ()` combined with a non-zero `line_count`. Phase 23's
    `IngestionStats` surfaces these explicitly so benchmark reports
    can attribute coverage gaps to parser failure.
    """
    if root is None:
        root = os.path.dirname(file_path)

    line_count = source.count("\n") + (0 if source.endswith("\n") else 1)
    module_name = _module_name_from_path(file_path, root)

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        # All Phase-24 semantic tuples default to empty — the fallback
        # record is explicitly "no structural signal". Phase-23's
        # IngestionStats uses this combination to flag SyntaxError files.
        return CodeMetadata(
            file_path=file_path, module_name=module_name,
            line_count=line_count, n_functions=0, n_classes=0, n_methods=0,
            imports=(), function_names=(), class_names=(),
            function_returns_none=(), function_n_args=(),
            function_is_async=(), function_is_test=(),
            is_test_file=False, has_docstring=False, n_docstring_chars=0,
        )

    imports: list[str] = []
    function_names: list[str] = []
    class_names: list[str] = []
    function_returns_none: list[bool] = []
    function_n_args: list[int] = []
    function_is_async: list[bool] = []
    function_is_test: list[bool] = []
    n_methods = 0

    # Phase 24: collect semantic analysis for every function (top-level
    # and methods). Parallel tuples are built in declaration order.
    semantic_function_names: list[str] = []
    fn_may_raise: list[bool] = []
    fn_is_recursive: list[bool] = []
    fn_may_write_global: list[bool] = []
    fn_calls_subprocess: list[bool] = []
    fn_calls_filesystem: list[bool] = []
    fn_calls_network: list[bool] = []

    # Precompute module-level context for the semantic analyzer once.
    from .code_semantics import (  # local import to avoid a hard cycle
        _collect_import_hints, _module_level_names,
    )
    import_hints = _collect_import_hints(tree)
    module_level_names = _module_level_names(tree)

    def _append_semantics(qualified_name: str, sem: FunctionSemantics) -> None:
        semantic_function_names.append(qualified_name)
        fn_may_raise.append(sem.may_raise)
        fn_is_recursive.append(sem.is_recursive)
        fn_may_write_global.append(sem.may_write_global)
        fn_calls_subprocess.append(sem.calls_subprocess)
        fn_calls_filesystem.append(sem.calls_filesystem)
        fn_calls_network.append(sem.calls_network)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # Record both the module and the dotted "module.name" forms.
            if mod:
                imports.append(mod)
            for n in node.names:
                imports.append(f"{mod}.{n.name}" if mod else n.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
            function_n_args.append(len(node.args.args))
            function_is_async.append(isinstance(node, ast.AsyncFunctionDef))
            function_returns_none.append(_function_returns_none(node))
            function_is_test.append(_is_test_name(node.name))
            sem = analyze_function(
                node, tree=tree, enclosing_class=None,
                import_hints=import_hints,
                module_level_names=module_level_names,
            )
            _append_semantics(node.name, sem)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n_methods += 1
                    sem = analyze_function(
                        sub, tree=tree, enclosing_class=node.name,
                        import_hints=import_hints,
                        module_level_names=module_level_names,
                    )
                    _append_semantics(f"{node.name}.{sub.name}", sem)

    docstr = ast.get_docstring(tree) or ""

    fname = os.path.basename(file_path)
    is_test_file = (fname.startswith("test_")
                    or fname.endswith("_test.py")
                    or "/tests/" in file_path.replace(os.sep, "/"))

    n_raise = sum(fn_may_raise)
    n_rec = sum(fn_is_recursive)
    n_gw = sum(fn_may_write_global)
    n_sp = sum(fn_calls_subprocess)
    n_fs = sum(fn_calls_filesystem)
    n_net = sum(fn_calls_network)
    n_io = sum(1 for a, b, c in zip(fn_calls_subprocess, fn_calls_filesystem,
                                    fn_calls_network) if a or b or c)

    return CodeMetadata(
        file_path=file_path,
        module_name=module_name,
        line_count=line_count,
        n_functions=len(function_names),
        n_classes=len(class_names),
        n_methods=n_methods,
        imports=tuple(imports),
        function_names=tuple(function_names),
        class_names=tuple(class_names),
        function_returns_none=tuple(function_returns_none),
        function_n_args=tuple(function_n_args),
        function_is_async=tuple(function_is_async),
        function_is_test=tuple(function_is_test),
        is_test_file=is_test_file,
        has_docstring=bool(docstr),
        n_docstring_chars=len(docstr),
        semantic_function_names=tuple(semantic_function_names),
        function_may_raise=tuple(fn_may_raise),
        function_is_recursive=tuple(fn_is_recursive),
        function_may_write_global=tuple(fn_may_write_global),
        function_calls_subprocess=tuple(fn_calls_subprocess),
        function_calls_filesystem=tuple(fn_calls_filesystem),
        function_calls_network=tuple(fn_calls_network),
        n_functions_may_raise=n_raise,
        n_functions_is_recursive=n_rec,
        n_functions_may_write_global=n_gw,
        n_functions_calls_subprocess=n_sp,
        n_functions_calls_filesystem=n_fs,
        n_functions_calls_network=n_net,
        n_functions_calls_external_io=n_io,
    )


# =============================================================================
# Directory walk + ingest
# =============================================================================


def _walk_python_files(root: str,
                        include_globs: tuple[str, ...] = ("*.py",),
                        exclude_dirs: tuple[str, ...] = (
                            "__pycache__", ".git", ".venv",
                        )) -> Iterator[str]:
    """Yield every .py file under `root`, deterministic order."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Edit dirnames in place so os.walk skips them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            yield os.path.join(dirpath, f)


@dataclass
class IngestionStats:
    """Coverage accounting for one `CodeIndexer.index_into` run — Phase 23.

    Every .py file under the root flows through four buckets:
      - `files_seen`: total .py files observed by the directory walk
        (bounded by `max_files` if set).
      - `files_parsed_ok`: files whose `ast.parse` succeeded and yielded
        non-trivial metadata (at least one import, function, or class,
        OR a module docstring). This is the strict "structural signal"
        bucket — files where the planner has *something* to operate on.
      - `files_syntax_error`: files where `ast.parse` raised SyntaxError.
        Recorded by `extract_metadata` via the fallback-record shape
        (n_functions=0, n_classes=0, imports=()). Flagged here for
        transparency; they still land in the ledger with minimal
        metadata.
      - `files_oversize_skipped`: files rejected because
        `len(source) > max_chars_per_file`.
      - `files_oserror`: files that raised an OSError on read.
      - `files_ledger_rejected`: files the ledger refused (capacity
        guards, duplicate CIDs, etc.).

    `metadata_completeness` is a scalar in [0, 1] used by Phase 23:
    it is `files_parsed_ok / max(1, files_seen - files_oversize_skipped
    - files_oserror)`. That denominator is the set of files we actually
    tried to parse. A low score means the corpus has a meaningful
    fraction of parseable-but-empty modules (e.g. `__init__.py`
    stubs or syntax-error files) — useful signal for interpreting a
    benchmark result.
    """

    files_seen: int = 0
    files_parsed_ok: int = 0
    files_trivial: int = 0         # parsed but no import/function/class/docstring
    files_syntax_error: int = 0
    files_oversize_skipped: int = 0
    files_oserror: int = 0
    files_ledger_rejected: int = 0
    total_lines: int = 0
    total_bytes: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0

    def attempted_parse(self) -> int:
        """Files where we actually called `extract_metadata`."""
        return self.files_seen - self.files_oversize_skipped - self.files_oserror

    @property
    def metadata_completeness(self) -> float:
        denom = max(1, self.attempted_parse())
        return self.files_parsed_ok / denom

    @property
    def parse_coverage(self) -> float:
        """Fraction of attempted files whose AST parsed successfully
        (includes files that parsed to a trivial metadata record)."""
        denom = max(1, self.attempted_parse())
        return (self.files_parsed_ok + self.files_trivial) / denom

    def as_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "files_parsed_ok": self.files_parsed_ok,
            "files_trivial": self.files_trivial,
            "files_syntax_error": self.files_syntax_error,
            "files_oversize_skipped": self.files_oversize_skipped,
            "files_oserror": self.files_oserror,
            "files_ledger_rejected": self.files_ledger_rejected,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "total_imports": self.total_imports,
            "metadata_completeness": round(self.metadata_completeness, 4),
            "parse_coverage": round(self.parse_coverage, 4),
        }


def _patch_with_interproc(
    collected: list[tuple[str, str, "CodeMetadata", "ast.Module | None"]],
) -> None:
    """Phase-25 post-pass. Given a list of `(file_path, source, md,
    tree)` tuples from `CodeIndexer.index_into`'s scan phase, build
    the corpus-wide call graph, propagate conservative effect
    predicates, and mutate each `md` in place to carry the Phase-25
    boolean tuples + aggregate counts.

    The function is a no-op on corpora with zero parseable trees; the
    metadata records keep their default-empty Phase-25 tuples.
    """
    from .code_interproc import (
        analyze_interproc, build_module_context,
    )
    from .code_semantics import FunctionSemantics

    modules = []
    intra_all: dict[str, FunctionSemantics] = {}
    qname_by_file: dict[str, list[str]] = {}
    for fp, _source, md, tree in collected:
        if tree is None:
            continue
        module_name = md.module_name
        ctx, intra_map = build_module_context(module_name, tree)
        modules.append(ctx)
        intra_all.update(intra_map)
        # Build the parallel qname list in the order the intra-
        # procedural analyzer used (which is declaration order:
        # top-level functions first, then methods in class order).
        qnames_here: list[str] = []
        for sub in ast.iter_child_nodes(tree):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qnames_here.append(f"{module_name}.{sub.name}")
            elif isinstance(sub, ast.ClassDef):
                for inner in ast.iter_child_nodes(sub):
                    if isinstance(inner, (ast.FunctionDef,
                                           ast.AsyncFunctionDef)):
                        qnames_here.append(
                            f"{module_name}.{sub.name}.{inner.name}")
        qname_by_file[fp] = qnames_here

    if not modules:
        return

    interproc_map, _cg = analyze_interproc(modules, intra_all)

    # Patch each file's metadata with parallel boolean tuples whose
    # order matches `semantic_function_names`.
    for fp, _source, md, tree in collected:
        if tree is None or md.semantic_function_names == ():
            continue
        qnames_here = qname_by_file.get(fp, [])
        # Sanity: per-file qname list should line up with the per-file
        # semantic_function_names length. If not, bail out of patching
        # this file (we'd rather ship empty trans-tuples than mis-
        # aligned ones — the gold depends on alignment).
        if len(qnames_here) != len(md.semantic_function_names):
            continue
        t_mr: list[bool] = []
        t_mwg: list[bool] = []
        t_cs: list[bool] = []
        t_cfs: list[bool] = []
        t_cn: list[bool] = []
        t_cyc: list[bool] = []
        t_unr: list[bool] = []
        for q in qnames_here:
            sem = interproc_map.get(q)
            if sem is None:
                # Should not happen, but be conservative — zero.
                t_mr.append(False); t_mwg.append(False)
                t_cs.append(False); t_cfs.append(False); t_cn.append(False)
                t_cyc.append(False); t_unr.append(False)
                continue
            t_mr.append(sem.trans_may_raise)
            t_mwg.append(sem.trans_may_write_global)
            t_cs.append(sem.trans_calls_subprocess)
            t_cfs.append(sem.trans_calls_filesystem)
            t_cn.append(sem.trans_calls_network)
            t_cyc.append(sem.participates_in_cycle)
            t_unr.append(sem.has_unresolved_callees)
        md.function_trans_may_raise = tuple(t_mr)
        md.function_trans_may_write_global = tuple(t_mwg)
        md.function_trans_calls_subprocess = tuple(t_cs)
        md.function_trans_calls_filesystem = tuple(t_cfs)
        md.function_trans_calls_network = tuple(t_cn)
        md.function_participates_in_cycle = tuple(t_cyc)
        md.function_has_unresolved_callees = tuple(t_unr)
        md.n_functions_trans_may_raise = sum(t_mr)
        md.n_functions_trans_may_write_global = sum(t_mwg)
        md.n_functions_trans_calls_subprocess = sum(t_cs)
        md.n_functions_trans_calls_filesystem = sum(t_cfs)
        md.n_functions_trans_calls_network = sum(t_cn)
        md.n_functions_trans_calls_external_io = sum(
            1 for a, b, c in zip(t_cs, t_cfs, t_cn) if a or b or c
        )
        md.n_functions_participates_in_cycle = sum(t_cyc)
        md.n_functions_has_unresolved_callees = sum(t_unr)


def _is_trivial(md: "CodeMetadata") -> bool:
    """True iff `md` carries no structural signal a planner can use."""
    return (md.n_functions == 0 and md.n_classes == 0
            and not md.imports and not md.has_docstring)


def _is_syntax_error_record(md: "CodeMetadata", source: str) -> bool:
    """True iff the metadata looks like the SyntaxError fallback record.

    The fallback produces (0,0,0, (), (), ...) on a non-empty source. We
    detect the case heuristically: all structural fields empty AND the
    source contains at least one non-blank line that is not a comment.
    For empty / comment-only files this returns False (they *are*
    trivial, but not parse failures)."""
    if md.n_functions or md.n_classes or md.imports or md.has_docstring:
        return False
    # A parseable empty/comment-only file is trivial but not an error.
    # A parseable file that defines nothing (imports alone live in
    # `md.imports`) is also trivial. The only way we land in the
    # SyntaxError path is if `ast.parse` itself raised — which we can
    # detect by trying again here with a cheap re-parse.
    try:
        ast.parse(source)
        return False
    except SyntaxError:
        return True


@dataclass
class CodeIndexer:
    """Walk a directory tree and ingest every Python file into a ledger.

    Usage:
        indexer = CodeIndexer(root="vision_mvp/core")
        ledger  = ContextLedger(embed_dim=64, embed_fn=hash_embedding)
        n = indexer.index_into(ledger)

    The indexer maintains an `IngestionStats` in `self.stats` after each
    `index_into` call — Phase 23 reports use this for coverage
    accounting across corpora.
    """

    root: str
    max_files: int | None = None
    max_chars_per_file: int = 64_000
    stats: IngestionStats = field(default_factory=IngestionStats)

    def files(self) -> list[str]:
        out: list[str] = []
        for p in _walk_python_files(self.root):
            out.append(p)
            if self.max_files is not None and len(out) >= self.max_files:
                break
        return out

    def index_into(self, ledger: ContextLedger,
                    progress_cb=None) -> list[Handle]:
        """Read each file, parse it, ingest into `ledger`. Returns the
        list of handles in ingestion order. Files larger than
        `max_chars_per_file` are skipped. Side effect: populates
        `self.stats` with ingestion coverage accounting.

        Phase 25 adds an interprocedural post-pass between scan and
        ingest: after every file has its intraprocedural metadata and
        parsed AST, the indexer builds a corpus-wide call graph and
        runs conservative effect propagation + SCC detection. The
        per-file metadata is then patched with the transitive boolean
        tuples and aggregate counts before each file is put into the
        ledger. This keeps handles self-describing — the planner reads
        interproc tuples through the same `metadata_dict()` API it
        uses for intraprocedural tuples, with no new lookup path.
        """
        # Fresh stats per call so successive runs don't accumulate.
        self.stats = IngestionStats()
        all_files = self.files()
        self.stats.files_seen = len(all_files)

        # Phase 1 — scan: parse each file, compute per-function intra
        # semantics, retain the AST for the interprocedural pass.
        collected: list[tuple[str, str, "CodeMetadata", "ast.Module | None"]] = []
        for fp in all_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except OSError:
                self.stats.files_oserror += 1
                continue
            if len(source) > self.max_chars_per_file:
                self.stats.files_oversize_skipped += 1
                continue
            md, tree = _extract_metadata_with_tree(
                fp, source, root=self.root)
            if _is_syntax_error_record(md, source):
                self.stats.files_syntax_error += 1
            elif _is_trivial(md):
                self.stats.files_trivial += 1
            else:
                self.stats.files_parsed_ok += 1
            self.stats.total_lines += md.line_count
            self.stats.total_bytes += len(source)
            self.stats.total_functions += md.n_functions
            self.stats.total_classes += md.n_classes
            self.stats.total_imports += len(md.imports)
            collected.append((fp, source, md, tree))

        # Phase 2 — corpus-wide interprocedural analysis. If every
        # file failed to parse, this is a no-op and the metadata
        # keeps empty trans-tuples.
        _patch_with_interproc(collected)

        # Phase 3 — ingest each file's patched metadata into the ledger.
        handles: list[Handle] = []
        for i, (fp, source, md, _tree) in enumerate(collected):
            try:
                h = ledger.put(source, metadata=md.as_dict())
            except Exception:
                self.stats.files_ledger_rejected += 1
                continue
            handles.append(h)
            if progress_cb is not None and (i + 1) % max(
                    1, len(collected) // 5) == 0:
                progress_cb(f"  [code-index] {i+1}/{len(collected)} ingested "
                            f"({len(handles)} kept)")
        return handles
