"""Python codebase task — Phase 22.

Treats a directory of Python source files as the corpus and generates a
deterministic question battery whose ground truth is computed directly
from the AST-derived metadata (the same metadata the indexer ingests
into the ledger). Because both the corpus and the gold come from the
SAME ast walk, the gold is exact by construction — no human curation
needed.

Question kinds:

  Counting:
    - count_files                    "How many Python files are in the corpus?"
    - count_functions_total          "How many functions are defined in total?"
    - count_classes_total            "How many classes are defined?"
    - count_test_files               "How many test files are in the corpus?"
    - count_distinct_imports         "How many distinct modules does the corpus import?"

  Listing:
    - list_files_importing_X         "List files importing {X}."  (X chosen from popular imports)
    - list_functions_returning_none  "List functions that return None."

  Top:
    - top_file_by_functions          "Which file has the most functions?"
    - largest_file                   "What is the largest file by line count?"

The list questions can have moderately large gold sets — we use the
NeedleQuestion `accept_all` field so scoring requires every expected
substring to appear.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ..core.code_index import (
    CodeIndexer, CodeMetadata, _patch_with_interproc,
    _extract_metadata_with_tree, extract_metadata,
)
from .needle_corpus import NeedleQuestion


_KIND_NAMES = (
    "count_files",
    "count_functions_total",
    "count_classes_total",
    "count_test_files",
    "count_distinct_imports",
    "list_files_importing",
    "list_functions_returning_none",
    "top_file_by_functions",
    "largest_file",
    # Phase 23 additions — broader direct-exact coverage across corpora
    "count_methods_total",
    "count_files_with_docstrings",
    "most_imported_module",
    # Phase 24 additions — conservative-semantic question battery.
    # The gold answer for each is computed by the corpus from the
    # SAME `core.code_semantics` analyzer that the planner reads, so
    # direct-exact is tautologically 1.0 when the corpus's semantics
    # are accepted as the ground-truth definition. (The predicate is
    # "does the conservative analyser flag this?", not "is this
    # actually raising at runtime?".)
    "count_may_raise",
    "list_may_raise",
    "count_is_recursive",
    "list_is_recursive",
    "count_may_write_global",
    "list_may_write_global",
    "count_calls_subprocess",
    "list_calls_subprocess",
    "count_calls_filesystem",
    "list_calls_filesystem",
    "count_calls_network",
    "list_calls_network",
    "count_calls_external_io",
    # Phase 25 additions — conservative *interprocedural* question
    # battery. Gold answers are computed by the corpus from the SAME
    # `core.code_interproc` propagation the planner consults, so
    # direct-exact is tautologically 1.0 on these. The interesting
    # experimental question is whether retrieval / LLM-mediated
    # paths can recover transitive flags (they empirically cannot).
    "count_trans_may_raise",
    "list_trans_may_raise",
    "count_trans_may_write_global",
    "list_trans_may_write_global",
    "count_trans_calls_subprocess",
    "list_trans_calls_subprocess",
    "count_trans_calls_filesystem",
    "list_trans_calls_filesystem",
    "count_trans_calls_network",
    "list_trans_calls_network",
    "count_trans_calls_external_io",
    "count_participates_in_cycle",
    "list_participates_in_cycle",
    "count_has_unresolved_callees",
)


@dataclass
class PythonCorpus:
    """A code-corpus task built from a directory walk."""

    root: str
    max_files: int | None = None
    max_chars_per_file: int = 64_000
    max_listing_size: int = 8         # cap "list X" gold sets for tractability
    seed: int = 22

    files: list[str] = field(default_factory=list)
    metadata: list[CodeMetadata] = field(default_factory=list)
    questions: list[NeedleQuestion] = field(default_factory=list)

    def build(self) -> None:
        indexer = CodeIndexer(
            root=self.root, max_files=self.max_files,
            max_chars_per_file=self.max_chars_per_file,
        )
        self.files = []
        self.metadata = []
        # Collect (file_path, source, md, tree) tuples in the same
        # shape `CodeIndexer.index_into` uses, so we can hand them to
        # the shared Phase-25 post-pass.
        collected: list = []
        for fp in indexer.files():
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    src = f.read()
            except OSError:
                continue
            if len(src) > self.max_chars_per_file:
                continue
            md, tree = _extract_metadata_with_tree(fp, src, root=self.root)
            collected.append((fp, src, md, tree))
        # Phase-25 post-pass: populate transitive boolean tuples +
        # aggregates by running the cross-file call-graph analysis.
        _patch_with_interproc(collected)
        for fp, _src, md, _tree in collected:
            self.files.append(fp)
            self.metadata.append(md)
        self._build_questions()

    # ---- Per-corpus aggregates used to compute gold answers ---------

    @property
    def n_files(self) -> int:
        return len(self.metadata)

    @property
    def n_functions_total(self) -> int:
        return sum(m.n_functions for m in self.metadata)

    @property
    def n_classes_total(self) -> int:
        return sum(m.n_classes for m in self.metadata)

    @property
    def n_test_files(self) -> int:
        return sum(1 for m in self.metadata if m.is_test_file)

    @property
    def all_imports(self) -> Counter:
        c: Counter = Counter()
        for m in self.metadata:
            c.update(m.imports)
        return c

    @property
    def n_distinct_imports(self) -> int:
        return len(self.all_imports)

    @property
    def n_methods_total(self) -> int:
        return sum(m.n_methods for m in self.metadata)

    @property
    def n_files_with_docstrings(self) -> int:
        return sum(1 for m in self.metadata if m.has_docstring)

    # ---- Phase-24 semantic aggregates --------------------------------

    @property
    def n_functions_may_raise(self) -> int:
        return sum(m.n_functions_may_raise for m in self.metadata)

    @property
    def n_functions_is_recursive(self) -> int:
        return sum(m.n_functions_is_recursive for m in self.metadata)

    @property
    def n_functions_may_write_global(self) -> int:
        return sum(m.n_functions_may_write_global for m in self.metadata)

    @property
    def n_functions_calls_subprocess(self) -> int:
        return sum(m.n_functions_calls_subprocess for m in self.metadata)

    @property
    def n_functions_calls_filesystem(self) -> int:
        return sum(m.n_functions_calls_filesystem for m in self.metadata)

    @property
    def n_functions_calls_network(self) -> int:
        return sum(m.n_functions_calls_network for m in self.metadata)

    @property
    def n_functions_calls_external_io(self) -> int:
        return sum(m.n_functions_calls_external_io for m in self.metadata)

    # ---- Phase-25 interprocedural aggregates ------------------------

    @property
    def n_functions_trans_may_raise(self) -> int:
        return sum(m.n_functions_trans_may_raise for m in self.metadata)

    @property
    def n_functions_trans_may_write_global(self) -> int:
        return sum(m.n_functions_trans_may_write_global for m in self.metadata)

    @property
    def n_functions_trans_calls_subprocess(self) -> int:
        return sum(m.n_functions_trans_calls_subprocess for m in self.metadata)

    @property
    def n_functions_trans_calls_filesystem(self) -> int:
        return sum(m.n_functions_trans_calls_filesystem for m in self.metadata)

    @property
    def n_functions_trans_calls_network(self) -> int:
        return sum(m.n_functions_trans_calls_network for m in self.metadata)

    @property
    def n_functions_trans_calls_external_io(self) -> int:
        return sum(m.n_functions_trans_calls_external_io for m in self.metadata)

    @property
    def n_functions_participates_in_cycle(self) -> int:
        return sum(m.n_functions_participates_in_cycle for m in self.metadata)

    @property
    def n_functions_has_unresolved_callees(self) -> int:
        return sum(m.n_functions_has_unresolved_callees for m in self.metadata)

    def _interproc_qualified_names(self, bool_field: str) -> list[str]:
        """Return sorted `module.qualified` names for functions whose
        `bool_field` interprocedural flag is True. Uses the same
        parallel tuples the planner uses, so the gold mirrors the
        planner's semantics exactly.

        The `trans_calls_external_io` pseudo-field is handled
        specially as the union of the three concrete trans-IO fields
        (matching `core.code_interproc.InterprocSemantics`).
        """
        out: list[str] = []
        for m in self.metadata:
            names = m.semantic_function_names
            if bool_field == "function_trans_calls_external_io":
                sp = m.function_trans_calls_subprocess
                fs = m.function_trans_calls_filesystem
                nw = m.function_trans_calls_network
                flags = tuple(
                    a or b or c for a, b, c in zip(sp, fs, nw)
                )
            else:
                flags = getattr(m, bool_field, ())
            for name, flag in zip(names, flags):
                if flag:
                    out.append(f"{m.module_name}.{name}")
        return sorted(out)

    def _semantic_qualified_names(self, bool_field: str) -> list[str]:
        """Return sorted `module.qualified` names for functions whose
        `bool_field` flag is True. Uses the same parallel tuples the
        planner uses, so the gold mirrors the planner's semantics
        exactly.

        The `calls_external_io` pseudo-field is handled specially: it
        is the union of the three concrete IO fields.
        """
        out: list[str] = []
        for m in self.metadata:
            names = m.semantic_function_names
            if bool_field == "function_calls_external_io":
                sp = m.function_calls_subprocess
                fs = m.function_calls_filesystem
                nw = m.function_calls_network
                flags = tuple(
                    a or b or c for a, b, c in zip(sp, fs, nw)
                )
            else:
                flags = getattr(m, bool_field, ())
            for name, flag in zip(names, flags):
                if flag:
                    out.append(f"{m.module_name}.{name}")
        return sorted(out)

    def most_imported_module(self) -> str | None:
        """The single most-frequent import across the corpus (by per-file
        presence). Tie-breaks on the Counter's insertion order, which
        matches the ledger's ingestion order."""
        c = self.all_imports
        if not c:
            return None
        top = c.most_common(1)[0]
        return top[0]

    def files_importing(self, target: str) -> list[str]:
        out = []
        for m in self.metadata:
            for imp in m.imports:
                if imp == target or imp.startswith(target + "."):
                    out.append(m.file_path)
                    break
        return sorted(out)

    def functions_returning_none(self) -> list[str]:
        """Returns 'module.func' strings, sorted."""
        out: list[str] = []
        for m in self.metadata:
            for n, ret in zip(m.function_names, m.function_returns_none):
                if ret:
                    out.append(f"{m.module_name}.{n}")
        return sorted(out)

    def top_file_by_functions(self) -> str | None:
        """Return the file path with max `n_functions`. Tie-break: the
        FIRST file (alphabetical) wins, to match the planner's
        deterministic insertion-order behaviour over the ledger."""
        if not self.metadata:
            return None
        m = max(self.metadata,
                 key=lambda x: (x.n_functions, -self.metadata.index(x)))
        return m.file_path

    def largest_file(self) -> str | None:
        """Return the file path with max `line_count`. Same tie-break."""
        if not self.metadata:
            return None
        m = max(self.metadata,
                 key=lambda x: (x.line_count, -self.metadata.index(x)))
        return m.file_path

    # ---- Question battery ------------------------------------------

    def _build_questions(self) -> None:
        import random
        rng = random.Random(self.seed)
        qs: list[NeedleQuestion] = []
        all_idx = tuple(range(self.n_files))

        # Count questions
        qs.append(NeedleQuestion(
            question="How many Python files are in the corpus?",
            gold=str(self.n_files),
            accept_any=(f"{self.n_files} files",
                        f"{self.n_files} python", str(self.n_files)),
            source_section=all_idx, kind="count_files",
        ))
        qs.append(NeedleQuestion(
            question="How many functions are defined in total in the corpus?",
            gold=str(self.n_functions_total),
            accept_any=(f"{self.n_functions_total} functions",
                        str(self.n_functions_total)),
            source_section=all_idx, kind="count_functions_total",
        ))
        qs.append(NeedleQuestion(
            question="How many classes are defined in the corpus?",
            gold=str(self.n_classes_total),
            accept_any=(f"{self.n_classes_total} classes",
                        str(self.n_classes_total)),
            source_section=all_idx, kind="count_classes_total",
        ))
        if self.n_test_files > 0:
            # Skip the question entirely if the corpus has no tests —
            # gold "0" is hard to score reliably (the digit "0" appears
            # in many incidental contexts).
            qs.append(NeedleQuestion(
                question="How many test files are in the corpus?",
                gold=str(self.n_test_files),
                accept_any=(f"{self.n_test_files} test files",
                            str(self.n_test_files)),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                      if m.is_test_file),
                kind="count_test_files",
            ))
        qs.append(NeedleQuestion(
            question="How many distinct modules are imported across the corpus?",
            gold=str(self.n_distinct_imports),
            accept_any=(f"{self.n_distinct_imports} distinct",
                        f"{self.n_distinct_imports} imports",
                        str(self.n_distinct_imports)),
            source_section=all_idx, kind="count_distinct_imports",
        ))

        # Listing: pick a popular import that appears in ≥ 2 files
        popular = [imp for imp, c in self.all_imports.most_common(20)
                   if c >= 2 and not imp.startswith("_")]
        if popular:
            target = rng.choice(popular[: min(5, len(popular))])
            files = self.files_importing(target)[: self.max_listing_size]
            base_names = sorted({os.path.basename(f) for f in files})
            qs.append(NeedleQuestion(
                question=f"List the Python files importing {target}.",
                gold=", ".join(base_names),
                accept_any=(),
                accept_all=tuple(base_names),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                      if m.file_path in files),
                kind="list_files_importing",
            ))

        # Listing: functions returning None
        none_funcs = self.functions_returning_none()
        if none_funcs:
            sample = none_funcs[: self.max_listing_size]
            base_names = sorted({n.rsplit(".", 1)[-1] for n in sample})
            qs.append(NeedleQuestion(
                question="List functions in the corpus that provably return None.",
                gold=", ".join(base_names),
                accept_any=(),
                accept_all=tuple(base_names),
                source_section=all_idx,
                kind="list_functions_returning_none",
            ))

        # Top: file with most functions
        top_fn_file = self.top_file_by_functions()
        if top_fn_file is not None:
            base = os.path.basename(top_fn_file)
            qs.append(NeedleQuestion(
                question="Which file has the most functions defined?",
                gold=base,
                accept_any=(top_fn_file, base),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                      if m.file_path == top_fn_file),
                kind="top_file_by_functions",
            ))

        # Top: largest file by lines
        big_file = self.largest_file()
        if big_file is not None:
            base = os.path.basename(big_file)
            qs.append(NeedleQuestion(
                question="What is the largest file by line count?",
                gold=base,
                accept_any=(big_file, base),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                      if m.file_path == big_file),
                kind="largest_file",
            ))

        # ---- Phase-23 additions ----------------------------------------

        # Count: methods total. Only asked if the corpus has ≥ 1 method
        # (otherwise "0 methods" is hard to score — the digit 0 appears
        # in many unrelated spans).
        if self.n_methods_total > 0:
            qs.append(NeedleQuestion(
                question="How many methods are defined in the corpus?",
                gold=str(self.n_methods_total),
                accept_any=(f"{self.n_methods_total} methods",
                            str(self.n_methods_total)),
                source_section=all_idx, kind="count_methods_total",
            ))

        # Count: files with docstrings.
        if self.n_files_with_docstrings > 0:
            n = self.n_files_with_docstrings
            qs.append(NeedleQuestion(
                question="How many files have docstrings?",
                gold=str(n),
                accept_any=(f"{n} files", str(n)),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                     if m.has_docstring),
                kind="count_files_with_docstrings",
            ))

        # Top-import (most frequently imported module). Skip if the
        # corpus has no imports at all (e.g. a pure stdlib module with
        # no cross-package imports).
        top_imp = self.most_imported_module()
        if top_imp is not None:
            qs.append(NeedleQuestion(
                question="Which module is imported most often across the corpus?",
                gold=top_imp,
                accept_any=(top_imp,),
                source_section=tuple(i for i, m in enumerate(self.metadata)
                                     if top_imp in m.imports),
                kind="most_imported_module",
            ))

        # ---- Phase-24 conservative-semantic questions -----------------
        #
        # Each (count_*, list_*) pair comes from one conservative
        # predicate populated by `core.code_semantics`. The gold is
        # the corpus's own computed count / list — it IS the reference
        # definition of "the conservative analyser flags this". That
        # means `direct-exact` is 100 % by construction on every
        # question here, which is EXACTLY the point: the substrate
        # makes the conservative analysis accessible without any LLM
        # reasoning, and the benchmark measures whether retrieval /
        # LLM-mediated paths can recover the same slice.
        self._append_semantic_questions(qs)

        # ---- Phase-25 interprocedural questions -----------------------
        self._append_interproc_questions(qs)

        self.questions = qs

    # ---- Semantic question emitter ---------------------------------

    def _append_semantic_questions(self, qs: list[NeedleQuestion]) -> None:
        """Append the Phase-24 conservative-semantic question battery
        to `qs`. Questions where the gold set would be empty are
        SKIPPED (scoring "0 functions" against a blob of source where
        the digit "0" might appear by chance is unreliable, same
        rationale as Phase 23)."""
        def _src_section_for(predicate_field: str) -> tuple[int, ...]:
            out: list[int] = []
            for i, m in enumerate(self.metadata):
                if predicate_field == "function_calls_external_io":
                    sp = m.function_calls_subprocess
                    fs = m.function_calls_filesystem
                    nw = m.function_calls_network
                    if any(a or b or c for a, b, c in zip(sp, fs, nw)):
                        out.append(i)
                else:
                    if any(getattr(m, predicate_field, ())):
                        out.append(i)
            return tuple(out)

        def _listing(names: list[str]) -> tuple[str, ...]:
            """Trim to `max_listing_size` and reduce to bare short
            names (`module.Foo.bar` → `bar`) so `accept_all`'s
            substring-rule can find them regardless of how the answer
            is rendered.
            """
            sample = names[: self.max_listing_size]
            bare = sorted({n.rsplit(".", 1)[-1] for n in sample})
            return tuple(bare)

        # Each entry: (count_kind, list_kind, predicate_field,
        #              count_attr, question_count, question_list, noun).
        specs = [
            ("count_may_raise", "list_may_raise",
             "function_may_raise", self.n_functions_may_raise,
             "How many functions in the corpus may raise an exception?",
             "List functions that may raise exceptions.",
             "may raise"),
            ("count_is_recursive", "list_is_recursive",
             "function_is_recursive", self.n_functions_is_recursive,
             "How many functions are recursive?",
             "List recursive functions.",
             "recursive"),
            ("count_may_write_global", "list_may_write_global",
             "function_may_write_global", self.n_functions_may_write_global,
             "How many functions may write to global state?",
             "List functions that may write to global state.",
             "may write global state"),
            ("count_calls_subprocess", "list_calls_subprocess",
             "function_calls_subprocess", self.n_functions_calls_subprocess,
             "How many functions call subprocess?",
             "List functions that call subprocess.",
             "call subprocess"),
            ("count_calls_filesystem", "list_calls_filesystem",
             "function_calls_filesystem", self.n_functions_calls_filesystem,
             "How many functions touch the filesystem?",
             "List functions that touch the filesystem.",
             "touch the filesystem"),
            ("count_calls_network", "list_calls_network",
             "function_calls_network", self.n_functions_calls_network,
             "How many functions make network calls?",
             "List functions that make network calls.",
             "make network calls"),
            # Single-kind union: "external side effects" — we only emit
            # the count question (listing the union would double-count
            # against the concrete-IO listings already present). Counts
            # are independently measurable.
            ("count_calls_external_io", None,
             "function_calls_external_io", self.n_functions_calls_external_io,
             "How many functions have external side effects?",
             None,
             "have external side effects"),
        ]
        for (ck, lk, field, n, q_count, q_list, _noun) in specs:
            if n <= 0:
                continue
            src = _src_section_for(field)
            qs.append(NeedleQuestion(
                question=q_count, gold=str(n),
                accept_any=(f"{n} functions", str(n)),
                source_section=src, kind=ck,
            ))
            if lk is not None:
                names = self._semantic_qualified_names(field)
                if not names:
                    continue
                bare = _listing(names)
                if not bare:
                    continue
                qs.append(NeedleQuestion(
                    question=q_list,
                    gold=", ".join(bare),
                    accept_any=(),
                    accept_all=bare,
                    source_section=src,
                    kind=lk,
                ))

    # ---- Interprocedural question emitter (Phase 25) ----------------

    def _append_interproc_questions(self, qs: list[NeedleQuestion]) -> None:
        """Append the Phase-25 interprocedural question battery. Gold
        answers come from the same parallel tuples the planner
        consults; `direct-exact` is 1.0 on matched questions by
        construction. Retrieval and wrap-LLM paths are the
        interesting experimental conditions.

        Skips any question whose support is empty on this corpus
        (same rationale as the Phase-23 / 24 emitters: scoring "0
        functions" against random source-body spans is unreliable).
        """

        def _src_section_for(predicate_field: str) -> tuple[int, ...]:
            out: list[int] = []
            for i, m in enumerate(self.metadata):
                if predicate_field == "function_trans_calls_external_io":
                    sp = m.function_trans_calls_subprocess
                    fs = m.function_trans_calls_filesystem
                    nw = m.function_trans_calls_network
                    if any(a or b or c for a, b, c in zip(sp, fs, nw)):
                        out.append(i)
                else:
                    if any(getattr(m, predicate_field, ())):
                        out.append(i)
            return tuple(out)

        def _listing(names: list[str]) -> tuple[str, ...]:
            """Same shape as the Phase-24 emitter: trim + dedupe bare
            names so `accept_all` can find them regardless of the
            answer's rendering style."""
            sample = names[: self.max_listing_size]
            bare = sorted({n.rsplit(".", 1)[-1] for n in sample})
            return tuple(bare)

        # Each entry: (count_kind, list_kind, predicate_field,
        #              count_attr, count_question, list_question).
        specs = [
            ("count_trans_may_raise", "list_trans_may_raise",
             "function_trans_may_raise", self.n_functions_trans_may_raise,
             "How many functions may transitively raise an exception "
             "through a helper?",
             "List functions that may transitively raise an exception."),
            ("count_trans_may_write_global",
             "list_trans_may_write_global",
             "function_trans_may_write_global",
             self.n_functions_trans_may_write_global,
             "How many functions may transitively mutate module "
             "globals through a helper?",
             "List functions that may transitively mutate module globals."),
            ("count_trans_calls_subprocess",
             "list_trans_calls_subprocess",
             "function_trans_calls_subprocess",
             self.n_functions_trans_calls_subprocess,
             "How many functions transitively invoke subprocess "
             "through a helper?",
             "List functions that transitively invoke subprocess."),
            ("count_trans_calls_filesystem",
             "list_trans_calls_filesystem",
             "function_trans_calls_filesystem",
             self.n_functions_trans_calls_filesystem,
             "How many functions transitively touch the filesystem "
             "through a helper?",
             "List functions that transitively touch the filesystem."),
            ("count_trans_calls_network",
             "list_trans_calls_network",
             "function_trans_calls_network",
             self.n_functions_trans_calls_network,
             "How many functions transitively make network calls "
             "through a helper?",
             "List functions that transitively make network calls."),
            # External-IO union: count-only — listing would duplicate
            # the concrete-IO listings already present.
            ("count_trans_calls_external_io", None,
             "function_trans_calls_external_io",
             self.n_functions_trans_calls_external_io,
             "How many functions have transitive external side effects "
             "through a helper?",
             None),
            ("count_participates_in_cycle",
             "list_participates_in_cycle",
             "function_participates_in_cycle",
             self.n_functions_participates_in_cycle,
             "How many functions participate in a recursion cycle?",
             "List functions that participate in a recursion cycle."),
            # Unresolved callees: count-only. Listing every function
            # with unresolved calls is noisy and not particularly
            # interesting — the count is the transparency signal.
            ("count_has_unresolved_callees", None,
             "function_has_unresolved_callees",
             self.n_functions_has_unresolved_callees,
             "How many functions call into unresolved helpers?",
             None),
        ]
        for (ck, lk, field, n, q_count, q_list) in specs:
            if n <= 0:
                continue
            src = _src_section_for(field)
            qs.append(NeedleQuestion(
                question=q_count, gold=str(n),
                accept_any=(f"{n} functions", str(n)),
                source_section=src, kind=ck,
            ))
            if lk is not None:
                names = self._interproc_qualified_names(field)
                if not names:
                    continue
                bare = _listing(names)
                if not bare:
                    continue
                qs.append(NeedleQuestion(
                    question=q_list, gold=", ".join(bare),
                    accept_any=(), accept_all=bare,
                    source_section=src, kind=lk,
                ))

    # ---- Convenience -----------------------------------------------

    @property
    def total_lines(self) -> int:
        return sum(m.line_count for m in self.metadata)

    def chunks(self) -> list[str]:
        """Return file source bodies (one per file)."""
        out = []
        for fp in self.files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    out.append(f.read())
            except OSError:
                out.append("")
        return out
