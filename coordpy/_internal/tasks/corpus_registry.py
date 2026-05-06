"""Multi-corpus registry for Phase-23 external validity benchmarking.

Phase 22 established the exact-memory + exact-planning pipeline on a
single real Python codebase (`coordpy/_internal/core/`). Phase 23 needs to
exercise the same pipeline across *several* real codebases of
different family, size, and metadata coverage. This module provides
the reusable loader.

Design goals:

  * **Generic**. A corpus is nothing more than a local directory root
    plus a couple of ingest knobs (`max_files`, `max_chars_per_file`).
    Anything the `CodeIndexer` can walk is a valid corpus.
  * **Deterministic**. Two runs over the same root produce the same
    corpus (same ordering, same gold answers). Relies on
    `CodeIndexer`'s sorted directory walk and `PythonCorpus`'s
    seed-controlled question generation.
  * **Transparent**. Every corpus reports its ingestion coverage
    (`IngestionStats`) so the benchmark can attribute accuracy
    differences to corpus-level parseability / metadata richness.

Typical usage:

    reg = CorpusRegistry([
        CorpusSpec(name="vision-core",   root="vision_mvp/core"),
        CorpusSpec(name="vision-tasks",  root="vision_mvp/tasks"),
        CorpusSpec(name="click",         root="<path-to>/click",
                   max_files=24),
    ])
    for entry in reg.build():
        print(entry.name, entry.coverage.metadata_completeness,
              len(entry.corpus.questions))
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterable

from ..core.code_index import CodeIndexer, IngestionStats
from ..core.context_ledger import ContextLedger, hash_embedding
from .python_corpus import PythonCorpus


# =============================================================================
# Specification
# =============================================================================


@dataclass(frozen=True)
class CorpusSpec:
    """Declarative description of a benchmark corpus.

    Fields:
      name: short identifier used in reports (e.g. "click", "json-stdlib").
      root: absolute or repo-relative directory to ingest.
      family: coarse bucket for the writeup ("research", "utility",
              "stdlib", "tests", "third-party-cli", ...). Informational
              only — does not affect ingestion.
      max_files: if set, truncate the walk after this many files. Used
              for scaling sweeps.
      max_chars_per_file: per-file size cap (defaults to 64 KiB, matching
              `CodeIndexer`'s default).
      seed: per-corpus random seed. `PythonCorpus` uses this to choose
              which popular import to quiz about in the `list_files_importing`
              question.
    """

    name: str
    root: str
    family: str = "unknown"
    max_files: int | None = None
    max_chars_per_file: int = 64_000
    seed: int = 23


# =============================================================================
# Built entry
# =============================================================================


@dataclass
class CorpusEntry:
    """One built corpus with its coverage accounting."""

    spec: CorpusSpec
    corpus: PythonCorpus
    coverage: IngestionStats
    build_seconds: float
    indexer_coverage_source: str = ""    # notes which indexer produced coverage

    @property
    def name(self) -> str:
        return self.spec.name

    def summary(self) -> dict:
        cov = self.coverage.as_dict()
        qs = self.corpus.questions
        by_kind: dict[str, int] = {}
        for q in qs:
            by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
        return {
            "name": self.spec.name,
            "root": self.spec.root,
            "family": self.spec.family,
            "max_files": self.spec.max_files,
            "n_files": self.corpus.n_files,
            "total_lines": self.corpus.total_lines,
            "n_functions_total": self.corpus.n_functions_total,
            "n_classes_total": self.corpus.n_classes_total,
            "n_methods_total": self.corpus.n_methods_total,
            "n_distinct_imports": self.corpus.n_distinct_imports,
            "n_files_with_docstrings": self.corpus.n_files_with_docstrings,
            "most_imported_module": self.corpus.most_imported_module(),
            "n_questions": len(qs),
            "questions_by_kind": by_kind,
            "build_seconds": round(self.build_seconds, 3),
            "coverage": cov,
        }


# =============================================================================
# Registry
# =============================================================================


@dataclass
class CorpusRegistry:
    """A list of corpus specs + deterministic construction.

    `build(only=)` constructs every spec in order. Corpus construction
    walks the directory, parses every file, and fills in gold answers.
    Ingestion cost is reported per corpus.
    """

    specs: list[CorpusSpec] = field(default_factory=list)

    def build(self, only: Iterable[str] | None = None) -> list[CorpusEntry]:
        wanted = set(only) if only is not None else None
        entries: list[CorpusEntry] = []
        for spec in self.specs:
            if wanted is not None and spec.name not in wanted:
                continue
            t0 = time.time()
            corpus = PythonCorpus(
                root=spec.root, max_files=spec.max_files,
                max_chars_per_file=spec.max_chars_per_file, seed=spec.seed,
            )
            corpus.build()
            # Coverage accounting: run a throwaway CodeIndexer against a
            # minimal throwaway ledger. This is deterministic AND cheap
            # (we do the same directory walk + parse that `PythonCorpus.build`
            # did, so there's no blow-up in cost). We keep the ledger
            # here so any ledger-level rejection (capacity) shows up in
            # stats; but we do NOT reuse this ledger elsewhere — the
            # benchmark rebuilds ledgers per condition anyway.
            throwaway_ledger = ContextLedger(
                embed_dim=16,
                embed_fn=lambda t: hash_embedding(t, dim=16),
                max_artifacts=None,
                max_artifact_chars=spec.max_chars_per_file,
            )
            cov_indexer = CodeIndexer(
                root=spec.root, max_files=spec.max_files,
                max_chars_per_file=spec.max_chars_per_file,
            )
            cov_indexer.index_into(throwaway_ledger)
            coverage = cov_indexer.stats
            entries.append(CorpusEntry(
                spec=spec, corpus=corpus, coverage=coverage,
                build_seconds=time.time() - t0,
                indexer_coverage_source="throwaway_ledger",
            ))
        return entries

    def summary(self, entries: list[CorpusEntry] | None = None) -> list[dict]:
        """Serialisable per-corpus summary. If `entries` is None, builds
        from spec."""
        es = entries if entries is not None else self.build()
        return [e.summary() for e in es]

    # ---- discovery helpers -------------------------------------------------

    def add(self, spec: CorpusSpec) -> None:
        self.specs.append(spec)

    def __len__(self) -> int:
        return len(self.specs)


# =============================================================================
# Default Phase-23 registry
# =============================================================================


def _maybe_add(reg: CorpusRegistry, spec: CorpusSpec) -> bool:
    """Append `spec` iff its root directory exists on disk and contains
    at least one .py file. Returns True iff appended."""
    if not os.path.isdir(spec.root):
        return False
    for _d, _ds, fs in os.walk(spec.root):
        if any(f.endswith(".py") for f in fs):
            reg.add(spec)
            return True
    return False


def default_phase23_registry(
    repo_root: str | None = None,
    extra_roots: Iterable[str] | None = None,
) -> CorpusRegistry:
    """Construct the registry used by the Phase-23 benchmark.

    The base set is all-local to the repo for reproducibility:
      - `vision_mvp/core`   (research/framework code, ~108 .py files)
      - `vision_mvp/tasks`  (corpus-generator code, ~16 .py files)
      - `vision_mvp/tests`  (test-file-dominated, ~54 .py files)
      - `vision_mvp/experiments` (experiment scripts, ~27 .py files)

    Additional roots can be passed via `extra_roots` — these are
    treated as OUTSIDE the repo (e.g. a local third-party library
    path) and named by their basename. We only add them if they
    exist AND contain at least one .py file.
    """
    if repo_root is None:
        # Assume we live inside the repo; climb to the repo root by
        # walking two levels up from this module.
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    reg = CorpusRegistry()
    for name, rel, fam in [
        ("vision-core",        "vision_mvp/core",        "research-framework"),
        ("vision-tasks",       "vision_mvp/tasks",       "research-utility"),
        ("vision-tests",       "vision_mvp/tests",       "test-suite"),
        ("vision-experiments", "vision_mvp/experiments", "research-scripts"),
    ]:
        _maybe_add(reg, CorpusSpec(
            name=name, root=os.path.join(repo_root, rel), family=fam,
        ))
    if extra_roots:
        for path in extra_roots:
            if not os.path.isdir(path):
                continue
            _maybe_add(reg, CorpusSpec(
                name=os.path.basename(os.path.normpath(path)) or "external",
                root=path, family="third-party-external",
            ))
    return reg
