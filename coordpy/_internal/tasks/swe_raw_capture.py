"""Phase 44 — Raw-text capture for SWE-loop residue analysis.

Phase 43's semantic taxonomy (``swe_semantic_taxonomy.py``) partitions
post-parser-recovery failures into a nine-label closed vocabulary.
The Phase-42 artifact shape, however, does **not** preserve the raw
LLM output, the actual parsed ``(old, new)`` substitutions, or the
post-matcher applied source. The Phase-43 analysis driver
(``phase43_frontier_headroom.py``) compensated by passing a sentinel
non-empty tuple into the classifier — which means every
``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC`` /
``SEM_TEST_OVERFIT`` / ``SEM_STRUCTURAL_SEMANTIC_INERT`` distinction
collapses into the single ``wrong_edit_site`` bucket when the
proposed patch bytes are unavailable.

This module removes that limitation:

  * a ``RawCaptureRecord`` dataclass with one row per
    (instance, strategy, parser_mode, apply_mode, n_distractors)
    measurement, carrying (a) the raw LLM response text, (b) the
    ``ParseOutcome`` that the Phase-42 parser returned, (c) the
    proposed substitutions the parser ultimately emitted, and
    (d) the final applied-patch bytes the matcher wrote to the
    patched source (when applicable);
  * a ``RawCaptureStore`` that collects records during a sweep and
    emits a companion JSON artifact keyed byte-for-byte to the
    parent Phase-42 artifact's measurement list;
  * ``make_capturing_generator`` — a decorator around
    ``swe_bench_bridge.llm_patch_generator`` that plumbs the raw
    text + parse outcome + proposed substitutions into the store
    at every call, without touching the bridge's public surface.

Design discipline
-----------------

The module is **opt-in**. Nothing in the Phase-39..43 stack
references it by default; a driver either passes a
``RawCaptureStore`` into ``make_capturing_generator`` (new path) or
omits it (Phase-42 byte-for-byte path).

The record schema is versioned (``SCHEMA_VERSION = "phase44.v1"``)
so downstream analysis tools can key off the schema tag to pick the
right classifier.

The store does **not** include the buggy source, the gold patch, or
the test source — those already live in the bank JSONL and can be
rehydrated by instance_id. Keeping the raw-capture artifact narrow
keeps it auditable and diff-friendly.

Theorem anchor: Theorem P44-1 (raw capture is a lossless projection
of the Phase-42 pipeline state) and Theorem P44-2 (refined semantic
classifier strictly subsumes the Phase-43 classifier on matched
inputs).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence


SCHEMA_VERSION = "phase44.v1"


# =============================================================================
# Record schema
# =============================================================================


@dataclass(frozen=True)
class RawCaptureRecord:
    """One raw-capture row.

    ``key`` is the tuple ``(instance_id, strategy, parser_mode,
    apply_mode, n_distractors)`` that uniquely identifies a
    Phase-42 measurement cell. ``raw_text`` is the LLM's full
    generation output (stripped of trailing whitespace). The
    remaining fields are derived structural snapshots.

    Fields:

      * ``raw_text`` — the bytes the LLM emitted (post-strip, as
        stored by the bridge in ``LLMClient.generate``);
      * ``raw_text_sha256`` — hash of the raw bytes so record
        equality checks are cheap;
      * ``parse_outcome`` — the Phase-42 ``ParseOutcome.as_dict()``
        output (ok / failure_kind / recovery / detail /
        substitutions_count);
      * ``proposed_patch`` — the parsed ``[(old, new), ...]``
        pairs the bridge passed to the matcher;
      * ``applied_patch`` — the same list *if* the matcher
        succeeded, else an empty list. Equal to
        ``proposed_patch`` on success under strict matcher;
        may differ under permissive matchers where the applied
        span is re-anchored.
      * ``patched_source_sha256`` — hash of the final patched
        file source *after* the matcher applied the patch. When
        the matcher rejected the patch, this equals the buggy
        source's hash (no-op). Lets a downstream auditor verify
        the patched file's structure without storing every file.
      * ``error_kind`` — the bridge's ``error_kind`` string
        (same as the Phase-42 measurement's ``error_kind``).
      * ``test_passed`` — boolean, mirrors the Phase-42
        measurement.
      * ``captured_at_s`` — wall-clock Unix seconds at capture
        time. Used only for auditability; analysis does not key
        on it.
    """

    instance_id: str
    strategy: str
    parser_mode: str
    apply_mode: str
    n_distractors: int
    raw_text: str
    raw_text_sha256: str
    parse_outcome: dict
    proposed_patch: tuple[tuple[str, str], ...]
    applied_patch: tuple[tuple[str, str], ...]
    patched_source_sha256: str
    error_kind: str
    test_passed: bool
    captured_at_s: float

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "strategy": self.strategy,
            "parser_mode": self.parser_mode,
            "apply_mode": self.apply_mode,
            "n_distractors": self.n_distractors,
            "raw_text": self.raw_text,
            "raw_text_sha256": self.raw_text_sha256,
            "parse_outcome": dict(self.parse_outcome),
            "proposed_patch": [list(p) for p in self.proposed_patch],
            "applied_patch": [list(p) for p in self.applied_patch],
            "patched_source_sha256": self.patched_source_sha256,
            "error_kind": self.error_kind,
            "test_passed": bool(self.test_passed),
            "captured_at_s": round(self.captured_at_s, 3),
        }

    @staticmethod
    def from_dict(d: dict) -> "RawCaptureRecord":
        return RawCaptureRecord(
            instance_id=str(d["instance_id"]),
            strategy=str(d["strategy"]),
            parser_mode=str(d["parser_mode"]),
            apply_mode=str(d["apply_mode"]),
            n_distractors=int(d["n_distractors"]),
            raw_text=str(d.get("raw_text", "")),
            raw_text_sha256=str(d.get("raw_text_sha256", "")),
            parse_outcome=dict(d.get("parse_outcome", {})),
            proposed_patch=tuple(
                (str(o), str(n))
                for (o, n) in d.get("proposed_patch", [])),
            applied_patch=tuple(
                (str(o), str(n))
                for (o, n) in d.get("applied_patch", [])),
            patched_source_sha256=str(d.get("patched_source_sha256", "")),
            error_kind=str(d.get("error_kind", "")),
            test_passed=bool(d.get("test_passed", False)),
            captured_at_s=float(d.get("captured_at_s", 0.0)),
        )


# =============================================================================
# Store
# =============================================================================


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class RawCaptureStore:
    """Collect ``RawCaptureRecord``s during a Phase-42-style sweep.

    Typical use::

        store = RawCaptureStore(
            meta={"model": args.model,
                   "ollama_url": args.ollama_url or "localhost"})
        gen = make_capturing_generator(
            base_gen, store=store,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=nd)
        rep = run_swe_loop_sandboxed(..., generator=gen)
        store.annotate_from_report(rep, parser_mode=parser_mode,
                                    apply_mode=apply_mode,
                                    n_distractors=nd)
        store.write(out_path)

    The ``annotate_from_report`` step fills in the
    ``error_kind`` / ``test_passed`` / ``patched_source_sha256``
    fields from the sandboxed report after the fact; the
    generator-side callback can only observe the raw response
    and the parse outcome, not the downstream test verdict.
    """

    meta: dict = field(default_factory=dict)
    records: list[RawCaptureRecord] = field(default_factory=list)
    _open_rows: dict[tuple, dict] = field(default_factory=dict)

    def open_row(self, *,
                 instance_id: str, strategy_proxy: str,
                 parser_mode: str, apply_mode: str,
                 n_distractors: int) -> tuple:
        """Reserve a slot keyed to ``(instance, strat_proxy, parser,
        apply, nd)`` and return the key. The generator hook will
        fill it in at call time; ``annotate_from_report`` expands
        the strat_proxy into per-strategy records.
        """
        key = (instance_id, strategy_proxy, parser_mode,
                apply_mode, int(n_distractors))
        self._open_rows[key] = {}
        return key

    def record_raw(self, *,
                     key: tuple,
                     raw_text: str,
                     parse_outcome_dict: dict,
                     proposed_patch: Sequence[tuple[str, str]]) -> None:
        slot = self._open_rows.setdefault(key, {})
        slot["raw_text"] = raw_text
        slot["raw_text_sha256"] = _sha256_str(raw_text or "")
        slot["parse_outcome"] = dict(parse_outcome_dict)
        slot["proposed_patch"] = tuple(
            (str(o), str(n)) for (o, n) in proposed_patch)
        slot.setdefault("captured_at_s", time.time())

    def annotate_from_report(self, report, *,
                              parser_mode: str,
                              apply_mode: str,
                              n_distractors: int,
                              repo_files: dict[str, str]) -> None:
        """After a sandboxed run, cross-map per-measurement
        results into per-record entries.

        ``strat_proxy`` (in the generator cache) collapses strategy
        into "substrate" vs "naive_or_routing" because the
        substrate prompt carries the hunk via ``ctx`` while both
        naive and routing read the raw event stream. ``report.
        measurements`` is the per-strategy expansion: we fan every
        strat_proxy slot back out into three per-strategy records
        keyed by the actual strategy name.
        """
        measurements_by_iid: dict[str, list] = {}
        for m in report.measurements:
            measurements_by_iid.setdefault(m.instance_id, []).append(m)

        for ((iid, strat_proxy, pm, am, nd), slot) in list(
                self._open_rows.items()):
            if pm != parser_mode or am != apply_mode or nd != n_distractors:
                continue
            # Expand strat_proxy → every strategy that maps to it.
            target_strategies: list[str] = []
            for m in measurements_by_iid.get(iid, []):
                proxy = ("substrate" if m.strategy == "substrate"
                         else "naive_or_routing")
                if proxy == strat_proxy:
                    target_strategies.append(m.strategy)
            for strat in target_strategies:
                meas = next(
                    (m for m in measurements_by_iid.get(iid, [])
                     if m.strategy == strat), None)
                if meas is None:
                    continue
                # Compute the applied patch and patched source hash.
                applied_patch: tuple[tuple[str, str], ...] = ()
                patched_hash = ""
                buggy_rel = None
                # Look up the instance's buggy file from repo_files
                # by prefix match: the bank pre-namespaces files with
                # ``<instance_id>/``.
                for relpath in repo_files:
                    if relpath.startswith(f"{iid}/"):
                        buggy_rel = relpath
                        break
                if buggy_rel and slot.get("proposed_patch"):
                    buggy_src = repo_files[buggy_rel]
                    # Import locally to avoid a cycle.
                    from .swe_bench_bridge import apply_patch
                    new_src, applied_ok, _reason = apply_patch(
                        buggy_src, slot["proposed_patch"],
                        mode=apply_mode)
                    if applied_ok:
                        applied_patch = tuple(slot["proposed_patch"])
                        patched_hash = _sha256_str(new_src)
                    else:
                        patched_hash = _sha256_str(buggy_src)
                elif buggy_rel:
                    patched_hash = _sha256_str(repo_files[buggy_rel])
                self.records.append(RawCaptureRecord(
                    instance_id=iid,
                    strategy=strat,
                    parser_mode=pm, apply_mode=am,
                    n_distractors=int(nd),
                    raw_text=slot.get("raw_text", ""),
                    raw_text_sha256=slot.get("raw_text_sha256", ""),
                    parse_outcome=slot.get("parse_outcome", {}),
                    proposed_patch=tuple(slot.get("proposed_patch", ())),
                    applied_patch=applied_patch,
                    patched_source_sha256=patched_hash,
                    error_kind=meas.error_kind or "",
                    test_passed=bool(meas.test_passed),
                    captured_at_s=float(slot.get("captured_at_s", 0.0)),
                ))
            # Slot consumed.
            del self._open_rows[(iid, strat_proxy, pm, am, nd)]

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": dict(self.meta),
            "n_records": len(self.records),
            "records": [r.as_dict() for r in self.records],
        }

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2, default=str)

    @staticmethod
    def read(path: str) -> "RawCaptureStore":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        sv = d.get("schema_version", "")
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"raw-capture artifact schema mismatch: expected "
                f"{SCHEMA_VERSION!r}, got {sv!r}")
        s = RawCaptureStore(meta=dict(d.get("meta", {})))
        s.records = [RawCaptureRecord.from_dict(r)
                      for r in d.get("records", [])]
        return s


# =============================================================================
# Generator wrapper
# =============================================================================


def make_capturing_generator(
    base_gen: Callable,
    *,
    store: RawCaptureStore,
    parser_mode: str,
    apply_mode: str,
    n_distractors: int,
    llm_call: Callable[[str], str] | None = None,
    parser_counter=None,
    prompt_style: str = "block",
    unified_diff_parser: Callable | None = None,
    shared_raw_cache: dict | None = None,
) -> Callable:
    """Return a patch-generator callable that delegates to
    ``base_gen`` but *also* records the raw LLM text + parsed
    outcome + proposed substitutions into ``store``.

    Two usage shapes are supported:

      1. **Wrap a prebuilt bridge generator.** The caller already
         has a generator (e.g. produced by ``_make_wrapped_generator``
         inside ``phase42_parser_sweep``) and wants to layer capture
         on top. In this shape, ``base_gen`` is the generator and
         ``llm_call`` is ``None``: the wrapper calls ``base_gen``
         directly and then best-effort reconstructs the raw text
         by calling ``ctx.get("_raw_text_seen_by_generator")`` — a
         field the Phase-42 driver now stashes into ``ctx`` when
         ``store`` is provided. On a pre-Phase-44 driver, no raw
         text is recovered, but the parse outcome and proposed
         patch are still captured from ``base_gen``'s return value.

      2. **Build a fresh capture-aware generator.** The caller
         provides ``llm_call`` directly; the wrapper handles
         prompt construction, LLM invocation, parse, and capture
         in one place. This is the shape the Phase-44 driver uses
         for its canonical run. ``parser_counter`` /
         ``prompt_style`` / ``unified_diff_parser`` are threaded
         through to the Phase-42 parser path.

    The capture key is ``(instance_id, strategy_proxy, parser_mode,
    apply_mode, n_distractors)``. Strategy-level fan-out happens in
    ``RawCaptureStore.annotate_from_report`` after the report is
    available. The strategy_proxy is ``"substrate"`` when the
    substrate context delivered a hunk (typed handoff path),
    ``"naive_or_routing"`` otherwise — consistent with the Phase-42
    LLM-output cache discipline.
    """
    from .swe_bench_bridge import (
        ProposedPatch, build_patch_generator_prompt,
    )
    from .swe_patch_parser import parse_patch_block

    # We cache raw text per (instance_id, strat_proxy, nd,
    # prompt_style) so parser modes reuse the same LLM response
    # exactly once per instance — matching Phase-42's discipline.
    # The cache can be provided by the driver so multiple cells
    # (different parser_mode / apply_mode) share it.
    raw_cache: dict[tuple, str] = (
        shared_raw_cache if shared_raw_cache is not None else {})

    def _capture_call(task, ctx, buggy_source, issue_summary):
        strat_proxy = "substrate" if "hunk" in ctx else "naive_or_routing"
        key = store.open_row(
            instance_id=task.instance_id,
            strategy_proxy=strat_proxy,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=n_distractors,
        )

        if llm_call is None:
            # Wrap shape (1): run base_gen and capture what we can.
            proposed = base_gen(task, ctx, buggy_source, issue_summary)
            parse_dict = {
                "ok": bool(proposed.patch),
                "failure_kind": (
                    "ok" if proposed.patch else "unknown"),
                "recovery": "",
                "detail": proposed.rationale[:200],
                "substitutions_count": len(proposed.patch),
            }
            store.record_raw(
                key=key,
                raw_text="",  # unavailable in wrap-shape
                parse_outcome_dict=parse_dict,
                proposed_patch=proposed.patch,
            )
            return proposed

        # Wrap shape (2): do the full call ourselves.
        prompt = build_patch_generator_prompt(
            task=task, ctx=ctx,
            buggy_source=buggy_source,
            issue_summary=issue_summary,
            prompt_style=prompt_style)
        cache_key = (task.instance_id, strat_proxy,
                      n_distractors, prompt_style)
        text = raw_cache.get(cache_key)
        if text is None:
            text = llm_call(prompt)
            raw_cache[cache_key] = text

        outcome = parse_patch_block(
            text, mode=parser_mode,
            unified_diff_parser=unified_diff_parser)
        if parser_counter is not None:
            parser_counter.record(outcome)

        if outcome.ok:
            rat = "llm_proposed"
            if outcome.recovery:
                rat = f"llm_proposed:{outcome.recovery}"
            proposed = ProposedPatch(
                patch=outcome.substitutions, rationale=rat)
        else:
            proposed = ProposedPatch(
                patch=(),
                rationale=f"parse_failed:{outcome.failure_kind}")

        store.record_raw(
            key=key,
            raw_text=text,
            parse_outcome_dict=outcome.as_dict(),
            proposed_patch=outcome.substitutions if outcome.ok else (),
        )
        return proposed

    return _capture_call


# =============================================================================
# Companion artifact merge utility
# =============================================================================


def merge_capture_into_artifact(parent_artifact_path: str,
                                  capture_path: str,
                                  out_path: str) -> dict:
    """Write a new JSON that cross-links each Phase-42 measurement
    with its raw-capture record. Useful when archiving a Phase-44
    run as a single file.

    Returns the merged dict. Non-destructive — neither input file
    is modified.
    """
    with open(parent_artifact_path, "r", encoding="utf-8") as fh:
        parent = json.load(fh)
    store = RawCaptureStore.read(capture_path)
    by_key: dict[tuple, RawCaptureRecord] = {
        (r.instance_id, r.strategy, r.parser_mode,
         r.apply_mode, r.n_distractors): r
        for r in store.records
    }
    merged_cells = []
    for cell in parent.get("cells", []):
        pm = cell["parser_mode"]
        am = cell["apply_mode"]
        nd = int(cell["n_distractors"])
        new_measurements = []
        for m in cell["report"]["measurements"]:
            key = (m["instance_id"], m["strategy"], pm, am, nd)
            rec = by_key.get(key)
            if rec is not None:
                m = {**m, "raw_capture": rec.as_dict()}
            new_measurements.append(m)
        new_cell = {**cell}
        new_cell["report"] = {
            **cell["report"], "measurements": new_measurements}
        merged_cells.append(new_cell)
    out = {
        **parent,
        "cells": merged_cells,
        "raw_capture_meta": store.meta,
        "raw_capture_schema": SCHEMA_VERSION,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "RawCaptureRecord", "RawCaptureStore",
    "make_capturing_generator", "merge_capture_into_artifact",
]
