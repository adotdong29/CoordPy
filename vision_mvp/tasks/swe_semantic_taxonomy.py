"""Phase 43 — Semantic failure taxonomy for post-parser-recovery SWE loops.

Phase 42 closed the *format* attribution layer (parser-compliance) and
the *matcher* attribution layer (byte-fidelity). On a coder-finetuned
14B model the robust parser lifted pass@1 from 1.8 % → 93.0 %, and the
remaining four failures per strategy did not respond to matcher
permissiveness either (Theorem P41-3 / P42-3 null-regress sets). That
residue is the programme's first *purely semantic* failure surface on
the real SWE loop: the generator's output is well-formed code that
applies cleanly against the buggy source but still fails the hidden
test.

This module classifies each such failure into one of a small closed
vocabulary so the Phase-43 attribution table has an honest account of
*what semantic mistake* remained. The labels are structural (derived
from the relationship between the LLM's proposed patch, the gold
patch, and the test outcome) — not subjective.

Label vocabulary (mutually exclusive, exhaustive over the failure
space given the Phase-42 error_kind axis):

    SEM_OK                       — test passed; no semantic failure.
    SEM_PARSE_FAIL               — parser returned no substitutions
                                     (handled entirely by Phase-42
                                     axis; kept here for completeness
                                     so the taxonomy is a total cover).
    SEM_WRONG_EDIT_SITE          — patch applies but matches a
                                     different source region than any
                                     hunk in the gold patch. Usually
                                     a hallucinated anchor.
    SEM_RIGHT_SITE_WRONG_LOGIC   — patch matches the same OLD window
                                     as a gold hunk but the NEW block
                                     differs from gold's NEW in a
                                     behaviour-changing way.
    SEM_INCOMPLETE_MULTI_HUNK    — gold patch has ≥ 2 hunks; the
                                     proposed patch covers strictly
                                     fewer of them. Common on
                                     multi-method class fixes.
    SEM_TEST_OVERFIT             — applies cleanly, matches the right
                                     site, but fails an assertion on
                                     a *subset* of test inputs. The
                                     generator fixed the example in
                                     the prompt but not the edge cases.
    SEM_STRUCTURAL_SEMANTIC_INERT — applies cleanly, syntactically
                                     valid, but throws at runtime
                                     (test_exception) because of a
                                     type-shape or missing-attribute
                                     mismatch the fix introduced.
    SEM_SYNTAX_INVALID           — proposed NEW block is syntactically
                                     broken Python; the patched file
                                     won't even compile.
    SEM_NO_MATCH_RESIDUAL        — post-recovery parse is non-empty
                                     but apply_patch rejects under
                                     strict and every permissive mode.
                                     Dominantly the anchor is
                                     hallucinated.

The taxonomy is *post-parser-recovery* (Theorem P42-2 ensures
recovery cannot manufacture a pass, so a post-recovery failure is a
genuine generator-side semantic defect, not a parser artifact). The
Phase-42 parser-compliance counter remains the authoritative signal
for format attribution; this module sits strictly above it on the
cell-by-cell analysis path.

Usage:

    from vision_mvp.tasks.swe_semantic_taxonomy import (
        classify_semantic_outcome, SemanticCounter,
    )

    ctr = SemanticCounter()
    for cell_dict in phase43_artifact["cells"]:
        for m in cell_dict["report"]["measurements"]:
            label = classify_semantic_outcome(
                instance=instance_by_id[m["instance_id"]],
                proposed_patch=proposed_by_id_strat[
                    (m["instance_id"], m["strategy"])],
                error_kind=m["error_kind"],
                test_passed=m["test_passed"],
            )
            ctr.record(label, strategy=m["strategy"])
    print(ctr.as_dict())

Theoretical anchor: Theorem P43-3 (semantic-ceiling regime
separation) in `RESULTS_PHASE43.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# =============================================================================
# Closed label vocabulary
# =============================================================================


SEM_OK = "ok"
SEM_PARSE_FAIL = "parse_fail"
SEM_WRONG_EDIT_SITE = "wrong_edit_site"
SEM_RIGHT_SITE_WRONG_LOGIC = "right_site_wrong_logic"
SEM_INCOMPLETE_MULTI_HUNK = "incomplete_multi_hunk"
SEM_TEST_OVERFIT = "test_overfit"
SEM_STRUCTURAL_SEMANTIC_INERT = "structural_semantic_inert"
SEM_SYNTAX_INVALID = "syntax_invalid"
SEM_NO_MATCH_RESIDUAL = "no_match_residual"

ALL_SEMANTIC_LABELS: tuple[str, ...] = (
    SEM_OK, SEM_PARSE_FAIL, SEM_WRONG_EDIT_SITE,
    SEM_RIGHT_SITE_WRONG_LOGIC, SEM_INCOMPLETE_MULTI_HUNK,
    SEM_TEST_OVERFIT, SEM_STRUCTURAL_SEMANTIC_INERT,
    SEM_SYNTAX_INVALID, SEM_NO_MATCH_RESIDUAL,
)


# -----------------------------------------------------------------
# Phase 44 — refined sub-labels (opt-in second-pass classifier).
# The Phase-43 closed vocabulary above remains canonical for
# artifact reporting. The Phase-44 sub-labels are a strictly
# additive partition of SEM_WRONG_EDIT_SITE / SEM_INCOMPLETE_MULTI_HUNK
# / SEM_STRUCTURAL_SEMANTIC_INERT that kicks in only when the
# proposed_patch *bytes* are available (raw capture). A Phase-43
# sentinel-path classification is the *coarsest* label that covers
# its match; the Phase-44 refinement is the *finest* label the raw
# bytes support.
# -----------------------------------------------------------------


SEM_RIGHT_FILE_WRONG_SPAN = "right_file_wrong_span"
"""Proposed OLD text anchors into the same buggy file as the gold
patch — i.e. ``apply_patch`` finds a match *somewhere* in the file —
but the anchored span does not overlap any gold hunk's OLD window.
Distinct from ``SEM_WRONG_EDIT_SITE`` which admits no anchoring
constraint. Typical shape: the generator named the right function
but edited a different statement inside it."""

SEM_RIGHT_SPAN_WRONG_LOGIC = "right_span_wrong_logic"
"""Phase-44 refinement of SEM_RIGHT_SITE_WRONG_LOGIC. Proposed OLD
overlaps a gold hunk's OLD window AND the proposed NEW bytes differ
from gold NEW by more than pure whitespace normalisation. Keeps the
Phase-43 label available as a synonym for backwards compatibility."""

SEM_PARTIAL_MULTI_HUNK_SUCCESS = "partial_multi_hunk_success"
"""Gold has ≥ 2 hunks; the proposed patch covers at least one hunk
AND at least one proposed NEW agrees with gold NEW (up to
whitespace normalisation) AND at least one other gold hunk is
missed. Strictly narrower than SEM_INCOMPLETE_MULTI_HUNK: the
partial-success variant records that *some* of the gold work was
done, separating "multi-hunk awareness" from "multi-hunk blindness"."""

SEM_NARROW_FIX_TEST_OVERFIT = "narrow_fix_test_overfit"
"""Phase-44 refinement of SEM_TEST_OVERFIT. Proposed NEW agrees
with gold NEW on a *subset* of byte ranges (i.e. the generator
correctly transcribed part of the gold fix) but introduced a
guard or conditional that narrows the fix to the primary example.
Keyed off the detection that (a) error_kind == test_assert,
(b) proposed OLD overlaps a gold OLD, and (c) proposed NEW shares
a non-trivial common substring with gold NEW."""

SEM_STRUCTURAL_VALID_INERT = "structural_valid_inert"
"""Phase-44 refinement of SEM_STRUCTURAL_SEMANTIC_INERT for the
specific failure where the patched file is byte-identical to the
buggy file under whitespace normalisation — the patch applied
(the matcher found an anchor), but the NEW payload is behaviourally
equivalent to OLD. Detected by comparing the normalised patched
source against the normalised buggy source."""


ALL_REFINED_LABELS: tuple[str, ...] = (
    SEM_RIGHT_FILE_WRONG_SPAN, SEM_RIGHT_SPAN_WRONG_LOGIC,
    SEM_PARTIAL_MULTI_HUNK_SUCCESS, SEM_NARROW_FIX_TEST_OVERFIT,
    SEM_STRUCTURAL_VALID_INERT,
)


# Mapping Phase-43 coarse labels → the set of Phase-44 refined
# labels they can partition into. Used by analysis tools to
# verify a refined classification is a *legal* refinement.
REFINEMENT_MAP: dict[str, tuple[str, ...]] = {
    SEM_WRONG_EDIT_SITE: (SEM_RIGHT_FILE_WRONG_SPAN,
                            SEM_WRONG_EDIT_SITE),
    SEM_RIGHT_SITE_WRONG_LOGIC: (SEM_RIGHT_SITE_WRONG_LOGIC,
                                    SEM_RIGHT_SPAN_WRONG_LOGIC,
                                    SEM_STRUCTURAL_VALID_INERT),
    SEM_INCOMPLETE_MULTI_HUNK: (SEM_PARTIAL_MULTI_HUNK_SUCCESS,
                                  SEM_INCOMPLETE_MULTI_HUNK),
    SEM_TEST_OVERFIT: (SEM_NARROW_FIX_TEST_OVERFIT,
                         SEM_TEST_OVERFIT),
    SEM_STRUCTURAL_SEMANTIC_INERT: (SEM_STRUCTURAL_VALID_INERT,
                                       SEM_STRUCTURAL_SEMANTIC_INERT),
}


ALL_SEMANTIC_LABELS_V2: tuple[str, ...] = (
    ALL_SEMANTIC_LABELS + ALL_REFINED_LABELS
)


# =============================================================================
# Classifier
# =============================================================================


def _normalise(s: str) -> str:
    """Whitespace-insensitive normalisation used for fuzzy anchor
    equality (cheap; not a full AST comparison)."""
    return " ".join(s.split())


def _overlaps(a: str, b: str, *, min_shared_lines: int = 2) -> bool:
    """True if the two OLD blocks share at least ``min_shared_lines``
    non-empty *normalised* lines."""
    if not a or not b:
        return False
    la = [_normalise(x) for x in a.splitlines() if x.strip()]
    lb = [_normalise(x) for x in b.splitlines() if x.strip()]
    if not la or not lb:
        return False
    shared = set(la) & set(lb)
    return len(shared) >= min_shared_lines


def _anchor_in_source(source: str, old: str) -> bool:
    """True if ``old`` matches a unique non-ambiguous span in
    ``source`` under whitespace-insensitive line-by-line equality.
    Approximates APPLY_MODE_WS_COLLAPSE's locator."""
    if not old or not source:
        return False
    src_lines = [_normalise(x) for x in source.splitlines()]
    old_lines = [_normalise(x) for x in old.splitlines() if x.strip()]
    if not old_lines:
        return False
    hits = 0
    span = len(old_lines)
    for i in range(0, len(src_lines) - span + 1):
        window = [ln for ln in src_lines[i:i + span] if ln]
        if window[:span] == old_lines:
            hits += 1
            if hits > 1:
                return False
    return hits == 1


def classify_semantic_outcome(
    *,
    buggy_source: str,
    gold_patch: Sequence[tuple[str, str]],
    proposed_patch: Sequence[tuple[str, str]],
    error_kind: str,
    test_passed: bool,
    error_detail: str = "",
) -> str:
    """Return one of ``ALL_SEMANTIC_LABELS`` describing the post-
    parser-recovery semantic outcome for a single (instance, strategy)
    cell.

    Arguments:
        buggy_source: the source file text *before* any patch.
        gold_patch: the oracle-verified ``(old, new)`` substitutions.
        proposed_patch: the LLM's parsed substitutions (empty tuple
            iff the parser returned no substitutions for this cell).
        error_kind: the Phase-42 bridge error_kind label.
        test_passed: True iff the hidden test passed.
        error_detail: optional prefix of the error message — used to
            disambiguate ``SEM_TEST_OVERFIT`` from
            ``SEM_RIGHT_SITE_WRONG_LOGIC`` when both are plausible.

    Classifier priority order:
        1. test passed → SEM_OK (regardless of other signals).
        2. no proposed patch → SEM_PARSE_FAIL.
        3. error_kind == "patch_no_match" → SEM_NO_MATCH_RESIDUAL
           if the generator emitted an OLD that did not match any
           anchor in the buggy source (even fuzzy).
        4. error_kind == "syntax" → SEM_SYNTAX_INVALID.
        5. gold has ≥ 2 hunks and proposed has strictly fewer
           (successfully applied) substitutions → SEM_INCOMPLETE_MULTI_HUNK.
        6. OLD site disjoint from all gold OLD anchors →
           SEM_WRONG_EDIT_SITE.
        7. test_exception → SEM_STRUCTURAL_SEMANTIC_INERT.
        8. test_assert with multi-case test body signal →
           SEM_TEST_OVERFIT (fixes some but not all).
        9. fallback: SEM_RIGHT_SITE_WRONG_LOGIC.

    Definitions 5 and 6 are mutually refined by the overlap check in
    `_overlaps`: a proposed OLD is "on the right site" iff it shares
    ≥ 2 non-empty normalised lines with at least one gold OLD.
    """
    # 1. pass
    if test_passed:
        return SEM_OK

    # 2. no substitutions
    if not proposed_patch:
        return SEM_PARSE_FAIL

    # 3. patch did not apply
    if error_kind == "patch_no_match":
        return SEM_NO_MATCH_RESIDUAL

    # 4. patched source doesn't compile
    if error_kind == "syntax":
        return SEM_SYNTAX_INVALID

    # 5. multi-hunk gap
    if len(gold_patch) >= 2 and len(proposed_patch) < len(gold_patch):
        # Only call this incomplete_multi_hunk when the proposed hunks
        # do not cover every distinct gold site.
        covered = 0
        for (g_old, _g_new) in gold_patch:
            for (p_old, _p_new) in proposed_patch:
                if _overlaps(g_old, p_old):
                    covered += 1
                    break
        if covered < len(gold_patch):
            return SEM_INCOMPLETE_MULTI_HUNK

    # 6. right-site check
    right_site = False
    for (g_old, _g_new) in gold_patch:
        for (p_old, _p_new) in proposed_patch:
            if _overlaps(g_old, p_old):
                right_site = True
                break
        if right_site:
            break
    if not right_site:
        return SEM_WRONG_EDIT_SITE

    # 7. runtime error after applying a right-site patch
    if error_kind == "test_exception":
        return SEM_STRUCTURAL_SEMANTIC_INERT

    # 8. overfit test detection — heuristic: the error_detail
    # typically contains a specific failing input; if the gold NEW
    # differs substantially from the proposed NEW for exactly one
    # hunk while the proposed patch otherwise matches gold site-wise,
    # it is likely a narrow fix that passes some cases.
    if error_kind == "test_assert":
        # Overfit proxy: test assertion failed but the proposed fix
        # is structurally close to gold's shape (same number of hunks
        # and all on right sites).
        if len(proposed_patch) == len(gold_patch):
            # Check if at least one proposed NEW differs from the
            # matching gold NEW. If all NEWs agree (up to normalisation)
            # something strange is going on; classify as wrong_logic
            # for safety. Otherwise: overfit.
            overlaps_ok = True
            new_differs = False
            for (g_old, g_new) in gold_patch:
                matched = False
                for (p_old, p_new) in proposed_patch:
                    if _overlaps(g_old, p_old):
                        matched = True
                        if _normalise(g_new) != _normalise(p_new):
                            new_differs = True
                        break
                if not matched:
                    overlaps_ok = False
                    break
            if overlaps_ok and new_differs:
                # Signal test_overfit when the error_detail mentions
                # a specific assertion — common shape of a narrow
                # fix that handled the primary example but not a
                # second assertion.
                if ("assert" in error_detail.lower() or
                        "expected" in error_detail.lower() or
                        error_detail):
                    return SEM_TEST_OVERFIT
        return SEM_RIGHT_SITE_WRONG_LOGIC

    # 9. fallback
    return SEM_RIGHT_SITE_WRONG_LOGIC


# =============================================================================
# Phase 44 — refined classifier
# =============================================================================


def _common_substring_len(a: str, b: str) -> int:
    """Length of the longest common contiguous substring of ``a`` and
    ``b`` under whitespace normalisation. Cheap O(|a| + |b|) via a
    token-set intersection — deliberately conservative; callers should
    treat a positive value as "partially overlapping content", not
    "substring equality"."""
    if not a or not b:
        return 0
    ta = _normalise(a).split()
    tb = _normalise(b).split()
    if not ta or not tb:
        return 0
    sa = set(ta)
    sb = set(tb)
    return sum(1 for t in tb if t in sa) + sum(1 for t in ta if t in sb) // 2


def _apply_normalised_equals(source_a: str, source_b: str) -> bool:
    """True iff two source strings agree after whitespace collapsing."""
    norm_a = "\n".join(_normalise(x) for x in source_a.splitlines())
    norm_b = "\n".join(_normalise(x) for x in source_b.splitlines())
    return norm_a == norm_b


def refine_semantic_outcome(
    *,
    coarse_label: str,
    buggy_source: str,
    gold_patch: Sequence[tuple[str, str]],
    proposed_patch: Sequence[tuple[str, str]],
    applied_patch: Sequence[tuple[str, str]] | None = None,
    patched_source: str | None = None,
    error_detail: str = "",
) -> str:
    """Second-pass classifier that refines a coarse Phase-43 label
    into a Phase-44 sub-label *when* the proposed-patch bytes
    support it.

    Rules (return the first that fires):

      * ``coarse_label == SEM_WRONG_EDIT_SITE``:
          - proposed OLD appears unambiguously somewhere in the
            buggy source (``apply_patch``-style locate) AND the
            match span does not overlap any gold OLD →
            ``SEM_RIGHT_FILE_WRONG_SPAN``;
          - otherwise → ``SEM_WRONG_EDIT_SITE`` (unchanged).

      * ``coarse_label == SEM_INCOMPLETE_MULTI_HUNK``:
          - at least one proposed NEW agrees with its matching
            gold NEW (by whitespace-normalised equality) →
            ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``;
          - otherwise → ``SEM_INCOMPLETE_MULTI_HUNK``.

      * ``coarse_label == SEM_STRUCTURAL_SEMANTIC_INERT``:
          - ``patched_source`` agrees with ``buggy_source`` under
            whitespace normalisation → ``SEM_STRUCTURAL_VALID_INERT``;
          - otherwise unchanged.

      * ``coarse_label == SEM_TEST_OVERFIT``:
          - proposed NEW shares ≥ 40 % of gold NEW's tokens with the
            matching gold hunk (by the cheap intersection) →
            ``SEM_NARROW_FIX_TEST_OVERFIT``;
          - otherwise unchanged.

      * ``coarse_label == SEM_RIGHT_SITE_WRONG_LOGIC``:
          - renamed to ``SEM_RIGHT_SPAN_WRONG_LOGIC`` — a direct
            synonym under Phase 44 — when raw bytes are available,
            so downstream analyses can filter on "Phase-44
            refined" cleanly. If ``proposed_patch`` is the sentinel
            (see Phase-43 § D.7), returns the coarse label unchanged.

    For any other coarse label (SEM_OK, SEM_PARSE_FAIL,
    SEM_SYNTAX_INVALID, SEM_NO_MATCH_RESIDUAL), this function
    returns the coarse label unchanged.

    Pre-condition: ``proposed_patch`` is the *real* list of parsed
    substitutions (not a sentinel). When sentinel is passed, the
    function returns ``coarse_label`` as-is.
    """
    # Sentinel-path opt-out: if proposed_patch was a sentinel, no
    # refinement is possible; return the coarse label.
    sentinel = (("__sentinel__", "__sentinel__"),)
    if tuple(proposed_patch) == sentinel:
        return coarse_label

    if coarse_label == SEM_WRONG_EDIT_SITE:
        # Does any proposed OLD match somewhere in the buggy source?
        for (p_old, _p_new) in proposed_patch:
            if not p_old:
                continue
            if _anchor_in_source(buggy_source, p_old):
                # Match lives in the file — but does it overlap any
                # gold OLD? (If so, the classifier would have called
                # this right_site in the first place; we guard anyway.)
                overlaps = False
                for (g_old, _g_new) in gold_patch:
                    if _overlaps(g_old, p_old):
                        overlaps = True
                        break
                if not overlaps:
                    return SEM_RIGHT_FILE_WRONG_SPAN
        return SEM_WRONG_EDIT_SITE

    if coarse_label == SEM_INCOMPLETE_MULTI_HUNK:
        # Any proposed NEW byte-normalised-agree with a gold NEW?
        # Match OLDs by (a) normalised-string equality (robust to
        # single-line hunks) or (b) the ≥2-line overlap check.
        for (g_old, g_new) in gold_patch:
            for (p_old, p_new) in proposed_patch:
                same_old = (_normalise(g_old) == _normalise(p_old)
                             or _overlaps(g_old, p_old))
                if same_old and _normalise(g_new) == _normalise(p_new):
                    return SEM_PARTIAL_MULTI_HUNK_SUCCESS
        return SEM_INCOMPLETE_MULTI_HUNK

    if coarse_label == SEM_STRUCTURAL_SEMANTIC_INERT:
        if patched_source is not None and buggy_source:
            if _apply_normalised_equals(patched_source, buggy_source):
                return SEM_STRUCTURAL_VALID_INERT
        return SEM_STRUCTURAL_SEMANTIC_INERT

    if coarse_label == SEM_TEST_OVERFIT:
        for (g_old, g_new) in gold_patch:
            for (p_old, p_new) in proposed_patch:
                if _overlaps(g_old, p_old):
                    # Token-intersection heuristic: ≥ 40 % of gold NEW
                    # tokens appear in proposed NEW.
                    tokens_g = _normalise(g_new).split()
                    tokens_p = set(_normalise(p_new).split())
                    if not tokens_g:
                        continue
                    shared = sum(1 for t in tokens_g if t in tokens_p)
                    if shared >= 0.4 * len(tokens_g):
                        return SEM_NARROW_FIX_TEST_OVERFIT
        return SEM_TEST_OVERFIT

    if coarse_label == SEM_RIGHT_SITE_WRONG_LOGIC:
        # Confirm the match is real (raw bytes available) and
        # return the Phase-44 spelling.
        for (g_old, _g_new) in gold_patch:
            for (p_old, _p_new) in proposed_patch:
                if _overlaps(g_old, p_old):
                    return SEM_RIGHT_SPAN_WRONG_LOGIC
        return SEM_RIGHT_SITE_WRONG_LOGIC

    return coarse_label


def classify_semantic_outcome_v2(
    *,
    buggy_source: str,
    gold_patch: Sequence[tuple[str, str]],
    proposed_patch: Sequence[tuple[str, str]],
    applied_patch: Sequence[tuple[str, str]] = (),
    patched_source: str | None = None,
    error_kind: str,
    test_passed: bool,
    error_detail: str = "",
) -> str:
    """Phase-44 single-call classifier.

    Runs the Phase-43 ``classify_semantic_outcome`` to obtain the
    coarse label, then applies ``refine_semantic_outcome`` to
    narrow it. Returns a label from
    ``ALL_SEMANTIC_LABELS_V2`` (coarse ∪ refined).

    When ``proposed_patch`` is empty or the sentinel, behaves
    identically to the Phase-43 classifier (safety guarantee:
    Theorem P44-2).
    """
    coarse = classify_semantic_outcome(
        buggy_source=buggy_source,
        gold_patch=gold_patch,
        proposed_patch=proposed_patch,
        error_kind=error_kind,
        test_passed=test_passed,
        error_detail=error_detail,
    )
    return refine_semantic_outcome(
        coarse_label=coarse,
        buggy_source=buggy_source,
        gold_patch=gold_patch,
        proposed_patch=proposed_patch,
        applied_patch=applied_patch,
        patched_source=patched_source,
        error_detail=error_detail,
    )


# =============================================================================
# Aggregation counter
# =============================================================================


@dataclass
class SemanticCounter:
    """Per-strategy histogram over the semantic-label vocabulary.

    Records one label per (instance, strategy) cell. ``by_strategy``
    maps strategy name → dict[label] → count. ``pooled`` is the
    aggregate across strategies.
    """

    n_records: int = 0
    by_strategy: dict[str, dict[str, int]] = field(default_factory=dict)
    pooled: dict[str, int] = field(default_factory=dict)

    def record(self, label: str, *, strategy: str) -> None:
        self.n_records += 1
        strat_map = self.by_strategy.setdefault(strategy, {})
        strat_map[label] = strat_map.get(label, 0) + 1
        self.pooled[label] = self.pooled.get(label, 0) + 1

    def pass_rate(self, *, strategy: str | None = None) -> float:
        if strategy is None:
            total = sum(self.pooled.values())
            if total == 0:
                return 0.0
            return self.pooled.get(SEM_OK, 0) / total
        strat_map = self.by_strategy.get(strategy, {})
        total = sum(strat_map.values())
        if total == 0:
            return 0.0
        return strat_map.get(SEM_OK, 0) / total

    def failure_mix(self, *, strategy: str | None = None
                      ) -> dict[str, float]:
        """Return the fraction of *failures* (non-``SEM_OK``) in each
        label bucket. Useful for comparing failure composition across
        models when pass-rates differ.
        """
        if strategy is None:
            base = self.pooled
        else:
            base = self.by_strategy.get(strategy, {})
        total_fail = sum(v for (k, v) in base.items() if k != SEM_OK)
        if total_fail == 0:
            return {}
        return {
            k: round(v / total_fail, 4)
            for (k, v) in base.items() if k != SEM_OK
        }

    def as_dict(self) -> dict:
        out = {
            "n_records": self.n_records,
            "pooled": dict(sorted(self.pooled.items())),
            "by_strategy": {
                s: dict(sorted(m.items()))
                for (s, m) in sorted(self.by_strategy.items())
            },
            "pooled_pass_rate": round(self.pass_rate(), 4),
            "pooled_failure_mix": self.failure_mix(),
        }
        return out


__all__ = [
    "SEM_OK", "SEM_PARSE_FAIL", "SEM_WRONG_EDIT_SITE",
    "SEM_RIGHT_SITE_WRONG_LOGIC", "SEM_INCOMPLETE_MULTI_HUNK",
    "SEM_TEST_OVERFIT", "SEM_STRUCTURAL_SEMANTIC_INERT",
    "SEM_SYNTAX_INVALID", "SEM_NO_MATCH_RESIDUAL",
    "ALL_SEMANTIC_LABELS",
    # Phase 44 refined labels + classifier
    "SEM_RIGHT_FILE_WRONG_SPAN", "SEM_RIGHT_SPAN_WRONG_LOGIC",
    "SEM_PARTIAL_MULTI_HUNK_SUCCESS",
    "SEM_NARROW_FIX_TEST_OVERFIT", "SEM_STRUCTURAL_VALID_INERT",
    "ALL_REFINED_LABELS", "ALL_SEMANTIC_LABELS_V2",
    "REFINEMENT_MAP",
    "refine_semantic_outcome", "classify_semantic_outcome_v2",
    "classify_semantic_outcome", "SemanticCounter",
]
