"""Phase 42 — LLM patch-output parser with an explicit compliance layer.

Phase 41 § D.4 surfaced a new attribution boundary the programme had
not named before: the LLM-output *parser*. On the Phase-41 bank,
``gemma2:9b`` emits the semantically correct fix on every instance
but fails to close the ``<<<`` delimiter of the bridge's ``OLD>>>``
/ ``<<<NEW>>>`` / ``<<<`` block, so every patch parses as the empty
tuple and lands as ``patch_no_match``. The byte-strict Phase-40
parser is brittle in a specific, measurable way: it silently drops
patches that are semantically correct but stop short of the final
delimiter.

This module adds an attribution layer *above* the matcher axis:

  * **Explicit parser-failure taxonomy.** Every call to
    ``parse_patch_block`` returns a ``ParseOutcome`` carrying an
    ``ok`` boolean, a list of (old, new) substitutions, a
    ``failure_kind`` drawn from a small closed vocabulary, and a
    ``recovery`` label naming whether a recovery heuristic fired
    and which one.
  * **Unified-diff fallback.** The parser accepts both the native
    ``OLD>>>...<<<NEW>>>...<<<`` block shape (Phase-40 contract)
    and a ``--- a/<path>`` / ``+++ b/<path>`` unified-diff shape —
    ``parse_unified_diff`` is called via the Phase-40 code path and
    the per-file substitutions are returned.
  * **Tolerant block closing at end-of-generation.** If the NEW
    block starts with ``<<<NEW>>>`` but no closing ``<<<`` appears
    before end-of-generation, the parser synthesises the closing
    delimiter at EOS *and records it as recovery="closed_at_eos"*.
    The recovery label is preserved all the way to the failure
    taxonomy so a downstream analyst can tell a byte-clean parse
    from a recovered parse.
  * **Partial-block recovery.** If the generator emits partial
    formatting (``OLD:`` / ``NEW:`` free-form, or only a code
    fence), the parser tries three deterministic heuristics in
    declared order (``fenced_code_heuristic`` →
    ``label_prefix_heuristic`` → ``json_block_heuristic``) and
    records which one (if any) recovered. Heuristics are pure and
    order-invariant on their *own* applicable inputs; the explicit
    ordering documents priority when more than one would succeed.
  * **Honest separation between parser recovery and patch
    quality.** Recovery never synthesises *content*; it only
    reconstructs a missing delimiter or label. If the content
    after recovery is semantically wrong, the downstream matcher
    still rejects it cleanly. Recovery therefore cannot manufacture
    a false pass — it can only convert a ``parse_failed`` outcome
    into a legitimate matcher-axis outcome (``old_not_found`` /
    ``old_ambiguous`` / ``passed`` depending on the patch).

Scope discipline — what the parser does NOT do:

  * It does not *guess* at the fix. If the generator emits only
    prose ("the bug is on line 42"), no heuristic fabricates an
    OLD/NEW pair; the outcome is ``failure_kind = "prose_only"``.
  * It does not retry the LLM. Retry is a driver-level concern —
    the parser is pure.
  * It does not re-rank substitutions. The parser emits them in
    the order the model wrote them; the matcher applies them
    left-to-right.

Theoretical anchor: Theorem P42-1 (parser-compliance is an
independent attribution axis above matcher precision) and
Theorem P42-2 (recovery cannot produce a false pass).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence


# =============================================================================
# Failure taxonomy — closed vocabulary above the matcher axis
# =============================================================================


# Parser outcomes partition the LLM-output space into a small,
# exhaustive vocabulary. Each kind is mutually exclusive — the
# parser emits *one* of these per call.

PARSE_OK = "ok"
"""Successful parse; the produced substitutions are the matcher's
responsibility from here on."""

PARSE_EMPTY_OUTPUT = "empty_output"
"""The LLM returned an empty string (or whitespace-only)."""

PARSE_NO_BLOCK = "no_block"
"""Non-empty output but no OLD>>>/<<<NEW>>>/unified-diff anchor
anywhere in it. Typically prose explaining the fix without code."""

PARSE_UNCLOSED_NEW = "unclosed_new"
"""A block start is present (``OLD>>>`` + ``<<<NEW>>>``) but the
final ``<<<`` delimiter is missing. Either recovered via the
end-of-generation closing heuristic (``ok`` + recovery =
``closed_at_eos``) or surfaced as failure when the NEW payload is
empty too."""

PARSE_UNCLOSED_OLD = "unclosed_old"
"""The OLD>>> header is present but the <<<NEW>>> separator is
missing. Rare; almost always paired with prose_only."""

PARSE_MALFORMED_DIFF = "malformed_diff"
"""A unified-diff shape was detected but ``parse_unified_diff``
returned no hunks or every hunk was a pure addition/deletion the
substitution model cannot apply."""

PARSE_EMPTY_PATCH = "empty_patch"
"""The block parsed cleanly but the OLD payload was empty (generator
emitted ``OLD>>>\\n<<<NEW>>>...<<<`` with a blank OLD). Distinct from
``unclosed_new`` — here the structure is correct but the content is
unusable."""

PARSE_MULTI_BLOCK = "multi_block"
"""Multiple OLD/NEW blocks were emitted where the bridge contract
allows exactly one. The parser keeps all of them (the matcher
applies them left-to-right) but records the multi-block shape so
the attribution analyst sees it."""

PARSE_PROSE_ONLY = "prose_only"
"""The output is prose that explicitly *describes* the fix (contains
trigger words like "change X to Y", "replace", "the bug is") but
does not contain any OLD/NEW or diff shape. The parser does NOT
fabricate substitutions from prose — this is a distinct failure."""

PARSE_FENCED_ONLY = "fenced_only"
"""The output contains a code fence (```python``` ... ```) but no
OLD/NEW block. The fenced-code heuristic can sometimes recover if
the fence contains exactly two code blocks of compatible shape;
otherwise surfaces as a distinct failure."""


ALL_PARSE_KINDS: tuple[str, ...] = (
    PARSE_OK, PARSE_EMPTY_OUTPUT, PARSE_NO_BLOCK, PARSE_UNCLOSED_NEW,
    PARSE_UNCLOSED_OLD, PARSE_MALFORMED_DIFF, PARSE_EMPTY_PATCH,
    PARSE_MULTI_BLOCK, PARSE_PROSE_ONLY, PARSE_FENCED_ONLY,
)


# Recovery labels — which heuristic (if any) fired.
RECOVERY_NONE = ""
RECOVERY_CLOSED_AT_EOS = "closed_at_eos"
RECOVERY_FENCED_CODE = "fenced_code_heuristic"
RECOVERY_LABEL_PREFIX = "label_prefix_heuristic"
RECOVERY_UNIFIED_DIFF = "unified_diff_fallback"
RECOVERY_LOOSE_DELIM = "loose_delimiter_heuristic"
RECOVERY_FENCE_WRAPPED = "fence_wrapped_payload"

ALL_RECOVERY_LABELS: tuple[str, ...] = (
    RECOVERY_NONE, RECOVERY_CLOSED_AT_EOS, RECOVERY_FENCED_CODE,
    RECOVERY_LABEL_PREFIX, RECOVERY_UNIFIED_DIFF,
    RECOVERY_LOOSE_DELIM, RECOVERY_FENCE_WRAPPED,
)


# =============================================================================
# Parse outcome — what a parser call returns
# =============================================================================


@dataclass(frozen=True)
class ParseOutcome:
    """Structured outcome of one parser call.

    ``substitutions`` is the ordered list of ``(old, new)`` pairs
    suitable for the Phase-41 ``apply_patch`` matcher. On a
    non-``ok`` outcome the list is empty (by convention) except for
    ``PARSE_MULTI_BLOCK`` which returns every block found.
    """

    ok: bool
    substitutions: tuple[tuple[str, str], ...]
    failure_kind: str
    recovery: str = RECOVERY_NONE
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "substitutions_count": len(self.substitutions),
            "failure_kind": self.failure_kind,
            "recovery": self.recovery,
            "detail": self.detail[:200],
        }


# =============================================================================
# Parser modes — selects which heuristic bundle to apply
# =============================================================================


PARSER_STRICT = "strict"
"""Phase-40/41 parser. Only the ``OLD>>>(.*?)<<<NEW>>>(.*?)<<<``
regex. Kept as a no-recovery baseline for the attribution study."""

PARSER_ROBUST = "robust"
"""Phase-42 default. OLD/NEW block with tolerant block closing at
EOS + unified-diff fallback + fenced-code heuristic + label-prefix
heuristic. Unique-match discipline preserved."""

PARSER_UNIFIED = "unified"
"""Only the unified-diff path. Useful when the prompt instructs the
LLM to emit ``--- a/<path>`` / ``+++ b/<path>`` directly."""

ALL_PARSER_MODES: tuple[str, ...] = (
    PARSER_STRICT, PARSER_ROBUST, PARSER_UNIFIED,
)


# =============================================================================
# Regex library — compiled once, used everywhere
# =============================================================================


_RE_OLD_START = re.compile(r"OLD>>>", re.IGNORECASE)
_RE_NEW_SEP = re.compile(r"<<<NEW>>>", re.IGNORECASE)
_RE_NEW_END = re.compile(r"<<<", re.IGNORECASE)

# Full strict block (same as Phase-41 llm_patch_generator).
_RE_STRICT_BLOCK = re.compile(
    r"OLD>>>(.*?)<<<NEW>>>(.*?)<<<", re.DOTALL | re.IGNORECASE)

# A unified diff is detectable by ``--- a/<path>`` / ``+++ b/<path>`` +
# ``@@`` headers. We only need the first header to decide we are
# looking at diff-shaped output.
_RE_DIFF_HEADER = re.compile(
    r"^---\s+[ab]/\S+$\s+^\+\+\+\s+[ab]/\S+$", re.MULTILINE)

# Code-fence heuristic: three-backtick block, optional language tag.
_RE_FENCE = re.compile(
    r"```[A-Za-z0-9_+\-]*\s*\n(.*?)\n```", re.DOTALL)

# Label-prefix heuristic: models sometimes emit:
#   OLD:
#   <code>
#   NEW:
#   <code>
# (optionally with <code> ... </code> wrappers). We keep the
# pattern deliberately strict so prose-only responses cannot
# accidentally match.
_RE_LABEL_OLD = re.compile(
    r"(?:^|\n)\s*(?:OLD|BEFORE|ORIGINAL)\s*[:=]\s*\n",
    re.IGNORECASE)
_RE_LABEL_NEW = re.compile(
    r"(?:^|\n)\s*(?:NEW|AFTER|FIXED|REPLACEMENT)\s*[:=]\s*\n",
    re.IGNORECASE)


# =============================================================================
# Parser entry-point
# =============================================================================


def parse_patch_block(text: str,
                       *,
                       mode: str = PARSER_ROBUST,
                       unified_diff_parser:
                           Callable[[str], dict[str, tuple[tuple[str, str], ...]]] | None = None,
                       ) -> ParseOutcome:
    """Parse ``text`` into a ``ParseOutcome``.

    ``mode`` selects the parser bundle:

      * ``PARSER_STRICT`` — Phase-40/41 regex, no recovery.
      * ``PARSER_ROBUST`` — Phase-42 default. Tries, in order:
          (1) strict OLD/NEW block,
          (2) loose-delimiter OLD/NEW (tolerant of missing closing
              ``<<<`` at end-of-generation),
          (3) unified-diff parse (via ``unified_diff_parser``),
          (4) fenced-code heuristic (exactly two code fences —
              first is OLD, second is NEW),
          (5) label-prefix heuristic (``OLD:`` / ``NEW:``
              sections).
        Stops at the first heuristic that yields a non-empty pair.
      * ``PARSER_UNIFIED`` — only the unified-diff path.

    ``unified_diff_parser`` is injected to keep this module free
    of the Phase-40 unified-diff parser import (callers pass in
    ``vision_mvp.tasks.swe_bench_bridge.parse_unified_diff``).
    """
    if text is None or not text or not text.strip():
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_EMPTY_OUTPUT,
            recovery=RECOVERY_NONE, detail="")

    if mode == PARSER_STRICT:
        return _parse_strict(text)

    if mode == PARSER_UNIFIED:
        return _parse_unified_only(text, unified_diff_parser)

    # Default: robust.
    # (1) Byte-exact strict block first. On a successful strict
    # parse, apply the Phase-42 fence-wrapped-payload unwrap
    # post-processor so the captured OLD/NEW are free of
    # surrounding ```lang ... ``` fences. The post-processor is
    # a byte-safe projection (Theorem P42-2).
    out = _parse_strict(text)
    if out.ok:
        return _post_strict_fence_unwrap(out)

    # (2) Loose delimiter — tolerate missing closing ``<<<``.
    out2 = _parse_loose_delimiters(text)
    if out2.ok:
        return _post_strict_fence_unwrap(out2)

    # (3) Unified-diff fallback.
    out3 = _parse_unified_only(text, unified_diff_parser)
    if out3.ok:
        return out3

    # (4) Fenced-code heuristic.
    out4 = _parse_fenced_code(text)
    if out4.ok:
        return out4

    # (5) Label-prefix heuristic.
    out5 = _parse_label_prefix(text)
    if out5.ok:
        return out5

    # None recovered — pick the most informative failure kind.
    candidates = [out, out2, out3, out4, out5]
    # Prefer the most *specific* failure kind in this ordering
    # (every candidate has ok=False here).
    preference = [
        PARSE_UNCLOSED_NEW, PARSE_UNCLOSED_OLD, PARSE_EMPTY_PATCH,
        PARSE_MALFORMED_DIFF, PARSE_FENCED_ONLY, PARSE_PROSE_ONLY,
        PARSE_NO_BLOCK,
    ]
    by_kind = {c.failure_kind: c for c in candidates}
    for kind in preference:
        if kind in by_kind:
            return by_kind[kind]
    # Fallback.
    return candidates[0]


# -----------------------------------------------------------------
# Strict parse — Phase-41 baseline
# -----------------------------------------------------------------


_RE_FENCE_WRAPPED_PAYLOAD = re.compile(
    r"\A\s*```[A-Za-z0-9_+\-]*\s*\n(?P<body>.*?)\n\s*```\s*\Z",
    re.DOTALL)


def _unwrap_fenced_payload(payload: str) -> tuple[str, bool]:
    """If ``payload`` is a single ```lang ... ``` fence, return the
    body; otherwise return ``payload`` unchanged. The second return
    value flags whether the unwrap fired — used by callers to set
    ``RECOVERY_FENCE_WRAPPED``.

    We are deliberately strict: the entire payload (after stripping
    outer whitespace) must be exactly one fence. A payload that
    *contains* a fence plus other content is left alone — the
    caller's heuristics (fenced_code, label_prefix) handle those.
    """
    m = _RE_FENCE_WRAPPED_PAYLOAD.match(payload)
    if m is None:
        return payload, False
    return m.group("body"), True


def _parse_strict(text: str) -> ParseOutcome:
    m = _RE_STRICT_BLOCK.search(text)
    if m is None:
        # Decide: was there a partial shape?
        start = _RE_OLD_START.search(text)
        sep = _RE_NEW_SEP.search(text)
        if start is None:
            if _looks_prose(text):
                return ParseOutcome(
                    ok=False, substitutions=(),
                    failure_kind=PARSE_PROSE_ONLY,
                    recovery=RECOVERY_NONE,
                    detail=f"no OLD>>> anchor in {len(text)} chars")
            if "```" in text:
                return ParseOutcome(
                    ok=False, substitutions=(),
                    failure_kind=PARSE_FENCED_ONLY,
                    recovery=RECOVERY_NONE,
                    detail="code fence present, no OLD>>>")
            return ParseOutcome(
                ok=False, substitutions=(),
                failure_kind=PARSE_NO_BLOCK,
                recovery=RECOVERY_NONE,
                detail=f"no OLD>>> anchor in {len(text)} chars")
        # OLD>>> present.
        if sep is None:
            return ParseOutcome(
                ok=False, substitutions=(),
                failure_kind=PARSE_UNCLOSED_OLD,
                recovery=RECOVERY_NONE,
                detail="OLD>>> found but <<<NEW>>> missing")
        # start + sep but no closing <<<.
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_UNCLOSED_NEW,
            recovery=RECOVERY_NONE,
            detail="OLD>>> and <<<NEW>>> found but closing <<< missing")
    old_payload = m.group(1).strip("\n")
    new_payload = m.group(2).strip("\n")
    if not old_payload:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_EMPTY_PATCH,
            recovery=RECOVERY_NONE,
            detail="OLD payload empty")
    # Look for a second block; if one exists, surface PARSE_MULTI_BLOCK
    # but still return every pair (the matcher applies left-to-right).
    all_blocks = _RE_STRICT_BLOCK.findall(text)
    if len(all_blocks) > 1:
        subs = tuple(
            (o.strip("\n"), n.strip("\n"))
            for (o, n) in all_blocks if o.strip("\n"))
        return ParseOutcome(
            ok=True, substitutions=subs,
            failure_kind=PARSE_MULTI_BLOCK,
            recovery=RECOVERY_NONE,
            detail=f"{len(all_blocks)} blocks found")
    return ParseOutcome(
        ok=True, substitutions=((old_payload, new_payload),),
        failure_kind=PARSE_OK, recovery=RECOVERY_NONE, detail="")


def _post_strict_fence_unwrap(outcome: ParseOutcome) -> ParseOutcome:
    """Robust-mode post-processor that unwraps ```lang ... ```
    around each OLD/NEW payload of a strict-parse outcome. Only
    runs when strict already succeeded (``outcome.ok = True``);
    returns a new outcome with ``RECOVERY_FENCE_WRAPPED`` if any
    payload was unwrapped, otherwise returns the original outcome
    unchanged. Byte-safe (Theorem P42-2): the unwrapped payload is
    a substring of the fenced payload, so the matcher's
    ``(apply, test)`` verdict remains a pure function of generator
    bytes.
    """
    if not outcome.ok or not outcome.substitutions:
        return outcome
    unwrapped: list[tuple[str, str]] = []
    any_stripped = False
    for (o, n) in outcome.substitutions:
        o2, os_ = _unwrap_fenced_payload(o)
        n2, ns_ = _unwrap_fenced_payload(n)
        any_stripped = any_stripped or os_ or ns_
        unwrapped.append((o2, n2))
    if not any_stripped:
        return outcome
    return ParseOutcome(
        ok=True, substitutions=tuple(unwrapped),
        failure_kind=outcome.failure_kind,
        recovery=RECOVERY_FENCE_WRAPPED,
        detail=(outcome.detail + "; fence-wrapped payload unwrapped"
                 if outcome.detail
                 else "fence-wrapped payload unwrapped"))


# -----------------------------------------------------------------
# Loose delimiters — tolerate a missing closing <<< at EOS
# -----------------------------------------------------------------


def _parse_loose_delimiters(text: str) -> ParseOutcome:
    """Recover an OLD/NEW block that starts cleanly but does not
    close the NEW payload with ``<<<``.

    Contract: we only recover if the NEW payload is *non-empty*.
    An empty NEW payload is surfaced as ``PARSE_UNCLOSED_NEW`` with
    no recovery (the matcher can't do anything with it anyway).
    """
    start = _RE_OLD_START.search(text)
    if start is None:
        return ParseOutcome(
            ok=False, substitutions=(), failure_kind=PARSE_NO_BLOCK,
            recovery=RECOVERY_NONE, detail="")
    # Consume the OLD payload up to <<<NEW>>>.
    sep = _RE_NEW_SEP.search(text, pos=start.end())
    if sep is None:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_UNCLOSED_OLD,
            recovery=RECOVERY_NONE,
            detail="OLD>>> found but <<<NEW>>> not located after it")
    old_payload = text[start.end():sep.start()].strip("\n")
    # Search for a closing <<< strictly after <<<NEW>>>.
    # Phase-41's regex uses `<<<` (3 chars) which also matches the
    # opening of another ``<<<NEW>>>`` — so we scan carefully.
    new_start = sep.end()
    # Simple case: ``<<<`` literally at some later position that is
    # NOT the start of another ``<<<NEW>>>``.
    close = None
    pos = new_start
    while True:
        m = _RE_NEW_END.search(text, pos=pos)
        if m is None:
            break
        # Skip if this `<<<` is actually the opening of `<<<NEW>>>`.
        if text[m.start():m.start() + 9].upper() == "<<<NEW>>>":
            pos = m.start() + 9
            continue
        close = m
        break
    if close is not None:
        new_payload = text[new_start:close.start()].strip("\n")
        if not old_payload:
            return ParseOutcome(
                ok=False, substitutions=(),
                failure_kind=PARSE_EMPTY_PATCH,
                recovery=RECOVERY_NONE, detail="")
        return ParseOutcome(
            ok=True,
            substitutions=((old_payload, new_payload),),
            failure_kind=PARSE_OK,
            recovery=RECOVERY_LOOSE_DELIM,
            detail="closing <<< found after prose filtering")
    # No closing delimiter — recover at end-of-generation.
    new_payload = text[new_start:].strip("\n")
    # Strip common trailing prose artifacts the LLM may append after
    # the NEW block (e.g. a ``` fence close, or free text).
    new_payload = _strip_trailing_prose(new_payload)
    if not old_payload:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_EMPTY_PATCH,
            recovery=RECOVERY_NONE, detail="")
    if not new_payload:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_UNCLOSED_NEW,
            recovery=RECOVERY_NONE, detail="NEW payload empty at EOS")
    return ParseOutcome(
        ok=True, substitutions=((old_payload, new_payload),),
        failure_kind=PARSE_OK,
        recovery=RECOVERY_CLOSED_AT_EOS,
        detail=f"closed NEW block at end-of-generation "
                f"(payload_chars={len(new_payload)})")


_PROSE_TAILS = (
    r"\n\s*(?:This fixes|This change|Note that|Explanation|"
    r"After this|With this|The fix|Now|Let me know).*$",
    r"\n\s*```\s*$",
    r"\n\s*<</s>>.*$",
    r"\n\s*</?(?:code|pre|answer)>.*$",
    # Trailing partial/full delimiter at end-of-generation:
    # Some models (qwen3.5:35b in Phase 43 § D.4) close the NEW block
    # with ``<<`` instead of the canonical ``<<<``. Without this
    # stripper the partial delimiter lands inside the NEW payload and
    # produces a syntax error on the patched file. The stripper is
    # byte-safe under Theorem P42-2: ``<<`` / ``<<<`` at end-of-
    # generation is never content the generator intends to emit, and
    # the stripped substring is part of the generator's own output —
    # not synthesized.
    r"\n\s*<{2,4}\s*\Z",
)
_RE_PROSE_TAILS = [re.compile(p, re.DOTALL | re.IGNORECASE)
                    for p in _PROSE_TAILS]


def _strip_trailing_prose(payload: str) -> str:
    """Best-effort strip of LLM trailing prose from a NEW payload.

    Conservative: we only remove recognisable prose tails. If the
    payload ends mid-code, we leave it alone.
    """
    s = payload
    for pat in _RE_PROSE_TAILS:
        s = pat.sub("", s)
    return s.rstrip()


# -----------------------------------------------------------------
# Unified-diff fallback
# -----------------------------------------------------------------


def _parse_unified_only(text: str,
                         unified_diff_parser:
                             Callable[[str], dict[str, tuple[tuple[str, str], ...]]] | None,
                         ) -> ParseOutcome:
    """Parse ``text`` as a unified diff.

    If ``unified_diff_parser`` is None, returns ``PARSE_NO_BLOCK``
    (the parser does not reimplement unified-diff parsing here — we
    use the Phase-40 ``parse_unified_diff`` via dependency
    injection).

    Recovery: the unified-diff fallback also tolerates the diff
    being wrapped in a code fence (``\u0060\u0060\u0060diff\n...\n\u0060\u0060\u0060``).
    """
    if unified_diff_parser is None:
        if _RE_DIFF_HEADER.search(text):
            return ParseOutcome(
                ok=False, substitutions=(),
                failure_kind=PARSE_MALFORMED_DIFF,
                recovery=RECOVERY_NONE,
                detail="diff-shaped output but no parser injected")
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_NO_BLOCK,
            recovery=RECOVERY_NONE, detail="")

    # Unwrap a ```diff fence if present.
    diff_text = text
    fence_m = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text,
                         flags=re.DOTALL)
    if fence_m is not None:
        candidate = fence_m.group(1)
        if _RE_DIFF_HEADER.search(candidate):
            diff_text = candidate

    if not _RE_DIFF_HEADER.search(diff_text):
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_NO_BLOCK,
            recovery=RECOVERY_NONE, detail="")

    try:
        parsed = unified_diff_parser(diff_text)
    except Exception as ex:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_MALFORMED_DIFF,
            recovery=RECOVERY_NONE,
            detail=f"unified-diff parser raised: {type(ex).__name__}: {ex}")

    if not parsed:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_MALFORMED_DIFF,
            recovery=RECOVERY_NONE,
            detail="unified-diff parser returned no hunks")

    # Flatten to a single ordered list — callers (and the matcher)
    # do not distinguish across multiple files here; the multi-file
    # case is an upstream driver concern.
    subs: list[tuple[str, str]] = []
    for path in sorted(parsed.keys()):
        subs.extend(parsed[path])
    if not subs:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_MALFORMED_DIFF,
            recovery=RECOVERY_NONE,
            detail="unified-diff parser returned zero substitutions")
    return ParseOutcome(
        ok=True, substitutions=tuple(subs),
        failure_kind=PARSE_OK,
        recovery=RECOVERY_UNIFIED_DIFF,
        detail=f"{len(subs)} substitutions from unified diff")


# -----------------------------------------------------------------
# Fenced-code heuristic — exactly two fences, first is OLD
# -----------------------------------------------------------------


def _parse_fenced_code(text: str) -> ParseOutcome:
    """If the output contains exactly two code fences, treat the
    first as OLD and the second as NEW.

    Conservative: we refuse on 1 or ≥ 3 fences because the mapping
    is ambiguous.
    """
    fences = _RE_FENCE.findall(text)
    if len(fences) != 2:
        if fences:
            return ParseOutcome(
                ok=False, substitutions=(),
                failure_kind=PARSE_FENCED_ONLY,
                recovery=RECOVERY_NONE,
                detail=f"{len(fences)} code fences — heuristic "
                        "requires exactly 2")
        return ParseOutcome(
            ok=False, substitutions=(), failure_kind=PARSE_NO_BLOCK,
            recovery=RECOVERY_NONE, detail="")
    old_payload = fences[0].strip("\n")
    new_payload = fences[1].strip("\n")
    if not old_payload or not new_payload:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_EMPTY_PATCH,
            recovery=RECOVERY_NONE,
            detail="one of the two fences was empty")
    return ParseOutcome(
        ok=True, substitutions=((old_payload, new_payload),),
        failure_kind=PARSE_OK,
        recovery=RECOVERY_FENCED_CODE,
        detail="two-fence pairing")


# -----------------------------------------------------------------
# Label-prefix heuristic — OLD: / NEW: free-form labels
# -----------------------------------------------------------------


def _parse_label_prefix(text: str) -> ParseOutcome:
    """Look for a section headed by ``OLD:`` (or ``BEFORE:`` /
    ``ORIGINAL:``) followed by a section headed by ``NEW:`` (or
    ``AFTER:`` / ``FIXED:`` / ``REPLACEMENT:``).

    Conservative: the OLD section ends at the first NEW label match
    or at a following code-fence close; the NEW section ends at
    EOS, the next OLD label, or the next triple-backtick.
    """
    old_m = _RE_LABEL_OLD.search(text)
    new_m = _RE_LABEL_NEW.search(text)
    if old_m is None or new_m is None or new_m.start() <= old_m.end():
        return ParseOutcome(
            ok=False, substitutions=(), failure_kind=PARSE_NO_BLOCK,
            recovery=RECOVERY_NONE, detail="")
    old_body = text[old_m.end():new_m.start()]
    tail = text[new_m.end():]
    # NEW section ends at end-of-generation, another OLD label,
    # or a fenced-code close marker.
    end_candidates = []
    next_old = _RE_LABEL_OLD.search(tail)
    if next_old is not None:
        end_candidates.append(next_old.start())
    fence_close = re.search(r"```", tail)
    if fence_close is not None:
        end_candidates.append(fence_close.start())
    end = min(end_candidates) if end_candidates else len(tail)
    new_body = tail[:end]
    old_payload = _strip_label_body(old_body)
    new_payload = _strip_label_body(new_body)
    if not old_payload or not new_payload:
        return ParseOutcome(
            ok=False, substitutions=(),
            failure_kind=PARSE_EMPTY_PATCH,
            recovery=RECOVERY_NONE,
            detail="label-prefix: one body empty")
    return ParseOutcome(
        ok=True, substitutions=((old_payload, new_payload),),
        failure_kind=PARSE_OK,
        recovery=RECOVERY_LABEL_PREFIX,
        detail="OLD:/NEW: labelled sections")


def _strip_label_body(body: str) -> str:
    """Strip surrounding code-fence / whitespace from a labelled
    section body.
    """
    s = body
    # Drop a leading ```lang fence open, trailing ``` close.
    s = re.sub(r"^\s*```[A-Za-z0-9_+\-]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    # Drop a leading <code> / trailing </code> wrapper.
    s = re.sub(r"^\s*<code[^>]*>\s*\n?", "", s)
    s = re.sub(r"\n?\s*</code>\s*$", "", s)
    # Normalise surrounding whitespace.
    return s.strip("\n")


# -----------------------------------------------------------------
# Prose-detection heuristic
# -----------------------------------------------------------------


_PROSE_TRIGGERS = (
    "should be", "needs to", "the bug is", "change ", "replace ",
    "instead of", "the fix ", "to fix", "you can", "we can",
    "I recommend", "therefore", "let me", "here is", "here's",
)


def _looks_prose(text: str) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in _PROSE_TRIGGERS)


# =============================================================================
# Compliance counter — aggregate parser outcomes across a run
# =============================================================================


@dataclass
class ParserComplianceCounter:
    """Aggregate ``ParseOutcome`` records across a Phase-42 run.

    One counter is used per ``(model, parser_mode)`` cell in the
    sweep. The counter exposes two headline metrics:

      * ``compliance_rate`` — fraction of calls whose outcome was
        ``ok`` (with or without recovery);
      * ``raw_compliance_rate`` — fraction of calls whose outcome
        was ``ok`` with ``recovery == RECOVERY_NONE`` (parser
        recovery *not* credited).

    The gap ``compliance_rate − raw_compliance_rate`` is the
    fraction of calls that the Phase-42 robust parser rescued from
    the Phase-41 strict parser. Recovery labels are counted
    separately so a downstream analyst can attribute that gap to
    the specific heuristics that fired.
    """

    n_calls: int = 0
    n_raw_ok: int = 0
    n_recovered_ok: int = 0
    kind_counts: dict[str, int] = field(default_factory=dict)
    recovery_counts: dict[str, int] = field(default_factory=dict)

    def record(self, outcome: ParseOutcome) -> None:
        self.n_calls += 1
        self.kind_counts[outcome.failure_kind] = (
            self.kind_counts.get(outcome.failure_kind, 0) + 1)
        self.recovery_counts[outcome.recovery or RECOVERY_NONE] = (
            self.recovery_counts.get(outcome.recovery or RECOVERY_NONE, 0) + 1)
        if outcome.ok:
            if outcome.recovery == RECOVERY_NONE:
                self.n_raw_ok += 1
            else:
                self.n_recovered_ok += 1

    @property
    def compliance_rate(self) -> float:
        if self.n_calls == 0:
            return 0.0
        return (self.n_raw_ok + self.n_recovered_ok) / self.n_calls

    @property
    def raw_compliance_rate(self) -> float:
        if self.n_calls == 0:
            return 0.0
        return self.n_raw_ok / self.n_calls

    @property
    def recovery_lift(self) -> float:
        """How many percentage points the parser's recovery
        heuristics added above the strict baseline.
        """
        return self.compliance_rate - self.raw_compliance_rate

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_raw_ok": self.n_raw_ok,
            "n_recovered_ok": self.n_recovered_ok,
            "compliance_rate": round(self.compliance_rate, 4),
            "raw_compliance_rate": round(self.raw_compliance_rate, 4),
            "recovery_lift": round(self.recovery_lift, 4),
            "kind_counts": dict(sorted(self.kind_counts.items())),
            "recovery_counts": dict(sorted(self.recovery_counts.items())),
        }


# =============================================================================
# Module exports
# =============================================================================


__all__ = [
    # Failure taxonomy
    "PARSE_OK", "PARSE_EMPTY_OUTPUT", "PARSE_NO_BLOCK",
    "PARSE_UNCLOSED_NEW", "PARSE_UNCLOSED_OLD",
    "PARSE_MALFORMED_DIFF", "PARSE_EMPTY_PATCH",
    "PARSE_MULTI_BLOCK", "PARSE_PROSE_ONLY", "PARSE_FENCED_ONLY",
    "ALL_PARSE_KINDS",
    # Recovery labels
    "RECOVERY_NONE", "RECOVERY_CLOSED_AT_EOS",
    "RECOVERY_FENCED_CODE", "RECOVERY_LABEL_PREFIX",
    "RECOVERY_UNIFIED_DIFF", "RECOVERY_LOOSE_DELIM",
    "ALL_RECOVERY_LABELS",
    # Parser modes
    "PARSER_STRICT", "PARSER_ROBUST", "PARSER_UNIFIED",
    "ALL_PARSER_MODES",
    # API
    "ParseOutcome", "parse_patch_block",
    "ParserComplianceCounter",
]
