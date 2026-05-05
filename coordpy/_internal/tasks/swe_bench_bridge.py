"""Phase 39 — SWE-bench-style task bridge for the team substrate.

The Phase-30 ``swe_loop_harness`` proved that a single-aggregator
LLM benefits from substrate-bounded context on analyzer-derived
question banks. Phase 31..38 sharpened the *team-communication*
substrate: typed handoffs, dynamic threads, ensemble defense,
reply calibration.

The largest external-validity gap named at every milestone since
Phase 30 is end-to-end SWE-bench evaluation. SWE-bench's task
shape is *fundamentally team-shaped*: each instance is a tuple
``(repo, base_commit, problem_statement, gold_patch, test_patch)``
where solving requires (a) reading an issue, (b) searching the
codebase, (c) producing a patch, (d) running the hidden test
suite and reporting pass/fail.

This module ships the bridge piece toward that endgame *without*
introducing a Docker / SWE-bench download dependency:

  * A **SWE-bench-compatible task schema**
    (``SWEBenchStyleTask``) that mirrors the public
    SWE-bench instance shape (``instance_id``, ``repo``,
    ``problem_statement``, ``gold_patch``, ``test_patch``,
    ``run_tests``) so a future loader can drop in real
    SWE-bench instances by implementing one adapter
    function (``SWEBenchAdapter.from_swe_bench_dict``).

  * A **MiniSWEBank** of four hand-authored synthetic
    instances built into this repo. Each has a real
    Python source file with a real bug, a real
    issue-statement string, a real gold patch (a
    line-anchored string substitution), and a real
    in-process test that imports the patched module and
    asserts behaviour. Patches are applied by writing a
    temporary file with the substitution applied; tests
    run in a fresh ``exec`` namespace with no network.

  * A **multi-role team runner** built on the existing
    Phase-31 ``HandoffRouter`` and Phase-35 typed
    handoff vocabulary. Four roles —
    ``issue_reader`` / ``code_searcher`` /
    ``patch_generator`` / ``test_runner`` — exchange
    typed claims (``CLAIM_ISSUE_PARSED`` /
    ``CLAIM_FILE_LOCATED`` /
    ``CLAIM_PATCH_PROPOSED`` / ``CLAIM_TEST_RESULT``).
    The patch generator is the team's aggregator
    role and the only role that needs an LLM (or a
    deterministic mock).

  * A **strategy ablation** —
    ``STRATEGY_NAIVE`` / ``STRATEGY_ROUTING`` /
    ``STRATEGY_SUBSTRATE`` — the same shape as
    Phase-30 / Phase-31, so the cross-strategy
    accuracy-vs-token table reads identically to the
    other phases.

Scope discipline
----------------

  * **Not** SWE-bench end-to-end. The included MiniSWEBank
    has 4 instances; they are *self-authored*. The point is
    to deliver a *runnable, reproducible* artifact whose
    schema matches SWE-bench so the next milestone can
    import real SWE-bench instances without architectural
    change.
  * **Not** a claim that the substrate solves SWE-bench.
    The mini bank is constructed so a competent string
    matcher can pass; the LLM-based patch generator is
    optional. The headline is *substrate vs naive*
    *correctness preservation* on this task family, not a
    pass@1 leaderboard.
  * **Not** a replacement for the deterministic
    ``swe_loop_harness``. That harness drives a
    single-aggregator LLM; this module drives a
    multi-role team with typed inter-role handoffs.
  * **Hidden tests run in-process via ``exec``** in a
    sandboxed namespace — no shell, no network, no
    subprocess. This is safe for self-authored tasks; a
    real SWE-bench loader would route through Docker for
    untrusted patches.

Theoretical anchor: RESULTS_PHASE39.md § B (Theorem
P39-3 / P39-4).
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.role_handoff import (
    DeliveryAccount, HandoffLog, HandoffRouter, RoleInbox,
    RoleSubscriptionTable,
)


# =============================================================================
# Roles + claim kinds
# =============================================================================


ROLE_ISSUE_READER = "issue_reader"
ROLE_CODE_SEARCHER = "code_searcher"
ROLE_PATCH_GENERATOR = "patch_generator"
ROLE_TEST_RUNNER = "test_runner"

ALL_SWE_ROLES = (
    ROLE_ISSUE_READER, ROLE_CODE_SEARCHER,
    ROLE_PATCH_GENERATOR, ROLE_TEST_RUNNER,
)


CLAIM_ISSUE_PARSED = "ISSUE_PARSED"          # issue_reader → patch_generator
CLAIM_FILE_LOCATED = "FILE_LOCATED"          # code_searcher → patch_generator
CLAIM_HUNK_LOCATED = "HUNK_LOCATED"          # code_searcher → patch_generator
CLAIM_PATCH_PROPOSED = "PATCH_PROPOSED"       # patch_generator → test_runner
CLAIM_TEST_RESULT = "TEST_RESULT"            # test_runner → patch_generator (verdict)

ALL_SWE_CLAIM_KINDS = (
    CLAIM_ISSUE_PARSED, CLAIM_FILE_LOCATED, CLAIM_HUNK_LOCATED,
    CLAIM_PATCH_PROPOSED, CLAIM_TEST_RESULT,
)


# =============================================================================
# Strategies
# =============================================================================


STRATEGY_NAIVE = "naive"
STRATEGY_ROUTING = "routing"
STRATEGY_SUBSTRATE = "substrate"

ALL_SWE_STRATEGIES = (
    STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE,
)


# =============================================================================
# Task schema — SWE-bench-compatible
# =============================================================================


@dataclass(frozen=True)
class SWEBenchStyleTask:
    """One SWE-bench-compatible task instance.

    Mirrors the SWE-bench public schema closely enough that a
    future ``SWEBenchAdapter.from_swe_bench_dict`` can map a
    downloaded SWE-bench JSONL row into this shape with one
    adapter function. Fields:

      * ``instance_id``        — short stable identifier
        (e.g. ``"mini-swe-001"`` or
        ``"sympy__sympy-12345"``).
      * ``repo``               — short repo label
        (e.g. ``"mini-swe/calc"``).
      * ``base_commit``        — opaque tag; for the mini
        bank we use ``"v0"``.
      * ``problem_statement``  — natural-language issue
        text the issue_reader role consumes.
      * ``buggy_file_relpath`` — path within
        ``repo_root`` of the file that contains the bug.
      * ``buggy_function``     — qualified name of the
        function the gold patch fixes (used by the
        code_searcher's deterministic ranker).
      * ``gold_patch``         — list of
        ``(old_str, new_str)`` substitutions on the
        buggy file; applied left-to-right.
      * ``test_source``        — Python source of the
        hidden test. Must define ``def test() -> None``;
        receives a ``module`` argument bound to the
        patched module's namespace.
      * ``hidden_event_log``   — list of (event_type,
        body) tuples representing the *raw* event stream
        the four roles see *before* any extraction. The
        substrate ablation routes these to the relevant
        roles via subscriptions; naive sends every event
        to the patch_generator.
      * ``role_observable_events`` — for each of the
        four roles, the subset of ``hidden_event_log``
        indices the role can ever observe. Mirrors the
        Phase-31 role-observable-types map.

    A minted task may carry redundant / distractor events
    in ``hidden_event_log`` to make the substrate's
    bounded-context property load-bearing.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    buggy_file_relpath: str
    buggy_function: str
    gold_patch: tuple[tuple[str, str], ...]
    test_source: str
    hidden_event_log: tuple[tuple[str, str], ...] = ()
    role_observable_events: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "buggy_file_relpath": self.buggy_file_relpath,
            "buggy_function": self.buggy_function,
            "gold_patch": [list(p) for p in self.gold_patch],
            "test_source": self.test_source,
            "n_events": len(self.hidden_event_log),
        }


# =============================================================================
# Repo workspace — applies a patch and runs the hidden test in-process
# =============================================================================


@dataclass
class WorkspaceResult:
    """Outcome of one patch+test cycle."""

    patch_applied: bool
    syntax_ok: bool
    test_passed: bool
    error_kind: str   # "" | "patch_no_match" | "syntax" | "import" |
                       # "test_assert" | "test_exception"
    error_detail: str = ""

    def as_dict(self) -> dict:
        return {
            "patch_applied": self.patch_applied,
            "syntax_ok": self.syntax_ok,
            "test_passed": self.test_passed,
            "error_kind": self.error_kind,
            "error_detail": self.error_detail[:300],
        }


APPLY_MODE_STRICT = "strict"
APPLY_MODE_LSTRIP = "lstrip"
APPLY_MODE_WS_COLLAPSE = "ws_collapse"
APPLY_MODE_LINE_ANCHORED = "line_anchored"

ALL_APPLY_MODES = (
    APPLY_MODE_STRICT, APPLY_MODE_LSTRIP,
    APPLY_MODE_WS_COLLAPSE, APPLY_MODE_LINE_ANCHORED,
)


def apply_patch(source: str,
                patch: Sequence[tuple[str, str]],
                *,
                mode: str = APPLY_MODE_STRICT,
                ) -> tuple[str, bool, str]:
    """Apply a list of ``(old, new)`` substitutions left-to-right.

    Returns ``(new_source, applied, reason)``. ``applied`` is True
    iff every ``old`` was present at apply time. ``reason`` is one
    of ``""`` / ``"empty_patch"`` / ``"old_not_found"`` /
    ``"old_ambiguous"`` / ``"unknown_mode"``.

    Matcher modes (Phase 41):

      * ``"strict"`` (default, Phase-40 semantics) — byte-exact
        unique-substring match. The only regime under which the
        substrate's byte-for-byte invariants hold.
      * ``"lstrip"`` — tolerate leading-whitespace drift on every
        line of the OLD block. The matcher re-scans ``source`` line
        by line and finds a run of lines whose ``str.lstrip()`` side
        equals the OLD block's ``str.lstrip()`` side. The matched
        region is replaced in the ORIGINAL source (preserving its
        original indentation width); indentation of the NEW block
        is adjusted to match the source's indent if the OLD block's
        leading-whitespace prefix is consistent.
      * ``"ws_collapse"`` — tolerate internal-whitespace drift on
        every line. Two lines are considered equal when their
        ``" ".join(split())`` normalisation agrees. The actual
        replacement preserves the source's bytes exactly and
        substitutes the NEW block (with its original whitespace)
        over the matched-line span.
      * ``"line_anchored"`` — the strictest of the three permissive
        modes: the matcher requires both anchor sides to agree up
        to *trailing*-whitespace drift on every line (newline /
        carriage-return / stray spaces at end-of-line are ignored,
        but the leading-indent and internal-whitespace shape must
        still match). Captures the "generator emitted an extra
        space before the newline" class of failure without
        admitting semantic drift.

    All three permissive modes remain *unique-match* — a matched
    span that can be produced by more than one anchoring within
    the source is rejected with ``"old_ambiguous"``. This
    preserves the apply_patch invariant that every successful
    return corresponds to a uniquely-identifiable source region.
    """
    if not patch:
        return source, False, "empty_patch"
    if mode == APPLY_MODE_STRICT:
        return _apply_patch_strict(source, patch)
    if mode in (APPLY_MODE_LSTRIP, APPLY_MODE_WS_COLLAPSE,
                APPLY_MODE_LINE_ANCHORED):
        return _apply_patch_permissive(source, patch, mode=mode)
    return source, False, "unknown_mode"


def _apply_patch_strict(source: str,
                          patch: Sequence[tuple[str, str]],
                          ) -> tuple[str, bool, str]:
    cur = source
    for (old, new) in patch:
        if not old:
            return cur, False, "old_not_found"
        n_match = cur.count(old)
        if n_match == 0:
            return cur, False, "old_not_found"
        if n_match > 1:
            return cur, False, "old_ambiguous"
        cur = cur.replace(old, new, 1)
    return cur, True, ""


def _normalise_line(line: str, *, mode: str) -> str:
    """Project a line under the requested matcher normalisation.

    Used only for *locating* a match; the substitution still runs
    against the source's original bytes so the Phase-31 chain hash
    remains auditable.

    Every permissive mode normalises the trailing ``\\r`` / ``\\n``
    away: an OLD block whose final line is missing its ``\\n``
    (common when an LLM emits the block with a
    ``rstrip("\\n")`` / ``strip("\\n")`` step in its parser) must
    still match a source line that has a trailing ``\\n``. Strict
    mode is unchanged — it uses ``str.count`` / ``str.replace``
    on the raw bytes and goes through ``_apply_patch_strict``.
    """
    if mode == APPLY_MODE_LSTRIP:
        return line.lstrip().rstrip("\r\n")
    if mode == APPLY_MODE_WS_COLLAPSE:
        # " ".join(line.split()) already collapses every whitespace
        # run (including the final "\r\n") into a single token.
        return " ".join(line.split())
    if mode == APPLY_MODE_LINE_ANCHORED:
        # rstrip() strips every trailing whitespace including "\r\n".
        return line.rstrip()
    return line


def _apply_patch_permissive(source: str,
                              patch: Sequence[tuple[str, str]],
                              *, mode: str,
                              ) -> tuple[str, bool, str]:
    """Apply each hunk by locating a unique line-range in ``source``
    whose per-line normalisation equals the OLD block's per-line
    normalisation, then splicing the NEW block (in its own original
    form) into that range.

    Unique-match discipline: if the normalised OLD appears more than
    once in the normalised source, the hunk is rejected as
    ``old_ambiguous`` — the substrate's guarantee is that every
    successful ``apply_patch`` return corresponds to one unambiguous
    source region.
    """
    cur = source
    for (old, new) in patch:
        if not old:
            return cur, False, "old_not_found"
        src_lines = cur.splitlines(keepends=True)
        old_lines = old.splitlines(keepends=True)
        if not old_lines:
            return cur, False, "old_not_found"
        norm_src = [_normalise_line(ln, mode=mode) for ln in src_lines]
        norm_old = [_normalise_line(ln, mode=mode) for ln in old_lines]
        hits: list[int] = []
        last = len(norm_src) - len(norm_old)
        if last < 0:
            return cur, False, "old_not_found"
        for i in range(last + 1):
            if norm_src[i:i + len(norm_old)] == norm_old:
                hits.append(i)
        if not hits:
            return cur, False, "old_not_found"
        if len(hits) > 1:
            return cur, False, "old_ambiguous"
        start = hits[0]
        # Compose the replacement. The NEW block keeps its own
        # whitespace shape; we splice it into the source on line
        # boundaries so the surrounding bytes are preserved
        # verbatim.
        new_lines = new.splitlines(keepends=True)
        # Trailing-newline preservation: if the last source line
        # being replaced has a ``\n`` but the last NEW line does
        # not (common when an LLM emits the block with a
        # ``strip("\n")``-style parser step), append the missing
        # ``\n`` so the patched source keeps the original's
        # end-of-file shape. Without this, a line-based splice
        # can drop the trailing newline and surprise a hidden
        # test that asserts source structure.
        if new_lines and len(src_lines) >= start + len(old_lines):
            last_src_idx = start + len(old_lines) - 1
            src_last_nl = src_lines[last_src_idx].endswith("\n")
            if src_last_nl and not new_lines[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1] + "\n"
        cur = "".join(
            src_lines[:start] + new_lines +
            src_lines[start + len(old_lines):])
    return cur, True, ""


def run_patched_test(file_source: str,
                      patched_source: str,
                      test_source: str,
                      module_name: str = "patched_under_test",
                      ) -> WorkspaceResult:
    """Compile ``patched_source`` into a fresh module namespace and
    execute the hidden test against it. Captures syntax / import /
    assertion / exception errors into a ``WorkspaceResult``.

    ``file_source`` is the original buggy source — used only to
    report apply failures meaningfully.

    The test source must define ``def test(module): ...``.

    Sandboxing: a fresh ``__main__``-shaped dict, no shell, no
    subprocess access from this function. Tests are author-
    controlled in the mini bank.
    """
    # Compile the patched source.
    try:
        code = compile(patched_source, f"<{module_name}>", "exec")
    except SyntaxError as ex:
        return WorkspaceResult(
            patch_applied=True, syntax_ok=False,
            test_passed=False, error_kind="syntax",
            error_detail=f"{type(ex).__name__}: {ex}")
    mod_globals: dict = {
        "__name__": module_name, "__doc__": None,
        "__builtins__": __builtins__,
    }
    try:
        # Capture stdout/stderr to keep the harness output clean.
        with redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            exec(code, mod_globals)
    except Exception as ex:
        return WorkspaceResult(
            patch_applied=True, syntax_ok=True,
            test_passed=False, error_kind="import",
            error_detail=f"{type(ex).__name__}: {ex}")
    # Compile the test source.
    test_globals: dict = {
        "__name__": "<patched_test>", "__doc__": None,
        "__builtins__": __builtins__,
    }
    try:
        test_code = compile(test_source, "<patched_test>", "exec")
        exec(test_code, test_globals)
    except SyntaxError as ex:
        return WorkspaceResult(
            patch_applied=True, syntax_ok=True,
            test_passed=False, error_kind="syntax",
            error_detail=f"test syntax: {ex}")
    test_fn = test_globals.get("test")
    if not callable(test_fn):
        return WorkspaceResult(
            patch_applied=True, syntax_ok=True,
            test_passed=False, error_kind="test_exception",
            error_detail="test source did not define test(module)")
    # Wrap exec module dict as an attribute object the test sees.
    class _Mod:
        pass
    mod_obj = _Mod()
    for k, v in mod_globals.items():
        if not k.startswith("__"):
            setattr(mod_obj, k, v)
    try:
        with redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            test_fn(mod_obj)
    except AssertionError as ex:
        return WorkspaceResult(
            patch_applied=True, syntax_ok=True,
            test_passed=False, error_kind="test_assert",
            error_detail=str(ex)[:300])
    except Exception as ex:
        return WorkspaceResult(
            patch_applied=True, syntax_ok=True,
            test_passed=False, error_kind="test_exception",
            error_detail=f"{type(ex).__name__}: {ex}")
    return WorkspaceResult(
        patch_applied=True, syntax_ok=True,
        test_passed=True, error_kind="")


# =============================================================================
# Subscription table (the "who should know what" declaration)
# =============================================================================


def build_swe_role_subscriptions() -> RoleSubscriptionTable:
    """Default Phase-39 subscription table.

    issue_reader  : ISSUE_PARSED   → patch_generator
    code_searcher : FILE_LOCATED   → patch_generator
                  : HUNK_LOCATED   → patch_generator
    patch_generator: PATCH_PROPOSED → test_runner
    test_runner   : TEST_RESULT    → patch_generator   (verdict feedback)
    """
    s = RoleSubscriptionTable()
    s.subscribe(ROLE_ISSUE_READER, CLAIM_ISSUE_PARSED,
                [ROLE_PATCH_GENERATOR])
    s.subscribe(ROLE_CODE_SEARCHER, CLAIM_FILE_LOCATED,
                [ROLE_PATCH_GENERATOR])
    s.subscribe(ROLE_CODE_SEARCHER, CLAIM_HUNK_LOCATED,
                [ROLE_PATCH_GENERATOR])
    s.subscribe(ROLE_PATCH_GENERATOR, CLAIM_PATCH_PROPOSED,
                [ROLE_TEST_RUNNER])
    s.subscribe(ROLE_TEST_RUNNER, CLAIM_TEST_RESULT,
                [ROLE_PATCH_GENERATOR])
    return s


# =============================================================================
# Patch-generator interface
# =============================================================================


PatchGenerator = Callable[
    [SWEBenchStyleTask, dict, str, str], "ProposedPatch"
]


@dataclass(frozen=True)
class ProposedPatch:
    """One patch proposal from the patch_generator role.

    ``patch`` is a list of ``(old, new)`` substitutions; the runner
    will offer it to the test_runner. ``rationale`` is a short
    human-readable explanation included in the typed handoff
    payload (bounded by ``rationale_token_cap``).
    """

    patch: tuple[tuple[str, str], ...]
    rationale: str = ""


def deterministic_oracle_generator(task: SWEBenchStyleTask,
                                     ctx: dict,
                                     buggy_source: str,
                                     issue_summary: str,
                                     ) -> ProposedPatch:
    """Reference patch_generator that emits the gold patch verbatim.

    Used as the substrate's *correctness ceiling*. Any LLM-based
    generator whose accuracy under-shoots this is bottle-necked by
    the LLM, not by the substrate. Mirrors the role of
    ``MockAnswerLLM`` in the Phase-30 harness.
    """
    return ProposedPatch(
        patch=task.gold_patch,
        rationale=issue_summary[:64] or task.problem_statement[:64])


def llm_patch_generator(llm_call: Callable[[str], str],
                          *,
                          parser_mode: str | None = None,
                          parser_counter=None,
                          prompt_style: str = "block",
                          ) -> PatchGenerator:
    """Wrap an LLM call as a patch_generator.

    The LLM is shown the issue summary, the located file, the
    located hunk (when available) and asked to emit a single
    ``OLD>>>...<<<NEW>>>...<<<`` block (``prompt_style="block"``,
    Phase-40/41 default) *or* a unified-diff (``prompt_style=
    "unified_diff"``, Phase-42 opt-in).

    The ``parser_mode`` argument (Phase-42) picks the parser
    implementation:

      * ``None`` (default) — retain the Phase-41 byte-strict
        regex for backwards compatibility;
      * ``"strict"`` / ``"robust"`` / ``"unified"`` — route
        parsing through ``swe_patch_parser.parse_patch_block``
        with the requested mode. ``"robust"`` is the Phase-42
        default when a parser is selected.

    ``parser_counter`` is an optional
    ``swe_patch_parser.ParserComplianceCounter`` that gets
    ``record``-called on every LLM response; used by the
    Phase-42 driver to aggregate parser-compliance metrics.
    """
    _BLOCK_RE = re.compile(
        r"OLD>>>(.*?)<<<NEW>>>(.*?)<<<", re.DOTALL)

    # Lazy import to keep the bridge module free of a hard dependency
    # on the Phase-42 parser module when parser_mode is None.
    _parser = None
    if parser_mode is not None:
        from .swe_patch_parser import parse_patch_block
        _parser = parse_patch_block

    def _gen(task: SWEBenchStyleTask,
             ctx: dict,
             buggy_source: str,
             issue_summary: str) -> ProposedPatch:
        prompt = build_patch_generator_prompt(
            task=task, ctx=ctx, buggy_source=buggy_source,
            issue_summary=issue_summary,
            prompt_style=prompt_style)
        text = llm_call(prompt)
        if _parser is None:
            # Phase-41 byte-strict path — preserved for regression.
            m = _BLOCK_RE.search(text)
            if not m:
                return ProposedPatch(patch=(),
                                      rationale="parse_failed")
            return ProposedPatch(
                patch=((m.group(1).strip("\n"),
                        m.group(2).strip("\n")),),
                rationale="llm_proposed")
        outcome = _parser(
            text, mode=parser_mode,
            unified_diff_parser=parse_unified_diff)
        if parser_counter is not None:
            parser_counter.record(outcome)
        if not outcome.ok:
            return ProposedPatch(
                patch=(),
                rationale=f"parse_failed:{outcome.failure_kind}")
        rat = "llm_proposed"
        if outcome.recovery:
            rat = f"llm_proposed:{outcome.recovery}"
        return ProposedPatch(
            patch=outcome.substitutions,
            rationale=rat)

    return _gen


def build_patch_generator_prompt(*,
                                  task: SWEBenchStyleTask,
                                  ctx: dict,
                                  buggy_source: str,
                                  issue_summary: str,
                                  hunk_window: int = 12,
                                  prompt_style: str = "block",
                                  ) -> str:
    """Assemble a small, bounded prompt for the patch_generator LLM.

    Substrate-style: only the located hunk is included by default
    (``ctx["hunk"]`` if present); otherwise the function body
    block is shown. The point is *bounded context* — the patch
    generator does not need the entire repository, the entire
    issue, or the entire file.

    ``prompt_style`` (Phase 42):
      * ``"block"`` (default) — Phase-40/41 ``OLD>>>.../<<<NEW>>>
        .../<<<`` output contract, byte-exact.
      * ``"unified_diff"`` — request a ``--- a/<path>`` /
        ``+++ b/<path>`` / ``@@`` unified-diff output. Paired with
        ``parser_mode="robust"`` on ``llm_patch_generator``, this
        gives a second output shape the parser accepts.
    """
    hunk = ctx.get("hunk") or _slice_function(
        buggy_source, task.buggy_function, hunk_window)
    header = [
        ("You are the PATCH GENERATOR role on a multi-role "
         "code-fix team. Read the issue summary and the located "
         "code hunk, then emit ONE patch."),
        f"INSTANCE: {task.instance_id}  REPO: {task.repo}",
        f"ISSUE SUMMARY: {issue_summary}",
        f"FILE: {task.buggy_file_relpath}",
        f"FUNCTION: {task.buggy_function}",
        "HUNK:",
        "```python",
        hunk,
        "```",
    ]
    if prompt_style == "unified_diff":
        footer = [
            ("Reply with exactly one unified diff, no surrounding "
             "prose. Format (pay careful attention to indentation —"
             " lines after --- / +++ / @@ must preserve the "
             "source's indentation exactly):"),
            f"--- a/{task.buggy_file_relpath}",
            f"+++ b/{task.buggy_file_relpath}",
            "@@ -<line>,<count> +<line>,<count> @@",
            "<context line>",
            "-<line exactly as it appears in the hunk>",
            "+<line that fixes the bug>",
            "<context line>",
        ]
    else:
        footer = [
            ("Reply with exactly one OLD/NEW block, no surrounding "
             "prose. IMPORTANT: you MUST emit the closing '<<<' "
             "delimiter after the NEW section — do NOT stop early. "
             "Format:"),
            "OLD>>>",
            "<old text exactly as it appears in the hunk>",
            "<<<NEW>>>",
            "<new text that fixes the bug>",
            "<<<",
        ]
    return "\n".join(header + footer)


def _slice_function(source: str,
                     qualified_name: str,
                     window: int = 12) -> str:
    """Return at most ``window`` lines centered on the named def.

    Conservative; falls back to first 2*window lines if the def
    is not found.
    """
    lines = source.splitlines()
    pat = re.compile(r"^\s*def\s+" + re.escape(
        qualified_name.split(".")[-1]) + r"\b")
    for i, line in enumerate(lines):
        if pat.match(line):
            return "\n".join(lines[i: i + window])
    return "\n".join(lines[: 2 * window])


# =============================================================================
# Per-role event observation
# =============================================================================


_EVENT_TYPE_ISSUE = "ISSUE"
_EVENT_TYPE_FILE_HEADER = "FILE_HEADER"
_EVENT_TYPE_FUNCTION_BODY = "FUNCTION_BODY"
_EVENT_TYPE_TEST_FIXTURE = "TEST_FIXTURE"
_EVENT_TYPE_DISTRACTOR = "DISTRACTOR"


_DEFAULT_ROLE_OBSERVABLE_TYPES = {
    ROLE_ISSUE_READER: frozenset({_EVENT_TYPE_ISSUE,
                                    _EVENT_TYPE_DISTRACTOR}),
    ROLE_CODE_SEARCHER: frozenset({_EVENT_TYPE_FILE_HEADER,
                                     _EVENT_TYPE_FUNCTION_BODY,
                                     _EVENT_TYPE_DISTRACTOR}),
    ROLE_PATCH_GENERATOR: frozenset({_EVENT_TYPE_DISTRACTOR}),
    ROLE_TEST_RUNNER: frozenset({_EVENT_TYPE_TEST_FIXTURE,
                                   _EVENT_TYPE_DISTRACTOR}),
}


def role_observable(task: SWEBenchStyleTask, role: str
                      ) -> list[tuple[str, str]]:
    """Filter ``task.hidden_event_log`` to the events ``role`` can
    observe under the default per-role observable map.
    """
    obs = _DEFAULT_ROLE_OBSERVABLE_TYPES.get(role, frozenset())
    return [ev for ev in task.hidden_event_log if ev[0] in obs]


# =============================================================================
# Substrate-driven runner
# =============================================================================


@dataclass
class SWEMeasurement:
    """Per-(task, strategy) measurement."""

    instance_id: str
    repo: str
    strategy: str
    test_passed: bool
    patch_applied: bool
    error_kind: str
    n_events_to_patch_gen: int
    n_chars_patch_gen_prompt: int
    n_handoffs: int
    n_events_total: int
    log_length: int
    chain_ok: bool
    wall_seconds: float

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "strategy": self.strategy,
            "test_passed": self.test_passed,
            "patch_applied": self.patch_applied,
            "error_kind": self.error_kind,
            "n_events_to_patch_gen": self.n_events_to_patch_gen,
            "n_chars_patch_gen_prompt":
                self.n_chars_patch_gen_prompt,
            "n_handoffs": self.n_handoffs,
            "n_events_total": self.n_events_total,
            "log_length": self.log_length,
            "chain_ok": self.chain_ok,
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class SWEReport:
    """Aggregated report across (task, strategy) pairs."""

    n_tasks: int
    measurements: list[SWEMeasurement]
    config: dict

    def as_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "config": self.config,
            "measurements": [m.as_dict() for m in self.measurements],
            "pooled": self.pooled_summary(),
        }

    def pooled_summary(self) -> dict:
        by: dict[str, list[SWEMeasurement]] = {}
        for m in self.measurements:
            by.setdefault(m.strategy, []).append(m)
        out: dict[str, dict] = {}
        for strat, ms in by.items():
            n = len(ms)
            if not n:
                continue
            n_pass = sum(1 for m in ms if m.test_passed)
            n_apply = sum(1 for m in ms if m.patch_applied)
            mean_chars = sum(m.n_chars_patch_gen_prompt for m in ms) / n
            mean_events = sum(m.n_events_to_patch_gen for m in ms) / n
            mean_handoffs = sum(m.n_handoffs for m in ms) / n
            out[strat] = {
                "n": n,
                "pass_at_1": round(n_pass / n, 4),
                "patch_applied_rate": round(n_apply / n, 4),
                "mean_patch_gen_prompt_chars": round(mean_chars, 1),
                "mean_patch_gen_prompt_tokens_approx":
                    round(mean_chars / 4, 1),
                "mean_events_to_patch_gen": round(mean_events, 2),
                "mean_handoffs": round(mean_handoffs, 2),
            }
        return out


def _summarise_issue(text: str, cap: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= cap:
        return text
    return text[: cap - 1] + "…"


def _build_patch_gen_context(*,
                              strategy: str,
                              task: SWEBenchStyleTask,
                              buggy_source: str,
                              issue_events: list[tuple[str, str]],
                              code_events: list[tuple[str, str]],
                              all_events: list[tuple[str, str]],
                              ) -> tuple[dict, list[tuple[str, str]]]:
    """Return (ctx, delivered_events) for the patch generator under
    ``strategy``.

    * ``naive``     — every event in ``all_events`` becomes part of
      the prompt; no typed handoff is consumed.
    * ``routing``   — only events matching the patch_generator's
      observable types pass; in this task that is *empty* by
      construction (the role has no native observables).
    * ``substrate`` — typed handoffs only; the prompt receives the
      issue summary and the located hunk via ``ctx["hunk"]``.

    The function is the single place the strategy ablation is
    expressed; the rest of the runner is identical across cells.
    """
    ctx: dict = {}
    if strategy == STRATEGY_SUBSTRATE:
        # Typed handoffs surface only what the patch generator needs.
        if issue_events:
            ctx["issue_summary"] = _summarise_issue(
                " ".join(b for (_t, b) in issue_events))
        # Locate hunk via deterministic searcher signal.
        hunk = _slice_function(
            buggy_source, task.buggy_function, window=12)
        ctx["hunk"] = hunk
        return ctx, []
    if strategy == STRATEGY_ROUTING:
        # patch_generator observes only DISTRACTORs in the
        # default Phase-39 role-observable map; no useful info.
        delivered = role_observable(task, ROLE_PATCH_GENERATOR)
        return ctx, delivered
    # naive
    return ctx, list(all_events)


def _emit_substrate_handoffs(*,
                              task: SWEBenchStyleTask,
                              router: HandoffRouter,
                              issue_events: list[tuple[str, str]],
                              code_events: list[tuple[str, str]],
                              ) -> int:
    """Emit the four substrate handoffs (issue, file, hunk, ...)
    to the patch_generator. Returns the number of handoffs emitted.
    """
    n = 0
    if issue_events:
        router.emit(
            source_role=ROLE_ISSUE_READER, source_agent_id=0,
            claim_kind=CLAIM_ISSUE_PARSED,
            payload=_summarise_issue(
                " ".join(b for (_t, b) in issue_events)),
            source_event_ids=tuple(range(len(issue_events))),
            round=1)
        n += 1
    if code_events:
        router.emit(
            source_role=ROLE_CODE_SEARCHER, source_agent_id=0,
            claim_kind=CLAIM_FILE_LOCATED,
            payload=task.buggy_file_relpath,
            source_event_ids=(), round=1)
        router.emit(
            source_role=ROLE_CODE_SEARCHER, source_agent_id=0,
            claim_kind=CLAIM_HUNK_LOCATED,
            payload=task.buggy_function,
            source_event_ids=(), round=1)
        n += 2
    return n


def _build_patch_gen_prompt_for_strategy(*,
                                          strategy: str,
                                          task: SWEBenchStyleTask,
                                          buggy_source: str,
                                          ctx: dict,
                                          delivered_events:
                                              list[tuple[str, str]],
                                          ) -> str:
    """Build the actual prompt the patch_generator agent sees.

    Stable shape across strategies so the byte-accurate prompt
    measurement is fair.
    """
    issue_summary = ctx.get(
        "issue_summary", task.problem_statement)
    if strategy == STRATEGY_SUBSTRATE:
        return build_patch_generator_prompt(
            task=task, ctx=ctx, buggy_source=buggy_source,
            issue_summary=issue_summary)
    # naive / routing — embed the delivered event stream.
    lines = [
        ("You are the PATCH GENERATOR role on a multi-role "
         "code-fix team. The team substrate failed to deliver "
         "a typed summary; reconstruct what you can from the "
         "raw event stream below."),
        f"INSTANCE: {task.instance_id}  REPO: {task.repo}",
        ("Reply with exactly one OLD/NEW block:"),
        "OLD>>>",
        "<old text exactly as it appears in the file>",
        "<<<NEW>>>",
        "<new text that fixes the bug>",
        "<<<",
        "",
        "DELIVERED EVENTS:",
    ]
    for (t, b) in delivered_events:
        lines.append(f"- [{t}] {b}")
    return "\n".join(lines)


def run_swe_loop(*,
                 bank: Sequence[SWEBenchStyleTask],
                 repo_files: dict[str, str],
                 generator: PatchGenerator,
                 strategies: Sequence[str] = ALL_SWE_STRATEGIES,
                 apply_mode: str = APPLY_MODE_STRICT,
                 ) -> SWEReport:
    """Run ``generator`` over every (task, strategy) pair.

    ``repo_files`` maps ``relpath → source`` for the buggy
    workspace; the runner reads from this dict (not disk) so
    the bench stays hermetic and parallel-safe.

    The runner:
      1. Builds a fresh ``HandoffRouter`` per (task, strategy).
      2. Routes the issue/code/test events to the relevant
         roles' inboxes when ``strategy == "substrate"``.
      3. Calls ``generator(task, ctx, buggy_source, issue_summary)``.
      4. Applies the proposed patch to ``repo_files[task.
         buggy_file_relpath]`` (in memory).
      5. Compiles + runs the hidden test in a fresh namespace.
      6. Records a ``SWEMeasurement``.
    """
    measurements: list[SWEMeasurement] = []
    for task in bank:
        buggy_source = repo_files[task.buggy_file_relpath]
        all_events = list(task.hidden_event_log)
        issue_events = [
            ev for ev in all_events if ev[0] == _EVENT_TYPE_ISSUE]
        code_events = [
            ev for ev in all_events
            if ev[0] in (_EVENT_TYPE_FILE_HEADER,
                          _EVENT_TYPE_FUNCTION_BODY)]
        for strat in strategies:
            t0 = time.time()
            subs = build_swe_role_subscriptions()
            log = HandoffLog()
            account = DeliveryAccount()
            router = HandoffRouter(subs=subs, log=log,
                                     account=account)
            for r in ALL_SWE_ROLES:
                router.register_inbox(RoleInbox(role=r,
                                                  capacity=64))
            n_handoffs = 0
            if strat == STRATEGY_SUBSTRATE:
                n_handoffs = _emit_substrate_handoffs(
                    task=task, router=router,
                    issue_events=issue_events,
                    code_events=code_events)
            ctx, delivered = _build_patch_gen_context(
                strategy=strat, task=task,
                buggy_source=buggy_source,
                issue_events=issue_events,
                code_events=code_events,
                all_events=all_events)
            prompt = _build_patch_gen_prompt_for_strategy(
                strategy=strat, task=task,
                buggy_source=buggy_source, ctx=ctx,
                delivered_events=delivered)
            issue_summary = ctx.get(
                "issue_summary", task.problem_statement)
            try:
                proposed = generator(task, ctx, buggy_source,
                                       issue_summary)
            except Exception as ex:  # robustness net
                proposed = ProposedPatch(
                    patch=(), rationale=f"gen_error:{type(ex).__name__}")
            new_source, applied, _reason = apply_patch(
                buggy_source, proposed.patch, mode=apply_mode)
            if applied:
                # Patch_generator → test_runner handoff.
                router.emit(
                    source_role=ROLE_PATCH_GENERATOR,
                    source_agent_id=0,
                    claim_kind=CLAIM_PATCH_PROPOSED,
                    payload=f"{task.instance_id}:applied",
                    source_event_ids=(), round=2)
                n_handoffs += 1
                wr = run_patched_test(
                    file_source=buggy_source,
                    patched_source=new_source,
                    test_source=task.test_source,
                    module_name=f"patched_{task.instance_id}")
                # Test_runner → patch_generator handoff (verdict).
                verdict = "pass" if wr.test_passed else (
                    f"fail:{wr.error_kind}")
                router.emit(
                    source_role=ROLE_TEST_RUNNER,
                    source_agent_id=0,
                    claim_kind=CLAIM_TEST_RESULT,
                    payload=verdict, source_event_ids=(),
                    round=3)
                n_handoffs += 1
                error_kind = wr.error_kind
                test_passed = wr.test_passed
                patch_applied = True
            else:
                error_kind = "patch_no_match"
                test_passed = False
                patch_applied = False
            wall = time.time() - t0
            measurements.append(SWEMeasurement(
                instance_id=task.instance_id, repo=task.repo,
                strategy=strat, test_passed=test_passed,
                patch_applied=patch_applied,
                error_kind=error_kind,
                n_events_to_patch_gen=len(delivered),
                n_chars_patch_gen_prompt=len(prompt),
                n_handoffs=n_handoffs,
                n_events_total=len(all_events),
                log_length=router.log_length(),
                chain_ok=router.verify(),
                wall_seconds=wall))
    return SWEReport(
        n_tasks=len(bank),
        measurements=measurements,
        config={
            "strategies": list(strategies),
            "n_tasks": len(bank),
            "apply_mode": apply_mode,
        })


# =============================================================================
# MiniSWEBank — four hand-authored tasks
# =============================================================================


def _calc_bug_v0() -> str:
    """Tiny calculator with an off-by-one in ``factorial``."""
    return textwrap.dedent('''
        """Mini calculator module."""

        def add(a, b):
            return a + b

        def sub(a, b):
            return a - b

        def factorial(n):
            """Return n!. BUG: returns 1 for n=0 incorrectly via
            range(1, n+1) starting from 1 — that part is fine —
            but seeds with 0 instead of 1 so every result is 0."""
            result = 0
            for i in range(1, n + 1):
                result *= i
            return result

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        ''').strip() + "\n"


def _calc_test_v0() -> str:
    return textwrap.dedent('''
        def test(module):
            assert module.add(2, 3) == 5
            assert module.sub(10, 4) == 6
            assert module.factorial(0) == 1, "0! must be 1"
            assert module.factorial(1) == 1
            assert module.factorial(5) == 120
            assert module.is_prime(2) is True
            assert module.is_prime(9) is False
        ''').strip() + "\n"


def _strings_bug_v0() -> str:
    return textwrap.dedent('''
        """Mini string helper module."""

        def reverse(s):
            return s[::-1]

        def title_case(s):
            """Capitalise every word. BUG: lower-cases the first
            letter of each word instead of upper-casing it.
            """
            return " ".join(w[0].lower() + w[1:].lower()
                            for w in s.split() if w)

        def count_vowels(s):
            return sum(1 for c in s.lower() if c in "aeiou")
        ''').strip() + "\n"


def _strings_test_v0() -> str:
    return textwrap.dedent('''
        def test(module):
            assert module.reverse("abc") == "cba"
            assert module.title_case("hello world") == "Hello World"
            assert module.title_case("the quick brown fox") == \
                   "The Quick Brown Fox"
            assert module.count_vowels("Hello") == 2
            assert module.count_vowels("rhythm") == 0
        ''').strip() + "\n"


def _list_bug_v0() -> str:
    return textwrap.dedent('''
        """Mini list operations module."""

        def first(items):
            return items[0]

        def last(items):
            """Return the last item. BUG: off-by-one — returns the
            second-to-last when the list has ≥ 2 items."""
            if not items:
                raise IndexError("empty list")
            if len(items) == 1:
                return items[0]
            return items[-2]

        def unique(items):
            seen = set()
            out = []
            for x in items:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        ''').strip() + "\n"


def _list_test_v0() -> str:
    return textwrap.dedent('''
        def test(module):
            assert module.first([10, 20, 30]) == 10
            assert module.last([10, 20, 30]) == 30, "last must be 30"
            assert module.last([42]) == 42
            assert module.unique([1, 2, 2, 3, 1]) == [1, 2, 3]
        ''').strip() + "\n"


def _dict_bug_v0() -> str:
    return textwrap.dedent('''
        """Mini dict helpers."""

        def merge(a, b):
            """Return a new dict with a's keys overridden by b's.
            BUG: returns ``a`` directly mutated instead of a copy.
            """
            a.update(b)
            return a

        def keys_sorted(d):
            return sorted(d.keys())

        def value_count(d, value):
            return sum(1 for v in d.values() if v == value)
        ''').strip() + "\n"


def _dict_test_v0() -> str:
    return textwrap.dedent('''
        def test(module):
            a = {"x": 1, "y": 2}
            b = {"y": 20, "z": 30}
            merged = module.merge(a, b)
            assert merged == {"x": 1, "y": 20, "z": 30}
            assert a == {"x": 1, "y": 2}, (
                "merge must not mutate its first argument; got "
                + repr(a))
            assert module.keys_sorted({"b": 1, "a": 2}) == \
                   ["a", "b"]
            assert module.value_count({"a": 1, "b": 1, "c": 2}, 1) == 2
        ''').strip() + "\n"


def _make_event_log(*,
                     issue_text: str,
                     file_relpath: str,
                     buggy_function: str,
                     buggy_source: str,
                     n_distractors: int = 6,
                     ) -> list[tuple[str, str]]:
    """Compose a realistic raw-event stream visible to the four
    roles in different slices.

    ``n_distractors`` determines how much non-load-bearing chatter
    sits in the team's collective inbox.
    """
    out: list[tuple[str, str]] = []
    out.append((_EVENT_TYPE_ISSUE, issue_text))
    out.append((_EVENT_TYPE_FILE_HEADER,
                 f"path={file_relpath} loc=≈{buggy_source.count(chr(10))}"))
    fn_block = _slice_function(buggy_source, buggy_function,
                                 window=12)
    out.append((_EVENT_TYPE_FUNCTION_BODY,
                 f"name={buggy_function} body={fn_block!r}"))
    out.append((_EVENT_TYPE_TEST_FIXTURE,
                 "test_runner=in-process exec()"))
    distract_pool = [
        "ci_log: build green at 03:14:12",
        "lint_log: file passes ruff with 0 issues",
        "import_log: no circular imports detected",
        "coverage_log: 78.4% lines covered (file)",
        "git_log: last touched 7 days ago by alice",
        "pre_commit: black formatter clean",
        "vulnerability_scanner: 0 CVEs (mock)",
        "dependency_audit: pinned to numpy>=1.26",
        "doc_log: no docstring drift detected",
        "perf_log: function avg 0.001ms over 10000 calls",
        "issue_tracker: 3 related tickets closed as stale",
        "pr_discussion: reviewer approved",
        "sla_metric: uptime 99.97% last 30 days",
        "release_note: not scheduled for this release",
        "feature_flag: flag 'new_x' is off in prod",
        "config_audit: no drift from repo config",
        "log_volume: 12.3MB/day ingestion",
        "retention_policy: logs kept 90 days",
        "team_calendar: on-call rotates Friday",
        "dependency_graph: 24 inbound callers",
        "branch_policy: main protected, requires 2 approvals",
        "build_image: python:3.11-slim digest verified",
        "test_time: full suite 12.4s on laptop",
        "cold_start: import takes 18ms typical",
        "mem_footprint: module heap ≈ 0.4MB",
        "repl_cache: .pyc regenerated yesterday",
        "monkey_patch_audit: none detected",
        "cache_size: lru_cache keys=128 entries",
        "typing_check: mypy --strict passes",
        "api_version: v2.1 compatible",
    ]
    for i in range(min(n_distractors, len(distract_pool))):
        out.append((_EVENT_TYPE_DISTRACTOR, distract_pool[i]))
    return out


def build_mini_swe_bank(n_distractors: int = 6,
                          ) -> tuple[list[SWEBenchStyleTask],
                                     dict[str, str]]:
    """Build the four-instance mini SWE bank.

    Returns ``(tasks, repo_files)``. ``repo_files`` maps each
    ``buggy_file_relpath`` to the buggy file source string.
    """
    tasks: list[SWEBenchStyleTask] = []
    repo_files: dict[str, str] = {}

    def _emit(instance_id, repo, file_rel, fn, source, test,
                issue):
        repo_files[file_rel] = source
        ev = _make_event_log(
            issue_text=issue, file_relpath=file_rel,
            buggy_function=fn, buggy_source=source,
            n_distractors=n_distractors)
        # Gold patch derived from the bug doc (string substitution).
        return ev

    # Task 1 — calculator factorial seeds with 0 instead of 1
    src = _calc_bug_v0()
    issue = ("`factorial(n)` returns 0 for every n — "
              "the seed value is wrong. Expected: factorial(0)=1, "
              "factorial(5)=120.")
    ev = _emit("mini-swe-001", "mini-swe/calc",
                "calc.py", "factorial", src,
                _calc_test_v0(), issue)
    tasks.append(SWEBenchStyleTask(
        instance_id="mini-swe-001", repo="mini-swe/calc",
        base_commit="v0", problem_statement=issue,
        buggy_file_relpath="calc.py",
        buggy_function="factorial",
        gold_patch=(("    result = 0\n",
                     "    result = 1\n"),),
        test_source=_calc_test_v0(),
        hidden_event_log=tuple(ev)))

    # Task 2 — strings title_case lowercases first letter
    src = _strings_bug_v0()
    issue = ("`title_case(\"hello world\")` returns "
              "`'hello world'` instead of `'Hello World'`. "
              "Each word's first letter must be upper-cased, "
              "not lower-cased.")
    ev = _emit("mini-swe-002", "mini-swe/strings",
                "strings.py", "title_case", src,
                _strings_test_v0(), issue)
    tasks.append(SWEBenchStyleTask(
        instance_id="mini-swe-002", repo="mini-swe/strings",
        base_commit="v0", problem_statement=issue,
        buggy_file_relpath="strings.py",
        buggy_function="title_case",
        gold_patch=(("w[0].lower() + w[1:].lower()",
                     "w[0].upper() + w[1:].lower()"),),
        test_source=_strings_test_v0(),
        hidden_event_log=tuple(ev)))

    # Task 3 — list last returns penultimate
    src = _list_bug_v0()
    issue = ("`last([10, 20, 30])` returns `20` but it should "
              "return `30`. The function returns the wrong index "
              "when the list has ≥ 2 items.")
    ev = _emit("mini-swe-003", "mini-swe/list",
                "listops.py", "last", src,
                _list_test_v0(), issue)
    tasks.append(SWEBenchStyleTask(
        instance_id="mini-swe-003", repo="mini-swe/list",
        base_commit="v0", problem_statement=issue,
        buggy_file_relpath="listops.py",
        buggy_function="last",
        gold_patch=(("return items[-2]", "return items[-1]"),),
        test_source=_list_test_v0(),
        hidden_event_log=tuple(ev)))

    # Task 4 — dict merge mutates first argument
    src = _dict_bug_v0()
    issue = ("`merge(a, b)` mutates `a` in place; callers expect "
              "`a` to be unchanged after the call. Return a new "
              "dict instead.")
    ev = _emit("mini-swe-004", "mini-swe/dict",
                "dictops.py", "merge", src,
                _dict_test_v0(), issue)
    tasks.append(SWEBenchStyleTask(
        instance_id="mini-swe-004", repo="mini-swe/dict",
        base_commit="v0", problem_statement=issue,
        buggy_file_relpath="dictops.py",
        buggy_function="merge",
        gold_patch=(("    a.update(b)\n    return a\n",
                     "    out = dict(a)\n    out.update(b)\n"
                     "    return out\n"),),
        test_source=_dict_test_v0(),
        hidden_event_log=tuple(ev)))
    return tasks, repo_files


# =============================================================================
# SWE-bench external adapter shim
# =============================================================================


@dataclass
class SWEBenchAdapter:
    """Adapter shim for loading external SWE-bench instances.

    Phase 39 shipped this adapter as a schema-mapping shim;
    Phase 40 promotes it to a real loader path. Two forms of
    input are now supported:

      * ``from_dict(d)`` — a dict whose ``gold_patch`` is either
        the bridge's native ``[(old, new), ...]`` shape or a
        unified-diff string. Diff parsing is delegated to
        ``parse_unified_diff`` and applied against ``repo_files``
        (when provided) so the resulting substitutions are
        guaranteed unique-match against the buggy source.
      * ``from_swe_bench_dict(d, repo_files)`` — a real
        SWE-bench-shaped dict (``patch``, ``test_patch``,
        optional ``problem_statement``, …). The adapter
        derives ``buggy_file_relpath`` and ``buggy_function``
        from the diff when not supplied explicitly.

    Both paths return a fully-typed ``SWEBenchStyleTask`` that
    flows through the unchanged ``run_swe_loop``. The loader
    side does no network access; loading a JSONL artifact is
    the responsibility of ``load_jsonl_bank`` below.
    """

    @staticmethod
    def from_dict(d: dict, *, repo_files: dict[str, str] | None = None,
                  hidden_event_log: Sequence[tuple[str, str]] = (),
                  ) -> SWEBenchStyleTask:
        """Build a ``SWEBenchStyleTask`` from a dict whose keys are
        a subset of the SWE-bench schema. Required keys:

          ``instance_id``, ``repo``, ``base_commit``,
          ``problem_statement``, ``buggy_file_relpath``,
          ``buggy_function``, ``gold_patch``, ``test_source``.

        ``gold_patch`` is either a list of ``[old, new]`` pairs
        (this module's substitution shape) or a unified-diff
        string. When a unified diff is supplied, ``repo_files``
        must contain the buggy file source so the diff hunks
        can be anchored to unique substitution windows.
        """
        gp = d.get("gold_patch")
        if isinstance(gp, str):
            relpath = str(d.get("buggy_file_relpath", ""))
            if not repo_files or relpath not in repo_files:
                raise ValueError(
                    "unified-diff gold_patch requires "
                    "repo_files[buggy_file_relpath] to anchor "
                    "the substitution windows; got repo_files="
                    f"{None if repo_files is None else sorted(repo_files)}, "
                    f"buggy_file_relpath={relpath!r}.")
            file_patches = parse_unified_diff(gp)
            subs = file_patches.get(relpath)
            if subs is None:
                raise ValueError(
                    f"unified diff does not touch declared "
                    f"buggy_file_relpath={relpath!r}; diff "
                    f"covers {sorted(file_patches)}.")
            gp_pairs: tuple[tuple[str, str], ...] = subs
        else:
            gp_pairs = tuple((str(o), str(n)) for (o, n) in gp)
        return SWEBenchStyleTask(
            instance_id=str(d["instance_id"]),
            repo=str(d["repo"]),
            base_commit=str(d.get("base_commit", "v0")),
            problem_statement=str(d["problem_statement"]),
            buggy_file_relpath=str(d["buggy_file_relpath"]),
            buggy_function=str(d["buggy_function"]),
            gold_patch=gp_pairs,
            test_source=str(d["test_source"]),
            hidden_event_log=tuple(hidden_event_log),
        )

    @staticmethod
    def from_swe_bench_dict(d: dict, *,
                              repo_files: dict[str, str],
                              hidden_event_log: Sequence[tuple[str, str]] = (),
                              ) -> SWEBenchStyleTask:
        """Build a task from a real SWE-bench-shaped dict.

        Real SWE-bench instances ship the gold patch as a
        unified-diff string under key ``patch`` (and optionally a
        ``test_patch``). The adapter:

          * parses ``patch`` with ``parse_unified_diff``;
          * if the diff touches a single Python file, derives
            ``buggy_file_relpath`` from the diff header (unless
            an explicit override is in ``d``);
          * scans the buggy hunk for the enclosing ``def`` /
            ``async def`` to derive ``buggy_function`` (unless
            an explicit override is in ``d``);
          * promotes ``test_patch`` (if present) into
            ``test_source`` by extracting the *added* test body;
            otherwise expects ``d["test_source"]`` to be set.

        ``repo_files`` is the in-memory snapshot of the repo at
        ``base_commit``; the caller is responsible for materialising
        it (in tests, a 1- or 2-file dict suffices; in a real
        pipeline, the sandbox runner stages a checkout into a
        temp directory and reads files from there).
        """
        # Normalise the patch into per-file substitutions.
        diff_str = d.get("patch") or d.get("gold_patch") or ""
        if not isinstance(diff_str, str):
            # Already in the bridge's substitution shape — fall
            # through to from_dict.
            return SWEBenchAdapter.from_dict(
                d, repo_files=repo_files,
                hidden_event_log=hidden_event_log)
        file_patches = parse_unified_diff(diff_str)
        if not file_patches:
            raise ValueError(
                f"empty / unparseable unified diff for instance "
                f"{d.get('instance_id', '<unknown>')!r}: "
                f"{diff_str[:120]!r}…")
        relpath = str(d.get("buggy_file_relpath")
                       or _pick_primary_file(file_patches))
        if relpath not in file_patches:
            raise ValueError(
                f"buggy_file_relpath={relpath!r} not in diff; "
                f"diff covers {sorted(file_patches)}.")
        if relpath not in repo_files:
            raise ValueError(
                f"repo_files missing source for {relpath!r}; "
                f"have {sorted(repo_files)}.")
        subs = file_patches[relpath]
        # Derive function name from the first hunk header if not
        # supplied; fall back to the parsed @@ header function.
        function = str(d.get("buggy_function")
                        or _derive_function_name(repo_files[relpath], subs)
                        or "<unknown>")
        # Promote a test_patch into test_source if needed.
        test_source = str(d.get("test_source", "")) or _test_from_patch(
            d.get("test_patch", ""))
        if not test_source:
            raise ValueError(
                f"instance {d.get('instance_id', '<unknown>')!r} has "
                "neither test_source nor a usable test_patch")
        return SWEBenchStyleTask(
            instance_id=str(d["instance_id"]),
            repo=str(d.get("repo", "external/unknown")),
            base_commit=str(d.get("base_commit", "v0")),
            problem_statement=str(d.get("problem_statement", "")),
            buggy_file_relpath=relpath,
            buggy_function=function,
            gold_patch=subs,
            test_source=test_source,
            hidden_event_log=tuple(hidden_event_log),
        )


# =============================================================================
# Phase 40 — unified-diff parser + JSONL loader (real-task path)
# =============================================================================


_DIFF_HEADER_OLD = re.compile(r"^---\s+(?:a/)?(\S+)")
_DIFF_HEADER_NEW = re.compile(r"^\+\+\+\s+(?:b/)?(\S+)")
_DIFF_HUNK = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$")


def parse_unified_diff(diff_text: str
                        ) -> dict[str, tuple[tuple[str, str], ...]]:
    """Parse a unified diff into per-file ``(old, new)`` substitution
    tuples suitable for ``apply_patch``.

    The parser handles the standard SWE-bench output of
    ``git diff``: ``--- a/<path>`` / ``+++ b/<path>`` headers,
    one or more ``@@ -lo,llen +ro,rlen @@`` hunks, with
    ``" "`` (context), ``"-"`` (removed), and ``"+"`` (added)
    body lines. Each hunk becomes one ``(old_block, new_block)``
    pair; hunks within a file are returned in document order so
    ``apply_patch``'s left-to-right substitution stays valid.

    The parser is intentionally tolerant of:
      * ``\\ No newline at end of file`` markers (skipped);
      * file headers without an ``a/`` / ``b/`` prefix;
      * hunks whose context-only lines outnumber the
        change lines (unique-match shrinkage handled by
        ``_shrink_to_unique`` if the hunk is degenerate).

    The parser deliberately does NOT handle:
      * binary diffs (returns no entries for them);
      * file create / delete (the substitution model assumes
        in-place edit; create/delete map to ``apply_patch``'s
        ``empty_patch`` regime — which the runner will surface
        as ``patch_no_match``).

    Returns a dict ``{relpath: ((old, new), ...)}``.
    """
    if not diff_text:
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    cur_path: str | None = None
    cur_old: list[str] = []
    cur_new: list[str] = []
    in_hunk = False

    def _flush_hunk():
        nonlocal cur_old, cur_new
        if cur_path is None:
            cur_old = []
            cur_new = []
            return
        old_block = "".join(cur_old)
        new_block = "".join(cur_new)
        # An empty old means a pure addition with no anchor; an
        # empty new means a pure deletion. Both are valid in
        # unified-diff syntax, but apply_patch needs old to be
        # non-empty AND unique. We retain only hunks whose old
        # is non-empty; pure additions cannot be substituted
        # without context anchoring (caller's burden).
        if old_block and old_block != new_block:
            out.setdefault(cur_path, []).append((old_block, new_block))
        cur_old = []
        cur_new = []

    for raw in diff_text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        # File headers reset the per-file accumulator.
        m_old = _DIFF_HEADER_OLD.match(line)
        if m_old is not None:
            _flush_hunk()
            in_hunk = False
            continue
        m_new = _DIFF_HEADER_NEW.match(line)
        if m_new is not None:
            _flush_hunk()
            cur_path = m_new.group(1)
            in_hunk = False
            continue
        m_hunk = _DIFF_HUNK.match(line)
        if m_hunk is not None:
            _flush_hunk()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("\\ "):
            # "\ No newline at end of file" — informational.
            continue
        if line.startswith("-"):
            cur_old.append(line[1:] + "\n")
            continue
        if line.startswith("+"):
            cur_new.append(line[1:] + "\n")
            continue
        # Context line — present in both old and new.
        body = line[1:] if line.startswith(" ") else line
        cur_old.append(body + "\n")
        cur_new.append(body + "\n")
    _flush_hunk()
    return {p: tuple(s) for p, s in out.items()}


def _pick_primary_file(file_patches: dict
                        ) -> str:
    """When the caller did not specify ``buggy_file_relpath``,
    pick the file with the most substitutions; tie-break by
    lexicographic order so the choice is deterministic.
    """
    best = None
    best_count = -1
    for path, subs in sorted(file_patches.items()):
        if len(subs) > best_count:
            best = path
            best_count = len(subs)
    return best or ""


_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _derive_function_name(source: str,
                            subs: Sequence[tuple[str, str]]) -> str:
    """Find the function name the first substitution touches.

    Lookup order:
      1. inside the substitution's ``old`` block (the diff hunk
         frequently includes the ``def`` line as context, e.g.
         ``@@ ... @@`` followed by a context ``def foo():`` line);
      2. the *enclosing* ``def`` walking backward through ``source``
         from the substitution's anchor;
      3. otherwise ``""`` (caller records ``"<unknown>"``).

    Returns the bare function identifier, not its qualified name.
    """
    if not subs:
        return ""
    old = subs[0][0]
    # (1) Look inside the old block first.
    for line in old.splitlines():
        m = _DEF_RE.match(line)
        if m:
            return m.group(1)
    # (2) Walk backward through the source to find the enclosing def.
    idx = source.find(old)
    if idx < 0:
        return ""
    head = source[:idx].splitlines()
    for line in reversed(head):
        m = _DEF_RE.match(line)
        if m:
            return m.group(1)
    return ""


def _test_from_patch(test_patch: str) -> str:
    """Recover a runnable test source from a SWE-bench-style
    ``test_patch``. SWE-bench's ``test_patch`` is itself a unified
    diff; we extract its added (``+``) lines as a single Python
    module. The bridge contract requires ``def test(module):``;
    the caller is responsible for adapting a real SWE-bench
    pytest body if needed (the Phase 40 loader for the
    ``swe_bench_lite``-shaped JSONL pre-wraps tests on its way
    in — see ``load_jsonl_bank``).
    """
    if not test_patch:
        return ""
    added: list[str] = []
    in_hunk = False
    for line in test_patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+++") or line.startswith("---"):
            in_hunk = False
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added) + ("\n" if added else "")


def load_jsonl_bank(path: str,
                     *,
                     repo_files_resolver: Callable[[dict],
                                                     dict[str, str]] | None = None,
                     limit: int | None = None,
                     hidden_event_log_factory:
                         Callable[[SWEBenchStyleTask], Sequence[tuple[str, str]]] | None = None,
                     ) -> tuple[list[SWEBenchStyleTask], dict[str, str]]:
    """Load a JSONL file of SWE-bench-style instances into a
    bridge bank.

    Each line must be a JSON object compatible with
    ``SWEBenchAdapter.from_swe_bench_dict``. The instance may
    embed its repo snapshot inline (``repo_files`` key — a dict
    of relpath → source), or the caller may pass
    ``repo_files_resolver`` to materialise files from an
    external store (e.g. a git checkout).

    Returns ``(tasks, pooled_repo_files)``. Per-instance file
    namespaces are isolated by prefixing each relpath with
    ``f"{instance_id}/"`` so two instances editing
    ``"src/x.py"`` do not collide in the pooled dict.

    The loader is hermetic on the local file path it is
    pointed at — no network calls, no shell.
    """
    tasks: list[SWEBenchStyleTask] = []
    pooled: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(
                    f"jsonl row must be a JSON object, got {type(row).__name__}")
            inline = row.get("repo_files") or {}
            if not inline and repo_files_resolver is not None:
                inline = repo_files_resolver(row)
            if not isinstance(inline, dict):
                raise ValueError(
                    "repo_files must be a {relpath: source} dict")
            instance_id = str(row.get("instance_id") or
                              f"row-{len(tasks):04d}")
            # Namespace per-instance files so two instances editing
            # the same logical relpath cannot collide in the pool.
            prefix = f"{instance_id}/"
            namespaced_files = {prefix + k: v for k, v in inline.items()}
            # Rebuild the row so the adapter sees the namespaced
            # buggy file path; the original relpath (unprefixed)
            # is what appears inside the unified diff, so we map
            # it back when the diff is applied.
            row_for_adapter = dict(row)
            if "buggy_file_relpath" in row_for_adapter:
                row_for_adapter["buggy_file_relpath"] = (
                    prefix + row_for_adapter["buggy_file_relpath"])
            # Re-anchor the diff path headers to the namespaced
            # paths so the parser yields keys that exist in
            # `namespaced_files`.
            patch = row_for_adapter.get("patch") or row_for_adapter.get("gold_patch")
            if isinstance(patch, str) and inline:
                row_for_adapter["patch"] = _renamespace_diff(
                    patch, prefix, set(inline.keys()))
            task = SWEBenchAdapter.from_swe_bench_dict(
                row_for_adapter, repo_files=namespaced_files,
                hidden_event_log=())
            if hidden_event_log_factory is not None:
                ev = tuple(hidden_event_log_factory(task))
                # Re-pack with the new event log (frozen dataclass).
                task = SWEBenchStyleTask(
                    instance_id=task.instance_id, repo=task.repo,
                    base_commit=task.base_commit,
                    problem_statement=task.problem_statement,
                    buggy_file_relpath=task.buggy_file_relpath,
                    buggy_function=task.buggy_function,
                    gold_patch=task.gold_patch,
                    test_source=task.test_source,
                    hidden_event_log=ev,
                )
            tasks.append(task)
            pooled.update(namespaced_files)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks, pooled


def _renamespace_diff(diff_text: str, prefix: str,
                       known_paths: set[str]) -> str:
    """Rewrite ``--- a/<p>`` / ``+++ b/<p>`` headers to
    ``--- a/<prefix><p>`` / ``+++ b/<prefix><p>`` for every
    ``<p>`` in ``known_paths``. Diff hunks are unchanged.

    This preserves diff semantics (both sides shift in
    lockstep) while letting the JSONL loader pool files from
    multiple instances into a single bridge dict without
    name collisions.
    """
    out_lines: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("--- a/") or stripped.startswith("+++ b/"):
            head, _, rest = stripped.partition("/")
            head_prefix = head + "/"
            path = rest
            if path in known_paths:
                out_lines.append(head_prefix + prefix + path + "\n")
                continue
        out_lines.append(line if line.endswith("\n") else line + "\n")
    return "".join(out_lines)


def build_synthetic_event_log(task: SWEBenchStyleTask,
                                n_distractors: int = 6
                                ) -> list[tuple[str, str]]:
    """Default ``hidden_event_log_factory`` for the JSONL loader.

    Composes a realistic raw-event stream visible to the four
    bridge roles: one ISSUE event (the problem statement), one
    FILE_HEADER event, one FUNCTION_BODY event (sliced from the
    buggy file), one TEST_FIXTURE event, and ``n_distractors``
    DISTRACTOR events. Mirrors the mini-bank composition shape
    so naive / routing / substrate measurements remain
    comparable across the mini bank and the JSONL bank.
    """
    # The instance's buggy_file_relpath is namespaced; strip the
    # prefix for the FILE_HEADER body so the role observable
    # filter sees the shape it expects.
    rel = task.buggy_file_relpath
    if "/" in rel:
        rel = rel.split("/", 1)[1]
    return _make_event_log(
        issue_text=task.problem_statement,
        file_relpath=rel,
        buggy_function=task.buggy_function,
        buggy_source=task.test_source,
        n_distractors=n_distractors,
    )


# =============================================================================
# Module exports
# =============================================================================


__all__ = [
    # Roles + claims
    "ROLE_ISSUE_READER", "ROLE_CODE_SEARCHER",
    "ROLE_PATCH_GENERATOR", "ROLE_TEST_RUNNER", "ALL_SWE_ROLES",
    "CLAIM_ISSUE_PARSED", "CLAIM_FILE_LOCATED",
    "CLAIM_HUNK_LOCATED", "CLAIM_PATCH_PROPOSED",
    "CLAIM_TEST_RESULT", "ALL_SWE_CLAIM_KINDS",
    # Strategies
    "STRATEGY_NAIVE", "STRATEGY_ROUTING", "STRATEGY_SUBSTRATE",
    "ALL_SWE_STRATEGIES",
    # Schema + workspace
    "SWEBenchStyleTask", "WorkspaceResult",
    "apply_patch", "run_patched_test",
    # Phase 41 — matcher modes
    "APPLY_MODE_STRICT", "APPLY_MODE_LSTRIP",
    "APPLY_MODE_WS_COLLAPSE", "APPLY_MODE_LINE_ANCHORED",
    "ALL_APPLY_MODES",
    # Subscriptions + runner
    "build_swe_role_subscriptions", "run_swe_loop",
    "SWEMeasurement", "SWEReport",
    "ProposedPatch", "PatchGenerator",
    "deterministic_oracle_generator", "llm_patch_generator",
    "build_patch_generator_prompt",
    # Mini bank + external adapter
    "build_mini_swe_bank", "SWEBenchAdapter",
    "role_observable",
    # Phase 40 — real-task loader path
    "parse_unified_diff", "load_jsonl_bank",
    "build_synthetic_event_log",
]


# =============================================================================
# Phase 42 — re-export parser-compliance symbols so the bridge
# surface stays one import.
# =============================================================================


from .swe_patch_parser import (  # noqa: E402
    ALL_PARSE_KINDS as ALL_PARSE_KINDS,
    ALL_PARSER_MODES as ALL_PARSER_MODES,
    PARSER_ROBUST as PARSER_ROBUST,
    PARSER_STRICT as PARSER_STRICT,
    PARSER_UNIFIED as PARSER_UNIFIED,
    ParseOutcome as ParseOutcome,
    ParserComplianceCounter as ParserComplianceCounter,
    parse_patch_block as parse_patch_block,
)

__all__ += [
    "PARSER_ROBUST", "PARSER_STRICT", "PARSER_UNIFIED",
    "ALL_PARSER_MODES", "ALL_PARSE_KINDS",
    "ParseOutcome", "ParserComplianceCounter",
    "parse_patch_block",
]
