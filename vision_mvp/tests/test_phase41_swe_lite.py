"""Phase 41 — larger SWE-bench-Lite-style bank + permissive matcher tests.

Coverage:
  * strict apply_patch behaviour is byte-identical to Phase 40 (no
    regression on the default matcher);
  * permissive matcher modes recover indentation drift / internal
    whitespace drift / trailing whitespace drift on narrow cases;
  * permissive matchers still refuse ambiguous anchors;
  * the Phase-41 JSONL bank loads cleanly and the oracle saturates
    pass@1 on every instance through the subprocess sandbox;
  * bounded-context preservation holds at the larger bank size
    (Theorem P41-1);
  * apply_mode is threaded through run_swe_loop and
    run_swe_loop_sandboxed and recorded in the report config.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_APPLY_MODES, ALL_SWE_STRATEGIES, APPLY_MODE_LINE_ANCHORED,
    APPLY_MODE_LSTRIP, APPLY_MODE_STRICT, APPLY_MODE_WS_COLLAPSE,
    STRATEGY_NAIVE, STRATEGY_SUBSTRATE, apply_patch,
    build_synthetic_event_log, deterministic_oracle_generator,
    load_jsonl_bank, run_swe_loop,
)
from vision_mvp.tasks.swe_sandbox import (
    SubprocessSandbox, run_swe_loop_sandboxed,
)


_BANK_JSONL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl"))


# -----------------------------------------------------------------
# 1. Strict matcher regression — Phase-40 byte-exact semantics
# -----------------------------------------------------------------


def test_strict_mode_byte_exact_baseline():
    src = "def f():\n    return 0\n"
    p = (("    return 0\n", "    return 1\n"),)
    out, ok, r = apply_patch(src, p)  # default mode = strict
    assert ok and r == ""
    assert out == "def f():\n    return 1\n"


def test_strict_mode_refuses_missing_anchor():
    src = "x = 1\n"
    out, ok, r = apply_patch(src, (("nope\n", "yep\n"),),
                              mode=APPLY_MODE_STRICT)
    assert not ok
    assert r == "old_not_found"


def test_strict_mode_refuses_ambiguous_anchor():
    src = "x = 0\ny = 1\nx = 0\n"
    out, ok, r = apply_patch(src, (("x = 0\n", "x = 2\n"),),
                              mode=APPLY_MODE_STRICT)
    assert not ok
    assert r == "old_ambiguous"


# -----------------------------------------------------------------
# 2. Permissive matchers — narrow recovery surface
# -----------------------------------------------------------------


def test_lstrip_mode_recovers_indentation_drift():
    """Generator emitted 2-space indent, source has 4-space."""
    src = "def f():\n    if x:\n        return 0\n"
    old = "  if x:\n      return 0\n"
    new = "    if x:\n        return 1\n"
    # strict: the 2-space-indent OLD is NOT a byte-substring of the
    # 4-space-indent source (the 2-space prefix doesn't sit at the
    # right offset for str.replace to match cleanly as a run of
    # tokens).
    _, ok_strict, _ = apply_patch(src, ((old, new),),
                                    mode=APPLY_MODE_STRICT)
    assert not ok_strict
    out, ok, r = apply_patch(src, ((old, new),),
                              mode=APPLY_MODE_LSTRIP)
    assert ok, r
    assert "return 1" in out


def test_ws_collapse_mode_recovers_internal_whitespace_drift():
    src = "def f(x, y):\n    return x + y\n"
    old = "    return x +  y\n"   # double space before y
    new = "    return x * y\n"
    _, ok_strict, _ = apply_patch(src, ((old, new),),
                                    mode=APPLY_MODE_STRICT)
    assert not ok_strict
    out, ok, _ = apply_patch(src, ((old, new),),
                              mode=APPLY_MODE_WS_COLLAPSE)
    assert ok
    assert "return x * y" in out


def test_line_anchored_mode_recovers_trailing_whitespace_drift():
    src = "x = 0\n"
    old = "x = 0 \n"              # trailing space on OLD
    new = "x = 1\n"
    _, ok_strict, _ = apply_patch(src, ((old, new),),
                                    mode=APPLY_MODE_STRICT)
    assert not ok_strict
    out, ok, _ = apply_patch(src, ((old, new),),
                              mode=APPLY_MODE_LINE_ANCHORED)
    assert ok
    assert out == "x = 1\n"


def test_permissive_mode_refuses_ambiguous_match():
    src = "x = 0\ny = 0\nx = 0\n"
    out, ok, r = apply_patch(src, (("x = 0\n", "x = 2\n"),),
                              mode=APPLY_MODE_LSTRIP)
    assert not ok
    assert r == "old_ambiguous"


def test_unknown_apply_mode_surfaces_cleanly():
    src = "x = 0\n"
    out, ok, r = apply_patch(src, (("x = 0\n", "x = 1\n"),),
                              mode="definitely-not-a-mode")
    assert not ok
    assert r == "unknown_mode"


def test_permissive_modes_enumeration_contains_expected():
    assert APPLY_MODE_STRICT in ALL_APPLY_MODES
    assert APPLY_MODE_LSTRIP in ALL_APPLY_MODES
    assert APPLY_MODE_WS_COLLAPSE in ALL_APPLY_MODES
    assert APPLY_MODE_LINE_ANCHORED in ALL_APPLY_MODES


# -----------------------------------------------------------------
# 3. Over-acceptance guard — permissive must not match semantic drift
# -----------------------------------------------------------------


def test_lstrip_mode_does_not_match_semantic_drift():
    """lstrip must NOT treat two semantically-different lines as
    equal. Two lines with the same lstrip projection but different
    identifiers are *different*.
    """
    src = "x = 0\ny = 99\n"
    # generator-emitted OLD looks like a valid lstrip-normalised
    # version of a *different* line from the source:
    out, ok, r = apply_patch(
        src, (("z = 0\n", "z = 1\n"),), mode=APPLY_MODE_LSTRIP)
    assert not ok
    assert r == "old_not_found"


def test_ws_collapse_does_not_drop_commented_lines():
    """ws_collapse must not equate a commented-out line with its
    uncommented counterpart after whitespace normalisation.
    """
    src = "x = 0\n# x = 1\n"
    out, ok, _ = apply_patch(
        src, (("x = 1\n", "x = 2\n"),), mode=APPLY_MODE_WS_COLLAPSE)
    # The OLD "x = 1\n" is not a line in src (the matching line in
    # src starts with "# "); normalisation keeps the "#" so the
    # two are distinct.
    assert not ok


# -----------------------------------------------------------------
# 4. JSONL bank — loader smoke + oracle ceiling at scale
# -----------------------------------------------------------------


def test_phase41_jsonl_loads_expected_instance_count():
    tasks, files = load_jsonl_bank(
        _BANK_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 2))
    # Bank should have substantially more instances than Phase 40's
    # 6-instance mini bank so the law-of-large-numbers argument
    # applies.
    assert len(tasks) >= 20
    for t in tasks:
        assert t.gold_patch
        assert "def test(" in t.test_source
        assert t.buggy_file_relpath in files


def test_phase41_jsonl_oracle_saturates_under_sandbox_strict():
    tasks, files = load_jsonl_bank(
        _BANK_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 4))
    sb = SubprocessSandbox()
    rep = run_swe_loop_sandboxed(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        sandbox=sb, strategies=ALL_SWE_STRATEGIES, timeout_s=15.0,
        apply_mode=APPLY_MODE_STRICT)
    pooled = rep.pooled_summary()
    for strat in ALL_SWE_STRATEGIES:
        assert pooled[strat]["pass_at_1"] == 1.0, (strat, pooled)


def test_phase41_jsonl_oracle_saturates_under_sandbox_lstrip():
    """Theorem P41-2 infrastructure: the oracle still saturates
    pass@1 = 1.0 under a permissive matcher — permissive cannot
    *remove* correctness from a byte-exact patch path.
    """
    tasks, files = load_jsonl_bank(
        _BANK_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 4))
    sb = SubprocessSandbox()
    rep = run_swe_loop_sandboxed(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        sandbox=sb, strategies=ALL_SWE_STRATEGIES, timeout_s=15.0,
        apply_mode=APPLY_MODE_LSTRIP)
    pooled = rep.pooled_summary()
    for strat in ALL_SWE_STRATEGIES:
        assert pooled[strat]["pass_at_1"] == 1.0, (strat, pooled)


# -----------------------------------------------------------------
# 5. Bounded-context preservation at scale (Theorem P41-1)
# -----------------------------------------------------------------


def test_phase41_substrate_prompt_constant_across_distractors():
    """Theorem P41-1: on the larger bank, substrate prompt chars are
    constant under distractor sweep; naive grows.
    """
    sb = SubprocessSandbox()
    per_nd_sub: dict[int, float] = {}
    per_nd_naive: dict[int, float] = {}
    for nd in (0, 6, 24):
        tasks, files = load_jsonl_bank(
            _BANK_JSONL,
            hidden_event_log_factory=lambda t, k=nd: build_synthetic_event_log(t, k))
        rep = run_swe_loop_sandboxed(
            bank=tasks, repo_files=files,
            generator=deterministic_oracle_generator, sandbox=sb,
            strategies=(STRATEGY_SUBSTRATE, STRATEGY_NAIVE),
            timeout_s=10.0)
        pooled = rep.pooled_summary()
        per_nd_sub[nd] = pooled[STRATEGY_SUBSTRATE]["mean_patch_gen_prompt_chars"]
        per_nd_naive[nd] = pooled[STRATEGY_NAIVE]["mean_patch_gen_prompt_chars"]
    # Substrate is flat across the distractor axis.
    assert per_nd_sub[0] == per_nd_sub[6] == per_nd_sub[24]
    # Naive grows monotonically.
    assert per_nd_naive[0] < per_nd_naive[6] < per_nd_naive[24]


# -----------------------------------------------------------------
# 6. apply_mode threading through the runner surfaces in config
# -----------------------------------------------------------------


def test_apply_mode_recorded_in_run_swe_loop_config():
    tasks, files = load_jsonl_bank(
        _BANK_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 1),
        limit=2)
    rep = run_swe_loop(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        strategies=(STRATEGY_SUBSTRATE,),
        apply_mode=APPLY_MODE_LSTRIP)
    assert rep.config["apply_mode"] == APPLY_MODE_LSTRIP


def test_apply_mode_recorded_in_sandboxed_run_config():
    tasks, files = load_jsonl_bank(
        _BANK_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 1),
        limit=2)
    sb = SubprocessSandbox()
    rep = run_swe_loop_sandboxed(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        sandbox=sb, strategies=(STRATEGY_SUBSTRATE,),
        timeout_s=10.0, apply_mode=APPLY_MODE_LSTRIP)
    assert rep.config["apply_mode"] == APPLY_MODE_LSTRIP


# -----------------------------------------------------------------
# 7. Permissive matcher + bridge — end-to-end sandbox path
# -----------------------------------------------------------------


def test_permissive_matcher_end_to_end_recovers_drifted_patch():
    """Assemble an explicit indentation-drift patch and verify that
    strict rejects while lstrip accepts through the sandbox boundary.
    """
    sb = SubprocessSandbox()
    src = "def f():\n    if x:\n        return 0\n"
    # Generator-emitted OLD drops 2 spaces of indent from both lines.
    patch = (("  if x:\n      return 0\n",
              "    if x:\n        return 1\n"),)
    test_source = (
        "def test(module):\n"
        "    assert module.f.__code__.co_varnames == ()\n")  # type: ignore
    r_strict = sb.run(buggy_source=src, patch=patch,
                       test_source=test_source, module_name="m",
                       timeout_s=5.0, apply_mode=APPLY_MODE_STRICT)
    assert not r_strict.patch_applied
    # lstrip must recover the patch and it must at least apply.
    r_perm = sb.run(buggy_source=src, patch=patch,
                     test_source=test_source, module_name="m",
                     timeout_s=5.0, apply_mode=APPLY_MODE_LSTRIP)
    assert r_perm.patch_applied
