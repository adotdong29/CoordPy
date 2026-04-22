"""Phase 39 — SWE-bench-style bridge tests.

Coverage:
  * Bank construction is hermetic (no disk I/O).
  * Patch application semantics (apply_patch).
  * Hidden-test execution sandboxing (run_patched_test).
  * Substrate-driven runner produces reproducible measurements
    across naive / routing / substrate strategies.
  * Hash-chained log invariant survives every strategy.
  * Bounded-context invariant on the patch_generator role
    (Theorem P31-3 / P39-1 shape) — substrate prompt size is
    independent of n_distractors.
  * SWEBenchAdapter.from_dict round-trips a SWE-bench-shaped dict.
"""

from __future__ import annotations

import textwrap

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_SWE_STRATEGIES, CLAIM_HUNK_LOCATED, CLAIM_ISSUE_PARSED,
    CLAIM_PATCH_PROPOSED, CLAIM_TEST_RESULT, ProposedPatch,
    ROLE_PATCH_GENERATOR, ROLE_TEST_RUNNER,
    STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE,
    SWEBenchAdapter, SWEBenchStyleTask, apply_patch,
    build_mini_swe_bank, build_swe_role_subscriptions,
    deterministic_oracle_generator, run_patched_test,
    run_swe_loop,
)


# -----------------------------------------------------------------
# 1. Bank
# -----------------------------------------------------------------


def test_mini_bank_has_four_runnable_instances():
    tasks, files = build_mini_swe_bank(n_distractors=4)
    assert len(tasks) == 4
    assert len(files) == 4
    for t in tasks:
        assert t.instance_id.startswith("mini-swe-")
        assert t.gold_patch
        assert t.test_source
        assert t.buggy_file_relpath in files
        # Hidden test source must define ``def test(module): ...``.
        assert "def test(" in t.test_source


def test_default_role_subscriptions_cover_all_claim_kinds():
    s = build_swe_role_subscriptions()
    pairs = set(s.all_pairs())
    # Each claim kind has at least one declared producer.
    found_kinds = {k for (_role, k) in pairs}
    assert CLAIM_ISSUE_PARSED in found_kinds
    assert CLAIM_HUNK_LOCATED in found_kinds
    assert CLAIM_PATCH_PROPOSED in found_kinds
    assert CLAIM_TEST_RESULT in found_kinds


# -----------------------------------------------------------------
# 2. Patch + test sandbox
# -----------------------------------------------------------------


def test_apply_patch_substitutes_unique_match():
    src = "def f():\n    return 0\n"
    new, ok, _ = apply_patch(src, [("return 0", "return 1")])
    assert ok and new == "def f():\n    return 1\n"


def test_apply_patch_refuses_ambiguous_match():
    src = "x = 1\nx = 1\n"
    _, ok, reason = apply_patch(src, [("x = 1", "x = 2")])
    assert not ok and reason == "old_ambiguous"


def test_apply_patch_refuses_missing_old():
    src = "def f(): return 0\n"
    _, ok, reason = apply_patch(src, [("notinfile", "x")])
    assert not ok and reason == "old_not_found"


def test_run_patched_test_passes_on_correct_patch():
    src = "def f(): return 0\n"
    test = textwrap.dedent("""
        def test(module):
            assert module.f() == 1
    """)
    new, ok, _ = apply_patch(src, [("return 0", "return 1")])
    assert ok
    res = run_patched_test(file_source=src, patched_source=new,
                            test_source=test)
    assert res.test_passed
    assert res.error_kind == ""


def test_run_patched_test_reports_assertion_failure():
    src = "def f(): return 0\n"
    test = textwrap.dedent("""
        def test(module):
            assert module.f() == 99
    """)
    res = run_patched_test(file_source=src, patched_source=src,
                            test_source=test)
    assert not res.test_passed
    assert res.error_kind == "test_assert"


def test_run_patched_test_catches_syntax_error():
    bad_src = "def f( return 0\n"
    test = "def test(module): pass\n"
    res = run_patched_test(file_source=bad_src,
                            patched_source=bad_src,
                            test_source=test)
    assert not res.test_passed
    assert res.error_kind == "syntax"


# -----------------------------------------------------------------
# 3. Substrate runner — oracle-generator ceiling
# -----------------------------------------------------------------


def test_oracle_generator_passes_every_instance_under_every_strategy():
    tasks, files = build_mini_swe_bank(n_distractors=6)
    rep = run_swe_loop(bank=tasks, repo_files=files,
                        generator=deterministic_oracle_generator,
                        strategies=ALL_SWE_STRATEGIES)
    pooled = rep.pooled_summary()
    for strat in ALL_SWE_STRATEGIES:
        assert pooled[strat]["pass_at_1"] == 1.0, strat


def test_substrate_prompt_is_distractor_independent():
    tasks_a, files_a = build_mini_swe_bank(n_distractors=0)
    tasks_b, files_b = build_mini_swe_bank(n_distractors=24)
    rep_a = run_swe_loop(
        bank=tasks_a, repo_files=files_a,
        generator=deterministic_oracle_generator,
        strategies=(STRATEGY_SUBSTRATE,))
    rep_b = run_swe_loop(
        bank=tasks_b, repo_files=files_b,
        generator=deterministic_oracle_generator,
        strategies=(STRATEGY_SUBSTRATE,))
    sub_a = rep_a.pooled_summary()[STRATEGY_SUBSTRATE]
    sub_b = rep_b.pooled_summary()[STRATEGY_SUBSTRATE]
    # Substrate prompt size is independent of n_distractors.
    assert sub_a["mean_patch_gen_prompt_chars"] == \
           sub_b["mean_patch_gen_prompt_chars"]
    # Naive grows with distractors (sanity check on the
    # comparison axis).
    rep_n_a = run_swe_loop(
        bank=tasks_a, repo_files=files_a,
        generator=deterministic_oracle_generator,
        strategies=(STRATEGY_NAIVE,))
    rep_n_b = run_swe_loop(
        bank=tasks_b, repo_files=files_b,
        generator=deterministic_oracle_generator,
        strategies=(STRATEGY_NAIVE,))
    n_a = rep_n_a.pooled_summary()[STRATEGY_NAIVE]
    n_b = rep_n_b.pooled_summary()[STRATEGY_NAIVE]
    assert n_b["mean_patch_gen_prompt_chars"] > \
           n_a["mean_patch_gen_prompt_chars"]


def test_chain_hash_invariant_holds_under_every_strategy():
    tasks, files = build_mini_swe_bank(n_distractors=4)
    rep = run_swe_loop(bank=tasks, repo_files=files,
                        generator=deterministic_oracle_generator,
                        strategies=ALL_SWE_STRATEGIES)
    for m in rep.measurements:
        assert m.chain_ok, (m.instance_id, m.strategy)


def test_substrate_patch_proposed_handoff_is_recorded():
    tasks, files = build_mini_swe_bank(n_distractors=2)
    rep = run_swe_loop(bank=tasks, repo_files=files,
                        generator=deterministic_oracle_generator,
                        strategies=(STRATEGY_SUBSTRATE,))
    # For substrate, n_handoffs counts:
    #   issue_parsed (1) + file_located (1) + hunk_located (1) +
    #   patch_proposed (1) + test_result (1) = 5
    for m in rep.measurements:
        assert m.n_handoffs == 5, m.instance_id


# -----------------------------------------------------------------
# 4. LLM-style generator surface (without an actual LLM)
# -----------------------------------------------------------------


def test_llm_generator_parses_correct_block():
    from vision_mvp.tasks.swe_bench_bridge import (
        llm_patch_generator, build_mini_swe_bank,
    )
    tasks, files = build_mini_swe_bank(n_distractors=2)
    t = tasks[0]   # mini-swe-001 factorial
    src = files[t.buggy_file_relpath]

    canned_response = (
        "OLD>>>\n"
        "    result = 0\n"
        "<<<NEW>>>\n"
        "    result = 1\n"
        "<<<")

    def _stub(_prompt: str) -> str:
        return canned_response

    gen = llm_patch_generator(_stub)
    proposed = gen(t, {"hunk": ""}, src, "factorial bug")
    assert len(proposed.patch) == 1
    new, ok, _ = apply_patch(src, proposed.patch)
    assert ok
    res = run_patched_test(file_source=src, patched_source=new,
                            test_source=t.test_source)
    assert res.test_passed


def test_llm_generator_returns_empty_patch_on_parse_failure():
    from vision_mvp.tasks.swe_bench_bridge import llm_patch_generator
    tasks, files = build_mini_swe_bank(n_distractors=2)
    t = tasks[0]
    src = files[t.buggy_file_relpath]

    def _stub(_prompt: str) -> str:
        return "I think the fix is to change result = 0 to 1."

    gen = llm_patch_generator(_stub)
    proposed = gen(t, {"hunk": ""}, src, "x")
    assert proposed.patch == ()
    assert proposed.rationale == "parse_failed"


# -----------------------------------------------------------------
# 5. External-adapter shim
# -----------------------------------------------------------------


def test_swe_bench_adapter_round_trip():
    d = {
        "instance_id": "ext-001",
        "repo": "external/dummy",
        "base_commit": "abc123",
        "problem_statement": "x doesn't y",
        "buggy_file_relpath": "src.py",
        "buggy_function": "x",
        "gold_patch": [["return 0", "return 1"]],
        "test_source": ("def test(module):\n"
                          "    assert module.x() == 1\n"),
    }
    t = SWEBenchAdapter.from_dict(d)
    assert t.instance_id == "ext-001"
    assert len(t.gold_patch) == 1
    assert t.gold_patch[0] == ("return 0", "return 1")


def test_swe_bench_adapter_requires_repo_files_for_unified_diff():
    """Phase 40 promotes unified-diff parsing to a first-class
    feature (see ``test_phase40_real_swe_bridge``). Without
    ``repo_files``, the adapter cannot anchor the diff to a
    unique substitution window and must refuse — but with
    ``ValueError`` (a typed contract violation), not the
    Phase-39 ``NotImplementedError`` placeholder.
    """
    d = {
        "instance_id": "x", "repo": "r", "base_commit": "v",
        "problem_statement": "p", "buggy_file_relpath": "f.py",
        "buggy_function": "f",
        "gold_patch": "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-0\n+1\n",
        "test_source": "def test(module): pass\n",
    }
    try:
        SWEBenchAdapter.from_dict(d)
    except ValueError:
        return
    raise AssertionError("expected ValueError (no repo_files)")


# -----------------------------------------------------------------
# 6. Failure semantics
# -----------------------------------------------------------------


def test_failed_apply_marks_patch_no_match():
    tasks, files = build_mini_swe_bank(n_distractors=0)
    t = tasks[0]

    def _broken_gen(_t, _ctx, _src, _summary):
        return ProposedPatch(patch=(("nope", "nope2"),))

    rep = run_swe_loop(bank=[t], repo_files=files,
                        generator=_broken_gen,
                        strategies=(STRATEGY_SUBSTRATE,))
    m = rep.measurements[0]
    assert not m.patch_applied
    assert m.error_kind == "patch_no_match"
    assert not m.test_passed


def test_runner_recovers_from_generator_exception():
    tasks, files = build_mini_swe_bank(n_distractors=0)
    t = tasks[0]

    def _crashing_gen(_t, _ctx, _src, _summary):
        raise RuntimeError("boom")

    rep = run_swe_loop(bank=[t], repo_files=files,
                        generator=_crashing_gen,
                        strategies=(STRATEGY_SUBSTRATE,))
    m = rep.measurements[0]
    assert not m.patch_applied
    # No patch applied → workspace test never runs.
    assert m.error_kind == "patch_no_match"
