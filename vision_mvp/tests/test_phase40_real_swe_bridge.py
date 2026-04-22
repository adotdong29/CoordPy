"""Phase 40 — real SWE-bridge loader, adapter, and sandbox tests.

Coverage:
  * unified-diff parser round-trips on representative shapes;
  * SWEBenchAdapter.from_swe_bench_dict handles the real-shape
    SWE-bench dict (unified-diff ``patch``, optional
    ``test_patch``);
  * load_jsonl_bank materialises a runnable bank from the bundled
    real-shape JSONL artifact;
  * SubprocessSandbox preserves the Phase-39 oracle ceiling on the
    mini bank (Theorem P40-3 empirical equivalence);
  * SubprocessSandbox enforces wall-clock timeouts on infinite
    loops, captures syntax / import / assertion / exception
    failures, and refuses ambiguous patches via apply_patch;
  * the sandbox-aware run_swe_loop_sandboxed produces measurements
    whose pass@1 matches the in-process ceiling under the
    deterministic oracle;
  * load_jsonl_bank + SubprocessSandbox composition gives 100 %
    pass@1 on the bundled real-shape bank under the oracle (the
    Phase-40 precondition test);
  * substrate prompt size is independent of n_distractors on the
    real-shape JSONL bank (P40 analogue of P39-3).
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_SWE_STRATEGIES, SWEBenchAdapter, STRATEGY_NAIVE,
    STRATEGY_SUBSTRATE, apply_patch, build_mini_swe_bank,
    build_synthetic_event_log, deterministic_oracle_generator,
    load_jsonl_bank, parse_unified_diff, run_swe_loop,
)
from vision_mvp.tasks.swe_sandbox import (
    DockerSandbox, InProcessSandbox, SubprocessSandbox,
    run_swe_loop_sandboxed, select_sandbox,
)


_BUNDLED_JSONL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_real_shape_mini.jsonl"))


# -----------------------------------------------------------------
# 1. Unified-diff parser
# -----------------------------------------------------------------


def test_parse_unidiff_single_hunk_substitution():
    diff = textwrap.dedent("""\
        --- a/calc.py
        +++ b/calc.py
        @@ -7,5 +7,5 @@ def factorial(n):
             but seeds with 0 instead of 1 so every result is 0.\"\"\"
        -    result = 0
        +    result = 1
             for i in range(1, n + 1):
                 result *= i
                 return result
    """)
    out = parse_unified_diff(diff)
    assert "calc.py" in out
    subs = out["calc.py"]
    assert len(subs) == 1
    old, new = subs[0]
    assert "result = 0" in old
    assert "result = 1" in new
    # The change line is the only difference between old and new blocks.
    assert old.replace("result = 0", "result = 1") == new


def test_parse_unidiff_strips_a_b_prefixes():
    # SWE-bench ships diffs with `a/` / `b/` prefixes from `git diff`.
    diff = textwrap.dedent("""\
        --- a/src/x.py
        +++ b/src/x.py
        @@ -1,1 +1,1 @@
        -x = 0
        +x = 1
    """)
    out = parse_unified_diff(diff)
    assert list(out.keys()) == ["src/x.py"]


def test_parse_unidiff_handles_multi_file_diff():
    diff = textwrap.dedent("""\
        --- a/one.py
        +++ b/one.py
        @@ -1,1 +1,1 @@
        -A = 1
        +A = 2
        --- a/two.py
        +++ b/two.py
        @@ -1,1 +1,1 @@
        -B = 1
        +B = 2
    """)
    out = parse_unified_diff(diff)
    assert set(out.keys()) == {"one.py", "two.py"}
    assert len(out["one.py"]) == 1
    assert len(out["two.py"]) == 1


def test_parse_unidiff_handles_empty_diff():
    assert parse_unified_diff("") == {}


def test_parse_unidiff_skips_no_newline_marker():
    diff = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x = 0\n"
        "+x = 1\n"
        "\\ No newline at end of file\n"
    )
    out = parse_unified_diff(diff)
    assert "x.py" in out


# -----------------------------------------------------------------
# 2. SWEBenchAdapter.from_swe_bench_dict
# -----------------------------------------------------------------


def test_from_swe_bench_dict_round_trips_unidiff_patch():
    src = textwrap.dedent("""\
        def f():
            return 0
    """)
    diff = textwrap.dedent("""\
        --- a/m.py
        +++ b/m.py
        @@ -1,2 +1,2 @@
         def f():
        -    return 0
        +    return 1
    """)
    d = {
        "instance_id": "x-001",
        "repo": "ext/m",
        "base_commit": "v0",
        "problem_statement": "f returns wrong value",
        "patch": diff,
        "buggy_file_relpath": "m.py",
        "test_source": ("def test(module):\n"
                          "    assert module.f() == 1\n"),
    }
    t = SWEBenchAdapter.from_swe_bench_dict(d, repo_files={"m.py": src})
    assert t.buggy_function == "f"
    new, ok, _ = apply_patch(src, t.gold_patch)
    assert ok and "return 1" in new


def test_from_swe_bench_dict_derives_buggy_function_from_diff():
    src = textwrap.dedent("""\
        def helper():
            pass

        def target():
            return 0
    """)
    diff = textwrap.dedent("""\
        --- a/m.py
        +++ b/m.py
        @@ -4,2 +4,2 @@ def target():
         def target():
        -    return 0
        +    return 1
    """)
    d = {
        "instance_id": "x-002", "repo": "r", "base_commit": "v0",
        "problem_statement": "p",
        "patch": diff,
        "buggy_file_relpath": "m.py",
        "test_source": "def test(module):\n    assert module.target() == 1\n",
    }
    t = SWEBenchAdapter.from_swe_bench_dict(d, repo_files={"m.py": src})
    assert t.buggy_function == "target"


def test_from_swe_bench_dict_promotes_test_patch_to_test_source():
    src = "def f():\n    return 0\n"
    diff = textwrap.dedent("""\
        --- a/m.py
        +++ b/m.py
        @@ -1,2 +1,2 @@
         def f():
        -    return 0
        +    return 1
    """)
    test_patch = textwrap.dedent("""\
        --- a/test_m.py
        +++ b/test_m.py
        @@ -0,0 +1,3 @@
        +def test(module):
        +    assert module.f() == 1
        +
    """)
    d = {
        "instance_id": "x-003", "repo": "r", "base_commit": "v0",
        "problem_statement": "p",
        "patch": diff,
        "buggy_file_relpath": "m.py",
        "test_patch": test_patch,
    }
    t = SWEBenchAdapter.from_swe_bench_dict(d, repo_files={"m.py": src})
    assert "def test(module)" in t.test_source
    assert "module.f() == 1" in t.test_source


def test_from_swe_bench_dict_raises_when_diff_misses_repo_file():
    diff = ("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x\n+y\n")
    d = {"instance_id": "x", "repo": "r", "base_commit": "v0",
         "problem_statement": "p", "patch": diff,
         "buggy_file_relpath": "m.py",
         "test_source": "def test(module): pass\n"}
    with pytest.raises(ValueError):
        SWEBenchAdapter.from_swe_bench_dict(d, repo_files={"other.py": ""})


# -----------------------------------------------------------------
# 3. JSONL loader
# -----------------------------------------------------------------


def test_load_jsonl_bank_materialises_runnable_tasks():
    tasks, files = load_jsonl_bank(
        _BUNDLED_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 4))
    assert len(tasks) == 6
    for t in tasks:
        assert t.gold_patch
        assert "def test(" in t.test_source
        assert t.buggy_file_relpath in files


def test_load_jsonl_bank_namespaces_paths():
    tasks, files = load_jsonl_bank(_BUNDLED_JSONL)
    # Two distinct tasks should never share a relpath in the pool.
    assert len(set(t.buggy_file_relpath for t in tasks)) == len(tasks)


def test_load_jsonl_bank_respects_limit():
    tasks, _ = load_jsonl_bank(_BUNDLED_JSONL, limit=2)
    assert len(tasks) == 2


# -----------------------------------------------------------------
# 4. Subprocess sandbox — boundary preservation
# -----------------------------------------------------------------


def test_subprocess_sandbox_oracle_ceiling_on_mini_bank():
    """SubprocessSandbox preserves the Phase-39 oracle ceiling.

    Theorem P40-3 empirical signature: every (strategy, distractor)
    cell that passes under InProcessSandbox passes under
    SubprocessSandbox.
    """
    tasks, files = build_mini_swe_bank(n_distractors=4)
    sb = SubprocessSandbox()
    rep = run_swe_loop_sandboxed(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        sandbox=sb, strategies=ALL_SWE_STRATEGIES, timeout_s=15.0)
    pooled = rep.pooled_summary()
    for strat in ALL_SWE_STRATEGIES:
        assert pooled[strat]["pass_at_1"] == 1.0, (strat, pooled)


def test_subprocess_sandbox_handles_timeout():
    sb = SubprocessSandbox()
    src = "def loop_until():\n    return 0\n"
    patch = (("return 0", "i = 0\n    while True:\n        i += 1"),)
    test_src = "def test(module):\n    assert module.loop_until() == 1\n"
    res = sb.run(buggy_source=src, patch=patch, test_source=test_src,
                  module_name="m", timeout_s=2.0)
    assert not res.test_passed
    assert res.error_kind == "timeout"


def test_subprocess_sandbox_catches_syntax_error():
    sb = SubprocessSandbox()
    src = "def f():\n    return 0\n"
    patch = (("return 0", "def gibberish("),)
    test_src = "def test(module): pass\n"
    res = sb.run(buggy_source=src, patch=patch, test_source=test_src,
                  module_name="m", timeout_s=5.0)
    assert not res.test_passed
    assert res.error_kind == "syntax"


def test_subprocess_sandbox_attributes_patch_no_match():
    sb = SubprocessSandbox()
    res = sb.run(buggy_source="x = 1\n",
                  patch=(("nope", "yep"),),
                  test_source="def test(module): pass\n",
                  module_name="m", timeout_s=5.0)
    assert not res.patch_applied
    assert res.error_kind == "patch_no_match"


def test_subprocess_sandbox_catches_test_assertion():
    sb = SubprocessSandbox()
    src = "def f(): return 0\n"
    test_src = "def test(module):\n    assert module.f() == 99\n"
    res = sb.run(buggy_source=src, patch=(("return 0", "return 1"),),
                  test_source=test_src, module_name="m", timeout_s=5.0)
    assert not res.test_passed
    assert res.error_kind == "test_assert"


def test_subprocess_sandbox_isolates_subprocess_crash():
    """A patched module that calls os._exit(1) must not kill the
    bridge process — the subprocess catches it and reports
    sandbox_error / import (the runner exits cleanly with no JSON
    output, so the parser surfaces sandbox_error)."""
    sb = SubprocessSandbox()
    src = "x = 0\n"
    new = "import os; os._exit(1)\n"
    patch = (("x = 0", new),)
    test_src = "def test(module): pass\n"
    res = sb.run(buggy_source=src, patch=patch, test_source=test_src,
                  module_name="m", timeout_s=5.0)
    # Subprocess died before emitting JSON; should be reported as
    # sandbox_error, not silently as a pass.
    assert not res.test_passed
    assert res.error_kind == "sandbox_error"


# -----------------------------------------------------------------
# 5. Sandbox factory
# -----------------------------------------------------------------


def test_select_sandbox_in_process_always_available():
    sb = select_sandbox("in_process")
    assert sb.name() == "in_process"
    assert sb.is_available()


def test_select_sandbox_subprocess_available_locally():
    sb = select_sandbox("subprocess")
    assert sb.name() == "subprocess"
    assert sb.is_available()


def test_select_sandbox_auto_falls_through_to_subprocess_when_no_docker():
    # On developer machines without docker the auto choice is
    # subprocess; we don't assert the exact answer (would be
    # docker on a docker-host CI), only that the factory returns
    # an available backend with a stable name.
    sb = select_sandbox("auto")
    assert sb.name() in ("docker", "subprocess", "in_process")
    assert sb.is_available()


def test_select_sandbox_unknown_raises():
    with pytest.raises(ValueError):
        select_sandbox("totally_made_up_backend")


# -----------------------------------------------------------------
# 6. JSONL + sandbox + substrate end-to-end
# -----------------------------------------------------------------


def test_real_shape_jsonl_passes_oracle_ceiling_under_sandbox():
    tasks, files = load_jsonl_bank(
        _BUNDLED_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 4))
    sb = SubprocessSandbox()
    rep = run_swe_loop_sandboxed(
        bank=tasks, repo_files=files,
        generator=deterministic_oracle_generator,
        sandbox=sb, strategies=ALL_SWE_STRATEGIES, timeout_s=15.0)
    pooled = rep.pooled_summary()
    for strat in ALL_SWE_STRATEGIES:
        assert pooled[strat]["pass_at_1"] == 1.0, (strat, pooled)


def test_real_shape_substrate_prompt_distractor_independent():
    """Phase-40 analogue of Theorem P39-3 on the JSONL bank.

    Substrate prompt characters are constant in n_distractors;
    naive grows.
    """
    tasks_a, files_a = load_jsonl_bank(
        _BUNDLED_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 0))
    tasks_b, files_b = load_jsonl_bank(
        _BUNDLED_JSONL,
        hidden_event_log_factory=lambda t: build_synthetic_event_log(t, 24))
    sb = SubprocessSandbox()
    rep_a = run_swe_loop_sandboxed(
        bank=tasks_a, repo_files=files_a,
        generator=deterministic_oracle_generator, sandbox=sb,
        strategies=(STRATEGY_SUBSTRATE,), timeout_s=10.0)
    rep_b = run_swe_loop_sandboxed(
        bank=tasks_b, repo_files=files_b,
        generator=deterministic_oracle_generator, sandbox=sb,
        strategies=(STRATEGY_SUBSTRATE,), timeout_s=10.0)
    sub_a = rep_a.pooled_summary()[STRATEGY_SUBSTRATE]
    sub_b = rep_b.pooled_summary()[STRATEGY_SUBSTRATE]
    assert sub_a["mean_patch_gen_prompt_chars"] == \
           sub_b["mean_patch_gen_prompt_chars"]
    rep_n_a = run_swe_loop_sandboxed(
        bank=tasks_a, repo_files=files_a,
        generator=deterministic_oracle_generator, sandbox=sb,
        strategies=(STRATEGY_NAIVE,), timeout_s=10.0)
    rep_n_b = run_swe_loop_sandboxed(
        bank=tasks_b, repo_files=files_b,
        generator=deterministic_oracle_generator, sandbox=sb,
        strategies=(STRATEGY_NAIVE,), timeout_s=10.0)
    n_a = rep_n_a.pooled_summary()[STRATEGY_NAIVE]
    n_b = rep_n_b.pooled_summary()[STRATEGY_NAIVE]
    assert n_b["mean_patch_gen_prompt_chars"] > \
           n_a["mean_patch_gen_prompt_chars"]


def test_sandbox_choice_does_not_change_oracle_pass_rate():
    """Theorem P40-3 — sandbox-boundary preservation.

    InProcessSandbox and SubprocessSandbox must both deliver
    pass@1 = 1.000 on the mini bank under the deterministic
    oracle. (DockerSandbox is omitted from the assertion because
    test environments may not have a docker daemon.)
    """
    tasks, files = build_mini_swe_bank(n_distractors=2)
    for sb in (InProcessSandbox(), SubprocessSandbox()):
        rep = run_swe_loop_sandboxed(
            bank=tasks, repo_files=files,
            generator=deterministic_oracle_generator, sandbox=sb,
            strategies=ALL_SWE_STRATEGIES, timeout_s=10.0)
        for strat in ALL_SWE_STRATEGIES:
            assert rep.pooled_summary()[strat]["pass_at_1"] == 1.0, (
                sb.name(), strat)


def test_docker_sandbox_reports_unavailability_cleanly():
    d = DockerSandbox(docker_bin="this-binary-does-not-exist-xyz")
    assert not d.is_available()
    res = d.run(buggy_source="x = 0\n", patch=(("x = 0", "x = 1"),),
                 test_source="def test(module): pass\n",
                 module_name="m", timeout_s=5.0)
    assert not res.test_passed
    assert res.error_kind == "sandbox_error"
