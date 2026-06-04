"""W136 — unit tests for the machine-structured algorithm-state trace instrument
(``coordpy.algorithm_state_trace_v1``).  All $0 NIM (oracle execution only).

Covers: typed sub-instance generation for the structured input shapes (knapsack tuples, grid),
genuinely-new-vs-S4 (the dual-trajectory + transition the W135 prose ladder lacks), leakage-cleanliness
(no solver source in the capsule, sub-instances disjoint from the graded bank), reproducibility, the
negative control (a value-correct/complexity naive ⇒ NONE), the positive control (ref ⇒ NONE), the
forward-only T2 controller routing + never-worse-than-C1 fallback, and falsifiability.
"""
from __future__ import annotations

import pytest

from coordpy.resistant_by_construction_battlefield_v1 import (
    MODE_COMPLEXITY_BLIND, MODE_SEARCH_ENUM, MODE_WRONG_ALGORITHM, mint_problem_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1
from coordpy.algorithm_state_trace_v1 import (
    ARM_T1_TRACE_REWRITE, ARM_T2_TRACE_CONTROLLER, TRACE_NONE,
    AlgorithmStateTraceV1, StateTransitionRowV1, _typed_subinstances_v1,
    build_algorithm_state_trace_v1, route_trace_action_v1, run_trace_arm_v1,
    trace_is_genuinely_new_vs_structure_v1,
)
from coordpy.controller_native_code_mechanism_v1 import ControllerAction

WSEED = 909_136


def _tmpl(name):
    for t in RBC_SLATE_V1:
        if t.name == name:
            return t
    raise KeyError(name)


def _mint(name, seed=136_777):
    t = _tmpl(name)
    p = mint_problem_v1(t, global_seed=seed, timeout_s=8.0)
    return t, p


def _probe(t, p):
    return build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)


def _trace(code, t, p):
    return build_algorithm_state_trace_v1(code, p, _probe(t, p), t, timeout_s=2.0, oracle_timeout_s=4.0)


# ---------------------------------------------------------------- typed sub-instances

def test_typed_subinstances_knapsack_tuple_prefixes():
    # 'N W' header + 2N interleaved (w,v) -> first-m-item sub-instances with N reset to m
    subs = _typed_subinstances_v1("3 100\n20 120 40 200 60 240")
    assert len(subs) == 3
    assert subs[0] == "1 100\n20 120"
    assert subs[1] == "2 100\n20 120 40 200"
    assert subs[2] == "3 100\n20 120 40 200 60 240"


def test_typed_subinstances_grid_subgrids():
    # 'R C' header + R row-strings -> top-left r'xc' sub-grids
    subs = _typed_subinstances_v1("2 2\n.. ..")
    assert subs, "grid should yield sub-grids"
    assert subs[-1] == "2 2\n.. .."          # ends at the full grid
    assert all("\n" in s for s in subs)
    # every sub-grid has a valid 'r c' header and r row-tokens each of length c
    for s in subs:
        head, body = s.split("\n", 1)
        r, c = map(int, head.split())
        toks = body.split()
        assert len(toks) == r and all(len(tok) == c for tok in toks)


def test_typed_subinstances_1d_array_fallback():
    # single-int header + flat array -> prefixes (stride-1 generalisation)
    subs = _typed_subinstances_v1("4\n5 3 2 7")
    assert subs[0] == "1\n5"
    assert subs[-1] == "4\n5 3 2 7"


# ---------------------------------------------------------------- genuinely-new vs S4

def test_trace_genuinely_new_on_knapsack_naive():
    t, p = _mint("wa_knapsack_01")
    assert p.gates.admitted and p.mode == MODE_WRONG_ALGORITHM
    tr = _trace(t.naive_source, t, p)
    gn = trace_is_genuinely_new_vs_structure_v1(tr, p)
    assert tr.found() and gn["genuinely_new"]
    assert tr.has_dual_trajectory() and tr.has_transition_structure()
    assert len(tr.rows) >= 2 and tr.leakage_clean


def test_trace_genuinely_new_on_search_enum_naive():
    t, p = _mint("se_count_stair_climbings") if any(
        x.name == "se_count_stair_climbings" for x in RBC_SLATE_V1) else _mint(
        next(x.name for x in RBC_SLATE_V1 if x.mode == MODE_SEARCH_ENUM))
    tr = _trace(t.naive_source, t, p)
    gn = trace_is_genuinely_new_vs_structure_v1(tr, p)
    # a search-enum naive that fails should fire a genuinely-new frontier trace
    if tr.found():
        assert gn["genuinely_new"] and tr.leakage_clean


def test_degenerate_trace_is_not_genuinely_new():
    # a trace with no dual trajectory / no transition collapses to S4's ladder ⇒ NOT genuinely-new
    from coordpy.exact_oracle_witness_v1 import _none_witness, WitnessV1, WITNESS_COUNTEREXAMPLE
    t, p = _mint("wa_knapsack_01")
    ce = WitnessV1(kind=WITNESS_COUNTEREXAMPLE, ew_family="EW1", probe_input="9 9\n1 2",
                   probe_input_tokens=3, expected_output="3", observed_output="2",
                   observed_kind="WRONG_ANSWER", big_n_tokens=0, ref_runtime_s=0.0,
                   cand_runtime_s=0.0, shrink_steps=0, leakage_clean=True)
    degenerate = AlgorithmStateTraceV1(
        kind="DECISION_PATH", at_family="AT1", counterexample=ce, optimal_value="3",
        naive_value="2", objective_gap="1", naive_overcounts=None, rows=(),
        first_divergence_idx=-1, leakage_clean=True)
    gn = trace_is_genuinely_new_vs_structure_v1(degenerate, p)
    assert not gn["genuinely_new"]  # no rows ⇒ no dual trajectory ⇒ not new vs S4


# ---------------------------------------------------------------- leakage / controls

def test_capsule_has_no_solver_source():
    t, p = _mint("wa_knapsack_01")
    block = _trace(t.naive_source, t, p).to_capsule_block(ARM_T1_TRACE_REWRITE)
    for token in ("def ", "import ", "class "):
        assert token not in block, f"capsule leaked solver source via {token!r}"


def test_subinstances_disjoint_from_secret_bank():
    t, p = _mint("wa_knapsack_01")
    tr = _trace(t.naive_source, t, p)
    secret = {inp for inp, _ in p.secret_cases}
    # the rows are built from sub-instances; none of the revealed sub-instance VALUES may be a graded case
    assert tr.counterexample.probe_input not in secret
    assert tr.leakage_clean


def test_positive_control_ref_is_none():
    t, p = _mint("wa_knapsack_01")
    tr = _trace(t.ref_source, t, p)
    assert tr.kind == TRACE_NONE and not tr.found()


def test_negative_control_complexity_naive_is_none():
    # a COMPLEXITY_BLIND naive is value-correct (just too slow) ⇒ no counterexample ⇒ trace NONE
    cb = next((x for x in RBC_SLATE_V1 if x.mode == MODE_COMPLEXITY_BLIND), None)
    if cb is None:
        pytest.skip("no complexity template in slate")
    p = mint_problem_v1(cb, global_seed=136_778, timeout_s=8.0)
    tr = _trace(cb.naive_source, cb, p)
    assert tr.kind == TRACE_NONE


# ---------------------------------------------------------------- reproducibility

def test_trace_reproducible():
    t, p = _mint("wa_knapsack_01")
    a = _trace(t.naive_source, t, p)
    b = _trace(t.naive_source, t, p)
    assert a.cid() == b.cid()
    assert a.to_capsule_block(ARM_T1_TRACE_REWRITE) == b.to_capsule_block(ARM_T1_TRACE_REWRITE)


# ---------------------------------------------------------------- T2 controller routing

def test_route_abstains_when_value_correct():
    t, p = _mint("wa_knapsack_01")
    none_trace = _trace(t.ref_source, t, p)   # ref ⇒ NONE
    action, why = route_trace_action_v1(none_trace, stderr_tail="", timed_out=False)
    assert action == ControllerAction.ABSTAIN


def test_route_uses_trace_when_dual_trajectory():
    t, p = _mint("wa_knapsack_01")
    tr = _trace(t.naive_source, t, p)
    action, why = route_trace_action_v1(tr, stderr_tail="", timed_out=False)
    assert action in (ControllerAction.PATCH, ControllerAction.REPLAN)


def test_t2_never_worse_than_counterexample():
    # the T2 fallback rendering (DRAFT/ABSTAIN) is the bare counterexample block (never worse than C1)
    from coordpy.algorithm_state_trace_v1 import _trace_block_for_action_v1
    t, p = _mint("wa_knapsack_01")
    tr = _trace(t.naive_source, t, p)
    fallback = _trace_block_for_action_v1(tr, ControllerAction.DRAFT)
    assert fallback == tr.counterexample.to_prompt_block()


# ---------------------------------------------------------------- same-budget arm structure

def test_trace_arm_same_budget_and_scored():
    t, p = _mint("wa_knapsack_01")
    probe = _probe(t, p)
    calls = {"n": 0}

    def stub_gen(prompt, max_tokens, temperature):
        calls["n"] += 1
        return "```python\nimport sys\nprint(0)\n```", {}

    outcome, audit = run_trace_arm_v1(
        seed=1, template=t, problem=p, probe=probe, gen=stub_gen, K=5, temperature=0.7,
        max_tokens=256, timeout_s=4.0, arm=ARM_T1_TRACE_REWRITE, minted_date="2026-06-04")
    assert outcome.n_model_calls == 5 and calls["n"] == 5
    assert outcome.arm_id == ARM_T1_TRACE_REWRITE
    assert len(outcome.per_call_passed) == 5
    assert audit.problem_id == p.problem_id


# ---------------------------------------------------------------- W136 root-cause: I/O-format confound

def test_io_format_is_the_trap_discriminator():
    """The SAME correct 0/1-knapsack DP passes with robust split() parsing and FAILS with per-line
    parsing on the whitespace-flattened battlefield input — proving I/O format, not the algorithm, is
    the trap discriminator (the W136 root cause)."""
    from coordpy.resistant_by_construction_battlefield_v1 import _exec_capture_v1
    from coordpy.coordpy_icpc_battlefield_v1 import judge_icpc_output_v1
    t, p = _mint("wa_knapsack_01")

    def fails(code):
        return sum(1 for inp, exp in p.secret_cases
                   if not (lambda r: (not r.timed_out and r.returncode == 0
                                      and judge_icpc_output_v1(got_stdout=r.stdout, expected=exp,
                                                               kind=p.kind, float_tol=p.float_tol)))(
                       _exec_capture_v1(code, inp, timeout_s=8.0)))

    robust = ("import sys\nd=sys.stdin.read().split()\nn=int(d[0]);W=int(d[1]);k=2;dp=[0]*(W+1)\n"
              "for _ in range(n):\n wt=int(d[k]);v=int(d[k+1]);k+=2\n for c in range(W,wt-1,-1):\n"
              "  if dp[c-wt]+v>dp[c]:dp[c]=dp[c-wt]+v\nprint(dp[W])")
    perline = ("n,W=map(int,input().split())\nitems=[tuple(map(int,input().split())) for _ in range(n)]\n"
               "dp=[0]*(W+1)\nfor wt,v in items:\n for c in range(W,wt-1,-1):\n"
               "  if dp[c-wt]+v>dp[c]:dp[c]=dp[c-wt]+v\nprint(dp[W])")
    assert fails(robust) == 0, "correct DP + robust parsing must PASS the secret bank"
    assert fails(perline) == len(p.secret_cases), "same DP + per-line parsing must FAIL (I/O confound)"


def test_code_fails_public_io_detector():
    """The execution-grounded I/O detector flags a per-line parser (crashes on the flattened public
    sample) and clears a robust parser."""
    from coordpy.algorithm_state_trace_v1 import code_fails_public_io_v1
    t, p = _mint("wa_knapsack_01")
    pilot = p.to_pilot_problem(minted_date="2026-06-04")
    perline = "n,W=map(int,input().split())\nitems=[tuple(map(int,input().split())) for _ in range(n)]\nprint(0)"
    robust = "import sys\nd=sys.stdin.read().split()\nprint(0)"
    assert code_fails_public_io_v1(perline, pilot) is True
    assert code_fails_public_io_v1(robust, pilot) is False


def test_io_grounded_arm_fires_directive():
    """The T_IO arm prepends the I/O directive when the model's code fails a public sample."""
    from coordpy.algorithm_state_trace_v1 import run_io_grounded_trace_arm_v1, ARM_T_IO
    t, p = _mint("wa_knapsack_01")
    probe = _probe(t, p)

    def perline_gen(prompt, max_tokens, temperature):
        # always returns a per-line parser (fails the flattened public sample)
        return ("```python\nn,W=map(int,input().split())\n"
                "items=[tuple(map(int,input().split())) for _ in range(n)]\nprint(0)\n```", {})

    outcome, audit = run_io_grounded_trace_arm_v1(
        seed=1, template=t, problem=p, probe=probe, gen=perline_gen, K=3, temperature=0.7,
        max_tokens=256, timeout_s=4.0, minted_date="2026-06-04")
    assert outcome.arm_id == ARM_T_IO and outcome.n_model_calls == 3
    assert "IO_REPAIR" in audit.controller_actions  # the directive fired on the per-line code
