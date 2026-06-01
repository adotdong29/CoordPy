"""W128 tests — role-diverse algorithm search (Lane α mechanism) + earn gate + bridges.

Falsifiability-first + positive controls (the W125/W126/W127 discipline):
* the fake-diversity detector MUST bite (identical sketches -> FAKE_DIVERSE);
* the REAL synthesis bridge (W41/W42) MUST be load-bearing (abstain on divergence);
* the W79 substrate-controller literal bridge MUST be examined-and-killed as fake;
* the leakage guard positive control MUST catch a planted accepted block;
* the earn gate MUST reject a fake-diversity "win" and a sub-threshold net.

Direct-execution (no pytest required) + pytest-discoverable.  NIM-free (fake gens only).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import IcpcPilotProblemV1  # noqa: E402

_ANALYZE = """SPEC:
read integer n, print 2*n. 0<=n<=1e9.
INVARIANTS:
- output is even
- output >= 0 for n >= 0
COMPLEXITY:
O(1); brute force fits.
SKETCHES:
SKETCH A: multiply
1. read n
2. print n*2
SKETCH B: repeated addition
1. read n
2. accumulate n twice in a loop
3. print accumulator
COUNTEREXAMPLES:
CASE:
1
EXPECT:
2
===
CASE:
9
EXPECT:
18
"""


def _double_problem():
    return IcpcPilotProblemV1(
        problem_id="syn/double", short_name="double", source_repo="w128_test",
        contest_date="2024-01-01", statement="Read integer n; print 2*n.", kind="passfail",
        float_tol=1e-6, samples=[("3\n", "6\n"), ("5\n", "10\n")],
        secret_cases=[("7\n", "14\n"), ("0\n", "0\n"), ("11\n", "22\n")])


def _gen_correct_diverse(prompt, max_tokens, temperature):
    if "ANALYSIS team" in prompt:
        return (_ANALYZE, 1)
    if "multiply" in prompt:
        return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*2)\n```", 1)
    return ("```python\nimport sys\nn=int(sys.stdin.read())\ns=0\nfor _ in range(2):\n s+=n\nprint(s)\n```", 1)


# ---------------------------------------------------------------- stable boundary
def test_stable_boundary_untouched():
    assert coordpy.__version__ == "0.5.20"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"
    # the W128 module is explicit-import-only; it must NOT be re-exported by __init__
    assert not hasattr(coordpy, "role_diverse_algorithm_search_v1") or \
        "role_diverse_algorithm_search_v1" not in getattr(coordpy, "__all__", [])


# ---------------------------------------------------------------- fake-diversity detector
def test_fake_diversity_positive_control_bites():
    fd = R.fake_diversity_control_v1()
    assert fd.classify() == "FAKE_DIVERSE", "positive control must classify FAKE_DIVERSE"
    assert fd.diversity_real is False


def test_genuine_diversity_classifies_real():
    out = R.run_role_diverse_search_v1(_gen_correct_diverse, _double_problem(),
                                       K=5, n_sketches=4, timeout_s=8.0)
    assert out.diversity["classify"] == "REAL"
    assert out.n_calls == 5, "budget must be matched: 1 analyze + 4 implement = K=5"


def test_diversity_real_requires_distinct_sketches():
    # two identical-outline sketches -> high jaccard -> not diversity_real
    sk = (R.SketchV1("A", "x", "loop over all n and count"),
          R.SketchV1("B", "x", "loop over all n and count"))
    arts = R.RoleArtifactsV1(spec="s", invariants=("inv",), complexity="O(n)", sketches=sk,
                             counterexamples=(("99\n", None),), raw="")
    impls = (R.CandidateImplV1("A", "print(1)", True), R.CandidateImplV1("B", "print(2)", True))
    div = R.compute_diversity_v1(arts, impls, sample_inputs=["3\n"])
    assert div.classify() == "FAKE_DIVERSE"


# ---------------------------------------------------------------- selection / abstain
def test_public_filter_eliminates_sample_failers():
    # one impl right, one wrong-on-sample -> only the right one survives publics
    def gen(prompt, mt, t):
        if "ANALYSIS team" in prompt:
            return (_ANALYZE, 1)
        if "multiply" in prompt:
            return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*2)\n```", 1)
        return ("```python\nimport sys\nn=int(sys.stdin.read());print(n+1)\n```", 1)  # wrong
    out = R.run_role_diverse_search_v1(gen, _double_problem(), K=5, n_sketches=4, timeout_s=8.0)
    # RDA4 must commit a sample-passing (correct) candidate and pass secret
    assert out.committed_pass["RDA4"] is True
    assert out.pool_pass is True


def test_rda3_abstains_on_irreconcilable_divergence():
    # sample under-determines: n*2 and n*n both pass the single sample (n=2 -> 4); they DIVERGE
    prob = IcpcPilotProblemV1(
        problem_id="syn/amb", short_name="amb", source_repo="w128_test",
        contest_date="2024-01-01", statement="Read n; print f(n).", kind="passfail",
        float_tol=1e-6, samples=[("2\n", "4\n")], secret_cases=[("3\n", "6\n")])
    A = _ANALYZE.replace("EXPECT:\n2", "").replace("EXPECT:\n18", "")  # NO predicted-expected

    def gen(prompt, mt, t):
        if "ANALYSIS team" in prompt:
            return (A, 1)
        if "multiply" in prompt:
            return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*2)\n```", 1)
        return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*n)\n```", 1)
    out = R.run_role_diverse_search_v1(gen, prob, K=3, n_sketches=2, timeout_s=8.0)
    assert out.n_public_survivors == 2
    # RDA3 must ABSTAIN (no strict majority, no predicted-expected to break the tie)
    assert out.abstained["RDA3"] is True
    assert out.committed_pass["RDA3"] is False


def test_rda4_breaks_tie_with_predicted_expected():
    # same ambiguity BUT the analyze call supplies correct predicted-expected -> RDA4 commits right
    prob = IcpcPilotProblemV1(
        problem_id="syn/amb2", short_name="amb2", source_repo="w128_test",
        contest_date="2024-01-01", statement="Read n; print 2*n.", kind="passfail",
        float_tol=1e-6, samples=[("2\n", "4\n")], secret_cases=[("3\n", "6\n"), ("5\n", "10\n")])

    def gen(prompt, mt, t):
        if "ANALYSIS team" in prompt:
            return (_ANALYZE, 1)  # has EXPECT: 1->2, 9->18 (consistent with n*2)
        if "multiply" in prompt:
            return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*2)\n```", 1)
        return ("```python\nimport sys\nn=int(sys.stdin.read());print(n*n)\n```", 1)
    out = R.run_role_diverse_search_v1(gen, prob, K=3, n_sketches=2, timeout_s=8.0)
    assert out.committed_pass["RDA4"] is True, "RDA4 should commit the predicted-expected match"


# ---------------------------------------------------------------- leakage (§3) positive control
def test_leakage_guard_catches_planted_accepted_block():
    prob = _double_problem()
    leak_hits = {"called": False}

    def leak(code):
        # a candidate reproducing this 3-line block is NOT clean
        planted = "x = 1\ny = 2\nz = 3"
        leak_hits["called"] = True
        return planted not in code

    def gen(prompt, mt, t):
        if "ANALYSIS team" in prompt:
            return (_ANALYZE, 1)
        return ("```python\nx = 1\ny = 2\nz = 3\nimport sys\nn=int(sys.stdin.read());print(n*2)\n```", 1)
    out = R.run_role_diverse_search_v1(gen, prob, K=3, n_sketches=2, timeout_s=8.0,
                                       leakage_check=leak)
    assert leak_hits["called"] is True
    assert out.leakage_clean is False, "planted block must flip leakage_clean"


def test_leakage_clean_passes_clean_candidate():
    out = R.run_role_diverse_search_v1(_gen_correct_diverse, _double_problem(), K=5,
                                       n_sketches=4, timeout_s=8.0,
                                       leakage_check=lambda c: "PLANTED" not in c)
    assert out.leakage_clean is True


# ---------------------------------------------------------------- honest mining: RDA4 kill record
def test_substrate_controllers_examined_and_literal_bridge_killed():
    rep = R.examine_substrate_controller_applicability_v1()
    assert rep["all_substrate_specific"] is True
    for name in ("team_consensus_controller_v14", "consensus_fallback_controller_v25",
                 "hosted_cost_planner_v12", "hosted_real_handoff_coordinator_v11"):
        c = rep["controllers"][name]
        assert c.get("literal_bridge_would_be_fake") is True, name
    assert "role_invariant_synthesis.select_role_invariance_decision" in rep["consensus_provided_by"]


def test_real_synthesis_bridge_is_load_bearing():
    from coordpy.role_invariant_synthesis import (
        select_role_invariance_decision, W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED)
    b_ratify, _o, s1 = select_role_invariance_decision(
        integrated_services=["A"], expected_services=["A"], policy_match_found=True)
    b_abstain, _o2, _s2 = select_role_invariance_decision(
        integrated_services=["A"], expected_services=["B"], policy_match_found=True)
    assert b_abstain == W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED
    assert b_ratify != b_abstain


# ---------------------------------------------------------------- earn gate (R1')
def _res(short, fam, base, scaf, rda, pool, abst, divreal, clean, triv=False):
    return R.RdaDevBenchTargetResultV1(
        short_name=short, family=fam, baseline_pass=base, scaffold_pass=scaf,
        rda_committed_pass=rda, rda_pool_pass=pool, rda_abstained=abst,
        diversity_real=divreal, leakage_clean=clean, failure_was_trivial=triv)


def test_earn_gate_rejects_subthreshold_net():
    # only +1 net -> R1a fail
    res = [_res("a", "simulation_grid", False, False, True, True, False, True, True),
           _res("b", "adhoc_math", True, True, True, True, False, True, True),
           _res("c", "greedy_scheduling", False, False, False, False, True, True, True)]
    v = R.apply_rda_dev_bench_earn_gate_v1(res)
    assert v.net_rda_gain == 1 and v.earned is False


def test_earn_gate_rejects_fake_diversity_winner():
    # +2 net BUT a winner is FAKE_DIVERSE -> R1c fail (not a mechanism win)
    res = [_res("a", "simulation_grid", False, False, True, True, False, False, True),  # fake!
           _res("b", "adhoc_math", False, False, True, True, False, True, True)]
    v = R.apply_rda_dev_bench_earn_gate_v1(res)
    assert v.net_rda_gain == 2 and v.r1c_clean_and_real is False and v.earned is False


def test_earn_gate_earns_on_clean_real_named_cluster_win():
    res = [_res("a", "simulation_grid", False, False, True, True, False, True, True),
           _res("b", "adhoc_math", False, False, True, True, False, True, True),
           _res("c", "greedy_scheduling", True, False, True, True, False, True, True)]  # no regr
    v = R.apply_rda_dev_bench_earn_gate_v1(res)
    assert v.net_rda_gain == 2 and v.gain_includes_named_cluster is True
    assert v.r1c_clean_and_real and v.r1e_beats_scaffold and v.earned is True


def test_earn_gate_requires_beating_scaffold():
    # rda net +2 but scaffold net +3 -> R1e fail
    res = [_res("a", "simulation_grid", False, True, True, True, False, True, True),
           _res("b", "adhoc_math", False, True, True, True, False, True, True),
           _res("c", "greedy_scheduling", False, True, False, False, True, True, True)]
    v = R.apply_rda_dev_bench_earn_gate_v1(res)
    # scaffold solves 3 uniques, rda solves 2 -> rda must NOT earn over scaffold
    assert v.r1e_beats_scaffold is False and v.earned is False


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} W128 tests PASS")


if __name__ == "__main__":
    _run_all()
