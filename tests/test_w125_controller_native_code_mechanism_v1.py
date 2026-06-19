"""W125 — tests for the controller-native code mechanism (RUNBOOK W125 §§ 2,4,5,6).

Runnable by direct execution (``python3 tests/test_w125_controller_native_code_mechanism_v1.py``)
because the local pytest/attrs env is broken; also importable as pytest test functions.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy  # noqa: E402
from coordpy.coordpy_icpc_battlefield_v1 import KIND_PASSFAIL  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcPilotProblemV1, extract_candidate_code_v1)
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402


def _synthetic():
    prob = IcpcPilotProblemV1(
        problem_id="t/sum", short_name="sum", source_repo="t", contest_date="2025-01-01",
        statement="Read a b; print a+b.", kind=KIND_PASSFAIL, float_tol=0.0,
        samples=(("2 3\n", "5\n"),),
        secret_cases=(("10 20\n", "30\n"), ("1 1\n", "2\n")))
    g = extract_candidate_code_v1(
        response_text="```python\ndef main():\n a,b=map(int,input().split());print(a+b)\nmain()\n```")
    bd = extract_candidate_code_v1(
        response_text="```python\ndef main():\n a,b=map(int,input().split());print(a*b)\nmain()\n```")
    pool = M.PerProblemPoolV1(problem_id=prob.problem_id, a0_code=bd,
                              a1_codes=(bd, bd, bd, bd, g), b_codes=(bd, bd, bd, bd, bd))
    return prob, pool, g, bd


def test_stable_boundary_untouched():
    assert coordpy.__version__ == "1.2.1"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


def test_schema_version_present():
    assert M.W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION == (
        "coordpy.controller_native_code_mechanism_v1.v1")


def test_fake_different_test_bites():
    slate = M.evaluate_mechanism_slate()
    # reflexion B and the relabeled negative control MUST classify FAKE_DIFFERENT
    assert "reflexion_B" in slate.fake_candidates
    assert "C0_reflexion_relabeled" in slate.fake_candidates
    # the three real controllers MUST classify REAL; the lead is C3
    for name in ("C1_role_specialized_planner", "C2_router_selected_multi_candidate",
                 "C3_tool_substrate_audited_repair"):
        assert name in slate.real_candidates
    assert slate.lead == "C3_tool_substrate_audited_repair"


def test_fingerprint_classifier_logic():
    # a linear DRAFT-only chain is never REAL
    assert M.reflexion_b_fingerprint().classify() == "FAKE_DIFFERENT"
    # ≥2 native props AND non-linear => REAL
    assert M.C3DigestRoutedRepairControllerV1().fingerprint().classify() == "REAL"
    assert M.C2RouterSelectControllerV1().fingerprint().classify() == "REAL"


def test_audit_chain_rehash_and_tamper():
    prob, pool, _g, _bd = _synthetic()
    plane = M.AuditedGraderPlaneV1(prob, caller_agent_id="t")
    plane.grade_samples(pool.a1_codes[0])
    r1 = plane.merkle_root()
    assert r1 == plane.merkle_root()  # re-hash stable
    import dataclasses
    c, r = plane.chain.steps[-1]
    plane.chain.steps[-1] = (c, dataclasses.replace(r, result_bytes=r.result_bytes + b"X"))
    assert plane.chain.merkle_root() != r1  # tamper detected


def test_idempotent_recommit_refused():
    prob, pool, _g, _bd = _synthetic()
    plane = M.AuditedGraderPlaneV1(prob, caller_agent_id="t")
    plane.grade_samples(pool.a1_codes[0])
    call = plane.chain.steps[-1][0]
    assert plane.chain.already_committed(call, None) is True


def test_never_reads_secret_guard():
    prob, _pool, _g, _bd = _synthetic()
    plane = M.AuditedGraderPlaneV1(prob, caller_agent_id="t")
    assert plane.secret_leak_in("the answer is " + prob.secret_cases[0][1]) is True
    assert plane.secret_leak_in("Problem:\n" + prob.statement) is False


def test_routing_determinism():
    prob, pool, _g, _bd = _synthetic()
    c3 = M.C3DigestRoutedRepairControllerV1()
    o1 = c3.replay_on_pool(pool, prob, K=5)
    o2 = c3.replay_on_pool(pool, prob, K=5)
    assert o1.audit_merkle_root == o2.audit_merkle_root
    assert o1.action_trace == o2.action_trace
    assert o1.cid() == o2.cid()


def test_same_budget_accounting():
    prob, pool, _g, _bd = _synthetic()
    o = M.C3DigestRoutedRepairControllerV1().replay_on_pool(pool, prob, K=5)
    assert o.n_model_slots_used <= 5
    assert o.n_secret_grader_calls == 1


def test_c2_blind_selection_finds_good_candidate():
    prob, pool, _g, _bd = _synthetic()
    o = M.C2RouterSelectControllerV1().replay_on_pool(pool, prob, K=5)
    assert o.committed_passed_secret is True  # the sample-passing draft is selected


def test_contract_checks_all_pass():
    prob, pool, _g, _bd = _synthetic()
    cc = M.run_contract_checks(pool, prob, K=5)
    assert cc.all_pass is True


def test_headroom_probe_no_headroom_when_a1_already_passes():
    prob, pool, _g, _bd = _synthetic()
    hr = M.headroom_probe([prob], [pool], field="t")
    assert hr.a1_pass_count == 1
    assert hr.blind_selection_headroom == 0  # A1 already solves it; no extra headroom


def test_earn_gate_thresholds():
    slate = M.evaluate_mechanism_slate()
    cc = M.ContractCheckReportV1(
        schema="s", audit_chain_rehash_ok=True, audit_tamper_detected=True,
        idempotent_recommit_refused=True, grader_capture_complete=True,
        never_reads_secret=True, routing_determinism_ok=True, same_budget_ok=True)

    def hr(blind, diverge):
        return M.HeadroomReportV1(
            schema="s", field="t", n_problems=30, a1_pass_count=7,
            pool_union_secret_count=8, oracle_pool_headroom=1,
            c2_blind_committed_pass_count=7, c3_committed_pass_count=7,
            blind_selection_headroom=blind, blind_headroom_problem_ids=tuple(),
            reflexion_divergence=diverge, reflexion_divergence_problem_ids=tuple(),
            looks_right_fails_hidden=0)

    assert M.apply_pilot_earn_gate(cc, slate, hr(2, 3)).earned is True
    assert M.apply_pilot_earn_gate(cc, slate, hr(1, 3)).earned is False  # E3a fail
    assert M.apply_pilot_earn_gate(cc, slate, hr(2, 2)).earned is False  # E3b fail
    # contract fail => not earned regardless
    cc_bad = M.ContractCheckReportV1(
        schema="s", audit_chain_rehash_ok=False, audit_tamper_detected=True,
        idempotent_recommit_refused=True, grader_capture_complete=True,
        never_reads_secret=True, routing_determinism_ok=True, same_budget_ok=True)
    assert M.apply_pilot_earn_gate(cc_bad, slate, hr(2, 3)).earned is False


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            n_pass += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(fns)} W125 tests passed")
    return 0 if n_pass == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
