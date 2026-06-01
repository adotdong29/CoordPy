"""W126 — tests for family_adapted_repair_synthesis_v1 (falsifiability-first).

Validated by DIRECT EXECUTION (``python tests/test_w126_family_adapted_repair_synthesis_v1.py``)
because the local pytest/attrs env is broken (see W124/W125 notes).  Each ``test_*`` raises
on failure; ``main`` runs all and prints a PASS/FAIL tally.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_adapted_repair_synthesis_v1 as S  # noqa: E402
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402


def _synth():
    prob, pool = M.synthetic_contract_problem()
    corpus = S.load_exposed_teacher_corpus_v1(["sum"])
    motifs = S.derive_family_motifs_v1(corpus)
    return prob, pool, motifs


# ---------------------------------------------------------------- no-leakage guard

def test_leakage_guard_bites_on_full_secret_case():
    prob, _ = M.synthetic_contract_problem()
    g = S.SynthesisLeakageGuardV1(prob)
    # secret case ("10 20\n","30\n") -> full "10 20\n30" must be caught
    assert not g.check("noise 10 20\n30 noise").clean, "guard failed to catch full secret case"


def test_leakage_guard_passes_clean_code():
    prob, _ = M.synthetic_contract_problem()
    g = S.SynthesisLeakageGuardV1(prob)
    assert g.check("def main():\n    a,b=map(int,input().split())\n    print(a+b)").clean


def test_leakage_guard_provenance_clean_coincidence_not_flagged():
    """A secret byte-run that ALREADY appears in the provenance (a base-model generation
    literal, e.g. an emoticon coinciding with a secret answer) is NOT a leak."""
    prob, _ = M.synthetic_contract_problem()
    g = S.SynthesisLeakageGuardV1(prob)
    # synthetic secret has full-case "10 20\n30"; put it in provenance => coincidence
    g.set_provenance(["# a generation that happens to contain 10 20\n30 as a literal"])
    assert g.check("x = '10 20\\n30'  # provenance-clean").clean or \
        g.check("noise 10 20\n30 noise").clean, "provenance coincidence wrongly flagged"


def test_leakage_guard_flags_injection_absent_from_provenance():
    """A secret run that is NOT in the provenance is the real injection signature."""
    prob, _ = M.synthetic_contract_problem()
    g = S.SynthesisLeakageGuardV1(prob)
    g.set_provenance(["totally unrelated source code with no secrets"])
    assert not g.check("hardcoded 10 20\n30 injected").clean, \
        "injected secret run (absent from provenance) must be flagged"


def test_leakage_guard_catches_target_accepted_lines():
    prob, _ = M.synthetic_contract_problem()
    acc = "def solve_the_exact_target_problem_uniquely():\n    return 12345"
    g = S.SynthesisLeakageGuardV1(prob, target_accepted_texts=[acc])
    assert not g.check("x\n" + "def solve_the_exact_target_problem_uniquely():\n").clean


# ---------------------------------------------------------------- teacher corpus

def test_teacher_corpus_nonempty_and_disjoint():
    corpus = S.load_exposed_teacher_corpus_v1(["sum"])
    assert len(corpus) > 0, "exposed teacher corpus empty"
    # disjointness: no teacher whose problem short-name is a (fake) target
    fake_targets = [corpus[0].problem_short]
    filtered = S.load_exposed_teacher_corpus_v1(fake_targets)
    assert all(s.problem_short != corpus[0].problem_short for s in filtered), \
        "teacher corpus admitted a same-named target problem"


def test_motifs_derived():
    corpus = S.load_exposed_teacher_corpus_v1(["sum"])
    m = S.derive_family_motifs_v1(corpus)
    assert m.n_solutions == len(corpus) and m.n_problems >= 1
    assert sum(m.idiom_freq.values()) >= 1


# ---------------------------------------------------------------- synthesis operators

def test_s1_splice_produces_new_programs():
    _prob, pool, _m = _synth()
    out = S.synth_splice_v1(pool)
    originals = {M._code_norm_sha(c) for c in
                 [pool.a0_code, *pool.a1_codes, *pool.b_codes]}
    assert len(out) >= 1, "S1 splice produced nothing"
    assert any(M._code_norm_sha(c) not in originals for c in out), \
        "S1 splice produced only verbatim originals (not new trajectories)"


def test_s3_motif_harden_produces_candidates():
    _prob, pool, m = _synth()
    out = S.synth_motif_harden_v1(pool, m)
    assert len(out) >= 1, "S3 motif-harden produced nothing"


# ---------------------------------------------------------------- consensus (arsenal)

def test_consensus_trust_beats_majority_on_minority_correct():
    """FALSIFIABILITY: the synthetic pool has ONE correct minority generation (a+b) among
    ten wrong (a*b).  Naive MAJORITY consensus must FAIL (wrong is the plurality) while the
    arsenal's TRUST-WEIGHTED / sample-passers consensus must PASS ⇒ the trust mechanism is
    load-bearing, not decoration."""
    prob, pool, _m = _synth()
    plane = M.AuditedGraderPlaneV1(prob, caller_agent_id="t", timeout_s=4)
    evs = {e.variant: e for e in S.eval_output_consensus_v1(prob, pool, plane, timeout_s=3)}
    assert not evs["majority"].passed_all_secret, \
        "majority consensus unexpectedly passed (test does not discriminate)"
    assert evs["trust_weighted"].passed_all_secret, "trust-weighted consensus failed to recover minority"
    assert evs["sample_passers"].passed_all_secret, "sample-passers consensus failed"


# ---------------------------------------------------------------- capped runner safety

def test_capped_runner_bounds_infinite_output():
    out, status, _rc = S._run_capped_v1("while True:\n    print('x'*1000)\n", "",
                                        timeout_s=3, max_bytes=200000)
    assert status in ("capped", "timeout"), f"infinite-print not bounded: {status}"
    assert (out is None) or (len(out) <= 400000), "output not capped"


def test_capped_runner_normal_output():
    out, status, _rc = S._run_capped_v1(
        "print(sum(map(int,input().split())))", "2 3\n", timeout_s=3)
    assert status == "ok" and out is not None and out.decode().strip() == "5"


# ---------------------------------------------------------------- earn gate (P1/P2)

def _mk(pid, *, unsolved=True, blind=False, oracle=False, cons=False,
        cons_variant="", blind_op="", leak=True):
    return S.ProblemSynthResultV1(
        problem_id=pid, short_name=pid, was_unsolved=unsolved, n_synth_candidates=10,
        oracle_program_pass=oracle, oracle_program_op=("S1_splice" if oracle else ""),
        blind_program_pass=blind, blind_program_op=blind_op,
        consensus_pass=cons, consensus_variant=cons_variant, leakage_clean=leak,
        trace_cid="cid")


def test_earn_gate_earned_on_two_distinct_families():
    rs = [_mk("p1", blind=True, oracle=True, blind_op="S2_digest_repair"),
          _mk("p2", cons=True, oracle=True, cons_variant="trust_weighted")]
    v = S.apply_synthesis_earn_gate_v1(rs)
    assert v.earned and v.verdict_label == "FRESH_RESISTANT_PILOT_EARNED_SYNTHESIS_HEADROOM"
    assert v.blind_new_solved == 2 and v.p1_two_distinct_new and v.p2_two_distinct_families


def test_earn_gate_not_earned_thin_single_family():
    rs = [_mk("p1", blind=True, oracle=True, blind_op="S1_splice"),
          _mk("p2", blind=True, oracle=True, blind_op="S1_splice")]  # same op -> 1 family
    v = S.apply_synthesis_earn_gate_v1(rs)
    assert not v.earned and v.verdict_label == "FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_THIN"


def test_earn_gate_dead_on_zero_oracle():
    rs = [_mk("p1"), _mk("p2"), _mk("p3")]  # all unsolved, nothing solved
    v = S.apply_synthesis_earn_gate_v1(rs)
    assert not v.earned and v.verdict_label == "FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_DEAD"
    assert v.oracle_new_solved == 0 and v.blind_new_solved == 0


def test_earn_gate_killed_on_leakage():
    rs = [_mk("p1", blind=True, oracle=True, blind_op="S2_digest_repair", leak=False),
          _mk("p2", cons=True, oracle=True, cons_variant="majority")]
    v = S.apply_synthesis_earn_gate_v1(rs)
    assert not v.earned and v.verdict_label == "SYNTHESIS_INVALID_LEAKAGE"


def test_earn_gate_requires_two_not_one():
    rs = [_mk("p1", blind=True, oracle=True, blind_op="S2_digest_repair")]  # only ONE new
    v = S.apply_synthesis_earn_gate_v1(rs)
    assert not v.earned, "one blind win must NOT earn the pilot"


# ---------------------------------------------------------------- determinism + end-to-end

def test_trace_cid_deterministic():
    prob, pool, m = _synth()
    g = S.SynthesisLeakageGuardV1(prob)
    r1 = S.synthesize_and_measure_problem_v1(prob, pool, m, g, was_unsolved=True, timeout_s=4)
    r2 = S.synthesize_and_measure_problem_v1(prob, pool, m, g, was_unsolved=True, timeout_s=4)
    assert r1.trace_cid == r2.trace_cid, "synthesis trace CID not deterministic"


def test_end_to_end_synthetic_is_solvable_and_clean():
    """On the (easy) synthetic problem the pipeline CAN create a blind secret-passer and is
    leakage-clean — the positive control that the measurement is not vacuously zero."""
    prob, pool, m = _synth()
    g = S.SynthesisLeakageGuardV1(prob)
    r = S.synthesize_and_measure_problem_v1(prob, pool, m, g, was_unsolved=True, timeout_s=4)
    assert r.leakage_clean
    assert r.blind_new_pass, "pipeline failed to solve a solvable synthetic (vacuous)"


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    npass = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            npass += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n  {npass}/{len(tests)} passed")
    return 0 if npass == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
