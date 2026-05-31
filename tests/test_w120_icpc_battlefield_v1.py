"""W120 tests — official-ICPC multi-surface battlefield + stdin/stdout reflexion bench.

Pure / NIM-free.  Covers the admission rule, manifest CID determinism, the typed
exclusion audit, the deterministic float oracle, the >=30 count-gate certification flip,
a FALSIFIABILITY test (a truncated <30 listing must NOT earn a pilot), the LCB-inherited
decision-CID invariant, the stratified slice selector, and the bench report-shape
compatibility with the VERBATIM W108 gate evaluator (via a mock generator).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    EXTENDED_TIERS,
    ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
    KIND_PASSFAIL,
    KIND_PASSFAIL_FLOAT,
    TIER_CORE,
    W120_RAW_CLASSIFICATION_SHA256,
    assess_battlefield_admissibility_v1,
    build_battlefield_manifest_v1,
    classify_battlefield_listing_v1,
    core_slice_cid_v1,
    exclusion_audit_v1,
    grader_selftest_summary_v1,
    judge_icpc_output_v1,
    run_battlefield_construction_v1,
    select_battlefield_core_slice_v1,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1,
    IcpcPilotProblemV1,
    run_icpc_reflexion_bench_v1,
)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _evaluate_phase2_gates,
    _mlb_rates,
)

MAVERICK = "meta/llama-4-maverick-17b-128e-instruct"


# ---------------------------------------------------------------- admission / manifest
def test_listing_sha_pinned():
    import hashlib
    import json
    rows = [list(r) for r in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1]
    # the snapshot SHA is over the rich snapshot dicts (results file), not this tuple;
    # here we just assert the pinned constant is a 64-hex string and stable in-module.
    assert len(W120_RAW_CLASSIFICATION_SHA256) == 64
    assert all(c in "0123456789abcdef" for c in W120_RAW_CLASSIFICATION_SHA256)
    # tuple itself is deterministic / unique by (repo, short)
    keys = [(r[0], r[1]) for r in rows]
    assert len(keys) == len(set(keys)) == 51


def test_manifest_counts_and_core_ge_30():
    m = build_battlefield_manifest_v1(fetched_on="2026-05-31")
    assert m.n_seen == 51
    assert m.n_admitted == 49
    assert m.n_core_passfail == 45
    assert m.n_float == 3
    assert m.n_custom_validator == 1
    assert m.n_core_passfail >= 30
    assert m.date_min == "2024-11-11" and m.date_max == "2025-11-13"
    assert set(m.surfaces) == {"ECNA", "RMRC"}


def test_manifest_cid_deterministic():
    a = build_battlefield_manifest_v1(fetched_on="2026-05-31")
    b = build_battlefield_manifest_v1(fetched_on="2099-01-01")  # date must not affect CID
    assert a.manifest_cid() == b.manifest_cid()
    assert a.core_slice_cid() == b.core_slice_cid()
    assert len(a.manifest_cid()) == 64


def test_exclusion_audit_typed():
    probs = classify_battlefield_listing_v1()
    audit = exclusion_audit_v1(probs)
    assert audit.n_seen == 51 and audit.n_admitted == 49 and audit.n_excluded == 2
    assert audit.by_exclusion_reason == {
        "excluded_kind:custom_no_validator": 1,
        "excluded_kind:interactive": 1,
    }
    assert ("icpc_na-rocky-mountain-2025-2026-public_poetictournament"
            in audit.excluded_problem_ids)
    assert ("icpc_na-rocky-mountain-2024-2025-public_alwaysknowwhereyourtowelis"
            in audit.excluded_problem_ids)
    # draftlottery is admitted but as FLOAT (the W119 mislabel correction), not core
    df = [p for p in probs if p.short_name == "draftlottery"][0]
    assert df.admitted and df.tier != TIER_CORE and df.kind == KIND_PASSFAIL_FLOAT


# ---------------------------------------------------------------- the float oracle
def test_float_oracle():
    # tier-1 exact diff
    assert judge_icpc_output_v1(got_stdout="1 2 3", expected="1 2 3",
                                kind=KIND_PASSFAIL)
    assert not judge_icpc_output_v1(got_stdout="1 2", expected="1 2 3",
                                    kind=KIND_PASSFAIL)
    # tier-2 float within tolerance
    assert judge_icpc_output_v1(got_stdout="0.5000001", expected="0.5",
                                kind=KIND_PASSFAIL_FLOAT, float_tol=1e-5)
    # outside tolerance
    assert not judge_icpc_output_v1(got_stdout="0.6", expected="0.5",
                                    kind=KIND_PASSFAIL_FLOAT, float_tol=1e-5)
    # token-count mismatch
    assert not judge_icpc_output_v1(got_stdout="0.5 0.5", expected="0.5",
                                    kind=KIND_PASSFAIL_FLOAT, float_tol=1e-5)
    # non-numeric token must match exactly even under float kind
    assert not judge_icpc_output_v1(got_stdout="YES", expected="NO",
                                    kind=KIND_PASSFAIL_FLOAT, float_tol=1e-5)


# ---------------------------------------------------------------- certification flip
def test_certification_flips_at_30_and_maverick_pilot_admissible():
    res = run_battlefield_construction_v1(verified_on="2026-05-31")
    assert res.admissibility.pilot_admissible is True
    assert res.pilot_earned is True
    assert res.verdict == "CERTIFIABLE_STRONGER_MODEL"
    assert res.n_identity_certifiable_models == 1
    mav = [m for m in res.per_model if m.model_id == MAVERICK][0]
    assert mav.identity_certifiable and mav.pilot_admissible
    # the three UNKNOWN-cutoff models stay C1-blocked
    others = [m for m in res.per_model if m.model_id != MAVERICK]
    assert all((not m.identity_certifiable) for m in others)


def test_lcb_inherited_decision_cid_invariant():
    res = run_battlefield_construction_v1(verified_on="2026-05-31")
    cid = res.to_dict()["lcb_inherited_decision_cid"]
    assert cid.startswith("258b6ed7")  # byte-identical to W114..W119


def test_grader_selftest_each_surface():
    st = grader_selftest_summary_v1()
    assert st["rmrc"]["all_pass"] and st["ecna"]["all_pass"]
    assert st["grader_proven_executable_each_surface"] is True
    assert st["n_cases_passed"] == st["n_cases_run"] == 165


# ---------------------------------------------------------------- FALSIFIABILITY
def test_falsifiability_sub30_listing_earns_no_pilot():
    """A truncated listing with < 30 core pass-fail problems must NOT earn a pilot —
    proving the COUNT (not the grader) is the gate, and that the gate genuinely binds."""
    core_rows = [r for r in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1 if r[3] == KIND_PASSFAIL]
    short = tuple(core_rows[:20])  # only 20 core < 30
    res = run_battlefield_construction_v1(
        listing=short, verified_on="2026-05-31",
        raw_classification_sha256="deadbeef")
    assert res.manifest.n_core_passfail == 20
    assert res.admissibility.meets_min_slice is False
    assert res.admissibility.pilot_admissible is False
    assert res.pilot_earned is False
    assert res.verdict == VERDICT_NONE  # "NO_CERTIFIABLE_STRONGER_MODEL"
    mav = [m for m in res.per_model if m.model_id == MAVERICK][0]
    assert mav.identity_certifiable is False  # C2 fails at 20 < 30


def test_falsifiability_grader_clean_but_we_still_need_30():
    """Even with the grader self-test passing, 29 core must fail; 30 must pass —
    a tight boundary check on MIN_RESISTANT_SLICE."""
    core_rows = [r for r in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1 if r[3] == KIND_PASSFAIL]
    m29 = build_battlefield_manifest_v1(listing=tuple(core_rows[:29]),
                                        fetched_on="2026-05-31")
    a29 = assess_battlefield_admissibility_v1(m29, grader_selftest_summary_v1())
    assert a29.meets_min_slice is False and a29.pilot_admissible is False
    m30 = build_battlefield_manifest_v1(listing=tuple(core_rows[:30]),
                                        fetched_on="2026-05-31")
    a30 = assess_battlefield_admissibility_v1(m30, grader_selftest_summary_v1())
    assert a30.meets_min_slice is True and a30.pilot_admissible is True


# ---------------------------------------------------------------- slice selector
def test_core_slice_selector_deterministic_and_balanced():
    probs = classify_battlefield_listing_v1()
    s1 = select_battlefield_core_slice_v1(probs, n_problems=30)
    s2 = select_battlefield_core_slice_v1(probs, n_problems=30)
    assert len(s1) == 30
    assert [p.problem_id for p in s1] == [p.problem_id for p in s2]
    assert core_slice_cid_v1(s1).startswith("01bf9ef8")
    surfaces = {p.surface for p in s1}
    assert surfaces == {"ECNA", "RMRC"}  # spans both official surfaces
    assert all(p.tier == TIER_CORE for p in s1)  # pilot slice is strict core only


# ---------------------------------------------------------------- bench report shape
def _mock_gen(program: str):
    def gen(prompt, max_tokens, temperature):
        return f"```python\n{program}\n```", 1
    return gen


_DOUBLE_PROB = IcpcPilotProblemV1(
    problem_id="t_double", short_name="double", source_repo="t", contest_date="2025-01-01",
    statement="Read n, print 2n.", kind=KIND_PASSFAIL, float_tol=1e-6,
    samples=(("4\n", "8\n"),), secret_cases=(("4\n", "8\n"), ("10\n", "20\n")))


def test_bench_report_plugs_into_w108_gates_correct():
    gen = _mock_gen("n=int(input())\nprint(n*2)")
    cfg = IcpcBenchConfigV1(K_multi_sample=3, seeds=(1,))
    rep = run_icpc_reflexion_bench_v1(gen=gen, model_id="mock", subset=(_DOUBLE_PROB,),
                                      config=cfg)
    assert rep.a0_mean_pass_at_1 == 1.0 and rep.b_mean_pass_at_1 == 1.0
    # the VERBATIM W108 evaluator must accept the report shape without error
    mlb = _mlb_rates(rep)
    gates = _evaluate_phase2_gates(report=rep, mlb=mlb)
    assert "verdict_label" in gates and gates["a1_pct"] == 100.0
    # A0 passed at attempt 0 ⇒ reflexion NOT invoked
    assert mlb["n_b_invoked_reflexion"] == 0


def test_bench_report_wrong_solution_invokes_reflexion():
    gen = _mock_gen("n=int(input())\nprint(n*3)")  # always wrong
    cfg = IcpcBenchConfigV1(K_multi_sample=3, seeds=(1,))
    rep = run_icpc_reflexion_bench_v1(gen=gen, model_id="mock", subset=(_DOUBLE_PROB,),
                                      config=cfg)
    assert rep.b_mean_pass_at_1 == 0.0
    mlb = _mlb_rates(rep)
    assert mlb["n_b_invoked_reflexion"] == 1 and mlb["n_b_rescued_via_reflexion"] == 0
