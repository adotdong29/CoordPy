"""W140 unit tests — cross-tier tutor bench harness + capability-matched tutor controller (Lane β).

$0 (no NIM): a deterministic fake ``gen`` returns a fixed program so the K-loop, grading, and the
controller's KEEP/APPLY routing are verified without any model.  Asserts the W139 non-negativity
invariant is preserved structurally (an ineligible tier's controller is exactly A1) and the
fake-different + arm-slate discipline.
"""
import pytest

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, FUNC_FACTORIES
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy import family_tutor_compiler_v1 as T
from coordpy import cross_tier_tutor_bench_v1 as B

CX = CX_FACTORIES["count_pairs_sum_le_t"]
FN = FUNC_FACTORIES["subarrays_sum_and_range"]


def _mint(fac, knob, seed):
    tmpl = fac(knob)
    return tmpl, mint_problem_v1(tmpl.minted, global_seed=seed, timeout_s=8.0)


def _gen_returning(code):
    def gen(prompt, max_tokens, temperature):
        return (f"```python\n{code}\n```", {"toks": 1})
    return gen


def test_arm_slate_complete_and_lead_is_t4():
    ids = [a.arm_id for a in B.TUTOR_ARM_SLATE_V1]
    assert ids == ["A0", "A1", "B0", "C0", "T1", "T2", "T3", "T4", "T5", "T6"]
    lead = [a for a in B.TUTOR_ARM_SLATE_V1 if a.is_lead]
    assert len(lead) == 1 and lead[0].arm_id == "T4"
    neg = [a for a in B.TUTOR_ARM_SLATE_V1 if a.is_negative_control]
    assert len(neg) == 1 and neg[0].arm_id == "T6"


def test_observed_kind_is_family_level():
    cx_t, _ = _mint(CX, 50000, 1)
    fn_t, _ = _mint(FN, 1500, 1)
    assert B.tutor_observed_kind_for_template(cx_t) == B.OBS_TIMEOUT       # complexity -> timeout
    assert B.tutor_observed_kind_for_template(fn_t) == B.OBS_WRONG_ANSWER  # hidden-edge -> wrong answer


def test_tutor_arm_passes_with_efficient_gen_fails_with_naive():
    tmpl, prob = _mint(CX, 50000, 140_200_001)
    tut = B.compile_family_tutors_v1(tmpl)[T.TC2_REWRITE]
    o_ok, tr = B.run_tutor_arm_v1(seed=1, template=tmpl, problem=prob, tutor=tut,
                                  gen=_gen_returning(tmpl.minted.ref_source), K=5, temperature=0.7,
                                  max_tokens=1536, timeout_s=8.0, minted_date="2026-06-07", arm_id="T2")
    assert o_ok.final_passed is True and o_ok.n_model_calls == 5 and tr.n_tutor_attempts == 4
    o_bad, _ = B.run_tutor_arm_v1(seed=1, template=tmpl, problem=prob, tutor=tut,
                                  gen=_gen_returning(tmpl.minted.naive_source), K=5, temperature=0.7,
                                  max_tokens=1536, timeout_s=8.0, minted_date="2026-06-07", arm_id="T2")
    assert o_bad.final_passed is False


def test_controller_keep_equals_plain_on_ineligible_tier():
    """W139 non-negativity invariant: an ineligible tier's T4 NEVER applies the tutor (all KEEP/PLAIN),
    so T4 ≡ A1 by construction and cannot hurt."""
    tmpl, prob = _mint(CX, 50000, 140_200_002)
    tut = B.compile_family_tutors_v1(tmpl)[T.TC2_REWRITE]
    o, tr = B.run_tutor_controller_arm_v1(seed=1, template=tmpl, problem=prob, tutor=tut,
                                          gen=_gen_returning(tmpl.minted.naive_source), K=5,
                                          temperature=0.7, max_tokens=1536, timeout_s=8.0,
                                          minted_date="2026-06-07", tutor_eligible=False)
    assert set(tr.actions) <= {B.ACT_PLAIN, B.ACT_KEEP_PLAIN}
    assert B.ACT_TUTOR_APPLY not in tr.actions
    gn = B.tutor_controller_is_genuinely_new_v1(tr)
    assert gn["genuinely_new"] is False and gn["is_keep_noop"] is True


def test_controller_applies_tutor_on_eligible_tier():
    tmpl, prob = _mint(CX, 50000, 140_200_003)
    tut = B.compile_family_tutors_v1(tmpl)[T.TC2_REWRITE]
    o, tr = B.run_tutor_controller_arm_v1(seed=1, template=tmpl, problem=prob, tutor=tut,
                                          gen=_gen_returning(tmpl.minted.ref_source), K=5,
                                          temperature=0.7, max_tokens=1536, timeout_s=8.0,
                                          minted_date="2026-06-07", tutor_eligible=True)
    assert B.ACT_TUTOR_APPLY in tr.actions and o.final_passed is True
    assert B.tutor_controller_is_genuinely_new_v1(tr)["genuinely_new"] is True


def test_tutor_usability_dataclass_eligibility_threshold():
    """Eligibility math: rate >= tau AND n_eligible > 0."""
    elig = B.TutorUsabilityV1(model_id="m", tier="small", family="f", tc_kind=T.TC2_REWRITE,
                              n_probed=6, n_eligible=5, n_flipped=2, tau=0.34)
    assert abs(elig.rate - 0.4) < 1e-9 and elig.tutor_eligible is True
    inelig = B.TutorUsabilityV1(model_id="m", tier="small", family="f", tc_kind=T.TC2_REWRITE,
                                n_probed=6, n_eligible=5, n_flipped=0, tau=0.34)
    assert inelig.rate == 0.0 and inelig.tutor_eligible is False  # the W139 8B floor


def test_fake_different_bites_for_t6():
    fd = B.fake_different_report_v1(real_arm_ids=("T1", "T2", "T3", "T4")).to_dict()
    assert fd["bites"] is True


def test_parser_neutrality_preserved_on_shared_families():
    """W136 I/O-confound regression: every minted secret case of every shared family is parser-neutral."""
    from coordpy.parser_neutral_io_v1 import parser_neutrality_gate_v1
    for fac, knob in [(CX, 50000), (FN, 1500)]:
        tmpl, prob = _mint(fac, knob, 140_200_050)
        hc1 = parser_neutrality_gate_v1([i for i, _ in prob.secret_cases], tmpl.io_shape)
        assert hc1.is_parser_neutral, f"{tmpl.minted.family} must be parser-neutral"
