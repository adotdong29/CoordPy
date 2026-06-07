"""W139 unit tests — capability-matched controller + large-probe counterexample revival (no NIM)."""
from __future__ import annotations

from coordpy.headroom_band_slate_v3 import FUNC_FACTORIES
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.exact_oracle_witness_v1 import (
    build_witness_probe_set_v1, find_counterexample_witness_v1)
from coordpy.band_mechanism_bench_v1 import fake_different_report_v1
from coordpy.capability_matched_witness_compiler_v1 import (
    ACT_KEEP_PLAIN, ACT_PLAIN, ACT_WITNESS_APPLY, CM_ARM, DEFAULT_TAU_WU,
    WitnessUsabilityV1, build_combined_probe_set_v1, build_large_probe_set_v1,
    capability_matched_is_genuinely_new_v1, run_capability_matched_arm_v1)

_OK_CODE = "```python\nimport sys\nprint(0)\n```"


def _mock_gen(prompt, max_tokens, temperature):
    return (_OK_CODE, 5)


def _func_problem(knob=1500, seed=139_950_001):
    tmpl = FUNC_FACTORIES["subarrays_sum_and_range"](knob)
    mp = mint_problem_v1(tmpl.minted, global_seed=seed, timeout_s=1.0)
    return tmpl, mp


# ---- capability prior routing -------------------------------------------------------

def test_witness_usability_eligibility_threshold():
    weak = WitnessUsabilityV1("8b", "small", 6, 5, 1, DEFAULT_TAU_WU)     # 0.20 < 0.34
    strong = WitnessUsabilityV1("70b", "strong", 6, 5, 4, DEFAULT_TAU_WU)  # 0.80 >= 0.34
    assert not weak.witness_eligible
    assert strong.witness_eligible
    assert WitnessUsabilityV1("x", "mid", 6, 0, 0, DEFAULT_TAU_WU).witness_eligible is False  # no data


# ---- large-probe revival (the W138 2nd-mode repair) ---------------------------------

def test_large_probe_revives_dead_counterexample_mode():
    tmpl, mp = _func_problem()
    naive = mp.naive_source
    small = build_witness_probe_set_v1(tmpl.minted, mp, witness_seed=999_139, timeout_s=2.0)
    large = build_large_probe_set_v1(tmpl.minted, mp, witness_seed=999_139)
    w_small = find_counterexample_witness_v1(naive, mp, small, tmpl.minted, timeout_s=2.0)
    w_large = find_counterexample_witness_v1(naive, mp, large, tmpl.minted, timeout_s=2.0)
    # W138 small-probe cap (400 tok) misses it; the large-probe search finds it, leakage-clean
    assert not w_small.found()
    assert w_large.found()
    assert w_large.leakage_clean
    assert w_large.probe_input_tokens > 400


def test_combined_probe_carries_large_counterexamples():
    tmpl, mp = _func_problem()
    combined = build_combined_probe_set_v1(graded_template=tmpl.minted, probe_template=tmpl.minted,
                                           problem=mp, witness_seed=999_139)
    assert len(combined.small) > 0
    # disjoint from graded secret inputs
    secret = {i for i, _ in mp.secret_cases}
    assert all(inp not in secret for inp, _ in combined.small)


# ---- the capability-matched controller (same-budget K) ------------------------------

def test_controller_keeps_on_ineligible_tier_is_exactly_A1():
    tmpl, mp = _func_problem()
    probe = build_witness_probe_set_v1(tmpl.minted, mp, witness_seed=999_139, timeout_s=2.0)
    out, tr = run_capability_matched_arm_v1(
        seed=1, template=tmpl.minted, problem=mp, probe=probe, gen=_mock_gen, K=5,
        temperature=0.7, max_tokens=64, timeout_s=2.0, minted_date="2026-06-07",
        witness_eligible=False)
    # KEEP path: attempt 0 PLAIN, then all KEEP_PLAIN -> structurally == A1 (never the witness)
    assert out.n_model_calls == 5
    assert tr.actions[0] == ACT_PLAIN
    assert all(a == ACT_KEEP_PLAIN for a in tr.actions[1:])
    assert ACT_WITNESS_APPLY not in tr.actions
    # an ineligible (KEEP) run is honest non-action — NOT counted as a mechanism rescue
    audit = capability_matched_is_genuinely_new_v1(tr)
    assert audit["is_keep_noop"] is True
    assert audit["genuinely_new"] is False


def test_controller_same_budget_and_arm_id():
    tmpl, mp = _func_problem()
    probe = build_combined_probe_set_v1(graded_template=tmpl.minted, probe_template=tmpl.minted,
                                        problem=mp, witness_seed=999_139)
    out, tr = run_capability_matched_arm_v1(
        seed=1, template=tmpl.minted, problem=mp, probe=probe, gen=_mock_gen, K=5,
        temperature=0.7, max_tokens=64, timeout_s=2.0, minted_date="2026-06-07",
        witness_eligible=True)
    assert out.arm_id == CM_ARM
    assert out.n_model_calls == 5
    assert len(tr.actions) == 5
    assert tr.all_leakage_clean


# ---- fake-different discipline -------------------------------------------------------

def test_fake_different_bites_with_cm_and_nb_real():
    fd = fake_different_report_v1(real_arm_ids=("Cm", "Nb", "C0"))
    assert fd.bites
    assert "M3" in fd.fake_arms and "B0" in fd.fake_arms
    assert "Cm" in fd.real_arms and "Nb" in fd.real_arms
