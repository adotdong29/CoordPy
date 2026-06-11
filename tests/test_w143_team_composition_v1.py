"""W143 $0 self-tests (RUNBOOK §6) — team plumbing, ablation bite, NEG controls, budget parity,
earn gate.  All NIM-free via a deterministic MOCK generator that emits real Python programs (so the v2
verifier + AST extractor run for real).  These validate WIRING, not whether a real model discovers —
that is the NIM bench's job.

Run:  python tests/test_w143_team_composition_v1.py    (or pytest)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from coordpy.moderate_p_family_slate_v1 import build_screen_slate_v1
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.icpc_reflexion_bench_v1 import grade_on_secret_v1
from coordpy import multi_agent_discover_amortize_v1 as MA
from coordpy.role_diverse_algorithm_search_v1 import fake_diversity_control_v1

MINTED_DATE = "2026-06-08"
FAMILY = "count_pairs_sum_le_t"   # W142b-proven COMPLEXITY family; ref extracts a clean scaffold (G3)


TEST_KNOB = 600   # small problem size: SAME code paths, fast verifier subprocess runs (plumbing only;
                  # the real NIM bench uses the production knob so the family stays resistant/discriminating)


def _cand(fam):
    for c in build_screen_slate_v1(knob=TEST_KNOB):
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def _passes_secret(minted, code, timeout_s):
    if not code or not code.strip():
        return False
    p = minted.to_pilot_problem(minted_date=MINTED_DATE)
    ok, _, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
    return bool(ok)


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


_ANALYZE_TEXT = (
    "SPEC:\nRead n and T, then n integers; count something per the statement.\n\n"
    "INVARIANTS:\n- the count is non-negative\n- order of items does not matter\n\n"
    "COMPLEXITY:\nn up to large; brute O(n^2) is too slow; target O(n log n).\n\n"
    "SKETCHES:\n"
    "SKETCH A: sort then two pointers\nSort the array; move a right pointer down while the sum exceeds "
    "the threshold; accumulate the window width.\n"
    "SKETCH B: binary search per element\nSort; for each i binary-search the largest j with a[i]+a[j] "
    "within bound; sum the counts.\n"
    "SKETCH C: bucket / counting\nBucket values and convolve counts under the threshold.\n\n"
    "COUNTEREXAMPLES:\nCASE:\n3 5\n1 2 3\n===\nCASE:\n2 0\n-1 1\n")


class MockGen:
    """Deterministic oracle-mock: ANALYZE -> labelled sketches; brute/impl/scaffold/FNB -> the family
    reference (a correct, extractable program).  Counts calls.  Returns (text, wall_ms) like the NIM gen."""

    def __init__(self, ref_code: str, *, fnb_correct: bool = True, wrong_code: str = "print(0)"):
        self.ref = ref_code
        self.fnb_correct = fnb_correct
        self.wrong = wrong_code
        self.n_calls = 0
        self.kinds: list[str] = []

    def __call__(self, prompt: str, max_tokens: int, temperature: float):
        # NOTE: _efficient_prompt, _scaffold_prompt and _transcript_prompt ALL contain a ```python
        # marker, so we key on each prompt's DISTINCTIVE phrase (order matters), never on the fence.
        self.n_calls += 1
        p = prompt
        if "ANALYSIS team" in p:                              # STRATEGIST (build_analyze_prompt_v1)
            self.kinds.append("analyze")
            return (_ANALYZE_TEXT, 1)
        if "brute-force" in p or "INDEPENDENT reference checker" in p:  # BRUTE-AUTHOR quorum
            self.kinds.append("brute")
            return (_fence(self.ref), 1)
        if "following EXACTLY this approach" in p:            # IMPLEMENTER (sketch-guided) -> correct
            # tag each impl with a unique dead var per approach so distinct sketches -> distinct
            # normalized code (a real model writes different code per sketch); stays correct.
            import re as _re
            mm = _re.search(r"APPROACH \(([^)]+)\)", p)
            tag = (mm.group(1) if mm else "x").replace(" ", "_").replace("/", "_")[:30] or "x"
            self.kinds.append("implement")
            return (_fence(f"_approach_tag = {tag!r}\n" + self.ref), 1)
        if "filling in every blank" in p:                    # AMORTIZER via shared-state (_scaffold_prompt)
            self.kinds.append("scaffold")
            return (_fence(self.ref), 1)
        if "CLOSELY RELATED" in p:                            # AMORTIZER via transcript (_transcript_prompt)
            self.kinds.append("transcript")
            return (_fence(self.ref), 1)
        # otherwise the neutral FNB / efficient prompt (i.i.d. candidates) -> ref iff fnb_correct
        self.kinds.append("fnb")
        return (_fence(self.ref if self.fnb_correct else self.wrong), 1)


def _setup():
    c = _cand(FAMILY)
    template = c.factory(c.knob)
    teacher = mint_problem_v1(template.minted, global_seed=1)
    return c, template, teacher


# ==================================================================================================
def test_team_roles_and_budget_parity():
    """>=3 roles for every TEAM arm (pure config check, all 7 arms); real discovery on representative
    arms confirms the STRATEGIST fires for MA but not ST and the K_d+K_b budget identity holds."""
    c, template, teacher = _setup()
    ref = template.minted.ref_source
    # (i) pure config check, ALL arms (instant, no subprocess)
    for arm in MA.ARM_SLATE:
        cfg = MA.arm_config(arm)
        if cfg.is_team():
            assert cfg.n_active_roles() >= 3, f"{arm}: team arm needs >=3 active roles (got {cfg.n_active_roles()})"
    assert not MA.arm_config("ST").is_team(), "ST is the single-controller baseline, not a team"
    assert MA.arm_config("MA_FULL").is_team(), "MA_FULL must be a team"
    # (ii) real discovery for representative arms: STRATEGIST firing + budget parity
    K_d, K_b = 4, 2
    for arm in ("ST", "MA_FULL"):
        cfg = MA.arm_config(arm)
        res = MA.team_discover_v1(MockGen(ref), teacher, template, config=cfg, K_d=K_d, K_b=K_b,
                                  timeout_s=4.0, minted_date=MINTED_DATE, max_disc_tries=1,
                                  passes_secret_fn=_passes_secret)
        assert res.n_model_calls == K_d + K_b, f"{arm}: budget {res.n_model_calls} != {K_d + K_b}"
        assert res.n_analyze == (1 if cfg.role_diverse else 0), f"{arm}: STRATEGIST firing wrong (n_analyze={res.n_analyze})"
        assert res.discovered, f"{arm}: oracle-mock should discover+extract a scaffold"
    print("PASS test_team_roles_and_budget_parity")


def test_diversity_real_vs_fake():
    """MA_FULL with distinct sketches classifies REAL; the fake-diversity positive control is FAKE_DIVERSE;
    ST (i.i.d.) reports NA."""
    c, template, teacher = _setup()
    ref = template.minted.ref_source
    gen = MockGen(ref)
    ma = MA.team_discover_v1(gen, teacher, template, config=MA.arm_config("MA_FULL"),
                             K_d=4, K_b=2, timeout_s=4.0, minted_date=MINTED_DATE)
    assert ma.diversity_classify == "REAL", f"MA_FULL diversity={ma.diversity_classify} (expected REAL)"
    st = MA.team_discover_v1(MockGen(ref), teacher, template, config=MA.arm_config("ST"),
                             K_d=4, K_b=2, timeout_s=4.0, minted_date=MINTED_DATE)
    assert st.diversity_classify == "NA", f"ST diversity={st.diversity_classify} (expected NA)"
    assert fake_diversity_control_v1().classify() == "FAKE_DIVERSE", "fake-diversity control must BITE"
    print("PASS test_diversity_real_vs_fake")


def test_transfer_ablation_prompts_differ():
    """The shared-state / transcript / empty / none transfer modes produce DISTINCT amortizer prompts
    (RUNBOOK §14 Q2 3-arm).  Shared-state embeds the holed skeleton; transcript embeds the RAW winner;
    empty uses the content-free tutor; none == the FNB."""
    c, template, teacher = _setup()
    ref = template.minted.ref_source
    gen = MockGen(ref)
    disc = MA.team_discover_v1(gen, teacher, template, config=MA.arm_config("MA_FULL"),
                               K_d=4, K_b=2, timeout_s=4.0, minted_date=MINTED_DATE)
    assert disc.scaffold is not None and disc.winner_code
    mp = teacher.to_pilot_problem(minted_date=MINTED_DATE)
    empty = MA.make_negative_control_tutor_v1(template)
    p_ss = MA.amortize_prompt_v1(mp, transfer=MA.TRANSFER_SHARED_STATE, scaffold=disc.scaffold, winner_code=disc.winner_code, empty_tutor=empty)
    p_tr = MA.amortize_prompt_v1(mp, transfer=MA.TRANSFER_TRANSCRIPT, scaffold=disc.scaffold, winner_code=disc.winner_code, empty_tutor=empty)
    p_se = MA.amortize_prompt_v1(mp, transfer=MA.TRANSFER_EMPTY, scaffold=disc.scaffold, winner_code=disc.winner_code, empty_tutor=empty)
    p_no = MA.amortize_prompt_v1(mp, transfer=MA.TRANSFER_NONE, scaffold=disc.scaffold, winner_code=disc.winner_code, empty_tutor=empty)
    assert p_ss and p_tr and p_se and p_no
    assert len({p_ss, p_tr, p_se, p_no}) == 4, "all four transfer prompts must differ"
    assert disc.winner_code.strip() in p_tr, "transcript transfer must embed the RAW winner code"
    assert disc.winner_code.strip() not in p_ss, "shared-state must NOT paste the raw winner (it abstracts to a holed skeleton)"
    # the raw transcript carries >= the abstracted skeleton's tokens (a shared-state win is structure, not token count)
    assert len(p_tr) >= len(p_se), "transcript should carry >= the structure-empty tokens"
    print("PASS test_transfer_ablation_prompts_differ")


def test_fragile_mock_ma_discovers_where_st_fails():
    """The pipeline can EXPRESS the load-bearing effect: with a fragile i.i.d. distribution (FNB
    candidates WRONG) but sketch-guided implements CORRECT, MA_FULL discovers while ST fails — and the
    amortize then gives MA the win, ST a KEEP.  (The mock injects the effect; the NIM bench tests
    whether it is real.)"""
    c, template, teacher = _setup()
    ref = template.minted.ref_source
    # ST i.i.d. discovery: FNB candidates are WRONG -> no verified winner -> disc=False -> KEEP
    st = MA.team_discover_v1(MockGen(ref, fnb_correct=False), teacher, template,
                             config=MA.arm_config("ST"), K_d=4, K_b=2, timeout_s=4.0,
                             minted_date=MINTED_DATE, passes_secret_fn=_passes_secret)
    # MA role-diverse discovery: IMPLEMENT (sketch-guided) candidates are CORRECT -> disc=True
    ma = MA.team_discover_v1(MockGen(ref, fnb_correct=False), teacher, template,
                             config=MA.arm_config("MA_FULL"), K_d=4, K_b=2, timeout_s=4.0,
                             minted_date=MINTED_DATE, passes_secret_fn=_passes_secret)
    assert not st.discovered, "ST (fragile i.i.d.) should FAIL to discover when FNB candidates are wrong"
    assert ma.discovered, "MA (sketch-guided) should discover even when i.i.d. FNB is wrong"
    # amortize one member: ST KEEPs (scaffold None -> B0), MA solves via the scaffold
    member = mint_problem_v1(template.minted, global_seed=100)
    brutes = [ref]  # a correct brute anchors the cluster
    st_pass = MA.amortize_member_v1(MockGen(ref, fnb_correct=False), member, template,
                                    config=MA.arm_config("ST"), discover=st, brutes=brutes, K_a=4,
                                    timeout_s=4.0, minted_date=MINTED_DATE, b0_pass=False,
                                    passes_secret_fn=_passes_secret)
    ma_pass = MA.amortize_member_v1(MockGen(ref, fnb_correct=False), member, template,
                                    config=MA.arm_config("MA_FULL"), discover=ma, brutes=brutes, K_a=4,
                                    timeout_s=4.0, minted_date=MINTED_DATE, b0_pass=False,
                                    passes_secret_fn=_passes_secret)
    assert st_pass is False, "ST KEEP == B0(False) when discovery failed"
    assert ma_pass is True, "MA scaffolded amortize should solve the member"
    print("PASS test_fragile_mock_ma_discovers_where_st_fails")


def test_budget_parity_gate_bites():
    """team_budget_parity_v1 flips ok=False when an arm exceeds its declared discovery/amortize budget."""
    ok_report = MA.team_budget_parity_v1(M=10, K_a=4, K_d=10, K_b=5)
    assert ok_report.same_budget_identity_holds, "clean budgets should hold"
    bad = MA.team_budget_parity_v1(M=10, K_a=4, K_d=10, K_b=5,
                                   observed={"MA_FULL": {"discovery": 99, "amortize": 40}})
    assert not bad.same_budget_identity_holds, "an over-budget arm MUST flip the identity"
    print("PASS test_budget_parity_gate_bites")


def test_earn_gate_logic():
    """The strict earn gate: a clean fragile-band earn PASSES; DPI-band fail (ST disc-rate=1) FAILS;
    a tie-ST with no ablation collapse FAILS (not load-bearing)."""
    clean = MA.apply_team_earn_gate_v1(ma_full_pp_over_a1=40.0, ma_full_pp_over_b0=30.0,
                                       ma_minus_st_pp=30.0, n_modes=2, neg_le_b0=True, ma_gt_neg=True,
                                       diversity_real=True, st_disc_rate=0.3)
    assert clean.earned, f"clean fragile-band earn should pass: {clean.reasons}"
    dpi = MA.apply_team_earn_gate_v1(ma_full_pp_over_a1=40.0, ma_full_pp_over_b0=30.0,
                                     ma_minus_st_pp=30.0, n_modes=2, neg_le_b0=True, ma_gt_neg=True,
                                     diversity_real=True, st_disc_rate=1.0)
    assert not dpi.earned and not dpi.dpi_band_ok, "DPI-band fail (ST never fails to discover) must block the earn"
    tie = MA.apply_team_earn_gate_v1(ma_full_pp_over_a1=40.0, ma_full_pp_over_b0=30.0,
                                     ma_minus_st_pp=0.0, n_modes=2, neg_le_b0=True, ma_gt_neg=True,
                                     diversity_real=True, st_disc_rate=0.3)
    assert not tie.earned and not tie.load_bearing, "MA ties ST with no ablation collapse => not load-bearing"
    tie_ablation = MA.apply_team_earn_gate_v1(ma_full_pp_over_a1=40.0, ma_full_pp_over_b0=30.0,
                                              ma_minus_st_pp=0.0, n_modes=2, neg_le_b0=True, ma_gt_neg=True,
                                              diversity_real=True, st_disc_rate=0.3, ablation_collapse=True)
    assert tie_ablation.earned and tie_ablation.load_bearing, "ablation collapse proves load-bearing even if MA~ST aggregate"
    neg = MA.apply_team_earn_gate_v1(ma_full_pp_over_a1=40.0, ma_full_pp_over_b0=30.0,
                                     ma_minus_st_pp=30.0, n_modes=2, neg_le_b0=False, ma_gt_neg=True,
                                     diversity_real=True, st_disc_rate=0.3)
    assert not neg.earned, "NEG>B0 (fake-different team lifts) must block the earn"
    print("PASS test_earn_gate_logic")


def main() -> int:
    test_team_roles_and_budget_parity()
    test_diversity_real_vs_fake()
    test_transfer_ablation_prompts_differ()
    test_fragile_mock_ma_discovers_where_st_fails()
    test_budget_parity_gate_bites()
    test_earn_gate_logic()
    print("\nALL W143 $0 self-tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
