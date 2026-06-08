"""W142 self-tests — moderate-`p` family screen + slate + discover/amortize accounting.

Fast $0 tests (no NIM): the decisive gate is G3 (gated-accumulator extractability), which must ADMIT the
counting/two-deque veins and REJECT the prefix-hash + binary-search-on-answer negative controls — machine-
checking the W142 de-risk finding.  Plus: the gates BITE on deliberately-bad inputs, the mock-gen p
measurement + Wilson band admission, the budget identity, and the W141 regression fixture.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from coordpy.moderate_p_family_slate_v1 import (  # noqa: E402
    build_screen_slate_v1, screen_slate_fingerprint_cid_v1, MODERATE_P_LO, MODERATE_P_HI,
    NEW_CX_FACTORIES, VEIN_SORT_TWO_POINTER, VEIN_TWO_DEQUE)
from coordpy.moderate_p_family_screen_v1 import (  # noqa: E402
    dollar0_gates_v1, screen_family_v1, summarize_screen_v1, measure_fair_p_v1, FNB_PROMPTS)
from coordpy.discover_amortize_accounting_v1 import (  # noqa: E402
    amortized_budget_parity_v1, per_member_superiority_pp_v1, predicted_solve_rate_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402


# ----------------------------------------------------------------- slate

def test_slate_builds_and_fingerprint_is_deterministic():
    slate = build_screen_slate_v1()
    assert len(slate) >= 8
    fams = [c.family for c in slate]
    assert "count_pairs_sum_le_t" in fams                  # the W141 win, re-confirmed
    assert "count_pairs_product_le_t" in fams              # NEW
    assert "count_triples_sum_lt_t" in fams                # NEW
    assert "count_subarrays_range_le_l" in fams            # NEW two-deque
    # >= 2 distinct technique veins among the extractable candidates
    veins = {c.vein for c in slate if c.expect_extractable}
    assert VEIN_SORT_TWO_POINTER in veins and VEIN_TWO_DEQUE in veins
    assert screen_slate_fingerprint_cid_v1() == screen_slate_fingerprint_cid_v1()  # deterministic


# ----------------------------------------------------------------- G3 extractability (the decisive gate, $0)

def test_g3_admits_extractable_veins_and_rejects_controls():
    """G3 (compiled tutor AND n_pred_holes>=1) must match every candidate's locked prediction."""
    slate = build_screen_slate_v1()
    for c in slate:
        g = dollar0_gates_v1(c, timeout_s=3.0)
        assert g.g3_extractable == c.expect_extractable, (
            f"{c.family}: G3={g.g3_extractable} expected {c.expect_extractable} ({g.g3_reason})")


def test_g3_new_counting_families_have_gated_accumulator():
    for fam in ("count_pairs_product_le_t", "count_triples_sum_lt_t"):
        c = next(x for x in build_screen_slate_v1() if x.family == fam)
        g = dollar0_gates_v1(c, timeout_s=3.0)
        assert g.g3_extractable and g.n_pred_holes >= 1 and g.n_add_holes >= 1


def test_g3_rejects_prefix_hash_no_gating_predicate():
    c = next(x for x in build_screen_slate_v1() if x.family == "count_subarrays_sum_divisible_k")
    g = dollar0_gates_v1(c, timeout_s=3.0)
    assert g.g3_extractable is False          # prefix-mod-hash: technique in dict maintenance, no gate
    assert g.n_pred_holes == 0


def test_g3_rejects_binary_search_on_answer_no_accumulator():
    c = next(x for x in build_screen_slate_v1() if x.family == "kth_smallest_pair_distance")
    g = dollar0_gates_v1(c, timeout_s=3.0)
    assert g.g3_extractable is False          # printed answer is a reassignment, not an accumulator


# ----------------------------------------------------------------- gates BITE

def test_g4_novelty_bites_on_duplicate():
    c = next(x for x in build_screen_slate_v1() if x.family == "count_pairs_sum_le_t")
    mt = c.factory(c.knob).minted
    g = dollar0_gates_v1(c, known_algo_sigs=[mt.algo_sig], known_statements=[mt.statement])
    assert g.g4_novel is False                # its own sig/statement marked as already-known -> not novel


def test_g2_discriminating_on_a_counting_family():
    """ref passes the hidden bank, naive FAILS it (TLE).  Slow-ish (naive TLE), bounded timeout."""
    c = next(x for x in build_screen_slate_v1() if x.family == "count_triples_sum_lt_t")
    g = dollar0_gates_v1(c, timeout_s=3.0)
    assert g.g2_discriminating is True


# ----------------------------------------------------------------- mock-gen fair-p + admission

def _mock_gen_rate(rate_flags):
    """A deterministic mock generator: returns the family ref (correct) or a TLE-naive by a fixed pattern
    so the measured p matches ``rate_flags`` exactly (ref on True, a wrong stub on False)."""
    st = {"n": 0}

    def make(ref_src):
        def gen(prompt, max_tokens, temperature):
            i = st["n"]; st["n"] += 1
            ok = rate_flags[i % len(rate_flags)]
            code = ref_src if ok else "import sys\nprint(-1)\n"   # -1 != true answer -> fails secret
            return (f"```python\n{code}\n```", 1)
        return gen
    return make


def test_mock_p_measurement_and_band_admission():
    c = next(x for x in build_screen_slate_v1() if x.family == "count_pairs_sum_le_t")
    ref = c.factory(c.knob).minted.ref_source
    # 3/12 pass -> p_hat = 0.25, squarely in [0.10,0.50]
    flags = [True, False, False, False] * 3
    gen = _mock_gen_rate(flags)(ref)
    res = screen_family_v1(c, gen=gen, K_screen=12, prompt_indices=(0,), timeout_s=4.0)
    assert abs(res.p_median - 0.25) < 1e-9
    assert MODERATE_P_LO <= res.p_median <= MODERATE_P_HI
    assert res.wilson_lo > 0.0 and res.wilson_hi < 1.0    # Wilson excludes 0 and 1 at n=12, k=3
    assert res.in_band is True and res.admitted is True


def test_mock_p_out_of_band_not_admitted():
    c = next(x for x in build_screen_slate_v1() if x.family == "count_pairs_sum_le_t")
    ref = c.factory(c.knob).minted.ref_source
    # 11/12 pass -> p_hat ~ 0.92, ABOVE the band -> not admitted
    flags = [True, True, True, True, True, True, True, True, True, True, True, False]
    gen = _mock_gen_rate(flags)(ref)
    res = screen_family_v1(c, gen=gen, K_screen=12, prompt_indices=(0,), timeout_s=4.0)
    assert res.p_median > MODERATE_P_HI
    assert res.in_band is False and res.admitted is False


def test_fnb_prompt_bank_is_neutral():
    """The neutral-prompt bank must name NO technique / efficiency / size cue (the W141 inflators)."""
    banned = ("efficient", "time limit", "largest input", "O(", "complexity", "two pointer",
              "monotonic", "prefix", "fenwick", "binary search", "hash", "sort")
    for builder in FNB_PROMPTS:
        text = builder("PROBLEM_STATEMENT_PLACEHOLDER").lower()
        # strip the placeholder; check only the instruction wrapper
        wrapper = text.replace("problem_statement_placeholder", "")
        for w in banned:
            assert w not in wrapper, f"neutral prompt leaks cue {w!r}"


# ----------------------------------------------------------------- discover/amortize accounting

def test_amortized_budget_identity():
    rep = amortized_budget_parity_v1(M=20, K_a=4, K_d=12)
    assert rep.b0_family_total == 80
    assert rep.st_family_total == 12 + 80
    assert rep.per_member_discovery_overhead == 0.6           # 12/20
    assert rep.same_budget_identity_holds is True
    # a declared/observed mismatch flips the identity
    bad = amortized_budget_parity_v1(M=20, K_a=4, K_d=12, observed={"STd": 999})
    assert bad.same_budget_identity_holds is False


def test_per_member_superiority_matches_theory():
    # ST-B0 at equal amortize budget = (1-p)^K_a * 100 (q=1)
    assert per_member_superiority_pp_v1(0.10, 4) == pytest.approx(100 * 0.9 ** 4, abs=0.01)
    assert per_member_superiority_pp_v1(0.50, 4) == pytest.approx(100 * 0.5 ** 4, abs=0.01)  # ~6.25
    # vanishes at the high-p extreme (the W141 NSL/inversions collapse)
    assert per_member_superiority_pp_v1(0.83, 4) < 1.5
    assert predicted_solve_rate_v1(0.0, 5) == 0.0 and predicted_solve_rate_v1(1.0, 5) == 1.0


# ----------------------------------------------------------------- screen verdict span logic

def test_summarize_span_logic_two_modes_or_three_families():
    slate = build_screen_slate_v1()
    # fabricate three admitted results spanning 2 veins + 2 modes
    from coordpy.moderate_p_family_screen_v1 import FamilyScreenResultV1, Dollar0GatesV1, FairPResultV1
    g = Dollar0GatesV1(True, True, True, True, 1, 1, "ok", True)
    fp = FairPResultV1(0, 12, 3, tuple([True, False, False, False] * 3))
    def mk(fam, vein, mode):
        return FamilyScreenResultV1(fam, vein, mode, 50000, g, (fp,), 0.25, 0.25, 0.25,
                                    0.09, 0.53, 12, 3, True, True, True)
    res = [mk("count_pairs_sum_le_t", VEIN_SORT_TWO_POINTER, "COMPLEXITY_BLIND"),
           mk("count_pairs_product_le_t", VEIN_SORT_TWO_POINTER, "COMPLEXITY_BLIND"),
           mk("subarrays_sum_and_range", VEIN_TWO_DEQUE, "HIDDEN_EDGE_STATE_MISS")]
    v = summarize_screen_v1(res, slate)
    assert v.n_admitted == 3 and v.span_ok is True
    assert v.lane_alpha_success is True       # >=3 families AND >=2 veins
    assert len(v.admitted_modes) == 2


# ----------------------------------------------------------------- W141 regression fixture

def test_w141_count_pairs_sum_still_extracts():
    """The W141 win family must still compile a clean gated-accumulator tutor (extractor unchanged)."""
    c = next(x for x in build_screen_slate_v1() if x.family == "count_pairs_sum_le_t")
    g = dollar0_gates_v1(c, timeout_s=3.0)
    assert g.g3_extractable is True and g.n_pred_holes >= 1 and g.n_add_holes >= 1
