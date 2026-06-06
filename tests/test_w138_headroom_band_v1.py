"""W138 unit tests — headroom-band slate v3 + band calibration v2 + corpus v3 + mechanism bench.

All $0 (no NIM): structure, determinism, HC1 parser-neutrality (+ W136 confound regression),
HC2 exact-oracle discrimination (functional families + CX ref==brute), the Wilson interval, the
band-admission verdict (W137 bimodality-detector regression), and the fake-different discipline.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordpy.headroom_band_slate_v3 import (
    CX_FACTORIES, FUNC_FACTORIES, build_band_candidates_v3, band_slate_fingerprint_cid_v1,
    _cx_count_pairs_sum_le_t, _ms_mod_then_maxsub, _ce_subarrays_sum_and_range)
from coordpy.parser_neutral_io_v1 import (
    parse_all_tokens_v1, parser_neutrality_gate_v1, render_normal_form_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.headroom_band_calibration_v2 import (
    BAND_LO, BAND_HI, band_verdict_v1, wilson_interval_v1)
from coordpy.model_ladder_calibration_v1 import ModelTemplateStatV1, TemplateCalibrationV1
from coordpy.headroom_band_corpus_v3 import (
    split_seeds_v1, mint_split_v3, summarize_split_v3, corpus_cid_v3)
from coordpy.band_mechanism_bench_v1 import (
    BAND_ARM_SLATE_V1, BAND_ARM_DISPATCH_V1, arm_scored_on_problem_v1, fake_different_report_v1)


# ---------------------------------------------------------------- slate structure + determinism

def test_slate_grid_shape():
    cands = build_band_candidates_v3()
    fams = {c.family for c in cands}
    modes = {c.mode for c in cands}
    assert len(cands) == 18, "9 families x 2 knobs"
    assert len(fams) == 9
    assert modes == {"COMPLEXITY_BLIND", "WRONG_ALGORITHM_ADMISSIBLE", "HIDDEN_EDGE_STATE_MISS"}
    # at least 3 distinct COMPLEXITY families exist (the Path-B backbone)
    cx = {c.family for c in cands if c.mode == "COMPLEXITY_BLIND"}
    assert len(cx) >= 3


def test_slate_fingerprint_stable():
    assert band_slate_fingerprint_cid_v1() == band_slate_fingerprint_cid_v1()
    assert len(band_slate_fingerprint_cid_v1()) == 64


def test_mint_determinism():
    t = _cx_count_pairs_sum_le_t(20000)
    p1 = mint_problem_v1(t.minted, global_seed=900000, timeout_s=1.0)
    p2 = mint_problem_v1(t.minted, global_seed=900000, timeout_s=1.0)
    assert p1.content_cid() == p2.content_cid()


# ---------------------------------------------------------------- HC1 parser-neutrality + W136 regression

def test_hc1_normal_form_passes_flattened_fails():
    t = _cx_count_pairs_sum_le_t(50)
    shape = t.io_shape
    data = {"N": 4, "T": 10, "A": [1, 2, 3, 4]}
    normal = render_normal_form_v1(shape, data)
    flat = " ".join(normal.split()) + "\n"          # W136 confound: whole body on one line
    assert parser_neutrality_gate_v1([normal], shape).is_parser_neutral is True
    assert parser_neutrality_gate_v1([flat], shape).is_parser_neutral is False
    # the all-tokens reader STILL recovers the flattened body — that is exactly why it is a confound
    assert parse_all_tokens_v1(flat, shape) == {"N": 4, "T": 10, "A": [1, 2, 3, 4]}


def test_every_cell_is_parser_neutral_on_public_samples():
    # cheap: mint at tiny knobs so no TLE wait; HC1 is shape-only and knob-independent
    for c in build_band_candidates_v3(cx_knobs=(60,), func_knobs=(60,)):
        p = mint_problem_v1(c.template.minted, global_seed=900001, timeout_s=2.0)
        rec = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], c.template.io_shape)
        assert rec.is_parser_neutral, f"{c.cell_id} not parser-neutral"


# ---------------------------------------------------------------- HC2 exact-oracle discrimination

def test_functional_families_discriminate_wrong_answer():
    # functional families: naive drops a stage/constraint -> WRONG_ANSWER on hidden (fast, no TLE)
    for fac in (_ms_mod_then_maxsub, _ce_subarrays_sum_and_range):
        p = mint_problem_v1(fac(60).minted, global_seed=900002, timeout_s=3.0)
        assert p.gates.admitted, f"{p.name}: {p.gates.reason}"
        assert "WRONG_ANSWER" in p.gates.naive_fail_kinds


def test_cx_ref_equals_brute_on_small():
    # CX ref/brute agreement on small cases (knob tiny so no TLE); discrimination via TIMEOUT needs
    # large N (covered by the build self-test), but ref==brute must hold structurally
    for fam, fac in CX_FACTORIES.items():
        p = mint_problem_v1(fac(60).minted, global_seed=900003, timeout_s=3.0)
        assert p.gates.g_reference_solvable, f"{fam} ref not solvable"
        assert p.gates.g_oracle_small_agreement, f"{fam} brute != ref"
        assert p.gates.n_brute_checked >= 1


# ---------------------------------------------------------------- Wilson interval

def test_wilson_interval():
    lo, hi = wilson_interval_v1(0, 8)
    assert lo == 0.0 and hi < 1.0           # 0/8 -> lower bound 0 (touches extreme)
    lo, hi = wilson_interval_v1(8, 8)
    assert lo > 0.0 and hi == 1.0           # 8/8 -> upper bound 1 (touches extreme)
    lo, hi = wilson_interval_v1(4, 8)
    assert 0.0 < lo < 0.5 < hi < 1.0        # 4/8 -> strictly interior
    lo, hi = wilson_interval_v1(0, 0)
    assert (lo, hi) == (0.0, 1.0)


# ---------------------------------------------------------------- band-admission (W137 bimodality detector)

def _cal(name, s_a0, s_a1, sm_a0, mode="COMPLEXITY_BLIND"):
    strong = ModelTemplateStatV1(model_id="meta/llama-3.3-70b-instruct", tier="strong",
                                 a0_passed=tuple(s_a0), a1_passed=tuple(s_a1),
                                 n_calls=len(s_a0) + len(s_a1) * 5)
    small = ModelTemplateStatV1(model_id="meta/llama-3.1-8b-instruct", tier="small",
                                a0_passed=tuple(sm_a0), a1_passed=(), n_calls=len(sm_a0))
    return TemplateCalibrationV1(template_name=name, family=name, mode=mode,
                                 per_model=(strong, small), hc3_has_headroom=False,
                                 hc4_not_dead=False, discriminates=False, admitted=False, reason="")


def test_band_verdict_culls_saturated():
    T = True
    v = band_verdict_v1(_cal("sat", [T] * 5, [T] * 8, [T] * 5), cell_id="sat@1", knob_value=1)
    assert not v.admitted and v.reason.startswith("HC3")


def test_band_verdict_culls_dead():
    F = False
    v = band_verdict_v1(_cal("dead", [F] * 5, [F] * 8, [F] * 5), cell_id="dead@1", knob_value=1)
    assert not v.admitted and "DEAD" in v.reason


def test_band_verdict_admits_intermediate_discriminating():
    T, F = True, False
    # strong a0=0 (<0.8), a1=4/8=0.5 (in band, Wilson excludes 0,1), small best=0 -> discriminates
    v = band_verdict_v1(_cal("inter", [F] * 5, [T, T, T, T, F, F, F, F], [F] * 5),
                        cell_id="inter@1", knob_value=1)
    assert v.admitted, v.reason
    assert BAND_LO <= v.strong_a1_rate <= BAND_HI
    assert v.hb4_discriminates


def test_band_verdict_culls_no_discrimination():
    T, F = True, False
    # intermediate a1 but the small tier matches the strong best -> HB4 fails
    v = band_verdict_v1(_cal("nodisc", [F] * 5, [T, T, T, T, F, F, F, F], [T] * 5),
                        cell_id="nodisc@1", knob_value=1)
    assert not v.admitted and "DISCRIMINAT" in v.reason.upper()


# ---------------------------------------------------------------- corpus split disjointness

def test_split_seeds_disjoint():
    seeds = {sp: set(split_seeds_v1(sp, 50)) for sp in ("train", "dev", "eval", "frontier")}
    all_seeds = [s for ss in seeds.values() for s in ss]
    assert len(all_seeds) == len(set(all_seeds)), "splits share a seed"
    # disjoint from the W137 (137_*) and calibration (137_900_000) bases
    assert all(s >= 138_000_000 for s in all_seeds)


def test_corpus_cid_stable_and_summary():
    t = _ce_subarrays_sum_and_range(60)
    probs = mint_split_v3("dev", templates=[t], n_replicas=2, timeout_s=2.0, mint_timeout_s=1.0)
    summ = summarize_split_v3(probs)
    assert summ.n_admitted >= 1
    cid1 = corpus_cid_v3({"dev": probs})
    probs2 = mint_split_v3("dev", templates=[t], n_replicas=2, timeout_s=2.0, mint_timeout_s=1.0)
    assert corpus_cid_v3({"dev": probs2}) == cid1


# ---------------------------------------------------------------- mechanism-bench discipline

def test_arm_slate_structure():
    ids = [a.arm_id for a in BAND_ARM_SLATE_V1]
    assert ids == ["A0", "A1", "B0", "C0", "N0", "X1", "X2"]
    lead = [a for a in BAND_ARM_SLATE_V1 if a.is_lead]
    assert len(lead) == 1 and lead[0].arm_id == "X1"
    neg = [a for a in BAND_ARM_SLATE_V1 if a.is_negative_control]
    assert len(neg) == 1 and neg[0].arm_id == "X2"
    assert set(BAND_ARM_DISPATCH_V1) == {"C0", "N0", "X1"}


def test_arm_scored_on_routing():
    # C0 scored on complexity only; N0 on non-complexity only; X1/A*/B0 on all
    assert arm_scored_on_problem_v1("C0", "COMPLEXITY_BLIND") is True
    assert arm_scored_on_problem_v1("C0", "WRONG_ALGORITHM_ADMISSIBLE") is False
    assert arm_scored_on_problem_v1("N0", "COMPLEXITY_BLIND") is False
    assert arm_scored_on_problem_v1("N0", "HIDDEN_EDGE_STATE_MISS") is True
    assert arm_scored_on_problem_v1("X1", "COMPLEXITY_BLIND") is True
    assert arm_scored_on_problem_v1("A1", "WRONG_ALGORITHM_ADMISSIBLE") is True


def test_fake_different_bites():
    fd = fake_different_report_v1()
    assert fd.bites
    assert "X2" not in fd.real_arms        # X2 is the relabeled-reflexion negative control (= M3/B0)
    assert "M3" in fd.fake_arms and "B0" in fd.fake_arms
