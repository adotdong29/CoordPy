"""W137 unit tests — parser-neutral I/O kernel + hard slate v2 + corpus + calibration + bench.

Fast by design: the kernel + logic tests use no subprocesses; only the WRONG_ANSWER templates are
minted (the TIMEOUT templates are exercised end-to-end by
``scripts/run_w137_build_v2_battlefield_and_selftest_v1.py``).
"""
from __future__ import annotations

import pytest

from coordpy.parser_neutral_io_v1 import (
    io_shape, scalar_line, array_line, rows, grid, render_normal_form_v1,
    parse_per_line_v1, parse_all_tokens_v1, parser_neutrality_gate_v1, NormalFormError)
from coordpy.hard_battlefield_slate_v2 import (
    build_hard_slate_v2, slate_fingerprint_cid_v1, io_shape_registry_v2)
from coordpy.hard_battlefield_corpus_v2 import (
    template_diversity_v1, split_seeds_v1, SPLIT_NAMES, mint_split_v2, summarize_split_v2)
from coordpy.model_ladder_calibration_v1 import (
    ModelTemplateStatV1, calibrate_template_v1, LADDER_V1, CALIBRATION_SEED_BASE)
from coordpy.repaired_field_mechanism_bench_v1 import (
    fake_different_report_v1, m3_relabeled_reflexion_fingerprint_v1,
    witness_arm_fingerprint_v1, evaluate_gate_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1

LOCKED_SLATE_CID = "2ce207c567324e4322f308e58a1fc2c88a8d4bdd0e340d2ec8a1b867d82b3f70"


# ---------------------------------------------------------------- parser-neutral I/O kernel

def test_normal_form_roundtrip_rows():
    shp = io_shape(scalar_line("N", "W"), rows("ITEMS", "N", "w", "v"))
    data = {"N": 3, "W": 50, "ITEMS": [(10, 60), (20, 100), (30, 120)]}
    nf = render_normal_form_v1(shp, data)
    assert nf == "3 50\n10 60\n20 100\n30 120\n"
    assert parse_per_line_v1(nf, shp) == data
    assert parse_all_tokens_v1(nf, shp) == data


def test_hc1_passes_normal_form_and_fails_flattened():
    # the W136 confound regression: a flattened body must FAIL HC1; normal form must PASS
    shp = io_shape(scalar_line("N", "W"), rows("ITEMS", "N", "w", "v"))
    nf = render_normal_form_v1(shp, {"N": 3, "W": 50, "ITEMS": [(10, 60), (20, 100), (30, 120)]})
    flat = "3 50\n10 60 20 100 30 120\n"
    assert parser_neutrality_gate_v1([nf], shp).is_parser_neutral is True
    rec = parser_neutrality_gate_v1([flat], shp)
    assert rec.is_parser_neutral is False
    assert rec.first_failure  # a non-empty diagnostic


def test_hc1_grid_and_array_shapes():
    g = io_shape(scalar_line("R", "C"), grid("G", "R", "C"))
    gnf = render_normal_form_v1(g, {"R": 2, "C": 3, "G": ["..#", "#.."]})
    assert parser_neutrality_gate_v1([gnf], g).is_parser_neutral is True
    assert parser_neutrality_gate_v1(["2 3\n..# #..\n"], g).is_parser_neutral is False
    a = io_shape(scalar_line("N"), array_line("A", "N"))
    anf = render_normal_form_v1(a, {"N": 4, "A": [3, 1, 4, 1]})
    assert parser_neutrality_gate_v1([anf], a).is_parser_neutral is True


def test_per_line_reader_raises_on_wrong_token_count():
    shp = io_shape(scalar_line("N"), rows("R", "N", "a", "b"))
    with pytest.raises(NormalFormError):
        parse_per_line_v1("2\n1 2 3\n4 5\n", shp)  # first row has 3 tokens, stride is 2


# ---------------------------------------------------------------- slate v2

def test_slate_fingerprint_matches_lock():
    assert slate_fingerprint_cid_v1() == LOCKED_SLATE_CID


def test_slate_has_17_templates_4_modes():
    slate = build_hard_slate_v2()
    assert len(slate) == 17
    modes = {t.minted.mode for t in slate}
    assert len(modes) == 4
    # every template has an io_shape registered
    reg = io_shape_registry_v2()
    assert set(reg) == {t.minted.name for t in slate}


def test_wrong_answer_templates_admit_and_are_parser_neutral():
    # mint only the fast (OUTPUT_MISMATCH) templates; assert HC2 admitted + HC1 parser-neutral
    slate = build_hard_slate_v2()
    fast = [t for t in slate if t.minted.discriminator == "OUTPUT_MISMATCH"]
    assert len(fast) >= 8
    for t in fast:
        p = mint_problem_v1(t.minted, global_seed=137_001, timeout_s=5.0)
        assert p.gates.admitted, f"{t.minted.name}: {p.gates.reason}"
        hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], t.io_shape)
        assert hc1.is_parser_neutral, f"{t.minted.name} not parser-neutral"


# ---------------------------------------------------------------- corpus (HC5 + splits)

def test_hc5_template_diversity_all_distinct():
    div = template_diversity_v1()
    assert div.all_distinct is True
    assert div.max_pairwise_jaccard < 0.55
    assert div.n_modes == 4


def test_split_seeds_disjoint_across_splits():
    seeds = {sp: set(split_seeds_v1(sp, 10)) for sp in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a != b:
                assert seeds[a].isdisjoint(seeds[b])
    # and disjoint from calibration seeds
    cal = set(range(CALIBRATION_SEED_BASE, CALIBRATION_SEED_BASE + 100))
    for sp in SPLIT_NAMES:
        assert set(split_seeds_v1(sp, 50)).isdisjoint(cal)


def test_mint_split_admits_wrong_answer_template():
    slate = build_hard_slate_v2()
    t = next(t for t in slate if t.minted.name == "wa_knapsack_01_v2")
    probs = mint_split_v2("dev", n_replicas=2, timeout_s=5.0, templates=[t])
    assert len(probs) == 2
    assert all(p.admitted for p in probs)
    summ = summarize_split_v2(probs)
    assert summ.n_admitted == 2


# ---------------------------------------------------------------- calibration admission logic

def test_hc3_culls_saturated_template():
    # a fake gen that always returns a perfect solver -> strong A0 saturates -> HC3 culls
    slate = build_hard_slate_v2()
    t = next(t for t in slate if t.minted.name == "wa_knapsack_01_v2")
    ref = t.minted.ref_source

    def gen_for_model(model_id):
        def gen(prompt, max_tokens, temperature):
            return ("```python\n" + ref + "\n```", 1)
        return gen

    rec = calibrate_template_v1(t, gen_for_model=gen_for_model, ladder=LADDER_V1,
                               n_a0=2, n_a1=1, K_a1=2, hc3_ceiling=0.80, mint_timeout_s=5.0)
    anchor = rec.anchor()
    assert anchor.a0_rate == 1.0
    assert rec.hc3_has_headroom is False
    assert rec.admitted is False
    assert "HC3_SATURATED" in rec.reason


def test_hc4_culls_dead_template():
    # a fake gen that always returns a non-solution -> strong best_rate == 0 -> HC4 dead
    slate = build_hard_slate_v2()
    t = next(t for t in slate if t.minted.name == "wa_knapsack_01_v2")

    def gen_for_model(model_id):
        def gen(prompt, max_tokens, temperature):
            return ("```python\nprint(-999999)\n```", 1)
        return gen

    rec = calibrate_template_v1(t, gen_for_model=gen_for_model, ladder=LADDER_V1,
                               n_a0=2, n_a1=1, K_a1=2, hc3_ceiling=0.80, mint_timeout_s=5.0)
    assert rec.anchor().best_rate == 0.0
    assert rec.hc4_not_dead is False
    assert rec.admitted is False
    assert "HC4_DEAD" in rec.reason


def test_admission_band_logic_on_synthetic_stats():
    # mid-band: strong A0 0.33 (< ceiling) and > 0 -> admitted
    strong = ModelTemplateStatV1("strong", "strong", (True, False, False), (True,), 4)
    assert strong.a0_rate == pytest.approx(1 / 3)
    assert 0 < strong.a0_rate < 0.80 and strong.best_rate > 0


# ---------------------------------------------------------------- mechanism bench (fake-different + gates)

def test_fake_different_bites():
    fd = fake_different_report_v1()
    assert fd.bites is True
    assert "M3" in fd.fake_arms and "B0" in fd.fake_arms
    assert m3_relabeled_reflexion_fingerprint_v1().classify() == "FAKE_DIFFERENT"
    assert witness_arm_fingerprint_v1("M1").classify() == "REAL"


def test_gate_fails_on_zero_margin():
    n = 6
    same = [True, True, False, False, True, False]
    v = evaluate_gate_v1(name="dev_gate", per_lead=same, per_a1=same, per_b0=same,
                         modes=["COMPLEXITY_BLIND"] * n, families=[f"f{i}" for i in range(n)],
                         rescue_is_structural=[True] * n, margin_pp=3.33)
    assert v.passed is False
    assert "MARGIN_FAIL" in v.reason


def test_gate_excludes_non_structural_rescue():
    per_lead = [True, True, True, True, True, True]
    per_b0 = [False, False, False, True, True, True]
    per_a1 = [False, False, False, True, True, True]
    modes = ["COMPLEXITY_BLIND", "WRONG_ALGORITHM_ADMISSIBLE", "SEARCH_ENUM", "x", "x", "x"]
    fams = ["a", "b", "c", "d", "e", "f"]
    # one rescue flagged non-structural (parsing/formatting) -> all_structural False -> gate fails
    v = evaluate_gate_v1(name="dev_gate", per_lead=per_lead, per_a1=per_a1, per_b0=per_b0,
                         modes=modes, families=fams,
                         rescue_is_structural=[False, True, True, True, True, True], margin_pp=3.33)
    assert v.all_structural is False
    assert v.passed is False
