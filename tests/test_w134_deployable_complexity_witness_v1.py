"""W134 tests — deployable complexity witness + complexity-only corpus.

Mostly PURE unit tests (no subprocess) for speed; two minimal end-to-end subprocess checks for
the fire/no-fire verdict and the same-budget arm.  The full 20-mint corpus validation + the
all-train naive/ref separation gate live in the $0 build script
``scripts/run_w134_build_corpus_and_witness_selftest_v1.py`` (too slow for a unit test).
"""
from __future__ import annotations

import dataclasses
import inspect

import coordpy
from coordpy.deployable_complexity_witness_v1 import (
    ACTION_KEEP, ACTION_REWRITE, ARM_D1_REWRITE, ARM_D3_CONTROLLER, BudgetFactV1,
    DeployableWitnessV1, GrowthMeasurementV1, P_SUPERLINEAR, WALL_BUDGET_S,
    build_deployable_witness_v1, build_ladder_v1, derive_budget_fact_v1,
    deployable_witness_is_genuinely_new_v1, _fit_loglog, measure_growth_v1,
    parse_input_shape_v1, parse_value_hi_v1, run_deployable_witness_arm_v1,
    select_deployable_action_v1, synth_input_v1,
)
from coordpy.complexity_only_corpus_v1 import (
    DEV_SEEDS, EVAL_SEEDS, FRONTIER_SEEDS, MIN_FRONTIER, MIN_PER_SPLIT, TRAIN_SEEDS,
    complexity_slate_v1,
)
from coordpy.resistant_by_construction_battlefield_v1 import MODE_COMPLEXITY_BLIND
from coordpy.icpc_reflexion_bench_v1 import IcpcPilotProblemV1
from coordpy.coordpy_icpc_battlefield_v1 import KIND_PASSFAIL

_STMT = ("Count something.\n\nInput: first line N; second line N integers.\nOutput: a count.\n"
         "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9.")
_SAMPLES = (("4\n3 1 4 2\n", "2\n"),)
_ON2 = ("import sys\nd=sys.stdin.buffer.read().split()\nn=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "c=0\nfor i in range(n):\n for j in range(i+1,n):\n  c+=(a[i]>a[j])\nprint(c)\n")
_ON = ("import sys\nd=sys.stdin.buffer.read().split()\nn=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
       "print(sum(1 for x in a if x>0))\n")


# ---- stable boundary -----------------------------------------------------------------

def test_version_and_sdk_boundary_unchanged():
    assert coordpy.__version__ == "1.2.0"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


# ---- DW1 — constraint-derived budget -------------------------------------------------

def test_budget_parses_n_max_and_flags_quadratic_over_budget():
    b = derive_budget_fact_v1(_STMT)
    assert b.n_max == 100000
    assert b.quadratic_over_budget is True               # 1e10 ops >> 5e8 budget
    assert 1.6 < b.admissible_exponent_ceiling < 1.9     # log(5e8)/log(1e5) ~= 1.74

def test_budget_abstains_when_no_constraint():
    b = derive_budget_fact_v1("Solve the problem. Input: a list. Output: a number.")
    assert b.n_max is None and b.quadratic_over_budget is None

def test_parse_value_hi():
    assert parse_value_hi_v1("1 <= a_i <= 10^9.") == 10 ** 9
    assert parse_value_hi_v1("no bound here") == 10 ** 9   # default


# ---- DW2 — ladder + fit --------------------------------------------------------------

def test_fit_loglog_recovers_quadratic_and_linear():
    quad = [(float(n), float(n) ** 2) for n in (1000, 2000, 4000, 8000)]
    expo, r2 = _fit_loglog(quad)
    assert abs(expo - 2.0) < 1e-6 and r2 > 0.999
    lin = [(float(n), float(n)) for n in (1000, 2000, 4000, 8000)]
    expo2, _ = _fit_loglog(lin)
    assert abs(expo2 - 1.0) < 1e-6

def test_parse_input_shape_and_synth_sizes():
    shape = parse_input_shape_v1(_STMT, _SAMPLES)
    assert shape.parseable and shape.sample_n == 4
    for kind in ("random", "descending", "constant"):
        import random as _r
        inp = synth_input_v1(shape, size=2000, kind=kind, rng=_r.Random(1))
        lines = inp.split("\n")
        assert lines[0].split()[0] == "2000" and len(lines[1].split()) == 2000

def test_ladder_deterministic_and_spec_consistent():
    l1 = build_ladder_v1(_STMT, _SAMPLES)
    l2 = build_ladder_v1(_STMT, _SAMPLES)
    assert l1.parseable and l1.cid() == l2.cid()
    for sz, shps in l1.rungs:
        for _k, inp in shps:
            lines = inp.split("\n")
            assert lines[0].split()[0] == str(sz) and len(lines[1].split()) == sz

def test_ladder_unparseable_on_freeform_samples():
    lad = build_ladder_v1("Free text.", (("hello world\n", "ok\n"),))
    assert not lad.parseable


# ---- fake-different / genuinely-new + no-oracle structural ---------------------------

def _fired_witness():
    g = GrowthMeasurementV1(measurable=True, baseline_s=0.01, sizes=(1000, 2000),
                            compute_times_s=(0.1, 0.4), any_tle=True, tle_size=4000,
                            fitted_exponent=2.0, fit_r2=1.0, n_points=2)
    b = derive_budget_fact_v1(_STMT)
    return DeployableWitnessV1(kind="COMPLEXITY", fired=True, confidence_ok=True,
                              reason="LADDER_TLE", budget=b, growth=g,
                              extrapolated_s_at_n_max=900.0)

def test_genuinely_new_requires_curve_and_verdict():
    gn = deployable_witness_is_genuinely_new_v1(_fired_witness())
    assert gn["genuinely_new"] and gn["has_measured_curve_ge2_sizes"] and gn["has_growth_verdict"]
    assert gn["carries_no_oracle_output"] and gn["uses_no_reference_timing"]

def test_none_witness_is_not_genuinely_new():
    g = GrowthMeasurementV1(True, 0.01, (1000,), (0.001,), False, None, 0.1, 0.2, 1)
    w = DeployableWitnessV1("NONE", False, False, "ADMISSIBLE_GROWTH", derive_budget_fact_v1(_STMT),
                            g, None)
    assert not deployable_witness_is_genuinely_new_v1(w)["genuinely_new"]

def test_deployable_witness_has_no_oracle_output_field():
    # structural no-leakage: the witness record cannot carry an expected/oracle output
    fields = {f.name for f in dataclasses.fields(DeployableWitnessV1)}
    assert "expected_output" not in fields and "ref_runtime_s" not in fields

def test_witness_builder_and_arm_take_no_template_or_oracle():
    # the deployable witness consumes ONLY (code, statement, samples) — no template/ref/naive/secret
    params = set(inspect.signature(build_deployable_witness_v1).parameters)
    assert "statement" in params and "samples" in params
    assert not ({"template", "problem", "ref_source", "naive_source", "secret_cases"} & params)
    arm_params = set(inspect.signature(run_deployable_witness_arm_v1).parameters)
    assert "pilot" in arm_params and "template" not in arm_params and "problem" not in arm_params

def test_prompt_block_carries_no_expected_output():
    block = _fired_witness().to_prompt_block()
    assert "reference solution was used" in block  # explicitly states no reference
    assert "Correct output" not in block and "expected" not in block.lower()


# ---- end-to-end fire / no-fire verdict (subprocess) ----------------------------------

def test_witness_fires_on_quadratic_silent_on_linear():
    w_slow = build_deployable_witness_v1(_ON2, statement=_STMT, samples=_SAMPLES)
    assert w_slow.found()                              # O(N^2): TLE or super-linear+significant
    w_fast = build_deployable_witness_v1(_ON, statement=_STMT, samples=_SAMPLES)
    assert not w_fast.found()                          # O(N): admissible / below significance

def test_d3_controller_routes_rewrite_on_slow_defers_on_fast():
    from coordpy.deployable_complexity_witness_v1 import ACTION_ABSTAIN
    d_slow = select_deployable_action_v1(_ON2, statement=_STMT, samples=_SAMPLES,
                                         arm=ARM_D3_CONTROLLER)
    assert d_slow.action == ACTION_REWRITE
    d_fast = select_deployable_action_v1(_ON, statement=_STMT, samples=_SAMPLES,
                                         arm=ARM_D3_CONTROLLER)
    # a fast program is never rewritten — KEEP (measurably admissible) or ABSTAIN (unmeasurably
    # fast); both defer to blind reflexion, so the deployable arm is never worse than B0.
    assert d_fast.action in (ACTION_KEEP, ACTION_ABSTAIN)


# ---- same-budget arm (stub gen; no NIM) ----------------------------------------------

def test_deployable_arm_makes_exactly_K_calls():
    pilot = IcpcPilotProblemV1(
        problem_id="t_sum", short_name="sum", source_repo="test", contest_date="2026-06-03",
        statement="Sum.\nInput: N then N ints.\nConstraints: 1 <= N <= 100000.",
        kind=KIND_PASSFAIL, float_tol=0.0,
        samples=(("3\n1 2 3\n", "6\n"),), secret_cases=(("3\n1 2 3\n", "6\n"),))
    calls = {"n": 0}
    good = "import sys\nd=sys.stdin.buffer.read().split()\nprint(sum(int(x) for x in d[1:]))\n"

    def gen(prompt, max_tokens, temperature):
        calls["n"] += 1
        return (f"```python\n{good}\n```", 10)

    outcome, trace = run_deployable_witness_arm_v1(
        seed=1, pilot=pilot, gen=gen, K=2, temperature=0.7, max_tokens=64, timeout_s=8.0,
        arm=ARM_D1_REWRITE)
    assert outcome.n_model_calls == 2 and calls["n"] == 2
    assert outcome.final_passed and trace.all_oracle_free


# ---- complexity-only corpus structure (pure; no mint) --------------------------------

def test_complexity_slate_is_nine_cb_all_complexity_mode():
    slate = complexity_slate_v1()
    assert len(slate) == 9
    assert all(t.mode == MODE_COMPLEXITY_BLIND for t in slate)
    assert all(t.name.startswith("cb_") for t in slate)

def test_split_seeds_pairwise_disjoint():
    allseeds = list(TRAIN_SEEDS) + list(DEV_SEEDS) + list(EVAL_SEEDS) + list(FRONTIER_SEEDS)
    assert len(allseeds) == len(set(allseeds))           # all 20 distinct -> seed-disjoint splits
    assert MIN_PER_SPLIT == 36 and MIN_FRONTIER == 30
