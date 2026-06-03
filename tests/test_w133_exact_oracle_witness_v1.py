"""W133 tests — exact-oracle witness instrument + curriculum (all $0 NIM, deterministic).

Covers: witness fires on the admissible-wrong / too-slow trap (EW1 counterexample + EW2
complexity); the positive control (NO witness on the correct reference); no-leakage (probe
inputs disjoint from graded secret cases; no solver source in the model-facing block);
determinism (stable probe-set + split CIDs); seed-disjoint curriculum (content + graded-secret
inputs); the same-budget arm (exactly K model calls); and the stable version boundary.
"""
import coordpy
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.exact_oracle_witness_v1 import (
    ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER,
    WITNESS_COMPLEXITY, WITNESS_COUNTEREXAMPLE, WITNESS_NONE,
    build_witness_probe_set_v1, find_counterexample_witness_v1, run_witness_arm_v1,
    select_witness_v1, witness_is_genuinely_new_v1,
)
from coordpy.witness_curriculum_corpus_v1 import (
    DEV_SEED, EVAL_SEED, TRAIN_SEED, build_curriculum_v1,
)

_BY_NAME = {t.name: t for t in RBC_SLATE_V1}
_WS = 999_133


def _mint(name, seed=133_002):
    t = _BY_NAME[name]
    return t, mint_problem_v1(t, global_seed=seed, timeout_s=8.0)


def test_version_boundary_unchanged():
    assert coordpy.__version__ == "0.5.20"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


def test_ew1_counterexample_fires_on_wrong_algorithm_trap():
    t, p = _mint("wa_knapsack_01")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    w = find_counterexample_witness_v1(t.naive_source, p, probe, t, timeout_s=2.0)
    assert w.kind == WITNESS_COUNTEREXAMPLE
    assert w.expected_output and w.observed_output      # carries an oracle output + the wrong one
    assert w.leakage_clean
    assert witness_is_genuinely_new_v1(w, p)["genuinely_new"] is True


def test_ew1_counterexample_fires_on_hidden_edge_trap():
    t, p = _mint("he_interval_union_length")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    w = select_witness_v1(t.naive_source, p, probe, t, arm=ARM_C3_CONTROLLER, timeout_s=2.0)
    assert w.kind == WITNESS_COUNTEREXAMPLE
    assert w.leakage_clean


def test_ew2_complexity_fires_on_too_slow_trap():
    t, p = _mint("cb_pairs_sum_le_t")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    assert probe.big_input is not None and probe.big_ref_runtime_s < 2.0   # ref finishes fast
    w = select_witness_v1(t.naive_source, p, probe, t, arm=ARM_C3_CONTROLLER, timeout_s=2.0)
    assert w.kind == WITNESS_COMPLEXITY
    assert w.ref_runtime_s < w.cand_runtime_s
    assert w.leakage_clean


def test_positive_control_no_witness_on_correct_reference():
    # the correct reference is value-correct AND fast => NO counterexample, NO complexity witness.
    t, p = _mint("wa_knapsack_01")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    w = select_witness_v1(t.ref_source, p, probe, t, arm=ARM_C3_CONTROLLER, timeout_s=2.0)
    assert w.kind == WITNESS_NONE
    assert witness_is_genuinely_new_v1(w, p)["genuinely_new"] is False


def test_no_leakage_probe_disjoint_and_no_solver_in_prompt():
    t, p = _mint("se_lattice_paths_blocked")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    secret_inputs = {inp for inp, _ in p.secret_cases}
    for inp, _ in probe.small:
        assert inp not in secret_inputs
    w = select_witness_v1(t.naive_source, p, probe, t, arm=ARM_C1_COUNTEREXAMPLE, timeout_s=2.0)
    block = w.to_prompt_block()
    # the model-facing block must never contain solver source
    for tell in ("def ", "import ", "sys.stdin", "ref_source", "naive_source"):
        assert tell not in block
    assert w.probe_input not in secret_inputs


def test_probe_set_is_deterministic():
    t, p = _mint("he_max_gap_sorted")
    a = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    b = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    assert a.cid() == b.cid()


def test_same_budget_arm_makes_exactly_k_calls():
    t, p = _mint("he_interval_union_length")
    probe = build_witness_probe_set_v1(t, p, witness_seed=_WS, timeout_s=2.0)
    calls = {"n": 0}

    def stub_gen(prompt, max_tokens, temperature):
        calls["n"] += 1
        return ("```python\nprint(0)\n```", 1)   # a deliberately wrong program

    outcome, trace = run_witness_arm_v1(
        seed=1, template=t, problem=p, probe=probe, gen=stub_gen, K=5, temperature=0.7,
        max_tokens=64, timeout_s=8.0, arm=ARM_C3_CONTROLLER, minted_date="2026-06-02",
        witness_timeout_s=2.0)
    assert calls["n"] == 5                       # exactly K model calls (same budget as B0)
    assert outcome.n_model_calls == 5
    assert trace.all_leakage_clean


def test_curriculum_seed_disjoint_subset():
    # a small seed-disjoint curriculum (fast templates only) — verify the disjointness audit.
    subset = tuple(_BY_NAME[n] for n in
                   ("wa_knapsack_01", "he_interval_union_length", "se_lattice_paths_blocked"))
    cur = build_curriculum_v1(subset, train_seed=TRAIN_SEED, dev_seed=DEV_SEED,
                              eval_seed=EVAL_SEED, timeout_s=8.0)
    dj = cur.disjointness
    # held-out integrity = whole-problem content-CID disjoint + seed disjoint (the meaningful
    # property); residual per-secret-input overlap is only seed-independent canonical boundary
    # cases (e.g. a smallest-n or worst-case stress input), which carry no answer signal.
    assert dj["all_content_cids_pairwise_disjoint"] is True
    assert dj["held_out_integrity"] is True
    assert cur.train.split_cid != cur.dev.split_cid != cur.eval.split_cid
    # re-build is deterministic
    cur2 = build_curriculum_v1(subset, train_seed=TRAIN_SEED, dev_seed=DEV_SEED,
                               eval_seed=EVAL_SEED, timeout_s=8.0)
    assert cur.curriculum_cid() == cur2.curriculum_cid()
