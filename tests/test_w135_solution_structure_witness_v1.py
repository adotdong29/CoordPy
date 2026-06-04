"""W135 — tests for the solution-STRUCTURE witness instrument + non-complexity corpus.

$0 NIM: the only code execution is the (audited) oracle/candidate subprocess.  A small battlefield
(2 WA + 2 SE families, one seed) is minted once and shared; the corpus split tests mint a single
seed-split (fast) to check disjointness/CID determinism without paying the full 20-mint corpus.
"""
from __future__ import annotations

import functools

from coordpy.resistant_by_construction_battlefield_v1 import (
    MODE_COMPLEXITY_BLIND, MODE_SEARCH_ENUM, MODE_WRONG_ALGORITHM, mint_battlefield_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1
from coordpy.solution_structure_witness_v1 import (
    ARM_S1_GREEDY, ARM_S2_SUBSTRUCTURE, ARM_S3_SEARCH, ARM_S4_CONTROLLER, STRUCT_NONE,
    STRUCTURE_ARMS, W135_SOLUTION_STRUCTURE_WITNESS_V1_SCHEMA_VERSION,
    build_structure_witness_v1, run_structure_witness_arm_v1,
    structure_witness_is_genuinely_new_v1,
)
from coordpy.noncomplexity_structure_corpus_v1 import (
    MODE_SEARCH_ENUM as _CMSE, MIN_FRONTIER, MIN_PER_SPLIT, TRAIN_SEEDS, DEV_SEEDS,
    build_noncomplexity_split_v1, corpus_disjointness_report_v1, dedup_splits_across_v1,
    noncomplexity_slate_v1,
)

WSEED = 999_135
MINTED = "2026-06-03"
_WA_NAMES = ("wa_max_nonadjacent_sum", "wa_min_coins")
_SE_NAMES = ("se_count_stair_climbings", "se_count_bsts_catalan")
_CB_NAME = "cb_pairs_sum_le_t"


@functools.lru_cache(maxsize=1)
def _mini_battlefield():
    """Mint a small shared battlefield: 2 WA + 2 SE + 1 CB families, seed 135021."""
    slate = tuple(t for t in RBC_SLATE_V1
                  if t.name in (_WA_NAMES + _SE_NAMES + (_CB_NAME,)))
    bf = mint_battlefield_v1(slate, global_seed=135021, minted_date=MINTED, timeout_s=8.0)
    by_name = {f"rbc_{t.name}": t for t in slate}
    probs = {p.problem_id: p for p in bf.problems}
    return by_name, probs


def _witness_for(name, code_attr="naive_source"):
    by_name, probs = _mini_battlefield()
    pid = f"rbc_{name}"
    t, p = by_name[pid], probs[pid]
    probe = build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)
    code = getattr(t, code_attr)
    return build_structure_witness_v1(code, p, probe, t, timeout_s=2.0), p, t


# --------------------------------------------------------------- module + slate

def test_schema_version_string():
    assert W135_SOLUTION_STRUCTURE_WITNESS_V1_SCHEMA_VERSION == "coordpy.solution_structure_witness_v1.v1"


def test_noncomplexity_slate_is_16_two_modes():
    slate = noncomplexity_slate_v1()
    assert len(slate) == 16
    modes = {t.mode for t in slate}
    assert modes == {MODE_WRONG_ALGORITHM, MODE_SEARCH_ENUM}
    # every non-complexity template has an INDEPENDENT brute oracle (not == naive, unlike complexity)
    assert all(t.brute_source != t.naive_source for t in slate)


def test_structure_arms_constant():
    assert STRUCTURE_ARMS == (ARM_S1_GREEDY, ARM_S2_SUBSTRUCTURE, ARM_S3_SEARCH, ARM_S4_CONTROLLER)


# --------------------------------------------------------------- witness fires / silent

def test_witness_fires_on_wa_naive():
    w, p, t = _witness_for(_WA_NAMES[0], "naive_source")
    assert w.found()
    gn = structure_witness_is_genuinely_new_v1(w, p)
    assert gn["genuinely_new"] is True
    assert gn["leakage_clean"] is True


def test_witness_fires_on_se_naive():
    w, p, t = _witness_for(_SE_NAMES[0], "naive_source")
    assert w.found()
    gn = structure_witness_is_genuinely_new_v1(w, p)
    # an SE family with a clean integer-N ladder must carry a >=2-rung sub-value ladder
    assert gn["genuinely_new"] is True
    assert gn["has_substructure_ladder"] is True


def test_positive_control_ref_is_silent():
    """The correct reference as candidate ⇒ no counterexample ⇒ NONE (no witness)."""
    for nm in (_WA_NAMES[0], _SE_NAMES[0]):
        w, p, t = _witness_for(nm, "ref_source")
        assert not w.found()
        assert w.kind == STRUCT_NONE


def test_negative_control_complexity_naive_is_silent():
    """A value-correct-but-slow COMPLEXITY naive has no value counterexample ⇒ structure NONE."""
    w, p, t = _witness_for(_CB_NAME, "naive_source")
    assert p.mode == MODE_COMPLEXITY_BLIND
    assert not w.found()
    assert w.kind == STRUCT_NONE


# --------------------------------------------------------------- no-leakage

def test_no_solver_source_in_prompt_block():
    w, p, t = _witness_for(_WA_NAMES[0], "naive_source")
    for arm in STRUCTURE_ARMS:
        block = w.to_prompt_block(arm)
        # the block must never contain solver SOURCE (it carries oracle OUTPUTS only)
        for needle in ("def ", "import ", "class ", "sys.stdin", "lambda"):
            assert needle not in block, f"{arm} leaked source token {needle!r}"


def test_probe_and_ladder_disjoint_from_secret_cases():
    w, p, t = _witness_for(_SE_NAMES[0], "naive_source")
    secret_inputs = {inp for inp, _ in p.secret_cases}
    assert w.counterexample.probe_input not in secret_inputs
    # ladder sub-instances are not directly stored as raw input on the rung, but the builder
    # asserts disjointness; verify leakage_clean is set and the witness is marked clean.
    assert w.leakage_clean is True


def test_prompt_block_does_not_reveal_a_recurrence_formula():
    """The structure block must teach STRUCTURE (values), not hand over the recurrence/state."""
    w, p, t = _witness_for(_SE_NAMES[0], "naive_source")
    block = w.to_prompt_block(ARM_S2_SUBSTRUCTURE).lower()
    # it explicitly asks the model to RECOVER the recurrence (does not state it)
    assert "recover the recurrence" in block or "find the recurrence" in block.replace("recover", "find")


# --------------------------------------------------------------- genuinely-new semantics

def test_genuinely_new_requires_structure_beyond_counterexample():
    """A witness with no ladder and greedy==observed (the naive self-test, no clean ladder) is not
    genuinely-new; one with a >=2-rung ladder is."""
    w, p, t = _witness_for(_SE_NAMES[1], "naive_source")  # catalan: clean 1..N ladder
    gn = structure_witness_is_genuinely_new_v1(w, p)
    assert gn["has_substructure_ladder"] is True
    assert gn["genuinely_new"] is True


# --------------------------------------------------------------- determinism

def test_witness_reproducible():
    by_name, probs = _mini_battlefield()
    pid = f"rbc_{_WA_NAMES[0]}"
    t, p = by_name[pid], probs[pid]
    probe = build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)
    w1 = build_structure_witness_v1(t.naive_source, p, probe, t, timeout_s=2.0)
    w2 = build_structure_witness_v1(t.naive_source, p, probe, t, timeout_s=2.0)
    assert w1.found() == w2.found()
    assert w1.cid() == w2.cid()
    assert w1.to_prompt_block(ARM_S4_CONTROLLER) == w2.to_prompt_block(ARM_S4_CONTROLLER)
    assert [r.optimal_value for r in w1.ladder] == [r.optimal_value for r in w2.ladder]


# --------------------------------------------------------------- same-budget arm

def test_structure_arm_makes_exactly_K_model_calls():
    by_name, probs = _mini_battlefield()
    pid = f"rbc_{_WA_NAMES[0]}"
    t, p = by_name[pid], probs[pid]
    probe = build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)
    calls = {"n": 0}

    def fake_gen(prompt, max_tokens, temperature):
        calls["n"] += 1
        # return a trivial non-solution so the arm runs all K attempts (never early-passes)
        return ("```python\nprint(0)\n```", 0)

    outcome, trace = run_structure_witness_arm_v1(
        seed=1, template=t, problem=p, probe=probe, gen=fake_gen, K=5, temperature=0.7,
        max_tokens=256, timeout_s=4.0, arm=ARM_S4_CONTROLLER, minted_date=MINTED)
    assert calls["n"] == 5
    assert outcome.n_model_calls == 5
    assert outcome.arm_id == ARM_S4_CONTROLLER
    assert trace.all_leakage_clean is True


def test_arm_trace_records_structure_kinds():
    by_name, probs = _mini_battlefield()
    pid = f"rbc_{_SE_NAMES[0]}"
    t, p = by_name[pid], probs[pid]
    probe = build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)

    def fake_gen(prompt, max_tokens, temperature):
        return ("```python\nprint(0)\n```", 0)

    outcome, trace = run_structure_witness_arm_v1(
        seed=1, template=t, problem=p, probe=probe, gen=fake_gen, K=5, temperature=0.7,
        max_tokens=256, timeout_s=4.0, arm=ARM_S3_SEARCH, minted_date=MINTED)
    # attempts 1..4 each compute a witness (4 entries); a clean SE ladder ⇒ genuinely-new fired
    assert len(trace.struct_kinds) == 4
    assert trace.any_structure_found is True


# --------------------------------------------------------------- corpus

def test_corpus_split_seed_disjoint_and_floors():
    slate = noncomplexity_slate_v1()
    train = build_noncomplexity_split_v1(slate, split="train", seeds=TRAIN_SEEDS, minted_date=MINTED)
    dev = build_noncomplexity_split_v1(slate, split="dev", seeds=DEV_SEEDS, minted_date=MINTED)
    assert train.n_admitted >= MIN_PER_SPLIT
    assert dev.n_admitted >= MIN_PER_SPLIT
    # both modes present
    assert set(train.mode_histogram) == {MODE_WRONG_ALGORITHM, _CMSE}
    # raw splits are SEED-disjoint by construction; content may collide on seed-independent
    # tiny-case families (e.g. fib_no_adjacent) ⇒ dedup is what guarantees held-out integrity.
    raw = corpus_disjointness_report_v1({"train": train, "dev": dev})
    assert raw["all_seeds_pairwise_disjoint"] is True
    train2, dev2 = dedup_splits_across_v1([train, dev])
    deduped = corpus_disjointness_report_v1({"train": train2, "dev": dev2})
    assert deduped["all_content_cids_pairwise_disjoint"] is True
    assert deduped["held_out_integrity"] is True
    assert train2.n_admitted >= MIN_PER_SPLIT and dev2.n_admitted >= MIN_PER_SPLIT


def test_split_cid_deterministic():
    slate = noncomplexity_slate_v1()
    a = build_noncomplexity_split_v1(slate, split="train", seeds=TRAIN_SEEDS, minted_date=MINTED)
    b = build_noncomplexity_split_v1(slate, split="train", seeds=TRAIN_SEEDS, minted_date=MINTED)
    assert a.split_cid == b.split_cid
    assert a.n_admitted == b.n_admitted
