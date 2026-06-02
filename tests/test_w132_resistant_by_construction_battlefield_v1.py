"""W132 — tests for the resistant-by-construction battlefield framework + slate.

$0 / deterministic.  Exercises the per-problem quality gates, the novelty guard (positive
+ negative controls), no-leakage, deterministic regeneration, and Maverick resistance —
WITHOUT the full-slate mint (the 9 COMPLEXITY templates each pay an ~8s TLE; the build
SCRIPT verifies the >=30 admission and the verdict JSON is asserted here if present).
"""
from __future__ import annotations

import dataclasses
import json
import os

import coordpy
from coordpy.resistant_by_construction_battlefield_v1 import (
    DISC_OUTPUT_MISMATCH,
    MODE_COMPLEXITY_BLIND,
    TARGET_MODES,
    certify_resistance_v1,
    core_slice_cid_v1,
    mint_battlefield_v1,
    mint_problem_v1,
    novelty_filter_v1,
    select_core_slice_v1,
    statement_jaccard_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1
from coordpy.icpc_reflexion_bench_v1 import IcpcPilotProblemV1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 132
FAST = [t for t in RBC_SLATE_V1 if t.mode != MODE_COMPLEXITY_BLIND]


# ---------------------------------------------------------------- stable boundary
def test_version_boundary_unchanged():
    assert coordpy.__version__ == "0.5.20"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


# ---------------------------------------------------------------- slate structure
def test_slate_structure():
    assert len(RBC_SLATE_V1) == 33
    modes = {t.mode for t in RBC_SLATE_V1}
    assert modes == set(TARGET_MODES)
    names = [t.name for t in RBC_SLATE_V1]
    assert len(names) == len(set(names))            # unique problem identities
    fams = [t.family for t in RBC_SLATE_V1]
    assert len(fams) == len(set(fams))              # one family per problem
    # every mode is materially represented
    for m in TARGET_MODES:
        assert sum(1 for t in RBC_SLATE_V1 if t.mode == m) >= 7


# ---------------------------------------------------------------- per-problem gates
def test_fast_templates_all_admitted():
    bad = []
    for t in FAST:
        p = mint_problem_v1(t, global_seed=SEED, timeout_s=6.0)
        if not p.gates.admitted:
            bad.append((t.name, p.gates.reason))
    assert not bad, f"non-admitted fast templates: {bad}"


def test_one_complexity_template_timeout_discriminates():
    t = next(t for t in RBC_SLATE_V1 if t.mode == MODE_COMPLEXITY_BLIND)
    p = mint_problem_v1(t, global_seed=SEED, timeout_s=8.0)
    assert p.gates.admitted, p.gates.reason
    assert "TIMEOUT" in p.gates.naive_fail_kinds       # the O(N^2) naive really TLEs
    assert p.gates.naive_passes_all_public


def test_gate_components_present():
    p = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    g = p.gates
    assert g.g_passfail_only and g.g_reference_solvable
    assert g.g_oracle_small_agreement and g.n_brute_checked >= 1
    assert g.g_discriminating and g.g_split_integrity


# ------------------------------------------------------------- novelty guard controls
def test_novelty_positive_control_rejects_planted_duplicate():
    p = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    dup = dataclasses.replace(p, problem_id="rbc_planted_dup")   # identical statement
    accepted, rejected = novelty_filter_v1([p, dup])
    assert len(accepted) == 1 and len(rejected) == 1
    assert rejected[0].problem_id == "rbc_planted_dup"
    assert rejected[0].jaccard >= 0.55


def test_novelty_official_identity_guard():
    p = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    tainted = dataclasses.replace(
        p, problem_id="rbc_tainted",
        statement=p.statement + "\nThis is the spaceelevator problem.")
    accepted, rejected = novelty_filter_v1(
        [tainted], official_identities=["spaceelevator"])
    assert len(accepted) == 0 and len(rejected) == 1
    assert rejected[0].nearest_id.startswith("official:")


def test_distinct_minted_statements_are_novel():
    a = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    b = mint_problem_v1(FAST[3], global_seed=SEED, timeout_s=6.0)
    assert statement_jaccard_v1(a.statement, b.statement) < 0.55


# ------------------------------------------------------------- negative control (falsifiable)
def test_non_discriminating_template_is_rejected():
    """A template whose naive EQUALS the reference must NOT be admitted (the gate bites)."""
    base = FAST[0]
    fake = dataclasses.replace(base, name="fake_non_discriminating",
                               naive_source=base.ref_source)   # naive == ref => no trap
    p = mint_problem_v1(fake, global_seed=SEED, timeout_s=6.0)
    assert not p.gates.admitted
    assert "NOT_DISCRIMINATING" in p.gates.reason


# ------------------------------------------------------------- no-leakage
def test_pilot_problem_hides_solver_sources():
    p = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    pp = p.to_pilot_problem(minted_date="2026-06-02")
    assert isinstance(pp, IcpcPilotProblemV1)
    blob = json.dumps({"statement": pp.statement,
                       "samples": [list(s) for s in pp.samples]})
    assert "ref_source" not in dir(pp)
    # the model-facing payload contains no solver token tells
    for tell in ("def ", "import ", "lru_cache", "bisect"):
        assert tell not in blob
    # samples are a subset of the secret grader
    assert set(pp.samples).issubset(set(pp.secret_cases))


# ------------------------------------------------------------- determinism
def test_deterministic_regeneration_same_content_cid():
    a = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    b = mint_problem_v1(FAST[0], global_seed=SEED, timeout_s=6.0)
    assert a.content_cid() == b.content_cid()
    c = mint_problem_v1(FAST[0], global_seed=SEED + 1, timeout_s=6.0)
    assert c.content_cid() != a.content_cid()       # a different seed => fresh instance


# ------------------------------------------------------------- resistance certification
def test_maverick_resistance_by_date_and_construction():
    cert = certify_resistance_v1(model_id="meta/llama-4-maverick-17b-128e-instruct",
                                 minted_date="2026-06-02", n_core=33, raw_cid="deadbeef")
    assert cert.resistant
    assert cert.minted_after_cutoff and cert.novel_by_construction
    # a pre-cutoff mint date would NOT be date-resistant
    cert2 = certify_resistance_v1(model_id="meta/llama-4-maverick-17b-128e-instruct",
                                  minted_date="2024-01-01", n_core=33, raw_cid="deadbeef")
    assert not cert2.minted_after_cutoff


# ------------------------------------------------------------- small battlefield assembly
def test_small_battlefield_assembles_and_strata():
    small = list(FAST[:8])                          # fast (no COMPLEXITY TLE)
    bf = mint_battlefield_v1(small, global_seed=SEED, minted_date="2026-06-02",
                             timeout_s=6.0, min_slice=5)
    assert bf.manifest.n_admitted == 8
    assert bf.manifest.manifest_cid() == bf.manifest.manifest_cid()   # stable
    core = select_core_slice_v1(bf, n_problems=6)
    assert len(core) == 6
    assert core_slice_cid_v1(core)


# ------------------------------------------------------------- build verdict (if emitted)
def test_build_verdict_meets_min_slice_if_present():
    path = os.path.join(ROOT, "results", "w132", "battlefield",
                        "battlefield_verdict_v1.json")
    if not os.path.exists(path):
        return
    v = json.load(open(path))
    assert v["n_admitted"] >= 30
    assert v["meets_min_slice_30"] is True
    assert v["deterministic_regeneration"] is True
    assert v["resistance_cert"]["resistant"] is True
    assert v["battlefield_pilot_earned"] is True
    assert set(v["mode_histogram"]) == set(TARGET_MODES)
