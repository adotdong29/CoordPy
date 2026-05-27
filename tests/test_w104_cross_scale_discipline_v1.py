"""W104 — cross-scale discipline regression tests.

Codifies the four W104 guardrails so neither failure mode can
quietly recur on the W104 / W105+ code line:

1. **Sidecar resume-from-disk** — W102 buffered all 330 sidecar
   entries until pilot exit; W103 added per-write flush; W104
   EXTENDS this with the resume capability so a socket-hang or
   429 storm that requires a kill-and-restart does NOT lose
   evidence.

2. **Cross-scale comparator schema/provenance refuse-to-run** —
   the cross-scale comparator REFUSES to emit a diff when the
   two reports do not share a slice CID, a corpus SHA-256, or
   when either report has a missing MLB block / unrecognised
   schema.  This catches a 70B-vs-405B mix-up at write time.

3. **Cross-scale comparator cluster-shift correctness** — a
   synthetic 4-problem pair exercises all four shift values
   (stayed / improved / regressed / flipped).

4. **W104 target-selection-rule determinism** — the selection
   rule's anti-pattern token guard + cross-scale-legitimacy
   axes are unit-tested so a future target change cannot
   silently bypass them.

5. **W104 slice CID equality with W103** — the W104 driver
   reuses the W103 slice byte-for-byte; this test asserts the
   CID equality and the bench-iteration order equality.

Plus light validation that the W104 pilot driver loads the
W103 slice from the W103 provenance JSON unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
PILOT_SCRIPT = (
    ROOT / "scripts"
    / "run_w104_humaneval_plus_cross_scale_pilot.py")


def _load_pilot_module():
    """Load the W104 pilot driver as a module so we can exercise
    its pure-Python helpers without forking."""
    spec = importlib.util.spec_from_file_location(
        "w104_humaneval_plus_cross_scale_pilot", PILOT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------
# Cross-scale comparator schema/provenance refuse-to-run guards
# ---------------------------------------------------------------

def _minimal_bench_report(
        *, model_id: str = "x", b_minus_a1_pp: float = 5.0,
        mlb1: float = 0.4, mlb2: float = 0.4,
        per_problem_a0=(False,), per_problem_a1=(False,),
        per_problem_b=(True,), per_problem_b_fpi=(1,),
        bench_merkle_root: str = "deadbeef",
        a0_mean: float | None = None,
        a1_mean: float | None = None,
        b_mean: float | None = None,
):
    n = len(per_problem_a0)
    if a0_mean is None:
        a0_mean = sum(per_problem_a0) / n if n else 0.0
    if a1_mean is None:
        a1_mean = sum(per_problem_a1) / n if n else 0.0
    if b_mean is None:
        b_mean = sum(per_problem_b) / n if n else 0.0
    return {
        "schema": "coordpy.humaneval_plus_reflexion_bench_v1.v1",
        "model_id": model_id,
        "n_problems": int(n),
        "n_seeds": 1,
        "K_multi_sample": 5,
        "per_seed": [{
            "schema": "coordpy.humaneval_plus_reflexion_bench_v1.v1",
            "seed": 1,
            "n_problems": int(n),
            "a0_pass_at_1": float(a0_mean),
            "a1_pass_at_1": float(a1_mean),
            "b_pass_at_1": float(b_mean),
            "a0_total_wall_ms": 0,
            "a1_total_wall_ms": 0,
            "b_total_wall_ms": 0,
            "per_problem_a0_passed": list(per_problem_a0),
            "per_problem_a1_passed": list(per_problem_a1),
            "per_problem_b_passed": list(per_problem_b),
            "per_problem_b_first_pass_idx": list(
                per_problem_b_fpi),
            "outcome_cids": [],
            "seed_merkle_root": "x",
        }],
        "a0_mean_pass_at_1": float(a0_mean),
        "a1_mean_pass_at_1": float(a1_mean),
        "b_mean_pass_at_1": float(b_mean),
        "b_beats_a0_per_seed": [True],
        "b_beats_a1_per_seed": [True],
        "b_mean_strictly_beats_a0_mean": True,
        "b_mean_strictly_beats_a1_mean": True,
        "b_mean_minus_a1_mean_pp": float(b_minus_a1_pp),
        "bench_merkle_root": bench_merkle_root,
        "mlb": {
            "n_problems_total": int(n),
            "n_b_invoked_reflexion": 0,
            "n_b_rescued_via_reflexion": 0,
            "mlb1_invocation_rate": float(mlb1),
            "mlb2_rescue_rate": float(mlb2),
            "mlb1_floor": 0.33,
            "mlb2_floor": 0.33,
            "mlb1_passes": bool(mlb1 >= 0.33),
            "mlb2_passes": bool(mlb2 >= 0.33),
        },
    }


def _minimal_provenance(
        *, slice_cid="abc", corpus_sha="cba",
        task_ids=("HumanEval/0",)):
    return {
        "schema": "coordpy.w103_humaneval_plus_pilot.v1",
        "corpus_sha256": corpus_sha,
        "slice_cid_bench_order": slice_cid,
        "slice_cid_helper_priority": slice_cid,
        "bench_iteration_task_ids": list(task_ids),
    }


def test_w104_comparator_refuses_on_slice_cid_mismatch():
    from coordpy.cross_scale_comparator_v1 import (
        CrossScaleComparatorError,
        build_cross_scale_comparator_report_v1,
    )
    a = _minimal_bench_report(model_id="A")
    b = _minimal_bench_report(model_id="B")
    pa = _minimal_provenance(slice_cid="abc")
    pb = _minimal_provenance(slice_cid="xyz")
    with pytest.raises(CrossScaleComparatorError, match="slice"):
        build_cross_scale_comparator_report_v1(
            scale_a_bench_report=a, scale_a_provenance=pa,
            scale_b_bench_report=b, scale_b_provenance=pb)


def test_w104_comparator_refuses_on_corpus_sha_mismatch():
    from coordpy.cross_scale_comparator_v1 import (
        CrossScaleComparatorError,
        build_cross_scale_comparator_report_v1,
    )
    a = _minimal_bench_report()
    b = _minimal_bench_report()
    pa = _minimal_provenance(corpus_sha="111")
    pb = _minimal_provenance(corpus_sha="222")
    with pytest.raises(CrossScaleComparatorError, match="corpus"):
        build_cross_scale_comparator_report_v1(
            scale_a_bench_report=a, scale_a_provenance=pa,
            scale_b_bench_report=b, scale_b_provenance=pb)


def test_w104_comparator_refuses_on_missing_mlb_block():
    from coordpy.cross_scale_comparator_v1 import (
        CrossScaleComparatorError,
        build_cross_scale_comparator_report_v1,
    )
    a = _minimal_bench_report()
    b = _minimal_bench_report()
    del b["mlb"]
    pa = _minimal_provenance()
    pb = _minimal_provenance()
    with pytest.raises(CrossScaleComparatorError, match="mlb"):
        build_cross_scale_comparator_report_v1(
            scale_a_bench_report=a, scale_a_provenance=pa,
            scale_b_bench_report=b, scale_b_provenance=pb)


def test_w104_comparator_refuses_on_unrecognised_schema():
    from coordpy.cross_scale_comparator_v1 import (
        CrossScaleComparatorError,
        build_cross_scale_comparator_report_v1,
    )
    a = _minimal_bench_report()
    b = _minimal_bench_report()
    b["schema"] = "coordpy.completely_unknown.v0"
    pa = _minimal_provenance()
    pb = _minimal_provenance()
    with pytest.raises(CrossScaleComparatorError, match="schema"):
        build_cross_scale_comparator_report_v1(
            scale_a_bench_report=a, scale_a_provenance=pa,
            scale_b_bench_report=b, scale_b_provenance=pb)


# ---------------------------------------------------------------
# Cross-scale comparator cluster-shift correctness
# ---------------------------------------------------------------

def test_w104_comparator_cluster_shifts_synthetic_pair():
    from coordpy.cross_scale_comparator_v1 import (
        build_cross_scale_comparator_report_v1,
    )
    # 4-problem synthetic pair exercises all four shift values:
    # p0: (A0=F,A1=F,B=T) -> (A0=F,A1=F,B=T)   stayed
    # p1: (A0=F,A1=F,B=F) -> (A0=F,A1=T,B=T)   improved
    # p2: (A0=T,A1=T,B=T) -> (A0=T,A1=T,B=F)   regressed
    # p3: (A0=T,A1=F,B=F) -> (A0=F,A1=T,B=F)   flipped (A0 lost, A1 gained)
    a = _minimal_bench_report(
        per_problem_a0=[False, False, True, True],
        per_problem_a1=[False, False, True, False],
        per_problem_b=[True, False, True, False],
        per_problem_b_fpi=[1, -1, 0, -1])
    b = _minimal_bench_report(
        per_problem_a0=[False, False, True, False],
        per_problem_a1=[False, True, True, True],
        per_problem_b=[True, True, False, False],
        per_problem_b_fpi=[1, 0, -1, -1])
    pa = _minimal_provenance(
        task_ids=["HumanEval/0", "HumanEval/1",
                  "HumanEval/2", "HumanEval/3"])
    pb = _minimal_provenance(
        task_ids=["HumanEval/0", "HumanEval/1",
                  "HumanEval/2", "HumanEval/3"])
    rep = build_cross_scale_comparator_report_v1(
        scale_a_bench_report=a, scale_a_provenance=pa,
        scale_b_bench_report=b, scale_b_provenance=pb)
    shifts = [r.cluster_shift for r in rep.per_problem]
    assert shifts == ["stayed", "improved", "regressed", "flipped"]
    counts = rep.aggregate_cluster_shift_counts
    assert counts["stayed"] == 1
    assert counts["improved"] == 1
    assert counts["regressed"] == 1
    assert counts["flipped"] == 1


def test_w104_comparator_aggregate_b_minus_a1_shift():
    from coordpy.cross_scale_comparator_v1 import (
        build_cross_scale_comparator_report_v1,
    )
    # Scale A: A1=50%, B=70% -> +20pp; Scale B: A1=80%, B=60% -> -20pp.
    # Cross-scale shift on B-A1 = -40 pp.
    a = _minimal_bench_report(
        per_problem_a0=[False] * 10,
        per_problem_a1=[True, True, True, True, True,
                        False, False, False, False, False],
        per_problem_b=[True, True, True, True, True,
                       True, True, False, False, False],
        per_problem_b_fpi=[0, 0, 0, 0, 0, 1, 1, -1, -1, -1],
        a0_mean=0.0, a1_mean=0.5, b_mean=0.7,
        b_minus_a1_pp=20.0,
        bench_merkle_root="aaa")
    b = _minimal_bench_report(
        per_problem_a0=[False] * 10,
        per_problem_a1=[True, True, True, True, True,
                        True, True, True, False, False],
        per_problem_b=[True, True, True, True, True,
                       True, False, False, False, False],
        per_problem_b_fpi=[0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
        a0_mean=0.0, a1_mean=0.8, b_mean=0.6,
        b_minus_a1_pp=-20.0,
        bench_merkle_root="bbb")
    pa = _minimal_provenance(
        task_ids=[f"HumanEval/{i}" for i in range(10)])
    pb = _minimal_provenance(
        task_ids=[f"HumanEval/{i}" for i in range(10)])
    rep = build_cross_scale_comparator_report_v1(
        scale_a_bench_report=a, scale_a_provenance=pa,
        scale_b_bench_report=b, scale_b_provenance=pb)
    assert abs(
        rep.aggregate_b_minus_a1_pp_at_scale_a - 20.0) < 0.01
    assert abs(
        rep.aggregate_b_minus_a1_pp_at_scale_b - (-20.0)) < 0.01
    assert abs(
        rep.cross_scale_shift_on_b_minus_a1_pp - (-40.0)) < 0.01


# ---------------------------------------------------------------
# W104 target-selection-rule determinism
# ---------------------------------------------------------------

def test_w104_target_selection_anti_pattern_guard():
    """No W104-eligible target can carry a forbidden token in
    its name.  Hardcoded so any future expansion of the W104
    target list cannot silently regress this guard."""
    mod = _load_pilot_module()
    targets = [
        mod.W104_PRIMARY_TARGET_MODEL,
        mod.W104_BACKUP_TARGET_MODEL,
    ]
    forbidden = (
        "bounded_window", "compaction", "context_compaction",
        "prose_summary", "context_pruning", "summarizer")
    for t in targets:
        low = t.lower()
        for f in forbidden:
            assert f not in low, (
                f"W104 target {t!r} contains anti-pattern token "
                f"{f!r}")


def test_w104_target_selection_distinct_from_w103():
    """The cross-scale target must be structurally different from
    the W103 70B target."""
    mod = _load_pilot_module()
    assert (
        mod.W104_PRIMARY_TARGET_MODEL
        != "meta/llama-3.3-70b-instruct")
    assert (
        mod.W104_BACKUP_TARGET_MODEL
        != "meta/llama-3.3-70b-instruct")


def test_w104_target_selection_primary_is_405b():
    """W104 lead-lane primary is the 405B class per the
    pre-locked RUNBOOK."""
    mod = _load_pilot_module()
    assert (
        mod.W104_PRIMARY_TARGET_MODEL
        == "meta/llama-3.1-405b-instruct")


# ---------------------------------------------------------------
# W104 slice-CID equality with W103 (byte-equal reuse)
# ---------------------------------------------------------------

def test_w104_slice_cid_equals_w103_locked_constant():
    """The W104 driver pins the W103 slice CID as a constant; if
    a future change inadvertently rebuilds the slice instead of
    reusing it byte-for-byte, this test fires."""
    mod = _load_pilot_module()
    assert (
        mod.W103_HELPER_ANCHORED_SLICE_CID_HELPER_PRIORITY
        == "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2")
    assert (
        mod.W103_HELPER_ANCHORED_SLICE_CID_BENCH_ORDER
        == "d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052")


def test_w104_slice_recompute_matches_w103_provenance_on_disk():
    """If the W103 provenance JSON is on disk, the slice CIDs
    recomputed from it match the W104 locked constants."""
    mod = _load_pilot_module()
    w103_prov = (
        ROOT / "results" / "w103" / "humaneval_plus_pilot"
        / "w103_humaneval_plus_pilot_meta_llama-3.3-70b-instruct_"
          "20260526T022037Z" / "provenance.json")
    if not w103_prov.exists():
        pytest.skip("W103 provenance JSON not on disk")
    with open(w103_prov) as f:
        prov = json.load(f)
    assert (
        prov["slice_cid_helper_priority"]
        == mod.W103_HELPER_ANCHORED_SLICE_CID_HELPER_PRIORITY)
    assert (
        prov["slice_cid_bench_order"]
        == mod.W103_HELPER_ANCHORED_SLICE_CID_BENCH_ORDER)


# ---------------------------------------------------------------
# Sidecar resume-from-disk regression guard
# ---------------------------------------------------------------

def test_w104_sidecar_resume_helper_extracts_completed_keys():
    """If the pilot is killed mid-run, restarting it must skip
    already-completed (seed, p_idx, arm, attempt_idx) tuples.
    The helper parses the on-disk sidecar to extract those
    tuples; this test exercises that helper on a synthetic
    sidecar."""
    mod = _load_pilot_module()
    with tempfile.TemporaryDirectory() as td:
        side = Path(td) / "side.jsonl"
        with open(side, "w") as f:
            f.write(json.dumps({
                "seed": 1, "p_idx": 0, "arm": "A0",
                "attempt_idx": 0,
                "prompt_sha256": "p", "response_sha256": "r",
            }) + "\n")
            f.write(json.dumps({
                "seed": 1, "p_idx": 0, "arm": "B",
                "attempt_idx": 0,
                "prompt_sha256": "p", "response_sha256": "r",
            }) + "\n")
            # malformed trailing line — must NOT count as complete
            f.write("{ not json")
        completed = mod._extract_completed_sidecar_keys(side)
        assert (1, 0, "A0", 0) in completed
        assert (1, 0, "B", 0) in completed
        # the malformed trailing line is treated as not-yet-completed
        assert len(completed) == 2


def test_w104_sidecar_resume_handles_missing_file():
    """If the sidecar file does not exist (fresh run), the
    completed-key set is empty (no spurious crash)."""
    mod = _load_pilot_module()
    with tempfile.TemporaryDirectory() as td:
        side = Path(td) / "missing.jsonl"
        completed = mod._extract_completed_sidecar_keys(side)
        assert completed == set()


# ---------------------------------------------------------------
# Cross-scale comparator markdown smoke test
# ---------------------------------------------------------------

def test_w104_comparator_markdown_smoke():
    from coordpy.cross_scale_comparator_v1 import (
        build_cross_scale_comparator_report_v1,
        format_cross_scale_comparator_markdown_v1,
    )
    a = _minimal_bench_report(model_id="scale_a_model")
    b = _minimal_bench_report(model_id="scale_b_model")
    pa = _minimal_provenance()
    pb = _minimal_provenance()
    rep = build_cross_scale_comparator_report_v1(
        scale_a_bench_report=a, scale_a_provenance=pa,
        scale_b_bench_report=b, scale_b_provenance=pb)
    md = format_cross_scale_comparator_markdown_v1(report=rep)
    assert "scale_a_model" in md
    assert "scale_b_model" in md
    assert "Cross-scale shift on B" in md
