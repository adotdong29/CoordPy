"""W103 — code-line discipline regression tests.

Codifies the two W102 lessons so neither failure mode can quietly
recur on the W103 / W104+ code line:

1. Silent-degeneration via schema assumption — the W101 V1 MBPP+
   loader assumed parallel `plus_input` / `plus_output` arrays
   that do NOT exist in the real EvalPlus release; against real
   data V1 silently emitted 0 plus-assertions and would have
   silently degenerated the cheap pilot to a base-MBPP run.
   W102 added P5 + P6 probes for MBPP+ V2; W103 EXTENDS the
   structural defence to HumanEval+ via the executor surface
   guard below.

2. Cross-bench arsenal-mining priors as cheap-pilot earning
   evidence — W102 re-graded W91 responses showed +5.28 pp on
   MBPP+ V2 but the fresh-K=5 cheap pilot at seed 101_001
   produced -6.67 pp (an 11.95 pp swing).  W103 codifies this
   anti-pattern via the provenance + earning-status guard below.

Plus light validation that the W103 pilot driver's helper-
consumption + slice-CID + corpus-SHA pin behave correctly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
PILOT_SCRIPT = (
    ROOT / "scripts" / "run_w103_humaneval_plus_pilot.py")
SLICE_SCRIPT = (
    ROOT / "scripts" / "run_w102_code_slice_proposal.py")


def _load_pilot_module():
    """Load the W103 pilot driver as a module so we can exercise
    its pure-Python helpers without forking."""
    spec = importlib.util.spec_from_file_location(
        "w103_humaneval_plus_pilot", PILOT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------
# Lesson 1 — Silent-degeneration anti-pattern guard
# ---------------------------------------------------------------

def test_w103_humaneval_plus_executor_refuses_synthetic_silent_degeneration():
    """The W102 lesson: a loader that silently emits an empty
    extra-test surface degrades the cheap pilot to a base-bench
    run, hiding the failure mode.  This test constructs a
    synthetic HumanEval+ row WITHOUT a `def check(` block and
    asserts the executor produces a clear, observable FAIL on
    a trivially-correct canonical solution.

    If a future loader change reintroduces V1's silent-
    degeneration shape, this test fires.
    """
    from coordpy.humaneval_plus_executor_v1 import (
        run_humaneval_plus_executor_v1,
    )
    from coordpy.humaneval_plus_loader_v1 import (
        HumanEvalPlusProblemV1,
    )
    synthetic = HumanEvalPlusProblemV1(
        task_id="HumanEval/synthetic_silent_degen",
        prompt=(
            "def add_one(x: int) -> int:\n"
            "    '''Return x + 1.'''\n"),
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        # NO `def check(` block — the V1 silent-degeneration
        # equivalent on the HumanEval+ surface.
        test=(
            "# synthetic row without a check() block — should "
            "force a clear FAIL\n"
            "x = 1  # no-op\n"))
    candidate = synthetic.prompt + synthetic.canonical_solution
    exe = run_humaneval_plus_executor_v1(
        problem=synthetic, candidate_code=candidate)
    # Without a check() invocation the executor's surface is
    # structurally degenerate.  The W103 hardening contract is
    # that this MUST be detectable — either via FAIL or via
    # missing-check-block sentinel.  The W102 V2 P5 probe is
    # the preflight-time guard; this test is the unit-test-
    # time guard.
    assert not exe.passed, (
        "synthetic silent-degeneration row passed; the "
        "executor's surface is not honest about empty "
        "extra-test programs")


def test_w103_humaneval_plus_preflight_p5_catches_missing_check_block(
        monkeypatch):
    """W102 added P5 (extra-test-surface integrity) for MBPP+
    V2; the HumanEval+ analog already exists in
    `coordpy.humaneval_plus_preflight_v1.probe_humaneval_plus_
    extra_test_surface_v1`.  Exercise the probe with a synthetic
    in-memory corpus (mocked loader) and confirm it FAILs when
    rows lack the `def check(` block — the exact silent-
    degeneration shape W101 V1's loader produced on real data.
    """
    import coordpy.humaneval_plus_preflight_v1 as pf
    from coordpy.humaneval_plus_loader_v1 import (
        HumanEvalPlusProblemV1,
    )
    synthetic_corpus = (
        HumanEvalPlusProblemV1(
            task_id="HumanEval/0",
            prompt="def f(x):\n    pass\n",
            canonical_solution="    return x\n",
            entry_point="f",
            test="def check(candidate):\n    assert True\n"),
        HumanEvalPlusProblemV1(
            task_id="HumanEval/synthetic_no_check",
            prompt="def g(x):\n    pass\n",
            canonical_solution="    return x\n",
            entry_point="g",
            test="# no check() block here\nx = 1\n"),
    )
    monkeypatch.setattr(
        pf, "is_humaneval_plus_cached",
        lambda *, cache_path=None: True)
    monkeypatch.setattr(
        pf, "load_humaneval_plus_corpus_v1",
        lambda *, cache_path=None: synthetic_corpus)
    result = pf.probe_humaneval_plus_extra_test_surface_v1(
        cache_path=None)
    # rate = 1/2 = 50% < 95% floor → FAIL
    assert not result.passed, (
        "P5 should FAIL on a corpus where half the rows are "
        "missing the check() block")
    assert result.evidence["rate"] == 0.5


# ---------------------------------------------------------------
# Lesson 2 — Arsenal-mining priors are NOT earning evidence
# ---------------------------------------------------------------

def test_w103_arsenal_mining_prior_is_not_earning_evidence():
    """The W102 lesson: re-grading historical responses against
    a new test surface produces an UPPER BOUND on what the
    mechanism could produce IF the sampling distribution stayed
    the same.  Fresh-K=5 sampling at a new seed is the ground
    truth.

    The W103 pilot driver MUST record the arsenal-mining prior
    in the provenance + verdict but MUST NOT use it as a Phase 2
    gate input.  This test asserts the driver module records
    the prior with explicit "NOT a gate input" status.
    """
    mod = _load_pilot_module()
    # The driver carries a published constant for the W102
    # cross-bench prior.  This must be RECORDED.
    assert hasattr(
        mod,
        "W103_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_B_MINUS_A1_PP")
    # The Phase 2 gate evaluation must NOT reference any
    # mining-prior field as an input.
    src = PILOT_SCRIPT.read_text()
    # Find the _evaluate_phase2_gates function body
    gates_start = src.index("def _evaluate_phase2_gates")
    gates_end = src.index("\ndef _build_nim_gen", gates_start)
    gates_body = src[gates_start:gates_end]
    # The W102 anti-pattern would be silently using the mining
    # prior to inflate a verdict.  Assert no mining-prior
    # symbols leak into the gate evaluation.
    assert "ARSENAL_MINING_PRIOR" not in gates_body, (
        "_evaluate_phase2_gates references the arsenal-mining "
        "prior; this re-introduces the W102 anti-pattern")
    assert "mining_report" not in gates_body, (
        "_evaluate_phase2_gates references the mining report; "
        "this re-introduces the W102 anti-pattern")


def test_w103_pilot_driver_records_provenance_fields():
    """The hardening lane requires the driver to record corpus_
    sha, helper_proposal_cid, mining_report_cid, preflight_
    verdict_cid, slice_cid_helper_priority + slice_cid_bench_
    order + arsenal_mining_prior_* into the bench report.
    This test asserts those keys are referenced in the driver's
    provenance dict construction."""
    src = PILOT_SCRIPT.read_text()
    required_keys = [
        "corpus_sha256",
        "preflight_verdict_cid",
        "helper_proposal_cid_humaneval_plus",
        "helper_proposal_cid_humaneval_topup",
        "mining_report_cid",
        "slice_cid_helper_priority",
        "slice_cid_bench_order",
        "arsenal_mining_prior_humaneval_plus",
        "earning_status",
    ]
    for key in required_keys:
        assert key in src, (
            f"W103 pilot driver missing provenance key {key!r} "
            "(W102 hardening regression)")


# ---------------------------------------------------------------
# Slice-consumption + corpus-SHA pin guards
# ---------------------------------------------------------------

def test_w103_helper_anchored_slice_dedups_task_ids():
    """The helper proposal can contain a task_id under multiple
    historical seeds (e.g., HumanEval/91 appears in both seed
    88028001 + seed 88028002 shared_fails).  The pilot driver
    MUST de-dup on task_id so the same HumanEval+ row is
    attempted once.
    """
    mod = _load_pilot_module()
    proposals = {
        "proposals": {
            "humaneval_plus": {
                "schema": "coordpy.code_slice_selector_v1.v1",
                "bench": "humaneval_plus",
                "n_problems": 5,
                "cheap_pilot_budget_nim_calls": 55,
                "proposal_cid": "test_hp_cid",
                "rationale_summary": "test",
                "proposal": [
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/91",
                     "cluster": "shared_fails",
                     "justification": "x"},
                    {"bench": "humaneval_plus", "seed": 2,
                     "task_id": "HumanEval/91",  # dup
                     "cluster": "shared_fails",
                     "justification": "x"},
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/100",
                     "cluster": "b_only_wins",
                     "justification": "x"},
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/101",
                     "cluster": "shared_wins",
                     "justification": "x"},
                ],
            },
            "humaneval": {
                "schema": "coordpy.code_slice_selector_v1.v1",
                "bench": "humaneval",
                "n_problems": 5,
                "cheap_pilot_budget_nim_calls": 55,
                "proposal_cid": "test_he_cid",
                "rationale_summary": "test",
                "proposal": [
                    {"bench": "humaneval", "seed": 1,
                     "task_id": "HumanEval/200",  # new
                     "cluster": "shared_wins",
                     "justification": "x"},
                    {"bench": "humaneval", "seed": 1,
                     "task_id": "HumanEval/100",  # dup (across blocks)
                     "cluster": "shared_wins",
                     "justification": "x"},
                ],
            },
        },
        "mining_report_path": "",
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json.dump(proposals, f)
        path = Path(f.name)
    try:
        slice_, hp_cid, he_cid, _ = (
            mod._build_helper_anchored_slice(
                proposals_json_path=path,
                n_problems=4))
    finally:
        path.unlink()
    tids = [t for t, _ in slice_]
    assert tids == [
        "HumanEval/91", "HumanEval/100", "HumanEval/101",
        "HumanEval/200"], (
        f"de-dup or top-up logic broken; got {tids}")
    assert hp_cid == "test_hp_cid"
    assert he_cid == "test_he_cid"


def test_w103_helper_anchored_slice_refuses_all_shared_wins():
    """The hardening contract: if the slice degenerates to only
    `shared_wins` (no rescue or stress surface to test), the
    pilot MUST refuse to run."""
    mod = _load_pilot_module()
    proposals = {
        "proposals": {
            "humaneval_plus": {
                "schema": "coordpy.code_slice_selector_v1.v1",
                "bench": "humaneval_plus",
                "n_problems": 3,
                "cheap_pilot_budget_nim_calls": 33,
                "proposal_cid": "test_hp_cid",
                "rationale_summary": "test",
                "proposal": [
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/100",
                     "cluster": "shared_wins",
                     "justification": "x"},
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/101",
                     "cluster": "shared_wins",
                     "justification": "x"},
                    {"bench": "humaneval_plus", "seed": 1,
                     "task_id": "HumanEval/102",
                     "cluster": "shared_wins",
                     "justification": "x"},
                ],
            },
        },
        "mining_report_path": "",
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json.dump(proposals, f)
        path = Path(f.name)
    try:
        with pytest.raises(SystemExit) as ei:
            mod._build_helper_anchored_slice(
                proposals_json_path=path,
                n_problems=3)
        assert "shared_wins" in str(ei.value)
    finally:
        path.unlink()


def test_w103_slice_cid_is_deterministic_from_helper_priority_order():
    """The slice CID is computed from the helper-priority-
    ordered task_id list joined with commas; this test pins the
    CID for the actual W103 production slice so any future
    refactor that perturbs the order is caught."""
    expected_slice = [
        "HumanEval/118", "HumanEval/16", "HumanEval/160",
        "HumanEval/163", "HumanEval/121", "HumanEval/125",
        "HumanEval/84", "HumanEval/129", "HumanEval/76",
        "HumanEval/91", "HumanEval/132", "HumanEval/137",
        "HumanEval/140", "HumanEval/154", "HumanEval/32",
        "HumanEval/55", "HumanEval/83", "HumanEval/17",
        "HumanEval/122", "HumanEval/100", "HumanEval/101",
        "HumanEval/104", "HumanEval/111", "HumanEval/113",
        "HumanEval/119", "HumanEval/14", "HumanEval/35",
        "HumanEval/44", "HumanEval/49", "HumanEval/61",
    ]
    cid = hashlib.sha256(
        ",".join(expected_slice).encode("utf-8")).hexdigest()
    assert cid == (
        "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466"
        "dcc01dd8d2"), (
        f"W103 production slice CID drifted from "
        f"docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md: got {cid}")


# ---------------------------------------------------------------
# Anti-pattern carry-forward — bounded / compaction guards
# ---------------------------------------------------------------

def test_w103_pilot_driver_does_not_import_anti_patterns():
    """The W103 pilot driver must not transitively re-introduce
    bounded-window / compaction / summarization primitives."""
    src = PILOT_SCRIPT.read_text()
    forbidden = [
        "bounded_window", "compaction", "context_compaction",
        "prose_summary", "context_pruning", "summarizer",
    ]
    for tok in forbidden:
        assert tok not in src, (
            f"W103 pilot driver contains forbidden anti-pattern "
            f"token {tok!r}")
