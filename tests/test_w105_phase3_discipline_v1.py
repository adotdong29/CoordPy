"""W105 — Phase 3 discipline tests.

Covers (a) pack-CID-mismatch refuse-to-run, (b) corpus-SHA-
mismatch refuse-to-run, (c) per-seed shuffle reproducibility,
(d) resume-safe per-cell skipping, (e) evaluator PASS path on
synthetic 6 cells, (f) evaluator per-bar FAIL paths, (g)
cross-class entitlement ONLY on BOTH-PASS, (h) cross-class
comparator per-seed alignment, (i) cross-class comparator
refuse-to-run paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_class_comparator_v1 import (  # noqa: E402
    CrossClassComparatorError,
    build_cross_class_comparator_report_v1,
)
from coordpy.phase3_retirement_evaluator_v1 import (  # noqa: E402
    Phase3CellSummaryV1,
    Phase3RetirementEvaluatorError,
    W105_PHASE3_CROSS_CLASS_DIFF_PP_ABS_MAX,
    W105_PHASE3_MARGIN_FLOOR_PP,
    W105_PHASE3_PER_SEED_MAJORITY_MIN,
    build_phase3_retirement_verdict_v1,
    evaluate_cross_class_verdict_v1,
    evaluate_per_class_verdict_v1,
)
from coordpy.humaneval_plus_reflexion_bench_v1 import (  # noqa: E402
    select_humaneval_plus_subset_v1,
)


# W105 locked constants pulled in to test the driver-side
# verification path without importing the driver as a module
# (the driver uses `argparse` + main() and is not import-safe).
W105_PACK_CID_LOCKED: str = (
    "8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314")
W105_INNER_KERNEL_CID_LOCKED: str = (
    "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2")
W105_CORPUS_SHA_LOCKED: str = (
    "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492")
W105_PHASE3_SEEDS_LOCKED: tuple[int, ...] = (
    105_001, 105_002, 105_003)


def _slice_pack_with_overrides(*, overrides: dict) -> dict:
    """Build a synthetic W105-shape slice pack, then apply
    overrides on top."""
    tids = [f"HumanEval/{i}" for i in range(100)]
    pack = {
        "schema": "coordpy.w104_w105_phase3_slice_pack.v1",
        "n_problems": 100,
        "phase3_seeds": list(W105_PHASE3_SEEDS_LOCKED),
        "pack_cid": hashlib.sha256(
            ",".join(tids).encode()).hexdigest(),
        "inner_kernel_cid_w103_helper_priority":
            W105_INNER_KERNEL_CID_LOCKED,
        "task_ids_helper_priority": tids,
    }
    pack.update(overrides)
    return pack


def _make_cell_bench_report(
        *, model_id: str, seed: int, n_problems: int,
        a0: float, a1: float, b: float,
        mlb1: float, mlb2: float,
        per_problem_b_passed: list[bool] | None = None,
        per_problem_a1_passed: list[bool] | None = None,
        per_problem_a0_passed: list[bool] | None = None,
        per_problem_b_first_pass_idx: list[int] | None = None,
) -> dict:
    if per_problem_b_passed is None:
        # Distribute B passes deterministically.
        per_problem_b_passed = [True] * int(round(n_problems * b)) + (
            [False] * (n_problems - int(round(n_problems * b))))
    if per_problem_a1_passed is None:
        per_problem_a1_passed = (
            [True] * int(round(n_problems * a1)) + (
                [False] * (n_problems - int(round(n_problems * a1)))))
    if per_problem_a0_passed is None:
        per_problem_a0_passed = (
            [True] * int(round(n_problems * a0)) + (
                [False] * (n_problems - int(round(n_problems * a0)))))
    if per_problem_b_first_pass_idx is None:
        per_problem_b_first_pass_idx = [
            (0 if per_problem_b_passed[i]
             and per_problem_a1_passed[i] else
             (2 if per_problem_b_passed[i] else -1))
            for i in range(n_problems)]
    outcome_cids = [
        hashlib.sha256(
            f"cell_{model_id}_{seed}_{i}".encode()).hexdigest()
        for i in range(n_problems * 3)]
    payload = {
        "kind": "w102_humaneval_plus_bench_merkle_root_v1",
        "model_id": str(model_id),
        "outcome_cids": list(outcome_cids),
        "seeds": [int(seed)],
    }
    bench_merkle = hashlib.sha256(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()
    return {
        "schema": "coordpy.humaneval_plus_reflexion_bench_v1.v1",
        "model_id": str(model_id),
        "n_problems": int(n_problems),
        "n_seeds": 1,
        "K_multi_sample": 5,
        "per_seed": [{
            "schema":
                "coordpy.humaneval_plus_reflexion_bench_v1.v1",
            "seed": int(seed),
            "n_problems": int(n_problems),
            "a0_pass_at_1": float(a0),
            "a1_pass_at_1": float(a1),
            "b_pass_at_1": float(b),
            "a0_total_wall_ms": 1,
            "a1_total_wall_ms": 5,
            "b_total_wall_ms": 5,
            "per_problem_a0_passed": list(
                per_problem_a0_passed),
            "per_problem_a1_passed": list(
                per_problem_a1_passed),
            "per_problem_b_passed": list(per_problem_b_passed),
            "per_problem_b_first_pass_idx": list(
                per_problem_b_first_pass_idx),
            "outcome_cids": list(outcome_cids),
            "seed_merkle_root": "syn_seed_merkle",
        }],
        "a0_mean_pass_at_1": float(a0),
        "a1_mean_pass_at_1": float(a1),
        "b_mean_pass_at_1": float(b),
        "b_beats_a0_per_seed": [bool(b > a0)],
        "b_beats_a1_per_seed": [bool(b > a1)],
        "b_mean_strictly_beats_a0_mean": bool(b > a0),
        "b_mean_strictly_beats_a1_mean": bool(b > a1),
        "b_mean_minus_a1_mean_pp": float((b - a1) * 100.0),
        "bench_merkle_root": str(bench_merkle),
        "mlb": {
            "n_problems_total": int(n_problems),
            "n_b_invoked_reflexion": int(
                round(mlb1 * n_problems)),
            "n_b_rescued_via_reflexion": int(
                round(mlb1 * n_problems * mlb2)),
            "mlb1_invocation_rate": float(mlb1),
            "mlb2_rescue_rate": float(mlb2),
            "mlb1_floor": 0.33,
            "mlb2_floor": 0.33,
            "mlb1_passes": bool(mlb1 >= 0.33),
            "mlb2_passes": bool(mlb2 >= 0.33),
        },
    }


def _make_cell_provenance(
        *, model_id: str, seed: int, n_problems: int,
        slice_pack_cid: str = W105_PACK_CID_LOCKED,
        corpus_sha: str = W105_CORPUS_SHA_LOCKED,
        per_seed_iter: list[str] | None = None,
        is_canary: bool = False) -> dict:
    if per_seed_iter is None:
        tids = [f"HumanEval/{i}" for i in range(n_problems)]
        rng = random.Random(int(seed))
        idxs = list(range(len(tids)))
        rng.shuffle(idxs)
        per_seed_iter = [tids[i] for i in idxs]
    return {
        "schema": "coordpy.w105_phase3_cell.v1",
        "model_id": str(model_id),
        "seed": int(seed),
        "n_problems": int(n_problems),
        "K_multi_sample": 5,
        "corpus_sha256": str(corpus_sha),
        "slice_pack_cid": str(slice_pack_cid),
        "per_seed_iteration_task_ids": list(per_seed_iter),
        "is_canary": bool(is_canary),
        "cell_run_id": f"syn_cell_{seed}",
    }


class TestSlicePackCidIntegrity(unittest.TestCase):
    """W105 driver requires the slice pack's pack_cid to match
    the W105 locked constant and to re-derive cleanly from the
    comma-joined task_ids_helper_priority."""

    def test_pack_cid_re_derives_on_disk(self) -> None:
        pack_path = (
            ROOT / "data" / "w105" / "phase3_slice_pack"
            / "w105_phase3_slice_pack_20260526T215647Z"
            / "slice_pack.json")
        self.assertTrue(pack_path.exists())
        with open(pack_path) as f:
            pack = json.load(f)
        tids = list(pack["task_ids_helper_priority"])
        recomputed = hashlib.sha256(
            ",".join(tids).encode()).hexdigest()
        self.assertEqual(recomputed, pack["pack_cid"])
        self.assertEqual(pack["pack_cid"], W105_PACK_CID_LOCKED)
        self.assertEqual(
            pack["inner_kernel_cid_w103_helper_priority"],
            W105_INNER_KERNEL_CID_LOCKED)
        self.assertEqual(
            tuple(pack["phase3_seeds"]),
            W105_PHASE3_SEEDS_LOCKED)


class TestPerSeedShuffleReproducibility(unittest.TestCase):
    """The bench module's seed-driven shuffle MUST match the
    driver's seed-driven shuffle byte-for-byte so the cross-
    class comparator can align per-(matched seed)."""

    def test_shuffle_matches_bench_module(self) -> None:
        # Build a fake corpus by task_id.
        class FakeProblem:
            def __init__(self, tid: str) -> None:
                self.task_id = tid
        tids = [f"HumanEval/{i}" for i in range(100)]
        corpus = tuple(FakeProblem(t) for t in tids)
        for seed in (105_001, 105_002, 105_003, 105_999):
            via_bench = select_humaneval_plus_subset_v1(
                corpus=corpus, n_problems=100, seed=int(seed))
            via_bench_tids = [p.task_id for p in via_bench]
            # Replay shuffle the driver way.
            rng = random.Random(int(seed))
            idxs = list(range(len(tids)))
            rng.shuffle(idxs)
            via_driver = [tids[i] for i in idxs]
            self.assertEqual(
                via_bench_tids, via_driver,
                f"shuffle mismatch on seed {seed}")


class TestPerClassRetirementPassPath(unittest.TestCase):
    """All 6 bars PASS + MLB-2 load-bearing => RETIRED."""

    def test_pass_path_emits_retired(self) -> None:
        # 3 cells; per-class mean B-A1 = +10pp; per-seed
        # majority 3/3; per-problem majority 60/100 each cell.
        cells = []
        for s, b in zip((105_001, 105_002, 105_003),
                        (0.70, 0.65, 0.68)):
            cells.append(Phase3CellSummaryV1(
                model_class_id="meta/llama-3.3-70b-instruct",
                seed=int(s), n_problems=100,
                a0_pct=50.0, a1_pct=float(b * 100 - 10),
                b_pct=float(b * 100),
                b_minus_a1_pp=10.0,
                n_b_ge_a1=60,
                mlb1_invocation_rate=0.50,
                mlb2_rescue_rate=0.45,
                a1_lt_saturation_max=True,
                bench_merkle_root="m" * 64,
                slice_pack_cid=W105_PACK_CID_LOCKED,
                cell_run_id=f"syn_{s}"))
        v = evaluate_per_class_verdict_v1(
            model_class_id="meta/llama-3.3-70b-instruct",
            cells=tuple(cells),
            audit_chain_re_derives=3,
            audit_chain_total=3,
            canonical_pass_rate=1.0)
        self.assertEqual(v.verdict_label, "RETIRED")
        self.assertEqual(v.n_bars_passed_of_6, 6)
        self.assertTrue(v.mlb2_load_bearing)


class TestPerClassMarginFail(unittest.TestCase):
    def test_margin_fail_emits_fail_with_margin_reason(self) -> None:
        cells = [Phase3CellSummaryV1(
            model_class_id="m", seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0, b_pct=56.0,
            b_minus_a1_pp=1.0,
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.5, mlb2_rescue_rate=0.45,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        v = evaluate_per_class_verdict_v1(
            model_class_id="m", cells=tuple(cells),
            audit_chain_re_derives=3,
            audit_chain_total=3,
            canonical_pass_rate=1.0)
        self.assertTrue(v.verdict_label.startswith("FAIL"))
        self.assertIn("MARGIN", v.verdict_label)


class TestPerClassA1SaturationFail(unittest.TestCase):
    def test_a1_saturation_fail(self) -> None:
        cells = [Phase3CellSummaryV1(
            model_class_id="m", seed=int(s),
            n_problems=100,
            a0_pct=80.0, a1_pct=95.0, b_pct=99.0,
            b_minus_a1_pp=4.0,
            n_b_ge_a1=95,
            mlb1_invocation_rate=0.05, mlb2_rescue_rate=0.05,
            a1_lt_saturation_max=False,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        v = evaluate_per_class_verdict_v1(
            model_class_id="m", cells=tuple(cells),
            audit_chain_re_derives=3,
            audit_chain_total=3,
            canonical_pass_rate=1.0)
        self.assertTrue(v.verdict_label.startswith("FAIL"))
        self.assertIn("A1_SATURATION", v.verdict_label)


class TestPerClassMlb2NonLoadBearing(unittest.TestCase):
    """6/6 bars PASS but MLB-2 < 33 % => RETIRED_MARGIN_DRIVEN_
    NON_LOAD_BEARING (mirrors the W96-C / W100 precedent)."""

    def test_mlb2_non_load_bearing(self) -> None:
        cells = [Phase3CellSummaryV1(
            model_class_id="m", seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0, b_pct=65.0,
            b_minus_a1_pp=10.0,
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.50, mlb2_rescue_rate=0.20,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        v = evaluate_per_class_verdict_v1(
            model_class_id="m", cells=tuple(cells),
            audit_chain_re_derives=3,
            audit_chain_total=3,
            canonical_pass_rate=1.0)
        self.assertEqual(
            v.verdict_label,
            "RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING")
        self.assertFalse(v.mlb2_load_bearing)


class TestPerClassRefuseOnDuplicateSeed(unittest.TestCase):
    def test_duplicate_seed_refuses(self) -> None:
        cells = [Phase3CellSummaryV1(
            model_class_id="m", seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0, b_pct=65.0,
            b_minus_a1_pp=10.0,
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.5, mlb2_rescue_rate=0.45,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 1)]
        with self.assertRaises(
                Phase3RetirementEvaluatorError):
            evaluate_per_class_verdict_v1(
                model_class_id="m", cells=tuple(cells),
                audit_chain_re_derives=2,
                audit_chain_total=2,
                canonical_pass_rate=1.0)


class TestPerClassRefuseOnSlicePackMismatch(unittest.TestCase):
    def test_slice_pack_mismatch_refuses(self) -> None:
        cells = [Phase3CellSummaryV1(
            model_class_id="m", seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0, b_pct=65.0,
            b_minus_a1_pp=10.0,
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.5, mlb2_rescue_rate=0.45,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=(
                "a" * 64 if s == 1 else W105_PACK_CID_LOCKED),
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        with self.assertRaises(
                Phase3RetirementEvaluatorError):
            evaluate_per_class_verdict_v1(
                model_class_id="m", cells=tuple(cells),
                audit_chain_re_derives=3,
                audit_chain_total=3,
                canonical_pass_rate=1.0)


class TestCrossClassEntitlementBothPass(unittest.TestCase):
    def test_both_retired_entitled_if_within_envelope(self) -> None:
        a = self._mk_retired_class("A", margin=10.0)
        b = self._mk_retired_class("B", margin=8.0)
        cc = evaluate_cross_class_verdict_v1(
            class_a=a, class_b=b)
        self.assertTrue(cc.cross_class_retirement_entitled)
        self.assertEqual(cc.cross_class_claim_label,
                         "CROSS_CLASS_RETIRED")

    def test_both_retired_not_entitled_if_outside_envelope(self) -> None:
        a = self._mk_retired_class("A", margin=20.0)
        b = self._mk_retired_class("B", margin=8.0)
        cc = evaluate_cross_class_verdict_v1(
            class_a=a, class_b=b)
        # Diff = 12 pp; envelope = 5 pp; NOT entitled.
        self.assertFalse(cc.cross_class_retirement_entitled)
        self.assertEqual(
            cc.cross_class_claim_label,
            "CROSS_CLASS_PARTIAL_RETIRED_MARGIN_GAP_EXCEEDS_ENVELOPE")

    def test_class_a_pass_class_b_fail_bounded_claim(self) -> None:
        a = self._mk_retired_class("A", margin=10.0)
        b = self._mk_failing_class("B")
        cc = evaluate_cross_class_verdict_v1(
            class_a=a, class_b=b)
        self.assertFalse(cc.cross_class_retirement_entitled)
        self.assertEqual(
            cc.cross_class_claim_label,
            "CLASS_A_RETIRED_CLASS_B_FAIL_BOUNDED_CLAIM")

    @staticmethod
    def _mk_retired_class(name: str, margin: float):
        cells = [Phase3CellSummaryV1(
            model_class_id=name, seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0,
            b_pct=float(55.0 + margin),
            b_minus_a1_pp=float(margin),
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.5, mlb2_rescue_rate=0.45,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        return evaluate_per_class_verdict_v1(
            model_class_id=name, cells=tuple(cells),
            audit_chain_re_derives=3, audit_chain_total=3,
            canonical_pass_rate=1.0)

    @staticmethod
    def _mk_failing_class(name: str):
        cells = [Phase3CellSummaryV1(
            model_class_id=name, seed=int(s),
            n_problems=100,
            a0_pct=50.0, a1_pct=55.0, b_pct=56.0,
            b_minus_a1_pp=1.0,
            n_b_ge_a1=60,
            mlb1_invocation_rate=0.5, mlb2_rescue_rate=0.45,
            a1_lt_saturation_max=True,
            bench_merkle_root="m" * 64,
            slice_pack_cid=W105_PACK_CID_LOCKED,
            cell_run_id=f"syn_{s}") for s in (1, 2, 3)]
        return evaluate_per_class_verdict_v1(
            model_class_id=name, cells=tuple(cells),
            audit_chain_re_derives=3, audit_chain_total=3,
            canonical_pass_rate=1.0)


class TestCrossClassComparatorPerSeedAlignment(unittest.TestCase):
    """The per-seed-aligned comparator works when both classes
    share the same iteration task_id list for each seed."""

    def test_happy_path(self) -> None:
        n = 30
        tids = [f"HumanEval/{i}" for i in range(n)]
        per_seed_iter = list(tids)
        by_seed_a: dict[int, tuple] = {}
        by_seed_b: dict[int, tuple] = {}
        for seed in (105_001, 105_002, 105_003):
            br_a = _make_cell_bench_report(
                model_id="A", seed=seed,
                n_problems=n,
                a0=0.30, a1=0.50, b=0.70,
                mlb1=0.50, mlb2=0.40)
            br_b = _make_cell_bench_report(
                model_id="B", seed=seed,
                n_problems=n,
                a0=0.30, a1=0.50, b=0.60,
                mlb1=0.50, mlb2=0.35)
            by_seed_a[seed] = (
                br_a, _make_cell_provenance(
                    model_id="A", seed=seed,
                    n_problems=n,
                    per_seed_iter=per_seed_iter))
            by_seed_b[seed] = (
                br_b, _make_cell_provenance(
                    model_id="B", seed=seed,
                    n_problems=n,
                    per_seed_iter=per_seed_iter))
        comp = build_cross_class_comparator_report_v1(
            class_a_id="A", class_b_id="B",
            class_a_by_seed=by_seed_a,
            class_b_by_seed=by_seed_b)
        self.assertEqual(comp.class_a_model_id, "A")
        self.assertEqual(comp.class_b_model_id, "B")
        self.assertEqual(len(comp.per_seed), 3)
        # Class A B-A1=+20pp; Class B B-A1=+10pp; shift = -10pp
        self.assertAlmostEqual(
            comp.aggregate_cross_class_shift_on_b_minus_a1_pp,
            -10.0, places=2)


class TestCrossClassComparatorRefuseOnIterationMismatch(unittest.TestCase):
    """If the bench's per-seed shuffle differs between classes,
    the comparator refuses to run.  W105 alignment guarantee."""

    def test_iteration_mismatch_refuses(self) -> None:
        n = 30
        tids = [f"HumanEval/{i}" for i in range(n)]
        per_seed_iter_a = list(tids)
        per_seed_iter_b = list(reversed(tids))
        seed = 105_001
        br_a = _make_cell_bench_report(
            model_id="A", seed=seed, n_problems=n,
            a0=0.30, a1=0.50, b=0.70,
            mlb1=0.50, mlb2=0.40)
        br_b = _make_cell_bench_report(
            model_id="B", seed=seed, n_problems=n,
            a0=0.30, a1=0.50, b=0.60,
            mlb1=0.50, mlb2=0.35)
        prov_a = _make_cell_provenance(
            model_id="A", seed=seed, n_problems=n,
            per_seed_iter=per_seed_iter_a)
        prov_b = _make_cell_provenance(
            model_id="B", seed=seed, n_problems=n,
            per_seed_iter=per_seed_iter_b)
        with self.assertRaises(CrossClassComparatorError):
            build_cross_class_comparator_report_v1(
                class_a_id="A", class_b_id="B",
                class_a_by_seed={seed: (br_a, prov_a)},
                class_b_by_seed={seed: (br_b, prov_b)})


class TestCrossClassComparatorRefuseOnSlicePackMismatch(unittest.TestCase):
    def test_slice_pack_mismatch_refuses(self) -> None:
        n = 30
        tids = [f"HumanEval/{i}" for i in range(n)]
        per_seed_iter = list(tids)
        seed = 105_001
        br_a = _make_cell_bench_report(
            model_id="A", seed=seed, n_problems=n,
            a0=0.30, a1=0.50, b=0.70,
            mlb1=0.50, mlb2=0.40)
        br_b = _make_cell_bench_report(
            model_id="B", seed=seed, n_problems=n,
            a0=0.30, a1=0.50, b=0.60,
            mlb1=0.50, mlb2=0.35)
        prov_a = _make_cell_provenance(
            model_id="A", seed=seed, n_problems=n,
            per_seed_iter=per_seed_iter,
            slice_pack_cid=W105_PACK_CID_LOCKED)
        prov_b = _make_cell_provenance(
            model_id="B", seed=seed, n_problems=n,
            per_seed_iter=per_seed_iter,
            slice_pack_cid="b" * 64)
        with self.assertRaises(CrossClassComparatorError):
            build_cross_class_comparator_report_v1(
                class_a_id="A", class_b_id="B",
                class_a_by_seed={seed: (br_a, prov_a)},
                class_b_by_seed={seed: (br_b, prov_b)})


class TestCrossClassComparatorRefuseOnSeedSetMismatch(unittest.TestCase):
    def test_seed_set_mismatch_refuses(self) -> None:
        n = 30
        tids = [f"HumanEval/{i}" for i in range(n)]
        per_seed_iter = list(tids)
        br = _make_cell_bench_report(
            model_id="X", seed=105_001, n_problems=n,
            a0=0.30, a1=0.50, b=0.60,
            mlb1=0.50, mlb2=0.35)
        prov = _make_cell_provenance(
            model_id="X", seed=105_001, n_problems=n,
            per_seed_iter=per_seed_iter)
        with self.assertRaises(CrossClassComparatorError):
            build_cross_class_comparator_report_v1(
                class_a_id="A", class_b_id="B",
                class_a_by_seed={105_001: (br, prov)},
                class_b_by_seed={105_002: (br, prov)})


class TestPhase3VerdictEndToEnd(unittest.TestCase):
    """Synthetic full 6-cell run: 2 classes × 3 seeds.  Class A
    retires; Class B fails on margin."""

    def test_end_to_end_split(self) -> None:
        n = 100
        tids = [f"HumanEval/{i}" for i in range(n)]
        cells_by_class: dict[str, list] = {"A": [], "B": []}
        for seed in (105_001, 105_002, 105_003):
            br_a = _make_cell_bench_report(
                model_id="A", seed=seed, n_problems=n,
                a0=0.50, a1=0.55, b=0.70,
                mlb1=0.50, mlb2=0.45)
            br_b = _make_cell_bench_report(
                model_id="B", seed=seed, n_problems=n,
                a0=0.50, a1=0.55, b=0.56,
                mlb1=0.50, mlb2=0.40)
            cells_by_class["A"].append((
                br_a, _make_cell_provenance(
                    model_id="A", seed=seed, n_problems=n,
                    per_seed_iter=tids)))
            cells_by_class["B"].append((
                br_b, _make_cell_provenance(
                    model_id="B", seed=seed, n_problems=n,
                    per_seed_iter=tids)))
        verdict = build_phase3_retirement_verdict_v1(
            cells_by_class=cells_by_class,
            audit_chain_re_derives_by_class={"A": 3, "B": 3},
            audit_chain_total_by_class={"A": 3, "B": 3},
            canonical_pass_rate_by_class={"A": 1.0, "B": 1.0})
        per_class = {
            c.model_class_id: c.verdict_label
            for c in verdict.per_class}
        self.assertEqual(per_class["A"], "RETIRED")
        self.assertTrue(per_class["B"].startswith("FAIL"))
        self.assertIsNotNone(verdict.cross_class)
        self.assertFalse(
            verdict.cross_class.cross_class_retirement_entitled)


class TestResumeSafeCellSkip(unittest.TestCase):
    """The driver's resume rule: a cell is COMPLETE iff
    `phase3_cell_verdict.json` is on disk in its dir."""

    def test_phase3_cell_verdict_marks_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cell_dir = Path(td) / "cell"
            cell_dir.mkdir()
            from scripts.run_w105_phase3_retirement_bench import (
                _cell_run_complete)
            self.assertFalse(_cell_run_complete(cell_dir))
            (cell_dir / "phase3_cell_verdict.json").write_text(
                "{}")
            self.assertTrue(_cell_run_complete(cell_dir))


class TestPhase3VerdictRefuseOnCorpusShaMismatch(unittest.TestCase):
    def test_corpus_sha_mismatch_refuses(self) -> None:
        n = 100
        tids = [f"HumanEval/{i}" for i in range(n)]
        cells_by_class: dict[str, list] = {"A": []}
        # First cell has the locked SHA; second has a different
        # SHA — evaluator should refuse.
        for i, seed in enumerate((105_001, 105_002, 105_003)):
            br = _make_cell_bench_report(
                model_id="A", seed=seed, n_problems=n,
                a0=0.50, a1=0.55, b=0.70,
                mlb1=0.50, mlb2=0.45)
            sha = (
                W105_CORPUS_SHA_LOCKED if i == 0
                else "f" * 64)
            cells_by_class["A"].append((
                br, _make_cell_provenance(
                    model_id="A", seed=seed, n_problems=n,
                    per_seed_iter=tids, corpus_sha=sha)))
        with self.assertRaises(
                Phase3RetirementEvaluatorError):
            build_phase3_retirement_verdict_v1(
                cells_by_class=cells_by_class,
                audit_chain_re_derives_by_class={"A": 3},
                audit_chain_total_by_class={"A": 3},
                canonical_pass_rate_by_class={"A": 1.0})


if __name__ == "__main__":
    unittest.main()
