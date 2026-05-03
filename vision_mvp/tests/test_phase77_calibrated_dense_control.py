"""Phase 77 W30 calibrated geometry-aware dense control unit tests.

Covers every enumerated H1 failure mode in
``verify_calibrated_geometry_ratification`` + H2 byte-for-W29 invariant
+ H3 tamper-rejection + named falsifiers + cram-factor numerator/
denominator + calibration-prior reroute logic + ancestor-chain
construction.
"""
from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.coordpy.team_coord import (
    build_incident_triage_schema_capsule,
    SchemaCapsule,
    GeometryPartitionedRatificationEnvelope,
    SubspaceBasis,
    PartitionRegistration,
    GeometryPartitionRegistry,
    GeometryPartitionedOrchestrator,
    classify_partition_id_for_cell,
    verify_geometry_partition_ratification,
    build_trivial_partition_registry,
    build_three_partition_registry,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    BasisHistory, AncestorChain, PartitionCalibrationVector,
    CalibratedGeometryRatificationEnvelope,
    CalibratedGeometryRegistry,
    CalibratedGeometryOrchestrator,
    verify_calibrated_geometry_ratification,
    update_partition_calibration_running_mean,
    build_trivial_calibrated_registry,
    build_calibrated_registry,
    W30_CALIBRATED_SCHEMA_VERSION,
    W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD,
    W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
    W30_BRANCH_CALIBRATED_RESOLVED,
    W30_BRANCH_CALIBRATION_REROUTED,
    W30_BRANCH_DISAGREEMENT_ROUTED,
)


# ---------------------------------------------------------------------------
# BasisHistory primitive tests
# ---------------------------------------------------------------------------


class BasisHistoryPrimitiveTests(unittest.TestCase):

    def test_history_cid_is_deterministic(self) -> None:
        h1 = BasisHistory(stride=4, basis_cid_history=("aa" * 32,) * 4)
        h2 = BasisHistory(stride=4, basis_cid_history=("aa" * 32,) * 4)
        self.assertEqual(h1.history_cid, h2.history_cid)

    def test_history_cid_changes_on_content_tamper(self) -> None:
        h1 = BasisHistory(stride=2, basis_cid_history=("aa" * 32, "bb" * 32))
        h2 = BasisHistory(stride=2, basis_cid_history=("aa" * 32, "cc" * 32))
        self.assertNotEqual(h1.history_cid, h2.history_cid)

    def test_history_cid_changes_on_order(self) -> None:
        # Note: order matters (no canonical sort) — rotation is detectable.
        h1 = BasisHistory(stride=2, basis_cid_history=("aa" * 32, "bb" * 32))
        h2 = BasisHistory(stride=2, basis_cid_history=("bb" * 32, "aa" * 32))
        self.assertNotEqual(h1.history_cid, h2.history_cid)

    def test_n_history_bytes_grows_with_stride(self) -> None:
        h_small = BasisHistory(stride=2, basis_cid_history=("aa" * 32,) * 2)
        h_large = BasisHistory(stride=8, basis_cid_history=("aa" * 32,) * 8)
        self.assertGreater(h_large.n_history_bytes, h_small.n_history_bytes)


# ---------------------------------------------------------------------------
# PartitionCalibrationVector primitive tests
# ---------------------------------------------------------------------------


class PartitionCalibrationVectorTests(unittest.TestCase):

    def test_calibration_cid_is_deterministic(self) -> None:
        cv1 = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.95, 0.30),
            partition_ids=(0, 1, 2),
        )
        cv2 = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.95, 0.30),
            partition_ids=(0, 1, 2),
        )
        self.assertEqual(cv1.calibration_cid, cv2.calibration_cid)

    def test_partition_ids_canonically_sorted(self) -> None:
        # Reordered input produces same CID after canonical sort.
        cv1 = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.95, 0.30),
            partition_ids=(0, 1, 2),
        )
        cv2 = PartitionCalibrationVector(
            calibration_vector=(0.30, 0.95, 0.95),
            partition_ids=(2, 0, 1),
        )
        self.assertEqual(cv1.calibration_cid, cv2.calibration_cid)

    def test_below_threshold_detection(self) -> None:
        cv = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.95, 0.30),
            partition_ids=(0, 1, 2),
            threshold=0.5,
        )
        self.assertFalse(cv.is_below_threshold(0))
        self.assertFalse(cv.is_below_threshold(1))
        self.assertTrue(cv.is_below_threshold(2))

    def test_running_mean_update(self) -> None:
        cv = PartitionCalibrationVector(
            calibration_vector=(0.5,),
            partition_ids=(W29_PARTITION_CYCLIC,),
        )
        new = update_partition_calibration_running_mean(
            prev=cv,
            partition_id=W29_PARTITION_CYCLIC,
            observed_agreement=1.0,
            n_observations_prior=1,
        )
        self.assertAlmostEqual(
            new.prior_for(W29_PARTITION_CYCLIC), 0.75, places=2)
        # Other partitions unchanged.
        cv2 = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.5),
            partition_ids=(W29_PARTITION_LINEAR, W29_PARTITION_CYCLIC),
        )
        new2 = update_partition_calibration_running_mean(
            prev=cv2,
            partition_id=W29_PARTITION_CYCLIC,
            observed_agreement=1.0,
            n_observations_prior=1,
        )
        self.assertAlmostEqual(
            new2.prior_for(W29_PARTITION_LINEAR), 0.95, places=2)


# ---------------------------------------------------------------------------
# AncestorChain primitive tests
# ---------------------------------------------------------------------------


class AncestorChainPrimitiveTests(unittest.TestCase):

    def test_chain_cid_is_sorted_invariant(self) -> None:
        a1 = AncestorChain(ancestor_window=3,
                            ancestor_chain=("aa" * 32, "bb" * 32, "cc" * 32))
        a2 = AncestorChain(ancestor_window=3,
                            ancestor_chain=("cc" * 32, "aa" * 32, "bb" * 32))
        self.assertEqual(a1.chain_cid, a2.chain_cid)

    def test_chain_cid_detects_tamper(self) -> None:
        a1 = AncestorChain(ancestor_window=2,
                            ancestor_chain=("aa" * 32, "bb" * 32))
        a2 = AncestorChain(ancestor_window=2,
                            ancestor_chain=("aa" * 32, "cc" * 32))
        self.assertNotEqual(a1.chain_cid, a2.chain_cid)


# ---------------------------------------------------------------------------
# Registry factories
# ---------------------------------------------------------------------------


class CalibratedRegistryFactoryTests(unittest.TestCase):

    def test_trivial_registry_is_trivial(self) -> None:
        schema = build_incident_triage_schema_capsule()
        reg = build_trivial_calibrated_registry(schema=schema)
        self.assertTrue(reg.is_trivial)
        self.assertFalse(reg.has_wire_required_layer)
        self.assertEqual(int(reg.calibration_stride), 0)
        self.assertIsNone(reg.calibration_vector)

    def test_calibrated_registry_is_non_trivial(self) -> None:
        schema = build_incident_triage_schema_capsule()
        reg = build_calibrated_registry(
            schema=schema,
            calibration_stride=8,
            calibration_priors=(0.95, 0.95, 0.30),
            ancestor_window=4,
        )
        self.assertFalse(reg.is_trivial)
        self.assertTrue(reg.has_wire_required_layer)
        self.assertEqual(int(reg.calibration_stride), 8)
        self.assertIsNotNone(reg.calibration_vector)


# ---------------------------------------------------------------------------
# Verifier failure-mode tests (every enumerated H1 mode)
# ---------------------------------------------------------------------------


def _make_envelope(*, schema_version=W30_CALIBRATED_SCHEMA_VERSION,
                     schema_cid="ff" * 32, w29_partition_cid="aa" * 32,
                     basis_history=None, calibration=None,
                     ancestor_chain=None, disagreement_route_active=False,
                     disagreement_route_target_partition_id=0,
                     cell_index=0, wire_required=True,
                     ) -> CalibratedGeometryRatificationEnvelope:
    return CalibratedGeometryRatificationEnvelope(
        schema_version=schema_version,
        schema_cid=schema_cid,
        w29_partition_cid=w29_partition_cid,
        basis_history=basis_history,
        calibration=calibration,
        ancestor_chain=ancestor_chain,
        disagreement_route_active=disagreement_route_active,
        disagreement_route_target_partition_id=
            disagreement_route_target_partition_id,
        cell_index=cell_index,
        wire_required=wire_required,
    )


class CalibratedVerifierFailureModeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.basis_a = "aa" * 32
        self.basis_b = "bb" * 32

    def _make_default_envelope(self) -> (
            CalibratedGeometryRatificationEnvelope):
        bh = BasisHistory(stride=2, basis_cid_history=(self.basis_a,
                                                          self.basis_b))
        cv = PartitionCalibrationVector(
            calibration_vector=(0.95, 0.95, 0.30),
            partition_ids=(0, 1, 2),
        )
        ac = AncestorChain(ancestor_window=2,
                            ancestor_chain=("aa" * 32, "bb" * 32))
        return _make_envelope(
            schema_cid=self.schema.cid,
            basis_history=bh, calibration=cv, ancestor_chain=ac,
        )

    def test_empty_envelope(self) -> None:
        outcome = verify_calibrated_geometry_ratification(
            None, registered_schema=self.schema,
            registered_w29_partition_cid="",
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "empty_calibrated_envelope")

    def test_schema_version_unknown(self) -> None:
        env = _make_envelope(schema_version="bogus")
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "schema_version_unknown")

    def test_schema_cid_mismatch(self) -> None:
        env = _make_envelope(schema_cid="00" * 32)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "schema_cid_mismatch")

    def test_w29_parent_cid_mismatch(self) -> None:
        env = _make_envelope(schema_cid=self.schema.cid,
                                w29_partition_cid="aa" * 32)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid="bb" * 32,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "w29_parent_cid_mismatch")

    def test_basis_history_stride_mismatch(self) -> None:
        bh = BasisHistory(stride=3, basis_cid_history=("aa" * 32,) * 2)
        # Force mismatch via the dataclass replace.
        # Note: BasisHistory normalises basis_cid_history to whatever's
        # passed; the verifier checks len == stride.
        env = _make_envelope(schema_cid=self.schema.cid,
                                basis_history=bh)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=3,
            registered_basis_cids=frozenset({"aa" * 32}),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "basis_history_stride_mismatch")

    def test_basis_history_unregistered_cid(self) -> None:
        bh = BasisHistory(stride=2,
                            basis_cid_history=("aa" * 32, "ff" * 32))
        env = _make_envelope(schema_cid=self.schema.cid, basis_history=bh)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=2,
            registered_basis_cids=frozenset({"aa" * 32}),  # bb not registered
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason,
                          "basis_history_contains_unregistered_cid")

    def test_basis_history_cid_mismatch(self) -> None:
        bh = BasisHistory(stride=2,
                            basis_cid_history=("aa" * 32, "bb" * 32))
        # Tamper with stored history_cid.
        object.__setattr__(bh, "history_cid", "00" * 32)
        env = _make_envelope(schema_cid=self.schema.cid, basis_history=bh)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=2,
            registered_basis_cids=frozenset({"aa" * 32, "bb" * 32}),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "basis_history_cid_mismatch")

    def test_calibration_vector_dim_mismatch(self) -> None:
        cv = PartitionCalibrationVector(
            calibration_vector=(0.5, 0.5),
            partition_ids=(0, 1),
        )
        env = _make_envelope(schema_cid=self.schema.cid, calibration=cv)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(0, 1, 2),  # registry expects 3
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "calibration_vector_dim_mismatch")

    def test_calibration_vector_out_of_range(self) -> None:
        # Construct the CV first with valid values, then corrupt.
        cv = PartitionCalibrationVector(
            calibration_vector=(0.5, 0.5, 0.5),
            partition_ids=(0, 1, 2),
        )
        # Tamper with the vector to push out of range.
        object.__setattr__(cv, "calibration_vector",
                              (2.5, 0.5, 0.5))
        env = _make_envelope(schema_cid=self.schema.cid, calibration=cv)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(0, 1, 2),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "calibration_vector_out_of_range")

    def test_calibration_cid_mismatch(self) -> None:
        cv = PartitionCalibrationVector(
            calibration_vector=(0.5, 0.5, 0.5),
            partition_ids=(0, 1, 2),
        )
        # Tamper with calibration_cid.
        object.__setattr__(cv, "calibration_cid", "00" * 32)
        env = _make_envelope(schema_cid=self.schema.cid, calibration=cv)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(0, 1, 2),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "calibration_cid_mismatch")

    def test_ancestor_chain_unregistered_cid(self) -> None:
        ac = AncestorChain(ancestor_window=2,
                            ancestor_chain=("aa" * 32, "ff" * 32))
        env = _make_envelope(schema_cid=self.schema.cid, ancestor_chain=ac)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=2,
            registered_ancestor_cids=frozenset({"aa" * 32}),  # ff not registered
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "ancestor_chain_unregistered_cid")

    def test_ancestor_chain_cid_mismatch(self) -> None:
        ac = AncestorChain(ancestor_window=2,
                            ancestor_chain=("aa" * 32, "bb" * 32))
        # Tamper with chain_cid.
        object.__setattr__(ac, "chain_cid", "00" * 32)
        env = _make_envelope(schema_cid=self.schema.cid, ancestor_chain=ac)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=2,
            registered_ancestor_cids=frozenset({"aa" * 32, "bb" * 32}),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "ancestor_chain_cid_mismatch")

    def test_disagreement_route_unsealed_target_unregistered(self) -> None:
        # Route active but target_partition_id 99 is not registered.
        env = _make_envelope(
            schema_cid=self.schema.cid,
            disagreement_route_active=True,
            disagreement_route_target_partition_id=99,
        )
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset({0, 1, 2}),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "disagreement_route_unsealed")

    def test_calibrated_cid_hash_mismatch(self) -> None:
        env = _make_envelope(schema_cid=self.schema.cid)
        # Tamper with stored calibrated_cid.
        object.__setattr__(env, "calibrated_cid", "ff" * 32)
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=0,
            registered_basis_cids=frozenset(),
            registered_calibration_partition_ids=(),
            registered_ancestor_window=0,
            registered_ancestor_cids=frozenset(),
            registered_partition_ids_for_route=frozenset(),
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "calibrated_cid_hash_mismatch")

    def test_well_formed_envelope_passes(self) -> None:
        env = self._make_default_envelope()
        outcome = verify_calibrated_geometry_ratification(
            env, registered_schema=self.schema,
            registered_w29_partition_cid=env.w29_partition_cid,
            registered_calibration_stride=2,
            registered_basis_cids=frozenset({self.basis_a, self.basis_b}),
            registered_calibration_partition_ids=(0, 1, 2),
            registered_ancestor_window=2,
            registered_ancestor_cids=frozenset({"aa" * 32, "bb" * 32}),
            registered_partition_ids_for_route=frozenset({0, 1, 2}),
        )
        self.assertTrue(outcome.ok, msg=outcome.reason)


# ---------------------------------------------------------------------------
# H2 byte-for-W29 invariant on trivial registry
# ---------------------------------------------------------------------------


class TrivialCalibrationByteEquivalenceTests(unittest.TestCase):

    def test_trivial_registry_makes_w30_equal_w29(self) -> None:
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="trivial_calibration", n_eval=8, bank_seed=11)
        # H2 anchor: every cell yields trivial_calibration_passthrough.
        self.assertIn(W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
                       result["branch_hist"])
        self.assertEqual(
            result["branch_hist"][W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH],
            result["n_eval"])
        # Total visible tokens equal byte-for-byte.
        self.assertEqual(
            result["total_w30_visible_tokens"],
            result["total_w29_visible_tokens"])
        # Correctness equal.
        self.assertEqual(
            result["correctness_ratified_rate_w30"],
            result["correctness_ratified_rate_w29"])
        # No overhead at all.
        self.assertEqual(result["max_overhead_w30_vs_w29_per_cell"], 0)


# ---------------------------------------------------------------------------
# H6 cram-factor amplification
# ---------------------------------------------------------------------------


class CramFactorAmplificationTests(unittest.TestCase):

    def test_chain_cram_clears_8x_w28_bar_at_stride_28_window_12(self) -> None:
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="chain_cram", n_eval=16, bank_seed=11,
            calibration_stride=28, ancestor_window=12,
        )
        self.assertGreaterEqual(
            result["cram_ratio_w30_over_w28"], 8.0,
            msg=f"H6 8x bar missed: {result['cram_ratio_w30_over_w28']}")
        self.assertGreaterEqual(
            result["cram_ratio_w30_over_w29"], 2.0,
            msg=f"H6 2x bar missed: {result['cram_ratio_w30_over_w29']}")
        self.assertLessEqual(
            result["max_overhead_w30_vs_w29_per_cell"], 1)


# ---------------------------------------------------------------------------
# H7 calibration-prior strict gain
# ---------------------------------------------------------------------------


class CalibrationPriorStrictGainTests(unittest.TestCase):

    def test_calibration_prior_strictly_beats_uniform_at_seed_11(self) -> None:
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="calibration_prior", n_eval=16, bank_seed=11,
            calibration_priors=(0.95, 0.95, 0.30),
        )
        self.assertGreaterEqual(
            result["delta_w30_minus_w29"], 0.10,
            msg=f"H7 +0.10 bar missed: "
                f"delta={result['delta_w30_minus_w29']}")
        self.assertGreaterEqual(
            result["trust_precision_w30"], 0.95)
        self.assertGreater(int(result["n_calibration_rerouted"]), 0)


# ---------------------------------------------------------------------------
# H8 disagreement-routing strict gain
# ---------------------------------------------------------------------------


class DisagreementRoutingStrictGainTests(unittest.TestCase):

    def test_xhost_disagree_strictly_beats_w29_at_seed_11(self) -> None:
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="xhost_disagree", n_eval=16, bank_seed=11,
        )
        self.assertGreaterEqual(
            result["delta_w30_minus_w29"], 0.10,
            msg=f"H8 +0.10 bar missed: "
                f"delta={result['delta_w30_minus_w29']}")
        self.assertGreaterEqual(
            result["trust_precision_w30"], 0.95)
        self.assertGreater(int(result["n_disagreement_routed"]), 0)


# ---------------------------------------------------------------------------
# H3 tamper-rejection bench
# ---------------------------------------------------------------------------


class TamperRejectionTests(unittest.TestCase):

    def test_calibrated_tampered_rejects_at_least_95_percent(self) -> None:
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="calibrated_tampered", n_eval=16, bank_seed=11,
        )
        rate = result.get("tamper_reject_rate", 0.0)
        self.assertIsNotNone(rate)
        self.assertGreaterEqual(
            rate, 0.95,
            msg=f"H3 tamper-reject rate below 95%: {rate}")


# ---------------------------------------------------------------------------
# Falsifier tests (W30-Λ-non-calibratable, W30-Λ-degenerate-history)
# ---------------------------------------------------------------------------


class FalsifierBenchTests(unittest.TestCase):

    def test_non_calibratable_falsifier_W30_equals_W29(self) -> None:
        """W30-Λ-non-calibratable: uniform priors → no override → no
        correctness gain over W29.
        """
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="non_calibratable", n_eval=16, bank_seed=11,
        )
        # Correctness rate equal byte-for-byte.
        self.assertAlmostEqual(
            result["correctness_ratified_rate_w30"],
            result["correctness_ratified_rate_w29"],
            places=6)
        # n_calibration_rerouted == 0 (no override fires).
        self.assertEqual(int(result["n_calibration_rerouted"]), 0)

    def test_degenerate_history_falsifier_no_cram_amplification(self) -> None:
        """W30-Λ-degenerate-history: stride=1 → no real cram
        amplification.
        """
        from vision_mvp.experiments.phase77_calibrated_dense_control import (
            run_phase77,
        )
        result = run_phase77(
            bank="degenerate_history", n_eval=16, bank_seed=11,
        )
        # Cram ratio over W29 ≤ 1.20 (no real amplification at stride=1).
        self.assertLessEqual(
            result["cram_ratio_w30_over_w29"], 1.20,
            msg=f"degenerate_history cram_ratio_w30_over_w29="
                f"{result['cram_ratio_w30_over_w29']} > 1.20")


# ---------------------------------------------------------------------------
# Schema version + sanity
# ---------------------------------------------------------------------------


class SchemaSanityTests(unittest.TestCase):

    def test_schema_version_is_v1(self) -> None:
        self.assertEqual(W30_CALIBRATED_SCHEMA_VERSION,
                          "coordpy.calibrated_geometry_ratification.v1")

    def test_default_calibration_threshold(self) -> None:
        self.assertAlmostEqual(
            W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
