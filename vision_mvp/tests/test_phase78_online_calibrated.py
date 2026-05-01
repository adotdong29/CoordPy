"""Phase 78 W31 online self-calibrated geometry-aware dense control
unit tests.

Covers every enumerated H1 failure mode in
``verify_online_calibrated_ratification`` + H2 byte-for-W30 invariant
+ H3 tamper rejection + H5 named falsifiers + H6 nonstationary-prior
discharge + H7 adaptive vs frozen threshold + H8 manifest cross-
component tamper detection.
"""
from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    build_incident_triage_schema_capsule,
    SchemaCapsule,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    PartitionCalibrationVector,
    update_partition_calibration_running_mean,
    # W31
    PriorTrajectoryEntry,
    OnlineCalibratedRatificationEnvelope,
    OnlineCalibratedRegistry,
    W31OnlineResult,
    OnlineCalibratedOrchestrator,
    verify_online_calibrated_ratification,
    derive_per_cell_agreement_signal,
    compute_adaptive_threshold,
    build_trivial_online_registry,
    build_online_calibrated_registry,
    W31_ONLINE_SCHEMA_VERSION,
    W31_DEFAULT_THRESHOLD_MIN, W31_DEFAULT_THRESHOLD_MAX,
    W31_DEFAULT_TRAJECTORY_WINDOW,
    W31_BRANCH_ONLINE_RESOLVED,
    W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH,
    W31_BRANCH_ONLINE_REJECTED,
    W31_ALL_BRANCHES,
    _compute_w31_manifest_cid,
)


# ---------------------------------------------------------------------------
# W31 helper tests
# ---------------------------------------------------------------------------


class AgreementSignalTests(unittest.TestCase):

    def test_signal_ratified_no_disagreement(self) -> None:
        self.assertEqual(
            1.0,
            derive_per_cell_agreement_signal(
                ratified=True, cross_host_disagreement_count=0))

    def test_signal_ratified_with_disagreement(self) -> None:
        self.assertEqual(
            0.0,
            derive_per_cell_agreement_signal(
                ratified=True, cross_host_disagreement_count=2))

    def test_signal_not_ratified(self) -> None:
        self.assertEqual(
            0.0,
            derive_per_cell_agreement_signal(
                ratified=False, cross_host_disagreement_count=0))


class AdaptiveThresholdTests(unittest.TestCase):

    def test_clipped_to_max(self) -> None:
        # Median of (1.0, 1.0, 1.0) = 1.0, clipped to 0.8.
        thr = compute_adaptive_threshold(
            calibration_vector=(1.0, 1.0, 1.0),
            threshold_min=0.2, threshold_max=0.8)
        self.assertAlmostEqual(thr, 0.8)

    def test_clipped_to_min(self) -> None:
        thr = compute_adaptive_threshold(
            calibration_vector=(0.05, 0.05, 0.05),
            threshold_min=0.2, threshold_max=0.8)
        self.assertAlmostEqual(thr, 0.2)

    def test_median_within_band(self) -> None:
        thr = compute_adaptive_threshold(
            calibration_vector=(0.3, 0.5, 0.7),
            threshold_min=0.2, threshold_max=0.8)
        self.assertAlmostEqual(thr, 0.5)

    def test_empty_vector_returns_midpoint(self) -> None:
        thr = compute_adaptive_threshold(
            calibration_vector=(),
            threshold_min=0.2, threshold_max=0.8)
        self.assertAlmostEqual(thr, 0.5)


# ---------------------------------------------------------------------------
# PriorTrajectoryEntry primitive
# ---------------------------------------------------------------------------


class PriorTrajectoryEntryTests(unittest.TestCase):

    def test_as_dict_roundtrips_floats(self) -> None:
        e = PriorTrajectoryEntry(
            cell_idx=3, partition_id=2,
            observed_agreement=0.75, prior_after=0.6667)
        d = e.as_dict()
        self.assertEqual(d["cell_idx"], 3)
        self.assertEqual(d["partition_id"], 2)
        self.assertAlmostEqual(d["observed_agreement"], 0.75)
        self.assertAlmostEqual(d["prior_after"], 0.6667)


# ---------------------------------------------------------------------------
# OnlineCalibratedRatificationEnvelope primitive
# ---------------------------------------------------------------------------


def _make_simple_envelope(
        *,
        schema: SchemaCapsule,
        cell_index: int = 0,
        traj: tuple[PriorTrajectoryEntry, ...] = (),
        thresholds: tuple[float, ...] = (),
        wire_required: bool = True,
) -> OnlineCalibratedRatificationEnvelope:
    from vision_mvp.wevra.team_coord import (
        _compute_prior_trajectory_cid,
        _compute_threshold_trajectory_cid,
    )
    bh_cid = "aa" * 32
    cal_cid = "bb" * 32
    anc_cid = "cc" * 32
    route_cid = "dd" * 32
    ptraj_cid = _compute_prior_trajectory_cid(trajectory=traj)
    thr_cid = _compute_threshold_trajectory_cid(thresholds=thresholds)
    manifest_cid = _compute_w31_manifest_cid(
        basis_history_cid=bh_cid, calibration_cid=cal_cid,
        ancestor_chain_cid=anc_cid,
        prior_trajectory_cid=ptraj_cid,
        threshold_trajectory_cid=thr_cid,
        route_audit_cid=route_cid,
    )
    return OnlineCalibratedRatificationEnvelope(
        schema_version=W31_ONLINE_SCHEMA_VERSION,
        schema_cid=schema.cid,
        w30_calibrated_cid="ee" * 32,
        prior_trajectory=traj,
        prior_trajectory_cid=ptraj_cid,
        threshold_trajectory=thresholds,
        threshold_trajectory_cid=thr_cid,
        basis_history_cid=bh_cid,
        calibration_cid=cal_cid,
        ancestor_chain_cid=anc_cid,
        route_audit_cid=route_cid,
        manifest_cid=manifest_cid,
        cell_index=cell_index,
        wire_required=wire_required,
    )


class W31EnvelopePrimitiveTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def test_envelope_self_consistent_recomputes_to_same_cid(self) -> None:
        env = _make_simple_envelope(schema=self.schema)
        self.assertEqual(env.recompute_w31_cid(), env.w31_cid)

    def test_decoder_text_uses_w31_cid_prefix(self) -> None:
        env = _make_simple_envelope(schema=self.schema)
        self.assertTrue(env.to_decoder_text().startswith("<w31_ref:"))

    def test_n_wire_tokens_zero_when_not_required(self) -> None:
        env = _make_simple_envelope(schema=self.schema, wire_required=False)
        self.assertEqual(env.n_wire_tokens, 0)

    def test_n_wire_tokens_nonzero_when_required(self) -> None:
        env = _make_simple_envelope(schema=self.schema, wire_required=True)
        self.assertGreater(env.n_wire_tokens, 0)


# ---------------------------------------------------------------------------
# Verifier failure-mode tests (every enumerated H1 mode)
# ---------------------------------------------------------------------------


class W31VerifierFailureModeTests(unittest.TestCase):
    """Mechanically asserts every enumerated H1 failure mode in the
    W31 verifier.
    """

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.env = _make_simple_envelope(schema=self.schema)
        self.registered_pids = frozenset(
            (W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
             W29_PARTITION_CYCLIC))

    def _verify(self, env, **overrides) -> tuple[bool, str]:
        kwargs = dict(
            registered_schema=self.schema,
            registered_w30_calibrated_cid=self.env.w30_calibrated_cid,
            registered_partition_ids=self.registered_pids,
            registered_trajectory_window=8,
            registered_basis_history_cid=self.env.basis_history_cid,
            registered_calibration_cid=self.env.calibration_cid,
            registered_ancestor_chain_cid=self.env.ancestor_chain_cid,
            registered_route_audit_cid=self.env.route_audit_cid,
        )
        kwargs.update(overrides)
        outcome = verify_online_calibrated_ratification(env, **kwargs)
        return outcome.ok, outcome.reason

    def test_failure_empty_w31_envelope(self) -> None:
        ok, reason = self._verify(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "empty_w31_envelope")

    def test_failure_w31_schema_version_unknown(self) -> None:
        bad = dataclasses.replace(
            self.env, schema_version="foo.bar.v1", w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "w31_schema_version_unknown")

    def test_failure_w31_schema_cid_mismatch(self) -> None:
        bad = dataclasses.replace(
            self.env, schema_cid="ff" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "w31_schema_cid_mismatch")

    def test_failure_w30_parent_cid_mismatch(self) -> None:
        bad = dataclasses.replace(
            self.env, w30_calibrated_cid="ff" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "w30_parent_cid_mismatch")

    def test_failure_prior_trajectory_length_mismatch_long(self) -> None:
        # Trajectory longer than registered window cap.
        traj = tuple(
            PriorTrajectoryEntry(cell_idx=i, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0)
            for i in range(10))
        thresholds = (0.5,) * 10
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env, registered_trajectory_window=5)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_length_mismatch")

    def test_failure_prior_trajectory_length_mismatch_non_monotone(
            self) -> None:
        # Non-monotone cell indices.
        traj = (
            PriorTrajectoryEntry(cell_idx=2, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),
            PriorTrajectoryEntry(cell_idx=1, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),
        )
        thresholds = (0.5, 0.5)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env, registered_trajectory_window=8)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_length_mismatch")

    def test_failure_prior_trajectory_unregistered_partition(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=99,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_unregistered_partition")

    def test_failure_prior_trajectory_observed_out_of_range(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=2.5, prior_after=1.0),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_observed_out_of_range")

    def test_failure_prior_trajectory_prior_after_out_of_range(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.7),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_prior_after_out_of_range")

    def test_failure_prior_trajectory_cid_mismatch_internal(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        bad = dataclasses.replace(env, prior_trajectory_cid="00" * 32,
                                    w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_cid_mismatch")

    def test_failure_prior_trajectory_cid_mismatch_cross_cell(self) -> None:
        # Cross-cell swap: registered expects different CID.
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(
            env,
            registered_prior_trajectory_cid="ff" * 32,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_trajectory_cid_mismatch")

    def test_failure_threshold_trajectory_length_mismatch(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (0.5, 0.5, 0.5)  # length != trajectory length
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env)
        self.assertFalse(ok)
        self.assertEqual(reason, "threshold_trajectory_length_mismatch")

    def test_failure_threshold_trajectory_value_out_of_range(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (1.7,)  # > 1
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        ok, reason = self._verify(env)
        self.assertFalse(ok)
        self.assertEqual(reason, "threshold_trajectory_value_out_of_range")

    def test_failure_threshold_trajectory_cid_mismatch_internal(self) -> None:
        traj = (
            PriorTrajectoryEntry(cell_idx=0, partition_id=0,
                                  observed_agreement=1.0, prior_after=1.0),)
        thresholds = (0.5,)
        env = _make_simple_envelope(
            schema=self.schema, traj=traj, thresholds=thresholds)
        bad = dataclasses.replace(env, threshold_trajectory_cid="00" * 32,
                                    w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "threshold_trajectory_cid_mismatch")

    def test_failure_manifest_cid_mismatch_corruption(self) -> None:
        bad = dataclasses.replace(self.env, manifest_cid="00" * 32,
                                    w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        # Either manifest mismatch (when components match) or outer
        # mismatch.
        self.assertIn(reason, ("manifest_cid_mismatch",
                                "w31_outer_cid_mismatch"))

    def test_failure_w31_outer_cid_mismatch(self) -> None:
        bad = dataclasses.replace(self.env)
        # Bypass __post_init__ recompute by setting raw attribute.
        object.__setattr__(bad, "w31_cid", "00" * 32)
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "w31_outer_cid_mismatch")

    def test_basis_history_passthrough_mismatch_is_manifest_failure(
            self) -> None:
        # A swap of basis_history_cid is detected by the manifest
        # passthrough check — registered says one CID, env carries
        # another.
        bad = dataclasses.replace(
            self.env, basis_history_cid="ee" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "manifest_cid_mismatch")

    def test_calibration_cid_passthrough_mismatch(self) -> None:
        bad = dataclasses.replace(
            self.env, calibration_cid="ee" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "manifest_cid_mismatch")

    def test_ancestor_chain_cid_passthrough_mismatch(self) -> None:
        bad = dataclasses.replace(
            self.env, ancestor_chain_cid="ee" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "manifest_cid_mismatch")

    def test_route_audit_cid_passthrough_mismatch(self) -> None:
        bad = dataclasses.replace(
            self.env, route_audit_cid="ee" * 32, w31_cid="")
        ok, reason = self._verify(bad)
        self.assertFalse(ok)
        self.assertEqual(reason, "manifest_cid_mismatch")


# ---------------------------------------------------------------------------
# Registry factories
# ---------------------------------------------------------------------------


class W31RegistryFactoryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def test_trivial_registry_is_trivial(self) -> None:
        reg = build_trivial_online_registry(schema=self.schema)
        self.assertTrue(reg.is_trivial)
        self.assertFalse(reg.has_wire_required_layer)

    def test_nontrivial_registry_is_not_trivial(self) -> None:
        reg = build_online_calibrated_registry(
            schema=self.schema, online_enabled=True,
            adaptive_threshold=True, manifest_disabled=False,
            trajectory_window=8)
        self.assertFalse(reg.is_trivial)
        self.assertTrue(reg.has_wire_required_layer)

    def test_partial_nontrivial_only_window(self) -> None:
        # Enabling only trajectory_window > 0 also makes the registry
        # non-trivial.
        reg = build_online_calibrated_registry(
            schema=self.schema, online_enabled=False,
            adaptive_threshold=False, manifest_disabled=True,
            trajectory_window=4)
        self.assertFalse(reg.is_trivial)


# ---------------------------------------------------------------------------
# End-to-end (phase78) bench tests — H2 / H3 / H5 / H6 / H7 / H8
# ---------------------------------------------------------------------------


class W31BenchTests(unittest.TestCase):
    """End-to-end bench checks via the phase78 driver."""

    def test_h2_trivial_w31_byte_for_w30(self) -> None:
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78
        for seed in (11, 17, 23):
            r = run_phase78(bank="trivial_w31", bank_seed=seed)
            self.assertTrue(r["byte_equivalent_w31_w30"],
                              f"seed {seed} not byte-equivalent")
            self.assertEqual(r["correctness_ratified_rate_w30"],
                              r["correctness_ratified_rate_w31"])
            self.assertEqual(r["mean_overhead_w31_vs_w30_per_cell"], 0.0)

    def test_h6_nonstationary_prior_clears_delta_010(self) -> None:
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78_seed_sweep
        r = run_phase78_seed_sweep(bank="nonstationary_prior")
        self.assertGreaterEqual(r["min_delta_w31_minus_w30"], 0.10)
        self.assertGreaterEqual(r["min_correctness_w31"], 0.80)
        self.assertEqual(r["min_trust_precision_w31"], 1.0)

    def test_h7_adaptive_minus_frozen_at_least_005(self) -> None:
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78_seed_sweep
        adaptive = run_phase78_seed_sweep(bank="adaptive_threshold")
        frozen = run_phase78_seed_sweep(bank="frozen_threshold")
        adaptive_min = adaptive["min_correctness_w31"]
        frozen_max = frozen["max_correctness_w31"]
        self.assertGreaterEqual(adaptive_min - frozen_max, 0.05)

    def test_h8_manifest_tamper_reject_rate_above_095(self) -> None:
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78
        r = run_phase78(bank="manifest_tamper", bank_seed=11)
        self.assertGreaterEqual(r["tamper_reject_rate"], 0.95)

    def test_w31_lambda_no_drift_falsifier(self) -> None:
        # On a stationary regime, online learning gives no help.
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78_seed_sweep
        r = run_phase78_seed_sweep(bank="no_drift")
        self.assertEqual(r["min_delta_w31_minus_w30"], 0.0)

    def test_w31_lambda_frozen_threshold_falsifier(self) -> None:
        # With adaptive_threshold=False, the threshold never adapts.
        from vision_mvp.experiments.phase78_online_calibrated_dense_control \
            import run_phase78_seed_sweep
        r = run_phase78_seed_sweep(bank="frozen_threshold")
        # Frozen threshold yields zero gain on this regime.
        self.assertEqual(r["min_delta_w31_minus_w30"], 0.0)


if __name__ == "__main__":
    unittest.main()
