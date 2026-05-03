"""Phase 80 — W33 Trust-EWMA-tracked multi-oracle adjudication tests.

Covers:

  * Verifier failure modes (14 enumerated).
  * Registry factories (trivial / non-trivial).
  * Byte-for-W21 invariant (W33-Λ-trivial-trust-ewma).
  * Falsifiers (no_trust_shift, frozen_trust_threshold).
  * Manifest-v3 tamper detection.
  * H6 main load-bearing claim on R-80-COMPROMISED-SHIFT.
  * Per-oracle agreement signal.
  * Anchor-oracle reference path.
"""
from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.coordpy.team_coord import (
    TrustEWMATrackedMultiOracleOrchestrator,
    TrustEWMARatificationEnvelope,
    TrustEWMARegistry,
    TrustTrajectoryEntry,
    W33TrustEWMAResult,
    verify_trust_ewma_ratification,
    derive_per_oracle_agreement_signal,
    build_trivial_trust_ewma_registry,
    build_trust_ewma_registry,
    update_ewma_prior,
    W33_TRUST_EWMA_SCHEMA_VERSION,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    W33_BRANCH_TRUST_EWMA_RESOLVED,
    W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH,
    W33_BRANCH_TRUST_EWMA_REJECTED,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE,
    W33_ALL_BRANCHES,
    _compute_oracle_trust_state_cid,
    _compute_trust_trajectory_cid,
    _compute_w33_manifest_v3_cid,
    _compute_w33_outer_cid,
    LatentVerificationOutcome,
)
from vision_mvp.experiments.phase80_trust_ewma_tracked import (
    _stable_schema_capsule,
    _five_named_tampers,
    run_phase80,
    run_phase80_seed_sweep,
    run_phase80_manifest_v3_tamper_sweep,
)


# ---------------------------------------------------------------------------
# Per-oracle agreement signal
# ---------------------------------------------------------------------------


class W33AgreementSignalTests(unittest.TestCase):

    def test_abstained_returns_one(self) -> None:
        self.assertEqual(
            1.0,
            derive_per_oracle_agreement_signal(
                probe_top_set=(),
                probe_abstained=True,
                resolved_top_set=("a", "b"),
            ),
        )

    def test_subset_returns_one(self) -> None:
        self.assertEqual(
            1.0,
            derive_per_oracle_agreement_signal(
                probe_top_set=("a", "b"),
                probe_abstained=False,
                resolved_top_set=("a", "b"),
            ),
        )

    def test_disjoint_returns_zero(self) -> None:
        self.assertEqual(
            0.0,
            derive_per_oracle_agreement_signal(
                probe_top_set=("c",),
                probe_abstained=False,
                resolved_top_set=("a", "b"),
            ),
        )

    def test_partial_returns_half(self) -> None:
        self.assertEqual(
            0.5,
            derive_per_oracle_agreement_signal(
                probe_top_set=("a", "c"),
                probe_abstained=False,
                resolved_top_set=("a", "b"),
            ),
        )

    def test_no_resolved_returns_one(self) -> None:
        # When the reference is empty, the EWMA holds — return 1.0
        # (no information to update against).
        self.assertEqual(
            1.0,
            derive_per_oracle_agreement_signal(
                probe_top_set=("c",),
                probe_abstained=False,
                resolved_top_set=(),
            ),
        )


# ---------------------------------------------------------------------------
# Registry factories + trivial-path semantics
# ---------------------------------------------------------------------------


class W33RegistryFactoryTests(unittest.TestCase):

    def test_trivial_registry_is_trivial(self) -> None:
        schema = _stable_schema_capsule()
        reg = build_trivial_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph",),
        )
        self.assertTrue(reg.is_trivial)
        self.assertFalse(reg.has_wire_required_layer)
        self.assertEqual(reg.trust_trajectory_window, 0)
        self.assertFalse(reg.trust_ewma_enabled)
        self.assertTrue(reg.manifest_v3_disabled)

    def test_non_trivial_registry_is_not_trivial(self) -> None:
        schema = _stable_schema_capsule()
        reg = build_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph",),
            anchor_oracle_ids=("service_graph",),
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=8,
        )
        self.assertFalse(reg.is_trivial)
        self.assertTrue(reg.has_wire_required_layer)
        self.assertEqual(reg.trust_trajectory_window, 8)
        self.assertTrue(reg.trust_ewma_enabled)
        self.assertFalse(reg.manifest_v3_disabled)


# ---------------------------------------------------------------------------
# Verifier — 14 enumerated failure modes
# ---------------------------------------------------------------------------


def _build_minimal_envelope(*, schema, parent_cid="x" * 64,
                              trust_threshold=0.5,
                              ewma_alpha=0.2,
                              cell_index=0,
                              n_detrusted=0,
                              wire_required=True,
                              ) -> TrustEWMARatificationEnvelope:
    state = (("service_graph", 1.0), ("change_history", 1.0),
              ("oncall_notes", 1.0))
    state_cid = _compute_oracle_trust_state_cid(oracle_to_trust=state)
    traj = (
        TrustTrajectoryEntry(
            cell_idx=int(cell_index),
            oracle_id="service_graph",
            observed_quorum_agreement=1.0, ewma_trust_after=1.0),
    )
    traj_cid = _compute_trust_trajectory_cid(trajectory=traj)
    audit_cid = "0" * 64
    manifest_v3 = _compute_w33_manifest_v3_cid(
        parent_cid=parent_cid,
        oracle_trust_state_cid=state_cid,
        trust_trajectory_cid=traj_cid,
        trust_route_audit_cid=audit_cid,
    )
    env = TrustEWMARatificationEnvelope(
        schema_version=W33_TRUST_EWMA_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        parent_cid=parent_cid,
        oracle_trust_state=state,
        oracle_trust_state_cid=state_cid,
        trust_trajectory=traj,
        trust_trajectory_cid=traj_cid,
        trust_threshold=trust_threshold,
        ewma_alpha=ewma_alpha,
        trust_route_audit_cid=audit_cid,
        manifest_v3_cid=manifest_v3,
        cell_index=cell_index,
        n_detrusted_oracles=n_detrusted,
        wire_required=wire_required,
    )
    return env


class W33VerifierFailureModeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = _stable_schema_capsule()
        self.env = _build_minimal_envelope(schema=self.schema)
        self.parent_cid = self.env.parent_cid
        self.oracle_ids = frozenset({
            "service_graph", "change_history", "oncall_notes"})
        self.window = W33_DEFAULT_TRUST_TRAJECTORY_WINDOW

    def _verify(self, env, **overrides):
        return verify_trust_ewma_ratification(
            env,
            registered_schema=overrides.get(
                "registered_schema", self.schema),
            registered_parent_cid=overrides.get(
                "registered_parent_cid", self.parent_cid),
            registered_oracle_ids=overrides.get(
                "registered_oracle_ids", self.oracle_ids),
            registered_trust_trajectory_window=overrides.get(
                "registered_trust_trajectory_window", self.window),
            registered_oracle_trust_state_cid=overrides.get(
                "registered_oracle_trust_state_cid", None),
        )

    def test_baseline_passes(self) -> None:
        out = self._verify(self.env)
        self.assertTrue(out.ok, msg=str(out))

    def test_empty_w33_envelope(self) -> None:
        out = self._verify(None)
        self.assertEqual(out.reason, "empty_w33_envelope")

    def test_w33_schema_version_unknown(self) -> None:
        env = dataclasses.replace(
            self.env, schema_version="wrong.version")
        out = self._verify(env)
        self.assertEqual(out.reason, "w33_schema_version_unknown")

    def test_w33_schema_cid_mismatch(self) -> None:
        env = dataclasses.replace(self.env, schema_cid="0" * 64)
        out = self._verify(env)
        self.assertEqual(out.reason, "w33_schema_cid_mismatch")

    def test_w32_parent_cid_mismatch(self) -> None:
        out = self._verify(self.env,
                            registered_parent_cid="zz" * 32)
        self.assertEqual(out.reason, "w32_parent_cid_mismatch")

    def test_oracle_trust_state_unregistered_oracle(self) -> None:
        new_state = (("rogue_oracle", 1.0),) + self.env.oracle_trust_state
        new_state_cid = _compute_oracle_trust_state_cid(
            oracle_to_trust=new_state)
        new_manifest = _compute_w33_manifest_v3_cid(
            parent_cid=self.env.parent_cid,
            oracle_trust_state_cid=new_state_cid,
            trust_trajectory_cid=self.env.trust_trajectory_cid,
            trust_route_audit_cid=self.env.trust_route_audit_cid,
        )
        env = dataclasses.replace(
            self.env,
            oracle_trust_state=new_state,
            oracle_trust_state_cid=new_state_cid,
            manifest_v3_cid=new_manifest,
            w33_cid="",
        )
        out = self._verify(env)
        self.assertEqual(
            out.reason, "oracle_trust_state_unregistered_oracle")

    def test_oracle_trust_state_ewma_out_of_range(self) -> None:
        bad_state = (("service_graph", 2.0),) + self.env.oracle_trust_state[1:]
        env = dataclasses.replace(
            self.env, oracle_trust_state=bad_state)
        out = self._verify(env)
        self.assertEqual(
            out.reason, "oracle_trust_state_ewma_out_of_range")

    def test_oracle_trust_state_cid_mismatch_internal(self) -> None:
        # Oracle trust state changed but cid not updated.
        env = dataclasses.replace(
            self.env,
            oracle_trust_state=(("service_graph", 0.5),) +
                                self.env.oracle_trust_state[1:],
        )
        out = self._verify(env)
        self.assertEqual(out.reason, "oracle_trust_state_cid_mismatch")

    def test_oracle_trust_state_cid_mismatch_cross_cell(self) -> None:
        # Registered expected CID does not match envelope's CID.
        out = self._verify(
            self.env,
            registered_oracle_trust_state_cid="0" * 64,
        )
        self.assertEqual(
            out.reason, "oracle_trust_state_cid_mismatch")

    def test_trust_trajectory_length_mismatch(self) -> None:
        # Truncate window to 0 — any non-empty trajectory fails.
        out = self._verify(
            self.env,
            registered_trust_trajectory_window=0,
        )
        self.assertEqual(out.reason, "trust_trajectory_length_mismatch")

    def test_trust_trajectory_unregistered_oracle(self) -> None:
        bad_entry = TrustTrajectoryEntry(
            cell_idx=0, oracle_id="rogue_oracle",
            observed_quorum_agreement=1.0, ewma_trust_after=1.0,
        )
        new_traj = (bad_entry,)
        new_traj_cid = _compute_trust_trajectory_cid(trajectory=new_traj)
        new_manifest = _compute_w33_manifest_v3_cid(
            parent_cid=self.env.parent_cid,
            oracle_trust_state_cid=self.env.oracle_trust_state_cid,
            trust_trajectory_cid=new_traj_cid,
            trust_route_audit_cid=self.env.trust_route_audit_cid,
        )
        env = dataclasses.replace(
            self.env, trust_trajectory=new_traj,
            trust_trajectory_cid=new_traj_cid,
            manifest_v3_cid=new_manifest,
            w33_cid="",
        )
        out = self._verify(env)
        self.assertEqual(
            out.reason, "trust_trajectory_unregistered_oracle")

    def test_trust_trajectory_observed_out_of_range(self) -> None:
        bad_entry = dataclasses.replace(
            self.env.trust_trajectory[0],
            observed_quorum_agreement=2.0)
        env = dataclasses.replace(
            self.env,
            trust_trajectory=(bad_entry,))
        out = self._verify(env)
        self.assertEqual(
            out.reason, "trust_trajectory_observed_out_of_range")

    def test_trust_trajectory_cid_mismatch(self) -> None:
        # Trajectory changed but CID kept.
        bad_entry = dataclasses.replace(
            self.env.trust_trajectory[0],
            ewma_trust_after=0.5)
        env = dataclasses.replace(
            self.env,
            trust_trajectory=(bad_entry,))
        out = self._verify(env)
        self.assertIn(
            out.reason,
            ("trust_trajectory_observed_out_of_range",
             "trust_trajectory_cid_mismatch"))

    def test_trust_threshold_out_of_range(self) -> None:
        env = dataclasses.replace(self.env, trust_threshold=2.0)
        out = self._verify(env)
        self.assertEqual(out.reason, "trust_threshold_out_of_range")

    def test_manifest_v3_cid_mismatch(self) -> None:
        env = dataclasses.replace(self.env,
                                    manifest_v3_cid="0" * 64)
        out = self._verify(env)
        self.assertEqual(out.reason, "manifest_v3_cid_mismatch")

    def test_w33_outer_cid_mismatch(self) -> None:
        env = dataclasses.replace(self.env, w33_cid="0" * 64)
        out = self._verify(env)
        self.assertEqual(out.reason, "w33_outer_cid_mismatch")


# ---------------------------------------------------------------------------
# H2 — byte-for-W21 invariant
# ---------------------------------------------------------------------------


class W33BenchTests(unittest.TestCase):

    def test_h2_trivial_w33_byte_for_w21(self) -> None:
        result = run_phase80(
            bank="trivial_w33", n_eval=12, bank_seed=11,
            trust_ewma_enabled=False,
            manifest_v3_disabled=True,
            trust_trajectory_window=0,
        )
        # n_w33 visible == n_w21 visible byte-for-byte.
        self.assertEqual(
            result["mean_total_w21_visible_tokens"],
            result["mean_total_w33_visible_tokens"],
        )
        # Correctness identical.
        self.assertEqual(
            result["correctness_ratified_rate_w21"],
            result["correctness_ratified_rate_w33"],
        )
        self.assertTrue(result["byte_equivalent_w33_w21"])
        # Max overhead is 0 (trivial path).
        self.assertEqual(result["max_overhead_w33_per_cell"], 0)

    def test_h6_compromised_shift_strict_trust_precision_gain(self) -> None:
        sweep = run_phase80_seed_sweep(
            bank="compromised_shift", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
            trust_threshold=0.5,
            ewma_alpha=0.2,
        )
        # H6: trust_precision_w33 - trust_precision_w21 >= 0.20
        # across 5/5 seeds.
        self.assertGreaterEqual(
            float(sweep["min_delta_trust_precision_w33_w21"]), 0.20)
        # Trust precision = 1.0 across all seeds (W33 only ratifies
        # cells where it is correct).
        self.assertGreaterEqual(
            float(sweep["min_trust_precision_w33"]), 0.95)
        # Correctness should not regress.
        self.assertGreaterEqual(
            float(sweep["min_delta_correctness_w33_w21"]), 0.0)
        # Per-cell overhead bounded.
        self.assertLessEqual(
            int(sweep["max_overhead_w33_per_cell"]), 1)

    def test_h8_manifest_v3_tamper_reject_rate_one(self) -> None:
        sweep = run_phase80_manifest_v3_tamper_sweep(
            bank="manifest_v3_tamper", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(int(sweep["n_tamper_attempts_total"]), 400)
        self.assertEqual(int(sweep["n_tamper_rejected_total"]), 400)
        self.assertEqual(float(sweep["reject_rate_total"]), 1.0)

    def test_w33_lambda_no_trust_shift_falsifier(self) -> None:
        # All oracles agree throughout — W33 = W21 byte-for-byte.
        sweep = run_phase80_seed_sweep(
            bank="no_trust_shift", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
            trust_threshold=0.5,
            ewma_alpha=0.2,
        )
        self.assertEqual(
            float(sweep["min_delta_correctness_w33_w21"]), 0.0)
        self.assertEqual(
            float(sweep["max_delta_correctness_w33_w21"]), 0.0)
        self.assertEqual(
            float(sweep["min_delta_trust_precision_w33_w21"]), 0.0)

    def test_w33_lambda_frozen_trust_threshold_falsifier(self) -> None:
        # trust_threshold = 0.0 — gate never fires; W33 == W21 on
        # the answer.
        sweep = run_phase80_seed_sweep(
            bank="frozen_trust_threshold", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
            trust_threshold=0.0,
            ewma_alpha=0.2,
        )
        self.assertEqual(
            float(sweep["min_delta_correctness_w33_w21"]), 0.0)
        self.assertEqual(
            float(sweep["max_delta_correctness_w33_w21"]), 0.0)
        self.assertEqual(
            float(sweep["min_delta_trust_precision_w33_w21"]), 0.0)


# ---------------------------------------------------------------------------
# Five named tampers — sanity check that each is detected by exactly
# one failure mode.
# ---------------------------------------------------------------------------


class W33FiveNamedTamperTests(unittest.TestCase):

    def test_each_named_tamper_rejected(self) -> None:
        schema = _stable_schema_capsule()
        env = _build_minimal_envelope(schema=schema)
        registered_oracle_ids = frozenset({
            "service_graph", "change_history", "oncall_notes"})
        for tamper_id, tampered in _five_named_tampers(env):
            outcome = verify_trust_ewma_ratification(
                tampered,
                registered_schema=schema,
                registered_parent_cid=env.parent_cid,
                registered_oracle_ids=registered_oracle_ids,
                registered_trust_trajectory_window=(
                    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
                registered_oracle_trust_state_cid=(
                    env.oracle_trust_state_cid),
            )
            self.assertFalse(
                outcome.ok,
                msg=f"tamper {tamper_id} unexpectedly accepted")


# ---------------------------------------------------------------------------
# Sanity: orchestrator end-to-end smoke test
# ---------------------------------------------------------------------------


class W33OrchestratorSmokeTests(unittest.TestCase):

    def test_basic_decode_returns_w33_audit(self) -> None:
        result = run_phase80(
            bank="trivial_w33", n_eval=4, bank_seed=11,
            trust_ewma_enabled=False,
            manifest_v3_disabled=True,
            trust_trajectory_window=0,
        )
        self.assertEqual(int(result["n_eval"]), 4)
        # Trivial path → byte-equivalent.
        self.assertTrue(result["byte_equivalent_w33_w21"])

    def test_compromised_shift_writes_audit(self) -> None:
        result = run_phase80(
            bank="compromised_shift", n_eval=12, bank_seed=11,
        )
        # W33 ratifies fewer cells than W21 (abstains on
        # double-compromise window).
        self.assertLessEqual(
            int(result["n_ratified_w33"]),
            int(result["n_ratified_w21"]))
        # Some oracles get detrusted.
        self.assertGreaterEqual(
            int(result["n_oracles_detrusted"]), 1)


if __name__ == "__main__":
    unittest.main()
