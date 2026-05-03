"""Phase 82 — W35 trust-subspace dense-control tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.coordpy.team_coord import (
    TrustSubspaceBasisEntry,
    TrustSubspaceDenseRatificationEnvelope,
    W35_TRUST_SUBSPACE_SCHEMA_VERSION,
    W35_BRANCH_BASIS_HISTORY_REROUTED,
    W35_BRANCH_BASIS_HISTORY_UNSAFE,
    W35_BRANCH_TRUST_SUBSPACE_RESOLVED,
    select_trust_subspace_projection,
    verify_trust_subspace_dense_ratification,
    _compute_live_attestation_cid,
    _compute_trust_subspace_basis_state_cid,
    _compute_w35_projection_audit_cid,
    _compute_w35_manifest_v5_cid,
)
from vision_mvp.experiments.phase82_trust_subspace_dense_control import (
    _stable_schema_capsule,
    run_phase82_seed_sweep,
)


def _entry(
        oracle_id: str,
        top_set: tuple[str, ...],
        score: float,
        n_observations: int = 4,
) -> TrustSubspaceBasisEntry:
    return TrustSubspaceBasisEntry(
        cell_idx=0,
        oracle_id=oracle_id,
        top_set=top_set,
        ewma_trust_after=score,
        top_set_stability_ewma=score,
        response_feature_signature="0123456789abcdef",
        response_feature_stability=1.0,
        host_health=1.0,
        n_observations=n_observations,
        projection_score=score,
    )


def _clean_envelope() -> tuple[TrustSubspaceDenseRatificationEnvelope, dict]:
    schema = _stable_schema_capsule()
    parent_w34_cid = "ab" * 32
    entries = (
        _entry("change_history", ("gold",), 1.0),
        _entry("service_graph", ("decoy",), 0.7),
        _entry("oncall_notes", ("decoy",), 0.2),
    )
    basis_cid = _compute_trust_subspace_basis_state_cid(
        basis_entries=entries)
    live_cid = _compute_live_attestation_cid(attestations=())
    projection_cid = _compute_w35_projection_audit_cid(
        projection_branch=W35_BRANCH_BASIS_HISTORY_REROUTED,
        selected_oracle_id="change_history",
        projection_top_set=("gold",),
        projection_score=1.0,
        projection_margin=0.55,
        projection_threshold=0.9,
        projection_margin_min=0.05,
    )
    manifest = _compute_w35_manifest_v5_cid(
        parent_w34_cid=parent_w34_cid,
        basis_state_cid=basis_cid,
        live_attestation_cid=live_cid,
        projection_audit_cid=projection_cid,
    )
    env = TrustSubspaceDenseRatificationEnvelope(
        schema_version=W35_TRUST_SUBSPACE_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        parent_w34_cid=parent_w34_cid,
        basis_entries=entries,
        basis_state_cid=basis_cid,
        live_attestation_cid=live_cid,
        projection_branch=W35_BRANCH_BASIS_HISTORY_REROUTED,
        selected_oracle_id="change_history",
        projection_top_set=("gold",),
        projection_score=1.0,
        projection_margin=0.55,
        projection_threshold=0.9,
        projection_margin_min=0.05,
        projection_audit_cid=projection_cid,
        manifest_v5_cid=manifest,
        cell_index=0,
        wire_required=True,
    )
    kwargs = dict(
        registered_schema=schema,
        registered_parent_w34_cid=parent_w34_cid,
        registered_oracle_ids=frozenset(
            {"service_graph", "change_history", "oncall_notes"}),
    )
    return env, kwargs


class W35ProjectionTests(unittest.TestCase):

    def test_group_average_prefers_stable_singleton_over_bad_group(self) -> None:
        top, oid, score, margin, branch = select_trust_subspace_projection(
            basis_entries=(
                _entry("change_history", ("gold",), 1.0),
                _entry("service_graph", ("decoy",), 0.95),
                _entry("oncall_notes", ("decoy",), 0.10),
            ),
            projection_threshold=0.9,
            projection_margin_min=0.05,
            min_basis_observations=3,
        )
        self.assertEqual(("gold",), top)
        self.assertEqual("change_history", oid)
        self.assertEqual(W35_BRANCH_BASIS_HISTORY_REROUTED, branch)
        self.assertGreaterEqual(score, 0.9)
        self.assertGreaterEqual(margin, 0.05)

    def test_short_history_is_unsafe(self) -> None:
        top, oid, _score, _margin, branch = select_trust_subspace_projection(
            basis_entries=(_entry("change_history", ("gold",), 1.0, 1),),
            min_basis_observations=3,
        )
        self.assertEqual((), top)
        self.assertEqual("", oid)
        self.assertEqual(W35_BRANCH_BASIS_HISTORY_UNSAFE, branch)


class W35VerifierTests(unittest.TestCase):

    def test_clean_envelope_passes(self) -> None:
        env, kwargs = _clean_envelope()
        outcome = verify_trust_subspace_dense_ratification(env, **kwargs)
        self.assertTrue(outcome.ok, outcome.reason)

    def test_empty_envelope_rejected(self) -> None:
        _env, kwargs = _clean_envelope()
        outcome = verify_trust_subspace_dense_ratification(None, **kwargs)
        self.assertEqual("empty_w35_envelope", outcome.reason)

    def test_schema_version_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_version="wrong")
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_schema_version_unknown", outcome.reason)

    def test_schema_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_cid="wrong")
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_schema_cid_mismatch", outcome.reason)

    def test_parent_w34_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, parent_w34_cid="cd" * 32)
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w34_parent_cid_mismatch", outcome.reason)

    def test_projection_branch_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_branch="unknown_branch")
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_projection_branch_unknown", outcome.reason)

    def test_unregistered_basis_oracle_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.basis_entries)
        entries[0] = dataclasses.replace(entries[0], oracle_id="rogue")
        bad = dataclasses.replace(env, basis_entries=tuple(entries))
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual(
            "w35_basis_entry_unregistered_oracle", outcome.reason)

    def test_basis_score_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.basis_entries)
        entries[0] = dataclasses.replace(entries[0], projection_score=2.0)
        bad = dataclasses.replace(env, basis_entries=tuple(entries))
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_basis_score_out_of_range", outcome.reason)

    def test_basis_stability_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.basis_entries)
        entries[0] = dataclasses.replace(entries[0], ewma_trust_after=2.0)
        bad = dataclasses.replace(env, basis_entries=tuple(entries))
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_basis_stability_out_of_range", outcome.reason)

    def test_basis_state_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, basis_state_cid="00" * 32)
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_basis_state_cid_mismatch", outcome.reason)

    def test_projection_top_set_unregistered_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_top_set=("new_gold",))
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual(
            "w35_projection_top_set_unregistered", outcome.reason)

    def test_projection_margin_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_margin=2.0)
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_projection_margin_out_of_range", outcome.reason)

    def test_live_attestation_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, live_attestation_cid="bad")
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_live_attestation_cid_mismatch", outcome.reason)

    def test_manifest_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, manifest_v5_cid="00" * 32)
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_manifest_v5_cid_mismatch", outcome.reason)

    def test_outer_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, w35_cid="00" * 32)
        outcome = verify_trust_subspace_dense_ratification(bad, **kwargs)
        self.assertEqual("w35_outer_cid_mismatch", outcome.reason)

    def test_envelope_decoder_text(self) -> None:
        env, _kwargs = _clean_envelope()
        self.assertTrue(env.to_decoder_text().startswith("<w35_ref:"))


class W35Phase82BenchmarkTests(unittest.TestCase):

    def test_h6_w35_correctness_gain_over_w34_5_seeds(self) -> None:
        result = run_phase82_seed_sweep(
            bank="trust_subspace_shift",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertGreaterEqual(
            result["min_delta_correctness_w35_w34"], 0.25)
        self.assertEqual(1.0, result["min_trust_precision_w35"])
        self.assertLessEqual(result["max_overhead_w35_per_cell"], 1)

    def test_trivial_w35_byte_for_w34(self) -> None:
        result = run_phase82_seed_sweep(
            bank="trivial_w35",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            trust_subspace_enabled=False,
            manifest_v5_disabled=True,
        )
        self.assertTrue(result["all_byte_equivalent_w35_w34"])

    def test_no_anchor_disagreement_no_correctness_lift(self) -> None:
        result = run_phase82_seed_sweep(
            bank="no_anchor_disagreement",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(0.0, result["min_delta_correctness_w35_w34"])
        self.assertEqual(1.0, result["min_trust_precision_w35"])

    def test_all_anchor_compromised_falsifier_no_recovery(self) -> None:
        result = run_phase82_seed_sweep(
            bank="all_anchor_compromised",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertEqual(0.0, result["min_delta_correctness_w35_w34"])
        self.assertEqual(0, result["seed_results"][0]["n_w35_basis_rerouted"])


if __name__ == "__main__":
    unittest.main()
