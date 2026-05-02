"""Phase 83 — W36 host-diverse trust-subspace guard tests."""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    HostDiverseBasisEntry,
    HostDiverseRatificationEnvelope,
    HostRegistration,
    W36_HOST_DIVERSE_SCHEMA_VERSION,
    W36_BRANCH_HOST_DIVERSE_REROUTED,
    W36_BRANCH_HOST_DIVERSE_UNSAFE,
    select_host_diverse_projection,
    verify_host_diverse_ratification,
    _compute_host_diverse_basis_state_cid,
    _compute_w36_projection_audit_cid,
    _compute_w36_manifest_v6_cid,
    _compute_live_attestation_cid,
    _compute_host_topology_cid,
)
from vision_mvp.experiments.phase83_host_diverse_trust_subspace import (
    _stable_schema_capsule,
    run_phase83_seed_sweep,
)


def _entry(
        oracle_id: str,
        host_id: str,
        top_set: tuple[str, ...],
        score: float,
        host_health: float = 1.0,
) -> HostDiverseBasisEntry:
    return HostDiverseBasisEntry(
        cell_idx=0,
        oracle_id=oracle_id,
        host_id=host_id,
        model_id=f"model_{host_id}",
        top_set=top_set,
        projection_score=score,
        host_health=host_health,
        response_feature_signature="0123456789abcdef",
        host_attested=bool(host_id),
    )


def _registered_hosts() -> dict[str, HostRegistration]:
    return {
        "mac1": HostRegistration(
            host_id="mac1", model_id="model_mac1",
            base_url="mock://mac1", preflight_ok=True),
        "mac2": HostRegistration(
            host_id="mac2", model_id="model_mac2",
            base_url="mock://mac2", preflight_ok=True),
        "mac_bad": HostRegistration(
            host_id="mac_bad", model_id="model_bad",
            base_url="mock://bad", preflight_ok=False),
    }


def _clean_envelope() -> tuple[HostDiverseRatificationEnvelope, dict]:
    schema = _stable_schema_capsule()
    parent_w35_cid = "ab" * 32
    entries = (
        _entry("change_history", "mac1", ("gold",), 0.96),
        _entry("oncall_notes", "mac2", ("gold",), 0.92),
        _entry("service_graph", "mac1", ("decoy",), 0.99),
    )
    basis_cid = _compute_host_diverse_basis_state_cid(
        basis_entries=entries)
    host_topology_cid = _compute_host_topology_cid(
        registered_hosts={
            k: v.as_dict() for k, v in _registered_hosts().items()
        })
    live_cid = _compute_live_attestation_cid(attestations=())
    projection_cid = _compute_w36_projection_audit_cid(
        projection_branch=W36_BRANCH_HOST_DIVERSE_REROUTED,
        selected_oracle_ids=("change_history", "oncall_notes"),
        supporting_host_ids=("mac1", "mac2"),
        projection_top_set=("gold",),
        projection_score=0.94,
        projection_margin=0.10,
        host_diversity_threshold=0.86,
        host_diversity_margin_min=0.03,
        min_distinct_hosts=2,
    )
    manifest = _compute_w36_manifest_v6_cid(
        parent_w35_cid=parent_w35_cid,
        host_diverse_basis_state_cid=basis_cid,
        host_topology_cid=host_topology_cid,
        live_attestation_cid=live_cid,
        projection_audit_cid=projection_cid,
    )
    env = HostDiverseRatificationEnvelope(
        schema_version=W36_HOST_DIVERSE_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        parent_w35_cid=parent_w35_cid,
        host_diverse_basis_entries=entries,
        host_diverse_basis_state_cid=basis_cid,
        host_topology_cid=host_topology_cid,
        live_attestation_cid=live_cid,
        projection_branch=W36_BRANCH_HOST_DIVERSE_REROUTED,
        selected_oracle_ids=("change_history", "oncall_notes"),
        supporting_host_ids=("mac1", "mac2"),
        projection_top_set=("gold",),
        projection_score=0.94,
        projection_margin=0.10,
        host_diversity_threshold=0.86,
        host_diversity_margin_min=0.03,
        min_distinct_hosts=2,
        projection_audit_cid=projection_cid,
        manifest_v6_cid=manifest,
        cell_index=0,
        wire_required=True,
    )
    kwargs = dict(
        registered_schema=schema,
        registered_parent_w35_cid=parent_w35_cid,
        registered_oracle_ids=frozenset(
            {"service_graph", "change_history", "oncall_notes"}),
        registered_host_ids=frozenset({"mac1", "mac2", "mac_bad"}),
        registered_host_topology_cid=host_topology_cid,
    )
    return env, kwargs


class W36ProjectionTests(unittest.TestCase):

    def test_host_diverse_group_beats_single_high_score_host(self) -> None:
        top, oids, hosts, score, _margin, branch = (
            select_host_diverse_projection(
                basis_entries=(
                    _entry("service_graph", "mac1", ("decoy",), 0.99),
                    _entry("change_history", "mac1", ("gold",), 0.96),
                    _entry("oncall_notes", "mac2", ("gold",), 0.92),
                ),
                min_distinct_hosts=2,
            ))
        self.assertEqual(("gold",), top)
        self.assertEqual(("change_history", "oncall_notes"), oids)
        self.assertEqual(("mac1", "mac2"), hosts)
        self.assertGreaterEqual(score, 0.86)
        self.assertEqual(W36_BRANCH_HOST_DIVERSE_REROUTED, branch)

    def test_single_host_support_is_unsafe(self) -> None:
        top, oids, hosts, _score, _margin, branch = (
            select_host_diverse_projection(
                basis_entries=(
                    _entry("service_graph", "mac1", ("decoy",), 0.99),
                    _entry("change_history", "mac1", ("gold",), 0.96),
                ),
                min_distinct_hosts=2,
            ))
        self.assertEqual((), top)
        self.assertEqual((), oids)
        self.assertEqual((), hosts)
        self.assertEqual(W36_BRANCH_HOST_DIVERSE_UNSAFE, branch)


class W36VerifierTests(unittest.TestCase):

    def test_clean_envelope_passes(self) -> None:
        env, kwargs = _clean_envelope()
        outcome = verify_host_diverse_ratification(env, **kwargs)
        self.assertTrue(outcome.ok, outcome.reason)

    def test_empty_envelope_rejected(self) -> None:
        _env, kwargs = _clean_envelope()
        outcome = verify_host_diverse_ratification(None, **kwargs)
        self.assertEqual("empty_w36_envelope", outcome.reason)

    def test_schema_version_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_version="wrong")
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_schema_version_unknown", outcome.reason)

    def test_schema_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, schema_cid="wrong")
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_schema_cid_mismatch", outcome.reason)

    def test_parent_w35_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, parent_w35_cid="cd" * 32)
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w35_parent_cid_mismatch", outcome.reason)

    def test_projection_branch_unknown_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_branch="unknown")
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_projection_branch_unknown", outcome.reason)

    def test_unregistered_oracle_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.host_diverse_basis_entries)
        entries[0] = dataclasses.replace(entries[0], oracle_id="rogue")
        bad = dataclasses.replace(env, host_diverse_basis_entries=tuple(entries))
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual(
            "w36_basis_entry_unregistered_oracle", outcome.reason)

    def test_unregistered_host_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.host_diverse_basis_entries)
        entries[0] = dataclasses.replace(entries[0], host_id="rogue_host")
        bad = dataclasses.replace(env, host_diverse_basis_entries=tuple(entries))
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_basis_host_unregistered", outcome.reason)

    def test_basis_score_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.host_diverse_basis_entries)
        entries[0] = dataclasses.replace(entries[0], projection_score=2.0)
        bad = dataclasses.replace(env, host_diverse_basis_entries=tuple(entries))
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_basis_score_out_of_range", outcome.reason)

    def test_basis_health_out_of_range_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        entries = list(env.host_diverse_basis_entries)
        entries[0] = dataclasses.replace(entries[0], host_health=2.0)
        bad = dataclasses.replace(env, host_diverse_basis_entries=tuple(entries))
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_basis_health_out_of_range", outcome.reason)

    def test_basis_state_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(
            env, host_diverse_basis_state_cid="00" * 32)
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_basis_state_cid_mismatch", outcome.reason)

    def test_projection_top_set_unregistered_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, projection_top_set=("new_gold",))
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual(
            "w36_projection_top_set_unregistered", outcome.reason)

    def test_host_diversity_requirement_invalid_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, min_distinct_hosts=0)
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual(
            "w36_host_diversity_requirement_invalid", outcome.reason)

    def test_manifest_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, manifest_v6_cid="00" * 32)
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_manifest_v6_cid_mismatch", outcome.reason)

    def test_outer_cid_mismatch_rejected(self) -> None:
        env, kwargs = _clean_envelope()
        bad = dataclasses.replace(env, w36_cid="00" * 32)
        outcome = verify_host_diverse_ratification(bad, **kwargs)
        self.assertEqual("w36_outer_cid_mismatch", outcome.reason)


class W36Phase83BenchmarkTests(unittest.TestCase):

    def test_host_diverse_recover_correctness_gain_over_w35(self) -> None:
        result = run_phase83_seed_sweep(
            bank="host_diverse_recover",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertGreaterEqual(
            result["min_delta_correctness_w36_w35"], 0.25)
        self.assertEqual(1.0, result["min_trust_precision_w36"])
        self.assertLessEqual(result["max_overhead_w36_per_cell"], 1)

    def test_host_spoofed_consensus_improves_trust_precision(self) -> None:
        result = run_phase83_seed_sweep(
            bank="host_spoofed_consensus",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertGreaterEqual(
            result["min_delta_trust_precision_w36_w35"], 0.375)
        self.assertEqual(1.0, result["min_trust_precision_w36"])
        self.assertEqual(0.0, result["min_delta_correctness_w36_w35"])

    def test_trivial_w36_byte_for_w35(self) -> None:
        result = run_phase83_seed_sweep(
            bank="trivial_w36",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertTrue(result["all_byte_equivalent_w36_w35"])

    def test_no_live_attestation_is_named_falsifier(self) -> None:
        result = run_phase83_seed_sweep(
            bank="no_live_attestation",
            n_eval=16,
            seeds=(11, 17, 23, 29, 31),
        )
        self.assertLessEqual(result["max_delta_correctness_w36_w35"], -1.0)
        self.assertEqual(1.0, result["min_trust_precision_w36"])


if __name__ == "__main__":
    unittest.main()
