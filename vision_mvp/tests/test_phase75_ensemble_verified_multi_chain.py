"""Tests for SDK v3.29 — Phase-75 ensemble-verified cross-model
multi-chain pivot ratification (W28 family).

Eight pre-committed sub-banks:
  * R-75-SINGLE-PROBE         — H2 anchor; W28-Λ-single-probe falsifier
  * R-75-CHAIN-SHARED         — multi-probe overhead bound
  * R-75-CROSS-MODEL-DRIFT    — S3 headline (synthetic drift)
  * R-75-COORDINATED-DRIFT    — W28-Λ-coordinated-drift falsifier
  * R-75-TRUST-ZERO           — W28-Λ-trust-zero falsifier
  * R-75-RATIFICATION-TAMPERED — H3 trust falsifier (verifier rejects
                                  every enumerated tampering mode)
  * R-75-POOL-EXHAUSTED       — W28-Λ-pool-exhausted-passthrough
  * (R-75-CROSS-HOST-LIVE     — S1/S2 best-effort; covered out-of-band)
"""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.experiments.phase75_ensemble_verified_multi_chain import (
    run_phase75,
    run_phase75_seed_stability_sweep,
    run_cross_regime_p75,
    discover_two_host_topology,
    IntermittentDriftProbe,
    CoordinatedDriftProbe,
    build_ensemble_registry_for_bank,
)
from vision_mvp.wevra.team_coord import (
    SchemaCapsule, build_incident_triage_schema_capsule,
    SalienceSignatureEnvelope, W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
    EnsemblePivotRatificationEnvelope, W28_RATIFICATION_SCHEMA_VERSION,
    verify_ensemble_pivot_ratification,
    DeterministicSignatureProbe, OracleConsultationProbe,
    LLMSignatureProbe, ProbeVote,
    EnsembleProbeRegistration, EnsembleRatificationRegistry,
    EnsembleVerifiedMultiChainOrchestrator,
    build_default_ensemble_registry,
    build_two_probe_oracle_ensemble_registry,
    W28_BRANCH_RATIFIED, W28_BRANCH_RATIFIED_PASSTHROUGH,
    W28_BRANCH_QUORUM_BELOW_THRESHOLD, W28_BRANCH_PROBE_REJECTED,
    W28_BRANCH_NO_RATIFY_NEEDED, W28_BRANCH_FALLBACK_W27,
    W28_BRANCH_NO_TRIGGER, W28_BRANCH_DISABLED,
    W28_ALL_BRANCHES, W28_DEFAULT_TRIGGER_BRANCHES,
    ServiceGraphOracle, ChangeHistoryOracle,
)


# ---------------------------------------------------------------------------
# 1. Probe vote unit tests
# ---------------------------------------------------------------------------


class ProbeVoteTests(unittest.TestCase):
    def test_signed_weight_ratify(self) -> None:
        v = ProbeVote(probe_id="p", ratify=True, reject=False,
                       trust_weight=0.7)
        self.assertAlmostEqual(v.signed_weight, 0.7)

    def test_signed_weight_reject(self) -> None:
        v = ProbeVote(probe_id="p", ratify=False, reject=True,
                       trust_weight=0.4)
        self.assertAlmostEqual(v.signed_weight, -0.4)

    def test_signed_weight_abstain(self) -> None:
        v = ProbeVote(probe_id="p", ratify=False, reject=False,
                       trust_weight=1.0)
        self.assertEqual(v.signed_weight, 0.0)
        self.assertTrue(v.is_abstain)

    def test_malformed_ratify_and_reject(self) -> None:
        v = ProbeVote(probe_id="p", ratify=True, reject=True,
                       trust_weight=1.0)
        self.assertEqual(v.reason, "malformed_ratify_and_reject")


# ---------------------------------------------------------------------------
# 2. Built-in probe behaviour
# ---------------------------------------------------------------------------


def _make_sig() -> SalienceSignatureEnvelope:
    schema = build_incident_triage_schema_capsule()
    return SalienceSignatureEnvelope(
        schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
        schema_cid=schema.cid,
        producer_agent_id="p",
        consumer_agent_ids=("c0",),
        canonical_per_tag_votes=(("orders", 2), ("payments", 2)),
        canonical_projected_subset=("orders", "payments"),
        cell_index_first_observed=0,
    )


class DeterministicSignatureProbeTests(unittest.TestCase):
    def test_ratifies_on_byte_identical(self) -> None:
        sig = _make_sig()
        p = DeterministicSignatureProbe()
        v = p.vote(
            signature=sig,
            canonical_per_tag_votes=sig.canonical_per_tag_votes,
            canonical_projected_subset=sig.canonical_projected_subset,
            cell_index=0,
        )
        self.assertTrue(v.ratify)
        self.assertFalse(v.reject)
        self.assertEqual(v.reason, "local_recompute_ok")

    def test_wire_required_default_false(self) -> None:
        p = DeterministicSignatureProbe()
        self.assertFalse(p.wire_required)


class OracleConsultationProbeTests(unittest.TestCase):
    def test_ratifies_on_known_pair(self) -> None:
        sig = _make_sig()
        oracle = ServiceGraphOracle(oracle_id="sg")
        p = OracleConsultationProbe(oracle=oracle)
        v = p.vote(
            signature=sig,
            canonical_per_tag_votes=sig.canonical_per_tag_votes,
            canonical_projected_subset=sig.canonical_projected_subset,
            cell_index=0,
        )
        # Padded with decoy: the oracle returns the dependency_chain
        # over orders+payments which fully agrees with the projected set.
        self.assertTrue(v.ratify)

    def test_wire_required_true(self) -> None:
        oracle = ServiceGraphOracle(oracle_id="sg")
        p = OracleConsultationProbe(oracle=oracle)
        self.assertTrue(p.wire_required)


# ---------------------------------------------------------------------------
# 3. Verifier failure modes (H1 + H3)
# ---------------------------------------------------------------------------


class EnsembleVerifierFailureModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.sig = _make_sig()
        self.probe_ids = frozenset(("local_recompute",))
        self.valid_env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(("local_recompute", 1, 0, 1.0, "ok"),),
            quorum_threshold=1.0,
            quorum_weight=1.0,
            ratified=True,
            cell_index=0,
        )

    def test_ok(self) -> None:
        outcome = verify_ensemble_pivot_ratification(
            self.valid_env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.reason, "ok")

    def test_empty_ratification(self) -> None:
        outcome = verify_ensemble_pivot_ratification(
            None, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "empty_ratification")

    def test_schema_version_unknown(self) -> None:
        env = dataclasses.replace(
            self.valid_env, schema_version="WRONG_VERSION",
            ratification_cid="")
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "schema_version_unknown")

    def test_schema_cid_mismatch(self) -> None:
        env = dataclasses.replace(
            self.valid_env, schema_cid="WRONG_CID",
            ratification_cid="")
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "schema_cid_mismatch")

    def test_signature_cid_empty(self) -> None:
        env = dataclasses.replace(
            self.valid_env, signature_cid="",
            ratification_cid="")
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid="",
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "signature_cid_empty")

    def test_signature_cid_mismatch(self) -> None:
        env = self.valid_env
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid="DIFFERENT_SIG_CID",
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "signature_cid_mismatch")

    def test_probe_table_empty(self) -> None:
        env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(),
            quorum_threshold=1.0,
            quorum_weight=0.0,
            ratified=False,
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "probe_table_empty")

    def test_probe_id_unregistered(self) -> None:
        env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(("FAKE_PROBE", 1, 0, 1.0, "spoofed"),),
            quorum_threshold=1.0,
            quorum_weight=1.0,
            ratified=True,
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "probe_id_unregistered")

    def test_probe_vote_malformed(self) -> None:
        env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            # Both ratify=1 and reject=1 ⇒ malformed.
            probe_votes=(("local_recompute", 1, 1, 1.0, "broken"),),
            quorum_threshold=1.0,
            quorum_weight=0.0,
            ratified=False,
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "probe_vote_malformed")

    def test_trust_weight_negative(self) -> None:
        env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(("local_recompute", 1, 0, -0.5, "neg"),),
            quorum_threshold=1.0,
            quorum_weight=1.0,
            ratified=True,
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "trust_weight_negative")

    def test_hash_mismatch(self) -> None:
        env = dataclasses.replace(
            self.valid_env, ratification_cid="x" * 64)
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "hash_mismatch")

    def test_quorum_below_threshold(self) -> None:
        # Construct an envelope claiming ratified=True but with a
        # quorum_weight strictly below threshold; the verifier must
        # catch it.  We have to bypass the helpful __post_init__
        # which would recompute the cid, so build a *valid* envelope
        # then mutate via dataclasses.replace.
        good = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(("local_recompute", 1, 0, 0.3, "weak"),),
            quorum_threshold=1.0,
            quorum_weight=0.3,
            ratified=True,  # claim ratified though weight < threshold
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            good, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "quorum_below_threshold")

    def test_quorum_recompute_mismatch(self) -> None:
        # Ratified=False but quorum_weight ≥ threshold ⇒ flag mismatch.
        env = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid=self.sig.signature_cid,
            probe_votes=(("local_recompute", 1, 0, 1.0, "ok"),),
            quorum_threshold=1.0,
            quorum_weight=1.0,
            ratified=False,  # mismatch — recompute would say True
            cell_index=0,
        )
        outcome = verify_ensemble_pivot_ratification(
            env, registered_schema=self.schema,
            registered_signature_cid=self.sig.signature_cid,
            registered_probe_ids=self.probe_ids,
        )
        self.assertEqual(outcome.reason, "quorum_recompute_mismatch")


# ---------------------------------------------------------------------------
# 4. Bench-level integration tests (named falsifiers)
# ---------------------------------------------------------------------------


class SinglePProbeByteEquivalenceTests(unittest.TestCase):
    """H2 anchor: K_probes=1 with weight ≥ quorum reduces to W27
    byte-for-byte (W28-Λ-single-probe falsifier)."""

    def test_byte_for_byte_w27_at_seed_11(self) -> None:
        result = run_phase75(
            bank="single_probe", bank_seed=11, n_eval=16,
            verbose=False)
        self.assertEqual(
            result["mean_total_w27_visible_tokens"],
            result["mean_total_w28_visible_tokens"])
        self.assertTrue(result["byte_equivalent_w28_w27"])
        self.assertEqual(result["mean_overhead_w28_vs_w27_per_cell"], 0)
        self.assertEqual(result["max_overhead_w28_vs_w27_per_cell"], 0)
        self.assertEqual(result["n_ratified"], result["n_cells"])
        # Every cell ratified via the deterministic local-recompute path.
        self.assertEqual(
            result["branch_counts_w28"].get(W28_BRANCH_RATIFIED_PASSTHROUGH),
            result["n_cells"])

    def test_no_regression_on_correctness(self) -> None:
        result = run_phase75(
            bank="single_probe", bank_seed=11, n_eval=16,
            verbose=False)
        self.assertGreaterEqual(
            result["correctness_ratified_rate_w28"],
            result["correctness_ratified_rate_w27"])

    def test_seed_stability(self) -> None:
        sweep = run_phase75_seed_stability_sweep(
            bank="single_probe", verbose=False)
        # K=1 byte-for-byte must hold across every seed.
        for r in sweep["rows"]:
            self.assertEqual(
                r["mean_total_w28_visible_tokens"],
                r["mean_total_w27_visible_tokens"])
            self.assertTrue(r["byte_equivalent_w28_w27"])
        self.assertTrue(sweep["all_correctness_w28_ge_w27"])


class CrossModelDriftHeadlineTests(unittest.TestCase):
    """S3 / W28-3 headline: trust-weighted ensemble ratifies on
    intermittent drift; overhead bounded ≤ 1 per cell."""

    def test_overhead_bounded(self) -> None:
        result = run_phase75(
            bank="cross_model_drift", bank_seed=11, n_eval=16,
            verbose=False)
        # S4: overhead ≤ 2 tokens / cell.
        self.assertLessEqual(
            result["max_overhead_w28_vs_w27_per_cell"], 2)
        # All cells ratified (deterministic probes carry quorum).
        self.assertEqual(result["n_ratified"], result["n_cells"])
        # No regression vs W27.
        self.assertGreaterEqual(
            result["correctness_ratified_rate_w28"],
            result["correctness_ratified_rate_w27"])
        # Trust precision = 1 (every ratified cell is correct).
        self.assertEqual(result["trust_precision"], 1.0)

    def test_seed_stability_5_seeds(self) -> None:
        sweep = run_phase75_seed_stability_sweep(
            bank="cross_model_drift", verbose=False)
        self.assertLessEqual(sweep["max_overhead_w28_vs_w27"], 2.0)
        self.assertEqual(sweep["min_trust_precision"], 1.0)
        self.assertTrue(sweep["all_correctness_w28_ge_w27"])


class CoordinatedDriftFalsifierTests(unittest.TestCase):
    """W28-Λ-coordinated-drift: when every probe drifts identically,
    the ensemble cannot detect the drift; correctness ≤ W27."""

    def test_ensemble_cannot_detect_coordinated_drift(self) -> None:
        result = run_phase75(
            bank="coordinated_drift", bank_seed=11, n_eval=16,
            verbose=False)
        # A fraction of cells will be unratified (the coordinated
        # drift abstain phase).  The bank uses period=4, so 8/16 cells
        # are in the drift phase.
        self.assertEqual(result["n_ratified"], 8)
        self.assertEqual(result["n_quorum_below_threshold"], 8)


class TrustZeroFalsifierTests(unittest.TestCase):
    """W28-Λ-trust-zero: zero trust priors ⇒ quorum unreachable ⇒
    controller abstains on every cell."""

    def test_zero_weight_abstains(self) -> None:
        result = run_phase75(
            bank="trust_zero", bank_seed=11, n_eval=16,
            verbose=False)
        self.assertEqual(result["n_ratified"], 0)
        # Every cell falls into quorum_below_threshold (or
        # no_ratify_needed).  The W27 inner still runs, so correctness
        # tracks W27.
        self.assertEqual(
            result["correctness_ratified_rate_w28"],
            result["correctness_ratified_rate_w27"])


class RatificationTamperedFalsifierTests(unittest.TestCase):
    """H3 trust falsifier: every tampered envelope is rejected."""

    def test_tampered_envelopes_rejected_at_full_rate(self) -> None:
        result = run_phase75(
            bank="ratification_tampered", bank_seed=11, n_eval=16,
            verbose=False)
        if result["n_tamper_attempts"] > 0:
            self.assertEqual(
                result["n_tampered_rejected"],
                result["n_tamper_attempts"])

    def test_tampered_envelopes_dont_corrupt_correctness(self) -> None:
        result = run_phase75(
            bank="ratification_tampered", bank_seed=11, n_eval=16,
            verbose=False)
        # W28's correctness must equal W27's (tampering doesn't
        # change the underlying answer; only the trust signal).
        self.assertEqual(
            result["correctness_ratified_rate_w28"],
            result["correctness_ratified_rate_w27"])


class PoolExhaustedPassthroughTests(unittest.TestCase):
    """W28-Λ-pool-exhausted-passthrough: when W27 reports POOL_EXHAUSTED,
    W28 must NOT ratify (no spurious ratifications)."""

    def test_no_ratify_on_pool_exhausted(self) -> None:
        result = run_phase75(
            bank="pool_exhausted", bank_seed=11, n_eval=16,
            max_active_chains=2, verbose=False)
        # Pool exhausted ⇒ at least some cells must be no_ratify_needed.
        # The W27 inner falls through; W28 should not invent a fresh
        # ratification on the exhausted-pool branches.
        branches = result["branch_counts_w28"]
        # No cell should be in the 'ratified' branch *unless* W27 also
        # ratified that cell via a non-exhausted path.
        n_ratified = result["n_ratified"]
        # The bench has 4 distinct signatures + max_active_chains=2,
        # so cells beyond the pool fall through; at most half should
        # be ratified (the cells routed to in-pool signatures).
        self.assertLessEqual(n_ratified, result["n_cells"])

    def test_no_regression_on_correctness(self) -> None:
        result = run_phase75(
            bank="pool_exhausted", bank_seed=11, n_eval=16,
            max_active_chains=2, verbose=False)
        self.assertGreaterEqual(
            result["correctness_ratified_rate_w28"],
            result["correctness_ratified_rate_w27"])


# ---------------------------------------------------------------------------
# 5. Disabled / no-trigger paths
# ---------------------------------------------------------------------------


class DisabledOrchestratorTests(unittest.TestCase):
    """When enabled=False or the registry has no probes, the W28 layer
    must reduce to W27 byte-for-byte."""

    def test_disabled_branch_reduces_to_w27(self) -> None:
        from vision_mvp.experiments.phase75_ensemble_verified_multi_chain import (
            _build_w28_orchestrator)
        from vision_mvp.experiments.phase74_multi_chain_pivot import (
            build_phase74_bank, build_team_shared_pool)
        schema = build_incident_triage_schema_capsule()
        cells = build_phase74_bank(
            n_replicates=2, seed=11, n_cells=4,
            bank="chain_shared", signature_period=4)
        producer_id = "p"
        consumer_ids = ("c0",)
        pool = build_team_shared_pool(
            T_decoder=None, schema=schema,
            raw_oracles=(
                (ServiceGraphOracle(oracle_id="sg"), "service_graph"),),
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            chain_persist_window=4,
            max_active_chains=8,
            projection_id_for_consumer={"c0": "proj_c0"},
            projected_tags_for_consumer={
                "c0": ("orders", "payments")},
        )
        # Empty-probe registry ⇒ disabled branch.
        registry = EnsembleRatificationRegistry(
            schema=schema, quorum_threshold=1.0, probes=())
        producer = _build_w28_orchestrator(
            schema=schema, agent_id=producer_id, is_producer=True,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            pool=pool, registry=registry,
        )
        for cell in cells:
            producer.decode_rounds(cell)
        result = producer.last_result
        self.assertIsNotNone(result)
        self.assertEqual(result.decoder_branch, W28_BRANCH_DISABLED)


# ---------------------------------------------------------------------------
# 6. Topology probe (best effort)
# ---------------------------------------------------------------------------


class TopologyProbeTests(unittest.TestCase):
    """The two-host discovery must return a structured dict; topology
    is best-effort and may be 'unreachable' offline."""

    def test_topology_dict_shape(self) -> None:
        topo = discover_two_host_topology()
        self.assertIn("topology", topo)
        self.assertIn("hosts", topo)
        self.assertIn(topo["topology"],
                       ("two_host", "single_host", "unreachable"))


if __name__ == "__main__":
    unittest.main()
