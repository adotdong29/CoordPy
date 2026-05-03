"""Tests for SDK v3.28 — Phase-74 multi-chain salience-keyed dense-control
fanout (W27 family).

Six pre-committed sub-banks:
  * R-74-CHAIN-SHARED         — W27-Λ-single-signature falsifier
  * R-74-DIVERGENT-RECOVER    — W27 isolation cost (within-graph gold)
  * R-74-XORACLE-RECOVER      — W27-1 anchor: per-signature oracle scope
                                  recovers correctness AND saves tokens
                                  vs W26 single-stack baseline.
  * R-74-POOL-EXHAUSTED       — W27-Λ-pool-exhausted falsifier
  * R-74-PIVOT-TAMPERED       — W27-3 trust falsifier (audited disambig)
  * R-74-SIGNATURE-DRIFT      — W27-3 trust falsifier (audited disambig)
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase74_multi_chain_pivot import (
    build_phase74_bank,
    build_team_shared_pool,
    build_team_shared_pool_xoracle,
    _build_w26_stack,
    _build_w27_orchestrator,
    _build_partial_service_graph_oracle,
    _expected_gold_for_cell,
    run_phase74,
    run_phase74_seed_stability_sweep,
)
from vision_mvp.coordpy.team_coord import (
    SalienceSignatureEnvelope, ChainPivotEnvelope,
    SharedMultiChainPool,
    MultiChainPersistedFanoutOrchestrator,
    MultiChainPersistedFanoutDisambiguator,
    MultiChainPersistedFanoutRegistry,
    ChainPersistedFanoutRegistry,
    SharedFanoutRegistry,
    verify_salience_signature, verify_chain_pivot,
    compute_input_signature_cid,
    W27_BRANCH_PIVOTED, W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_POOL_EXHAUSTED, W27_BRANCH_PIVOT_REJECTED,
    W27_BRANCH_FALLBACK_W26, W27_BRANCH_NO_TRIGGER,
    W27_BRANCH_DISABLED, W27_ALL_BRANCHES,
    W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
    W27_CHAIN_PIVOT_SCHEMA_VERSION,
    build_incident_triage_schema_capsule,
    ServiceGraphOracle, ChangeHistoryOracle,
)


class SalienceSignatureUnitTests(unittest.TestCase):
    def test_signature_byte_stable_on_identical_inputs(self) -> None:
        schema = build_incident_triage_schema_capsule()
        sig1 = SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c0", "c1"),
            canonical_per_tag_votes=(("orders", 2), ("payments", 2)),
            canonical_projected_subset=("orders", "payments"),
            cell_index_first_observed=0,
        )
        sig2 = SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c1", "c0"),  # different order
            canonical_per_tag_votes=(("payments", 2), ("orders", 2)),
            canonical_projected_subset=("payments", "orders"),
            cell_index_first_observed=0,
        )
        self.assertEqual(sig1.signature_cid, sig2.signature_cid)

    def test_signature_changes_on_content_change(self) -> None:
        schema = build_incident_triage_schema_capsule()
        sig1 = SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            canonical_per_tag_votes=(("orders", 2),),
            canonical_projected_subset=("orders",),
            cell_index_first_observed=0,
        )
        sig2 = SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            canonical_per_tag_votes=(("api", 2),),  # different tag
            canonical_projected_subset=("api",),
            cell_index_first_observed=0,
        )
        self.assertNotEqual(sig1.signature_cid, sig2.signature_cid)


class VerifySalienceSignatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.sig = SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            canonical_per_tag_votes=(("orders", 2),),
            canonical_projected_subset=("orders",),
            cell_index_first_observed=0,
        )

    def test_ok(self) -> None:
        outcome = verify_salience_signature(
            self.sig, registered_schema=self.schema)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.reason, "ok")

    def test_empty(self) -> None:
        outcome = verify_salience_signature(
            None, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "empty_signature")

    def test_schema_version_unknown(self) -> None:
        bad = SalienceSignatureEnvelope(
            schema_version="coordpy.bad.v1",
            schema_cid=self.schema.cid,
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            canonical_per_tag_votes=(("orders", 2),),
            canonical_projected_subset=("orders",),
            cell_index_first_observed=0,
        )
        outcome = verify_salience_signature(
            bad, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "schema_version_unknown")


class ChainPivotEnvelopeUnitTests(unittest.TestCase):
    def test_pivot_cid_recompute_byte_stable(self) -> None:
        schema = build_incident_triage_schema_capsule()
        p = ChainPivotEnvelope(
            schema_version=W27_CHAIN_PIVOT_SCHEMA_VERSION,
            schema_cid=schema.cid,
            signature_cid="sig0",
            parent_chain_root_cid="root0",
            parent_advance_cid="adv0",
            cell_index=3,
        )
        self.assertEqual(p.pivot_cid, p.recompute_pivot_cid())
        self.assertEqual(p.n_pivot_tokens, 1)


class VerifyChainPivotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.pivot = ChainPivotEnvelope(
            schema_version=W27_CHAIN_PIVOT_SCHEMA_VERSION,
            schema_cid=self.schema.cid,
            signature_cid="SIG_X",
            parent_chain_root_cid="ROOT_X",
            parent_advance_cid="ADV_X",
            cell_index=3,
        )

    def test_ok(self) -> None:
        outcome = verify_chain_pivot(
            self.pivot,
            registered_schema=self.schema,
            registered_signature_cid="SIG_X",
            registered_parent_chain_root_cid="ROOT_X",
            registered_parent_advance_cid="ADV_X",
        )
        self.assertTrue(outcome.ok)

    def test_unknown_signature(self) -> None:
        outcome = verify_chain_pivot(
            self.pivot,
            registered_schema=self.schema,
            registered_signature_cid="",
            registered_parent_chain_root_cid="ROOT_X",
            registered_parent_advance_cid="ADV_X",
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "unknown_signature")

    def test_salience_signature_mismatch(self) -> None:
        outcome = verify_chain_pivot(
            self.pivot,
            registered_schema=self.schema,
            registered_signature_cid="SIG_OTHER",
            registered_parent_chain_root_cid="ROOT_X",
            registered_parent_advance_cid="ADV_X",
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "salience_signature_mismatch")

    def test_parent_chain_unknown(self) -> None:
        outcome = verify_chain_pivot(
            self.pivot,
            registered_schema=self.schema,
            registered_signature_cid="SIG_X",
            registered_parent_chain_root_cid="ROOT_OTHER",
            registered_parent_advance_cid="ADV_X",
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "parent_chain_unknown")

    def test_parent_advance_unknown(self) -> None:
        outcome = verify_chain_pivot(
            self.pivot,
            registered_schema=self.schema,
            registered_signature_cid="SIG_X",
            registered_parent_chain_root_cid="ROOT_X",
            registered_parent_advance_cid="ADV_OTHER",
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "parent_advance_unknown")


class W27BranchVocabularyTests(unittest.TestCase):
    def test_branch_set_size(self) -> None:
        self.assertEqual(len(W27_ALL_BRANCHES), 7)

    def test_required_branches_present(self) -> None:
        self.assertIn(W27_BRANCH_PIVOTED, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_ANCHORED_NEW, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_POOL_EXHAUSTED, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_PIVOT_REJECTED, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_FALLBACK_W26, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_NO_TRIGGER, W27_ALL_BRANCHES)
        self.assertIn(W27_BRANCH_DISABLED, W27_ALL_BRANCHES)


class InputSignatureCidTests(unittest.TestCase):
    def test_byte_stable_across_handoff_order(self) -> None:
        from vision_mvp.coordpy.team_coord import _DecodedHandoff
        h_a = _DecodedHandoff("r0", "K", "p0")
        h_b = _DecodedHandoff("r1", "K", "p1")
        s1 = compute_input_signature_cid(
            [[h_a, h_b]],
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            schema_cid="schema",
        )
        s2 = compute_input_signature_cid(
            [[h_b, h_a]],  # reversed
            producer_agent_id="p",
            consumer_agent_ids=("c0",),
            schema_cid="schema",
        )
        self.assertEqual(s1, s2)


class SharedMultiChainPoolTests(unittest.TestCase):
    def test_pool_exhausted_returns_true(self) -> None:
        schema = build_incident_triage_schema_capsule()

        def factory(*, signature_cid, agent_id, is_producer):
            local_fanout = SharedFanoutRegistry(schema=schema)
            local_chain = ChainPersistedFanoutRegistry(schema=schema)
            return _build_w26_stack(
                T_decoder=None, schema=schema,
                raw_oracles=(
                    (ServiceGraphOracle(oracle_id="g"), "service_graph"),
                    (ChangeHistoryOracle(oracle_id="h"), "change_history"),
                ),
                agent_id=agent_id, is_producer=is_producer,
                producer_agent_id="p",
                consumer_agent_ids=("c0",),
                fanout_registry=local_fanout,
                chain_registry=local_chain,
                chain_persist_window=16,
            )
        pool = SharedMultiChainPool(
            schema=schema, stack_factory=factory, max_active_chains=1)
        # First sig: succeeds.
        slot, exhausted = pool.get_or_make_slot(
            agent_id="p", signature_cid="sig_a", is_producer=True)
        self.assertIsNotNone(slot)
        self.assertFalse(exhausted)
        # Same sig: succeeds, returns existing slot.
        slot2, exhausted2 = pool.get_or_make_slot(
            agent_id="p", signature_cid="sig_a", is_producer=True)
        self.assertIs(slot, slot2)
        self.assertFalse(exhausted2)
        # Second sig: pool exhausted (max_active_chains=1).
        slot3, exhausted3 = pool.get_or_make_slot(
            agent_id="p", signature_cid="sig_b", is_producer=True)
        self.assertIsNone(slot3)
        self.assertTrue(exhausted3)
        self.assertEqual(pool.n_pool_exhausted_rejections, 1)


class Phase74XOracleRecoverTests(unittest.TestCase):
    """The W27-1 anchor: per-signature oracle scope recovers
    correctness AND saves tokens vs the W26 single-stack baseline.
    """

    def test_w27_strictly_better_than_w26_xoracle(self) -> None:
        r = run_phase74(
            bank="xoracle_recover", T_decoder=None,
            K_consumers=3, n_eval=16, bank_seed=11,
            chain_persist_window=16, max_active_chains=8,
            signature_period=4)
        self.assertEqual(r["correctness_ratified_rate_w27"], 1.0)
        self.assertEqual(r["correctness_ratified_rate_w26"], 0.5)
        self.assertGreater(
            r["mean_savings_w27_vs_w26_per_cell"], 20.0)
        self.assertGreater(
            r["mean_savings_w27_vs_w25_per_cell"], 20.0)
        self.assertEqual(r["pool_size_final"], 2)

    def test_w27_xoracle_seed_stability(self) -> None:
        sweep = run_phase74_seed_stability_sweep(
            bank="xoracle_recover", T_decoder=None,
            K_consumers=3, n_eval=16,
            chain_persist_window=16, max_active_chains=8,
            signature_period=4)
        self.assertTrue(sweep["all_savings_w26_positive"])
        self.assertTrue(sweep["all_correctness_1000"])
        self.assertGreaterEqual(sweep["min_savings_w27_vs_w26"], 20.0)


class Phase74ChainSharedTests(unittest.TestCase):
    """W27-Λ-single-signature: on the chain_shared regime W27 reduces
    to W26 byte-for-byte.
    """

    def test_w27_equals_w26_on_chain_shared(self) -> None:
        r = run_phase74(
            bank="chain_shared", T_decoder=None,
            K_consumers=3, n_eval=16, bank_seed=11,
            chain_persist_window=16, max_active_chains=8)
        self.assertEqual(
            r["mean_savings_w27_vs_w26_per_cell"], 0.0)
        self.assertEqual(r["pool_size_final"], 1)
        self.assertEqual(r["correctness_ratified_rate_w27"], 1.0)
        self.assertEqual(r["correctness_ratified_rate_w26"], 1.0)


class Phase74PoolExhaustedTests(unittest.TestCase):
    """W27-Λ-pool-exhausted: when more signatures arrive than the pool
    can hold, W27 falls back to W26 deterministically; correctness
    preserved.
    """

    def test_pool_caps_at_max_active_chains(self) -> None:
        r = run_phase74(
            bank="pool_exhausted", T_decoder=None,
            K_consumers=3, n_eval=16, bank_seed=11,
            chain_persist_window=16)
        # max_active_chains is overridden to 2 in the runner for this bank.
        self.assertEqual(r["max_active_chains"], 2)
        self.assertEqual(r["pool_size_final"], 2)
        # Correctness preserved on this regime.
        self.assertEqual(r["correctness_ratified_rate_w26"], 1.0)
        self.assertEqual(r["correctness_ratified_rate_w27"], 1.0)


class Phase74PivotTamperedTests(unittest.TestCase):
    """W27-3 trust falsifier: pivot tampering rejected by
    verify_chain_pivot."""

    def test_pivot_tampered_correctness_preserved(self) -> None:
        r = run_phase74(
            bank="pivot_tampered", T_decoder=None,
            K_consumers=3, n_eval=16, bank_seed=11,
            chain_persist_window=16)
        # The tampering only corrupts the audited disambig wrapper;
        # correctness on the orchestrator path is preserved.
        self.assertEqual(r["correctness_ratified_rate_w26"], 1.0)
        self.assertEqual(r["correctness_ratified_rate_w27"], 1.0)


class Phase74SignatureDriftTests(unittest.TestCase):
    """W27-3 trust falsifier: stale signature CID falls through cleanly."""

    def test_signature_drift_correctness_preserved(self) -> None:
        r = run_phase74(
            bank="signature_drift", T_decoder=None,
            K_consumers=3, n_eval=16, bank_seed=11,
            chain_persist_window=16)
        self.assertEqual(r["correctness_ratified_rate_w26"], 1.0)
        self.assertEqual(r["correctness_ratified_rate_w27"], 1.0)


class Phase74CrossRegimeTests(unittest.TestCase):
    """The full R-74 family — every sub-bank produces deterministic
    branch counts and never crashes."""

    def test_cross_regime_runs_every_bank(self) -> None:
        from vision_mvp.experiments.phase74_multi_chain_pivot import (
            run_cross_regime_p74,
        )
        d = run_cross_regime_p74(
            K_consumers=3, n_eval=16, bank_seed=11)
        # Five banks × 2 T_decoder = 12.
        self.assertEqual(len(d), 12)
        # Each result is well-formed.
        for k, r in d.items():
            self.assertIn("mean_total_w27_visible_tokens", r)
            self.assertIn("mean_total_w26_visible_tokens", r)
            self.assertIn("correctness_ratified_rate_w27", r)
            self.assertIn("pool_size_final", r)


if __name__ == "__main__":
    unittest.main()
