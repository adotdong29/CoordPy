"""Tests for SDK v3.26 — shared-fanout dense-control capsule +
cross-agent state reuse (W25 family + Phase 72 driver).

Theorem anchors:

* **W25-1 (efficiency)** — :class:`Phase72SharedFanoutTests`.
  On R-72-FANOUT-SHARED the :class:`SharedFanoutDisambiguator`
  strictly reduces ``mean_total_w25_visible_tokens`` over
  ``mean_total_w24_visible_tokens`` AND records
  ``correctness_ratified_rate = 1.000`` AND
  ``fanout_consumer_resolved_rate = 1.000``.  Stable across 5/5 seeds
  with ≥ 40 tokens/cell mean savings (−69.87% on K=3, 16 cells).

* **W25-Λ-disjoint (named falsifier)** — :class:`Phase72DisjointTests`.
  With no shared registry, W25 reduces to W24 per-agent on every agent.
  ``mean_savings_tokens_per_cell = 0.000`` and
  ``mean_total_w25_visible_tokens == mean_total_w24_visible_tokens``.

* **W25-3 (trust-boundary soundness)** —
  :class:`Phase72PoisonedTests`. An unauthorised consumer_id is
  rejected on every cell: ``fanout_consumer_rejected_rate = 0.333``
  (1 of 3 consumers poisoned), 0 spurious resolutions of the
  poisoned consumer.

* **W25-2 (seed stability)** — :class:`Phase72SeedStabilityTests`.
  W25-1 success criterion holds across all 5 pre-committed seeds
  (11, 17, 23, 29, 31).
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase72_shared_fanout import (
    run_phase72,
    run_phase72_seed_stability_sweep,
    run_cross_regime_p72,
)
from vision_mvp.coordpy.team_coord import (
    FanoutEnvelope,
    SharedFanoutRegistry,
    SharedFanoutDisambiguator,
    verify_fanout,
    W25_FANOUT_SCHEMA_VERSION,
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED,
    W25_BRANCH_NO_TRIGGER,
    W25_BRANCH_DISABLED,
    W25_ALL_BRANCHES,
    LatentVerificationOutcome,
    build_incident_triage_schema_capsule,
)


# =============================================================================
# FanoutEnvelope unit tests
# =============================================================================


class FanoutEnvelopeUnitTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.env = FanoutEnvelope(
            schema_version=W25_FANOUT_SCHEMA_VERSION,
            producer_agent_id="prod",
            consumer_agent_ids=("c0", "c1", "c2"),
            compact_per_tag_votes=(("orders", 3), ("payments", 3)),
            compact_projected_subset=("orders", "payments"),
            schema_cid=str(self.schema.cid),
            cell_index=5,
            n_resolved_in_window=4,
        )

    def test_fanout_cid_computed_at_construction(self) -> None:
        self.assertEqual(len(self.env.fanout_cid), 64)
        self.assertEqual(self.env.fanout_cid,
                          self.env.recompute_fanout_cid())

    def test_canonicalises_unsorted_votes(self) -> None:
        env_a = FanoutEnvelope(
            schema_version=W25_FANOUT_SCHEMA_VERSION,
            producer_agent_id="prod",
            consumer_agent_ids=("c0",),
            compact_per_tag_votes=(("payments", 3), ("orders", 3)),
            compact_projected_subset=("orders", "payments"),
            schema_cid=str(self.schema.cid),
            cell_index=0,
            n_resolved_in_window=0,
        )
        env_b = FanoutEnvelope(
            schema_version=W25_FANOUT_SCHEMA_VERSION,
            producer_agent_id="prod",
            consumer_agent_ids=("c0",),
            compact_per_tag_votes=(("orders", 3), ("payments", 3)),
            compact_projected_subset=("orders", "payments"),
            schema_cid=str(self.schema.cid),
            cell_index=0,
            n_resolved_in_window=0,
        )
        self.assertEqual(env_a.fanout_cid, env_b.fanout_cid)

    def test_consumer_ref_token_format(self) -> None:
        token = self.env.consumer_ref_token("c0")
        self.assertTrue(token.startswith("<fanout_ref:"))
        self.assertTrue(token.endswith(">"))

    def test_n_fanout_ref_tokens_is_one(self) -> None:
        self.assertEqual(self.env.n_fanout_ref_tokens, 1)

    def test_n_fanout_bytes_positive(self) -> None:
        self.assertGreater(self.env.n_fanout_bytes, 0)

    def test_tampered_schema_cid_changes_fanout_cid(self) -> None:
        env_tampered = FanoutEnvelope(
            schema_version=self.env.schema_version,
            producer_agent_id=self.env.producer_agent_id,
            consumer_agent_ids=self.env.consumer_agent_ids,
            compact_per_tag_votes=self.env.compact_per_tag_votes,
            compact_projected_subset=self.env.compact_projected_subset,
            schema_cid="TAMPERED" + "0" * 56,
            cell_index=self.env.cell_index,
            n_resolved_in_window=self.env.n_resolved_in_window,
        )
        self.assertNotEqual(env_tampered.fanout_cid, self.env.fanout_cid)


# =============================================================================
# verify_fanout unit tests
# =============================================================================


class VerifyFanoutUnitTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.env = FanoutEnvelope(
            schema_version=W25_FANOUT_SCHEMA_VERSION,
            producer_agent_id="prod",
            consumer_agent_ids=("c0", "c1"),
            compact_per_tag_votes=(("orders", 2),),
            compact_projected_subset=("orders",),
            schema_cid=str(self.schema.cid),
            cell_index=0,
            n_resolved_in_window=0,
        )

    def test_valid_consumer_verifies_ok(self) -> None:
        outcome = verify_fanout(
            self.env,
            registered_schema=self.schema,
            consumer_id="c0")
        self.assertTrue(outcome.ok)

    def test_unauthorized_consumer_rejected(self) -> None:
        outcome = verify_fanout(
            self.env,
            registered_schema=self.schema,
            consumer_id="UNAUTHORIZED")
        self.assertFalse(outcome.ok)
        self.assertIn("not_authorized", outcome.reason)

    def test_wrong_schema_cid_rejected(self) -> None:
        env_wrong = FanoutEnvelope(
            schema_version=self.env.schema_version,
            producer_agent_id=self.env.producer_agent_id,
            consumer_agent_ids=self.env.consumer_agent_ids,
            compact_per_tag_votes=self.env.compact_per_tag_votes,
            compact_projected_subset=self.env.compact_projected_subset,
            schema_cid="WRONG" + "0" * 59,
            cell_index=self.env.cell_index,
            n_resolved_in_window=self.env.n_resolved_in_window,
        )
        outcome = verify_fanout(
            env_wrong,
            registered_schema=self.schema,
            consumer_id="c0")
        self.assertFalse(outcome.ok)
        self.assertIn("schema_cid_mismatch", outcome.reason)

    def test_unknown_schema_version_rejected(self) -> None:
        env_bad = FanoutEnvelope(
            schema_version="coordpy.shared_fanout.UNKNOWN",
            producer_agent_id=self.env.producer_agent_id,
            consumer_agent_ids=self.env.consumer_agent_ids,
            compact_per_tag_votes=self.env.compact_per_tag_votes,
            compact_projected_subset=self.env.compact_projected_subset,
            schema_cid=str(self.schema.cid),
            cell_index=self.env.cell_index,
            n_resolved_in_window=self.env.n_resolved_in_window,
        )
        outcome = verify_fanout(
            env_bad,
            registered_schema=self.schema,
            consumer_id="c0")
        self.assertFalse(outcome.ok)
        self.assertIn("schema_version_unknown", outcome.reason)

    def test_hash_mismatch_rejected(self) -> None:
        # Manually construct an envelope with wrong fanout_cid
        import dataclasses
        env_tampered = dataclasses.replace(
            self.env, fanout_cid="TAMPERED" + "0" * 56)
        outcome = verify_fanout(
            env_tampered,
            registered_schema=self.schema,
            consumer_id="c0")
        self.assertFalse(outcome.ok)
        self.assertIn("hash_mismatch", outcome.reason)


# =============================================================================
# SharedFanoutRegistry unit tests
# =============================================================================


class SharedFanoutRegistryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.registry = SharedFanoutRegistry(schema=self.schema)
        self.env = FanoutEnvelope(
            schema_version=W25_FANOUT_SCHEMA_VERSION,
            producer_agent_id="prod",
            consumer_agent_ids=("c0", "c1"),
            compact_per_tag_votes=(("orders", 2),),
            compact_projected_subset=("orders",),
            schema_cid=str(self.schema.cid),
            cell_index=0,
            n_resolved_in_window=0,
        )

    def test_register_and_get_by_producer(self) -> None:
        self.registry.register(self.env)
        retrieved = self.registry.get_by_producer("prod", 0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.fanout_cid, self.env.fanout_cid)

    def test_get_missing_returns_none(self) -> None:
        result = self.registry.get_by_producer("prod", 99)
        self.assertIsNone(result)

    def test_resolve_valid_consumer(self) -> None:
        self.registry.register(self.env)
        env_out, reason = self.registry.resolve(
            self.env.fanout_cid, "c0")
        self.assertIsNotNone(env_out)
        self.assertEqual(reason, "ok")

    def test_resolve_invalid_consumer_rejected(self) -> None:
        self.registry.register(self.env)
        env_out, reason = self.registry.resolve(
            self.env.fanout_cid, "UNAUTHORIZED")
        self.assertIsNone(env_out)
        self.assertNotEqual(reason, "ok")

    def test_counters_increment(self) -> None:
        self.registry.register(self.env)
        self.assertEqual(self.registry.n_registered, 1)
        self.registry.resolve(self.env.fanout_cid, "c0")
        self.assertEqual(self.registry.n_resolved, 1)


# =============================================================================
# W25 branch vocabulary
# =============================================================================


class W25BranchVocabularyTests(unittest.TestCase):

    def test_all_branches_defined(self) -> None:
        expected = {
            W25_BRANCH_FANOUT_PRODUCER_EMITTED,
            W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
            W25_BRANCH_FANOUT_CONSUMER_REJECTED,
            W25_BRANCH_NO_TRIGGER,
            W25_BRANCH_DISABLED,
        }
        self.assertEqual(set(W25_ALL_BRANCHES), expected)


# =============================================================================
# Phase 72 — W25-1 efficiency anchor
# =============================================================================


class Phase72SharedFanoutTests(unittest.TestCase):
    """W25-1: shared fanout strictly reduces total visible tokens."""

    def test_w25_saves_tokens_over_w24_loose(self) -> None:
        result = run_phase72(bank="fanout_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertLess(result["mean_total_w25_visible_tokens"],
                         result["mean_total_w24_visible_tokens"],
                         "W25 must be strictly cheaper than W24 total")
        self.assertGreaterEqual(result["mean_savings_tokens_per_cell"],
                                  40.0,
                                  "mean savings ≥ 40 tokens/cell on K=3")
        self.assertEqual(result["correctness_ratified_rate"], 1.0,
                          "producer correctness must be 1.000")
        self.assertEqual(result["fanout_consumer_resolved_rate"], 1.0,
                          "all consumers must resolve on shared bank")

    def test_w25_savings_pct_exceeds_60pct(self) -> None:
        result = run_phase72(bank="fanout_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertGreater(result["savings_pct"], 60.0,
                            "W25 savings must exceed 60% on K=3")

    def test_fanout_bytes_positive(self) -> None:
        result = run_phase72(bank="fanout_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertGreater(result["n_fanout_bytes_total"], 0,
                            "fanout envelopes must have positive bytes")

    def test_registry_registered_equals_n_cells(self) -> None:
        result = run_phase72(bank="fanout_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["registry_n_registered"],
                          result["n_cells"],
                          "producer must register exactly n_cells envelopes")


# =============================================================================
# Phase 72 — W25-Λ-disjoint named falsifier
# =============================================================================


class Phase72DisjointTests(unittest.TestCase):
    """W25-Λ-disjoint: without shared registry, W25 = W24."""

    def test_w25_equals_w24_on_disjoint(self) -> None:
        result = run_phase72(bank="disjoint", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["mean_total_w25_visible_tokens"],
                          result["mean_total_w24_visible_tokens"],
                          "disjoint: W25 must equal W24 total")
        self.assertEqual(result["mean_savings_tokens_per_cell"], 0.0,
                          "disjoint: no savings")

    def test_disjoint_no_fanout_fired(self) -> None:
        result = run_phase72(bank="disjoint", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["fanout_consumer_resolved_rate"], 0.0)
        self.assertEqual(result["fanout_consumer_rejected_rate"], 0.0)
        self.assertEqual(result["registry_n_registered"], 0)


# =============================================================================
# Phase 72 — W25-3 trust-boundary soundness
# =============================================================================


class Phase72PoisonedTests(unittest.TestCase):
    """W25-3: unauthorised consumer_id is rejected on every cell."""

    def test_poisoned_consumer_rejected(self) -> None:
        result = run_phase72(bank="fanout_poisoned", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # 1 of 3 consumers is poisoned → rejected_rate ≈ 1/3
        self.assertAlmostEqual(result["fanout_consumer_rejected_rate"],
                                  1.0 / 3, places=2,
                                  msg="exactly 1/3 consumers must be rejected")
        self.assertEqual(result["n_consumer_rejected"],
                          result["n_cells"],
                          "poisoned consumer rejected on every cell")

    def test_correct_consumers_still_resolve(self) -> None:
        result = run_phase72(bank="fanout_poisoned", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # 2 of 3 consumers are authorised → resolved_rate ≈ 2/3
        self.assertAlmostEqual(result["fanout_consumer_resolved_rate"],
                                  2.0 / 3, places=2,
                                  msg="authorised consumers must still resolve")

    def test_correctness_preserved_despite_poison(self) -> None:
        result = run_phase72(bank="fanout_poisoned", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["correctness_ratified_rate"], 1.0,
                          "producer correctness must be 1.000 even with poisoned consumer")


# =============================================================================
# Phase 72 — W25-2 seed stability
# =============================================================================


class Phase72SeedStabilityTests(unittest.TestCase):
    """W25-2: W25-1 success criterion holds across all 5 seeds."""

    def test_seed_stability_5_of_5(self) -> None:
        sweep = run_phase72_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            bank="fanout_shared", T_decoder=None, K_consumers=3)
        self.assertTrue(sweep["all_savings_positive"],
                         "all seeds must show positive savings")
        self.assertTrue(sweep["all_correctness_1000"],
                         "all seeds must show correctness = 1.000")
        self.assertGreaterEqual(sweep["min_savings"], 40.0,
                                  "min savings ≥ 40 tokens/cell across all seeds")

    def test_seed_stability_correctness_invariant(self) -> None:
        sweep = run_phase72_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            bank="fanout_shared", T_decoder=None, K_consumers=3)
        self.assertEqual(sweep["min_correctness"], 1.0,
                          "min correctness must be 1.000 across all seeds")


# =============================================================================
# Phase 72 — cross-regime evaluation
# =============================================================================


class Phase72CrossRegimeTests(unittest.TestCase):
    """Cross-regime: all 3 banks × 2 T_decoder settings."""

    def test_cross_regime_structure(self) -> None:
        results = run_cross_regime_p72(n_eval=8, K_consumers=3,
                                        bank_seed=11, verbose=False)
        # Keys are f"{bank}_T{T_decoder}" e.g. "fanout_shared_TNone"
        self.assertTrue(any(k.startswith("fanout_shared") for k in results))
        self.assertTrue(any(k.startswith("disjoint") for k in results))
        self.assertTrue(any(k.startswith("fanout_poisoned") for k in results))

    def test_cross_regime_fanout_shared_saves_on_both_decoders(self) -> None:
        results = run_cross_regime_p72(n_eval=8, K_consumers=3,
                                        bank_seed=11, verbose=False)
        for key, row in results.items():
            if key.startswith("fanout_shared"):
                self.assertGreater(row["mean_savings_tokens_per_cell"], 0.0,
                                    f"fanout_shared should save tokens: {key}")

    def test_cross_regime_disjoint_zero_savings(self) -> None:
        results = run_cross_regime_p72(n_eval=8, K_consumers=3,
                                        bank_seed=11, verbose=False)
        for key, row in results.items():
            if key.startswith("disjoint"):
                self.assertEqual(row["mean_savings_tokens_per_cell"], 0.0,
                                   f"disjoint must have zero savings: {key}")


if __name__ == "__main__":
    unittest.main()
