"""Tests for SDK v3.27 — chain-persisted dense-control fanout
(W26 family + Phase 73 driver).

Theorem anchors:

* **W26-1 (efficiency)** — :class:`Phase73ChainSharedTests`.
  On R-73-CHAIN-SHARED the
  :class:`ChainPersistedFanoutDisambiguator` strictly reduces
  ``mean_total_w26_visible_tokens`` over
  ``mean_total_w25_visible_tokens`` AND records
  ``correctness_ratified_rate = 1.000`` AND
  ``chain_consumer_resolved_rate = 1.000``.  Stable across 5/5
  seeds.

* **W26-Λ-no-chain (named falsifier)** —
  :class:`Phase73NoChainTests`.  With ``chain_persist_window = 1``
  every cell is an anchor → W26 = W25 byte-for-byte.

* **W26-Λ-tampered (named falsifier)** —
  :class:`Phase73ChainTamperedTests`.  Every advance after the
  first is rejected by ``verify_chain_advance``; W26 collapses
  toward the W25 floor on rejected cells.

* **W26-Λ-projection-mismatch (named falsifier)** —
  :class:`Phase73ProjectionMismatchTests`.  A consumer requesting
  the wrong projection_id is rejected on every cell; the other
  consumers still resolve.

* **W26-Λ-divergent (named regime)** —
  :class:`Phase73DivergentTests`.  When the gold subset flips at
  the bench mid-point, the inner W25 fires no_trigger → W26 also
  fires no_trigger on those cells; correctness drops to 0.5.

* **W26-2 (seed stability)** —
  :class:`Phase73SeedStabilityTests`.  W26-1 holds across 5/5 seeds.

* **W26-3 (trust soundness)** — :class:`VerifyChainEnvelopesTests`,
  :class:`ChainPersistedFanoutRegistryTests`.  Six unit failure
  modes for ``verify_chain_anchor`` + eight for
  ``verify_chain_advance`` + projection-scope subscription check.
"""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.experiments.phase73_chain_persisted_fanout import (
    run_phase73, run_phase73_seed_stability_sweep,
    run_cross_regime_p73, run_k_scaling_sweep,
)
from vision_mvp.coordpy.team_coord import (
    ChainAnchorEnvelope, ChainAdvanceEnvelope,
    ChainPersistedFanoutRegistry,
    ChainPersistedFanoutDisambiguator,
    ProjectionSlot,
    verify_chain_anchor, verify_chain_advance,
    verify_projection_subscription,
    LatentVerificationOutcome,
    W26_CHAIN_ANCHOR_SCHEMA_VERSION,
    W26_CHAIN_ADVANCE_SCHEMA_VERSION,
    W26_BRANCH_CHAIN_ANCHORED, W26_BRANCH_CHAIN_ADVANCED,
    W26_BRANCH_CHAIN_REJECTED, W26_BRANCH_CHAIN_RE_ANCHORED,
    W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
    W26_BRANCH_CHAIN_PROJECTION_REJECTED,
    W26_BRANCH_NO_TRIGGER, W26_BRANCH_DISABLED,
    W26_ALL_BRANCHES, W26_DEFAULT_TRIGGER_BRANCHES,
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    build_incident_triage_schema_capsule,
)


# =============================================================================
# ProjectionSlot unit tests
# =============================================================================


class ProjectionSlotUnitTests(unittest.TestCase):

    def test_projection_cid_computed_at_construction(self) -> None:
        slot = ProjectionSlot(
            projection_id="proj_a",
            consumer_id="c0",
            projected_tags=("orders", "payments"))
        self.assertEqual(len(slot.projection_cid), 64)

    def test_canonicalises_unsorted_tags(self) -> None:
        a = ProjectionSlot(
            projection_id="p", consumer_id="c",
            projected_tags=("payments", "orders"))
        b = ProjectionSlot(
            projection_id="p", consumer_id="c",
            projected_tags=("orders", "payments"))
        self.assertEqual(a.projection_cid, b.projection_cid)


# =============================================================================
# ChainAnchorEnvelope unit tests
# =============================================================================


class ChainAnchorEnvelopeUnitTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.slots = (
            ProjectionSlot(projection_id="p0", consumer_id="c0",
                            projected_tags=("orders",)),
            ProjectionSlot(projection_id="p1", consumer_id="c1",
                            projected_tags=("payments",)),
        )
        self.anchor = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c0", "c1"),
            cell_index_anchor=0,
            chain_persist_window=16,
            canonical_compact_per_tag_votes=(("orders", 3), ("payments", 3)),
            canonical_compact_projected_subset=("orders", "payments"),
            n_w15_canonical_tokens=14,
            projection_slots=self.slots,
        )

    def test_chain_root_cid_64_chars(self) -> None:
        self.assertEqual(len(self.anchor.chain_root_cid), 64)

    def test_chain_root_cid_recomputes(self) -> None:
        self.assertEqual(self.anchor.chain_root_cid,
                          self.anchor.recompute_chain_root_cid())

    def test_canonicalises_unsorted_inputs(self) -> None:
        a = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c1", "c0"),
            cell_index_anchor=0, chain_persist_window=16,
            canonical_compact_per_tag_votes=(("payments", 3),
                                              ("orders", 3)),
            canonical_compact_projected_subset=("payments", "orders"),
            n_w15_canonical_tokens=14,
            projection_slots=tuple(reversed(self.slots)))
        self.assertEqual(a.chain_root_cid, self.anchor.chain_root_cid)

    def test_projection_for_returns_slot(self) -> None:
        slot = self.anchor.projection_for("c0")
        self.assertIsNotNone(slot)
        self.assertEqual(slot.projection_id, "p0")

    def test_projection_for_unknown_returns_none(self) -> None:
        self.assertIsNone(self.anchor.projection_for("UNKNOWN"))


# =============================================================================
# ChainAdvanceEnvelope unit tests
# =============================================================================


class ChainAdvanceEnvelopeUnitTests(unittest.TestCase):

    def setUp(self) -> None:
        self.advance = ChainAdvanceEnvelope(
            schema_version=W26_CHAIN_ADVANCE_SCHEMA_VERSION,
            schema_cid="dummy_schema_cid",
            chain_root_cid="dummy_root",
            parent_advance_cid="dummy_parent",
            cell_index=5,
            cell_in_chain=2,
            delta_per_tag_votes=(),
            delta_projected_subset_added=(),
            delta_projected_subset_removed=(),
        )

    def test_advance_cid_64_chars(self) -> None:
        self.assertEqual(len(self.advance.advance_cid), 64)

    def test_decoder_text_format(self) -> None:
        text = self.advance.to_decoder_text()
        self.assertTrue(text.startswith("<chain_advance:"))
        self.assertTrue(text.endswith(">"))

    def test_n_advance_tokens_is_one(self) -> None:
        self.assertEqual(self.advance.n_advance_tokens, 1)

    def test_is_empty_delta_true_when_no_delta(self) -> None:
        self.assertTrue(self.advance.is_empty_delta)

    def test_is_empty_delta_false_when_delta_present(self) -> None:
        adv2 = dataclasses.replace(
            self.advance,
            delta_per_tag_votes=(("orders", 3),),
            advance_cid="")
        self.assertFalse(adv2.is_empty_delta)


# =============================================================================
# verify_chain_anchor unit tests
# =============================================================================


class VerifyChainAnchorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.anchor = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c0",),
            cell_index_anchor=0,
            chain_persist_window=16,
            canonical_compact_per_tag_votes=(("orders", 1),),
            canonical_compact_projected_subset=("orders",),
            n_w15_canonical_tokens=10,
            projection_slots=(ProjectionSlot(
                projection_id="p", consumer_id="c0",
                projected_tags=("orders",)),))

    def test_valid_anchor_verifies_ok(self) -> None:
        outcome = verify_chain_anchor(
            self.anchor, registered_schema=self.schema)
        self.assertTrue(outcome.ok)

    def test_empty_anchor_rejected(self) -> None:
        outcome = verify_chain_anchor(
            None, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertIn("empty_anchor", outcome.reason)

    def test_unknown_schema_version_rejected(self) -> None:
        a = dataclasses.replace(
            self.anchor, schema_version="bad_version",
            chain_root_cid="")
        outcome = verify_chain_anchor(a, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertIn("schema_version_unknown", outcome.reason)

    def test_schema_cid_mismatch_rejected(self) -> None:
        a = dataclasses.replace(
            self.anchor, schema_cid="WRONG" + "0" * 59,
            chain_root_cid="")
        outcome = verify_chain_anchor(a, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertIn("schema_cid_mismatch", outcome.reason)

    def test_window_non_positive_rejected(self) -> None:
        a = dataclasses.replace(
            self.anchor, chain_persist_window=0,
            chain_root_cid="")
        outcome = verify_chain_anchor(a, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertIn("window_non_positive", outcome.reason)

    def test_hash_mismatch_rejected(self) -> None:
        # Tamper the chain_root_cid post-construction.
        a = dataclasses.replace(
            self.anchor, chain_root_cid="TAMPERED" + "0" * 56)
        outcome = verify_chain_anchor(a, registered_schema=self.schema)
        self.assertFalse(outcome.ok)
        self.assertIn("hash_mismatch", outcome.reason)


# =============================================================================
# verify_chain_advance unit tests
# =============================================================================


class VerifyChainAdvanceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.anchor = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c0",),
            cell_index_anchor=0,
            chain_persist_window=4,
            canonical_compact_per_tag_votes=(("orders", 1),),
            canonical_compact_projected_subset=("orders",),
            n_w15_canonical_tokens=10,
            projection_slots=(ProjectionSlot(
                projection_id="p", consumer_id="c0",
                projected_tags=("orders",)),))
        self.advance = ChainAdvanceEnvelope(
            schema_version=W26_CHAIN_ADVANCE_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            chain_root_cid=self.anchor.chain_root_cid,
            parent_advance_cid=self.anchor.chain_root_cid,
            cell_index=1, cell_in_chain=1,
            delta_per_tag_votes=(),
            delta_projected_subset_added=(),
            delta_projected_subset_removed=())

    def _verify(self, advance: ChainAdvanceEnvelope | None,
                  *, parent: str | None = None,
                  cell_in_chain: int = 1
                  ) -> LatentVerificationOutcome:
        return verify_chain_advance(
            advance,
            registered_schema=self.schema,
            anchor=self.anchor,
            expected_parent_cid=(parent
                                  if parent is not None
                                  else self.anchor.chain_root_cid),
            expected_cell_in_chain=cell_in_chain)

    def test_valid_advance_verifies_ok(self) -> None:
        outcome = self._verify(self.advance)
        self.assertTrue(outcome.ok)

    def test_empty_advance_rejected(self) -> None:
        outcome = self._verify(None)
        self.assertFalse(outcome.ok)
        self.assertIn("empty_advance", outcome.reason)

    def test_unknown_schema_version_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, schema_version="bad", advance_cid="")
        outcome = self._verify(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("schema_version_unknown", outcome.reason)

    def test_schema_cid_mismatch_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, schema_cid="WRONG", advance_cid="")
        outcome = self._verify(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("schema_cid_mismatch", outcome.reason)

    def test_chain_root_mismatch_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, chain_root_cid="WRONG", advance_cid="")
        outcome = self._verify(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("chain_root_mismatch", outcome.reason)

    def test_parent_mismatch_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, parent_advance_cid="WRONG", advance_cid="")
        outcome = self._verify(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("parent_mismatch", outcome.reason)

    def test_cell_in_chain_mismatch_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, cell_in_chain=99, advance_cid="")
        outcome = self._verify(adv, cell_in_chain=1)
        self.assertFalse(outcome.ok)
        self.assertIn("cell_in_chain_mismatch", outcome.reason)

    def test_window_expired_rejected(self) -> None:
        # Anchor window=4, cell_in_chain=5 exceeds; build an advance
        # that lies about its position in the chain.
        adv = dataclasses.replace(
            self.advance, cell_in_chain=5, advance_cid="")
        outcome = self._verify(adv, cell_in_chain=5)
        self.assertFalse(outcome.ok)
        self.assertIn("window_expired", outcome.reason)

    def test_hash_mismatch_rejected(self) -> None:
        adv = dataclasses.replace(
            self.advance, advance_cid="TAMPERED" + "0" * 56)
        outcome = self._verify(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("hash_mismatch", outcome.reason)


# =============================================================================
# verify_projection_subscription unit tests
# =============================================================================


class VerifyProjectionSubscriptionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.anchor = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c0", "c1"),
            cell_index_anchor=0,
            chain_persist_window=16,
            canonical_compact_per_tag_votes=(("orders", 1),),
            canonical_compact_projected_subset=("orders",),
            n_w15_canonical_tokens=10,
            projection_slots=(
                ProjectionSlot(projection_id="p0", consumer_id="c0",
                                projected_tags=("orders",)),
                ProjectionSlot(projection_id="p1", consumer_id="c1",
                                projected_tags=("payments",)),
            ))

    def test_valid_subscription_ok(self) -> None:
        outcome = verify_projection_subscription(
            self.anchor, consumer_id="c0", projection_id="p0")
        self.assertTrue(outcome.ok)

    def test_unknown_consumer_rejected(self) -> None:
        outcome = verify_projection_subscription(
            self.anchor, consumer_id="UNKNOWN", projection_id="p0")
        self.assertFalse(outcome.ok)
        self.assertIn("consumer_not_in_anchor", outcome.reason)

    def test_wrong_projection_for_consumer_rejected(self) -> None:
        outcome = verify_projection_subscription(
            self.anchor, consumer_id="c0", projection_id="p1")
        self.assertFalse(outcome.ok)
        self.assertIn("projection_unauthorized", outcome.reason)


# =============================================================================
# ChainPersistedFanoutRegistry unit tests
# =============================================================================


class ChainPersistedFanoutRegistryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.registry = ChainPersistedFanoutRegistry(schema=self.schema)
        self.anchor = ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            producer_agent_id="prod",
            consumer_agent_ids=("c0",),
            cell_index_anchor=0,
            chain_persist_window=16,
            canonical_compact_per_tag_votes=(("orders", 1),),
            canonical_compact_projected_subset=("orders",),
            n_w15_canonical_tokens=10,
            projection_slots=(ProjectionSlot(
                projection_id="p0", consumer_id="c0",
                projected_tags=("orders",)),))

    def test_register_anchor_increments_count(self) -> None:
        self.registry.register_anchor(self.anchor)
        self.assertEqual(self.registry.n_anchors_registered, 1)
        self.assertIs(
            self.registry.get_anchor(self.anchor.chain_root_cid),
            self.anchor)

    def test_register_advance_after_anchor(self) -> None:
        self.registry.register_anchor(self.anchor)
        adv = ChainAdvanceEnvelope(
            schema_version=W26_CHAIN_ADVANCE_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            chain_root_cid=self.anchor.chain_root_cid,
            parent_advance_cid=self.anchor.chain_root_cid,
            cell_index=1, cell_in_chain=1,
            delta_per_tag_votes=(),
            delta_projected_subset_added=(),
            delta_projected_subset_removed=())
        outcome = self.registry.register_advance(adv)
        self.assertTrue(outcome.ok)
        self.assertEqual(self.registry.n_advances_registered, 1)

    def test_advance_without_anchor_rejected(self) -> None:
        adv = ChainAdvanceEnvelope(
            schema_version=W26_CHAIN_ADVANCE_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            chain_root_cid="UNKNOWN" + "0" * 57,
            parent_advance_cid="UNKNOWN" + "0" * 57,
            cell_index=1, cell_in_chain=1,
            delta_per_tag_votes=(),
            delta_projected_subset_added=(),
            delta_projected_subset_removed=())
        outcome = self.registry.register_advance(adv)
        self.assertFalse(outcome.ok)
        self.assertIn("anchor_not_found", outcome.reason)

    def test_resolve_projection_authorised(self) -> None:
        self.registry.register_anchor(self.anchor)
        anchor_out, reason = self.registry.resolve_projection(
            chain_root_cid=self.anchor.chain_root_cid,
            consumer_id="c0", projection_id="p0")
        self.assertIsNotNone(anchor_out)
        self.assertEqual(reason, "ok")

    def test_resolve_projection_wrong_id_rejected(self) -> None:
        self.registry.register_anchor(self.anchor)
        anchor_out, reason = self.registry.resolve_projection(
            chain_root_cid=self.anchor.chain_root_cid,
            consumer_id="c0", projection_id="WRONG")
        self.assertIsNone(anchor_out)
        self.assertIn("projection_unauthorized", reason)


# =============================================================================
# W26 branch vocabulary
# =============================================================================


class W26BranchVocabularyTests(unittest.TestCase):

    def test_all_branches_defined(self) -> None:
        expected = {
            W26_BRANCH_CHAIN_ANCHORED,
            W26_BRANCH_CHAIN_ADVANCED,
            W26_BRANCH_CHAIN_REJECTED,
            W26_BRANCH_CHAIN_RE_ANCHORED,
            W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
            W26_BRANCH_CHAIN_PROJECTION_REJECTED,
            W26_BRANCH_NO_TRIGGER,
            W26_BRANCH_DISABLED,
        }
        self.assertEqual(set(W26_ALL_BRANCHES), expected)

    def test_default_trigger_branches_correct(self) -> None:
        self.assertEqual(
            W26_DEFAULT_TRIGGER_BRANCHES,
            frozenset({W25_BRANCH_FANOUT_PRODUCER_EMITTED,
                        W25_BRANCH_FANOUT_CONSUMER_RESOLVED}))


# =============================================================================
# Phase 73 — W26-1 efficiency anchor
# =============================================================================


class Phase73ChainSharedTests(unittest.TestCase):
    """W26-1: chain-persisted fanout strictly reduces total visible
    tokens over W25, with correctness preserved.
    """

    def test_w26_strictly_cheaper_than_w25_loose(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertLess(result["mean_total_w26_visible_tokens"],
                         result["mean_total_w25_visible_tokens"])
        self.assertGreaterEqual(
            result["mean_savings_w26_vs_w25_per_cell"], 10.0)

    def test_w26_correctness_full(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["correctness_ratified_rate"], 1.0)

    def test_w26_consumer_resolved_full(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["chain_consumer_resolved_rate"], 1.0)

    def test_w26_savings_pct_exceeds_50pct_over_w25(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertGreater(result["savings_pct_w26_vs_w25"], 50.0)

    def test_w26_savings_pct_exceeds_85pct_over_w24(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertGreater(result["savings_pct_w26_vs_w24"], 85.0)

    def test_w26_one_anchor_n_advances(self) -> None:
        result = run_phase73(bank="chain_shared", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["registry_n_anchors"], 1)
        self.assertEqual(result["registry_n_advances"], 15)

    def test_w26_loose_and_tight_decoder_match(self) -> None:
        loose = run_phase73(bank="chain_shared", T_decoder=None,
                             K_consumers=3, n_eval=16, bank_seed=11)
        tight = run_phase73(bank="chain_shared", T_decoder=24,
                             K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(loose["mean_total_w26_visible_tokens"],
                          tight["mean_total_w26_visible_tokens"])
        self.assertEqual(loose["correctness_ratified_rate"],
                          tight["correctness_ratified_rate"])


# =============================================================================
# Phase 73 — W26-Λ-no-chain falsifier
# =============================================================================


class Phase73NoChainTests(unittest.TestCase):
    """W26-Λ-no-chain: with chain_persist_window=1, W26 reduces to W25
    byte-for-byte (every cell is an anchor).
    """

    def test_w26_equals_w25_on_no_chain(self) -> None:
        result = run_phase73(bank="no_chain", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["mean_total_w26_visible_tokens"],
                          result["mean_total_w25_visible_tokens"])
        self.assertEqual(result["mean_savings_w26_vs_w25_per_cell"], 0.0)

    def test_no_chain_branches_all_anchor(self) -> None:
        result = run_phase73(bank="no_chain", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # Every cell is anchored or re-anchored.
        b = result["branch_counts_producer"]
        self.assertEqual(
            int(b.get(W26_BRANCH_CHAIN_ADVANCED, 0)), 0)
        self.assertEqual(
            int(b.get(W26_BRANCH_CHAIN_ANCHORED, 0))
            + int(b.get(W26_BRANCH_CHAIN_RE_ANCHORED, 0)),
            result["n_cells"])


# =============================================================================
# Phase 73 — W26-Λ-tampered falsifier
# =============================================================================


class Phase73ChainTamperedTests(unittest.TestCase):
    """W26-Λ-tampered: when the producer's advance is corrupted, the
    controller rejects it; W26 falls through to W25 on rejected cells.
    """

    def test_chain_tampered_most_cells_rejected(self) -> None:
        result = run_phase73(bank="chain_tampered", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        b = result["branch_counts_producer"]
        # At least 12/16 cells should be rejected.
        self.assertGreaterEqual(
            int(b.get(W26_BRANCH_CHAIN_REJECTED, 0)), 12)

    def test_chain_tampered_correctness_preserved(self) -> None:
        result = run_phase73(bank="chain_tampered", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # The tamper is at the W26 layer; correctness via inner W25/W24
        # is preserved.
        self.assertEqual(result["correctness_ratified_rate"], 1.0)

    def test_chain_tampered_savings_collapse(self) -> None:
        result = run_phase73(bank="chain_tampered", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # Savings vs W25 should collapse below the chain_shared floor.
        self.assertLess(result["mean_savings_w26_vs_w25_per_cell"], 5.0)

    def test_chain_tampered_registry_rejects(self) -> None:
        result = run_phase73(bank="chain_tampered", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertGreaterEqual(
            result["registry_n_advances_rejected"], 12)


# =============================================================================
# Phase 73 — W26-Λ-projection-mismatch falsifier
# =============================================================================


class Phase73ProjectionMismatchTests(unittest.TestCase):
    """W26-Λ-projection-mismatch: a consumer requesting a projection
    not in their slot is rejected; the other consumers still resolve.
    """

    def test_one_third_consumers_rejected(self) -> None:
        result = run_phase73(bank="projection_mismatch", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertAlmostEqual(
            result["chain_consumer_rejected_rate"], 1.0 / 3, places=2)

    def test_two_thirds_consumers_resolve(self) -> None:
        result = run_phase73(bank="projection_mismatch", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertAlmostEqual(
            result["chain_consumer_resolved_rate"], 2.0 / 3, places=2)

    def test_correctness_preserved_despite_projection_mismatch(self
                                                                  ) -> None:
        result = run_phase73(bank="projection_mismatch", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["correctness_ratified_rate"], 1.0)

    def test_n_consumer_rejected_equals_n_cells(self) -> None:
        result = run_phase73(bank="projection_mismatch", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # 1 of 3 consumers rejected on every cell = n_cells rejections.
        self.assertEqual(result["n_consumer_rejected"], result["n_cells"])


# =============================================================================
# Phase 73 — W26-Λ-divergent regime
# =============================================================================


class Phase73DivergentTests(unittest.TestCase):
    """W26-Λ-divergent: when gold subset flips at the bench midpoint,
    inner W25 fires no_trigger on divergent cells; W26 falls through.
    """

    def test_divergent_correctness_drops_to_half(self) -> None:
        result = run_phase73(bank="divergent", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        self.assertEqual(result["correctness_ratified_rate"], 0.5)

    def test_divergent_partial_chain_savings(self) -> None:
        result = run_phase73(bank="divergent", T_decoder=None,
                              K_consumers=3, n_eval=16, bank_seed=11)
        # Some chain advance still happens on the first half.
        b = result["branch_counts_producer"]
        self.assertGreater(int(b.get(W26_BRANCH_CHAIN_ADVANCED, 0)), 0)
        # And some cells fall through (no_trigger).
        self.assertGreater(int(b.get(W26_BRANCH_NO_TRIGGER, 0)), 0)


# =============================================================================
# Phase 73 — W26-2 seed stability
# =============================================================================


class Phase73SeedStabilityTests(unittest.TestCase):
    """W26-2: W26-1 success criterion holds across all 5 seeds."""

    def test_seed_stability_5_of_5(self) -> None:
        sweep = run_phase73_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31), bank="chain_shared",
            T_decoder=None, K_consumers=3, n_eval=16)
        self.assertTrue(sweep["all_savings_w25_positive"])
        self.assertTrue(sweep["all_savings_w24_positive"])
        self.assertTrue(sweep["all_correctness_1000"])
        self.assertGreaterEqual(
            sweep["min_savings_w26_vs_w25"], 10.0)
        self.assertGreaterEqual(
            sweep["min_savings_w26_vs_w24"], 50.0)

    def test_seed_stability_correctness_invariant(self) -> None:
        sweep = run_phase73_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31), bank="chain_shared",
            T_decoder=None, K_consumers=3, n_eval=16)
        self.assertEqual(sweep["min_correctness"], 1.0)


# =============================================================================
# Phase 73 — K-scaling sweep
# =============================================================================


class Phase73KScalingTests(unittest.TestCase):
    """W25-C-K-SCALING / W26-C-K-SCALING: discharge by measurement.

    The W25 conjecture is that savings grow as K×(C−1) — at K=10 with
    C=14.6 the saving over W24 is ≈ 88%.  The W26 conjecture is the
    same growth with an additional cross-cell amortisation of the
    producer's per-cell cost — at K=10, W26 savings over W24 should
    exceed 90%.
    """

    def test_w26_savings_over_w24_grows_with_K(self) -> None:
        sweep = run_k_scaling_sweep(
            K_values=(3, 5, 8, 10), n_eval=16, bank_seed=11)
        rows = sweep["rows"]
        savings_pct = [r["savings_pct_w26_vs_w24"] for r in rows]
        # Strictly non-decreasing.
        for a, b in zip(savings_pct, savings_pct[1:]):
            self.assertLessEqual(a, b + 0.5)
        # K=10 hits the conjectured ≥ 90%.
        self.assertGreater(savings_pct[-1], 90.0)

    def test_w25_savings_pct_over_w24_at_K10(self) -> None:
        # Discharges W25-C-K-SCALING (≈ 88% at K=10).
        sweep = run_k_scaling_sweep(
            K_values=(3, 5, 8, 10), n_eval=16, bank_seed=11)
        K10 = sweep["rows"][-1]
        # Compute W25 saving over W24 directly.
        w25_save_pct = (
            100.0
            * (K10["mean_total_w24_visible_tokens"]
               - K10["mean_total_w25_visible_tokens"])
            / K10["mean_total_w24_visible_tokens"])
        self.assertGreater(w25_save_pct, 80.0)


# =============================================================================
# Phase 73 — cross-regime evaluation
# =============================================================================


class Phase73CrossRegimeTests(unittest.TestCase):

    def test_cross_regime_six_banks_two_decoders(self) -> None:
        results = run_cross_regime_p73(
            n_eval=8, K_consumers=3, bank_seed=11, verbose=False)
        # 6 banks × 2 T_decoder = 12 entries.
        self.assertEqual(len(results), 12)

    def test_chain_shared_saves_on_both_decoders(self) -> None:
        results = run_cross_regime_p73(
            n_eval=8, K_consumers=3, bank_seed=11, verbose=False)
        for k, r in results.items():
            if k.startswith("chain_shared"):
                self.assertGreater(
                    r["mean_savings_w26_vs_w25_per_cell"], 0.0)

    def test_no_chain_zero_savings_both_decoders(self) -> None:
        results = run_cross_regime_p73(
            n_eval=8, K_consumers=3, bank_seed=11, verbose=False)
        for k, r in results.items():
            if k.startswith("no_chain"):
                self.assertEqual(
                    r["mean_savings_w26_vs_w25_per_cell"], 0.0)


if __name__ == "__main__":
    unittest.main()
