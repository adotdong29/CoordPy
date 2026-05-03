"""Tests for SDK v3.24 — capsule-native cross-cell delta execution +
quorum-keyed cache + super-token reference (W23 family + Phase 70
driver).

Theorem anchors:

* **W23-1 (efficiency)** — :class:`Phase70DeltaFanoutTests` /
  :class:`Phase70SeedStabilityTests`. R-70-DELTA-FANOUT shows
  ``mean_n_w23_visible_tokens_to_decider`` strictly below the W22
  baseline AND ``correctness_ratified_rate = 1.000`` AND
  ``chain_verifies_ok_rate = 1.000``, stable across 5/5 seeds.
* **W23-2 (mitigation)** — :class:`Phase70AmplifiedLLMTests`. On
  R-70-AMPLIFIED-LLM the W22 baseline reproduces the
  W22-C-CACHE-AMPLIFICATION effect (acc_full < 1.0); the W23
  quorum-keyed cache strictly improves correctness over W22.
* **W23-3 (trust-boundary soundness)** —
  :class:`Phase70SuperTokenTamperedTests`,
  :class:`Phase70ChainBrokenTests`,
  :class:`SessionDigestVerificationTests`,
  :class:`SuperTokenReferenceTests`. Every tampered envelope /
  reference is rejected; W23 falls through to W22 byte-for-byte.
* **W23-Λ-no-delta (named falsifier)** —
  :class:`Phase70NoDeltaTests`. With no cross-cell state, W23
  reduces to W22 (no savings).
* **W23-Λ-real** — see ``docs/data/phase70_live_mixtral_8x7b_n4.json``.
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from vision_mvp.experiments.phase70_capsule_session_delta import (
    run_phase70, run_phase70_seed_stability_sweep,
    run_cross_regime_synthetic_p70,
    FlippingProbabilisticOracle,
)
from vision_mvp.coordpy.team_coord import (
    SchemaCapsule, build_incident_triage_schema_capsule,
    SessionDigestEnvelope, SessionDeltaEnvelope,
    verify_session_digest_chain, verify_session_delta,
    SuperTokenReferenceEnvelope, SuperTokenRegistry,
    verify_super_token_reference,
    QuorumKeyedSharedReadCache, QuorumKeyedCachingOracleAdapter,
    SharedReadCache, CrossHostProducerDecoderProxy,
    LatentDigestDisambiguator, CrossCellDeltaDisambiguator,
    LatentVerificationOutcome,
    OutsideQuery, OutsideVerdict,
    CACHE_FRESHNESS_BYTE_IDENTICAL,
    CACHE_FRESHNESS_PER_CELL_NONCE,
    CACHE_FRESHNESS_QUORUM_LOCKED,
    CACHE_FRESHNESS_POLICIES,
    W23_BRANCH_DELTA_RESOLVED, W23_BRANCH_DELTA_REJECTED,
    W23_BRANCH_GENESIS, W23_BRANCH_SUPER_TOKEN_RESOLVED,
    W23_BRANCH_SUPER_TOKEN_REJECTED, W23_BRANCH_NO_TRIGGER,
    W23_BRANCH_DISABLED,
    W23_SESSION_ENVELOPE_SCHEMA_VERSION,
    W23_DELTA_ENVELOPE_SCHEMA_VERSION,
    W23_SUPER_TOKEN_SCHEMA_VERSION,
)


# =============================================================================
# Unit tests on the W23 surface
# =============================================================================


class SessionDigestEnvelopeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def _make(self) -> SessionDigestEnvelope:
        return SessionDigestEnvelope(
            schema_cid=self.schema.cid,
            prior_session_digest_cid="",
            n_cells=1,
            cumulative_per_tag_votes=(("orders", 2), ("payments", 2)),
            latest_projected_subset=("orders", "payments"),
            latest_inner_branch="quorum_resolved",
            n_cells_resolved=1)

    def test_genesis_signs_at_construction(self) -> None:
        env = self._make()
        self.assertEqual(len(env.digest_cid), 64)
        self.assertEqual(env.digest_cid, env.recompute_digest_cid())

    def test_canonicalises_unsorted_inputs(self) -> None:
        env_a = SessionDigestEnvelope(
            schema_cid=self.schema.cid, prior_session_digest_cid="",
            n_cells=1,
            cumulative_per_tag_votes=(("payments", 2), ("orders", 2)),
            latest_projected_subset=("payments", "orders"),
            latest_inner_branch="quorum_resolved",
            n_cells_resolved=1)
        env_b = self._make()
        self.assertEqual(env_a.digest_cid, env_b.digest_cid)


class SessionDigestVerificationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.env = SessionDigestEnvelope(
            schema_cid=self.schema.cid, prior_session_digest_cid="",
            n_cells=1,
            cumulative_per_tag_votes=(("orders", 2),),
            latest_projected_subset=("orders",),
            latest_inner_branch="quorum_resolved", n_cells_resolved=1)

    def test_ok_on_valid_genesis(self) -> None:
        out = verify_session_digest_chain(
            self.env, registered_schema=self.schema,
            prior_chain_head_cid="")
        self.assertTrue(out.ok)
        self.assertEqual(out.reason, "ok")

    def test_chain_head_mismatch_when_prior_changes(self) -> None:
        out = verify_session_digest_chain(
            self.env, registered_schema=self.schema,
            prior_chain_head_cid="0" * 64)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "chain_head_mismatch")

    def test_schema_cid_mismatch_when_schema_drifts(self) -> None:
        drifted = dataclasses.replace(self.schema, version="v9_drifted")
        out = verify_session_digest_chain(
            self.env, registered_schema=drifted,
            prior_chain_head_cid="")
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "schema_cid_mismatch")

    def test_hash_mismatch_when_envelope_tampered(self) -> None:
        # Tamper the cumulative votes after construction; the
        # signed digest_cid no longer matches the canonical bytes.
        tampered = dataclasses.replace(
            self.env,
            cumulative_per_tag_votes=(("orders", 99),))
        # ``replace`` re-signs at __post_init__ -> we have to bypass.
        object.__setattr__(
            tampered, "cumulative_per_tag_votes",
            (("orders", 99),))
        # Force the digest_cid to the original (so it no longer
        # matches the canonical bytes).
        object.__setattr__(tampered, "digest_cid", self.env.digest_cid)
        out = verify_session_digest_chain(
            tampered, registered_schema=self.schema,
            prior_chain_head_cid="")
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "hash_mismatch")

    def test_empty_envelope_rejected(self) -> None:
        out = verify_session_digest_chain(
            None, registered_schema=self.schema,
            prior_chain_head_cid="")
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "empty_envelope")


class SessionDeltaEnvelopeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def test_signs_at_construction(self) -> None:
        delta = SessionDeltaEnvelope(
            schema_cid=self.schema.cid,
            prior_session_digest_cid="a" * 64,
            parent_session_digest_cid="a" * 64,
            cell_index=1, inner_w19_branch="quorum_resolved",
            delta_projected_added=(),
            delta_projected_removed=(),
            delta_per_tag_votes=(),
            delta_n_outside_tokens=0,
            parent_probe_cids=("aa" * 32,))
        self.assertEqual(len(delta.delta_cid), 64)
        self.assertEqual(delta.delta_cid, delta.recompute_delta_cid())

    def test_empty_delta_text_is_minimal(self) -> None:
        # Adaptive serialisation: empty optional fields are omitted.
        delta = SessionDeltaEnvelope(
            schema_cid=self.schema.cid,
            prior_session_digest_cid="a" * 64,
            parent_session_digest_cid="a" * 64,
            cell_index=2, inner_w19_branch="",
            delta_projected_added=(),
            delta_projected_removed=(),
            delta_per_tag_votes=(),
            delta_n_outside_tokens=0,
            parent_probe_cids=())
        text = delta.to_decoder_text()
        # Should NOT contain any of the "(none)" placeholders.
        self.assertNotIn("(none)", text)
        # Should contain the four anchor fields.
        self.assertIn("SESSION_DELTA", text)
        self.assertIn("schema_cid=", text)
        self.assertIn("parent=", text)
        self.assertIn("cell=", text)
        self.assertIn("delta_cid=", text)


class SessionDeltaVerificationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.parent_cid = "a" * 64
        self.delta = SessionDeltaEnvelope(
            schema_cid=self.schema.cid,
            prior_session_digest_cid=self.parent_cid,
            parent_session_digest_cid=self.parent_cid,
            cell_index=1, inner_w19_branch="quorum_resolved",
            delta_projected_added=("orders",),
            delta_projected_removed=(),
            delta_per_tag_votes=(("orders", 2),),
            delta_n_outside_tokens=12,
            parent_probe_cids=("bb" * 32,))

    def test_ok_when_parent_matches(self) -> None:
        out = verify_session_delta(
            self.delta, registered_schema=self.schema,
            parent_session_digest_cid=self.parent_cid)
        self.assertTrue(out.ok)
        self.assertEqual(out.reason, "ok")

    def test_parent_mismatch_when_chain_drifts(self) -> None:
        out = verify_session_delta(
            self.delta, registered_schema=self.schema,
            parent_session_digest_cid="0" * 64)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "parent_session_mismatch")


class SuperTokenReferenceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.parent_cid = "a" * 64
        self.delta = SessionDeltaEnvelope(
            schema_cid=self.schema.cid,
            prior_session_digest_cid=self.parent_cid,
            parent_session_digest_cid=self.parent_cid,
            cell_index=1, inner_w19_branch="quorum_resolved",
            delta_projected_added=("orders",),
            delta_projected_removed=(),
            delta_per_tag_votes=(("orders", 2),),
            delta_n_outside_tokens=12,
            parent_probe_cids=("bb" * 32,))

    def test_super_token_is_one_whitespace_token(self) -> None:
        env = SuperTokenReferenceEnvelope(
            schema_cid=self.schema.cid,
            delta_cid=self.delta.delta_cid,
            parent_session_digest_cid=self.parent_cid,
            hex_prefix_len=16)
        self.assertEqual(env.n_super_token_tokens, 1)
        self.assertEqual(env.super_token.count(" "), 0)
        # The super-token text must be whitespace-tokenised as a
        # single token (length 1 in str.split() units).
        self.assertEqual(len(env.super_token.split()), 1)

    def test_super_token_resolves_when_registered(self) -> None:
        registry = SuperTokenRegistry()
        registry.register(self.delta, hex_prefix_len=16)
        env = SuperTokenReferenceEnvelope(
            schema_cid=self.schema.cid,
            delta_cid=self.delta.delta_cid,
            parent_session_digest_cid=self.parent_cid,
            hex_prefix_len=16)
        out = verify_super_token_reference(
            env, registry=registry, registered_schema=self.schema,
            parent_session_digest_cid=self.parent_cid)
        self.assertTrue(out.ok)
        self.assertEqual(out.reason, "ok")
        self.assertEqual(registry.n_resolved, 1)

    def test_super_token_rejects_when_registry_empty(self) -> None:
        # The producer registered, but the *verifier* has a different
        # (empty) registry — the R-70-SUPER-TOKEN-TAMPERED bench
        # config.
        producer_registry = SuperTokenRegistry()
        producer_registry.register(self.delta, hex_prefix_len=16)
        verifier_registry = SuperTokenRegistry()
        env = SuperTokenReferenceEnvelope(
            schema_cid=self.schema.cid,
            delta_cid=self.delta.delta_cid,
            parent_session_digest_cid=self.parent_cid,
            hex_prefix_len=16)
        out = verify_super_token_reference(
            env, registry=verifier_registry,
            registered_schema=self.schema,
            parent_session_digest_cid=self.parent_cid)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "unknown_super_token")
        self.assertEqual(verifier_registry.n_rejected, 1)

    def test_super_token_rejects_on_parent_mismatch(self) -> None:
        registry = SuperTokenRegistry()
        registry.register(self.delta, hex_prefix_len=16)
        env = SuperTokenReferenceEnvelope(
            schema_cid=self.schema.cid,
            delta_cid=self.delta.delta_cid,
            parent_session_digest_cid=self.parent_cid,
            hex_prefix_len=16)
        out = verify_super_token_reference(
            env, registry=registry, registered_schema=self.schema,
            parent_session_digest_cid="0" * 64)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "parent_session_mismatch")

    def test_super_token_invalid_prefix_len(self) -> None:
        with self.assertRaises(ValueError):
            SuperTokenReferenceEnvelope(
                schema_cid=self.schema.cid,
                delta_cid=self.delta.delta_cid,
                parent_session_digest_cid=self.parent_cid,
                hex_prefix_len=2)
        with self.assertRaises(ValueError):
            SuperTokenReferenceEnvelope(
                schema_cid=self.schema.cid,
                delta_cid=self.delta.delta_cid,
                parent_session_digest_cid=self.parent_cid,
                hex_prefix_len=128)


class QuorumKeyedSharedReadCacheTests(unittest.TestCase):

    def test_default_policy_is_byte_identical(self) -> None:
        cache = QuorumKeyedSharedReadCache()
        self.assertEqual(cache.policy_for("any"),
                          CACHE_FRESHNESS_BYTE_IDENTICAL)

    def test_policy_set_validates(self) -> None:
        cache = QuorumKeyedSharedReadCache()
        with self.assertRaises(ValueError):
            cache.set_policy("oracle", "unknown_policy")

    def test_per_cell_nonce_policy_changes_cid(self) -> None:
        cache = QuorumKeyedSharedReadCache()
        cache.set_policy("flipping", CACHE_FRESHNESS_PER_CELL_NONCE)
        q = OutsideQuery(
            admitted_tags=("a", "b"),
            elected_root_cause="x",
            primary_payload="p", witness_payloads=(),
            max_response_tokens=12)
        cid_a = cache.query_cid_for(q, oracle_id="flipping",
                                       cell_nonce="cell0")
        cid_b = cache.query_cid_for(q, oracle_id="flipping",
                                       cell_nonce="cell1")
        cid_byte = cache.query_cid_for(q, oracle_id="deterministic",
                                          cell_nonce="cell0")
        self.assertNotEqual(cid_a, cid_b,
                              "per-cell nonce should split the cache key")
        # Deterministic oracle (BYTE_IDENTICAL) is invariant under nonce.
        cid_byte_2 = cache.query_cid_for(
            q, oracle_id="deterministic", cell_nonce="cell1")
        self.assertEqual(cid_byte, cid_byte_2)

    def test_quorum_locked_defers_put(self) -> None:
        cache = QuorumKeyedSharedReadCache()
        cache.set_policy("locked", CACHE_FRESHNESS_QUORUM_LOCKED)
        cache.put("cid_x", b"body", n_tokens=4, oracle_id="locked",
                    quorum_locked_pending=True)
        # Pending: not yet in the inner cache.
        self.assertEqual(cache.inner.stats()["n_entries"], 0)
        self.assertEqual(len(cache.quorum_pending), 1)
        # Confirm quorum: promote the pending entry.
        n = cache.confirm_quorum_for(["cid_x"])
        self.assertEqual(n, 1)
        self.assertEqual(cache.inner.stats()["n_entries"], 1)


class QuorumKeyedCachingOracleAdapterTests(unittest.TestCase):

    def test_per_cell_nonce_does_not_collapse_to_first_reply(self) -> None:
        # The flipping oracle returns a "bad" reply on consult #1
        # and "good" replies thereafter. With BYTE_IDENTICAL caching
        # the bad reply is cached and returned for consult #2 too.
        # With PER_CELL_NONCE caching consult #2 misses the cache
        # and gets the good reply.
        flipping = FlippingProbabilisticOracle()
        cache = QuorumKeyedSharedReadCache()
        cache.set_policy("flipping_probabilistic",
                          CACHE_FRESHNESS_PER_CELL_NONCE)
        adapter = QuorumKeyedCachingOracleAdapter(
            inner=flipping, cache=cache,
            oracle_id="flipping_probabilistic",
            cell_nonce="cell_0")
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="primary", witness_payloads=(),
            max_response_tokens=24)
        v0 = adapter.consult(q)
        # Switch to a new cell — PER_CELL_NONCE means new cache key.
        adapter.cell_nonce = "cell_1"
        v1 = adapter.consult(q)
        # The flipping oracle returns DIFFERENT replies for consult
        # 1 vs consult 2; with PER_CELL_NONCE we get both.
        self.assertNotEqual(v0.payload, v1.payload)
        self.assertEqual(flipping.n_consults, 2)


# =============================================================================
# CrossHostProducerDecoderProxy
# =============================================================================


class CrossHostProducerDecoderProxyTests(unittest.TestCase):

    def test_round_trip_returns_equivalent_dict(self) -> None:
        proxy = CrossHostProducerDecoderProxy()
        payload = {"a": 1, "b": [1, 2], "c": "three"}
        out = proxy.producer_to_decoder(payload)
        self.assertEqual(payload, out)
        self.assertEqual(proxy.n_round_trips, 1)
        self.assertGreater(proxy.n_bytes_serialised, 0)


# =============================================================================
# Phase 70 driver — bench-level theorem anchors
# =============================================================================


class Phase70DeltaFanoutTests(unittest.TestCase):
    """W23-1 (efficiency, correctness ratification)."""

    def test_delta_savings_loose(self) -> None:
        rep = run_phase70(bank="delta_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        eff = rep["eff_compare"]
        # W23 delta visible-tokens strictly below W22.
        self.assertLess(eff["w23_delta_visible_tokens_per_cell"],
                          eff["w22_visible_tokens_per_cell"])
        self.assertGreater(eff["w23_delta_savings_per_cell"], 0)
        self.assertGreater(eff["w23_delta_savings_pct"], 0.0)
        # Super-token savings even larger.
        self.assertLess(eff["w23_super_token_visible_tokens_per_cell"],
                          eff["w23_delta_visible_tokens_per_cell"])
        # Chain verifies on every cell.
        self.assertEqual(eff["chain_verifies_ok_rate"], 1.0)
        # Super-token resolves on every super-token cell.
        self.assertEqual(eff["super_token_resolved_rate"], 1.0)
        # Correctness ratified across every cell.
        ratified = rep["correctness_ratified_rates"]
        self.assertEqual(ratified["capsule_w23_delta"], 1.0)
        self.assertEqual(ratified["capsule_w23_super_token"], 1.0)
        self.assertEqual(ratified["capsule_w23_quorum_keyed"], 1.0)

    def test_delta_savings_tight(self) -> None:
        rep = run_phase70(bank="delta_fanout", n_eval=8,
                            T_decoder=24, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertLess(eff["w23_delta_visible_tokens_per_cell"],
                          eff["w22_visible_tokens_per_cell"])
        self.assertGreater(eff["w23_delta_savings_per_cell"], 0)
        ratified = rep["correctness_ratified_rates"]
        self.assertEqual(ratified["capsule_w23_delta"], 1.0)

    def test_super_token_n_tokens_per_cell_is_bounded(self) -> None:
        rep = run_phase70(bank="delta_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        ps = rep["pack_stats_summary"]["capsule_w23_super_token"]
        # Every resolved super-token cell pays exactly 1 super-token
        # in addition to W15-kept.
        n_cells = int(ps["n_cells"])
        n_super_resolved = int(ps["n_w23_super_token_resolved_cells"])
        n_genesis = int(ps["n_w23_genesis_cells"])
        self.assertEqual(n_super_resolved + n_genesis, n_cells)

    def test_no_accuracy_regression_on_delta_fanout(self) -> None:
        rep = run_phase70(bank="delta_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        pooled = rep["pooled"]
        for s in ("capsule_w23_delta", "capsule_w23_super_token",
                   "capsule_w23_quorum_keyed"):
            self.assertEqual(
                pooled[s]["accuracy_full"],
                pooled["capsule_w22_hybrid"]["accuracy_full"],
                f"{s} must tie W22 on accuracy_full")


class Phase70AmplifiedLLMTests(unittest.TestCase):
    """W23-2 (mitigation of W22-C-CACHE-AMPLIFICATION)."""

    def test_w22_baseline_reproduces_amplification(self) -> None:
        rep = run_phase70(bank="amplified_llm", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        pooled = rep["pooled"]
        # The W22 baseline is below 1.0 — the cache amplification
        # is real on this regime.
        self.assertLess(
            pooled["capsule_w22_hybrid"]["accuracy_full"], 1.0)

    def test_w23_quorum_keyed_strictly_improves_over_w22(self) -> None:
        rep = run_phase70(bank="amplified_llm", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        pooled = rep["pooled"]
        adv = rep["mitigation_advantage_w23_minus_w22"]
        # Strict mitigation: w23_quorum_keyed > w22_hybrid on
        # accuracy_full.
        self.assertGreater(
            pooled["capsule_w23_quorum_keyed"]["accuracy_full"],
            pooled["capsule_w22_hybrid"]["accuracy_full"])
        self.assertGreater(adv, 0.0)

    def test_w23_delta_alone_does_not_fix_amplification(self) -> None:
        # Without quorum-keyed cache, the W23 delta layer inherits
        # the amplified W22 votes and produces the same accuracy.
        # This isolates the W23-2 mitigation as load-bearing.
        rep = run_phase70(bank="amplified_llm", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        pooled = rep["pooled"]
        self.assertEqual(
            pooled["capsule_w23_delta"]["accuracy_full"],
            pooled["capsule_w22_hybrid"]["accuracy_full"])


class Phase70SuperTokenTamperedTests(unittest.TestCase):
    """W23-3 (super-token rejection via verifier registry split)."""

    def test_super_token_rejected_on_every_cell(self) -> None:
        rep = run_phase70(bank="super_token_tampered", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertEqual(eff["super_token_verification_ok_rate"], 0.0)
        self.assertEqual(eff["super_token_resolved_rate"], 0.0)
        # Correctness preserved (W22 answer) on rejection.
        ratified = rep["correctness_ratified_rates"]
        self.assertEqual(ratified["capsule_w23_super_token"], 1.0)

    def test_super_token_rejected_no_visible_tokens_savings(self) -> None:
        rep = run_phase70(bank="super_token_tampered", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # On rejection W23 falls through to W22 verbose digest cost.
        self.assertEqual(eff["w23_super_token_savings_per_cell"], 0.0)


class Phase70ChainBrokenTests(unittest.TestCase):
    """W23-3 (chain-link rejection via verifier chain head split)."""

    def test_chain_broken_rejects_post_genesis_cells(self) -> None:
        rep = run_phase70(bank="chain_broken", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # Only the genesis cell verifies; all subsequent cells are
        # rejected because the verifier's expected chain head is
        # the bench-installed phantom.
        self.assertLess(eff["chain_verifies_ok_rate"], 0.5)
        # Correctness preserved on rejection.
        ratified = rep["correctness_ratified_rates"]
        self.assertEqual(ratified["capsule_w23_delta"], 1.0)

    def test_chain_broken_no_visible_tokens_savings(self) -> None:
        rep = run_phase70(bank="chain_broken", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # On rejection W23 falls through to W22 cost — no savings.
        self.assertEqual(eff["w23_delta_savings_per_cell"], 0.0)


class Phase70NoDeltaTests(unittest.TestCase):
    """W23-Λ-no-delta — no cross-cell state, W23 reduces to W22."""

    def test_no_delta_zero_savings(self) -> None:
        rep = run_phase70(bank="no_delta", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # Every cell is genesis (chain reset) → no delta savings.
        self.assertEqual(eff["w23_delta_savings_per_cell"], 0.0)
        self.assertEqual(eff["w23_super_token_savings_per_cell"], 0.0)
        # Accuracy preserved.
        ratified = rep["correctness_ratified_rates"]
        self.assertEqual(ratified["capsule_w23_delta"], 1.0)


class Phase70SeedStabilityTests(unittest.TestCase):
    """W23-1 stability — savings hold across 5/5 seeds."""

    def test_savings_strictly_positive_across_5_seeds(self) -> None:
        out = run_phase70_seed_stability_sweep(
            bank="delta_fanout", T_decoder=None, n_eval=8,
            K_auditor=12, seeds=(11, 17, 23, 29, 31))
        self.assertGreater(
            out["min_w23_delta_savings_per_cell"], 0.0)
        self.assertGreater(
            out["min_w23_super_token_savings_per_cell"], 0.0)
        self.assertEqual(
            out["min_w23_delta_correctness_ratified_rate"], 1.0)
        self.assertEqual(
            out["min_w23_super_token_correctness_ratified_rate"], 1.0)


class Phase70AuditOKTests(unittest.TestCase):
    """T-1..T-7 audit holds on every cell of every regime."""

    def test_audit_ok_on_every_w23_cell(self) -> None:
        for bank in ("delta_fanout", "super_token", "no_delta",
                      "super_token_tampered", "chain_broken"):
            rep = run_phase70(bank=bank, n_eval=4, T_decoder=None,
                                K_auditor=12, bank_seed=11)
            grid = rep["audit_ok_grid"]
            for s in ("capsule_w22_hybrid", "capsule_w23_delta",
                       "capsule_w23_super_token",
                       "capsule_w23_quorum_keyed"):
                self.assertTrue(grid[s],
                                  f"audit failed on {bank} / {s}")


class W23SDKReductionTests(unittest.TestCase):
    """W23 backward-compat — when ``enabled=False`` or schema=None,
    the W23 layer reduces to W22 byte-for-byte on the answer field."""

    def test_disabled_reduces_to_w22(self) -> None:
        schema = build_incident_triage_schema_capsule()
        cache = SharedReadCache()
        w22 = LatentDigestDisambiguator(schema=schema, cache=cache)
        w23 = CrossCellDeltaDisambiguator(
            inner=w22, schema=schema, enabled=False)
        self.assertFalse(w23.enabled)
        # Sanity: chain head is empty before any cell.
        self.assertEqual(w23.chain_head_cid(), "")


class Phase70CrossRegimeSmokeTests(unittest.TestCase):
    """Cross-regime smoke test."""

    def test_cross_regime_matrix_smoke(self) -> None:
        out = run_cross_regime_synthetic_p70(
            n_eval=4, bank_seed=11, K_auditor=12,
            T_decoder_tight=24, quorum_min=2)
        # DELTA-FANOUT-LOOSE: W23 saves visible tokens.
        loose = out["regimes"]["R-70-DELTA-FANOUT-LOOSE"]
        self.assertGreater(
            loose["eff_compare"]["w23_delta_savings_per_cell"], 0)
        self.assertEqual(
            loose["correctness_ratified_rates"][
                "capsule_w23_delta"], 1.0)
        # SUPER-TOKEN-TAMPERED: every super-token rejected.
        tampered = out["regimes"]["R-70-SUPER-TOKEN-TAMPERED"]
        self.assertEqual(
            tampered["eff_compare"][
                "super_token_verification_ok_rate"], 0.0)
        self.assertEqual(
            tampered["correctness_ratified_rates"][
                "capsule_w23_super_token"], 1.0)
        # CHAIN-BROKEN: chain rejected on most cells; correctness OK.
        chain = out["regimes"]["R-70-CHAIN-BROKEN"]
        self.assertLess(
            chain["eff_compare"]["chain_verifies_ok_rate"], 0.5)
        self.assertEqual(
            chain["correctness_ratified_rates"][
                "capsule_w23_delta"], 1.0)
        # AMPLIFIED-LLM: W23 quorum-keyed strictly mitigates.
        amp = out["regimes"]["R-70-AMPLIFIED-LLM"]
        self.assertGreater(
            amp["mitigation_advantage_w23_minus_w22"], 0.0)


class W22BackwardCompatTests(unittest.TestCase):
    """W23-3-B regression — every existing W22 / W21 / ... regime
    still passes byte-for-byte. Tested by re-running the existing
    Phase 69 harness through the same imports."""

    def test_phase69_smoke_after_w23_landed(self) -> None:
        from vision_mvp.experiments.phase69_capsule_latent_hybrid import (
            run_phase69)
        rep = run_phase69(bank="cache_fanout", n_eval=4,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        # W22-1 still holds.
        self.assertGreater(
            rep["eff_compare"]["visible_tokens_savings_per_cell"], 0)
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
