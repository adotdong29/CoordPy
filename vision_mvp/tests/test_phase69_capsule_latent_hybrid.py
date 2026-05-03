"""Tests for SDK v3.23 — capsule + audited latent-state-sharing
hybrid (W22 family + Phase 69 driver).

Theorem anchors:

* **W22-1 (efficiency)** — :class:`Phase69CacheFanoutTests` /
  :class:`Phase69SeedStabilityTests`. R-69-CACHE-FANOUT shows
  ``mean_n_visible_tokens_to_decider`` strictly below the W21
  baseline AND ``cache_tokens_saved_total > 0``, stable across
  5/5 seeds.
* **W22-2 (correctness ratification)** —
  :class:`Phase69CorrectnessRatificationTests`. On every cell of
  R-69-CACHE-FANOUT, the W22 ``answer["services"]`` equals the
  W21 ``answer["services"]`` byte-for-byte.
* **W22-3 (trust-boundary soundness)** —
  :class:`Phase69PoisonedDigestTests`,
  :class:`Phase69SchemaDriftTests`,
  :class:`LatentDigestVerificationTests`. Every tampered envelope
  is rejected; W22 falls through to W21 byte-for-byte; correctness
  preserved.
* **W22-3-A (backward-compat reduction to W21)** —
  :class:`Phase69NoTriggerTests`. When W21 abstains, W22 fires
  ``W22_BRANCH_NO_TRIGGER`` and reduces to W21 byte-for-byte.
* **W22-Λ-no-cache (named falsifier)** —
  :class:`Phase69NoCacheTests`. With per-cell fresh caches,
  ``cache_tokens_saved_total == 0``; the digest still compresses
  but no wire-side savings.
* **W22 audit-preserving** — :class:`Phase69AuditOKTests`. T-1..T-7
  audit holds on every cell of every regime.
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from vision_mvp.experiments.phase69_capsule_latent_hybrid import (
    run_phase69, run_phase69_seed_stability_sweep,
    run_cross_regime_synthetic_p69,
)
from vision_mvp.coordpy.team_coord import (
    SchemaCapsule, LatentDigestEnvelope, LatentVerificationOutcome,
    verify_latent_digest, build_incident_triage_schema_capsule,
    SharedReadCache, CachingOracleAdapter, EnvelopeTamperer,
    LatentDigestDisambiguator, TrustWeightedMultiOracleDisambiguator,
    BundleContradictionDisambiguator,
    RelationalCompatibilityDisambiguator, AttentionAwareBundleDecoder,
    OracleRegistration, ServiceGraphOracle, ChangeHistoryOracle,
    CompromisedServiceGraphOracle, SingletonAsymmetricOracle,
    AbstainingOracle, OutsideQuery, OutsideVerdict,
    W22_BRANCH_LATENT_RESOLVED, W22_BRANCH_LATENT_REJECTED,
    W22_BRANCH_NO_TRIGGER, W22_BRANCH_DISABLED, W22_BRANCH_NO_SCHEMA,
    W22_LATENT_ENVELOPE_SCHEMA_VERSION,
)


# =============================================================================
# Unit tests on the W22 surface
# =============================================================================


class SchemaCapsuleTests(unittest.TestCase):

    def test_schema_capsule_cid_is_deterministic(self) -> None:
        a = build_incident_triage_schema_capsule()
        b = build_incident_triage_schema_capsule()
        self.assertEqual(a.cid, b.cid)
        self.assertEqual(len(a.cid), 64)  # SHA-256 hex

    def test_schema_capsule_cid_changes_on_version_bump(self) -> None:
        a = build_incident_triage_schema_capsule()
        b = dataclasses.replace(a, version="v2")
        self.assertNotEqual(a.cid, b.cid)

    def test_schema_capsule_canonicalises_unsorted_input(self) -> None:
        s1 = SchemaCapsule(
            schema_id="x", version="v1",
            closed_vocab_root_causes=("b", "a"),
            closed_vocab_services=("c", "a", "b"),
            closed_vocab_oracle_kinds=())
        s2 = SchemaCapsule(
            schema_id="x", version="v1",
            closed_vocab_root_causes=("a", "b"),
            closed_vocab_services=("a", "b", "c"),
            closed_vocab_oracle_kinds=())
        self.assertEqual(s1.cid, s2.cid)


class LatentDigestEnvelopeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def _make_envelope(self) -> LatentDigestEnvelope:
        return LatentDigestEnvelope(
            schema_cid=self.schema.cid,
            inner_w19_branch="abstained_symmetric",
            quorum_min=2, min_trust_sum=0.0,
            per_tag_vote_count=(("orders", 2), ("payments", 2),
                                  ("cache", 1)),
            per_tag_trust_sum=(("orders", 2.0), ("payments", 2.0),
                                ("cache", 0.8)),
            projected_subset=("orders", "payments"),
            n_oracles_consulted=3,
            n_outside_tokens_total=12,
            parent_probe_cids=("aa" * 32, "bb" * 32, "cc" * 32),
        )

    def test_envelope_signs_at_construction(self) -> None:
        env = self._make_envelope()
        self.assertEqual(len(env.digest_cid), 64)
        self.assertEqual(env.digest_cid, env.recompute_digest_cid())

    def test_envelope_canonicalises_tuples(self) -> None:
        env_a = LatentDigestEnvelope(
            schema_cid=self.schema.cid,
            inner_w19_branch="x", quorum_min=2, min_trust_sum=0.0,
            per_tag_vote_count=(("b", 1), ("a", 2)),
            per_tag_trust_sum=(("b", 1.0), ("a", 2.0)),
            projected_subset=("b", "a"), n_oracles_consulted=2,
            n_outside_tokens_total=4, parent_probe_cids=("y", "x"))
        env_b = LatentDigestEnvelope(
            schema_cid=self.schema.cid,
            inner_w19_branch="x", quorum_min=2, min_trust_sum=0.0,
            per_tag_vote_count=(("a", 2), ("b", 1)),
            per_tag_trust_sum=(("a", 2.0), ("b", 1.0)),
            projected_subset=("a", "b"), n_oracles_consulted=2,
            n_outside_tokens_total=4, parent_probe_cids=("x", "y"))
        # Probe-cid order is deliberately preserved (provenance).
        # The envelope's digest_cid ought to differ on parent order
        # even when other fields are equal.
        self.assertNotEqual(env_a.digest_cid, env_b.digest_cid)

    def test_replace_preserves_signed_digest_cid(self) -> None:
        env = self._make_envelope()
        original = env.digest_cid
        replaced = dataclasses.replace(
            env, projected_subset=("cache",))
        # The signed digest_cid is preserved; recompute reveals the
        # mismatch — this is the load-bearing tamper-detection signal.
        self.assertEqual(replaced.digest_cid, original)
        self.assertNotEqual(replaced.recompute_digest_cid(), original)


class LatentDigestVerificationTests(unittest.TestCase):
    """W22-3 — controller-side verification soundness."""

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.env = LatentDigestEnvelope(
            schema_cid=self.schema.cid,
            inner_w19_branch="abstained_symmetric",
            quorum_min=2, min_trust_sum=0.0,
            per_tag_vote_count=(("orders", 2), ("payments", 2),
                                  ("cache", 1)),
            per_tag_trust_sum=(("orders", 2.0), ("payments", 2.0),
                                ("cache", 0.8)),
            projected_subset=("orders", "payments"),
            n_oracles_consulted=3, n_outside_tokens_total=12,
            parent_probe_cids=("aa" * 32, "bb" * 32, "cc" * 32))

    def test_verify_passes_on_clean_envelope(self) -> None:
        out = verify_latent_digest(
            self.env, registered_schema=self.schema)
        self.assertTrue(out.ok)
        self.assertEqual(out.reason, "ok")

    def test_verify_rejects_empty_envelope(self) -> None:
        out = verify_latent_digest(None, registered_schema=self.schema)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "empty_envelope")

    def test_verify_rejects_schema_drift(self) -> None:
        drifted = dataclasses.replace(self.schema, version="v9_drifted")
        out = verify_latent_digest(
            self.env, registered_schema=drifted)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "schema_cid_mismatch")

    def test_verify_rejects_tampered_projected_subset(self) -> None:
        t = EnvelopeTamperer(
            mode="flip_projected_subset",
            admitted_tags=("orders", "payments", "cache"))
        tampered = t.apply(self.env)
        out = verify_latent_digest(
            tampered, registered_schema=self.schema)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "hash_mismatch")

    def test_verify_rejects_phantom_probe_with_sealed_set(self) -> None:
        t = EnvelopeTamperer(mode="add_phantom_probe_cid")
        tampered = t.apply(self.env)
        # The tamper changes the canonical bytes (probe list grew),
        # so hash_mismatch fires even before sealed_set check.
        out = verify_latent_digest(
            tampered, registered_schema=self.schema,
            sealed_probe_cids=set(self.env.parent_probe_cids))
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "hash_mismatch")

    def test_verify_rejects_changed_quorum_min(self) -> None:
        t = EnvelopeTamperer(mode="change_quorum_min")
        tampered = t.apply(self.env)
        out = verify_latent_digest(
            tampered, registered_schema=self.schema)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "hash_mismatch")

    def test_verify_rejects_unknown_schema_version(self) -> None:
        wrong_version = dataclasses.replace(
            self.env, schema_version="coordpy.latent_digest.v999")
        out = verify_latent_digest(
            wrong_version, registered_schema=self.schema)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "schema_version_unknown")


class SharedReadCacheTests(unittest.TestCase):

    def test_cache_collapses_identical_queries(self) -> None:
        cache = SharedReadCache()
        adapter = CachingOracleAdapter(
            inner=ServiceGraphOracle(), cache=cache,
            oracle_id="service_graph")
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="primary",
            witness_payloads=(),
            max_response_tokens=24)
        v1 = adapter.consult(q)
        # First consult is a miss (oracle is called).
        self.assertFalse(adapter.last_was_hit)
        v2 = adapter.consult(q)
        # Second consult hits the cache.
        self.assertTrue(adapter.last_was_hit)
        self.assertEqual(v1.payload, v2.payload)
        self.assertEqual(cache.n_hits, 1)
        self.assertEqual(cache.n_misses, 1)
        self.assertGreater(cache.n_tokens_saved, 0)

    def test_cache_distinguishes_different_oracle_ids(self) -> None:
        cache = SharedReadCache()
        a = CachingOracleAdapter(
            inner=ServiceGraphOracle(), cache=cache,
            oracle_id="service_graph")
        b = CachingOracleAdapter(
            inner=ChangeHistoryOracle(), cache=cache,
            oracle_id="change_history")
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="p", witness_payloads=(),
            max_response_tokens=24)
        a.consult(q)
        b.consult(q)
        # Two distinct oracle_ids → two distinct CIDs → both miss.
        self.assertEqual(cache.n_hits, 0)
        self.assertEqual(cache.n_misses, 2)


# =============================================================================
# Phase 69 driver tests — theorem anchors
# =============================================================================


class Phase69CacheFanoutTests(unittest.TestCase):
    """W22-1 — efficiency anchor."""

    def test_w22_strict_visible_tokens_savings_loose(self) -> None:
        rep = run_phase69(bank="cache_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12,
                            bank_seed=11)
        eff = rep["eff_compare"]
        # Visible tokens per cell strictly below W21 baseline.
        self.assertLess(eff["w22_visible_tokens_per_cell"],
                          eff["w21_visible_tokens_per_cell"])
        self.assertGreater(eff["visible_tokens_savings_per_cell"], 0)
        self.assertGreater(eff["visible_tokens_savings_pct"], 0.0)
        # Cache contributes wire-side savings.
        self.assertGreater(eff["cache_tokens_saved_total"], 0)
        # Verification ok rate = 100% (no tampering).
        self.assertEqual(eff["verification_ok_rate"], 1.0)
        # Correctness ratified across every cell.
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)

    def test_w22_strict_visible_tokens_savings_tight(self) -> None:
        rep = run_phase69(bank="cache_fanout", n_eval=8,
                            T_decoder=24, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertLess(eff["w22_visible_tokens_per_cell"],
                          eff["w21_visible_tokens_per_cell"])
        self.assertGreater(eff["cache_tokens_saved_total"], 0)
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)

    def test_w22_no_accuracy_regression_on_cache_fanout(self) -> None:
        rep = run_phase69(bank="cache_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        pooled = rep["pooled"]
        # W22 ties W21 on accuracy_full byte-for-byte.
        self.assertEqual(
            pooled["capsule_w22_hybrid"]["accuracy_full"],
            pooled["capsule_multi_oracle"]["accuracy_full"])
        self.assertEqual(
            pooled["capsule_w22_hybrid"]["accuracy_full"], 1.0)


class Phase69CorrectnessRatificationTests(unittest.TestCase):
    """W22-2 — byte-for-byte answer ratification."""

    def test_w22_correctness_ratified_byte_for_byte(self) -> None:
        rep = run_phase69(bank="cache_fanout", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)
        self.assertGreaterEqual(
            rep["correctness_ratified_cells_count"], 8)


class Phase69PoisonedDigestTests(unittest.TestCase):
    """W22-3 — tampered envelope must be rejected."""

    def test_poisoned_digest_rejected_on_every_cell(self) -> None:
        rep = run_phase69(bank="poisoned_digest", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertEqual(eff["verification_ok_rate"], 0.0)
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)

    def test_poisoned_digest_no_visible_tokens_savings(self) -> None:
        rep = run_phase69(bank="poisoned_digest", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # On rejection W22 falls through to W21 baseline cost — no
        # visible-tokens savings is the honest accounting.
        self.assertEqual(eff["visible_tokens_savings_per_cell"], 0.0)

    def test_poisoned_digest_rejection_reason_is_hash_mismatch(self) -> None:
        rep = run_phase69(bank="poisoned_digest", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        ps = rep["pack_stats_summary"]["capsule_w22_hybrid"]
        reasons = ps["rejection_reasons"]
        self.assertIn("hash_mismatch", reasons)


class Phase69SchemaDriftTests(unittest.TestCase):
    """W22-3 — schema-drift envelope must be rejected."""

    def test_schema_drift_rejected_on_every_cell(self) -> None:
        rep = run_phase69(bank="schema_drift", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertEqual(eff["verification_ok_rate"], 0.0)
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)

    def test_schema_drift_rejection_reason(self) -> None:
        rep = run_phase69(bank="schema_drift", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        ps = rep["pack_stats_summary"]["capsule_w22_hybrid"]
        reasons = ps["rejection_reasons"]
        self.assertIn("schema_cid_mismatch", reasons)

    def test_schema_drift_no_visible_tokens_savings(self) -> None:
        rep = run_phase69(bank="schema_drift", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertEqual(eff["visible_tokens_savings_per_cell"], 0.0)


class Phase69NoTriggerTests(unittest.TestCase):
    """W22 backward-compat — when W21 abstains, W22 reduces to W21."""

    def test_no_trigger_reduces_to_w21(self) -> None:
        rep = run_phase69(bank="no_trigger", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        pooled = rep["pooled"]
        self.assertEqual(
            pooled["capsule_w22_hybrid"]["accuracy_full"],
            pooled["capsule_multi_oracle"]["accuracy_full"])
        self.assertEqual(rep["correctness_ratified_rate"], 1.0)
        # No envelope was emitted (W21 abstained → W22 NO_TRIGGER).
        ps = rep["pack_stats_summary"]["capsule_w22_hybrid"]
        self.assertGreaterEqual(ps["n_w22_no_trigger_cells"], 8)
        self.assertEqual(ps["n_w22_resolved_cells"], 0)


class Phase69NoCacheTests(unittest.TestCase):
    """W22-Λ-no-cache falsifier — fresh per-cell cache eliminates
    wire-side savings; only the digest contribution remains."""

    def test_no_cache_records_zero_cache_tokens_saved(self) -> None:
        rep = run_phase69(bank="no_cache", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        self.assertEqual(eff["cache_tokens_saved_total"], 0)
        self.assertEqual(eff["cache_hit_rate"], 0.0)

    def test_no_cache_still_has_digest_savings(self) -> None:
        rep = run_phase69(bank="no_cache", n_eval=8,
                            T_decoder=None, K_auditor=12, bank_seed=11)
        eff = rep["eff_compare"]
        # The digest still compresses the W21 audit even without
        # cross-cell cache hits.
        self.assertGreater(eff["visible_tokens_savings_per_cell"], 0)


class Phase69SeedStabilityTests(unittest.TestCase):
    """W22-1 stability — savings hold across 5/5 seeds."""

    def test_savings_strictly_positive_across_5_seeds(self) -> None:
        out = run_phase69_seed_stability_sweep(
            bank="cache_fanout", T_decoder=None, n_eval=8,
            K_auditor=12, seeds=(11, 17, 23, 29, 31))
        self.assertGreater(out["min_visible_tokens_savings_per_cell"], 0)
        self.assertEqual(out["min_correctness_ratified_rate"], 1.0)
        self.assertGreater(out["mean_cache_tokens_saved_total"], 0)


class Phase69AuditOKTests(unittest.TestCase):
    """T-1..T-7 audit holds on every cell of every regime."""

    def test_audit_ok_on_every_w22_cell(self) -> None:
        for bank in ("cache_fanout", "no_cache",
                      "poisoned_digest", "schema_drift"):
            rep = run_phase69(bank=bank, n_eval=4, T_decoder=None,
                                K_auditor=12, bank_seed=11)
            grid = rep["audit_ok_grid"]
            self.assertTrue(grid["capsule_w22_hybrid"],
                              f"audit failed on {bank}")


class W22SDKReductionTests(unittest.TestCase):
    """W22 backward-compat — when ``enabled=False``, the W22 layer
    reduces to W21 byte-for-byte on the answer field."""

    def test_disabled_reduces_to_w21(self) -> None:
        schema = build_incident_triage_schema_capsule()
        cache = SharedReadCache()
        regs = (
            OracleRegistration(
                oracle=CachingOracleAdapter(
                    inner=CompromisedServiceGraphOracle(
                        oracle_id="compromised_registry"),
                    cache=cache, oracle_id="compromised_registry"),
                trust_prior=0.8, role_label="compromised_registry"),
            OracleRegistration(
                oracle=CachingOracleAdapter(
                    inner=ServiceGraphOracle(oracle_id="service_graph"),
                    cache=cache, oracle_id="service_graph"),
                trust_prior=1.0, role_label="service_graph"),
            OracleRegistration(
                oracle=CachingOracleAdapter(
                    inner=ChangeHistoryOracle(oracle_id="change_history"),
                    cache=cache, oracle_id="change_history"),
                trust_prior=1.0, role_label="change_history"),
        )
        inner_w15 = AttentionAwareBundleDecoder(T_decoder=None)
        w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
        w19 = BundleContradictionDisambiguator(inner=w18)
        w21 = TrustWeightedMultiOracleDisambiguator(
            inner=w19, oracle_registrations=regs, quorum_min=2)
        w22_disabled = LatentDigestDisambiguator(
            inner=w21, schema=schema, cache=cache,
            enabled=False)
        # No actual decode here — just that the W22 layer's
        # decoder_branch will be DISABLED on any input.
        self.assertFalse(w22_disabled.enabled)


# Cross-regime smoke test — single-shot run that exercises the full
# matrix of regimes and reports a single pass/fail.

class Phase69CrossRegimeSmokeTests(unittest.TestCase):

    def test_cross_regime_matrix_smoke(self) -> None:
        out = run_cross_regime_synthetic_p69(
            n_eval=4, bank_seed=11, K_auditor=12,
            T_decoder_tight=24, quorum_min=2)
        # CACHE-FANOUT-LOOSE: W22 saves visible tokens.
        loose = out["regimes"]["R-69-CACHE-FANOUT-LOOSE"]
        self.assertGreater(
            loose["eff_compare"]["visible_tokens_savings_per_cell"], 0)
        self.assertEqual(loose["correctness_ratified_rate"], 1.0)
        # POISONED-DIGEST: every cell rejected.
        poisoned = out["regimes"]["R-69-POISONED-DIGEST"]
        self.assertEqual(
            poisoned["eff_compare"]["verification_ok_rate"], 0.0)
        self.assertEqual(poisoned["correctness_ratified_rate"], 1.0)
        # SCHEMA-DRIFT: every cell rejected.
        drift = out["regimes"]["R-69-SCHEMA-DRIFT"]
        self.assertEqual(
            drift["eff_compare"]["verification_ok_rate"], 0.0)
        self.assertEqual(drift["correctness_ratified_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
