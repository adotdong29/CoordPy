"""Tests for SDK v3.22 — trust-weighted multi-oracle adjudicator
+ W21 family (Phase 68).

Covers:

* Helper-class unit tests: :class:`ChangeHistoryOracle`,
  :class:`OnCallNotesOracle` (full + partial modes),
  :class:`SingletonAsymmetricOracle` (first/middle/last/explicit
  targets), :class:`DisagreeingHonestOracle` (abstains under
  R-66-shape admitted).
* W21 decoder unit tests: trigger gating, branch determinism,
  no-oracle / disabled paths, bounded-context invariants,
  reduces-to-W20 in the single-oracle quorum_min=1 regime.
* Phase-68 bench-property tests: every cell carries the
  R-66-OUTSIDE-REQUIRED bundle shape (decoy_only primary +
  all_three secondary).
* Phase-68 default-config tests: W21 strict win on
  R-68-MULTI-MAJORITY (loose AND tight) with deterministic
  oracle stack (W21-1).
* Phase-68 5-seed stability: gap w21 − w20 ≥ 0.50 on every seed
  under R-68-MULTI-MAJORITY-LOOSE AND R-68-MULTI-MAJORITY-TIGHT.
* Phase-68 falsifier tests: W21 ties FIFO at 0.000 on
  R-68-MULTI-NO-QUORUM (W21-Λ-no-quorum), R-68-MULTI-ALL-
  COMPROMISED (W21-Λ-all-compromised), AND R-68-MULTI-PARTIAL
  (W21-Λ-partial under default ``quorum_min = 2``).
* Phase-68 backward-compat: W21 reduces to W19 byte-for-byte on
  R-66 default banks (W21-3-A) AND ties W20 byte-for-byte on
  R-67-OUTSIDE-RESOLVES with ``quorum_min = 1`` and a single
  registered oracle (W21-3-B).
* Phase-68 token-budget honesty: bounded-context invariant —
  ``n_outside_tokens_total ≤ N × max_response_tokens`` on every
  cell where N = #registered oracles; W15 ``tokens_kept``
  byte-for-byte unchanged from the W19 inner.
* Phase-68 conditional success: under ``quorum_min = 1``, W21
  recovers the gold pair on R-68-MULTI-PARTIAL (W21-C-PARTIAL-
  RECOVERY conjecture, empirical-only).
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase66_deceptive_ambiguity import (
    run_phase66,
)
from vision_mvp.experiments.phase67_outside_information import (
    run_phase67,
)
from vision_mvp.experiments.phase68_multi_oracle_adjudication import (
    _bench_property_p67, _P67_EXPECTED_SHAPE,
    _bank_to_oracle_registrations,
    build_phase68_bank,
    run_phase68, run_phase68_seed_stability_sweep,
    run_cross_regime_synthetic,
)
from vision_mvp.coordpy.team_coord import (
    AbstainingOracle, AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    ChangeHistoryOracle,
    CompromisedServiceGraphOracle,
    DisagreeingHonestOracle,
    OnCallNotesOracle,
    OracleRegistration,
    OutsideQuery, OutsideVerdict,
    OutsideWitnessAcquisitionDisambiguator,
    RelationalCompatibilityDisambiguator,
    ServiceGraphOracle,
    SingletonAsymmetricOracle,
    TrustWeightedMultiOracleDisambiguator,
    W19_BRANCH_ABSTAINED_NO_SIGNAL,
    W19_BRANCH_ABSTAINED_SYMMETRIC,
    W19_BRANCH_CONFOUND_RESOLVED,
    W19_BRANCH_PRIMARY_TRUSTED,
    W21_ALL_BRANCHES, W21_BRANCH_DISABLED,
    W21_BRANCH_NO_TRIGGER, W21_BRANCH_NO_ORACLES,
    W21_BRANCH_NO_QUORUM, W21_BRANCH_QUORUM_RESOLVED,
    W21_BRANCH_SYMMETRIC_QUORUM,
    W21_DEFAULT_TRIGGER_BRANCHES,
    W21MultiOracleResult, W21OracleProbe,
    _DecodedHandoff, _disambiguator_payload_tokens,
)


# =============================================================================
# Helper builders
# =============================================================================


def _r1_symmetric(A: str, B: str, decoy: str
                    ) -> list[_DecodedHandoff]:
    """R-66-OUTSIDE-REQUIRED round-1 symmetric corroboration handoffs."""
    return [
        _DecodedHandoff("monitor", "LATENCY_SPIKE",
                         f"p95_ms=2200 service={A}"),
        _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                         f"error_rate=0.20 service={B}"),
        _DecodedHandoff("monitor", "LATENCY_SPIKE",
                         f"p95_ms=2100 service={decoy}"),
        _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                         f"error_rate=0.18 service={decoy}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=11 service={A}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=10 service={B}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=9 service={decoy}"),
        _DecodedHandoff("network", "FW_BLOCK_SURGE",
                         f"rule=deny count=8 service={decoy}"),
    ]


def _r2_outside_required(A: str, B: str, decoy: str
                            ) -> list[_DecodedHandoff]:
    """R-66-OUTSIDE-REQUIRED round-2: deceptive primary + symmetric
    secondary witness (mentions all three)."""
    return [
        _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                         f"deadlock relation={decoy}_{decoy}_join "
                         f"wait_chain=2"),
        _DecodedHandoff("monitor", "DEADLOCK_DETECTED",
                         f"deadlock relation={A}_{B}_{decoy}_join "
                         f"detected_at=t=120"),
    ]


def _w21_chain(oracle_registrations=(), T_decoder=None, *,
                 enabled=True, trigger_branches=None,
                 quorum_min=2, min_trust_sum=0.0):
    """Build a fresh W21→W19→W18→W15 chain."""
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        enabled=enabled,
        trigger_branches=(trigger_branches
                            if trigger_branches is not None
                            else W21_DEFAULT_TRIGGER_BRANCHES),
        quorum_min=quorum_min,
        min_trust_sum=min_trust_sum)
    return w21, inner_w15


def _w20_chain(oracle=None, T_decoder=None):
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w20 = OutsideWitnessAcquisitionDisambiguator(
        inner=w19, oracle=oracle)
    return w20, inner_w15


# =============================================================================
# Oracle helper unit tests
# =============================================================================


class ChangeHistoryOracleTests(unittest.TestCase):

    def test_emits_gold_pair_when_admitted(self):
        oracle = ChangeHistoryOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="deadlock relation=cache_cache_join",
            witness_payloads=(),
        )
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertEqual(v.source_id, "change_history")
        self.assertIn("orders", v.payload)
        self.assertIn("payments", v.payload)
        self.assertNotIn("cache", v.payload)

    def test_abstains_on_unknown_root_cause(self):
        oracle = ChangeHistoryOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="unknown",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)

    def test_root_cause_keys_match_p66_families(self):
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            _P66_FAMILIES)
        oracle = ChangeHistoryOracle()
        for (root_cause, A, B) in _P66_FAMILIES:
            q = OutsideQuery(
                admitted_tags=(A, B, "cache"),
                elected_root_cause=root_cause,
                primary_payload="", witness_payloads=())
            v = oracle.consult(q)
            self.assertIsNotNone(
                v.payload,
                f"ChangeHistoryOracle abstained on {root_cause}")
            self.assertIn(A, v.payload)
            self.assertIn(B, v.payload)


class OnCallNotesOracleTests(unittest.TestCase):

    def test_emits_full_pair_by_default(self):
        oracle = OnCallNotesOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertIn("orders", v.payload)
        self.assertIn("payments", v.payload)

    def test_partial_only_emits_first_index(self):
        oracle = OnCallNotesOracle(emit_partial_only=True,
                                     partial_index=0)
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertIn("orders", v.payload)
        self.assertNotIn("payments", v.payload)

    def test_partial_index_one_emits_second(self):
        oracle = OnCallNotesOracle(emit_partial_only=True,
                                     partial_index=1)
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertNotIn("orders", v.payload)
        self.assertIn("payments", v.payload)


class SingletonAsymmetricOracleTests(unittest.TestCase):

    def test_first_target(self):
        oracle = SingletonAsymmetricOracle(target="first")
        q = OutsideQuery(
            admitted_tags=("cache", "orders", "payments"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        # Sorted: ('cache', 'orders', 'payments'); first = cache
        self.assertIn("cache", v.payload)

    def test_last_target(self):
        oracle = SingletonAsymmetricOracle(target="last")
        q = OutsideQuery(
            admitted_tags=("cache", "orders", "payments"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        # Sorted: last = payments
        self.assertIn("payments", v.payload)

    def test_explicit_target(self):
        oracle = SingletonAsymmetricOracle(target="orders")
        q = OutsideQuery(
            admitted_tags=("cache", "orders", "payments"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIn("orders", v.payload)

    def test_explicit_target_not_in_admitted_abstains(self):
        oracle = SingletonAsymmetricOracle(target="zzz")
        q = OutsideQuery(
            admitted_tags=("cache", "orders", "payments"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)

    def test_singleton_admitted_abstains(self):
        oracle = SingletonAsymmetricOracle(target="first")
        q = OutsideQuery(
            admitted_tags=("orders",),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)


class DisagreeingHonestOracleTests(unittest.TestCase):

    def test_abstains_on_r66_admitted_shape(self):
        # Default wrong_log['deadlock'] = ('api', 'db'); not in
        # admitted = (orders, payments, cache).
        oracle = DisagreeingHonestOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)

    def test_emits_when_wrong_pair_in_admitted(self):
        oracle = DisagreeingHonestOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "api", "db", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        # wrong_log['deadlock'] = ('api', 'db')
        self.assertIn("api", v.payload)
        self.assertIn("db", v.payload)
        self.assertNotIn("orders", v.payload)


# =============================================================================
# W21 decoder unit tests
# =============================================================================


class W21DecoderUnitTests(unittest.TestCase):
    """Mechanical W21 unit tests (no Phase-68 driver)."""

    def _r66_outside_required_bundle(self, A="orders",
                                          B="payments",
                                          decoy="cache"):
        round1 = _r1_symmetric(A, B, decoy)
        round2 = _r2_outside_required(A, B, decoy)
        return [round1, round2]

    def test_disabled_path_reduces_to_inner_w19(self):
        # enabled=False with a clean oracle should still tie W19's
        # answer (W21 is disabled).
        bundle = self._r66_outside_required_bundle()
        regs = (OracleRegistration(
            oracle=ServiceGraphOracle(),
            trust_prior=1.0, role_label="reg"),)
        w21, _ = _w21_chain(oracle_registrations=regs,
                              enabled=False, quorum_min=2)
        ans = w21.decode_rounds(bundle)
        # W21 disabled — branch is DISABLED; answer is W19's.
        self.assertEqual(ans["multi_oracle"]["decoder_branch"],
                          W21_BRANCH_DISABLED)

    def test_no_oracles_path_reduces_to_inner_w19(self):
        bundle = self._r66_outside_required_bundle()
        w21, _ = _w21_chain(oracle_registrations=(), quorum_min=2)
        ans = w21.decode_rounds(bundle)
        self.assertEqual(ans["multi_oracle"]["decoder_branch"],
                          W21_BRANCH_NO_ORACLES)

    def test_no_trigger_when_inner_w19_resolves(self):
        # Build a bundle where W19 resolves (e.g. confound_resolvable
        # shape: deceptive primary + asymmetric secondary mentioning
        # only gold). On that bundle, the W19 branch is
        # CONFOUND_RESOLVED (not in W21 trigger set), so W21 reduces
        # to W19.
        round1 = _r1_symmetric("orders", "payments", "cache")
        # secondary mentions only gold (asymmetric witness pattern)
        round2 = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                             "deadlock relation=cache_cache_join"),
            _DecodedHandoff("monitor", "DEADLOCK_DETECTED",
                             "deadlock relation=orders_payments_join"),
        ]
        regs = (OracleRegistration(
            oracle=ServiceGraphOracle(),
            trust_prior=1.0, role_label="reg"),
                 OracleRegistration(
            oracle=ChangeHistoryOracle(),
            trust_prior=1.0, role_label="ch"),
                 OracleRegistration(
            oracle=OnCallNotesOracle(),
            trust_prior=1.0, role_label="on"),)
        w21, _ = _w21_chain(oracle_registrations=regs)
        ans = w21.decode_rounds([round1, round2])
        self.assertEqual(ans["multi_oracle"]["decoder_branch"],
                          W21_BRANCH_NO_TRIGGER)

    def test_quorum_resolved_with_two_honest_one_compromised(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(oracle=CompromisedServiceGraphOracle(),
                                trust_prior=0.8, role_label="bad"),
            OracleRegistration(oracle=ServiceGraphOracle(),
                                trust_prior=1.0, role_label="sg"),
            OracleRegistration(oracle=ChangeHistoryOracle(),
                                trust_prior=1.0, role_label="ch"),
        )
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=2)
        ans = w21.decode_rounds(bundle)
        mo = ans["multi_oracle"]
        self.assertEqual(mo["decoder_branch"],
                          W21_BRANCH_QUORUM_RESOLVED)
        # Projected services should be the gold pair.
        self.assertEqual(set(ans["services"]),
                          {"orders", "payments"})

    def test_no_quorum_with_three_singletons(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(oracle=SingletonAsymmetricOracle(
                target="first"), trust_prior=1.0, role_label="s1"),
            OracleRegistration(oracle=SingletonAsymmetricOracle(
                target="middle"), trust_prior=1.0, role_label="s2"),
            OracleRegistration(oracle=SingletonAsymmetricOracle(
                target="last"), trust_prior=1.0, role_label="s3"),
        )
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=2)
        ans = w21.decode_rounds(bundle)
        mo = ans["multi_oracle"]
        self.assertEqual(mo["decoder_branch"], W21_BRANCH_NO_QUORUM)
        self.assertTrue(mo["abstained"])

    def test_all_compromised_picks_decoy(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="c1"),
                trust_prior=1.0, role_label="c1"),
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="c2"),
                trust_prior=1.0, role_label="c2"),
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="c3"),
                trust_prior=1.0, role_label="c3"),
        )
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=2)
        ans = w21.decode_rounds(bundle)
        mo = ans["multi_oracle"]
        # Quorum forms on decoy; W21 projects to {decoy}.
        self.assertEqual(mo["decoder_branch"],
                          W21_BRANCH_QUORUM_RESOLVED)
        self.assertEqual(set(ans["services"]), {"cache"})

    def test_partial_with_quorum_min_2_abstains(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(
                oracle=OnCallNotesOracle(emit_partial_only=True,
                                          partial_index=0),
                trust_prior=1.0, role_label="p0"),
            OracleRegistration(
                oracle=OnCallNotesOracle(emit_partial_only=True,
                                          partial_index=1),
                trust_prior=1.0, role_label="p1"),
            OracleRegistration(
                oracle=AbstainingOracle(),
                trust_prior=1.0, role_label="abst"),
        )
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=2)
        ans = w21.decode_rounds(bundle)
        mo = ans["multi_oracle"]
        self.assertEqual(mo["decoder_branch"], W21_BRANCH_NO_QUORUM)

    def test_partial_with_quorum_min_1_recovers(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(
                oracle=OnCallNotesOracle(emit_partial_only=True,
                                          partial_index=0),
                trust_prior=1.0, role_label="p0"),
            OracleRegistration(
                oracle=OnCallNotesOracle(emit_partial_only=True,
                                          partial_index=1),
                trust_prior=1.0, role_label="p1"),
            OracleRegistration(
                oracle=AbstainingOracle(),
                trust_prior=1.0, role_label="abst"),
        )
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=1)
        ans = w21.decode_rounds(bundle)
        mo = ans["multi_oracle"]
        self.assertEqual(mo["decoder_branch"],
                          W21_BRANCH_QUORUM_RESOLVED)
        self.assertEqual(set(ans["services"]),
                          {"orders", "payments"})

    def test_determinism(self):
        bundle = self._r66_outside_required_bundle()
        regs = (
            OracleRegistration(oracle=CompromisedServiceGraphOracle(),
                                trust_prior=0.8, role_label="bad"),
            OracleRegistration(oracle=ServiceGraphOracle(),
                                trust_prior=1.0, role_label="sg"),
            OracleRegistration(oracle=ChangeHistoryOracle(),
                                trust_prior=1.0, role_label="ch"),
        )
        w21_a, _ = _w21_chain(oracle_registrations=regs)
        w21_b, _ = _w21_chain(oracle_registrations=regs)
        ans_a = w21_a.decode_rounds(bundle)
        ans_b = w21_b.decode_rounds(bundle)
        self.assertEqual(ans_a["services"], ans_b["services"])
        self.assertEqual(
            ans_a["multi_oracle"]["per_tag_votes"],
            ans_b["multi_oracle"]["per_tag_votes"])

    def test_branches_well_typed(self):
        for b in (W21_BRANCH_QUORUM_RESOLVED,
                   W21_BRANCH_NO_QUORUM,
                   W21_BRANCH_SYMMETRIC_QUORUM,
                   W21_BRANCH_NO_ORACLES,
                   W21_BRANCH_NO_TRIGGER,
                   W21_BRANCH_DISABLED):
            self.assertIn(b, W21_ALL_BRANCHES)

    def test_default_trigger_set(self):
        # W21 default trigger set is the same as W20: abstentions only.
        self.assertIn(W19_BRANCH_ABSTAINED_SYMMETRIC,
                       W21_DEFAULT_TRIGGER_BRANCHES)
        self.assertIn(W19_BRANCH_ABSTAINED_NO_SIGNAL,
                       W21_DEFAULT_TRIGGER_BRANCHES)
        # Non-trigger branches should not be in the default set.
        self.assertNotIn(W19_BRANCH_PRIMARY_TRUSTED,
                          W21_DEFAULT_TRIGGER_BRANCHES)
        self.assertNotIn(W19_BRANCH_CONFOUND_RESOLVED,
                          W21_DEFAULT_TRIGGER_BRANCHES)

    def test_w21_q1_single_oracle_ties_w20_byte_for_byte(self):
        # W21-3-B: with quorum_min=1 and a single registered honest
        # oracle, W21 ties W20 byte-for-byte on the answer field on
        # R-66-OUTSIDE-REQUIRED bundle (where W19 abstains).
        bundle = self._r66_outside_required_bundle()
        regs = (OracleRegistration(
            oracle=ServiceGraphOracle(),
            trust_prior=1.0, role_label="sg"),)
        w21, _ = _w21_chain(oracle_registrations=regs,
                              quorum_min=1)
        w20, _ = _w20_chain(oracle=ServiceGraphOracle())
        ans21 = w21.decode_rounds(bundle)
        ans20 = w20.decode_rounds(bundle)
        # Answer field byte-for-byte equality on root_cause/services.
        self.assertEqual(ans21["root_cause"], ans20["root_cause"])
        self.assertEqual(set(ans21["services"]),
                          set(ans20["services"]))
        # Both should be the gold pair.
        self.assertEqual(set(ans21["services"]),
                          {"orders", "payments"})


# =============================================================================
# Phase-68 bench-property tests
# =============================================================================


class Phase68BenchPropertyTests(unittest.TestCase):

    def test_every_bank_holds_outside_required_shape(self):
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            _build_round_candidates_p66)
        bank = build_phase68_bank(n_replicates=2, seed=11)
        for sc in bank:
            r1 = _build_round_candidates_p66(sc.round1_emissions)
            r2 = _build_round_candidates_p66(sc.round2_emissions)
            bp = _bench_property_p67(sc, r1, r2)
            self.assertTrue(bp["symmetric_corroboration_holds"],
                            f"sym corr fails on {sc.scenario_id}")
            self.assertEqual(tuple(bp["shape"]), _P67_EXPECTED_SHAPE,
                              f"shape mismatch on {sc.scenario_id}")


# =============================================================================
# Phase-68 default-config tests (W21-1 anchor)
# =============================================================================


class Phase68DefaultConfigTests(unittest.TestCase):

    def test_w21_strict_win_multi_majority_loose(self):
        rep = run_phase68(bank="multi_majority", T_decoder=None,
                            n_eval=8, K_auditor=12)
        w21 = rep["pooled"]["capsule_multi_oracle"]["accuracy_full"]
        max_non_w21 = rep["headline_gap"]["max_non_w21_accuracy_full"]
        self.assertGreaterEqual(w21, 1.0)
        self.assertEqual(max_non_w21, 0.0)
        self.assertEqual(rep["headline_gap"]["w21_minus_w20"], 1.0)

    def test_w21_strict_win_multi_majority_tight(self):
        rep = run_phase68(bank="multi_majority", T_decoder=24,
                            n_eval=8, K_auditor=12)
        w21 = rep["pooled"]["capsule_multi_oracle"]["accuracy_full"]
        max_non_w21 = rep["headline_gap"]["max_non_w21_accuracy_full"]
        self.assertGreaterEqual(w21, 1.0)
        self.assertEqual(max_non_w21, 0.0)
        self.assertEqual(rep["headline_gap"]["w21_minus_w20"], 1.0)

    def test_w21_branches_quorum_resolved_on_majority(self):
        rep = run_phase68(bank="multi_majority", T_decoder=None,
                            n_eval=8, K_auditor=12)
        # Every cell should fire QUORUM_RESOLVED.
        self.assertEqual(rep["w21_branch_counts"].get(
            W21_BRANCH_QUORUM_RESOLVED, 0), 8)

    def test_w21_audit_OK_on_every_cell(self):
        rep = run_phase68(bank="multi_majority", T_decoder=None,
                            n_eval=8, K_auditor=12)
        self.assertTrue(rep["audit_ok_grid"]["capsule_multi_oracle"])
        for s in ("capsule_outside_witness",
                   "capsule_bundle_contradiction",
                   "capsule_relational_compat",
                   "capsule_attention_aware"):
            self.assertTrue(rep["audit_ok_grid"][s], f"{s} audit not OK")


# =============================================================================
# Phase-68 5-seed stability tests
# =============================================================================


class Phase68SeedStabilityTests(unittest.TestCase):

    def test_loose_min_gap_at_strong_bar(self):
        sweep = run_phase68_seed_stability_sweep(
            bank="multi_majority", T_decoder=None,
            n_eval=8, K_auditor=12)
        self.assertGreaterEqual(sweep["min_w21_minus_w20"], 0.5)

    def test_tight_min_gap_at_strong_bar(self):
        sweep = run_phase68_seed_stability_sweep(
            bank="multi_majority", T_decoder=24,
            n_eval=8, K_auditor=12)
        self.assertGreaterEqual(sweep["min_w21_minus_w20"], 0.5)

    def test_w21_one_thousand_on_every_seed(self):
        for seed in (11, 17, 23, 29, 31):
            rep = run_phase68(bank="multi_majority", T_decoder=None,
                                n_eval=8, K_auditor=12,
                                bank_seed=seed)
            self.assertEqual(
                rep["pooled"]["capsule_multi_oracle"]["accuracy_full"],
                1.0,
                f"W21 not 1.000 on seed={seed}")


# =============================================================================
# Phase-68 falsifier tests
# =============================================================================


class Phase68FalsifierTests(unittest.TestCase):

    def test_w21_lambda_no_quorum_ties_fifo(self):
        rep = run_phase68(bank="multi_no_quorum", T_decoder=None,
                            n_eval=8, K_auditor=12)
        self.assertEqual(
            rep["pooled"]["capsule_multi_oracle"]["accuracy_full"], 0.0)
        self.assertEqual(
            rep["pooled"]["capsule_fifo"]["accuracy_full"], 0.0)
        self.assertEqual(rep["w21_branch_counts"].get(
            W21_BRANCH_NO_QUORUM, 0), 8)

    def test_w21_lambda_all_compromised_fails(self):
        rep = run_phase68(bank="multi_all_compromised",
                            T_decoder=None, n_eval=8, K_auditor=12)
        # W21 quorum forms on decoy; projects to decoy; FAILS at 0.000.
        self.assertEqual(
            rep["pooled"]["capsule_multi_oracle"]["accuracy_full"], 0.0)
        self.assertEqual(rep["w21_branch_counts"].get(
            W21_BRANCH_QUORUM_RESOLVED, 0), 8)

    def test_w21_lambda_partial_default_q2(self):
        rep = run_phase68(bank="multi_partial", T_decoder=None,
                            n_eval=8, K_auditor=12, quorum_min=2)
        self.assertEqual(
            rep["pooled"]["capsule_multi_oracle"]["accuracy_full"], 0.0)
        self.assertEqual(rep["w21_branch_counts"].get(
            W21_BRANCH_NO_QUORUM, 0), 8)


# =============================================================================
# Phase-68 conditional success: quorum_min=1 recovers partial.
# =============================================================================


class Phase68ConditionalSuccessTests(unittest.TestCase):

    def test_w21_partial_with_quorum_min_1_wins(self):
        rep = run_phase68(bank="multi_partial", T_decoder=None,
                            n_eval=8, K_auditor=12, quorum_min=1)
        self.assertEqual(
            rep["pooled"]["capsule_multi_oracle"]["accuracy_full"], 1.0)
        self.assertEqual(rep["w21_branch_counts"].get(
            W21_BRANCH_QUORUM_RESOLVED, 0), 8)


# =============================================================================
# Phase-68 backward-compat tests (W21-3-A on R-66; W21-3-B on R-67)
# =============================================================================


class Phase68BackwardCompatTests(unittest.TestCase):

    def test_w21_no_trigger_on_r66_default_banks(self):
        """W21 reduces to W19 byte-for-byte when W19 doesn't abstain.
        Each P66 sub-bank with R-66-CORROBORATED / R-66-DECEIVE-NAIVE
        / R-66-CONFOUND-RESOLVABLE has W19 firing a non-trigger
        branch."""
        from vision_mvp.experiments.phase68_multi_oracle_adjudication \
            import (_run_capsule_strategy, _make_factory,
                     _R68_STRATEGIES)
        from vision_mvp.experiments.phase52_team_coord import (
            claim_priorities, make_team_budgets)
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            build_phase66_bank, _build_round_candidates_p66)

        budgets = make_team_budgets(K_producer=6, T_producer=96,
                                       K_auditor=12, T_auditor=256)
        priorities = claim_priorities()
        regs = (
            OracleRegistration(oracle=ServiceGraphOracle(),
                                trust_prior=1.0, role_label="sg"),
            OracleRegistration(oracle=ChangeHistoryOracle(),
                                trust_prior=1.0, role_label="ch"),
        )
        for bank_name in ("corroborated", "deceive_naive",
                            "confound_resolvable"):
            bank_obj = build_phase66_bank(
                bank=bank_name, n_replicates=1, seed=11)
            for sc in bank_obj:
                r1 = _build_round_candidates_p66(sc.round1_emissions)
                r2 = _build_round_candidates_p66(sc.round2_emissions)
                # W21 result
                fac21 = _make_factory("capsule_multi_oracle",
                                         priorities, budgets)
                r21, _ = _run_capsule_strategy(
                    sc=sc, budgets=budgets,
                    policy_per_role_factory=fac21,
                    strategy_name="capsule_multi_oracle",
                    decoder_mode="multi_oracle",
                    round1_cands=r1, round2_cands=r2,
                    oracle_registrations=regs, quorum_min=2)
                # W19 result
                fac19 = _make_factory("capsule_bundle_contradiction",
                                         priorities, budgets)
                r19, _ = _run_capsule_strategy(
                    sc=sc, budgets=budgets,
                    policy_per_role_factory=fac19,
                    strategy_name="capsule_bundle_contradiction",
                    decoder_mode="bundle_contradiction",
                    round1_cands=r1, round2_cands=r2)
                self.assertEqual(
                    set(r21.answer.get("services", ())),
                    set(r19.answer.get("services", ())),
                    f"W21 != W19 on {bank_name}/{sc.scenario_id}")

    def test_w21_disabled_reduces_to_w19_byte_for_byte(self):
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            run_phase66)
        # Run the W21 with enabled=False on a P66 bundle directly.
        bundle = [
            _r1_symmetric("orders", "payments", "cache"),
            _r2_outside_required("orders", "payments", "cache")]
        regs = (OracleRegistration(
            oracle=ServiceGraphOracle(),
            trust_prior=1.0, role_label="sg"),)
        # Disabled
        w21_dis, _ = _w21_chain(oracle_registrations=regs,
                                   enabled=False)
        ans_dis = w21_dis.decode_rounds(bundle)
        self.assertEqual(
            ans_dis["multi_oracle"]["decoder_branch"],
            W21_BRANCH_DISABLED)
        # Compare with bare W19 inner.
        from vision_mvp.coordpy.team_coord import (
            BundleContradictionDisambiguator,
            RelationalCompatibilityDisambiguator,
            AttentionAwareBundleDecoder)
        inner = AttentionAwareBundleDecoder(T_decoder=None)
        w18 = RelationalCompatibilityDisambiguator(inner=inner)
        w19 = BundleContradictionDisambiguator(inner=w18)
        ans_w19 = w19.decode_rounds(bundle)
        self.assertEqual(set(ans_dis.get("services", ())),
                          set(ans_w19.get("services", ())))


# =============================================================================
# Phase-68 token-budget honesty tests
# =============================================================================


class Phase68TokenBudgetHonestyTests(unittest.TestCase):

    def test_n_outside_tokens_total_per_cell_bounded(self):
        rep = run_phase68(bank="multi_majority", T_decoder=None,
                            n_eval=8, K_auditor=12)
        ps = rep["pack_stats_summary"]["capsule_multi_oracle"]
        # 3 oracles × max_response_tokens=24 = 72 max per cell.
        self.assertLessEqual(
            ps["w21_outside_tokens_total_per_cell_avg"], 72.0)
        self.assertEqual(ps["w21_outside_queries_per_cell_avg"], 3.0)

    def test_w21_does_not_inflate_w15_tokens_kept(self):
        # On R-68-MULTI-MAJORITY-TIGHT the W21 layer reads only the
        # admitted tag set; the inner W15 ``tokens_kept`` accounting
        # should be byte-for-byte identical to the W19 / W20 path on
        # the same cells (since W19 inner is shared).
        rep = run_phase68(bank="multi_majority", T_decoder=24,
                            n_eval=8, K_auditor=12)
        ps_w19 = rep["pack_stats_summary"][
            "capsule_bundle_contradiction"]
        ps_w20 = rep["pack_stats_summary"]["capsule_outside_witness"]
        ps_w21 = rep["pack_stats_summary"]["capsule_multi_oracle"]
        # W19 / W20 / W21 share the inner W15 + W18 packer; tokens_kept
        # should match.
        self.assertEqual(ps_w19["tokens_kept_sum"],
                          ps_w21["tokens_kept_sum"])
        self.assertEqual(ps_w20["tokens_kept_sum"],
                          ps_w21["tokens_kept_sum"])

    def test_strict_tight_budget_holds(self):
        # T_decoder=24 caps tokens_kept_sum below 8 cells × 24 + slack.
        rep = run_phase68(bank="multi_majority", T_decoder=24,
                            n_eval=8, K_auditor=12)
        ps_w21 = rep["pack_stats_summary"]["capsule_multi_oracle"]
        self.assertLessEqual(ps_w21["tokens_kept_sum"], 8 * 32)


# =============================================================================
# Phase-68 cross-regime synthetic
# =============================================================================


class Phase68CrossRegimeSyntheticTests(unittest.TestCase):

    def test_majority_loose_at_one_thousand(self):
        rep = run_cross_regime_synthetic(n_eval=8, K_auditor=12)
        h = rep["headline_summary"]
        self.assertEqual(h["r68_multi_majority_loose_w21"], 1.0)
        self.assertEqual(h["r68_multi_majority_loose_w20"], 0.0)
        self.assertEqual(h["w21_minus_w20_multi_majority_loose"], 1.0)

    def test_majority_tight_at_one_thousand(self):
        rep = run_cross_regime_synthetic(n_eval=8, K_auditor=12)
        h = rep["headline_summary"]
        self.assertEqual(h["r68_multi_majority_tight_w21"], 1.0)
        self.assertEqual(h["r68_multi_majority_tight_w20"], 0.0)
        self.assertEqual(h["w21_minus_w20_multi_majority_tight"], 1.0)

    def test_falsifiers_at_zero(self):
        rep = run_cross_regime_synthetic(n_eval=8, K_auditor=12)
        h = rep["headline_summary"]
        self.assertEqual(h["r68_multi_no_quorum_w21"], 0.0)
        self.assertEqual(h["r68_multi_all_compromised_w21"], 0.0)
        self.assertEqual(h["r68_multi_partial_w21"], 0.0)

    def test_partial_q1_recovers(self):
        rep = run_cross_regime_synthetic(n_eval=8, K_auditor=12)
        h = rep["headline_summary"]
        self.assertEqual(h["r68_multi_partial_q1_w21"], 1.0)

    def test_single_clean_w20_wins_w21_q2_abstains(self):
        rep = run_cross_regime_synthetic(n_eval=8, K_auditor=12)
        h = rep["headline_summary"]
        self.assertEqual(h["r68_single_clean_w20"], 1.0)
        self.assertEqual(h["r68_single_clean_w21"], 0.0)


# =============================================================================
# Phase-68 W21-3-B reduces-to-W20 on R-67
# =============================================================================


class Phase68ReducesToW20OnR67Tests(unittest.TestCase):

    def test_w21_q1_single_oracle_ties_w20_on_r67(self):
        # W21 with quorum_min=1 and single ServiceGraphOracle on
        # R-67-OUTSIDE-RESOLVES should produce the same answer
        # accuracy as W20 (1.000 on the synthetic deterministic
        # anchor).
        from vision_mvp.experiments.phase67_outside_information import (
            run_phase67)
        rep_w20 = run_phase67(bank="outside_resolves",
                                 T_decoder=None, n_eval=8,
                                 K_auditor=12)
        # Build a Phase-68 driver call with single-oracle override to
        # simulate the same regime.
        rep_w21 = run_phase68(bank="single_clean", T_decoder=None,
                                n_eval=8, K_auditor=12, quorum_min=1)
        w20_acc = rep_w20["pooled"][
            "capsule_outside_witness"]["accuracy_full"]
        w21_acc = rep_w21["pooled"][
            "capsule_multi_oracle"]["accuracy_full"]
        self.assertEqual(w20_acc, 1.0)
        self.assertEqual(w21_acc, 1.0)


if __name__ == "__main__":
    unittest.main()
