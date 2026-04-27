"""Tests for SDK v3.16 attention-aware context packing + W15 family.

Covers:

* Unit tests for :class:`CapsuleContextPacker` and
  :class:`FifoContextPacker` — salience scoring, hypothesis
  preservation, byte-for-byte determinism, budget enforcement
  (W15-2).
* Unit tests for :class:`AttentionAwareBundleDecoder` —
  backward-compat with the W13 layered decoder when
  ``T_decoder=None`` (W15-3) and pack-stats exposure.
* Phase-62 closure-property tests — multi-hypothesis comparable-
  magnitude invariants, asymmetric corroboration shape.
* Phase-62 default config tests — strict W15 separation under
  decoder-side budget pressure (W15-1) and saturation falsifier
  (W15-Λ-budget).
* Phase-62 5-seed stability — gap attention − fifo_pack ≥ 0.50 on
  every seed.
"""

from __future__ import annotations

import unittest

from vision_mvp.wevra.team_coord import (
    AttentionAwareBundleDecoder, CapsuleContextPacker,
    FifoContextPacker, LayeredRobustMultiRoundBundleDecoder,
    W15PackResult, W15PackedHandoff,
    W15_DEFAULT_CCK_WEIGHT, W15_DEFAULT_TIER_WEIGHT,
    W15_DEFAULT_CORROBORATION_WEIGHT, W15_DEFAULT_MAGNITUDE_WEIGHT,
    W15_DEFAULT_ROUND_WEIGHT,
    _DecodedHandoff, _SPECIFIC_TIER_CLAIM_KINDS,
    _payload_magnitude, _service_tag_of, _handoff_n_tokens,
)
from vision_mvp.experiments.phase62_attention_aware_packing import (
    IdentityExtractor, build_phase62_bank, run_phase62,
    run_cross_regime_summary,
    run_phase62_seed_stability_sweep,
    _bench_property,
)


# =============================================================================
# W15-2 — salience scoring + determinism
# =============================================================================


def _h(role: str, kind: str, payload: str) -> _DecodedHandoff:
    return _DecodedHandoff(source_role=role, claim_kind=kind, payload=payload)


class SalienceScoringTests(unittest.TestCase):

    def test_specific_tier_kind_set_includes_canonical_specific_kinds(self):
        for k in ("DEADLOCK_SUSPECTED", "POOL_EXHAUSTION",
                   "DISK_FILL_CRITICAL", "SLOW_QUERY_OBSERVED",
                   "OOM_KILL", "TLS_EXPIRED", "DNS_MISROUTE",
                   "CRON_OVERRUN"):
            self.assertIn(k, _SPECIFIC_TIER_CLAIM_KINDS,
                msg=f"{k!r} should be in _SPECIFIC_TIER_CLAIM_KINDS")

    def test_specific_tier_kind_set_excludes_generic_noise(self):
        for k in ("LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE"):
            self.assertNotIn(k, _SPECIFIC_TIER_CLAIM_KINDS,
                msg=f"{k!r} should be in generic-noise tier")

    def test_payload_magnitude_extracts_p95(self):
        self.assertAlmostEqual(_payload_magnitude("p95_ms=2500 service=x"),
                                  0.5, places=3)

    def test_payload_magnitude_extracts_error_rate(self):
        self.assertAlmostEqual(_payload_magnitude("error_rate=0.25 service=x"),
                                  0.5, places=3)

    def test_payload_magnitude_extracts_fw_count(self):
        self.assertAlmostEqual(_payload_magnitude("rule=deny count=15 service=x"),
                                  0.5, places=3)

    def test_payload_magnitude_returns_zero_when_no_field(self):
        self.assertEqual(_payload_magnitude("deadlock relation=foo"), 0.0)

    def test_service_tag_extraction(self):
        self.assertEqual(_service_tag_of("p95_ms=2100 service=orders"),
                          "orders")
        self.assertEqual(_service_tag_of("deadlock relation=orders_payments"),
                          "")

    def test_n_tokens_proxy(self):
        self.assertEqual(_handoff_n_tokens(_h("monitor", "LATENCY_SPIKE",
                                                  "p95_ms=2100 service=orders")), 2)
        self.assertEqual(_handoff_n_tokens(_h("monitor", "X", "")), 1)


class CapsuleContextPackerDeterminismTests(unittest.TestCase):

    def test_pack_is_byte_deterministic(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("network", "FW_BLOCK_SURGE", "rule=deny count=10 service=cache"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        a = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=None,
            round_index_hint=[1, 1, 2])
        b = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=None,
            round_index_hint=[1, 1, 2])
        self.assertEqual([k.handoff for k in a.kept],
                          [k.handoff for k in b.kept])
        self.assertEqual(a.position_of_first_causal_claim,
                          b.position_of_first_causal_claim)

    def test_round2_specific_at_rank_zero(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("monitor", "ERROR_RATE_SPIKE",
               "error_rate=0.22 service=payments"),
            _h("network", "FW_BLOCK_SURGE",
               "rule=deny count=10 service=cache"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        result = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=None,
            round_index_hint=[1, 1, 1, 2])
        self.assertEqual(result.position_of_first_causal_claim, 0)
        self.assertEqual(result.kept[0].handoff.claim_kind,
                          "DEADLOCK_SUSPECTED")

    def test_no_budget_keeps_everything(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("monitor", "ERROR_RATE_SPIKE",
               "error_rate=0.22 service=payments"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        result = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=None)
        self.assertEqual(result.n_kept, 3)
        self.assertEqual(result.n_dropped_budget, 0)

    def test_huge_budget_keeps_everything(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        result = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=99999)
        self.assertEqual(result.n_kept, 2)

    def test_tight_budget_drops_lowest_salience(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        # T_decoder = 3 forces only the round-2 specific claim
        # (n_tokens=3 for "deadlock relation=... wait_chain=2") to be
        # kept.
        result = CapsuleContextPacker().pack(
            hs, elected_root_cause="deadlock", T_decoder=3,
            round_index_hint=[1, 2])
        self.assertEqual(result.n_kept, 1)
        self.assertEqual(result.kept[0].handoff.claim_kind,
                          "DEADLOCK_SUSPECTED")
        self.assertEqual(result.n_dropped_budget, 1)

    def test_round_index_hint_length_must_match(self):
        hs = [_h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders")]
        with self.assertRaises(ValueError):
            CapsuleContextPacker().pack(
                hs, elected_root_cause="deadlock", T_decoder=None,
                round_index_hint=[1, 2, 3])

    def test_hypothesis_preservation_keeps_per_tag_role(self):
        """W15 multi-hypothesis: hypothesis preservation keeps (tag,
        role) representatives so multi-role decoys retain ≥ 2 distinct
        roles in the kept bundle (which is what the W11 noise-decoy
        drop needs to fire)."""
        # Two services: gold (1 role only) + decoy (2 roles).
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=4500 service=gold"),
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=decoy"),
            _h("network", "FW_BLOCK_SURGE",
               "rule=deny count=10 service=decoy"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=foo wait_chain=2"),
        ]
        # Tight budget — without hypothesis preservation the salience-
        # greedy packer would prefer decoy mentions (corr=2) over gold
        # mention (corr=1). With preservation, gold's monitor mention
        # AND decoy's network mention AND decoy's monitor mention all
        # survive (per-(tag, role) preservation).
        result = CapsuleContextPacker(preserve_hypotheses=True).pack(
            hs, elected_root_cause="deadlock", T_decoder=15,
            round_index_hint=[1, 1, 1, 2])
        kept_payloads = [k.handoff.payload for k in result.kept]
        self.assertTrue(any("service=gold" in p for p in kept_payloads),
            msg="hypothesis preservation should keep at least one "
                 "gold mention even when decoys are more corroborated")


class FifoContextPackerTests(unittest.TestCase):

    def test_fifo_keeps_arrival_order(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("monitor", "ERROR_RATE_SPIKE",
               "error_rate=0.22 service=payments"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=foo wait_chain=2"),
        ]
        result = FifoContextPacker().pack(
            hs, T_decoder=None)
        self.assertEqual([k.handoff for k in result.kept], hs)

    def test_fifo_drops_tail_under_budget(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE", "p95_ms=2100 service=orders"),
            _h("monitor", "ERROR_RATE_SPIKE",
               "error_rate=0.22 service=payments"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=foo wait_chain=2"),
        ]
        # Each n_tokens = 2, 2, 3. T_decoder=3 keeps first one.
        result = FifoContextPacker().pack(
            hs, T_decoder=3)
        self.assertEqual(result.n_kept, 1)
        self.assertEqual(result.kept[0].handoff.claim_kind,
                          "LATENCY_SPIKE")
        # Round-2 specific claim was DROPPED (the W15-Λ-budget failure
        # mode under FIFO).
        self.assertEqual(result.position_of_first_causal_claim, -1)


# =============================================================================
# W15-3 — backward-compat with W13 layered decoder
# =============================================================================


class AttentionAwareBackwardCompatTests(unittest.TestCase):

    def test_w15_no_budget_ties_w13_byte_for_byte(self):
        """W15-3: with ``T_decoder=None``, the W15 decoder reduces to
        the W13 layered decoder on the answer field."""
        hs = [
            _h("monitor", "LATENCY_SPIKE",
               "p95_ms=2100 service=orders"),
            _h("monitor", "ERROR_RATE_SPIKE",
               "error_rate=0.22 service=payments"),
            _h("monitor", "LATENCY_SPIKE",
               "p95_ms=180 service=cache"),
            _h("network", "FW_BLOCK_SURGE",
               "rule=deny count=10 service=cache"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=orders_payments_join wait_chain=2"),
        ]
        w15 = AttentionAwareBundleDecoder(T_decoder=None)
        w13 = LayeredRobustMultiRoundBundleDecoder()
        a15 = w15.decode_rounds([hs])
        a13 = w13.decode_rounds([hs])
        self.assertEqual(a15.get("root_cause"), a13.get("root_cause"))
        self.assertEqual(a15.get("services"), a13.get("services"))
        self.assertEqual(a15.get("remediation"), a13.get("remediation"))

    def test_w15_pack_stats_exposed(self):
        hs = [
            _h("monitor", "LATENCY_SPIKE",
               "p95_ms=2100 service=orders"),
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=foo wait_chain=2"),
        ]
        w15 = AttentionAwareBundleDecoder(T_decoder=None)
        ans = w15.decode_rounds([hs])
        self.assertIn("pack_stats", ans)
        self.assertIn("n_input", ans["pack_stats"])
        self.assertIn("n_kept", ans["pack_stats"])
        self.assertIn("position_of_first_causal_claim", ans["pack_stats"])

    def test_w15_first_pass_root_cause_recorded(self):
        hs = [
            _h("db_admin", "DEADLOCK_SUSPECTED",
               "deadlock relation=foo wait_chain=2"),
        ]
        w15 = AttentionAwareBundleDecoder(T_decoder=None)
        ans = w15.decode_rounds([hs])
        self.assertEqual(ans["first_pass_root_cause"], "deadlock")


# =============================================================================
# Phase-62 bank shape — multi-hypothesis closure invariants
# =============================================================================


class Phase62BankShapeTests(unittest.TestCase):

    def test_bank_size_default(self):
        bank = build_phase62_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 8)

    def test_every_scenario_has_two_distinct_decoys(self):
        bank = build_phase62_bank(n_replicates=2, seed=11)
        for sc in bank:
            ext = IdentityExtractor()
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            bp = _bench_property(sc, r1, r2)
            self.assertGreaterEqual(len(set(bp["decoys"])), 2,
                msg=f"{sc.scenario_id}: < 2 distinct decoys")

    def test_every_scenario_satisfies_r62_property(self):
        bank = build_phase62_bank(n_replicates=2, seed=11)
        ext = IdentityExtractor()
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            bp = _bench_property(sc, r1, r2)
            self.assertTrue(bp["r62_property_holds"],
                msg=f"{sc.scenario_id}: bench property fails: {bp}")

    def test_all_decoys_corroborated_all_golds_single_role(self):
        """Asymmetric corroboration property: decoys ≥ 2 roles, golds
        exactly 1 role. This is what makes W11 drop work for decoys
        but not golds."""
        bank = build_phase62_bank(n_replicates=2, seed=11)
        ext = IdentityExtractor()
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            bp = _bench_property(sc, r1, r2)
            self.assertTrue(bp["all_decoys_round1_corroborated"],
                msg=f"{sc.scenario_id}: decoy corroboration fails")
            self.assertTrue(bp["all_golds_single_role"],
                msg=f"{sc.scenario_id}: gold single-role property fails")


# =============================================================================
# Phase-62 default config — W15-1 + W15-Λ-budget anchors
# =============================================================================


class Phase62DefaultTests(unittest.TestCase):
    """Pre-committed default config:
    K_auditor=12, T_auditor=256, n_eval=8, bank_seed=11.
    """

    @classmethod
    def setUpClass(cls):
        cls.r_default = run_phase62(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=None, verbose=False)
        cls.r_tight = run_phase62(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=24, verbose=False)
        cls.r_saturation = run_phase62(
            n_eval=8, K_auditor=12, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            T_decoder=9999, verbose=False)

    def _gap(self, rep, a, b):
        return (rep["pooled"][a]["accuracy_full"]
                - rep["pooled"][b]["accuracy_full"])

    def test_default_property_holds_8_of_8(self):
        self.assertEqual(
            self.r_default["bench_summary"]["scenarios_with_property"], 8)

    def test_default_w15_ties_w13(self):
        """W15-3 backward-compat: with no budget pressure, W15
        accuracy ties the W13 layered decoder."""
        gap = self._gap(self.r_default,
                          "capsule_attention_aware",
                          "capsule_layered_multi_round")
        self.assertEqual(gap, 0.000,
            msg=f"W15-3: w15 - w13 = {gap}; expected 0.000")
        # Both at 1.000.
        self.assertEqual(
            self.r_default["pooled"]["capsule_attention_aware"]
                ["accuracy_full"], 1.000)

    def test_tightbudget_w15_strict_win_over_fifo_pack(self):
        """W15-1: under decoder-side budget pressure, salience-aware
        packing strictly beats FIFO packing on the cross-round bundle.
        """
        gap = self._gap(self.r_tight,
                          "capsule_attention_aware",
                          "capsule_layered_fifo_packed")
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W15-1: w15 - fifo_packed_layered = {gap}; "
                 f"expected ≥ 0.50")

    def test_tightbudget_fifo_pack_ties_substrate_fifo(self):
        """W15-Λ-budget structural sketch: under FIFO packing with a
        tight T_decoder, the cross-round decoder cannot retain the
        round-2 specific claim — every method ties FIFO at 0.000 by
        construction (the FIFO pack drops the disambiguator)."""
        self.assertEqual(
            self.r_tight["pooled"]["capsule_layered_fifo_packed"]
                ["accuracy_full"], 0.000)

    def test_tightbudget_w15_keeps_round2_specific_claim(self):
        """The W15 hypothesis-preserving packer puts the round-2
        specific claim at rank 0 in every cell."""
        ps = self.r_tight["pack_stats_summary"]["capsule_attention_aware"]
        # Position of first causal claim averaged over 8 cells; should be 0.0.
        self.assertEqual(ps["position_of_first_causal_claim_avg"], 0.0)
        # Round-2 specific claim kept in every cell.
        self.assertEqual(ps["n_with_causal_claim_kept"], 8)

    def test_tightbudget_fifo_pack_drops_round2_specific_claim(self):
        """Under FIFO packing the round-2 specific claim arrives last
        and is the first casualty of truncation."""
        ps = self.r_tight["pack_stats_summary"]["capsule_layered_fifo_packed"]
        self.assertEqual(ps["n_with_causal_claim_kept"], 0)
        self.assertEqual(ps["position_of_first_causal_claim_avg"], -1.0)

    def test_saturation_falsifier_w15_ties_fifo_pack(self):
        """W15-Λ-budget: under no budget pressure (T_decoder=9999),
        salience reordering is a no-op on the answer field. Both
        packers at 1.000."""
        gap = self._gap(self.r_saturation,
                          "capsule_attention_aware",
                          "capsule_layered_fifo_packed")
        self.assertEqual(gap, 0.000,
            msg=f"W15-Λ-budget: w15 - fifo_packed = {gap}; "
                 f"expected 0.000 under saturation budget")

    def test_audit_OK_on_every_capsule_strategy(self):
        for rep_name, rep in (("default", self.r_default),
                                ("tight", self.r_tight),
                                ("saturation", self.r_saturation)):
            for s, ok in rep["audit_ok_grid"].items():
                if s == "substrate":
                    continue
                self.assertTrue(ok,
                    msg=f"{rep_name}/{s} failed audit")

    def test_w15_token_efficiency_under_tight_budget(self):
        """Honest token-efficiency check: W15 on R-62-tightbudget uses
        ≤ FIFO-pack token sum (both are bounded by T_decoder by
        construction). Reported as a tokens_kept_over_input ratio."""
        ps_w15 = self.r_tight["pack_stats_summary"]["capsule_attention_aware"]
        ps_fifo = self.r_tight["pack_stats_summary"]["capsule_layered_fifo_packed"]
        self.assertLessEqual(
            ps_w15["tokens_kept_over_input"],
            1.0)
        self.assertLessEqual(
            ps_fifo["tokens_kept_over_input"],
            1.0)


# =============================================================================
# 5-seed stability — gap attention − fifo_pack ≥ 0.50 on every seed
# =============================================================================


class Phase62SeedStabilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sweep = run_phase62_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=8, K_auditor=12, T_auditor=256, T_decoder=24)

    def test_w15_one_thousand_on_every_seed(self):
        for seed, rep in self.sweep["per_seed"].items():
            v = (rep["pooled"]["capsule_attention_aware"]
                  ["accuracy_full"])
            self.assertEqual(v, 1.000,
                msg=f"seed={seed}: w15={v}, expected 1.000 under "
                     f"T_decoder=24")

    def test_fifo_pack_zero_on_every_seed(self):
        for seed, rep in self.sweep["per_seed"].items():
            v = (rep["pooled"]["capsule_layered_fifo_packed"]
                  ["accuracy_full"])
            self.assertEqual(v, 0.000,
                msg=f"seed={seed}: fifo_packed_layered={v}; expected "
                     f"0.000 under T_decoder=24")

    def test_gap_attention_minus_fifo_pack_holds_across_5_seeds(self):
        """W15-1 stability: for every seed, the salience-pack accuracy
        − FIFO-pack accuracy ≥ 0.50."""
        for seed in (11, 17, 23, 29, 31):
            gap = (self.sweep["per_seed"][seed]["headline_gap"]
                    ["attention_minus_fifo_packed"])
            self.assertGreaterEqual(gap, 0.50,
                msg=f"seed={seed}: gap = {gap}")


# =============================================================================
# Cross-regime — three Phase-62 sub-banks
# =============================================================================


class Phase62CrossRegimeTests(unittest.TestCase):

    def test_three_subbanks_separate_cleanly(self):
        rep = run_cross_regime_summary(n_eval=8, bank_seed=11)
        # Default (T_decoder=None) — sanity: w15 = layered = 1.000.
        d = rep["r62_default"]
        self.assertEqual(
            d["pooled"]["capsule_attention_aware"]["accuracy_full"],
            1.000)
        self.assertEqual(
            d["pooled"]["capsule_layered_multi_round"]["accuracy_full"],
            1.000)
        # Tight-budget — W15-1: w15 strict win.
        t = rep["r62_tightbudget"]
        self.assertEqual(
            t["pooled"]["capsule_attention_aware"]["accuracy_full"],
            1.000)
        self.assertEqual(
            t["pooled"]["capsule_layered_fifo_packed"]["accuracy_full"],
            0.000)
        # Saturation — W15-Λ-budget: both at 1.000.
        s = rep["r62_saturation"]
        self.assertEqual(
            s["pooled"]["capsule_attention_aware"]["accuracy_full"],
            1.000)
        self.assertEqual(
            s["pooled"]["capsule_layered_fifo_packed"]["accuracy_full"],
            1.000)


if __name__ == "__main__":
    unittest.main()
