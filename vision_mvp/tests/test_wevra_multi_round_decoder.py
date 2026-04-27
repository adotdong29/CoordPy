"""Tests for SDK v3.12 multi-round bundle decoder + W11 family.

Covers:

* Unit tests for :class:`MultiRoundBundleDecoder` — single-bundle
  reduction, round-union, contradiction-aware noise-decoy drop.
* Phase-58 bank-shape tests (the delayed-causal-evidence property
  is mechanically verified).
* Phase-58 default-config tests: W11-Λ single-round limit + W11-1
  multi-round strict win + 5/5 seed stability.
* Phase-58 falsifier tests (W11-4 — round-1 noise floods budget).
* Cross-regime backward-compat tests (W11-3) — single-round
  application of the W11 decoder matches W10 on R-54/R-55/R-56/R-57.
* Lifecycle-audit (T-1..T-7) preserved on every cell of Phase 58.
"""

from __future__ import annotations

import unittest

from vision_mvp.wevra.team_coord import (
    BundleAwareTeamDecoder, MultiRoundBundleDecoder,
    _DecodedHandoff, _GENERIC_NOISE_CLAIM_KINDS,
    collect_admitted_handoffs,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    build_phase58_bank, build_phase58_falsifier_bank,
    _bench_property, run_phase58, run_phase58_seed_stability_sweep,
    run_cross_regime_summary,
)


# =============================================================================
# Unit tests — decoder semantics
# =============================================================================


class MultiRoundDecoderUnitTests(unittest.TestCase):

    def test_single_bundle_reduction(self):
        """W11-3 (single-bundle): the decoder reduces to its inner
        on a single round bundle (specific root_cause, no decoy)."""
        d = MultiRoundBundleDecoder()
        bundle = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                              "service=orders relation=t"),
            _DecodedHandoff("db_admin", "POOL_EXHAUSTION",
                              "service=payments waiters=88"),
        ]
        r = d.decode_rounds([bundle])
        self.assertEqual(r["root_cause"], "deadlock")
        self.assertEqual(set(r["services"]), {"orders", "payments"})

    def test_round_union_promotes_root_cause(self):
        """W11-1 anchor unit: round-1 generic-only + round-2
        specific-only → union elects specific root_cause."""
        d = MultiRoundBundleDecoder()
        r1 = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),
            _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                              "rate=0.22 service=payments"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=180 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "deny count=10 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "deny count=11 service=cache"),
        ]
        r2 = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                              "deadlock relation=orders_payments"),
        ]
        out = d.decode_rounds([r1, r2])
        self.assertEqual(out["root_cause"], "deadlock")
        # cache is corroborated decoy via two roles, both noise-only —
        # dropped by contradiction-aware step.
        self.assertEqual(set(out["services"]), {"orders", "payments"})

    def test_round1_only_collapses(self):
        """W11-Λ unit witness: round-1 alone elects a generic
        root_cause and the decoder cannot exclude the decoy."""
        d = MultiRoundBundleDecoder()
        r1 = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=180 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "deny count=10 service=cache"),
        ]
        out = d.decode_rounds([r1])
        # Generic root_cause: noise-decoy step is skipped.
        self.assertIn(out["root_cause"], ("latency_spike", "error_spike",
                                            "fw_block"))

    def test_noise_decoy_floor(self):
        """A single-role generic-noise mention is preserved (floor
        default = 2)."""
        d = MultiRoundBundleDecoder()
        bundle = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),  # 1 role noise
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                              "deadlock relation=t"),
        ]
        out = d.decode_rounds([bundle])
        self.assertEqual(out["root_cause"], "deadlock")
        self.assertIn("orders", out["services"])

    def test_generic_noise_kinds_set(self):
        """The contradiction-aware drop uses exactly the closed
        vocabulary of generic-noise kinds."""
        self.assertEqual(_GENERIC_NOISE_CLAIM_KINDS,
                          frozenset({"LATENCY_SPIKE",
                                      "ERROR_RATE_SPIKE",
                                      "FW_BLOCK_SURGE"}))


# =============================================================================
# Phase 58 bank shape — pre-committed delayed-causal-evidence property
# =============================================================================


class Phase58BankShapeTests(unittest.TestCase):

    def test_default_bank_property_holds(self):
        """The default Phase-58 bank has the delayed-causal-evidence
        property on every scenario."""
        bank = build_phase58_bank(n_replicates=2, seed=11)
        for sc in bank[:8]:
            props = _bench_property(sc)
            self.assertTrue(
                props["delayed_causal_evidence_property_holds"],
                msg=f"{sc.scenario_id}: {props}")
            self.assertTrue(props["round1_only_generic_noise"])
            self.assertTrue(props["round2_only_specific"])
            self.assertTrue(props["decoy_only_in_round1"])
            self.assertTrue(props["round1_decoy_corroborated"])

    def test_falsifier_bank_breaks_budget(self):
        """W11-4 falsifier: round-1 noise count exceeds K_auditor."""
        bank = build_phase58_falsifier_bank(n_replicates=2, seed=11)
        for sc in bank[:8]:
            props = _bench_property(sc)
            self.assertGreater(props["n_round1_to_auditor"], 4)


# =============================================================================
# Phase 58 default — W11-Λ + W11-1 anchor
# =============================================================================


class Phase58DefaultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = run_phase58(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2, verbose=False)

    def test_multi_round_strict_win(self):
        """W11-1: capsule_multi_round achieves accuracy_full = 1.000."""
        p = self.report["pooled"]
        self.assertEqual(p["capsule_multi_round"]["accuracy_full"], 1.0)

    def test_W11_Lambda_single_round_limit(self):
        """W11-Λ: every per-round / single-round-bundle strategy ties
        FIFO at 0.000 on Phase-58 default."""
        p = self.report["pooled"]
        for s in ("substrate", "capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_cohort_buffered",
                   "capsule_corroboration", "capsule_multi_service",
                   "capsule_bundle_decoder"):
            self.assertEqual(
                p[s]["accuracy_full"], 0.0,
                msg=f"{s} unexpectedly non-zero on Phase-58 default")

    def test_audit_OK_on_every_capsule_strategy(self):
        """T-1..T-7 holds for every capsule strategy on Phase-58."""
        grid = self.report["audit_ok_grid"]
        for s, ok in grid.items():
            if s == "substrate":
                continue
            self.assertTrue(ok, msg=f"audit not OK for {s}")

    def test_headline_gap_at_least_0_20(self):
        """Strong success bar: gap ≥ 0.20 vs FIFO and vs SDK v3.11
        bundle decoder on R-58."""
        g = self.report["headline_gap"]
        self.assertGreaterEqual(
            g["multi_round_minus_fifo_accuracy_full"], 0.20)
        self.assertGreaterEqual(
            g["multi_round_minus_bundle_decoder_accuracy_full"], 0.20)


# =============================================================================
# Phase 58 falsifier — W11-4
# =============================================================================


class Phase58FalsifierTests(unittest.TestCase):

    def test_falsifier_collapses_to_zero(self):
        """W11-4: when round-1 noise floods K_auditor=4, multi_round
        ties FIFO at 0.000 — the named falsifier regime."""
        rep = run_phase58(
            n_eval=8, K_auditor=4, T_auditor=128,
            bank_seed=11, bank_replicates=2,
            use_falsifier_bank=True, verbose=False)
        p = rep["pooled"]
        self.assertEqual(p["capsule_multi_round"]["accuracy_full"], 0.0)
        self.assertEqual(p["capsule_fifo"]["accuracy_full"], 0.0)


# =============================================================================
# Phase 58 seed stability — 5/5
# =============================================================================


class Phase58SeedStabilityTests(unittest.TestCase):

    def test_gap_stable_across_5_seeds(self):
        rep = run_phase58_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=8, K_auditor=8, T_auditor=256)
        for seed, d in rep["per_seed"].items():
            g = d["headline_gap"]
            self.assertGreaterEqual(
                g["multi_round_minus_fifo_accuracy_full"], 0.20,
                msg=f"seed {seed}")
            self.assertEqual(
                g["max_single_round_accuracy_full"], 0.0,
                msg=f"seed {seed}")


# =============================================================================
# Cross-regime — W11-3 backward compatibility
# =============================================================================


class Phase58CrossRegimeTests(unittest.TestCase):
    """W11-3: the SDK v3.12 milestone preserves SDK v3.8/v3.9/v3.10/
    v3.11 wins on R-54..R-57 with no regression."""

    @classmethod
    def setUpClass(cls):
        cls.xr = run_cross_regime_summary(
            n_eval=8, bank_seed=11, bank_replicates=2)

    def test_R54_W7_2_preserved(self):
        p = self.xr["phase54_default"]["pooled"]
        self.assertEqual(p["capsule_cohort_buffered"]["accuracy_full"], 1.0)

    def test_R55_W8_preserved(self):
        p = self.xr["phase55_default"]["pooled"]
        self.assertEqual(p["capsule_corroboration"]["accuracy_full"], 1.0)

    def test_R56_W9_preserved(self):
        p = self.xr["phase56_default"]["pooled"]
        self.assertEqual(p["capsule_multi_service"]["accuracy_full"], 1.0)

    def test_R57_W10_preserved(self):
        p = self.xr["phase57_default"]["pooled"]
        self.assertEqual(p["capsule_bundle_decoder"]["accuracy_full"], 1.0)

    def test_R58_W11_strict_gain(self):
        p = self.xr["phase58_default"]["pooled"]
        self.assertEqual(p["capsule_multi_round"]["accuracy_full"], 1.0)


# =============================================================================
# Single-round W11 ≡ W10 (decoder-level reduction)
# =============================================================================


class W11SingleRoundReductionTests(unittest.TestCase):
    """W11-3 unit-level: the W11 decoder applied as a single-bundle
    decoder produces the same ``services`` set as W10
    BundleAwareTeamDecoder on a (specific root_cause, single-role
    decoy) input — i.e. it does not over-filter on the easy case."""

    def test_single_role_decoy_is_kept(self):
        # W10 keeps a single-role decoy via CCK predicate failure;
        # W11 keeps it via the noise-decoy floor=2.
        bundle = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                              "service=orders relation=t"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=180 service=cache"),  # single role
        ]
        w11 = MultiRoundBundleDecoder()
        out = w11.decode_rounds([bundle])
        self.assertEqual(out["root_cause"], "deadlock")
        # Single-role noise — preserved (floor=2, only 1 role).
        self.assertIn("cache", out["services"])
        self.assertIn("orders", out["services"])


if __name__ == "__main__":
    unittest.main()
