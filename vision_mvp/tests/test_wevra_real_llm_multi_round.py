"""Tests for SDK v3.13 real-LLM-robust multi-round bundle decoder + W12 family.

Covers:

* Unit tests for :class:`RobustMultiRoundBundleDecoder` — synonym
  rewriting, payload rewriting, single-bundle reduction, multi-round
  union, contradiction-aware noise-decoy drop on the post-normalisation
  stream.
* Unit tests for the closed-vocabulary :data:`CLAIM_KIND_SYNONYMS`
  table (W12-2 mechanical witness).
* Phase-59 bench-property tests (delayed-causal-evidence holds *after
  normalisation* under bounded synonym + payload drift).
* Phase-59 default-config tests: W12-Λ collapse of un-normalised
  W11 + W12-1 strict win + 5/5 seed stability.
* Phase-59 falsifier tests (W12-4 — out-of-vocabulary kinds).
* Phase-59 backward-compat tests (W12-3 — clean-LLM mode reduces
  byte-for-byte to R-58 W11 result).
* Cross-regime audit (R-54..R-58 unchanged + R-59 default holds).
* Lifecycle audit (T-1..T-7) preserved on every cell of Phase 59.
"""

from __future__ import annotations

import unittest

from vision_mvp.wevra.team_coord import (
    BundleAwareTeamDecoder, CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE,
    CLAIM_KIND_SYNONYMS, MultiRoundBundleDecoder,
    RobustMultiRoundBundleDecoder, _DECODER_PRIORITY, _DecodedHandoff,
    normalize_claim_kind, normalize_handoff, normalize_payload,
)
from vision_mvp.experiments.phase59_real_llm_multi_round import (
    NOISY_KIND_VARIANTS, NoisyLLMExtractor, NoisyLLMExtractorConfig,
    OUT_OF_VOCAB_KINDS, _bench_property,
    run_cross_regime_summary, run_phase59,
    run_phase59_seed_stability_sweep,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    build_phase58_bank,
)


# =============================================================================
# Unit tests — closed-vocabulary normalisation
# =============================================================================


class NormalisationTableTests(unittest.TestCase):
    """W12-2 mechanical witness: every synonym maps to a canonical
    kind that the priority decoder recognises."""

    def test_every_synonym_target_is_canonical(self):
        """The synonym table's image is a subset of the canonical
        ``_DECODER_PRIORITY`` kinds."""
        canonical = {kind for (kind, _label, _remed) in _DECODER_PRIORITY}
        for variant, target in CLAIM_KIND_SYNONYMS.items():
            self.assertIn(
                target, canonical,
                msg=f"synonym {variant!r}->{target!r}: target not in "
                     f"canonical priority list")

    def test_canonical_kinds_self_map(self):
        """Every canonical kind is in the synonym table mapping to
        itself (idempotency anchor)."""
        canonical = {kind for (kind, _label, _remed) in _DECODER_PRIORITY}
        for k in canonical:
            self.assertEqual(
                CLAIM_KIND_SYNONYMS.get(k), k,
                msg=f"canonical kind {k!r} is missing self-map")

    def test_normaliser_is_idempotent_on_canonical(self):
        for k in (kind for (kind, _l, _r) in _DECODER_PRIORITY):
            self.assertEqual(normalize_claim_kind(k), k)

    def test_normaliser_rewrites_known_drift(self):
        self.assertEqual(
            normalize_claim_kind("DEADLOCK_DETECTED"), "DEADLOCK_SUSPECTED")
        self.assertEqual(
            normalize_claim_kind("POOL_EXHAUSTED"), "POOL_EXHAUSTION")
        self.assertEqual(
            normalize_claim_kind("DISK_FULL"), "DISK_FILL_CRITICAL")

    def test_unknown_kind_passes_through(self):
        # Out-of-vocab kinds (W12-4 falsifier regime) survive
        # normalisation unchanged so the closed-vocabulary contract
        # is mechanically falsifiable.
        self.assertEqual(
            normalize_claim_kind("DEADLOCK_PROBABLY_DETECTED_MAYBE"),
            "DEADLOCK_PROBABLY_DETECTED_MAYBE")

    def test_payload_normaliser_rewrites_alt_spellings(self):
        self.assertIn("service=orders", normalize_payload("svc=orders"))
        self.assertIn("service=orders",
                      normalize_payload("for service orders"))
        self.assertIn("service=orders",
                      normalize_payload("on service orders"))
        # Idempotent on canonical.
        self.assertEqual("service=orders waiters=88",
                          normalize_payload("service=orders waiters=88"))


# =============================================================================
# Unit tests — RobustMultiRoundBundleDecoder semantics
# =============================================================================


class RobustDecoderUnitTests(unittest.TestCase):

    def test_single_bundle_reduction_with_drift(self):
        """A single bundle with drifted kinds + drifted service tokens
        is rescued by normalisation and decoded correctly."""
        d = RobustMultiRoundBundleDecoder()
        bundle = [
            _DecodedHandoff("db_admin", "DEADLOCK_DETECTED",
                              "svc=orders relation=t"),
            _DecodedHandoff("db_admin", "POOL_EXHAUSTED",
                              "for service payments waiters=88"),
        ]
        r = d.decode_rounds([bundle])
        self.assertEqual(r["root_cause"], "deadlock")
        self.assertEqual(set(r["services"]), {"orders", "payments"})
        # Both handoffs needed kind rewriting; both needed payload
        # rewriting.
        self.assertEqual(d.last_n_kind_rewrites, 2)
        self.assertEqual(d.last_n_payload_rewrites, 2)

    def test_round_union_with_round_1_drift(self):
        """Round-1 drift (synonym kinds + alt service tags) is
        normalised; round-2 specific-tier kind elects deadlock; the
        contradiction-aware drop fires on the noise-corroborated
        decoy."""
        d = RobustMultiRoundBundleDecoder()
        r1 = [
            _DecodedHandoff("monitor", "LATENCY",
                              "p95=2100 svc=orders"),
            _DecodedHandoff("monitor", "ERROR_SURGE",
                              "rate=0.22 svc=payments"),
            _DecodedHandoff("monitor", "P95_HIGH",
                              "p95=180 service:cache"),
            _DecodedHandoff("network", "FW_DENY",
                              "deny count=10 service:cache"),
            _DecodedHandoff("network", "BLOCKED_PACKETS",
                              "deny count=11 service:cache"),
        ]
        r2 = [
            _DecodedHandoff("db_admin", "DEADLOCK",
                              "deadlock relation=orders_payments"),
        ]
        out = d.decode_rounds([r1, r2])
        self.assertEqual(out["root_cause"], "deadlock")
        self.assertEqual(set(out["services"]), {"orders", "payments"})

    def test_clean_input_matches_w11(self):
        """W12-3 unit: on an input where every kind is canonical and
        every payload uses ``service=``, the robust decoder produces
        the same answer as :class:`MultiRoundBundleDecoder` and the
        rewrite counters are zero."""
        clean_r1 = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=180 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "deny count=10 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "deny count=11 service=cache"),
        ]
        clean_r2 = [
            _DecodedHandoff("db_admin", "DEADLOCK_SUSPECTED",
                              "deadlock relation=orders_payments"),
        ]
        w11 = MultiRoundBundleDecoder().decode_rounds([clean_r1, clean_r2])
        d = RobustMultiRoundBundleDecoder()
        w12 = d.decode_rounds([clean_r1, clean_r2])
        self.assertEqual(w11, w12)
        self.assertEqual(d.last_n_kind_rewrites, 0)
        self.assertEqual(d.last_n_payload_rewrites, 0)

    def test_oov_kind_does_not_get_rescued(self):
        """W12-4 unit: an out-of-vocabulary kind like
        ``DEADLOCK_PROBABLY_DETECTED_MAYBE`` is NOT in the synonym
        table and survives normalisation unchanged. The priority
        decoder cannot match it; the elected root_cause stays generic
        or unknown."""
        d = RobustMultiRoundBundleDecoder()
        r1 = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),
        ]
        r2 = [
            _DecodedHandoff("db_admin", "DEADLOCK_PROBABLY_DETECTED_MAYBE",
                              "looks like deadlock"),
        ]
        out = d.decode_rounds([r1, r2])
        # OOV → root_cause is generic (latency_spike) not deadlock.
        self.assertNotEqual(out["root_cause"], "deadlock")


# =============================================================================
# Synthetic noisy LLM extractor — closed-vocabulary witnesses
# =============================================================================


class NoisyExtractorTests(unittest.TestCase):

    def test_clean_extractor_matches_phase58_canonical(self):
        """W12-3 anchor (extractor side): clean mode produces the
        canonical Phase-58 candidate stream (modulo deterministic
        ordering)."""
        bank = build_phase58_bank(n_replicates=1, seed=11)[:4]
        ext = NoisyLLMExtractor(NoisyLLMExtractorConfig(
            synonym_prob=0.0, svc_token_alt_prob=0.0,
            oov_prob=0.0, drop_claim_prob=0.0, seed=11))
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            for (_s, _t, kind, payload, _e) in r1 + r2:
                self.assertEqual(
                    normalize_claim_kind(kind), kind,
                    msg=f"clean extractor emitted drifted kind {kind!r}")

    def test_noisy_extractor_drifts_some_kinds(self):
        """The default noisy config produces measurable drift on the
        scenario bank."""
        bank = build_phase58_bank(n_replicates=2, seed=11)[:8]
        ext = NoisyLLMExtractor(NoisyLLMExtractorConfig(
            synonym_prob=0.50, svc_token_alt_prob=0.30, seed=11))
        n_total = 0
        n_drifted = 0
        for sc in bank:
            for cands in (ext.extract_round(sc, 1),
                          ext.extract_round(sc, 2)):
                for (_s, _t, kind, _p, _e) in cands:
                    n_total += 1
                    if normalize_claim_kind(kind) != kind:
                        n_drifted += 1
        self.assertGreater(n_total, 0)
        # Default config should drift roughly 25-50% of kinds; we
        # require at least 10 drifted of the bank's ~70 to make sure
        # the bench actually exercises normalisation.
        self.assertGreater(
            n_drifted, 10,
            msg=f"only {n_drifted}/{n_total} drifted; expected ≥ 10")

    def test_noisy_variants_all_in_synonym_table(self):
        """W12-2 mechanical: every variant the noisy extractor can
        emit is in the synonym table (so normalisation is guaranteed
        to rescue it under default noise)."""
        for canonical, variants in NOISY_KIND_VARIANTS.items():
            for v in variants:
                self.assertIn(
                    v, CLAIM_KIND_SYNONYMS,
                    msg=f"variant {v!r} for canonical {canonical!r} "
                         f"is missing from CLAIM_KIND_SYNONYMS")
                self.assertEqual(
                    CLAIM_KIND_SYNONYMS[v], canonical,
                    msg=f"variant {v!r} maps to "
                         f"{CLAIM_KIND_SYNONYMS[v]!r}, expected "
                         f"{canonical!r}")

    def test_oov_kinds_are_not_in_synonym_table(self):
        """W12-4 mechanical: OOV kinds are not in the synonym table
        (so the falsifier regime cannot be rescued)."""
        for canonical, oov in OUT_OF_VOCAB_KINDS.items():
            self.assertNotIn(
                oov, CLAIM_KIND_SYNONYMS,
                msg=f"OOV kind {oov!r} unexpectedly present in "
                     f"synonym table — falsifier no longer sharp")


# =============================================================================
# Phase 59 bench-property tests
# =============================================================================


class Phase59BenchPropertyTests(unittest.TestCase):

    def test_bench_property_holds_after_normalisation(self):
        """The delayed-causal-evidence property holds *after
        normalisation* on every Phase-59 default scenario."""
        bank = build_phase58_bank(n_replicates=2, seed=11)[:8]
        ext = NoisyLLMExtractor(NoisyLLMExtractorConfig(
            synonym_prob=0.50, svc_token_alt_prob=0.30, seed=11))
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            props = _bench_property(sc, r1, r2)
            self.assertTrue(
                props["delayed_causal_evidence_property_holds"],
                msg=f"{sc.scenario_id}: {props}")
            self.assertTrue(props["round1_only_generic_noise"])
            self.assertTrue(props["round2_only_specific"])
            self.assertTrue(props["decoy_only_in_round1"])
            self.assertTrue(props["round1_decoy_corroborated"])

    def test_clean_extractor_zero_drift(self):
        bank = build_phase58_bank(n_replicates=2, seed=11)[:8]
        ext = NoisyLLMExtractor(NoisyLLMExtractorConfig(
            synonym_prob=0.0, svc_token_alt_prob=0.0, seed=11))
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            props = _bench_property(sc, r1, r2)
            self.assertEqual(props["n_kind_drifted_round1"], 0)
            self.assertEqual(props["n_kind_drifted_round2"], 0)


# =============================================================================
# Phase 59 default — W12-Λ + W12-1 anchor
# =============================================================================


class Phase59DefaultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = run_phase59(
            n_eval=12, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=3,
            llm_mode="synthetic_noisy_llm",
            llm_synonym_prob=0.50, llm_svc_alt_prob=0.30,
            llm_seed=11, verbose=False)

    def test_robust_strict_win(self):
        """W12-1: capsule_robust_multi_round achieves accuracy_full
        = 1.000 on Phase-59 default."""
        p = self.report["pooled"]
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 1.0)

    def test_W12_Lambda_un_normalised_collapse(self):
        """W12-Λ: every un-normalised single-round / multi-round
        capsule strategy ties FIFO at 0.000 on Phase-59 default."""
        p = self.report["pooled"]
        # Every other capsule strategy must be at 0.000 because the
        # synonym drift breaks `_DECODER_PRIORITY` lookups.
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_cohort_buffered",
                   "capsule_corroboration", "capsule_multi_service",
                   "capsule_bundle_decoder", "capsule_multi_round"):
            self.assertEqual(
                p[s]["accuracy_full"], 0.0,
                msg=f"{s} unexpectedly non-zero on Phase-59 default")

    def test_audit_OK_on_every_capsule_strategy(self):
        """T-1..T-7 holds for every capsule strategy on Phase-59."""
        grid = self.report["audit_ok_grid"]
        for s, ok in grid.items():
            if s == "substrate":
                continue
            self.assertTrue(ok, msg=f"audit not OK for {s}")

    def test_headline_gap_at_least_0_20(self):
        """Strong success bar: gap ≥ 0.20 vs FIFO and vs SDK v3.12
        (un-normalised) MultiRoundBundleDecoder on R-59."""
        g = self.report["headline_gap"]
        self.assertGreaterEqual(g["robust_multi_round_minus_fifo"], 0.20)
        self.assertGreaterEqual(
            g["robust_multi_round_minus_multi_round"], 0.20)


# =============================================================================
# Phase 59 falsifier — W12-4 anchor
# =============================================================================


class Phase59FalsifierTests(unittest.TestCase):

    def test_W12_4_oov_collapses_to_zero(self):
        """W12-4: on the OOV-saturated falsifier bank, every method
        (including capsule_robust_multi_round) ties FIFO at 0.000 on
        accuracy_full."""
        report = run_phase59(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            use_falsifier_bank=True,
            llm_mode="synthetic_noisy_llm", verbose=False)
        p = report["pooled"]
        # accuracy_full is the load-bearing comparison; root_cause /
        # services may diverge, that's fine — the strong-bar metric
        # is set-equality services_correct ∧ root_cause_correct.
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 0.0,
            msg="W12-4 falsifier no longer collapses W12 to 0.000")
        self.assertEqual(p["capsule_fifo"]["accuracy_full"], 0.0)
        # All "winners" tied at 0.000.
        for s in ("capsule_robust_multi_round", "capsule_fifo",
                   "capsule_multi_round", "capsule_bundle_decoder"):
            self.assertEqual(p[s]["accuracy_full"], 0.0)


# =============================================================================
# Phase 59 backward-compat — W12-3 anchor
# =============================================================================


class Phase59BackwardCompatTests(unittest.TestCase):

    def test_clean_mode_robust_matches_w11(self):
        """W12-3: on synthetic_clean_llm mode, robust ties W11 at
        accuracy_full = 1.000."""
        report = run_phase59(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            llm_mode="synthetic_clean_llm", verbose=False)
        p = report["pooled"]
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            p["capsule_multi_round"]["accuracy_full"], 1.0)

    def test_cross_regime_R54_to_R58_preserved(self):
        """W12-3 cross-regime: R-54 (W7-2), R-55 (W8), R-56 (W9),
        R-57 (W10), R-58 (W11) all still hit the prior anchor at
        accuracy_full = 1.000 with their respective strategies."""
        d = run_cross_regime_summary(n_eval=8, bank_seed=11)
        self.assertEqual(
            d["phase54_default"]["pooled"]["capsule_cohort_buffered"][
                "accuracy_full"], 1.0)
        self.assertEqual(
            d["phase55_default"]["pooled"]["capsule_corroboration"][
                "accuracy_full"], 1.0)
        self.assertEqual(
            d["phase56_default"]["pooled"]["capsule_multi_service"][
                "accuracy_full"], 1.0)
        self.assertEqual(
            d["phase57_default"]["pooled"]["capsule_bundle_decoder"][
                "accuracy_full"], 1.0)
        self.assertEqual(
            d["phase58_default"]["pooled"]["capsule_multi_round"][
                "accuracy_full"], 1.0)


# =============================================================================
# Seed stability sweep — strong-bar bar-3
# =============================================================================


class Phase59SeedStabilityTests(unittest.TestCase):

    def test_gap_holds_across_5_seeds(self):
        """W12-1 stability anchor: the gap robust − w11 ≥ 0.50 on
        every one of 5 alternate seeds (the strong-bar bar-3
        cross-bank stability requirement)."""
        sweep = run_phase59_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=12, K_auditor=8, T_auditor=256)
        seen = sweep["per_seed"]
        n_seeds_pass = 0
        for seed, rep in seen.items():
            g = rep["headline_gap"]["robust_multi_round_minus_multi_round"]
            self.assertGreaterEqual(
                g, 0.50,
                msg=f"seed={seed}: robust−w11 = {g:.3f} < 0.50")
            self.assertTrue(
                rep["audit_ok_grid"]["capsule_robust_multi_round"],
                msg=f"seed={seed}: audit not OK")
            n_seeds_pass += 1
        self.assertEqual(n_seeds_pass, 5)


if __name__ == "__main__":
    unittest.main()
