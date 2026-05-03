"""Tests for SDK v3.14 layered open-world normalisation + W13 family.

Covers:

* Unit tests for :class:`LayeredClaimNormalizer` — exact + heuristic +
  abstention layers, idempotency on canonical, soundness on the
  heuristic rule set.
* Unit tests for :class:`LayeredRobustMultiRoundBundleDecoder` —
  W13-1 sufficiency, W13-3 backward-compat, W13-4 closure boundary.
* Phase-60 closure-property tests — every wide-OOV variant resolves
  to its canonical kind under W13; every cosmic-OOV variant escapes
  both layers.
* Phase-60 default config tests — W13-1 strict win on the wide-OOV
  bank, audit OK on every cell.
* Phase-60 falsifier tests — W13-4 cosmic-OOV ties FIFO at 0.000.
* Phase-60 backward-compat tests — clean LLM mode reduces to W11/W12
  byte-for-byte; cross-regime R-54..R-58 preserved.
* Phase-60 5-seed stability — gap layered − robust ≥ 0.50 on every
  seed.
"""

from __future__ import annotations

import unittest

from vision_mvp.coordpy.team_coord import (
    CLAIM_KIND_SYNONYMS, HeuristicAbstractionRule,
    LAYERED_NORMALIZER_ABSTAIN,
    LayeredClaimNormalizer, LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder, RobustMultiRoundBundleDecoder,
    _DECODER_PRIORITY, _DecodedHandoff, _HEURISTIC_KIND_RULES,
)
from vision_mvp.experiments.phase60_open_world_normalization import (
    COSMIC_OOV_KINDS, HEURISTIC_RESCUABLE_OOV_KINDS,
    Phase60ExtractorConfig, Phase60SyntheticExtractor,
    run_cross_regime_summary, run_phase60,
    run_phase60_seed_stability_sweep,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    build_phase58_bank,
)


_CANONICAL_KINDS = {kind for (kind, _l, _r) in _DECODER_PRIORITY}


# =============================================================================
# Heuristic rule soundness (W13-2)
# =============================================================================


class HeuristicRuleSoundnessTests(unittest.TestCase):

    def test_every_rule_canonical_is_in_priority_table(self):
        """W13-2(a): every heuristic rule's output is a key in
        ``_DECODER_PRIORITY``."""
        for rule in _HEURISTIC_KIND_RULES:
            self.assertIn(
                rule.canonical, _CANONICAL_KINDS,
                msg=f"heuristic rule {rule.name!r} -> "
                     f"{rule.canonical!r} not in canonical set")

    def test_every_canonical_self_maps_through_layered(self):
        """W13-2(b): the layered normaliser is idempotent on every
        canonical kind (the heuristic layer never disagrees with the
        exact-table layer on canonical input — checked via a fully-
        layered call where exact wins)."""
        n = LayeredClaimNormalizer()
        for k in _CANONICAL_KINDS:
            self.assertEqual(n.normalize(k), k,
                              msg=f"canonical {k} self-map failed")

    def test_heuristic_layer_idempotent_on_canonical_when_exact_disabled(
            self):
        """Even with the exact table emptied, the heuristic layer
        should resolve every canonical kind to itself (the rules are
        designed so each canonical witnesses its own rule)."""
        # Strip the exact table to force the heuristic layer.
        bare = LayeredClaimNormalizer(synonyms={})
        for k in _CANONICAL_KINDS:
            out = bare.normalize(k)
            self.assertIn(
                out, _CANONICAL_KINDS,
                msg=f"heuristic-only normalisation of canonical {k} "
                     f"produced {out!r}, not a canonical kind")
            # And specifically it should resolve to itself OR to a
            # canonical kind in the same tier; the Phase-60 driver
            # uses the eq-canonical check, which is sufficient for
            # backward-compat.
            self.assertEqual(
                out, k,
                msg=f"heuristic-only normalisation of canonical {k} "
                     f"produced {out!r}, expected self-map")


# =============================================================================
# W13-1 closure widening — every wide-OOV variant rescued
# =============================================================================


class W13ClosureTests(unittest.TestCase):

    def test_every_wide_oov_variant_outside_w12_inside_w13(self):
        """W13-1 closure contract: every entry in
        :data:`HEURISTIC_RESCUABLE_OOV_KINDS` is (a) NOT in
        :data:`CLAIM_KIND_SYNONYMS` and (b) resolves to its named
        canonical via the layered normaliser."""
        n = LayeredClaimNormalizer()
        for canonical, variants in HEURISTIC_RESCUABLE_OOV_KINDS.items():
            for v in variants:
                self.assertNotIn(
                    v.upper(), CLAIM_KIND_SYNONYMS,
                    msg=f"wide-OOV variant {v!r} unexpectedly in W12 "
                         f"table — falsifier not sharp")
                n.reset_counters()
                out = n.normalize(v)
                self.assertEqual(
                    out, canonical,
                    msg=f"wide-OOV variant {v!r} -> {out!r}, "
                         f"expected {canonical!r}")
                self.assertGreaterEqual(
                    n.last_n_heuristic, 1,
                    msg=f"heuristic layer did not fire on {v!r}")

    def test_every_cosmic_oov_variant_escapes_both_layers(self):
        """W13-4 closure boundary: every entry in :data:`COSMIC_OOV_KINDS`
        escapes both layers and the priority decoder cannot match the
        result."""
        n = LayeredClaimNormalizer()
        for canonical, variants in COSMIC_OOV_KINDS.items():
            for v in variants:
                self.assertNotIn(
                    v.upper(), CLAIM_KIND_SYNONYMS,
                    msg=f"cosmic OOV {v!r} unexpectedly in W12 table")
                out = n.normalize(v)
                self.assertNotIn(
                    out, _CANONICAL_KINDS,
                    msg=f"cosmic OOV {v!r} unexpectedly resolved to "
                         f"canonical {out!r} — closure boundary not sharp")

    def test_abstain_mode_returns_sentinel(self):
        n = LayeredClaimNormalizer(abstain_on_unknown=True)
        out = n.normalize("XYZZY_QQQQ")
        self.assertEqual(out, LAYERED_NORMALIZER_ABSTAIN)
        self.assertEqual(n.last_n_abstained, 1)
        # Canonical input should still resolve.
        self.assertEqual(n.normalize("DEADLOCK_SUSPECTED"),
                          "DEADLOCK_SUSPECTED")


# =============================================================================
# W13 decoder unit semantics
# =============================================================================


class LayeredDecoderUnitTests(unittest.TestCase):

    def test_W13_1_unit_w12_4_oov_rescued(self):
        """Single-bundle W13 unit: a W12-4 OOV kind that the fixed
        table cannot rescue is heuristically rescued; the W11
        contradiction-aware drop fires on the noise-corroborated
        decoy (cache, 2 producer roles, generic-only kinds)."""
        d = LayeredRobustMultiRoundBundleDecoder()
        bundle = [
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=2100 service=orders"),
            _DecodedHandoff("monitor", "ERROR_RATE_SPIKE",
                              "rate=0.22 service=payments"),
            # 2 distinct roles mention cache via generic noise → the
            # W11 noise-decoy drop fires on the deadlock root_cause.
            _DecodedHandoff("monitor", "LATENCY_SPIKE",
                              "p95=180 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "count=10 service=cache"),
            _DecodedHandoff("network", "FW_BLOCK_SURGE",
                              "count=11 service=cache"),
        ]
        r2 = [
            _DecodedHandoff("db_admin",
                              "DEADLOCK_PROBABLY_DETECTED_MAYBE",
                              "looks like deadlock relation=orders_payments"),
        ]
        out = d.decode_rounds([bundle, r2])
        self.assertEqual(out["root_cause"], "deadlock")
        self.assertEqual(set(out["services"]), {"orders", "payments"})
        stats = d.normalizer_stats()
        self.assertGreaterEqual(stats["n_heuristic"], 1,
                                 msg="heuristic layer should have fired")

    def test_W13_3_clean_input_matches_w11(self):
        """W13-3: on a candidate stream where every kind is canonical,
        the W13 decoder produces the same answer as W11/W12 and
        ``n_heuristic = n_passthrough = 0``."""
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
        d = LayeredRobustMultiRoundBundleDecoder()
        w13 = d.decode_rounds([clean_r1, clean_r2])
        self.assertEqual(w11, w13)
        stats = d.normalizer_stats()
        self.assertEqual(stats["n_heuristic"], 0)
        self.assertEqual(stats["n_abstained"], 0)
        self.assertEqual(stats["n_passthrough"], 0)
        self.assertGreater(stats["n_exact"], 0)

    def test_W13_3_w12_synonym_input_matches_w12(self):
        """W13-3 cross-version: a W12-rescuable synonym (POOL_EXHAUSTED)
        produces the same answer through both W12 and W13."""
        r1 = [_DecodedHandoff("monitor", "LATENCY",
                                 "p95=200 service=cache"),
              _DecodedHandoff("monitor", "ERROR_RATE",
                                 "rate=0.05 service=cache")]
        r2 = [_DecodedHandoff("db_admin", "POOL_EXHAUSTED",
                                 "pool active=200/200 service=api")]
        w12 = RobustMultiRoundBundleDecoder().decode_rounds([r1, r2])
        d = LayeredRobustMultiRoundBundleDecoder()
        w13 = d.decode_rounds([r1, r2])
        self.assertEqual(w12, w13)
        # All kinds resolved via the exact table; heuristic layer is
        # not exercised on this input.
        self.assertEqual(d.normalizer_stats()["n_heuristic"], 0)

    def test_W13_4_cosmic_oov_does_not_get_rescued(self):
        """W13-4: a kind outside both layers (COSMIC_RAY_FLIP) does
        not resolve; the priority decoder elects a non-deadlock
        root_cause."""
        d = LayeredRobustMultiRoundBundleDecoder()
        r1 = [_DecodedHandoff("monitor", "LATENCY_SPIKE",
                                 "p95=2100 service=orders")]
        r2 = [_DecodedHandoff("db_admin", "COSMIC_RAY_FLIP",
                                 "looks like a deadlock")]
        out = d.decode_rounds([r1, r2])
        self.assertNotEqual(out["root_cause"], "deadlock")
        # The OOV kind passed through (no exact, no heuristic).
        self.assertGreaterEqual(d.normalizer_stats()["n_passthrough"], 1)


# =============================================================================
# Phase 60 default — W13-1 anchor
# =============================================================================


class Phase60DefaultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = run_phase60(
            n_eval=12, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=3,
            llm_mode="synthetic_wide_oov_llm",
            wide_oov_prob=0.50, capture_raw=False, verbose=False)

    def test_layered_strict_win(self):
        """W13-1: capsule_layered_multi_round = 1.000 on Phase-60
        wide-OOV default."""
        p = self.report["pooled"]
        self.assertEqual(
            p["capsule_layered_multi_round"]["accuracy_full"], 1.0)

    def test_w12_robust_collapses_on_wide_oov(self):
        """W13-Λ-fixed: the W12 fixed-table normaliser ties FIFO at
        0.000 on the wide-OOV bank because every drifted variant is
        outside :data:`CLAIM_KIND_SYNONYMS`."""
        p = self.report["pooled"]
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 0.0)
        self.assertEqual(p["capsule_fifo"]["accuracy_full"], 0.0)

    def test_drift_breakdown_is_w13_only(self):
        """The drift summary must show every drifted kind is W13-
        rescuable (and not W12-rescuable; not OOV)."""
        d = self.report["drift_summary"]
        self.assertGreater(d["n_w13_rescuable_total"], 0)
        self.assertEqual(d["n_w12_rescuable_total"], 0)
        self.assertEqual(d["n_oov_total"], 0)

    def test_audit_OK_on_every_capsule_strategy(self):
        grid = self.report["audit_ok_grid"]
        for s, ok in grid.items():
            if s == "substrate":
                continue
            self.assertTrue(ok, msg=f"audit not OK for {s}")

    def test_strong_bar_gap_at_least_0_50(self):
        """Strong bar 11 (R-60 layered split): gap layered − robust
        ≥ 0.50 on the pre-committed default."""
        g = self.report["headline_gap"]
        self.assertGreaterEqual(g["layered_minus_robust"], 0.50)
        self.assertGreaterEqual(g["layered_minus_fifo"], 0.50)
        self.assertGreaterEqual(g["layered_minus_multi_round"], 0.50)


# =============================================================================
# Phase 60 falsifier — W13-4 anchor
# =============================================================================


class Phase60FalsifierTests(unittest.TestCase):

    def test_W13_4_cosmic_oov_collapses_layered_to_zero(self):
        """W13-4: on the cosmic-OOV bank, even the layered normaliser
        cannot rescue the run. capsule_layered_multi_round ties FIFO
        at accuracy_full = 0.000."""
        report = run_phase60(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            llm_mode="synthetic_cosmic_oov_llm",
            capture_raw=False, verbose=False)
        p = report["pooled"]
        self.assertEqual(
            p["capsule_layered_multi_round"]["accuracy_full"], 0.0)
        self.assertEqual(p["capsule_fifo"]["accuracy_full"], 0.0)
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 0.0)
        self.assertEqual(
            p["capsule_multi_round"]["accuracy_full"], 0.0)
        # Drift breakdown must show non-zero OOV.
        self.assertGreater(report["drift_summary"]["n_oov_total"], 0)


# =============================================================================
# Phase 60 backward-compat — W13-3 anchor
# =============================================================================


class Phase60BackwardCompatTests(unittest.TestCase):

    def test_clean_mode_layered_matches_w11(self):
        """W13-3: on synthetic_clean_llm mode, layered ties W11 / W12
        at accuracy_full = 1.000 on Phase-60 default."""
        report = run_phase60(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            llm_mode="synthetic_clean_llm",
            capture_raw=False, verbose=False)
        p = report["pooled"]
        self.assertEqual(
            p["capsule_layered_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            p["capsule_multi_round"]["accuracy_full"], 1.0)

    def test_synonym_mode_layered_matches_w12(self):
        """W13-3 cross-mode: on synthetic_synonym_llm mode (Phase-59
        default), layered ties W12 at 1.000 — the heuristic layer is
        a no-op when the exact layer already covers the drift."""
        report = run_phase60(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            llm_mode="synthetic_synonym_llm",
            capture_raw=False, verbose=False)
        p = report["pooled"]
        self.assertEqual(
            p["capsule_layered_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            p["capsule_robust_multi_round"]["accuracy_full"], 1.0)

    def test_cross_regime_R54_to_R59_preserved(self):
        """Cross-regime audit: R-54..R-58 + R-59 noisy + R-60 clean +
        R-60 wide-OOV all preserve their respective anchors."""
        d = run_cross_regime_summary(n_eval=8, bank_seed=11)
        self.assertEqual(
            d["phase54_default"]["pooled"][
                "capsule_cohort_buffered"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase55_default"]["pooled"][
                "capsule_corroboration"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase56_default"]["pooled"][
                "capsule_multi_service"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase57_default"]["pooled"][
                "capsule_bundle_decoder"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase58_default"]["pooled"][
                "capsule_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase59_noisy"]["pooled"][
                "capsule_robust_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase60_clean"]["pooled"][
                "capsule_layered_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            d["phase60_wide_oov"]["pooled"][
                "capsule_layered_multi_round"]["accuracy_full"], 1.0)


# =============================================================================
# 5-seed stability — strong-bar bar-3
# =============================================================================


class Phase60SeedStabilityTests(unittest.TestCase):

    def test_layered_minus_robust_gap_holds_across_5_seeds(self):
        """W13-1 stability: gap layered − robust ≥ 0.50 on every one
        of 5 alternate seeds (the strong-bar bar-3 cross-bank
        stability requirement)."""
        sweep = run_phase60_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=12, K_auditor=8, T_auditor=256,
            llm_mode="synthetic_wide_oov_llm",
            wide_oov_prob=0.50)
        n_pass = 0
        for seed, rep in sweep["per_seed"].items():
            g = rep["headline_gap"]["layered_minus_robust"]
            self.assertGreaterEqual(
                g, 0.50,
                msg=f"seed={seed}: layered − robust = {g:.3f} < 0.50")
            self.assertTrue(
                rep["audit_ok_grid"]["capsule_layered_multi_round"])
            n_pass += 1
        self.assertEqual(n_pass, 5)


# =============================================================================
# Phase-60 noisy extractor witnesses
# =============================================================================


class Phase60ExtractorWitnessTests(unittest.TestCase):

    def test_clean_extractor_zero_drift(self):
        bank = build_phase58_bank(n_replicates=2, seed=11)[:8]
        ext = Phase60SyntheticExtractor(Phase60ExtractorConfig(
            synonym_prob=0.0, wide_oov_prob=0.0,
            cosmic_oov_prob=0.0, svc_token_alt_prob=0.0, seed=11))
        for sc in bank:
            for cands in (ext.extract_round(sc, 1),
                          ext.extract_round(sc, 2)):
                for (_s, _t, kind, _p, _e) in cands:
                    self.assertEqual(
                        kind.upper(),
                        # canonical kinds always self-map under
                        # CLAIM_KIND_SYNONYMS
                        CLAIM_KIND_SYNONYMS.get(kind.upper(), kind),
                        msg=f"clean extractor emitted non-canonical "
                             f"kind {kind!r}")

    def test_wide_oov_extractor_drifts_into_w13_closure(self):
        bank = build_phase58_bank(n_replicates=2, seed=11)[:8]
        ext = Phase60SyntheticExtractor(Phase60ExtractorConfig(
            synonym_prob=0.0, wide_oov_prob=0.50,
            cosmic_oov_prob=0.0, svc_token_alt_prob=0.30, seed=11))
        n_total = 0
        n_drifted = 0
        n = LayeredClaimNormalizer()
        for sc in bank:
            for cands in (ext.extract_round(sc, 1),
                          ext.extract_round(sc, 2)):
                for (_s, _t, kind, _p, _e) in cands:
                    n_total += 1
                    if kind not in _CANONICAL_KINDS:
                        n_drifted += 1
                        # Every drifted kind must resolve via W13.
                        out = n.normalize(kind)
                        self.assertIn(
                            out, _CANONICAL_KINDS,
                            msg=f"wide-OOV extractor emitted {kind!r} "
                                 f"that escapes the W13 closure")
        self.assertGreater(n_drifted, 5,
                            msg=f"wide-OOV extractor produced only "
                                 f"{n_drifted}/{n_total} drift; expected ≥ 5")


if __name__ == "__main__":
    unittest.main()
