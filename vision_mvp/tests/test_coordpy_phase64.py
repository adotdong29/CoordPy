"""Tests for SDK v3.18 / W17 family.

* magnitude-hinted producer protocol (W17 protocol surface)
* Phase-64 synthetic R-64-baseline / R-64-W14H-only / R-64-SYM
* W17-Λ-symmetric wall (every capsule strategy ties FIFO at 0.000)
* magnitude-hint backward-compat (W17-3) with the W14 structured
  prompt on R-58/R-61 default banks
"""

from __future__ import annotations

import unittest

from vision_mvp.tasks.incident_triage import (
    ROLE_DB_ADMIN, ROLE_MONITOR, ROLE_NETWORK,
)
from vision_mvp.coordpy.team_coord import (
    INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS,
    OperationalThreshold,
    PRODUCER_PROMPT_MAGNITUDE_HINTED,
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    StructuredProducerProtocol,
    incident_triage_magnitude_thresholds,
    incident_triage_role_schemas,
)
from vision_mvp.experiments.phase64_live_composition import (
    _phase64_sym_property, build_phase64_sym_bank,
    run_cross_regime_synthetic, run_phase64,
)
from vision_mvp.experiments.phase61_producer_ambiguity_preservation import (
    IdentityExtractor,
)


# =============================================================================
# Magnitude-hinted prompt rendering invariants (W17-2)
# =============================================================================


class MagnitudeHintedProtocolRenderingTests(unittest.TestCase):
    """W17-2: magnitude-hinted prompt rendering is byte-deterministic
    and includes the threshold table when ``schema.magnitude_thresholds``
    is non-empty."""

    def setUp(self) -> None:
        self.proto = StructuredProducerProtocol(
            mode=PRODUCER_PROMPT_MAGNITUDE_HINTED)
        self.schemas = incident_triage_role_schemas(
            magnitude_hinted=True)
        self.events = [
            ("LATENCY_SPIKE", "p95_ms=4100 service=web"),
            ("ERROR_RATE_SPIKE", "error_rate=0.15 service=db"),
            ("LATENCY_SPIKE", "p95_ms=2900 service=archival"),
            ("ERROR_RATE_SPIKE", "error_rate=0.13 service=archival"),
        ]

    def test_threshold_block_renders_under_magnitude_hinted_mode(self):
        res = self.proto.render_prompt(
            role=ROLE_MONITOR, round_idx=1,
            events=self.events,
            schema=self.schemas[ROLE_MONITOR])
        self.assertEqual(res.mode, PRODUCER_PROMPT_MAGNITUDE_HINTED)
        self.assertIn("OPERATIONAL QUALIFYING THRESHOLDS", res.text)
        self.assertIn("LATENCY_SPIKE qualifies for any p95_ms >= 1000",
                       res.text)
        self.assertIn("ERROR_RATE_SPIKE qualifies for any "
                       "error_rate >= 0.1", res.text)
        # Anti-relative-magnitude clause is the load-bearing
        # methodological instruction.
        self.assertIn(
            "Each event is judged on its own ABSOLUTE magnitude",
            res.text)
        self.assertIn(
            "Do NOT skip an event because another event in this round",
            res.text)

    def test_threshold_block_filters_out_of_round_kinds(self):
        # Round-2 (diagnosis) should NOT render the round-1
        # observation kinds' thresholds because none of round-2's
        # allowed kinds are in the threshold table.
        # Use db_admin's schema since db_admin emits diagnosis kinds.
        res = self.proto.render_prompt(
            role=ROLE_DB_ADMIN, round_idx=2,
            events=[("DEADLOCK_SUSPECTED", "wait_chain=2")],
            schema=self.schemas[ROLE_DB_ADMIN])
        self.assertEqual(res.mode, PRODUCER_PROMPT_MAGNITUDE_HINTED)
        # Round-2 allowed kinds (DEADLOCK_SUSPECTED etc.) have no
        # thresholds in the default table → block is suppressed.
        self.assertNotIn("OPERATIONAL QUALIFYING THRESHOLDS", res.text)

    def test_empty_threshold_table_reduces_to_structured_with_clause(self):
        # W17-3 backward-compat: with an empty
        # ``magnitude_thresholds`` field, the prompt reduces to the
        # structured prompt with the anti-relative-magnitude clause
        # appended.
        empty_schemas = incident_triage_role_schemas(
            magnitude_hinted=False)
        res = self.proto.render_prompt(
            role=ROLE_MONITOR, round_idx=1,
            events=self.events,
            schema=empty_schemas[ROLE_MONITOR])
        self.assertEqual(res.mode, PRODUCER_PROMPT_MAGNITUDE_HINTED)
        self.assertNotIn("OPERATIONAL QUALIFYING THRESHOLDS", res.text)
        # The anti-relative-magnitude clause is the methodological
        # instruction that survives even without thresholds.
        self.assertIn(
            "Each event is judged on its own ABSOLUTE magnitude",
            res.text)

    def test_render_byte_deterministic_under_magnitude_hinted(self):
        a = self.proto.render_prompt(
            role=ROLE_MONITOR, round_idx=1,
            events=self.events,
            schema=self.schemas[ROLE_MONITOR])
        b = self.proto.render_prompt(
            role=ROLE_MONITOR, round_idx=1,
            events=self.events,
            schema=self.schemas[ROLE_MONITOR])
        self.assertEqual(a.text, b.text)

    def test_magnitude_hinted_mode_in_all_modes(self):
        from vision_mvp.coordpy.team_coord import (
            ALL_PRODUCER_PROMPT_MODES)
        self.assertIn(PRODUCER_PROMPT_MAGNITUDE_HINTED,
                       ALL_PRODUCER_PROMPT_MODES)
        self.assertIn(PRODUCER_PROMPT_NAIVE, ALL_PRODUCER_PROMPT_MODES)
        self.assertIn(PRODUCER_PROMPT_STRUCTURED,
                       ALL_PRODUCER_PROMPT_MODES)


class IncidentTriageMagnitudeHintsTests(unittest.TestCase):
    """W17-2: the default operational-threshold table and schema
    factory invariants."""

    def test_default_thresholds_cover_observation_tier(self):
        ts = incident_triage_magnitude_thresholds()
        kinds = {t.kind for t in ts}
        self.assertEqual(
            kinds,
            {"LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE"})

    def test_threshold_defaults_match_extractor_calibration(self):
        tab = {t.kind: t for t in
                INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS}
        # The synthetic ``MagnitudeFilteringExtractor`` defaults are
        # the calibration anchor; the prompt thresholds must equal
        # them exactly so the prompt is operationally honest.
        self.assertEqual(tab["LATENCY_SPIKE"].threshold, 1000.0)
        self.assertEqual(tab["LATENCY_SPIKE"].field, "p95_ms")
        self.assertEqual(tab["ERROR_RATE_SPIKE"].threshold, 0.10)
        self.assertEqual(tab["ERROR_RATE_SPIKE"].field, "error_rate")
        self.assertEqual(tab["FW_BLOCK_SURGE"].threshold, 5.0)
        self.assertEqual(tab["FW_BLOCK_SURGE"].field, "count")

    def test_schema_factory_default_has_no_thresholds(self):
        plain = incident_triage_role_schemas()
        for role, schema in plain.items():
            self.assertEqual(
                schema.magnitude_thresholds, (),
                f"role={role} should have empty thresholds by default")

    def test_schema_factory_magnitude_hinted_carries_thresholds(self):
        hinted = incident_triage_role_schemas(magnitude_hinted=True)
        for role, schema in hinted.items():
            self.assertEqual(
                schema.magnitude_thresholds,
                INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS,
                f"role={role} should carry the default thresholds")


# =============================================================================
# Phase-64 synthetic R-64-baseline / R-64-W14H-only sanity (W17-3)
# =============================================================================


class Phase64SyntheticBaselineTests(unittest.TestCase):
    """R-64-baseline: identity producer + magnitude-hinted prompt
    + ``T_decoder=None``. Sanity anchor — every cross-round capsule
    decoder hits 1.000."""

    def test_baseline_property_holds_8_of_8(self):
        rep = run_phase64(
            n_eval=8, K_auditor=8, T_auditor=256,
            T_decoder=None, bank="phase61",
            extractor="identity",
            prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
            magnitude_hinted_schema=True,
            verbose=False)
        self.assertEqual(
            rep["bench_summary"]["scenarios_with_property"], 8)

    def test_baseline_layered_strict_win(self):
        rep = run_phase64(
            n_eval=8, K_auditor=8, T_auditor=256,
            T_decoder=None, bank="phase61",
            extractor="identity",
            prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
            magnitude_hinted_schema=True,
            verbose=False)
        pooled = rep["pooled"]
        self.assertEqual(
            pooled["capsule_layered_multi_round"]["accuracy_full"], 1.0)
        self.assertEqual(
            pooled["capsule_attention_aware"]["accuracy_full"], 1.0)
        self.assertEqual(pooled["capsule_fifo"]["accuracy_full"], 0.0)


class Phase64SyntheticW14HOnlyTests(unittest.TestCase):
    """R-64-W14H-only: synthetic mag-filter producer + magnitude-
    hinted prompt + ``T_decoder=None``. Synthetic counterpart of
    the W17-1 anchor."""

    def test_w14h_only_property_holds_8_of_8(self):
        rep = run_phase64(
            n_eval=8, K_auditor=8, T_auditor=256,
            T_decoder=None, bank="phase61",
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
            magnitude_hinted_schema=True,
            verbose=False)
        self.assertEqual(
            rep["bench_summary"]["scenarios_with_property"], 8)
        self.assertEqual(
            rep["pooled"]["capsule_layered_multi_round"]["accuracy_full"],
            1.0)


# =============================================================================
# Phase-64-SYM bank shape + W17-Λ-symmetric wall
# =============================================================================


class Phase64SymBankShapeTests(unittest.TestCase):
    """The R-64-SYM bench property: every gold service AND the
    decoy service are mentioned by ≥ 2 distinct routed producer
    roles in round 1 via generic-noise kinds."""

    def setUp(self) -> None:
        self.bank = build_phase64_sym_bank(n_replicates=2, seed=11)
        self.ext = IdentityExtractor(seed=11)

    def test_bank_size(self):
        self.assertEqual(len(self.bank), 8)

    def test_every_scenario_is_symmetric(self):
        for sc in self.bank:
            r1 = self.ext.extract_round(sc, 1)
            r2 = self.ext.extract_round(sc, 2)
            prop = _phase64_sym_property(sc, r1, r2)
            self.assertTrue(
                prop["both_golds_cross_role_corroborated"],
                f"{sc.scenario_id}: gold not multi-role corroborated")
            self.assertTrue(
                prop["decoy_cross_role_corroborated"],
                f"{sc.scenario_id}: decoy not multi-role corroborated")
            self.assertTrue(
                prop["symmetric_corroboration_holds"],
                f"{sc.scenario_id}: symmetry property fails")

    def test_routed_role_count_matches_subscription_table(self):
        # The only routed roles for generic-noise kinds are
        # monitor + network. Both gold and decoy must use exactly
        # these two roles in the symmetric bank.
        for sc in self.bank:
            r1 = self.ext.extract_round(sc, 1)
            prop = _phase64_sym_property(sc, r1, [])
            # Each gold's role count must equal exactly 2.
            self.assertEqual(prop["gold_a_role_count"], 2)
            self.assertEqual(prop["gold_b_role_count"], 2)
            self.assertEqual(prop["decoy_role_count"], 2)


class Phase64SymWallTests(unittest.TestCase):
    """W17-Λ-symmetric: on R-64-SYM every capsule strategy in the
    SDK ties FIFO at ``accuracy_full = 0.000`` — the named
    structural limit the programme has been pointing toward since
    SDK v3.16. **Fires under both ``T_decoder = None`` and
    ``T_decoder = 24``.**"""

    def _run(self, T_decoder):
        return run_phase64(
            n_eval=8, K_auditor=12, T_auditor=256,
            T_decoder=T_decoder, bank="symmetric",
            extractor="identity",
            prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
            magnitude_hinted_schema=True,
            verbose=False)

    def test_every_capsule_strategy_ties_fifo_loose(self):
        rep = self._run(None)
        pooled = rep["pooled"]
        for s in (
                "capsule_fifo", "capsule_priority", "capsule_coverage",
                "capsule_cohort_buffered", "capsule_corroboration",
                "capsule_multi_service", "capsule_multi_round",
                "capsule_robust_multi_round",
                "capsule_layered_multi_round",
                "capsule_layered_fifo_packed",
                "capsule_attention_aware"):
            self.assertEqual(
                pooled[s]["accuracy_full"], 0.0,
                f"{s} should tie FIFO at 0.000 on R-64-SYM (loose)")

    def test_every_capsule_strategy_ties_fifo_tight(self):
        rep = self._run(24)
        pooled = rep["pooled"]
        for s in (
                "capsule_fifo", "capsule_priority", "capsule_coverage",
                "capsule_cohort_buffered", "capsule_corroboration",
                "capsule_multi_service", "capsule_multi_round",
                "capsule_robust_multi_round",
                "capsule_layered_multi_round",
                "capsule_layered_fifo_packed",
                "capsule_attention_aware"):
            self.assertEqual(
                pooled[s]["accuracy_full"], 0.0,
                f"{s} should tie FIFO at 0.000 on R-64-SYM (tight)")

    def test_root_cause_still_correct_under_symmetry(self):
        # The W17-Λ-symmetric wall is a *services_correct* wall: the
        # priority decoder still elects the right specific-tier
        # root_cause from the round-2 disambiguator. The failure is
        # downstream of root_cause election (the W11 drop fires
        # symmetrically on gold and decoy).
        rep = self._run(None)
        for s in ("capsule_multi_round",
                   "capsule_layered_multi_round",
                   "capsule_attention_aware"):
            self.assertEqual(
                rep["pooled"][s]["accuracy_root_cause"], 1.0,
                f"{s} should still elect the right root_cause")


class Phase64CrossRegimeSyntheticTests(unittest.TestCase):
    """Synthetic cross-regime summary — single test that runs all
    four synthetic cells (baseline, W14H-only, SYM-loose, SYM-tight)
    and verifies they separate cleanly:
    * baseline + W14H = 1.000 (sanity + backward compat);
    * SYM loose + tight = 0.000 (W17-Λ-symmetric)."""

    def test_four_cells_separate_cleanly(self):
        rep = run_cross_regime_synthetic(
            n_eval=8, bank_seed=11, K_auditor=8,
            T_auditor=256, T_decoder_tight=14)
        h = rep["headline_summary"]
        self.assertEqual(h["r64_baseline_attention_aware"], 1.0)
        self.assertEqual(h["r64_w14h_only_attention_aware"], 1.0)
        self.assertEqual(h["r64_sym_loose_max_capsule"], 0.0)
        self.assertEqual(h["r64_sym_tight_max_capsule"], 0.0)


# =============================================================================
# W17-3 backward-compat — magnitude-hinted prompt does not break the
# W14 / W13 anchors on R-58 / R-61 default banks.
# =============================================================================


class W17BackwardCompatTests(unittest.TestCase):
    """W17-3: with empty ``schema.magnitude_thresholds``, the
    magnitude-hinted prompt's downstream behaviour matches the
    structured prompt's downstream behaviour on the R-61 default
    bank (the synthetic side does not consult the prompt
    thresholds; the magnitude-filter extractor's behaviour is
    governed by its own threshold parameters)."""

    def test_synthetic_mag_hinted_matches_structured_on_r61(self):
        s_rep = run_phase64(
            n_eval=8, K_auditor=8, T_auditor=256,
            T_decoder=None, bank="phase61",
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED,
            magnitude_hinted_schema=False,
            verbose=False)
        h_rep = run_phase64(
            n_eval=8, K_auditor=8, T_auditor=256,
            T_decoder=None, bank="phase61",
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
            magnitude_hinted_schema=True,
            verbose=False)
        # Both must produce the same downstream answer on the
        # synthetic side (the synthetic extractor does not consult
        # the prompt thresholds — the prompt is rendered but only
        # affects a real LLM's behaviour).
        for s in ("capsule_layered_multi_round",
                   "capsule_attention_aware"):
            self.assertEqual(
                s_rep["pooled"][s]["accuracy_full"],
                h_rep["pooled"][s]["accuracy_full"],
                f"{s}: structured vs magnitude_hinted should match "
                f"on synthetic bank (W17-3 backward-compat)")


if __name__ == "__main__":
    unittest.main()
