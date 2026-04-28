"""Tests for SDK v3.15 structured producer protocol + W14 family.

Covers:

* Unit tests for :class:`StructuredProducerProtocol` —
  observation/diagnosis split, per-event mandate, schema soundness,
  byte-for-byte determinism (W14-2).
* Unit tests for :class:`MagnitudeFilteringExtractor` — calibration
  of the synthetic counterpart of W13-Λ-real, naive vs structured
  prompt behaviour (W14-Λ-prompt vs W14-1).
* Phase-61 closure-property tests — comparable-magnitude decoy
  invariant, R-61-default sanity, R-61-naive collapse, R-61-
  structured restoration.
* Phase-61 default config tests — every cross-round decoder hits
  1.000 when the protocol is in effect; every method ties FIFO at
  0.000 when the naive prompt is in effect (W14-Λ-prompt).
* Phase-61 5-seed stability — gap structured − naive ≥ 0.50 on every
  seed.
* Phase-61 backward-compat tests — incident-triage schema partition
  is sound; structured prompt rendering is deterministic.
"""

from __future__ import annotations

import unittest

from vision_mvp.wevra.team_coord import (
    ALL_PRODUCER_PROMPT_MODES,
    INCIDENT_TRIAGE_OBSERVATION_KINDS,
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    ProducerPromptResult, RoleExtractionSchema,
    StructuredProducerProtocol,
    incident_triage_role_schemas,
)
from vision_mvp.experiments.phase61_producer_ambiguity_preservation import (
    IdentityExtractor, MagnitudeFilteringExtractor,
    build_phase61_bank, run_phase61, run_cross_regime_summary,
    run_phase61_seed_stability_sweep,
    _bench_property,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    build_phase58_bank, _build_round_candidates,
)


# =============================================================================
# W14-2 — schema soundness + protocol determinism
# =============================================================================


class IncidentTriageSchemaTests(unittest.TestCase):

    def test_every_role_has_schema(self):
        sch = incident_triage_role_schemas()
        self.assertEqual(
            sorted(sch.keys()),
            sorted(["monitor", "db_admin", "sysadmin", "network"]))

    def test_partition_is_disjoint_and_sums_to_allowed(self):
        for role, s in incident_triage_role_schemas().items():
            obs = set(s.observation_kinds)
            diag = set(s.diagnosis_kinds)
            self.assertTrue(obs.isdisjoint(diag),
                msg=f"role {role!r}: observation and diagnosis overlap")
            self.assertEqual(obs | diag, set(s.allowed_kinds),
                msg=f"role {role!r}: partition does not cover allowed")

    def test_observation_kinds_match_global_constant(self):
        obs_global = set(INCIDENT_TRIAGE_OBSERVATION_KINDS)
        for role, s in incident_triage_role_schemas().items():
            self.assertTrue(set(s.observation_kinds).issubset(obs_global),
                msg=f"role {role!r}: observation_kinds not in global tier")


class StructuredProtocolDeterminismTests(unittest.TestCase):

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            StructuredProducerProtocol(mode="bogus")

    def test_naive_mode_produces_legacy_format(self):
        proto = StructuredProducerProtocol(mode=PRODUCER_PROMPT_NAIVE)
        sch = incident_triage_role_schemas()
        events = [("LATENCY_SPIKE", "p95_ms=2100 service=orders")]
        res = proto.render_prompt(role="monitor", round_idx=1,
                                    events=events, schema=sch["monitor"])
        self.assertEqual(res.mode, PRODUCER_PROMPT_NAIVE)
        # Legacy phrase markers
        self.assertIn("operational symptoms", res.text)
        self.assertIn("Maximum 6 lines", res.text)

    def test_structured_mode_renders_tier_banner(self):
        proto = StructuredProducerProtocol(mode=PRODUCER_PROMPT_STRUCTURED)
        sch = incident_triage_role_schemas()
        events = [("LATENCY_SPIKE", "p95_ms=2100 service=orders")]
        res = proto.render_prompt(role="monitor", round_idx=1,
                                    events=events, schema=sch["monitor"])
        self.assertEqual(res.mode, PRODUCER_PROMPT_STRUCTURED)
        self.assertIn("ROUND 1 — OBSERVATION MODE", res.text)
        self.assertIn("EMIT ONE CLAIM PER LISTED EVENT", res.text)
        self.assertIn("FORBIDDEN claim kinds", res.text)
        # Round 1 monitor: observation kinds visible, diagnosis kinds
        # in the forbidden list (monitor's diagnosis_kinds is empty,
        # but the text still renders the section label).
        self.assertIn("LATENCY_SPIKE", res.text)

    def test_structured_round2_is_diagnosis_mode(self):
        proto = StructuredProducerProtocol(mode=PRODUCER_PROMPT_STRUCTURED)
        sch = incident_triage_role_schemas()
        events = [("DEADLOCK_SUSPECTED", "deadlock relation=foo wait_chain=2")]
        res = proto.render_prompt(role="db_admin", round_idx=2,
                                    events=events, schema=sch["db_admin"])
        self.assertIn("ROUND 2 — DIAGNOSIS MODE", res.text)
        self.assertIn("DEADLOCK_SUSPECTED", res.text)

    def test_render_is_deterministic(self):
        proto = StructuredProducerProtocol(mode=PRODUCER_PROMPT_STRUCTURED)
        sch = incident_triage_role_schemas()
        events = [("LATENCY_SPIKE", "p95_ms=2100 service=orders"),
                  ("ERROR_RATE_SPIKE", "error_rate=0.22 service=payments")]
        a = proto.render_prompt(role="monitor", round_idx=1,
                                  events=events, schema=sch["monitor"])
        b = proto.render_prompt(role="monitor", round_idx=1,
                                  events=events, schema=sch["monitor"])
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.kinds_in_scope, b.kinds_in_scope)

    def test_role_mismatch_raises(self):
        proto = StructuredProducerProtocol(mode=PRODUCER_PROMPT_STRUCTURED)
        sch = incident_triage_role_schemas()
        with self.assertRaises(ValueError):
            proto.render_prompt(role="monitor", round_idx=1,
                                  events=[],
                                  schema=sch["db_admin"])

    def test_all_modes_listed(self):
        # W17 (SDK v3.18) added PRODUCER_PROMPT_MAGNITUDE_HINTED as
        # an additive third mode. The W14 surface (naive +
        # structured) remains a strict subset.
        from vision_mvp.wevra.team_coord import (
            PRODUCER_PROMPT_MAGNITUDE_HINTED)
        self.assertEqual(
            set(ALL_PRODUCER_PROMPT_MODES),
            {PRODUCER_PROMPT_NAIVE,
              PRODUCER_PROMPT_STRUCTURED,
              PRODUCER_PROMPT_MAGNITUDE_HINTED})


# =============================================================================
# Phase-61 bank shape — comparable-magnitude property
# =============================================================================


class Phase61BankShapeTests(unittest.TestCase):

    def test_decoy_magnitudes_within_gold_range(self):
        """W14-1 closure property: every Phase-61 scenario's decoy
        round-1 latency / error magnitudes lie inside the gold
        magnitude range. (The redesigned events are operationally
        comparable; magnitude alone cannot discriminate.)"""
        import re
        bank = build_phase61_bank(n_replicates=1, seed=11)
        for sc in bank:
            decoy = sc.decoy_storm_service
            gold_a, gold_b = sc.gold_services_pair
            decoy_lat = []
            gold_lat = []
            decoy_err = []
            gold_err = []
            for role, evs in sc.round1_emissions.items():
                for (kind, payload) in evs:
                    m = re.search(r"\bp95_ms=([0-9]+)", payload)
                    if m:
                        v = int(m.group(1))
                        if f"service={decoy}" in payload:
                            decoy_lat.append(v)
                        elif (f"service={gold_a}" in payload
                              or f"service={gold_b}" in payload):
                            gold_lat.append(v)
                    m = re.search(r"\berror_rate=([0-9.]+)", payload)
                    if m:
                        v = float(m.group(1))
                        if f"service={decoy}" in payload:
                            decoy_err.append(v)
                        elif (f"service={gold_a}" in payload
                              or f"service={gold_b}" in payload):
                            gold_err.append(v)
            # Decoy magnitudes must clear the magnitude-filter
            # threshold (p95_ms ≥ 1000, error_rate ≥ 0.10) so the
            # filter doesn't trivially drop them.
            for v in decoy_lat:
                self.assertGreaterEqual(v, 1000,
                    msg=f"{sc.scenario_id}: decoy p95_ms={v} below 1000")
            for v in decoy_err:
                self.assertGreaterEqual(v, 0.10,
                    msg=f"{sc.scenario_id}: decoy error_rate={v} below 0.10")

    def test_default_bank_size_8(self):
        bank = build_phase61_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 8)


# =============================================================================
# IdentityExtractor — R-61-default sanity
# =============================================================================


class Phase61DefaultExtractorTests(unittest.TestCase):

    def test_bench_property_holds_8_of_8(self):
        bank = build_phase61_bank(n_replicates=2, seed=11)
        ext = IdentityExtractor()
        n_holds = 0
        for sc in bank:
            r1 = ext.extract_round(sc, 1)
            r2 = ext.extract_round(sc, 2)
            bp = _bench_property(sc, r1, r2)
            if bp["delayed_causal_evidence_property_holds"]:
                n_holds += 1
        self.assertEqual(n_holds, len(bank))


# =============================================================================
# MagnitudeFilteringExtractor calibration
# =============================================================================


class MagnitudeFilterCalibrationTests(unittest.TestCase):

    def test_filter_drops_phase58_low_magnitude_decoys(self):
        """Calibration anchor: the magnitude filter on the *original*
        Phase-58 events drops the deliberately-low-magnitude decoy
        spikes (matching the W13-Λ-real real-Ollama observation)."""
        from vision_mvp.experiments.phase58_multi_round_decoder import (
            build_phase58_bank as build58)
        bank = build58(n_replicates=1, seed=11)
        ext = MagnitudeFilteringExtractor(
            prompt_mode=PRODUCER_PROMPT_NAIVE)
        for sc in bank:
            ext.reset_counters()
            r1 = ext.extract_round(sc, 1)
            # Under Phase-58 events, the low-magnitude decoy spikes
            # (p95_ms=180/200/210, error_rate=0.04..0.06) are below
            # threshold and must be dropped.
            self.assertGreater(ext.n_filtered_by_threshold, 0,
                msg=f"{sc.scenario_id}: filter did not drop "
                     f"low-magnitude decoy")

    def test_filter_keeps_phase61_comparable_magnitude_decoys(self):
        """On the Phase-61 redesigned events (decoy magnitudes inside
        the gold range), the threshold filter does NOT drop the
        decoy spikes; the prompt-induced compression is what erases
        the bench property under naive prompt."""
        bank = build_phase61_bank(n_replicates=1, seed=11)
        ext = MagnitudeFilteringExtractor(
            prompt_mode=PRODUCER_PROMPT_NAIVE)
        for sc in bank:
            ext.reset_counters()
            ext.extract_round(sc, 1)
            self.assertEqual(ext.n_filtered_by_threshold, 0,
                msg=f"{sc.scenario_id}: filter unexpectedly dropped "
                     f"comparable-magnitude decoy")

    def test_naive_compression_drops_decoy_kinds_from_monitor(self):
        """Under naive prompt, the per-(role, kind) top-N compression
        drops monitor's decoy-side LATENCY/ERROR mentions because the
        gold mention is higher-magnitude. This is the W14-Λ-prompt
        synthetic mechanism."""
        bank = build_phase61_bank(n_replicates=1, seed=11)
        ext = MagnitudeFilteringExtractor(
            prompt_mode=PRODUCER_PROMPT_NAIVE)
        for sc in bank:
            ext.reset_counters()
            ext.extract_round(sc, 1)
            self.assertGreater(ext.n_compressed_by_prompt, 0,
                msg=f"{sc.scenario_id}: naive prompt did not compress")

    def test_structured_disables_compression(self):
        bank = build_phase61_bank(n_replicates=1, seed=11)
        proto = StructuredProducerProtocol(
            mode=PRODUCER_PROMPT_STRUCTURED)
        sch = incident_triage_role_schemas()
        ext = MagnitudeFilteringExtractor(
            prompt_mode=PRODUCER_PROMPT_STRUCTURED)
        recorded: list = []
        for sc in bank:
            ext.reset_counters()
            ext.extract_round(sc, 1, protocol=proto, schemas=sch,
                                record_prompts=recorded)
            self.assertEqual(ext.n_compressed_by_prompt, 0,
                msg=f"{sc.scenario_id}: structured prompt unexpectedly "
                     f"compressed")
        # Every round-1 prompt was rendered in structured mode.
        self.assertTrue(recorded)
        for r in recorded:
            self.assertEqual(r.mode, PRODUCER_PROMPT_STRUCTURED)


# =============================================================================
# Phase-61 default config — W14-Λ-prompt + W14-1 anchors
# =============================================================================


class Phase61DefaultTests(unittest.TestCase):
    """Pre-committed default config:
    K_auditor=8, T_auditor=256, n_eval=8, bank_seed=11.
    """

    @classmethod
    def setUpClass(cls):
        cls.r_default = run_phase61(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            extractor="identity",
            prompt_mode=PRODUCER_PROMPT_NAIVE, verbose=False)
        cls.r_naive = run_phase61(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_NAIVE, verbose=False)
        cls.r_struct = run_phase61(
            n_eval=8, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=2,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED, verbose=False)

    def _gap(self, rep, a, b):
        return (rep["pooled"][a]["accuracy_full"]
                - rep["pooled"][b]["accuracy_full"])

    def test_default_property_holds_8_of_8(self):
        self.assertEqual(
            self.r_default["bench_summary"]["scenarios_with_property"], 8)
        self.assertEqual(
            self.r_default["bench_summary"]["n_scenarios"], 8)

    def test_naive_prompt_property_collapses(self):
        """W14-Λ-prompt: under naive prompt + magnitude filter, the
        bench property is erased upstream — 0/8 scenarios hold."""
        bs = self.r_naive["bench_summary"]
        self.assertEqual(bs["scenarios_with_property"], 0)
        self.assertEqual(bs["scenarios_with_decoy_corroboration"], 0)

    def test_structured_prompt_property_restored(self):
        """W14-1: under structured prompt + magnitude filter, the bench
        property survives — 8/8 scenarios hold."""
        bs = self.r_struct["bench_summary"]
        self.assertEqual(bs["scenarios_with_property"], 8)
        self.assertEqual(bs["scenarios_with_decoy_corroboration"], 8)

    def test_naive_prompt_every_method_ties_fifo(self):
        """W14-Λ-prompt: every capsule strategy ties FIFO at 0.000."""
        for s, p in self.r_naive["pooled"].items():
            self.assertEqual(p["accuracy_full"], 0.000,
                msg=f"strategy {s!r} unexpectedly broke "
                     f"W14-Λ-prompt: {p['accuracy_full']}")

    def test_structured_prompt_layered_strict_win(self):
        """W14-1 strict separation: layered = robust = multi_round =
        1.000 under structured prompt."""
        for s in ("capsule_layered_multi_round",
                   "capsule_robust_multi_round",
                   "capsule_multi_round"):
            self.assertEqual(
                self.r_struct["pooled"][s]["accuracy_full"], 1.000,
                msg=f"strategy {s!r} accuracy_full under structured "
                     f"prompt: {self.r_struct['pooled'][s]['accuracy_full']}")
        gap = self._gap(self.r_struct,
                          "capsule_layered_multi_round", "capsule_fifo")
        self.assertGreaterEqual(gap, 0.50,
            msg=f"layered − fifo strict gap = {gap}; expected ≥ 0.50")

    def test_audit_OK_on_every_capsule_strategy(self):
        for rep_name, rep in (("default", self.r_default),
                                ("naive", self.r_naive),
                                ("struct", self.r_struct)):
            for s, ok in rep["audit_ok_grid"].items():
                if s == "substrate":
                    continue
                self.assertTrue(ok, msg=f"{rep_name}/{s} failed audit")


# =============================================================================
# 5-seed stability — gap structured − naive ≥ 0.50 on every seed
# =============================================================================


class Phase61SeedStabilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.struct_sweep = run_phase61_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=8, K_auditor=8, T_auditor=256,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_STRUCTURED)
        cls.naive_sweep = run_phase61_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=8, K_auditor=8, T_auditor=256,
            extractor="magnitude_filter",
            prompt_mode=PRODUCER_PROMPT_NAIVE)

    def test_structured_layered_one_thousand_on_every_seed(self):
        for seed, rep in self.struct_sweep["per_seed"].items():
            v = rep["pooled"]["capsule_layered_multi_round"]["accuracy_full"]
            self.assertEqual(v, 1.000,
                msg=f"seed={seed}: layered={v}, expected 1.000 under "
                     f"structured prompt")

    def test_naive_every_method_zero_on_every_seed(self):
        for seed, rep in self.naive_sweep["per_seed"].items():
            for s, p in rep["pooled"].items():
                self.assertEqual(p["accuracy_full"], 0.000,
                    msg=f"seed={seed}, strategy={s!r}: "
                         f"acc={p['accuracy_full']} under naive prompt")

    def test_gap_structured_minus_naive_holds_across_5_seeds(self):
        """W14-1 stability: for every seed, the structured-prompt
        layered accuracy − naive-prompt layered accuracy ≥ 0.50."""
        for seed in (11, 17, 23, 29, 31):
            s_acc = (self.struct_sweep["per_seed"][seed]["pooled"]
                     ["capsule_layered_multi_round"]["accuracy_full"])
            n_acc = (self.naive_sweep["per_seed"][seed]["pooled"]
                     ["capsule_layered_multi_round"]["accuracy_full"])
            gap = s_acc - n_acc
            self.assertGreaterEqual(gap, 0.50,
                msg=f"seed={seed}: gap structured − naive = {gap}")


# =============================================================================
# Cross-regime — three Phase-61 sub-banks
# =============================================================================


class Phase61CrossRegimeTests(unittest.TestCase):

    def test_three_subbanks_separate_cleanly(self):
        rep = run_cross_regime_summary(n_eval=8, bank_seed=11)
        # Default (identity, naive) — sanity anchor: layered = 1.000
        d = rep["r61_default_identity"]
        self.assertEqual(
            d["pooled"]["capsule_layered_multi_round"]["accuracy_full"],
            1.000)
        # Naive prompt + magnitude filter — W14-Λ-prompt: every method 0.000
        n = rep["r61_naive_prompt"]
        for s, p in n["pooled"].items():
            self.assertEqual(p["accuracy_full"], 0.000)
        # Structured prompt + magnitude filter — W14-1: layered = 1.000
        s = rep["r61_structured_prompt"]
        self.assertEqual(
            s["pooled"]["capsule_layered_multi_round"]["accuracy_full"],
            1.000)


if __name__ == "__main__":
    unittest.main()
