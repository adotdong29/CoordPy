"""Tests for the SDK v3.11 bundle-aware team decoder + W10 family.

Covers:

* Unit-level tests for ``BundleAwareTeamDecoder`` and the closed-
  vocabulary ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` table.
* Phase-57 bank-shape tests (the bench property is mechanically
  verified).
* Phase-57 default-config tests: W10-Λ admission limit witness +
  W10-1 bundle decoder strict win.
* Phase-57 seed-stability tests: gap holds across 5/5 bank seeds.
* Phase-57 falsifier tests: W10-4 (decoy CCK promotion ties FIFO).
* Cross-regime backward-compat tests: W10-3 (bundle decoder ties
  prior best on R-54 / R-55 / R-56).
* Lifecycle-audit tests: W10-aud (T-1..T-7 OK on every cell).
"""

from __future__ import annotations

import unittest

from vision_mvp.coordpy.team_coord import (
    BundleAwareTeamDecoder, CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE,
    MultiServiceCorroborationAdmissionPolicy,
    _decoded_root_cause, _DecodedHandoff, decode_admitted_role_view,
)
from vision_mvp.experiments.phase57_decoder_forcing import (
    build_phase57_bank, build_phase57_falsifier_bank,
    build_candidate_stream as build_p57_stream,
    _candidate_stats, run_phase57, run_phase57_seed_stability_sweep,
)
from vision_mvp.experiments.phase56_multi_service_corroboration import (
    build_phase56_bank,
    build_candidate_stream as build_p56_stream,
    _as_incident_scenario as as_incident_p56,
)
from vision_mvp.experiments.phase55_decoy_plurality import (
    build_phase55_bank,
    build_candidate_stream as build_p55_stream,
    _as_incident_scenario as as_incident_p55,
)
from vision_mvp.experiments.phase54_cross_role_coherence import (
    build_phase54_bank,
    build_candidate_stream as build_p54_stream,
    _as_incident_scenario as as_incident_p54,
)
from vision_mvp.tasks.incident_triage import grade_answer, ROLE_AUDITOR
from vision_mvp.experiments.phase52_team_coord import _format_canonical_answer

import re
_TAG_RE = re.compile(r"service=(\w+)")


# =============================================================================
# Unit tests — CCK table + decoder rule
# =============================================================================


class CCKTableTests(unittest.TestCase):
    """W10-2: CCK structural correctness."""

    def test_cck_data_tier_includes_data_tier_kinds(self) -> None:
        # The data-tier root_causes (deadlock / pool_exhaustion /
        # slow_query_cascade) all include the data-tier causal kinds.
        for r in ("deadlock", "pool_exhaustion", "slow_query_cascade"):
            cck = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE[r]
            self.assertIn("POOL_EXHAUSTION", cck, f"r={r}")
            self.assertIn("DEADLOCK_SUSPECTED", cck, f"r={r}")
            self.assertIn("SLOW_QUERY_OBSERVED", cck, f"r={r}")
            # Generic-noise kinds are NOT in data-tier CCK.
            self.assertNotIn("LATENCY_SPIKE", cck, f"r={r}")
            self.assertNotIn("ERROR_RATE_SPIKE", cck, f"r={r}")
            self.assertNotIn("FW_BLOCK_SURGE", cck, f"r={r}")

    def test_cck_storage_tier(self) -> None:
        cck = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE["disk_fill"]
        self.assertIn("DISK_FILL_CRITICAL", cck)
        self.assertIn("CRON_OVERRUN", cck)
        self.assertNotIn("LATENCY_SPIKE", cck)

    def test_cck_generic_tier_is_all_noise(self) -> None:
        # The generic-tier root_causes (error_spike / latency_spike)
        # have CCK = {ERROR_RATE_SPIKE, LATENCY_SPIKE} — bundle
        # decoder is a no-op here (W10-1 honest scope).
        cck_e = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE["error_spike"]
        cck_l = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE["latency_spike"]
        self.assertEqual(cck_e, frozenset({"ERROR_RATE_SPIKE",
                                              "LATENCY_SPIKE"}))
        self.assertEqual(cck_l, frozenset({"ERROR_RATE_SPIKE",
                                              "LATENCY_SPIKE"}))

    def test_unknown_root_cause_has_empty_cck(self) -> None:
        self.assertEqual(CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE["unknown"],
                          frozenset())


class BundleDecoderUnitTests(unittest.TestCase):
    """Direct unit tests of the decoder rule."""

    def _h(self, role: str, kind: str, tag: str) -> _DecodedHandoff:
        return _DecodedHandoff(role, kind, f"service={tag}")

    def test_filters_corroborated_decoy_via_noise(self) -> None:
        # gold_root_cause via DEADLOCK; decoy via LATENCY+FW only.
        hs = [
            self._h("db_admin", "DEADLOCK_SUSPECTED", "orders"),
            self._h("db_admin", "POOL_EXHAUSTION", "payments"),
            self._h("monitor", "LATENCY_SPIKE", "orders"),
            self._h("monitor", "LATENCY_SPIKE", "cache"),
            self._h("network", "FW_BLOCK_SURGE", "cache"),
        ]
        ans = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2).decode(hs)
        self.assertEqual(ans["root_cause"], "deadlock")
        self.assertEqual(ans["services"], ("orders", "payments"))

    def test_fallback_trusts_admission_on_small_set(self) -> None:
        # Admitted = {web, db}; |set|=2 ≤ threshold=2 → trust admission.
        hs = [
            self._h("db_admin", "SLOW_QUERY_OBSERVED", "db"),
            self._h("monitor", "LATENCY_SPIKE", "web"),
            self._h("network", "FW_BLOCK_SURGE", "web"),
        ]
        ans = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2).decode(hs)
        self.assertEqual(ans["root_cause"], "slow_query_cascade")
        self.assertEqual(ans["services"], ("db", "web"))

    def test_fallback_disabled_filters_strictly(self) -> None:
        # With fallback disabled, web is filtered (no CCK mention).
        hs = [
            self._h("db_admin", "SLOW_QUERY_OBSERVED", "db"),
            self._h("monitor", "LATENCY_SPIKE", "web"),
            self._h("network", "FW_BLOCK_SURGE", "web"),
        ]
        ans = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=0).decode(hs)
        self.assertEqual(ans["services"], ("db",))

    def test_unknown_root_cause_admits_all_tags(self) -> None:
        # No admitted handoffs → root_cause=unknown; CCK empty;
        # everything is admitted (no filter).
        hs = [
            self._h("monitor", "FOO_BAR", "x"),
        ]
        ans = BundleAwareTeamDecoder().decode(hs)
        self.assertEqual(ans["root_cause"], "unknown")
        self.assertEqual(ans["services"], ("x",))

    def test_priority_decoder_reproduces_substrate_priority(self) -> None:
        # Highest-priority kind in the admitted set wins.
        rc, _ = _decoded_root_cause(
            {"DISK_FILL_CRITICAL", "POOL_EXHAUSTION", "LATENCY_SPIKE"})
        self.assertEqual(rc, "disk_fill")
        rc, _ = _decoded_root_cause(
            {"POOL_EXHAUSTION", "LATENCY_SPIKE"})
        self.assertEqual(rc, "pool_exhaustion")
        rc, _ = _decoded_root_cause(set())
        self.assertEqual(rc, "unknown")


# =============================================================================
# Phase-57 bank-shape tests — bench property mechanically verified
# =============================================================================


class Phase57BankShapeTests(unittest.TestCase):
    """Pre-committed bench property: every Phase-57 default scenario
    satisfies the decoder-forcing property."""

    def test_default_bank_decoder_forcing_property_holds(self) -> None:
        bank = build_phase57_bank(seed=11)[:10]
        for sc in bank:
            cands = build_p57_stream(sc)
            stats = _candidate_stats(
                cands, sc.gold_services_pair,
                sc.decoy_storm_service, sc.gold_root_cause)
            self.assertTrue(
                stats["decoder_forcing_property_holds"],
                msg=f"{sc.scenario_id}: {stats}")
            # The decoy is corroborated by ≥ 2 distinct producer roles.
            self.assertGreaterEqual(stats["decoy_role_count"], 2,
                                       msg=sc.scenario_id)
            # The decoy is mentioned only via generic-noise kinds.
            self.assertTrue(stats["decoy_noise_only"],
                              msg=sc.scenario_id)
            # Both gold services are CCK-corroborated.
            self.assertTrue(
                stats["both_gold_corroborated_via_cck"],
                msg=sc.scenario_id)
            # Surplus over K_auditor=8 is present (or at least
            # |candidates_to_auditor| > K_producer=6 is present).
            self.assertGreaterEqual(stats["n_candidates_to_auditor"], 6,
                                       msg=sc.scenario_id)

    def test_falsifier_bank_decoy_cck_promoted(self) -> None:
        bank = build_phase57_falsifier_bank(seed=11)[:10]
        for sc in bank:
            cands = build_p57_stream(sc)
            stats = _candidate_stats(
                cands, sc.gold_services_pair,
                sc.decoy_storm_service, sc.gold_root_cause)
            self.assertTrue(stats["decoy_cck_promoted"],
                             msg=f"{sc.scenario_id}: {stats}")


# =============================================================================
# Phase-57 default-config tests — W10-Λ admission limit + W10-1 win
# =============================================================================


class Phase57DefaultConfigTests(unittest.TestCase):
    """Pre-committed default config (K_auditor=8, T_auditor=256,
    n_eval=10, bank_seed=11): bundle decoder strict win + admission
    limit witness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_phase57(
            n_eval=10, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=3, verbose=False)

    def test_bundle_decoder_strictly_beats_w9_at_K8(self) -> None:
        bd = self.report["pooled"]["capsule_bundle_decoder"]
        w9 = self.report["pooled"]["capsule_multi_service"]
        self.assertEqual(bd["accuracy_full"], 1.000,
                          msg=self.report["pooled"])
        self.assertEqual(w9["accuracy_full"], 0.000,
                          msg=self.report["pooled"])
        self.assertGreaterEqual(
            bd["accuracy_full"] - w9["accuracy_full"], 0.20)

    def test_w10_lambda_admission_limit_witness(self) -> None:
        # W10-Λ: every service-blind admission policy ties FIFO at 0.000.
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_cohort_buffered",
                   "capsule_corroboration", "capsule_multi_service"):
            self.assertEqual(
                self.report["pooled"][s]["accuracy_full"], 0.000,
                msg=f"{s} should tie FIFO at 0.000 by W10-Λ")
        self.assertEqual(
            self.report["headline_gap"]["max_admission_only_accuracy_full"],
            0.000)

    def test_audit_ok_grid_holds_for_every_capsule_strategy(self) -> None:
        # W10-aud: T-1..T-7 OK on every cell of every capsule strategy.
        grid = self.report["audit_ok_grid"]
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_cohort_buffered",
                   "capsule_corroboration", "capsule_multi_service",
                   "capsule_bundle_decoder"):
            self.assertTrue(grid[s], msg=f"{s} audit_ok_grid")

    def test_bench_property_holds_on_every_scenario(self) -> None:
        self.assertEqual(
            self.report["bench_summary"]["scenarios_with_decoder_forcing_property"],
            self.report["bench_summary"]["n_scenarios"])
        self.assertEqual(
            self.report["bench_summary"]["scenarios_with_decoy_cck_promotion"],
            0)


# =============================================================================
# Phase-57 seed-stability — gap holds across 5/5 seeds
# =============================================================================


class Phase57SeedStabilityTests(unittest.TestCase):
    """W10-1 stability anchor: bundle_decoder − fifo gap stable
    across 5/5 alternate bank seeds."""

    def test_gap_holds_across_five_seeds(self) -> None:
        rep = run_phase57_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31), n_eval=10,
            K_auditor=8, T_auditor=256)
        for seed, v in rep["per_seed"].items():
            gap = v["headline_gap"]["bundle_decoder_minus_fifo_accuracy_full"]
            self.assertGreaterEqual(
                gap, 0.20, msg=f"seed={seed}: gap={gap}")
            self.assertEqual(
                v["pooled"]["capsule_bundle_decoder"]["accuracy_full"],
                1.000, msg=f"seed={seed}")
            # Every admission policy ties FIFO at 0.
            for s in ("capsule_fifo", "capsule_multi_service",
                       "capsule_corroboration", "capsule_cohort_buffered"):
                self.assertEqual(
                    v["pooled"][s]["accuracy_full"], 0.000,
                    msg=f"seed={seed}, strategy={s}")


# =============================================================================
# Phase-57 falsifier — W10-4
# =============================================================================


class Phase57FalsifierTests(unittest.TestCase):
    """W10-4: when decoy is CCK-promoted, bundle decoder cannot
    exclude it; ties FIFO at 0.000."""

    def test_bundle_decoder_ties_fifo_on_falsifier(self) -> None:
        rep = run_phase57(
            n_eval=10, K_auditor=8, T_auditor=256,
            bank_seed=11, bank_replicates=3,
            use_falsifier_bank=True, verbose=False)
        bd = rep["pooled"]["capsule_bundle_decoder"]
        fifo = rep["pooled"]["capsule_fifo"]
        self.assertEqual(bd["accuracy_full"], 0.000)
        self.assertEqual(fifo["accuracy_full"], 0.000)
        self.assertEqual(
            rep["headline_gap"]["bundle_decoder_minus_fifo_accuracy_full"],
            0.0)
        # All scenarios satisfy the falsifier property.
        self.assertEqual(
            rep["bench_summary"]["scenarios_with_decoy_cck_promotion"],
            rep["bench_summary"]["n_scenarios"])
        # Audit still holds.
        for s, ok in rep["audit_ok_grid"].items():
            if s == "substrate":
                continue
            self.assertTrue(ok, msg=f"{s} audit broken on falsifier")


# =============================================================================
# Backward-compat — W10-3 across R-54 / R-55 / R-56
# =============================================================================


class Phase57BackwardCompatTests(unittest.TestCase):
    """W10-3: bundle decoder paired with the prior milestone's
    admission policy ties prior best on R-54 / R-55 / R-56 default.

    These tests run a *minimal* simulation of the joint admit-and-
    decode path on each prior bench rather than re-invoking the full
    coordinator (which would double the test wall-time). The result
    is the *services_correct* set the bundle decoder would emit
    given the pre-committed dominant-tag set from each prior
    milestone's admission policy.
    """

    def _eval_bundle_with_admission(self, bank, build_stream,
                                       as_incident, top_k):
        bd = BundleAwareTeamDecoder()
        n_full = 0
        for sc in bank:
            cands = build_stream(sc)
            cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            policy = (
                MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                    [(c[0], c[3]) for c in cands_aud], top_k=top_k,
                    min_corroborated_roles=2))
            admitted: list[_DecodedHandoff] = []
            for (src, _to, kind, payload, _evs) in cands_aud:
                m = _TAG_RE.search(payload)
                if m and (policy.fixed_dominant_tags is None
                            or m.group(1) in policy.fixed_dominant_tags):
                    admitted.append(_DecodedHandoff(src, kind, payload))
                    if len(admitted) >= 8:
                        break
            ans = bd.decode(admitted)
            inc = as_incident(sc)
            g = grade_answer(inc, _format_canonical_answer(ans))
            if g["full_correct"]:
                n_full += 1
        return n_full / len(bank)

    def test_bundle_decoder_matches_w7_2_on_phase54(self) -> None:
        # Phase 54 default: single-service gold; admission with top_k=1.
        acc = self._eval_bundle_with_admission(
            build_phase54_bank(seed=11)[:10],
            build_p54_stream, as_incident_p54, top_k=1)
        self.assertEqual(acc, 1.000)

    def test_bundle_decoder_matches_w8_on_phase55(self) -> None:
        # Phase 55 default: single-service gold + corroborated.
        acc = self._eval_bundle_with_admission(
            build_phase55_bank(seed=11)[:10],
            build_p55_stream, as_incident_p55, top_k=1)
        self.assertEqual(acc, 1.000)

    def test_bundle_decoder_matches_w9_on_phase56(self) -> None:
        # Phase 56 default: multi-service gold; admission top_k=2.
        acc = self._eval_bundle_with_admission(
            build_phase56_bank(seed=11)[:10],
            build_p56_stream, as_incident_p56, top_k=2)
        self.assertEqual(acc, 1.000)


# =============================================================================
# decode_admitted_role_view — convenience helper
# =============================================================================


class DecodeAdmittedRoleViewTests(unittest.TestCase):
    """``decode_admitted_role_view`` returns the unknown answer when
    the role-view CID is missing or absent from the ledger."""

    def test_missing_cid_returns_unknown(self) -> None:
        from vision_mvp.coordpy.capsule import CapsuleLedger
        ans = decode_admitted_role_view(CapsuleLedger(), "")
        self.assertEqual(ans["root_cause"], "unknown")
        self.assertEqual(ans["services"], ())


if __name__ == "__main__":
    unittest.main()
