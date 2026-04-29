"""Tests for SDK v3.21 — outside-witness acquisition disambiguator
+ W20 family.

Covers:

* Helper-class unit tests: :class:`ServiceGraphOracle` (deterministic
  asymmetric reply on connected gold pairs, abstention on isolated
  decoys), :class:`AbstainingOracle` (always None),
  :class:`CompromisedServiceGraphOracle` (decoy-asymmetric reply via
  gold-set blacklist).
* W20 decoder unit tests: trigger gating
  (:data:`W20_DEFAULT_TRIGGER_BRANCHES`) + branch determinism +
  bounded-context invariants.
* Phase-67 bench-property tests: every cell in every sub-bank carries
  the named R-66-OUTSIDE-REQUIRED bundle shape (decoy_only primary +
  all_three secondary).
* Phase-67 default-config tests: W20 strict win on
  R-67-OUTSIDE-RESOLVES (loose AND tight) with deterministic oracle
  (W20-1).
* Phase-67 5-seed stability: gap w20 − w19 ≥ 0.50 on every seed
  under R-67-OUTSIDE-RESOLVES-LOOSE AND R-67-OUTSIDE-RESOLVES-TIGHT.
* Phase-67 falsifier tests: W20 ties FIFO at 0.000 on
  R-67-OUTSIDE-NONE (W20-Λ-none), R-67-OUTSIDE-COMPROMISED
  (W20-Λ-compromised), AND R-67-JOINT-DECEPTION
  (W20-Λ-joint-deception).
* Phase-67 backward-compat: W20 reduces to W19 byte-for-byte on
  R-66 default banks (W20-3) AND on
  R-67-OUTSIDE-REQUIRED-BASELINE (no-oracle path).
* Phase-67 token-budget honesty: bounded-context invariant —
  ``n_outside_tokens ≤ max_response_tokens`` on every cell;
  W15 ``tokens_kept`` byte-for-byte unchanged from the W19 inner.
* Phase-67 audit T-1..T-7 preservation on every cell of every regime.
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase66_deceptive_ambiguity import (
    build_phase66_bank, run_phase66,
)
from vision_mvp.experiments.phase67_outside_information import (
    _bench_property_p67, _P67_EXPECTED_SHAPE,
    build_phase67_bank, run_phase67,
    run_phase67_seed_stability_sweep,
    run_cross_regime_synthetic,
)
from vision_mvp.wevra.team_coord import (
    AbstainingOracle, AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    CompromisedServiceGraphOracle,
    OutsideQuery, OutsideVerdict, OutsideWitnessAcquisitionDisambiguator,
    OutsideWitnessOracle,
    RelationalCompatibilityDisambiguator, ServiceGraphOracle,
    W19_BRANCH_ABSTAINED_NO_SIGNAL,
    W19_BRANCH_ABSTAINED_SYMMETRIC,
    W19_BRANCH_CONFOUND_RESOLVED,
    W19_BRANCH_PRIMARY_TRUSTED,
    W20_ALL_BRANCHES, W20_BRANCH_DISABLED, W20_BRANCH_NO_TRIGGER,
    W20_BRANCH_OUTSIDE_ABSTAINED, W20_BRANCH_OUTSIDE_RESOLVED,
    W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC,
    W20_DEFAULT_TRIGGER_BRANCHES,
    W20OutsideResult,
    _DecodedHandoff,
    _disambiguator_payload_tokens,
    build_incident_triage_service_graph,
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


def _w20_chain(oracle=None, T_decoder=None, *, enabled=True,
                 trigger_branches=None):
    """Build a fresh W20→W19→W18→W15 chain."""
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w20 = OutsideWitnessAcquisitionDisambiguator(
        inner=w19, oracle=oracle, enabled=enabled,
        trigger_branches=(trigger_branches
                            if trigger_branches is not None
                            else W20_DEFAULT_TRIGGER_BRANCHES))
    return w20, inner_w15


# =============================================================================
# Service-graph oracle unit tests
# =============================================================================


class ServiceGraphOracleTests(unittest.TestCase):

    def test_default_graph_has_gold_pairs(self):
        g = build_incident_triage_service_graph()
        # Each gold pair from _P66_FAMILIES must have a bidirectional edge.
        for a, b in (("orders", "payments"),
                      ("api", "db"),
                      ("storage", "logs_pipeline"),
                      ("web", "db_query")):
            self.assertIn(b, g[a])
            self.assertIn(a, g[b])

    def test_default_graph_decoys_isolated(self):
        g = build_incident_triage_service_graph()
        for d in ("search_index", "archival", "metrics", "telemetry",
                   "audit_jobs", "sessions", "cache", "scratch_pool"):
            self.assertEqual(g[d], frozenset())

    def test_oracle_emits_gold_pair_when_admitted(self):
        oracle = ServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="deadlock relation=cache_cache_join",
            witness_payloads=(),
        )
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertEqual(v.source_id, "service_graph")
        self.assertIn("orders", v.payload)
        self.assertIn("payments", v.payload)
        self.assertNotIn("cache", v.payload)

    def test_oracle_abstains_when_no_connected_pair(self):
        oracle = ServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("cache", "search_index", "metrics"),
            elected_root_cause="deadlock",
            primary_payload="",
            witness_payloads=(),
        )
        v = oracle.consult(q)
        self.assertIsNone(v.payload)
        self.assertEqual(v.n_tokens, 0)

    def test_oracle_abstains_when_singleton_admitted(self):
        oracle = ServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders",), elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)

    def test_oracle_truncates_to_max_response_tokens(self):
        oracle = ServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=(),
            max_response_tokens=3,
        )
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertLessEqual(v.n_tokens, 3)
        self.assertLessEqual(len(v.payload.split()), 3)

    def test_oracle_determinism(self):
        oracle = ServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="x", witness_payloads=("y",),
        )
        v1 = oracle.consult(q)
        v2 = oracle.consult(q)
        self.assertEqual(v1.payload, v2.payload)


class AbstainingOracleTests(unittest.TestCase):

    def test_always_returns_none(self):
        oracle = AbstainingOracle()
        for tags in (("orders", "payments", "cache"),
                      ("a", "b"), ("x",), ()):
            v = oracle.consult(OutsideQuery(
                admitted_tags=tags, elected_root_cause="x",
                primary_payload="", witness_payloads=()))
            self.assertIsNone(v.payload)
            self.assertEqual(v.n_tokens, 0)


class CompromisedOracleTests(unittest.TestCase):

    def test_emits_decoy_asymmetric_reply(self):
        oracle = CompromisedServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments", "cache"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNotNone(v.payload)
        self.assertIn("cache", v.payload)
        self.assertNotIn("orders", v.payload)
        self.assertNotIn("payments", v.payload)

    def test_abstains_when_all_admitted_are_gold(self):
        oracle = CompromisedServiceGraphOracle()
        q = OutsideQuery(
            admitted_tags=("orders", "payments"),
            elected_root_cause="deadlock",
            primary_payload="", witness_payloads=())
        v = oracle.consult(q)
        self.assertIsNone(v.payload)


# =============================================================================
# W20 decoder unit tests
# =============================================================================


class W20DecoderUnitTests(unittest.TestCase):

    def test_outside_resolved_on_outside_required_shape(self):
        """W19 abstains via SYMMETRIC; W20 + ServiceGraphOracle resolves."""
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20, _ = _w20_chain(oracle=ServiceGraphOracle())
        ans = w20.decode_rounds(per_round)
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        out = ans["outside"]
        self.assertEqual(out["decoder_branch"], W20_BRANCH_OUTSIDE_RESOLVED)
        self.assertTrue(out["triggered"])
        self.assertEqual(out["inner_branch"],
                         W19_BRANCH_ABSTAINED_SYMMETRIC)
        self.assertGreater(out["n_outside_tokens"], 0)

    def test_outside_abstain_with_abstaining_oracle(self):
        """W19 abstains via SYMMETRIC; W20 + AbstainingOracle abstains."""
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20, _ = _w20_chain(oracle=AbstainingOracle())
        ans = w20.decode_rounds(per_round)
        out = ans["outside"]
        self.assertEqual(out["decoder_branch"], W20_BRANCH_OUTSIDE_ABSTAINED)
        self.assertTrue(out["triggered"])
        self.assertEqual(out["n_outside_tokens"], 0)
        # Falls through to W19's empty answer.
        self.assertEqual(tuple(ans["services"]), ())

    def test_outside_compromised_picks_decoy(self):
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20, _ = _w20_chain(oracle=CompromisedServiceGraphOracle())
        ans = w20.decode_rounds(per_round)
        # W20 trusts the compromised oracle and projects to {cache}.
        self.assertEqual(tuple(ans["services"]), ("cache",))

    def test_no_trigger_when_inner_w19_resolves(self):
        """On a regime where W19 already resolves (R-66-DECEIVE-NAIVE
        shape), W20 should reduce to W19 byte-for-byte (no trigger)."""
        # Build an R-66-DECEIVE-NAIVE bundle: primary = decoy only,
        # secondary = gold only.
        A, B, decoy = "orders", "payments", "cache"
        per_round = [
            _r1_symmetric(A, B, decoy),
            [_DecodedHandoff(
                "db_admin", "DEADLOCK_SUSPECTED",
                f"deadlock relation={decoy}_{decoy}_join wait_chain=2"),
             _DecodedHandoff(
                "monitor", "DEADLOCK_DETECTED",
                f"deadlock relation={A}_{B}_join detected_at=t=120")],
        ]
        w20, _ = _w20_chain(oracle=ServiceGraphOracle())
        ans = w20.decode_rounds(per_round)
        # W19 would resolve to {orders, payments} via confound branch.
        self.assertEqual(set(ans["services"]), {"orders", "payments"})
        out = ans["outside"]
        self.assertEqual(out["decoder_branch"], W20_BRANCH_NO_TRIGGER)
        self.assertFalse(out["triggered"])
        self.assertEqual(out["n_outside_tokens"], 0)

    def test_disabled_path_reduces_to_inner_w19(self):
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20_dis, _ = _w20_chain(oracle=ServiceGraphOracle(), enabled=False)
        ans_dis = w20_dis.decode_rounds(per_round)
        out = ans_dis["outside"]
        self.assertEqual(out["decoder_branch"], W20_BRANCH_DISABLED)
        # Compare against the W19-only inner pipeline.
        w19 = BundleContradictionDisambiguator(
            inner=RelationalCompatibilityDisambiguator(
                inner=AttentionAwareBundleDecoder()))
        ans_w19 = w19.decode_rounds(per_round)
        self.assertEqual(tuple(ans_dis["services"]),
                         tuple(ans_w19["services"]))

    def test_no_oracle_path_treated_like_abstain(self):
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20, _ = _w20_chain(oracle=None)
        ans = w20.decode_rounds(per_round)
        out = ans["outside"]
        self.assertEqual(out["decoder_branch"], W20_BRANCH_OUTSIDE_ABSTAINED)
        self.assertEqual(out["oracle_consulted"], "no_oracle")

    def test_determinism(self):
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w20a, _ = _w20_chain(oracle=ServiceGraphOracle())
        w20b, _ = _w20_chain(oracle=ServiceGraphOracle())
        a = w20a.decode_rounds(per_round)
        b = w20b.decode_rounds(per_round)
        self.assertEqual(tuple(a["services"]), tuple(b["services"]))
        self.assertEqual(a["outside"]["per_tag_outside_count"],
                         b["outside"]["per_tag_outside_count"])

    def test_branches_well_typed(self):
        for branch in W20_ALL_BRANCHES:
            self.assertIsInstance(branch, str)
            self.assertGreater(len(branch), 0)

    def test_default_trigger_set(self):
        self.assertIn(W19_BRANCH_ABSTAINED_SYMMETRIC,
                       W20_DEFAULT_TRIGGER_BRANCHES)
        self.assertIn(W19_BRANCH_ABSTAINED_NO_SIGNAL,
                       W20_DEFAULT_TRIGGER_BRANCHES)
        self.assertNotIn(W19_BRANCH_CONFOUND_RESOLVED,
                          W20_DEFAULT_TRIGGER_BRANCHES)
        self.assertNotIn(W19_BRANCH_PRIMARY_TRUSTED,
                          W20_DEFAULT_TRIGGER_BRANCHES)


# =============================================================================
# Phase-67 bench-property tests
# =============================================================================


class Phase67BenchPropertyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from vision_mvp.experiments.phase66_deceptive_ambiguity import (
            _build_round_candidates_p66,
        )
        cls._build_p66 = staticmethod(_build_round_candidates_p66)
        cls.banks = ("outside_required_baseline", "outside_resolves",
                      "outside_none", "outside_compromised",
                      "joint_deception")

    def test_every_bank_holds_outside_required_shape(self):
        for bank in self.banks:
            scenarios = build_phase67_bank(bank=bank, n_replicates=2,
                                              seed=11)
            for sc in scenarios:
                round1 = self._build_p66(sc.round1_emissions)
                round2 = self._build_p66(sc.round2_emissions)
                bench = _bench_property_p67(sc, round1, round2)
                self.assertTrue(
                    bench["symmetric_corroboration_holds"],
                    f"sym corr fails on {bank}/{sc.scenario_id}")
                self.assertEqual(
                    tuple(bench["shape"]), _P67_EXPECTED_SHAPE,
                    f"shape mismatch on {bank}/{sc.scenario_id}: "
                    f"{bench['shape']}")


# =============================================================================
# Phase-67 default-config tests
# =============================================================================


class Phase67DefaultConfigTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r_resolves_loose = run_phase67(
            bank="outside_resolves", T_decoder=None,
            n_eval=8, K_auditor=12)
        cls.r_resolves_tight = run_phase67(
            bank="outside_resolves", T_decoder=24,
            n_eval=8, K_auditor=12)
        cls.r_baseline = run_phase67(
            bank="outside_required_baseline", T_decoder=None,
            n_eval=8, K_auditor=12)

    def test_w20_strict_win_resolves_loose(self):
        p = self.r_resolves_loose["pooled"]
        self.assertGreaterEqual(p["capsule_outside_witness"]["accuracy_full"],
                                 0.99)
        self.assertLessEqual(
            p["capsule_bundle_contradiction"]["accuracy_full"], 0.01)
        gap = self.r_resolves_loose["headline_gap"]["w20_minus_w19"]
        self.assertGreaterEqual(gap, 0.99)

    def test_w20_strict_win_resolves_tight(self):
        p = self.r_resolves_tight["pooled"]
        self.assertGreaterEqual(p["capsule_outside_witness"]["accuracy_full"],
                                 0.99)
        gap = self.r_resolves_tight["headline_gap"]["w20_minus_w19"]
        self.assertGreaterEqual(gap, 0.99)

    def test_w20_baseline_ties_w19_at_zero(self):
        p = self.r_baseline["pooled"]
        # No oracle ⇒ W20 reduces to W19 abstention; ties FIFO.
        self.assertLessEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.01)
        self.assertLessEqual(
            p["capsule_bundle_contradiction"]["accuracy_full"], 0.01)

    def test_w20_branches_outside_resolved_on_resolves(self):
        bc = self.r_resolves_loose["w20_branch_counts"]
        self.assertEqual(bc.get(W20_BRANCH_OUTSIDE_RESOLVED, 0), 8)

    def test_w20_audit_OK_on_every_cell(self):
        for r in (self.r_resolves_loose, self.r_resolves_tight,
                  self.r_baseline):
            grid = r["audit_ok_grid"]
            for k, v in grid.items():
                if k == "substrate":
                    continue
                self.assertTrue(v, f"audit failed on {k}")


# =============================================================================
# Phase-67 5-seed stability
# =============================================================================


class Phase67SeedStabilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sweep_loose = run_phase67_seed_stability_sweep(
            bank="outside_resolves", T_decoder=None,
            n_eval=8, K_auditor=12)
        cls.sweep_tight = run_phase67_seed_stability_sweep(
            bank="outside_resolves", T_decoder=24,
            n_eval=8, K_auditor=12)

    def test_loose_min_gap_at_strong_bar(self):
        self.assertGreaterEqual(self.sweep_loose["min_w20_minus_w19"], 0.99)

    def test_tight_min_gap_at_strong_bar(self):
        self.assertGreaterEqual(self.sweep_tight["min_w20_minus_w19"], 0.99)

    def test_w20_one_thousand_on_every_seed(self):
        for s, v in self.sweep_loose["per_seed"].items():
            self.assertGreaterEqual(
                v["pooled"]["capsule_outside_witness"], 0.99,
                f"loose seed {s} below 0.99")
        for s, v in self.sweep_tight["per_seed"].items():
            self.assertGreaterEqual(
                v["pooled"]["capsule_outside_witness"], 0.99,
                f"tight seed {s} below 0.99")


# =============================================================================
# Phase-67 falsifier tests (W20-Λ-* family)
# =============================================================================


class Phase67FalsifierTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r_none = run_phase67(bank="outside_none", T_decoder=None,
                                  n_eval=8, K_auditor=12)
        cls.r_compromised = run_phase67(
            bank="outside_compromised", T_decoder=None,
            n_eval=8, K_auditor=12)
        cls.r_joint = run_phase67(
            bank="joint_deception", T_decoder=None,
            n_eval=8, K_auditor=12)

    def test_w20_lambda_none_ties_fifo(self):
        p = self.r_none["pooled"]
        self.assertLessEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.01)
        bc = self.r_none["w20_branch_counts"]
        self.assertEqual(bc.get(W20_BRANCH_OUTSIDE_ABSTAINED, 0), 8)

    def test_w20_lambda_compromised_fails(self):
        p = self.r_compromised["pooled"]
        # W20 trusts the compromised oracle and projects to decoy.
        self.assertLessEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.01)
        # The branch fired (oracle replied), but the answer is wrong.
        bc = self.r_compromised["w20_branch_counts"]
        self.assertEqual(bc.get(W20_BRANCH_OUTSIDE_RESOLVED, 0), 8)

    def test_w20_lambda_joint_deception_fails(self):
        p = self.r_joint["pooled"]
        self.assertLessEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.01)


# =============================================================================
# Phase-67 backward-compat
# =============================================================================


class Phase67BackwardCompatTests(unittest.TestCase):

    def test_w20_matches_w19_on_phase66_default_banks(self):
        """W20 with deterministic ServiceGraphOracle ties W19
        byte-for-byte on R-66 default banks (no-trigger path because
        W19 resolves)."""
        for bank in ("corroborated", "deceive_naive",
                      "confound_resolvable"):
            r = run_phase66(bank=bank, T_decoder=None,
                             n_eval=4, K_auditor=12)
            w19_acc = r["pooled"]["capsule_bundle_contradiction"][
                "accuracy_full"]
            # Construct a W20 chain inside the same per-cell loop and
            # compare its answers cell-by-cell with W19. We do this
            # by running the bench-equivalent chain manually.
            scenarios = build_phase66_bank(bank=bank, n_replicates=2,
                                              seed=11)[:4]
            from vision_mvp.experiments.phase66_deceptive_ambiguity import (
                _build_round_candidates_p66,
            )
            for sc in scenarios:
                r1 = _build_round_candidates_p66(sc.round1_emissions)
                r2 = _build_round_candidates_p66(sc.round2_emissions)
                # Build flat per-round handoff lists for the auditor
                # by taking the candidates routed to the auditor.
                auditor_r1 = [
                    _DecodedHandoff(s, k, p)
                    for (s, t, k, p, _e) in r1 if t == "auditor"]
                auditor_r2 = [
                    _DecodedHandoff(s, k, p)
                    for (s, t, k, p, _e) in r2 if t == "auditor"]
                w19 = BundleContradictionDisambiguator(
                    inner=RelationalCompatibilityDisambiguator(
                        inner=AttentionAwareBundleDecoder()))
                w20, _ = _w20_chain(oracle=ServiceGraphOracle())
                a19 = w19.decode_rounds([auditor_r1, auditor_r2])
                a20 = w20.decode_rounds([auditor_r1, auditor_r2])
                self.assertEqual(
                    tuple(a19["services"]), tuple(a20["services"]),
                    f"W20 deviates from W19 on {bank}/{sc.scenario_id}")

    def test_w20_disabled_reduces_to_w19_byte_for_byte(self):
        per_round = [_r1_symmetric("orders", "payments", "cache"),
                      _r2_outside_required("orders", "payments", "cache")]
        w19 = BundleContradictionDisambiguator(
            inner=RelationalCompatibilityDisambiguator(
                inner=AttentionAwareBundleDecoder()))
        a19 = w19.decode_rounds(per_round)
        w20_dis, _ = _w20_chain(oracle=ServiceGraphOracle(), enabled=False)
        a20 = w20_dis.decode_rounds(per_round)
        self.assertEqual(tuple(a19["services"]), tuple(a20["services"]))


# =============================================================================
# Phase-67 bounded-context honesty (token-budget invariants)
# =============================================================================


class Phase67TokenBudgetHonestyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r_loose = run_phase67(
            bank="outside_resolves", T_decoder=None,
            n_eval=8, K_auditor=12)
        cls.r_tight = run_phase67(
            bank="outside_resolves", T_decoder=24,
            n_eval=8, K_auditor=12)

    def test_n_outside_tokens_per_cell_bounded(self):
        """Bounded-context invariant: n_outside_tokens per cell ≤ 24
        (the default :class:`OutsideQuery.max_response_tokens`)."""
        for r in (self.r_loose, self.r_tight):
            ps = r["pack_stats_summary"]["capsule_outside_witness"]
            avg = ps["outside_tokens_per_cell_avg"]
            self.assertLessEqual(avg, 24.0,
                                  f"avg {avg} exceeds budget")
            self.assertGreater(avg, 0.0,
                                "expected nonzero outside tokens")

    def test_w20_does_not_inflate_w15_tokens_kept(self):
        """W20 must not change the inner W15 ``tokens_kept`` figure
        relative to W19 — the W20 layer reads the same W15-packed
        bundle the W19 layer reads."""
        w19_ps = self.r_tight["pack_stats_summary"][
            "capsule_bundle_contradiction"]
        w20_ps = self.r_tight["pack_stats_summary"][
            "capsule_outside_witness"]
        self.assertEqual(w19_ps["tokens_kept_sum"],
                         w20_ps["tokens_kept_sum"])
        self.assertEqual(w19_ps["handoffs_decoder_input_sum"],
                         w20_ps["handoffs_decoder_input_sum"])

    def test_strict_tight_budget_holds(self):
        """Under T_decoder=24, W15 ``tokens_kept_over_input`` ≤ 1.0."""
        w20_ps = self.r_tight["pack_stats_summary"][
            "capsule_outside_witness"]
        self.assertLessEqual(w20_ps["tokens_kept_over_input"], 1.0)


# =============================================================================
# Phase-67 cross-regime synthetic summary
# =============================================================================


class Phase67CrossRegimeSyntheticTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = run_cross_regime_synthetic(
            n_eval=8, K_auditor=12)

    def test_outside_resolves_loose_at_one_thousand(self):
        p = self.r["r67_outside_resolves_loose"]["pooled"]
        self.assertGreaterEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.99)

    def test_outside_resolves_tight_at_one_thousand(self):
        p = self.r["r67_outside_resolves_tight"]["pooled"]
        self.assertGreaterEqual(
            p["capsule_outside_witness"]["accuracy_full"], 0.99)

    def test_falsifiers_at_zero(self):
        for cell in ("r67_outside_required_baseline",
                      "r67_outside_none",
                      "r67_outside_compromised",
                      "r67_joint_deception"):
            p = self.r[cell]["pooled"]
            self.assertLessEqual(
                p["capsule_outside_witness"]["accuracy_full"], 0.01,
                f"{cell} should fail by construction")

    def test_strict_gain_summary(self):
        h = self.r["headline_summary"]
        self.assertGreaterEqual(h["w20_minus_w19_outside_resolves_loose"],
                                 0.99)
        self.assertGreaterEqual(h["w20_minus_w19_outside_resolves_tight"],
                                 0.99)


if __name__ == "__main__":
    unittest.main()
