"""Tests for SDK v3.10 — multi-service-corroboration multi-agent
benchmark (``vision_mvp.experiments.phase56_multi_service_corroboration``
+ ``vision_mvp.coordpy.team_coord.MultiServiceCorroborationAdmissionPolicy``).

These tests anchor the W9 family theorems empirically. They are
deterministic and CI-safe (no LLM IO).
"""

from __future__ import annotations

import unittest

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
)
from vision_mvp.coordpy.team_coord import (
    CohortCoherenceAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy,
    MultiServiceCorroborationAdmissionPolicy,
    REASON_ADMIT, REASON_SCORE_LOW,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_team_handoff, _dominant_tag_set,
)
from vision_mvp.experiments.phase52_team_coord import (
    make_team_budgets, pool,
)
from vision_mvp.experiments.phase55_decoy_plurality import (
    build_phase55_bank,
    build_candidate_stream as p55_stream,
    _run_capsule_strategy as p55_run,
)
from vision_mvp.experiments.phase56_multi_service_corroboration import (
    build_phase56_bank, build_phase56_falsifier_bank,
    build_candidate_stream, _candidate_stats,
    run_phase56, run_phase56_seed_stability_sweep,
    run_cross_regime_summary, _BASE_BUILDERS,
)


# =============================================================================
# MultiServiceCorroborationAdmissionPolicy — unit tests (W9 surface)
# =============================================================================


class MultiServicePolicyUnitTests(unittest.TestCase):
    """Direct ``decide(...)`` and ``from_candidate_stream`` tests."""

    def _budget(self) -> RoleBudget:
        return RoleBudget(role="auditor", K_role=8, T_role=128)

    def _cap(self, *, source_role: str, kind: str, payload: str):
        return capsule_team_handoff(
            source_role=source_role, to_role="auditor",
            claim_kind=kind, payload=payload, round=1)

    # -- W9-2 structural unit tests on the dominant-set fitter ------------

    def test_dominant_set_filters_single_role_decoy_storm(self) -> None:
        """W9-2 structural: a decoy with raw-count advantage but only
        one distinct producer role is excluded by the
        ``min_corroborated_roles`` threshold."""
        stream = [
            ("monitor", "p1 service=decoy"),
            ("monitor", "p2 service=decoy"),
            ("monitor", "p3 service=decoy"),
            ("monitor", "p4 service=decoy"),
            ("db_admin", "p5 service=api"),
            ("sysadmin", "p6 service=api"),
            ("monitor", "p7 service=db"),
            ("network", "p8 service=db"),
        ]
        p = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        self.assertEqual(set(p.fixed_dominant_tags), {"api", "db"})

    def test_dominant_set_admits_multi_service_gold(self) -> None:
        """W9-1 anchor: with both gold tags 2-role-corroborated and a
        single-role decoy storm, dominant set = {gold_A, gold_B}."""
        stream = [
            ("monitor", "p service=archival"),
            ("monitor", "p service=archival"),
            ("monitor", "p service=archival"),
            ("monitor", "p service=api"),
            ("monitor", "p service=db"),
            ("db_admin", "p service=api"),
            ("db_admin", "p service=db"),
        ]
        p = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        self.assertEqual(set(p.fixed_dominant_tags), {"api", "db"})

    def test_dominant_set_collapses_to_w8_under_role_count_argmax(
            self) -> None:
        """W9-3 backward-compat: when only one tag has the maximum
        distinct-role count, the argmax-by-role-count tier has size 1
        and W9 admits the same singleton as W8 single-tag
        corroboration. Phase 55 default has this property by
        construction (gold has 3 roles, every decoy ≤ 2 roles)."""
        stream = [
            ("monitor", "p service=decoy"),
            ("monitor", "p service=decoy"),
            ("db_admin", "p service=decoy"),  # decoy: 2 roles
            ("monitor", "p service=gold"),
            ("db_admin", "p service=gold"),
            ("network", "p service=gold"),    # gold: 3 roles
        ]
        p9 = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        p8 = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
            stream)
        self.assertEqual(set(p9.fixed_dominant_tags), {"gold"})
        self.assertEqual(p8.fixed_dominant_tag, "gold")

    def test_dominant_set_falsifier_admits_decoy_when_role_count_ties(
            self) -> None:
        """W9-4 falsifier: when the decoy is also corroborated by the
        max distinct-role count, the argmax tier includes it; the
        top-K cap then admits the decoy by score."""
        stream = [
            ("monitor", "p service=decoy"),
            ("monitor", "p service=decoy"),
            ("monitor", "p service=decoy"),
            ("monitor", "p service=decoy"),
            ("db_admin", "p service=decoy"),  # decoy: 2 roles, 5 raw
            ("monitor", "p service=api"),
            ("db_admin", "p service=api"),    # api: 2 roles, 2 raw
            ("network", "p service=db"),
            ("sysadmin", "p service=db"),     # db: 2 roles, 2 raw
        ]
        p = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        # All three tags are 2-role; argmax tier = {api, db, decoy}.
        # Scores: api=202, db=202, decoy=205. Top-2 = {decoy, api}
        # (lex tie on api over db at 202).
        self.assertIn("decoy", p.fixed_dominant_tags)
        self.assertEqual(len(p.fixed_dominant_tags), 2)

    def test_dominant_tag_set_helper_is_deterministic(self) -> None:
        """``_dominant_tag_set`` must produce a sorted tuple
        (deterministic admission set across tied scores / lex
        tie-break)."""
        d = _dominant_tag_set(
            count_per_tag={"a": 3, "b": 3, "c": 3},
            role_per_tag={"a": {"r1", "r2"}, "b": {"r1", "r2"},
                            "c": {"r1", "r2"}},
            role_weight=100, top_k=2, min_corroborated_roles=2)
        # All three tied on score 203; lex tie-break picks {a, b}.
        self.assertEqual(d, ("a", "b"))

    def test_dominant_set_empty_when_no_tag_passes_threshold(self) -> None:
        """If every tag has ``|distinct_roles| < min_corroborated_roles``,
        the dominant set is empty and the buffered policy admits
        nothing tagged."""
        stream = [
            ("monitor", "p service=alpha"),
            ("monitor", "p service=beta"),
            ("monitor", "p service=gamma"),
        ]
        p = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        self.assertEqual(p.fixed_dominant_tags, frozenset())

    # -- decide() integration ----------------------------------------------

    def test_decide_admits_dominant_tag(self) -> None:
        p = MultiServiceCorroborationAdmissionPolicy(
            fixed_dominant_tags=frozenset({"api", "db"}))
        cap = self._cap(source_role="db_admin",
                         kind="POOL_EXHAUSTION",
                         payload="pool active=200/200 service=api")
        d = p.decide(candidate=cap, role="auditor",
                      budget=self._budget(),
                      current_admitted=[], current_n_tokens=0)
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)

    def test_decide_rejects_non_dominant_tag(self) -> None:
        p = MultiServiceCorroborationAdmissionPolicy(
            fixed_dominant_tags=frozenset({"api", "db"}))
        cap = self._cap(source_role="monitor",
                         kind="ERROR_RATE_SPIKE",
                         payload="error_rate=0.04 service=archival")
        d = p.decide(candidate=cap, role="auditor",
                      budget=self._budget(),
                      current_admitted=[], current_n_tokens=0)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_SCORE_LOW)

    def test_decide_admits_untagged_candidate(self) -> None:
        """An untagged candidate cannot violate corroboration on a
        missing key; it is admitted (consistent with W7-2 / W8 design)."""
        p = MultiServiceCorroborationAdmissionPolicy(
            fixed_dominant_tags=frozenset({"api"}))
        cap = self._cap(source_role="monitor",
                         kind="LATENCY_SPIKE", payload="p95=1200")
        d = p.decide(candidate=cap, role="auditor",
                      budget=self._budget(),
                      current_admitted=[], current_n_tokens=0)
        self.assertTrue(d.admit)


# =============================================================================
# Phase 56 bench shape — pre-committed mechanical witnesses
# =============================================================================


class Phase56BankShapeTests(unittest.TestCase):
    """Mechanical contract tests on the Phase-56 bank construction.

    These tests verify the pre-committed bench properties hold *as
    constructed* — the success-criterion document forbids retroactively
    relaxing the property after seeing results.
    """

    def test_default_bank_size(self) -> None:
        bank = build_phase56_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 10)

    def test_falsifier_bank_size(self) -> None:
        bank = build_phase56_falsifier_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 10)

    def test_default_bank_multi_service_gold_property_holds(self) -> None:
        """Every default-bank scenario has BOTH gold services
        corroborated by ≥ 2 distinct producer roles."""
        bank = build_phase56_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(
                cands, sc.gold_services_pair, sc.decoy_storm_service)
            self.assertEqual(stats["n_gold_corroborated_roles_geq2"], 2,
                              msg=f"scenario={sc.scenario_id}")

    def test_default_bank_decoy_is_single_role_only(self) -> None:
        """Every default-bank scenario has every decoy service
        corroborated by ≤ 1 distinct producer role."""
        bank = build_phase56_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(
                cands, sc.gold_services_pair, sc.decoy_storm_service)
            self.assertLessEqual(stats["max_decoy_role_count"], 1,
                                  msg=f"scenario={sc.scenario_id}")

    def test_default_bank_has_budget_pressure(self) -> None:
        """Every default-bank scenario has |candidates_to_auditor| > K
        at K=4 — i.e. budget pressure forces the policy to choose."""
        bank = build_phase56_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            n_aud = sum(1 for c in cands if c[1] == ROLE_AUDITOR)
            self.assertGreater(n_aud, 4, msg=f"scenario={sc.scenario_id}")

    def test_falsifier_bank_breaks_decoy_single_role_property(
            self) -> None:
        """Every falsifier-bank scenario has at least one decoy
        corroborated by ≥ 2 distinct producer roles. This is the
        named W9-4 falsifier regime."""
        bank = build_phase56_falsifier_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(
                cands, sc.gold_services_pair, sc.decoy_storm_service)
            self.assertTrue(stats["decoy_corroboration_holds"],
                              msg=f"scenario={sc.scenario_id}")

    def test_default_bank_gold_pair_size_is_two(self) -> None:
        """Every default-bank scenario has gold_services of size 2
        (the multi-service axis Phase 56 introduces)."""
        bank = build_phase56_bank(n_replicates=2, seed=11)
        for sc in bank:
            self.assertEqual(len(set(sc.gold_services_pair)), 2,
                              msg=f"scenario={sc.scenario_id}")

    def test_each_base_builder_has_distinct_scenario_id(self) -> None:
        ids = [b().scenario_id for b in _BASE_BUILDERS]
        self.assertEqual(len(set(ids)), len(_BASE_BUILDERS))


# =============================================================================
# W9-1 — strict-separation default config win
# =============================================================================


class Phase56DefaultConfigTests(unittest.TestCase):
    """W9-1 anchor: on the Phase-56 default config (K_auditor=4,
    n_eval=10, bank_seed=11), capsule_multi_service strictly improves
    accuracy_full over substrate FIFO, capsule_fifo, and W7-2 buffered
    cohort, AND over W8 single-tag corroboration."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_phase56(
            n_eval=10, K_auditor=4, T_auditor=128,
            bank_seed=11, bank_replicates=2, verbose=False)

    def test_multi_service_strictly_beats_fifo_at_K4(self) -> None:
        gap = self.report["headline_gap"][
            "multi_service_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W9-1 strict separation: gap (multi_service − fifo) "
                f"= {gap:+.3f} ≱ +0.50")

    def test_multi_service_strictly_beats_w7_2_at_K4(self) -> None:
        gap = self.report["headline_gap"][
            "multi_service_minus_cohort_buffered_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W9-1 strict separation: gap (multi_service − W7-2) "
                f"= {gap:+.3f} ≱ +0.50")

    def test_multi_service_strictly_beats_w8_at_K4(self) -> None:
        """The hardest test: W9 multi-service strictly beats W8
        single-tag corroboration on Phase 56."""
        gap = self.report["headline_gap"][
            "multi_service_minus_corroboration_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50,
            msg=f"W9-1 strict separation: gap (multi_service − W8) "
                f"= {gap:+.3f} ≱ +0.50")

    def test_audit_ok_grid_holds_for_every_capsule_strategy(self) -> None:
        grid = self.report["audit_ok_grid"]
        for strategy, ok in grid.items():
            if strategy == "substrate":
                continue
            self.assertTrue(ok, msg=f"audit_ok=False for {strategy}")

    def test_bench_property_holds_on_all_default_scenarios(self) -> None:
        bench = self.report["bench_summary"]
        self.assertEqual(bench["scenarios_with_multi_service_gold_corroboration"],
                          bench["n_scenarios"])
        self.assertEqual(bench["scenarios_with_decoy_corroboration"], 0)

    def test_w8_falsifies_on_phase56(self) -> None:
        """W8 single-tag corroboration ties FIFO on Phase 56 (the W8
        multi-service-gold falsifier — named in
        ``HOW_NOT_TO_OVERSTATE.md``)."""
        w8_acc = self.report["pooled"]["capsule_corroboration"][
            "accuracy_full"]
        fifo_acc = self.report["pooled"]["capsule_fifo"][
            "accuracy_full"]
        self.assertEqual(w8_acc, fifo_acc,
            msg=f"W8 should tie FIFO on Phase 56, "
                f"got W8={w8_acc:.3f} fifo={fifo_acc:.3f}")


# =============================================================================
# W9-1 cross-bank stability — pre-committed seed sweep
# =============================================================================


class Phase56SeedStabilityTests(unittest.TestCase):
    """W9-1 stability anchor: the +0.50 gap holds across ≥ 3 distinct
    bank seeds. Pre-committed seed set: {11, 17, 23, 29, 31}."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_phase56_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31),
            n_eval=10, K_auditor=4, T_auditor=128)

    def test_gap_holds_across_five_seeds(self) -> None:
        gaps = []
        for seed, v in self.report["per_seed"].items():
            gap = v["headline_gap"][
                "multi_service_minus_corroboration_accuracy_full"]
            gaps.append((seed, gap))
        for seed, gap in gaps:
            self.assertGreaterEqual(gap, 0.50,
                msg=f"seed={seed}: gap (W9 − W8) = {gap:+.3f} ≱ +0.50")
        # And: ALL seeds agree on positive gap (no zero-stability cell)
        n_positive = sum(1 for (_s, g) in gaps if g >= 0.50)
        self.assertGreaterEqual(n_positive, 3,
            msg=f"only {n_positive}/5 seeds clear +0.50; need ≥ 3")


# =============================================================================
# W9-3 — backward-compat with Phase 55 / W8
# =============================================================================


class W9BackwardCompatTests(unittest.TestCase):
    """W9-3: on Phase-55 default (single-service-gold-corroborated),
    multi_service collapses to W8 single-tag corroboration via the
    argmax-by-role-count gate. Both achieve accuracy_full = 1.000."""

    def test_w9_matches_w8_on_phase55_default(self) -> None:
        bank = build_phase55_bank(n_replicates=2, seed=11)[:10]
        budgets = make_team_budgets(
            K_producer=6, T_producer=96, K_auditor=4, T_auditor=128)
        results = []
        for sc in bank:
            cands = p55_stream(sc)
            cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            w9 = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_aud],
                top_k=2, min_corroborated_roles=2)
            ppr = {r: w9 for r in budgets}
            res = p55_run(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=ppr,
                strategy_name="capsule_multi_service")
            results.append(res)
        p = pool(results, "capsule_multi_service").as_dict()
        # W8 achieves 1.000 on Phase 55 default; W9 must too (W9-3).
        self.assertGreaterEqual(p["accuracy_full"], 0.95)
        # And every scenario passes audit (T-1..T-7).
        for r in results:
            self.assertTrue(r.audit_ok)

    def test_w9_admits_same_set_as_w8_when_one_tag_dominates(
            self) -> None:
        """Mechanical: when the role-count argmax tier has size 1,
        W9's dominant set is identical to W8's fixed_dominant_tag."""
        # gold has strictly more distinct producer roles than every
        # decoy — the argmax tier is {gold}.
        stream = [
            ("monitor", "p service=decoy"),
            ("db_admin", "p service=decoy"),
            ("monitor", "p service=gold"),
            ("db_admin", "p service=gold"),
            ("network", "p service=gold"),
        ]
        w9 = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
            stream, top_k=2, min_corroborated_roles=2)
        w8 = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
            stream)
        self.assertEqual(set(w9.fixed_dominant_tags),
                          {w8.fixed_dominant_tag})


# =============================================================================
# W9-4 — falsifier regime
# =============================================================================


class Phase56FalsifierTests(unittest.TestCase):
    """W9-4 anchor: when the bench property *fails* (decoy is also
    cross-role-corroborated), capsule_multi_service ties FIFO at
    accuracy_full = 0.000."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_phase56(
            n_eval=10, K_auditor=4, T_auditor=128,
            bank_seed=11, bank_replicates=2,
            use_falsifier_bank=True, verbose=False)

    def test_multi_service_ties_fifo_on_falsifier(self) -> None:
        ms_acc = self.report["pooled"]["capsule_multi_service"][
            "accuracy_full"]
        fifo_acc = self.report["pooled"]["capsule_fifo"][
            "accuracy_full"]
        self.assertEqual(ms_acc, fifo_acc,
            msg=f"W9-4 falsifier: expected ms == fifo, "
                f"got ms={ms_acc:.3f} fifo={fifo_acc:.3f}")

    def test_decoy_corroboration_property_holds_on_falsifier_bank(
            self) -> None:
        bench = self.report["bench_summary"]
        self.assertEqual(bench["scenarios_with_decoy_corroboration"],
                          bench["n_scenarios"])

    def test_falsifier_bank_drops_multi_service_gold_property(
            self) -> None:
        """Mechanical: the falsifier bank promotes a decoy to ≥ 2
        distinct roles. Either the bench multi-service-gold property
        no longer holds OR the decoy-corroboration property holds."""
        bench = self.report["bench_summary"]
        self.assertGreater(bench["scenarios_with_decoy_corroboration"], 0)


# =============================================================================
# W9 audit-grid contract
# =============================================================================


class W9LifecycleAuditTests(unittest.TestCase):
    """T-1..T-7 hold for the multi_service strategy on every cell of
    the Phase-56 default and falsifier banks."""

    def test_default_bank_audit_ok_on_every_cell(self) -> None:
        bank = build_phase56_bank(n_replicates=2, seed=11)
        budgets = make_team_budgets(
            K_producer=6, T_producer=96, K_auditor=4, T_auditor=128)
        from vision_mvp.experiments.phase56_multi_service_corroboration import (
            _run_capsule_strategy)
        for sc in bank:
            cands = build_candidate_stream(sc)
            cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            w9 = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_aud],
                top_k=2, min_corroborated_roles=2)
            ppr = {r: w9 for r in budgets}
            res = _run_capsule_strategy(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=ppr,
                strategy_name="capsule_multi_service")
            self.assertTrue(res.audit_ok,
                msg=f"audit failed on {sc.scenario_id}")


# =============================================================================
# Cross-regime contract (R-53/R-54/R-55/R-56 + R-56-falsifier)
# =============================================================================


class CrossRegimeContractTests(unittest.TestCase):
    """The cross-regime summary report is the load-bearing artefact
    for the SDK v3.10 strong-success-bar claim."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_cross_regime_summary(
            n_eval=10, bank_seed=11, bank_replicates=2)

    def test_phase56_default_multi_service_wins(self) -> None:
        gap = self.report["phase56_default"]["headline_gap"][
            "multi_service_minus_corroboration_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50)

    def test_phase56_falsifier_multi_service_ties_fifo(self) -> None:
        ms = self.report["phase56_falsifier"]["pooled"][
            "capsule_multi_service"]["accuracy_full"]
        fifo = self.report["phase56_falsifier"]["pooled"][
            "capsule_fifo"]["accuracy_full"]
        self.assertEqual(ms, fifo)

    def test_phase55_w8_still_wins(self) -> None:
        """No regression on Phase 55 default — W8 corroboration
        still wins by +1.000 vs FIFO."""
        gap = self.report["phase55_default"]["headline_gap"][
            "corroboration_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50)


# =============================================================================
# Public-API: SDK v3.10 export contract
# =============================================================================


class PublicAPITests(unittest.TestCase):
    """The W9 multi_service policy is exported under the canonical
    SDK alias and SDK_VERSION advertises v3.10."""

    def test_canonical_alias_is_exported(self) -> None:
        from vision_mvp.coordpy import (
            TeamMultiServiceCorroborationAdmissionPolicy,
        )
        self.assertIs(
            TeamMultiServiceCorroborationAdmissionPolicy,
            MultiServiceCorroborationAdmissionPolicy)

    def test_sdk_version_is_at_least_v3_11(self) -> None:
        from vision_mvp.coordpy import SDK_VERSION
        # SDK v3.11+ — public surface preserved across v3.11/v3.12.
        self.assertTrue(
            SDK_VERSION.startswith("coordpy.sdk.v3."),
            msg=SDK_VERSION)
        minor = int(SDK_VERSION.split(".")[-1])
        self.assertGreaterEqual(minor, 11)

    def test_w8_export_is_preserved(self) -> None:
        """SDK v3.9 surface preserved: the W8 alias still works."""
        from vision_mvp.coordpy import (
            TeamCrossRoleCorroborationAdmissionPolicy,
        )
        self.assertIs(
            TeamCrossRoleCorroborationAdmissionPolicy,
            CrossRoleCorroborationAdmissionPolicy)


if __name__ == "__main__":
    unittest.main()
