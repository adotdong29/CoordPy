"""Tests for SDK v3.9 — cross-role corroboration multi-agent benchmark
(``vision_mvp.experiments.phase55_decoy_plurality`` +
``vision_mvp.coordpy.team_coord.CrossRoleCorroborationAdmissionPolicy``).

These tests anchor the W8 family theorems empirically. They are
designed to be runnable in CI without network IO; the bench is a
deterministic synthetic candidate stream.
"""

from __future__ import annotations

import re
import unittest

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    CrossRoleCorroborationAdmissionPolicy, FifoAdmissionPolicy,
    REASON_ADMIT, REASON_SCORE_LOW, REASON_BUDGET_FULL,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_team_handoff,
)
from vision_mvp.experiments.phase55_decoy_plurality import (
    build_phase55_bank, build_phase55_falsifier_bank,
    build_candidate_stream, _candidate_stats,
    run_phase55, run_phase55_budget_sweep,
    run_seed_stability_sweep, run_cross_regime_summary,
    _BASE_BUILDERS,
)


# ---------------------------------------------------------------------------
# CrossRoleCorroborationAdmissionPolicy — unit tests
# ---------------------------------------------------------------------------


class CorroborationPolicyUnitTests(unittest.TestCase):
    """Direct ``decide(...)`` and ``from_candidate_stream`` tests."""

    def _budget(self) -> RoleBudget:
        return RoleBudget(role="auditor", K_role=8, T_role=128)

    def _cap(self, *, source_role: str, kind: str, payload: str):
        return capsule_team_handoff(
            source_role=source_role, to_role="auditor",
            claim_kind=kind, payload=payload, round=1)

    # -- from_candidate_stream --------------------------------------------

    def test_buffered_picks_corroborated_over_raw_plurality(self) -> None:
        """The W8-2 strict-ordering anchor: a decoy with raw-count
        advantage of `Δr ≤ 99` and role-coverage disadvantage of
        `Δr_role ≥ 1` is dominated by the corroboration score."""
        stream = [
            ("monitor",  "p1 service=archival"),
            ("monitor",  "p2 service=archival"),
            ("monitor",  "p3 service=archival"),
            ("monitor",  "p4 service=archival"),  # archival: 4 mentions, 1 role
            ("db_admin", "p5 service=api"),
            ("sysadmin", "p6 service=api"),
            ("network",  "p7 service=api"),       # api: 3 mentions, 3 roles
        ]
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        self.assertEqual(p.fixed_dominant_tag, "api")

    def test_buffered_matches_W7_2_on_gold_plurality(self) -> None:
        """W8-3 backward compatibility: when gold has both raw
        plurality AND distinct-role coverage, corroboration agrees
        with W7-2 single-tag plurality."""
        stream = [
            ("monitor",  "p1 service=api"),
            ("db_admin", "p2 service=api"),
            ("sysadmin", "p3 service=api"),
            ("monitor",  "p4 service=archival"),
            ("db_admin", "p5 service=archival"),
        ]
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        self.assertEqual(p.fixed_dominant_tag, "api")

    def test_buffered_falsifier_decoy_wins(self) -> None:
        """The W8-1 falsifier: when a decoy has BOTH more raw mentions
        AND more distinct roles than gold, the policy picks the
        decoy. The W8-1 win does not hold in this regime — by design."""
        stream = [
            ("monitor",  "p1 service=archival"),
            ("db_admin", "p2 service=archival"),
            ("sysadmin", "p3 service=archival"),
            ("network",  "p4 service=archival"),  # 4 mentions, 4 roles
            ("monitor",  "p5 service=api"),       # 1 mention, 1 role
        ]
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        self.assertEqual(p.fixed_dominant_tag, "archival")

    def test_buffered_no_tags_returns_streaming_default(self) -> None:
        stream = [("monitor", "p1 no tag"), ("db_admin", "p2 q#1")]
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        self.assertIsNone(p.fixed_dominant_tag)

    def test_buffered_lex_breaks_score_ties(self) -> None:
        """When two tags score identically, lex order breaks
        deterministically."""
        stream = [
            ("monitor",  "p1 service=zebra"),
            ("db_admin", "p2 service=zebra"),
            ("monitor",  "p3 service=alpha"),
            ("db_admin", "p4 service=alpha"),
        ]
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        # Both: 2 mentions, 2 roles → tied score → lex picks alpha.
        self.assertEqual(p.fixed_dominant_tag, "alpha")

    # -- W8-2 score-function structural property --------------------------

    def test_w8_2_role_weight_dominates_raw_count(self) -> None:
        """W8-2 anchor: with default ``role_weight=100``, no raw
        count difference up to 99 can override a 1-role coverage
        advantage."""
        # gold: 1 mention, 1 role. score = 100*1 + 1 = 101.
        # decoy: 99 mentions, 0 roles ... but every mention requires
        # a role. So the realistic maximum: decoy with 99 mentions
        # all from the same role = 1 role. Score = 100*1 + 99 = 199.
        # gold MUST also be 1+ role to be in the stream. Test the
        # boundary case where decoy has 1 role + 50 mentions and gold
        # has 2 roles + 1 mention each:
        stream = []
        for i in range(50):
            stream.append(("monitor", f"p{i} service=decoy"))
        stream.append(("monitor",  "g1 service=gold"))
        stream.append(("db_admin", "g2 service=gold"))
        # decoy: 50 mentions, 1 role → score 100*1 + 50 = 150
        # gold:   2 mentions, 2 roles → score 100*2 + 2 = 202
        p = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
        self.assertEqual(p.fixed_dominant_tag, "gold")

    # -- decide() per-call ------------------------------------------------

    def test_decide_admits_dominant_tag(self) -> None:
        p = CrossRoleCorroborationAdmissionPolicy(fixed_dominant_tag="api")
        cand = self._cap(source_role="db_admin", kind="POOL_EXHAUSTION",
                          payload="pool service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)

    def test_decide_rejects_foreign_tag(self) -> None:
        p = CrossRoleCorroborationAdmissionPolicy(fixed_dominant_tag="api")
        cand = self._cap(source_role="db_admin", kind="POOL_EXHAUSTION",
                          payload="pool service=archival")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_SCORE_LOW)

    def test_decide_admits_no_tag(self) -> None:
        """Untagged candidates cannot violate corroboration on a
        missing key — admit (consistent with W7-2 design)."""
        p = CrossRoleCorroborationAdmissionPolicy(fixed_dominant_tag="api")
        cand = self._cap(source_role="db_admin", kind="DEADLOCK_SUSPECTED",
                          payload="deadlock relation=orders_payments")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)

    def test_budget_pre_check_overrides_corroboration_rule(self) -> None:
        """Structural budget pre-checks (K_role full) deny before
        the corroboration rule runs."""
        p = CrossRoleCorroborationAdmissionPolicy(fixed_dominant_tag="api")
        budget = RoleBudget(role="auditor", K_role=1, T_role=128)
        already = self._cap(source_role="monitor", kind="POOL_EXHAUSTION",
                              payload="pool service=api")
        cand = self._cap(source_role="db_admin", kind="OOM_KILL",
                          payload="oom service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=budget, current_admitted=(already,),
                      current_n_tokens=1)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_BUDGET_FULL)

    def test_streaming_admits_first_no_cohort(self) -> None:
        """Streaming mode admits the first tagged candidate before
        any cohort exists."""
        p = CrossRoleCorroborationAdmissionPolicy()
        cand = self._cap(source_role="monitor", kind="LATENCY_SPIKE",
                          payload="p service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)


# ---------------------------------------------------------------------------
# Phase 55 bench shape — pre-committed bench-property tests
# ---------------------------------------------------------------------------


class Phase55BankShapeTests(unittest.TestCase):
    """The Phase 55 default bank must satisfy *both* the
    decoy-plurality property AND the gold-corroboration property
    on every scenario. Otherwise the W8-1 anchor is invalid."""

    def test_bank_size_matches_replicates(self) -> None:
        bank = build_phase55_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 10)
        bank3 = build_phase55_bank(n_replicates=3, seed=11)
        self.assertEqual(len(bank3), 15)

    def test_every_scenario_has_decoy_plurality(self) -> None:
        """W8-1 prerequisite: in every default scenario, some decoy
        service must have STRICTLY MORE raw mentions than the gold
        service. (This makes W7-2 buffered cohort fail by construction.)"""
        bank = build_phase55_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(cands, sc.real_service)
            self.assertTrue(
                stats["decoy_plurality_holds"],
                msg=f"{sc.scenario_id}: gold_raw={stats['real_raw']}; "
                     f"max_decoy_raw={stats['max_decoy_raw']}")

    def test_every_scenario_has_gold_corroboration(self) -> None:
        """W8-1 prerequisite: in every default scenario, the gold
        service must be mentioned by STRICTLY MORE distinct producer
        roles than any decoy service."""
        bank = build_phase55_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(cands, sc.real_service)
            self.assertTrue(
                stats["gold_corroboration_holds"],
                msg=f"{sc.scenario_id}: gold_roles={stats['real_roles']}; "
                     f"max_decoy_roles={stats['max_decoy_roles']}")

    def test_every_scenario_has_surplus(self) -> None:
        """At K_auditor=4, every scenario's auditor candidate count
        must exceed the budget. (Otherwise no admission is needed and
        FIFO ties trivially.)"""
        bank = build_phase55_bank(n_replicates=2, seed=11)
        K_auditor = 4
        for sc in bank:
            cands = build_candidate_stream(sc)
            n_aud = sum(1 for c in cands if c[1] == ROLE_AUDITOR)
            self.assertGreater(
                n_aud, K_auditor,
                msg=f"{sc.scenario_id}: |cands_to_auditor|={n_aud} <= "
                     f"K_auditor={K_auditor}")

    def test_scenarios_have_at_least_three_corroborating_roles(self) -> None:
        """The strong cross-role hypothesis: gold is mentioned by at
        least 3 distinct producer roles in every scenario. This
        sharpens the W8-1 win to require *cross-role* corroboration,
        not just two-role agreement."""
        bank = build_phase55_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(cands, sc.real_service)
            self.assertGreaterEqual(
                stats["real_roles"], 3,
                msg=f"{sc.scenario_id}: gold_roles={stats['real_roles']} < 3")

    def test_decoy_services_disjoint_from_real_and_each_other(self) -> None:
        bank = build_phase55_bank(n_replicates=2, seed=11)
        for sc in bank:
            self.assertNotIn(sc.real_service, (sc.decoy_storm_service,
                                                  sc.decoy_secondary_service))
            self.assertNotEqual(sc.decoy_storm_service,
                                 sc.decoy_secondary_service)

    def test_falsifier_bank_drops_corroboration_property(self) -> None:
        """The W8-1 falsifier bank must NOT satisfy gold_corroboration."""
        bank = build_phase55_falsifier_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            stats = _candidate_stats(cands, sc.real_service)
            self.assertFalse(
                stats["gold_corroboration_holds"],
                msg=(f"{sc.scenario_id}: falsifier still has "
                      f"gold_corroboration: gold_roles="
                      f"{stats['real_roles']} > "
                      f"max_decoy_roles={stats['max_decoy_roles']}"))


# ---------------------------------------------------------------------------
# Phase 55 driver — pre-committed default config result (W8-1 anchor)
# ---------------------------------------------------------------------------


class Phase55DefaultConfigTests(unittest.TestCase):
    """W8-1 anchor: on the pre-committed default
    (K_auditor=4, T_auditor=128, n_eval=10, bank_seed=11),
    capsule_corroboration strictly outperforms BOTH substrate FIFO
    AND SDK v3.8 W7-2 buffered cohort by ≥ 0.50 on accuracy_full.
    This is the canonical Phase-55 contract test; any code change
    that takes the gap below +0.50 breaks W8-1."""

    def test_corroboration_strictly_beats_fifo_at_K4(self) -> None:
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        gap = rep["headline_gap"][
            "corroboration_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(
            gap, 0.50,
            msg=f"W8-1 contract: corroboration − fifo accuracy_full "
                 f"={gap} < 0.50 falsifier")

    def test_corroboration_strictly_beats_W7_2_at_K4(self) -> None:
        """The strict separation from W7-2 — the load-bearing W8-1
        claim. If W7-2 ties or beats corroboration on this bench,
        the milestone is null."""
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        gap = rep["headline_gap"][
            "corroboration_minus_cohort_buffered_accuracy_full"]
        self.assertGreaterEqual(
            gap, 0.50,
            msg=f"W8-1 strict-separation contract: corroboration − "
                 f"cohort_buffered accuracy_full ={gap} < 0.50 falsifier")

    def test_corroboration_perfect_services_at_K4(self) -> None:
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        self.assertEqual(
            rep["pooled"]["capsule_corroboration"]["accuracy_services"],
            1.0,
            msg="corroboration must achieve perfect accuracy_services "
                 "on Phase 55 default — the gold service is the only "
                 "one in the cohort.")

    def test_W7_2_loses_on_phase55(self) -> None:
        """W7-2 buffered cohort ties FIFO on Phase 55 default
        (the W7-2 falsifier instantiation)."""
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        gap = rep["headline_gap"][
            "cohort_buffered_minus_fifo_accuracy_full"]
        self.assertLessEqual(gap, 0.10)

    def test_audit_ok_grid_holds_for_all_capsule_strategies(self) -> None:
        """W6-1 generalisation to Phase 55: T-1..T-7 hold on every
        capsule strategy cell of the new bench, including the new
        capsule_corroboration cell."""
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        grid = rep["audit_ok_grid"]
        for s in ("capsule_fifo", "capsule_priority", "capsule_coverage",
                   "capsule_cohort_buffered", "capsule_corroboration"):
            self.assertTrue(grid[s], msg=f"{s} audit_ok = False")


class Phase55SeedStabilityTests(unittest.TestCase):
    """W8-1 cross-seed stability: the +0.50 corroboration−fifo gap
    must hold across ≥ 3 distinct bank_seed values. This is the
    cross-bank-stability requirement of the success bar."""

    def test_gap_holds_across_five_seeds(self) -> None:
        sweep = run_seed_stability_sweep(
            seeds=(11, 17, 23, 29, 31), n_eval=10,
            K_auditor=4, T_auditor=128)
        for seed, data in sweep["per_seed"].items():
            gap = data["headline_gap"][
                "corroboration_minus_fifo_accuracy_full"]
            self.assertGreaterEqual(
                gap, 0.50,
                msg=f"seed={seed}: corroboration−fifo gap "
                     f"={gap} < 0.50")

    def test_gap_holds_across_three_minimum(self) -> None:
        """Per success-criterion bar 1.1: ≥ 3 distinct bank seeds
        with stable gap. Cheaper variant for fast CI."""
        for seed in (11, 17, 23):
            rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                              bank_seed=seed, bank_replicates=2,
                              verbose=False)
            gap = rep["headline_gap"][
                "corroboration_minus_fifo_accuracy_full"]
            self.assertGreaterEqual(
                gap, 0.50,
                msg=f"seed={seed}: gap={gap} < 0.50")


# ---------------------------------------------------------------------------
# Falsifier regime — W8-1 conditional (named falsifier)
# ---------------------------------------------------------------------------


class Phase55FalsifierTests(unittest.TestCase):
    """The W8-1 win is *conditional* on cross-role-corroborated gold.
    This test class confirms that on the named falsifier regime
    (where the decoy is corroborated instead), corroboration ties
    FIFO — exactly as predicted."""

    def test_corroboration_ties_fifo_on_falsifier(self) -> None:
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          use_falsifier_bank=True, verbose=False)
        gap = rep["headline_gap"][
            "corroboration_minus_fifo_accuracy_full"]
        self.assertLessEqual(
            gap, 0.10,
            msg=f"falsifier: corroboration must tie or barely beat "
                 f"FIFO; saw gap={gap}")

    def test_falsifier_bank_loses_for_all_strategies(self) -> None:
        """On the falsifier regime (gold has no cross-role
        corroboration; decoy does), no admission policy can recover
        — services pollution is unavoidable."""
        rep = run_phase55(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          use_falsifier_bank=True, verbose=False)
        for s in ("capsule_fifo", "capsule_priority",
                   "capsule_coverage", "capsule_cohort_buffered",
                   "capsule_corroboration"):
            self.assertEqual(
                rep["pooled"][s]["accuracy_full"], 0.0,
                msg=f"{s}: expected 0.0 on falsifier; saw "
                     f"{rep['pooled'][s]['accuracy_full']}")


# ---------------------------------------------------------------------------
# Budget sweep — W8-1-conditional structure-win window
# ---------------------------------------------------------------------------


class Phase55BudgetSweepTests(unittest.TestCase):
    """W8-1-conditional: identify the K-window where corroboration
    strictly outperforms FIFO."""

    def test_budget_sweep_runs_end_to_end(self) -> None:
        rep = run_phase55_budget_sweep(
            K_values=(2, 4, 6), n_eval=10, bank_seed=11,
            bank_replicates=2)
        self.assertEqual(rep["schema"],
                          "phase55.decoy_plurality_budget_sweep.v1")
        self.assertEqual(set(rep["pooled_per_K"]), {2, 4, 6})

    def test_corroboration_wins_at_K4_in_sweep(self) -> None:
        rep = run_phase55_budget_sweep(
            K_values=(4,), n_eval=10, bank_seed=11, bank_replicates=2)
        gap = rep["headline_gap_per_K"][4][
            "corroboration_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50)

    def test_W7_2_loses_across_full_K_range(self) -> None:
        """Sharper observation: even with K = 8 (no budget pressure),
        W7-2 ties FIFO on Phase 55 because it's structurally blind to
        cross-role corroboration. Cohort_buffered's accuracy_full ≤
        0.10 across the full K range."""
        rep = run_phase55_budget_sweep(
            K_values=(3, 4, 6, 8), n_eval=10, bank_seed=11,
            bank_replicates=2)
        for K in (3, 4, 6, 8):
            gap = rep["headline_gap_per_K"][K][
                "cohort_buffered_minus_fifo_accuracy_full"]
            self.assertLessEqual(
                gap, 0.10,
                msg=f"W7-2 should not beat FIFO on Phase 55; "
                     f"K={K}, gap={gap}")


# ---------------------------------------------------------------------------
# Cross-regime — backward compat (W8-3) + cross-regime audit (PART D)
# ---------------------------------------------------------------------------


class CrossRegimeBackwardCompatibilityTests(unittest.TestCase):
    """W8-3: on the Phase-54 default (gold-plurality), corroboration
    matches W7-2 cohort_buffered — i.e. the new policy is a strict
    generalisation, not a replacement that loses prior wins."""

    def test_corroboration_matches_W7_2_on_phase54(self) -> None:
        from vision_mvp.experiments.phase54_cross_role_coherence import (
            build_phase54_bank, build_candidate_stream as p54_cands,
            _run_capsule_strategy as p54_run,
        )
        from vision_mvp.experiments.phase52_team_coord import (
            make_team_budgets, pool,
        )
        from vision_mvp.coordpy.team_coord import (
            CohortCoherenceAdmissionPolicy as Cohort,
            CrossRoleCorroborationAdmissionPolicy as Corr,
        )
        bank = build_phase54_bank(n_replicates=2, seed=11)
        budgets = make_team_budgets(K_auditor=4, T_auditor=128)
        rs_corr, rs_cohort = [], []
        for sc in bank:
            cands = p54_cands(sc)
            cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            corr = Corr.from_candidate_stream(
                [(c[0], c[3]) for c in cands_aud])
            cohort = Cohort.from_candidate_payloads(
                [c[3] for c in cands_aud])
            rs_corr.append(p54_run(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role={r: corr for r in budgets},
                strategy_name="corr"))
            rs_cohort.append(p54_run(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role={r: cohort for r in budgets},
                strategy_name="cohort"))
        a_corr = pool(rs_corr, "corr").accuracy_full
        a_cohort = pool(rs_cohort, "cohort").accuracy_full
        # W8-3 backward-compat: corroboration must match W7-2 within
        # 0.05 on the Phase 54 default config.
        self.assertAlmostEqual(
            a_corr, a_cohort, delta=0.05,
            msg=f"W8-3: corroboration ({a_corr}) must match W7-2 "
                 f"({a_cohort}) on Phase 54 default")
        # Both must clear the W7-2 +0.50 vs FIFO bar.
        self.assertGreaterEqual(a_corr, 0.50)


class CrossRegimeNoRegressionTests(unittest.TestCase):
    """PART D no-regression: corroboration must not regress on Phase
    53 synthetic (the W7-1 low-surplus regime). All strategies tie
    here; corroboration ties too."""

    def test_no_regression_on_phase53_synthetic_K4(self) -> None:
        from vision_mvp.experiments.phase53_scale_vs_structure import (
            expand_bank, split_bank, _build_synthetic_candidates,
            _run_capsule_strategy_with_candidates,
            _run_substrate_with_candidates,
        )
        from vision_mvp.experiments.phase52_team_coord import (
            make_team_budgets, pool,
        )
        from vision_mvp.coordpy.team_coord import (
            CrossRoleCorroborationAdmissionPolicy as Corr,
        )
        bank = expand_bank(seeds=(31, 32, 33), distractors_per_role=(8,))
        train, evald = split_bank(bank, train_fraction=0.66, seed=0)
        evald = list(evald)[:5]
        budgets = make_team_budgets(K_auditor=4, T_auditor=128)
        rs_sub, rs_corr = [], []
        for sc in evald:
            cands = _build_synthetic_candidates(sc)
            cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            corr = Corr.from_candidate_stream(
                [(c[0], c[3]) for c in cands_aud])
            rs_sub.append(_run_substrate_with_candidates(sc, cands, 4))
            rs_corr.append(_run_capsule_strategy_with_candidates(
                scenario=sc, candidates=cands, budgets=budgets,
                policy_per_role={r: corr for r in budgets},
                strategy_name="corr"))
        a_sub = pool(rs_sub, "substrate").accuracy_full
        a_corr = pool(rs_corr, "corr").accuracy_full
        # Per success-criterion bar 1.1 condition 4: ≤ 0.05 regression.
        self.assertGreaterEqual(
            a_corr, a_sub - 0.05,
            msg=f"corroboration ({a_corr}) regresses against "
                 f"substrate ({a_sub}) on Phase 53 synthetic by > 0.05")


class CrossRegimeSummaryTests(unittest.TestCase):
    """PART D: a single end-to-end report that bundles
    Phase 53 (synthetic only) + Phase 54 + Phase 55 + falsifier
    so a reviewer can audit the cross-regime picture in one place."""

    def test_cross_regime_summary_runs(self) -> None:
        rep = run_cross_regime_summary(
            n_eval=10, bank_seed=11, bank_replicates=2)
        self.assertEqual(rep["schema"], "phase55.cross_regime.v1")
        # Phase 55 default: corroboration > cohort_buffered.
        p55 = rep["phase55_default"]
        self.assertGreaterEqual(
            p55["headline_gap"][
                "corroboration_minus_cohort_buffered_accuracy_full"],
            0.50)
        # Phase 54 default: cohort_buffered wins (preserved W7-2).
        p54 = rep["phase54_default"]
        self.assertGreaterEqual(
            p54["headline_gap"][
                "cohort_buffered_minus_fifo_accuracy_full"], 0.50)
        # Phase 55 falsifier: corroboration ties FIFO.
        p55f = rep["phase55_falsifier"]
        self.assertLessEqual(
            p55f["headline_gap"][
                "corroboration_minus_fifo_accuracy_full"], 0.10)


# ---------------------------------------------------------------------------
# End-to-end TeamCoordinator smoke — single-scenario sanity
# ---------------------------------------------------------------------------


class CorroborationCoordinatorSmokeTests(unittest.TestCase):
    """End-to-end smoke: drive one Phase 55 scenario through the
    real TeamCoordinator and verify (a) the team-lifecycle audit
    holds T-1..T-7 and (b) the auditor's role view contains only
    gold-tagged handoffs."""

    def test_one_scenario_corroboration_admits_only_gold(self) -> None:
        sc = _BASE_BUILDERS[0](decoy_storm="archival",
                                 decoy_secondary="logs")
        cands = build_candidate_stream(sc)
        cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
        corr = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
            [(c[0], c[3]) for c in cands_aud])
        ledger = CapsuleLedger()
        budgets = {
            ROLE_MONITOR:  RoleBudget(role=ROLE_MONITOR,  K_role=6, T_role=96),
            ROLE_DB_ADMIN: RoleBudget(role=ROLE_DB_ADMIN, K_role=6, T_role=96),
            ROLE_SYSADMIN: RoleBudget(role=ROLE_SYSADMIN, K_role=6, T_role=96),
            ROLE_NETWORK:  RoleBudget(role=ROLE_NETWORK,  K_role=6, T_role=96),
            ROLE_AUDITOR:  RoleBudget(role=ROLE_AUDITOR,  K_role=4, T_role=128),
        }
        coord = TeamCoordinator(
            ledger=ledger, role_budgets=budgets,
            policy_per_role={r: corr for r in budgets},
            team_tag="phase55_smoke")
        coord.advance_round(1)
        for (src, to, kind, payload, _evs) in cands:
            coord.emit_handoff(
                source_role=src, to_role=to, claim_kind=kind, payload=payload)
        coord.seal_all_role_views()
        rv = ledger.get(coord.role_view_cid(ROLE_AUDITOR))
        # Every admitted handoff must carry the gold service tag.
        gold_tag_pat = re.compile(rf"service={sc.real_service}\b")
        for p_cid in rv.parents:
            cap = ledger.get(p_cid)
            payload = cap.payload.get("payload", "")
            self.assertRegex(
                payload, gold_tag_pat,
                msg=f"corroboration admitted non-gold handoff: {payload!r}")
        coord.seal_team_decision(team_role=ROLE_AUDITOR,
                                   decision={"root_cause": "memory_leak"})
        report = audit_team_lifecycle(ledger)
        self.assertTrue(report.is_ok(),
                          msg=f"audit failed: {report.violations}")


if __name__ == "__main__":
    unittest.main()
