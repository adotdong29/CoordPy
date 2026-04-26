"""Tests for SDK v3.8 — cross-role cohort-coherence multi-agent
benchmark (``vision_mvp.experiments.phase54_cross_role_coherence`` +
``vision_mvp.wevra.team_coord.CohortCoherenceAdmissionPolicy``).

These tests anchor the W7 family theorems empirically. They are
designed to be runnable in CI without network IO; the bench is a
deterministic synthetic candidate stream.
"""

from __future__ import annotations

import unittest

from vision_mvp.tasks.incident_triage import ROLE_AUDITOR
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    CohortCoherenceAdmissionPolicy, FifoAdmissionPolicy,
    REASON_ADMIT, REASON_SCORE_LOW, REASON_BUDGET_FULL,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_team_handoff,
)
from vision_mvp.experiments.phase54_cross_role_coherence import (
    build_phase54_bank, build_candidate_stream,
    run_phase54, run_phase54_budget_sweep,
    _build_scenario_oom_api,
    _build_scenario_disk_with_backup_decoy,
    _build_scenario_tls_with_cache_decoy,
    _build_scenario_dns_with_users_decoy,
    _build_scenario_deadlock_with_logs_decoy,
)


# ---------------------------------------------------------------------------
# CohortCoherenceAdmissionPolicy — unit tests
# ---------------------------------------------------------------------------


class CohortPolicyUnitTests(unittest.TestCase):
    """Direct ``decide(...)`` tests on the cohort-coherence policy."""

    def _budget(self) -> RoleBudget:
        return RoleBudget(role="auditor", K_role=8, T_role=128)

    def _cap(self, *, kind: str, payload: str) -> object:
        return capsule_team_handoff(
            source_role="db_admin", to_role="auditor",
            claim_kind=kind, payload=payload, round=1)

    def test_streaming_admits_first_no_cohort(self) -> None:
        """Streaming mode admits the first candidate (no cohort)."""
        p = CohortCoherenceAdmissionPolicy()
        cand = self._cap(kind="POOL_EXHAUSTION",
                          payload="pool active=200/200 service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)

    def test_streaming_locks_to_first_admitted_tag(self) -> None:
        """Streaming mode rejects subsequent foreign-tag candidates
        (W7-1-aux: streaming is arrival-order-sensitive)."""
        p = CohortCoherenceAdmissionPolicy()
        first = self._cap(kind="POOL_EXHAUSTION",
                            payload="pool active=200/200 service=archival")
        second = self._cap(kind="POOL_EXHAUSTION",
                             payload="pool active=200/200 service=api")
        d = p.decide(candidate=second, role="auditor",
                      budget=self._budget(), current_admitted=(first,),
                      current_n_tokens=1)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_SCORE_LOW)

    def test_buffered_factory_picks_strict_plurality(self) -> None:
        """``from_candidate_payloads`` selects the strictly-most-
        common service tag across the candidate stream."""
        payloads = [
            "p1 service=api",
            "p2 service=api",
            "p3 service=api",
            "p4 service=archival",
            "p5 service=backup",
        ]
        p = CohortCoherenceAdmissionPolicy.from_candidate_payloads(payloads)
        self.assertEqual(p.fixed_plurality_tag, "api")

    def test_buffered_factory_lex_breaks_ties(self) -> None:
        """Ties on plurality count break by lex order (deterministic)."""
        payloads = [
            "p1 service=zebra",
            "p2 service=zebra",
            "p3 service=alpha",
            "p4 service=alpha",
        ]
        p = CohortCoherenceAdmissionPolicy.from_candidate_payloads(payloads)
        self.assertEqual(p.fixed_plurality_tag, "alpha")

    def test_buffered_factory_no_tags_returns_streaming_default(self) -> None:
        """If no candidate carries a service tag, the buffered factory
        returns a streaming policy (``fixed_plurality_tag is None``)."""
        payloads = ["p1 no tag", "p2 q#1"]
        p = CohortCoherenceAdmissionPolicy.from_candidate_payloads(payloads)
        self.assertIsNone(p.fixed_plurality_tag)

    def test_buffered_admits_plurality_tag(self) -> None:
        """Buffered mode admits candidates whose tag matches the
        pre-fitted plurality (independent of admission order)."""
        p = CohortCoherenceAdmissionPolicy(fixed_plurality_tag="api")
        cand = self._cap(kind="OOM_KILL",
                          payload="oom comm=app.py service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)

    def test_buffered_rejects_foreign_tag(self) -> None:
        """Buffered mode rejects foreign-tag candidates regardless of
        whether the cohort is empty."""
        p = CohortCoherenceAdmissionPolicy(fixed_plurality_tag="api")
        cand = self._cap(kind="OOM_KILL",
                          payload="oom comm=cron service=archival")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_SCORE_LOW)

    def test_buffered_admits_no_tag_candidates(self) -> None:
        """A candidate without a ``service=<tag>`` token can never
        violate cohort coherence — admit unconditionally."""
        p = CohortCoherenceAdmissionPolicy(fixed_plurality_tag="api")
        cand = self._cap(kind="DEADLOCK_SUSPECTED",
                          payload="deadlock relation=orders_payments")
        d = p.decide(candidate=cand, role="auditor",
                      budget=self._budget(), current_admitted=(),
                      current_n_tokens=0)
        self.assertTrue(d.admit)

    def test_budget_pre_check_overrides_cohort_rule(self) -> None:
        """Budget pre-checks (K_role full) deny before cohort rule runs."""
        p = CohortCoherenceAdmissionPolicy(fixed_plurality_tag="api")
        budget = RoleBudget(role="auditor", K_role=1, T_role=128)
        already = self._cap(kind="POOL_EXHAUSTION",
                              payload="pool service=api")
        cand = self._cap(kind="OOM_KILL",
                          payload="oom service=api")
        d = p.decide(candidate=cand, role="auditor",
                      budget=budget, current_admitted=(already,),
                      current_n_tokens=1)
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_BUDGET_FULL)


# ---------------------------------------------------------------------------
# Phase 54 bench — bank construction + scenario shape
# ---------------------------------------------------------------------------


class Phase54BankShapeTests(unittest.TestCase):

    def test_bank_size_matches_replicates(self) -> None:
        bank = build_phase54_bank(n_replicates=2, seed=11)
        self.assertEqual(len(bank), 10)
        bank3 = build_phase54_bank(n_replicates=3, seed=11)
        self.assertEqual(len(bank3), 15)

    def test_every_scenario_has_gold_plurality(self) -> None:
        """W7-2 anchor: every Phase-54 scenario has strict gold
        plurality in the auditor's candidate stream."""
        bank = build_phase54_bank(n_replicates=2, seed=11)
        for sc in bank:
            cands = build_candidate_stream(sc)
            cands_to_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
            counts: dict[str, int] = {}
            for (_src, _to, _kind, payload, _evs) in cands_to_aud:
                import re
                m = re.search(r"service=(\w+)", payload)
                if m:
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            real = sc.real_service
            real_count = counts.get(real, 0)
            other_max = max((c for k, c in counts.items() if k != real),
                             default=0)
            self.assertGreater(
                real_count, other_max,
                msg=(f"{sc.scenario_id}: real={real} count={real_count}; "
                      f"foreign max={other_max}"))

    def test_decoy_services_disjoint_from_real(self) -> None:
        bank = build_phase54_bank(n_replicates=2, seed=11)
        for sc in bank:
            self.assertNotIn(sc.real_service, sc.decoy_services)

    def test_scenarios_carry_role_emissions_for_at_least_two_roles(self) -> None:
        """Cross-role coherence requires evidence from ≥ 2 producer
        roles in every scenario."""
        bank = build_phase54_bank(n_replicates=2, seed=11)
        for sc in bank:
            n_roles_with_emissions = sum(
                1 for r, ems in sc.role_emissions.items() if ems)
            self.assertGreaterEqual(
                n_roles_with_emissions, 2,
                msg=f"{sc.scenario_id}: {n_roles_with_emissions} roles emit")


# ---------------------------------------------------------------------------
# Phase 54 driver — pre-committed default config result
# ---------------------------------------------------------------------------


class Phase54DefaultConfigTests(unittest.TestCase):
    """Anchors the W7-2 empirical result on the pre-committed config
    (K_auditor=4, n_eval=10, bank_seed=11). Run end-to-end and assert
    the headline gap is at least the pre-committed +0.50 bar.

    This is the canonical Phase-54 contract test: any code change
    that takes ``capsule_cohort_buffered − capsule_fifo`` below
    +0.50 on accuracy_full breaks W7-2.
    """

    def test_buffered_cohort_strictly_beats_fifo_at_K4(self) -> None:
        rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        gap = rep["headline_gap"][
            "cohort_buffered_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(
            gap, 0.50,
            msg=f"W7-2 contract: buffered cohort − fifo accuracy_full "
                 f"={gap} < 0.50 falsifier")

    def test_buffered_cohort_perfect_services_at_K4(self) -> None:
        rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        # Cohort-coherence on this bench should produce exactly the
        # gold service tag (no foreign service pollution).
        self.assertEqual(
            rep["pooled"]["capsule_cohort_buffered"]["accuracy_services"],
            1.0)

    def test_fifo_loses_on_services_at_K4(self) -> None:
        rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        self.assertEqual(
            rep["pooled"]["capsule_fifo"]["accuracy_services"], 0.0,
            msg="FIFO cannot filter foreign-service decoys; expected 0.0")

    def test_streaming_cohort_does_not_beat_fifo_at_K4(self) -> None:
        """W7-1-aux: streaming cohort is arrival-order-unstable; on
        this bench (decoys arrive first per role), streaming locks
        onto the decoy plurality and ties FIFO."""
        rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        gap_stream = rep["headline_gap"][
            "cohort_streaming_minus_fifo_accuracy_full"]
        self.assertLessEqual(gap_stream, 0.10)

    def test_audit_ok_grid_holds_for_all_capsule_strategies(self) -> None:
        """W6-1 generalisation to Phase 54: T-1..T-7 hold on every
        capsule strategy cell of the bench, including the new
        cohort_streaming and cohort_buffered cells."""
        rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                          bank_seed=11, bank_replicates=2,
                          verbose=False)
        grid = rep["audit_ok_grid"]
        # Substrate is not in the capsule ledger; the grid records False.
        for s in ("capsule_fifo", "capsule_priority", "capsule_coverage",
                   "capsule_cohort_streaming", "capsule_cohort_buffered"):
            self.assertTrue(grid[s], msg=f"{s} audit_ok = False")

    def test_result_stable_across_bank_seeds(self) -> None:
        """The +0.50 gap holds for at least 3/3 distinct bank seeds.

        Stability across bank seeds is the W7-2 strict-direction
        signal: this is *not* a single-seed lucky win.
        """
        for seed in (11, 17, 23):
            rep = run_phase54(K_auditor=4, T_auditor=128, n_eval=10,
                              bank_seed=seed, bank_replicates=2,
                              verbose=False)
            gap = rep["headline_gap"][
                "cohort_buffered_minus_fifo_accuracy_full"]
            self.assertGreaterEqual(
                gap, 0.50,
                msg=f"seed={seed}: gap={gap} < 0.50")


# ---------------------------------------------------------------------------
# Budget sweep — W7-2 falsifier search
# ---------------------------------------------------------------------------


class Phase54BudgetSweepTests(unittest.TestCase):
    """Anchors the W7-2-conditional empirical result: cohort
    coherence wins iff K_auditor is in the structure-win window."""

    def test_budget_sweep_runs_end_to_end(self) -> None:
        rep = run_phase54_budget_sweep(
            K_values=(2, 4, 6), n_eval=10, bank_seed=11,
            bank_replicates=2)
        self.assertEqual(rep["schema"],
                          "phase54.cross_role_coherence_budget_sweep.v1")
        self.assertEqual(set(rep["pooled_per_K"]), {2, 4, 6})

    def test_buffered_cohort_wins_at_K4_in_sweep(self) -> None:
        rep = run_phase54_budget_sweep(
            K_values=(4,), n_eval=10, bank_seed=11, bank_replicates=2)
        gap = rep["headline_gap_per_K"][4][
            "cohort_buffered_minus_fifo_accuracy_full"]
        self.assertGreaterEqual(gap, 0.50)


if __name__ == "__main__":
    unittest.main()
