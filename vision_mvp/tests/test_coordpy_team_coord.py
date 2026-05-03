"""Tests for SDK v3.5 — capsule-native multi-agent team coordination
(``vision_mvp.coordpy.team_coord`` + ``vision_mvp.coordpy.team_policy``)."""

from __future__ import annotations

import unittest

from vision_mvp.coordpy.capsule import (
    CapsuleAdmissionError, CapsuleKind, CapsuleLedger, CapsuleLifecycle,
)
from vision_mvp.coordpy.team_coord import (
    DEFAULT_ROLE_BUDGETS, ClaimPriorityAdmissionPolicy,
    CoverageGuidedAdmissionPolicy, FifoAdmissionPolicy,
    REASON_BUDGET_FULL, REASON_DUPLICATE, REASON_TOKENS_FULL,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_role_view, capsule_team_decision, capsule_team_handoff,
)
from vision_mvp.coordpy.team_policy import (
    LearnedTeamAdmissionPolicy, TrainSample, featurise_team_handoff,
    train_team_admission_policy,
)


# ---------------------------------------------------------------------------
# Capsule constructors
# ---------------------------------------------------------------------------


class TeamHandoffCapsuleTests(unittest.TestCase):

    def test_handoff_capsule_round_trips(self) -> None:
        cap = capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE",
            payload="error_rate=0.5 service=web", round=1)
        self.assertEqual(cap.kind, CapsuleKind.TEAM_HANDOFF)
        self.assertEqual(cap.lifecycle, CapsuleLifecycle.PROPOSED)
        self.assertEqual(cap.payload["source_role"], "monitor")
        self.assertEqual(cap.payload["claim_kind"], "ERROR_RATE_SPIKE")
        self.assertIn("payload_sha256", cap.payload)
        self.assertEqual(len(cap.payload["payload_sha256"]), 64)

    def test_two_byte_identical_handoffs_share_cid(self) -> None:
        a = capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="x", round=1)
        b = capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="x", round=1)
        self.assertEqual(a.cid, b.cid)

    def test_role_view_capsule_enforces_K_role_at_construction(self) -> None:
        # K_role=2 but we pass 3 admitted CIDs.
        budget = RoleBudget(role="auditor", K_role=2, T_role=64)
        with self.assertRaises(ValueError):
            capsule_role_view(
                role="auditor", round=1,
                admitted_handoff_cids=("a"*64, "b"*64, "c"*64),
                admitted_claim_kinds=("k1", "k2", "k3"),
                n_tokens_admitted=10,
                budget=budget,
            )

    def test_role_view_capsule_enforces_T_role_at_construction(self) -> None:
        budget = RoleBudget(role="auditor", K_role=8, T_role=4)
        with self.assertRaises(ValueError):
            capsule_role_view(
                role="auditor", round=1,
                admitted_handoff_cids=(),
                admitted_claim_kinds=(),
                n_tokens_admitted=99,
                budget=budget,
            )

    def test_team_decision_capsule_round_trips(self) -> None:
        cap = capsule_team_decision(
            team_tag="t", round=2,
            decision={"root_cause": "disk_fill"},
            evidence_summary="cron + slow_query",
            n_role_views=1, gate_passed=True,
        )
        self.assertEqual(cap.kind, CapsuleKind.TEAM_DECISION)
        self.assertEqual(cap.payload["team_tag"], "t")
        self.assertTrue(cap.payload["gate_passed"])
        self.assertEqual(cap.payload["decision"]["root_cause"], "disk_fill")


# ---------------------------------------------------------------------------
# TeamCoordinator
# ---------------------------------------------------------------------------


def _budgets(K=8, T=128) -> dict[str, RoleBudget]:
    return {
        "monitor":  RoleBudget("monitor",  K, T),
        "db_admin": RoleBudget("db_admin", K, T),
        "sysadmin": RoleBudget("sysadmin", K, T),
        "network":  RoleBudget("network",  K, T),
        "auditor":  RoleBudget("auditor",  K, T),
    }


class TeamCoordinatorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.ledger = CapsuleLedger()
        self.coord = TeamCoordinator(
            ledger=self.ledger,
            role_budgets=_budgets(K=8, T=128),
            team_tag="incident_smoke",
        )
        self.coord.advance_round(1)

    def test_emit_handoff_admits_when_budget_permits(self) -> None:
        cap, decision = self.coord.emit_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE",
            payload="error_rate=0.3 service=web")
        self.assertTrue(decision.admit)
        self.assertEqual(cap.kind, CapsuleKind.TEAM_HANDOFF)
        self.assertIn(cap.cid, self.ledger)

    def test_emit_handoff_rejects_unknown_to_role(self) -> None:
        with self.assertRaises(KeyError):
            self.coord.emit_handoff(
                source_role="monitor", to_role="not-a-role",
                claim_kind="X", payload="p")

    def test_seal_role_view_creates_role_view_capsule(self) -> None:
        for i in range(3):
            self.coord.emit_handoff(
                source_role="monitor", to_role="auditor",
                claim_kind="ERROR_RATE_SPIKE",
                payload=f"error_rate=0.{30+i}")
        rv = self.coord.seal_role_view("auditor")
        self.assertEqual(rv.kind, CapsuleKind.ROLE_VIEW)
        self.assertEqual(rv.lifecycle, CapsuleLifecycle.SEALED)
        self.assertEqual(rv.payload["n_admitted"], 3)
        self.assertEqual(rv.payload["role"], "auditor")

    def test_capsule_idempotency_under_byte_identical_emissions(self) -> None:
        c1, _ = self.coord.emit_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="x")
        c2, _ = self.coord.emit_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="x")
        self.assertEqual(c1.cid, c2.cid)
        self.assertEqual(self.coord.stats()["n_team_handoff"], 1)

    def test_K_role_budget_rejects_overflow(self) -> None:
        budgets = _budgets(K=2, T=128)
        coord = TeamCoordinator(
            ledger=CapsuleLedger(), role_budgets=budgets,
            team_tag="K_test")
        coord.advance_round(1)
        for i in range(5):
            coord.emit_handoff(
                source_role="monitor", to_role="auditor",
                claim_kind="ERROR_RATE_SPIKE",
                payload=f"e{i}")
        rv = coord.seal_role_view("auditor")
        # K=2 → only 2 admitted, rest dropped budget_full.
        self.assertEqual(rv.payload["n_admitted"], 2)
        self.assertGreaterEqual(rv.payload["n_dropped_budget"], 3)

    def test_T_role_budget_rejects_overflow(self) -> None:
        budgets = {
            "auditor": RoleBudget("auditor", K_role=8, T_role=8),
        }
        # only auditor; producers not exercised
        coord = TeamCoordinator(
            ledger=CapsuleLedger(), role_budgets=budgets,
            team_tag="T_test")
        coord.advance_round(1)
        # Each handoff has 5 tokens.
        for i in range(4):
            coord.emit_handoff(
                source_role="monitor", to_role="auditor",
                claim_kind="LATENCY_SPIKE",
                payload=f"a b c d e f{i}",  # 7 tokens
            )
        rv = coord.seal_role_view("auditor")
        # T=8, each handoff ~7 tokens — only one fits.
        self.assertEqual(rv.payload["n_admitted"], 1)
        self.assertGreaterEqual(rv.payload["n_dropped_capacity"], 0)

    def test_seal_team_decision_links_to_role_view(self) -> None:
        self.coord.emit_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="x")
        self.coord.seal_role_view("auditor")
        td = self.coord.seal_team_decision(
            team_role="auditor",
            decision={"root_cause": "x"})
        self.assertEqual(td.kind, CapsuleKind.TEAM_DECISION)
        self.assertEqual(len(td.parents), 1)
        rv_cid = self.coord.role_view_cid("auditor")
        self.assertEqual(td.parents[0], rv_cid)


# ---------------------------------------------------------------------------
# Lifecycle audit (T-1..T-7)
# ---------------------------------------------------------------------------


class TeamLifecycleAuditTests(unittest.TestCase):

    def _make_round(self) -> tuple[CapsuleLedger, TeamCoordinator]:
        ledger = CapsuleLedger()
        coord = TeamCoordinator(
            ledger=ledger, role_budgets=_budgets(K=8, T=128),
            team_tag="audit_t")
        coord.advance_round(1)
        coord.emit_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE", payload="e1")
        coord.emit_handoff(
            source_role="db_admin", to_role="auditor",
            claim_kind="SLOW_QUERY_OBSERVED", payload="q1")
        coord.seal_all_role_views()
        coord.seal_team_decision(
            team_role="auditor", decision={"x": 1})
        return ledger, coord

    def test_w4_1_audit_ok_on_well_formed_run(self) -> None:
        ledger, _coord = self._make_round()
        report = audit_team_lifecycle(ledger)
        self.assertEqual(report.verdict, "OK", msg=report.violations)
        self.assertEqual(report.counts["n_team_decision"], 1)
        self.assertGreaterEqual(report.counts["n_team_handoff"], 2)
        self.assertEqual(report.counts["n_role_view"], 5)

    def test_w4_1_audit_empty_ledger_returns_empty(self) -> None:
        report = audit_team_lifecycle(CapsuleLedger())
        self.assertEqual(report.verdict, "EMPTY")

    def test_t7_violated_when_role_view_admits_wrong_to_role(self) -> None:
        # Construct a pathological situation: a ROLE_VIEW for
        # "auditor" whose parent handoff has to_role="db_admin".
        ledger = CapsuleLedger()
        bad = capsule_team_handoff(
            source_role="monitor", to_role="db_admin",
            claim_kind="ERROR_RATE_SPIKE", payload="z", round=1)
        bad = ledger.admit_and_seal(bad)
        rv = capsule_role_view(
            role="auditor", round=1,
            admitted_handoff_cids=(bad.cid,),
            admitted_claim_kinds=("ERROR_RATE_SPIKE",),
            n_tokens_admitted=1,
            budget=RoleBudget(role="auditor", K_role=4, T_role=64))
        ledger.admit_and_seal(rv)
        report = audit_team_lifecycle(ledger)
        self.assertEqual(report.verdict, "BAD")
        invariants = {v["invariant"] for v in report.violations}
        self.assertIn("T-7", invariants)


# ---------------------------------------------------------------------------
# Admission policies
# ---------------------------------------------------------------------------


class AdmissionPolicyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.budget = RoleBudget(role="auditor", K_role=2, T_role=64)
        self.coord = TeamCoordinator(
            ledger=CapsuleLedger(),
            role_budgets={"auditor": self.budget},
            team_tag="ap")
        self.coord.advance_round(1)

    def _h(self, *, kind: str = "ERROR_RATE_SPIKE",
            payload: str = "p") -> any:
        return capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind=kind, payload=payload, round=1)

    def test_fifo_admits_until_budget_full(self) -> None:
        pol = FifoAdmissionPolicy()
        admitted: list = []
        for i in range(5):
            cap = self._h(payload=f"e{i}")
            d = pol.decide(
                candidate=cap, role="auditor", budget=self.budget,
                current_admitted=admitted, current_n_tokens=0)
            if d.admit:
                admitted.append(cap)
        self.assertEqual(len(admitted), 2)

    def test_priority_rejects_below_threshold(self) -> None:
        pol = ClaimPriorityAdmissionPolicy(
            priorities={"HIGH": 1.0, "LOW": 0.1},
            threshold=0.5)
        cap_low = self._h(kind="LOW")
        d_low = pol.decide(
            candidate=cap_low, role="auditor", budget=self.budget,
            current_admitted=[], current_n_tokens=0)
        self.assertFalse(d_low.admit)
        cap_high = self._h(kind="HIGH")
        d_high = pol.decide(
            candidate=cap_high, role="auditor", budget=self.budget,
            current_admitted=[], current_n_tokens=0)
        self.assertTrue(d_high.admit)

    def test_coverage_guided_dedupes_by_kind(self) -> None:
        pol = CoverageGuidedAdmissionPolicy()
        cap1 = self._h(kind="A", payload="x")
        d1 = pol.decide(
            candidate=cap1, role="auditor", budget=self.budget,
            current_admitted=[], current_n_tokens=0)
        self.assertTrue(d1.admit)
        cap2 = self._h(kind="A", payload="y")
        d2 = pol.decide(
            candidate=cap2, role="auditor", budget=self.budget,
            current_admitted=[cap1], current_n_tokens=0)
        self.assertFalse(d2.admit)
        self.assertEqual(d2.reason, REASON_DUPLICATE)


# ---------------------------------------------------------------------------
# Learned policy
# ---------------------------------------------------------------------------


class LearnedAdmissionPolicyTests(unittest.TestCase):

    def test_separable_pattern_train_to_perfect_accuracy(self) -> None:
        # Causal claims have "service=" tokens; non-causal don't.
        samples = []
        for i in range(40):
            cap = capsule_team_handoff(
                source_role="monitor", to_role="auditor",
                claim_kind="ERROR_RATE_SPIKE",
                payload=f"error_rate=0.{50+i} service=web")
            samples.append(TrainSample(role="auditor", capsule=cap, label=1))
        for i in range(40):
            cap = capsule_team_handoff(
                source_role="monitor", to_role="auditor",
                claim_kind="LATENCY_SPIKE",
                payload=f"noise{i}")
            samples.append(TrainSample(role="auditor", capsule=cap, label=0))
        policy, stats = train_team_admission_policy(
            samples, epochs=300, seed=7, threshold=0.5)
        self.assertGreaterEqual(stats.train_accuracy, 0.95)
        # Held-out positive sample.
        budget = RoleBudget(role="auditor", K_role=8, T_role=128)
        pos = capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE",
            payload="error_rate=0.99 service=api")
        d = policy.decide(
            candidate=pos, role="auditor", budget=budget,
            current_admitted=[], current_n_tokens=0)
        self.assertTrue(d.admit, msg=f"policy rejected positive: score={d.score}")

    def test_featurise_team_handoff_returns_six_features(self) -> None:
        cap = capsule_team_handoff(
            source_role="monitor", to_role="auditor",
            claim_kind="ERROR_RATE_SPIKE",
            payload="error_rate=0.3 service=web")
        v = featurise_team_handoff(cap).as_array()
        self.assertEqual(v.shape, (6,))


# ---------------------------------------------------------------------------
# W4-2 / W4-3 — coverage-implies-correctness + local-view limitation
# ---------------------------------------------------------------------------


class TeamLevelCorrectnessTests(unittest.TestCase):
    """Tests that anchor W4-2 (coverage → correct) and W4-3 (limitation)."""

    def test_w4_2_coverage_implies_correct(self) -> None:
        """If the auditor admits every required (role, kind) handoff,
        the deterministic decoder is correct on the gold scenario."""
        from vision_mvp.experiments.phase52_team_coord import (
            run_strategy, claim_priorities,
        )
        from vision_mvp.tasks.incident_triage import (
            build_scenario_bank,
        )
        from vision_mvp.core.extractor_noise import NoiseConfig
        sc = build_scenario_bank(seed=31, distractors_per_role=4)[0]
        # K_auditor large enough to absorb every causal handoff —
        # under noise=identity, only causal handoffs are emitted.
        budgets = {
            "monitor":  RoleBudget("monitor",  K_role=16, T_role=512),
            "db_admin": RoleBudget("db_admin", K_role=16, T_role=512),
            "sysadmin": RoleBudget("sysadmin", K_role=16, T_role=512),
            "network":  RoleBudget("network",  K_role=16, T_role=512),
            "auditor":  RoleBudget("auditor",  K_role=32, T_role=2048),
        }
        policies = {r: FifoAdmissionPolicy() for r in budgets}
        result = run_strategy(
            scenario=sc, noise=NoiseConfig(), budgets=budgets,
            policy_per_role=policies, strategy_name="capsule_fifo_uncapped")
        # With identity noise + uncapped budgets, FIFO admits every
        # causal handoff → coverage is total → decoder is correct.
        self.assertTrue(result.grading["root_cause_correct"],
                          msg=f"answer={result.answer}; "
                              f"failure={result.failure_kind}")

    def test_w4_3_local_view_limitation_at_tight_budget(self) -> None:
        """Under tight per-role budgets, even sound capsule strategies
        can fail the team gate. Expressed as: with K_auditor=1 and
        a multi-claim scenario, accuracy drops below 1.0 — i.e. the
        local-view budget is below the role's causal-share floor."""
        from vision_mvp.experiments.phase52_team_coord import (
            run_strategy,
        )
        from vision_mvp.tasks.incident_triage import build_scenario_bank
        from vision_mvp.core.extractor_noise import NoiseConfig
        bank = build_scenario_bank(seed=31, distractors_per_role=4)
        budgets = {
            "monitor":  RoleBudget("monitor",  K_role=2, T_role=64),
            "db_admin": RoleBudget("db_admin", K_role=2, T_role=64),
            "sysadmin": RoleBudget("sysadmin", K_role=2, T_role=64),
            "network":  RoleBudget("network",  K_role=2, T_role=64),
            "auditor":  RoleBudget("auditor",  K_role=1, T_role=32),
        }
        policies = {r: FifoAdmissionPolicy() for r in budgets}
        n_correct = 0
        for sc in bank:
            r = run_strategy(
                scenario=sc, noise=NoiseConfig(), budgets=budgets,
                policy_per_role=policies, strategy_name="cap")
            if r.grading["full_correct"]:
                n_correct += 1
        # With K_auditor=1 we cannot cover any scenario whose decoder
        # needs services from multiple handoffs.
        self.assertLess(n_correct / len(bank), 1.0,
                          msg="K_auditor=1 should prevent full coverage "
                              "on any multi-claim scenario.")


if __name__ == "__main__":
    unittest.main()
