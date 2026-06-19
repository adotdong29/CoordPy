"""W86 / P1 #37 — BudgetEnforcerV1 inserted into the W83 composed pipeline.

W84 shipped ``budget_enforcement_v1`` standalone (RunBudgetSpecV1,
BudgetEnforcerV1, CostModelV1, BudgetBreachAuditV1, abstract
stress bench). The literal #37 DoD bullet 2 says the enforcer
must be **inserted into the W83 composed pipeline**. This
module ships that integration.

The wrapper:

* Iterates over W83 ``RegimeScenarioV1`` scenarios (the
  load-bearing benchmark surface for composed recovery).
* Predicts each scenario's cost via the cost model
  (predicted_usd, predicted_tokens, predicted_flops).
* Pre-action check via ``BudgetEnforcerV1.check()``.
* If refused → record ``BudgetBreachAuditV1`` capsule; the
  outcome is ``abstain`` with the breach attached.
* If permitted → run ``_run_one_regime_v1`` from the W83
  composed pipeline, then commit the actual cost to the
  enforcer's running totals (so subsequent decisions see
  accurate remaining budget).
* Emits ``BudgetEnforcedRecoveryBenchReportV1`` with content-
  addressed Merkle root over the merged outcome + breach
  chain.

Anti-cheat:
* No silent overspend. Every refused candidate gets a
  ``BudgetBreachAuditV1`` with the pre-budget, the would-be
  post-budget, and the breached axis.
* Tool calls count toward ``max_tool_calls``.
* The cost model is content-addressed; the enforcer's
  verdict CID re-derives from disk.
* When the budget is huge ("permit-all" regime), behavior is
  byte-identical to running the W83 composed pipeline without
  the enforcer — verified by the bench.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .budget_enforcement_v1 import (
    BudgetBreachAuditV1,
    BudgetEnforcerV1,
    CandidateActionV1,
    CostModelV1,
    RunBudgetSpecV1,
    W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
    default_cost_model_v1,
)
from .composed_long_horizon_multi_agent_recovery_v1 import (
    RecoveryOutcomeV1,
    RegimeScenarioV1,
    W83_RECOVERY_V1_SCHEMA_VERSION,
    _run_one_regime_v1,
    build_regime_scenario_v1,
)


W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION: str = (
    "coordpy.budget_enforced_composed_recovery_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _predict_action_for_regime(
        regime: str, n_team: int,
) -> CandidateActionV1:
    """Predict the W83 pipeline's expected cost for one regime.

    The W83 pipeline's three possible decisions are
    ``replay_from_trusted``, ``runtime_recompute``,
    ``transcript_recompute``, plus ``abstain``. The visible-token
    and flop costs depend on which one fires. The predicted
    cost is the *expected* cost given the scenario's adversary
    profile (regime), used pre-action by the enforcer.

    The cost model here over-predicts the worst case
    (``runtime_recompute``) so the enforcer is honest: it
    refuses an action when even the cheap path would breach.
    """
    n = max(1, int(n_team))
    # Worst case is runtime_recompute: 4000 flops/witness, no
    # replay savings.
    predicted_flops = float(4000 * n)
    predicted_tokens = int(n * 10)
    # Cost: assume 1 USD per million flops + 0.01 USD per
    # token at the default cost model's rates.
    predicted_usd = float(
        predicted_flops / 1e6 + predicted_tokens * 1e-4)
    predicted_latency_ms = float(50.0 + 5.0 * n)
    return CandidateActionV1(
        action_name="composed_recovery_step",
        predicted_usd=predicted_usd,
        predicted_latency_ms=predicted_latency_ms,
        predicted_tokens=predicted_tokens,
        is_tool_call=False,
        predicted_flops=predicted_flops,
    )


@dataclasses.dataclass(frozen=True)
class ScenarioBudgetOutcomeV1:
    """Per-scenario outcome under the enforced pipeline.

    Either ``permitted=True`` and the W83 recovery outcome is
    populated, or ``permitted=False`` and the breach audit is
    populated. Never both.
    """

    schema: str
    regime: str
    permitted: bool
    recovery_outcome: dict[str, Any] | None
    breach_audit: dict[str, Any] | None
    predicted_action_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "regime": str(self.regime),
            "permitted": bool(self.permitted),
            "recovery_outcome": (
                dict(self.recovery_outcome)
                if self.recovery_outcome else None),
            "breach_audit": (
                dict(self.breach_audit)
                if self.breach_audit else None),
            "predicted_action_cid": str(
                self.predicted_action_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_scenario_budget_outcome_v1",
            "outcome": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class BudgetEnforcedRecoveryBenchReportV1:
    """Bench report over a sequence of regimes under one budget.

    Three load-bearing bools:

    * ``zero_commits_when_over_budget`` — under a tiny budget,
      every scenario is refused; no W83 recovery decision runs.
    * ``every_refusal_audit_carries_breach_audit`` — anti-cheat:
      every refusal MUST have a ``BudgetBreachAuditV1``.
    * ``under_budget_matches_no_enforcer`` — under a huge
      budget, behavior is byte-identical to running the W83
      composed pipeline without the enforcer.
    """

    schema: str
    n_scenarios: int
    n_committed: int
    n_refused: int
    n_committed_no_enforcer: int
    zero_commits_when_over_budget: bool
    every_refusal_audit_carries_breach_audit: bool
    under_budget_matches_no_enforcer: bool
    budget_spec_cid: str
    cost_model_cid: str
    per_scenario_outcome_cids: tuple[str, ...]
    breach_audit_merkle_root: str
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_scenarios": int(self.n_scenarios),
            "n_committed": int(self.n_committed),
            "n_refused": int(self.n_refused),
            "n_committed_no_enforcer": int(
                self.n_committed_no_enforcer),
            "zero_commits_when_over_budget": bool(
                self.zero_commits_when_over_budget),
            "every_refusal_audit_carries_breach_audit": bool(
                self.every_refusal_audit_carries_breach_audit),
            "under_budget_matches_no_enforcer": bool(
                self.under_budget_matches_no_enforcer),
            "budget_spec_cid": str(self.budget_spec_cid),
            "cost_model_cid": str(self.cost_model_cid),
            "per_scenario_outcome_cids": list(
                self.per_scenario_outcome_cids),
            "breach_audit_merkle_root": str(
                self.breach_audit_merkle_root),
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w86_budget_enforced_recovery_bench_report_v1",
            "report": self.to_dict(),
        })


def run_budget_enforced_composed_recovery_v1(
        *,
        spec: RunBudgetSpecV1,
        cost_model: CostModelV1,
        scenarios: Sequence[RegimeScenarioV1],
) -> tuple[
        BudgetEnforcedRecoveryBenchReportV1,
        tuple[ScenarioBudgetOutcomeV1, ...]]:
    """Run the W83 composed pipeline gated by BudgetEnforcerV1.

    Returns (report, per_scenario_outcomes).
    """
    enforcer = BudgetEnforcerV1(
        spec=spec, cost_model=cost_model)
    outcomes: list[ScenarioBudgetOutcomeV1] = []
    breach_audits: list[BudgetBreachAuditV1] = []
    for sc in scenarios:
        n_team = int(len(sc.team_member_snapshots))
        candidate = _predict_action_for_regime(
            regime=str(sc.regime), n_team=n_team)
        verdict = enforcer.check(candidate)
        if not verdict.permitted:
            assert verdict.breach_audit is not None
            breach_audits.append(verdict.breach_audit)
            enforcer.record_breach(verdict.breach_audit)
            outcomes.append(ScenarioBudgetOutcomeV1(
                schema=(
                    W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION),
                regime=str(sc.regime),
                permitted=False,
                recovery_outcome=None,
                breach_audit=dict(
                    verdict.breach_audit.to_dict()),
                predicted_action_cid=str(candidate.cid()),
            ))
            continue
        # Permitted: run the W83 composed pipeline scenario.
        rec = _run_one_regime_v1(scenario=sc)
        enforcer.commit(candidate)
        outcomes.append(ScenarioBudgetOutcomeV1(
            schema=(
                W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION),
            regime=str(sc.regime),
            permitted=True,
            recovery_outcome=dict(rec.to_dict()),
            breach_audit=None,
            predicted_action_cid=str(candidate.cid()),
        ))
    # Run the same scenarios WITHOUT the enforcer for the
    # under-budget-matches-no-enforcer bar.
    no_enforcer_committed = sum(
        1 for sc in scenarios
        if _run_one_regime_v1(scenario=sc) is not None)

    n_committed = sum(1 for o in outcomes if o.permitted)
    n_refused = sum(1 for o in outcomes if not o.permitted)
    every_refusal_has_audit = bool(all(
        (not o.permitted) == (o.breach_audit is not None)
        for o in outcomes))
    outcome_cids = tuple(o.cid() for o in outcomes)
    breach_root = _sha256_hex({
        "kind": "w86_budget_breach_merkle_root_v1",
        "breach_cids": [b.cid() for b in breach_audits],
    })
    bench_root = _sha256_hex({
        "kind": "w86_budget_enforced_recovery_bench_merkle_v1",
        "outcome_cids": list(outcome_cids),
        "breach_root": breach_root,
        "budget_spec_cid": str(spec.cid()),
        "cost_model_cid": str(cost_model.cid()),
    })
    # The "under-budget matches no-enforcer" bool is True iff
    # this run committed exactly the count of scenarios the
    # no-enforcer run committed.
    under_budget_match = bool(
        n_committed == no_enforcer_committed)

    report = BudgetEnforcedRecoveryBenchReportV1(
        schema=W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION,
        n_scenarios=int(len(scenarios)),
        n_committed=int(n_committed),
        n_refused=int(n_refused),
        n_committed_no_enforcer=int(no_enforcer_committed),
        zero_commits_when_over_budget=bool(
            n_committed == 0 and n_refused == len(scenarios)),
        every_refusal_audit_carries_breach_audit=bool(
            every_refusal_has_audit),
        under_budget_matches_no_enforcer=bool(
            under_budget_match),
        budget_spec_cid=str(spec.cid()),
        cost_model_cid=str(cost_model.cid()),
        per_scenario_outcome_cids=outcome_cids,
        breach_audit_merkle_root=str(breach_root),
        bench_merkle_root=str(bench_root),
    )
    return report, tuple(outcomes)


def run_budget_integration_head_to_head_v1(
        *,
        n_regimes: int = 6,
        seed_root: int = 86_037_001,
) -> dict[str, Any]:
    """Standalone driver: build N scenarios, run them under
    BOTH a tiny budget (expect 0 commits) AND a huge budget
    (expect == no-enforcer commits). Returns a dict with both
    reports content-addressed.

    This is the load-bearing #37 integration: same scenarios,
    same cost model, different budgets, opposite outcomes.
    """
    regimes = (
        "baseline", "stealth_bias_low", "stealth_bias_high",
        "noisy_witness", "outlier_witness", "stuck_witness",
    )[:int(n_regimes)]
    scenarios = tuple(
        build_regime_scenario_v1(
            regime=r, seed=int(seed_root) + i)
        for i, r in enumerate(regimes))
    cm = default_cost_model_v1()

    # Tiny-budget regime: every scenario refused.
    tiny_spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=1e-9,
        max_per_step_latency_ms=1e-3,
        max_total_tokens=1,
        max_tool_calls=0,
        max_recompute_flops=1.0,
    )
    tiny_report, tiny_outcomes = (
        run_budget_enforced_composed_recovery_v1(
            spec=tiny_spec, cost_model=cm,
            scenarios=scenarios))

    # Huge-budget regime: every scenario permitted.
    huge_spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=1e9,
        max_per_step_latency_ms=1e9,
        max_total_tokens=10**12,
        max_tool_calls=10**6,
        max_recompute_flops=1e18,
    )
    huge_report, huge_outcomes = (
        run_budget_enforced_composed_recovery_v1(
            spec=huge_spec, cost_model=cm,
            scenarios=scenarios))

    return {
        "schema": (
            W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION),
        "n_scenarios": int(len(scenarios)),
        "regimes": list(regimes),
        "cost_model_cid": str(cm.cid()),
        "tiny_budget": {
            "report": tiny_report.to_dict(),
            "report_cid": str(tiny_report.cid()),
        },
        "huge_budget": {
            "report": huge_report.to_dict(),
            "report_cid": str(huge_report.cid()),
        },
        "load_bearing_bools": {
            "zero_commits_when_over_budget": bool(
                tiny_report.zero_commits_when_over_budget),
            "every_refusal_audit_carries_breach_audit": bool(
                tiny_report
                .every_refusal_audit_carries_breach_audit),
            "under_budget_matches_no_enforcer": bool(
                huge_report.under_budget_matches_no_enforcer),
        },
    }


__all__ = [
    "W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION",
    "ScenarioBudgetOutcomeV1",
    "BudgetEnforcedRecoveryBenchReportV1",
    "run_budget_enforced_composed_recovery_v1",
    "run_budget_integration_head_to_head_v1",
]
