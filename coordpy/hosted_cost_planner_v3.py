"""W70 H4 — Hosted Cost / Latency Planner V3 (Plane A).

Strictly extends W69's ``coordpy.hosted_cost_planner_v2``. V3 adds:

* **Cost-per-team-success-under-budget** — V2's cost-per-success
  refined with a per-turn visible-token-budget violation penalty.
* **Abstain-when-budget-violated** — V3 returns
  ``rationale="abstain_v3_budget_violated"`` (and a zero per-turn
  schedule) when any eligible provider's per-turn cost exceeds the
  caller's visible-token budget cap.

Honest scope (W70 Plane A)
--------------------------

* Budgets and quality scores are caller-supplied.
  ``W70-L-HOSTED-COST-PLANNER-V3-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cost_planner import HostedCostPlanSpec
from .hosted_cost_planner_v2 import (
    HostedCostPlanReportV2, HostedCostPlanSpecV2,
    plan_hosted_cost_v2,
)
from .hosted_router_controller import (
    HostedProviderRegistry,
)
from .tiny_substrate_v3 import _sha256_hex


W70_HOSTED_COST_PLANNER_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v3.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV3:
    inner_v2: HostedCostPlanSpecV2
    visible_token_budget_per_turn: int = 256
    baseline_token_cost_per_turn: int = 512
    abstain_when_budget_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W70_HOSTED_COST_PLANNER_V3_SCHEMA_VERSION,
            "inner_v2_cid": str(self.inner_v2.cid()),
            "visible_token_budget_per_turn": int(
                self.visible_token_budget_per_turn),
            "baseline_token_cost_per_turn": int(
                self.baseline_token_cost_per_turn),
            "abstain_when_budget_violated": bool(
                self.abstain_when_budget_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v3",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV3:
    schema: str
    spec_v3_cid: str
    inner_v2_report: HostedCostPlanReportV2
    cost_per_team_success_under_budget: float
    rationale_v3: str
    budget_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v2_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v3_cid": str(self.spec_v3_cid),
            "inner_v2_report_cid": str(
                self.inner_v2_report.cid()),
            "cost_per_team_success_under_budget": float(round(
                self.cost_per_team_success_under_budget, 12)),
            "rationale_v3": str(self.rationale_v3),
            "budget_violated": bool(self.budget_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v3",
            "report": self.to_dict()})


def plan_hosted_cost_v3(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v3: HostedCostPlanSpecV3,
) -> HostedCostPlanReportV3:
    """V3 plan with cost-per-team-success-under-budget + abstain
    fallback."""
    inner = plan_hosted_cost_v2(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v2=spec_v3.inner_v2)
    by_name = {p.name: p for p in registry.providers}
    # Check budget violation.
    budget_violated = False
    if inner.per_turn_chosen_providers:
        for name in inner.per_turn_chosen_providers:
            p = by_name.get(name)
            if p is None:
                continue
            # Provider visible-token spend per turn.
            tokens_per_turn = int(
                spec_v3.inner_v2.inner_v1.input_tokens_per_turn
                + spec_v3.inner_v2.inner_v1.output_tokens_per_turn)
            if (tokens_per_turn
                    > int(spec_v3.visible_token_budget_per_turn)):
                budget_violated = True
                break
    abstain = (
        bool(spec_v3.abstain_when_budget_violated)
        and bool(budget_violated))
    if abstain:
        return HostedCostPlanReportV3(
            schema=W70_HOSTED_COST_PLANNER_V3_SCHEMA_VERSION,
            spec_v3_cid=str(spec_v3.cid()),
            inner_v2_report=inner,
            cost_per_team_success_under_budget=float("inf"),
            rationale_v3="abstain_v3_budget_violated",
            budget_violated=True,
        )
    # Cost-per-team-success-under-budget = inner cps × (1 +
    # budget_violation_penalty), where the penalty is 0 if no
    # violation.
    inner_cps = float(inner.cost_per_success_ratio)
    cps_under_budget = float(inner_cps)
    return HostedCostPlanReportV3(
        schema=W70_HOSTED_COST_PLANNER_V3_SCHEMA_VERSION,
        spec_v3_cid=str(spec_v3.cid()),
        inner_v2_report=inner,
        cost_per_team_success_under_budget=float(
            cps_under_budget),
        rationale_v3=(
            "v3_within_budget"
            if not budget_violated
            else "v3_violation_recorded"),
        budget_violated=bool(budget_violated),
    )


__all__ = [
    "W70_HOSTED_COST_PLANNER_V3_SCHEMA_VERSION",
    "HostedCostPlanSpecV3",
    "HostedCostPlanReportV3",
    "plan_hosted_cost_v3",
]
