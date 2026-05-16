"""W71 H4 — Hosted Cost / Latency Planner V4 (Plane A).

Strictly extends W70's ``coordpy.hosted_cost_planner_v3``. V4 adds:

* **Cost-per-repair-success-under-budget** — V3's
  cost-per-team-success-under-budget refined with a per-turn
  repair-dominance penalty.
* **Abstain-when-restart-pressure-violated** — V4 returns
  ``rationale="abstain_v4_restart_violated"`` (and a zero per-turn
  schedule) when caller-declared restart pressure exceeds the
  caller's restart-pressure cap.

Honest scope (W71 Plane A)
--------------------------

* Budgets, quality scores, and restart-pressure are caller-
  supplied. ``W71-L-HOSTED-COST-PLANNER-V4-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v3 import (
    HostedCostPlanReportV3, HostedCostPlanSpecV3,
    plan_hosted_cost_v3,
)
from .hosted_router_controller import (
    HostedProviderRegistry,
)
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_COST_PLANNER_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v4.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV4:
    inner_v3: HostedCostPlanSpecV3
    restart_pressure: float = 0.0
    restart_pressure_cap: float = 0.7
    abstain_when_restart_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W71_HOSTED_COST_PLANNER_V4_SCHEMA_VERSION,
            "inner_v3_cid": str(self.inner_v3.cid()),
            "restart_pressure": float(round(
                self.restart_pressure, 12)),
            "restart_pressure_cap": float(round(
                self.restart_pressure_cap, 12)),
            "abstain_when_restart_violated": bool(
                self.abstain_when_restart_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v4",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV4:
    schema: str
    spec_v4_cid: str
    inner_v3_report: HostedCostPlanReportV3
    cost_per_repair_success_under_budget: float
    rationale_v4: str
    restart_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v3_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v4_cid": str(self.spec_v4_cid),
            "inner_v3_report_cid": str(
                self.inner_v3_report.cid()),
            "cost_per_repair_success_under_budget": float(round(
                self.cost_per_repair_success_under_budget, 12)),
            "rationale_v4": str(self.rationale_v4),
            "restart_violated": bool(self.restart_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v4",
            "report": self.to_dict()})


def plan_hosted_cost_v4(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v4: HostedCostPlanSpecV4,
) -> HostedCostPlanReportV4:
    """V4 plan with cost-per-repair-success-under-budget +
    abstain-when-restart-pressure-violated fallback."""
    inner = plan_hosted_cost_v3(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v3=spec_v4.inner_v3)
    pressure = float(max(0.0, min(
        1.0, float(spec_v4.restart_pressure))))
    restart_violated = bool(
        pressure > float(spec_v4.restart_pressure_cap))
    abstain = (
        bool(spec_v4.abstain_when_restart_violated)
        and bool(restart_violated))
    if abstain or inner.budget_violated:
        return HostedCostPlanReportV4(
            schema=W71_HOSTED_COST_PLANNER_V4_SCHEMA_VERSION,
            spec_v4_cid=str(spec_v4.cid()),
            inner_v3_report=inner,
            cost_per_repair_success_under_budget=float("inf"),
            rationale_v4=(
                "abstain_v4_restart_violated"
                if abstain else "abstain_v4_budget_violated"),
            restart_violated=bool(restart_violated),
        )
    inner_cps = float(
        inner.cost_per_team_success_under_budget)
    # Penalty for restart pressure (multiplicative).
    cps_under_repair = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV4(
        schema=W71_HOSTED_COST_PLANNER_V4_SCHEMA_VERSION,
        spec_v4_cid=str(spec_v4.cid()),
        inner_v3_report=inner,
        cost_per_repair_success_under_budget=float(
            cps_under_repair),
        rationale_v4=(
            "v4_within_budget_and_restart_safe"
            if not restart_violated
            else "v4_restart_violation_recorded"),
        restart_violated=bool(restart_violated),
    )


__all__ = [
    "W71_HOSTED_COST_PLANNER_V4_SCHEMA_VERSION",
    "HostedCostPlanSpecV4",
    "HostedCostPlanReportV4",
    "plan_hosted_cost_v4",
]
