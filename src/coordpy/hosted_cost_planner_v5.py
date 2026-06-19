"""W72 H4 — Hosted Cost / Latency Planner V5 (Plane A).

Strictly extends W71's ``coordpy.hosted_cost_planner_v4``. V5 adds:

* **Cost-per-rejoin-success-under-budget** — V4's
  cost-per-repair-success-under-budget refined with a per-turn
  rejoin-pressure penalty.
* **Abstain-when-rejoin-pressure-violated** — V5 returns
  ``rationale="abstain_v5_rejoin_violated"`` (and a zero per-turn
  schedule) when caller-declared rejoin pressure exceeds the
  caller's rejoin-pressure cap.

Honest scope (W72 Plane A)
--------------------------

* Budgets, quality scores, restart pressure, and rejoin pressure
  are caller-supplied. ``W72-L-HOSTED-COST-PLANNER-V5-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v4 import (
    HostedCostPlanReportV4, HostedCostPlanSpecV4,
    plan_hosted_cost_v4,
)
from .hosted_router_controller import (
    HostedProviderRegistry,
)
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_COST_PLANNER_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v5.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV5:
    inner_v4: HostedCostPlanSpecV4
    rejoin_pressure: float = 0.0
    rejoin_pressure_cap: float = 0.7
    abstain_when_rejoin_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W72_HOSTED_COST_PLANNER_V5_SCHEMA_VERSION,
            "inner_v4_cid": str(self.inner_v4.cid()),
            "rejoin_pressure": float(round(
                self.rejoin_pressure, 12)),
            "rejoin_pressure_cap": float(round(
                self.rejoin_pressure_cap, 12)),
            "abstain_when_rejoin_violated": bool(
                self.abstain_when_rejoin_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v5",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV5:
    schema: str
    spec_v5_cid: str
    inner_v4_report: HostedCostPlanReportV4
    cost_per_rejoin_success_under_budget: float
    rationale_v5: str
    rejoin_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v4_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v5_cid": str(self.spec_v5_cid),
            "inner_v4_report_cid": str(
                self.inner_v4_report.cid()),
            "cost_per_rejoin_success_under_budget": float(round(
                self.cost_per_rejoin_success_under_budget, 12)),
            "rationale_v5": str(self.rationale_v5),
            "rejoin_violated": bool(self.rejoin_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v5",
            "report": self.to_dict()})


def plan_hosted_cost_v5(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v5: HostedCostPlanSpecV5,
) -> HostedCostPlanReportV5:
    """V5 plan with cost-per-rejoin-success-under-budget +
    abstain-when-rejoin-pressure-violated fallback."""
    inner = plan_hosted_cost_v4(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v4=spec_v5.inner_v4)
    pressure = float(max(0.0, min(
        1.0, float(spec_v5.rejoin_pressure))))
    rejoin_violated = bool(
        pressure > float(spec_v5.rejoin_pressure_cap))
    abstain = (
        bool(spec_v5.abstain_when_rejoin_violated)
        and bool(rejoin_violated))
    if abstain or inner.restart_violated:
        return HostedCostPlanReportV5(
            schema=W72_HOSTED_COST_PLANNER_V5_SCHEMA_VERSION,
            spec_v5_cid=str(spec_v5.cid()),
            inner_v4_report=inner,
            cost_per_rejoin_success_under_budget=float("inf"),
            rationale_v5=(
                "abstain_v5_rejoin_violated"
                if abstain
                else "abstain_v5_restart_violated"),
            rejoin_violated=bool(rejoin_violated),
        )
    inner_cps = float(
        inner.cost_per_repair_success_under_budget)
    # Penalty for rejoin pressure (multiplicative).
    cps_under_rejoin = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV5(
        schema=W72_HOSTED_COST_PLANNER_V5_SCHEMA_VERSION,
        spec_v5_cid=str(spec_v5.cid()),
        inner_v4_report=inner,
        cost_per_rejoin_success_under_budget=float(
            cps_under_rejoin),
        rationale_v5=(
            "v5_within_budget_and_rejoin_safe"
            if not rejoin_violated
            else "v5_rejoin_violation_recorded"),
        rejoin_violated=bool(rejoin_violated),
    )


__all__ = [
    "W72_HOSTED_COST_PLANNER_V5_SCHEMA_VERSION",
    "HostedCostPlanSpecV5",
    "HostedCostPlanReportV5",
    "plan_hosted_cost_v5",
]
