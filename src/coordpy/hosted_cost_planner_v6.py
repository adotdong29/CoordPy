"""W73 H4 — Hosted Cost / Latency Planner V6 (Plane A).

Strictly extends W72's ``coordpy.hosted_cost_planner_v5``. V6 adds:

* **Cost-per-replacement-rejoin-success-under-budget** — V5's
  cost-per-rejoin-success-under-budget refined with a per-turn
  replacement-pressure penalty.
* **Abstain-when-replacement-pressure-violated** — V6 returns
  ``rationale="abstain_v6_replacement_violated"`` (and a zero
  per-turn schedule) when caller-declared replacement pressure
  exceeds the caller's replacement-pressure cap.

Honest scope (W73 Plane A)
--------------------------

* Budgets, quality scores, restart pressure, rejoin pressure, AND
  replacement pressure are caller-supplied.
  ``W73-L-HOSTED-COST-PLANNER-V6-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v5 import (
    HostedCostPlanReportV5, HostedCostPlanSpecV5,
    plan_hosted_cost_v5,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_COST_PLANNER_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v6.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV6:
    inner_v5: HostedCostPlanSpecV5
    replacement_pressure: float = 0.0
    replacement_pressure_cap: float = 0.7
    abstain_when_replacement_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W73_HOSTED_COST_PLANNER_V6_SCHEMA_VERSION,
            "inner_v5_cid": str(self.inner_v5.cid()),
            "replacement_pressure": float(round(
                self.replacement_pressure, 12)),
            "replacement_pressure_cap": float(round(
                self.replacement_pressure_cap, 12)),
            "abstain_when_replacement_violated": bool(
                self.abstain_when_replacement_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v6",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV6:
    schema: str
    spec_v6_cid: str
    inner_v5_report: HostedCostPlanReportV5
    cost_per_replacement_rejoin_success_under_budget: float
    rationale_v6: str
    replacement_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v5_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v6_cid": str(self.spec_v6_cid),
            "inner_v5_report_cid": str(
                self.inner_v5_report.cid()),
            "cost_per_replacement_rejoin_success_under_budget":
                float(round(
                    self
                    .cost_per_replacement_rejoin_success_under_budget,
                    12)),
            "rationale_v6": str(self.rationale_v6),
            "replacement_violated": bool(
                self.replacement_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v6",
            "report": self.to_dict()})


def plan_hosted_cost_v6(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v6: HostedCostPlanSpecV6,
) -> HostedCostPlanReportV6:
    """V6 plan with cost-per-replacement-rejoin-success-under-
    budget + abstain-when-replacement-pressure-violated fallback."""
    inner = plan_hosted_cost_v5(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v5=spec_v6.inner_v5)
    pressure = float(max(0.0, min(
        1.0, float(spec_v6.replacement_pressure))))
    replacement_violated = bool(
        pressure > float(spec_v6.replacement_pressure_cap))
    abstain = (
        bool(spec_v6.abstain_when_replacement_violated)
        and bool(replacement_violated))
    if abstain or inner.rejoin_violated:
        return HostedCostPlanReportV6(
            schema=W73_HOSTED_COST_PLANNER_V6_SCHEMA_VERSION,
            spec_v6_cid=str(spec_v6.cid()),
            inner_v5_report=inner,
            cost_per_replacement_rejoin_success_under_budget=(
                float("inf")),
            rationale_v6=(
                "abstain_v6_replacement_violated"
                if abstain
                else "abstain_v6_rejoin_violated"),
            replacement_violated=bool(replacement_violated),
        )
    inner_cps = float(
        inner.cost_per_rejoin_success_under_budget)
    # Penalty for replacement pressure (multiplicative).
    cps_under_rep = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV6(
        schema=W73_HOSTED_COST_PLANNER_V6_SCHEMA_VERSION,
        spec_v6_cid=str(spec_v6.cid()),
        inner_v5_report=inner,
        cost_per_replacement_rejoin_success_under_budget=float(
            cps_under_rep),
        rationale_v6=(
            "v6_within_budget_and_replacement_safe"
            if not replacement_violated
            else "v6_replacement_violation_recorded"),
        replacement_violated=bool(replacement_violated),
    )


__all__ = [
    "W73_HOSTED_COST_PLANNER_V6_SCHEMA_VERSION",
    "HostedCostPlanSpecV6",
    "HostedCostPlanReportV6",
    "plan_hosted_cost_v6",
]
