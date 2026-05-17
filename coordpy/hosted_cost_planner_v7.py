"""W74 H4 — Hosted Cost / Latency Planner V7 (Plane A).

Strictly extends W73's ``coordpy.hosted_cost_planner_v6``. V7 adds:

* **Cost-per-compound-success-under-budget** — V6's cost-per-
  replacement-rejoin-success-under-budget refined with a per-turn
  compound-pressure penalty.
* **Abstain-when-compound-pressure-violated** — V7 returns
  ``rationale="abstain_v7_compound_violated"`` (and a zero
  per-turn schedule) when caller-declared compound pressure
  exceeds the caller's compound-pressure cap.

Honest scope (W74 Plane A)
--------------------------

* Budgets, quality scores, restart pressure, rejoin pressure,
  replacement pressure, AND compound pressure are caller-supplied.
  ``W74-L-HOSTED-COST-PLANNER-V7-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v6 import (
    HostedCostPlanReportV6, HostedCostPlanSpecV6,
    plan_hosted_cost_v6,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W74_HOSTED_COST_PLANNER_V7_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v7.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV7:
    inner_v6: HostedCostPlanSpecV6
    compound_pressure: float = 0.0
    compound_pressure_cap: float = 0.7
    abstain_when_compound_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W74_HOSTED_COST_PLANNER_V7_SCHEMA_VERSION,
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_pressure": float(round(
                self.compound_pressure, 12)),
            "compound_pressure_cap": float(round(
                self.compound_pressure_cap, 12)),
            "abstain_when_compound_violated": bool(
                self.abstain_when_compound_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v7",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV7:
    schema: str
    spec_v7_cid: str
    inner_v6_report: HostedCostPlanReportV6
    cost_per_compound_success_under_budget: float
    rationale_v7: str
    compound_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v6_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v7_cid": str(self.spec_v7_cid),
            "inner_v6_report_cid": str(
                self.inner_v6_report.cid()),
            "cost_per_compound_success_under_budget":
                float(round(
                    self
                    .cost_per_compound_success_under_budget,
                    12)),
            "rationale_v7": str(self.rationale_v7),
            "compound_violated": bool(
                self.compound_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v7",
            "report": self.to_dict()})


def plan_hosted_cost_v7(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v7: HostedCostPlanSpecV7,
) -> HostedCostPlanReportV7:
    """V7 plan with cost-per-compound-success-under-budget +
    abstain-when-compound-pressure-violated fallback."""
    inner = plan_hosted_cost_v6(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v6=spec_v7.inner_v6)
    pressure = float(max(0.0, min(
        1.0, float(spec_v7.compound_pressure))))
    compound_violated = bool(
        pressure > float(spec_v7.compound_pressure_cap))
    abstain = (
        bool(spec_v7.abstain_when_compound_violated)
        and bool(compound_violated))
    if abstain or inner.replacement_violated:
        return HostedCostPlanReportV7(
            schema=W74_HOSTED_COST_PLANNER_V7_SCHEMA_VERSION,
            spec_v7_cid=str(spec_v7.cid()),
            inner_v6_report=inner,
            cost_per_compound_success_under_budget=(
                float("inf")),
            rationale_v7=(
                "abstain_v7_compound_violated"
                if abstain
                else "abstain_v7_replacement_violated"),
            compound_violated=bool(compound_violated),
        )
    inner_cps = float(
        inner.cost_per_replacement_rejoin_success_under_budget)
    # Penalty for compound pressure (multiplicative).
    cps_under_cmp = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV7(
        schema=W74_HOSTED_COST_PLANNER_V7_SCHEMA_VERSION,
        spec_v7_cid=str(spec_v7.cid()),
        inner_v6_report=inner,
        cost_per_compound_success_under_budget=float(
            cps_under_cmp),
        rationale_v7=(
            "v7_within_budget_and_compound_safe"
            if not compound_violated
            else "v7_compound_violation_recorded"),
        compound_violated=bool(compound_violated),
    )


__all__ = [
    "W74_HOSTED_COST_PLANNER_V7_SCHEMA_VERSION",
    "HostedCostPlanSpecV7",
    "HostedCostPlanReportV7",
    "plan_hosted_cost_v7",
]
