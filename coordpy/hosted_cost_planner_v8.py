"""W75 H4 — Hosted Cost / Latency Planner V8 (Plane A).

Strictly extends W74's ``coordpy.hosted_cost_planner_v7``. V8 adds:

* **Cost-per-compound-chain-success-under-budget** — V7's cost-
  per-compound-success-under-budget refined with a per-turn
  compound-chain-pressure penalty.
* **Abstain-when-compound-chain-pressure-violated** — V8 returns
  ``rationale="abstain_v8_compound_chain_violated"`` (and a zero
  per-turn schedule) when caller-declared compound-chain pressure
  exceeds the caller's compound-chain-pressure cap.

Honest scope (W75 Plane A)
--------------------------

* Budgets, quality scores, restart pressure, rejoin pressure,
  replacement pressure, compound pressure, AND compound-chain
  pressure are caller-supplied.
  ``W75-L-HOSTED-COST-PLANNER-V8-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v7 import (
    HostedCostPlanReportV7, HostedCostPlanSpecV7,
    plan_hosted_cost_v7,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_COST_PLANNER_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v8.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV8:
    inner_v7: HostedCostPlanSpecV7
    compound_chain_pressure: float = 0.0
    compound_chain_pressure_cap: float = 0.7
    abstain_when_compound_chain_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W75_HOSTED_COST_PLANNER_V8_SCHEMA_VERSION,
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_pressure": float(round(
                self.compound_chain_pressure, 12)),
            "compound_chain_pressure_cap": float(round(
                self.compound_chain_pressure_cap, 12)),
            "abstain_when_compound_chain_violated": bool(
                self.abstain_when_compound_chain_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v8",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV8:
    schema: str
    spec_v8_cid: str
    inner_v7_report: HostedCostPlanReportV7
    cost_per_compound_chain_success_under_budget: float
    rationale_v8: str
    compound_chain_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v7_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v8_cid": str(self.spec_v8_cid),
            "inner_v7_report_cid": str(
                self.inner_v7_report.cid()),
            "cost_per_compound_chain_success_under_budget":
                float(round(
                    self
                    .cost_per_compound_chain_success_under_budget,
                    12)),
            "rationale_v8": str(self.rationale_v8),
            "compound_chain_violated": bool(
                self.compound_chain_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v8",
            "report": self.to_dict()})


def plan_hosted_cost_v8(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v8: HostedCostPlanSpecV8,
) -> HostedCostPlanReportV8:
    """V8 plan with cost-per-compound-chain-success-under-budget +
    abstain-when-compound-chain-pressure-violated fallback."""
    inner = plan_hosted_cost_v7(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v7=spec_v8.inner_v7)
    pressure = float(max(0.0, min(
        1.0, float(spec_v8.compound_chain_pressure))))
    chain_violated = bool(
        pressure > float(spec_v8.compound_chain_pressure_cap))
    abstain = (
        bool(spec_v8.abstain_when_compound_chain_violated)
        and bool(chain_violated))
    if abstain or inner.compound_violated:
        return HostedCostPlanReportV8(
            schema=W75_HOSTED_COST_PLANNER_V8_SCHEMA_VERSION,
            spec_v8_cid=str(spec_v8.cid()),
            inner_v7_report=inner,
            cost_per_compound_chain_success_under_budget=(
                float("inf")),
            rationale_v8=(
                "abstain_v8_compound_chain_violated"
                if abstain
                else "abstain_v8_compound_violated"),
            compound_chain_violated=bool(chain_violated),
        )
    inner_cps = float(
        inner.cost_per_compound_success_under_budget)
    # Penalty for compound-chain pressure (multiplicative).
    cps_under_chain = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV8(
        schema=W75_HOSTED_COST_PLANNER_V8_SCHEMA_VERSION,
        spec_v8_cid=str(spec_v8.cid()),
        inner_v7_report=inner,
        cost_per_compound_chain_success_under_budget=float(
            cps_under_chain),
        rationale_v8=(
            "v8_within_budget_and_compound_chain_safe"
            if not chain_violated
            else "v8_compound_chain_violation_recorded"),
        compound_chain_violated=bool(chain_violated),
    )


__all__ = [
    "W75_HOSTED_COST_PLANNER_V8_SCHEMA_VERSION",
    "HostedCostPlanSpecV8",
    "HostedCostPlanReportV8",
    "plan_hosted_cost_v8",
]
