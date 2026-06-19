"""W76 H4 — Hosted Cost / Latency Planner V9 (Plane A).

Strictly extends W75's ``coordpy.hosted_cost_planner_v8``. V9 adds:

* **Cost-per-chain-then-restart-success-under-budget** — V8's
  cost-per-compound-chain-success-under-budget refined with a
  per-turn chain-then-restart-pressure penalty.
* **Abstain-when-chain-then-restart-pressure-violated** — V9
  returns ``rationale="abstain_v9_chain_then_restart_violated"``
  (and a zero per-turn schedule) when caller-declared chain-then-
  restart pressure exceeds the caller's cap.

Honest scope (W76 Plane A): all pressures are caller-supplied.
``W76-L-HOSTED-COST-PLANNER-V9-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v8 import (
    HostedCostPlanReportV8, HostedCostPlanSpecV8,
    plan_hosted_cost_v8,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_COST_PLANNER_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v9.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV9:
    inner_v8: HostedCostPlanSpecV8
    chain_then_restart_pressure: float = 0.0
    chain_then_restart_pressure_cap: float = 0.7
    abstain_when_chain_then_restart_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W76_HOSTED_COST_PLANNER_V9_SCHEMA_VERSION,
            "inner_v8_cid": str(self.inner_v8.cid()),
            "chain_then_restart_pressure": float(round(
                self.chain_then_restart_pressure, 12)),
            "chain_then_restart_pressure_cap": float(round(
                self.chain_then_restart_pressure_cap, 12)),
            "abstain_when_chain_then_restart_violated": bool(
                self.abstain_when_chain_then_restart_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v9",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV9:
    schema: str
    spec_v9_cid: str
    inner_v8_report: HostedCostPlanReportV8
    cost_per_chain_then_restart_success_under_budget: float
    rationale_v9: str
    chain_then_restart_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v8_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v9_cid": str(self.spec_v9_cid),
            "inner_v8_report_cid": str(
                self.inner_v8_report.cid()),
            "cost_per_chain_then_restart_success_under_budget":
                float(round(
                    self
                    .cost_per_chain_then_restart_success_under_budget,
                    12)),
            "rationale_v9": str(self.rationale_v9),
            "chain_then_restart_violated": bool(
                self.chain_then_restart_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v9",
            "report": self.to_dict()})


def plan_hosted_cost_v9(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v9: HostedCostPlanSpecV9,
) -> HostedCostPlanReportV9:
    """V9 plan with cost-per-chain-then-restart-success-under-
    budget + abstain-when-chain-then-restart-pressure-violated
    fallback."""
    inner = plan_hosted_cost_v8(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v8=spec_v9.inner_v8)
    pressure = float(max(0.0, min(
        1.0, float(spec_v9.chain_then_restart_pressure))))
    ctr_violated = bool(
        pressure > float(spec_v9.chain_then_restart_pressure_cap))
    abstain = (
        bool(spec_v9.abstain_when_chain_then_restart_violated)
        and bool(ctr_violated))
    if abstain or inner.compound_chain_violated:
        return HostedCostPlanReportV9(
            schema=W76_HOSTED_COST_PLANNER_V9_SCHEMA_VERSION,
            spec_v9_cid=str(spec_v9.cid()),
            inner_v8_report=inner,
            cost_per_chain_then_restart_success_under_budget=(
                float("inf")),
            rationale_v9=(
                "abstain_v9_chain_then_restart_violated"
                if abstain
                else "abstain_v9_compound_chain_violated"),
            chain_then_restart_violated=bool(ctr_violated),
        )
    inner_cps = float(
        inner.cost_per_compound_chain_success_under_budget)
    # Penalty for chain-then-restart pressure (multiplicative).
    cps_under_ctr = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV9(
        schema=W76_HOSTED_COST_PLANNER_V9_SCHEMA_VERSION,
        spec_v9_cid=str(spec_v9.cid()),
        inner_v8_report=inner,
        cost_per_chain_then_restart_success_under_budget=float(
            cps_under_ctr),
        rationale_v9=(
            "v9_within_budget_and_chain_then_restart_safe"
            if not ctr_violated
            else "v9_chain_then_restart_violation_recorded"),
        chain_then_restart_violated=bool(ctr_violated),
    )


__all__ = [
    "W76_HOSTED_COST_PLANNER_V9_SCHEMA_VERSION",
    "HostedCostPlanSpecV9",
    "HostedCostPlanReportV9",
    "plan_hosted_cost_v9",
]
