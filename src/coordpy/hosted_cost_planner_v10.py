"""W77 H4 — Hosted Cost / Latency Planner V10 (Plane A).

Strictly extends W76's ``coordpy.hosted_cost_planner_v9``. V10
adds:

* **Cost-per-post-restart-replacement-success-under-budget** —
  V9's cost-per-chain-then-restart-success-under-budget refined
  with a per-turn post-restart-replacement-pressure penalty.
* **Abstain-when-post-restart-replacement-pressure-violated** —
  V10 returns ``rationale_v10="abstain_v10_post_restart_replacement_violated"``
  (and a zero per-turn schedule) when caller-declared post-restart-
  replacement pressure exceeds the caller's cap.

Honest scope (W77 Plane A): all pressures are caller-supplied.
``W77-L-HOSTED-COST-PLANNER-V10-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v9 import (
    HostedCostPlanReportV9, HostedCostPlanSpecV9,
    plan_hosted_cost_v9,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_COST_PLANNER_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v10.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV10:
    inner_v9: HostedCostPlanSpecV9
    post_restart_replacement_pressure: float = 0.0
    post_restart_replacement_pressure_cap: float = 0.7
    abstain_when_post_restart_replacement_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W77_HOSTED_COST_PLANNER_V10_SCHEMA_VERSION,
            "inner_v9_cid": str(self.inner_v9.cid()),
            "post_restart_replacement_pressure": float(round(
                self.post_restart_replacement_pressure, 12)),
            "post_restart_replacement_pressure_cap": float(round(
                self.post_restart_replacement_pressure_cap, 12)),
            "abstain_when_post_restart_replacement_violated":
                bool(
                    self
                    .abstain_when_post_restart_replacement_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v10",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV10:
    schema: str
    spec_v10_cid: str
    inner_v9_report: HostedCostPlanReportV9
    cost_per_post_restart_replacement_success_under_budget: float
    rationale_v10: str
    post_restart_replacement_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v9_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v10_cid": str(self.spec_v10_cid),
            "inner_v9_report_cid": str(
                self.inner_v9_report.cid()),
            "cost_per_post_restart_replacement_success_under_budget":
                float(round(
                    self
                    .cost_per_post_restart_replacement_success_under_budget,
                    12)),
            "rationale_v10": str(self.rationale_v10),
            "post_restart_replacement_violated": bool(
                self.post_restart_replacement_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v10",
            "report": self.to_dict()})


def plan_hosted_cost_v10(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v10: HostedCostPlanSpecV10,
) -> HostedCostPlanReportV10:
    """V10 plan with cost-per-post-restart-replacement-success-
    under-budget + abstain-when-post-restart-replacement-pressure-
    violated fallback."""
    inner = plan_hosted_cost_v9(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v9=spec_v10.inner_v9)
    pressure = float(max(0.0, min(
        1.0, float(spec_v10.post_restart_replacement_pressure))))
    pcr_violated = bool(
        pressure
        > float(spec_v10.post_restart_replacement_pressure_cap))
    abstain = (
        bool(
            spec_v10
            .abstain_when_post_restart_replacement_violated)
        and bool(pcr_violated))
    if (abstain
            or inner.chain_then_restart_violated
            or inner.inner_v8_report.compound_chain_violated):
        return HostedCostPlanReportV10(
            schema=W77_HOSTED_COST_PLANNER_V10_SCHEMA_VERSION,
            spec_v10_cid=str(spec_v10.cid()),
            inner_v9_report=inner,
            cost_per_post_restart_replacement_success_under_budget=(
                float("inf")),
            rationale_v10=(
                "abstain_v10_post_restart_replacement_violated"
                if abstain
                else "abstain_v10_inner_violated"),
            post_restart_replacement_violated=bool(pcr_violated),
        )
    inner_cps = float(
        inner.cost_per_chain_then_restart_success_under_budget)
    # Penalty for post-restart-replacement pressure
    # (multiplicative).
    cps_under_pcr = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV10(
        schema=W77_HOSTED_COST_PLANNER_V10_SCHEMA_VERSION,
        spec_v10_cid=str(spec_v10.cid()),
        inner_v9_report=inner,
        cost_per_post_restart_replacement_success_under_budget=(
            float(cps_under_pcr)),
        rationale_v10=(
            "v10_within_budget_and_post_restart_replacement_safe"
            if not pcr_violated
            else "v10_post_restart_replacement_violation_recorded"),
        post_restart_replacement_violated=bool(pcr_violated),
    )


__all__ = [
    "W77_HOSTED_COST_PLANNER_V10_SCHEMA_VERSION",
    "HostedCostPlanSpecV10",
    "HostedCostPlanReportV10",
    "plan_hosted_cost_v10",
]
