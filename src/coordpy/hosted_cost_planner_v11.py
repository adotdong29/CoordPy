"""W78 H4 — Hosted Cost / Latency Planner V11 (Plane A).

Strictly extends W77's ``coordpy.hosted_cost_planner_v10``. V11
adds:

* **Cost-per-long-horizon-reconstruction-success-under-budget** —
  V10's cost-per-post-restart-replacement-success-under-budget
  refined with a per-turn long-horizon-reconstruction-pressure
  penalty.
* **Abstain-when-long-horizon-reconstruction-pressure-violated** —
  V11 returns ``rationale_v11=
  "abstain_v11_long_horizon_reconstruction_violated"`` (and an
  infinite cost) when caller-declared long-horizon-reconstruction
  pressure exceeds the caller's cap.

Honest scope (W78 Plane A): all pressures are caller-supplied.
``W78-L-HOSTED-COST-PLANNER-V11-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_cost_planner_v10 import (
    HostedCostPlanReportV10, HostedCostPlanSpecV10,
    plan_hosted_cost_v10,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_COST_PLANNER_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v11.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV11:
    inner_v10: HostedCostPlanSpecV10
    long_horizon_reconstruction_pressure: float = 0.0
    long_horizon_reconstruction_pressure_cap: float = 0.7
    abstain_when_long_horizon_reconstruction_violated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W78_HOSTED_COST_PLANNER_V11_SCHEMA_VERSION,
            "inner_v10_cid": str(self.inner_v10.cid()),
            "long_horizon_reconstruction_pressure": float(round(
                self.long_horizon_reconstruction_pressure, 12)),
            "long_horizon_reconstruction_pressure_cap": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_cap,
                    12)),
            "abstain_when_long_horizon_reconstruction_violated":
                bool(
                    self
                    .abstain_when_long_horizon_reconstruction_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v11",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV11:
    schema: str
    spec_v11_cid: str
    inner_v10_report: HostedCostPlanReportV10
    cost_per_long_horizon_reconstruction_success_under_budget: (
        float)
    rationale_v11: str
    long_horizon_reconstruction_violated: bool

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v10_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v11_cid": str(self.spec_v11_cid),
            "inner_v10_report_cid": str(
                self.inner_v10_report.cid()),
            "cost_per_long_horizon_reconstruction_success_under_budget":
                float(round(
                    self
                    .cost_per_long_horizon_reconstruction_success_under_budget,
                    12)),
            "rationale_v11": str(self.rationale_v11),
            "long_horizon_reconstruction_violated": bool(
                self.long_horizon_reconstruction_violated),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v11",
            "report": self.to_dict()})


def plan_hosted_cost_v11(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v11: HostedCostPlanSpecV11,
) -> HostedCostPlanReportV11:
    """V11 plan with cost-per-long-horizon-reconstruction-success-
    under-budget + abstain-when-long-horizon-reconstruction-
    pressure-violated fallback."""
    inner = plan_hosted_cost_v10(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec_v10=spec_v11.inner_v10)
    pressure = float(max(0.0, min(
        1.0,
        float(spec_v11.long_horizon_reconstruction_pressure))))
    lhr_violated = bool(
        pressure
        > float(
            spec_v11.long_horizon_reconstruction_pressure_cap))
    abstain = (
        bool(
            spec_v11
            .abstain_when_long_horizon_reconstruction_violated)
        and bool(lhr_violated))
    if (abstain
            or inner.post_restart_replacement_violated
            or inner.inner_v9_report.chain_then_restart_violated):
        return HostedCostPlanReportV11(
            schema=W78_HOSTED_COST_PLANNER_V11_SCHEMA_VERSION,
            spec_v11_cid=str(spec_v11.cid()),
            inner_v10_report=inner,
            cost_per_long_horizon_reconstruction_success_under_budget=(
                float("inf")),
            rationale_v11=(
                "abstain_v11_long_horizon_reconstruction_violated"
                if abstain
                else "abstain_v11_inner_violated"),
            long_horizon_reconstruction_violated=bool(
                lhr_violated),
        )
    inner_cps = float(
        inner
        .cost_per_post_restart_replacement_success_under_budget)
    # Penalty for long-horizon-reconstruction pressure
    # (multiplicative).
    cps_under_lhr = float(
        inner_cps * (1.0 + 0.5 * float(pressure)))
    return HostedCostPlanReportV11(
        schema=W78_HOSTED_COST_PLANNER_V11_SCHEMA_VERSION,
        spec_v11_cid=str(spec_v11.cid()),
        inner_v10_report=inner,
        cost_per_long_horizon_reconstruction_success_under_budget=(
            float(cps_under_lhr)),
        rationale_v11=(
            "v11_within_budget_and_long_horizon_reconstruction_safe"
            if not lhr_violated
            else (
                "v11_long_horizon_reconstruction_violation_"
                "recorded")),
        long_horizon_reconstruction_violated=bool(lhr_violated),
    )


__all__ = [
    "W78_HOSTED_COST_PLANNER_V11_SCHEMA_VERSION",
    "HostedCostPlanSpecV11",
    "HostedCostPlanReportV11",
    "plan_hosted_cost_v11",
]
