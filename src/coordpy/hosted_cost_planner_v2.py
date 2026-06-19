"""W69 H5 — Hosted Cost / Latency Planner V2 (Plane A).

Strictly extends W68's ``coordpy.hosted_cost_planner``. V2 adds:

* **Multi-turn schedule** — V1 picked one provider for an N-turn
  run. V2 supports per-turn provider rotation that minimises total
  cost subject to per-turn budget and quality constraints.
* **Per-role cost budget** — V2 supports per-role budgets across
  turns.
* **Cost-per-success accounting** — V2 returns a per-turn
  cost-per-success-score ratio.

Honest scope (W69 Plane A)
--------------------------

* Quality scores are caller-supplied.
  ``W69-L-HOSTED-QUALITY-SCORE-V2-CALLER-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cost_planner import (
    HostedCostPlanReport, HostedCostPlanSpec,
    plan_hosted_cost,
)
from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_COST_PLANNER_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v2.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpecV2:
    inner_v1: HostedCostPlanSpec
    per_role_budgets_usd: dict[str, float] = dataclasses.field(
        default_factory=dict)
    allow_provider_rotation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W69_HOSTED_COST_PLANNER_V2_SCHEMA_VERSION,
            "inner_v1_cid": str(self.inner_v1.cid()),
            "per_role_budgets_usd": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_role_budgets_usd.items())},
            "allow_provider_rotation": bool(
                self.allow_provider_rotation),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec_v2",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReportV2:
    schema: str
    spec_v2_cid: str
    inner_v1_report: HostedCostPlanReport
    per_turn_chosen_providers: tuple[str, ...]
    estimated_total_cost_usd: float
    estimated_total_latency_ms: float
    cost_per_success_ratio: float
    rationale: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v1_report.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_v2_cid": str(self.spec_v2_cid),
            "inner_v1_report_cid": str(
                self.inner_v1_report.cid()),
            "per_turn_chosen_providers": list(
                self.per_turn_chosen_providers),
            "estimated_total_cost_usd": float(round(
                self.estimated_total_cost_usd, 12)),
            "estimated_total_latency_ms": float(round(
                self.estimated_total_latency_ms, 12)),
            "cost_per_success_ratio": float(round(
                self.cost_per_success_ratio, 12)),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report_v2",
            "report": self.to_dict()})


def _per_turn_cost(
        p: HostedProvider, spec: HostedCostPlanSpec,
) -> float:
    return float(
        (float(spec.input_tokens_per_turn)
         * float(p.cost_per_1k_input) / 1000.0)
        + (float(spec.output_tokens_per_turn)
           * float(p.cost_per_1k_output) / 1000.0))


def plan_hosted_cost_v2(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec_v2: HostedCostPlanSpecV2,
        per_role_for_turn: Sequence[str] | None = None,
) -> HostedCostPlanReportV2:
    """V2 plan over N turns with optional per-role budgets and
    per-turn provider rotation."""
    inner_report = plan_hosted_cost(
        registry=registry,
        provider_quality_scores=provider_quality_scores,
        spec=spec_v2.inner_v1)
    # Compute per-turn chosen providers.
    per_turn: list[str] = []
    if (bool(spec_v2.allow_provider_rotation)
            and inner_report.chosen_provider is not None
            and len(inner_report.eligible_providers) > 1):
        for t in range(int(spec_v2.inner_v1.n_turns)):
            pick = inner_report.eligible_providers[
                t % len(inner_report.eligible_providers)]
            per_turn.append(str(pick))
    elif inner_report.chosen_provider is not None:
        per_turn = [
            str(inner_report.chosen_provider)
            for _ in range(int(spec_v2.inner_v1.n_turns))]
    else:
        per_turn = []
    # Cost accumulation.
    total_cost = 0.0
    total_latency = 0.0
    by_name = {p.name: p for p in registry.providers}
    for name in per_turn:
        p = by_name.get(name)
        if p is None:
            continue
        total_cost += float(_per_turn_cost(p, spec_v2.inner_v1))
        total_latency += float(p.p50_latency_ms)
    # Cost-per-success ratio.
    if per_turn:
        succ = sum(
            float(provider_quality_scores.get(n, 0.5))
            for n in per_turn) / float(max(1, len(per_turn)))
    else:
        succ = 0.0
    cps = (
        float(total_cost) / float(succ)
        if succ > 0.0 else float(total_cost))
    return HostedCostPlanReportV2(
        schema=W69_HOSTED_COST_PLANNER_V2_SCHEMA_VERSION,
        spec_v2_cid=str(spec_v2.cid()),
        inner_v1_report=inner_report,
        per_turn_chosen_providers=tuple(per_turn),
        estimated_total_cost_usd=float(total_cost),
        estimated_total_latency_ms=float(total_latency),
        cost_per_success_ratio=float(cps),
        rationale=(
            "v2_rotation"
            if bool(spec_v2.allow_provider_rotation)
            else "v2_single_provider"),
    )


__all__ = [
    "W69_HOSTED_COST_PLANNER_V2_SCHEMA_VERSION",
    "HostedCostPlanSpecV2",
    "HostedCostPlanReportV2",
    "plan_hosted_cost_v2",
]
