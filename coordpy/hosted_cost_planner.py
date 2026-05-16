"""W68 H5 — Hosted Cost / Latency Planner (Plane A).

Cost/latency-aware provider selection over the hosted control plane.
Optimises a per-turn budget (cost in USD and p50 latency in ms)
under a matched-quality constraint (caller-supplied quality scores
per provider). Returns a per-turn allocation that minimises total
cost subject to the constraint.

Honest scope (W68 Plane A)
--------------------------

* Quality scores are caller-supplied; the planner does NOT measure
  output quality. ``W68-L-HOSTED-QUALITY-SCORE-CALLER-CAP``.
* The planner does NOT itself call hosted providers.
* The optimisation is a greedy knapsack over providers; for small
  N (≤ 20 providers) this is exact-equivalent on cost-equal
  candidates.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_COST_PLANNER_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner.v1")


@dataclasses.dataclass(frozen=True)
class HostedCostPlanSpec:
    n_turns: int
    input_tokens_per_turn: int
    output_tokens_per_turn: int
    max_cost_per_turn_usd: float
    max_latency_per_turn_ms: float
    min_quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W68_HOSTED_COST_PLANNER_SCHEMA_VERSION,
            "n_turns": int(self.n_turns),
            "input_tokens_per_turn": int(
                self.input_tokens_per_turn),
            "output_tokens_per_turn": int(
                self.output_tokens_per_turn),
            "max_cost_per_turn_usd": float(round(
                self.max_cost_per_turn_usd, 12)),
            "max_latency_per_turn_ms": float(round(
                self.max_latency_per_turn_ms, 12)),
            "min_quality_score": float(round(
                self.min_quality_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_spec",
            "spec": self.to_dict()})


def _per_turn_cost(
        p: HostedProvider, spec: HostedCostPlanSpec,
) -> float:
    return float(
        (float(spec.input_tokens_per_turn)
         * float(p.cost_per_1k_input) / 1000.0)
        + (float(spec.output_tokens_per_turn)
           * float(p.cost_per_1k_output) / 1000.0))


@dataclasses.dataclass(frozen=True)
class HostedCostPlanReport:
    schema: str
    spec_cid: str
    registry_cid: str
    chosen_provider: str | None
    estimated_total_cost_usd: float
    estimated_total_latency_ms: float
    rationale: str
    eligible_providers: tuple[str, ...]
    ineligible_providers: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "spec_cid": str(self.spec_cid),
            "registry_cid": str(self.registry_cid),
            "chosen_provider": (
                None if self.chosen_provider is None
                else str(self.chosen_provider)),
            "estimated_total_cost_usd": float(round(
                self.estimated_total_cost_usd, 12)),
            "estimated_total_latency_ms": float(round(
                self.estimated_total_latency_ms, 12)),
            "rationale": str(self.rationale),
            "eligible_providers": list(self.eligible_providers),
            "ineligible_providers": [
                [str(n), str(r)]
                for n, r in self.ineligible_providers],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_plan_report",
            "report": self.to_dict()})


def plan_hosted_cost(
        *, registry: HostedProviderRegistry,
        provider_quality_scores: dict[str, float],
        spec: HostedCostPlanSpec,
) -> HostedCostPlanReport:
    eligible: list[HostedProvider] = []
    ineligible: list[tuple[str, str]] = []
    for p in registry.providers:
        q = float(provider_quality_scores.get(p.name, 0.0))
        if q < float(spec.min_quality_score):
            ineligible.append((p.name, "quality"))
            continue
        if float(p.p50_latency_ms) > float(
                spec.max_latency_per_turn_ms):
            ineligible.append((p.name, "latency"))
            continue
        if _per_turn_cost(p, spec) > float(
                spec.max_cost_per_turn_usd):
            ineligible.append((p.name, "cost"))
            continue
        eligible.append(p)
    if not eligible:
        return HostedCostPlanReport(
            schema=W68_HOSTED_COST_PLANNER_SCHEMA_VERSION,
            spec_cid=str(spec.cid()),
            registry_cid=str(registry.cid()),
            chosen_provider=None,
            estimated_total_cost_usd=0.0,
            estimated_total_latency_ms=0.0,
            rationale="no_eligible_provider",
            eligible_providers=(),
            ineligible_providers=tuple(ineligible),
        )
    # Lexicographic preference: cheapest then fastest.
    def key(p: HostedProvider) -> tuple[float, float, str]:
        return (
            _per_turn_cost(p, spec),
            float(p.p50_latency_ms),
            str(p.name))
    sorted_eligible = sorted(eligible, key=key)
    winner = sorted_eligible[0]
    total_cost = float(
        _per_turn_cost(winner, spec) * spec.n_turns)
    total_latency = float(
        float(winner.p50_latency_ms) * spec.n_turns)
    return HostedCostPlanReport(
        schema=W68_HOSTED_COST_PLANNER_SCHEMA_VERSION,
        spec_cid=str(spec.cid()),
        registry_cid=str(registry.cid()),
        chosen_provider=str(winner.name),
        estimated_total_cost_usd=float(total_cost),
        estimated_total_latency_ms=float(total_latency),
        rationale="cheapest_then_fastest_with_quality_floor",
        eligible_providers=tuple(
            p.name for p in sorted_eligible),
        ineligible_providers=tuple(ineligible),
    )


__all__ = [
    "W68_HOSTED_COST_PLANNER_SCHEMA_VERSION",
    "HostedCostPlanSpec",
    "HostedCostPlanReport",
    "plan_hosted_cost",
]
