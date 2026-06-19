"""W70 H1 — Hosted Router Controller V3 (Plane A).

Strictly extends W69's
``coordpy.hosted_router_controller_v2``. V3 adds:

* **Budget-aware weighted scoring** — adds two new weights:
  ``weight_budget_efficiency`` and ``weight_repair_dominance_match``.
* **Per-budget routing CID** — exposes a content-addressed routing
  decision CID that depends on the declared visible-token budget.
* **Routing-cost-per-team-success-under-budget** — refines V2's
  bookkeeping to weight by both team-success and budget violation.

Honest scope (W70 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Budgets and success scores are caller-declared (the router does
  not measure live success). ``W70-L-HOSTED-V3-NO-SUBSTRATE-CAP``,
  ``W70-L-HOSTED-V3-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
    HostedRoutingDecision, HostedRoutingRequest,
    HostedRouterController,
    W68_HOSTED_ROUTER_SCHEMA_VERSION,
)
from .hosted_router_controller_v2 import (
    HostedRouterControllerV2, HostedRoutingDecisionV2,
    HostedRoutingRequestV2,
)
from .tiny_substrate_v3 import _sha256_hex


W70_HOSTED_ROUTER_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v3.v1")


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV3:
    inner_v2: HostedRoutingRequestV2
    visible_token_budget: int = 256
    baseline_token_cost: int = 512
    repair_dominance_label: int = 0
    weight_budget_efficiency: float = 1.0
    weight_repair_dominance_match: float = 0.5

    @property
    def request_cid(self) -> str:
        return self.inner_v2.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_HOSTED_ROUTER_V3_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "visible_token_budget": int(
                self.visible_token_budget),
            "baseline_token_cost": int(self.baseline_token_cost),
            "repair_dominance_label": int(
                self.repair_dominance_label),
            "weight_budget_efficiency": float(round(
                self.weight_budget_efficiency, 12)),
            "weight_repair_dominance_match": float(round(
                self.weight_repair_dominance_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV3:
    schema: str
    inner_v2: HostedRoutingDecisionV2
    budget_efficiency_score: float
    repair_dominance_match_score: float
    per_budget_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v2.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v2_cid": str(self.inner_v2.cid()),
            "budget_efficiency_score": float(round(
                self.budget_efficiency_score, 12)),
            "repair_dominance_match_score": float(round(
                self.repair_dominance_match_score, 12)),
            "per_budget_routing_cid": str(
                self.per_budget_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v3",
            "decision": self.to_dict()})


@dataclasses.dataclass
class HostedRouterControllerV3:
    inner_v2: HostedRouterControllerV2
    repair_dominance_match_table: dict[int, str] = (
        dataclasses.field(default_factory=lambda: {
            # 0 no_repair → text_only; 1..6 → tier preferring caches.
            0: "openai_paid",
            1: "openrouter_paid",
            2: "openrouter_paid",
            3: "openai_paid",
            4: "openrouter_paid",
            5: "openai_paid",
            6: "openrouter_paid",
        }))
    audit_v3: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
    ) -> "HostedRouterControllerV3":
        inner_v2 = HostedRouterControllerV2.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(inner_v2=inner_v2)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_HOSTED_ROUTER_V3_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "repair_dominance_match_table": {
                str(k): str(v)
                for k, v in sorted(
                    self
                    .repair_dominance_match_table.items())},
        })

    def _budget_efficiency_score(
            self, p: HostedProvider | None,
            req_v3: HostedRoutingRequestV3,
    ) -> float:
        if p is None:
            return 0.0
        budget_ratio = (
            float(req_v3.visible_token_budget)
            / float(max(1, req_v3.baseline_token_cost)))
        cost_per_1k = float(p.cost_per_1k_output)
        return float(
            float(req_v3.weight_budget_efficiency)
            * (1.0 / (1.0 + cost_per_1k))
            * float(max(0.05, min(1.0, budget_ratio))))

    def _repair_dominance_match_score(
            self, p: HostedProvider | None,
            req_v3: HostedRoutingRequestV3,
    ) -> float:
        if p is None:
            return 0.0
        target = self.repair_dominance_match_table.get(
            int(req_v3.repair_dominance_label), "")
        if str(p.name) == str(target):
            return float(req_v3.weight_repair_dominance_match)
        return 0.0

    def decide_v3(
            self, req_v3: HostedRoutingRequestV3,
    ) -> HostedRoutingDecisionV3:
        d2 = self.inner_v2.decide_v2(req_v3.inner_v2)
        winner = (
            d2.chosen_provider
            if d2.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr in self
                 .inner_v2.inner_v1.registry.providers
                 if pr.name == winner), None)
        be = float(self._budget_efficiency_score(p, req_v3))
        rd = float(self._repair_dominance_match_score(p, req_v3))
        per_budget_cid = _sha256_hex({
            "schema": W70_HOSTED_ROUTER_V3_SCHEMA_VERSION,
            "kind": "per_budget_routing_cid",
            "request_v3_cid": str(req_v3.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v3.visible_token_budget),
            "baseline_token_cost": int(
                req_v3.baseline_token_cost),
            "repair_dominance_label": int(
                req_v3.repair_dominance_label),
        })
        self.audit_v3.append({
            "request_cid": str(req_v3.request_cid),
            "winner": str(winner or ""),
            "budget_efficiency_score": float(be),
            "repair_dominance_match_score": float(rd),
            "per_budget_routing_cid": str(per_budget_cid),
        })
        return HostedRoutingDecisionV3(
            schema=W70_HOSTED_ROUTER_V3_SCHEMA_VERSION,
            inner_v2=d2,
            budget_efficiency_score=float(be),
            repair_dominance_match_score=float(rd),
            per_budget_routing_cid=str(per_budget_cid),
        )

    def routing_cost_per_team_success_under_budget(
            self) -> float:
        """Aggregate routing-cost-per-team-success-under-budget."""
        n = float(len(self.audit_v3))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v3:
            be = float(entry.get("budget_efficiency_score", 0.0))
            rd = float(
                entry.get("repair_dominance_match_score", 0.0))
            total += (
                1.0 - 0.5 * float(be) - 0.5 * float(rd))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV3Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_team_success_under_budget: float
    inner_v2_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_team_success_under_budget": float(
                round(
                    self
                    .routing_cost_per_team_success_under_budget,
                    12)),
            "inner_v2_witness_cid": str(self.inner_v2_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v3_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v3_witness(
        controller: HostedRouterControllerV3,
) -> HostedRouterControllerV3Witness:
    from .hosted_router_controller_v2 import (
        emit_hosted_router_controller_v2_witness,
    )
    inner_w = emit_hosted_router_controller_v2_witness(
        controller.inner_v2)
    return HostedRouterControllerV3Witness(
        schema=W70_HOSTED_ROUTER_V3_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v3)),
        routing_cost_per_team_success_under_budget=float(
            controller
            .routing_cost_per_team_success_under_budget()),
        inner_v2_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W70_HOSTED_ROUTER_V3_SCHEMA_VERSION",
    "HostedRoutingRequestV3",
    "HostedRoutingDecisionV3",
    "HostedRouterControllerV3",
    "HostedRouterControllerV3Witness",
    "emit_hosted_router_controller_v3_witness",
]
