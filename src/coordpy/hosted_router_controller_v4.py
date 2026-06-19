"""W71 H1 — Hosted Router Controller V4 (Plane A).

Strictly extends W70's
``coordpy.hosted_router_controller_v3``. V4 adds:

* **Restart-pressure weighting** — adds a new weight
  ``weight_restart_pressure`` and folds caller-declared restart
  pressure into the routing score.
* **Delayed-repair match table** — V3 had a per-label
  repair-dominance match table; V4 adds a per-(label, restart)
  table so that a high restart pressure can swing the match to a
  different provider tier.
* **Per-budget+restart routing CID** — exposes a content-addressed
  routing CID that depends on both the declared visible-token
  budget AND the declared restart pressure.

Honest scope (W71 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Restart pressure is caller-declared (no live measurement).
  ``W71-L-HOSTED-V4-NO-SUBSTRATE-CAP``,
  ``W71-L-HOSTED-V4-RESTART-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
    HostedRoutingRequest,
)
from .hosted_router_controller_v2 import (
    HostedRoutingRequestV2,
)
from .hosted_router_controller_v3 import (
    HostedRouterControllerV3, HostedRoutingDecisionV3,
    HostedRoutingRequestV3,
)
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_ROUTER_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v4.v1")
W71_DEFAULT_RESTART_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV4:
    inner_v3: HostedRoutingRequestV3
    restart_pressure: float = 0.0
    weight_restart_pressure: float = 0.6
    weight_delayed_repair_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v3.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v3.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v3.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v3.repair_dominance_label)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_HOSTED_ROUTER_V4_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "restart_pressure": float(round(
                self.restart_pressure, 12)),
            "weight_restart_pressure": float(round(
                self.weight_restart_pressure, 12)),
            "weight_delayed_repair_match": float(round(
                self.weight_delayed_repair_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV4:
    schema: str
    inner_v3: HostedRoutingDecisionV3
    restart_pressure_score: float
    delayed_repair_match_score: float
    per_budget_restart_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v3.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v3_cid": str(self.inner_v3.cid()),
            "restart_pressure_score": float(round(
                self.restart_pressure_score, 12)),
            "delayed_repair_match_score": float(round(
                self.delayed_repair_match_score, 12)),
            "per_budget_restart_routing_cid": str(
                self.per_budget_restart_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v4",
            "decision": self.to_dict()})


@dataclasses.dataclass
class HostedRouterControllerV4:
    inner_v3: HostedRouterControllerV3
    # Match (repair_dominance_label, restart_pressure_floor_crossed)
    # → provider tier. The W70 table picks providers per repair
    # label; V4 also picks differently when restart pressure is high.
    delayed_repair_match_table: dict[
        tuple[int, bool], str] = dataclasses.field(
        default_factory=lambda: {
            (0, False): "openai_paid",
            (1, False): "openrouter_paid",
            (2, False): "openrouter_paid",
            (3, False): "openai_paid",
            (4, False): "openrouter_paid",
            (5, False): "openai_paid",
            (6, False): "openrouter_paid",
            # When restart pressure crosses the floor we prefer
            # the paid-OpenAI tier to absorb the recovery cost.
            (0, True): "openai_paid",
            (1, True): "openai_paid",
            (2, True): "openai_paid",
            (3, True): "openai_paid",
            (4, True): "openai_paid",
            (5, True): "openai_paid",
            (6, True): "openai_paid",
        })
    restart_pressure_floor: float = (
        W71_DEFAULT_RESTART_PRESSURE_FLOOR)
    audit_v4: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            restart_pressure_floor: float = (
                W71_DEFAULT_RESTART_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV4":
        inner_v3 = HostedRouterControllerV3.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v3=inner_v3,
            restart_pressure_floor=float(
                restart_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_HOSTED_ROUTER_V4_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "delayed_repair_match_table": [
                [int(k[0]), bool(k[1]), str(v)]
                for k, v in sorted(
                    self.delayed_repair_match_table.items())],
            "restart_pressure_floor": float(round(
                self.restart_pressure_floor, 12)),
        })

    def _restart_pressure_score(
            self, p: HostedProvider | None,
            req_v4: HostedRoutingRequestV4,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0, float(req_v4.restart_pressure))))
        # paid-OpenAI tier preferred at high pressure.
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(req_v4.weight_restart_pressure)
            * pressure * provider_lift)

    def _delayed_repair_match_score(
            self, p: HostedProvider | None,
            req_v4: HostedRoutingRequestV4,
    ) -> float:
        if p is None:
            return 0.0
        above = bool(
            float(req_v4.restart_pressure)
            >= float(self.restart_pressure_floor))
        key = (int(req_v4.repair_dominance_label), bool(above))
        target = self.delayed_repair_match_table.get(key, "")
        if str(p.name) == str(target):
            return float(req_v4.weight_delayed_repair_match)
        return 0.0

    def decide_v4(
            self, req_v4: HostedRoutingRequestV4,
    ) -> HostedRoutingDecisionV4:
        d3 = self.inner_v3.decide_v3(req_v4.inner_v3)
        winner = (
            d3.chosen_provider
            if d3.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr in self.inner_v3
                 .inner_v2.inner_v1.registry.providers
                 if pr.name == winner), None)
        rp = float(self._restart_pressure_score(p, req_v4))
        dr = float(self._delayed_repair_match_score(p, req_v4))
        per_routing_cid = _sha256_hex({
            "schema": W71_HOSTED_ROUTER_V4_SCHEMA_VERSION,
            "kind": "per_budget_restart_routing_cid",
            "request_v4_cid": str(req_v4.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v4.visible_token_budget),
            "baseline_token_cost": int(
                req_v4.baseline_token_cost),
            "repair_dominance_label": int(
                req_v4.repair_dominance_label),
            "restart_pressure": float(round(
                req_v4.restart_pressure, 12)),
        })
        self.audit_v4.append({
            "request_cid": str(req_v4.request_cid),
            "winner": str(winner or ""),
            "restart_pressure_score": float(rp),
            "delayed_repair_match_score": float(dr),
            "per_budget_restart_routing_cid": str(
                per_routing_cid),
        })
        return HostedRoutingDecisionV4(
            schema=W71_HOSTED_ROUTER_V4_SCHEMA_VERSION,
            inner_v3=d3,
            restart_pressure_score=float(rp),
            delayed_repair_match_score=float(dr),
            per_budget_restart_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_repair_success_under_budget(
            self) -> float:
        """Aggregate cost-per-repair-success-under-budget for V4."""
        n = float(len(self.audit_v4))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v4:
            rp = float(entry.get(
                "restart_pressure_score", 0.0))
            dr = float(entry.get(
                "delayed_repair_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(rp) - 0.6 * float(dr))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV4Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_repair_success_under_budget: float
    inner_v3_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_repair_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_repair_success_under_budget,
                    12)),
            "inner_v3_witness_cid": str(
                self.inner_v3_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v4_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v4_witness(
        controller: HostedRouterControllerV4,
) -> HostedRouterControllerV4Witness:
    from .hosted_router_controller_v3 import (
        emit_hosted_router_controller_v3_witness,
    )
    inner_w = emit_hosted_router_controller_v3_witness(
        controller.inner_v3)
    return HostedRouterControllerV4Witness(
        schema=W71_HOSTED_ROUTER_V4_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v4)),
        routing_cost_per_repair_success_under_budget=float(
            controller
            .routing_cost_per_repair_success_under_budget()),
        inner_v3_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W71_HOSTED_ROUTER_V4_SCHEMA_VERSION",
    "W71_DEFAULT_RESTART_PRESSURE_FLOOR",
    "HostedRoutingRequestV4",
    "HostedRoutingDecisionV4",
    "HostedRouterControllerV4",
    "HostedRouterControllerV4Witness",
    "emit_hosted_router_controller_v4_witness",
]
