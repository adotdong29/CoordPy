"""W72 H1 — Hosted Router Controller V5 (Plane A).

Strictly extends W71's
``coordpy.hosted_router_controller_v4``. V5 adds:

* **Rejoin-pressure weighting** — adds a new weight
  ``weight_rejoin_pressure`` and folds caller-declared rejoin
  pressure into the routing score.
* **Delayed-rejoin match table** — V4 had a per-(label, restart)
  delayed-repair match table; V5 adds a per-(label, restart,
  rejoin) table so that a high rejoin pressure can swing the
  match to a different provider tier.
* **Per-budget+restart+rejoin routing CID** — exposes a content-
  addressed routing CID that depends on the declared visible-token
  budget AND the declared restart pressure AND the declared rejoin
  pressure.

Honest scope (W72 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Rejoin pressure is caller-declared (no live measurement).
  ``W72-L-HOSTED-V5-NO-SUBSTRATE-CAP``,
  ``W72-L-HOSTED-V5-REJOIN-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v4 import (
    HostedRouterControllerV4, HostedRoutingDecisionV4,
    HostedRoutingRequestV4,
)
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_ROUTER_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v5.v1")
W72_DEFAULT_REJOIN_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV5:
    inner_v4: HostedRoutingRequestV4
    rejoin_pressure: float = 0.0
    weight_rejoin_pressure: float = 0.6
    weight_delayed_rejoin_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v4.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v4.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v4.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v4.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v4.restart_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_HOSTED_ROUTER_V5_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "rejoin_pressure": float(round(
                self.rejoin_pressure, 12)),
            "weight_rejoin_pressure": float(round(
                self.weight_rejoin_pressure, 12)),
            "weight_delayed_rejoin_match": float(round(
                self.weight_delayed_rejoin_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV5:
    schema: str
    inner_v4: HostedRoutingDecisionV4
    rejoin_pressure_score: float
    delayed_rejoin_match_score: float
    per_budget_restart_rejoin_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v4.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "rejoin_pressure_score": float(round(
                self.rejoin_pressure_score, 12)),
            "delayed_rejoin_match_score": float(round(
                self.delayed_rejoin_match_score, 12)),
            "per_budget_restart_rejoin_routing_cid": str(
                self.per_budget_restart_rejoin_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v5",
            "decision": self.to_dict()})


def _default_delayed_rejoin_match_table() -> dict[
        tuple[int, bool, bool], str]:
    out: dict[tuple[int, bool, bool], str] = {}
    # Default per-label provider with neither restart nor rejoin
    # pressure crossed.
    for label in range(8):
        out[(int(label), False, False)] = (
            "openai_paid" if label % 2 == 0 else "openrouter_paid")
    # When only restart pressure crosses, prefer paid-OpenAI tier
    # (mirrors V4 behaviour).
    for label in range(8):
        out[(int(label), True, False)] = "openai_paid"
    # When only rejoin pressure crosses, prefer paid-OpenAI tier
    # (rejoin recovery favours higher trust).
    for label in range(8):
        out[(int(label), False, True)] = "openai_paid"
    # When both restart AND rejoin pressure cross, paid-OpenAI tier.
    for label in range(8):
        out[(int(label), True, True)] = "openai_paid"
    return out


@dataclasses.dataclass
class HostedRouterControllerV5:
    inner_v4: HostedRouterControllerV4
    delayed_rejoin_match_table: dict[
        tuple[int, bool, bool], str] = dataclasses.field(
        default_factory=_default_delayed_rejoin_match_table)
    rejoin_pressure_floor: float = (
        W72_DEFAULT_REJOIN_PRESSURE_FLOOR)
    audit_v5: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            rejoin_pressure_floor: float = (
                W72_DEFAULT_REJOIN_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV5":
        inner_v4 = HostedRouterControllerV4.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v4=inner_v4,
            rejoin_pressure_floor=float(
                rejoin_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_HOSTED_ROUTER_V5_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "delayed_rejoin_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]), str(v)]
                for k, v in sorted(
                    self.delayed_rejoin_match_table.items())],
            "rejoin_pressure_floor": float(round(
                self.rejoin_pressure_floor, 12)),
        })

    def _rejoin_pressure_score(
            self, p: HostedProvider | None,
            req_v5: HostedRoutingRequestV5,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0, float(req_v5.rejoin_pressure))))
        # paid-OpenAI tier preferred at high rejoin pressure.
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(req_v5.weight_rejoin_pressure)
            * pressure * provider_lift)

    def _delayed_rejoin_match_score(
            self, p: HostedProvider | None,
            req_v5: HostedRoutingRequestV5,
    ) -> float:
        if p is None:
            return 0.0
        restart_above = bool(
            float(req_v5.restart_pressure)
            >= float(
                self.inner_v4.restart_pressure_floor))
        rejoin_above = bool(
            float(req_v5.rejoin_pressure)
            >= float(self.rejoin_pressure_floor))
        key = (
            int(req_v5.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above))
        target = self.delayed_rejoin_match_table.get(key, "")
        if str(p.name) == str(target):
            return float(req_v5.weight_delayed_rejoin_match)
        return 0.0

    def decide_v5(
            self, req_v5: HostedRoutingRequestV5,
    ) -> HostedRoutingDecisionV5:
        d4 = self.inner_v4.decide_v4(req_v5.inner_v4)
        winner = (
            d4.chosen_provider
            if d4.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr in self.inner_v4.inner_v3.inner_v2
                 .inner_v1.registry.providers
                 if pr.name == winner), None)
        rj = float(self._rejoin_pressure_score(p, req_v5))
        dr = float(self._delayed_rejoin_match_score(p, req_v5))
        per_routing_cid = _sha256_hex({
            "schema": W72_HOSTED_ROUTER_V5_SCHEMA_VERSION,
            "kind": "per_budget_restart_rejoin_routing_cid",
            "request_v5_cid": str(req_v5.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v5.visible_token_budget),
            "baseline_token_cost": int(
                req_v5.baseline_token_cost),
            "repair_dominance_label": int(
                req_v5.repair_dominance_label),
            "restart_pressure": float(round(
                req_v5.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v5.rejoin_pressure, 12)),
        })
        self.audit_v5.append({
            "request_cid": str(req_v5.request_cid),
            "winner": str(winner or ""),
            "rejoin_pressure_score": float(rj),
            "delayed_rejoin_match_score": float(dr),
            "per_budget_restart_rejoin_routing_cid": str(
                per_routing_cid),
        })
        return HostedRoutingDecisionV5(
            schema=W72_HOSTED_ROUTER_V5_SCHEMA_VERSION,
            inner_v4=d4,
            rejoin_pressure_score=float(rj),
            delayed_rejoin_match_score=float(dr),
            per_budget_restart_rejoin_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_rejoin_success_under_budget(
            self) -> float:
        """Aggregate cost-per-rejoin-success-under-budget for V5."""
        n = float(len(self.audit_v5))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v5:
            rj = float(entry.get(
                "rejoin_pressure_score", 0.0))
            dr = float(entry.get(
                "delayed_rejoin_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(rj) - 0.6 * float(dr))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV5Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_rejoin_success_under_budget: float
    inner_v4_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_rejoin_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_rejoin_success_under_budget,
                    12)),
            "inner_v4_witness_cid": str(
                self.inner_v4_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v5_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v5_witness(
        controller: HostedRouterControllerV5,
) -> HostedRouterControllerV5Witness:
    from .hosted_router_controller_v4 import (
        emit_hosted_router_controller_v4_witness,
    )
    inner_w = emit_hosted_router_controller_v4_witness(
        controller.inner_v4)
    return HostedRouterControllerV5Witness(
        schema=W72_HOSTED_ROUTER_V5_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v5)),
        routing_cost_per_rejoin_success_under_budget=float(
            controller
            .routing_cost_per_rejoin_success_under_budget()),
        inner_v4_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W72_HOSTED_ROUTER_V5_SCHEMA_VERSION",
    "W72_DEFAULT_REJOIN_PRESSURE_FLOOR",
    "HostedRoutingRequestV5",
    "HostedRoutingDecisionV5",
    "HostedRouterControllerV5",
    "HostedRouterControllerV5Witness",
    "emit_hosted_router_controller_v5_witness",
]
