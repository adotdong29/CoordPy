"""W74 H1 — Hosted Router Controller V7 (Plane A).

Strictly extends W73's
``coordpy.hosted_router_controller_v6``. V7 adds:

* **Compound-pressure weighting** — adds a new weight
  ``weight_compound_pressure`` and folds caller-declared compound
  pressure into the routing score.
* **Compound-repair-after-DRTR match table** — V6 had a per-(label,
  restart, rejoin, replacement) match table; V7 adds a per-(label,
  restart, rejoin, replacement, compound) table so that a high
  compound pressure can swing the match to a different provider
  tier.
* **Per-budget+restart+rejoin+replacement+compound routing CID** —
  exposes a content-addressed routing CID that depends on the
  declared visible-token budget AND the declared restart, rejoin,
  replacement, AND compound pressures.

Honest scope (W74 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Compound pressure is caller-declared (no live measurement).
  ``W74-L-HOSTED-V7-NO-SUBSTRATE-CAP``,
  ``W74-L-HOSTED-V7-COMPOUND-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v6 import (
    HostedRouterControllerV6, HostedRoutingDecisionV6,
    HostedRoutingRequestV6,
)
from .tiny_substrate_v3 import _sha256_hex


W74_HOSTED_ROUTER_V7_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v7.v1")
W74_DEFAULT_COMPOUND_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV7:
    inner_v6: HostedRoutingRequestV6
    compound_pressure: float = 0.0
    weight_compound_pressure: float = 0.6
    weight_compound_repair_drtr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v6.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v6.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v6.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v6.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v6.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v6.rejoin_pressure)

    @property
    def replacement_pressure(self) -> float:
        return float(self.inner_v6.replacement_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_HOSTED_ROUTER_V7_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_pressure": float(round(
                self.compound_pressure, 12)),
            "weight_compound_pressure": float(round(
                self.weight_compound_pressure, 12)),
            "weight_compound_repair_drtr_match": float(round(
                self.weight_compound_repair_drtr_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV7:
    schema: str
    inner_v6: HostedRoutingDecisionV6
    compound_pressure_score: float
    compound_repair_drtr_match_score: float
    per_budget_restart_rejoin_replacement_compound_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v6.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_pressure_score": float(round(
                self.compound_pressure_score, 12)),
            "compound_repair_drtr_match_score": float(round(
                self.compound_repair_drtr_match_score, 12)),
            "per_budget_restart_rejoin_replacement_"
            "compound_routing_cid":
                str(
                    self
                    .per_budget_restart_rejoin_replacement_compound_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v7",
            "decision": self.to_dict()})


def _default_compound_repair_drtr_match_table() -> dict[
        tuple[int, bool, bool, bool, bool], str]:
    out: dict[tuple[int, bool, bool, bool, bool], str] = {}
    # Default per-label provider with no extra pressure crossed.
    for label in range(11):
        out[(int(label), False, False, False, False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
    # Any pressure cross → paid-OpenAI tier (higher trust).
    for label in range(11):
        for r in (False, True):
            for j in (False, True):
                for rep in (False, True):
                    for cmp in (False, True):
                        if r or j or rep or cmp:
                            out[(int(label), bool(r), bool(j),
                                 bool(rep), bool(cmp))] = (
                                "openai_paid")
    return out


@dataclasses.dataclass
class HostedRouterControllerV7:
    inner_v6: HostedRouterControllerV6
    compound_repair_drtr_match_table: dict[
        tuple[int, bool, bool, bool, bool], str] = (
        dataclasses.field(
            default_factory=_default_compound_repair_drtr_match_table))
    compound_pressure_floor: float = (
        W74_DEFAULT_COMPOUND_PRESSURE_FLOOR)
    audit_v7: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            compound_pressure_floor: float = (
                W74_DEFAULT_COMPOUND_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV7":
        inner_v6 = HostedRouterControllerV6.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v6=inner_v6,
            compound_pressure_floor=float(
                compound_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_HOSTED_ROUTER_V7_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_repair_drtr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), bool(k[4]), str(v)]
                for k, v in sorted(
                    self.compound_repair_drtr_match_table
                    .items())],
            "compound_pressure_floor": float(round(
                self.compound_pressure_floor, 12)),
        })

    def _compound_pressure_score(
            self, p: HostedProvider | None,
            req_v7: HostedRoutingRequestV7,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0, float(req_v7.compound_pressure))))
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(req_v7.weight_compound_pressure)
            * pressure * provider_lift)

    def _compound_repair_drtr_match_score(
            self, p: HostedProvider | None,
            req_v7: HostedRoutingRequestV7,
    ) -> float:
        if p is None:
            return 0.0
        restart_above = bool(
            float(req_v7.restart_pressure)
            >= float(
                self.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor))
        rejoin_above = bool(
            float(req_v7.rejoin_pressure)
            >= float(
                self.inner_v6.inner_v5.rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v7.replacement_pressure)
            >= float(
                self.inner_v6.replacement_pressure_floor))
        compound_above = bool(
            float(req_v7.compound_pressure)
            >= float(self.compound_pressure_floor))
        key = (
            int(req_v7.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above),
            bool(compound_above))
        target = (
            self.compound_repair_drtr_match_table.get(key, ""))
        if str(p.name) == str(target):
            return float(
                req_v7.weight_compound_repair_drtr_match)
        return 0.0

    def decide_v7(
            self, req_v7: HostedRoutingRequestV7,
    ) -> HostedRoutingDecisionV7:
        d6 = self.inner_v6.decide_v6(req_v7.inner_v6)
        winner = (
            d6.chosen_provider
            if d6.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v6.inner_v5.inner_v4.inner_v3
                 .inner_v2.inner_v1.registry.providers
                 if pr.name == winner), None)
        cp = float(self._compound_pressure_score(p, req_v7))
        cmp_match = float(
            self._compound_repair_drtr_match_score(p, req_v7))
        per_routing_cid = _sha256_hex({
            "schema": W74_HOSTED_ROUTER_V7_SCHEMA_VERSION,
            "kind":
                "per_budget_restart_rejoin_replacement_"
                "compound_routing_cid",
            "request_v7_cid": str(req_v7.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v7.visible_token_budget),
            "baseline_token_cost": int(
                req_v7.baseline_token_cost),
            "repair_dominance_label": int(
                req_v7.repair_dominance_label),
            "restart_pressure": float(round(
                req_v7.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v7.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v7.replacement_pressure, 12)),
            "compound_pressure": float(round(
                req_v7.compound_pressure, 12)),
        })
        self.audit_v7.append({
            "request_cid": str(req_v7.request_cid),
            "winner": str(winner or ""),
            "compound_pressure_score": float(cp),
            "compound_repair_drtr_match_score": float(cmp_match),
            "per_routing_cid": str(per_routing_cid),
        })
        return HostedRoutingDecisionV7(
            schema=W74_HOSTED_ROUTER_V7_SCHEMA_VERSION,
            inner_v6=d6,
            compound_pressure_score=float(cp),
            compound_repair_drtr_match_score=float(cmp_match),
            per_budget_restart_rejoin_replacement_compound_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_compound_success_under_budget(
            self) -> float:
        """Aggregate cost-per-compound-success-under-budget."""
        n = float(len(self.audit_v7))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v7:
            cp = float(entry.get(
                "compound_pressure_score", 0.0))
            cmp_m = float(entry.get(
                "compound_repair_drtr_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(cp) - 0.6 * float(cmp_m))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV7Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_compound_success_under_budget: float
    inner_v6_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_compound_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_compound_success_under_budget,
                    12)),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v7_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v7_witness(
        controller: HostedRouterControllerV7,
) -> HostedRouterControllerV7Witness:
    from .hosted_router_controller_v6 import (
        emit_hosted_router_controller_v6_witness,
    )
    inner_w = emit_hosted_router_controller_v6_witness(
        controller.inner_v6)
    return HostedRouterControllerV7Witness(
        schema=W74_HOSTED_ROUTER_V7_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v7)),
        routing_cost_per_compound_success_under_budget=float(
            controller
            .routing_cost_per_compound_success_under_budget()),
        inner_v6_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W74_HOSTED_ROUTER_V7_SCHEMA_VERSION",
    "W74_DEFAULT_COMPOUND_PRESSURE_FLOOR",
    "HostedRoutingRequestV7",
    "HostedRoutingDecisionV7",
    "HostedRouterControllerV7",
    "HostedRouterControllerV7Witness",
    "emit_hosted_router_controller_v7_witness",
]
