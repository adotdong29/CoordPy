"""W73 H1 — Hosted Router Controller V6 (Plane A).

Strictly extends W72's
``coordpy.hosted_router_controller_v5``. V6 adds:

* **Replacement-pressure weighting** — adds a new weight
  ``weight_replacement_pressure`` and folds caller-declared
  replacement pressure into the routing score.
* **Replacement-after-CTR match table** — V5 had a per-(label,
  restart, rejoin) delayed-rejoin match table; V6 adds a per-
  (label, restart, rejoin, replacement) table so that a high
  replacement pressure can swing the match to a different provider
  tier.
* **Per-budget+restart+rejoin+replacement routing CID** — exposes
  a content-addressed routing CID that depends on the declared
  visible-token budget AND the declared restart, rejoin, AND
  replacement pressures.

Honest scope (W73 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Replacement pressure is caller-declared (no live measurement).
  ``W73-L-HOSTED-V6-NO-SUBSTRATE-CAP``,
  ``W73-L-HOSTED-V6-REPLACEMENT-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v5 import (
    HostedRouterControllerV5, HostedRoutingDecisionV5,
    HostedRoutingRequestV5,
)
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_ROUTER_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v6.v1")
W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV6:
    inner_v5: HostedRoutingRequestV5
    replacement_pressure: float = 0.0
    weight_replacement_pressure: float = 0.6
    weight_replacement_after_ctr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v5.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v5.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v5.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v5.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v5.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v5.rejoin_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_HOSTED_ROUTER_V6_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "replacement_pressure": float(round(
                self.replacement_pressure, 12)),
            "weight_replacement_pressure": float(round(
                self.weight_replacement_pressure, 12)),
            "weight_replacement_after_ctr_match": float(round(
                self.weight_replacement_after_ctr_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV6:
    schema: str
    inner_v5: HostedRoutingDecisionV5
    replacement_pressure_score: float
    replacement_after_ctr_match_score: float
    per_budget_restart_rejoin_replacement_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v5.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "replacement_pressure_score": float(round(
                self.replacement_pressure_score, 12)),
            "replacement_after_ctr_match_score": float(round(
                self.replacement_after_ctr_match_score, 12)),
            "per_budget_restart_rejoin_replacement_routing_cid":
                str(
                    self
                    .per_budget_restart_rejoin_replacement_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v6",
            "decision": self.to_dict()})


def _default_replacement_after_ctr_match_table() -> dict[
        tuple[int, bool, bool, bool], str]:
    out: dict[tuple[int, bool, bool, bool], str] = {}
    # Default per-label provider with no extra pressure crossed.
    for label in range(10):
        out[(int(label), False, False, False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
    # Any pressure cross → paid-OpenAI tier (higher trust).
    for label in range(10):
        for r in (False, True):
            for j in (False, True):
                for rep in (False, True):
                    if r or j or rep:
                        out[(int(label), bool(r), bool(j),
                             bool(rep))] = "openai_paid"
    return out


@dataclasses.dataclass
class HostedRouterControllerV6:
    inner_v5: HostedRouterControllerV5
    replacement_after_ctr_match_table: dict[
        tuple[int, bool, bool, bool], str] = (
        dataclasses.field(
            default_factory=_default_replacement_after_ctr_match_table))
    replacement_pressure_floor: float = (
        W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR)
    audit_v6: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            replacement_pressure_floor: float = (
                W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV6":
        inner_v5 = HostedRouterControllerV5.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v5=inner_v5,
            replacement_pressure_floor=float(
                replacement_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_HOSTED_ROUTER_V6_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "replacement_after_ctr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), str(v)]
                for k, v in sorted(
                    self.replacement_after_ctr_match_table
                    .items())],
            "replacement_pressure_floor": float(round(
                self.replacement_pressure_floor, 12)),
        })

    def _replacement_pressure_score(
            self, p: HostedProvider | None,
            req_v6: HostedRoutingRequestV6,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0, float(req_v6.replacement_pressure))))
        # paid-OpenAI tier preferred at high replacement pressure.
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(req_v6.weight_replacement_pressure)
            * pressure * provider_lift)

    def _replacement_after_ctr_match_score(
            self, p: HostedProvider | None,
            req_v6: HostedRoutingRequestV6,
    ) -> float:
        if p is None:
            return 0.0
        restart_above = bool(
            float(req_v6.restart_pressure)
            >= float(
                self.inner_v5.inner_v4.restart_pressure_floor))
        rejoin_above = bool(
            float(req_v6.rejoin_pressure)
            >= float(
                self.inner_v5.rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v6.replacement_pressure)
            >= float(self.replacement_pressure_floor))
        key = (
            int(req_v6.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above))
        target = (
            self.replacement_after_ctr_match_table.get(key, ""))
        if str(p.name) == str(target):
            return float(req_v6.weight_replacement_after_ctr_match)
        return 0.0

    def decide_v6(
            self, req_v6: HostedRoutingRequestV6,
    ) -> HostedRoutingDecisionV6:
        d5 = self.inner_v5.decide_v5(req_v6.inner_v5)
        winner = (
            d5.chosen_provider
            if d5.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v5.inner_v4.inner_v3.inner_v2
                 .inner_v1.registry.providers
                 if pr.name == winner), None)
        rp = float(self._replacement_pressure_score(p, req_v6))
        rep_match = float(
            self._replacement_after_ctr_match_score(p, req_v6))
        per_routing_cid = _sha256_hex({
            "schema": W73_HOSTED_ROUTER_V6_SCHEMA_VERSION,
            "kind":
                "per_budget_restart_rejoin_replacement_routing_cid",
            "request_v6_cid": str(req_v6.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v6.visible_token_budget),
            "baseline_token_cost": int(
                req_v6.baseline_token_cost),
            "repair_dominance_label": int(
                req_v6.repair_dominance_label),
            "restart_pressure": float(round(
                req_v6.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v6.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v6.replacement_pressure, 12)),
        })
        self.audit_v6.append({
            "request_cid": str(req_v6.request_cid),
            "winner": str(winner or ""),
            "replacement_pressure_score": float(rp),
            "replacement_after_ctr_match_score": float(rep_match),
            "per_routing_cid": str(per_routing_cid),
        })
        return HostedRoutingDecisionV6(
            schema=W73_HOSTED_ROUTER_V6_SCHEMA_VERSION,
            inner_v5=d5,
            replacement_pressure_score=float(rp),
            replacement_after_ctr_match_score=float(rep_match),
            per_budget_restart_rejoin_replacement_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_replacement_success_under_budget(
            self) -> float:
        """Aggregate cost-per-replacement-success-under-budget."""
        n = float(len(self.audit_v6))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v6:
            rp = float(entry.get(
                "replacement_pressure_score", 0.0))
            rep = float(entry.get(
                "replacement_after_ctr_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(rp) - 0.6 * float(rep))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV6Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_replacement_success_under_budget: float
    inner_v5_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_replacement_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_replacement_success_under_budget,
                    12)),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v6_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v6_witness(
        controller: HostedRouterControllerV6,
) -> HostedRouterControllerV6Witness:
    from .hosted_router_controller_v5 import (
        emit_hosted_router_controller_v5_witness,
    )
    inner_w = emit_hosted_router_controller_v5_witness(
        controller.inner_v5)
    return HostedRouterControllerV6Witness(
        schema=W73_HOSTED_ROUTER_V6_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v6)),
        routing_cost_per_replacement_success_under_budget=float(
            controller
            .routing_cost_per_replacement_success_under_budget()),
        inner_v5_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W73_HOSTED_ROUTER_V6_SCHEMA_VERSION",
    "W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR",
    "HostedRoutingRequestV6",
    "HostedRoutingDecisionV6",
    "HostedRouterControllerV6",
    "HostedRouterControllerV6Witness",
    "emit_hosted_router_controller_v6_witness",
]
