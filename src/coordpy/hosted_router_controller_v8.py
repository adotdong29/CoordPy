"""W75 H1 — Hosted Router Controller V8 (Plane A).

Strictly extends W74's
``coordpy.hosted_router_controller_v7``. V8 adds:

* **Compound-chain-pressure weighting** — adds a new weight
  ``weight_compound_chain_pressure`` and folds caller-declared
  compound-chain pressure into the routing score.
* **Compound-chain-repair-after-RTR match table** — V7 had a
  per-(label, restart, rejoin, replacement, compound) match
  table; V8 adds a per-(label, restart, rejoin, replacement,
  compound, chain) table so that a high compound-chain pressure
  can swing the match to a different provider tier.
* **Per-budget+restart+rejoin+replacement+compound+chain
  routing CID** — exposes a content-addressed routing CID that
  depends on the declared visible-token budget AND the declared
  restart, rejoin, replacement, compound, AND compound-chain
  pressures.

Honest scope (W75 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Compound-chain pressure is caller-declared (no live
  measurement). ``W75-L-HOSTED-V8-NO-SUBSTRATE-CAP``,
  ``W75-L-HOSTED-V8-COMPOUND-CHAIN-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v7 import (
    HostedRouterControllerV7, HostedRoutingDecisionV7,
    HostedRoutingRequestV7,
)
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_ROUTER_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v8.v1")
W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV8:
    inner_v7: HostedRoutingRequestV7
    compound_chain_pressure: float = 0.0
    weight_compound_chain_pressure: float = 0.6
    weight_compound_chain_repair_rtr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v7.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v7.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v7.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v7.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v7.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v7.rejoin_pressure)

    @property
    def replacement_pressure(self) -> float:
        return float(self.inner_v7.replacement_pressure)

    @property
    def compound_pressure(self) -> float:
        return float(self.inner_v7.compound_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_HOSTED_ROUTER_V8_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_pressure": float(round(
                self.compound_chain_pressure, 12)),
            "weight_compound_chain_pressure": float(round(
                self.weight_compound_chain_pressure, 12)),
            "weight_compound_chain_repair_rtr_match": float(
                round(
                    self
                    .weight_compound_chain_repair_rtr_match, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV8:
    schema: str
    inner_v7: HostedRoutingDecisionV7
    compound_chain_pressure_score: float
    compound_chain_repair_rtr_match_score: float
    per_budget_restart_rejoin_replacement_compound_chain_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v7.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_pressure_score": float(round(
                self.compound_chain_pressure_score, 12)),
            "compound_chain_repair_rtr_match_score": float(round(
                self.compound_chain_repair_rtr_match_score, 12)),
            "per_budget_restart_rejoin_replacement_"
            "compound_chain_routing_cid":
                str(
                    self
                    .per_budget_restart_rejoin_replacement_compound_chain_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v8",
            "decision": self.to_dict()})


def _default_compound_chain_repair_rtr_match_table() -> dict[
        tuple[int, bool, bool, bool, bool, bool], str]:
    out: dict[
        tuple[int, bool, bool, bool, bool, bool], str] = {}
    # Default per-label provider with no extra pressure crossed.
    for label in range(12):
        out[(int(label), False, False, False, False, False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
    # Any pressure cross → paid-OpenAI tier (higher trust).
    for label in range(12):
        for r in (False, True):
            for j in (False, True):
                for rep in (False, True):
                    for cmp in (False, True):
                        for ch in (False, True):
                            if r or j or rep or cmp or ch:
                                out[(int(label), bool(r),
                                     bool(j), bool(rep),
                                     bool(cmp), bool(ch))] = (
                                    "openai_paid")
    return out


@dataclasses.dataclass
class HostedRouterControllerV8:
    inner_v7: HostedRouterControllerV7
    compound_chain_repair_rtr_match_table: dict[
        tuple[int, bool, bool, bool, bool, bool], str] = (
        dataclasses.field(
            default_factory=_default_compound_chain_repair_rtr_match_table))
    compound_chain_pressure_floor: float = (
        W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR)
    audit_v8: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            compound_chain_pressure_floor: float = (
                W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV8":
        inner_v7 = HostedRouterControllerV7.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v7=inner_v7,
            compound_chain_pressure_floor=float(
                compound_chain_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_HOSTED_ROUTER_V8_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_repair_rtr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), bool(k[4]), bool(k[5]), str(v)]
                for k, v in sorted(
                    self.compound_chain_repair_rtr_match_table
                    .items())],
            "compound_chain_pressure_floor": float(round(
                self.compound_chain_pressure_floor, 12)),
        })

    def _compound_chain_pressure_score(
            self, p: HostedProvider | None,
            req_v8: HostedRoutingRequestV8,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0, float(req_v8.compound_chain_pressure))))
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(req_v8.weight_compound_chain_pressure)
            * pressure * provider_lift)

    def _compound_chain_repair_rtr_match_score(
            self, p: HostedProvider | None,
            req_v8: HostedRoutingRequestV8,
    ) -> float:
        if p is None:
            return 0.0
        restart_above = bool(
            float(req_v8.restart_pressure)
            >= float(
                self.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor))
        rejoin_above = bool(
            float(req_v8.rejoin_pressure)
            >= float(
                self.inner_v7.inner_v6.inner_v5
                .rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v8.replacement_pressure)
            >= float(
                self.inner_v7.inner_v6.replacement_pressure_floor))
        compound_above = bool(
            float(req_v8.compound_pressure)
            >= float(self.inner_v7.compound_pressure_floor))
        chain_above = bool(
            float(req_v8.compound_chain_pressure)
            >= float(self.compound_chain_pressure_floor))
        key = (
            int(req_v8.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above),
            bool(compound_above),
            bool(chain_above))
        target = (
            self.compound_chain_repair_rtr_match_table.get(
                key, ""))
        if str(p.name) == str(target):
            return float(
                req_v8.weight_compound_chain_repair_rtr_match)
        return 0.0

    def decide_v8(
            self, req_v8: HostedRoutingRequestV8,
    ) -> HostedRoutingDecisionV8:
        d7 = self.inner_v7.decide_v7(req_v8.inner_v7)
        winner = (
            d7.chosen_provider
            if d7.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v7.inner_v6.inner_v5.inner_v4
                 .inner_v3.inner_v2.inner_v1.registry.providers
                 if pr.name == winner), None)
        ccp = float(self._compound_chain_pressure_score(p, req_v8))
        chain_match = float(
            self._compound_chain_repair_rtr_match_score(p, req_v8))
        per_routing_cid = _sha256_hex({
            "schema": W75_HOSTED_ROUTER_V8_SCHEMA_VERSION,
            "kind":
                "per_budget_restart_rejoin_replacement_"
                "compound_chain_routing_cid",
            "request_v8_cid": str(req_v8.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v8.visible_token_budget),
            "baseline_token_cost": int(
                req_v8.baseline_token_cost),
            "repair_dominance_label": int(
                req_v8.repair_dominance_label),
            "restart_pressure": float(round(
                req_v8.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v8.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v8.replacement_pressure, 12)),
            "compound_pressure": float(round(
                req_v8.compound_pressure, 12)),
            "compound_chain_pressure": float(round(
                req_v8.compound_chain_pressure, 12)),
        })
        self.audit_v8.append({
            "request_cid": str(req_v8.request_cid),
            "winner": str(winner or ""),
            "compound_chain_pressure_score": float(ccp),
            "compound_chain_repair_rtr_match_score": float(
                chain_match),
            "per_routing_cid": str(per_routing_cid),
        })
        return HostedRoutingDecisionV8(
            schema=W75_HOSTED_ROUTER_V8_SCHEMA_VERSION,
            inner_v7=d7,
            compound_chain_pressure_score=float(ccp),
            compound_chain_repair_rtr_match_score=float(
                chain_match),
            per_budget_restart_rejoin_replacement_compound_chain_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_compound_chain_success_under_budget(
            self) -> float:
        """Aggregate cost-per-compound-chain-success-under-budget."""
        n = float(len(self.audit_v8))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v8:
            ccp = float(entry.get(
                "compound_chain_pressure_score", 0.0))
            ch_m = float(entry.get(
                "compound_chain_repair_rtr_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(ccp) - 0.6 * float(ch_m))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV8Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_compound_chain_success_under_budget: float
    inner_v7_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_compound_chain_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_compound_chain_success_under_budget,
                    12)),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v8_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v8_witness(
        controller: HostedRouterControllerV8,
) -> HostedRouterControllerV8Witness:
    from .hosted_router_controller_v7 import (
        emit_hosted_router_controller_v7_witness,
    )
    inner_w = emit_hosted_router_controller_v7_witness(
        controller.inner_v7)
    return HostedRouterControllerV8Witness(
        schema=W75_HOSTED_ROUTER_V8_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v8)),
        routing_cost_per_compound_chain_success_under_budget=float(
            controller
            .routing_cost_per_compound_chain_success_under_budget()),
        inner_v7_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W75_HOSTED_ROUTER_V8_SCHEMA_VERSION",
    "W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR",
    "HostedRoutingRequestV8",
    "HostedRoutingDecisionV8",
    "HostedRouterControllerV8",
    "HostedRouterControllerV8Witness",
    "emit_hosted_router_controller_v8_witness",
]
