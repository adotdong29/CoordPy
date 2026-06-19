"""W76 H1 — Hosted Router Controller V9 (Plane A).

Strictly extends W75's
``coordpy.hosted_router_controller_v8``. V9 adds:

* **Compound-chain-then-restart-pressure weighting** — adds a new
  weight ``weight_compound_chain_then_restart_pressure`` and folds
  caller-declared chain-then-restart pressure into the routing
  score.
* **Compound-chain-then-restart-after-RTR match table** — V8 had
  a per-(label, restart, rejoin, replacement, compound, chain)
  match table; V9 adds a per-(label, restart, rejoin, replacement,
  compound, chain, chain-then-restart) table so a high chain-then-
  restart pressure can swing the match to a different provider
  tier.
* **Per-budget+restart+rejoin+replacement+compound+chain+
  chain-then-restart routing CID** — exposes a content-addressed
  routing CID that depends on the declared visible-token budget
  AND the declared pressures including chain-then-restart.

Honest scope (W76 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Chain-then-restart pressure is caller-declared (no live
  measurement). ``W76-L-HOSTED-V9-NO-SUBSTRATE-CAP``,
  ``W76-L-HOSTED-V9-COMPOUND-CHAIN-THEN-RESTART-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v8 import (
    HostedRouterControllerV8, HostedRoutingDecisionV8,
    HostedRoutingRequestV8,
)
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_ROUTER_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v9.v1")
W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV9:
    inner_v8: HostedRoutingRequestV8
    compound_chain_then_restart_pressure: float = 0.0
    weight_compound_chain_then_restart_pressure: float = 0.6
    weight_chain_then_restart_after_rtr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v8.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v8.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v8.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v8.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v8.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v8.rejoin_pressure)

    @property
    def replacement_pressure(self) -> float:
        return float(self.inner_v8.replacement_pressure)

    @property
    def compound_pressure(self) -> float:
        return float(self.inner_v8.compound_pressure)

    @property
    def compound_chain_pressure(self) -> float:
        return float(self.inner_v8.compound_chain_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_HOSTED_ROUTER_V9_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "compound_chain_then_restart_pressure": float(round(
                self.compound_chain_then_restart_pressure, 12)),
            "weight_compound_chain_then_restart_pressure": float(
                round(
                    self
                    .weight_compound_chain_then_restart_pressure,
                    12)),
            "weight_chain_then_restart_after_rtr_match": float(
                round(
                    self
                    .weight_chain_then_restart_after_rtr_match,
                    12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV9:
    schema: str
    inner_v8: HostedRoutingDecisionV8
    chain_then_restart_pressure_score: float
    chain_then_restart_after_rtr_match_score: float
    per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v8.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "chain_then_restart_pressure_score": float(round(
                self.chain_then_restart_pressure_score, 12)),
            "chain_then_restart_after_rtr_match_score": float(
                round(
                    self
                    .chain_then_restart_after_rtr_match_score,
                    12)),
            "per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid":
                str(
                    self
                    .per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v9",
            "decision": self.to_dict()})


def _default_chain_then_restart_after_rtr_match_table() -> dict[
        tuple[int, bool, bool, bool, bool, bool, bool], str]:
    out: dict[
        tuple[int, bool, bool, bool, bool, bool, bool],
        str] = {}
    # Default per-label provider with no extra pressure crossed.
    for label in range(13):
        out[(int(label), False, False, False, False, False,
             False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
    # Any pressure cross → paid-OpenAI tier (higher trust).
    for label in range(13):
        for r in (False, True):
            for j in (False, True):
                for rep in (False, True):
                    for cmp in (False, True):
                        for ch in (False, True):
                            for ctr in (False, True):
                                if (r or j or rep or cmp
                                        or ch or ctr):
                                    out[(
                                        int(label), bool(r),
                                        bool(j), bool(rep),
                                        bool(cmp), bool(ch),
                                        bool(ctr))] = (
                                            "openai_paid")
    return out


@dataclasses.dataclass
class HostedRouterControllerV9:
    inner_v8: HostedRouterControllerV8
    chain_then_restart_after_rtr_match_table: dict[
        tuple[int, bool, bool, bool, bool, bool, bool],
        str] = dataclasses.field(
            default_factory=(
                _default_chain_then_restart_after_rtr_match_table))
    chain_then_restart_pressure_floor: float = (
        W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR)
    audit_v9: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            chain_then_restart_pressure_floor: float = (
                W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV9":
        inner_v8 = HostedRouterControllerV8.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v8=inner_v8,
            chain_then_restart_pressure_floor=float(
                chain_then_restart_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_HOSTED_ROUTER_V9_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "chain_then_restart_after_rtr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), bool(k[4]), bool(k[5]),
                 bool(k[6]), str(v)]
                for k, v in sorted(
                    self
                    .chain_then_restart_after_rtr_match_table
                    .items())],
            "chain_then_restart_pressure_floor": float(round(
                self.chain_then_restart_pressure_floor, 12)),
        })

    def _chain_then_restart_pressure_score(
            self, p: HostedProvider | None,
            req_v9: HostedRoutingRequestV9,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0,
            float(req_v9.compound_chain_then_restart_pressure))))
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(
                req_v9
                .weight_compound_chain_then_restart_pressure)
            * pressure * provider_lift)

    def _chain_then_restart_after_rtr_match_score(
            self, p: HostedProvider | None,
            req_v9: HostedRoutingRequestV9,
    ) -> float:
        if p is None:
            return 0.0
        restart_above = bool(
            float(req_v9.restart_pressure)
            >= float(
                self.inner_v8.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor))
        rejoin_above = bool(
            float(req_v9.rejoin_pressure)
            >= float(
                self.inner_v8.inner_v7.inner_v6.inner_v5
                .rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v9.replacement_pressure)
            >= float(
                self.inner_v8.inner_v7.inner_v6
                .replacement_pressure_floor))
        compound_above = bool(
            float(req_v9.compound_pressure)
            >= float(self.inner_v8.inner_v7.compound_pressure_floor))
        chain_above = bool(
            float(req_v9.compound_chain_pressure)
            >= float(self.inner_v8.compound_chain_pressure_floor))
        ctr_above = bool(
            float(req_v9.compound_chain_then_restart_pressure)
            >= float(self.chain_then_restart_pressure_floor))
        key = (
            int(req_v9.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above),
            bool(compound_above),
            bool(chain_above),
            bool(ctr_above))
        target = (
            self.chain_then_restart_after_rtr_match_table.get(
                key, ""))
        if str(p.name) == str(target):
            return float(
                req_v9.weight_chain_then_restart_after_rtr_match)
        return 0.0

    def decide_v9(
            self, req_v9: HostedRoutingRequestV9,
    ) -> HostedRoutingDecisionV9:
        d8 = self.inner_v8.decide_v8(req_v9.inner_v8)
        winner = (
            d8.chosen_provider
            if d8.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v8.inner_v7.inner_v6.inner_v5
                 .inner_v4.inner_v3.inner_v2.inner_v1.registry
                 .providers
                 if pr.name == winner), None)
        ctr_p = float(
            self._chain_then_restart_pressure_score(p, req_v9))
        ctr_match = float(
            self._chain_then_restart_after_rtr_match_score(
                p, req_v9))
        per_routing_cid = _sha256_hex({
            "schema": W76_HOSTED_ROUTER_V9_SCHEMA_VERSION,
            "kind":
                "per_budget_restart_rejoin_replacement_"
                "compound_chain_then_restart_routing_cid",
            "request_v9_cid": str(req_v9.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v9.visible_token_budget),
            "baseline_token_cost": int(
                req_v9.baseline_token_cost),
            "repair_dominance_label": int(
                req_v9.repair_dominance_label),
            "restart_pressure": float(round(
                req_v9.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v9.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v9.replacement_pressure, 12)),
            "compound_pressure": float(round(
                req_v9.compound_pressure, 12)),
            "compound_chain_pressure": float(round(
                req_v9.compound_chain_pressure, 12)),
            "compound_chain_then_restart_pressure": float(round(
                req_v9.compound_chain_then_restart_pressure, 12)),
        })
        self.audit_v9.append({
            "request_cid": str(req_v9.request_cid),
            "winner": str(winner or ""),
            "chain_then_restart_pressure_score": float(ctr_p),
            "chain_then_restart_after_rtr_match_score": float(
                ctr_match),
            "per_routing_cid": str(per_routing_cid),
        })
        return HostedRoutingDecisionV9(
            schema=W76_HOSTED_ROUTER_V9_SCHEMA_VERSION,
            inner_v8=d8,
            chain_then_restart_pressure_score=float(ctr_p),
            chain_then_restart_after_rtr_match_score=float(
                ctr_match),
            per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid=str(
                per_routing_cid),
        )

    def routing_cost_per_chain_then_restart_success_under_budget(
            self) -> float:
        """Aggregate cost-per-chain-then-restart-success-under-
        budget."""
        n = float(len(self.audit_v9))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v9:
            ctr_p = float(entry.get(
                "chain_then_restart_pressure_score", 0.0))
            ctr_m = float(entry.get(
                "chain_then_restart_after_rtr_match_score", 0.0))
            total += (
                1.0 - 0.4 * float(ctr_p) - 0.6 * float(ctr_m))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV9Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_chain_then_restart_success_under_budget: (
        float)
    inner_v8_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_chain_then_restart_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_chain_then_restart_success_under_budget,
                    12)),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v9_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v9_witness(
        controller: HostedRouterControllerV9,
) -> HostedRouterControllerV9Witness:
    from .hosted_router_controller_v8 import (
        emit_hosted_router_controller_v8_witness,
    )
    inner_w = emit_hosted_router_controller_v8_witness(
        controller.inner_v8)
    return HostedRouterControllerV9Witness(
        schema=W76_HOSTED_ROUTER_V9_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v9)),
        routing_cost_per_chain_then_restart_success_under_budget=(
            float(
                controller
                .routing_cost_per_chain_then_restart_success_under_budget())),
        inner_v8_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W76_HOSTED_ROUTER_V9_SCHEMA_VERSION",
    "W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR",
    "HostedRoutingRequestV9",
    "HostedRoutingDecisionV9",
    "HostedRouterControllerV9",
    "HostedRouterControllerV9Witness",
    "emit_hosted_router_controller_v9_witness",
]
