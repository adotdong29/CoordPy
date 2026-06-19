"""W77 H1 — Hosted Router Controller V10 (Plane A).

Strictly extends W76's ``coordpy.hosted_router_controller_v9``. V10
adds:

* **Post-restart-replacement-pressure weighting** — adds a new
  weight ``weight_post_restart_replacement_pressure`` and folds
  caller-declared post-restart-replacement pressure into the
  routing score.
* **Post-restart-replacement-after-PCR match table** — V9 had a
  per-(label, restart, rejoin, replacement, compound, chain,
  chain-then-restart) match table; V10 adds a per-(…, post-
  restart-replacement) table so a high post-restart-replacement
  pressure can swing the match to a different provider tier.
* **Per-budget+...+post-restart-replacement routing CID** —
  exposes a content-addressed routing CID that depends on the
  declared visible-token budget AND the declared pressures
  including post-restart-replacement.

Honest scope (W77 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Post-restart-replacement pressure is caller-declared (no live
  measurement). ``W77-L-HOSTED-V10-NO-SUBSTRATE-CAP``,
  ``W77-L-HOSTED-V10-POST-RESTART-REPLACEMENT-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v9 import (
    HostedRouterControllerV9, HostedRoutingDecisionV9,
    HostedRoutingRequestV9,
)
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_ROUTER_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v10.v1")
W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV10:
    inner_v9: HostedRoutingRequestV9
    post_restart_replacement_pressure: float = 0.0
    weight_post_restart_replacement_pressure: float = 0.6
    weight_post_restart_replacement_after_pcr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v9.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v9.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v9.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v9.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v9.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v9.rejoin_pressure)

    @property
    def replacement_pressure(self) -> float:
        return float(self.inner_v9.replacement_pressure)

    @property
    def compound_pressure(self) -> float:
        return float(self.inner_v9.compound_pressure)

    @property
    def compound_chain_pressure(self) -> float:
        return float(self.inner_v9.compound_chain_pressure)

    @property
    def compound_chain_then_restart_pressure(self) -> float:
        return float(
            self.inner_v9.compound_chain_then_restart_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_HOSTED_ROUTER_V10_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "post_restart_replacement_pressure": float(round(
                self.post_restart_replacement_pressure, 12)),
            "weight_post_restart_replacement_pressure": float(
                round(
                    self
                    .weight_post_restart_replacement_pressure,
                    12)),
            "weight_post_restart_replacement_after_pcr_match":
                float(round(
                    self
                    .weight_post_restart_replacement_after_pcr_match,
                    12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV10:
    schema: str
    inner_v9: HostedRoutingDecisionV9
    post_restart_replacement_pressure_score: float
    post_restart_replacement_after_pcr_match_score: float
    per_routing_cid_v10: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v9.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "post_restart_replacement_pressure_score": float(round(
                self.post_restart_replacement_pressure_score, 12)),
            "post_restart_replacement_after_pcr_match_score":
                float(round(
                    self
                    .post_restart_replacement_after_pcr_match_score,
                    12)),
            "per_routing_cid_v10": str(self.per_routing_cid_v10),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v10",
            "decision": self.to_dict()})


def _default_post_restart_replacement_after_pcr_match_table(
) -> dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool],
        str]:
    out: dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool],
        str] = {}
    for label in range(14):
        out[(int(label), False, False, False, False, False,
             False, False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
    for label in range(14):
        for r in (False, True):
            for j in (False, True):
                for rep in (False, True):
                    for cmp in (False, True):
                        for ch in (False, True):
                            for ctr in (False, True):
                                for pcr in (False, True):
                                    if (r or j or rep or cmp
                                            or ch or ctr or pcr):
                                        out[(
                                            int(label), bool(r),
                                            bool(j), bool(rep),
                                            bool(cmp), bool(ch),
                                            bool(ctr),
                                            bool(pcr))] = (
                                                "openai_paid")
    return out


@dataclasses.dataclass
class HostedRouterControllerV10:
    inner_v9: HostedRouterControllerV9
    post_restart_replacement_after_pcr_match_table: dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool],
        str] = dataclasses.field(
            default_factory=(
                _default_post_restart_replacement_after_pcr_match_table))
    post_restart_replacement_pressure_floor: float = (
        W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR)
    audit_v10: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            post_restart_replacement_pressure_floor: float = (
                W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV10":
        inner_v9 = HostedRouterControllerV9.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v9=inner_v9,
            post_restart_replacement_pressure_floor=float(
                post_restart_replacement_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_HOSTED_ROUTER_V10_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "post_restart_replacement_after_pcr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), bool(k[4]), bool(k[5]),
                 bool(k[6]), bool(k[7]), str(v)]
                for k, v in sorted(
                    self
                    .post_restart_replacement_after_pcr_match_table
                    .items())],
            "post_restart_replacement_pressure_floor": float(round(
                self.post_restart_replacement_pressure_floor, 12)),
        })

    def _post_restart_replacement_pressure_score(
            self, p: HostedProvider | None,
            req_v10: HostedRoutingRequestV10,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0,
            float(req_v10.post_restart_replacement_pressure))))
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(
                req_v10
                .weight_post_restart_replacement_pressure)
            * pressure * provider_lift)

    def _post_restart_replacement_after_pcr_match_score(
            self, p: HostedProvider | None,
            req_v10: HostedRoutingRequestV10,
    ) -> float:
        if p is None:
            return 0.0
        v9 = self.inner_v9
        restart_above = bool(
            float(req_v10.restart_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor))
        rejoin_above = bool(
            float(req_v10.rejoin_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6.inner_v5
                .rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v10.replacement_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6
                .replacement_pressure_floor))
        compound_above = bool(
            float(req_v10.compound_pressure)
            >= float(
                v9.inner_v8.inner_v7.compound_pressure_floor))
        chain_above = bool(
            float(req_v10.compound_chain_pressure)
            >= float(
                v9.inner_v8.compound_chain_pressure_floor))
        ctr_above = bool(
            float(req_v10.compound_chain_then_restart_pressure)
            >= float(v9.chain_then_restart_pressure_floor))
        pcr_above = bool(
            float(req_v10.post_restart_replacement_pressure)
            >= float(self.post_restart_replacement_pressure_floor))
        key = (
            int(req_v10.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above),
            bool(compound_above),
            bool(chain_above),
            bool(ctr_above),
            bool(pcr_above))
        target = (
            self
            .post_restart_replacement_after_pcr_match_table.get(
                key, ""))
        if str(p.name) == str(target):
            return float(
                req_v10
                .weight_post_restart_replacement_after_pcr_match)
        return 0.0

    def decide_v10(
            self, req_v10: HostedRoutingRequestV10,
    ) -> HostedRoutingDecisionV10:
        d9 = self.inner_v9.decide_v9(req_v10.inner_v9)
        winner = (
            d9.chosen_provider
            if d9.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v9.inner_v8.inner_v7.inner_v6
                 .inner_v5.inner_v4.inner_v3.inner_v2.inner_v1
                 .registry.providers
                 if pr.name == winner), None)
        pcr_p = float(
            self._post_restart_replacement_pressure_score(
                p, req_v10))
        pcr_match = float(
            self._post_restart_replacement_after_pcr_match_score(
                p, req_v10))
        per_routing_cid = _sha256_hex({
            "schema": W77_HOSTED_ROUTER_V10_SCHEMA_VERSION,
            "kind": "per_routing_cid_v10",
            "request_v10_cid": str(req_v10.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v10.visible_token_budget),
            "baseline_token_cost": int(
                req_v10.baseline_token_cost),
            "repair_dominance_label": int(
                req_v10.repair_dominance_label),
            "restart_pressure": float(round(
                req_v10.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v10.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v10.replacement_pressure, 12)),
            "compound_pressure": float(round(
                req_v10.compound_pressure, 12)),
            "compound_chain_pressure": float(round(
                req_v10.compound_chain_pressure, 12)),
            "compound_chain_then_restart_pressure": float(round(
                req_v10.compound_chain_then_restart_pressure,
                12)),
            "post_restart_replacement_pressure": float(round(
                req_v10.post_restart_replacement_pressure, 12)),
        })
        self.audit_v10.append({
            "request_cid": str(req_v10.request_cid),
            "winner": str(winner or ""),
            "post_restart_replacement_pressure_score": float(
                pcr_p),
            "post_restart_replacement_after_pcr_match_score":
                float(pcr_match),
            "per_routing_cid_v10": str(per_routing_cid),
        })
        return HostedRoutingDecisionV10(
            schema=W77_HOSTED_ROUTER_V10_SCHEMA_VERSION,
            inner_v9=d9,
            post_restart_replacement_pressure_score=float(pcr_p),
            post_restart_replacement_after_pcr_match_score=float(
                pcr_match),
            per_routing_cid_v10=str(per_routing_cid),
        )

    def routing_cost_per_post_restart_replacement_success_under_budget(
            self) -> float:
        """Aggregate cost-per-post-restart-replacement-success-
        under-budget."""
        n = float(len(self.audit_v10))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v10:
            pcr_p = float(entry.get(
                "post_restart_replacement_pressure_score", 0.0))
            pcr_m = float(entry.get(
                "post_restart_replacement_after_pcr_match_score",
                0.0))
            total += (
                1.0 - 0.4 * float(pcr_p) - 0.6 * float(pcr_m))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV10Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_post_restart_replacement_success_under_budget: (
        float)
    inner_v9_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_post_restart_replacement_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_post_restart_replacement_success_under_budget,
                    12)),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v10_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v10_witness(
        controller: HostedRouterControllerV10,
) -> HostedRouterControllerV10Witness:
    from .hosted_router_controller_v9 import (
        emit_hosted_router_controller_v9_witness,
    )
    inner_w = emit_hosted_router_controller_v9_witness(
        controller.inner_v9)
    return HostedRouterControllerV10Witness(
        schema=W77_HOSTED_ROUTER_V10_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v10)),
        routing_cost_per_post_restart_replacement_success_under_budget=(
            float(
                controller
                .routing_cost_per_post_restart_replacement_success_under_budget())),
        inner_v9_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W77_HOSTED_ROUTER_V10_SCHEMA_VERSION",
    "W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR",
    "HostedRoutingRequestV10",
    "HostedRoutingDecisionV10",
    "HostedRouterControllerV10",
    "HostedRouterControllerV10Witness",
    "emit_hosted_router_controller_v10_witness",
]
