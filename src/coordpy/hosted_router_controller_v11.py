"""W78 H1 — Hosted Router Controller V11 (Plane A).

Strictly extends W77's ``coordpy.hosted_router_controller_v10``.
V11 adds:

* **Long-horizon-reconstruction-pressure weighting** — adds a new
  weight ``weight_long_horizon_reconstruction_pressure`` and
  folds caller-declared long-horizon-reconstruction pressure
  into the routing score.
* **Long-horizon-reconstruction-after-PCR match table** — V10
  had an 8-axis match key; V11 adds a 9-axis match key
  including long-horizon-reconstruction-above-floor.
* **Per-budget+...+long-horizon-reconstruction routing CID** —
  exposes a content-addressed routing CID that depends on the
  declared visible-token budget AND the declared pressures
  including long-horizon-reconstruction.

Honest scope (W78 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Long-horizon-reconstruction pressure is caller-declared (no
  live measurement). ``W78-L-HOSTED-V11-NO-SUBSTRATE-CAP``,
  ``W78-L-HOSTED-V11-LONG-HORIZON-RECONSTRUCTION-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
)
from .hosted_router_controller_v10 import (
    HostedRouterControllerV10, HostedRoutingDecisionV10,
    HostedRoutingRequestV10,
    W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR,
)
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_ROUTER_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v11.v1")
W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV11:
    inner_v10: HostedRoutingRequestV10
    long_horizon_reconstruction_pressure: float = 0.0
    weight_long_horizon_reconstruction_pressure: float = 0.6
    weight_long_horizon_reconstruction_after_pcr_match: float = 0.4

    @property
    def request_cid(self) -> str:
        return self.inner_v10.request_cid

    @property
    def visible_token_budget(self) -> int:
        return int(self.inner_v10.visible_token_budget)

    @property
    def baseline_token_cost(self) -> int:
        return int(self.inner_v10.baseline_token_cost)

    @property
    def repair_dominance_label(self) -> int:
        return int(self.inner_v10.repair_dominance_label)

    @property
    def restart_pressure(self) -> float:
        return float(self.inner_v10.restart_pressure)

    @property
    def rejoin_pressure(self) -> float:
        return float(self.inner_v10.rejoin_pressure)

    @property
    def replacement_pressure(self) -> float:
        return float(self.inner_v10.replacement_pressure)

    @property
    def compound_pressure(self) -> float:
        return float(self.inner_v10.compound_pressure)

    @property
    def compound_chain_pressure(self) -> float:
        return float(self.inner_v10.compound_chain_pressure)

    @property
    def compound_chain_then_restart_pressure(self) -> float:
        return float(
            self.inner_v10.compound_chain_then_restart_pressure)

    @property
    def post_restart_replacement_pressure(self) -> float:
        return float(
            self.inner_v10.post_restart_replacement_pressure)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_HOSTED_ROUTER_V11_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "long_horizon_reconstruction_pressure": float(round(
                self.long_horizon_reconstruction_pressure, 12)),
            "weight_long_horizon_reconstruction_pressure": float(
                round(
                    self
                    .weight_long_horizon_reconstruction_pressure,
                    12)),
            "weight_long_horizon_reconstruction_after_pcr_match":
                float(round(
                    self
                    .weight_long_horizon_reconstruction_after_pcr_match,
                    12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV11:
    schema: str
    inner_v10: HostedRoutingDecisionV10
    long_horizon_reconstruction_pressure_score: float
    long_horizon_reconstruction_after_pcr_match_score: float
    per_routing_cid_v11: str

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v10.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "long_horizon_reconstruction_pressure_score": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_score,
                    12)),
            "long_horizon_reconstruction_after_pcr_match_score":
                float(round(
                    self
                    .long_horizon_reconstruction_after_pcr_match_score,
                    12)),
            "per_routing_cid_v11": str(self.per_routing_cid_v11),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v11",
            "decision": self.to_dict()})


def _default_long_horizon_reconstruction_after_pcr_match_table(
) -> dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool,
              bool],
        str]:
    out: dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool,
              bool],
        str] = {}
    for label in range(15):
        out[(int(label), False, False, False, False, False,
             False, False, False)] = (
            "openai_paid" if label % 2 == 0
            else "openrouter_paid")
        out[(int(label), False, False, False, False, False,
             False, False, True)] = "openai_paid"
    return out


@dataclasses.dataclass
class HostedRouterControllerV11:
    inner_v10: HostedRouterControllerV10
    long_horizon_reconstruction_after_pcr_match_table: dict[
        tuple[int, bool, bool, bool, bool, bool, bool, bool,
              bool],
        str] = dataclasses.field(
            default_factory=(
                _default_long_horizon_reconstruction_after_pcr_match_table))
    long_horizon_reconstruction_pressure_floor: float = (
        W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR)
    audit_v11: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
            long_horizon_reconstruction_pressure_floor: float = (
                W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR),
    ) -> "HostedRouterControllerV11":
        inner_v10 = HostedRouterControllerV10.init(
            registry,
            success_score_per_provider=(
                success_score_per_provider or {}))
        return cls(
            inner_v10=inner_v10,
            long_horizon_reconstruction_pressure_floor=float(
                long_horizon_reconstruction_pressure_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_HOSTED_ROUTER_V11_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "long_horizon_reconstruction_after_pcr_match_table": [
                [int(k[0]), bool(k[1]), bool(k[2]),
                 bool(k[3]), bool(k[4]), bool(k[5]),
                 bool(k[6]), bool(k[7]), bool(k[8]), str(v)]
                for k, v in sorted(
                    self
                    .long_horizon_reconstruction_after_pcr_match_table
                    .items())],
            "long_horizon_reconstruction_pressure_floor": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_floor,
                    12)),
        })

    def _long_horizon_reconstruction_pressure_score(
            self, p: HostedProvider | None,
            req_v11: HostedRoutingRequestV11,
    ) -> float:
        if p is None:
            return 0.0
        pressure = float(max(0.0, min(
            1.0,
            float(req_v11.long_horizon_reconstruction_pressure))))
        provider_lift = (
            1.0 if str(p.name) == "openai_paid" else 0.5)
        return float(
            float(
                req_v11
                .weight_long_horizon_reconstruction_pressure)
            * pressure * provider_lift)

    def _long_horizon_reconstruction_after_pcr_match_score(
            self, p: HostedProvider | None,
            req_v11: HostedRoutingRequestV11,
    ) -> float:
        if p is None:
            return 0.0
        # Re-use the V10 above-floor flags + new LHR-above-floor.
        v10 = self.inner_v10
        # Read V9 floors for the previous axes.
        v9 = v10.inner_v9
        restart_above = bool(
            float(req_v11.restart_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor))
        rejoin_above = bool(
            float(req_v11.rejoin_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6.inner_v5
                .rejoin_pressure_floor))
        replacement_above = bool(
            float(req_v11.replacement_pressure)
            >= float(
                v9.inner_v8.inner_v7.inner_v6
                .replacement_pressure_floor))
        compound_above = bool(
            float(req_v11.compound_pressure)
            >= float(
                v9.inner_v8.inner_v7.compound_pressure_floor))
        chain_above = bool(
            float(req_v11.compound_chain_pressure)
            >= float(
                v9.inner_v8.compound_chain_pressure_floor))
        ctr_above = bool(
            float(req_v11.compound_chain_then_restart_pressure)
            >= float(v9.chain_then_restart_pressure_floor))
        pcr_above = bool(
            float(req_v11.post_restart_replacement_pressure)
            >= float(
                v10.post_restart_replacement_pressure_floor))
        lhr_above = bool(
            float(req_v11.long_horizon_reconstruction_pressure)
            >= float(
                self
                .long_horizon_reconstruction_pressure_floor))
        key = (
            int(req_v11.repair_dominance_label),
            bool(restart_above),
            bool(rejoin_above),
            bool(replacement_above),
            bool(compound_above),
            bool(chain_above),
            bool(ctr_above),
            bool(pcr_above),
            bool(lhr_above))
        target = (
            self
            .long_horizon_reconstruction_after_pcr_match_table
            .get(key, ""))
        if str(p.name) == str(target):
            return float(
                req_v11
                .weight_long_horizon_reconstruction_after_pcr_match)
        return 0.0

    def decide_v11(
            self, req_v11: HostedRoutingRequestV11,
    ) -> HostedRoutingDecisionV11:
        d10 = self.inner_v10.decide_v10(req_v11.inner_v10)
        winner = (
            d10.chosen_provider
            if d10.chosen_provider is not None else None)
        p = None
        if winner is not None:
            p = next(
                (pr for pr
                 in self.inner_v10.inner_v9.inner_v8.inner_v7
                 .inner_v6.inner_v5.inner_v4.inner_v3.inner_v2
                 .inner_v1.registry.providers
                 if pr.name == winner), None)
        lhr_p = float(
            self._long_horizon_reconstruction_pressure_score(
                p, req_v11))
        lhr_match = float(
            self
            ._long_horizon_reconstruction_after_pcr_match_score(
                p, req_v11))
        per_routing_cid = _sha256_hex({
            "schema": W78_HOSTED_ROUTER_V11_SCHEMA_VERSION,
            "kind": "per_routing_cid_v11",
            "request_v11_cid": str(req_v11.cid()),
            "winner": str(winner or ""),
            "visible_token_budget": int(
                req_v11.visible_token_budget),
            "baseline_token_cost": int(
                req_v11.baseline_token_cost),
            "repair_dominance_label": int(
                req_v11.repair_dominance_label),
            "restart_pressure": float(round(
                req_v11.restart_pressure, 12)),
            "rejoin_pressure": float(round(
                req_v11.rejoin_pressure, 12)),
            "replacement_pressure": float(round(
                req_v11.replacement_pressure, 12)),
            "compound_pressure": float(round(
                req_v11.compound_pressure, 12)),
            "compound_chain_pressure": float(round(
                req_v11.compound_chain_pressure, 12)),
            "compound_chain_then_restart_pressure": float(round(
                req_v11.compound_chain_then_restart_pressure,
                12)),
            "post_restart_replacement_pressure": float(round(
                req_v11.post_restart_replacement_pressure, 12)),
            "long_horizon_reconstruction_pressure": float(round(
                req_v11.long_horizon_reconstruction_pressure, 12)),
        })
        self.audit_v11.append({
            "request_cid": str(req_v11.request_cid),
            "winner": str(winner or ""),
            "long_horizon_reconstruction_pressure_score": float(
                lhr_p),
            "long_horizon_reconstruction_after_pcr_match_score":
                float(lhr_match),
            "per_routing_cid_v11": str(per_routing_cid),
        })
        return HostedRoutingDecisionV11(
            schema=W78_HOSTED_ROUTER_V11_SCHEMA_VERSION,
            inner_v10=d10,
            long_horizon_reconstruction_pressure_score=float(
                lhr_p),
            long_horizon_reconstruction_after_pcr_match_score=(
                float(lhr_match)),
            per_routing_cid_v11=str(per_routing_cid),
        )

    def routing_cost_per_long_horizon_reconstruction_success_under_budget(
            self) -> float:
        n = float(len(self.audit_v11))
        if n <= 0.0:
            return 0.0
        total = 0.0
        for entry in self.audit_v11:
            lhr_p = float(entry.get(
                "long_horizon_reconstruction_pressure_score", 0.0))
            lhr_m = float(entry.get(
                "long_horizon_reconstruction_after_pcr_match_score",
                0.0))
            total += (
                1.0 - 0.4 * float(lhr_p) - 0.6 * float(lhr_m))
        return float(total / n)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV11Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    routing_cost_per_long_horizon_reconstruction_success_under_budget: (
        float)
    inner_v10_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "routing_cost_per_long_horizon_reconstruction_success_under_budget":
                float(round(
                    self
                    .routing_cost_per_long_horizon_reconstruction_success_under_budget,
                    12)),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v11_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v11_witness(
        controller: HostedRouterControllerV11,
) -> HostedRouterControllerV11Witness:
    from .hosted_router_controller_v10 import (
        emit_hosted_router_controller_v10_witness,
    )
    inner_w = emit_hosted_router_controller_v10_witness(
        controller.inner_v10)
    return HostedRouterControllerV11Witness(
        schema=W78_HOSTED_ROUTER_V11_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v11)),
        routing_cost_per_long_horizon_reconstruction_success_under_budget=(
            float(
                controller
                .routing_cost_per_long_horizon_reconstruction_success_under_budget())),
        inner_v10_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W78_HOSTED_ROUTER_V11_SCHEMA_VERSION",
    "W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR",
    "HostedRoutingRequestV11",
    "HostedRoutingDecisionV11",
    "HostedRouterControllerV11",
    "HostedRouterControllerV11Witness",
    "emit_hosted_router_controller_v11_witness",
]
