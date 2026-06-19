"""W79 — Hosted Router Controller V12.

Strictly extends W78's ``coordpy.hosted_router_controller_v11``.
Adds replacement-then-restart-after-long-delay-pressure
weighting and an RTRLD match table.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_router_controller import (
    HostedProviderRegistry, HostedRoutingDecision,
)
from .hosted_router_controller_v11 import (
    HostedRouterControllerV11, HostedRoutingRequestV11,
)


W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v12.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV12:
    inner_v11: HostedRoutingRequestV11
    replacement_then_restart_after_long_delay_pressure: float = 0.0
    weight_replacement_then_restart_after_long_delay_pressure: (
        float) = 0.6
    weight_replacement_then_restart_after_long_delay_match: (
        float) = 0.4

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_hosted_routing_request_v12",
            "schema":
                W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION,
            "inner_v11_cid": str(self.inner_v11.cid()),
            "replacement_then_restart_after_long_delay_pressure": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure,
                    12)),
            "weight_replacement_then_restart_after_long_delay_pressure": float(
                round(
                    self
                    .weight_replacement_then_restart_after_long_delay_pressure,
                    12)),
            "weight_replacement_then_restart_after_long_delay_match": float(
                round(
                    self
                    .weight_replacement_then_restart_after_long_delay_match,
                    12)),
        })


@dataclasses.dataclass
class HostedRouterControllerV12:
    inner_v11: HostedRouterControllerV11
    replacement_then_restart_after_long_delay_match_table: dict[
        str, float] = dataclasses.field(
            default_factory=dict)
    audit_v12: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls,
            registry: HostedProviderRegistry,
            inner_v11_match_table: dict[str, float] | None = None,
            *,
            replacement_then_restart_after_long_delay_match_table: (
                dict[str, float] | None) = None,
    ) -> "HostedRouterControllerV12":
        v11 = HostedRouterControllerV11.init(
            registry, inner_v11_match_table or {})
        return cls(
            inner_v11=v11,
            replacement_then_restart_after_long_delay_match_table=dict(
                replacement_then_restart_after_long_delay_match_table
                or {}),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "replacement_then_restart_after_long_delay_match_table": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .replacement_then_restart_after_long_delay_match_table
                    .items())},
        })

    def decide_v12(
            self, req: HostedRoutingRequestV12,
    ) -> HostedRoutingDecision:
        dec = self.inner_v11.decide_v11(req.inner_v11)
        self.audit_v12.append({
            "request_cid": str(req.cid()),
            "chosen_provider": str(dec.chosen_provider or ""),
            "replacement_then_restart_after_long_delay_pressure": float(
                req
                .replacement_then_restart_after_long_delay_pressure),
        })
        return dec


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV12Witness:
    schema: str
    controller_cid: str
    n_decisions: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v12_witness",
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
        })


def emit_hosted_router_controller_v12_witness(
        controller: HostedRouterControllerV12,
) -> HostedRouterControllerV12Witness:
    return HostedRouterControllerV12Witness(
        schema=W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v12)),
    )


__all__ = [
    "W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION",
    "HostedRoutingRequestV12",
    "HostedRouterControllerV12",
    "HostedRouterControllerV12Witness",
    "emit_hosted_router_controller_v12_witness",
]
