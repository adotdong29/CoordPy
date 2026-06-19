"""W79 — Hosted Logprob Router V12.

Strictly extends W78's ``coordpy.hosted_logprob_router_v11``.
Adds replacement-then-restart-after-long-delay-aware abstain
floor and a 10-pressure tiebreak."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_logprob_router_v11 import HostedLogprobRouterV11


W79_HOSTED_LOGPROB_ROUTER_V12_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v12.v1")
W79_HOSTED_LOGPROB_ROUTER_V12_N_TIEBREAK_PRESSURES: int = 10


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class HostedLogprobRouterV12:
    inner_v11: HostedLogprobRouterV11 = dataclasses.field(
        default_factory=HostedLogprobRouterV11)
    replacement_then_restart_after_long_delay_aware_abstain_floor: (
        float) = 0.35

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W79_HOSTED_LOGPROB_ROUTER_V12_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "replacement_then_restart_after_long_delay_aware_abstain_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_aware_abstain_floor,
                    12)),
            "n_tiebreak_pressures": int(
                W79_HOSTED_LOGPROB_ROUTER_V12_N_TIEBREAK_PRESSURES),
        })


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV12Witness:
    schema: str
    router_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v12_witness",
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
        })


def emit_hosted_logprob_router_v12_witness(
        router: HostedLogprobRouterV12,
) -> HostedLogprobRouterV12Witness:
    return HostedLogprobRouterV12Witness(
        schema=W79_HOSTED_LOGPROB_ROUTER_V12_SCHEMA_VERSION,
        router_cid=str(router.cid()),
    )


__all__ = [
    "W79_HOSTED_LOGPROB_ROUTER_V12_SCHEMA_VERSION",
    "W79_HOSTED_LOGPROB_ROUTER_V12_N_TIEBREAK_PRESSURES",
    "HostedLogprobRouterV12",
    "HostedLogprobRouterV12Witness",
    "emit_hosted_logprob_router_v12_witness",
]
