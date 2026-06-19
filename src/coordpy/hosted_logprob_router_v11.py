"""W78 H2 — Hosted Logprob Router V11 (Plane A).

Strictly extends W77's ``coordpy.hosted_logprob_router_v10``. V11
adds:

* **Long-horizon-reconstruction-aware abstain floor** — V11
  further lowers the effective abstain threshold when caller-
  declared long-horizon-reconstruction pressure is high.
* **Per-budget+...+long-horizon-reconstruction tiebreak** — V11
  takes pressures including long-horizon-reconstruction and
  further shrinks effective top-k under joint chain + restart +
  replacement + LHR pressure.

Honest scope (W78 Plane A): long-horizon-reconstruction pressure
is caller-declared. ``W78-L-HOSTED-V11-NO-SUBSTRATE-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_logprob_router import (
    TopKLogprobsPayload, W68_DEFAULT_TOP_K,
)
from .hosted_logprob_router_v10 import (
    HostedLogprobRouterV10,
    W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_FLOOR_DELTA,
    W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR,
    abstain_or_fuse_logprobs_v10,
)
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_LOGPROB_ROUTER_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v11.v1")
W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_FLOOR_DELTA: (
    float) = 1.6
W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR: (
    float) = 0.5


def abstain_or_fuse_logprobs_v11(
        payloads: Sequence[TopKLogprobsPayload], *,
        long_horizon_reconstruction_pressure: float = 0.0,
        long_horizon_reconstruction_pressure_floor: float = (
            W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR),
        long_horizon_reconstruction_abstain_floor_delta: float = (
            W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_FLOOR_DELTA),
        abstain_entropy_floor: float = 1.0,
        top_k: int = W68_DEFAULT_TOP_K,
        **v10_kwargs: Any,
) -> dict[str, Any]:
    """V11 Bayesian fusion with long-horizon-reconstruction-aware
    abstain.

    When ``long_horizon_reconstruction_pressure >= long_horizon_
    reconstruction_pressure_floor``, V11 lowers the
    abstain-entropy-floor by ``long_horizon_reconstruction_
    abstain_floor_delta`` and further shrinks the effective top-k.
    """
    lhr_pressure = float(max(0.0, min(
        1.0, float(long_horizon_reconstruction_pressure))))
    lhr_floor_active = bool(
        lhr_pressure
        >= float(long_horizon_reconstruction_pressure_floor))
    lhr_floor_delta = (
        float(long_horizon_reconstruction_abstain_floor_delta)
        if lhr_floor_active else 0.0)
    effective_abstain_floor = float(
        max(0.0,
            float(abstain_entropy_floor) - lhr_floor_delta))
    res = abstain_or_fuse_logprobs_v10(
        payloads,
        abstain_entropy_floor=float(effective_abstain_floor),
        top_k=int(top_k),
        **v10_kwargs)
    lhr_shrink = 0.5 if lhr_floor_active else 1.0
    effective_top_k = max(
        2, int(round(
            float(res.get("effective_top_k_v10", top_k))
            * float(lhr_shrink))))
    out = dict(res)
    out["schema"] = W78_HOSTED_LOGPROB_ROUTER_V11_SCHEMA_VERSION
    out["long_horizon_reconstruction_pressure"] = float(round(
        lhr_pressure, 12))
    out["long_horizon_reconstruction_floor_active"] = bool(
        lhr_floor_active)
    out["effective_abstain_entropy_floor_v11"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v11"] = int(effective_top_k)
    if str(out.get("fusion_kind", "")).startswith("abstain_"):
        out["fusion_kind"] = "abstain_v11"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV11:
    inner_v10: HostedLogprobRouterV10 = dataclasses.field(
        default_factory=HostedLogprobRouterV10)
    long_horizon_reconstruction_pressure_floor: float = (
        W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR)
    long_horizon_reconstruction_abstain_floor_delta: float = (
        W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_FLOOR_DELTA)
    audit_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W78_HOSTED_LOGPROB_ROUTER_V11_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "long_horizon_reconstruction_pressure_floor": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_floor,
                    12)),
            "long_horizon_reconstruction_abstain_floor_delta":
                float(round(
                    self
                    .long_horizon_reconstruction_abstain_floor_delta,
                    12)),
        })

    def fuse_v11(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            long_horizon_reconstruction_pressure: float = 0.0,
            post_restart_replacement_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        v10 = self.inner_v10.inner_v9
        v9 = v10.inner_v8
        res = abstain_or_fuse_logprobs_v11(
            payloads,
            long_horizon_reconstruction_pressure=float(
                long_horizon_reconstruction_pressure),
            long_horizon_reconstruction_pressure_floor=float(
                self.long_horizon_reconstruction_pressure_floor),
            long_horizon_reconstruction_abstain_floor_delta=float(
                self
                .long_horizon_reconstruction_abstain_floor_delta),
            post_restart_replacement_pressure=float(
                post_restart_replacement_pressure),
            post_restart_replacement_pressure_floor=float(
                self.inner_v10
                .post_restart_replacement_pressure_floor),
            post_restart_replacement_abstain_floor_delta=float(
                self.inner_v10
                .post_restart_replacement_abstain_floor_delta),
            chain_then_restart_pressure_floor=float(
                self.inner_v10.inner_v9
                .chain_then_restart_pressure_floor),
            chain_then_restart_abstain_floor_delta=float(
                self.inner_v10.inner_v9
                .chain_then_restart_abstain_floor_delta),
            provider_trust=dict(
                v9.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .inner_v2.provider_trust),
            abstain_entropy_floor=float(
                v9.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .abstain_entropy_floor),
            **kwargs)
        self.audit_v11.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "long_horizon_reconstruction_pressure": float(round(
                float(long_horizon_reconstruction_pressure), 12)),
            "long_horizon_reconstruction_floor_active": bool(
                res.get(
                    "long_horizon_reconstruction_floor_active",
                    False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV11Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_long_horizon_reconstruction_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_long_horizon_reconstruction_floor_active": int(
                self.n_long_horizon_reconstruction_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v11_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v11_witness(
        router: HostedLogprobRouterV11,
) -> HostedLogprobRouterV11Witness:
    n_abst = sum(
        1 for e in router.audit_v11
        if str(e.get("kind", "")).startswith("abstain_"))
    n_lhr = sum(
        1 for e in router.audit_v11
        if bool(
            e.get("long_horizon_reconstruction_floor_active",
                  False)))
    return HostedLogprobRouterV11Witness(
        schema=W78_HOSTED_LOGPROB_ROUTER_V11_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v11)),
        n_abstain=int(n_abst),
        n_long_horizon_reconstruction_floor_active=int(n_lhr),
    )


__all__ = [
    "W78_HOSTED_LOGPROB_ROUTER_V11_SCHEMA_VERSION",
    "W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_FLOOR_DELTA",
    "W78_DEFAULT_LOGPROB_V11_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v11",
    "HostedLogprobRouterV11",
    "HostedLogprobRouterV11Witness",
    "emit_hosted_logprob_router_v11_witness",
]
