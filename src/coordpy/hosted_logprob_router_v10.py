"""W77 H2 — Hosted Logprob Router V10 (Plane A).

Strictly extends W76's ``coordpy.hosted_logprob_router_v9``. V10
adds:

* **Post-restart-replacement-aware abstain floor** — V10 further
  lowers the effective abstain threshold when caller-declared
  post-restart-replacement pressure is high.
* **Per-budget+…+post-restart-replacement tiebreak** — V10 takes
  pressures including post-restart-replacement and further shrinks
  effective top-k under joint chain + restart + replacement
  pressure.

Honest scope (W77 Plane A): post-restart-replacement pressure is
caller-declared. ``W77-L-HOSTED-V10-NO-SUBSTRATE-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_logprob_router import (
    TopKLogprobsPayload, W68_DEFAULT_TOP_K,
)
from .hosted_logprob_router_v9 import (
    HostedLogprobRouterV9,
    W76_DEFAULT_LOGPROB_V9_CHAIN_THEN_RESTART_FLOOR_DELTA,
    W76_DEFAULT_LOGPROB_V9_CHAIN_THEN_RESTART_PRESSURE_FLOOR,
    abstain_or_fuse_logprobs_v9,
)
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_LOGPROB_ROUTER_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v10.v1")
W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_FLOOR_DELTA: (
    float) = 1.5
W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR: (
    float) = 0.5


def abstain_or_fuse_logprobs_v10(
        payloads: Sequence[TopKLogprobsPayload], *,
        post_restart_replacement_pressure: float = 0.0,
        post_restart_replacement_pressure_floor: float = (
            W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR),
        post_restart_replacement_abstain_floor_delta: float = (
            W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_FLOOR_DELTA),
        abstain_entropy_floor: float = 1.0,
        top_k: int = W68_DEFAULT_TOP_K,
        **v9_kwargs: Any,
) -> dict[str, Any]:
    """V10 Bayesian fusion with post-restart-replacement-aware
    abstain.

    When ``post_restart_replacement_pressure >= post_restart_
    replacement_pressure_floor``, V10 lowers the abstain-entropy-
    floor by ``post_restart_replacement_abstain_floor_delta`` and
    further shrinks the effective top-k.
    """
    pcr_pressure = float(max(0.0, min(
        1.0, float(post_restart_replacement_pressure))))
    pcr_floor_active = bool(
        pcr_pressure
        >= float(post_restart_replacement_pressure_floor))
    pcr_floor_delta = (
        float(post_restart_replacement_abstain_floor_delta)
        if pcr_floor_active else 0.0)
    effective_abstain_floor = float(
        max(0.0,
            float(abstain_entropy_floor) - pcr_floor_delta))
    res = abstain_or_fuse_logprobs_v9(
        payloads,
        abstain_entropy_floor=float(effective_abstain_floor),
        top_k=int(top_k),
        **v9_kwargs)
    pcr_shrink = 0.5 if pcr_floor_active else 1.0
    effective_top_k = max(
        2, int(round(
            float(res.get("effective_top_k_v9", top_k))
            * float(pcr_shrink))))
    out = dict(res)
    out["schema"] = W77_HOSTED_LOGPROB_ROUTER_V10_SCHEMA_VERSION
    out["post_restart_replacement_pressure"] = float(round(
        pcr_pressure, 12))
    out["post_restart_replacement_floor_active"] = bool(
        pcr_floor_active)
    out["effective_abstain_entropy_floor_v10"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v10"] = int(effective_top_k)
    if str(out.get("fusion_kind", "")).startswith("abstain_"):
        out["fusion_kind"] = "abstain_v10"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV10:
    inner_v9: HostedLogprobRouterV9 = dataclasses.field(
        default_factory=HostedLogprobRouterV9)
    post_restart_replacement_pressure_floor: float = (
        W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR)
    post_restart_replacement_abstain_floor_delta: float = (
        W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_FLOOR_DELTA)
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W77_HOSTED_LOGPROB_ROUTER_V10_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "post_restart_replacement_pressure_floor": float(
                round(
                    self
                    .post_restart_replacement_pressure_floor,
                    12)),
            "post_restart_replacement_abstain_floor_delta": float(
                round(
                    self
                    .post_restart_replacement_abstain_floor_delta,
                    12)),
        })

    def fuse_v10(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            post_restart_replacement_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        # Mirror inner V9.fuse_v9 wiring (no direct call to keep
        # things lean — pass payloads to v10 abstain wrapper).
        v9 = self.inner_v9.inner_v8
        res = abstain_or_fuse_logprobs_v10(
            payloads,
            post_restart_replacement_pressure=float(
                post_restart_replacement_pressure),
            post_restart_replacement_pressure_floor=float(
                self.post_restart_replacement_pressure_floor),
            post_restart_replacement_abstain_floor_delta=float(
                self
                .post_restart_replacement_abstain_floor_delta),
            chain_then_restart_pressure_floor=float(
                self.inner_v9.chain_then_restart_pressure_floor),
            chain_then_restart_abstain_floor_delta=float(
                self.inner_v9
                .chain_then_restart_abstain_floor_delta),
            provider_trust=dict(
                v9.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .inner_v2.provider_trust),
            abstain_entropy_floor=float(
                v9.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .abstain_entropy_floor),
            **kwargs)
        self.audit_v10.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "post_restart_replacement_pressure": float(round(
                float(post_restart_replacement_pressure), 12)),
            "post_restart_replacement_floor_active": bool(
                res.get(
                    "post_restart_replacement_floor_active",
                    False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV10Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_post_restart_replacement_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_post_restart_replacement_floor_active": int(
                self.n_post_restart_replacement_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v10_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v10_witness(
        router: HostedLogprobRouterV10,
) -> HostedLogprobRouterV10Witness:
    n_abst = sum(
        1 for e in router.audit_v10
        if str(e.get("kind", "")).startswith("abstain_"))
    n_pcr = sum(
        1 for e in router.audit_v10
        if bool(
            e.get("post_restart_replacement_floor_active",
                  False)))
    return HostedLogprobRouterV10Witness(
        schema=W77_HOSTED_LOGPROB_ROUTER_V10_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v10)),
        n_abstain=int(n_abst),
        n_post_restart_replacement_floor_active=int(n_pcr),
    )


__all__ = [
    "W77_HOSTED_LOGPROB_ROUTER_V10_SCHEMA_VERSION",
    "W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_FLOOR_DELTA",
    "W77_DEFAULT_LOGPROB_V10_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v10",
    "HostedLogprobRouterV10",
    "HostedLogprobRouterV10Witness",
    "emit_hosted_logprob_router_v10_witness",
]
