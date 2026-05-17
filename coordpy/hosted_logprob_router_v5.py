"""W72 H2 — Hosted Logprob Router V5 (Plane A).

Strictly extends W71's ``coordpy.hosted_logprob_router_v4``. V5
adds:

* **Rejoin-aware abstain floor** — V4 had a restart-aware abstain
  floor delta. V5 *further* lowers the effective abstain threshold
  when caller-declared rejoin pressure is high, so V5 abstains
  more aggressively when the team is recovering from branch
  divergence after a recent restart.
* **Per-budget+restart+rejoin tiebreak** — V5 takes
  ``visible_token_budget``, ``restart_pressure``, and
  ``rejoin_pressure`` hints and shrinks effective top-k under
  tight budgets, high restart pressure, AND high rejoin pressure.

Honest scope (W72 Plane A)
--------------------------

* Rejoin pressure is a caller-declared signal, not a substrate-
  state read. ``W72-L-HOSTED-V5-REJOIN-DECLARED-CAP``,
  ``W72-L-HOSTED-V5-NO-SUBSTRATE-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_logprob_router import (
    TopKLogprobsPayload, W68_DEFAULT_TOP_K,
    W68_DEFAULT_LOGPROB_FUSION_FLOOR,
)
from .hosted_logprob_router_v2 import (
    W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR,
)
from .hosted_logprob_router_v3 import (
    W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR,
)
from .hosted_logprob_router_v4 import (
    HostedLogprobRouterV4,
    W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA,
    W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR,
    abstain_or_fuse_logprobs_v4,
)
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_LOGPROB_ROUTER_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v5.v1")
W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA: float = 1.2
W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR: float = 0.5


def abstain_or_fuse_logprobs_v5(
        payloads: Sequence[TopKLogprobsPayload], *,
        provider_trust: dict[str, float] | None = None,
        prior_alpha: float = 1.0,
        top_k: int = W68_DEFAULT_TOP_K,
        floor: float = W68_DEFAULT_LOGPROB_FUSION_FLOOR,
        tiebreak_entropy_floor: float = (
            W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR),
        abstain_entropy_floor: float = (
            W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR),
        visible_token_budget: int = 256,
        baseline_token_cost: int = 512,
        restart_pressure: float = 0.0,
        restart_pressure_floor: float = (
            W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR),
        restart_abstain_floor_delta: float = (
            W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA),
        rejoin_pressure: float = 0.0,
        rejoin_pressure_floor: float = (
            W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR),
        rejoin_abstain_floor_delta: float = (
            W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA),
) -> dict[str, Any]:
    """V5 Bayesian fusion with rejoin-aware abstain and per-budget+
    restart+rejoin tiebreak.

    When ``rejoin_pressure >= rejoin_pressure_floor``, V5 lowers
    the abstain-entropy-floor by ``rejoin_abstain_floor_delta``
    (so abstain fires at lower entropy) and further shrinks the
    effective top-k.
    """
    rj_pressure = float(max(0.0, min(
        1.0, float(rejoin_pressure))))
    rj_floor_active = bool(
        rj_pressure >= float(rejoin_pressure_floor))
    rj_floor_delta = (
        float(rejoin_abstain_floor_delta)
        if rj_floor_active else 0.0)
    effective_abstain_floor = float(
        max(0.0, float(abstain_entropy_floor) - rj_floor_delta))
    res = abstain_or_fuse_logprobs_v4(
        payloads,
        provider_trust=dict(provider_trust or {}),
        prior_alpha=float(prior_alpha),
        top_k=int(top_k), floor=float(floor),
        tiebreak_entropy_floor=float(tiebreak_entropy_floor),
        abstain_entropy_floor=float(effective_abstain_floor),
        visible_token_budget=int(visible_token_budget),
        baseline_token_cost=int(baseline_token_cost),
        restart_pressure=float(restart_pressure),
        restart_pressure_floor=float(restart_pressure_floor),
        restart_abstain_floor_delta=float(
            restart_abstain_floor_delta))
    rj_shrink = 0.5 if rj_floor_active else 1.0
    effective_top_k = max(
        2, int(round(
            float(res.get("effective_top_k_v4", top_k))
            * float(rj_shrink))))
    out = dict(res)
    out["schema"] = W72_HOSTED_LOGPROB_ROUTER_V5_SCHEMA_VERSION
    out["rejoin_pressure"] = float(round(rj_pressure, 12))
    out["rejoin_floor_active"] = bool(rj_floor_active)
    out["effective_abstain_entropy_floor"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v5"] = int(effective_top_k)
    if str(out.get("fusion_kind", "")) in (
            "abstain_v3", "abstain_v4"):
        out["fusion_kind"] = "abstain_v5"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV5:
    inner_v4: HostedLogprobRouterV4 = dataclasses.field(
        default_factory=HostedLogprobRouterV4)
    rejoin_pressure_floor: float = (
        W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR)
    rejoin_abstain_floor_delta: float = (
        W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA)
    audit_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W72_HOSTED_LOGPROB_ROUTER_V5_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "rejoin_pressure_floor": float(round(
                self.rejoin_pressure_floor, 12)),
            "rejoin_abstain_floor_delta": float(round(
                self.rejoin_abstain_floor_delta, 12)),
        })

    def fuse_v5(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            restart_pressure: float = 0.0,
            rejoin_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = abstain_or_fuse_logprobs_v5(
            payloads,
            provider_trust=dict(
                self.inner_v4.inner_v3.inner_v2.provider_trust),
            abstain_entropy_floor=float(
                self.inner_v4.inner_v3.abstain_entropy_floor),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=int(baseline_token_cost),
            restart_pressure=float(restart_pressure),
            restart_pressure_floor=float(
                self.inner_v4.restart_pressure_floor),
            restart_abstain_floor_delta=float(
                self.inner_v4.restart_abstain_floor_delta),
            rejoin_pressure=float(rejoin_pressure),
            rejoin_pressure_floor=float(
                self.rejoin_pressure_floor),
            rejoin_abstain_floor_delta=float(
                self.rejoin_abstain_floor_delta),
            **kwargs)
        self.audit_v5.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "visible_token_budget": int(visible_token_budget),
            "restart_pressure": float(round(
                float(restart_pressure), 12)),
            "rejoin_pressure": float(round(
                float(rejoin_pressure), 12)),
            "rejoin_floor_active": bool(
                res.get("rejoin_floor_active", False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV5Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_bayesian: int
    n_tiebreak: int
    n_rejoin_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
            "n_rejoin_floor_active": int(
                self.n_rejoin_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v5_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v5_witness(
        router: HostedLogprobRouterV5,
) -> HostedLogprobRouterV5Witness:
    n_abst = sum(
        1 for e in router.audit_v5
        if str(e.get("kind", "")) in (
            "abstain_v3", "abstain_v4", "abstain_v5"))
    n_bayes = sum(
        1 for e in router.audit_v5
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit_v5
        if "tiebreak" in str(e.get("kind", "")))
    n_rfa = sum(
        1 for e in router.audit_v5
        if bool(e.get("rejoin_floor_active", False)))
    return HostedLogprobRouterV5Witness(
        schema=W72_HOSTED_LOGPROB_ROUTER_V5_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v5)),
        n_abstain=int(n_abst),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
        n_rejoin_floor_active=int(n_rfa),
    )


__all__ = [
    "W72_HOSTED_LOGPROB_ROUTER_V5_SCHEMA_VERSION",
    "W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA",
    "W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v5",
    "HostedLogprobRouterV5",
    "HostedLogprobRouterV5Witness",
    "emit_hosted_logprob_router_v5_witness",
]
