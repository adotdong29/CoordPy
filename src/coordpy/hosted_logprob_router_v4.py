"""W71 H2 — Hosted Logprob Router V4 (Plane A).

Strictly extends W70's ``coordpy.hosted_logprob_router_v3``. V4
adds:

* **Restart-aware abstain floor** — V3 had a single
  abstain-entropy-floor. V4 *lowers* the effective abstain
  threshold when caller-declared restart pressure is high, so V4
  abstains more aggressively when the team is recovering from
  recent restarts.
* **Per-budget+restart tiebreak** — V4 takes both
  ``visible_token_budget`` and ``restart_pressure`` hints and
  shrinks effective top-k under tight budgets AND under high
  restart pressure.

Honest scope (W71 Plane A)
--------------------------

* Restart pressure is a caller-declared signal, not a substrate-
  state read. ``W71-L-HOSTED-V4-RESTART-DECLARED-CAP``,
  ``W71-L-HOSTED-V4-NO-SUBSTRATE-CAP``.
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
    HostedLogprobRouterV3,
    W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR,
    abstain_or_fuse_logprobs_v3,
)
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_LOGPROB_ROUTER_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v4.v1")
W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA: float = 1.0
W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR: float = 0.5


def abstain_or_fuse_logprobs_v4(
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
) -> dict[str, Any]:
    """V4 Bayesian fusion with restart-aware abstain and per-budget+
    restart tiebreak.

    When ``restart_pressure >= restart_pressure_floor``, V4 lowers
    the abstain-entropy-floor by ``restart_abstain_floor_delta``
    (so abstain fires at lower entropy) and further shrinks the
    effective top-k.
    """
    pressure = float(max(0.0, min(
        1.0, float(restart_pressure))))
    floor_active = bool(
        pressure >= float(restart_pressure_floor))
    effective_abstain_floor = float(
        max(0.0, float(abstain_entropy_floor)
            - (float(restart_abstain_floor_delta)
               if floor_active else 0.0)))
    # Per-budget tiebreak from V3 + per-restart shrink.
    budget_ratio = float(
        max(0.0, min(1.0, float(visible_token_budget)
                     / float(max(1, baseline_token_cost)))))
    restart_shrink = (
        0.5 if floor_active else 1.0)
    effective_top_k = max(
        2, int(round(
            float(top_k) * (0.5 + 0.5 * budget_ratio)
            * float(restart_shrink))))
    res = abstain_or_fuse_logprobs_v3(
        payloads,
        provider_trust=dict(provider_trust or {}),
        prior_alpha=float(prior_alpha),
        top_k=int(effective_top_k), floor=float(floor),
        tiebreak_entropy_floor=float(tiebreak_entropy_floor),
        abstain_entropy_floor=float(effective_abstain_floor),
        visible_token_budget=int(visible_token_budget),
        baseline_token_cost=int(baseline_token_cost))
    out = dict(res)
    out["schema"] = W71_HOSTED_LOGPROB_ROUTER_V4_SCHEMA_VERSION
    out["restart_pressure"] = float(round(pressure, 12))
    out["restart_floor_active"] = bool(floor_active)
    out["effective_abstain_entropy_floor"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v4"] = int(effective_top_k)
    # Rename V3 fusion_kind to V4 abstain marker if abstain.
    if str(out.get("fusion_kind", "")) == "abstain_v3":
        out["fusion_kind"] = "abstain_v4"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV4:
    inner_v3: HostedLogprobRouterV3 = dataclasses.field(
        default_factory=HostedLogprobRouterV3)
    restart_pressure_floor: float = (
        W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR)
    restart_abstain_floor_delta: float = (
        W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA)
    audit_v4: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W71_HOSTED_LOGPROB_ROUTER_V4_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "restart_pressure_floor": float(round(
                self.restart_pressure_floor, 12)),
            "restart_abstain_floor_delta": float(round(
                self.restart_abstain_floor_delta, 12)),
        })

    def fuse_v4(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            restart_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = abstain_or_fuse_logprobs_v4(
            payloads,
            provider_trust=dict(
                self.inner_v3.inner_v2.provider_trust),
            abstain_entropy_floor=float(
                self.inner_v3.abstain_entropy_floor),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=int(baseline_token_cost),
            restart_pressure=float(restart_pressure),
            restart_pressure_floor=float(
                self.restart_pressure_floor),
            restart_abstain_floor_delta=float(
                self.restart_abstain_floor_delta),
            **kwargs)
        self.audit_v4.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "visible_token_budget": int(visible_token_budget),
            "restart_pressure": float(round(
                float(restart_pressure), 12)),
            "restart_floor_active": bool(
                res.get("restart_floor_active", False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV4Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_bayesian: int
    n_tiebreak: int
    n_restart_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
            "n_restart_floor_active": int(
                self.n_restart_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v4_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v4_witness(
        router: HostedLogprobRouterV4,
) -> HostedLogprobRouterV4Witness:
    n_abst = sum(
        1 for e in router.audit_v4
        if str(e.get("kind", "")) in (
            "abstain_v3", "abstain_v4"))
    n_bayes = sum(
        1 for e in router.audit_v4
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit_v4
        if "tiebreak" in str(e.get("kind", "")))
    n_rfa = sum(
        1 for e in router.audit_v4
        if bool(e.get("restart_floor_active", False)))
    return HostedLogprobRouterV4Witness(
        schema=W71_HOSTED_LOGPROB_ROUTER_V4_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v4)),
        n_abstain=int(n_abst),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
        n_restart_floor_active=int(n_rfa),
    )


__all__ = [
    "W71_HOSTED_LOGPROB_ROUTER_V4_SCHEMA_VERSION",
    "W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA",
    "W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v4",
    "HostedLogprobRouterV4",
    "HostedLogprobRouterV4Witness",
    "emit_hosted_logprob_router_v4_witness",
]
