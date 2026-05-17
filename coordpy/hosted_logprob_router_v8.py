"""W75 H2 — Hosted Logprob Router V8 (Plane A).

Strictly extends W74's ``coordpy.hosted_logprob_router_v7``. V8
adds:

* **Compound-chain-aware abstain floor** — V7 had a compound-aware
  abstain floor delta. V8 *further* lowers the effective abstain
  threshold when caller-declared compound-chain pressure is high,
  so V8 abstains more aggressively when the team is recovering
  from a replacement-then-delayed-repair-then-rejoin compound-
  chain failure.
* **Per-budget+restart+rejoin+replacement+compound+chain tiebreak**
  — V8 takes ``visible_token_budget``, ``restart_pressure``,
  ``rejoin_pressure``, ``replacement_pressure``,
  ``compound_pressure``, AND ``compound_chain_pressure`` hints and
  shrinks effective top-k under tight budgets, high restart
  pressure, high rejoin pressure, high replacement pressure, high
  compound pressure, AND high compound-chain pressure.

Honest scope (W75 Plane A)
--------------------------

* Compound-chain pressure is a caller-declared signal, not a
  substrate-state read. ``W75-L-HOSTED-V8-COMPOUND-CHAIN-DECLARED-
  CAP``, ``W75-L-HOSTED-V8-NO-SUBSTRATE-CAP``.
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
    W71_DEFAULT_LOGPROB_V4_RESTART_FLOOR_DELTA,
    W71_DEFAULT_LOGPROB_V4_RESTART_PRESSURE_FLOOR,
)
from .hosted_logprob_router_v5 import (
    W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA,
    W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR,
)
from .hosted_logprob_router_v6 import (
    W73_DEFAULT_LOGPROB_V6_REPLACEMENT_FLOOR_DELTA,
    W73_DEFAULT_LOGPROB_V6_REPLACEMENT_PRESSURE_FLOOR,
)
from .hosted_logprob_router_v7 import (
    HostedLogprobRouterV7,
    W74_DEFAULT_LOGPROB_V7_COMPOUND_FLOOR_DELTA,
    W74_DEFAULT_LOGPROB_V7_COMPOUND_PRESSURE_FLOOR,
    abstain_or_fuse_logprobs_v7,
)
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_LOGPROB_ROUTER_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v8.v1")
W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_FLOOR_DELTA: float = 1.4
W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_PRESSURE_FLOOR: float = 0.5


def abstain_or_fuse_logprobs_v8(
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
        replacement_pressure: float = 0.0,
        replacement_pressure_floor: float = (
            W73_DEFAULT_LOGPROB_V6_REPLACEMENT_PRESSURE_FLOOR),
        replacement_abstain_floor_delta: float = (
            W73_DEFAULT_LOGPROB_V6_REPLACEMENT_FLOOR_DELTA),
        compound_pressure: float = 0.0,
        compound_pressure_floor: float = (
            W74_DEFAULT_LOGPROB_V7_COMPOUND_PRESSURE_FLOOR),
        compound_abstain_floor_delta: float = (
            W74_DEFAULT_LOGPROB_V7_COMPOUND_FLOOR_DELTA),
        compound_chain_pressure: float = 0.0,
        compound_chain_pressure_floor: float = (
            W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_PRESSURE_FLOOR),
        compound_chain_abstain_floor_delta: float = (
            W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_FLOOR_DELTA),
) -> dict[str, Any]:
    """V8 Bayesian fusion with compound-chain-aware abstain and
    per-budget+restart+rejoin+replacement+compound+chain tiebreak.

    When ``compound_chain_pressure >= compound_chain_pressure_
    floor``, V8 lowers the abstain-entropy-floor by
    ``compound_chain_abstain_floor_delta`` (so abstain fires at
    lower entropy) and further shrinks the effective top-k.
    """
    chain_pressure = float(max(0.0, min(
        1.0, float(compound_chain_pressure))))
    chain_floor_active = bool(
        chain_pressure >= float(compound_chain_pressure_floor))
    chain_floor_delta = (
        float(compound_chain_abstain_floor_delta)
        if chain_floor_active else 0.0)
    effective_abstain_floor = float(
        max(0.0,
            float(abstain_entropy_floor) - chain_floor_delta))
    res = abstain_or_fuse_logprobs_v7(
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
            restart_abstain_floor_delta),
        rejoin_pressure=float(rejoin_pressure),
        rejoin_pressure_floor=float(rejoin_pressure_floor),
        rejoin_abstain_floor_delta=float(
            rejoin_abstain_floor_delta),
        replacement_pressure=float(replacement_pressure),
        replacement_pressure_floor=float(
            replacement_pressure_floor),
        replacement_abstain_floor_delta=float(
            replacement_abstain_floor_delta),
        compound_pressure=float(compound_pressure),
        compound_pressure_floor=float(
            compound_pressure_floor),
        compound_abstain_floor_delta=float(
            compound_abstain_floor_delta))
    chain_shrink = 0.5 if chain_floor_active else 1.0
    effective_top_k = max(
        2, int(round(
            float(res.get("effective_top_k_v7", top_k))
            * float(chain_shrink))))
    out = dict(res)
    out["schema"] = W75_HOSTED_LOGPROB_ROUTER_V8_SCHEMA_VERSION
    out["compound_chain_pressure"] = float(round(
        chain_pressure, 12))
    out["compound_chain_floor_active"] = bool(
        chain_floor_active)
    out["effective_abstain_entropy_floor"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v8"] = int(effective_top_k)
    if str(out.get("fusion_kind", "")) in (
            "abstain_v3", "abstain_v4", "abstain_v5",
            "abstain_v6", "abstain_v7"):
        out["fusion_kind"] = "abstain_v8"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV8:
    inner_v7: HostedLogprobRouterV7 = dataclasses.field(
        default_factory=HostedLogprobRouterV7)
    compound_chain_pressure_floor: float = (
        W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_PRESSURE_FLOOR)
    compound_chain_abstain_floor_delta: float = (
        W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_FLOOR_DELTA)
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W75_HOSTED_LOGPROB_ROUTER_V8_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_pressure_floor": float(round(
                self.compound_chain_pressure_floor, 12)),
            "compound_chain_abstain_floor_delta": float(round(
                self.compound_chain_abstain_floor_delta, 12)),
        })

    def fuse_v8(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            restart_pressure: float = 0.0,
            rejoin_pressure: float = 0.0,
            replacement_pressure: float = 0.0,
            compound_pressure: float = 0.0,
            compound_chain_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = abstain_or_fuse_logprobs_v8(
            payloads,
            provider_trust=dict(
                self.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .inner_v2.provider_trust),
            abstain_entropy_floor=float(
                self.inner_v7.inner_v6.inner_v5.inner_v4.inner_v3
                .abstain_entropy_floor),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=int(baseline_token_cost),
            restart_pressure=float(restart_pressure),
            restart_pressure_floor=float(
                self.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_pressure_floor),
            restart_abstain_floor_delta=float(
                self.inner_v7.inner_v6.inner_v5.inner_v4
                .restart_abstain_floor_delta),
            rejoin_pressure=float(rejoin_pressure),
            rejoin_pressure_floor=float(
                self.inner_v7.inner_v6.inner_v5
                .rejoin_pressure_floor),
            rejoin_abstain_floor_delta=float(
                self.inner_v7.inner_v6.inner_v5
                .rejoin_abstain_floor_delta),
            replacement_pressure=float(replacement_pressure),
            replacement_pressure_floor=float(
                self.inner_v7.inner_v6.replacement_pressure_floor),
            replacement_abstain_floor_delta=float(
                self.inner_v7.inner_v6
                .replacement_abstain_floor_delta),
            compound_pressure=float(compound_pressure),
            compound_pressure_floor=float(
                self.inner_v7.compound_pressure_floor),
            compound_abstain_floor_delta=float(
                self.inner_v7.compound_abstain_floor_delta),
            compound_chain_pressure=float(
                compound_chain_pressure),
            compound_chain_pressure_floor=float(
                self.compound_chain_pressure_floor),
            compound_chain_abstain_floor_delta=float(
                self.compound_chain_abstain_floor_delta),
            **kwargs)
        self.audit_v8.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "visible_token_budget": int(visible_token_budget),
            "restart_pressure": float(round(
                float(restart_pressure), 12)),
            "rejoin_pressure": float(round(
                float(rejoin_pressure), 12)),
            "replacement_pressure": float(round(
                float(replacement_pressure), 12)),
            "compound_pressure": float(round(
                float(compound_pressure), 12)),
            "compound_chain_pressure": float(round(
                float(compound_chain_pressure), 12)),
            "compound_chain_floor_active": bool(
                res.get("compound_chain_floor_active", False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV8Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_bayesian: int
    n_tiebreak: int
    n_compound_chain_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
            "n_compound_chain_floor_active": int(
                self.n_compound_chain_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v8_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v8_witness(
        router: HostedLogprobRouterV8,
) -> HostedLogprobRouterV8Witness:
    n_abst = sum(
        1 for e in router.audit_v8
        if str(e.get("kind", "")) in (
            "abstain_v3", "abstain_v4",
            "abstain_v5", "abstain_v6",
            "abstain_v7", "abstain_v8"))
    n_bayes = sum(
        1 for e in router.audit_v8
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit_v8
        if "tiebreak" in str(e.get("kind", "")))
    n_cfa = sum(
        1 for e in router.audit_v8
        if bool(e.get("compound_chain_floor_active", False)))
    return HostedLogprobRouterV8Witness(
        schema=W75_HOSTED_LOGPROB_ROUTER_V8_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v8)),
        n_abstain=int(n_abst),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
        n_compound_chain_floor_active=int(n_cfa),
    )


__all__ = [
    "W75_HOSTED_LOGPROB_ROUTER_V8_SCHEMA_VERSION",
    "W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_FLOOR_DELTA",
    "W75_DEFAULT_LOGPROB_V8_COMPOUND_CHAIN_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v8",
    "HostedLogprobRouterV8",
    "HostedLogprobRouterV8Witness",
    "emit_hosted_logprob_router_v8_witness",
]
