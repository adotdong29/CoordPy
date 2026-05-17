"""W73 H2 — Hosted Logprob Router V6 (Plane A).

Strictly extends W72's ``coordpy.hosted_logprob_router_v5``. V6
adds:

* **Replacement-aware abstain floor** — V5 had a rejoin-aware
  abstain floor delta. V6 *further* lowers the effective abstain
  threshold when caller-declared replacement pressure is high,
  so V6 abstains more aggressively when the team is recovering
  from agent replacement after contradiction.
* **Per-budget+restart+rejoin+replacement tiebreak** — V6 takes
  ``visible_token_budget``, ``restart_pressure``, ``rejoin_pressure``,
  and ``replacement_pressure`` hints and shrinks effective top-k
  under tight budgets, high restart pressure, high rejoin pressure,
  AND high replacement pressure.

Honest scope (W73 Plane A)
--------------------------

* Replacement pressure is a caller-declared signal, not a
  substrate-state read. ``W73-L-HOSTED-V6-REPLACEMENT-DECLARED-CAP``,
  ``W73-L-HOSTED-V6-NO-SUBSTRATE-CAP``.
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
    HostedLogprobRouterV5,
    W72_DEFAULT_LOGPROB_V5_REJOIN_FLOOR_DELTA,
    W72_DEFAULT_LOGPROB_V5_REJOIN_PRESSURE_FLOOR,
    abstain_or_fuse_logprobs_v5,
)
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_LOGPROB_ROUTER_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v6.v1")
W73_DEFAULT_LOGPROB_V6_REPLACEMENT_FLOOR_DELTA: float = 1.5
W73_DEFAULT_LOGPROB_V6_REPLACEMENT_PRESSURE_FLOOR: float = 0.5


def abstain_or_fuse_logprobs_v6(
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
) -> dict[str, Any]:
    """V6 Bayesian fusion with replacement-aware abstain and
    per-budget+restart+rejoin+replacement tiebreak.

    When ``replacement_pressure >= replacement_pressure_floor``,
    V6 lowers the abstain-entropy-floor by
    ``replacement_abstain_floor_delta`` (so abstain fires at lower
    entropy) and further shrinks the effective top-k.
    """
    rep_pressure = float(max(0.0, min(
        1.0, float(replacement_pressure))))
    rep_floor_active = bool(
        rep_pressure >= float(replacement_pressure_floor))
    rep_floor_delta = (
        float(replacement_abstain_floor_delta)
        if rep_floor_active else 0.0)
    effective_abstain_floor = float(
        max(0.0, float(abstain_entropy_floor) - rep_floor_delta))
    res = abstain_or_fuse_logprobs_v5(
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
            rejoin_abstain_floor_delta))
    rep_shrink = 0.5 if rep_floor_active else 1.0
    effective_top_k = max(
        2, int(round(
            float(res.get("effective_top_k_v5", top_k))
            * float(rep_shrink))))
    out = dict(res)
    out["schema"] = W73_HOSTED_LOGPROB_ROUTER_V6_SCHEMA_VERSION
    out["replacement_pressure"] = float(round(rep_pressure, 12))
    out["replacement_floor_active"] = bool(rep_floor_active)
    out["effective_abstain_entropy_floor"] = float(round(
        effective_abstain_floor, 12))
    out["effective_top_k_v6"] = int(effective_top_k)
    if str(out.get("fusion_kind", "")) in (
            "abstain_v3", "abstain_v4", "abstain_v5"):
        out["fusion_kind"] = "abstain_v6"
    return out


@dataclasses.dataclass
class HostedLogprobRouterV6:
    inner_v5: HostedLogprobRouterV5 = dataclasses.field(
        default_factory=HostedLogprobRouterV5)
    replacement_pressure_floor: float = (
        W73_DEFAULT_LOGPROB_V6_REPLACEMENT_PRESSURE_FLOOR)
    replacement_abstain_floor_delta: float = (
        W73_DEFAULT_LOGPROB_V6_REPLACEMENT_FLOOR_DELTA)
    audit_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W73_HOSTED_LOGPROB_ROUTER_V6_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "replacement_pressure_floor": float(round(
                self.replacement_pressure_floor, 12)),
            "replacement_abstain_floor_delta": float(round(
                self.replacement_abstain_floor_delta, 12)),
        })

    def fuse_v6(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            restart_pressure: float = 0.0,
            rejoin_pressure: float = 0.0,
            replacement_pressure: float = 0.0,
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = abstain_or_fuse_logprobs_v6(
            payloads,
            provider_trust=dict(
                self.inner_v5.inner_v4.inner_v3.inner_v2
                .provider_trust),
            abstain_entropy_floor=float(
                self.inner_v5.inner_v4.inner_v3
                .abstain_entropy_floor),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=int(baseline_token_cost),
            restart_pressure=float(restart_pressure),
            restart_pressure_floor=float(
                self.inner_v5.inner_v4.restart_pressure_floor),
            restart_abstain_floor_delta=float(
                self.inner_v5.inner_v4
                .restart_abstain_floor_delta),
            rejoin_pressure=float(rejoin_pressure),
            rejoin_pressure_floor=float(
                self.inner_v5.rejoin_pressure_floor),
            rejoin_abstain_floor_delta=float(
                self.inner_v5.rejoin_abstain_floor_delta),
            replacement_pressure=float(replacement_pressure),
            replacement_pressure_floor=float(
                self.replacement_pressure_floor),
            replacement_abstain_floor_delta=float(
                self.replacement_abstain_floor_delta),
            **kwargs)
        self.audit_v6.append({
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
            "replacement_floor_active": bool(
                res.get("replacement_floor_active", False)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV6Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_bayesian: int
    n_tiebreak: int
    n_replacement_floor_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
            "n_replacement_floor_active": int(
                self.n_replacement_floor_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v6_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v6_witness(
        router: HostedLogprobRouterV6,
) -> HostedLogprobRouterV6Witness:
    n_abst = sum(
        1 for e in router.audit_v6
        if str(e.get("kind", "")) in (
            "abstain_v3", "abstain_v4",
            "abstain_v5", "abstain_v6"))
    n_bayes = sum(
        1 for e in router.audit_v6
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit_v6
        if "tiebreak" in str(e.get("kind", "")))
    n_rfa = sum(
        1 for e in router.audit_v6
        if bool(e.get("replacement_floor_active", False)))
    return HostedLogprobRouterV6Witness(
        schema=W73_HOSTED_LOGPROB_ROUTER_V6_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v6)),
        n_abstain=int(n_abst),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
        n_replacement_floor_active=int(n_rfa),
    )


__all__ = [
    "W73_HOSTED_LOGPROB_ROUTER_V6_SCHEMA_VERSION",
    "W73_DEFAULT_LOGPROB_V6_REPLACEMENT_FLOOR_DELTA",
    "W73_DEFAULT_LOGPROB_V6_REPLACEMENT_PRESSURE_FLOOR",
    "abstain_or_fuse_logprobs_v6",
    "HostedLogprobRouterV6",
    "HostedLogprobRouterV6Witness",
    "emit_hosted_logprob_router_v6_witness",
]
