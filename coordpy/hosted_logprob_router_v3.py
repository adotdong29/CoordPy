"""W70 H2 — Hosted Logprob Router V3 (Plane A).

Strictly extends W69's ``coordpy.hosted_logprob_router_v2``. V3
adds:

* **Abstain-when-disagree fusion** — when fused entropy exceeds
  the V3 disagree-floor, V3 returns an explicit ``abstain`` rather
  than falling back to V1 weighted-mean.
* **Per-budget tiebreak** — V3 takes a ``visible_token_budget``
  hint and selects a more conservative top-k under tight budgets.

Honest scope (W70 Plane A)
--------------------------

* Abstain is a *signal*, not a substrate-state read.
* ``W70-L-HOSTED-V3-NO-SUBSTRATE-CAP`` carries forward.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from .hosted_logprob_router import (
    TopKLogprobsPayload,
    W68_DEFAULT_LOGPROB_FUSION_FLOOR,
    W68_DEFAULT_TOP_K,
)
from .hosted_logprob_router_v2 import (
    HostedLogprobRouterV2,
    W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR,
    bayesian_fuse_logprobs_v2,
)
from .tiny_substrate_v3 import _sha256_hex


W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v3.v1")
W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR: float = 2.5


def _entropy(probs: Sequence[float]) -> float:
    s = 0.0
    for p in probs:
        if float(p) > 0.0:
            s -= float(p) * math.log(float(p) + 1e-12)
    return float(s)


def abstain_or_fuse_logprobs_v3(
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
) -> dict[str, Any]:
    """Bayesian fusion with abstain-when-disagree and per-budget
    tiebreak."""
    if not payloads:
        return {
            "schema":
                W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION,
            "fusion_kind": "no_payloads",
            "fused_distribution": [],
            "entropy": 0.0,
        }
    # Per-budget top-k clamp.
    budget_ratio = float(
        max(0.0, min(1.0, float(visible_token_budget)
                     / float(max(1, baseline_token_cost)))))
    effective_top_k = max(
        2, int(round(float(top_k) * (0.5 + 0.5 * budget_ratio))))
    res = bayesian_fuse_logprobs_v2(
        payloads,
        provider_trust=dict(provider_trust or {}),
        prior_alpha=float(prior_alpha),
        top_k=int(effective_top_k), floor=float(floor),
        tiebreak_entropy_floor=float(tiebreak_entropy_floor),
    )
    ent = float(res.get("entropy", 0.0))
    if ent > float(abstain_entropy_floor):
        # Explicit abstain — V3 refuses to make a decision.
        return {
            "schema":
                W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION,
            "fusion_kind": "abstain_v3",
            "entropy": float(ent),
            "abstain_entropy_floor": float(abstain_entropy_floor),
            "fused_distribution": [],
            "effective_top_k": int(effective_top_k),
        }
    return {
        "schema":
            W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION,
        "fusion_kind": str(res.get("fusion_kind", "")),
        "fused_distribution": res.get(
            "fused_distribution", []),
        "entropy": float(ent),
        "effective_top_k": int(effective_top_k),
        "n_shared_top_k": int(res.get("n_shared_top_k", 0)),
    }


@dataclasses.dataclass
class HostedLogprobRouterV3:
    inner_v2: HostedLogprobRouterV2 = dataclasses.field(
        default_factory=HostedLogprobRouterV2)
    abstain_entropy_floor: float = (
        W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR)
    audit_v3: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "abstain_entropy_floor": float(round(
                self.abstain_entropy_floor, 12)),
        })

    def fuse_v3(
            self, payloads: Sequence[TopKLogprobsPayload],
            *,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = abstain_or_fuse_logprobs_v3(
            payloads,
            provider_trust=dict(self.inner_v2.provider_trust),
            abstain_entropy_floor=float(
                self.abstain_entropy_floor),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=int(baseline_token_cost),
            **kwargs)
        self.audit_v3.append({
            "kind": str(res.get("fusion_kind", "")),
            "entropy": float(res.get("entropy", 0.0)),
            "n_payloads": int(len(payloads)),
            "visible_token_budget": int(visible_token_budget),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV3Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_abstain: int
    n_bayesian: int
    n_tiebreak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_abstain": int(self.n_abstain),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v3_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v3_witness(
        router: HostedLogprobRouterV3,
) -> HostedLogprobRouterV3Witness:
    n_abst = sum(
        1 for e in router.audit_v3
        if str(e.get("kind", "")) == "abstain_v3")
    n_bayes = sum(
        1 for e in router.audit_v3
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit_v3
        if "tiebreak" in str(e.get("kind", "")))
    return HostedLogprobRouterV3Witness(
        schema=W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit_v3)),
        n_abstain=int(n_abst),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
    )


__all__ = [
    "W70_HOSTED_LOGPROB_ROUTER_V3_SCHEMA_VERSION",
    "W70_DEFAULT_LOGPROB_V3_ABSTAIN_ENTROPY_FLOOR",
    "abstain_or_fuse_logprobs_v3",
    "HostedLogprobRouterV3",
    "HostedLogprobRouterV3Witness",
    "emit_hosted_logprob_router_v3_witness",
]
