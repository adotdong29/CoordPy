"""W69 H2 — Hosted Logprob Router V2 (Plane A).

Strictly extends W68's ``coordpy.hosted_logprob_router``. V2 adds:

* **Bayesian fusion over top-k logprobs** — V1 used weighted-mean;
  V2 supports a Dirichlet-style Bayesian update with a uniform
  prior over the shared top-k vocabulary.
* **Per-provider trust calibration** — V2 supports a per-provider
  trust prior in [0, 1] that scales the contribution.
* **Tie-breaker fallback** — when fused entropies exceed a floor
  V2 falls back to a text-only quorum + tiebreak by Bayesian
  posterior.

Honest scope (W69 Plane A)
--------------------------

* Logprob fusion does NOT recover hidden state.
  ``W69-L-HOSTED-LOGPROB-V2-NOT-HIDDEN-CAP``.
* Trust calibration scores are caller-supplied.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from .hosted_logprob_router import (
    TopKLogprobsPayload,
    W68_DEFAULT_LOGPROB_FUSION_FLOOR,
    W68_DEFAULT_TOP_K,
    fuse_logprobs,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router_v2.v1")
W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR: float = 2.0


def _entropy(probs: Sequence[float]) -> float:
    s = 0.0
    for p in probs:
        if float(p) > 0.0:
            s -= float(p) * math.log(float(p) + 1e-12)
    return float(s)


def bayesian_fuse_logprobs_v2(
        payloads: Sequence[TopKLogprobsPayload], *,
        provider_trust: dict[str, float] | None = None,
        prior_alpha: float = 1.0,
        top_k: int = W68_DEFAULT_TOP_K,
        floor: float = W68_DEFAULT_LOGPROB_FUSION_FLOOR,
        tiebreak_entropy_floor: float = (
            W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR),
) -> dict[str, Any]:
    """Bayesian-fusion over top-k logprobs. Returns the V2 fused
    distribution + audit + tiebreak fallback if entropy is too
    high."""
    if not payloads:
        return {
            "schema": W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
            "fusion_kind": "no_payloads",
            "fused_distribution": [],
            "entropy": 0.0,
        }
    # Run V1 weighted-mean for baseline fallback.
    v1 = fuse_logprobs(
        payloads, top_k=int(top_k), floor=float(floor))
    trust = dict(provider_trust or {})
    # Gather shared top-k tokens (intersection).
    sets = [set(p.as_dict().keys()) for p in payloads]
    shared = set.intersection(*sets) if sets else set()
    if not shared:
        return {
            "schema": W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
            "fusion_kind": "no_shared_top_k",
            "fused_distribution": v1.get(
                "fused_distribution", []),
            "entropy": float(_entropy([
                p for _, p in v1.get("fused_distribution", [])])),
        }
    # Bayesian-update posterior with Dirichlet-style prior.
    counts = {t: float(prior_alpha) for t in shared}
    for p in payloads:
        tr = float(trust.get(p.provider, 1.0))
        d = p.as_dict()
        # Convert log-probs to weights.
        max_lp = max(float(d[t]) for t in shared)
        for t in shared:
            counts[t] += float(tr) * float(
                math.exp(float(d[t]) - max_lp))
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {
            "schema": W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
            "fusion_kind": "zero_total",
            "fused_distribution": [],
            "entropy": 0.0,
        }
    probs = {t: float(counts[t] / total) for t in counts}
    fused = sorted(
        probs.items(), key=lambda kv: -kv[1])[:int(top_k)]
    entropy = float(_entropy([p for _, p in fused]))
    if entropy > float(tiebreak_entropy_floor):
        # Tiebreak: fall back to v1 weighted-mean.
        return {
            "schema": W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
            "fusion_kind": "tiebreak_to_v1_weighted_mean",
            "fused_distribution": v1.get(
                "fused_distribution", []),
            "entropy": float(entropy),
            "n_shared_top_k": int(len(shared)),
        }
    return {
        "schema": W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
        "fusion_kind": "bayesian_dirichlet_v2",
        "fused_distribution": [
            [str(t), float(round(p, 12))] for t, p in fused],
        "entropy": float(entropy),
        "n_shared_top_k": int(len(shared)),
    }


@dataclasses.dataclass
class HostedLogprobRouterV2:
    provider_trust: dict[str, float] = dataclasses.field(
        default_factory=dict)
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
            "kind": "hosted_logprob_router_v2",
            "provider_trust": {
                k: float(round(v, 12))
                for k, v in sorted(self.provider_trust.items())},
        })

    def fuse(
            self, payloads: Sequence[TopKLogprobsPayload],
            **kwargs: Any,
    ) -> dict[str, Any]:
        res = bayesian_fuse_logprobs_v2(
            payloads,
            provider_trust=dict(self.provider_trust),
            **kwargs)
        self.audit.append({
            "kind": str(res.get("fusion_kind", "")),
            "n_payloads": int(len(payloads)),
            "entropy": float(res.get("entropy", 0.0)),
        })
        return res


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterV2Witness:
    schema: str
    router_cid: str
    n_fusions: int
    n_bayesian: int
    n_tiebreak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusions": int(self.n_fusions),
            "n_bayesian": int(self.n_bayesian),
            "n_tiebreak": int(self.n_tiebreak),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_v2_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_v2_witness(
        router: HostedLogprobRouterV2,
) -> HostedLogprobRouterV2Witness:
    n_bayes = sum(
        1 for e in router.audit
        if str(e.get("kind", "")) == "bayesian_dirichlet_v2")
    n_tie = sum(
        1 for e in router.audit
        if "tiebreak" in str(e.get("kind", "")))
    return HostedLogprobRouterV2Witness(
        schema=W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusions=int(len(router.audit)),
        n_bayesian=int(n_bayes),
        n_tiebreak=int(n_tie),
    )


__all__ = [
    "W69_HOSTED_LOGPROB_ROUTER_V2_SCHEMA_VERSION",
    "W69_DEFAULT_LOGPROB_V2_TIEBREAK_ENTROPY_FLOOR",
    "bayesian_fuse_logprobs_v2",
    "HostedLogprobRouterV2",
    "HostedLogprobRouterV2Witness",
    "emit_hosted_logprob_router_v2_witness",
]
