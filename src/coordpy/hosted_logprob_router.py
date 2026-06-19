"""W68 H2 — Hosted Logprob Router (Plane A).

Logprob-aware fallback router for hosted-API providers that expose
top-k logprobs at the HTTP surface (e.g. some OpenRouter routes,
Groq, OpenAI). Implements the **honest** logprob-fusion path:

* When two providers both reply with top-k logprobs, the router
  fuses them into a consensus distribution via weighted-mean of
  the log-probabilities clipped to the shared top-k vocabulary.
* When only one provider replies with logprobs, the router falls
  back to its single distribution.
* When no provider replies with logprobs, the router falls back
  to a text-only quorum.

Honest scope (W68 Plane A)
--------------------------

* This is a **planning + fusion** layer; the router does not call
  hosted APIs. The caller passes in the logprob payloads it
  already obtained. ``W68-L-HOSTED-LOGPROB-CALLER-FETCHED-CAP``.
* Logprob fusion does NOT recover hidden state. It is an honest
  text-surface fusion of distributions over visible tokens.
  ``W68-L-HOSTED-LOGPROB-NOT-HIDDEN-CAP``.
* Top-k is bounded by the providers' shared exposed top-k (usually
  ≤ 20).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION: str = (
    "coordpy.hosted_logprob_router.v1")
W68_DEFAULT_TOP_K: int = 20
W68_DEFAULT_LOGPROB_FUSION_FLOOR: float = -20.0


@dataclasses.dataclass(frozen=True)
class TopKLogprobsPayload:
    provider: str
    token_to_logprob: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": str(self.provider),
            "token_to_logprob": [
                [str(t), float(round(lp, 12))]
                for t, lp in self.token_to_logprob],
        }

    def as_dict(self) -> dict[str, float]:
        return {str(t): float(lp)
                for t, lp in self.token_to_logprob}

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "top_k_logprobs_payload",
            "payload": self.to_dict()})


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return float(m + math.log(
        sum(math.exp(float(x) - float(m)) for x in xs)))


def fuse_logprobs(
        payloads: Sequence[TopKLogprobsPayload], *,
        weights: Sequence[float] | None = None,
        top_k: int = W68_DEFAULT_TOP_K,
        floor: float = W68_DEFAULT_LOGPROB_FUSION_FLOOR,
) -> dict[str, Any]:
    """Honest top-k logprob fusion across hosted providers."""
    if not payloads:
        return {
            "schema": W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
            "kind": "fuse_logprobs",
            "fused_distribution": [],
            "fusion_kind": "abstain",
            "rationale": "no_payloads",
        }
    if weights is None:
        weights = [1.0] * len(payloads)
    weights = [float(w) for w in weights]
    sw = float(sum(weights))
    if sw <= 0.0:
        weights = [1.0] * len(payloads)
        sw = float(len(payloads))
    # Shared top-k = intersection of tokens across all providers.
    token_sets = [set(p.as_dict().keys()) for p in payloads]
    shared = set(token_sets[0])
    for s in token_sets[1:]:
        shared &= s
    if not shared:
        # Fall back to single best provider (highest max-prob token).
        best_idx = 0
        best_val = float("-inf")
        for i, p in enumerate(payloads):
            d = p.as_dict()
            if d:
                m = max(d.values())
                if m > best_val:
                    best_val = m
                    best_idx = int(i)
        d = payloads[best_idx].as_dict()
        sorted_tokens = sorted(
            d.items(), key=lambda kv: -float(kv[1]))[:int(top_k)]
        return {
            "schema": W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
            "kind": "fuse_logprobs",
            "fused_distribution": [
                [str(t), float(round(lp, 12))]
                for t, lp in sorted_tokens],
            "fusion_kind": "single_provider_no_overlap",
            "rationale": "no_shared_top_k",
            "provider_used": payloads[best_idx].provider,
        }
    fused: dict[str, float] = {}
    for tok in shared:
        # Weighted-mean of log-probabilities (clipped at floor) on
        # shared tokens. This is honest text-surface fusion.
        s = 0.0
        for w, p in zip(weights, payloads):
            d = p.as_dict()
            lp = float(max(float(d.get(tok, float(floor))),
                           float(floor)))
            s += float(w) * lp
        fused[tok] = float(s / sw)
    sorted_tokens = sorted(
        fused.items(), key=lambda kv: -float(kv[1]))[:int(top_k)]
    return {
        "schema": W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
        "kind": "fuse_logprobs",
        "fused_distribution": [
            [str(t), float(round(lp, 12))]
            for t, lp in sorted_tokens],
        "fusion_kind": "weighted_mean_shared_top_k",
        "rationale": "shared_top_k_fused",
        "n_providers": int(len(payloads)),
        "n_shared_tokens": int(len(shared)),
    }


def text_only_quorum(
        responses: Sequence[str],
) -> dict[str, Any]:
    """Text-only quorum (plurality on exact string)."""
    if not responses:
        return {
            "schema": W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
            "kind": "text_only_quorum",
            "decision": None,
            "votes": {},
        }
    votes: dict[str, int] = {}
    for r in responses:
        votes[str(r)] = int(votes.get(str(r), 0)) + 1
    winner = max(votes.items(), key=lambda kv: kv[1])
    return {
        "schema": W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
        "kind": "text_only_quorum",
        "decision": str(winner[0]),
        "votes": {str(k): int(v) for k, v in votes.items()},
        "winning_count": int(winner[1]),
    }


@dataclasses.dataclass
class HostedLogprobRouter:
    top_k: int = W68_DEFAULT_TOP_K
    floor: float = W68_DEFAULT_LOGPROB_FUSION_FLOOR
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
            "kind": "hosted_logprob_router",
            "top_k": int(self.top_k),
            "floor": float(round(self.floor, 12)),
        })

    def fuse(
            self,
            logprob_payloads: Sequence[TopKLogprobsPayload],
            *, weights: Sequence[float] | None = None,
            text_only_responses: Sequence[str] = (),
    ) -> dict[str, Any]:
        if logprob_payloads:
            result = fuse_logprobs(
                logprob_payloads, weights=weights,
                top_k=int(self.top_k),
                floor=float(self.floor))
            self.audit.append({
                "kind": str(result["fusion_kind"]),
                "n_providers": int(len(logprob_payloads)),
            })
            return result
        text = text_only_quorum(list(text_only_responses))
        self.audit.append({
            "kind": "text_only_quorum",
            "n_responses": int(len(text_only_responses)),
        })
        return text


@dataclasses.dataclass(frozen=True)
class HostedLogprobRouterWitness:
    schema: str
    router_cid: str
    n_fusion_calls: int
    n_text_only_quorum_calls: int
    fusion_kinds_seen: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "router_cid": str(self.router_cid),
            "n_fusion_calls": int(self.n_fusion_calls),
            "n_text_only_quorum_calls": int(
                self.n_text_only_quorum_calls),
            "fusion_kinds_seen": list(self.fusion_kinds_seen),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_logprob_router_witness",
            "witness": self.to_dict()})


def emit_hosted_logprob_router_witness(
        router: HostedLogprobRouter,
) -> HostedLogprobRouterWitness:
    fusion = 0
    text = 0
    kinds: set[str] = set()
    for e in router.audit:
        k = str(e.get("kind", ""))
        kinds.add(k)
        if k == "text_only_quorum":
            text += 1
        else:
            fusion += 1
    return HostedLogprobRouterWitness(
        schema=W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION,
        router_cid=str(router.cid()),
        n_fusion_calls=int(fusion),
        n_text_only_quorum_calls=int(text),
        fusion_kinds_seen=tuple(sorted(kinds)),
    )


__all__ = [
    "W68_HOSTED_LOGPROB_ROUTER_SCHEMA_VERSION",
    "W68_DEFAULT_TOP_K",
    "W68_DEFAULT_LOGPROB_FUSION_FLOOR",
    "TopKLogprobsPayload",
    "fuse_logprobs",
    "text_only_quorum",
    "HostedLogprobRouter",
    "HostedLogprobRouterWitness",
    "emit_hosted_logprob_router_witness",
]
