"""W68 H1 — Hosted Router Controller (Plane A).

The first load-bearing W68 hosted-control-plane mechanism. Provides
a pure-Python honest provider-routing graph over OpenRouter, Groq,
OpenAI-compatible HTTP surfaces, and any other text-only hosted
backend. **Plane A only — does NOT pierce the hosted substrate
boundary.**

Honest scope (W68 Plane A)
--------------------------

* The router is a **planning** layer. It does not itself call
  hosted APIs; it accepts a registry of providers + per-provider
  capability flags + per-provider cost/latency estimates and
  produces a routing decision plus an audit trail.
* The router does NOT claim hidden-state / KV / attention access
  at the hosted surface. ``W68-L-HOSTED-NO-SUBSTRATE-CAP``.
* The cost/latency estimates are caller-supplied (the router does
  not measure live traffic). ``W68-L-HOSTED-ESTIMATES-CALLER-CAP``.
* Routing is deterministic on (registry CID, request CID).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_ROUTER_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller.v1")

# Honest capability tiers at the HOSTED surface only.
W68_HOSTED_TIER_TEXT_ONLY: str = "text_only"
W68_HOSTED_TIER_LOGPROBS: str = "logprobs"
W68_HOSTED_TIER_PREFIX_CACHE: str = "prefix_cache"
W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE: str = (
    "logprobs_and_prefix_cache")
W68_HOSTED_TIERS: tuple[str, ...] = (
    W68_HOSTED_TIER_TEXT_ONLY,
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
)


@dataclasses.dataclass(frozen=True)
class HostedProvider:
    """A hosted-API provider description. The fields are caller-
    declared; the router does not validate them against the live
    hosted API."""
    name: str
    base_url: str
    tier: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    p50_latency_ms: float
    data_policy: str   # e.g., "no_log", "log", "train"
    accepts_logprobs: bool
    accepts_prefix_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W68_HOSTED_ROUTER_SCHEMA_VERSION,
            "name": str(self.name),
            "base_url": str(self.base_url),
            "tier": str(self.tier),
            "cost_per_1k_input": float(round(
                self.cost_per_1k_input, 12)),
            "cost_per_1k_output": float(round(
                self.cost_per_1k_output, 12)),
            "p50_latency_ms": float(round(
                self.p50_latency_ms, 12)),
            "data_policy": str(self.data_policy),
            "accepts_logprobs": bool(self.accepts_logprobs),
            "accepts_prefix_cache": bool(
                self.accepts_prefix_cache),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider",
            "provider": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class HostedProviderRegistry:
    """A content-addressed registry of hosted providers. The
    router's decisions are CID-stable on the registry."""
    providers: tuple[HostedProvider, ...]

    def by_name(self) -> dict[str, HostedProvider]:
        return {p.name: p for p in self.providers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W68_HOSTED_ROUTER_SCHEMA_VERSION,
            "kind": "hosted_provider_registry",
            "providers": [
                p.to_dict() for p in sorted(
                    self.providers, key=lambda x: x.name)],
        }

    def cid(self) -> str:
        return _sha256_hex(self.to_dict())


def default_hosted_registry() -> HostedProviderRegistry:
    """A sample registry with the providers commonly available at
    the HTTP surface. None of these are connected; the registry is
    declarative."""
    return HostedProviderRegistry(providers=(
        HostedProvider(
            name="openrouter_free",
            base_url="https://openrouter.ai/api/v1",
            tier=W68_HOSTED_TIER_PREFIX_CACHE,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            p50_latency_ms=1200.0,
            data_policy="log",
            accepts_logprobs=False,
            accepts_prefix_cache=True),
        HostedProvider(
            name="openrouter_paid",
            base_url="https://openrouter.ai/api/v1",
            tier=W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
            cost_per_1k_input=0.50,
            cost_per_1k_output=1.50,
            p50_latency_ms=800.0,
            data_policy="no_log",
            accepts_logprobs=True,
            accepts_prefix_cache=True),
        HostedProvider(
            name="groq_free",
            base_url="https://api.groq.com/openai/v1",
            tier=W68_HOSTED_TIER_LOGPROBS,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            p50_latency_ms=300.0,
            data_policy="log",
            accepts_logprobs=True,
            accepts_prefix_cache=False),
        HostedProvider(
            name="openai_paid",
            base_url="https://api.openai.com/v1",
            tier=W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
            cost_per_1k_input=2.50,
            cost_per_1k_output=10.0,
            p50_latency_ms=600.0,
            data_policy="no_log",
            accepts_logprobs=True,
            accepts_prefix_cache=True),
        HostedProvider(
            name="ollama_local",
            base_url="http://localhost:11434",
            tier=W68_HOSTED_TIER_TEXT_ONLY,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            p50_latency_ms=2000.0,
            data_policy="no_log",
            accepts_logprobs=False,
            accepts_prefix_cache=False),
    ))


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequest:
    request_cid: str
    input_tokens: int
    expected_output_tokens: int
    require_logprobs: bool
    require_prefix_cache: bool
    data_policy_required: str   # any of "any", "no_log", "no_train"
    max_latency_ms: float
    max_cost_usd: float


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecision:
    schema: str
    request_cid: str
    chosen_provider: str | None
    rationale: str
    eligible_providers: tuple[str, ...]
    rejected_providers: tuple[tuple[str, str], ...]
    estimated_cost_usd: float
    estimated_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "request_cid": str(self.request_cid),
            "chosen_provider": (
                None if self.chosen_provider is None
                else str(self.chosen_provider)),
            "rationale": str(self.rationale),
            "eligible_providers": list(self.eligible_providers),
            "rejected_providers": [
                [str(n), str(r)]
                for n, r in self.rejected_providers],
            "estimated_cost_usd": float(round(
                self.estimated_cost_usd, 12)),
            "estimated_latency_ms": float(round(
                self.estimated_latency_ms, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision",
            "decision": self.to_dict()})


@dataclasses.dataclass
class HostedRouterController:
    registry: HostedProviderRegistry
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_HOSTED_ROUTER_SCHEMA_VERSION,
            "kind": "hosted_router_controller",
            "registry_cid": str(self.registry.cid()),
        })

    def _est_cost(
            self, p: HostedProvider, req: HostedRoutingRequest,
    ) -> float:
        return float(
            (float(req.input_tokens)
             * float(p.cost_per_1k_input) / 1000.0)
            + (float(req.expected_output_tokens)
               * float(p.cost_per_1k_output) / 1000.0))

    def decide(
            self, req: HostedRoutingRequest,
    ) -> HostedRoutingDecision:
        eligible: list[HostedProvider] = []
        rejected: list[tuple[str, str]] = []
        for p in self.registry.providers:
            if (req.require_logprobs
                    and not p.accepts_logprobs):
                rejected.append((p.name, "no_logprobs"))
                continue
            if (req.require_prefix_cache
                    and not p.accepts_prefix_cache):
                rejected.append((p.name, "no_prefix_cache"))
                continue
            if (str(req.data_policy_required) == "no_log"
                    and p.data_policy != "no_log"):
                rejected.append((p.name, "data_policy"))
                continue
            if (str(req.data_policy_required) == "no_train"
                    and p.data_policy in ("train",)):
                rejected.append((p.name, "data_policy"))
                continue
            est_lat = float(p.p50_latency_ms)
            if est_lat > float(req.max_latency_ms):
                rejected.append((p.name, "max_latency"))
                continue
            est_cost = self._est_cost(p, req)
            if est_cost > float(req.max_cost_usd):
                rejected.append((p.name, "max_cost"))
                continue
            eligible.append(p)
        if not eligible:
            decision = HostedRoutingDecision(
                schema=W68_HOSTED_ROUTER_SCHEMA_VERSION,
                request_cid=str(req.request_cid),
                chosen_provider=None,
                rationale="no_eligible_provider",
                eligible_providers=(),
                rejected_providers=tuple(rejected),
                estimated_cost_usd=0.0,
                estimated_latency_ms=0.0,
            )
            self.audit.append({
                "request_cid": str(req.request_cid),
                "outcome": "no_eligible_provider",
            })
            return decision
        # Lexicographic preference: cheapest then fastest.
        def key(p: HostedProvider) -> tuple[float, float, str]:
            return (
                self._est_cost(p, req),
                float(p.p50_latency_ms),
                str(p.name))
        eligible_sorted = sorted(eligible, key=key)
        winner = eligible_sorted[0]
        decision = HostedRoutingDecision(
            schema=W68_HOSTED_ROUTER_SCHEMA_VERSION,
            request_cid=str(req.request_cid),
            chosen_provider=str(winner.name),
            rationale="cheapest_then_fastest",
            eligible_providers=tuple(
                p.name for p in eligible_sorted),
            rejected_providers=tuple(rejected),
            estimated_cost_usd=float(self._est_cost(
                winner, req)),
            estimated_latency_ms=float(winner.p50_latency_ms),
        )
        self.audit.append({
            "request_cid": str(req.request_cid),
            "winner": str(winner.name),
        })
        return decision


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerWitness:
    schema: str
    controller_cid: str
    n_decisions: int
    n_no_provider: int
    eligible_providers_seen: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "n_no_provider": int(self.n_no_provider),
            "eligible_providers_seen": list(
                self.eligible_providers_seen),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_witness(
        controller: HostedRouterController,
) -> HostedRouterControllerWitness:
    seen: set[str] = set()
    n_no = 0
    for entry in controller.audit:
        w = entry.get("winner")
        if w is None or w == "":
            n_no += 1
        else:
            seen.add(str(w))
    return HostedRouterControllerWitness(
        schema=W68_HOSTED_ROUTER_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit)),
        n_no_provider=int(n_no),
        eligible_providers_seen=tuple(sorted(seen)),
    )


__all__ = [
    "W68_HOSTED_ROUTER_SCHEMA_VERSION",
    "W68_HOSTED_TIER_TEXT_ONLY",
    "W68_HOSTED_TIER_LOGPROBS",
    "W68_HOSTED_TIER_PREFIX_CACHE",
    "W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE",
    "W68_HOSTED_TIERS",
    "HostedProvider",
    "HostedProviderRegistry",
    "default_hosted_registry",
    "HostedRoutingRequest",
    "HostedRoutingDecision",
    "HostedRouterController",
    "HostedRouterControllerWitness",
    "emit_hosted_router_controller_witness",
]
