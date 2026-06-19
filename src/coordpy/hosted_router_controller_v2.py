"""W69 H1 — Hosted Router Controller V2 (Plane A).

Strictly extends W68's ``coordpy.hosted_router_controller``. V2
adds:

* **Multi-objective lexicographic preference** — V1 was
  cheapest-then-fastest. V2 supports per-request weighted scoring
  over (cost, latency, data_policy_strictness, success_score).
* **Provider blacklist + sticky routing** — a request can pin to
  a specific provider; the router records sticky decisions for
  audit.
* **Routing-cost-per-success bookkeeping** — the V2 router records
  declared success rates and exposes a routing-cost-per-success
  ratio.

Honest scope (W69 Plane A)
--------------------------

* All extensions are HTTP-text-only.
* Success rates are caller-declared (the router does not measure
  live success). ``W69-L-HOSTED-V2-SUCCESS-DECLARED-CAP``.
* Sticky routing does not pierce the substrate boundary.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_router_controller import (
    HostedProvider, HostedProviderRegistry,
    HostedRoutingDecision, HostedRoutingRequest,
    HostedRouterController,
    W68_HOSTED_ROUTER_SCHEMA_VERSION,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_ROUTER_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_router_controller_v2.v1")


@dataclasses.dataclass(frozen=True)
class HostedRoutingRequestV2:
    inner_v1: HostedRoutingRequest
    weight_cost: float = 1.0
    weight_latency: float = 0.5
    weight_data_policy: float = 0.0
    weight_success: float = 0.0
    sticky_provider: str | None = None
    provider_blacklist: tuple[str, ...] = ()

    @property
    def request_cid(self) -> str:
        return self.inner_v1.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_HOSTED_ROUTER_V2_SCHEMA_VERSION,
            "kind": "hosted_routing_request_v2",
            "inner_v1_request_cid": str(
                self.inner_v1.request_cid),
            "weight_cost": float(round(self.weight_cost, 12)),
            "weight_latency": float(round(
                self.weight_latency, 12)),
            "weight_data_policy": float(round(
                self.weight_data_policy, 12)),
            "weight_success": float(round(self.weight_success, 12)),
            "sticky_provider": (
                None if self.sticky_provider is None
                else str(self.sticky_provider)),
            "provider_blacklist": list(self.provider_blacklist),
        })


@dataclasses.dataclass(frozen=True)
class HostedRoutingDecisionV2:
    schema: str
    inner_v1: HostedRoutingDecision
    score: float
    sticky_applied: bool
    blacklist_applied: tuple[str, ...]

    @property
    def chosen_provider(self) -> str | None:
        return self.inner_v1.chosen_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "score": float(round(self.score, 12)),
            "sticky_applied": bool(self.sticky_applied),
            "blacklist_applied": list(self.blacklist_applied),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_routing_decision_v2",
            "decision": self.to_dict()})


@dataclasses.dataclass
class HostedRouterControllerV2:
    inner_v1: HostedRouterController
    success_score_per_provider: dict[str, float] = (
        dataclasses.field(default_factory=dict))
    data_policy_strictness: dict[str, float] = (
        dataclasses.field(default_factory=lambda: {
            "no_log": 1.0, "no_train": 0.7,
            "log": 0.3, "train": 0.0}))
    audit_v2: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))

    @classmethod
    def init(
            cls, registry: HostedProviderRegistry,
            success_score_per_provider: (
                dict[str, float] | None) = None,
    ) -> "HostedRouterControllerV2":
        return cls(
            inner_v1=HostedRouterController(registry=registry),
            success_score_per_provider=dict(
                success_score_per_provider or {}),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_HOSTED_ROUTER_V2_SCHEMA_VERSION,
            "kind": "hosted_router_controller_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
            "success_score_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.success_score_per_provider.items())},
            "data_policy_strictness": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.data_policy_strictness.items())},
        })

    def _provider_score(
            self, p: HostedProvider,
            req_v2: HostedRoutingRequestV2,
    ) -> float:
        cost = self.inner_v1._est_cost(p, req_v2.inner_v1)
        lat = float(p.p50_latency_ms)
        dp_strict = float(
            self.data_policy_strictness.get(p.data_policy, 0.0))
        success = float(self.success_score_per_provider.get(
            p.name, 0.5))
        return float(
            float(req_v2.weight_cost) * float(cost)
            + float(req_v2.weight_latency) * float(lat)
            - float(req_v2.weight_data_policy) * float(dp_strict)
            - float(req_v2.weight_success) * float(success))

    def decide_v2(
            self, req_v2: HostedRoutingRequestV2,
    ) -> HostedRoutingDecisionV2:
        # First: blacklist filter.
        blacklist = set(req_v2.provider_blacklist)
        # Sticky routing.
        sticky = (
            str(req_v2.sticky_provider)
            if req_v2.sticky_provider is not None else None)
        sticky_applied = False
        if sticky is not None and sticky not in blacklist:
            sticky_provider = next(
                (p for p in self.inner_v1.registry.providers
                 if p.name == sticky),
                None)
            if sticky_provider is not None:
                d1 = HostedRoutingDecision(
                    schema=W68_HOSTED_ROUTER_SCHEMA_VERSION,
                    request_cid=str(req_v2.request_cid),
                    chosen_provider=str(sticky_provider.name),
                    rationale="sticky_provider",
                    eligible_providers=(sticky_provider.name,),
                    rejected_providers=(),
                    estimated_cost_usd=float(
                        self.inner_v1._est_cost(
                            sticky_provider, req_v2.inner_v1)),
                    estimated_latency_ms=float(
                        sticky_provider.p50_latency_ms),
                )
                sticky_applied = True
                self.audit_v2.append({
                    "request_cid": str(req_v2.request_cid),
                    "winner": str(sticky_provider.name),
                    "sticky_applied": True,
                })
                return HostedRoutingDecisionV2(
                    schema=W69_HOSTED_ROUTER_V2_SCHEMA_VERSION,
                    inner_v1=d1,
                    score=float(self._provider_score(
                        sticky_provider, req_v2)),
                    sticky_applied=True,
                    blacklist_applied=tuple(sorted(blacklist)),
                )
        # Otherwise: filter blacklist + v1 eligibility, then score.
        candidate_providers = [
            p for p in self.inner_v1.registry.providers
            if p.name not in blacklist]
        # Build a per-request registry-equivalent and call V1.
        filtered_registry = HostedProviderRegistry(
            providers=tuple(candidate_providers))
        sub_v1 = HostedRouterController(
            registry=filtered_registry)
        d1 = sub_v1.decide(req_v2.inner_v1)
        score = 0.0
        if d1.chosen_provider is not None:
            winner = next(
                (p for p in candidate_providers
                 if p.name == d1.chosen_provider), None)
            if winner is not None:
                score = float(
                    self._provider_score(winner, req_v2))
                # V2 may re-pick a provider that scores lower if a
                # non-cost weight is large.
                if (float(req_v2.weight_success) > 0.0
                        or float(req_v2.weight_data_policy) > 0.0):
                    eligible_in_decision = [
                        p for p in candidate_providers
                        if p.name in set(d1.eligible_providers)]
                    if eligible_in_decision:
                        best = min(
                            eligible_in_decision,
                            key=lambda p: float(
                                self._provider_score(p, req_v2)))
                        if best.name != d1.chosen_provider:
                            d1 = HostedRoutingDecision(
                                schema=W68_HOSTED_ROUTER_SCHEMA_VERSION,
                                request_cid=str(
                                    req_v2.request_cid),
                                chosen_provider=str(best.name),
                                rationale="v2_weighted_score",
                                eligible_providers=tuple(
                                    d1.eligible_providers),
                                rejected_providers=tuple(
                                    d1.rejected_providers),
                                estimated_cost_usd=float(
                                    self.inner_v1._est_cost(
                                        best, req_v2.inner_v1)),
                                estimated_latency_ms=float(
                                    best.p50_latency_ms))
                            score = float(self._provider_score(
                                best, req_v2))
        self.audit_v2.append({
            "request_cid": str(req_v2.request_cid),
            "winner": (
                str(d1.chosen_provider)
                if d1.chosen_provider is not None else ""),
            "sticky_applied": False,
            "blacklist_n": int(len(blacklist)),
        })
        return HostedRoutingDecisionV2(
            schema=W69_HOSTED_ROUTER_V2_SCHEMA_VERSION,
            inner_v1=d1, score=float(score),
            sticky_applied=False,
            blacklist_applied=tuple(sorted(blacklist)),
        )

    def routing_cost_per_success_ratio(self) -> float:
        """Aggregate routing-cost-per-success across audit."""
        n = float(len(self.audit_v2))
        if n <= 0.0:
            return 0.0
        n_success = 0.0
        for entry in self.audit_v2:
            w = entry.get("winner", "")
            if not w:
                continue
            n_success += float(
                self.success_score_per_provider.get(w, 0.5))
        if n_success <= 0.0:
            return float(n)
        return float(n) / float(n_success)


@dataclasses.dataclass(frozen=True)
class HostedRouterControllerV2Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    n_no_provider: int
    n_sticky: int
    eligible_providers_seen: tuple[str, ...]
    routing_cost_per_success_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "n_no_provider": int(self.n_no_provider),
            "n_sticky": int(self.n_sticky),
            "eligible_providers_seen": list(
                self.eligible_providers_seen),
            "routing_cost_per_success_ratio": float(round(
                self.routing_cost_per_success_ratio, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_router_controller_v2_witness",
            "witness": self.to_dict()})


def emit_hosted_router_controller_v2_witness(
        controller: HostedRouterControllerV2,
) -> HostedRouterControllerV2Witness:
    seen: set[str] = set()
    n_no = 0
    n_sticky = 0
    for entry in controller.audit_v2:
        w = entry.get("winner")
        if w is None or w == "":
            n_no += 1
        else:
            seen.add(str(w))
        if bool(entry.get("sticky_applied", False)):
            n_sticky += 1
    return HostedRouterControllerV2Witness(
        schema=W69_HOSTED_ROUTER_V2_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v2)),
        n_no_provider=int(n_no),
        n_sticky=int(n_sticky),
        eligible_providers_seen=tuple(sorted(seen)),
        routing_cost_per_success_ratio=float(
            controller.routing_cost_per_success_ratio()),
    )


__all__ = [
    "W69_HOSTED_ROUTER_V2_SCHEMA_VERSION",
    "HostedRoutingRequestV2",
    "HostedRoutingDecisionV2",
    "HostedRouterControllerV2",
    "HostedRouterControllerV2Witness",
    "emit_hosted_router_controller_v2_witness",
]
