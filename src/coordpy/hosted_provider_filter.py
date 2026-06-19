"""W68 H4 — Hosted Provider Filter (Plane A).

Data-policy-aware provider filter for the hosted control plane.
Filters a ``HostedProviderRegistry`` against a declared data policy
(``no_log``, ``no_train``, etc.) and an allowed-tier set. Returns a
filtered registry whose CID is a deterministic function of the
input registry CID plus the filter parameters.

Honest scope (W68 Plane A)
--------------------------

* The filter trusts the provider's **declared** data policy; it
  does not verify it against the live API.
  ``W68-L-HOSTED-DATA-POLICY-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    HostedProviderRegistry,
    W68_HOSTED_TIERS,
)
from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_PROVIDER_FILTER_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter.v1")


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpec:
    require_data_policy: str   # "any", "no_log", "no_train"
    allowed_tiers: tuple[str, ...]
    max_p50_latency_ms: float
    max_cost_per_1k_output: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W68_HOSTED_PROVIDER_FILTER_SCHEMA_VERSION,
            "require_data_policy": str(
                self.require_data_policy),
            "allowed_tiers": list(self.allowed_tiers),
            "max_p50_latency_ms": float(round(
                self.max_p50_latency_ms, 12)),
            "max_cost_per_1k_output": float(round(
                self.max_cost_per_1k_output, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec",
            "spec": self.to_dict()})


def filter_hosted_registry(
        registry: HostedProviderRegistry,
        spec: HostedProviderFilterSpec,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    kept = []
    dropped: list[tuple[str, str]] = []
    for p in registry.providers:
        if spec.require_data_policy == "no_log":
            if p.data_policy not in ("no_log",):
                dropped.append((p.name, "data_policy"))
                continue
        if spec.require_data_policy == "no_train":
            if p.data_policy in ("train",):
                dropped.append((p.name, "data_policy_train"))
                continue
        if p.tier not in tuple(spec.allowed_tiers):
            dropped.append((p.name, "tier"))
            continue
        if float(p.p50_latency_ms) > float(
                spec.max_p50_latency_ms):
            dropped.append((p.name, "max_latency"))
            continue
        if float(p.cost_per_1k_output) > float(
                spec.max_cost_per_1k_output):
            dropped.append((p.name, "max_cost"))
            continue
        kept.append(p)
    new_registry = HostedProviderRegistry(
        providers=tuple(kept))
    report = {
        "schema": W68_HOSTED_PROVIDER_FILTER_SCHEMA_VERSION,
        "kind": "filter_report",
        "input_registry_cid": str(registry.cid()),
        "filter_spec_cid": str(spec.cid()),
        "output_registry_cid": str(new_registry.cid()),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(dropped)),
        "dropped_providers": [
            [str(n), str(r)] for n, r in dropped],
        "allowed_tiers": list(spec.allowed_tiers),
    }
    return new_registry, report


def all_known_tiers() -> tuple[str, ...]:
    return tuple(W68_HOSTED_TIERS)


__all__ = [
    "W68_HOSTED_PROVIDER_FILTER_SCHEMA_VERSION",
    "HostedProviderFilterSpec",
    "filter_hosted_registry",
    "all_known_tiers",
]
