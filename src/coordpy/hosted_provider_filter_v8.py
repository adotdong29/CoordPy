"""W76 H7 — Hosted Provider Filter V8 (Plane A).

Strictly extends W75's ``coordpy.hosted_provider_filter_v7``. V8
adds:

* **Chain-then-restart-aware filter** — V8 takes a caller-declared
  ``chain_then_restart_pressure`` and a
  ``max_chain_then_restart_noise_per_provider`` map; providers
  whose declared chain-then-restart-noise score exceeds the cap
  are filtered out under high chain-then-restart pressure.
* **Per-chain-then-restart tier weights** — V8 attaches a seventh
  set of per-tier weights.

Honest scope (W76): caller-declared. ``W76-L-HOSTED-PROVIDER-
FILTER-V8-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v7 import (
    HostedProviderFilterSpecV7,
    filter_hosted_registry_v7,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_PROVIDER_FILTER_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v8.v1")
W76_DEFAULT_PROVIDER_FILTER_V8_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV8:
    inner_v7: HostedProviderFilterSpecV7
    chain_then_restart_pressure: float = 0.0
    chain_then_restart_pressure_floor: float = (
        W76_DEFAULT_PROVIDER_FILTER_V8_PRESSURE_FLOOR)
    max_chain_then_restart_noise_per_provider: dict[
        str, float] = dataclasses.field(default_factory=dict)
    chain_then_restart_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W76_HOSTED_PROVIDER_FILTER_V8_SCHEMA_VERSION,
            "inner_v7_cid": str(self.inner_v7.cid()),
            "chain_then_restart_pressure": float(round(
                self.chain_then_restart_pressure, 12)),
            "chain_then_restart_pressure_floor": float(round(
                self.chain_then_restart_pressure_floor, 12)),
            "max_chain_then_restart_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_chain_then_restart_noise_per_provider
                    .items())},
            "chain_then_restart_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.chain_then_restart_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v8",
            "spec": self.to_dict()})


def filter_hosted_registry_v8(
        registry: HostedProviderRegistry,
        spec_v8: HostedProviderFilterSpecV8,
        provider_restart_noise: dict[str, float] | None = None,
        provider_rejoin_noise: dict[str, float] | None = None,
        provider_replacement_noise: (
            dict[str, float] | None) = None,
        provider_compound_noise: (
            dict[str, float] | None) = None,
        provider_compound_chain_noise: (
            dict[str, float] | None) = None,
        provider_chain_then_restart_noise: (
            dict[str, float] | None) = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Chain-then-restart-aware filter. Returns (filtered_registry,
    report_v8).

    First applies the V7 chain-aware filter chain. If
    ``chain_then_restart_pressure >= chain_then_restart_pressure_
    floor``, additionally drops providers whose caller-declared
    ``chain_then_restart_noise_score`` exceeds
    ``max_chain_then_restart_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v7(
        registry, spec_v8.inner_v7,
        provider_restart_noise=dict(
            provider_restart_noise or {}),
        provider_rejoin_noise=dict(provider_rejoin_noise or {}),
        provider_replacement_noise=dict(
            provider_replacement_noise or {}),
        provider_compound_noise=dict(
            provider_compound_noise or {}),
        provider_compound_chain_noise=dict(
            provider_compound_chain_noise or {}))
    pressure = float(max(0.0, min(
        1.0, float(spec_v8.chain_then_restart_pressure))))
    floor_active = bool(
        pressure
        >= float(spec_v8.chain_then_restart_pressure_floor))
    cn = dict(provider_chain_then_restart_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W76_HOSTED_PROVIDER_FILTER_V8_SCHEMA_VERSION,
            "v7_report": dict(inner_rep),
            "chain_then_restart_pressure": float(round(
                pressure, 12)),
            "chain_then_restart_pressure_floor_active": False,
            "n_dropped_under_chain_then_restart": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v8.max_chain_then_restart_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(cn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W76_HOSTED_PROVIDER_FILTER_V8_SCHEMA_VERSION,
        "v7_report": dict(inner_rep),
        "chain_then_restart_pressure": float(round(pressure, 12)),
        "chain_then_restart_pressure_floor_active": True,
        "n_dropped_under_chain_then_restart": int(len(dropped)),
        "dropped_under_chain_then_restart": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W76_HOSTED_PROVIDER_FILTER_V8_SCHEMA_VERSION",
    "W76_DEFAULT_PROVIDER_FILTER_V8_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV8",
    "filter_hosted_registry_v8",
]
