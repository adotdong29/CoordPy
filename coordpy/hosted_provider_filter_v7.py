"""W75 H7 — Hosted Provider Filter V7 (Plane A).

Strictly extends W74's ``coordpy.hosted_provider_filter_v6``. V7
adds:

* **Compound-chain-aware filter** — V7 takes a caller-declared
  ``compound_chain_pressure`` and a
  ``max_compound_chain_noise_per_provider`` map; providers whose
  declared compound-chain-noise score exceeds the cap are
  filtered out under high compound-chain pressure.
* **Per-compound-chain tier weights** — V7 attaches a sixth set
  of per-tier weights that downstream callers (cost planner V8,
  router V8) can prefer under compound-chain conditions.

Honest scope (W75): all extensions are HTTP-text-only; the
``compound_chain_noise_score`` is caller-declared.
``W75-L-HOSTED-PROVIDER-FILTER-V7-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v6 import (
    HostedProviderFilterSpecV6,
    filter_hosted_registry_v6,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_PROVIDER_FILTER_V7_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v7.v1")
W75_DEFAULT_PROVIDER_FILTER_V7_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV7:
    inner_v6: HostedProviderFilterSpecV6
    compound_chain_pressure: float = 0.0
    compound_chain_pressure_floor: float = (
        W75_DEFAULT_PROVIDER_FILTER_V7_PRESSURE_FLOOR)
    max_compound_chain_noise_per_provider: dict[
        str, float] = dataclasses.field(default_factory=dict)
    compound_chain_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W75_HOSTED_PROVIDER_FILTER_V7_SCHEMA_VERSION,
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_chain_pressure": float(round(
                self.compound_chain_pressure, 12)),
            "compound_chain_pressure_floor": float(round(
                self.compound_chain_pressure_floor, 12)),
            "max_compound_chain_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_compound_chain_noise_per_provider
                    .items())},
            "compound_chain_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.compound_chain_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v7",
            "spec": self.to_dict()})


def filter_hosted_registry_v7(
        registry: HostedProviderRegistry,
        spec_v7: HostedProviderFilterSpecV7,
        provider_restart_noise: dict[str, float] | None = None,
        provider_rejoin_noise: dict[str, float] | None = None,
        provider_replacement_noise: (
            dict[str, float] | None) = None,
        provider_compound_noise: (
            dict[str, float] | None) = None,
        provider_compound_chain_noise: (
            dict[str, float] | None) = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Compound-chain-aware filter. Returns (filtered_registry,
    report_v7).

    First applies the V6 compound-aware filter chain. If
    ``compound_chain_pressure >= compound_chain_pressure_floor``,
    additionally drops providers whose caller-declared
    ``compound_chain_noise_score`` exceeds
    ``max_compound_chain_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v6(
        registry, spec_v7.inner_v6,
        provider_restart_noise=dict(
            provider_restart_noise or {}),
        provider_rejoin_noise=dict(provider_rejoin_noise or {}),
        provider_replacement_noise=dict(
            provider_replacement_noise or {}),
        provider_compound_noise=dict(
            provider_compound_noise or {}))
    pressure = float(max(0.0, min(
        1.0, float(spec_v7.compound_chain_pressure))))
    floor_active = bool(
        pressure
        >= float(spec_v7.compound_chain_pressure_floor))
    cn = dict(provider_compound_chain_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W75_HOSTED_PROVIDER_FILTER_V7_SCHEMA_VERSION,
            "v6_report": dict(inner_rep),
            "compound_chain_pressure": float(round(
                pressure, 12)),
            "compound_chain_pressure_floor_active": False,
            "n_dropped_under_compound_chain": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v7.max_compound_chain_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(cn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W75_HOSTED_PROVIDER_FILTER_V7_SCHEMA_VERSION,
        "v6_report": dict(inner_rep),
        "compound_chain_pressure": float(round(pressure, 12)),
        "compound_chain_pressure_floor_active": True,
        "n_dropped_under_compound_chain": int(len(dropped)),
        "dropped_under_compound_chain": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W75_HOSTED_PROVIDER_FILTER_V7_SCHEMA_VERSION",
    "W75_DEFAULT_PROVIDER_FILTER_V7_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV7",
    "filter_hosted_registry_v7",
]
