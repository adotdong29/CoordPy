"""W74 H7 — Hosted Provider Filter V6 (Plane A).

Strictly extends W73's ``coordpy.hosted_provider_filter_v5``. V6
adds:

* **Compound-aware filter** — V6 takes a caller-declared
  ``compound_pressure`` and a ``max_compound_noise_per_provider``
  map; providers whose declared compound-noise score exceeds the
  cap are filtered out under high compound pressure.
* **Per-compound tier weights** — V6 attaches a fifth set of
  per-tier weights that downstream callers (cost planner V7,
  router V7) can prefer under compound conditions.

Honest scope (W74): all extensions are HTTP-text-only; the
``compound_noise_score`` is caller-declared.
``W74-L-HOSTED-PROVIDER-FILTER-V6-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v5 import (
    HostedProviderFilterSpecV5,
    filter_hosted_registry_v5,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W74_HOSTED_PROVIDER_FILTER_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v6.v1")
W74_DEFAULT_PROVIDER_FILTER_V6_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV6:
    inner_v5: HostedProviderFilterSpecV5
    compound_pressure: float = 0.0
    compound_pressure_floor: float = (
        W74_DEFAULT_PROVIDER_FILTER_V6_PRESSURE_FLOOR)
    max_compound_noise_per_provider: dict[str, float] = (
        dataclasses.field(default_factory=dict))
    compound_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W74_HOSTED_PROVIDER_FILTER_V6_SCHEMA_VERSION,
            "inner_v5_cid": str(self.inner_v5.cid()),
            "compound_pressure": float(round(
                self.compound_pressure, 12)),
            "compound_pressure_floor": float(round(
                self.compound_pressure_floor, 12)),
            "max_compound_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_compound_noise_per_provider
                    .items())},
            "compound_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.compound_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v6",
            "spec": self.to_dict()})


def filter_hosted_registry_v6(
        registry: HostedProviderRegistry,
        spec_v6: HostedProviderFilterSpecV6,
        provider_restart_noise: dict[str, float] | None = None,
        provider_rejoin_noise: dict[str, float] | None = None,
        provider_replacement_noise: (
            dict[str, float] | None) = None,
        provider_compound_noise: (
            dict[str, float] | None) = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Compound-aware filter. Returns (filtered_registry,
    report_v6).

    First applies the V5 replacement-aware filter chain. If
    ``compound_pressure >= compound_pressure_floor``,
    additionally drops providers whose caller-declared
    ``compound_noise_score`` exceeds
    ``max_compound_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v5(
        registry, spec_v6.inner_v5,
        provider_restart_noise=dict(
            provider_restart_noise or {}),
        provider_rejoin_noise=dict(provider_rejoin_noise or {}),
        provider_replacement_noise=dict(
            provider_replacement_noise or {}))
    pressure = float(max(0.0, min(
        1.0, float(spec_v6.compound_pressure))))
    floor_active = bool(
        pressure >= float(spec_v6.compound_pressure_floor))
    cn = dict(provider_compound_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W74_HOSTED_PROVIDER_FILTER_V6_SCHEMA_VERSION,
            "v5_report": dict(inner_rep),
            "compound_pressure": float(round(pressure, 12)),
            "compound_pressure_floor_active": False,
            "n_dropped_under_compound": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v6.max_compound_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(cn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W74_HOSTED_PROVIDER_FILTER_V6_SCHEMA_VERSION,
        "v5_report": dict(inner_rep),
        "compound_pressure": float(round(pressure, 12)),
        "compound_pressure_floor_active": True,
        "n_dropped_under_compound": int(len(dropped)),
        "dropped_under_compound": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W74_HOSTED_PROVIDER_FILTER_V6_SCHEMA_VERSION",
    "W74_DEFAULT_PROVIDER_FILTER_V6_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV6",
    "filter_hosted_registry_v6",
]
