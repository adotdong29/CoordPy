"""W72 H7 — Hosted Provider Filter V4 (Plane A).

Strictly extends W71's ``coordpy.hosted_provider_filter_v3``. V4
adds:

* **Rejoin-aware filter** — V4 takes a caller-declared
  ``rejoin_pressure`` and a ``max_rejoin_noise_per_provider``
  map; providers whose declared rejoin-noise score exceeds the
  cap are filtered out under high rejoin pressure.
* **Per-rejoin tier weights** — V4 attaches a third set of
  per-tier weights that downstream callers (cost planner V5,
  router V5) can prefer under rejoin conditions.

Honest scope (W72): all extensions are HTTP-text-only; the
``rejoin_noise_score`` is caller-declared.
``W72-L-HOSTED-PROVIDER-FILTER-V4-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v3 import (
    HostedProviderFilterSpecV3,
    filter_hosted_registry_v3,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_PROVIDER_FILTER_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v4.v1")
W72_DEFAULT_PROVIDER_FILTER_V4_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV4:
    inner_v3: HostedProviderFilterSpecV3
    rejoin_pressure: float = 0.0
    rejoin_pressure_floor: float = (
        W72_DEFAULT_PROVIDER_FILTER_V4_PRESSURE_FLOOR)
    max_rejoin_noise_per_provider: dict[str, float] = (
        dataclasses.field(default_factory=dict))
    rejoin_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W72_HOSTED_PROVIDER_FILTER_V4_SCHEMA_VERSION,
            "inner_v3_cid": str(self.inner_v3.cid()),
            "rejoin_pressure": float(round(
                self.rejoin_pressure, 12)),
            "rejoin_pressure_floor": float(round(
                self.rejoin_pressure_floor, 12)),
            "max_rejoin_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_rejoin_noise_per_provider.items())},
            "rejoin_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.rejoin_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v4",
            "spec": self.to_dict()})


def filter_hosted_registry_v4(
        registry: HostedProviderRegistry,
        spec_v4: HostedProviderFilterSpecV4,
        provider_restart_noise: dict[str, float] | None = None,
        provider_rejoin_noise: dict[str, float] | None = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Rejoin-aware filter. Returns (filtered_registry, report_v4).

    First applies the V3 restart-aware filter chain. If
    ``rejoin_pressure >= rejoin_pressure_floor``, additionally
    drops providers whose caller-declared ``rejoin_noise_score``
    exceeds ``max_rejoin_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v3(
        registry, spec_v4.inner_v3,
        provider_restart_noise=dict(
            provider_restart_noise or {}))
    pressure = float(max(0.0, min(
        1.0, float(spec_v4.rejoin_pressure))))
    floor_active = bool(
        pressure >= float(spec_v4.rejoin_pressure_floor))
    pn = dict(provider_rejoin_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W72_HOSTED_PROVIDER_FILTER_V4_SCHEMA_VERSION,
            "v3_report": dict(inner_rep),
            "rejoin_pressure": float(round(pressure, 12)),
            "rejoin_pressure_floor_active": False,
            "n_dropped_under_rejoin": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v4.max_rejoin_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W72_HOSTED_PROVIDER_FILTER_V4_SCHEMA_VERSION,
        "v3_report": dict(inner_rep),
        "rejoin_pressure": float(round(pressure, 12)),
        "rejoin_pressure_floor_active": True,
        "n_dropped_under_rejoin": int(len(dropped)),
        "dropped_under_rejoin": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W72_HOSTED_PROVIDER_FILTER_V4_SCHEMA_VERSION",
    "W72_DEFAULT_PROVIDER_FILTER_V4_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV4",
    "filter_hosted_registry_v4",
]
