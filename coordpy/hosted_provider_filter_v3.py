"""W71 H7 — Hosted Provider Filter V3 (Plane A).

Strictly extends W69's ``coordpy.hosted_provider_filter_v2``. V3
adds:

* **Restart-aware filter** — V3 takes a caller-declared
  ``restart_pressure`` and a ``max_restart_pressure_per_provider``
  map; providers whose declared restart-noise score exceeds the
  cap are filtered out under high restart pressure.
* **Per-restart tier weights** — V3 attaches a second set of
  per-tier weights that downstream callers (cost planner V4,
  router V4) can prefer under restart conditions.

Honest scope (W71): all extensions are HTTP-text-only; the
``restart_noise_score`` is caller-declared.
``W71-L-HOSTED-PROVIDER-FILTER-V3-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v2 import (
    HostedProviderFilterSpecV2,
    filter_hosted_registry_v2,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_PROVIDER_FILTER_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v3.v1")
W71_DEFAULT_PROVIDER_FILTER_V3_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV3:
    inner_v2: HostedProviderFilterSpecV2
    restart_pressure: float = 0.0
    restart_pressure_floor: float = (
        W71_DEFAULT_PROVIDER_FILTER_V3_PRESSURE_FLOOR)
    max_restart_noise_per_provider: dict[str, float] = (
        dataclasses.field(default_factory=dict))
    restart_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W71_HOSTED_PROVIDER_FILTER_V3_SCHEMA_VERSION,
            "inner_v2_cid": str(self.inner_v2.cid()),
            "restart_pressure": float(round(
                self.restart_pressure, 12)),
            "restart_pressure_floor": float(round(
                self.restart_pressure_floor, 12)),
            "max_restart_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_restart_noise_per_provider.items())},
            "restart_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.restart_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v3",
            "spec": self.to_dict()})


def filter_hosted_registry_v3(
        registry: HostedProviderRegistry,
        spec_v3: HostedProviderFilterSpecV3,
        provider_restart_noise: dict[str, float] | None = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Restart-aware filter. Returns (filtered_registry, report_v3).

    First applies the V2 compositional filter chain. If
    ``restart_pressure >= restart_pressure_floor``, additionally
    drops providers whose caller-declared ``restart_noise_score``
    exceeds ``max_restart_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v2(
        registry, spec_v3.inner_v2)
    pressure = float(max(0.0, min(
        1.0, float(spec_v3.restart_pressure))))
    floor_active = bool(
        pressure >= float(spec_v3.restart_pressure_floor))
    pn = dict(provider_restart_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W71_HOSTED_PROVIDER_FILTER_V3_SCHEMA_VERSION,
            "v2_report": dict(inner_rep),
            "restart_pressure": float(round(pressure, 12)),
            "restart_pressure_floor_active": False,
            "n_dropped_under_restart": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v3.max_restart_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W71_HOSTED_PROVIDER_FILTER_V3_SCHEMA_VERSION,
        "v2_report": dict(inner_rep),
        "restart_pressure": float(round(pressure, 12)),
        "restart_pressure_floor_active": True,
        "n_dropped_under_restart": int(len(dropped)),
        "dropped_under_restart": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W71_HOSTED_PROVIDER_FILTER_V3_SCHEMA_VERSION",
    "W71_DEFAULT_PROVIDER_FILTER_V3_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV3",
    "filter_hosted_registry_v3",
]
