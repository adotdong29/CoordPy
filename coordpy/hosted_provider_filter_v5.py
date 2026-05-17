"""W73 H7 — Hosted Provider Filter V5 (Plane A).

Strictly extends W72's ``coordpy.hosted_provider_filter_v4``. V5
adds:

* **Replacement-aware filter** — V5 takes a caller-declared
  ``replacement_pressure`` and a ``max_replacement_noise_per_provider``
  map; providers whose declared replacement-noise score exceeds the
  cap are filtered out under high replacement pressure.
* **Per-replacement tier weights** — V5 attaches a fourth set of
  per-tier weights that downstream callers (cost planner V6,
  router V6) can prefer under replacement conditions.

Honest scope (W73): all extensions are HTTP-text-only; the
``replacement_noise_score`` is caller-declared.
``W73-L-HOSTED-PROVIDER-FILTER-V5-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v4 import (
    HostedProviderFilterSpecV4,
    filter_hosted_registry_v4,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_PROVIDER_FILTER_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v5.v1")
W73_DEFAULT_PROVIDER_FILTER_V5_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV5:
    inner_v4: HostedProviderFilterSpecV4
    replacement_pressure: float = 0.0
    replacement_pressure_floor: float = (
        W73_DEFAULT_PROVIDER_FILTER_V5_PRESSURE_FLOOR)
    max_replacement_noise_per_provider: dict[str, float] = (
        dataclasses.field(default_factory=dict))
    replacement_tier_weights: dict[str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W73_HOSTED_PROVIDER_FILTER_V5_SCHEMA_VERSION,
            "inner_v4_cid": str(self.inner_v4.cid()),
            "replacement_pressure": float(round(
                self.replacement_pressure, 12)),
            "replacement_pressure_floor": float(round(
                self.replacement_pressure_floor, 12)),
            "max_replacement_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.max_replacement_noise_per_provider
                    .items())},
            "replacement_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.replacement_tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v5",
            "spec": self.to_dict()})


def filter_hosted_registry_v5(
        registry: HostedProviderRegistry,
        spec_v5: HostedProviderFilterSpecV5,
        provider_restart_noise: dict[str, float] | None = None,
        provider_rejoin_noise: dict[str, float] | None = None,
        provider_replacement_noise: (
            dict[str, float] | None) = None,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Replacement-aware filter. Returns (filtered_registry,
    report_v5).

    First applies the V4 rejoin-aware filter chain. If
    ``replacement_pressure >= replacement_pressure_floor``,
    additionally drops providers whose caller-declared
    ``replacement_noise_score`` exceeds
    ``max_replacement_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v4(
        registry, spec_v5.inner_v4,
        provider_restart_noise=dict(
            provider_restart_noise or {}),
        provider_rejoin_noise=dict(provider_rejoin_noise or {}))
    pressure = float(max(0.0, min(
        1.0, float(spec_v5.replacement_pressure))))
    floor_active = bool(
        pressure >= float(spec_v5.replacement_pressure_floor))
    pn = dict(provider_replacement_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W73_HOSTED_PROVIDER_FILTER_V5_SCHEMA_VERSION,
            "v4_report": dict(inner_rep),
            "replacement_pressure": float(round(pressure, 12)),
            "replacement_pressure_floor_active": False,
            "n_dropped_under_replacement": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v5.max_replacement_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W73_HOSTED_PROVIDER_FILTER_V5_SCHEMA_VERSION,
        "v4_report": dict(inner_rep),
        "replacement_pressure": float(round(pressure, 12)),
        "replacement_pressure_floor_active": True,
        "n_dropped_under_replacement": int(len(dropped)),
        "dropped_under_replacement": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W73_HOSTED_PROVIDER_FILTER_V5_SCHEMA_VERSION",
    "W73_DEFAULT_PROVIDER_FILTER_V5_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV5",
    "filter_hosted_registry_v5",
]
