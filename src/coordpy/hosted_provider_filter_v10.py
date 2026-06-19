"""W78 H7 — Hosted Provider Filter V10 (Plane A).

Strictly extends W77's ``coordpy.hosted_provider_filter_v9``. V10
adds:

* **Long-horizon-reconstruction-aware filter** — V10 takes a
  caller-declared ``long_horizon_reconstruction_pressure`` and a
  ``max_long_horizon_reconstruction_noise_per_provider`` map;
  providers whose declared long-horizon-reconstruction-noise
  score exceeds the cap are filtered out under high long-horizon-
  reconstruction pressure.

Honest scope (W78): caller-declared. ``W78-L-HOSTED-PROVIDER-
FILTER-V10-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v9 import (
    HostedProviderFilterSpecV9,
    filter_hosted_registry_v9,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_PROVIDER_FILTER_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v10.v1")
W78_DEFAULT_PROVIDER_FILTER_V10_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV10:
    inner_v9: HostedProviderFilterSpecV9
    long_horizon_reconstruction_pressure: float = 0.0
    long_horizon_reconstruction_pressure_floor: float = (
        W78_DEFAULT_PROVIDER_FILTER_V10_PRESSURE_FLOOR)
    max_long_horizon_reconstruction_noise_per_provider: dict[
        str, float] = dataclasses.field(default_factory=dict)
    long_horizon_reconstruction_tier_weights: dict[
        str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W78_HOSTED_PROVIDER_FILTER_V10_SCHEMA_VERSION,
            "inner_v9_cid": str(self.inner_v9.cid()),
            "long_horizon_reconstruction_pressure": float(round(
                self.long_horizon_reconstruction_pressure, 12)),
            "long_horizon_reconstruction_pressure_floor": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_floor,
                    12)),
            "max_long_horizon_reconstruction_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .max_long_horizon_reconstruction_noise_per_provider
                    .items())},
            "long_horizon_reconstruction_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .long_horizon_reconstruction_tier_weights
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v10",
            "spec": self.to_dict()})


def filter_hosted_registry_v10(
        registry: HostedProviderRegistry,
        spec_v10: HostedProviderFilterSpecV10,
        provider_long_horizon_reconstruction_noise: (
            dict[str, float] | None) = None,
        **inner_kwargs: Any,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Long-horizon-reconstruction-aware filter. Returns
    (filtered_registry, report_v10)."""
    inner_filtered, inner_rep = filter_hosted_registry_v9(
        registry, spec_v10.inner_v9, **inner_kwargs)
    pressure = float(max(0.0, min(
        1.0,
        float(spec_v10.long_horizon_reconstruction_pressure))))
    floor_active = bool(
        pressure
        >= float(
            spec_v10.long_horizon_reconstruction_pressure_floor))
    pn = dict(
        provider_long_horizon_reconstruction_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W78_HOSTED_PROVIDER_FILTER_V10_SCHEMA_VERSION,
            "v9_report": dict(inner_rep),
            "long_horizon_reconstruction_pressure": float(round(
                pressure, 12)),
            "long_horizon_reconstruction_pressure_floor_active": (
                False),
            "n_dropped_under_long_horizon_reconstruction": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v10
            .max_long_horizon_reconstruction_noise_per_provider
            .get(str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W78_HOSTED_PROVIDER_FILTER_V10_SCHEMA_VERSION,
        "v9_report": dict(inner_rep),
        "long_horizon_reconstruction_pressure": float(round(
            pressure, 12)),
        "long_horizon_reconstruction_pressure_floor_active": True,
        "n_dropped_under_long_horizon_reconstruction": int(
            len(dropped)),
        "dropped_under_long_horizon_reconstruction": list(
            dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W78_HOSTED_PROVIDER_FILTER_V10_SCHEMA_VERSION",
    "W78_DEFAULT_PROVIDER_FILTER_V10_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV10",
    "filter_hosted_registry_v10",
]
