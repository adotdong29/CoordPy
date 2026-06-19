"""W77 H7 — Hosted Provider Filter V9 (Plane A).

Strictly extends W76's ``coordpy.hosted_provider_filter_v8``. V9
adds:

* **Post-restart-replacement-aware filter** — V9 takes a caller-
  declared ``post_restart_replacement_pressure`` and a
  ``max_post_restart_replacement_noise_per_provider`` map;
  providers whose declared post-restart-replacement-noise score
  exceeds the cap are filtered out under high post-restart-
  replacement pressure.
* **Per-post-restart-replacement tier weights** — V9 attaches an
  eighth set of per-tier weights.

Honest scope (W77): caller-declared. ``W77-L-HOSTED-PROVIDER-
FILTER-V9-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_provider_filter_v8 import (
    HostedProviderFilterSpecV8,
    filter_hosted_registry_v8,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_PROVIDER_FILTER_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v9.v1")
W77_DEFAULT_PROVIDER_FILTER_V9_PRESSURE_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV9:
    inner_v8: HostedProviderFilterSpecV8
    post_restart_replacement_pressure: float = 0.0
    post_restart_replacement_pressure_floor: float = (
        W77_DEFAULT_PROVIDER_FILTER_V9_PRESSURE_FLOOR)
    max_post_restart_replacement_noise_per_provider: dict[
        str, float] = dataclasses.field(default_factory=dict)
    post_restart_replacement_tier_weights: dict[
        str, float] = (
        dataclasses.field(default_factory=dict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W77_HOSTED_PROVIDER_FILTER_V9_SCHEMA_VERSION,
            "inner_v8_cid": str(self.inner_v8.cid()),
            "post_restart_replacement_pressure": float(round(
                self.post_restart_replacement_pressure, 12)),
            "post_restart_replacement_pressure_floor": float(round(
                self.post_restart_replacement_pressure_floor, 12)),
            "max_post_restart_replacement_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .max_post_restart_replacement_noise_per_provider
                    .items())},
            "post_restart_replacement_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .post_restart_replacement_tier_weights
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v9",
            "spec": self.to_dict()})


def filter_hosted_registry_v9(
        registry: HostedProviderRegistry,
        spec_v9: HostedProviderFilterSpecV9,
        provider_post_restart_replacement_noise: (
            dict[str, float] | None) = None,
        **inner_kwargs: Any,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Post-restart-replacement-aware filter. Returns
    (filtered_registry, report_v9).

    First applies the V8 chain-then-restart-aware filter. If
    ``post_restart_replacement_pressure >= post_restart_replacement_
    pressure_floor``, additionally drops providers whose caller-
    declared ``post_restart_replacement_noise_score`` exceeds
    ``max_post_restart_replacement_noise_per_provider.get(provider_name)``.
    """
    inner_filtered, inner_rep = filter_hosted_registry_v8(
        registry, spec_v9.inner_v8, **inner_kwargs)
    pressure = float(max(0.0, min(
        1.0,
        float(spec_v9.post_restart_replacement_pressure))))
    floor_active = bool(
        pressure
        >= float(
            spec_v9.post_restart_replacement_pressure_floor))
    pn = dict(provider_post_restart_replacement_noise or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W77_HOSTED_PROVIDER_FILTER_V9_SCHEMA_VERSION,
            "v8_report": dict(inner_rep),
            "post_restart_replacement_pressure": float(round(
                pressure, 12)),
            "post_restart_replacement_pressure_floor_active": (
                False),
            "n_dropped_under_post_restart_replacement": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v9
            .max_post_restart_replacement_noise_per_provider.get(
                str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema":
            W77_HOSTED_PROVIDER_FILTER_V9_SCHEMA_VERSION,
        "v8_report": dict(inner_rep),
        "post_restart_replacement_pressure": float(round(
            pressure, 12)),
        "post_restart_replacement_pressure_floor_active": True,
        "n_dropped_under_post_restart_replacement": int(
            len(dropped)),
        "dropped_under_post_restart_replacement": list(dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W77_HOSTED_PROVIDER_FILTER_V9_SCHEMA_VERSION",
    "W77_DEFAULT_PROVIDER_FILTER_V9_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV9",
    "filter_hosted_registry_v9",
]
