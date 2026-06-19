"""W79 H7 — Hosted Provider Filter V11.

Strictly extends W78's V10. Adds replacement-then-restart-after-
long-delay-aware filter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_provider_filter_v10 import (
    HostedProviderFilterSpecV10, filter_hosted_registry_v10,
)
from .hosted_router_controller import HostedProviderRegistry


W79_HOSTED_PROVIDER_FILTER_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v11.v1")
W79_DEFAULT_PROVIDER_FILTER_V11_PRESSURE_FLOOR: float = 0.5


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV11:
    inner_v10: HostedProviderFilterSpecV10
    replacement_then_restart_after_long_delay_pressure: float = 0.0
    replacement_then_restart_after_long_delay_pressure_floor: (
        float) = W79_DEFAULT_PROVIDER_FILTER_V11_PRESSURE_FLOOR
    max_replacement_then_restart_after_long_delay_noise_per_provider: dict[
        str, float] = dataclasses.field(default_factory=dict)
    replacement_then_restart_after_long_delay_tier_weights: dict[
        str, float] = dataclasses.field(default_factory=dict)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v11",
            "schema":
                W79_HOSTED_PROVIDER_FILTER_V11_SCHEMA_VERSION,
            "inner_v10_cid": str(self.inner_v10.cid()),
            "replacement_then_restart_after_long_delay_pressure": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure,
                    12)),
            "replacement_then_restart_after_long_delay_pressure_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor,
                    12)),
            "max_replacement_then_restart_after_long_delay_noise_per_provider": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .max_replacement_then_restart_after_long_delay_noise_per_provider
                    .items())},
            "replacement_then_restart_after_long_delay_tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .replacement_then_restart_after_long_delay_tier_weights
                    .items())},
        })


def filter_hosted_registry_v11(
        registry: HostedProviderRegistry,
        spec_v11: HostedProviderFilterSpecV11,
        provider_replacement_then_restart_after_long_delay_noise: (
            dict[str, float] | None) = None,
        **inner_kwargs: Any,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    inner_filtered, inner_rep = filter_hosted_registry_v10(
        registry, spec_v11.inner_v10, **inner_kwargs)
    pressure = float(max(0.0, min(
        1.0,
        float(
            spec_v11
            .replacement_then_restart_after_long_delay_pressure))))
    floor_active = bool(
        pressure
        >= float(
            spec_v11
            .replacement_then_restart_after_long_delay_pressure_floor))
    pn = dict(
        provider_replacement_then_restart_after_long_delay_noise
        or {})
    if not floor_active:
        return inner_filtered, {
            "schema":
                W79_HOSTED_PROVIDER_FILTER_V11_SCHEMA_VERSION,
            "v10_report": dict(inner_rep),
            "replacement_then_restart_after_long_delay_pressure": float(
                round(pressure, 12)),
            "replacement_then_restart_after_long_delay_pressure_floor_active": (
                False),
            "n_dropped_under_replacement_then_restart_after_long_delay": 0,
            "kept_providers": [
                p.name for p in inner_filtered.providers],
        }
    kept = []
    dropped: list[str] = []
    for p in inner_filtered.providers:
        cap = float(
            spec_v11
            .max_replacement_then_restart_after_long_delay_noise_per_provider
            .get(str(p.name), 1.0))
        noise = float(pn.get(str(p.name), 0.0))
        if noise <= cap:
            kept.append(p)
        else:
            dropped.append(str(p.name))
    filt = HostedProviderRegistry(providers=tuple(kept))
    return filt, {
        "schema": W79_HOSTED_PROVIDER_FILTER_V11_SCHEMA_VERSION,
        "v10_report": dict(inner_rep),
        "replacement_then_restart_after_long_delay_pressure": float(
            round(pressure, 12)),
        "replacement_then_restart_after_long_delay_pressure_floor_active": True,
        "n_dropped_under_replacement_then_restart_after_long_delay": int(
            len(dropped)),
        "dropped_under_replacement_then_restart_after_long_delay": list(
            dropped),
        "kept_providers": [p.name for p in kept],
    }


__all__ = [
    "W79_HOSTED_PROVIDER_FILTER_V11_SCHEMA_VERSION",
    "W79_DEFAULT_PROVIDER_FILTER_V11_PRESSURE_FLOOR",
    "HostedProviderFilterSpecV11",
    "filter_hosted_registry_v11",
]
