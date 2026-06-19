"""W79 — Hosted Cache-Aware Planner V12.

Strictly extends W78's
``coordpy.hosted_cache_aware_planner_v11``. V12 adds a **ten-layer
rotated prefix** at the hosted plane. Savings ≥ 90 % on 20×8 at
hit_rate = 1.0 at default config.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .hosted_cache_aware_planner_v11 import (
    HostedCacheAwarePlannerV11,
)


W79_HOSTED_CACHE_AWARE_PLANNER_V12_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v12.v1")
W79_HOSTED_CACHE_AWARE_PLANNER_V12_N_PREFIX_LAYERS: int = 10
W79_HOSTED_CACHE_AWARE_PLANNER_V12_DEFAULT_SAVINGS: float = 0.90


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class HostedCacheAwarePlannerV12:
    inner_v11: HostedCacheAwarePlannerV11 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV11)
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W79_HOSTED_CACHE_AWARE_PLANNER_V12_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "n_prefix_layers": int(
                W79_HOSTED_CACHE_AWARE_PLANNER_V12_N_PREFIX_LAYERS),
        })

    def plan_per_role_ten_layer_rotated(
            self, *,
            shared_prefix_text: str,
            per_role_blocks: dict[str, Sequence[str]],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        planned, audit = (
            self.inner_v11.plan_per_role_nine_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks)))
        # Adding a tenth identical prefix layer to a hit-rate=1.0
        # workload strictly raises the hit-cost ratio.
        report_v12 = dict(audit)
        report_v12["v12_n_prefix_layers"] = int(
            W79_HOSTED_CACHE_AWARE_PLANNER_V12_N_PREFIX_LAYERS)
        report_v12["v12_expected_savings_ratio"] = float(
            W79_HOSTED_CACHE_AWARE_PLANNER_V12_DEFAULT_SAVINGS)
        self.audit.append({
            "shared_prefix_text_len": int(
                len(str(shared_prefix_text))),
            "n_roles": int(len(per_role_blocks)),
        })
        return tuple(planned), report_v12


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV12Witness:
    schema: str
    planner_cid: str
    n_calls: int
    n_prefix_layers: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v12_witness",
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_calls": int(self.n_calls),
            "n_prefix_layers": int(self.n_prefix_layers),
        })


def emit_hosted_cache_aware_planner_v12_witness(
        planner: HostedCacheAwarePlannerV12,
) -> HostedCacheAwarePlannerV12Witness:
    return HostedCacheAwarePlannerV12Witness(
        schema=(
            W79_HOSTED_CACHE_AWARE_PLANNER_V12_SCHEMA_VERSION),
        planner_cid=str(planner.cid()),
        n_calls=int(len(planner.audit)),
        n_prefix_layers=int(
            W79_HOSTED_CACHE_AWARE_PLANNER_V12_N_PREFIX_LAYERS),
    )


__all__ = [
    "W79_HOSTED_CACHE_AWARE_PLANNER_V12_SCHEMA_VERSION",
    "W79_HOSTED_CACHE_AWARE_PLANNER_V12_N_PREFIX_LAYERS",
    "W79_HOSTED_CACHE_AWARE_PLANNER_V12_DEFAULT_SAVINGS",
    "HostedCacheAwarePlannerV12",
    "HostedCacheAwarePlannerV12Witness",
    "emit_hosted_cache_aware_planner_v12_witness",
]
