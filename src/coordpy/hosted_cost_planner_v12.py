"""W79 — Hosted Cost Planner V12.

Cost-per-replacement-then-restart-after-long-delay-success-under-
budget + abstain-when-RTRLD-violated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_cost_planner_v11 import HostedCostPlanSpecV11


W79_HOSTED_COST_PLANNER_V12_SCHEMA_VERSION: str = (
    "coordpy.hosted_cost_planner_v12.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class HostedCostPlannerV12:
    inner_v11_spec: HostedCostPlanSpecV11 | None = None
    replacement_then_restart_after_long_delay_violation_abstain: (
        bool) = True
    replacement_then_restart_after_long_delay_cost_alpha: float = (
        1.20)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W79_HOSTED_COST_PLANNER_V12_SCHEMA_VERSION,
            "kind": "hosted_cost_planner_v12",
            "inner_v11_spec_cid": (
                str(self.inner_v11_spec.cid())
                if self.inner_v11_spec is not None else "none"),
            "replacement_then_restart_after_long_delay_violation_abstain": bool(
                self
                .replacement_then_restart_after_long_delay_violation_abstain),
            "replacement_then_restart_after_long_delay_cost_alpha": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_cost_alpha,
                    12)),
        })


@dataclasses.dataclass(frozen=True)
class HostedCostPlannerV12Witness:
    schema: str
    planner_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cost_planner_v12_witness",
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
        })


def emit_hosted_cost_planner_v12_witness(
        planner: HostedCostPlannerV12,
) -> HostedCostPlannerV12Witness:
    return HostedCostPlannerV12Witness(
        schema=W79_HOSTED_COST_PLANNER_V12_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
    )


__all__ = [
    "W79_HOSTED_COST_PLANNER_V12_SCHEMA_VERSION",
    "HostedCostPlannerV12",
    "HostedCostPlannerV12Witness",
    "emit_hosted_cost_planner_v12_witness",
]
