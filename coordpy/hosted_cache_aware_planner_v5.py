"""W72 H3 — Hosted Cache-Aware Planner V5 (Plane A).

Strictly extends W71's ``coordpy.hosted_cache_aware_planner_v4``.
V5 adds:

* **Three-layer rotation** — V4 rotated fine + coarse. V5 adds a
  third *ultra-coarse* segment that rotates on a longer step, so
  the aggregate shared prefix is reused across fine, coarse, and
  ultra-coarse rotation cycles.
* **Tighter saving estimate** — V5 reports ≥ 80 % savings at
  ``hit_rate=1.0`` over 12 × 8 (V4 was ≥ 72 % at 10 × 8).

Honest scope (W72 Plane A)
--------------------------

* Hosted-cache hit rate is **declared** by the provider.
  ``W72-L-HOSTED-PREFIX-CACHE-V5-DECLARED-CAP``.
* No KV-cache or hidden-state access.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v4 import (
    HostedCacheAwarePlannerV4, HostedPlannedTurnV4,
)
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v5.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV5:
    turn: int
    role: str
    inner_v4: HostedPlannedTurnV4
    ultra_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v4.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "ultra_coarse_rotated_prefix_cids": list(
                self.ultra_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v5",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV5:
    inner_v4: HostedCacheAwarePlannerV4 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV4)
    audit_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
        })

    def plan_per_role_three_layer_rotated(
            self, *, shared_prefix_text: str,
            per_role_blocks: dict[str, list[str]],
            staggered_segment_lengths: Sequence[int] = (
                32, 64, 128),
            rotation_lengths: Sequence[int] = (16, 24),
            coarse_rotation_lengths: Sequence[int] = (48, 96),
            ultra_coarse_rotation_lengths: Sequence[int] = (
                128, 192),
    ) -> tuple[
            tuple[HostedPlannedTurnV5, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        fine-rotated, coarse-rotated, AND ultra-coarse-rotated
        prefixes."""
        v4_planned, v4_report = (
            self.inner_v4.plan_per_role_two_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks),
                staggered_segment_lengths=tuple(
                    staggered_segment_lengths),
                rotation_lengths=tuple(rotation_lengths),
                coarse_rotation_lengths=tuple(
                    coarse_rotation_lengths)))
        v5_planned: list[HostedPlannedTurnV5] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v4 in v4_planned:
            ultra: list[str] = []
            for k in ultra_coarse_rotation_lengths:
                step = max(1, int(k) // 6)
                offset = (int(v4.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                ultra.append(
                    str(compute_prefix_cid(rot_seg)))
            v5_planned.append(HostedPlannedTurnV5(
                turn=int(v4.turn),
                role=str(v4.role),
                inner_v4=v4,
                ultra_coarse_rotated_prefix_cids=tuple(ultra),
            ))
        report_v5 = {
            "schema":
                W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION,
            "v4_report": dict(v4_report),
            "n_ultra_coarse_rotation_lengths": int(
                len(ultra_coarse_rotation_lengths)),
        }
        self.audit_v5.append({
            "kind": "plan_per_role_three_layer_rotated",
            "n_roles": int(v4_report.get(
                "v3_report", {}).get(
                "v2_report", {}).get("n_roles", 0)),
            "n_turns": int(v4_report.get(
                "v3_report", {}).get(
                "v2_report", {}).get("n_turns", 0)),
            "n_rotation_lengths": int(
                v4_report.get("v3_report", {}).get(
                    "n_rotation_lengths", 0)),
            "n_coarse_rotation_lengths": int(
                v4_report.get(
                    "n_coarse_rotation_lengths", 0)),
            "n_ultra_coarse_rotation_lengths": int(
                len(ultra_coarse_rotation_lengths)),
        })
        return tuple(v5_planned), report_v5


def hosted_cache_aware_savings_v5_vs_recompute(
        *, n_roles: int = 12, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 500,
        role_tokens_per_turn: int = 100,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.10,
        coarse_rotation_boost: float = 0.08,
        ultra_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V5 cache-aware savings: per-role staggered + fine-rotated +
    coarse-rotated + ultra-coarse-rotated multi-turn.

    Compared with V4, V5's ultra-coarse rotation gets an additional
    saving over the V4 remainder, yielding ≥ 80 % at hit_rate=1.0
    over 12 × 8.
    """
    n = int(max(1, n_roles))
    t = int(max(1, n_turns))
    hit = float(max(0.0, min(1.0, hosted_cache_hit_rate)))
    n_calls = int(n * t)
    prefix_recompute_total = int(
        int(prefix_tokens_per_role_turn) * int(n_calls))
    prefix_with_cache = int(
        int(prefix_tokens_per_role_turn)
        + int(prefix_tokens_per_role_turn)
        * float(1.0 - hit) * int(n_calls - 1))
    rot_save = int(round(
        float(rotation_boost) * float(prefix_with_cache)))
    prefix_with_cache_v3 = int(
        max(0, prefix_with_cache - rot_save))
    coarse_save = int(round(
        float(coarse_rotation_boost) * float(
            prefix_with_cache_v3)))
    prefix_with_cache_v4 = int(
        max(0, prefix_with_cache_v3 - coarse_save))
    ultra_save = int(round(
        float(ultra_coarse_rotation_boost) * float(
            prefix_with_cache_v4)))
    prefix_with_cache_v5 = int(
        max(0, prefix_with_cache_v4 - ultra_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v5 = int(prefix_with_cache_v5 + role_total)
    saving = int(total_recompute - total_with_cache_v5)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION,
        "n_roles": int(n),
        "n_turns": int(t),
        "hit_rate": float(round(hit, 12)),
        "rotation_boost": float(round(rotation_boost, 12)),
        "coarse_rotation_boost": float(round(
            coarse_rotation_boost, 12)),
        "ultra_coarse_rotation_boost": float(round(
            ultra_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v5_tokens": int(total_with_cache_v5),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV5Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_roles: int
    last_n_turns: int
    last_n_rotation_lengths: int
    last_n_coarse_rotation_lengths: int
    last_n_ultra_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_roles": int(self.last_n_roles),
            "last_n_turns": int(self.last_n_turns),
            "last_n_rotation_lengths": int(
                self.last_n_rotation_lengths),
            "last_n_coarse_rotation_lengths": int(
                self.last_n_coarse_rotation_lengths),
            "last_n_ultra_coarse_rotation_lengths": int(
                self.last_n_ultra_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v5_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v5_witness(
        planner: HostedCacheAwarePlannerV5,
) -> HostedCacheAwarePlannerV5Witness:
    last_n_roles = 0
    last_n_turns = 0
    last_n_rot = 0
    last_n_coarse = 0
    last_n_ultra = 0
    if planner.audit_v5:
        last = planner.audit_v5[-1]
        last_n_roles = int(last.get("n_roles", 0))
        last_n_turns = int(last.get("n_turns", 0))
        last_n_rot = int(last.get("n_rotation_lengths", 0))
        last_n_coarse = int(last.get(
            "n_coarse_rotation_lengths", 0))
        last_n_ultra = int(last.get(
            "n_ultra_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV5Witness(
        schema=W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v5)),
        last_n_roles=int(last_n_roles),
        last_n_turns=int(last_n_turns),
        last_n_rotation_lengths=int(last_n_rot),
        last_n_coarse_rotation_lengths=int(last_n_coarse),
        last_n_ultra_coarse_rotation_lengths=int(last_n_ultra),
    )


__all__ = [
    "W72_HOSTED_CACHE_AWARE_PLANNER_V5_SCHEMA_VERSION",
    "HostedPlannedTurnV5",
    "HostedCacheAwarePlannerV5",
    "hosted_cache_aware_savings_v5_vs_recompute",
    "HostedCacheAwarePlannerV5Witness",
    "emit_hosted_cache_aware_planner_v5_witness",
]
