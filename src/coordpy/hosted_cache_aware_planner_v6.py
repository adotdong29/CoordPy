"""W73 H3 — Hosted Cache-Aware Planner V6 (Plane A).

Strictly extends W72's ``coordpy.hosted_cache_aware_planner_v5``.
V6 adds:

* **Four-layer rotation** — V5 rotated fine + coarse + ultra-coarse.
  V6 adds a fourth *mega-coarse* segment that rotates on a still
  longer step, so the aggregate shared prefix is reused across
  fine, coarse, ultra-coarse, AND mega-coarse rotation cycles.
* **Tighter saving estimate** — V6 reports ≥ 85 % savings at
  ``hit_rate=1.0`` over 14 × 8 (V5 was ≥ 80 % at 12 × 8).

Honest scope (W73 Plane A)
--------------------------

* Hosted-cache hit rate is **declared** by the provider.
  ``W73-L-HOSTED-PREFIX-CACHE-V6-DECLARED-CAP``.
* No KV-cache or hidden-state access.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v5 import (
    HostedCacheAwarePlannerV5, HostedPlannedTurnV5,
)
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v6.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV6:
    turn: int
    role: str
    inner_v5: HostedPlannedTurnV5
    mega_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v5.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "mega_coarse_rotated_prefix_cids": list(
                self.mega_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v6",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV6:
    inner_v5: HostedCacheAwarePlannerV5 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV5)
    audit_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
        })

    def plan_per_role_four_layer_rotated(
            self, *, shared_prefix_text: str,
            per_role_blocks: dict[str, list[str]],
            staggered_segment_lengths: Sequence[int] = (
                32, 64, 128),
            rotation_lengths: Sequence[int] = (16, 24),
            coarse_rotation_lengths: Sequence[int] = (48, 96),
            ultra_coarse_rotation_lengths: Sequence[int] = (
                128, 192),
            mega_coarse_rotation_lengths: Sequence[int] = (
                256, 384),
    ) -> tuple[
            tuple[HostedPlannedTurnV6, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        fine-rotated, coarse-rotated, ultra-coarse-rotated, AND
        mega-coarse-rotated prefixes."""
        v5_planned, v5_report = (
            self.inner_v5.plan_per_role_three_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks),
                staggered_segment_lengths=tuple(
                    staggered_segment_lengths),
                rotation_lengths=tuple(rotation_lengths),
                coarse_rotation_lengths=tuple(
                    coarse_rotation_lengths),
                ultra_coarse_rotation_lengths=tuple(
                    ultra_coarse_rotation_lengths)))
        v6_planned: list[HostedPlannedTurnV6] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v5 in v5_planned:
            mega: list[str] = []
            for k in mega_coarse_rotation_lengths:
                step = max(1, int(k) // 8)
                offset = (int(v5.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                mega.append(
                    str(compute_prefix_cid(rot_seg)))
            v6_planned.append(HostedPlannedTurnV6(
                turn=int(v5.turn),
                role=str(v5.role),
                inner_v5=v5,
                mega_coarse_rotated_prefix_cids=tuple(mega),
            ))
        report_v6 = {
            "schema":
                W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION,
            "v5_report": dict(v5_report),
            "n_mega_coarse_rotation_lengths": int(
                len(mega_coarse_rotation_lengths)),
        }
        self.audit_v6.append({
            "kind": "plan_per_role_four_layer_rotated",
            "n_mega_coarse_rotation_lengths": int(
                len(mega_coarse_rotation_lengths)),
        })
        return tuple(v6_planned), report_v6


def hosted_cache_aware_savings_v6_vs_recompute(
        *, n_roles: int = 14, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 600,
        role_tokens_per_turn: int = 100,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.10,
        coarse_rotation_boost: float = 0.08,
        ultra_coarse_rotation_boost: float = 0.06,
        mega_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V6 cache-aware savings: per-role staggered + fine-rotated +
    coarse-rotated + ultra-coarse-rotated + mega-coarse-rotated
    multi-turn.

    Compared with V5, V6's mega-coarse rotation gets an additional
    saving over the V5 remainder, yielding ≥ 85 % at hit_rate=1.0
    over 14 × 8.
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
    mega_save = int(round(
        float(mega_coarse_rotation_boost) * float(
            prefix_with_cache_v5)))
    prefix_with_cache_v6 = int(
        max(0, prefix_with_cache_v5 - mega_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v6 = int(prefix_with_cache_v6 + role_total)
    saving = int(total_recompute - total_with_cache_v6)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION,
        "n_roles": int(n),
        "n_turns": int(t),
        "hit_rate": float(round(hit, 12)),
        "rotation_boost": float(round(rotation_boost, 12)),
        "coarse_rotation_boost": float(round(
            coarse_rotation_boost, 12)),
        "ultra_coarse_rotation_boost": float(round(
            ultra_coarse_rotation_boost, 12)),
        "mega_coarse_rotation_boost": float(round(
            mega_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v6_tokens": int(total_with_cache_v6),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV6Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_mega_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_mega_coarse_rotation_lengths": int(
                self.last_n_mega_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v6_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v6_witness(
        planner: HostedCacheAwarePlannerV6,
) -> HostedCacheAwarePlannerV6Witness:
    last_n_mega = 0
    if planner.audit_v6:
        last = planner.audit_v6[-1]
        last_n_mega = int(last.get(
            "n_mega_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV6Witness(
        schema=W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v6)),
        last_n_mega_coarse_rotation_lengths=int(last_n_mega),
    )


__all__ = [
    "W73_HOSTED_CACHE_AWARE_PLANNER_V6_SCHEMA_VERSION",
    "HostedPlannedTurnV6",
    "HostedCacheAwarePlannerV6",
    "hosted_cache_aware_savings_v6_vs_recompute",
    "HostedCacheAwarePlannerV6Witness",
    "emit_hosted_cache_aware_planner_v6_witness",
]
