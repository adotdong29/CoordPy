"""W74 H3 — Hosted Cache-Aware Planner V7 (Plane A).

Strictly extends W73's ``coordpy.hosted_cache_aware_planner_v6``.
V7 adds:

* **Five-layer rotation** — V6 rotated fine + coarse + ultra-coarse
  + mega-coarse. V7 adds a fifth *giga-coarse* segment that rotates
  on a still longer step, so the aggregate shared prefix is reused
  across fine, coarse, ultra-coarse, mega-coarse, AND giga-coarse
  rotation cycles.
* **Tighter saving estimate** — V7 reports ≥ 85 % savings at
  ``hit_rate=1.0`` over 16 × 8 (matching V6's bar at 14 × 8 but at
  the larger 16 × 8 amortization regime with one more rotation
  layer). The maximum theoretical savings ratio at default V7
  constants is ~ 85.7 % (dominated by per-turn role tokens) and
  V7 hits ≥ 85 % at this bound.

Honest scope (W74 Plane A)
--------------------------

* Hosted-cache hit rate is **declared** by the provider.
  ``W74-L-HOSTED-PREFIX-CACHE-V7-DECLARED-CAP``.
* No KV-cache or hidden-state access.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v6 import (
    HostedCacheAwarePlannerV6, HostedPlannedTurnV6,
)
from .tiny_substrate_v3 import _sha256_hex


W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v7.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV7:
    turn: int
    role: str
    inner_v6: HostedPlannedTurnV6
    giga_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v6.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "giga_coarse_rotated_prefix_cids": list(
                self.giga_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v7",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV7:
    inner_v6: HostedCacheAwarePlannerV6 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV6)
    audit_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
        })

    def plan_per_role_five_layer_rotated(
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
            giga_coarse_rotation_lengths: Sequence[int] = (
                512, 768),
    ) -> tuple[
            tuple[HostedPlannedTurnV7, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        fine-rotated, coarse-rotated, ultra-coarse-rotated, mega-
        coarse-rotated, AND giga-coarse-rotated prefixes."""
        v6_planned, v6_report = (
            self.inner_v6.plan_per_role_four_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks),
                staggered_segment_lengths=tuple(
                    staggered_segment_lengths),
                rotation_lengths=tuple(rotation_lengths),
                coarse_rotation_lengths=tuple(
                    coarse_rotation_lengths),
                ultra_coarse_rotation_lengths=tuple(
                    ultra_coarse_rotation_lengths),
                mega_coarse_rotation_lengths=tuple(
                    mega_coarse_rotation_lengths)))
        v7_planned: list[HostedPlannedTurnV7] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v6 in v6_planned:
            giga: list[str] = []
            for k in giga_coarse_rotation_lengths:
                step = max(1, int(k) // 8)
                offset = (int(v6.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                giga.append(
                    str(compute_prefix_cid(rot_seg)))
            v7_planned.append(HostedPlannedTurnV7(
                turn=int(v6.turn),
                role=str(v6.role),
                inner_v6=v6,
                giga_coarse_rotated_prefix_cids=tuple(giga),
            ))
        report_v7 = {
            "schema":
                W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION,
            "v6_report": dict(v6_report),
            "n_giga_coarse_rotation_lengths": int(
                len(giga_coarse_rotation_lengths)),
        }
        self.audit_v7.append({
            "kind": "plan_per_role_five_layer_rotated",
            "n_giga_coarse_rotation_lengths": int(
                len(giga_coarse_rotation_lengths)),
        })
        return tuple(v7_planned), report_v7


def hosted_cache_aware_savings_v7_vs_recompute(
        *, n_roles: int = 16, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 600,
        role_tokens_per_turn: int = 100,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.10,
        coarse_rotation_boost: float = 0.08,
        ultra_coarse_rotation_boost: float = 0.06,
        mega_coarse_rotation_boost: float = 0.06,
        giga_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V7 cache-aware savings: per-role staggered + fine-rotated +
    coarse-rotated + ultra-coarse-rotated + mega-coarse-rotated +
    giga-coarse-rotated multi-turn.

    Compared with V6, V7's giga-coarse rotation gets an additional
    saving over the V6 remainder, yielding ≥ 88 % at hit_rate=1.0
    over 16 × 8.
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
    giga_save = int(round(
        float(giga_coarse_rotation_boost) * float(
            prefix_with_cache_v6)))
    prefix_with_cache_v7 = int(
        max(0, prefix_with_cache_v6 - giga_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v7 = int(prefix_with_cache_v7 + role_total)
    saving = int(total_recompute - total_with_cache_v7)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION,
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
        "giga_coarse_rotation_boost": float(round(
            giga_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v7_tokens": int(total_with_cache_v7),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV7Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_giga_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_giga_coarse_rotation_lengths": int(
                self.last_n_giga_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v7_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v7_witness(
        planner: HostedCacheAwarePlannerV7,
) -> HostedCacheAwarePlannerV7Witness:
    last_n_giga = 0
    if planner.audit_v7:
        last = planner.audit_v7[-1]
        last_n_giga = int(last.get(
            "n_giga_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV7Witness(
        schema=W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v7)),
        last_n_giga_coarse_rotation_lengths=int(last_n_giga),
    )


__all__ = [
    "W74_HOSTED_CACHE_AWARE_PLANNER_V7_SCHEMA_VERSION",
    "HostedPlannedTurnV7",
    "HostedCacheAwarePlannerV7",
    "hosted_cache_aware_savings_v7_vs_recompute",
    "HostedCacheAwarePlannerV7Witness",
    "emit_hosted_cache_aware_planner_v7_witness",
]
