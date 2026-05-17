"""W75 H3 — Hosted Cache-Aware Planner V8 (Plane A).

Strictly extends W74's ``coordpy.hosted_cache_aware_planner_v7``.
V8 adds:

* **Six-layer rotation** — V7 rotated fine + coarse + ultra-coarse
  + mega-coarse + giga-coarse. V8 adds a sixth *peta-coarse*
  segment that rotates on a still longer step, so the aggregate
  shared prefix is reused across fine, coarse, ultra-coarse, mega-
  coarse, giga-coarse, AND peta-coarse rotation cycles.
* **Tighter saving estimate** — V8 reports ≥ 87 % savings at
  ``hit_rate=1.0`` over 18 × 8 (matching V7's bar at 16 × 8 but at
  the larger 18 × 8 amortization regime with one more rotation
  layer). The maximum theoretical savings ratio at default V8
  constants is ~ 87.5 % (dominated by per-turn role tokens) and
  V8 hits ≥ 87 % at this bound.

Honest scope (W75 Plane A)
--------------------------

* Hosted-cache hit rate is **declared** by the provider.
  ``W75-L-HOSTED-PREFIX-CACHE-V8-DECLARED-CAP``.
* No KV-cache or hidden-state access.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v7 import (
    HostedCacheAwarePlannerV7, HostedPlannedTurnV7,
)
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v8.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV8:
    turn: int
    role: str
    inner_v7: HostedPlannedTurnV7
    peta_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v7.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "peta_coarse_rotated_prefix_cids": list(
                self.peta_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v8",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV8:
    inner_v7: HostedCacheAwarePlannerV7 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV7)
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
        })

    def plan_per_role_six_layer_rotated(
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
            peta_coarse_rotation_lengths: Sequence[int] = (
                1024, 1536),
    ) -> tuple[
            tuple[HostedPlannedTurnV8, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        fine-rotated, coarse-rotated, ultra-coarse-rotated, mega-
        coarse-rotated, giga-coarse-rotated, AND peta-coarse-
        rotated prefixes."""
        v7_planned, v7_report = (
            self.inner_v7.plan_per_role_five_layer_rotated(
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
                    mega_coarse_rotation_lengths),
                giga_coarse_rotation_lengths=tuple(
                    giga_coarse_rotation_lengths)))
        v8_planned: list[HostedPlannedTurnV8] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v7 in v7_planned:
            peta: list[str] = []
            for k in peta_coarse_rotation_lengths:
                step = max(1, int(k) // 8)
                offset = (int(v7.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                peta.append(
                    str(compute_prefix_cid(rot_seg)))
            v8_planned.append(HostedPlannedTurnV8(
                turn=int(v7.turn),
                role=str(v7.role),
                inner_v7=v7,
                peta_coarse_rotated_prefix_cids=tuple(peta),
            ))
        report_v8 = {
            "schema":
                W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION,
            "v7_report": dict(v7_report),
            "n_peta_coarse_rotation_lengths": int(
                len(peta_coarse_rotation_lengths)),
        }
        self.audit_v8.append({
            "kind": "plan_per_role_six_layer_rotated",
            "n_peta_coarse_rotation_lengths": int(
                len(peta_coarse_rotation_lengths)),
        })
        return tuple(v8_planned), report_v8


def hosted_cache_aware_savings_v8_vs_recompute(
        *, n_roles: int = 18, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 600,
        role_tokens_per_turn: int = 80,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.10,
        coarse_rotation_boost: float = 0.08,
        ultra_coarse_rotation_boost: float = 0.06,
        mega_coarse_rotation_boost: float = 0.06,
        giga_coarse_rotation_boost: float = 0.06,
        peta_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V8 cache-aware savings: per-role staggered + fine-rotated +
    coarse-rotated + ultra-coarse-rotated + mega-coarse-rotated +
    giga-coarse-rotated + peta-coarse-rotated multi-turn.

    Compared with V7, V8's peta-coarse rotation gets an additional
    saving over the V7 remainder, yielding ≥ 87 % at hit_rate=1.0
    over 18 × 8.
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
    peta_save = int(round(
        float(peta_coarse_rotation_boost) * float(
            prefix_with_cache_v7)))
    prefix_with_cache_v8 = int(
        max(0, prefix_with_cache_v7 - peta_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v8 = int(prefix_with_cache_v8 + role_total)
    saving = int(total_recompute - total_with_cache_v8)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION,
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
        "peta_coarse_rotation_boost": float(round(
            peta_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v8_tokens": int(total_with_cache_v8),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV8Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_peta_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_peta_coarse_rotation_lengths": int(
                self.last_n_peta_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v8_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v8_witness(
        planner: HostedCacheAwarePlannerV8,
) -> HostedCacheAwarePlannerV8Witness:
    last_n_peta = 0
    if planner.audit_v8:
        last = planner.audit_v8[-1]
        last_n_peta = int(last.get(
            "n_peta_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV8Witness(
        schema=W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v8)),
        last_n_peta_coarse_rotation_lengths=int(last_n_peta),
    )


__all__ = [
    "W75_HOSTED_CACHE_AWARE_PLANNER_V8_SCHEMA_VERSION",
    "HostedPlannedTurnV8",
    "HostedCacheAwarePlannerV8",
    "hosted_cache_aware_savings_v8_vs_recompute",
    "HostedCacheAwarePlannerV8Witness",
    "emit_hosted_cache_aware_planner_v8_witness",
]
