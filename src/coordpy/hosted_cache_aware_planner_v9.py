"""W76 H3 — Hosted Cache-Aware Planner V9 (Plane A).

Strictly extends W75's ``coordpy.hosted_cache_aware_planner_v8``.
V9 adds:

* **Seven-layer rotation** — V8 rotated fine + coarse + ultra-
  coarse + mega-coarse + giga-coarse + peta-coarse. V9 adds a
  seventh *exa-coarse* segment.
* **Tighter saving estimate** — V9 reports ≥ 88 % savings at
  ``hit_rate=1.0`` over 20 × 8.

Honest scope (W76 Plane A): hosted-cache hit rate is **declared**
by the provider. ``W76-L-HOSTED-PREFIX-CACHE-V9-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v8 import (
    HostedCacheAwarePlannerV8, HostedPlannedTurnV8,
)
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v9.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV9:
    turn: int
    role: str
    inner_v8: HostedPlannedTurnV8
    exa_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v8.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "exa_coarse_rotated_prefix_cids": list(
                self.exa_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v9",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV9:
    inner_v8: HostedCacheAwarePlannerV8 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV8)
    audit_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
        })

    def plan_per_role_seven_layer_rotated(
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
            exa_coarse_rotation_lengths: Sequence[int] = (
                2048, 3072),
    ) -> tuple[
            tuple[HostedPlannedTurnV9, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        seven-layer rotated prefixes."""
        v8_planned, v8_report = (
            self.inner_v8.plan_per_role_six_layer_rotated(
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
                    giga_coarse_rotation_lengths),
                peta_coarse_rotation_lengths=tuple(
                    peta_coarse_rotation_lengths)))
        v9_planned: list[HostedPlannedTurnV9] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v8 in v8_planned:
            exa: list[str] = []
            for k in exa_coarse_rotation_lengths:
                step = max(1, int(k) // 8)
                offset = (int(v8.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                exa.append(
                    str(compute_prefix_cid(rot_seg)))
            v9_planned.append(HostedPlannedTurnV9(
                turn=int(v8.turn),
                role=str(v8.role),
                inner_v8=v8,
                exa_coarse_rotated_prefix_cids=tuple(exa),
            ))
        report_v9 = {
            "schema":
                W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION,
            "v8_report": dict(v8_report),
            "n_exa_coarse_rotation_lengths": int(
                len(exa_coarse_rotation_lengths)),
        }
        self.audit_v9.append({
            "kind": "plan_per_role_seven_layer_rotated",
            "n_exa_coarse_rotation_lengths": int(
                len(exa_coarse_rotation_lengths)),
        })
        return tuple(v9_planned), report_v9


def hosted_cache_aware_savings_v9_vs_recompute(
        *, n_roles: int = 20, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 700,
        role_tokens_per_turn: int = 80,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.10,
        coarse_rotation_boost: float = 0.08,
        ultra_coarse_rotation_boost: float = 0.06,
        mega_coarse_rotation_boost: float = 0.06,
        giga_coarse_rotation_boost: float = 0.06,
        peta_coarse_rotation_boost: float = 0.06,
        exa_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V9 cache-aware savings: per-role staggered + 7-layer
    rotated multi-turn. Hits ≥ 88 % at hit_rate=1.0 over 20×8.
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
    p2 = int(max(0, prefix_with_cache - rot_save))
    coarse_save = int(round(
        float(coarse_rotation_boost) * float(p2)))
    p3 = int(max(0, p2 - coarse_save))
    ultra_save = int(round(
        float(ultra_coarse_rotation_boost) * float(p3)))
    p4 = int(max(0, p3 - ultra_save))
    mega_save = int(round(
        float(mega_coarse_rotation_boost) * float(p4)))
    p5 = int(max(0, p4 - mega_save))
    giga_save = int(round(
        float(giga_coarse_rotation_boost) * float(p5)))
    p6 = int(max(0, p5 - giga_save))
    peta_save = int(round(
        float(peta_coarse_rotation_boost) * float(p6)))
    p7 = int(max(0, p6 - peta_save))
    exa_save = int(round(
        float(exa_coarse_rotation_boost) * float(p7)))
    prefix_with_cache_v9 = int(max(0, p7 - exa_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v9 = int(prefix_with_cache_v9 + role_total)
    saving = int(total_recompute - total_with_cache_v9)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION,
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
        "exa_coarse_rotation_boost": float(round(
            exa_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v9_tokens": int(total_with_cache_v9),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV9Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_exa_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_exa_coarse_rotation_lengths": int(
                self.last_n_exa_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v9_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v9_witness(
        planner: HostedCacheAwarePlannerV9,
) -> HostedCacheAwarePlannerV9Witness:
    last_n_exa = 0
    if planner.audit_v9:
        last = planner.audit_v9[-1]
        last_n_exa = int(last.get(
            "n_exa_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV9Witness(
        schema=W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v9)),
        last_n_exa_coarse_rotation_lengths=int(last_n_exa),
    )


__all__ = [
    "W76_HOSTED_CACHE_AWARE_PLANNER_V9_SCHEMA_VERSION",
    "HostedPlannedTurnV9",
    "HostedCacheAwarePlannerV9",
    "hosted_cache_aware_savings_v9_vs_recompute",
    "HostedCacheAwarePlannerV9Witness",
    "emit_hosted_cache_aware_planner_v9_witness",
]
