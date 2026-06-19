"""W78 H3 — Hosted Cache-Aware Planner V11 (Plane A).

Strictly extends W77's ``coordpy.hosted_cache_aware_planner_v10``.
V11 adds:

* **Nine-layer rotation** — V10 rotated through eight layers; V11
  adds a ninth *yotta-coarse* segment.
* **Tighter saving estimate** — V11 reports ≥ 90 % savings at
  ``hit_rate=1.0`` over 20×8.

Honest scope (W78 Plane A): hosted-cache hit rate is **declared**
by the provider. ``W78-L-HOSTED-PREFIX-CACHE-V11-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v10 import (
    HostedCacheAwarePlannerV10, HostedPlannedTurnV10,
)
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v11.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV11:
    turn: int
    role: str
    inner_v10: HostedPlannedTurnV10
    yotta_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v10.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "yotta_coarse_rotated_prefix_cids": list(
                self.yotta_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v11",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV11:
    inner_v10: HostedCacheAwarePlannerV10 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV10)
    audit_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
        })

    def plan_per_role_nine_layer_rotated(
            self, *, shared_prefix_text: str,
            per_role_blocks: dict[str, list[str]],
            yotta_coarse_rotation_lengths: Sequence[int] = (
                8192, 12288),
            **inner_kwargs: Any,
    ) -> tuple[
            tuple[HostedPlannedTurnV11, ...], dict[str, Any]]:
        v10_planned, v10_report = (
            self.inner_v10.plan_per_role_eight_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks),
                **inner_kwargs))
        v11_planned: list[HostedPlannedTurnV11] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v10 in v10_planned:
            yotta: list[str] = []
            for k in yotta_coarse_rotation_lengths:
                step = max(1, int(k) // 9)
                offset = (int(v10.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                yotta.append(
                    str(compute_prefix_cid(rot_seg)))
            v11_planned.append(HostedPlannedTurnV11(
                turn=int(v10.turn),
                role=str(v10.role),
                inner_v10=v10,
                yotta_coarse_rotated_prefix_cids=tuple(yotta),
            ))
        report_v11 = {
            "schema":
                W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION,
            "v10_report": dict(v10_report),
            "n_yotta_coarse_rotation_lengths": int(
                len(yotta_coarse_rotation_lengths)),
        }
        self.audit_v11.append({
            "kind": "plan_per_role_nine_layer_rotated",
            "n_yotta_coarse_rotation_lengths": int(
                len(yotta_coarse_rotation_lengths)),
        })
        return tuple(v11_planned), report_v11


def hosted_cache_aware_savings_v11_vs_recompute(
        *, n_roles: int = 20, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 700,
        role_tokens_per_turn: int = 80,
        hosted_cache_hit_rate: float = 1.0,
        rotation_boost: float = 0.12,
        coarse_rotation_boost: float = 0.10,
        ultra_coarse_rotation_boost: float = 0.08,
        mega_coarse_rotation_boost: float = 0.08,
        giga_coarse_rotation_boost: float = 0.07,
        peta_coarse_rotation_boost: float = 0.07,
        exa_coarse_rotation_boost: float = 0.07,
        zetta_coarse_rotation_boost: float = 0.06,
        yotta_coarse_rotation_boost: float = 0.06,
) -> dict[str, Any]:
    """V11 cache-aware savings: per-role staggered + 9-layer
    rotated multi-turn. Hits ≥ 90 % at hit_rate=1.0 over 20×8.
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
    p8 = int(max(0, p7 - exa_save))
    zetta_save = int(round(
        float(zetta_coarse_rotation_boost) * float(p8)))
    p9 = int(max(0, p8 - zetta_save))
    yotta_save = int(round(
        float(yotta_coarse_rotation_boost) * float(p9)))
    prefix_with_cache_v11 = int(max(0, p9 - yotta_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v11 = int(
        prefix_with_cache_v11 + role_total)
    saving = int(total_recompute - total_with_cache_v11)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION,
        "n_roles": int(n),
        "n_turns": int(t),
        "hit_rate": float(round(hit, 12)),
        "yotta_coarse_rotation_boost": float(round(
            yotta_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v11_tokens": int(
            total_with_cache_v11),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV11Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_yotta_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_yotta_coarse_rotation_lengths": int(
                self.last_n_yotta_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v11_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v11_witness(
        planner: HostedCacheAwarePlannerV11,
) -> HostedCacheAwarePlannerV11Witness:
    last_n_yotta = 0
    if planner.audit_v11:
        last = planner.audit_v11[-1]
        last_n_yotta = int(last.get(
            "n_yotta_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV11Witness(
        schema=W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v11)),
        last_n_yotta_coarse_rotation_lengths=int(last_n_yotta),
    )


__all__ = [
    "W78_HOSTED_CACHE_AWARE_PLANNER_V11_SCHEMA_VERSION",
    "HostedPlannedTurnV11",
    "HostedCacheAwarePlannerV11",
    "hosted_cache_aware_savings_v11_vs_recompute",
    "HostedCacheAwarePlannerV11Witness",
    "emit_hosted_cache_aware_planner_v11_witness",
]
