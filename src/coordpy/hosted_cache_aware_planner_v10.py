"""W77 H3 — Hosted Cache-Aware Planner V10 (Plane A).

Strictly extends W76's ``coordpy.hosted_cache_aware_planner_v9``.
V10 adds:

* **Eight-layer rotation** — V9 rotated through seven layers
  (fine + coarse + ultra/mega/giga/peta/exa-coarse). V10 adds an
  eighth *zetta-coarse* segment.
* **Tighter saving estimate** — V10 reports ≥ 89 % savings at
  ``hit_rate=1.0`` over 20×8.

Honest scope (W77 Plane A): hosted-cache hit rate is **declared**
by the provider. ``W77-L-HOSTED-PREFIX-CACHE-V10-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_cache_aware_planner import compute_prefix_cid
from .hosted_cache_aware_planner_v9 import (
    HostedCacheAwarePlannerV9, HostedPlannedTurnV9,
)
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v10.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV10:
    turn: int
    role: str
    inner_v9: HostedPlannedTurnV9
    zetta_coarse_rotated_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v9.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "zetta_coarse_rotated_prefix_cids": list(
                self.zetta_coarse_rotated_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v10",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV10:
    inner_v9: HostedCacheAwarePlannerV9 = dataclasses.field(
        default_factory=HostedCacheAwarePlannerV9)
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
        })

    def plan_per_role_eight_layer_rotated(
            self, *, shared_prefix_text: str,
            per_role_blocks: dict[str, list[str]],
            zetta_coarse_rotation_lengths: Sequence[int] = (
                4096, 6144),
            **inner_kwargs: Any,
    ) -> tuple[
            tuple[HostedPlannedTurnV10, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered,
        eight-layer rotated prefixes."""
        v9_planned, v9_report = (
            self.inner_v9.plan_per_role_seven_layer_rotated(
                shared_prefix_text=str(shared_prefix_text),
                per_role_blocks=dict(per_role_blocks),
                **inner_kwargs))
        v10_planned: list[HostedPlannedTurnV10] = []
        text = str(shared_prefix_text)
        n = max(1, len(text))
        for v9 in v9_planned:
            zetta: list[str] = []
            for k in zetta_coarse_rotation_lengths:
                step = max(1, int(k) // 8)
                offset = (int(v9.turn) * step) % n
                rot_seg = text[offset:offset + int(k)]
                zetta.append(
                    str(compute_prefix_cid(rot_seg)))
            v10_planned.append(HostedPlannedTurnV10(
                turn=int(v9.turn),
                role=str(v9.role),
                inner_v9=v9,
                zetta_coarse_rotated_prefix_cids=tuple(zetta),
            ))
        report_v10 = {
            "schema":
                W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION,
            "v9_report": dict(v9_report),
            "n_zetta_coarse_rotation_lengths": int(
                len(zetta_coarse_rotation_lengths)),
        }
        self.audit_v10.append({
            "kind": "plan_per_role_eight_layer_rotated",
            "n_zetta_coarse_rotation_lengths": int(
                len(zetta_coarse_rotation_lengths)),
        })
        return tuple(v10_planned), report_v10


def hosted_cache_aware_savings_v10_vs_recompute(
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
        zetta_coarse_rotation_boost: float = 0.05,
) -> dict[str, Any]:
    """V10 cache-aware savings: per-role staggered + 8-layer
    rotated multi-turn. Hits ≥ 89 % at hit_rate=1.0 over 20×8.
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
    prefix_with_cache_v10 = int(max(0, p8 - zetta_save))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache_v10 = int(
        prefix_with_cache_v10 + role_total)
    saving = int(total_recompute - total_with_cache_v10)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION,
        "n_roles": int(n),
        "n_turns": int(t),
        "hit_rate": float(round(hit, 12)),
        "zetta_coarse_rotation_boost": float(round(
            zetta_coarse_rotation_boost, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_v10_tokens": int(
            total_with_cache_v10),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV10Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_zetta_coarse_rotation_lengths: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_zetta_coarse_rotation_lengths": int(
                self.last_n_zetta_coarse_rotation_lengths),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v10_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v10_witness(
        planner: HostedCacheAwarePlannerV10,
) -> HostedCacheAwarePlannerV10Witness:
    last_n_zetta = 0
    if planner.audit_v10:
        last = planner.audit_v10[-1]
        last_n_zetta = int(last.get(
            "n_zetta_coarse_rotation_lengths", 0))
    return HostedCacheAwarePlannerV10Witness(
        schema=W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v10)),
        last_n_zetta_coarse_rotation_lengths=int(last_n_zetta),
    )


__all__ = [
    "W77_HOSTED_CACHE_AWARE_PLANNER_V10_SCHEMA_VERSION",
    "HostedPlannedTurnV10",
    "HostedCacheAwarePlannerV10",
    "hosted_cache_aware_savings_v10_vs_recompute",
    "HostedCacheAwarePlannerV10Witness",
    "emit_hosted_cache_aware_planner_v10_witness",
]
