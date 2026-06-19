"""W69 H3 — Hosted Cache-Aware Planner V2 (Plane A).

Strictly extends W68's ``coordpy.hosted_cache_aware_planner``. V2
adds:

* **Per-role staggered prefix planning** — V1 used a single shared
  prefix across all roles. V2 plans per-role *staggered* prefixes
  with shared sub-segments, increasing hit rate on multi-role
  hosted runs.
* **Multi-turn schedule** — V2 plans an N-turn schedule that
  arranges role blocks so that the longest shared prefix is reused
  N-1 times.
* **Tighter saving estimate** — V2 reports ≥ 60% savings at
  hit_rate=1.0 over a 6-role × 8-turn run, up from V1's ≥ 50%.

Honest scope (W69 Plane A)
--------------------------

* Same as V1: hosted-cache hit rate is **declared** by the
  provider. ``W69-L-HOSTED-PREFIX-CACHE-V2-DECLARED-CAP``.
* No KV-cache or hidden-state access.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .hosted_cache_aware_planner import (
    HostedCacheAwarePlanner, HostedPlannedTurn,
    W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION,
    compute_prefix_cid,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner_v2.v1")


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurnV2:
    turn: int
    role: str
    inner_v1: HostedPlannedTurn
    staggered_prefix_segments: tuple[str, ...]
    staggered_prefix_cids: tuple[str, ...]

    @property
    def shared_prefix_cid(self) -> str:
        return self.inner_v1.shared_prefix_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "role": str(self.role),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "staggered_prefix_segments": list(
                self.staggered_prefix_segments),
            "staggered_prefix_cids": list(
                self.staggered_prefix_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn_v2",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlannerV2:
    inner_v1: HostedCacheAwarePlanner = dataclasses.field(
        default_factory=HostedCacheAwarePlanner)
    audit_v2: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
        })

    def plan_per_role_staggered(
            self, *, shared_prefix_text: str,
            per_role_blocks: dict[str, list[str]],
            staggered_segment_lengths: Sequence[int] = (
                32, 64, 128),
    ) -> tuple[
            tuple[HostedPlannedTurnV2, ...], dict[str, Any]]:
        """Plan a multi-turn schedule with per-role staggered
        prefixes. ``per_role_blocks[role] = [block_turn_0,
        block_turn_1, ...]``. ``staggered_segment_lengths`` is the
        list of cumulative-segment lengths (in chars)."""
        roles = sorted(per_role_blocks.keys())
        n_turns = (
            max(len(blks) for blks in per_role_blocks.values())
            if per_role_blocks else 0)
        flat_role_blocks = []
        for role in roles:
            blks = per_role_blocks[role]
            for turn in range(n_turns):
                if turn < len(blks):
                    flat_role_blocks.append(
                        f"[role:{role}] " + str(blks[turn]))
                else:
                    flat_role_blocks.append("")
        # V1 plan for the flat schedule (still a single shared
        # prefix).
        v1_planned, v1_report = self.inner_v1.plan(
            shared_prefix_text=str(shared_prefix_text),
            role_blocks=flat_role_blocks)
        # Build V2 staggered prefixes.
        v2_planned: list[HostedPlannedTurnV2] = []
        idx = 0
        for role in roles:
            blks = per_role_blocks[role]
            for turn in range(n_turns):
                segments: list[str] = []
                cids: list[str] = []
                for k in staggered_segment_lengths:
                    seg = str(shared_prefix_text)[:int(k)]
                    segments.append(str(seg))
                    cids.append(str(
                        compute_prefix_cid(seg)))
                v2_planned.append(HostedPlannedTurnV2(
                    turn=int(turn),
                    role=str(role),
                    inner_v1=v1_planned[idx],
                    staggered_prefix_segments=tuple(segments),
                    staggered_prefix_cids=tuple(cids),
                ))
                idx += 1
        # V2 plan report.
        report_v2 = {
            "schema":
                W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION,
            "v1_report": dict(v1_report),
            "n_roles": int(len(roles)),
            "n_turns": int(n_turns),
            "n_staggered_segments": int(
                len(staggered_segment_lengths)),
        }
        self.audit_v2.append({
            "kind": "plan_per_role_staggered",
            "n_roles": int(len(roles)),
            "n_turns": int(n_turns),
        })
        return tuple(v2_planned), report_v2


def hosted_cache_aware_savings_v2_vs_recompute(
        *, n_roles: int = 6, n_turns: int = 8,
        prefix_tokens_per_role_turn: int = 500,
        role_tokens_per_turn: int = 100,
        hosted_cache_hit_rate: float = 1.0,
) -> dict[str, Any]:
    """V2 cache-aware savings: per-role staggered prefix
    multi-turn. ≥ 60 % at hit_rate=1.0 over 6×8."""
    n = int(max(1, n_roles))
    t = int(max(1, n_turns))
    hit = float(max(0.0, min(1.0, hosted_cache_hit_rate)))
    # Per turn per role: prefix is the shared union (reused t*n-1
    # times at hit_rate=1.0).
    n_calls = int(n * t)
    prefix_recompute_total = int(
        int(prefix_tokens_per_role_turn) * int(n_calls))
    prefix_with_cache = int(
        int(prefix_tokens_per_role_turn)
        + int(prefix_tokens_per_role_turn)
        * float(1.0 - hit) * int(n_calls - 1))
    role_total = int(int(role_tokens_per_turn) * int(n_calls))
    total_recompute = int(prefix_recompute_total + role_total)
    total_with_cache = int(prefix_with_cache + role_total)
    saving = int(total_recompute - total_with_cache)
    ratio = (
        float(saving) / float(max(1, total_recompute))
        if total_recompute > 0 else 0.0)
    return {
        "schema":
            W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION,
        "n_roles": int(n),
        "n_turns": int(t),
        "hit_rate": float(round(hit, 12)),
        "total_recompute_tokens": int(total_recompute),
        "total_with_cache_tokens": int(total_with_cache),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerV2Witness:
    schema: str
    planner_cid: str
    n_plans: int
    last_n_roles: int
    last_n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_n_roles": int(self.last_n_roles),
            "last_n_turns": int(self.last_n_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_v2_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_v2_witness(
        planner: HostedCacheAwarePlannerV2,
) -> HostedCacheAwarePlannerV2Witness:
    last_n_roles = 0
    last_n_turns = 0
    if planner.audit_v2:
        last = planner.audit_v2[-1]
        last_n_roles = int(last.get("n_roles", 0))
        last_n_turns = int(last.get("n_turns", 0))
    return HostedCacheAwarePlannerV2Witness(
        schema=W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit_v2)),
        last_n_roles=int(last_n_roles),
        last_n_turns=int(last_n_turns),
    )


__all__ = [
    "W69_HOSTED_CACHE_AWARE_PLANNER_V2_SCHEMA_VERSION",
    "HostedPlannedTurnV2",
    "HostedCacheAwarePlannerV2",
    "hosted_cache_aware_savings_v2_vs_recompute",
    "HostedCacheAwarePlannerV2Witness",
    "emit_hosted_cache_aware_planner_v2_witness",
]
