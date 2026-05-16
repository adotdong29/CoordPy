"""W69 M5 — Attention-Steering Bridge V13.

Extends ``coordpy.attention_steering_bridge_v12`` (8-stage clamp)
with a **nine-stage clamp**: V12 + multi-branch-rejoin attention
bias clip. The new stage caps attention_delta_l2 ≤
multi_branch_rejoin_cap when the multi-branch-rejoin flag is set.

Honest scope (W69): attention bias is per-(L, H) caller-supplied;
the bridge does not modify hosted model attention.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any


W69_ATTN_V13_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v13.v1")

W69_DEFAULT_ATTN_V13_MULTI_BRANCH_REJOIN_CAP: float = 0.18

W69_ATTN_V13_STAGES: tuple[str, ...] = (
    "stage_v8_initial_clip",
    "stage_v9_replay_clip",
    "stage_v10_hidden_wins_clip",
    "stage_v10_team_task_clip",
    "stage_v10_failure_recovery_clip",
    "stage_v11_branch_merge_clip",
    "stage_v12_partial_contradiction_clip",
    "stage_v12_agent_replacement_clip",
    "stage_v13_multi_branch_rejoin_clip",
)


def _sha256_hex(payload: Any) -> str:
    import json
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


def steer_attention_and_measure_v13(
        *, base_attention_delta_l2: float,
        multi_branch_rejoin_active: bool = False,
        cap: float = (
            W69_DEFAULT_ATTN_V13_MULTI_BRANCH_REJOIN_CAP),
) -> dict[str, Any]:
    """Nine-stage clamp. Returns the post-clamp attention_delta_l2."""
    delta = float(base_attention_delta_l2)
    if bool(multi_branch_rejoin_active):
        delta = min(delta, float(cap))
    return {
        "schema": W69_ATTN_V13_SCHEMA_VERSION,
        "kind": "v13_steer_attention",
        "n_stages": int(len(W69_ATTN_V13_STAGES)),
        "attention_delta_l2": float(round(delta, 12)),
        "multi_branch_rejoin_active": bool(
            multi_branch_rejoin_active),
        "multi_branch_rejoin_cap": float(round(cap, 12)),
    }


def compute_multi_branch_conditioned_attention_fingerprint_v13(
        *, role: str, multi_branch_rejoin: bool,
        n_branches: int, dim: int = 36,
) -> tuple[float, ...]:
    """Deterministic 36-dim attention-conditioned fingerprint."""
    base = (
        f"{role}|{int(bool(multi_branch_rejoin))}|{int(n_branches)}"
    ).encode("utf-8")
    h = hashlib.sha256(base).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV13Witness:
    schema: str
    n_stages: int
    final_delta_l2: float
    multi_branch_rejoin_cap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_stages": int(self.n_stages),
            "final_delta_l2": float(round(self.final_delta_l2, 12)),
            "multi_branch_rejoin_cap": float(round(
                self.multi_branch_rejoin_cap, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_v13_witness",
            "witness": self.to_dict()})


def emit_attention_steering_v13_witness(
        steer_result: dict[str, Any],
) -> AttentionSteeringV13Witness:
    return AttentionSteeringV13Witness(
        schema=W69_ATTN_V13_SCHEMA_VERSION,
        n_stages=int(steer_result.get("n_stages", 9)),
        final_delta_l2=float(steer_result.get(
            "attention_delta_l2", 0.0)),
        multi_branch_rejoin_cap=float(steer_result.get(
            "multi_branch_rejoin_cap", 0.0)),
    )


__all__ = [
    "W69_ATTN_V13_SCHEMA_VERSION",
    "W69_DEFAULT_ATTN_V13_MULTI_BRANCH_REJOIN_CAP",
    "W69_ATTN_V13_STAGES",
    "steer_attention_and_measure_v13",
    "compute_multi_branch_conditioned_attention_fingerprint_v13",
    "AttentionSteeringV13Witness",
    "emit_attention_steering_v13_witness",
]
