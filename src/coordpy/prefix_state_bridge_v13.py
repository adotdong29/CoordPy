"""W69 M4 — Prefix-State Bridge V13.

Extends ``coordpy.prefix_state_bridge_v12`` with a K=256 drift curve
(2x V12's K=128 path, expanded again from V12's K=192) and an 8-way
comparator over drift-curve L1 areas including a multi-branch-rejoin
column.

Honest scope (W69): K=256 drift curve is structural (no new ridge
fit; uses last-value extrapolation of V12 predictions).
``W69-L-V13-PREFIX-K256-STRUCTURAL-CAP``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .prefix_state_bridge_v12 import (
    W68_DEFAULT_PREFIX_V12_K_STEPS,
)
from .tiny_substrate_v3 import _sha256_hex


W69_PREFIX_V13_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v13.v1")

W69_DEFAULT_PREFIX_V13_K_STEPS: int = 256

W69_PREFIX_V13_COMPARATOR_AXES: tuple[str, ...] = (
    "prefix", "hidden", "replay", "team",
    "recover", "branch", "contradict", "multi_branch_rejoin",
)


def fit_prefix_drift_curve_predictor_v13(
        *, k_steps: int = W69_DEFAULT_PREFIX_V13_K_STEPS,
) -> dict[str, Any]:
    """Structural V13 drift curve: K=256 via last-value extrapolation
    of the V12 K=192 predictor. No new ridge fit."""
    if int(k_steps) < int(W68_DEFAULT_PREFIX_V12_K_STEPS):
        raise ValueError(
            "k_steps must be >= V12's K=192")
    return {
        "schema": W69_PREFIX_V13_SCHEMA_VERSION,
        "k_steps": int(k_steps),
        "structural": True,
        "v12_k_steps": int(W68_DEFAULT_PREFIX_V12_K_STEPS),
    }


def compute_role_task_team_branch_corruption_fingerprint_v13(
        *, role: str, task_name: str, team_id: str,
        branch_id: str = "main", corrupted: bool = False,
        rejoin: bool = False, dim: int = 60,
) -> tuple[float, ...]:
    """Deterministic 60-dim fingerprint."""
    base = (
        f"{role}|{task_name}|{team_id}|{branch_id}|"
        f"{int(bool(corrupted))}|{int(bool(rejoin))}"
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


def compare_prefix_v13_eight_way(
        *,
        prefix_l1_area: float,
        hidden_l1_area: float,
        replay_l1_area: float,
        team_l1_area: float,
        recover_l1_area: float,
        branch_l1_area: float,
        contradict_l1_area: float,
        multi_branch_rejoin_l1_area: float,
) -> dict[str, Any]:
    """Eight-way comparator. Lower L1 area wins."""
    arms: dict[str, float] = {
        "prefix": float(prefix_l1_area),
        "hidden": float(hidden_l1_area),
        "replay": float(replay_l1_area),
        "team": float(team_l1_area),
        "recover": float(recover_l1_area),
        "branch": float(branch_l1_area),
        "contradict": float(contradict_l1_area),
        "multi_branch_rejoin": float(multi_branch_rejoin_l1_area),
    }
    winner = min(arms.items(), key=lambda kv: kv[1])
    return {
        "schema": W69_PREFIX_V13_SCHEMA_VERSION,
        "kind": "v13_prefix_eight_way_compare",
        "arms": arms,
        "winner": str(winner[0]),
        "winner_l1_area": float(round(winner[1], 12)),
    }


@dataclasses.dataclass(frozen=True)
class PrefixStateV13Witness:
    schema: str
    k_steps: int
    comparator_winner: str
    fingerprint_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "k_steps": int(self.k_steps),
            "comparator_winner": str(self.comparator_winner),
            "fingerprint_l1": float(round(
                self.fingerprint_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_v13_witness",
            "witness": self.to_dict()})


def emit_prefix_state_v13_witness(
        *, k_steps: int = W69_DEFAULT_PREFIX_V13_K_STEPS,
        comparator_winner: str = "prefix",
        fingerprint_l1: float = 0.0,
) -> PrefixStateV13Witness:
    return PrefixStateV13Witness(
        schema=W69_PREFIX_V13_SCHEMA_VERSION,
        k_steps=int(k_steps),
        comparator_winner=str(comparator_winner),
        fingerprint_l1=float(fingerprint_l1),
    )


__all__ = [
    "W69_PREFIX_V13_SCHEMA_VERSION",
    "W69_DEFAULT_PREFIX_V13_K_STEPS",
    "W69_PREFIX_V13_COMPARATOR_AXES",
    "fit_prefix_drift_curve_predictor_v13",
    "compute_role_task_team_branch_corruption_fingerprint_v13",
    "compare_prefix_v13_eight_way",
    "PrefixStateV13Witness",
    "emit_prefix_state_v13_witness",
]
