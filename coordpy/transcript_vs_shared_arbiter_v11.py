"""W62 — Transcript-vs-Shared Arbiter V11.

Strictly extends W61's
``coordpy.transcript_vs_shared_arbiter_v10``. V10 had 11 arms.
V11 adds:

  * ``replay_dominance`` — fires when the W62 replay controller V3
    reports a strongly-dominant decision (margin ≥ threshold).

The 12 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``replay_dominance_fidelity = 0`` V11
reduces to V10 (no replay-dominance arm).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v10 import (
    ElevenArmCompareResult,
    W61_TVS_V10_ARMS,
    eleven_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W62_TVS_V11_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v11.v1")
W62_TVS_V11_ARM_REPLAY_DOMINANCE: str = "replay_dominance"


def _build_v11_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W61_TVS_V10_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(W62_TVS_V11_ARM_REPLAY_DOMINANCE)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W62_TVS_V11_ARM_REPLAY_DOMINANCE)
    return tuple(out)


W62_TVS_V11_ARMS: tuple[str, ...] = _build_v11_arms()


@dataclasses.dataclass(frozen=True)
class TwelveArmCompareResult:
    schema: str
    inner_v10: ElevenArmCompareResult
    pick_rates: dict[str, float]
    replay_dominance_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "replay_dominance_used": bool(
                self.replay_dominance_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v11_result",
            "result": self.to_dict()})


def twelve_arm_compare(
        *, per_turn_replay_dominance_fidelities: Sequence[float],
        **v10_kwargs: Any,
) -> TwelveArmCompareResult:
    v10 = eleven_arm_compare(**v10_kwargs)
    # If all replay_dominance_fidelities are zero, V11 reduces to
    # V10 (no replay-dominance arm).
    rd_any = any(
        float(x) > 0.0
        for x in per_turn_replay_dominance_fidelities)
    if not rd_any:
        pick_rates_v11 = dict(v10.pick_rates)
        pick_rates_v11[W62_TVS_V11_ARM_REPLAY_DOMINANCE] = 0.0
        # Renormalise so sum = 1 exactly.
        s = float(sum(pick_rates_v11.values()))
        if s > 0:
            pick_rates_v11 = {
                k: v / s for k, v in pick_rates_v11.items()}
        else:
            pick_rates_v11 = {
                k: (1.0 / float(len(W62_TVS_V11_ARMS)))
                for k in W62_TVS_V11_ARMS}
        return TwelveArmCompareResult(
            schema=W62_TVS_V11_SCHEMA_VERSION,
            inner_v10=v10,
            pick_rates=pick_rates_v11,
            replay_dominance_used=False,
        )
    # Compute replay_dominance arm pick rate as proportional to
    # mean replay-dominance fidelity, redistributing equally from
    # the other arms.
    mean_rd = float(
        sum(per_turn_replay_dominance_fidelities)
        / float(len(per_turn_replay_dominance_fidelities)))
    rd_weight = float(min(1.0, max(0.0, mean_rd)))
    new_rates: dict[str, float] = {
        a: v10.pick_rates.get(a, 0.0) * (1.0 - rd_weight)
        for a in W61_TVS_V10_ARMS}
    new_rates[W62_TVS_V11_ARM_REPLAY_DOMINANCE] = rd_weight
    # Renormalise.
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W62_TVS_V11_ARMS)))
            for k in W62_TVS_V11_ARMS}
    return TwelveArmCompareResult(
        schema=W62_TVS_V11_SCHEMA_VERSION,
        inner_v10=v10,
        pick_rates=new_rates,
        replay_dominance_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV11Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    replay_dominance_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "replay_dominance_used": bool(
                self.replay_dominance_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v11_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v11_witness(
        result: TwelveArmCompareResult,
) -> TVSArbiterV11Witness:
    return TVSArbiterV11Witness(
        schema=W62_TVS_V11_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W62_TVS_V11_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        replay_dominance_used=bool(result.replay_dominance_used),
    )


__all__ = [
    "W62_TVS_V11_SCHEMA_VERSION",
    "W62_TVS_V11_ARM_REPLAY_DOMINANCE",
    "W62_TVS_V11_ARMS",
    "TwelveArmCompareResult",
    "twelve_arm_compare",
    "TVSArbiterV11Witness",
    "emit_tvs_arbiter_v11_witness",
]
