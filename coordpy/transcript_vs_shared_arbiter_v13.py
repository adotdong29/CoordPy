"""W64 — Transcript-vs-Shared Arbiter V13.

Strictly extends W63's ``coordpy.transcript_vs_shared_arbiter_v12``.
V12 had 13 arms. V13 adds:

  * ``replay_dominance_primary`` — fires when the W64 V9 substrate
    reports a positive replay_dominance_witness mean.

The 14 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``replay_dominance_primary_fidelity = 0``
V13 reduces to V12 (no replay-dominance-primary arm).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v12 import (
    ThirteenArmCompareResult,
    W63_TVS_V12_ARMS,
    thirteen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W64_TVS_V13_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v13.v1")
W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY: str = (
    "replay_dominance_primary")


def _build_v13_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W63_TVS_V12_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(
                W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY)
    return tuple(out)


W64_TVS_V13_ARMS: tuple[str, ...] = _build_v13_arms()


@dataclasses.dataclass(frozen=True)
class FourteenArmCompareResult:
    schema: str
    inner_v12: ThirteenArmCompareResult
    pick_rates: dict[str, float]
    replay_dominance_primary_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v12_cid": str(self.inner_v12.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "replay_dominance_primary_used": bool(
                self.replay_dominance_primary_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v13_result",
            "result": self.to_dict()})


def fourteen_arm_compare(
        *, per_turn_replay_dominance_primary_fidelities: (
            Sequence[float]),
        **v12_kwargs: Any,
) -> FourteenArmCompareResult:
    v12 = thirteen_arm_compare(**v12_kwargs)
    rdp_any = any(
        float(x) > 0.0
        for x in per_turn_replay_dominance_primary_fidelities)
    if not rdp_any:
        pick_rates_v13 = dict(v12.pick_rates)
        pick_rates_v13[
            W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY] = 0.0
        s = float(sum(pick_rates_v13.values()))
        if s > 0:
            pick_rates_v13 = {
                k: v / s for k, v in pick_rates_v13.items()}
        else:
            pick_rates_v13 = {
                k: (1.0 / float(len(W64_TVS_V13_ARMS)))
                for k in W64_TVS_V13_ARMS}
        return FourteenArmCompareResult(
            schema=W64_TVS_V13_SCHEMA_VERSION,
            inner_v12=v12,
            pick_rates=pick_rates_v13,
            replay_dominance_primary_used=False,
        )
    mean_rdp = float(
        sum(per_turn_replay_dominance_primary_fidelities)
        / float(len(
            per_turn_replay_dominance_primary_fidelities)))
    rdp_weight = float(min(1.0, max(0.0, mean_rdp)))
    new_rates: dict[str, float] = {
        a: v12.pick_rates.get(a, 0.0) * (1.0 - rdp_weight)
        for a in W63_TVS_V12_ARMS}
    new_rates[
        W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY] = rdp_weight
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W64_TVS_V13_ARMS)))
            for k in W64_TVS_V13_ARMS}
    return FourteenArmCompareResult(
        schema=W64_TVS_V13_SCHEMA_VERSION,
        inner_v12=v12,
        pick_rates=new_rates,
        replay_dominance_primary_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV13Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    replay_dominance_primary_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "replay_dominance_primary_used": bool(
                self.replay_dominance_primary_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v13_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v13_witness(
        result: FourteenArmCompareResult,
) -> TVSArbiterV13Witness:
    return TVSArbiterV13Witness(
        schema=W64_TVS_V13_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W64_TVS_V13_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        replay_dominance_primary_used=bool(
            result.replay_dominance_primary_used),
    )


__all__ = [
    "W64_TVS_V13_SCHEMA_VERSION",
    "W64_TVS_V13_ARM_REPLAY_DOMINANCE_PRIMARY",
    "W64_TVS_V13_ARMS",
    "FourteenArmCompareResult",
    "fourteen_arm_compare",
    "TVSArbiterV13Witness",
    "emit_tvs_arbiter_v13_witness",
]
