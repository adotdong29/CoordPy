"""W63 — Transcript-vs-Shared Arbiter V12.

Strictly extends W62's ``coordpy.transcript_vs_shared_arbiter_v11``.
V11 had 12 arms. V12 adds:

  * ``hidden_wins`` — fires when the W63 hidden-state bridge V7
    reports a positive hidden-wins margin.

The 13 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``hidden_wins_fidelity = 0`` V12 reduces
to V11 (no hidden-wins arm).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v11 import (
    TwelveArmCompareResult,
    W62_TVS_V11_ARMS,
    twelve_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W63_TVS_V12_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v12.v1")
W63_TVS_V12_ARM_HIDDEN_WINS: str = "hidden_wins"


def _build_v12_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W62_TVS_V11_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(W63_TVS_V12_ARM_HIDDEN_WINS)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W63_TVS_V12_ARM_HIDDEN_WINS)
    return tuple(out)


W63_TVS_V12_ARMS: tuple[str, ...] = _build_v12_arms()


@dataclasses.dataclass(frozen=True)
class ThirteenArmCompareResult:
    schema: str
    inner_v11: TwelveArmCompareResult
    pick_rates: dict[str, float]
    hidden_wins_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_cid": str(self.inner_v11.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "hidden_wins_used": bool(self.hidden_wins_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v12_result",
            "result": self.to_dict()})


def thirteen_arm_compare(
        *, per_turn_hidden_wins_fidelities: Sequence[float],
        **v11_kwargs: Any,
) -> ThirteenArmCompareResult:
    v11 = twelve_arm_compare(**v11_kwargs)
    hw_any = any(
        float(x) > 0.0
        for x in per_turn_hidden_wins_fidelities)
    if not hw_any:
        pick_rates_v12 = dict(v11.pick_rates)
        pick_rates_v12[W63_TVS_V12_ARM_HIDDEN_WINS] = 0.0
        s = float(sum(pick_rates_v12.values()))
        if s > 0:
            pick_rates_v12 = {
                k: v / s for k, v in pick_rates_v12.items()}
        else:
            pick_rates_v12 = {
                k: (1.0 / float(len(W63_TVS_V12_ARMS)))
                for k in W63_TVS_V12_ARMS}
        return ThirteenArmCompareResult(
            schema=W63_TVS_V12_SCHEMA_VERSION,
            inner_v11=v11,
            pick_rates=pick_rates_v12,
            hidden_wins_used=False,
        )
    mean_hw = float(
        sum(per_turn_hidden_wins_fidelities)
        / float(len(per_turn_hidden_wins_fidelities)))
    hw_weight = float(min(1.0, max(0.0, mean_hw)))
    new_rates: dict[str, float] = {
        a: v11.pick_rates.get(a, 0.0) * (1.0 - hw_weight)
        for a in W62_TVS_V11_ARMS}
    new_rates[W63_TVS_V12_ARM_HIDDEN_WINS] = hw_weight
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W63_TVS_V12_ARMS)))
            for k in W63_TVS_V12_ARMS}
    return ThirteenArmCompareResult(
        schema=W63_TVS_V12_SCHEMA_VERSION,
        inner_v11=v11,
        pick_rates=new_rates,
        hidden_wins_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV12Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    hidden_wins_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "hidden_wins_used": bool(self.hidden_wins_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v12_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v12_witness(
        result: ThirteenArmCompareResult,
) -> TVSArbiterV12Witness:
    return TVSArbiterV12Witness(
        schema=W63_TVS_V12_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W63_TVS_V12_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        hidden_wins_used=bool(result.hidden_wins_used),
    )


__all__ = [
    "W63_TVS_V12_SCHEMA_VERSION",
    "W63_TVS_V12_ARM_HIDDEN_WINS",
    "W63_TVS_V12_ARMS",
    "ThirteenArmCompareResult",
    "thirteen_arm_compare",
    "TVSArbiterV12Witness",
    "emit_tvs_arbiter_v12_witness",
]
