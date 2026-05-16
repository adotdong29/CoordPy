"""W65 M19 — Transcript-vs-Shared Arbiter V14.

Strictly extends W64's ``coordpy.transcript_vs_shared_arbiter_v13``.
V13 had 14 arms. V14 adds a 15th:

  * ``team_substrate_coordination`` — fires when the W65
    multi-agent coordinator reports team-task-success above
    threshold.

The 15 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``team_coordination_fidelity = 0`` V14
reduces to V13.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v13 import (
    FourteenArmCompareResult, W64_TVS_V13_ARMS,
    fourteen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W65_TVS_V14_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v14.v1")
W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION: str = (
    "team_substrate_coordination")


def _build_v14_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W64_TVS_V13_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(
                W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION)
    return tuple(out)


W65_TVS_V14_ARMS: tuple[str, ...] = _build_v14_arms()


@dataclasses.dataclass(frozen=True)
class FifteenArmCompareResult:
    schema: str
    inner_v13: FourteenArmCompareResult
    pick_rates: dict[str, float]
    team_substrate_coordination_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v13_cid": str(self.inner_v13.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "team_substrate_coordination_used": bool(
                self.team_substrate_coordination_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v14_result",
            "result": self.to_dict()})


def fifteen_arm_compare(
        *, per_turn_team_substrate_coordination_fidelities: (
            Sequence[float]),
        **v13_kwargs: Any,
) -> FifteenArmCompareResult:
    v13 = fourteen_arm_compare(**v13_kwargs)
    ts_any = any(
        float(x) > 0.0
        for x in
        per_turn_team_substrate_coordination_fidelities)
    if not ts_any:
        rates_v14 = dict(v13.pick_rates)
        rates_v14[
            W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION] = 0.0
        s = float(sum(rates_v14.values()))
        if s > 0:
            rates_v14 = {
                k: v / s for k, v in rates_v14.items()}
        else:
            rates_v14 = {
                k: (1.0 / float(len(W65_TVS_V14_ARMS)))
                for k in W65_TVS_V14_ARMS}
        return FifteenArmCompareResult(
            schema=W65_TVS_V14_SCHEMA_VERSION,
            inner_v13=v13,
            pick_rates=rates_v14,
            team_substrate_coordination_used=False,
        )
    mean_ts = float(
        sum(per_turn_team_substrate_coordination_fidelities)
        / float(len(
            per_turn_team_substrate_coordination_fidelities)))
    ts_weight = float(min(1.0, max(0.0, mean_ts)))
    new_rates: dict[str, float] = {
        a: v13.pick_rates.get(a, 0.0) * (1.0 - ts_weight)
        for a in W64_TVS_V13_ARMS}
    new_rates[
        W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION] = ts_weight
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W65_TVS_V14_ARMS)))
            for k in W65_TVS_V14_ARMS}
    return FifteenArmCompareResult(
        schema=W65_TVS_V14_SCHEMA_VERSION,
        inner_v13=v13,
        pick_rates=new_rates,
        team_substrate_coordination_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV14Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    team_substrate_coordination_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "team_substrate_coordination_used": bool(
                self.team_substrate_coordination_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v14_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v14_witness(
        result: FifteenArmCompareResult,
) -> TVSArbiterV14Witness:
    return TVSArbiterV14Witness(
        schema=W65_TVS_V14_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W65_TVS_V14_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        team_substrate_coordination_used=bool(
            result.team_substrate_coordination_used),
    )


__all__ = [
    "W65_TVS_V14_SCHEMA_VERSION",
    "W65_TVS_V14_ARM_TEAM_SUBSTRATE_COORDINATION",
    "W65_TVS_V14_ARMS",
    "FifteenArmCompareResult",
    "fifteen_arm_compare",
    "TVSArbiterV14Witness",
    "emit_tvs_arbiter_v14_witness",
]
