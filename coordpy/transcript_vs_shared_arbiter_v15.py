"""W66 M19 — Transcript-vs-Shared Arbiter V15.

Strictly extends W65's ``coordpy.transcript_vs_shared_arbiter_v14``.
V14 had 15 arms. V15 adds a 16th:

  * ``team_failure_recovery`` — fires when the W66 multi-agent
    team-failure-recovery scalar is above threshold.

The 16 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``team_failure_recovery_fidelity = 0`` V15
reduces to V14.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v14 import (
    FifteenArmCompareResult, W65_TVS_V14_ARMS,
    fifteen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W66_TVS_V15_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v15.v1")
W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY: str = (
    "team_failure_recovery")


def _build_v15_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W65_TVS_V14_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY)
    return tuple(out)


W66_TVS_V15_ARMS: tuple[str, ...] = _build_v15_arms()


@dataclasses.dataclass(frozen=True)
class SixteenArmCompareResult:
    schema: str
    inner_v14: FifteenArmCompareResult
    pick_rates: dict[str, float]
    team_failure_recovery_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v14_cid": str(self.inner_v14.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "team_failure_recovery_used": bool(
                self.team_failure_recovery_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v15_result",
            "result": self.to_dict()})


def sixteen_arm_compare(
        *, per_turn_team_failure_recovery_fidelities: (
            Sequence[float]),
        **v14_kwargs: Any,
) -> SixteenArmCompareResult:
    v14 = fifteen_arm_compare(**v14_kwargs)
    tfr_any = any(
        float(x) > 0.0
        for x in per_turn_team_failure_recovery_fidelities)
    if not tfr_any:
        rates_v15 = dict(v14.pick_rates)
        rates_v15[
            W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY] = 0.0
        s = float(sum(rates_v15.values()))
        if s > 0:
            rates_v15 = {
                k: v / s for k, v in rates_v15.items()}
        else:
            rates_v15 = {
                k: (1.0 / float(len(W66_TVS_V15_ARMS)))
                for k in W66_TVS_V15_ARMS}
        return SixteenArmCompareResult(
            schema=W66_TVS_V15_SCHEMA_VERSION,
            inner_v14=v14,
            pick_rates=rates_v15,
            team_failure_recovery_used=False,
        )
    mean_tfr = float(
        sum(per_turn_team_failure_recovery_fidelities)
        / float(len(
            per_turn_team_failure_recovery_fidelities)))
    tfr_weight = float(min(1.0, max(0.0, mean_tfr)))
    new_rates: dict[str, float] = {
        a: v14.pick_rates.get(a, 0.0) * (1.0 - tfr_weight)
        for a in W65_TVS_V14_ARMS}
    new_rates[
        W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY] = tfr_weight
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W66_TVS_V15_ARMS)))
            for k in W66_TVS_V15_ARMS}
    return SixteenArmCompareResult(
        schema=W66_TVS_V15_SCHEMA_VERSION,
        inner_v14=v14,
        pick_rates=new_rates,
        team_failure_recovery_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV15Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    team_failure_recovery_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "team_failure_recovery_used": bool(
                self.team_failure_recovery_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v15_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v15_witness(
        result: SixteenArmCompareResult,
) -> TVSArbiterV15Witness:
    return TVSArbiterV15Witness(
        schema=W66_TVS_V15_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W66_TVS_V15_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        team_failure_recovery_used=bool(
            result.team_failure_recovery_used),
    )


__all__ = [
    "W66_TVS_V15_SCHEMA_VERSION",
    "W66_TVS_V15_ARM_TEAM_FAILURE_RECOVERY",
    "W66_TVS_V15_ARMS",
    "SixteenArmCompareResult",
    "sixteen_arm_compare",
    "TVSArbiterV15Witness",
    "emit_tvs_arbiter_v15_witness",
]
