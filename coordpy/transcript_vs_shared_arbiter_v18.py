"""W69 M17 — Transcript-vs-Shared Arbiter V18.

Strictly extends W68's ``coordpy.transcript_vs_shared_arbiter_v17``.
V17 had 18 arms. V18 adds a 19th:

  * ``multi_branch_rejoin_resolution`` — fires when the W69
    multi-agent multi-branch-rejoin scalar is above threshold.

The 19 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When
``multi_branch_rejoin_resolution_fidelity = 0`` V18 reduces to V17.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v17 import (
    EighteenArmCompareResult,
    W68_TVS_V17_ARMS,
    eighteen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W69_TVS_V18_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v18.v1")
W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION: str = (
    "multi_branch_rejoin_resolution")


def _build_v18_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W68_TVS_V17_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(
                W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(
            W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION)
    return tuple(out)


W69_TVS_V18_ARMS: tuple[str, ...] = _build_v18_arms()


@dataclasses.dataclass(frozen=True)
class NineteenArmCompareResult:
    schema: str
    inner_v17: EighteenArmCompareResult
    pick_rates: dict[str, float]
    multi_branch_rejoin_resolution_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v17_cid": str(self.inner_v17.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "multi_branch_rejoin_resolution_used": bool(
                self.multi_branch_rejoin_resolution_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v18_result",
            "result": self.to_dict()})


def nineteen_arm_compare(
        *, per_turn_multi_branch_rejoin_resolution_fidelities: (
            Sequence[float]),
        **v17_kwargs: Any,
) -> NineteenArmCompareResult:
    v17 = eighteen_arm_compare(**v17_kwargs)
    mbr_any = any(
        float(x) > 0.0
        for x in (
            per_turn_multi_branch_rejoin_resolution_fidelities))
    if not mbr_any:
        rates_v18 = dict(v17.pick_rates)
        rates_v18[
            W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION] = 0.0
        s = float(sum(rates_v18.values()))
        if s > 0:
            rates_v18 = {
                k: v / s for k, v in rates_v18.items()}
        else:
            rates_v18 = {
                k: (1.0 / float(len(W69_TVS_V18_ARMS)))
                for k in W69_TVS_V18_ARMS}
        return NineteenArmCompareResult(
            schema=W69_TVS_V18_SCHEMA_VERSION,
            inner_v17=v17,
            pick_rates=rates_v18,
            multi_branch_rejoin_resolution_used=False,
        )
    mean_mbr = float(
        sum(per_turn_multi_branch_rejoin_resolution_fidelities)
        / float(len(
            per_turn_multi_branch_rejoin_resolution_fidelities))
    )
    mbr_weight = float(min(1.0, max(0.0, mean_mbr)))
    new_rates: dict[str, float] = {
        a: v17.pick_rates.get(a, 0.0) * (1.0 - mbr_weight)
        for a in W68_TVS_V17_ARMS}
    new_rates[
        W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION] = (
            mbr_weight)
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W69_TVS_V18_ARMS)))
            for k in W69_TVS_V18_ARMS}
    return NineteenArmCompareResult(
        schema=W69_TVS_V18_SCHEMA_VERSION,
        inner_v17=v17,
        pick_rates=new_rates,
        multi_branch_rejoin_resolution_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV18Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    multi_branch_rejoin_resolution_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "multi_branch_rejoin_resolution_used": bool(
                self.multi_branch_rejoin_resolution_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v18_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v18_witness(
        result: NineteenArmCompareResult,
) -> TVSArbiterV18Witness:
    return TVSArbiterV18Witness(
        schema=W69_TVS_V18_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W69_TVS_V18_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        multi_branch_rejoin_resolution_used=bool(
            result.multi_branch_rejoin_resolution_used),
    )


__all__ = [
    "W69_TVS_V18_SCHEMA_VERSION",
    "W69_TVS_V18_ARM_MULTI_BRANCH_REJOIN_RESOLUTION",
    "W69_TVS_V18_ARMS",
    "NineteenArmCompareResult",
    "nineteen_arm_compare",
    "TVSArbiterV18Witness",
    "emit_tvs_arbiter_v18_witness",
]
