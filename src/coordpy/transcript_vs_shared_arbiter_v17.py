"""W68 M15 — Transcript-vs-Shared Arbiter V17.

Strictly extends W67's ``coordpy.transcript_vs_shared_arbiter_v16``.
V16 had 17 arms. V17 adds an 18th:

  * ``partial_contradiction_resolution`` — fires when the W68
    multi-agent partial-contradiction scalar is above threshold.

The 18 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When
``partial_contradiction_resolution_fidelity = 0`` V17 reduces to V16.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v16 import (
    SeventeenArmCompareResult,
    W67_TVS_V16_ARMS,
    seventeen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W68_TVS_V17_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v17.v1")
W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION: str = (
    "partial_contradiction_resolution")


def _build_v17_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W67_TVS_V16_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(
                W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(
            W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION)
    return tuple(out)


W68_TVS_V17_ARMS: tuple[str, ...] = _build_v17_arms()


@dataclasses.dataclass(frozen=True)
class EighteenArmCompareResult:
    schema: str
    inner_v16: SeventeenArmCompareResult
    pick_rates: dict[str, float]
    partial_contradiction_resolution_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v16_cid": str(self.inner_v16.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "partial_contradiction_resolution_used": bool(
                self.partial_contradiction_resolution_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v17_result",
            "result": self.to_dict()})


def eighteen_arm_compare(
        *, per_turn_partial_contradiction_resolution_fidelities: (
            Sequence[float]),
        **v16_kwargs: Any,
) -> EighteenArmCompareResult:
    v16 = seventeen_arm_compare(**v16_kwargs)
    pcr_any = any(
        float(x) > 0.0
        for x in (
            per_turn_partial_contradiction_resolution_fidelities))
    if not pcr_any:
        rates_v17 = dict(v16.pick_rates)
        rates_v17[
            W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION] = 0.0
        s = float(sum(rates_v17.values()))
        if s > 0:
            rates_v17 = {
                k: v / s for k, v in rates_v17.items()}
        else:
            rates_v17 = {
                k: (1.0 / float(len(W68_TVS_V17_ARMS)))
                for k in W68_TVS_V17_ARMS}
        return EighteenArmCompareResult(
            schema=W68_TVS_V17_SCHEMA_VERSION,
            inner_v16=v16,
            pick_rates=rates_v17,
            partial_contradiction_resolution_used=False,
        )
    mean_pcr = float(
        sum(per_turn_partial_contradiction_resolution_fidelities)
        / float(len(
            per_turn_partial_contradiction_resolution_fidelities))
    )
    pcr_weight = float(min(1.0, max(0.0, mean_pcr)))
    new_rates: dict[str, float] = {
        a: v16.pick_rates.get(a, 0.0) * (1.0 - pcr_weight)
        for a in W67_TVS_V16_ARMS}
    new_rates[
        W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION] = (
            pcr_weight)
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W68_TVS_V17_ARMS)))
            for k in W68_TVS_V17_ARMS}
    return EighteenArmCompareResult(
        schema=W68_TVS_V17_SCHEMA_VERSION,
        inner_v16=v16,
        pick_rates=new_rates,
        partial_contradiction_resolution_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV17Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    partial_contradiction_resolution_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "partial_contradiction_resolution_used": bool(
                self.partial_contradiction_resolution_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v17_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v17_witness(
        result: EighteenArmCompareResult,
) -> TVSArbiterV17Witness:
    return TVSArbiterV17Witness(
        schema=W68_TVS_V17_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W68_TVS_V17_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        partial_contradiction_resolution_used=bool(
            result.partial_contradiction_resolution_used),
    )


__all__ = [
    "W68_TVS_V17_SCHEMA_VERSION",
    "W68_TVS_V17_ARM_PARTIAL_CONTRADICTION_RESOLUTION",
    "W68_TVS_V17_ARMS",
    "EighteenArmCompareResult",
    "eighteen_arm_compare",
    "TVSArbiterV17Witness",
    "emit_tvs_arbiter_v17_witness",
]
