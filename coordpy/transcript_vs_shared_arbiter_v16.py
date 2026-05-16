"""W67 M19 — Transcript-vs-Shared Arbiter V16.

Strictly extends W66's ``coordpy.transcript_vs_shared_arbiter_v15``.
V15 had 16 arms. V16 adds a 17th:

  * ``branch_merge_reconciliation`` — fires when the W67
    multi-agent branch-merge scalar is above threshold.

The 17 arms over their respective fidelities; pick rates sum to
1.0 within 1e-9. When ``branch_merge_reconciliation_fidelity = 0``
V16 reduces to V15.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .transcript_vs_shared_arbiter_v15 import (
    SixteenArmCompareResult,
    W66_TVS_V15_ARMS,
    sixteen_arm_compare,
)
from .tiny_substrate_v3 import _sha256_hex


W67_TVS_V16_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v16.v1")
W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION: str = (
    "branch_merge_reconciliation")


def _build_v16_arms() -> tuple[str, ...]:
    out = []
    inserted = False
    for arm in W66_TVS_V15_ARMS:
        if (not inserted and arm == "abstain"):
            out.append(W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION)
            inserted = True
        out.append(arm)
    if not inserted:
        out.append(W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION)
    return tuple(out)


W67_TVS_V16_ARMS: tuple[str, ...] = _build_v16_arms()


@dataclasses.dataclass(frozen=True)
class SeventeenArmCompareResult:
    schema: str
    inner_v15: SixteenArmCompareResult
    pick_rates: dict[str, float]
    branch_merge_reconciliation_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v15_cid": str(self.inner_v15.cid()),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "branch_merge_reconciliation_used": bool(
                self.branch_merge_reconciliation_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v16_result",
            "result": self.to_dict()})


def seventeen_arm_compare(
        *, per_turn_branch_merge_reconciliation_fidelities: (
            Sequence[float]),
        **v15_kwargs: Any,
) -> SeventeenArmCompareResult:
    v15 = sixteen_arm_compare(**v15_kwargs)
    bmr_any = any(
        float(x) > 0.0
        for x in per_turn_branch_merge_reconciliation_fidelities)
    if not bmr_any:
        rates_v16 = dict(v15.pick_rates)
        rates_v16[
            W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION] = 0.0
        s = float(sum(rates_v16.values()))
        if s > 0:
            rates_v16 = {
                k: v / s for k, v in rates_v16.items()}
        else:
            rates_v16 = {
                k: (1.0 / float(len(W67_TVS_V16_ARMS)))
                for k in W67_TVS_V16_ARMS}
        return SeventeenArmCompareResult(
            schema=W67_TVS_V16_SCHEMA_VERSION,
            inner_v15=v15,
            pick_rates=rates_v16,
            branch_merge_reconciliation_used=False,
        )
    mean_bmr = float(
        sum(per_turn_branch_merge_reconciliation_fidelities)
        / float(len(
            per_turn_branch_merge_reconciliation_fidelities)))
    bmr_weight = float(min(1.0, max(0.0, mean_bmr)))
    new_rates: dict[str, float] = {
        a: v15.pick_rates.get(a, 0.0) * (1.0 - bmr_weight)
        for a in W66_TVS_V15_ARMS}
    new_rates[
        W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION] = bmr_weight
    s = float(sum(new_rates.values()))
    if s > 0:
        new_rates = {k: v / s for k, v in new_rates.items()}
    else:
        new_rates = {
            k: (1.0 / float(len(W67_TVS_V16_ARMS)))
            for k in W67_TVS_V16_ARMS}
    return SeventeenArmCompareResult(
        schema=W67_TVS_V16_SCHEMA_VERSION,
        inner_v15=v15,
        pick_rates=new_rates,
        branch_merge_reconciliation_used=True,
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV16Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum: float
    branch_merge_reconciliation_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "branch_merge_reconciliation_used": bool(
                self.branch_merge_reconciliation_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v16_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v16_witness(
        result: SeventeenArmCompareResult,
) -> TVSArbiterV16Witness:
    return TVSArbiterV16Witness(
        schema=W67_TVS_V16_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W67_TVS_V16_ARMS)),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        branch_merge_reconciliation_used=bool(
            result.branch_merge_reconciliation_used),
    )


__all__ = [
    "W67_TVS_V16_SCHEMA_VERSION",
    "W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION",
    "W67_TVS_V16_ARMS",
    "SeventeenArmCompareResult",
    "seventeen_arm_compare",
    "TVSArbiterV16Witness",
    "emit_tvs_arbiter_v16_witness",
]
