"""W59 M14 — TVS Arbiter V8 (Nine-Arm).

Strictly extends W58's
``coordpy.transcript_vs_shared_arbiter_v7``. V8 adds a ninth arm:

  1. transcript
  2. shared
  3. merge_consensus
  4. trust_weighted_merge
  5. substrate_replay
  6. substrate_hidden_inject
  7. cache_reuse_replay
  8. **retrieval_replay** (NEW)
  9. abstain

The new ``retrieval_replay`` arm is preferred when the
*retrieval fidelity* (e.g. cache-controller-V2 retrieval-top-K
agreement × argmax-preserved frequency) is the strict highest
score among the substrate / hidden / cache / retrieval axes.

V8 strictly extends V7: with ``per_turn_retrieval_fidelities``
None or all zeros, V8 reduces to V7 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W59_TVS_V8_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v8.v1")

W59_TVS_V8_ARMS: tuple[str, ...] = (
    "transcript",
    "shared",
    "merge_consensus",
    "trust_weighted_merge",
    "substrate_replay",
    "substrate_hidden_inject",
    "cache_reuse_replay",
    "retrieval_replay",
    "abstain",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class NineArmCompareResult:
    schema: str
    pick_rates: dict[str, float]
    budget_fractions: dict[str, float]
    pick_log: tuple[str, ...]
    substrate_used: bool
    hidden_inject_used: bool
    cache_reuse_used: bool
    retrieval_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "budget_fractions": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.budget_fractions.items())},
            "pick_log": list(self.pick_log),
            "substrate_used": bool(self.substrate_used),
            "hidden_inject_used": bool(self.hidden_inject_used),
            "cache_reuse_used": bool(self.cache_reuse_used),
            "retrieval_used": bool(self.retrieval_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v8_result",
            "result": self.to_dict()})


def nine_arm_compare(
        *,
        per_turn_confidences: Sequence[float],
        per_turn_trust_scores: Sequence[float],
        per_turn_merge_retentions: Sequence[float],
        per_turn_tw_retentions: Sequence[float],
        per_turn_substrate_fidelities: (
            Sequence[float] | None) = None,
        per_turn_hidden_fidelities: (
            Sequence[float] | None) = None,
        per_turn_cache_fidelities: (
            Sequence[float] | None) = None,
        per_turn_retrieval_fidelities: (
            Sequence[float] | None) = None,
        budget_tokens: int = 4,
) -> NineArmCompareResult:
    n = max(
        len(per_turn_confidences),
        len(per_turn_trust_scores),
        len(per_turn_merge_retentions),
        len(per_turn_tw_retentions),
        len(per_turn_substrate_fidelities or []),
        len(per_turn_hidden_fidelities or []),
        len(per_turn_cache_fidelities or []),
        len(per_turn_retrieval_fidelities or []))
    counts = {a: 0 for a in W59_TVS_V8_ARMS}
    log: list[str] = []
    sub_used = False
    hid_used = False
    cache_used = False
    ret_used = False
    for i in range(int(n)):
        c = float(per_turn_confidences[i]
                   if i < len(per_turn_confidences) else 0.0)
        t = float(per_turn_trust_scores[i]
                   if i < len(per_turn_trust_scores) else 0.0)
        m = float(per_turn_merge_retentions[i]
                   if i < len(per_turn_merge_retentions) else 0.0)
        w = float(per_turn_tw_retentions[i]
                   if i < len(per_turn_tw_retentions) else 0.0)
        sf = (
            float(per_turn_substrate_fidelities[i])
            if (per_turn_substrate_fidelities is not None
                and i < len(per_turn_substrate_fidelities))
            else 0.0)
        hf = (
            float(per_turn_hidden_fidelities[i])
            if (per_turn_hidden_fidelities is not None
                and i < len(per_turn_hidden_fidelities))
            else 0.0)
        cf = (
            float(per_turn_cache_fidelities[i])
            if (per_turn_cache_fidelities is not None
                and i < len(per_turn_cache_fidelities))
            else 0.0)
        rf = (
            float(per_turn_retrieval_fidelities[i])
            if (per_turn_retrieval_fidelities is not None
                and i < len(per_turn_retrieval_fidelities))
            else 0.0)
        scores = {
            "transcript": c,
            "shared": c * 0.9,
            "merge_consensus": m,
            "trust_weighted_merge": t * w,
            "substrate_replay": sf,
            "substrate_hidden_inject": hf,
            "cache_reuse_replay": cf,
            "retrieval_replay": rf,
        }
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] >= 0.5:
            arm = str(best[0])
        elif c >= 0.25:
            arm = "transcript"
        else:
            arm = "abstain"
        counts[arm] = counts.get(arm, 0) + 1
        log.append(arm)
        if arm == "substrate_replay":
            sub_used = True
        elif arm == "substrate_hidden_inject":
            sub_used = True
            hid_used = True
        elif arm == "cache_reuse_replay":
            cache_used = True
        elif arm == "retrieval_replay":
            ret_used = True
    rates = {a: float(counts[a]) / float(max(1, n))
              for a in W59_TVS_V8_ARMS}
    budget = max(0, int(budget_tokens))
    total_picked = sum(counts[a] for a in W59_TVS_V8_ARMS
                        if a != "abstain") or 1
    budget_fracs: dict[str, float] = {}
    for a in W59_TVS_V8_ARMS:
        if a == "abstain":
            budget_fracs[a] = 0.0
        else:
            budget_fracs[a] = float(
                counts[a]) / float(total_picked)
    return NineArmCompareResult(
        schema=W59_TVS_V8_SCHEMA_VERSION,
        pick_rates=rates,
        budget_fractions=budget_fracs,
        pick_log=tuple(log),
        substrate_used=bool(sub_used),
        hidden_inject_used=bool(hid_used),
        cache_reuse_used=bool(cache_used),
        retrieval_used=bool(ret_used),
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV8Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum_to_one: bool
    substrate_used: bool
    hidden_inject_used: bool
    cache_reuse_used: bool
    retrieval_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum_to_one": bool(
                self.pick_rates_sum_to_one),
            "substrate_used": bool(self.substrate_used),
            "hidden_inject_used": bool(self.hidden_inject_used),
            "cache_reuse_used": bool(self.cache_reuse_used),
            "retrieval_used": bool(self.retrieval_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v8_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v8_witness(
        *, result: NineArmCompareResult,
) -> TVSArbiterV8Witness:
    s = float(sum(result.pick_rates.values()))
    return TVSArbiterV8Witness(
        schema=W59_TVS_V8_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=len(W59_TVS_V8_ARMS),
        pick_rates_sum_to_one=bool(
            abs(s - 1.0) < 1e-9 or s == 0.0),
        substrate_used=bool(result.substrate_used),
        hidden_inject_used=bool(result.hidden_inject_used),
        cache_reuse_used=bool(result.cache_reuse_used),
        retrieval_used=bool(result.retrieval_used),
    )


__all__ = [
    "W59_TVS_V8_SCHEMA_VERSION",
    "W59_TVS_V8_ARMS",
    "NineArmCompareResult",
    "TVSArbiterV8Witness",
    "nine_arm_compare",
    "emit_tvs_arbiter_v8_witness",
]
