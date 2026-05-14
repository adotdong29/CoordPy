"""W58 M15 — TVS Arbiter V7 (Eight-Arm).

Strictly extends W57's
``coordpy.transcript_vs_shared_arbiter_v6``. V7 adds an eighth
arm:

  1. transcript
  2. shared
  3. merge_consensus
  4. trust_weighted_merge
  5. substrate_replay
  6. substrate_hidden_inject
  7. **cache_reuse_replay** (NEW)
  8. abstain

The new ``cache_reuse_replay`` arm is preferred when the cache
fidelity (e.g. flop savings × argmax-preserved frequency) is the
strict highest score among the substrate / hidden / cache axes.

V7 strictly extends V6: with ``per_turn_cache_fidelities = None``
or all zeros, V7 reduces to V6 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W58_TVS_V7_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v7.v1")

W58_TVS_V7_ARMS: tuple[str, ...] = (
    "transcript",
    "shared",
    "merge_consensus",
    "trust_weighted_merge",
    "substrate_replay",
    "substrate_hidden_inject",
    "cache_reuse_replay",
    "abstain",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class EightArmCompareResult:
    schema: str
    pick_rates: dict[str, float]
    budget_fractions: dict[str, float]
    pick_log: tuple[str, ...]
    substrate_used: bool
    hidden_inject_used: bool
    cache_reuse_used: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v7_result",
            "result": self.to_dict()})


def eight_arm_compare(
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
        budget_tokens: int = 4,
) -> EightArmCompareResult:
    """8-arm comparator with per-arm budget allocator.

    Decision rule per turn:

    * ``cache_reuse_replay`` if cache_fidelity is the strict
      highest and ≥ 0.5;
    * else ``substrate_hidden_inject`` if hidden_fidelity is
      highest and ≥ 0.5;
    * else ``substrate_replay`` if substrate_fidelity is highest
      and ≥ 0.5;
    * else ``trust_weighted_merge`` if trust × tw_retention ≥ 0.5;
    * else ``merge_consensus`` if merge_retention ≥ 0.5;
    * else ``shared`` if confidence ≥ 0.5;
    * else ``transcript`` if confidence ≥ 0.25;
    * else ``abstain``.
    """
    n = max(
        len(per_turn_confidences),
        len(per_turn_trust_scores),
        len(per_turn_merge_retentions),
        len(per_turn_tw_retentions),
        len(per_turn_substrate_fidelities or []),
        len(per_turn_hidden_fidelities or []),
        len(per_turn_cache_fidelities or []))
    counts = {a: 0 for a in W58_TVS_V7_ARMS}
    log: list[str] = []
    sub_used = False
    hid_used = False
    cache_used = False
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
        scores = {
            "transcript": c,
            "shared": c * 0.9,
            "merge_consensus": m,
            "trust_weighted_merge": t * w,
            "substrate_replay": sf,
            "substrate_hidden_inject": hf,
            "cache_reuse_replay": cf,
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
    rates = {a: float(counts[a]) / float(max(1, n))
              for a in W58_TVS_V7_ARMS}
    budget = max(0, int(budget_tokens))
    total_picked = sum(counts[a] for a in W58_TVS_V7_ARMS
                        if a != "abstain") or 1
    budget_fracs: dict[str, float] = {}
    for a in W58_TVS_V7_ARMS:
        if a == "abstain":
            budget_fracs[a] = 0.0
        else:
            budget_fracs[a] = float(
                counts[a]) / float(total_picked)
    return EightArmCompareResult(
        schema=W58_TVS_V7_SCHEMA_VERSION,
        pick_rates=rates,
        budget_fractions=budget_fracs,
        pick_log=tuple(log),
        substrate_used=bool(sub_used),
        hidden_inject_used=bool(hid_used),
        cache_reuse_used=bool(cache_used),
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV7Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum_to_one: bool
    substrate_used: bool
    hidden_inject_used: bool
    cache_reuse_used: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v7_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v7_witness(
        *, result: EightArmCompareResult,
) -> TVSArbiterV7Witness:
    s = float(sum(result.pick_rates.values()))
    return TVSArbiterV7Witness(
        schema=W58_TVS_V7_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=len(W58_TVS_V7_ARMS),
        pick_rates_sum_to_one=bool(
            abs(s - 1.0) < 1e-9 or s == 0.0),
        substrate_used=bool(result.substrate_used),
        hidden_inject_used=bool(result.hidden_inject_used),
        cache_reuse_used=bool(result.cache_reuse_used),
    )


__all__ = [
    "W58_TVS_V7_SCHEMA_VERSION",
    "W58_TVS_V7_ARMS",
    "EightArmCompareResult",
    "TVSArbiterV7Witness",
    "eight_arm_compare",
    "emit_tvs_arbiter_v7_witness",
]
