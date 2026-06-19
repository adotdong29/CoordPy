"""W61 M18 — TVS Arbiter V10 (Eleven-Arm).

Strictly extends W60's
``coordpy.transcript_vs_shared_arbiter_v9``. V10 adds an eleventh
arm:

  1. transcript
  2. shared
  3. merge_consensus
  4. trust_weighted_merge
  5. substrate_replay
  6. substrate_hidden_inject
  7. cache_reuse_replay
  8. retrieval_replay
  9. replay_controller_choice
  10. **attention_pattern_steer** (NEW)
  11. abstain

The new ``attention_pattern_steer`` arm is preferred when the
*attention-pattern fidelity* (attention-steering-V5 top-K Jaccard
to a reference pattern) is the strict highest score across the
substrate / hidden / cache / retrieval / replay / attention-pattern
axes.

V10 strictly extends V9: with ``per_turn_attention_pattern_fidelities``
None or all zeros, V10 reduces to V9 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W61_TVS_V10_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v10.v1")

W61_TVS_V10_ARMS: tuple[str, ...] = (
    "transcript",
    "shared",
    "merge_consensus",
    "trust_weighted_merge",
    "substrate_replay",
    "substrate_hidden_inject",
    "cache_reuse_replay",
    "retrieval_replay",
    "replay_controller_choice",
    "attention_pattern_steer",
    "abstain",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class ElevenArmCompareResult:
    schema: str
    pick_rates: dict[str, float]
    budget_fractions: dict[str, float]
    pick_log: tuple[str, ...]
    substrate_used: bool
    hidden_inject_used: bool
    cache_reuse_used: bool
    retrieval_used: bool
    replay_controller_used: bool
    attention_pattern_used: bool

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
            "hidden_inject_used": bool(
                self.hidden_inject_used),
            "cache_reuse_used": bool(self.cache_reuse_used),
            "retrieval_used": bool(self.retrieval_used),
            "replay_controller_used": bool(
                self.replay_controller_used),
            "attention_pattern_used": bool(
                self.attention_pattern_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v10_result",
            "result": self.to_dict()})


def eleven_arm_compare(
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
        per_turn_replay_fidelities: (
            Sequence[float] | None) = None,
        per_turn_attention_pattern_fidelities: (
            Sequence[float] | None) = None,
        budget_tokens: int = 4,
) -> ElevenArmCompareResult:
    n = max(
        len(per_turn_confidences),
        len(per_turn_trust_scores),
        len(per_turn_merge_retentions),
        len(per_turn_tw_retentions),
        len(per_turn_substrate_fidelities or []),
        len(per_turn_hidden_fidelities or []),
        len(per_turn_cache_fidelities or []),
        len(per_turn_retrieval_fidelities or []),
        len(per_turn_replay_fidelities or []),
        len(per_turn_attention_pattern_fidelities or []))
    counts = {a: 0 for a in W61_TVS_V10_ARMS}
    log: list[str] = []
    used = {
        "substrate": False, "hidden_inject": False,
        "cache_reuse": False, "retrieval": False,
        "replay_controller": False,
        "attention_pattern": False,
    }
    for i in range(int(n)):
        def gv(seq, i):
            if seq is None: return 0.0
            return float(seq[i]) if i < len(seq) else 0.0
        c = gv(per_turn_confidences, i)
        t = gv(per_turn_trust_scores, i)
        m = gv(per_turn_merge_retentions, i)
        w = gv(per_turn_tw_retentions, i)
        sf = gv(per_turn_substrate_fidelities, i)
        hf = gv(per_turn_hidden_fidelities, i)
        cf = gv(per_turn_cache_fidelities, i)
        rf = gv(per_turn_retrieval_fidelities, i)
        rp = gv(per_turn_replay_fidelities, i)
        ap = gv(per_turn_attention_pattern_fidelities, i)
        # Rank candidates.
        scores = {
            "transcript": c,
            "shared": c * t,
            "merge_consensus": m,
            "trust_weighted_merge": w,
            "substrate_replay": sf,
            "substrate_hidden_inject": hf,
            "cache_reuse_replay": cf,
            "retrieval_replay": rf,
            "replay_controller_choice": rp,
            "attention_pattern_steer": ap,
        }
        best_arm, best_v = max(
            scores.items(), key=lambda kv: kv[1])
        # Abstain when best is degenerate.
        if best_v < 1e-9:
            best_arm = "abstain"
        counts[best_arm] += 1
        log.append(str(best_arm))
        if best_arm == "substrate_replay":
            used["substrate"] = True
        if best_arm == "substrate_hidden_inject":
            used["hidden_inject"] = True
        if best_arm == "cache_reuse_replay":
            used["cache_reuse"] = True
        if best_arm == "retrieval_replay":
            used["retrieval"] = True
        if best_arm == "replay_controller_choice":
            used["replay_controller"] = True
        if best_arm == "attention_pattern_steer":
            used["attention_pattern"] = True
    total = max(int(n), 1)
    pick_rates = {
        a: float(counts[a]) / float(total)
        for a in W61_TVS_V10_ARMS}
    budget_fractions = {
        a: float(counts[a]) / float(total)
        for a in W61_TVS_V10_ARMS}
    return ElevenArmCompareResult(
        schema=W61_TVS_V10_SCHEMA_VERSION,
        pick_rates=pick_rates,
        budget_fractions=budget_fractions,
        pick_log=tuple(log),
        substrate_used=bool(used["substrate"]),
        hidden_inject_used=bool(used["hidden_inject"]),
        cache_reuse_used=bool(used["cache_reuse"]),
        retrieval_used=bool(used["retrieval"]),
        replay_controller_used=bool(
            used["replay_controller"]),
        attention_pattern_used=bool(used["attention_pattern"]),
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV10Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rate_sum: float
    attention_pattern_arm_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rate_sum": float(round(
                self.pick_rate_sum, 12)),
            "attention_pattern_arm_active": bool(
                self.attention_pattern_arm_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v10_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v10_witness(
        result: ElevenArmCompareResult,
) -> TVSArbiterV10Witness:
    return TVSArbiterV10Witness(
        schema=W61_TVS_V10_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=int(len(W61_TVS_V10_ARMS)),
        pick_rate_sum=float(sum(result.pick_rates.values())),
        attention_pattern_arm_active=bool(
            result.attention_pattern_used),
    )


__all__ = [
    "W61_TVS_V10_SCHEMA_VERSION",
    "W61_TVS_V10_ARMS",
    "ElevenArmCompareResult",
    "TVSArbiterV10Witness",
    "eleven_arm_compare",
    "emit_tvs_arbiter_v10_witness",
]
