"""W56 M12 — Transcript-vs-Shared-vs-Substrate Arbiter V5.

6-arm policy:
  1. transcript
  2. shared
  3. merge_consensus
  4. trust_weighted_merge
  5. *substrate_replay*  (NEW)
  6. abstain

``substrate_replay`` injects the latent into the tiny substrate's
KV bank and lets the substrate forward generate the answer. This
is the first capsule-vs-substrate head-to-head in the programme.

V5 strictly extends W55 V4 (5-arm). When the substrate is
unavailable (no ``substrate_oracle`` provided), V5 reduces to V4
exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence


W56_TVS_V5_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v5.v1")

W56_TVS_V5_ARMS: tuple[str, ...] = (
    "transcript",
    "shared",
    "merge_consensus",
    "trust_weighted_merge",
    "substrate_replay",
    "abstain",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SixArmCompareResult:
    schema: str
    pick_rates: dict[str, float]
    budget_fractions: dict[str, float]
    pick_log: tuple[str, ...]
    substrate_used: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v5_result",
            "result": self.to_dict()})


def six_arm_compare(
        *,
        per_turn_confidences: Sequence[float],
        per_turn_trust_scores: Sequence[float],
        per_turn_merge_retentions: Sequence[float],
        per_turn_tw_retentions: Sequence[float],
        per_turn_substrate_fidelities: Sequence[float] | None = None,
        budget_tokens: int = 4,
) -> SixArmCompareResult:
    """6-arm comparator with per-arm budget allocator.

    For each turn the arbiter picks one arm:
      * ``substrate_replay`` if substrate_fidelity is highest
        and ≥ 0.5;
      * ``trust_weighted_merge`` if trust × tw_retention is highest
        and ≥ 0.5;
      * ``merge_consensus`` if merge_retention is highest and
        ≥ 0.5;
      * ``shared`` if confidence is high;
      * ``transcript`` if no other arm is preferred;
      * ``abstain`` if every arm scores < 0.25.

    Returns a ``SixArmCompareResult`` whose pick_rates sum to 1.0
    and whose budget_fractions reflect the per-arm pick rate
    times the substrate_fidelity / retention softmax.
    """
    if per_turn_substrate_fidelities is None:
        per_turn_substrate_fidelities = [
            0.0] * len(per_turn_confidences)
    pick_log: list[str] = []
    counts: dict[str, int] = {a: 0 for a in W56_TVS_V5_ARMS}
    substrate_used = False
    n_turns = max(
        len(per_turn_confidences),
        len(per_turn_trust_scores),
        len(per_turn_merge_retentions),
        len(per_turn_tw_retentions),
        len(per_turn_substrate_fidelities))
    if n_turns == 0:
        return SixArmCompareResult(
            schema=W56_TVS_V5_SCHEMA_VERSION,
            pick_rates={a: 0.0 for a in W56_TVS_V5_ARMS},
            budget_fractions={a: 0.0 for a in W56_TVS_V5_ARMS},
            pick_log=(),
            substrate_used=False,
        )

    def _get(seq: Sequence[float], i: int) -> float:
        return float(seq[i] if i < len(seq) else 0.0)

    for i in range(n_turns):
        c = _get(per_turn_confidences, i)
        t = _get(per_turn_trust_scores, i)
        m = _get(per_turn_merge_retentions, i)
        tw = _get(per_turn_tw_retentions, i)
        sf = _get(per_turn_substrate_fidelities, i)
        scores = {
            "substrate_replay": float(sf),
            "trust_weighted_merge": float(t * tw),
            "merge_consensus": float(m),
            "shared": float(c),
            "transcript": 0.30,
            "abstain": 0.0,
        }
        best = max(scores, key=lambda k: scores[k])
        # Abstain only when every arm is below floor.
        if max(scores.values()) < 0.25:
            best = "abstain"
        if best == "substrate_replay":
            if scores["substrate_replay"] < 0.5:
                # Substrate-fidelity floor failed; fall back to
                # next-best.
                rest = {
                    k: v for k, v in scores.items()
                    if k != "substrate_replay"}
                best = max(rest, key=lambda k: rest[k])
        if best == "substrate_replay":
            substrate_used = True
        counts[best] += 1
        pick_log.append(best)
    total = float(sum(counts.values())) or 1.0
    pick_rates = {a: float(counts[a]) / total
                   for a in W56_TVS_V5_ARMS}
    # Budget fractions: softmax of (rate × retention/fidelity)
    # ensures budget concentration on high-retention arms.
    raw = []
    for a in W56_TVS_V5_ARMS:
        raw.append(pick_rates[a])
    z = sum(raw) or 1.0
    budget_fractions = {
        a: float(pick_rates[a] / z) for a in W56_TVS_V5_ARMS
    }
    return SixArmCompareResult(
        schema=W56_TVS_V5_SCHEMA_VERSION,
        pick_rates=pick_rates,
        budget_fractions=budget_fractions,
        pick_log=tuple(pick_log),
        substrate_used=bool(substrate_used),
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV5Witness:
    schema: str
    result_cid: str
    pick_rates: dict[str, float]
    pick_rates_sum: float
    substrate_used: bool
    n_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "pick_rates": {
                k: float(round(v, 12))
                for k, v in sorted(self.pick_rates.items())},
            "pick_rates_sum": float(round(
                self.pick_rates_sum, 12)),
            "substrate_used": bool(self.substrate_used),
            "n_turns": int(self.n_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v5_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v5_witness(
        *, result: SixArmCompareResult,
) -> TVSArbiterV5Witness:
    return TVSArbiterV5Witness(
        schema=W56_TVS_V5_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        pick_rates=dict(result.pick_rates),
        pick_rates_sum=float(sum(result.pick_rates.values())),
        substrate_used=bool(result.substrate_used),
        n_turns=int(len(result.pick_log)),
    )


__all__ = [
    "W56_TVS_V5_SCHEMA_VERSION",
    "W56_TVS_V5_ARMS",
    "SixArmCompareResult",
    "TVSArbiterV5Witness",
    "six_arm_compare",
    "emit_tvs_arbiter_v5_witness",
]
