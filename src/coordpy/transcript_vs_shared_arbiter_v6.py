"""W57 M13 — TVS Arbiter V6 (Seven-Arm).

Extends W56 V5's 6-arm policy with a 7th:

  1. transcript
  2. shared
  3. merge_consensus
  4. trust_weighted_merge
  5. substrate_replay              (substrate full forward)
  6. **substrate_hidden_inject**   (NEW — hidden-state bridge)
  7. abstain

The new ``substrate_hidden_inject`` arm is preferred when the
hidden-state fidelity is high (the substrate's intermediate
hidden state carries enough signal to make hidden-state injection
the right play). The arbiter picks based on the relative
strengths of substrate-final-hidden vs substrate-intermediate-
hidden vs the V5 arms.

V6 strictly extends V5: with ``per_turn_hidden_fidelities = None``
or all zeros, V6 reduces to V5 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W57_TVS_V6_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v6.v1")

W57_TVS_V6_ARMS: tuple[str, ...] = (
    "transcript",
    "shared",
    "merge_consensus",
    "trust_weighted_merge",
    "substrate_replay",
    "substrate_hidden_inject",
    "abstain",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SevenArmCompareResult:
    schema: str
    pick_rates: dict[str, float]
    budget_fractions: dict[str, float]
    pick_log: tuple[str, ...]
    substrate_used: bool
    hidden_inject_used: bool

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v6_result",
            "result": self.to_dict()})


def seven_arm_compare(
        *,
        per_turn_confidences: Sequence[float],
        per_turn_trust_scores: Sequence[float],
        per_turn_merge_retentions: Sequence[float],
        per_turn_tw_retentions: Sequence[float],
        per_turn_substrate_fidelities: (
            Sequence[float] | None) = None,
        per_turn_hidden_fidelities: (
            Sequence[float] | None) = None,
        budget_tokens: int = 4,
) -> SevenArmCompareResult:
    """7-arm comparator with per-arm budget allocator.

    Decision rule per turn:

    * ``substrate_hidden_inject`` if hidden_fidelity is highest
      and ≥ 0.5;
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
        len(per_turn_hidden_fidelities or []))
    counts = {a: 0 for a in W57_TVS_V6_ARMS}
    log: list[str] = []
    sub_used = False
    hid_used = False
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
        scores = {
            "transcript": c,
            "shared": c * 0.9,
            "merge_consensus": m,
            "trust_weighted_merge": t * w,
            "substrate_replay": sf,
            "substrate_hidden_inject": hf,
        }
        # Pick the arm with the highest score above 0.5; else
        # transcript-or-abstain.
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
    rates = {a: float(counts[a]) / float(max(1, n))
              for a in W57_TVS_V6_ARMS}
    # Per-arm budget allocator (proportional to pick rate).
    budget = max(0, int(budget_tokens))
    total_picked = sum(counts[a] for a in W57_TVS_V6_ARMS
                        if a != "abstain") or 1
    budget_fracs: dict[str, float] = {}
    for a in W57_TVS_V6_ARMS:
        if a == "abstain":
            budget_fracs[a] = 0.0
        else:
            budget_fracs[a] = float(counts[a]) / float(total_picked)
    return SevenArmCompareResult(
        schema=W57_TVS_V6_SCHEMA_VERSION,
        pick_rates=rates,
        budget_fractions=budget_fracs,
        pick_log=tuple(log),
        substrate_used=bool(sub_used),
        hidden_inject_used=bool(hid_used),
    )


@dataclasses.dataclass(frozen=True)
class TVSArbiterV6Witness:
    schema: str
    result_cid: str
    n_arms: int
    pick_rates_sum_to_one: bool
    substrate_used: bool
    hidden_inject_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "result_cid": str(self.result_cid),
            "n_arms": int(self.n_arms),
            "pick_rates_sum_to_one": bool(
                self.pick_rates_sum_to_one),
            "substrate_used": bool(self.substrate_used),
            "hidden_inject_used": bool(self.hidden_inject_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tvs_v6_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v6_witness(
        *, result: SevenArmCompareResult,
) -> TVSArbiterV6Witness:
    s = float(sum(result.pick_rates.values()))
    return TVSArbiterV6Witness(
        schema=W57_TVS_V6_SCHEMA_VERSION,
        result_cid=str(result.cid()),
        n_arms=len(W57_TVS_V6_ARMS),
        pick_rates_sum_to_one=bool(abs(s - 1.0) < 1e-9 or s == 0.0),
        substrate_used=bool(result.substrate_used),
        hidden_inject_used=bool(result.hidden_inject_used),
    )


__all__ = [
    "W57_TVS_V6_SCHEMA_VERSION",
    "W57_TVS_V6_ARMS",
    "SevenArmCompareResult",
    "TVSArbiterV6Witness",
    "seven_arm_compare",
    "emit_tvs_arbiter_v6_witness",
]
