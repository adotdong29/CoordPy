"""W65 M11 — Multi-Hop Translator V15.

Strictly extends W64's ``coordpy.multi_hop_translator_v14``. V14
had 27 backends, 702 directed edges, chain-length 21, 9-axis
composite. V15 adds:

* **35 backends** — V14's 27 + 8 new (BB..II).
* **1190 directed edges** — 35 × 34.
* **Chain length 25**.
* **10-axis composite trust** — adds ``team_coordination_trust``.
* Compromise threshold in [1, 10].

Honest scope (W65)
------------------

* ``W65-L-MULTI-HOP-V15-SYNTHETIC-BACKENDS-CAP`` — the 35 backends
  are NAMED, not EXECUTED.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .multi_hop_translator_v14 import (
    DecBackendChainPathV14,
    W64_DEFAULT_MH_V14_AGREEMENT_FLOOR,
)
from .tiny_substrate_v3 import _sha256_hex


W65_MULTI_HOP_V15_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v15.v1")

W65_DEFAULT_MH_V15_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "AA", "BB", "CC", "DD", "EE", "FF",
    "GG", "HH", "II")
W65_DEFAULT_MH_V15_CHAIN_LEN: int = 25
W65_DEFAULT_MH_V15_AGREEMENT_FLOOR: float = (
    W64_DEFAULT_MH_V14_AGREEMENT_FLOOR)


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV15:
    inner_v14: DecBackendChainPathV14
    team_coordination_trust: float = 1.0

    @property
    def chain(self) -> tuple[str, ...]:
        return self.inner_v14.chain

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v14.payload

    @property
    def confidence(self) -> float:
        return float(self.inner_v14.confidence)

    @property
    def composite_trust_v15(self) -> float:
        return float(
            self.inner_v14.composite_trust
            * self.team_coordination_trust)


def ten_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV15],
        agreement_floor: float = (
            W65_DEFAULT_MH_V15_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    surviving = [p for p in paths
                 if p.composite_trust_v15 >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    payloads = [list(p.payload) for p in surviving]
    weights = [
        float(p.composite_trust_v15 * p.confidence)
        for p in surviving]
    if sum(weights) < 1e-9:
        return [], {
            "n_paths": int(len(surviving)),
            "kind": "abstain",
            "rationale": "zero_weights"}
    Dim = max(len(p) for p in payloads)
    out = [0.0] * Dim
    for pl, w in zip(payloads, weights):
        for i in range(min(Dim, len(pl))):
            out[i] += float(pl[i]) * float(w)
    sumw = float(sum(weights))
    for i in range(Dim):
        out[i] /= sumw
    return out, {
        "n_paths": int(len(surviving)),
        "kind": "ten_axis_weighted_mean",
        "rationale": "ten_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v15(
        *, paths: Sequence[DecBackendChainPathV15],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration outcome.
    Bounded in [1, 10]."""
    if not paths:
        return 10
    base, _ = ten_axis_trust_arbitration(paths=paths)
    if not base:
        return 10
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust_v15)
    dominant = sorted_paths[0]
    for k in range(1, 11):
        if k == 10:
            inner_attack = dataclasses.replace(
                dominant.inner_v14,
                substrate_trust=0.0, hidden_trust=0.0,
                attention_trust=0.0, retrieval_trust=0.0,
                replay_trust=0.0,
                attention_pattern_trust=0.0,
                replay_dominance_trust=0.0,
                hidden_wins_trust=0.0,
                replay_dominance_primary_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v14=inner_attack,
                team_coordination_trust=0.0)
        elif k < 10:
            # Attack k inner axes; team_coordination_trust still up.
            field_names = (
                "substrate_trust", "hidden_trust",
                "attention_trust", "retrieval_trust",
                "replay_trust", "attention_pattern_trust",
                "replay_dominance_trust", "hidden_wins_trust",
                "replay_dominance_primary_trust")
            kwargs = {f: 0.0 for f in field_names[:k]}
            inner_attack = dataclasses.replace(
                dominant.inner_v14, **kwargs)
            attacked = dataclasses.replace(
                dominant, inner_v14=inner_attack)
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = ten_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 10


@dataclasses.dataclass(frozen=True)
class MultiHopV15Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    ten_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "ten_axis_used": bool(self.ten_axis_used),
            "compromise_threshold": int(self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v15_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len25_fidelity(
        *, backends: Sequence[str] = (
            W65_DEFAULT_MH_V15_BACKENDS),
        chain_length: int = W65_DEFAULT_MH_V15_CHAIN_LEN,
        seed: int = 0,
) -> dict[str, Any]:
    """Synthetic chain-length-25 fidelity probe."""
    n_b = int(len(backends))
    # Synthetic paths: one strong + a few moderates.
    strong = DecBackendChainPathV15(
        inner_v14=DecBackendChainPathV14(
            chain=tuple(backends[:int(chain_length)]),
            payload=(0.6, 0.4, 0.2),
            confidence=0.9, substrate_trust=0.9,
            hidden_trust=0.9, attention_trust=0.9,
            retrieval_trust=0.9, replay_trust=0.9,
            attention_pattern_trust=0.9,
            replay_dominance_trust=0.9,
            hidden_wins_trust=0.9,
            replay_dominance_primary_trust=0.9),
        team_coordination_trust=0.9)
    paths = [strong]
    for i in range(3):
        paths.append(DecBackendChainPathV15(
            inner_v14=DecBackendChainPathV14(
                chain=tuple(backends[i:int(chain_length) + i]),
                payload=(0.5, 0.5, 0.5),
                confidence=0.5, substrate_trust=0.5,
                hidden_trust=0.5, attention_trust=0.5,
                retrieval_trust=0.5, replay_trust=0.5,
                attention_pattern_trust=0.5,
                replay_dominance_trust=0.5,
                hidden_wins_trust=0.5,
                replay_dominance_primary_trust=0.5),
            team_coordination_trust=0.5))
    out, audit = ten_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v15(paths=paths)
    return {
        "schema": W65_MULTI_HOP_V15_SCHEMA_VERSION,
        "n_backends": int(n_b),
        "chain_length": int(chain_length),
        "n_edges": int(n_b * (n_b - 1)),
        "compromise_threshold": int(threshold),
        "kind": str(audit["kind"]),
        "rationale": str(audit["rationale"]),
        "n_paths": int(audit["n_paths"]),
    }


def emit_multi_hop_v15_witness(
        *, backends: Sequence[str] = (
            W65_DEFAULT_MH_V15_BACKENDS),
        chain_length: int = W65_DEFAULT_MH_V15_CHAIN_LEN,
        seed: int = 0,
) -> MultiHopV15Witness:
    res = evaluate_dec_chain_len25_fidelity(
        backends=backends, chain_length=int(chain_length),
        seed=int(seed))
    return MultiHopV15Witness(
        schema=W65_MULTI_HOP_V15_SCHEMA_VERSION,
        n_backends=int(res["n_backends"]),
        chain_length=int(res["chain_length"]),
        n_edges=int(res["n_edges"]),
        ten_axis_used=bool("ten_axis" in str(res["kind"])),
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W65_MULTI_HOP_V15_SCHEMA_VERSION",
    "W65_DEFAULT_MH_V15_BACKENDS",
    "W65_DEFAULT_MH_V15_CHAIN_LEN",
    "W65_DEFAULT_MH_V15_AGREEMENT_FLOOR",
    "DecBackendChainPathV15",
    "ten_axis_trust_arbitration",
    "estimate_compromise_threshold_v15",
    "MultiHopV15Witness",
    "evaluate_dec_chain_len25_fidelity",
    "emit_multi_hop_v15_witness",
]
