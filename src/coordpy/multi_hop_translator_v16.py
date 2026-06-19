"""W66 M11 — Multi-Hop Translator V16.

Strictly extends W65's ``coordpy.multi_hop_translator_v15``. V15
had 35 backends, 1190 directed edges, chain-length 25, 10-axis
composite. V16 adds:

* **36 backends** — V15's 35 + 1 new (JJ).
* **1260 directed edges** — 36 × 35.
* **Chain length 26**.
* **11-axis composite trust** — adds ``team_substrate_coordination_trust``.
* Compromise threshold in [1, 11].

Honest scope (W66)
------------------

* ``W66-L-MULTI-HOP-V16-SYNTHETIC-BACKENDS-CAP`` — backends are
  named, not executed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .multi_hop_translator_v15 import (
    DecBackendChainPathV15,
    W65_DEFAULT_MH_V15_AGREEMENT_FLOOR,
    W65_DEFAULT_MH_V15_BACKENDS,
)
from .multi_hop_translator_v14 import DecBackendChainPathV14
from .tiny_substrate_v3 import _sha256_hex


W66_MULTI_HOP_V16_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v16.v1")

W66_DEFAULT_MH_V16_BACKENDS: tuple[str, ...] = (
    *W65_DEFAULT_MH_V15_BACKENDS, "JJ")
W66_DEFAULT_MH_V16_CHAIN_LEN: int = 26
W66_DEFAULT_MH_V16_AGREEMENT_FLOOR: float = (
    W65_DEFAULT_MH_V15_AGREEMENT_FLOOR)


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV16:
    inner_v15: DecBackendChainPathV15
    team_substrate_coordination_trust: float = 1.0

    @property
    def chain(self) -> tuple[str, ...]:
        return self.inner_v15.chain

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v15.payload

    @property
    def confidence(self) -> float:
        return float(self.inner_v15.confidence)

    @property
    def composite_trust_v16(self) -> float:
        return float(
            self.inner_v15.composite_trust_v15
            * self.team_substrate_coordination_trust)


def eleven_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV16],
        agreement_floor: float = (
            W66_DEFAULT_MH_V16_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    surviving = [p for p in paths
                 if p.composite_trust_v16 >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    payloads = [list(p.payload) for p in surviving]
    weights = [
        float(p.composite_trust_v16 * p.confidence)
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
        "kind": "eleven_axis_weighted_mean",
        "rationale": "eleven_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v16(
        *, paths: Sequence[DecBackendChainPathV16],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration. Bounded
    in [1, 11]."""
    if not paths:
        return 11
    base, _ = eleven_axis_trust_arbitration(paths=paths)
    if not base:
        return 11
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust_v16)
    dominant = sorted_paths[0]
    inner_field_names = (
        "substrate_trust", "hidden_trust",
        "attention_trust", "retrieval_trust",
        "replay_trust", "attention_pattern_trust",
        "replay_dominance_trust", "hidden_wins_trust",
        "replay_dominance_primary_trust")
    for k in range(1, 12):
        if k <= 9:
            kwargs = {f: 0.0 for f in inner_field_names[:k]}
            inner_attack = dataclasses.replace(
                dominant.inner_v15.inner_v14, **kwargs)
            v15_attack = dataclasses.replace(
                dominant.inner_v15, inner_v14=inner_attack)
            attacked = dataclasses.replace(
                dominant, inner_v15=v15_attack)
        elif k == 10:
            inner_attack = dataclasses.replace(
                dominant.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v15, inner_v14=inner_attack,
                team_coordination_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v15=v15_attack)
        else:  # k == 11
            inner_attack = dataclasses.replace(
                dominant.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v15, inner_v14=inner_attack,
                team_coordination_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = eleven_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 11


@dataclasses.dataclass(frozen=True)
class MultiHopV16Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    eleven_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "eleven_axis_used": bool(self.eleven_axis_used),
            "compromise_threshold": int(self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v16_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len26_fidelity(
        *, backends: Sequence[str] = (
            W66_DEFAULT_MH_V16_BACKENDS),
        chain_length: int = W66_DEFAULT_MH_V16_CHAIN_LEN,
        seed: int = 0,
) -> dict[str, Any]:
    """Synthetic chain-length-26 fidelity probe."""
    n_b = int(len(backends))
    strong_inner = DecBackendChainPathV14(
        chain=tuple(backends[:int(chain_length)]),
        payload=(0.6, 0.4, 0.2),
        confidence=0.9, substrate_trust=0.9,
        hidden_trust=0.9, attention_trust=0.9,
        retrieval_trust=0.9, replay_trust=0.9,
        attention_pattern_trust=0.9,
        replay_dominance_trust=0.9,
        hidden_wins_trust=0.9,
        replay_dominance_primary_trust=0.9)
    strong = DecBackendChainPathV16(
        inner_v15=DecBackendChainPathV15(
            inner_v14=strong_inner,
            team_coordination_trust=0.9),
        team_substrate_coordination_trust=0.9)
    paths = [strong]
    for i in range(3):
        weak_inner = DecBackendChainPathV14(
            chain=tuple(backends[i:int(chain_length) + i]),
            payload=(0.5, 0.5, 0.5),
            confidence=0.5, substrate_trust=0.5,
            hidden_trust=0.5, attention_trust=0.5,
            retrieval_trust=0.5, replay_trust=0.5,
            attention_pattern_trust=0.5,
            replay_dominance_trust=0.5,
            hidden_wins_trust=0.5,
            replay_dominance_primary_trust=0.5)
        paths.append(DecBackendChainPathV16(
            inner_v15=DecBackendChainPathV15(
                inner_v14=weak_inner,
                team_coordination_trust=0.5),
            team_substrate_coordination_trust=0.5))
    out, audit = eleven_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v16(paths=paths)
    return {
        "schema": W66_MULTI_HOP_V16_SCHEMA_VERSION,
        "n_backends": int(n_b),
        "chain_length": int(chain_length),
        "n_edges": int(n_b * (n_b - 1)),
        "compromise_threshold": int(threshold),
        "kind": str(audit["kind"]),
        "rationale": str(audit["rationale"]),
        "n_paths": int(audit["n_paths"]),
    }


def emit_multi_hop_v16_witness(
        *, backends: Sequence[str] = (
            W66_DEFAULT_MH_V16_BACKENDS),
        chain_length: int = W66_DEFAULT_MH_V16_CHAIN_LEN,
        seed: int = 0,
) -> MultiHopV16Witness:
    res = evaluate_dec_chain_len26_fidelity(
        backends=backends, chain_length=int(chain_length),
        seed=int(seed))
    return MultiHopV16Witness(
        schema=W66_MULTI_HOP_V16_SCHEMA_VERSION,
        n_backends=int(res["n_backends"]),
        chain_length=int(res["chain_length"]),
        n_edges=int(res["n_edges"]),
        eleven_axis_used=bool(
            "eleven_axis" in str(res["kind"])),
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W66_MULTI_HOP_V16_SCHEMA_VERSION",
    "W66_DEFAULT_MH_V16_BACKENDS",
    "W66_DEFAULT_MH_V16_CHAIN_LEN",
    "W66_DEFAULT_MH_V16_AGREEMENT_FLOOR",
    "DecBackendChainPathV16",
    "eleven_axis_trust_arbitration",
    "estimate_compromise_threshold_v16",
    "MultiHopV16Witness",
    "evaluate_dec_chain_len26_fidelity",
    "emit_multi_hop_v16_witness",
]
