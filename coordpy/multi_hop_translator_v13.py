"""W63 M12 — Multi-Hop Translator V13.

Strictly extends W62's ``coordpy.multi_hop_translator_v12``. V12
ran a 7-axis composite trust at chain-length 17 over 20 backends
and 380 directed edges. V13 adds:

* **24 backends** (vs V12's 20) — A..X.
* **chain-length 19** (vs V12's 17).
* **8-axis composite** — adds ``hidden_wins_trust`` as the eighth
  trust axis.
* **552 directed edges** (vs V12's 380, = 24 × 23).
* **Compromise threshold ∈ [1, 8]**.

Honest scope
------------

* Backends are NAMED, not EXECUTED.
  ``W63-L-MULTI-HOP-V13-SYNTHETIC-BACKENDS-CAP`` documents.
* The 8-axis composite is the product of eight scalars in [0,1].
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .multi_hop_translator_v12 import DecBackendChainPathV12
from .tiny_substrate_v3 import _sha256_hex


W63_MULTI_HOP_V13_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v13.v1")

W63_DEFAULT_MH_V13_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X")
W63_DEFAULT_MH_V13_CHAIN_LEN: int = 19
W63_DEFAULT_MH_V13_AGREEMENT_FLOOR: float = 0.3


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV13:
    chain: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    substrate_trust: float = 1.0
    hidden_trust: float = 1.0
    attention_trust: float = 1.0
    retrieval_trust: float = 1.0
    replay_trust: float = 1.0
    attention_pattern_trust: float = 1.0
    replay_dominance_trust: float = 1.0
    hidden_wins_trust: float = 1.0

    @property
    def composite_trust(self) -> float:
        return float(
            self.substrate_trust
            * self.hidden_trust
            * self.attention_trust
            * self.retrieval_trust
            * self.replay_trust
            * self.attention_pattern_trust
            * self.replay_dominance_trust
            * self.hidden_wins_trust)


def eight_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV13],
        agreement_floor: float = (
            W63_DEFAULT_MH_V13_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    surviving = [p for p in paths
                 if p.composite_trust >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    payloads = [list(p.payload) for p in surviving]
    weights = [
        float(p.composite_trust * p.confidence)
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
        "kind": "eight_axis_weighted_mean",
        "rationale": "eight_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v13(
        *, paths: Sequence[DecBackendChainPathV13],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration outcome.
    Bounded in [1, 8]."""
    if not paths:
        return 8
    base, _ = eight_axis_trust_arbitration(paths=paths)
    if not base:
        return 8
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust)
    dominant = sorted_paths[0]
    axes = [
        "substrate_trust", "hidden_trust", "attention_trust",
        "retrieval_trust", "replay_trust",
        "attention_pattern_trust", "replay_dominance_trust",
        "hidden_wins_trust"]
    for k in range(1, 9):
        d = dict(dataclasses.asdict(dominant))
        for ax in axes[:k]:
            d[ax] = 0.0
        attacked = DecBackendChainPathV13(
            chain=dominant.chain,
            payload=dominant.payload,
            confidence=dominant.confidence,
            substrate_trust=d["substrate_trust"],
            hidden_trust=d["hidden_trust"],
            attention_trust=d["attention_trust"],
            retrieval_trust=d["retrieval_trust"],
            replay_trust=d["replay_trust"],
            attention_pattern_trust=d[
                "attention_pattern_trust"],
            replay_dominance_trust=d[
                "replay_dominance_trust"],
            hidden_wins_trust=d["hidden_wins_trust"])
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = eight_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 8


@dataclasses.dataclass(frozen=True)
class MultiHopV13Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    eight_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "eight_axis_used": bool(self.eight_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v13_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len19_fidelity(
        *, backends: Sequence[str] = (
            W63_DEFAULT_MH_V13_BACKENDS),
        chain_length: int = W63_DEFAULT_MH_V13_CHAIN_LEN,
        seed: int = 63130,
) -> dict[str, Any]:
    import numpy as _np
    rng = _np.random.default_rng(int(seed))
    n_b = int(len(backends))
    chains = []
    for _ in range(8):
        idx = rng.integers(0, n_b, size=int(chain_length))
        chains.append(tuple(backends[i] for i in idx))
    paths = []
    for ch in chains:
        paths.append(DecBackendChainPathV13(
            chain=ch,
            payload=tuple(rng.standard_normal(4).tolist()),
            confidence=float(rng.uniform(0.5, 1.0)),
            substrate_trust=float(rng.uniform(0.5, 1.0)),
            hidden_trust=float(rng.uniform(0.5, 1.0)),
            attention_trust=float(rng.uniform(0.5, 1.0)),
            retrieval_trust=float(rng.uniform(0.5, 1.0)),
            replay_trust=float(rng.uniform(0.5, 1.0)),
            attention_pattern_trust=float(
                rng.uniform(0.5, 1.0)),
            replay_dominance_trust=float(
                rng.uniform(0.5, 1.0)),
            hidden_wins_trust=float(rng.uniform(0.5, 1.0)),
        ))
    out, info = eight_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v13(paths=paths)
    return {
        "schema": W63_MULTI_HOP_V13_SCHEMA_VERSION,
        "n_paths": int(len(paths)),
        "kind": info.get("kind", ""),
        "payload": list(out),
        "compromise_threshold": int(threshold),
    }


def emit_multi_hop_v13_witness(
        *, backends: Sequence[str] = (
            W63_DEFAULT_MH_V13_BACKENDS),
        chain_length: int = W63_DEFAULT_MH_V13_CHAIN_LEN,
        seed: int = 63130,
) -> MultiHopV13Witness:
    n_b = int(len(backends))
    n_edges = int(n_b * (n_b - 1))
    res = evaluate_dec_chain_len19_fidelity(
        backends=backends, chain_length=chain_length, seed=seed)
    return MultiHopV13Witness(
        schema=W63_MULTI_HOP_V13_SCHEMA_VERSION,
        n_backends=int(n_b),
        chain_length=int(chain_length),
        n_edges=int(n_edges),
        eight_axis_used=True,
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W63_MULTI_HOP_V13_SCHEMA_VERSION",
    "W63_DEFAULT_MH_V13_BACKENDS",
    "W63_DEFAULT_MH_V13_CHAIN_LEN",
    "W63_DEFAULT_MH_V13_AGREEMENT_FLOOR",
    "DecBackendChainPathV13",
    "eight_axis_trust_arbitration",
    "estimate_compromise_threshold_v13",
    "MultiHopV13Witness",
    "evaluate_dec_chain_len19_fidelity",
    "emit_multi_hop_v13_witness",
]
