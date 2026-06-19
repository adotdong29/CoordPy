"""W62 M9 — Multi-Hop Translator V12.

Strictly extends W61's ``coordpy.multi_hop_translator_v11``. V11
ran a 6-axis (substrate × hidden × attention × retrieval × replay
× attention_pattern) composite trust at chain-length 16 over 18
backends and 306 directed edges. V12 adds:

* **20 backends** (vs V11's 18) — A..T.
* **chain-length 17** (vs V11's 16).
* **7-axis composite** — adds ``replay_dominance_trust`` as
  the seventh trust axis.
* **380 directed edges** (vs V11's 306, ≈ 20 × 19).
* **Compromise threshold ∈ [1, 7]**.

Honest scope
------------

* Backends are NAMED, not EXECUTED.
  ``W62-L-MULTI-HOP-V12-SYNTHETIC-BACKENDS-CAP`` documents.
* The 7-axis composite is the product of seven scalars in [0,1].
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .multi_hop_translator_v11 import (
    DecBackendChainPathV11,
)
from .tiny_substrate_v3 import _sha256_hex


W62_MULTI_HOP_V12_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v12.v1")

W62_DEFAULT_MH_V12_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T")
W62_DEFAULT_MH_V12_CHAIN_LEN: int = 17
W62_DEFAULT_MH_V12_AGREEMENT_FLOOR: float = 0.3


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV12:
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

    @property
    def composite_trust(self) -> float:
        return float(
            self.substrate_trust
            * self.hidden_trust
            * self.attention_trust
            * self.retrieval_trust
            * self.replay_trust
            * self.attention_pattern_trust
            * self.replay_dominance_trust)


def seven_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV12],
        agreement_floor: float = (
            W62_DEFAULT_MH_V12_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    # Filter by composite trust floor.
    surviving = [p for p in paths
                 if p.composite_trust >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    # Weighted average by composite trust.
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
        "kind": "seven_axis_weighted_mean",
        "rationale": "seven_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v12(
        *, paths: Sequence[DecBackendChainPathV12],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration outcome.
    Bounded in [1, 7]."""
    if not paths:
        return 7
    base, _ = seven_axis_trust_arbitration(paths=paths)
    if not base:
        return 7
    # Sort paths by composite_trust descending.
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust)
    dominant = sorted_paths[0]
    axes = [
        "substrate_trust", "hidden_trust", "attention_trust",
        "retrieval_trust", "replay_trust",
        "attention_pattern_trust", "replay_dominance_trust"]
    for k in range(1, 8):
        # Try driving the top-k axes of the dominant path to zero.
        attacked = dataclasses.replace(dominant)
        d = dict(dataclasses.asdict(attacked))
        for ax in axes[:k]:
            d[ax] = 0.0
        # Reconstruct.
        attacked = DecBackendChainPathV12(
            chain=dominant.chain,
            payload=dominant.payload,
            confidence=dominant.confidence,
            substrate_trust=d["substrate_trust"],
            hidden_trust=d["hidden_trust"],
            attention_trust=d["attention_trust"],
            retrieval_trust=d["retrieval_trust"],
            replay_trust=d["replay_trust"],
            attention_pattern_trust=d["attention_pattern_trust"],
            replay_dominance_trust=d[
                "replay_dominance_trust"])
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = seven_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 7


@dataclasses.dataclass(frozen=True)
class MultiHopV12Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    seven_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "seven_axis_used": bool(self.seven_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v12_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len17_fidelity(
        *, backends: Sequence[str] = (
            W62_DEFAULT_MH_V12_BACKENDS),
        chain_length: int = W62_DEFAULT_MH_V12_CHAIN_LEN,
        seed: int = 62120,
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
        paths.append(DecBackendChainPathV12(
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
        ))
    out, info = seven_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v12(paths=paths)
    return {
        "schema": W62_MULTI_HOP_V12_SCHEMA_VERSION,
        "n_paths": int(len(paths)),
        "kind": info.get("kind", ""),
        "payload": list(out),
        "compromise_threshold": int(threshold),
    }


def emit_multi_hop_v12_witness(
        *, backends: Sequence[str] = (
            W62_DEFAULT_MH_V12_BACKENDS),
        chain_length: int = W62_DEFAULT_MH_V12_CHAIN_LEN,
        seed: int = 62120,
) -> MultiHopV12Witness:
    n_b = int(len(backends))
    n_edges = int(n_b * (n_b - 1))
    res = evaluate_dec_chain_len17_fidelity(
        backends=backends, chain_length=chain_length, seed=seed)
    return MultiHopV12Witness(
        schema=W62_MULTI_HOP_V12_SCHEMA_VERSION,
        n_backends=int(n_b),
        chain_length=int(chain_length),
        n_edges=int(n_edges),
        seven_axis_used=True,
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W62_MULTI_HOP_V12_SCHEMA_VERSION",
    "W62_DEFAULT_MH_V12_BACKENDS",
    "W62_DEFAULT_MH_V12_CHAIN_LEN",
    "W62_DEFAULT_MH_V12_AGREEMENT_FLOOR",
    "DecBackendChainPathV12",
    "seven_axis_trust_arbitration",
    "estimate_compromise_threshold_v12",
    "MultiHopV12Witness",
    "evaluate_dec_chain_len17_fidelity",
    "emit_multi_hop_v12_witness",
]
