"""W64 M11 — Multi-Hop Translator V14.

Strictly extends W63's ``coordpy.multi_hop_translator_v13``. V13
ran an 8-axis composite trust at chain-length 19 over 24 backends
and 552 directed edges. V14 adds:

* **27 backends** (vs V13's 24) — A..AA.
* **chain-length 21** (vs V13's 19).
* **9-axis composite** — adds ``replay_dominance_primary_trust``
  as the ninth trust axis.
* **702 directed edges** (vs V13's 552, = 27 × 26).
* **Compromise threshold ∈ [1, 9]**.

Honest scope (W64)
------------------

* Backends are NAMED, not EXECUTED.
  ``W64-L-MULTI-HOP-V14-SYNTHETIC-BACKENDS-CAP`` documents.
* The 9-axis composite is the product of nine scalars in [0,1].
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .tiny_substrate_v3 import _sha256_hex


W64_MULTI_HOP_V14_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v14.v1")

W64_DEFAULT_MH_V14_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "AA")
W64_DEFAULT_MH_V14_CHAIN_LEN: int = 21
W64_DEFAULT_MH_V14_AGREEMENT_FLOOR: float = 0.3


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV14:
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
    replay_dominance_primary_trust: float = 1.0

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
            * self.hidden_wins_trust
            * self.replay_dominance_primary_trust)


def nine_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV14],
        agreement_floor: float = (
            W64_DEFAULT_MH_V14_AGREEMENT_FLOOR),
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
        "kind": "nine_axis_weighted_mean",
        "rationale": "nine_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v14(
        *, paths: Sequence[DecBackendChainPathV14],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration outcome.
    Bounded in [1, 9]."""
    if not paths:
        return 9
    base, _ = nine_axis_trust_arbitration(paths=paths)
    if not base:
        return 9
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust)
    dominant = sorted_paths[0]
    axes = [
        "substrate_trust", "hidden_trust", "attention_trust",
        "retrieval_trust", "replay_trust",
        "attention_pattern_trust", "replay_dominance_trust",
        "hidden_wins_trust",
        "replay_dominance_primary_trust"]
    for k in range(1, 10):
        d = dict(dataclasses.asdict(dominant))
        for ax in axes[:k]:
            d[ax] = 0.0
        attacked = DecBackendChainPathV14(
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
            hidden_wins_trust=d["hidden_wins_trust"],
            replay_dominance_primary_trust=d[
                "replay_dominance_primary_trust"])
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = nine_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 9


@dataclasses.dataclass(frozen=True)
class MultiHopV14Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    nine_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "nine_axis_used": bool(self.nine_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v14_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len21_fidelity(
        *, backends: Sequence[str] = (
            W64_DEFAULT_MH_V14_BACKENDS),
        chain_length: int = W64_DEFAULT_MH_V14_CHAIN_LEN,
        seed: int = 64140,
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
        paths.append(DecBackendChainPathV14(
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
            replay_dominance_primary_trust=float(
                rng.uniform(0.5, 1.0)),
        ))
    out, info = nine_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v14(paths=paths)
    return {
        "schema": W64_MULTI_HOP_V14_SCHEMA_VERSION,
        "n_paths": int(len(paths)),
        "kind": info.get("kind", ""),
        "payload": list(out),
        "compromise_threshold": int(threshold),
    }


def emit_multi_hop_v14_witness(
        *, backends: Sequence[str] = (
            W64_DEFAULT_MH_V14_BACKENDS),
        chain_length: int = W64_DEFAULT_MH_V14_CHAIN_LEN,
        seed: int = 64140,
) -> MultiHopV14Witness:
    n_b = int(len(backends))
    n_edges = int(n_b * (n_b - 1))
    res = evaluate_dec_chain_len21_fidelity(
        backends=backends, chain_length=chain_length, seed=seed)
    return MultiHopV14Witness(
        schema=W64_MULTI_HOP_V14_SCHEMA_VERSION,
        n_backends=int(n_b),
        chain_length=int(chain_length),
        n_edges=int(n_edges),
        nine_axis_used=True,
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W64_MULTI_HOP_V14_SCHEMA_VERSION",
    "W64_DEFAULT_MH_V14_BACKENDS",
    "W64_DEFAULT_MH_V14_CHAIN_LEN",
    "W64_DEFAULT_MH_V14_AGREEMENT_FLOOR",
    "DecBackendChainPathV14",
    "nine_axis_trust_arbitration",
    "estimate_compromise_threshold_v14",
    "MultiHopV14Witness",
    "evaluate_dec_chain_len21_fidelity",
    "emit_multi_hop_v14_witness",
]
