"""W67 M11 — Multi-Hop Translator V17.

Strictly extends W66's ``coordpy.multi_hop_translator_v16``. V16 had
36 backends, 1260 directed edges, chain-length 26, 11-axis composite.
V17 adds:

* **40 backends** — V16's 36 + 4 new (KK, LL, MM, NN).
* **40 × 39 = 1560 directed edges**.
* **Chain length 30**.
* **12-axis composite trust** — adds ``branch_merge_reconciliation_trust``.
* Compromise threshold in [1, 12].

Honest scope (W67)
------------------

* ``W67-L-MULTI-HOP-V17-SYNTHETIC-BACKENDS-CAP`` — backends are
  named, not executed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .multi_hop_translator_v16 import (
    DecBackendChainPathV16,
    W66_DEFAULT_MH_V16_AGREEMENT_FLOOR,
    W66_DEFAULT_MH_V16_BACKENDS,
)
from .multi_hop_translator_v15 import DecBackendChainPathV15
from .multi_hop_translator_v14 import DecBackendChainPathV14
from .tiny_substrate_v3 import _sha256_hex


W67_MULTI_HOP_V17_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v17.v1")

W67_DEFAULT_MH_V17_BACKENDS: tuple[str, ...] = (
    *W66_DEFAULT_MH_V16_BACKENDS, "KK", "LL", "MM", "NN")
W67_DEFAULT_MH_V17_CHAIN_LEN: int = 30
W67_DEFAULT_MH_V17_AGREEMENT_FLOOR: float = (
    W66_DEFAULT_MH_V16_AGREEMENT_FLOOR)


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV17:
    inner_v16: DecBackendChainPathV16
    branch_merge_reconciliation_trust: float = 1.0

    @property
    def chain(self) -> tuple[str, ...]:
        return self.inner_v16.chain

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v16.payload

    @property
    def confidence(self) -> float:
        return float(self.inner_v16.confidence)

    @property
    def composite_trust_v17(self) -> float:
        return float(
            self.inner_v16.composite_trust_v16
            * self.branch_merge_reconciliation_trust)


def twelve_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV17],
        agreement_floor: float = (
            W67_DEFAULT_MH_V17_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    surviving = [p for p in paths
                 if p.composite_trust_v17 >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    payloads = [list(p.payload) for p in surviving]
    weights = [
        float(p.composite_trust_v17 * p.confidence)
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
        "kind": "twelve_axis_weighted_mean",
        "rationale": "twelve_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v17(
        *, paths: Sequence[DecBackendChainPathV17],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration. Bounded
    in [1, 12]."""
    if not paths:
        return 12
    base, _ = twelve_axis_trust_arbitration(paths=paths)
    if not base:
        return 12
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust_v17)
    dominant = sorted_paths[0]
    inner_field_names = (
        "substrate_trust", "hidden_trust",
        "attention_trust", "retrieval_trust",
        "replay_trust", "attention_pattern_trust",
        "replay_dominance_trust", "hidden_wins_trust",
        "replay_dominance_primary_trust")
    for k in range(1, 13):
        if k <= 9:
            kwargs = {f: 0.0 for f in inner_field_names[:k]}
            inner_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15.inner_v14, **kwargs)
            v15_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15,
                inner_v14=inner_attack)
            v16_attack = dataclasses.replace(
                dominant.inner_v16, inner_v15=v15_attack)
            attacked = dataclasses.replace(
                dominant, inner_v16=v16_attack)
        elif k == 10:
            inner_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v16, inner_v15=v15_attack)
            attacked = dataclasses.replace(
                dominant, inner_v16=v16_attack)
        elif k == 11:
            inner_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v16, inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v16=v16_attack)
        else:   # k == 12
            inner_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v16, inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v16=v16_attack,
                branch_merge_reconciliation_trust=0.0)
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = twelve_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 12


@dataclasses.dataclass(frozen=True)
class MultiHopV17Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    twelve_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "twelve_axis_used": bool(self.twelve_axis_used),
            "compromise_threshold": int(self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v17_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len30_fidelity(
        *, backends: Sequence[str] = (
            W67_DEFAULT_MH_V17_BACKENDS),
        chain_length: int = W67_DEFAULT_MH_V17_CHAIN_LEN,
        seed: int = 0,
) -> dict[str, Any]:
    """Synthetic chain-length-30 fidelity probe."""
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
    strong = DecBackendChainPathV17(
        inner_v16=DecBackendChainPathV16(
            inner_v15=DecBackendChainPathV15(
                inner_v14=strong_inner,
                team_coordination_trust=0.9),
            team_substrate_coordination_trust=0.9),
        branch_merge_reconciliation_trust=0.9)
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
        paths.append(DecBackendChainPathV17(
            inner_v16=DecBackendChainPathV16(
                inner_v15=DecBackendChainPathV15(
                    inner_v14=weak_inner,
                    team_coordination_trust=0.5),
                team_substrate_coordination_trust=0.5),
            branch_merge_reconciliation_trust=0.5))
    out, audit = twelve_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v17(paths=paths)
    return {
        "schema": W67_MULTI_HOP_V17_SCHEMA_VERSION,
        "n_backends": int(n_b),
        "chain_length": int(chain_length),
        "n_edges": int(n_b * (n_b - 1)),
        "compromise_threshold": int(threshold),
        "kind": str(audit["kind"]),
        "rationale": str(audit["rationale"]),
        "n_paths": int(audit["n_paths"]),
    }


def emit_multi_hop_v17_witness(
        *, backends: Sequence[str] = (
            W67_DEFAULT_MH_V17_BACKENDS),
        chain_length: int = W67_DEFAULT_MH_V17_CHAIN_LEN,
        seed: int = 0,
) -> MultiHopV17Witness:
    res = evaluate_dec_chain_len30_fidelity(
        backends=backends, chain_length=int(chain_length),
        seed=int(seed))
    return MultiHopV17Witness(
        schema=W67_MULTI_HOP_V17_SCHEMA_VERSION,
        n_backends=int(res["n_backends"]),
        chain_length=int(res["chain_length"]),
        n_edges=int(res["n_edges"]),
        twelve_axis_used=bool(
            "twelve_axis" in str(res["kind"])),
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W67_MULTI_HOP_V17_SCHEMA_VERSION",
    "W67_DEFAULT_MH_V17_BACKENDS",
    "W67_DEFAULT_MH_V17_CHAIN_LEN",
    "W67_DEFAULT_MH_V17_AGREEMENT_FLOOR",
    "DecBackendChainPathV17",
    "twelve_axis_trust_arbitration",
    "estimate_compromise_threshold_v17",
    "MultiHopV17Witness",
    "evaluate_dec_chain_len30_fidelity",
    "emit_multi_hop_v17_witness",
]
