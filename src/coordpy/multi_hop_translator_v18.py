"""W68 M9 — Multi-Hop Translator V18.

Strictly extends W67's ``coordpy.multi_hop_translator_v17``. V17 had
40 backends, 1560 directed edges, chain-length 30, 12-axis composite.
V18 adds:

* **44 backends** — V17's 40 + 4 new (OO, PP, QQ, RR).
* **44 × 43 = 1892 directed edges**.
* **Chain length 34**.
* **13-axis composite trust** — adds
  ``partial_contradiction_reconciliation_trust``.
* Compromise threshold in [1, 13].

Honest scope (W68)
------------------

* ``W68-L-MULTI-HOP-V18-SYNTHETIC-BACKENDS-CAP`` — backends are
  named, not executed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .multi_hop_translator_v17 import (
    DecBackendChainPathV17,
    W67_DEFAULT_MH_V17_AGREEMENT_FLOOR,
    W67_DEFAULT_MH_V17_BACKENDS,
)
from .multi_hop_translator_v16 import DecBackendChainPathV16
from .multi_hop_translator_v15 import DecBackendChainPathV15
from .multi_hop_translator_v14 import DecBackendChainPathV14
from .tiny_substrate_v3 import _sha256_hex


W68_MULTI_HOP_V18_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v18.v1")

W68_DEFAULT_MH_V18_BACKENDS: tuple[str, ...] = (
    *W67_DEFAULT_MH_V17_BACKENDS, "OO", "PP", "QQ", "RR")
W68_DEFAULT_MH_V18_CHAIN_LEN: int = 34
W68_DEFAULT_MH_V18_AGREEMENT_FLOOR: float = (
    W67_DEFAULT_MH_V17_AGREEMENT_FLOOR)


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV18:
    inner_v17: DecBackendChainPathV17
    partial_contradiction_reconciliation_trust: float = 1.0

    @property
    def chain(self) -> tuple[str, ...]:
        return self.inner_v17.chain

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v17.payload

    @property
    def confidence(self) -> float:
        return float(self.inner_v17.confidence)

    @property
    def composite_trust_v18(self) -> float:
        return float(
            self.inner_v17.composite_trust_v17
            * self.partial_contradiction_reconciliation_trust)


def thirteen_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV18],
        agreement_floor: float = (
            W68_DEFAULT_MH_V18_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    surviving = [p for p in paths
                 if p.composite_trust_v18 >= float(trust_floor)]
    if not surviving:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "trust_floor"}
    payloads = [list(p.payload) for p in surviving]
    weights = [
        float(p.composite_trust_v18 * p.confidence)
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
        "kind": "thirteen_axis_weighted_mean",
        "rationale": "thirteen_axis_composite_trust",
        "agreement_floor": float(agreement_floor),
    }


def estimate_compromise_threshold_v18(
        *, paths: Sequence[DecBackendChainPathV18],
) -> int:
    """Returns the minimum number of axes an adversary must drive
    to zero on the dominant path to flip the arbitration. Bounded
    in [1, 13]."""
    if not paths:
        return 13
    base, _ = thirteen_axis_trust_arbitration(paths=paths)
    if not base:
        return 13
    sorted_paths = sorted(
        paths, key=lambda p: -p.composite_trust_v18)
    dominant = sorted_paths[0]
    # Walk axes 1..12 mirroring V17 plus axis 13 (partial-contradiction).
    inner_field_names = (
        "substrate_trust", "hidden_trust",
        "attention_trust", "retrieval_trust",
        "replay_trust", "attention_pattern_trust",
        "replay_dominance_trust", "hidden_wins_trust",
        "replay_dominance_primary_trust")
    for k in range(1, 14):
        if k <= 9:
            kwargs = {f: 0.0 for f in inner_field_names[:k]}
            inner_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15.inner_v14,
                **kwargs)
            v15_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15,
                inner_v14=inner_attack)
            v16_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16,
                inner_v15=v15_attack)
            v17_attack = dataclasses.replace(
                dominant.inner_v17, inner_v16=v16_attack)
            attacked = dataclasses.replace(
                dominant, inner_v17=v17_attack)
        elif k == 10:
            inner_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16, inner_v15=v15_attack)
            v17_attack = dataclasses.replace(
                dominant.inner_v17, inner_v16=v16_attack)
            attacked = dataclasses.replace(
                dominant, inner_v17=v17_attack)
        elif k == 11:
            inner_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16,
                inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
            v17_attack = dataclasses.replace(
                dominant.inner_v17, inner_v16=v16_attack)
            attacked = dataclasses.replace(
                dominant, inner_v17=v17_attack)
        elif k == 12:
            inner_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16,
                inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
            v17_attack = dataclasses.replace(
                dominant.inner_v17, inner_v16=v16_attack,
                branch_merge_reconciliation_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v17=v17_attack)
        else:   # k == 13
            inner_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15.inner_v14,
                **{f: 0.0 for f in inner_field_names})
            v15_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16.inner_v15,
                inner_v14=inner_attack,
                team_coordination_trust=0.0)
            v16_attack = dataclasses.replace(
                dominant.inner_v17.inner_v16,
                inner_v15=v15_attack,
                team_substrate_coordination_trust=0.0)
            v17_attack = dataclasses.replace(
                dominant.inner_v17, inner_v16=v16_attack,
                branch_merge_reconciliation_trust=0.0)
            attacked = dataclasses.replace(
                dominant, inner_v17=v17_attack,
                partial_contradiction_reconciliation_trust=0.0)
        new_paths = [attacked] + list(sorted_paths[1:])
        new_out, _ = thirteen_axis_trust_arbitration(
            paths=new_paths)
        if not new_out or any(
                abs(new_out[i] - base[i]) > 1e-6
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 13


@dataclasses.dataclass(frozen=True)
class MultiHopV18Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    thirteen_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "thirteen_axis_used": bool(self.thirteen_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v18_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len34_fidelity(
        *, backends: Sequence[str] = (
            W68_DEFAULT_MH_V18_BACKENDS),
        chain_length: int = W68_DEFAULT_MH_V18_CHAIN_LEN,
        seed: int = 0,
) -> dict[str, Any]:
    """Synthetic chain-length-34 fidelity probe."""
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
    strong = DecBackendChainPathV18(
        inner_v17=DecBackendChainPathV17(
            inner_v16=DecBackendChainPathV16(
                inner_v15=DecBackendChainPathV15(
                    inner_v14=strong_inner,
                    team_coordination_trust=0.9),
                team_substrate_coordination_trust=0.9),
            branch_merge_reconciliation_trust=0.9),
        partial_contradiction_reconciliation_trust=0.9)
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
        paths.append(DecBackendChainPathV18(
            inner_v17=DecBackendChainPathV17(
                inner_v16=DecBackendChainPathV16(
                    inner_v15=DecBackendChainPathV15(
                        inner_v14=weak_inner,
                        team_coordination_trust=0.5),
                    team_substrate_coordination_trust=0.5),
                branch_merge_reconciliation_trust=0.5),
            partial_contradiction_reconciliation_trust=0.5))
    out, audit = thirteen_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v18(paths=paths)
    return {
        "schema": W68_MULTI_HOP_V18_SCHEMA_VERSION,
        "n_backends": int(n_b),
        "chain_length": int(chain_length),
        "n_edges": int(n_b * (n_b - 1)),
        "compromise_threshold": int(threshold),
        "kind": str(audit["kind"]),
        "rationale": str(audit["rationale"]),
        "n_paths": int(audit["n_paths"]),
    }


def emit_multi_hop_v18_witness(
        *, backends: Sequence[str] = (
            W68_DEFAULT_MH_V18_BACKENDS),
        chain_length: int = W68_DEFAULT_MH_V18_CHAIN_LEN,
        seed: int = 0,
) -> MultiHopV18Witness:
    res = evaluate_dec_chain_len34_fidelity(
        backends=backends, chain_length=int(chain_length),
        seed=int(seed))
    return MultiHopV18Witness(
        schema=W68_MULTI_HOP_V18_SCHEMA_VERSION,
        n_backends=int(res["n_backends"]),
        chain_length=int(res["chain_length"]),
        n_edges=int(res["n_edges"]),
        thirteen_axis_used=bool(
            "thirteen_axis" in str(res["kind"])),
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W68_MULTI_HOP_V18_SCHEMA_VERSION",
    "W68_DEFAULT_MH_V18_BACKENDS",
    "W68_DEFAULT_MH_V18_CHAIN_LEN",
    "W68_DEFAULT_MH_V18_AGREEMENT_FLOOR",
    "DecBackendChainPathV18",
    "thirteen_axis_trust_arbitration",
    "estimate_compromise_threshold_v18",
    "MultiHopV18Witness",
    "evaluate_dec_chain_len34_fidelity",
    "emit_multi_hop_v18_witness",
]
