"""W69 M9 — Multi-Hop Translator V19.

Strictly extends W68's ``coordpy.multi_hop_translator_v18``. V18
had 44 backends, 1892 edges, chain-length 34, 13-axis composite.
V19 adds:

* **48 backends** — V18's 44 + 4 new (SS, TT, UU, VV).
* **48 × 47 = 2256 directed edges**.
* **Chain length 38**.
* **14-axis composite trust** — adds
  ``multi_branch_rejoin_reconciliation_trust``.
* Compromise threshold in [1, 14].

Honest scope (W69)
------------------

* ``W69-L-MULTI-HOP-V19-SYNTHETIC-BACKENDS-CAP`` — backends are
  named, not executed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .multi_hop_translator_v18 import (
    DecBackendChainPathV18,
    W68_DEFAULT_MH_V18_AGREEMENT_FLOOR,
    W68_DEFAULT_MH_V18_BACKENDS,
)
from .multi_hop_translator_v17 import DecBackendChainPathV17
from .multi_hop_translator_v16 import DecBackendChainPathV16
from .multi_hop_translator_v15 import DecBackendChainPathV15
from .multi_hop_translator_v14 import DecBackendChainPathV14
from .tiny_substrate_v3 import _sha256_hex


W69_MULTI_HOP_V19_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v19.v1")

W69_DEFAULT_MH_V19_BACKENDS: tuple[str, ...] = (
    *W68_DEFAULT_MH_V18_BACKENDS, "SS", "TT", "UU", "VV")
W69_DEFAULT_MH_V19_CHAIN_LEN: int = 38
W69_DEFAULT_MH_V19_AGREEMENT_FLOOR: float = (
    W68_DEFAULT_MH_V18_AGREEMENT_FLOOR)


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV19:
    inner_v18: DecBackendChainPathV18
    multi_branch_rejoin_reconciliation_trust: float = 1.0

    @property
    def chain(self) -> tuple[str, ...]:
        return self.inner_v18.chain

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v18.payload

    @property
    def confidence(self) -> float:
        return float(self.inner_v18.confidence)

    @property
    def composite_trust_v19(self) -> float:
        return float(
            self.inner_v18.composite_trust_v18
            * self.multi_branch_rejoin_reconciliation_trust)


def fourteen_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV19],
        agreement_floor: float = (
            W69_DEFAULT_MH_V19_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    eligible = [
        p for p in paths
        if p.composite_trust_v19 >= float(trust_floor)]
    if not eligible:
        return [], {
            "n_paths": int(len(paths)),
            "kind": "abstain",
            "rationale": "all_under_trust_floor"}
    weights = [
        max(0.0, float(p.composite_trust_v19))
        for p in eligible]
    total = float(sum(weights))
    if total <= 0.0:
        return [], {
            "n_paths": int(len(eligible)),
            "kind": "abstain",
            "rationale": "zero_weight_sum"}
    pay_len = len(eligible[0].payload)
    out = []
    for i in range(pay_len):
        acc = 0.0
        for w, p in zip(weights, eligible):
            v = (
                p.payload[i] if i < len(p.payload)
                else 0.0)
            acc += float(w) * float(v)
        out.append(float(acc / total))
    agreement = float(
        sum(abs(out[i] - eligible[0].payload[i])
            for i in range(pay_len))
        / max(1.0, pay_len))
    kind = (
        "fourteen_axis_arbitration_agree"
        if agreement <= float(agreement_floor)
        else "fourteen_axis_arbitration_compromise")
    return out, {
        "n_paths": int(len(eligible)),
        "kind": kind,
        "rationale": f"agreement_l1={agreement:.6f}",
    }


def estimate_compromise_threshold_v19(
        *, paths: Sequence[DecBackendChainPathV19],
) -> int:
    """Returns the smallest k in [1, 14] s.t. dropping the worst-k
    paths flips arbitration kind to ``agree``."""
    if not paths:
        return 14
    sorted_paths = sorted(
        paths, key=lambda p: -float(p.composite_trust_v19))
    base, audit_base = fourteen_axis_trust_arbitration(
        paths=sorted_paths)
    for k in range(1, 14):
        if len(sorted_paths) - k < 1:
            break
        new_out, audit = fourteen_axis_trust_arbitration(
            paths=sorted_paths[:len(sorted_paths) - k])
        if (str(audit.get("kind", ""))
                .endswith("agree")
                and len(new_out) > 0):
            return int(k)
        if all(abs(float(new_out[i]) - float(base[i])) < 1e-9
                for i in range(min(len(new_out), len(base)))):
            return int(k)
    return 14


@dataclasses.dataclass(frozen=True)
class MultiHopV19Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_edges: int
    fourteen_axis_used: bool
    compromise_threshold: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "fourteen_axis_used": bool(self.fourteen_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v19_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len38_fidelity(
        *, backends: Sequence[str] = (
            W69_DEFAULT_MH_V19_BACKENDS),
        chain_length: int = W69_DEFAULT_MH_V19_CHAIN_LEN,
        seed: int = 0,
) -> dict[str, Any]:
    """Synthetic chain-length-38 fidelity probe."""
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
    strong = DecBackendChainPathV19(
        inner_v18=DecBackendChainPathV18(
            inner_v17=DecBackendChainPathV17(
                inner_v16=DecBackendChainPathV16(
                    inner_v15=DecBackendChainPathV15(
                        inner_v14=strong_inner,
                        team_coordination_trust=0.9),
                    team_substrate_coordination_trust=0.9),
                branch_merge_reconciliation_trust=0.9),
            partial_contradiction_reconciliation_trust=0.9),
        multi_branch_rejoin_reconciliation_trust=0.9)
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
        paths.append(DecBackendChainPathV19(
            inner_v18=DecBackendChainPathV18(
                inner_v17=DecBackendChainPathV17(
                    inner_v16=DecBackendChainPathV16(
                        inner_v15=DecBackendChainPathV15(
                            inner_v14=weak_inner,
                            team_coordination_trust=0.5),
                        team_substrate_coordination_trust=0.5),
                    branch_merge_reconciliation_trust=0.5),
                partial_contradiction_reconciliation_trust=0.5),
            multi_branch_rejoin_reconciliation_trust=0.5))
    out, audit = fourteen_axis_trust_arbitration(paths=paths)
    threshold = estimate_compromise_threshold_v19(paths=paths)
    return {
        "schema": W69_MULTI_HOP_V19_SCHEMA_VERSION,
        "n_backends": int(n_b),
        "chain_length": int(chain_length),
        "n_edges": int(n_b * (n_b - 1)),
        "compromise_threshold": int(threshold),
        "kind": str(audit["kind"]),
        "rationale": str(audit["rationale"]),
        "n_paths": int(audit["n_paths"]),
    }


def emit_multi_hop_v19_witness(
        *, backends: Sequence[str] = (
            W69_DEFAULT_MH_V19_BACKENDS),
        chain_length: int = W69_DEFAULT_MH_V19_CHAIN_LEN,
        seed: int = 0,
) -> MultiHopV19Witness:
    res = evaluate_dec_chain_len38_fidelity(
        backends=backends, chain_length=int(chain_length),
        seed=int(seed))
    return MultiHopV19Witness(
        schema=W69_MULTI_HOP_V19_SCHEMA_VERSION,
        n_backends=int(res["n_backends"]),
        chain_length=int(res["chain_length"]),
        n_edges=int(res["n_edges"]),
        fourteen_axis_used=bool(
            "fourteen_axis" in str(res["kind"])),
        compromise_threshold=int(res["compromise_threshold"]),
        arbitration_kind=str(res["kind"]),
    )


__all__ = [
    "W69_MULTI_HOP_V19_SCHEMA_VERSION",
    "W69_DEFAULT_MH_V19_BACKENDS",
    "W69_DEFAULT_MH_V19_CHAIN_LEN",
    "W69_DEFAULT_MH_V19_AGREEMENT_FLOOR",
    "DecBackendChainPathV19",
    "fourteen_axis_trust_arbitration",
    "estimate_compromise_threshold_v19",
    "MultiHopV19Witness",
    "evaluate_dec_chain_len38_fidelity",
    "emit_multi_hop_v19_witness",
]
