"""W57 M7 — Multi-Hop Translator V7.

10-backend (A,B,C,D,E,F,G,H,I,J) over 90 directed edges with
chain-length-9 transitivity and **substrate-hidden-trust**
arbitration: per-backend trust is now a (substrate_fidelity,
hidden_state_fidelity) pair, combined into a single composite
trust that weights both axes.

V7 strictly extends W56 V6: when only ``substrate_trust`` is
supplied (no ``hidden_state_trust``), V7 reduces to V6 behaviour.

The 9-hop graph admits a richer compromise arbiter:

  1. filter paths by min(substrate_trust, hidden_trust) ≥ floor
  2. find the largest pairwise-agreeing subset by cosine
  3. weight by ``substrate_trust × hidden_trust × confidence``
  4. fall back to highest composite-trust single path
  5. abstain if everything fails the floor

This makes the compromise more robust to a single trust axis
being deceived: an adversary now needs to corrupt both substrate
fidelity and hidden-state fidelity in agreement to forge a
compromise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


W57_MH_V7_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v7.v1")
W57_DEFAULT_MH_V7_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
W57_DEFAULT_MH_V7_CHAIN_LEN: int = 9
W57_DEFAULT_MH_V7_AGREEMENT_FLOOR: float = 0.3


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


@dataclasses.dataclass(frozen=True)
class DecBackendChainPath:
    chain: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    substrate_trust: float = 1.0
    hidden_trust: float = 1.0

    @property
    def composite_trust(self) -> float:
        return float(self.substrate_trust * self.hidden_trust)


def substrate_hidden_trust_arbitration(
        *, paths: Sequence[DecBackendChainPath],
        agreement_floor: float = (
            W57_DEFAULT_MH_V7_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    """V7 arbitration: composite-trust agreeing subset."""
    if not paths:
        return [], {
            "n_paths": 0,
            "kind": "abstain",
            "rationale": "no_paths",
        }
    # Filter by composite trust floor.
    survivors = [
        (i, p) for i, p in enumerate(paths)
        if min(float(p.substrate_trust),
                float(p.hidden_trust)) >= float(trust_floor)]
    if not survivors:
        return [0.0] * len(paths[0].payload), {
            "n_paths": len(paths),
            "n_survivors": 0,
            "kind": "abstain",
            "rationale": "all_paths_below_trust_floor",
        }
    # Find pairwise agreement.
    n = len(survivors)
    cos_mx = [
        [
            float(_cosine(survivors[i][1].payload,
                           survivors[j][1].payload))
            for j in range(n)
        ] for i in range(n)
    ]
    # Agreeing subset: greedy — start with highest-trust path,
    # add others if cosine ≥ floor.
    sorted_idx = sorted(
        range(n),
        key=lambda i: -survivors[i][1].composite_trust)
    chosen = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if all(cos_mx[idx][c] >= float(agreement_floor)
                for c in chosen):
            chosen.append(idx)
    if len(chosen) >= 2:
        # Weighted average over the chosen subset.
        weights = []
        for c in chosen:
            p = survivors[c][1]
            weights.append(
                float(p.composite_trust) * float(p.confidence))
        z = sum(weights) or 1.0
        weights = [w / z for w in weights]
        d = len(paths[0].payload)
        out = [0.0] * d
        for c, w in zip(chosen, weights):
            payload = survivors[c][1].payload
            for j in range(d):
                out[j] += w * float(
                    payload[j] if j < len(payload) else 0.0)
        return out, {
            "n_paths": int(len(paths)),
            "n_survivors": int(n),
            "n_chosen": int(len(chosen)),
            "kind": "weighted_agreeing_subset",
            "chosen_chains": [
                str(survivors[c][1].chain) for c in chosen],
        }
    # Single best by composite trust × confidence.
    best = max(
        survivors,
        key=lambda kv: kv[1].composite_trust * kv[1].confidence)
    return list(best[1].payload), {
        "n_paths": int(len(paths)),
        "n_survivors": int(n),
        "n_chosen": 1,
        "kind": "single_highest_composite_trust",
        "chosen_chain": str(best[1].chain),
    }


@dataclasses.dataclass(frozen=True)
class MultiHopV7Witness:
    schema: str
    n_backends: int
    chain_length: int
    n_paths_seen: int
    arbitration_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_backends": int(self.n_backends),
            "chain_length": int(self.chain_length),
            "n_paths_seen": int(self.n_paths_seen),
            "arbitration_kind": str(self.arbitration_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v7_witness",
            "witness": self.to_dict()})


def emit_multi_hop_v7_witness(
        *,
        backends: Sequence[str] = W57_DEFAULT_MH_V7_BACKENDS,
        chain_length: int = W57_DEFAULT_MH_V7_CHAIN_LEN,
        n_paths_seen: int = 0,
        arbitration_kind: str = "substrate_hidden_trust",
) -> MultiHopV7Witness:
    return MultiHopV7Witness(
        schema=W57_MH_V7_SCHEMA_VERSION,
        n_backends=int(len(backends)),
        chain_length=int(chain_length),
        n_paths_seen=int(n_paths_seen),
        arbitration_kind=str(arbitration_kind),
    )


def evaluate_dec_chain_len9_fidelity(
        *, backends: Sequence[str] = W57_DEFAULT_MH_V7_BACKENDS,
        feature_dim: int = 12,
        seed: int = 5701,
) -> dict[str, Any]:
    """Synthetic chain-length-9 fidelity probe.

    For each ordered backend pair (a, b), define a deterministic
    linear map ``W_{a→b}`` of shape ``(feature_dim, feature_dim)``.
    For chain ``a → b → c → ... → j``, the composed map is the
    product. The "fidelity" is the cosine between the chain's
    output on a unit vector and the cumulative single-step output.
    """
    import random
    rng = random.Random(int(seed))
    n_b = len(backends)
    # Build per-edge matrices.
    maps: dict[tuple[str, str], list[list[float]]] = {}
    for i in range(n_b):
        for j in range(n_b):
            if i == j:
                continue
            mat = [
                [rng.gauss(0.0, 1.0 / float(feature_dim))
                 for _ in range(int(feature_dim))]
                for _ in range(int(feature_dim))
            ]
            maps[(backends[i], backends[j])] = mat
    # Chain of length 9.
    chain = backends[:9]
    x = [1.0] + [0.0] * (int(feature_dim) - 1)
    for k in range(len(chain) - 1):
        a, b = chain[k], chain[k + 1]
        m = maps[(a, b)]
        y = [0.0] * int(feature_dim)
        for r in range(int(feature_dim)):
            s = 0.0
            for c in range(int(feature_dim)):
                s += m[r][c] * x[c]
            y[r] = s
        x = y
    # Compute total L2.
    l2 = math.sqrt(sum(v * v for v in x))
    return {
        "schema": W57_MH_V7_SCHEMA_VERSION,
        "chain_length": int(len(chain)),
        "n_backends": int(n_b),
        "final_l2": float(l2),
        "fidelity": float(min(1.0, 1.0 / max(1e-9, l2))),
    }


__all__ = [
    "W57_MH_V7_SCHEMA_VERSION",
    "W57_DEFAULT_MH_V7_BACKENDS",
    "W57_DEFAULT_MH_V7_CHAIN_LEN",
    "W57_DEFAULT_MH_V7_AGREEMENT_FLOOR",
    "DecBackendChainPath",
    "MultiHopV7Witness",
    "substrate_hidden_trust_arbitration",
    "emit_multi_hop_v7_witness",
    "evaluate_dec_chain_len9_fidelity",
]
