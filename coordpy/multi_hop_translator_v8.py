"""W58 M9 — Multi-Hop Translator V8.

Strictly extends W57's ``coordpy.multi_hop_translator_v7``. V8
adds:

* **12 backends** (A..L) — V7 had 10. Edge count grows
  combinatorially: 12 × 11 = 132 directed edges.
* **Chain-length 11** transitivity probes.
* **Attention-trust composite** — the trust composite is now
  ``substrate_trust × hidden_trust × attention_trust``, a
  three-axis composite. An adversary must corrupt all three
  axes in agreement to forge a compromise.

V8 strictly extends V7: when ``attention_trust = 1.0`` is
supplied everywhere, V8's composite reduces to V7's.

Honest scope
------------

* The "backends" here are *named*, not *executed*. This is a
  graph + trust arbiter, not a multi-machine harness. The W58
  benchmark uses synthetic payloads where the backends are
  simulated. ``W58-L-MULTI-HOP-V8-SYNTHETIC-BACKENDS-CAP``
  documents this.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


W58_MH_V8_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v8.v1")
W58_DEFAULT_MH_V8_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
W58_DEFAULT_MH_V8_CHAIN_LEN: int = 11
W58_DEFAULT_MH_V8_AGREEMENT_FLOOR: float = 0.3


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
class DecBackendChainPathV8:
    chain: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    substrate_trust: float = 1.0
    hidden_trust: float = 1.0
    attention_trust: float = 1.0

    @property
    def composite_trust(self) -> float:
        return float(
            self.substrate_trust
            * self.hidden_trust
            * self.attention_trust)


def substrate_hidden_attention_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV8],
        agreement_floor: float = (
            W58_DEFAULT_MH_V8_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    """V8 arbitration: three-axis trust composite + greedy
    agreeing subset."""
    if not paths:
        return [], {
            "n_paths": 0,
            "kind": "abstain",
            "rationale": "no_paths",
        }
    survivors = [
        (i, p) for i, p in enumerate(paths)
        if min(
            float(p.substrate_trust),
            float(p.hidden_trust),
            float(p.attention_trust),
        ) >= float(trust_floor)]
    if not survivors:
        return [0.0] * len(paths[0].payload), {
            "n_paths": len(paths),
            "n_survivors": 0,
            "kind": "abstain",
            "rationale": "all_paths_below_trust_floor",
        }
    n = len(survivors)
    sorted_idx = sorted(
        range(n),
        key=lambda i: -float(survivors[i][1].composite_trust))
    chosen: list[int] = []
    for k in sorted_idx:
        ok = True
        for j in chosen:
            cs = _cosine(
                survivors[k][1].payload,
                survivors[j][1].payload)
            if cs < float(agreement_floor):
                ok = False
                break
        if ok:
            chosen.append(k)
    if len(chosen) < 2:
        # Highest composite single.
        best = sorted_idx[0]
        return list(survivors[best][1].payload), {
            "n_paths": len(paths),
            "n_survivors": n,
            "kind": "single",
            "best_idx": int(survivors[best][0]),
            "rationale": "no_agreeing_subset",
        }
    # Weighted average over agreeing subset.
    total_weight = 0.0
    n_dim = len(survivors[chosen[0]][1].payload)
    out = [0.0] * n_dim
    for k in chosen:
        path = survivors[k][1]
        w = (path.composite_trust * float(path.confidence))
        total_weight += w
        for j in range(n_dim):
            out[j] += w * float(path.payload[j])
    if total_weight > 0:
        out = [v / total_weight for v in out]
    return out, {
        "n_paths": len(paths),
        "n_survivors": n,
        "n_chosen": len(chosen),
        "kind": "agreeing_subset",
        "chosen_indices": [survivors[k][0] for k in chosen],
        "agreement_floor": float(agreement_floor),
        "trust_floor": float(trust_floor),
    }


@dataclasses.dataclass(frozen=True)
class MultiHopV8Witness:
    schema: str
    backends: tuple[str, ...]
    chain_length: int
    n_edges: int
    arbitration_kind: str
    composite_trust_used: bool
    payload_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backends": list(self.backends),
            "chain_length": int(self.chain_length),
            "n_edges": int(self.n_edges),
            "arbitration_kind": str(self.arbitration_kind),
            "composite_trust_used": bool(
                self.composite_trust_used),
            "payload_l2": float(round(self.payload_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v8_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len11_fidelity(
        *, backends: Sequence[str] = W58_DEFAULT_MH_V8_BACKENDS,
        chain_length: int = W58_DEFAULT_MH_V8_CHAIN_LEN,
        payload_dim: int = 8,
        seed: int = 580720,
) -> MultiHopV8Witness:
    """A constructive 11-hop probe over 12 backends.

    Generates synthetic payloads where each backend applies a
    seeded rotation to its incoming payload. Runs a chain of
    11 hops, then arbitrates over a small bundle of paths.
    """
    import random
    rng = random.Random(int(seed))
    base_payload = tuple(
        rng.uniform(-1.0, 1.0) for _ in range(int(payload_dim)))
    backends_t = tuple(str(b) for b in backends)
    paths: list[DecBackendChainPathV8] = []
    # Build 4 random chains of length=chain_length over the 12
    # backends.
    n_back = len(backends_t)
    for pi in range(4):
        chain = tuple(
            backends_t[rng.randrange(n_back)]
            for _ in range(int(chain_length)))
        # Each hop applies a small rotation; final payload is
        # a fixed function of the chain.
        payload = list(base_payload)
        for hop_idx, b in enumerate(chain):
            angle = (
                (ord(b) - ord("A") + 1)
                / float(n_back) * 0.20 * (hop_idx + 1))
            for j in range(payload_dim):
                payload[j] = (
                    payload[j] * math.cos(angle)
                    + (payload[(j + 1) % payload_dim])
                    * math.sin(angle) * 0.05)
        sub_t = 0.6 + 0.35 * rng.random()
        hid_t = 0.6 + 0.35 * rng.random()
        att_t = 0.5 + 0.40 * rng.random()
        paths.append(DecBackendChainPathV8(
            chain=chain,
            payload=tuple(float(round(v, 12)) for v in payload),
            confidence=0.5 + 0.5 * rng.random(),
            substrate_trust=sub_t,
            hidden_trust=hid_t,
            attention_trust=att_t,
        ))
    out, info = (
        substrate_hidden_attention_trust_arbitration(
            paths=paths))
    payload_l2 = math.sqrt(sum(v * v for v in out))
    return MultiHopV8Witness(
        schema=W58_MH_V8_SCHEMA_VERSION,
        backends=backends_t,
        chain_length=int(chain_length),
        n_edges=int(n_back * (n_back - 1)),
        arbitration_kind=str(info["kind"]),
        composite_trust_used=True,
        payload_l2=float(payload_l2),
    )


def emit_multi_hop_v8_witness(
        backends: Sequence[str] = W58_DEFAULT_MH_V8_BACKENDS,
        *,
        chain_length: int = W58_DEFAULT_MH_V8_CHAIN_LEN,
        seed: int = 580720,
) -> MultiHopV8Witness:
    return evaluate_dec_chain_len11_fidelity(
        backends=backends,
        chain_length=int(chain_length),
        seed=int(seed))


__all__ = [
    "W58_MH_V8_SCHEMA_VERSION",
    "W58_DEFAULT_MH_V8_BACKENDS",
    "W58_DEFAULT_MH_V8_CHAIN_LEN",
    "W58_DEFAULT_MH_V8_AGREEMENT_FLOOR",
    "DecBackendChainPathV8",
    "MultiHopV8Witness",
    "substrate_hidden_attention_trust_arbitration",
    "evaluate_dec_chain_len11_fidelity",
    "emit_multi_hop_v8_witness",
]
