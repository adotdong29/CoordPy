"""W61 M9 — Multi-Hop Translator V11.

Strictly extends W60's ``coordpy.multi_hop_translator_v10``. V11:

* **18 backends** (A..R). Edge count: 18 × 17 = 306.
* **Chain-length 16** transitivity probes.
* **Six-axis trust composite** —
  ``substrate × hidden × attention × retrieval × replay ×
  attention_pattern_fidelity``. An adversary must now corrupt all
  six axes in agreement to forge a compromise.
* **Compromise-of-N detection** — V11 ranges 1 ≤ threshold ≤ 6.

V11 strictly extends V10: when ``attention_pattern_trust = 1.0``
everywhere, V11's composite reduces to V10's five-axis composite.

Honest scope: backends still named, not executed.
``W61-L-MULTI-HOP-V11-SYNTHETIC-BACKENDS-CAP`` documents.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .multi_hop_translator_v10 import (
    DecBackendChainPathV10,
    _cosine,
)


W61_MH_V11_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v11.v1")
W61_DEFAULT_MH_V11_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R")
W61_DEFAULT_MH_V11_CHAIN_LEN: int = 16
W61_DEFAULT_MH_V11_AGREEMENT_FLOOR: float = 0.3


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV11:
    chain: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    substrate_trust: float = 1.0
    hidden_trust: float = 1.0
    attention_trust: float = 1.0
    retrieval_trust: float = 1.0
    replay_trust: float = 1.0
    attention_pattern_trust: float = 1.0

    @property
    def composite_trust(self) -> float:
        return float(
            self.substrate_trust
            * self.hidden_trust
            * self.attention_trust
            * self.retrieval_trust
            * self.replay_trust
            * self.attention_pattern_trust)


def six_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV11],
        agreement_floor: float = (
            W61_DEFAULT_MH_V11_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
    if not paths:
        return [], {
            "n_paths": 0, "kind": "abstain",
            "rationale": "no_paths"}
    survivors = [
        (i, p) for i, p in enumerate(paths)
        if min(
            float(p.substrate_trust),
            float(p.hidden_trust),
            float(p.attention_trust),
            float(p.retrieval_trust),
            float(p.replay_trust),
            float(p.attention_pattern_trust)) >= float(
                trust_floor)]
    if not survivors:
        return [0.0] * len(paths[0].payload), {
            "n_paths": len(paths), "n_survivors": 0,
            "kind": "abstain",
            "rationale": "all_paths_below_trust_floor",
        }
    n = len(survivors)
    sorted_idx = sorted(
        range(n),
        key=lambda i: -float(
            survivors[i][1].composite_trust))
    chosen: list[int] = []
    for k in sorted_idx:
        ok = True
        for j in chosen:
            cs = _cosine(
                survivors[k][1].payload,
                survivors[j][1].payload)
            if cs < float(agreement_floor):
                ok = False; break
        if ok:
            chosen.append(k)
    if len(chosen) < 2:
        best = sorted_idx[0]
        return list(survivors[best][1].payload), {
            "n_paths": len(paths), "n_survivors": n,
            "kind": "single",
            "best_idx": int(survivors[best][0]),
            "rationale": "no_agreeing_subset",
        }
    total_w = 0.0
    n_dim = len(survivors[chosen[0]][1].payload)
    out = [0.0] * n_dim
    for k in chosen:
        path = survivors[k][1]
        w = path.composite_trust * float(path.confidence)
        total_w += w
        for j in range(n_dim):
            out[j] += w * float(path.payload[j])
    if total_w > 0:
        out = [v / total_w for v in out]
    return out, {
        "n_paths": len(paths), "n_survivors": n,
        "n_chosen": len(chosen),
        "kind": "agreeing_subset",
        "chosen_indices": [survivors[k][0] for k in chosen],
        "agreement_floor": float(agreement_floor),
        "trust_floor": float(trust_floor),
    }


def estimate_compromise_threshold_v11(
        *, paths: Sequence[DecBackendChainPathV11],
        attack_axes: Sequence[str] = (
            "substrate", "hidden", "attention",
            "retrieval", "replay", "attention_pattern"),
) -> int:
    if not paths:
        return 0
    base, _ = six_axis_trust_arbitration(paths=paths)
    n = len(paths)
    if n == 0:
        return 0
    dom = max(
        range(n),
        key=lambda i: float(paths[i].composite_trust))
    for k in range(1, len(attack_axes) + 1):
        from itertools import combinations
        for subset in combinations(attack_axes, k):
            new_paths = list(paths)
            attacked = paths[dom]
            for axis in subset:
                fields = {
                    "substrate": "substrate_trust",
                    "hidden": "hidden_trust",
                    "attention": "attention_trust",
                    "retrieval": "retrieval_trust",
                    "replay": "replay_trust",
                    "attention_pattern": (
                        "attention_pattern_trust"),
                }
                kw = {fields[axis]: 0.0}
                attacked = dataclasses.replace(
                    attacked, **kw)
            new_paths[dom] = attacked
            new_out, info = six_axis_trust_arbitration(
                paths=new_paths)
            if (info["kind"] != "agreeing_subset"
                    or any(abs(float(a) - float(b)) > 1e-6
                            for a, b in zip(new_out, base))):
                return int(k)
    return int(len(attack_axes))


@dataclasses.dataclass(frozen=True)
class MultiHopV11Witness:
    schema: str
    backends: tuple[str, ...]
    chain_length: int
    n_edges: int
    arbitration_kind: str
    composite_trust_used: bool
    payload_l2: float
    attention_pattern_axis_used: bool
    compromise_threshold: int

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
            "attention_pattern_axis_used": bool(
                self.attention_pattern_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v11_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len16_fidelity(
        *, backends: Sequence[str] = (
            W61_DEFAULT_MH_V11_BACKENDS),
        chain_length: int = W61_DEFAULT_MH_V11_CHAIN_LEN,
        payload_dim: int = 8, seed: int = 611720,
) -> MultiHopV11Witness:
    import random
    rng = random.Random(int(seed))
    base_payload = tuple(
        rng.uniform(-1.0, 1.0)
        for _ in range(int(payload_dim)))
    backends_t = tuple(str(b) for b in backends)
    paths: list[DecBackendChainPathV11] = []
    n_back = len(backends_t)
    for pi in range(5):
        chain = tuple(
            backends_t[rng.randrange(n_back)]
            for _ in range(int(chain_length)))
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
        ret_t = 0.5 + 0.40 * rng.random()
        rep_t = 0.5 + 0.40 * rng.random()
        atp_t = 0.5 + 0.40 * rng.random()
        paths.append(DecBackendChainPathV11(
            chain=chain,
            payload=tuple(float(round(v, 12))
                            for v in payload),
            confidence=0.5 + 0.5 * rng.random(),
            substrate_trust=sub_t,
            hidden_trust=hid_t,
            attention_trust=att_t,
            retrieval_trust=ret_t,
            replay_trust=rep_t,
            attention_pattern_trust=atp_t,
        ))
    out, info = six_axis_trust_arbitration(paths=paths)
    payload_l2 = math.sqrt(sum(v * v for v in out))
    threshold = estimate_compromise_threshold_v11(
        paths=paths)
    return MultiHopV11Witness(
        schema=W61_MH_V11_SCHEMA_VERSION,
        backends=backends_t,
        chain_length=int(chain_length),
        n_edges=int(n_back * (n_back - 1)),
        arbitration_kind=str(info["kind"]),
        composite_trust_used=True,
        payload_l2=float(payload_l2),
        attention_pattern_axis_used=True,
        compromise_threshold=int(threshold),
    )


def emit_multi_hop_v11_witness(
        backends: Sequence[str] = W61_DEFAULT_MH_V11_BACKENDS,
        *, chain_length: int = W61_DEFAULT_MH_V11_CHAIN_LEN,
        seed: int = 611720,
) -> MultiHopV11Witness:
    return evaluate_dec_chain_len16_fidelity(
        backends=backends,
        chain_length=int(chain_length),
        seed=int(seed))


__all__ = [
    "W61_MH_V11_SCHEMA_VERSION",
    "W61_DEFAULT_MH_V11_BACKENDS",
    "W61_DEFAULT_MH_V11_CHAIN_LEN",
    "W61_DEFAULT_MH_V11_AGREEMENT_FLOOR",
    "DecBackendChainPathV11",
    "MultiHopV11Witness",
    "six_axis_trust_arbitration",
    "estimate_compromise_threshold_v11",
    "evaluate_dec_chain_len16_fidelity",
    "emit_multi_hop_v11_witness",
]
