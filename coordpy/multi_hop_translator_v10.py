"""W60 M9 — Multi-Hop Translator V10.

Strictly extends W59's ``coordpy.multi_hop_translator_v9``. V10:

* **16 backends** (A..P) — V9 had 14. Edge count: 16 × 15 = 240
  directed edges.
* **Chain-length 15** transitivity probes.
* **Five-axis trust composite** —
  ``substrate × hidden × attention × retrieval × replay``. An
  adversary must now corrupt all five axes in agreement to forge
  a compromise; corrupting any single axis collapses the
  composite.
* **Compromise-of-N detection** — V10 reports the *minimum number
  of axes* an adversary must compromise to flip the arbitration
  outcome. The W60 R-127 H-bar uses this for the
  ``compromise_threshold`` claim.

V10 strictly extends V9: when ``replay_trust = 1.0`` everywhere,
V10's composite reduces to V9's four-axis composite.

Honest scope
------------

* The 16 "backends" are still *named*, not *executed*. This is a
  graph + trust arbiter, not a multi-machine harness.
  ``W60-L-MULTI-HOP-V10-SYNTHETIC-BACKENDS-CAP`` documents this.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence


W60_MH_V10_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v10.v1")
W60_DEFAULT_MH_V10_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P")
W60_DEFAULT_MH_V10_CHAIN_LEN: int = 15
W60_DEFAULT_MH_V10_AGREEMENT_FLOOR: float = 0.3


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
        ai = float(a[i]); bi = float(b[i])
        dot += ai * bi; na += ai * ai; nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


@dataclasses.dataclass(frozen=True)
class DecBackendChainPathV10:
    chain: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    substrate_trust: float = 1.0
    hidden_trust: float = 1.0
    attention_trust: float = 1.0
    retrieval_trust: float = 1.0
    replay_trust: float = 1.0

    @property
    def composite_trust(self) -> float:
        return float(
            self.substrate_trust
            * self.hidden_trust
            * self.attention_trust
            * self.retrieval_trust
            * self.replay_trust)


def five_axis_trust_arbitration(
        *, paths: Sequence[DecBackendChainPathV10],
        agreement_floor: float = (
            W60_DEFAULT_MH_V10_AGREEMENT_FLOOR),
        trust_floor: float = 0.1,
) -> tuple[list[float], dict[str, Any]]:
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
            float(p.retrieval_trust),
            float(p.replay_trust),
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
        best = sorted_idx[0]
        return list(survivors[best][1].payload), {
            "n_paths": len(paths),
            "n_survivors": n,
            "kind": "single",
            "best_idx": int(survivors[best][0]),
            "rationale": "no_agreeing_subset",
        }
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


def estimate_compromise_threshold(
        *, paths: Sequence[DecBackendChainPathV10],
        attack_axes: Sequence[str] = (
            "substrate", "hidden", "attention",
            "retrieval", "replay"),
) -> int:
    """Return the *minimum number of axes* an adversary must drive
    to zero on the dominant path to flip the arbitration outcome.
    1 ≤ threshold ≤ 5."""
    if not paths:
        return 0
    base, _ = five_axis_trust_arbitration(paths=paths)
    n = len(paths)
    if n == 0:
        return 0
    # Identify the dominant path under base.
    dom = max(
        range(n),
        key=lambda i: float(paths[i].composite_trust))
    for k in range(1, len(attack_axes) + 1):
        from itertools import combinations
        for subset in combinations(attack_axes, k):
            new_paths = list(paths)
            new_attack = paths[dom]
            for axis in subset:
                fields = {
                    "substrate": "substrate_trust",
                    "hidden": "hidden_trust",
                    "attention": "attention_trust",
                    "retrieval": "retrieval_trust",
                    "replay": "replay_trust",
                }
                kw = {fields[axis]: 0.0}
                new_attack = dataclasses.replace(
                    new_attack, **kw)
            new_paths[dom] = new_attack
            new_out, info = five_axis_trust_arbitration(
                paths=new_paths)
            if (info["kind"] != "agreeing_subset"
                    or any(abs(float(a) - float(b)) > 1e-6
                            for a, b in zip(new_out, base))):
                return int(k)
    return int(len(attack_axes))


@dataclasses.dataclass(frozen=True)
class MultiHopV10Witness:
    schema: str
    backends: tuple[str, ...]
    chain_length: int
    n_edges: int
    arbitration_kind: str
    composite_trust_used: bool
    payload_l2: float
    retrieval_axis_used: bool
    replay_axis_used: bool
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
            "retrieval_axis_used": bool(
                self.retrieval_axis_used),
            "replay_axis_used": bool(self.replay_axis_used),
            "compromise_threshold": int(
                self.compromise_threshold),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "multi_hop_v10_witness",
            "witness": self.to_dict()})


def evaluate_dec_chain_len15_fidelity(
        *, backends: Sequence[str] = W60_DEFAULT_MH_V10_BACKENDS,
        chain_length: int = W60_DEFAULT_MH_V10_CHAIN_LEN,
        payload_dim: int = 8,
        seed: int = 600720,
) -> MultiHopV10Witness:
    import random
    rng = random.Random(int(seed))
    base_payload = tuple(
        rng.uniform(-1.0, 1.0) for _ in range(int(payload_dim)))
    backends_t = tuple(str(b) for b in backends)
    paths: list[DecBackendChainPathV10] = []
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
        paths.append(DecBackendChainPathV10(
            chain=chain,
            payload=tuple(float(round(v, 12))
                            for v in payload),
            confidence=0.5 + 0.5 * rng.random(),
            substrate_trust=sub_t,
            hidden_trust=hid_t,
            attention_trust=att_t,
            retrieval_trust=ret_t,
            replay_trust=rep_t,
        ))
    out, info = five_axis_trust_arbitration(paths=paths)
    payload_l2 = math.sqrt(sum(v * v for v in out))
    threshold = estimate_compromise_threshold(paths=paths)
    return MultiHopV10Witness(
        schema=W60_MH_V10_SCHEMA_VERSION,
        backends=backends_t,
        chain_length=int(chain_length),
        n_edges=int(n_back * (n_back - 1)),
        arbitration_kind=str(info["kind"]),
        composite_trust_used=True,
        payload_l2=float(payload_l2),
        retrieval_axis_used=True,
        replay_axis_used=True,
        compromise_threshold=int(threshold),
    )


def emit_multi_hop_v10_witness(
        backends: Sequence[str] = W60_DEFAULT_MH_V10_BACKENDS,
        *,
        chain_length: int = W60_DEFAULT_MH_V10_CHAIN_LEN,
        seed: int = 600720,
) -> MultiHopV10Witness:
    return evaluate_dec_chain_len15_fidelity(
        backends=backends,
        chain_length=int(chain_length),
        seed=int(seed))


__all__ = [
    "W60_MH_V10_SCHEMA_VERSION",
    "W60_DEFAULT_MH_V10_BACKENDS",
    "W60_DEFAULT_MH_V10_CHAIN_LEN",
    "W60_DEFAULT_MH_V10_AGREEMENT_FLOOR",
    "DecBackendChainPathV10",
    "MultiHopV10Witness",
    "five_axis_trust_arbitration",
    "estimate_compromise_threshold",
    "evaluate_dec_chain_len15_fidelity",
    "emit_multi_hop_v10_witness",
]
