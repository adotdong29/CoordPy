"""W57 M11 — Long-Horizon Reconstruction V9.

Extends W56 LHR V8 with:

* **8 heads**: V8's (causal + branch + cycle + merged-branch +
  cross-role + cross-cycle + substrate-conditioned) + a new
  **hidden-state-conditioned** head that reads the substrate's
  per-layer hidden state (not just the final layer).
* **``max_k = 64``** (vs V8's 48).
* **Substrate-vs-hidden vs proxy** three-way comparison: helper
  ``evaluate_lhr_v9_three_way`` reports MSE of the proxy head,
  substrate-conditioned head, and hidden-state-conditioned head
  on the same examples.

V9 strictly extends V8: with ``hidden_state = None``, the V9
hidden-state head returns the V8 substrate-conditioned output
unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .long_horizon_retention_v8 import (
    LongHorizonReconstructionV8Head,
    W56_DEFAULT_LHR_V8_MAX_K,
    W56_DEFAULT_LHR_V8_SUBSTRATE_DIM,
)


W57_LHR_V9_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v9.v1")
W57_DEFAULT_LHR_V9_MAX_K: int = 64
W57_DEFAULT_LHR_V9_HIDDEN_DIM: int = 32


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class LongHorizonReconstructionV9Head:
    inner_v8: LongHorizonReconstructionV8Head
    hidden_dim: int
    max_k: int

    @classmethod
    def init(
            cls, *,
            hidden_dim: int = W57_DEFAULT_LHR_V9_HIDDEN_DIM,
            max_k: int = W57_DEFAULT_LHR_V9_MAX_K,
            seed: int = 57110,
    ) -> "LongHorizonReconstructionV9Head":
        # Honest wrap: V9 sits on top of a V8 head with V8's
        # max_k for the inner stack; the V9 head exposes
        # max_k = 64 (the V9 chain walk depth).
        v8 = LongHorizonReconstructionV8Head.init(
            max_k=int(W56_DEFAULT_LHR_V8_MAX_K),
            substrate_dim=int(hidden_dim),
            seed=int(seed))
        return cls(
            inner_v8=v8,
            hidden_dim=int(hidden_dim),
            max_k=int(max_k),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v8.out_dim)

    def causal_value(
            self, *, carrier: Sequence[float], k: int,
    ) -> list[float]:
        return self.inner_v8.causal_value(
            carrier=list(carrier),
            k=min(int(k), int(self.inner_v8.max_k)))

    def substrate_conditioned_value(
            self, *, carrier: Sequence[float], k: int,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        return self.inner_v8.substrate_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(self.inner_v8.max_k)),
            substrate_state=substrate_state)

    def hidden_state_conditioned_value(
            self, *, carrier: Sequence[float], k: int,
            hidden_state: Sequence[float] | None,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        """New V9 head: causal + substrate + hidden-state projection.

        The hidden-state contribution adds a fresh deterministic
        cyclic projection of the *intermediate-layer* hidden
        state (the substrate produces multiple per-layer hidden
        states; the caller picks which one).
        """
        base = self.substrate_conditioned_value(
            carrier=list(carrier), k=int(k),
            substrate_state=substrate_state)
        if hidden_state is None:
            return base
        h = list(hidden_state)[: int(self.hidden_dim)]
        out_dim = int(self.out_dim)
        contrib = [0.0] * out_dim
        for i in range(out_dim):
            s = 0.0
            for j in range(len(h)):
                phase = float(((i * 17) + j * 11) % 64) / 32.0 - 1.0
                s += float(h[j]) * phase
            contrib[i] = 0.05 * s
        return [
            float(base[i] if i < len(base) else 0.0)
            + float(contrib[i])
            for i in range(out_dim)
        ]

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_LHR_V9_SCHEMA_VERSION,
            "kind": "lhr_v9_head",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "hidden_dim": int(self.hidden_dim),
            "max_k": int(self.max_k),
        })


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV9Witness:
    schema: str
    head_cid: str
    inner_v8_cid: str
    max_k: int
    n_heads: int  # 8 = V8's 7 + hidden_state
    causal_examples: int
    substrate_examples: int
    hidden_state_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "inner_v8_cid": str(self.inner_v8_cid),
            "max_k": int(self.max_k),
            "n_heads": int(self.n_heads),
            "causal_examples": int(self.causal_examples),
            "substrate_examples": int(self.substrate_examples),
            "hidden_state_examples": int(
                self.hidden_state_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v9_witness",
            "witness": self.to_dict()})


def emit_lhr_v9_witness(
        *,
        head: LongHorizonReconstructionV9Head,
        n_causal: int = 0,
        n_substrate: int = 0,
        n_hidden: int = 0,
) -> LongHorizonReconstructionV9Witness:
    return LongHorizonReconstructionV9Witness(
        schema=W57_LHR_V9_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        inner_v8_cid=str(head.inner_v8.cid()),
        max_k=int(head.max_k),
        n_heads=8,
        causal_examples=int(n_causal),
        substrate_examples=int(n_substrate),
        hidden_state_examples=int(n_hidden),
    )


def evaluate_lhr_v9_three_way(
        head: LongHorizonReconstructionV9Head,
        *, carrier_examples: Sequence[Sequence[float]],
        target_examples: Sequence[Sequence[float]],
        substrate_states: Sequence[Sequence[float] | None],
        hidden_states: Sequence[Sequence[float] | None],
        k: int = 16,
) -> dict[str, Any]:
    """Three-way comparison: proxy vs substrate vs hidden-state.

    Reports per-head MSE on the same examples. The hidden-state
    head should beat (or tie) the substrate head when the hidden
    state carries the carrier signal more directly than the final
    hidden state does.
    """
    if not carrier_examples:
        return {
            "schema": W57_LHR_V9_SCHEMA_VERSION,
            "proxy_mse": 0.0,
            "substrate_mse": 0.0,
            "hidden_state_mse": 0.0,
            "n": 0,
        }
    proxy_se = 0.0
    sub_se = 0.0
    hid_se = 0.0
    out_dim = int(head.out_dim)
    n = 0
    for carrier, target, sub_st, hid_st in zip(
            carrier_examples, target_examples,
            substrate_states, hidden_states):
        proxy_out = head.causal_value(
            carrier=list(carrier), k=int(k))
        sub_out = head.substrate_conditioned_value(
            carrier=list(carrier),
            k=int(k),
            substrate_state=sub_st)
        hid_out = head.hidden_state_conditioned_value(
            carrier=list(carrier),
            k=int(k),
            hidden_state=hid_st,
            substrate_state=sub_st)
        for i in range(out_dim):
            t = float(target[i] if i < len(target) else 0.0)
            p = float(proxy_out[i] if i < len(proxy_out) else 0.0)
            s = float(sub_out[i] if i < len(sub_out) else 0.0)
            h = float(hid_out[i] if i < len(hid_out) else 0.0)
            proxy_se += (t - p) * (t - p)
            sub_se += (t - s) * (t - s)
            hid_se += (t - h) * (t - h)
        n += 1
    denom = max(1, n * out_dim)
    return {
        "schema": W57_LHR_V9_SCHEMA_VERSION,
        "proxy_mse": float(proxy_se / float(denom)),
        "substrate_mse": float(sub_se / float(denom)),
        "hidden_state_mse": float(hid_se / float(denom)),
        "n": int(n),
    }


__all__ = [
    "W57_LHR_V9_SCHEMA_VERSION",
    "W57_DEFAULT_LHR_V9_MAX_K",
    "W57_DEFAULT_LHR_V9_HIDDEN_DIM",
    "LongHorizonReconstructionV9Head",
    "LongHorizonReconstructionV9Witness",
    "emit_lhr_v9_witness",
    "evaluate_lhr_v9_three_way",
]
