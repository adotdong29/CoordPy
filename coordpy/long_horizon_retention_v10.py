"""W58 M13 — Long-Horizon Retention V10.

Strictly extends W57's
``coordpy.long_horizon_retention_v9.LongHorizonReconstructionV9Head``.
V10 adds a *ninth* head — **attention-conditioned reconstruction**
— and raises ``max_k`` to **72** (vs V9's 64). The new head reads
the W58 substrate V3's attention pattern at the chosen layer and
projects it into the reconstruction output dimension.

V10 strictly extends V9: when ``attention_state = None``, V10
reduces to V9 byte-for-byte.

Honest scope
------------

* The V10 head is *initialised but not trained* end-to-end.
  ``W58-L-V10-LHR-NOT-TRAINED-CAP`` documents the boundary.
* The H90 bar asserts ``evaluate_lhr_v10_four_way`` runs without
  crashing and reports four MSEs (proxy / substrate / hidden /
  attention). The *quality* claim that the new head beats older
  heads on attention-aligned targets is *constructive only*
  (synthetic targets explicitly projected via the attention head
  by definition).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .long_horizon_retention_v9 import (
    LongHorizonReconstructionV9Head,
    W57_DEFAULT_LHR_V9_HIDDEN_DIM,
    W57_DEFAULT_LHR_V9_MAX_K,
)


W58_LHR_V10_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v10.v1")
W58_DEFAULT_LHR_V10_MAX_K: int = 72
W58_DEFAULT_LHR_V10_HIDDEN_DIM: int = (
    W57_DEFAULT_LHR_V9_HIDDEN_DIM)
W58_DEFAULT_LHR_V10_ATTENTION_DIM: int = 32


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class LongHorizonReconstructionV10Head:
    inner_v9: LongHorizonReconstructionV9Head
    attention_dim: int
    max_k: int

    @classmethod
    def init(
            cls, *,
            hidden_dim: int = W58_DEFAULT_LHR_V10_HIDDEN_DIM,
            attention_dim: int = (
                W58_DEFAULT_LHR_V10_ATTENTION_DIM),
            max_k: int = W58_DEFAULT_LHR_V10_MAX_K,
            seed: int = 58120,
    ) -> "LongHorizonReconstructionV10Head":
        v9 = LongHorizonReconstructionV9Head.init(
            hidden_dim=int(hidden_dim),
            max_k=int(W57_DEFAULT_LHR_V9_MAX_K),
            seed=int(seed))
        return cls(
            inner_v9=v9,
            attention_dim=int(attention_dim),
            max_k=int(max_k),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v9.out_dim)

    def attention_conditioned_value(
            self, *, carrier: Sequence[float], k: int,
            attention_state: Sequence[float] | None,
            hidden_state: Sequence[float] | None,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        """V10 head: causal + substrate + hidden + attention.

        The attention contribution adds a fresh deterministic
        cyclic projection of the *attention-pattern* signal.
        """
        base = self.inner_v9.hidden_state_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(self.inner_v9.max_k)),
            hidden_state=hidden_state,
            substrate_state=substrate_state)
        if attention_state is None:
            return base
        a = list(attention_state)[: int(self.attention_dim)]
        out_dim = int(self.out_dim)
        contrib = [0.0] * out_dim
        for i in range(out_dim):
            s = 0.0
            for j in range(len(a)):
                phase = (
                    float(((i * 19) + j * 13) % 64) / 32.0 - 1.0)
                s += float(a[j]) * phase
            contrib[i] = 0.04 * s
        return [
            float(base[i] if i < len(base) else 0.0)
            + float(contrib[i])
            for i in range(out_dim)
        ]

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_LHR_V10_SCHEMA_VERSION,
            "kind": "lhr_v10_head",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "attention_dim": int(self.attention_dim),
            "max_k": int(self.max_k),
        })


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV10Witness:
    schema: str
    head_cid: str
    inner_v9_cid: str
    max_k: int
    n_heads: int  # 9 = V9's 8 + attention
    causal_examples: int
    substrate_examples: int
    hidden_state_examples: int
    attention_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "inner_v9_cid": str(self.inner_v9_cid),
            "max_k": int(self.max_k),
            "n_heads": int(self.n_heads),
            "causal_examples": int(self.causal_examples),
            "substrate_examples": int(self.substrate_examples),
            "hidden_state_examples": int(
                self.hidden_state_examples),
            "attention_examples": int(self.attention_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v10_witness",
            "witness": self.to_dict()})


def emit_lhr_v10_witness(
        *,
        head: LongHorizonReconstructionV10Head,
        n_causal: int = 0,
        n_substrate: int = 0,
        n_hidden: int = 0,
        n_attention: int = 0,
) -> LongHorizonReconstructionV10Witness:
    return LongHorizonReconstructionV10Witness(
        schema=W58_LHR_V10_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        inner_v9_cid=str(head.inner_v9.cid()),
        max_k=int(head.max_k),
        n_heads=9,
        causal_examples=int(n_causal),
        substrate_examples=int(n_substrate),
        hidden_state_examples=int(n_hidden),
        attention_examples=int(n_attention),
    )


def evaluate_lhr_v10_four_way(
        head: LongHorizonReconstructionV10Head,
        *, carrier_examples: Sequence[Sequence[float]],
        target_examples: Sequence[Sequence[float]],
        substrate_states: Sequence[Sequence[float] | None],
        hidden_states: Sequence[Sequence[float] | None],
        attention_states: Sequence[Sequence[float] | None],
        k: int = 16,
) -> dict[str, Any]:
    """Four-way comparison: proxy vs substrate vs hidden vs
    attention. Reports per-head MSE."""
    if not carrier_examples:
        return {
            "schema": W58_LHR_V10_SCHEMA_VERSION,
            "proxy_mse": 0.0,
            "substrate_mse": 0.0,
            "hidden_state_mse": 0.0,
            "attention_mse": 0.0,
            "n": 0,
        }
    proxy_se = 0.0
    sub_se = 0.0
    hid_se = 0.0
    att_se = 0.0
    out_dim = int(head.out_dim)
    n = 0
    for carrier, target, sub_st, hid_st, att_st in zip(
            carrier_examples, target_examples,
            substrate_states, hidden_states,
            attention_states):
        proxy_out = head.inner_v9.causal_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v9.max_k)))
        sub_out = head.inner_v9.substrate_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v9.max_k)),
            substrate_state=sub_st)
        hid_out = head.inner_v9.hidden_state_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v9.max_k)),
            hidden_state=hid_st, substrate_state=sub_st)
        att_out = head.attention_conditioned_value(
            carrier=list(carrier), k=int(k),
            attention_state=att_st,
            hidden_state=hid_st,
            substrate_state=sub_st)
        t = list(target)[:out_dim]
        while len(t) < out_dim:
            t.append(0.0)
        for i in range(out_dim):
            d_p = float(proxy_out[i] if i < len(proxy_out) else 0.0) - float(t[i])
            d_s = float(sub_out[i] if i < len(sub_out) else 0.0) - float(t[i])
            d_h = float(hid_out[i] if i < len(hid_out) else 0.0) - float(t[i])
            d_a = float(att_out[i] if i < len(att_out) else 0.0) - float(t[i])
            proxy_se += d_p * d_p
            sub_se += d_s * d_s
            hid_se += d_h * d_h
            att_se += d_a * d_a
        n += 1
    denom = max(1, n) * max(1, out_dim)
    return {
        "schema": W58_LHR_V10_SCHEMA_VERSION,
        "proxy_mse": float(proxy_se / float(denom)),
        "substrate_mse": float(sub_se / float(denom)),
        "hidden_state_mse": float(hid_se / float(denom)),
        "attention_mse": float(att_se / float(denom)),
        "n": int(n),
    }


__all__ = [
    "W58_LHR_V10_SCHEMA_VERSION",
    "W58_DEFAULT_LHR_V10_MAX_K",
    "W58_DEFAULT_LHR_V10_HIDDEN_DIM",
    "W58_DEFAULT_LHR_V10_ATTENTION_DIM",
    "LongHorizonReconstructionV10Head",
    "LongHorizonReconstructionV10Witness",
    "emit_lhr_v10_witness",
    "evaluate_lhr_v10_four_way",
]
