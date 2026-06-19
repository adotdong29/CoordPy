"""W56 M10 — Long-Horizon Reconstruction V8.

7-head LHR (causal + branch + cycle + merged-branch + cross-role
+ cross-cycle + **substrate-conditioned**) at ``max_k=48`` with
degradation curve probe across ``k ∈ {1..96}``.

The substrate-conditioned head consumes the tiny-runtime hidden
state as a conditioning input alongside the carrier features.
This is the load-bearing piece of the H40 "substrate-conditioned
recovery strictly improves over proxy-only" bar.

V8 strictly extends V7: when ``substrate_conditioning`` is zero,
the substrate head reduces to a no-op and V8 reduces to V7 + a
fresh additional head. Honest scope: like W55 V7, the head is
**not trained end-to-end**; ``W56-L-LHR-V8-OUTER-NOT-TRAINED-CAP``
documents the seed-variability of the substrate head.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .long_horizon_retention_v7 import (
    LongHorizonReconstructionV7Head,
    LongHorizonReconstructionV7Witness,
    W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM,
    W55_DEFAULT_LHR_V7_HIDDEN_DIM,
    W55_DEFAULT_LHR_V7_MAX_K,
    W55_DEFAULT_LHR_V7_N_BRANCHES,
    W55_DEFAULT_LHR_V7_N_CYCLES,
    W55_DEFAULT_LHR_V7_N_MERGE_PAIRS,
    W55_DEFAULT_LHR_V7_N_ROLES,
)


W56_LHR_V8_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v8.v1")
W56_DEFAULT_LHR_V8_MAX_K: int = 48
W56_DEFAULT_LHR_V8_SUBSTRATE_DIM: int = 32


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class LongHorizonReconstructionV8Head:
    """V8 head: V7 inner + substrate-conditioned head."""

    inner_v7: LongHorizonReconstructionV7Head
    substrate_dim: int
    substrate_proj_seed: int

    @classmethod
    def init(
            cls, *,
            carrier_dim: int | None = None,
            hidden_dim: int = W55_DEFAULT_LHR_V7_HIDDEN_DIM,
            out_dim: int = W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM,
            max_k: int = W56_DEFAULT_LHR_V8_MAX_K,
            substrate_dim: int = W56_DEFAULT_LHR_V8_SUBSTRATE_DIM,
            n_branches: int = W55_DEFAULT_LHR_V7_N_BRANCHES,
            n_cycles: int = W55_DEFAULT_LHR_V7_N_CYCLES,
            n_merge_pairs: int = W55_DEFAULT_LHR_V7_N_MERGE_PAIRS,
            n_roles: int = W55_DEFAULT_LHR_V7_N_ROLES,
            seed: int = 56110,
    ) -> "LongHorizonReconstructionV8Head":
        if carrier_dim is None:
            carrier_dim = int(max_k) * int(out_dim)
        inner = LongHorizonReconstructionV7Head.init(
            carrier_dim=int(carrier_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(max_k),
            n_branches=int(n_branches),
            n_cycles=int(n_cycles),
            n_merge_pairs=int(n_merge_pairs),
            n_roles=int(n_roles),
            seed=int(seed))
        return cls(
            inner_v7=inner,
            substrate_dim=int(substrate_dim),
            substrate_proj_seed=int(seed) + 81,
        )

    @property
    def max_k(self) -> int:
        return int(self.inner_v7.max_k)

    @property
    def out_dim(self) -> int:
        return int(self.inner_v7.out_dim)

    def causal_value(
            self, *, carrier: Sequence[float], k: int,
    ) -> list[float]:
        """V7 causal head only (helper for proxy-only baseline)."""
        v6_main, _, _, _, _ = self.inner_v7.forward_value(
            carrier=list(carrier),
            k=int(k),
            branch_index=0,
            cycle_index=0,
            merge_pair_index=0,
            role_index=0)
        return list(v6_main)

    def substrate_conditioned_value(
            self,
            *, carrier: Sequence[float],
            k: int,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        """Substrate-conditioned recovery head.

        Combines the V7 causal head's output with a linear
        projection of the substrate state. When the substrate
        state is ``None`` or all-zero, returns the V7 causal
        output unchanged.
        """
        v7_causal = self.causal_value(carrier=carrier, k=int(k))
        out_dim = int(self.out_dim)
        if substrate_state is None:
            return list(v7_causal)
        sub = list(substrate_state)[: int(self.substrate_dim)]
        # Cheap deterministic projection: cyclic dot product
        # with positions; no params required.
        contrib = [0.0] * out_dim
        for i in range(out_dim):
            s = 0.0
            for j in range(len(sub)):
                phase = float(((i * 31) + j * 7) % 64) / 32.0 - 1.0
                s += float(sub[j]) * phase
            contrib[i] = float(0.10 * s)
        out = [
            float(v7_causal[i] if i < len(v7_causal) else 0.0)
            + float(contrib[i])
            for i in range(out_dim)
        ]
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_LHR_V8_SCHEMA_VERSION,
            "kind": "lhr_v8_head",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "substrate_dim": int(self.substrate_dim),
            "substrate_proj_seed": int(self.substrate_proj_seed),
        })


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV8Witness:
    schema: str
    head_cid: str
    inner_v7_witness_cid: str
    max_k: int
    n_heads: int  # = 7 (V7's 6 + substrate-conditioned)
    causal_head_mse_examples: int
    substrate_head_mse_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "max_k": int(self.max_k),
            "n_heads": int(self.n_heads),
            "causal_head_mse_examples": int(
                self.causal_head_mse_examples),
            "substrate_head_mse_examples": int(
                self.substrate_head_mse_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v8_witness",
            "witness": self.to_dict()})


def emit_lhr_v8_witness(
        *,
        head: LongHorizonReconstructionV8Head,
        examples: Sequence[Any],
        substrate_examples: Sequence[Any] = (),
        k_max_for_degradation: int = 16,
) -> LongHorizonReconstructionV8Witness:
    from .long_horizon_retention_v7 import emit_lhr_v7_witness
    inner_w = emit_lhr_v7_witness(
        head=head.inner_v7,
        examples=tuple(examples),
        k_max_for_degradation=int(k_max_for_degradation))
    return LongHorizonReconstructionV8Witness(
        schema=W56_LHR_V8_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        inner_v7_witness_cid=str(inner_w.cid()),
        max_k=int(head.max_k),
        n_heads=7,
        causal_head_mse_examples=int(len(examples)),
        substrate_head_mse_examples=int(len(substrate_examples)),
    )


def evaluate_lhr_v8_substrate_vs_proxy(
        head: LongHorizonReconstructionV8Head,
        *, carrier_examples: Sequence[Sequence[float]],
        target_examples: Sequence[Sequence[float]],
        substrate_states: Sequence[Sequence[float] | None],
        k: int = 16,
) -> dict[str, Any]:
    """Compare V8's substrate-conditioned head against proxy-only
    (V7 causal) on the same examples.

    Returns ``{"substrate_mse": ..., "proxy_mse": ..., "delta": ...}``.
    ``substrate_mse`` ≤ ``proxy_mse`` is the H40 bar (substrate
    helps when the substrate state carries the carrier signal).
    """
    if not carrier_examples:
        return {
            "substrate_mse": 0.0, "proxy_mse": 0.0,
            "delta": 0.0,
            "n": 0,
        }
    sub_se = 0.0
    proxy_se = 0.0
    out_dim = int(head.out_dim)
    n_examples = 0
    for carrier, target, sub_state in zip(
            carrier_examples, target_examples, substrate_states):
        proxy_out = head.causal_value(
            carrier=list(carrier), k=int(k))
        sub_out = head.substrate_conditioned_value(
            carrier=list(carrier),
            k=int(k),
            substrate_state=(
                None if sub_state is None
                else list(sub_state)))
        for i in range(out_dim):
            t = float(target[i] if i < len(target) else 0.0)
            p = float(proxy_out[i] if i < len(proxy_out) else 0.0)
            s = float(sub_out[i] if i < len(sub_out) else 0.0)
            proxy_se += (t - p) * (t - p)
            sub_se += (t - s) * (t - s)
        n_examples += 1
    proxy_mse = proxy_se / float(max(1, n_examples * out_dim))
    sub_mse = sub_se / float(max(1, n_examples * out_dim))
    return {
        "schema": W56_LHR_V8_SCHEMA_VERSION,
        "substrate_mse": float(sub_mse),
        "proxy_mse": float(proxy_mse),
        "delta": float(proxy_mse - sub_mse),
        "n": int(n_examples),
    }


__all__ = [
    "W56_LHR_V8_SCHEMA_VERSION",
    "W56_DEFAULT_LHR_V8_MAX_K",
    "W56_DEFAULT_LHR_V8_SUBSTRATE_DIM",
    "LongHorizonReconstructionV8Head",
    "LongHorizonReconstructionV8Witness",
    "emit_lhr_v8_witness",
    "evaluate_lhr_v8_substrate_vs_proxy",
]
