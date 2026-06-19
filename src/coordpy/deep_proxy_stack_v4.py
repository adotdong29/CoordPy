"""W53 M4 — Deep Proxy Stack V4 (depth 10 + merge-aware +
   corruption-aware heads).

Wraps a depth-10 ``DeepProxyStackV3`` and adds two heads:

* **merge-aware head** — when two parallel branch outputs are
  available, blends them via a learned per-dim gate
* **corruption-aware head** — emits a per-output confidence
  scalar reflecting the L2-stability of the per-layer output;
  large per-layer L2 jumps (or vanishing) are flagged as
  potentially corrupted

Pure-Python only — reuses W52's ``DeepProxyStackV3``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .deep_proxy_stack_v3 import (
    DeepProxyStackV3,
    DeepProxyStackV3ForwardWitness,
    W52_DEFAULT_DEEP_V3_FACTOR_DIM,
    W52_DEFAULT_DEEP_V3_IN_DIM,
    W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS,
    W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS,
    W52_DEFAULT_DEEP_V3_N_HEADS,
    W52_DEFAULT_DEEP_V3_N_ROLES,
    emit_deep_proxy_stack_v3_forward_witness,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_DEEP_V4_SCHEMA_VERSION: str = (
    "coordpy.deep_proxy_stack_v4.v1")

W53_DEFAULT_DEEP_V4_N_LAYERS: int = 10
W53_DEFAULT_DEEP_V4_IN_DIM: int = W52_DEFAULT_DEEP_V3_IN_DIM
W53_DEFAULT_DEEP_V4_FACTOR_DIM: int = (
    W52_DEFAULT_DEEP_V3_FACTOR_DIM)
W53_DEFAULT_DEEP_V4_N_HEADS: int = (
    W52_DEFAULT_DEEP_V3_N_HEADS)
W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS: int = (
    W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS)
W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS: int = (
    W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS)
W53_DEFAULT_DEEP_V4_N_ROLES: int = (
    W52_DEFAULT_DEEP_V3_N_ROLES)


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _l2(values: Sequence[float]) -> float:
    s = 0.0
    for v in values:
        s += float(v) * float(v)
    return float(math.sqrt(s))


# =============================================================================
# DeepProxyStackV4
# =============================================================================


@dataclasses.dataclass
class DeepProxyStackV4:
    """L=10 V3 stack + merge-aware head + corruption-aware head."""

    inner_v3: DeepProxyStackV3
    factor_dim: int
    w_merge: ParamTensor  # (factor_dim, 2*factor_dim)
    b_merge: ParamTensor  # (factor_dim,)
    w_corr: ParamTensor  # (1, factor_dim)
    b_corr: ParamTensor  # (1,)
    layer_l2_floor: float
    layer_l2_ceiling: float

    @classmethod
    def init(
            cls, *,
            n_layers: int = W53_DEFAULT_DEEP_V4_N_LAYERS,
            in_dim: int = W53_DEFAULT_DEEP_V4_IN_DIM,
            factor_dim: int = (
                W53_DEFAULT_DEEP_V4_FACTOR_DIM),
            n_heads: int = W53_DEFAULT_DEEP_V4_N_HEADS,
            n_branch_heads: int = (
                W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS),
            n_roles: int = W53_DEFAULT_DEEP_V4_N_ROLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            layer_l2_floor: float = 0.05,
            layer_l2_ceiling: float = 50.0,
    ) -> "DeepProxyStackV4":
        rng = _DeterministicLCG(seed=int(seed))
        inner = DeepProxyStackV3.init(
            n_layers=int(n_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            n_roles=int(n_roles),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        w_merge = ParamTensor(
            shape=(int(in_dim), 2 * int(in_dim)), values=[])
        w_merge.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        b_merge = ParamTensor(
            shape=(int(in_dim),),
            values=[0.0] * int(in_dim))
        w_corr = ParamTensor(
            shape=(1, int(in_dim)), values=[])
        w_corr.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.3)
        b_corr = ParamTensor(shape=(1,), values=[1.0])
        return cls(
            inner_v3=inner,
            factor_dim=int(in_dim),
            w_merge=w_merge, b_merge=b_merge,
            w_corr=w_corr, b_corr=b_corr,
            layer_l2_floor=float(layer_l2_floor),
            layer_l2_ceiling=float(layer_l2_ceiling))

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v3.params()) + [
            self.w_merge, self.b_merge,
            self.w_corr, self.b_corr]

    @property
    def n_layers(self) -> int:
        return int(self.inner_v3.n_layers)

    @property
    def n_branch_heads(self) -> int:
        return int(self.inner_v3.n_branch_heads)

    @property
    def n_cycle_heads(self) -> int:
        return int(self.inner_v3.n_cycle_heads)

    @property
    def n_roles(self) -> int:
        return int(self.inner_v3.n_roles)

    @property
    def in_dim(self) -> int:
        return int(self.inner_v3.in_dim)

    def merge_branch_outputs(
            self,
            out_a: Sequence[float],
            out_b: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        """Apply merge-aware head between two parallel outputs.

        Returns (merged, alpha_per_dim).
        """
        sd = int(self.factor_dim)
        cat = [
            float(out_a[i] if i < len(out_a) else 0.0)
            for i in range(sd)
        ] + [
            float(out_b[i] if i < len(out_b) else 0.0)
            for i in range(sd)
        ]
        wm = self.w_merge.values
        bm = self.b_merge.values
        alpha = [0.0] * sd
        for r in range(sd):
            s = 0.0
            base = r * (2 * sd)
            for j in range(2 * sd):
                s += float(wm[base + j]) * float(cat[j])
            s += float(bm[r])
            alpha[r] = float(_stable_sigmoid(s))
        merged = [
            float(alpha[i]) * float(
                out_a[i] if i < len(out_a) else 0.0)
            + (1.0 - float(alpha[i])) * float(
                out_b[i] if i < len(out_b) else 0.0)
            for i in range(sd)
        ]
        return merged, alpha

    def detect_corruption_value(
            self,
            output: Sequence[float],
            per_layer_l2_norms: Sequence[float],
    ) -> tuple[float, bool, str]:
        """Confidence scalar + corruption flag + reason.

        - L2 norm out of bounds → flag corruption
        - any layer's L2 below floor (vanishing) → flag corruption
        - any layer's L2 above ceiling (exploding) → flag corruption
        - else: confidence = sigmoid(w_corr · output + b_corr)
        """
        sd = int(self.factor_dim)
        wc = self.w_corr.values
        bc = self.b_corr.values
        s = 0.0
        for j in range(sd):
            s += float(wc[j]) * float(
                output[j] if j < len(output) else 0.0)
        s += float(bc[0])
        confidence = float(_stable_sigmoid(s))
        flag = False
        reason = ""
        for k, l2 in enumerate(per_layer_l2_norms):
            if float(l2) < float(self.layer_l2_floor):
                flag = True
                reason = (
                    f"layer_{k}_l2_below_floor:{float(l2):.4g}")
                break
            if float(l2) > float(self.layer_l2_ceiling):
                flag = True
                reason = (
                    f"layer_{k}_l2_above_ceiling:"
                    f"{float(l2):.4g}")
                break
        if flag:
            confidence = float(min(0.2, confidence))
        return float(confidence), bool(flag), str(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W53_DEEP_V4_SCHEMA_VERSION),
            "inner_v3": self.inner_v3.to_dict(),
            "factor_dim": int(self.factor_dim),
            "w_merge": self.w_merge.to_dict(),
            "b_merge": self.b_merge.to_dict(),
            "w_corr": self.w_corr.to_dict(),
            "b_corr": self.b_corr.to_dict(),
            "layer_l2_floor": float(round(
                self.layer_l2_floor, 12)),
            "layer_l2_ceiling": float(round(
                self.layer_l2_ceiling, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_deep_proxy_stack_v4",
            "stack": self.to_dict()})


# =============================================================================
# Forward witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DeepProxyStackV4ForwardWitness:
    stack_cid: str
    n_layers: int
    inner_witness_cid: str
    output_l2: float
    corruption_confidence: float
    corruption_flag: bool
    corruption_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_cid": str(self.stack_cid),
            "n_layers": int(self.n_layers),
            "inner_witness_cid": str(self.inner_witness_cid),
            "output_l2": float(round(self.output_l2, 12)),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
            "corruption_flag": bool(self.corruption_flag),
            "corruption_reason": str(self.corruption_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_deep_proxy_stack_v4_forward",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_v4_forward_witness(
        *,
        stack: DeepProxyStackV4,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
) -> tuple[
        DeepProxyStackV4ForwardWitness, list[float]]:
    inner_w, h_out = (
        emit_deep_proxy_stack_v3_forward_witness(
            stack=stack.inner_v3,
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index)))
    # Per-layer L2 norms for corruption detection.
    h_internal, l2s, _, _, _, _ = (
        stack.inner_v3.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index)))
    confidence, flag, reason = stack.detect_corruption_value(
        h_internal, l2s)
    output_l2 = float(_l2(h_internal))
    witness = DeepProxyStackV4ForwardWitness(
        stack_cid=str(stack.cid()),
        n_layers=int(stack.n_layers),
        inner_witness_cid=str(inner_w.cid()),
        output_l2=float(output_l2),
        corruption_confidence=float(confidence),
        corruption_flag=bool(flag),
        corruption_reason=str(reason),
    )
    return witness, h_internal


# =============================================================================
# Verifier
# =============================================================================

W53_DEEP_V4_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_deep_v4_stack_cid_mismatch",
    "w53_deep_v4_n_layers_mismatch",
    "w53_deep_v4_inner_witness_cid_mismatch",
    "w53_deep_v4_corruption_confidence_out_of_bounds",
    "w53_deep_v4_output_l2_pathology",
)


def verify_deep_proxy_stack_v4_forward_witness(
        witness: DeepProxyStackV4ForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        expected_n_layers: int | None = None,
        expected_inner_witness_cid: str | None = None,
        max_output_l2: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid
            != str(expected_stack_cid)):
        failures.append("w53_deep_v4_stack_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w53_deep_v4_n_layers_mismatch")
    if (expected_inner_witness_cid is not None
            and witness.inner_witness_cid
            != str(expected_inner_witness_cid)):
        failures.append(
            "w53_deep_v4_inner_witness_cid_mismatch")
    if not (0.0 <= float(witness.corruption_confidence)
            <= 1.0):
        failures.append(
            "w53_deep_v4_corruption_confidence_out_of_bounds")
    if (max_output_l2 is not None
            and witness.output_l2 > float(max_output_l2)):
        failures.append("w53_deep_v4_output_l2_pathology")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W53_DEEP_V4_SCHEMA_VERSION",
    "W53_DEFAULT_DEEP_V4_N_LAYERS",
    "W53_DEFAULT_DEEP_V4_IN_DIM",
    "W53_DEFAULT_DEEP_V4_FACTOR_DIM",
    "W53_DEFAULT_DEEP_V4_N_HEADS",
    "W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS",
    "W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS",
    "W53_DEFAULT_DEEP_V4_N_ROLES",
    "W53_DEEP_V4_VERIFIER_FAILURE_MODES",
    "DeepProxyStackV4",
    "DeepProxyStackV4ForwardWitness",
    "emit_deep_proxy_stack_v4_forward_witness",
    "verify_deep_proxy_stack_v4_forward_witness",
]
