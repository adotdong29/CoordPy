"""W55 M6 — Deep Proxy Stack V6 (L=14, trust-projected gating,
   disagreement-algebra head, adaptive abstain).

Extends W54 V5 with three additions:

* **L=14 depth** (vs V5's L=12) — wraps V5 with two more outer
  layers; outputs accumulate via residual scaled by a trust
  projection.
* **Trust-projected gating** — a per-layer scalar trust input
  modulates the residual contribution per layer. With trust=0
  the layer passes through unchanged (identity); with trust=1
  it adds the full residual.
* **Disagreement-algebra head** — given paired inputs ``a, b``,
  the head emits ⟨merged, ⊖ difference, ⊗ agreement_mean,
  agreement_mask_sum⟩ at the output stage.
* **Adaptive abstain threshold** — instead of a static threshold,
  V6 uses ``threshold = base_threshold * (1 + log(1 + ||x||₂))``
  so pathological large-norm inputs trigger tighter abstain
  thresholds.

Honest scope: pure-Python only, capsule-layer only.
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
from .deep_proxy_stack_v4 import (
    W53_DEFAULT_DEEP_V4_FACTOR_DIM,
    W53_DEFAULT_DEEP_V4_IN_DIM,
    W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
    W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
    W53_DEFAULT_DEEP_V4_N_HEADS,
    W53_DEFAULT_DEEP_V4_N_ROLES,
)
from .deep_proxy_stack_v5 import (
    DeepProxyStackV5,
    DeepProxyStackV5ForwardWitness,
    W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD,
    W54_DEFAULT_DEEP_V5_N_LAYERS,
    W54_DEFAULT_DEEP_V5_OUTER_LAYERS,
    emit_deep_proxy_stack_v5_forward_witness,
)
from .disagreement_algebra import (
    difference_op, intersection_op, merge_op,
    W55_DEFAULT_AGREEMENT_FLOOR,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_DEEP_V6_SCHEMA_VERSION: str = (
    "coordpy.deep_proxy_stack_v6.v1")

W55_DEFAULT_DEEP_V6_N_LAYERS: int = 14
W55_DEFAULT_DEEP_V6_OUTER_LAYERS: int = 2
W55_DEFAULT_DEEP_V6_BASE_ABSTAIN_THRESHOLD: float = 0.15
W55_DEFAULT_DEEP_V6_ADAPTIVE_GAIN: float = 0.5
W55_DEFAULT_DEEP_V6_DEFAULT_TRUST: float = 1.0


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


def _l2(values: Sequence[float]) -> float:
    s = 0.0
    for v in values:
        s += float(v) * float(v)
    return float(math.sqrt(s))


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# =============================================================================
# DeepProxyStackV6
# =============================================================================


@dataclasses.dataclass
class DeepProxyStackV6:
    """L=14 stack: V5 inner + 2 outer layers + trust gating +
    disagreement-algebra head + adaptive abstain."""

    inner_v5: DeepProxyStackV5
    n_outer_layers: int
    w_outer: tuple[ParamTensor, ...]
    b_outer: tuple[ParamTensor, ...]
    w_trust_gate: ParamTensor
    base_abstain_threshold: float
    adaptive_gain: float

    @classmethod
    def init(
            cls, *,
            n_layers: int = W55_DEFAULT_DEEP_V6_N_LAYERS,
            in_dim: int = W53_DEFAULT_DEEP_V4_IN_DIM,
            factor_dim: int = (
                W53_DEFAULT_DEEP_V4_FACTOR_DIM),
            n_heads: int = W53_DEFAULT_DEEP_V4_N_HEADS,
            n_branch_heads: int = (
                W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS),
            n_roles: int = W53_DEFAULT_DEEP_V4_N_ROLES,
            n_outer_layers: int = (
                W55_DEFAULT_DEEP_V6_OUTER_LAYERS),
            base_abstain_threshold: float = (
                W55_DEFAULT_DEEP_V6_BASE_ABSTAIN_THRESHOLD),
            adaptive_gain: float = (
                W55_DEFAULT_DEEP_V6_ADAPTIVE_GAIN),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepProxyStackV6":
        # V5 inner: layers = n_layers - n_outer_layers.
        v5_layers = max(
            W54_DEFAULT_DEEP_V5_N_LAYERS,
            int(n_layers) - int(n_outer_layers))
        rng = _DeterministicLCG(seed=int(seed) + 101)
        inner = DeepProxyStackV5.init(
            n_layers=int(v5_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            n_roles=int(n_roles),
            n_outer_layers=int(W54_DEFAULT_DEEP_V5_OUTER_LAYERS),
            abstain_threshold=float(
                base_abstain_threshold),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        w_out: list[ParamTensor] = []
        b_out: list[ParamTensor] = []
        for layer in range(int(n_outer_layers)):
            w = ParamTensor(
                shape=(int(in_dim), int(in_dim)),
                values=[])
            vals = [0.0] * (int(in_dim) * int(in_dim))
            for k in range(int(in_dim)):
                vals[k * int(in_dim) + k] = 1.0
            w.values = vals
            b = ParamTensor(
                shape=(int(in_dim),),
                values=[0.0] * int(in_dim))
            w_out.append(w)
            b_out.append(b)
        # Trust gate: 1×in_dim row, projects trust scalar to
        # per-dim residual scale.
        w_trust = ParamTensor(
            shape=(int(in_dim),),
            values=[1.0] * int(in_dim))
        return cls(
            inner_v5=inner,
            n_outer_layers=int(n_outer_layers),
            w_outer=tuple(w_out),
            b_outer=tuple(b_out),
            w_trust_gate=w_trust,
            base_abstain_threshold=float(
                base_abstain_threshold),
            adaptive_gain=float(adaptive_gain))

    def params(self) -> list[ParamTensor]:
        return (
            list(self.inner_v5.params())
            + list(self.w_outer) + list(self.b_outer)
            + [self.w_trust_gate])

    @property
    def n_layers(self) -> int:
        return int(self.inner_v5.n_layers) + int(
            self.n_outer_layers)

    @property
    def in_dim(self) -> int:
        return int(self.inner_v5.in_dim)

    @property
    def n_branch_heads(self) -> int:
        return int(self.inner_v5.n_branch_heads)

    @property
    def n_cycle_heads(self) -> int:
        return int(self.inner_v5.n_cycle_heads)

    @property
    def n_roles(self) -> int:
        return int(self.inner_v5.n_roles)

    def compute_adaptive_threshold(
            self, input_l2: float,
    ) -> float:
        """Adaptive threshold scales with input norm."""
        return float(
            self.base_abstain_threshold
            * (1.0 + float(self.adaptive_gain)
                * math.log1p(float(input_l2))))

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            role_index: int = 0,
            branch_index: int = 0,
            cycle_index: int = 0,
            trust_scalar: float = (
                W55_DEFAULT_DEEP_V6_DEFAULT_TRUST),
            uncertainty_scale: float = 1.0,
            paired_input: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Forward pass with trust gating + disagreement head."""
        v5_witness, v5_out = (
            emit_deep_proxy_stack_v5_forward_witness(
                stack=self.inner_v5,
                query_input=query_input,
                slot_keys=slot_keys,
                slot_values=slot_values,
                role_index=int(role_index),
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                uncertainty_scale=float(uncertainty_scale)))
        in_l2 = _l2(query_input)
        adapt_thr = self.compute_adaptive_threshold(in_l2)
        adaptive_abstain = (
            float(v5_witness.corruption_confidence)
            < float(adapt_thr))
        trust = float(max(0.0, min(1.0, float(trust_scalar))))
        # Apply outer layers with trust gating.
        cur = list(v5_out)
        for layer in range(self.n_outer_layers):
            w = self.w_outer[layer].values
            b = self.b_outer[layer].values
            sd = self.in_dim
            new = [0.0] * sd
            for r in range(sd):
                s = 0.0
                for j in range(sd):
                    s += float(
                        w[r * sd + j]) * float(cur[j])
                s += float(b[r])
                # Trust gating: blend new with old.
                gate_w = float(
                    self.w_trust_gate.values[r])
                effective = trust * gate_w
                new[r] = float(
                    (1.0 - effective) * float(cur[r])
                    + effective * math.tanh(s))
            cur = new
        # Disagreement-algebra head.
        if paired_input is not None:
            merged = merge_op(cur, paired_input)
            diff = difference_op(cur, paired_input)
            agree_vec, agree_mask = intersection_op(
                cur, paired_input,
                agreement_floor=W55_DEFAULT_AGREEMENT_FLOOR)
            agree_mask_sum = int(sum(agree_mask))
        else:
            merged = list(cur)
            diff = [0.0] * len(cur)
            agree_vec = list(cur)
            agree_mask = [1] * len(cur)
            agree_mask_sum = len(cur)
        # If adaptive abstain triggers, zero the merged output.
        if adaptive_abstain:
            out = [0.0] * len(cur)
        else:
            out = list(merged)
        return {
            "v5_witness": v5_witness,
            "v5_output": list(v5_out),
            "outer_output": list(cur),
            "merged_output": list(merged),
            "disagreement_vec": list(diff),
            "agreement_vec": list(agree_vec),
            "agreement_mask_sum": int(agree_mask_sum),
            "out": out,
            "trust_scalar": float(trust),
            "input_l2": float(in_l2),
            "adaptive_threshold": float(adapt_thr),
            "adaptive_abstain": bool(adaptive_abstain),
            "corruption_confidence": float(
                v5_witness.corruption_confidence),
            "corruption_flag": bool(
                v5_witness.corruption_flag),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_deep_v6_stack",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "n_outer_layers": int(self.n_outer_layers),
            "w_outer": [
                w.to_dict() for w in self.w_outer],
            "b_outer": [
                b.to_dict() for b in self.b_outer],
            "w_trust_gate": self.w_trust_gate.to_dict(),
            "base_abstain_threshold": float(round(
                self.base_abstain_threshold, 12)),
            "adaptive_gain": float(round(
                self.adaptive_gain, 12)),
        })


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DeepProxyStackV6ForwardWitness:
    stack_cid: str
    v5_witness_cid: str
    out_l2: float
    merged_l2: float
    disagreement_l2: float
    agreement_mask_sum: int
    trust_scalar: float
    input_l2: float
    adaptive_threshold: float
    adaptive_abstain: bool
    corruption_confidence: float
    corruption_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_cid": str(self.stack_cid),
            "v5_witness_cid": str(self.v5_witness_cid),
            "out_l2": float(round(self.out_l2, 12)),
            "merged_l2": float(round(self.merged_l2, 12)),
            "disagreement_l2": float(round(
                self.disagreement_l2, 12)),
            "agreement_mask_sum": int(
                self.agreement_mask_sum),
            "trust_scalar": float(round(
                self.trust_scalar, 12)),
            "input_l2": float(round(self.input_l2, 12)),
            "adaptive_threshold": float(round(
                self.adaptive_threshold, 12)),
            "adaptive_abstain": bool(
                self.adaptive_abstain),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
            "corruption_flag": bool(self.corruption_flag),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_deep_v6_witness",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_v6_forward_witness(
        *,
        stack: DeepProxyStackV6,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        trust_scalar: float = (
            W55_DEFAULT_DEEP_V6_DEFAULT_TRUST),
        uncertainty_scale: float = 1.0,
        paired_input: Sequence[float] | None = None,
) -> tuple[DeepProxyStackV6ForwardWitness, list[float]]:
    res = stack.forward_value(
        query_input=query_input,
        slot_keys=slot_keys,
        slot_values=slot_values,
        role_index=int(role_index),
        branch_index=int(branch_index),
        cycle_index=int(cycle_index),
        trust_scalar=float(trust_scalar),
        uncertainty_scale=float(uncertainty_scale),
        paired_input=paired_input)
    w = DeepProxyStackV6ForwardWitness(
        stack_cid=str(stack.cid()),
        v5_witness_cid=str(res["v5_witness"].cid()),
        out_l2=float(_l2(res["out"])),
        merged_l2=float(_l2(res["merged_output"])),
        disagreement_l2=float(_l2(
            res["disagreement_vec"])),
        agreement_mask_sum=int(res["agreement_mask_sum"]),
        trust_scalar=float(res["trust_scalar"]),
        input_l2=float(res["input_l2"]),
        adaptive_threshold=float(
            res["adaptive_threshold"]),
        adaptive_abstain=bool(res["adaptive_abstain"]),
        corruption_confidence=float(
            res["corruption_confidence"]),
        corruption_flag=bool(res["corruption_flag"]),
    )
    return w, list(res["out"])


# =============================================================================
# Verifier
# =============================================================================

W55_DEEP_V6_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_deep_v6_stack_cid_mismatch",
    "w55_deep_v6_trust_out_of_bounds",
    "w55_deep_v6_input_l2_negative",
    "w55_deep_v6_adaptive_threshold_negative",
    "w55_deep_v6_corruption_confidence_invalid",
    "w55_deep_v6_disagreement_l2_negative",
)


def verify_deep_proxy_stack_v6_witness(
        witness: DeepProxyStackV6ForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        max_input_l2: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid != str(expected_stack_cid)):
        failures.append("w55_deep_v6_stack_cid_mismatch")
    if not (0.0 <= float(witness.trust_scalar) <= 1.0):
        failures.append("w55_deep_v6_trust_out_of_bounds")
    if float(witness.input_l2) < 0.0:
        failures.append("w55_deep_v6_input_l2_negative")
    if float(witness.adaptive_threshold) < 0.0:
        failures.append(
            "w55_deep_v6_adaptive_threshold_negative")
    if not (
            0.0
            <= float(witness.corruption_confidence)
            <= 1.0):
        failures.append(
            "w55_deep_v6_corruption_confidence_invalid")
    if float(witness.disagreement_l2) < 0.0:
        failures.append(
            "w55_deep_v6_disagreement_l2_negative")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_DEEP_V6_SCHEMA_VERSION",
    "W55_DEFAULT_DEEP_V6_N_LAYERS",
    "W55_DEFAULT_DEEP_V6_OUTER_LAYERS",
    "W55_DEFAULT_DEEP_V6_BASE_ABSTAIN_THRESHOLD",
    "W55_DEFAULT_DEEP_V6_ADAPTIVE_GAIN",
    "W55_DEFAULT_DEEP_V6_DEFAULT_TRUST",
    "W55_DEEP_V6_VERIFIER_FAILURE_MODES",
    "DeepProxyStackV6",
    "DeepProxyStackV6ForwardWitness",
    "emit_deep_proxy_stack_v6_forward_witness",
    "verify_deep_proxy_stack_v6_witness",
]
