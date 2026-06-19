"""W54 M6 — Deep Proxy Stack V5 (L=12, disagreement-aware,
   uncertainty-projected gating, abstain short-circuit).

Extends W53 V4 with three additions:

* **L=12 depth** (vs V4's L=10) — wraps V4 with two more
  outer layers that read the V4 output and apply a residual
  scaled by the composite uncertainty signal
* **disagreement-aware head** — given two parallel outputs
  ``out_a`` and ``out_b``, emits a per-dim disagreement
  scalar ``|out_a[i] - out_b[i]|`` AND the standard merged
  output. The disagreement is normalised to [0, ∞) and bounded
  by ``||out_a - out_b||₂``.
* **abstain short-circuit** — if the V4 corruption confidence
  falls below ``abstain_threshold``, the forward pass returns
  a zero output + abstain flag without computing the outer
  layers (saves compute and signals to upstream).

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
    DeepProxyStackV4,
    DeepProxyStackV4ForwardWitness,
    W53_DEFAULT_DEEP_V4_FACTOR_DIM,
    W53_DEFAULT_DEEP_V4_IN_DIM,
    W53_DEFAULT_DEEP_V4_N_BRANCH_HEADS,
    W53_DEFAULT_DEEP_V4_N_CYCLE_HEADS,
    W53_DEFAULT_DEEP_V4_N_HEADS,
    W53_DEFAULT_DEEP_V4_N_LAYERS,
    W53_DEFAULT_DEEP_V4_N_ROLES,
    emit_deep_proxy_stack_v4_forward_witness,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_DEEP_V5_SCHEMA_VERSION: str = (
    "coordpy.deep_proxy_stack_v5.v1")

W54_DEFAULT_DEEP_V5_N_LAYERS: int = 12
W54_DEFAULT_DEEP_V5_OUTER_LAYERS: int = 2
W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD: float = 0.15


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
# DeepProxyStackV5
# =============================================================================


@dataclasses.dataclass
class DeepProxyStackV5:
    """L=12 stack: V4 inner + 2 outer layers + disagreement head."""

    inner_v4: DeepProxyStackV4
    n_outer_layers: int
    w_outer: tuple[ParamTensor, ...]
    b_outer: tuple[ParamTensor, ...]
    w_disagreement: ParamTensor
    abstain_threshold: float

    @classmethod
    def init(
            cls, *,
            n_layers: int = W54_DEFAULT_DEEP_V5_N_LAYERS,
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
                W54_DEFAULT_DEEP_V5_OUTER_LAYERS),
            abstain_threshold: float = (
                W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepProxyStackV5":
        inner_layers = max(
            W53_DEFAULT_DEEP_V4_N_LAYERS,
            int(n_layers) - int(n_outer_layers))
        rng = _DeterministicLCG(seed=int(seed) + 71)
        inner = DeepProxyStackV4.init(
            n_layers=int(inner_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            n_roles=int(n_roles),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        w_out: list[ParamTensor] = []
        b_out: list[ParamTensor] = []
        for layer in range(int(n_outer_layers)):
            w = ParamTensor(
                shape=(int(in_dim), int(in_dim)),
                values=[])
            # Identity-init for stability + small noise.
            vals = [0.0] * (int(in_dim) * int(in_dim))
            for k in range(int(in_dim)):
                vals[k * int(in_dim) + k] = 1.0
            w.values = vals
            b = ParamTensor(
                shape=(int(in_dim),),
                values=[0.0] * int(in_dim))
            w_out.append(w)
            b_out.append(b)
        w_disagree = ParamTensor(
            shape=(int(in_dim), int(in_dim)),
            values=[1.0 if (
                k // int(in_dim) == k % int(in_dim)) else 0.0
                for k in range(int(in_dim) * int(in_dim))])
        return cls(
            inner_v4=inner,
            n_outer_layers=int(n_outer_layers),
            w_outer=tuple(w_out), b_outer=tuple(b_out),
            w_disagreement=w_disagree,
            abstain_threshold=float(abstain_threshold))

    def params(self) -> list[ParamTensor]:
        return (
            list(self.inner_v4.params())
            + list(self.w_outer) + list(self.b_outer)
            + [self.w_disagreement])

    @property
    def n_layers(self) -> int:
        return int(self.inner_v4.n_layers) + int(
            self.n_outer_layers)

    @property
    def in_dim(self) -> int:
        return int(self.inner_v4.in_dim)

    @property
    def n_branch_heads(self) -> int:
        return int(self.inner_v4.n_branch_heads)

    @property
    def n_cycle_heads(self) -> int:
        return int(self.inner_v4.n_cycle_heads)

    @property
    def n_roles(self) -> int:
        return int(self.inner_v4.n_roles)

    def disagreement_head(
            self,
            out_a: Sequence[float],
            out_b: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        """Return (merged_output, per_dim_disagreement)."""
        sd = int(self.in_dim)
        disagreement = [
            float(abs(float(out_a[i] if i < len(out_a) else 0.0)
                       - float(out_b[i]
                               if i < len(out_b) else 0.0)))
            for i in range(sd)
        ]
        merged, _ = self.inner_v4.merge_branch_outputs(
            out_a, out_b)
        return merged, disagreement

    def forward_outer_value(
            self,
            inner_output: Sequence[float],
            uncertainty_scale: float,
    ) -> list[float]:
        """Apply the outer layers with uncertainty-gated residuals.

        For each outer layer:
            y' = y + uncertainty_scale * (W_outer y + b_outer)
        Uncertainty scale ∈ [0, 1]; high uncertainty → small residual.
        """
        cur = list(inner_output)
        u = float(max(0.0, min(1.0, float(uncertainty_scale))))
        sd = int(self.in_dim)
        for w, b in zip(self.w_outer, self.b_outer):
            new = [0.0] * sd
            for r in range(sd):
                base = r * sd
                s = 0.0
                for j in range(sd):
                    s += float(w.values[base + j]) * float(
                        cur[j] if j < len(cur) else 0.0)
                s += float(b.values[r])
                new[r] = float(cur[r]) + float(u) * float(s)
            cur = new
        return cur

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W54_DEEP_V5_SCHEMA_VERSION),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "n_outer_layers": int(self.n_outer_layers),
            "w_outer": [
                p.to_dict() for p in self.w_outer],
            "b_outer": [
                p.to_dict() for p in self.b_outer],
            "w_disagreement": self.w_disagreement.to_dict(),
            "abstain_threshold": float(round(
                self.abstain_threshold, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_deep_proxy_stack_v5",
            "stack": self.to_dict()})


# =============================================================================
# Forward witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DeepProxyStackV5ForwardWitness:
    stack_cid: str
    n_layers: int
    inner_v4_witness_cid: str
    output_l2: float
    corruption_confidence: float
    corruption_flag: bool
    corruption_reason: str
    abstain_short_circuit: bool
    disagreement_l1: float
    uncertainty_scale_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_cid": str(self.stack_cid),
            "n_layers": int(self.n_layers),
            "inner_v4_witness_cid": str(
                self.inner_v4_witness_cid),
            "output_l2": float(round(self.output_l2, 12)),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
            "corruption_flag": bool(self.corruption_flag),
            "corruption_reason": str(self.corruption_reason),
            "abstain_short_circuit": bool(
                self.abstain_short_circuit),
            "disagreement_l1": float(round(
                self.disagreement_l1, 12)),
            "uncertainty_scale_used": float(round(
                self.uncertainty_scale_used, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_deep_proxy_stack_v5_forward",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_v5_forward_witness(
        *,
        stack: DeepProxyStackV5,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        uncertainty_scale: float = 1.0,
        paired_input: Sequence[float] | None = None,
) -> tuple[DeepProxyStackV5ForwardWitness, list[float]]:
    """Forward through V5 with abstain short-circuit semantics.

    Optionally takes ``paired_input`` to exercise the disagreement
    head; if provided, the witness records per-dim disagreement L1.
    """
    inner_witness, inner_out = (
        emit_deep_proxy_stack_v4_forward_witness(
            stack=stack.inner_v4,
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index)))
    abstain = (
        float(inner_witness.corruption_confidence)
        < float(stack.abstain_threshold))
    disagreement_l1 = 0.0
    if paired_input is not None:
        merged, disagreement = stack.disagreement_head(
            inner_out, paired_input)
        disagreement_l1 = float(
            sum(abs(float(v)) for v in disagreement))
    if abstain:
        out = [0.0] * int(stack.in_dim)
        u_used = 0.0
    else:
        out = stack.forward_outer_value(
            inner_out, float(uncertainty_scale))
        u_used = float(uncertainty_scale)
    out_l2 = float(_l2(out))
    witness = DeepProxyStackV5ForwardWitness(
        stack_cid=str(stack.cid()),
        n_layers=int(stack.n_layers),
        inner_v4_witness_cid=str(inner_witness.cid()),
        output_l2=float(out_l2),
        corruption_confidence=float(
            inner_witness.corruption_confidence),
        corruption_flag=bool(
            inner_witness.corruption_flag),
        corruption_reason=str(
            inner_witness.corruption_reason),
        abstain_short_circuit=bool(abstain),
        disagreement_l1=float(disagreement_l1),
        uncertainty_scale_used=float(u_used),
    )
    return witness, list(out)


# =============================================================================
# Verifier
# =============================================================================

W54_DEEP_V5_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_deep_v5_stack_cid_mismatch",
    "w54_deep_v5_n_layers_mismatch",
    "w54_deep_v5_inner_v4_witness_cid_mismatch",
    "w54_deep_v5_corruption_confidence_out_of_bounds",
    "w54_deep_v5_output_l2_pathology",
    "w54_deep_v5_disagreement_l1_negative",
    "w54_deep_v5_uncertainty_scale_out_of_bounds",
)


def verify_deep_proxy_stack_v5_forward_witness(
        witness: DeepProxyStackV5ForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        expected_n_layers: int | None = None,
        expected_inner_v4_witness_cid: str | None = None,
        max_output_l2: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid
            != str(expected_stack_cid)):
        failures.append("w54_deep_v5_stack_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers
            != int(expected_n_layers)):
        failures.append("w54_deep_v5_n_layers_mismatch")
    if (expected_inner_v4_witness_cid is not None
            and witness.inner_v4_witness_cid
            != str(expected_inner_v4_witness_cid)):
        failures.append(
            "w54_deep_v5_inner_v4_witness_cid_mismatch")
    if not (0.0 <= float(witness.corruption_confidence)
            <= 1.0):
        failures.append(
            "w54_deep_v5_corruption_confidence_out_of_bounds")
    if (max_output_l2 is not None
            and witness.output_l2 > float(max_output_l2)):
        failures.append("w54_deep_v5_output_l2_pathology")
    if witness.disagreement_l1 < 0.0:
        failures.append("w54_deep_v5_disagreement_l1_negative")
    if not (0.0 <= float(witness.uncertainty_scale_used)
            <= 1.0):
        failures.append(
            "w54_deep_v5_uncertainty_scale_out_of_bounds")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_DEEP_V5_SCHEMA_VERSION",
    "W54_DEFAULT_DEEP_V5_N_LAYERS",
    "W54_DEFAULT_DEEP_V5_OUTER_LAYERS",
    "W54_DEFAULT_DEEP_V5_ABSTAIN_THRESHOLD",
    "W54_DEEP_V5_VERIFIER_FAILURE_MODES",
    "DeepProxyStackV5",
    "DeepProxyStackV5ForwardWitness",
    "emit_deep_proxy_stack_v5_forward_witness",
    "verify_deep_proxy_stack_v5_forward_witness",
]
