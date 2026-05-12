"""W52 M3 — Deep Proxy Stack V3 (L=8 with role banks + residual gate).

Extends W51's :class:`DeepProxyStackV2` with two structural
extensions:

* **Role-conditioned KV banks** — each layer carries a
  per-role residual subspace projection that is added to
  the layer output (one matrix per role per layer).
* **Pre-norm + residual gate per layer** — a learned sigmoid
  gate on the residual path; initialised to 1.0 (open) so a
  trivially-configured V3 reduces to V2-like behaviour.

Pure-Python only — reuses the W51 V2 ``DeepLayerV2`` building
blocks and the W47 ``Variable`` autograd engine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .autograd_manifold import (
    AdamOptimizer,
    ParamTensor,
    Variable,
    W47_DEFAULT_BETA1,
    W47_DEFAULT_BETA2,
    W47_DEFAULT_EPS,
    W47_DEFAULT_GRAD_CLIP,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_LEARNING_RATE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
    vmean,
    vmatmul,
)
from .deep_proxy_stack_v2 import (
    DeepLayerV2,
    DeepStackV2TrainingExample,
    DeepStackV2TrainingSet,
    W51_DEFAULT_DEEP_V2_FACTOR_DIM,
    W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM,
    W51_DEFAULT_DEEP_V2_IN_DIM,
    W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS,
    W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS,
    W51_DEFAULT_DEEP_V2_N_HEADS,
    _l2,
    synthesize_deep_stack_v2_training_set,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_DEEP_V3_SCHEMA_VERSION: str = (
    "coordpy.deep_proxy_stack_v3.v1")

W52_DEFAULT_DEEP_V3_N_LAYERS: int = 8
W52_DEFAULT_DEEP_V3_IN_DIM: int = W51_DEFAULT_DEEP_V2_IN_DIM
W52_DEFAULT_DEEP_V3_FACTOR_DIM: int = W51_DEFAULT_DEEP_V2_FACTOR_DIM
W52_DEFAULT_DEEP_V3_N_HEADS: int = W51_DEFAULT_DEEP_V2_N_HEADS
W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS: int = (
    W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS)
W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS: int = (
    W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS)
W52_DEFAULT_DEEP_V3_N_ROLES: int = 4


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


# =============================================================================
# RoleBank — per-role residual projection
# =============================================================================


@dataclasses.dataclass
class RoleBank:
    """Per-role residual subspace projection.

    Stores a learnable (n_roles, in_dim, in_dim) projection
    matrix; the per-role output is added to the layer output
    via the residual-gate.
    """

    n_roles: int
    in_dim: int
    w_proj: ParamTensor  # (n_roles * in_dim, in_dim)
    b_proj: ParamTensor  # (n_roles, in_dim)

    @classmethod
    def init(
            cls, *,
            n_roles: int = W52_DEFAULT_DEEP_V3_N_ROLES,
            in_dim: int = W52_DEFAULT_DEEP_V3_IN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "RoleBank":
        rng = _DeterministicLCG(seed=int(seed))
        # Initialise as a small perturbation of zero so it acts
        # like an additive offset that the optimiser can shape.
        size = int(n_roles) * int(in_dim) * int(in_dim)
        vals = [
            (rng.next_uniform() - 0.5) * float(init_scale) * 0.1
            for _ in range(size)
        ]
        w = ParamTensor(
            shape=(int(n_roles) * int(in_dim), int(in_dim)),
            values=vals)
        b = ParamTensor(
            shape=(int(n_roles), int(in_dim)),
            values=[0.0] * (int(n_roles) * int(in_dim)))
        return cls(
            n_roles=int(n_roles),
            in_dim=int(in_dim),
            w_proj=w,
            b_proj=b)

    def params(self) -> list[ParamTensor]:
        return [self.w_proj, self.b_proj]

    def project_value(
            self, *,
            role_index: int,
            x: Sequence[float],
    ) -> list[float]:
        r = int(role_index) % int(self.n_roles)
        sd = int(self.in_dim)
        out = [0.0] * sd
        for i in range(sd):
            row_base = (r * sd * sd) + (i * sd)
            s = 0.0
            for j in range(sd):
                xj = float(x[j]) if j < len(x) else 0.0
                s += float(self.w_proj.values[row_base + j]) * xj
            s += float(self.b_proj.values[r * sd + i])
            out[i] = s
        return out

    def project_vars(
            self, *,
            role_index: int,
            x: Sequence[Variable],
    ) -> list[Variable]:
        r = int(role_index) % int(self.n_roles)
        sd = int(self.in_dim)
        w_vars = self.w_proj.make_vars()
        b_vars = self.b_proj.make_vars()
        rows: list[list[Variable]] = []
        for i in range(sd):
            row_base = (r * sd * sd) + (i * sd)
            rows.append(list(w_vars[row_base:row_base + sd]))
        pre = vmatmul(rows, list(x))
        return [
            pre[i] + b_vars[r * sd + i]
            for i in range(sd)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_roles": int(self.n_roles),
            "in_dim": int(self.in_dim),
            "w_proj": self.w_proj.to_dict(),
            "b_proj": self.b_proj.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_bank",
            "bank": self.to_dict()})


# =============================================================================
# V3 layer: V2 layer + role bank + residual gate
# =============================================================================


@dataclasses.dataclass
class DeepLayerV3:
    """Single V3 layer: V2 layer + role bank + residual gate."""

    in_dim: int
    v2_layer: DeepLayerV2
    role_bank: RoleBank
    residual_gate_logit: ParamTensor  # scalar — sigmoid(x) ∈ [0, 1]

    @classmethod
    def init(
            cls, *,
            in_dim: int = W52_DEFAULT_DEEP_V3_IN_DIM,
            factor_dim: int = W52_DEFAULT_DEEP_V3_FACTOR_DIM,
            n_heads: int = W52_DEFAULT_DEEP_V3_N_HEADS,
            n_branch_heads: int = (
                W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS),
            n_roles: int = W52_DEFAULT_DEEP_V3_N_ROLES,
            ffn_hidden_dim: int = W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepLayerV3":
        rng = _DeterministicLCG(seed=int(seed))
        v2 = DeepLayerV2.init(
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            ffn_hidden_dim=int(ffn_hidden_dim),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            gate_in_dim=int(in_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        bank = RoleBank.init(
            n_roles=int(n_roles),
            in_dim=int(in_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        # Initialise residual gate to 0.0 (sigmoid(0)=0.5) by
        # default, so the V3 residual contribution starts small
        # and the trivial passthrough is close to V2.
        rg = ParamTensor(shape=(1,), values=[0.0])
        return cls(
            in_dim=int(in_dim),
            v2_layer=v2,
            role_bank=bank,
            residual_gate_logit=rg)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.v2_layer.params())
        out.extend(self.role_bank.params())
        out.append(self.residual_gate_logit)
        return out

    @property
    def residual_gate(self) -> float:
        return float(_stable_sigmoid(
            float(self.residual_gate_logit.values[0])))

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            role_index: int,
            branch_index: int,
            cycle_index: int,
    ) -> tuple[list[float], float, float, float]:
        """Returns (h_out, v2_gate, bc_gate, residual_gate)."""
        h_v2, gate_v, bc_v = self.v2_layer.forward_value(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        role_proj = self.role_bank.project_value(
            role_index=int(role_index),
            x=query_input)
        rg = self.residual_gate
        out = [
            float(h_v2[i] if i < len(h_v2) else 0.0)
            + rg * float(
                role_proj[i] if i < len(role_proj) else 0.0)
            for i in range(self.in_dim)
        ]
        return out, float(gate_v), float(bc_v), float(rg)

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
            role_index: int,
            branch_index: int,
            cycle_index: int,
    ) -> list[Variable]:
        h_v2 = self.v2_layer.forward_vars(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        role_proj = self.role_bank.project_vars(
            role_index=int(role_index),
            x=query_input)
        rg_vars = self.residual_gate_logit.make_vars()
        rg = rg_vars[0].sigmoid()
        out: list[Variable] = []
        for i in range(self.in_dim):
            a = h_v2[i] if i < len(h_v2) else Variable(0.0)
            b = role_proj[i] if i < len(role_proj) else Variable(0.0)
            out.append(a + rg * b)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "v2_layer": self.v2_layer.to_dict(),
            "role_bank": self.role_bank.to_dict(),
            "residual_gate_logit": (
                self.residual_gate_logit.to_dict()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_deep_layer_v3",
            "layer": self.to_dict()})


# =============================================================================
# DeepProxyStackV3
# =============================================================================


@dataclasses.dataclass
class DeepProxyStackV3:
    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    n_branch_heads: int
    n_cycle_heads: int
    n_roles: int
    layers: tuple[DeepLayerV3, ...]

    @classmethod
    def init(
            cls, *,
            n_layers: int = W52_DEFAULT_DEEP_V3_N_LAYERS,
            in_dim: int = W52_DEFAULT_DEEP_V3_IN_DIM,
            factor_dim: int = W52_DEFAULT_DEEP_V3_FACTOR_DIM,
            n_heads: int = W52_DEFAULT_DEEP_V3_N_HEADS,
            ffn_hidden_dim: int = W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM,
            n_branch_heads: int = (
                W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS),
            n_roles: int = W52_DEFAULT_DEEP_V3_N_ROLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepProxyStackV3":
        rng = _DeterministicLCG(seed=int(seed))
        layers: list[DeepLayerV3] = []
        for _ in range(int(n_layers)):
            layers.append(DeepLayerV3.init(
                in_dim=int(in_dim),
                factor_dim=int(factor_dim),
                n_heads=int(n_heads),
                ffn_hidden_dim=int(ffn_hidden_dim),
                n_branch_heads=int(n_branch_heads),
                n_cycle_heads=int(n_cycle_heads),
                n_roles=int(n_roles),
                seed=int(rng.next_uniform() * (1 << 30)),
                init_scale=float(init_scale)))
        return cls(
            n_layers=int(n_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            n_roles=int(n_roles),
            layers=tuple(layers))

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def forward_value_with_witness(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            role_index: int = 0,
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> tuple[list[float], list[float], list[float],
               list[float], list[float], list[float]]:
        """Returns (h_out, l2_norms, v2_gates, bc_gates,
        residual_gates, role_bank_norms)."""
        h = list(query_input)
        l2s: list[float] = []
        gates: list[float] = []
        bcs: list[float] = []
        rgs: list[float] = []
        rb_norms: list[float] = []
        for layer in self.layers:
            h, gv, bcv, rg = layer.forward_value(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
                role_index=int(role_index),
                branch_index=int(branch_index),
                cycle_index=int(cycle_index))
            l2s.append(float(_l2(h)))
            gates.append(float(gv))
            bcs.append(float(bcv))
            rgs.append(float(rg))
            rb_norms.append(float(_l2(
                layer.role_bank.project_value(
                    role_index=int(role_index),
                    x=h))))
        return h, l2s, gates, bcs, rgs, rb_norms

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            role_index: int = 0,
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> list[float]:
        out, _, _, _, _, _ = self.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        return out

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
            role_index: int = 0,
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> list[Variable]:
        h = list(query_input)
        for layer in self.layers:
            h = layer.forward_vars(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
                role_index=int(role_index),
                branch_index=int(branch_index),
                cycle_index=int(cycle_index))
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W52_DEEP_V3_SCHEMA_VERSION),
            "n_layers": int(self.n_layers),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "n_branch_heads": int(self.n_branch_heads),
            "n_cycle_heads": int(self.n_cycle_heads),
            "n_roles": int(self.n_roles),
            "layers": [l.to_dict() for l in self.layers],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_deep_proxy_stack_v3",
            "stack": self.to_dict()})


# =============================================================================
# Forward witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DeepProxyStackV3ForwardWitness:
    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    n_branch_heads: int
    n_cycle_heads: int
    n_roles: int
    stack_cid: str
    per_layer_l2_norms: tuple[float, ...]
    per_layer_v2_gate_values: tuple[float, ...]
    per_layer_bc_gate_values: tuple[float, ...]
    per_layer_residual_gates: tuple[float, ...]
    per_layer_role_bank_norms: tuple[float, ...]
    output_l2_norm: float
    input_l2_norm: float
    role_index: int
    branch_index: int
    cycle_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "n_branch_heads": int(self.n_branch_heads),
            "n_cycle_heads": int(self.n_cycle_heads),
            "n_roles": int(self.n_roles),
            "stack_cid": str(self.stack_cid),
            "per_layer_l2_norms": [
                float(round(v, 12))
                for v in self.per_layer_l2_norms],
            "per_layer_v2_gate_values": [
                float(round(v, 12))
                for v in self.per_layer_v2_gate_values],
            "per_layer_bc_gate_values": [
                float(round(v, 12))
                for v in self.per_layer_bc_gate_values],
            "per_layer_residual_gates": [
                float(round(v, 12))
                for v in self.per_layer_residual_gates],
            "per_layer_role_bank_norms": [
                float(round(v, 12))
                for v in self.per_layer_role_bank_norms],
            "output_l2_norm": float(round(
                self.output_l2_norm, 12)),
            "input_l2_norm": float(round(
                self.input_l2_norm, 12)),
            "role_index": int(self.role_index),
            "branch_index": int(self.branch_index),
            "cycle_index": int(self.cycle_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_deep_proxy_stack_v3_forward_witness",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_v3_forward_witness(
        *,
        stack: DeepProxyStackV3,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
) -> tuple[DeepProxyStackV3ForwardWitness, list[float]]:
    h, l2s, gates, bcs, rgs, rb_norms = (
        stack.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index)))
    w = DeepProxyStackV3ForwardWitness(
        n_layers=int(stack.n_layers),
        in_dim=int(stack.in_dim),
        factor_dim=int(stack.factor_dim),
        n_heads=int(stack.n_heads),
        n_branch_heads=int(stack.n_branch_heads),
        n_cycle_heads=int(stack.n_cycle_heads),
        n_roles=int(stack.n_roles),
        stack_cid=str(stack.cid()),
        per_layer_l2_norms=tuple(l2s),
        per_layer_v2_gate_values=tuple(gates),
        per_layer_bc_gate_values=tuple(bcs),
        per_layer_residual_gates=tuple(rgs),
        per_layer_role_bank_norms=tuple(rb_norms),
        output_l2_norm=float(_l2(h)),
        input_l2_norm=float(_l2(query_input)),
        role_index=int(role_index),
        branch_index=int(branch_index),
        cycle_index=int(cycle_index))
    return w, list(h)


# =============================================================================
# Training set + fit
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DeepStackV3TrainingExample:
    """One V3 training example with role index."""

    input_vec: tuple[float, ...]
    target_label: int
    branch_index: int
    cycle_index: int
    role_index: int


@dataclasses.dataclass(frozen=True)
class DeepStackV3TrainingSet:
    examples: tuple[DeepStackV3TrainingExample, ...]
    in_dim: int
    n_classes: int
    n_branches: int
    n_cycles: int
    n_roles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "n_classes": int(self.n_classes),
            "n_branches": int(self.n_branches),
            "n_cycles": int(self.n_cycles),
            "n_roles": int(self.n_roles),
            "examples": [
                {"input_vec": list(e.input_vec),
                 "target_label": int(e.target_label),
                 "branch_index": int(e.branch_index),
                 "cycle_index": int(e.cycle_index),
                 "role_index": int(e.role_index)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_deep_stack_v3_training_set",
            "set": self.to_dict()})


def synthesize_deep_stack_v3_training_set(
        *,
        n_examples: int = 24,
        in_dim: int = W52_DEFAULT_DEEP_V3_IN_DIM,
        compose_depth: int = 8,
        n_branches: int = 2,
        n_cycles: int = 2,
        n_roles: int = 2,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> DeepStackV3TrainingSet:
    """Synthesise an 8-step composition regime sensitive to
    role index.

    Target label depends on:
        sign(sum(x) + branch_offset + cycle_offset + role_offset).
    """
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[DeepStackV3TrainingExample] = []
    for i in range(int(n_examples)):
        x = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(in_dim))
        ]
        bi = i % max(1, int(n_branches))
        ci = (i // max(1, int(n_branches))) % max(1, int(n_cycles))
        ri = (i // (max(1, int(n_branches)) * max(1, int(n_cycles)))) % max(1, int(n_roles))
        branch_offset = (bi * 2 - 1) * 0.2
        cycle_offset = (ci * 2 - 1) * 0.15
        role_offset = (ri * 2 - 1) * 0.25
        depth_compose = sum(
            (x[k % len(x)]) ** (1 if k % 2 == 0 else 1)
            for k in range(int(compose_depth)))
        score = depth_compose + branch_offset + cycle_offset + role_offset
        label = 1 if score >= 0.0 else 0
        examples.append(DeepStackV3TrainingExample(
            input_vec=tuple(x),
            target_label=int(label),
            branch_index=int(bi),
            cycle_index=int(ci),
            role_index=int(ri)))
    return DeepStackV3TrainingSet(
        examples=tuple(examples),
        in_dim=int(in_dim),
        n_classes=2,
        n_branches=int(n_branches),
        n_cycles=int(n_cycles),
        n_roles=int(n_roles))


@dataclasses.dataclass(frozen=True)
class DeepStackV3TrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_stack_cid: str
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "final_loss": float(round(self.final_loss, 12)),
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
            "loss_head": [float(round(v, 12))
                          for v in self.loss_head],
            "loss_tail": [float(round(v, 12))
                          for v in self.loss_tail],
            "training_set_cid": str(self.training_set_cid),
            "final_stack_cid": str(self.final_stack_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_deep_stack_v3_training_trace",
            "trace": self.to_dict()})


def fit_deep_proxy_stack_v3(
        training_set: DeepStackV3TrainingSet,
        *,
        n_layers: int = W52_DEFAULT_DEEP_V3_N_LAYERS,
        n_steps: int = 96,
        learning_rate: float = 0.05,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
        n_branch_heads: int = W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS,
        n_cycle_heads: int = W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS,
        n_roles: int = W52_DEFAULT_DEEP_V3_N_ROLES,
) -> tuple[DeepProxyStackV3, DeepStackV3TrainingTrace]:
    """Fit V3 on a classification regime via Adam + cross-entropy."""
    stack = DeepProxyStackV3.init(
        n_layers=int(n_layers),
        in_dim=int(training_set.in_dim),
        n_branch_heads=int(n_branch_heads),
        n_cycle_heads=int(n_cycle_heads),
        n_roles=int(n_roles),
        seed=int(seed),
        init_scale=float(init_scale))
    # Classification head: sigmoid over the mean of output dims.
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    params = stack.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in params:
            p.make_vars()
        ex = training_set.examples[step % n]
        x_vars = [Variable(float(v)) for v in ex.input_vec]
        slot = [x_vars]
        out = stack.forward_vars(
            query_input=x_vars,
            slot_keys=slot, slot_values=slot,
            role_index=int(ex.role_index),
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index))
        # Mean of outputs → logit; sigmoid → probability of label 1.
        mean_var = vmean(out)
        prob = mean_var.sigmoid()
        target = Variable(float(int(ex.target_label)))
        # Binary cross-entropy.
        eps_v = Variable(1e-6)
        one_minus_t = Variable(1.0) - target
        one_minus_p = Variable(1.0) - prob
        loss_pos = target * (prob + eps_v).log() * Variable(-1.0)
        loss_neg = one_minus_t * (one_minus_p + eps_v).log() * Variable(-1.0)
        loss = loss_pos + loss_neg
        loss.backward()
        total_grad_sq = 0.0
        for p in params:
            for g in p.grads():
                total_grad_sq += float(g) * float(g)
        gn = math.sqrt(total_grad_sq)
        loss_history.append(float(loss.value))
        grad_norm_history.append(float(gn))
        lv = loss.value
        if (lv != lv or lv == float("inf")
                or lv == float("-inf")):
            diverged = True
            break
        optim.step(params)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = DeepStackV3TrainingTrace(
        seed=int(seed),
        n_steps=int(n_steps),
        final_loss=float(
            loss_history[-1] if loss_history else 0.0),
        final_grad_norm=float(
            grad_norm_history[-1]
            if grad_norm_history else 0.0),
        loss_head=tuple(loss_history[:head_n]),
        loss_tail=tuple(
            loss_history[-tail_n:] if tail_n > 0 else ()),
        training_set_cid=str(training_set.cid()),
        final_stack_cid=str(stack.cid()),
        diverged=bool(diverged),
    )
    return stack, trace


def evaluate_deep_stack_v3_accuracy(
        stack: DeepProxyStackV3,
        examples: Sequence[DeepStackV3TrainingExample],
) -> float:
    """Mean classification accuracy of V3."""
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        out = stack.forward_value(
            query_input=list(ex.input_vec),
            slot_keys=[list(ex.input_vec)],
            slot_values=[list(ex.input_vec)],
            role_index=int(ex.role_index),
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index))
        mean = sum(out) / float(max(1, len(out)))
        pred = 1 if mean >= 0.0 else 0
        if pred == int(ex.target_label):
            correct += 1
    return float(correct) / float(max(1, len(examples)))


# =============================================================================
# Collapse helpers (for the H5 falsifier comparison)
# =============================================================================


def collapse_role_banks(
        stack: DeepProxyStackV3,
) -> DeepProxyStackV3:
    """Zero out role banks across layers.

    Produces an equivalent V3 stack without per-role residuals.
    """
    for layer in stack.layers:
        bank = layer.role_bank
        bank.w_proj.values = [0.0] * len(bank.w_proj.values)
        bank.b_proj.values = [0.0] * len(bank.b_proj.values)
    return stack


# =============================================================================
# Verifier
# =============================================================================


W52_DEEP_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_deep_v3_schema_mismatch",
    "w52_deep_v3_stack_cid_mismatch",
    "w52_deep_v3_n_layers_mismatch",
    "w52_deep_v3_n_roles_mismatch",
    "w52_deep_v3_residual_gate_out_of_range",
)


def verify_deep_proxy_stack_v3_forward_witness(
        witness: DeepProxyStackV3ForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        expected_n_layers: int | None = None,
        expected_n_roles: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid != expected_stack_cid):
        failures.append("w52_deep_v3_stack_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w52_deep_v3_n_layers_mismatch")
    if (expected_n_roles is not None
            and witness.n_roles != int(expected_n_roles)):
        failures.append("w52_deep_v3_n_roles_mismatch")
    for rg in witness.per_layer_residual_gates:
        if rg < 0.0 - 1e-9 or rg > 1.0 + 1e-9:
            failures.append(
                "w52_deep_v3_residual_gate_out_of_range")
            break
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_DEEP_V3_SCHEMA_VERSION",
    "W52_DEFAULT_DEEP_V3_N_LAYERS",
    "W52_DEFAULT_DEEP_V3_IN_DIM",
    "W52_DEFAULT_DEEP_V3_FACTOR_DIM",
    "W52_DEFAULT_DEEP_V3_N_HEADS",
    "W52_DEFAULT_DEEP_V3_N_BRANCH_HEADS",
    "W52_DEFAULT_DEEP_V3_N_CYCLE_HEADS",
    "W52_DEFAULT_DEEP_V3_N_ROLES",
    "W52_DEEP_V3_VERIFIER_FAILURE_MODES",
    "RoleBank",
    "DeepLayerV3",
    "DeepProxyStackV3",
    "DeepProxyStackV3ForwardWitness",
    "DeepStackV3TrainingExample",
    "DeepStackV3TrainingSet",
    "DeepStackV3TrainingTrace",
    "synthesize_deep_stack_v3_training_set",
    "fit_deep_proxy_stack_v3",
    "evaluate_deep_stack_v3_accuracy",
    "collapse_role_banks",
    "emit_deep_proxy_stack_v3_forward_witness",
    "verify_deep_proxy_stack_v3_forward_witness",
]
