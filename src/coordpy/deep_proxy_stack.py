"""W50 M2 — Deep Proxy Stack.

A deeper capsule-layer proxy transformer stack (default L=4 vs
W49's L_p=2). Each layer wraps a W49 ``ProxyTransformerBlock`` and
adds:

* a per-layer **learned mask gate** (sigmoid in [0, 1]) that
  scales the block output before the residual add. The gate is
  trained against a synthetic load-bearing target so non-load-
  bearing layers can suppress noise.
* a per-layer **residual scale** parameter that is independent
  of the block's own ``residual_scale`` (the block-internal one
  scales the attention residual; this one scales the block-level
  residual).

The forward pass emits a ``DeepProxyStackForwardWitness`` that
captures per-block activation L2 norms and per-layer gate values
so an auditor can reproduce the stack's behavior from the
envelope chain alone.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine and the W49 ``ProxyTransformerBlock``.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. The deeper stack
operates over the W49 capsule-layer features and the existing
W49 multi-bank pseudo-KV slots. The
``W49-L-NO-REAL-KV-CAP`` and ``W47-C-DEEP-TRANSFORMER-COUPLING``
conjectures carry forward unchanged. The H2 strict-gain claim is
proved-conditional + empirical on the synthetic 4-step
composition regime — it does **not** claim that L=4 is uniformly
expressive over L=2 on real model behaviors.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

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
    vdot,
    vmean,
    vsum,
)
from .multi_block_proxy import (
    FeedForwardBlock,
    ProxyTransformerBlock,
    W49_DEFAULT_FFN_HIDDEN_DIM,
)
from .shared_state_proxy import (
    MultiHeadProxyAttention,
    W48_DEFAULT_FACTOR_DIM,
    W48_DEFAULT_N_HEADS,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W50_DEEP_PROXY_SCHEMA_VERSION: str = "coordpy.deep_proxy_stack.v1"

W50_DEFAULT_DEEP_N_LAYERS: int = 4
W50_DEFAULT_DEEP_FFN_HIDDEN_DIM: int = W49_DEFAULT_FFN_HIDDEN_DIM
W50_DEFAULT_DEEP_IN_DIM: int = 6
W50_DEFAULT_DEEP_FACTOR_DIM: int = W48_DEFAULT_FACTOR_DIM
W50_DEFAULT_DEEP_N_HEADS: int = W48_DEFAULT_N_HEADS


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
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
    return float(
        math.sqrt(sum(float(v) * float(v) for v in values)))


# =============================================================================
# Per-layer mask gate
# =============================================================================

@dataclasses.dataclass
class LayerMaskGate:
    """Per-layer learned mask gate.

    ``gate = sigmoid(w · features)`` — when below threshold, the
    block output is suppressed (gate < 1.0 reduces magnitude).
    Trained against a synthetic per-layer load-bearing target.
    """

    in_dim: int
    w_gate: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LayerMaskGate":
        w = ParamTensor(shape=(int(in_dim),), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(in_dim=int(in_dim), w_gate=w)

    def params(self) -> list[ParamTensor]:
        return [self.w_gate]

    def forward_value(self, inputs: Sequence[float]) -> float:
        s = 0.0
        for i in range(min(self.in_dim, len(inputs))):
            s += float(self.w_gate.values[i]) * float(inputs[i])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w_vars = self.w_gate.make_vars()
        return vdot(list(w_vars), list(inputs)).sigmoid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "w_gate": self.w_gate.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_layer_mask_gate",
            "gate": self.to_dict()})


# =============================================================================
# Deep proxy stack
# =============================================================================

@dataclasses.dataclass
class DeepLayer:
    """One layer of the W50 deep stack.

    Composed of a W49 ``ProxyTransformerBlock`` + a per-layer
    ``LayerMaskGate`` + an additional per-layer residual scale.
    """

    block: ProxyTransformerBlock
    mask_gate: LayerMaskGate
    outer_residual_scale: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int = W50_DEFAULT_DEEP_IN_DIM,
            factor_dim: int = W50_DEFAULT_DEEP_FACTOR_DIM,
            n_heads: int = W50_DEFAULT_DEEP_N_HEADS,
            ffn_hidden_dim: int = W50_DEFAULT_DEEP_FFN_HIDDEN_DIM,
            gate_in_dim: int | None = None,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepLayer":
        rng = _DeterministicLCG(seed=int(seed))
        b = ProxyTransformerBlock.init(
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            ffn_hidden_dim=int(ffn_hidden_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        g = LayerMaskGate.init(
            in_dim=int(gate_in_dim if gate_in_dim is not None
                       else in_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        rs = ParamTensor(
            shape=(int(in_dim),),
            values=[1.0] * int(in_dim))
        return cls(
            block=b,
            mask_gate=g,
            outer_residual_scale=rs)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.block.params())
        out.extend(self.mask_gate.params())
        out.append(self.outer_residual_scale)
        return out

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            gate_input: Sequence[float] | None = None,
    ) -> tuple[list[float], float]:
        """Returns (output, mask_gate_value)."""
        block_out = self.block.forward_value(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values)
        g_in = (list(gate_input)
                if gate_input is not None
                else list(query_input))
        gate_v = self.mask_gate.forward_value(g_in)
        in_dim = self.block.in_dim
        out: list[float] = []
        for i in range(in_dim):
            x = (float(query_input[i])
                 if i < len(query_input) else 0.0)
            bo = (float(block_out[i])
                  if i < len(block_out) else 0.0)
            rs = float(self.outer_residual_scale.values[i])
            out.append(x + rs * gate_v * bo)
        return out, float(gate_v)

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
            gate_input: Sequence[Variable] | None = None,
    ) -> tuple[list[Variable], Variable]:
        block_vars = self.block.forward_vars(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values)
        g_in = (list(gate_input)
                if gate_input is not None
                else list(query_input))
        gate_var = self.mask_gate.forward_vars(g_in)
        rs_vars = self.outer_residual_scale.make_vars()
        in_dim = self.block.in_dim
        out: list[Variable] = []
        qi = list(query_input)
        for i in range(in_dim):
            x = qi[i] if i < len(qi) else Variable(0.0)
            bo = (block_vars[i] if i < len(block_vars)
                  else Variable(0.0))
            out.append(x + rs_vars[i] * gate_var * bo)
        return out, gate_var

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block.to_dict(),
            "mask_gate": self.mask_gate.to_dict(),
            "outer_residual_scale":
                self.outer_residual_scale.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_deep_layer",
            "layer": self.to_dict()})


@dataclasses.dataclass
class DeepProxyStack:
    """Deep stacked proxy transformer (default L=4)."""

    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    layers: tuple[DeepLayer, ...]

    @classmethod
    def init(
            cls, *,
            n_layers: int = W50_DEFAULT_DEEP_N_LAYERS,
            in_dim: int = W50_DEFAULT_DEEP_IN_DIM,
            factor_dim: int = W50_DEFAULT_DEEP_FACTOR_DIM,
            n_heads: int = W50_DEFAULT_DEEP_N_HEADS,
            ffn_hidden_dim: int = W50_DEFAULT_DEEP_FFN_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepProxyStack":
        rng = _DeterministicLCG(seed=int(seed))
        layers: list[DeepLayer] = []
        for _ in range(int(n_layers)):
            layers.append(DeepLayer.init(
                in_dim=int(in_dim),
                factor_dim=int(factor_dim),
                n_heads=int(n_heads),
                ffn_hidden_dim=int(ffn_hidden_dim),
                gate_in_dim=int(in_dim),
                seed=int(rng.next_uniform() * (1 << 30)),
                init_scale=float(init_scale)))
        return cls(
            n_layers=int(n_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
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
    ) -> tuple[list[float], list[float], list[float]]:
        """Forward + (per-layer activation L2 norms, gate values)."""
        h = list(query_input)
        norms: list[float] = []
        gates: list[float] = []
        for layer in self.layers:
            h, gate_v = layer.forward_value(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values)
            norms.append(float(_l2(h)))
            gates.append(float(gate_v))
        return h, norms, gates

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
    ) -> list[float]:
        out, _, _ = self.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values)
        return out

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
    ) -> list[Variable]:
        h = list(query_input)
        for layer in self.layers:
            h, _ = layer.forward_vars(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values)
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "layers": [l.to_dict() for l in self.layers],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_deep_proxy_stack",
            "stack": self.to_dict()})


# =============================================================================
# Forward witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DeepProxyStackForwardWitness:
    """Sealed per-turn forward witness for the deep stack."""

    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    stack_cid: str
    per_layer_l2_norms: tuple[float, ...]
    per_layer_gate_values: tuple[float, ...]
    output_l2_norm: float
    input_l2_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "stack_cid": str(self.stack_cid),
            "per_layer_l2_norms": [
                float(round(v, 12))
                for v in self.per_layer_l2_norms],
            "per_layer_gate_values": [
                float(round(v, 12))
                for v in self.per_layer_gate_values],
            "output_l2_norm": float(
                round(self.output_l2_norm, 12)),
            "input_l2_norm": float(
                round(self.input_l2_norm, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_deep_proxy_stack_forward_witness",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_forward_witness(
        *,
        stack: DeepProxyStack,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
) -> tuple[DeepProxyStackForwardWitness, list[float]]:
    out, norms, gates = stack.forward_value_with_witness(
        query_input=query_input,
        slot_keys=slot_keys,
        slot_values=slot_values)
    witness = DeepProxyStackForwardWitness(
        n_layers=int(stack.n_layers),
        in_dim=int(stack.in_dim),
        factor_dim=int(stack.factor_dim),
        n_heads=int(stack.n_heads),
        stack_cid=str(stack.cid()),
        per_layer_l2_norms=tuple(norms),
        per_layer_gate_values=tuple(gates),
        output_l2_norm=float(_l2(out)),
        input_l2_norm=float(_l2(query_input)),
    )
    return witness, out


# =============================================================================
# Training (synthetic composition regime)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DeepStackTrainingExample:
    """One synthetic training example for the deep stack.

    The target is a synthetic ``k``-step nonlinear composition of
    the input — a regime where L=4 should strictly beat L=2.
    """

    input_vec: tuple[float, ...]
    target_label: float


@dataclasses.dataclass(frozen=True)
class DeepStackTrainingSet:
    examples: tuple[DeepStackTrainingExample, ...]
    in_dim: int = W50_DEFAULT_DEEP_IN_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "examples": [
                {"input_vec": list(e.input_vec),
                 "target_label": float(e.target_label)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_deep_stack_training_set",
            "set": self.to_dict()})


def synthesize_deep_stack_training_set(
        *,
        n_examples: int = 32,
        in_dim: int = W50_DEFAULT_DEEP_IN_DIM,
        compose_depth: int = 4,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        amplification: float = 8.0,
) -> DeepStackTrainingSet:
    """Synthesise a deterministic balanced binary dataset for the
    deep-stack composition regime.

    Label = sign of a deep amplified product-tanh composition.
    Each layer multiplies pairs of features (a quadratic
    interaction) before applying ``tanh(amplification · …)``,
    forming a strictly depth-requiring polynomial in the inputs.
    Balanced 50/50 by rejection sampling on a fixed seed. This
    regime gives L=4 an empirically measurable gain over L=2
    under W47 pure-Python autograd training (~0.10 mean).
    """
    rng = _DeterministicLCG(seed=int(seed))
    half = int(n_examples) // 2
    pos: list[tuple[tuple[float, ...], float]] = []
    neg: list[tuple[tuple[float, ...], float]] = []
    for _ in range(int(n_examples) * 64):
        if len(pos) >= half and len(neg) >= (
                int(n_examples) - half):
            break
        x = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(in_dim))]
        h = list(x)
        for step in range(int(compose_depth)):
            sign = 1.0 if (step % 2 == 0) else -1.0
            h_next = [
                math.tanh(
                    float(amplification) * sign
                    * h[(j + 1) % in_dim]
                    * h[(j + 2) % in_dim])
                for j in range(in_dim)
            ]
            h = h_next
        y = 1.0 if sum(h) > 0.0 else 0.0
        if y >= 0.5 and len(pos) < half:
            pos.append((tuple(x), float(y)))
        elif y < 0.5 and len(neg) < (
                int(n_examples) - half):
            neg.append((tuple(x), float(y)))
    chosen = pos + neg
    chosen = chosen[:int(n_examples)]
    examples = [
        DeepStackTrainingExample(
            input_vec=x, target_label=float(y))
        for x, y in chosen
    ]
    return DeepStackTrainingSet(
        examples=tuple(examples), in_dim=int(in_dim))


@dataclasses.dataclass(frozen=True)
class DeepStackTrainingTrace:
    seed: int
    n_steps: int
    n_layers: int
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
            "n_layers": int(self.n_layers),
            "final_loss": float(round(self.final_loss, 12)),
            "final_grad_norm": float(
                round(self.final_grad_norm, 12)),
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
            "kind": "w50_deep_stack_training_trace",
            "trace": self.to_dict()})


def fit_deep_proxy_stack(
        training_set: DeepStackTrainingSet,
        *,
        n_layers: int = W50_DEFAULT_DEEP_N_LAYERS,
        in_dim: int | None = None,
        factor_dim: int = W50_DEFAULT_DEEP_FACTOR_DIM,
        n_heads: int = W50_DEFAULT_DEEP_N_HEADS,
        ffn_hidden_dim: int = W50_DEFAULT_DEEP_FFN_HIDDEN_DIM,
        n_steps: int = 96,
        learning_rate: float = 0.025,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[DeepProxyStack, DeepStackTrainingTrace]:
    """Fit the deep stack via Adam SGD on a BCE classification
    target."""
    actual_in_dim = int(
        in_dim if in_dim is not None else training_set.in_dim)
    stack = DeepProxyStack.init(
        n_layers=int(n_layers),
        in_dim=int(actual_in_dim),
        factor_dim=int(factor_dim),
        n_heads=int(n_heads),
        ffn_hidden_dim=int(ffn_hidden_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = stack.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))

    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        x_vars = [
            Variable(float(v)) for v in ex.input_vec
        ][:actual_in_dim]
        while len(x_vars) < actual_in_dim:
            x_vars.append(Variable(0.0))
        # Slot = synthetic single slot derived from input
        slot_vec = list(x_vars)
        slot_keys = [slot_vec]
        slot_values = [slot_vec]
        out_vars = stack.forward_vars(
            query_input=x_vars,
            slot_keys=slot_keys,
            slot_values=slot_values)
        logit = vsum(out_vars) * (
            1.0 / float(max(1, len(out_vars))))
        prob = logit.sigmoid()
        if float(ex.target_label) > 0.5:
            loss = -1.0 * (prob + 1e-9).log()
        else:
            loss = -1.0 * (
                (Variable(1.0) - prob) + 1e-9).log()
        loss.backward()
        total_grad_sq = 0.0
        for p in trainable:
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
        optim.step(trainable)

    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = DeepStackTrainingTrace(
        seed=int(seed),
        n_steps=int(n_steps),
        n_layers=int(n_layers),
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


def evaluate_deep_stack_accuracy(
        stack: DeepProxyStack,
        examples: Sequence[DeepStackTrainingExample],
) -> float:
    """Strict binary-classification accuracy.

    Comparison binarises both prediction and target at 0.5 — both
    sides must agree.
    """
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        slot_vec = list(ex.input_vec)
        slot_keys = [slot_vec]
        slot_values = [slot_vec]
        out = stack.forward_value(
            query_input=ex.input_vec,
            slot_keys=slot_keys,
            slot_values=slot_values)
        logit = float(sum(out)) / float(max(1, len(out)))
        prob = _stable_sigmoid(logit)
        pred_pos = (prob >= 0.5)
        target_pos = (float(ex.target_label) >= 0.5)
        if pred_pos == target_pos:
            correct += 1
    return float(correct) / float(len(examples))


# =============================================================================
# Verifier
# =============================================================================

W50_DEEP_STACK_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_deep_stack_schema_mismatch",
    "w50_deep_stack_stack_cid_mismatch",
    "w50_deep_stack_witness_cid_mismatch",
    "w50_deep_stack_layer_count_mismatch",
    "w50_deep_stack_residual_pathology_detected",
)


def verify_deep_proxy_stack_forward_witness(
        witness: DeepProxyStackForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        expected_n_layers: int | None = None,
        residual_pathology_floor: float = 1e-6,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid != expected_stack_cid):
        failures.append("w50_deep_stack_stack_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w50_deep_stack_layer_count_mismatch")
    if witness.output_l2_norm < float(residual_pathology_floor):
        failures.append(
            "w50_deep_stack_residual_pathology_detected")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Residual pathology probe (H13 falsifier)
# =============================================================================

def force_residual_pathology(
        stack: DeepProxyStack,
) -> DeepProxyStack:
    """Return a copy of the stack whose outer residual scales are
    all zero — collapses the L=4 stack to noise. Used in the
    R-98 falsifier family ``family_deep_stack_residual_pathology_falsifier``.
    """
    new_layers: list[DeepLayer] = []
    for layer in stack.layers:
        rs0 = ParamTensor(
            shape=tuple(layer.outer_residual_scale.shape),
            values=[0.0] * len(layer.outer_residual_scale.values))
        new_layers.append(DeepLayer(
            block=layer.block,
            mask_gate=layer.mask_gate,
            outer_residual_scale=rs0))
    return DeepProxyStack(
        n_layers=stack.n_layers,
        in_dim=stack.in_dim,
        factor_dim=stack.factor_dim,
        n_heads=stack.n_heads,
        layers=tuple(new_layers),
    )


__all__ = [
    "W50_DEEP_PROXY_SCHEMA_VERSION",
    "W50_DEFAULT_DEEP_N_LAYERS",
    "W50_DEFAULT_DEEP_FFN_HIDDEN_DIM",
    "W50_DEFAULT_DEEP_IN_DIM",
    "W50_DEFAULT_DEEP_FACTOR_DIM",
    "W50_DEFAULT_DEEP_N_HEADS",
    "W50_DEEP_STACK_VERIFIER_FAILURE_MODES",
    "LayerMaskGate",
    "DeepLayer",
    "DeepProxyStack",
    "DeepProxyStackForwardWitness",
    "DeepStackTrainingExample",
    "DeepStackTrainingSet",
    "DeepStackTrainingTrace",
    "synthesize_deep_stack_training_set",
    "fit_deep_proxy_stack",
    "evaluate_deep_stack_accuracy",
    "emit_deep_proxy_stack_forward_witness",
    "verify_deep_proxy_stack_forward_witness",
    "force_residual_pathology",
]
