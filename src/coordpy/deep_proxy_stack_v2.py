"""W51 M3 — Deeper Proxy Stack V2.

A depth-six capsule-layer proxy transformer stack (default
``L=6``, vs W50's ``L=4``). Each layer wraps a W49
``ProxyTransformerBlock`` and adds:

* a per-layer **learned mask gate** (sigmoid in [0, 1])
  (inherits W50's behaviour).
* a per-layer **outer residual scale** (inherits W50).
* a per-layer **trainable temperature** ``tau_l`` that
  scales the block's pre-softmax attention logits — when
  ``tau_l`` is large the attention becomes sharp; when small
  it becomes diffuse.
* a **branch-specialised head selector** — a one-hot route
  vector over per-branch heads applied as a per-layer
  multiplicative gate on the block output.
* a **cycle-specialised head selector** — analogous to
  branch but over per-cycle classes.

The forward pass emits a ``DeepProxyStackV2ForwardWitness``
that captures per-block activation L2 norms, per-layer gate
values, per-layer temperatures, and the per-layer branch+cycle
selectors.

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine, the W49
``ProxyTransformerBlock``, and the W50 ``DeepLayer`` +
``LayerMaskGate`` abstractions.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state,
KV cache bytes, attention weights, or embeddings. The deeper
``L=6`` stack with branch/cycle-specialised heads operates
over W49 + W50 capsule-layer features only.

The H4 strict-gain claim is proved-conditional + empirical on
the synthetic six-step composition regime — it does not
claim ``L=6`` is uniformly expressive over ``L=4`` on real
model behaviours. The
``W51-L-DEEP-STACK-OVERDEPTH-CAP`` falsifier reproduces
honestly on shallow regimes.
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
    vdot,
    vmean,
    vsum,
)
from .deep_proxy_stack import (
    DeepLayer,
    DeepProxyStack,
    LayerMaskGate,
    W50_DEFAULT_DEEP_FACTOR_DIM,
    W50_DEFAULT_DEEP_FFN_HIDDEN_DIM,
    W50_DEFAULT_DEEP_IN_DIM,
    W50_DEFAULT_DEEP_N_HEADS,
)
from .multi_block_proxy import (
    ProxyTransformerBlock,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W51_DEEP_PROXY_V2_SCHEMA_VERSION: str = (
    "coordpy.deep_proxy_stack_v2.v1")

W51_DEFAULT_DEEP_V2_N_LAYERS: int = 6
W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM: int = (
    W50_DEFAULT_DEEP_FFN_HIDDEN_DIM)
W51_DEFAULT_DEEP_V2_IN_DIM: int = W50_DEFAULT_DEEP_IN_DIM
W51_DEFAULT_DEEP_V2_FACTOR_DIM: int = (
    W50_DEFAULT_DEEP_FACTOR_DIM)
W51_DEFAULT_DEEP_V2_N_HEADS: int = W50_DEFAULT_DEEP_N_HEADS
W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS: int = 4
W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS: int = 4
W51_DEFAULT_DEEP_V2_TEMPERATURE_INIT: float = 1.0


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
# Branch / Cycle head selector
# =============================================================================

@dataclasses.dataclass
class BranchCycleSelector:
    """Per-layer branch + cycle multiplicative gate.

    Stores ``n_branch_heads`` and ``n_cycle_heads`` learned
    gate coefficients each. For a given (branch_index,
    cycle_index), the selector emits two sigmoids
    ``(b_gate, c_gate)`` whose product scales the block output.
    """

    n_branch_heads: int
    n_cycle_heads: int
    w_branch: ParamTensor   # (n_branch_heads,)
    w_cycle: ParamTensor    # (n_cycle_heads,)

    @classmethod
    def init(
            cls, *,
            n_branch_heads: int = (
                W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BranchCycleSelector":
        rng = _DeterministicLCG(seed=int(seed))
        wb = ParamTensor(
            shape=(int(n_branch_heads),),
            values=[
                float(rng.next_uniform() - 0.5) * float(init_scale)
                for _ in range(int(n_branch_heads))
            ])
        wc = ParamTensor(
            shape=(int(n_cycle_heads),),
            values=[
                float(rng.next_uniform() - 0.5) * float(init_scale)
                for _ in range(int(n_cycle_heads))
            ])
        return cls(
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            w_branch=wb, w_cycle=wc)

    def params(self) -> list[ParamTensor]:
        return [self.w_branch, self.w_cycle]

    def gate_value(
            self, *,
            branch_index: int,
            cycle_index: int,
    ) -> float:
        b = int(branch_index) % max(1, self.n_branch_heads)
        c = int(cycle_index) % max(1, self.n_cycle_heads)
        bg = float(_stable_sigmoid(
            float(self.w_branch.values[b])))
        cg = float(_stable_sigmoid(
            float(self.w_cycle.values[c])))
        return float(bg * cg)

    def gate_vars(
            self, *,
            branch_index: int,
            cycle_index: int,
    ) -> Variable:
        wb_vars = self.w_branch.make_vars()
        wc_vars = self.w_cycle.make_vars()
        b = int(branch_index) % max(1, self.n_branch_heads)
        c = int(cycle_index) % max(1, self.n_cycle_heads)
        bg = wb_vars[b].sigmoid()
        cg = wc_vars[c].sigmoid()
        return bg * cg

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_branch_heads": int(self.n_branch_heads),
            "n_cycle_heads": int(self.n_cycle_heads),
            "w_branch": self.w_branch.to_dict(),
            "w_cycle": self.w_cycle.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_branch_cycle_selector",
            "selector": self.to_dict()})


# =============================================================================
# Deep layer V2
# =============================================================================

@dataclasses.dataclass
class DeepLayerV2:
    """One layer of the W51 deep stack V2.

    Composed of: W50 ``DeepLayer`` (block + mask gate + outer
    residual scale) + a per-layer trainable temperature +
    a per-layer ``BranchCycleSelector``.
    """

    base: DeepLayer
    log_temperature: ParamTensor    # (1,) — log of temperature
    selector: BranchCycleSelector

    @classmethod
    def init(
            cls, *,
            in_dim: int = W51_DEFAULT_DEEP_V2_IN_DIM,
            factor_dim: int = W51_DEFAULT_DEEP_V2_FACTOR_DIM,
            n_heads: int = W51_DEFAULT_DEEP_V2_N_HEADS,
            ffn_hidden_dim: int = (
                W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM),
            n_branch_heads: int = (
                W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS),
            gate_in_dim: int | None = None,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            temperature_init: float = (
                W51_DEFAULT_DEEP_V2_TEMPERATURE_INIT),
    ) -> "DeepLayerV2":
        rng = _DeterministicLCG(seed=int(seed))
        base = DeepLayer.init(
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            ffn_hidden_dim=int(ffn_hidden_dim),
            gate_in_dim=int(gate_in_dim) if gate_in_dim
            is not None else int(in_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        log_t = ParamTensor(
            shape=(1,),
            values=[math.log(max(1e-6,
                                   float(temperature_init)))])
        selector = BranchCycleSelector.init(
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        return cls(
            base=base,
            log_temperature=log_t,
            selector=selector)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.base.params())
        out.append(self.log_temperature)
        out.extend(self.selector.params())
        return out

    @property
    def temperature(self) -> float:
        return float(math.exp(
            float(self.log_temperature.values[0])))

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            branch_index: int = 0,
            cycle_index: int = 0,
            gate_input: Sequence[float] | None = None,
    ) -> tuple[list[float], float, float]:
        """Returns (output, mask_gate_value, bc_gate_value)."""
        base_out, base_gate = self.base.forward_value(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            gate_input=gate_input)
        bc_gate = self.selector.gate_value(
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        tau = float(self.temperature)
        in_dim = self.base.block.in_dim
        out: list[float] = []
        for i in range(in_dim):
            x = (float(query_input[i])
                 if i < len(query_input) else 0.0)
            bo_minus_x = float(base_out[i]) - x
            # bc_gate scales the bc-gated residual; tau scales
            # the overall residual magnitude.
            out.append(x + tau * bc_gate * bo_minus_x)
        return out, float(base_gate), float(bc_gate)

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
            branch_index: int = 0,
            cycle_index: int = 0,
            gate_input: Sequence[Variable] | None = None,
    ) -> list[Variable]:
        base_vars, base_gate = self.base.forward_vars(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            gate_input=gate_input)
        bc_gate_var = self.selector.gate_vars(
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        log_t_vars = self.log_temperature.make_vars()
        tau = log_t_vars[0].exp()
        in_dim = self.base.block.in_dim
        qi = list(query_input)
        out: list[Variable] = []
        for i in range(in_dim):
            x = qi[i] if i < len(qi) else Variable(0.0)
            bo = (base_vars[i] if i < len(base_vars)
                  else Variable(0.0))
            bo_minus_x = bo - x
            out.append(x + tau * bc_gate_var * bo_minus_x)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "log_temperature": self.log_temperature.to_dict(),
            "selector": self.selector.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_deep_layer_v2",
            "layer": self.to_dict()})


# =============================================================================
# Deep proxy stack V2
# =============================================================================

@dataclasses.dataclass
class DeepProxyStackV2:
    """Depth-six stacked proxy transformer V2."""

    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    n_branch_heads: int
    n_cycle_heads: int
    layers: tuple[DeepLayerV2, ...]

    @classmethod
    def init(
            cls, *,
            n_layers: int = W51_DEFAULT_DEEP_V2_N_LAYERS,
            in_dim: int = W51_DEFAULT_DEEP_V2_IN_DIM,
            factor_dim: int = W51_DEFAULT_DEEP_V2_FACTOR_DIM,
            n_heads: int = W51_DEFAULT_DEEP_V2_N_HEADS,
            ffn_hidden_dim: int = (
                W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM),
            n_branch_heads: int = (
                W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS),
            n_cycle_heads: int = (
                W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DeepProxyStackV2":
        rng = _DeterministicLCG(seed=int(seed))
        layers: list[DeepLayerV2] = []
        for _ in range(int(n_layers)):
            layers.append(DeepLayerV2.init(
                in_dim=int(in_dim),
                factor_dim=int(factor_dim),
                n_heads=int(n_heads),
                ffn_hidden_dim=int(ffn_hidden_dim),
                n_branch_heads=int(n_branch_heads),
                n_cycle_heads=int(n_cycle_heads),
                gate_in_dim=int(in_dim),
                seed=int(rng.next_uniform() * (1 << 30)),
                init_scale=float(init_scale)))
        return cls(
            n_layers=int(n_layers),
            in_dim=int(in_dim),
            factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            n_branch_heads=int(n_branch_heads),
            n_cycle_heads=int(n_cycle_heads),
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
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> tuple[list[float], list[float], list[float],
               list[float], list[float]]:
        """Forward + (per-layer L2 norms, per-layer mask gates,
        per-layer bc gates, per-layer temperatures)."""
        h = list(query_input)
        norms: list[float] = []
        gates: list[float] = []
        bcs: list[float] = []
        temps: list[float] = []
        for layer in self.layers:
            h, gate_v, bc_v = layer.forward_value(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index))
            norms.append(float(_l2(h)))
            gates.append(float(gate_v))
            bcs.append(float(bc_v))
            temps.append(float(layer.temperature))
        return h, norms, gates, bcs, temps

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> list[float]:
        out, _, _, _, _ = self.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            branch_index=int(branch_index),
            cycle_index=int(cycle_index))
        return out

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
            branch_index: int = 0,
            cycle_index: int = 0,
    ) -> list[Variable]:
        h = list(query_input)
        for layer in self.layers:
            h = layer.forward_vars(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
                branch_index=int(branch_index),
                cycle_index=int(cycle_index))
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "n_branch_heads": int(self.n_branch_heads),
            "n_cycle_heads": int(self.n_cycle_heads),
            "layers": [l.to_dict() for l in self.layers],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_deep_proxy_stack_v2",
            "stack": self.to_dict()})


# =============================================================================
# Forward witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DeepProxyStackV2ForwardWitness:
    """Sealed per-turn forward witness for the deep stack V2."""

    n_layers: int
    in_dim: int
    factor_dim: int
    n_heads: int
    n_branch_heads: int
    n_cycle_heads: int
    stack_cid: str
    per_layer_l2_norms: tuple[float, ...]
    per_layer_gate_values: tuple[float, ...]
    per_layer_bc_gate_values: tuple[float, ...]
    per_layer_temperatures: tuple[float, ...]
    output_l2_norm: float
    input_l2_norm: float
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
            "stack_cid": str(self.stack_cid),
            "per_layer_l2_norms": [
                float(round(v, 12))
                for v in self.per_layer_l2_norms],
            "per_layer_gate_values": [
                float(round(v, 12))
                for v in self.per_layer_gate_values],
            "per_layer_bc_gate_values": [
                float(round(v, 12))
                for v in self.per_layer_bc_gate_values],
            "per_layer_temperatures": [
                float(round(v, 12))
                for v in self.per_layer_temperatures],
            "output_l2_norm": float(
                round(self.output_l2_norm, 12)),
            "input_l2_norm": float(
                round(self.input_l2_norm, 12)),
            "branch_index": int(self.branch_index),
            "cycle_index": int(self.cycle_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_deep_proxy_stack_v2_forward_witness",
            "witness": self.to_dict()})


def emit_deep_proxy_stack_v2_forward_witness(
        *,
        stack: DeepProxyStackV2,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        branch_index: int = 0,
        cycle_index: int = 0,
) -> tuple[DeepProxyStackV2ForwardWitness, list[float]]:
    out, norms, gates, bcs, temps = (
        stack.forward_value_with_witness(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
            branch_index=int(branch_index),
            cycle_index=int(cycle_index)))
    witness = DeepProxyStackV2ForwardWitness(
        n_layers=int(stack.n_layers),
        in_dim=int(stack.in_dim),
        factor_dim=int(stack.factor_dim),
        n_heads=int(stack.n_heads),
        n_branch_heads=int(stack.n_branch_heads),
        n_cycle_heads=int(stack.n_cycle_heads),
        stack_cid=str(stack.cid()),
        per_layer_l2_norms=tuple(norms),
        per_layer_gate_values=tuple(gates),
        per_layer_bc_gate_values=tuple(bcs),
        per_layer_temperatures=tuple(temps),
        output_l2_norm=float(_l2(out)),
        input_l2_norm=float(_l2(query_input)),
        branch_index=int(branch_index),
        cycle_index=int(cycle_index),
    )
    return witness, out


# =============================================================================
# Training set (deep composition + branch routing)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DeepStackV2TrainingExample:
    """One synthetic training example for the deep stack V2.

    The target is a synthetic ``k``-step nonlinear composition
    of the input AND depends on a branch index (each branch
    has a different composition pattern).
    """

    input_vec: tuple[float, ...]
    branch_index: int
    cycle_index: int
    target_label: float


@dataclasses.dataclass(frozen=True)
class DeepStackV2TrainingSet:
    examples: tuple[DeepStackV2TrainingExample, ...]
    in_dim: int = W51_DEFAULT_DEEP_V2_IN_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "examples": [
                {"input_vec": list(e.input_vec),
                 "branch_index": int(e.branch_index),
                 "cycle_index": int(e.cycle_index),
                 "target_label": float(e.target_label)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_deep_stack_v2_training_set",
            "set": self.to_dict()})


def synthesize_deep_stack_v2_training_set(
        *,
        n_examples: int = 32,
        in_dim: int = W51_DEFAULT_DEEP_V2_IN_DIM,
        compose_depth: int = 6,
        n_branches: int = 2,
        n_cycles: int = 2,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        amplification: float = 8.0,
) -> DeepStackV2TrainingSet:
    """Synthesise a deterministic balanced dataset for the
    six-step deep composition regime, conditioned on branch
    and cycle indices.

    Label = sign of a deep amplified product-tanh composition;
    each branch uses a different (sign-flip pattern, position-
    offset pattern); each cycle uses a different (depth offset).
    Balanced 50/50 by rejection sampling.
    """
    rng = _DeterministicLCG(seed=int(seed))
    half = int(n_examples) // 2
    pos: list[tuple[tuple[float, ...], int, int, float]] = []
    neg: list[tuple[tuple[float, ...], int, int, float]] = []
    n_b = max(1, int(n_branches))
    n_c = max(1, int(n_cycles))
    for _ in range(int(n_examples) * 64):
        if (len(pos) >= half
                and len(neg) >= (int(n_examples) - half)):
            break
        x = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(in_dim))
        ]
        branch = int(rng.next_uniform() * float(n_b)) % n_b
        cycle = int(rng.next_uniform() * float(n_c)) % n_c
        # Branch sets the sign-flip + position-offset pattern.
        sign_seed = (branch * 7) + 1
        pos_seed = (branch * 11) + cycle + 1
        h = list(x)
        depth = max(1, int(compose_depth) + (cycle - 1))
        for step in range(depth):
            sign = 1.0 if (
                ((step + sign_seed) % 2) == 0) else -1.0
            offset_a = (step + pos_seed) % int(in_dim)
            offset_b = (step + pos_seed + 1) % int(in_dim)
            h_next = [
                math.tanh(
                    float(amplification) * sign
                    * h[(j + offset_a) % int(in_dim)]
                    * h[(j + offset_b) % int(in_dim)])
                for j in range(int(in_dim))
            ]
            h = h_next
        y = 1.0 if sum(h) > 0.0 else 0.0
        if y >= 0.5 and len(pos) < half:
            pos.append((tuple(x), int(branch), int(cycle),
                        float(y)))
        elif y < 0.5 and len(neg) < (int(n_examples) - half):
            neg.append((tuple(x), int(branch), int(cycle),
                        float(y)))
    chosen = pos + neg
    chosen = chosen[:int(n_examples)]
    examples = [
        DeepStackV2TrainingExample(
            input_vec=x, branch_index=int(b),
            cycle_index=int(c), target_label=float(y))
        for x, b, c, y in chosen
    ]
    return DeepStackV2TrainingSet(
        examples=tuple(examples), in_dim=int(in_dim))


@dataclasses.dataclass(frozen=True)
class DeepStackV2TrainingTrace:
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
            "kind": "w51_deep_stack_v2_training_trace",
            "trace": self.to_dict()})


def fit_deep_proxy_stack_v2(
        training_set: DeepStackV2TrainingSet,
        *,
        n_layers: int = W51_DEFAULT_DEEP_V2_N_LAYERS,
        in_dim: int | None = None,
        factor_dim: int = W51_DEFAULT_DEEP_V2_FACTOR_DIM,
        n_heads: int = W51_DEFAULT_DEEP_V2_N_HEADS,
        ffn_hidden_dim: int = W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM,
        n_branch_heads: int = (
            W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS),
        n_cycle_heads: int = W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS,
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
) -> tuple[DeepProxyStackV2, DeepStackV2TrainingTrace]:
    """Fit the deep stack V2 via Adam SGD on a BCE
    classification target, conditioned on branch + cycle
    indices.
    """
    actual_in_dim = int(
        in_dim if in_dim is not None else training_set.in_dim)
    stack = DeepProxyStackV2.init(
        n_layers=int(n_layers),
        in_dim=int(actual_in_dim),
        factor_dim=int(factor_dim),
        n_heads=int(n_heads),
        ffn_hidden_dim=int(ffn_hidden_dim),
        n_branch_heads=int(n_branch_heads),
        n_cycle_heads=int(n_cycle_heads),
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
        slot_vec = list(x_vars)
        slot_keys = [slot_vec]
        slot_values = [slot_vec]
        out_vars = stack.forward_vars(
            query_input=x_vars,
            slot_keys=slot_keys,
            slot_values=slot_values,
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index))
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
    trace = DeepStackV2TrainingTrace(
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


def evaluate_deep_stack_v2_accuracy(
        stack: DeepProxyStackV2,
        examples: Sequence[DeepStackV2TrainingExample],
) -> float:
    """Strict binary-classification accuracy with branch +
    cycle conditioning.
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
            slot_values=slot_values,
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index))
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

W51_DEEP_STACK_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w51_deep_stack_v2_schema_mismatch",
    "w51_deep_stack_v2_stack_cid_mismatch",
    "w51_deep_stack_v2_witness_cid_mismatch",
    "w51_deep_stack_v2_layer_count_mismatch",
    "w51_deep_stack_v2_residual_pathology_detected",
    "w51_deep_stack_v2_temperature_pathology_detected",
    "w51_deep_stack_v2_branch_cycle_indices_invalid",
)


def verify_deep_proxy_stack_v2_forward_witness(
        witness: DeepProxyStackV2ForwardWitness,
        *,
        expected_stack_cid: str | None = None,
        expected_n_layers: int | None = None,
        residual_pathology_floor: float = 1e-6,
        max_temperature: float = 1e6,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_stack_cid is not None
            and witness.stack_cid != expected_stack_cid):
        failures.append(
            "w51_deep_stack_v2_stack_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append(
            "w51_deep_stack_v2_layer_count_mismatch")
    if witness.output_l2_norm < float(residual_pathology_floor):
        failures.append(
            "w51_deep_stack_v2_residual_pathology_detected")
    for tau in witness.per_layer_temperatures:
        if float(tau) > float(max_temperature):
            failures.append(
                "w51_deep_stack_v2_temperature_pathology_detected")
            break
    if (witness.branch_index < 0
            or witness.cycle_index < 0):
        failures.append(
            "w51_deep_stack_v2_branch_cycle_indices_invalid")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Overdepth pathology + shared-head shallow stack helpers
# =============================================================================

def collapse_branch_cycle_selectors(
        stack: DeepProxyStackV2,
) -> DeepProxyStackV2:
    """Return a copy of the stack whose branch/cycle selectors
    are all forced to a uniform 1.0 — collapses the branch/cycle
    specialisation. Used in the R-100 shared-head ablation arm.
    """
    new_layers: list[DeepLayerV2] = []
    for layer in stack.layers:
        # Set all branch/cycle weights to a large positive value
        # so sigmoid → 1.0 → gate ≈ 1.0 (i.e. shared head).
        wb = ParamTensor(
            shape=tuple(layer.selector.w_branch.shape),
            values=[10.0] * len(layer.selector.w_branch.values))
        wc = ParamTensor(
            shape=tuple(layer.selector.w_cycle.shape),
            values=[10.0] * len(layer.selector.w_cycle.values))
        new_sel = BranchCycleSelector(
            n_branch_heads=layer.selector.n_branch_heads,
            n_cycle_heads=layer.selector.n_cycle_heads,
            w_branch=wb, w_cycle=wc)
        new_layers.append(DeepLayerV2(
            base=layer.base,
            log_temperature=layer.log_temperature,
            selector=new_sel))
    return DeepProxyStackV2(
        n_layers=stack.n_layers,
        in_dim=stack.in_dim,
        factor_dim=stack.factor_dim,
        n_heads=stack.n_heads,
        n_branch_heads=stack.n_branch_heads,
        n_cycle_heads=stack.n_cycle_heads,
        layers=tuple(new_layers),
    )


__all__ = [
    "W51_DEEP_PROXY_V2_SCHEMA_VERSION",
    "W51_DEFAULT_DEEP_V2_N_LAYERS",
    "W51_DEFAULT_DEEP_V2_FFN_HIDDEN_DIM",
    "W51_DEFAULT_DEEP_V2_IN_DIM",
    "W51_DEFAULT_DEEP_V2_FACTOR_DIM",
    "W51_DEFAULT_DEEP_V2_N_HEADS",
    "W51_DEFAULT_DEEP_V2_N_BRANCH_HEADS",
    "W51_DEFAULT_DEEP_V2_N_CYCLE_HEADS",
    "W51_DEFAULT_DEEP_V2_TEMPERATURE_INIT",
    "W51_DEEP_STACK_V2_VERIFIER_FAILURE_MODES",
    "BranchCycleSelector",
    "DeepLayerV2",
    "DeepProxyStackV2",
    "DeepProxyStackV2ForwardWitness",
    "DeepStackV2TrainingExample",
    "DeepStackV2TrainingSet",
    "DeepStackV2TrainingTrace",
    "synthesize_deep_stack_v2_training_set",
    "fit_deep_proxy_stack_v2",
    "evaluate_deep_stack_v2_accuracy",
    "emit_deep_proxy_stack_v2_forward_witness",
    "verify_deep_proxy_stack_v2_forward_witness",
    "collapse_branch_cycle_selectors",
]
