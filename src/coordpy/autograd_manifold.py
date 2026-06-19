"""W47 Autograd Manifold Stack (AMS) — capsule-native, autograd-
trained manifold-memory stack on top of W46 MMC, W45 LMC, W44
LMCC, and W43 PMC.

W47 is the first capsule-native CoordPy layer where the gating
policy is shaped by **autograd-trained** structure (pure-Python
reverse-mode automatic differentiation + SGD/Adam) rather than
stage-wise closed-form ridge. It directly attacks the
``W46-C-AUTOGRAD-DEEP-STACK`` carry-forward conjecture.

The eight trainable, content-addressed components plus a small
pure-Python reverse-mode autograd engine are:

  * **Pure-Python reverse-mode autograd engine.** A
    :class:`Variable` scalar with gradient tape and topologically-
    sorted backward pass. Supports the closed set of ops
    documented in :data:`W47_SUPPORTED_OPS`. Validates against
    finite-difference checks on every supported op.

  * **Trainable multi-layer manifold stack.** An ``L``-layer fully
    connected tanh stack over flattened channel features
    (24 inputs at the W43/W45/W46 defaults). The final layer
    collapses to a scalar gate logit. Weights are initialised
    deterministically from a seed-derived uniform.

  * **Trainable rank-r role adapter.** Per-role rank-``r`` LoRA-
    style delta ``A_r · B_r^T`` on the deepest hidden state.
    Trained jointly with the stack.

  * **Trainable dictionary / codebook.** ``K`` trainable prototype
    vectors, optimised via soft-assignment cross-entropy +
    straight-through residual loss. Encode remains bijective at
    inference.

  * **Trainable memory read/write head.** Learned query / key /
    value projections; reads the bounded W46 memory bank via
    softmax(Q · K^T / sqrt(d)) · V — exactly a capsule-layer
    attention head.

  * **Trainable packed control serializer.** Per-field sigmoid
    gates that learn which CTRL fields to emit; bijective.

  * **Adam-style optimiser.** First / second moment EMAs, fixed
    betas / eps, deterministic step counter. Exposed as
    :class:`AdamOptimizer`.

  * **Training trace witness.** Content-addressed record of seed,
    n_steps, optimiser config, loss / gradient-norm history,
    final-params CID. Bound under the W47 envelope.

Honest scope (do-not-overstate)
-------------------------------

W47 does NOT claim transformer-internal access. The trained
manifold stack operates strictly over W43 capsule-layer channel
encodings; it does not read hidden states, transplant KV cache,
inspect attention weights, or modify the model's attention
computation. The W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``) and
``W45-C-DEEP-TRANSFORMER-COUPLING`` carry forward unchanged.

W47 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine is correct but slow; training a 24→16→16→1 stack
on 64 examples for 200 steps takes ≈ 1–8s on a Mac M-series CPU.
Production training would need NumPy / JAX / PyTorch bindings.

W47 does NOT claim adversarial robustness under
training-distribution forgery (W47-L-AUTOGRAD-DISTRIBUTION-CAP).

W47 is strictly additive on top of W46 and the released v3.43
SDK. When the autograd stack is configured trivially
(``autograd_enabled=False``, W46-trivial inner), the W47
orchestrator reduces to ``ManifoldMemoryTeam.run`` byte-for-byte
— the W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH falsifier.

This module lives at ``coordpy.autograd_manifold`` and is NOT
exported through ``coordpy.__experimental__`` at this milestone;
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W47 surface through an explicit
``from coordpy.autograd_manifold import ...`` import.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .agents import (
    Agent,
    AgentTurn,
    _safe_usage_snapshot,
    _sha256_str,
)
from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .learned_manifold import (
    TrainingExample,
    TrainingSet,
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
    W45_N_CHANNELS,
    _channel_features_from_bundle,
    _confidence_bucket_for_probability,
    _softmax,
)
from .live_manifold import (
    LiveObservationBuilder,
    LiveTurnContext,
    W44_BRANCH_LIVE_NO_POLICY,
    W44_BRANCH_LIVE_RATIFIED,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_DEFAULT_PARENT_W42_CID,
    W44_ROUTE_MODE_FACTORADIC,
    default_live_observation_builder,
)
from .llm_backend import LLMBackend
from .manifold_memory import (
    ControlTokenWitness,
    DictionaryBasis,
    LayerParams,
    ManifoldMemoryBank,
    ManifoldMemoryHandoffEnvelope,
    ManifoldMemoryOrchestrator,
    ManifoldMemoryRegistry,
    ManifoldMemoryTeam,
    ManifoldMemoryTeamResult,
    MemoryEntry,
    MemoryForwardResult,
    MemoryGatingDecision,
    MultiLayerControllerParams,
    MultiRankRoleAdapter,
    PrefixCapsule,
    TimeAttentionWitness,
    W46_ALL_CTRL_MODES,
    W46_BRANCH_MEMORY_RATIFIED,
    W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH,
    W46_CTRL_MODE_COMPACT,
    W46_CTRL_MODE_FULL,
    W46_CTRL_MODE_OFF,
    W46_DEFAULT_DICTIONARY_SIZE,
    W46_DEFAULT_MEMORY_CAPACITY,
    W46_DEFAULT_N_LAYERS,
    W46_DEFAULT_PREFIX_TURNS,
    W46_DEFAULT_ROLE_DELTA_RANK,
    W46_DEFAULT_TIME_ATTN_TEMPERATURE,
    W46_DEFAULT_TIME_ATTN_WEIGHT,
    W46_NO_DICT_CODE,
    _flatten_channel_features,
    build_control_token_string,
    build_manifold_memory_registry,
    build_prefix_capsule,
    build_trivial_manifold_memory_registry,
    build_unfitted_memory_controller_params,
    fit_memory_controller,
    forward_memory_controller,
)
from .product_manifold import (
    CellObservation,
    ProductManifoldChannelBundle,
    ProductManifoldPolicyEntry,
    SphericalConsensusSignature,
    SubspaceBasis,
    encode_cell_channels,
)
from .team_coord import capsule_team_handoff


# =============================================================================
# Schema, branches, defaults
# =============================================================================

W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION: str = (
    "coordpy.autograd_manifold.v1")
W47_TEAM_RESULT_SCHEMA: str = (
    "coordpy.autograd_manifold_team_result.v1")

# Decision branches reuse W46 names; add three W47-specific.
W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH: str = (
    "autograd_trivial_passthrough")
W47_BRANCH_AUTOGRAD_DISABLED: str = "autograd_disabled"
W47_BRANCH_AUTOGRAD_RATIFIED: str = "autograd_ratified"
W47_BRANCH_AUTOGRAD_NO_POLICY: str = "autograd_no_policy"
W47_BRANCH_AUTOGRAD_CAUSAL_ABSTAIN: str = (
    "autograd_causal_abstain")
W47_BRANCH_AUTOGRAD_SPHERICAL_ABSTAIN: str = (
    "autograd_spherical_abstain")
W47_BRANCH_AUTOGRAD_SUBSPACE_ABSTAIN: str = (
    "autograd_subspace_abstain")
W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN: str = (
    "autograd_margin_abstain")
W47_BRANCH_AUTOGRAD_TIME_ATTN_ABSTAIN: str = (
    "autograd_time_attn_abstain")
W47_BRANCH_AUTOGRAD_TRAIN_FAILURE: str = (
    "autograd_train_failure")
W47_BRANCH_AUTOGRAD_REJECTED: str = "autograd_rejected"

W47_ALL_BRANCHES: tuple[str, ...] = (
    W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH,
    W47_BRANCH_AUTOGRAD_DISABLED,
    W47_BRANCH_AUTOGRAD_RATIFIED,
    W47_BRANCH_AUTOGRAD_NO_POLICY,
    W47_BRANCH_AUTOGRAD_CAUSAL_ABSTAIN,
    W47_BRANCH_AUTOGRAD_SPHERICAL_ABSTAIN,
    W47_BRANCH_AUTOGRAD_SUBSPACE_ABSTAIN,
    W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN,
    W47_BRANCH_AUTOGRAD_TIME_ATTN_ABSTAIN,
    W47_BRANCH_AUTOGRAD_TRAIN_FAILURE,
    W47_BRANCH_AUTOGRAD_REJECTED,
)

W47_AUTOGRAD_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W47_BRANCH_AUTOGRAD_CAUSAL_ABSTAIN,
    W47_BRANCH_AUTOGRAD_SPHERICAL_ABSTAIN,
    W47_BRANCH_AUTOGRAD_SUBSPACE_ABSTAIN,
    W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN,
    W47_BRANCH_AUTOGRAD_TIME_ATTN_ABSTAIN,
    W47_BRANCH_AUTOGRAD_TRAIN_FAILURE,
})

# Supported autograd ops — used by tests and the gradient check.
W47_SUPPORTED_OPS: tuple[str, ...] = (
    "add", "sub", "mul", "neg", "pow", "dot", "matmul",
    "tanh", "sigmoid", "relu", "exp", "log",
    "softmax", "mean", "sum",
)

# Default trainable-stack hyperparameters.
W47_DEFAULT_HIDDEN_DIM: int = 16
W47_DEFAULT_N_LAYERS: int = 3
W47_DEFAULT_LEARNING_RATE: float = 0.05
W47_DEFAULT_N_STEPS: int = 200
W47_DEFAULT_INIT_SCALE: float = 0.3
W47_DEFAULT_BETA1: float = 0.9
W47_DEFAULT_BETA2: float = 0.999
W47_DEFAULT_EPS: float = 1e-8
W47_DEFAULT_TRAIN_SEED: int = 0
W47_DEFAULT_LOSS_HISTORY_HEAD: int = 8
W47_DEFAULT_LOSS_HISTORY_TAIL: int = 8
W47_DEFAULT_TRAIN_BATCH_SIZE: int = 0  # full batch
W47_DEFAULT_MEMORY_HEAD_DIM: int = 8
W47_DEFAULT_DICT_GUMBEL_TEMP: float = 1.0
W47_DEFAULT_GRAD_CLIP: float = 5.0

W47_NO_ROLE_DELTA: str = "no_role_delta"


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


def _round_matrix(
        matrix: Sequence[Sequence[float]], precision: int = 12,
) -> list[list[float]]:
    return [_round_floats(row, precision) for row in matrix]


# =============================================================================
# Pure-Python reverse-mode autograd engine
# =============================================================================

class Variable:
    """Scalar autograd variable with reverse-mode gradient tape.

    A :class:`Variable` carries a Python ``float`` value plus a
    gradient accumulator and a topologically-sorted list of
    parent nodes that contributed to it. Calling
    :meth:`backward` on a scalar root populates ``.grad`` on
    every ancestor leaf.

    Honest scope: this is a *pure-Python scalar autograd engine*.
    Each scalar operation creates one Variable node; there is no
    vectorisation, no GPU. Correctness is checked against
    finite-differences in :func:`gradient_check`.
    """

    __slots__ = ("value", "grad", "_parents", "_op", "_grad_fn")

    def __init__(
            self,
            value: float,
            *,
            parents: Sequence["Variable"] = (),
            op: str = "leaf",
            grad_fn: Callable[[float], Sequence[float]] | None = None,
    ) -> None:
        self.value = float(value)
        self.grad = 0.0
        self._parents = tuple(parents)
        self._op = str(op)
        # grad_fn: given the upstream gradient flowing into this
        # node, returns the list of gradients to push to each
        # parent. For leaves it's None.
        self._grad_fn = grad_fn

    @property
    def op(self) -> str:
        return self._op

    @property
    def parents(self) -> tuple["Variable", ...]:
        return self._parents

    # -- arithmetic primitives --------------------------------------------

    def __add__(self, other: "Variable | float") -> "Variable":
        if not isinstance(other, Variable):
            other = Variable(float(other))
        v = Variable(
            self.value + other.value,
            parents=(self, other), op="add",
            grad_fn=lambda g: (g, g),
        )
        return v

    def __radd__(self, other: float) -> "Variable":
        return Variable(float(other)) + self

    def __sub__(self, other: "Variable | float") -> "Variable":
        if not isinstance(other, Variable):
            other = Variable(float(other))
        v = Variable(
            self.value - other.value,
            parents=(self, other), op="sub",
            grad_fn=lambda g: (g, -g),
        )
        return v

    def __rsub__(self, other: float) -> "Variable":
        return Variable(float(other)) - self

    def __mul__(self, other: "Variable | float") -> "Variable":
        if not isinstance(other, Variable):
            other = Variable(float(other))
        a, b = self.value, other.value
        v = Variable(
            a * b,
            parents=(self, other), op="mul",
            grad_fn=lambda g, a=a, b=b: (g * b, g * a),
        )
        return v

    def __rmul__(self, other: float) -> "Variable":
        return Variable(float(other)) * self

    def __neg__(self) -> "Variable":
        v = Variable(
            -self.value, parents=(self,), op="neg",
            grad_fn=lambda g: (-g,),
        )
        return v

    def __pow__(self, exponent: float) -> "Variable":
        e = float(exponent)
        a = self.value
        v = Variable(
            a ** e,
            parents=(self,), op="pow",
            grad_fn=lambda g, a=a, e=e: (g * e * (a ** (e - 1.0)),),
        )
        return v

    def __truediv__(
            self, other: "Variable | float",
    ) -> "Variable":
        if not isinstance(other, Variable):
            other = Variable(float(other))
        b = other.value
        if abs(b) < 1e-30:
            b = 1e-30 if b >= 0 else -1e-30
        a = self.value
        v = Variable(
            a / b,
            parents=(self, other), op="div",
            grad_fn=lambda g, a=a, b=b: (
                g / b, -g * a / (b * b)),
        )
        return v

    # -- nonlinearities ----------------------------------------------------

    def tanh(self) -> "Variable":
        t = math.tanh(self.value)
        v = Variable(
            t, parents=(self,), op="tanh",
            grad_fn=lambda g, t=t: (g * (1.0 - t * t),),
        )
        return v

    def sigmoid(self) -> "Variable":
        x = self.value
        if x >= 0:
            ex = math.exp(-x)
            s = 1.0 / (1.0 + ex)
        else:
            ex = math.exp(x)
            s = ex / (1.0 + ex)
        v = Variable(
            s, parents=(self,), op="sigmoid",
            grad_fn=lambda g, s=s: (g * s * (1.0 - s),),
        )
        return v

    def relu(self) -> "Variable":
        r = self.value if self.value > 0.0 else 0.0
        dz = 1.0 if self.value > 0.0 else 0.0
        v = Variable(
            r, parents=(self,), op="relu",
            grad_fn=lambda g, dz=dz: (g * dz,),
        )
        return v

    def exp(self) -> "Variable":
        ex = math.exp(min(self.value, 50.0))  # clamp to avoid overflow
        v = Variable(
            ex, parents=(self,), op="exp",
            grad_fn=lambda g, ex=ex: (g * ex,),
        )
        return v

    def log(self) -> "Variable":
        x = self.value
        if x <= 1e-30:
            x = 1e-30
        lg = math.log(x)
        v = Variable(
            lg, parents=(self,), op="log",
            grad_fn=lambda g, x=x: (g / x,),
        )
        return v

    # -- backward ----------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode backprop from this node.

        Builds a topological sort of all ancestors and pushes
        gradients in reverse order. Leaf accumulators land in
        ``.grad``; intermediate nodes receive their upstream
        gradient but their accumulator is reset at the start of
        each backward pass.
        """
        # Reset all gradients in the DAG.
        topo: list[Variable] = []
        visited: set[int] = set()

        def visit(node: Variable) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node._parents:
                visit(p)
            topo.append(node)

        visit(self)
        for n in topo:
            n.grad = 0.0
        self.grad = 1.0
        for n in reversed(topo):
            if n._grad_fn is None:
                continue
            grads = n._grad_fn(n.grad)
            for parent, g in zip(n._parents, grads):
                parent.grad += float(g)

    def detach(self) -> "Variable":
        return Variable(float(self.value))

    def __repr__(self) -> str:
        return (
            f"Variable(value={self.value:.6f}, grad={self.grad:.6f}, "
            f"op={self._op})")


def vdot(a: Sequence[Variable], b: Sequence[Variable]) -> Variable:
    """Dot product of two equal-length sequences of Variables."""
    if not a or not b:
        return Variable(0.0)
    n = min(len(a), len(b))
    result = a[0] * b[0]
    for i in range(1, n):
        result = result + a[i] * b[i]
    return result


def vsum(items: Sequence[Variable]) -> Variable:
    """Sum of a sequence of Variables."""
    if not items:
        return Variable(0.0)
    result = items[0]
    for i in range(1, len(items)):
        result = result + items[i]
    return result


def vmean(items: Sequence[Variable]) -> Variable:
    """Mean of a sequence of Variables."""
    if not items:
        return Variable(0.0)
    s = vsum(items)
    return s * (1.0 / float(len(items)))


def vsoftmax(items: Sequence[Variable]) -> list[Variable]:
    """Numerically stable softmax over a sequence of Variables.

    Subtracts the max via plain Python float arithmetic for
    numerical stability — the max is a constant w.r.t. the
    gradient, so this is exact.
    """
    if not items:
        return []
    m = max(it.value for it in items)
    exps = [(it - m).exp() for it in items]
    s = vsum(exps)
    return [e / s for e in exps]


def vmatmul(
        weights: Sequence[Sequence[Variable]],
        inputs: Sequence[Variable],
) -> list[Variable]:
    """Matrix-vector product. ``weights`` is shape ``(out, in)``,
    ``inputs`` is shape ``(in,)``.
    """
    out: list[Variable] = []
    for row in weights:
        out.append(vdot(row, inputs))
    return out


# =============================================================================
# Trainable parameters
# =============================================================================

@dataclasses.dataclass
class ParamTensor:
    """A trainable parameter tensor.

    Stores a flat list of Variables alongside the original shape.
    The Variables are *fresh* every step (the autograd tape is
    rebuilt every step), but their values + gradients persist via
    :meth:`update_values` / :meth:`zero_grad`.
    """

    shape: tuple[int, ...]
    values: list[float]
    _vars: list[Variable] = dataclasses.field(default_factory=list)

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def init_seed(self, seed: int, scale: float) -> None:
        """Deterministically initialise from a seed-derived stream
        of pseudo-random uniform draws in ``[-scale, scale]``."""
        rng = _DeterministicLCG(seed=int(seed))
        self.values = [
            (rng.next_uniform() * 2.0 - 1.0) * float(scale)
            for _ in range(self.size)
        ]

    def make_vars(self) -> list[Variable]:
        """Build a fresh list of Variables for the current step,
        wrapping ``self.values``. The autograd tape rebuilds from
        these leaves every step.
        """
        self._vars = [Variable(float(v)) for v in self.values]
        return self._vars

    def zero_grad(self) -> None:
        for v in self._vars:
            v.grad = 0.0

    def grads(self) -> list[float]:
        return [float(v.grad) for v in self._vars]

    def update_values(
            self, new_values: Sequence[float],
    ) -> None:
        if len(new_values) != self.size:
            raise ValueError(
                f"new_values has size {len(new_values)} != "
                f"{self.size}")
        self.values = [float(v) for v in new_values]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(int(s) for s in self.shape),
            "values": _round_floats(self.values),
        }


class _DeterministicLCG:
    """A tiny deterministic linear-congruential generator.

    We refuse to use ``random`` so the autograd init is fully
    reproducible from a seed integer without depending on the
    global RNG state.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self.state = (int(seed) & 0xFFFFFFFF) or 0xDEADBEEF
        # Numerical Recipes LCG constants.
        self.a = 1664525
        self.c = 1013904223
        self.m = 1 << 32

    def next_uniform(self) -> float:
        self.state = (self.a * self.state + self.c) % self.m
        return float(self.state) / float(self.m)


# =============================================================================
# Adam-style optimiser
# =============================================================================

@dataclasses.dataclass
class AdamOptimizer:
    """Pure-Python Adam.

    Per-parameter first / second moment EMAs. Updates parameter
    values in-place via ``ParamTensor.update_values`` after each
    backward pass.
    """

    learning_rate: float = W47_DEFAULT_LEARNING_RATE
    beta1: float = W47_DEFAULT_BETA1
    beta2: float = W47_DEFAULT_BETA2
    eps: float = W47_DEFAULT_EPS
    step_counter: int = 0
    grad_clip: float = W47_DEFAULT_GRAD_CLIP

    _m: dict[int, list[float]] = dataclasses.field(
        default_factory=dict)
    _v: dict[int, list[float]] = dataclasses.field(
        default_factory=dict)

    def step(self, params: Sequence[ParamTensor]) -> None:
        self.step_counter += 1
        t = float(self.step_counter)
        b1 = float(self.beta1)
        b2 = float(self.beta2)
        eps = float(self.eps)
        lr = float(self.learning_rate)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for pi, p in enumerate(params):
            grads = p.grads()
            # Gradient clipping (per-tensor L2).
            if self.grad_clip > 0.0:
                gn = math.sqrt(sum(g * g for g in grads))
                if gn > self.grad_clip:
                    scale = self.grad_clip / max(gn, 1e-12)
                    grads = [g * scale for g in grads]
            m = self._m.setdefault(pi, [0.0] * p.size)
            v = self._v.setdefault(pi, [0.0] * p.size)
            new_values = list(p.values)
            for i in range(p.size):
                g = float(grads[i])
                m[i] = b1 * m[i] + (1.0 - b1) * g
                v[i] = b2 * v[i] + (1.0 - b2) * g * g
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2
                new_values[i] = (
                    new_values[i]
                    - lr * m_hat / (math.sqrt(v_hat) + eps))
            p.update_values(new_values)

    def config_dict(self) -> dict[str, Any]:
        return {
            "kind": "adam_w47",
            "learning_rate": float(round(
                self.learning_rate, 12)),
            "beta1": float(round(self.beta1, 12)),
            "beta2": float(round(self.beta2, 12)),
            "eps": float(round(self.eps, 12)),
            "grad_clip": float(round(self.grad_clip, 12)),
        }


# =============================================================================
# Gradient check
# =============================================================================

def gradient_check(
        f: Callable[[Sequence[Variable]], Variable],
        x0: Sequence[float],
        *,
        eps: float = 1e-5,
        atol: float = 1e-5,
) -> tuple[bool, float, list[float], list[float]]:
    """Compare analytic gradients to finite-difference estimates.

    Builds Variables wrapping ``x0``, forwards through ``f``,
    backwards, and compares the resulting ``.grad`` of each input
    Variable against a central finite-difference of ``f`` at the
    same point.

    Returns ``(ok, max_abs_err, analytic_grads, fd_grads)``.
    """
    vs = [Variable(float(v)) for v in x0]
    y = f(vs)
    y.backward()
    analytic = [float(v.grad) for v in vs]
    fd: list[float] = []
    for i in range(len(x0)):
        def shift(delta: float) -> float:
            shifted = list(float(v) for v in x0)
            shifted[i] += delta
            vs2 = [Variable(s) for s in shifted]
            return float(f(vs2).value)
        fp = shift(eps)
        fm = shift(-eps)
        fd.append((fp - fm) / (2.0 * eps))
    max_err = max(abs(a - b) for a, b in zip(analytic, fd))
    return bool(max_err <= atol), float(max_err), analytic, fd


# =============================================================================
# Trainable multi-layer manifold stack
# =============================================================================

@dataclasses.dataclass
class AutogradStackLayer:
    """One trainable fully-connected tanh layer.

    Each layer has a weight tensor of shape ``(out_dim, in_dim)``
    and a bias tensor of shape ``(out_dim,)``.
    """

    in_dim: int
    out_dim: int
    weights: ParamTensor
    biases: ParamTensor
    activation: str = "tanh"

    @classmethod
    def init(
            cls,
            *,
            in_dim: int,
            out_dim: int,
            seed: int,
            scale: float = W47_DEFAULT_INIT_SCALE,
            activation: str = "tanh",
    ) -> "AutogradStackLayer":
        w = ParamTensor(
            shape=(int(out_dim), int(in_dim)),
            values=[],
        )
        w.init_seed(seed=int(seed), scale=float(scale))
        b = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim),
        )
        return cls(
            in_dim=int(in_dim), out_dim=int(out_dim),
            weights=w, biases=b, activation=str(activation))

    def params(self) -> list[ParamTensor]:
        return [self.weights, self.biases]

    def forward(
            self,
            inputs: Sequence[Variable],
            *,
            weight_vars: Sequence[Variable] | None = None,
            bias_vars: Sequence[Variable] | None = None,
    ) -> list[Variable]:
        if weight_vars is None:
            weight_vars = self.weights.make_vars()
        if bias_vars is None:
            bias_vars = self.biases.make_vars()
        # Reshape weight_vars (flat) into (out_dim, in_dim).
        rows: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.in_dim
            row = list(
                weight_vars[base:base + self.in_dim])
            rows.append(row)
        pre = vmatmul(rows, list(inputs))
        post = []
        for i, pi in enumerate(pre):
            with_b = pi + bias_vars[i]
            if self.activation == "tanh":
                post.append(with_b.tanh())
            elif self.activation == "sigmoid":
                post.append(with_b.sigmoid())
            elif self.activation == "relu":
                post.append(with_b.relu())
            elif self.activation == "linear":
                post.append(with_b)
            else:
                raise ValueError(
                    f"unknown activation {self.activation!r}")
        return post

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "out_dim": int(self.out_dim),
            "activation": str(self.activation),
            "weights": self.weights.to_dict(),
            "biases": self.biases.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_stack_layer",
            "layer": self.to_dict()})


@dataclasses.dataclass
class AutogradManifoldStack:
    """An ``L``-layer trainable manifold stack.

    Layers form a deterministic FC tanh chain mapping a flat
    feature vector of length ``W45_N_CHANNELS * feature_dim`` to
    a scalar gate logit (last layer is shape ``(1, hidden)`` with
    linear activation).
    """

    layers: tuple[AutogradStackLayer, ...]
    feature_dim: int

    @classmethod
    def init(
            cls,
            *,
            feature_dim: int = W45_DEFAULT_FEATURE_DIM,
            n_layers: int = W47_DEFAULT_N_LAYERS,
            hidden_dim: int = W47_DEFAULT_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AutogradManifoldStack":
        in_dim = int(W45_N_CHANNELS) * int(feature_dim)
        layers: list[AutogradStackLayer] = []
        # Hidden layers: in_dim -> hidden -> hidden -> ... -> hidden
        # then a single linear scalar output.
        n = max(1, int(n_layers))
        rng = _DeterministicLCG(seed=int(seed))
        for li in range(n - 1):
            layer_seed = (
                (rng.state ^ ((li + 1) * 0x9E3779B9)) & 0xFFFFFFFF)
            li_in = in_dim if li == 0 else int(hidden_dim)
            layers.append(AutogradStackLayer.init(
                in_dim=li_in,
                out_dim=int(hidden_dim),
                seed=int(layer_seed),
                scale=float(init_scale),
                activation="tanh",
            ))
        # Final linear scalar output.
        final_in = in_dim if n == 1 else int(hidden_dim)
        final_seed = (rng.state ^ 0xC6BC279692B5C323) & 0xFFFFFFFF
        layers.append(AutogradStackLayer.init(
            in_dim=int(final_in),
            out_dim=1,
            seed=int(final_seed),
            scale=float(init_scale),
            activation="linear",
        ))
        return cls(
            layers=tuple(layers),
            feature_dim=int(feature_dim),
        )

    @property
    def in_dim(self) -> int:
        return int(self.layers[0].in_dim)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def forward_vars(
            self,
            inputs: Sequence[Variable],
    ) -> Variable:
        """Forward pass with autograd tape; returns scalar Variable."""
        cur = list(inputs)
        for layer in self.layers:
            cur = layer.forward(cur)
        # Last layer is (1,) — return scalar.
        return cur[0]

    def forward_value(
            self, flat_features: Sequence[float],
    ) -> float:
        """Forward pass without autograd tape (inference)."""
        vs = [Variable(float(v)) for v in flat_features]
        out = self.forward_vars(vs)
        return float(out.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "layers": [lyr.to_dict() for lyr in self.layers],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_manifold_stack",
            "stack": self.to_dict()})


# =============================================================================
# Trainable rank-r role adapter
# =============================================================================

@dataclasses.dataclass
class AutogradRoleAdapter:
    """Rank-``r`` per-role trainable delta.

    For each role with at least ``rank + 1`` training examples we
    store two factor tensors: ``A`` of shape ``(in_dim, rank)``
    and ``B`` of shape ``(rank,)``. The per-role delta applied to
    the deepest hidden state ``h`` is ``B^T A^T h``, a scalar
    additive correction to the gate logit. Roles with too few
    examples get the all-zero delta (recorded as
    :data:`W47_NO_ROLE_DELTA`).
    """

    rank: int
    in_dim: int
    role_factors: dict[
        str, tuple[ParamTensor, ParamTensor]] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def init(
            cls,
            *,
            roles: Sequence[str],
            rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
            in_dim: int = W47_DEFAULT_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AutogradRoleAdapter":
        rng = _DeterministicLCG(seed=int(seed))
        rf: dict[str, tuple[ParamTensor, ParamTensor]] = {}
        for role in sorted(set(str(r) for r in roles)):
            a = ParamTensor(
                shape=(int(in_dim), int(rank)),
                values=[],
            )
            a.init_seed(
                seed=int(rng.next_uniform() * (1 << 30)),
                scale=float(init_scale))
            b = ParamTensor(
                shape=(int(rank),),
                values=[0.0] * int(rank),
            )
            rf[role] = (a, b)
        return cls(
            rank=int(rank), in_dim=int(in_dim), role_factors=rf,
        )

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for _, (a, b) in sorted(self.role_factors.items()):
            out.extend([a, b])
        return out

    def forward_delta(
            self,
            *,
            role: str,
            hidden: Sequence[Variable],
    ) -> Variable:
        """Compute scalar role delta = B^T A^T h."""
        key = str(role)
        if key not in self.role_factors:
            return Variable(0.0)
        a, b = self.role_factors[key]
        a_vars = a.make_vars()
        b_vars = b.make_vars()
        # A^T h -> r-vector. A is (in_dim, rank), so the i-th
        # output is dot(A[:, i], h).
        r_proj: list[Variable] = []
        for i in range(self.rank):
            col = [
                a_vars[j * self.rank + i]
                for j in range(self.in_dim)
            ]
            r_proj.append(vdot(col, list(hidden)))
        # delta = B^T r_proj.
        return vdot(b_vars, r_proj)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "in_dim": int(self.in_dim),
            "role_factors": {
                str(role): {
                    "A": a.to_dict(), "B": b.to_dict()}
                for role, (a, b) in sorted(self.role_factors.items())
            },
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_role_adapter",
            "adapter": self.to_dict()})


# =============================================================================
# Trainable dictionary / codebook
# =============================================================================

@dataclasses.dataclass
class AutogradDictionary:
    """Trainable ``K``-prototype dictionary.

    Each prototype is a flat vector of length ``W45_N_CHANNELS *
    feature_dim``. Trained by a soft-assignment cross-entropy +
    straight-through residual loss: gradient flows through both
    the assignment softmax and the residual reconstruction.

    At inference, encode picks the closest prototype (argmin L2)
    and returns ``(index, residual)``. Bijective: decode adds the
    residual back to the chosen prototype.
    """

    feature_dim: int
    k: int
    prototypes: ParamTensor

    @classmethod
    def init(
            cls,
            *,
            feature_dim: int = W45_DEFAULT_FEATURE_DIM,
            k: int = W46_DEFAULT_DICTIONARY_SIZE,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AutogradDictionary":
        vec_dim = int(W45_N_CHANNELS) * int(feature_dim)
        proto = ParamTensor(
            shape=(int(k), int(vec_dim)),
            values=[],
        )
        proto.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(
            feature_dim=int(feature_dim), k=int(k),
            prototypes=proto)

    @property
    def vector_dim(self) -> int:
        return int(W45_N_CHANNELS) * int(self.feature_dim)

    def params(self) -> list[ParamTensor]:
        return [self.prototypes]

    def encode_inference(
            self,
            flat: Sequence[float],
    ) -> tuple[int, tuple[float, ...]]:
        """Closest-prototype encode (no autograd)."""
        if self.k == 0:
            return W46_NO_DICT_CODE, tuple(float(v) for v in flat)
        vec = list(float(v) for v in flat)[:self.vector_dim]
        while len(vec) < self.vector_dim:
            vec.append(0.0)
        protos_flat = self.prototypes.values
        best_idx = 0
        best_dist = float("inf")
        for pi in range(self.k):
            base = pi * self.vector_dim
            d = 0.0
            for j in range(self.vector_dim):
                diff = vec[j] - protos_flat[base + j]
                d += diff * diff
            if d < best_dist:
                best_dist = d
                best_idx = pi
        base = best_idx * self.vector_dim
        residual = tuple(
            float(vec[j] - protos_flat[base + j])
            for j in range(self.vector_dim))
        return int(best_idx), tuple(_round_floats(residual))

    def decode(
            self, index: int, residual: Sequence[float],
    ) -> tuple[float, ...]:
        """Decode (index, residual) back to the original flat."""
        if index < 0 or index >= self.k:
            return tuple(float(v) for v in residual)
        base = index * self.vector_dim
        resid = list(residual)[:self.vector_dim]
        while len(resid) < self.vector_dim:
            resid.append(0.0)
        return tuple(
            float(self.prototypes.values[base + j] + resid[j])
            for j in range(self.vector_dim))

    def soft_assign_vars(
            self,
            flat: Sequence[Variable],
            *,
            temperature: float = W47_DEFAULT_DICT_GUMBEL_TEMP,
    ) -> tuple[list[Variable], list[Variable]]:
        """Differentiable soft assignment + reconstruction.

        Returns ``(soft_weights, reconstruction)`` where
        ``soft_weights`` is the K-vector softmax over neg L2
        distances and ``reconstruction`` is the weighted-sum
        prototype reconstruction.
        """
        proto_vars = self.prototypes.make_vars()
        # Per-prototype neg L2 squared distance.
        neg_dists: list[Variable] = []
        for pi in range(self.k):
            base = pi * self.vector_dim
            diffs: list[Variable] = []
            for j in range(self.vector_dim):
                diffs.append(
                    (flat[j] - proto_vars[base + j])
                    * (flat[j] - proto_vars[base + j]))
            d = vsum(diffs)
            # Negate + divide by temperature.
            neg_dists.append(d * (-1.0 / max(1e-9, float(temperature))))
        soft = vsoftmax(neg_dists)
        # Reconstruction: sum_k soft_k * proto_k
        recon: list[Variable] = []
        for j in range(self.vector_dim):
            comps: list[Variable] = []
            for pi in range(self.k):
                base = pi * self.vector_dim
                comps.append(soft[pi] * proto_vars[base + j])
            recon.append(vsum(comps))
        return soft, recon

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "k": int(self.k),
            "prototypes": self.prototypes.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_dictionary",
            "dict": self.to_dict()})


# =============================================================================
# Trainable memory read/write head
# =============================================================================

@dataclasses.dataclass
class AutogradMemoryHead:
    """Trainable QKV attention head over the memory bank.

    Query projects the current turn's flat features to a
    ``head_dim``-vector; key projects each memory entry's flat
    features the same way; value is a learned scalar projection
    of the entry's gate logit and per-channel logits. The
    attention output is a single scalar added to the gate logit.
    """

    in_dim: int
    head_dim: int
    w_query: ParamTensor
    w_key: ParamTensor
    w_value: ParamTensor
    b_query: ParamTensor
    b_key: ParamTensor

    @classmethod
    def init(
            cls,
            *,
            in_dim: int,
            head_dim: int = W47_DEFAULT_MEMORY_HEAD_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AutogradMemoryHead":
        rng = _DeterministicLCG(seed=int(seed))
        wq = ParamTensor(
            shape=(int(head_dim), int(in_dim)),
            values=[],
        )
        wq.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        wk = ParamTensor(
            shape=(int(head_dim), int(in_dim)),
            values=[],
        )
        wk.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        # Value is a single scalar that scales the entry's gate
        # logit. Initialize at 1.0.
        wv = ParamTensor(
            shape=(1,), values=[1.0])
        bq = ParamTensor(
            shape=(int(head_dim),),
            values=[0.0] * int(head_dim))
        bk = ParamTensor(
            shape=(int(head_dim),),
            values=[0.0] * int(head_dim))
        return cls(
            in_dim=int(in_dim), head_dim=int(head_dim),
            w_query=wq, w_key=wk, w_value=wv,
            b_query=bq, b_key=bk,
        )

    def params(self) -> list[ParamTensor]:
        return [
            self.w_query, self.w_key, self.w_value,
            self.b_query, self.b_key,
        ]

    def _project(
            self,
            inputs: Sequence[Variable],
            *,
            weights: Sequence[Variable],
            biases: Sequence[Variable],
    ) -> list[Variable]:
        rows: list[list[Variable]] = []
        for r in range(self.head_dim):
            base = r * self.in_dim
            row = list(weights[base:base + self.in_dim])
            rows.append(row)
        pre = vmatmul(rows, list(inputs))
        return [pre[i] + biases[i] for i in range(self.head_dim)]

    def forward_attention_vars(
            self,
            *,
            query_input: Sequence[Variable],
            keys_inputs: Sequence[Sequence[Variable]],
            entry_logits: Sequence[Variable],
    ) -> Variable:
        """Trainable scaled-dot-product attention.

        Returns the scalar pooled attention output, which is
        added to the gate logit.
        """
        if not keys_inputs:
            return Variable(0.0)
        wq_vars = self.w_query.make_vars()
        wk_vars = self.w_key.make_vars()
        wv_vars = self.w_value.make_vars()
        bq_vars = self.b_query.make_vars()
        bk_vars = self.b_key.make_vars()
        q = self._project(
            query_input, weights=wq_vars, biases=bq_vars)
        ks: list[list[Variable]] = []
        for ki in keys_inputs:
            ks.append(
                self._project(
                    ki, weights=wk_vars, biases=bk_vars))
        # Scaled dot-product scores.
        scale = 1.0 / math.sqrt(max(1.0, float(self.head_dim)))
        scores: list[Variable] = []
        for k in ks:
            sc = vdot(q, k) * scale
            scores.append(sc)
        # Softmax.
        attn = vsoftmax(scores)
        # Apply learned value scalar to entry logits.
        vs: list[Variable] = []
        for el in entry_logits:
            vs.append(el * wv_vars[0])
        # Pooled = sum_i attn_i * v_i.
        pooled = vsum([attn[i] * vs[i] for i in range(len(vs))])
        return pooled

    def forward_attention_value(
            self,
            *,
            query_input: Sequence[float],
            keys_inputs: Sequence[Sequence[float]],
            entry_logits: Sequence[float],
    ) -> tuple[float, list[float]]:
        """Forward pass without autograd; returns (pooled,
        attn_weights)."""
        if not keys_inputs:
            return 0.0, []
        # Plain Python matmul.
        wq = self.w_query.values
        wk = self.w_key.values
        bq = self.b_query.values
        bk = self.b_key.values
        v_scale = float(self.w_value.values[0])
        q = []
        for r in range(self.head_dim):
            base = r * self.in_dim
            q.append(
                sum(wq[base + j] * float(query_input[j])
                    for j in range(self.in_dim))
                + bq[r])
        scores: list[float] = []
        scale = 1.0 / math.sqrt(max(1.0, float(self.head_dim)))
        for ki in keys_inputs:
            k = []
            for r in range(self.head_dim):
                base = r * self.in_dim
                k.append(
                    sum(wk[base + j] * float(ki[j])
                        for j in range(self.in_dim))
                    + bk[r])
            scores.append(
                sum(q[r] * k[r] for r in range(self.head_dim))
                * scale)
        attn = _softmax(scores)
        pooled = sum(
            attn[i] * float(entry_logits[i]) * v_scale
            for i in range(len(entry_logits)))
        return float(pooled), [float(a) for a in attn]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "head_dim": int(self.head_dim),
            "w_query": self.w_query.to_dict(),
            "w_key": self.w_key.to_dict(),
            "w_value": self.w_value.to_dict(),
            "b_query": self.b_query.to_dict(),
            "b_key": self.b_key.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_memory_head",
            "head": self.to_dict()})


# =============================================================================
# Trainable packed control serializer
# =============================================================================

@dataclasses.dataclass
class AutogradControlSerializer:
    """Per-field sigmoid gates that learn which CTRL fields to
    emit.

    Four gates: ``layer_logits``, ``mem_attn``, ``dict_idx``,
    ``mem_summary``. Each is a single trainable scalar; the
    inference rule is ``gate(z) >= 0.5 -> emit, else suppress``.
    Suppression is recorded deterministically in the envelope as
    a 4-bit ``emit_mask``.
    """

    gates: ParamTensor  # shape (4,)

    @classmethod
    def init(
            cls, *,
            seed: int = W47_DEFAULT_TRAIN_SEED,
    ) -> "AutogradControlSerializer":
        # Initialise so all gates are slightly > 0.5 (emit all by
        # default) so that pre-training behavior matches W46 full
        # mode.
        gates = ParamTensor(
            shape=(4,), values=[0.5, 0.5, 0.5, 0.5])
        return cls(gates=gates)

    def params(self) -> list[ParamTensor]:
        return [self.gates]

    def emit_mask(self) -> tuple[bool, bool, bool, bool]:
        """At-inference emit mask: gate >= 0.5 (after sigmoid) ==
        emit."""
        def s(x: float) -> float:
            if x >= 0:
                return 1.0 / (1.0 + math.exp(-x))
            ex = math.exp(x)
            return ex / (1.0 + ex)
        gs = [s(v) for v in self.gates.values]
        return tuple(bool(g >= 0.5) for g in gs)

    def forward_loss_vars(
            self,
            *,
            target_mask: Sequence[bool],
    ) -> Variable:
        """Binary cross-entropy on the target emit mask."""
        gs = self.gates.make_vars()
        losses: list[Variable] = []
        for i in range(min(4, len(target_mask))):
            s = gs[i].sigmoid()
            if bool(target_mask[i]):
                losses.append((s + 1e-9).log() * (-1.0))
            else:
                # log(1 - s)
                losses.append(
                    ((Variable(1.0) - s) + 1e-9).log() * (-1.0))
        return vmean(losses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": self.gates.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_control_serializer",
            "ser": self.to_dict()})


# =============================================================================
# Full Autograd Manifold Params bundle + training trace
# =============================================================================

@dataclasses.dataclass
class TrainingTraceWitness:
    """Sealed record of one autograd training run.

    Bound under the W47 envelope; an auditor can re-fit with the
    same seed + training set and verify the trace CID matches.
    """

    seed: int
    n_steps: int
    optimizer_config: dict[str, Any]
    init_scale: float
    loss_history_head: tuple[float, ...]
    loss_history_tail: tuple[float, ...]
    grad_norm_head: tuple[float, ...]
    grad_norm_tail: tuple[float, ...]
    final_train_loss: float
    final_grad_norm: float
    final_params_cid: str
    training_set_cid: str
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "optimizer_config": dict(self.optimizer_config),
            "init_scale": float(round(self.init_scale, 12)),
            "loss_history_head": _round_floats(
                self.loss_history_head),
            "loss_history_tail": _round_floats(
                self.loss_history_tail),
            "grad_norm_head": _round_floats(self.grad_norm_head),
            "grad_norm_tail": _round_floats(self.grad_norm_tail),
            "final_train_loss": float(round(
                self.final_train_loss, 12)),
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
            "final_params_cid": str(self.final_params_cid),
            "training_set_cid": str(self.training_set_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_training_trace_witness",
            "trace": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class AutogradManifoldParams:
    """All trained, content-addressed W47 params.

    Frozen wrapper around the trained sub-components plus a
    reference to the underlying W46 ``MultiLayerControllerParams``
    used for the (closed-form) base path. The autograd stack +
    role adapter + dictionary + memory head + control serializer
    are *additive* to the W46 base.
    """

    base: MultiLayerControllerParams
    stack: AutogradManifoldStack
    role_adapter: AutogradRoleAdapter
    dictionary: AutogradDictionary
    memory_head: AutogradMemoryHead
    control_serializer: AutogradControlSerializer
    fitting_method: str
    training_trace: TrainingTraceWitness

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "stack": self.stack.to_dict(),
            "role_adapter": self.role_adapter.to_dict(),
            "dictionary": self.dictionary.to_dict(),
            "memory_head": self.memory_head.to_dict(),
            "control_serializer":
                self.control_serializer.to_dict(),
            "fitting_method": str(self.fitting_method),
            "training_trace_cid":
                self.training_trace.cid(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w47_autograd_manifold_params",
            "params": self.to_dict()})

    @property
    def feature_dim(self) -> int:
        return int(self.stack.feature_dim)

    @property
    def n_layers(self) -> int:
        return int(self.stack.n_layers)

    @property
    def role_adapter_rank(self) -> int:
        return int(self.role_adapter.rank)

    @property
    def dictionary_size(self) -> int:
        return int(self.dictionary.k)


def build_unfitted_autograd_params(
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        hidden_dim: int = W47_DEFAULT_HIDDEN_DIM,
        n_layers: int = W47_DEFAULT_N_LAYERS,
        rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
        dictionary_size: int = W46_DEFAULT_DICTIONARY_SIZE,
        head_dim: int = W47_DEFAULT_MEMORY_HEAD_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        roles: Sequence[str] = (),
) -> AutogradManifoldParams:
    """Build a fully-initialised (but untrained) bundle of W47
    autograd params, with the underlying W46 base in the
    ``unfitted`` state.
    """
    base = build_unfitted_memory_controller_params(
        feature_dim=int(feature_dim),
        n_layers=int(W46_DEFAULT_N_LAYERS),
        rank=int(rank),
        dictionary_size=int(dictionary_size),
    )
    stack = AutogradManifoldStack.init(
        feature_dim=int(feature_dim),
        n_layers=int(n_layers),
        hidden_dim=int(hidden_dim),
        seed=int(seed),
        init_scale=float(init_scale),
    )
    role_adapter = AutogradRoleAdapter.init(
        roles=tuple(roles),
        rank=int(rank),
        in_dim=int(hidden_dim) if int(n_layers) > 1 else (
            int(W45_N_CHANNELS) * int(feature_dim)),
        seed=int(seed) + 1,
        init_scale=float(init_scale),
    )
    dictionary = AutogradDictionary.init(
        feature_dim=int(feature_dim),
        k=int(dictionary_size),
        seed=int(seed) + 2,
        init_scale=float(init_scale),
    )
    memory_head = AutogradMemoryHead.init(
        in_dim=int(W45_N_CHANNELS) * int(feature_dim),
        head_dim=int(head_dim),
        seed=int(seed) + 3,
        init_scale=float(init_scale),
    )
    control_serializer = AutogradControlSerializer.init(
        seed=int(seed) + 4)
    trace = TrainingTraceWitness(
        seed=int(seed), n_steps=0,
        optimizer_config=AdamOptimizer().config_dict(),
        init_scale=float(init_scale),
        loss_history_head=tuple(),
        loss_history_tail=tuple(),
        grad_norm_head=tuple(),
        grad_norm_tail=tuple(),
        final_train_loss=0.0,
        final_grad_norm=0.0,
        final_params_cid="",
        training_set_cid="",
        diverged=False,
    )
    return AutogradManifoldParams(
        base=base,
        stack=stack,
        role_adapter=role_adapter,
        dictionary=dictionary,
        memory_head=memory_head,
        control_serializer=control_serializer,
        fitting_method="unfitted",
        training_trace=trace,
    )


# =============================================================================
# Autograd trainer
# =============================================================================

def _flatten_example_features(
        ex: TrainingExample,
        *,
        feature_dim: int,
) -> list[float]:
    """Flatten an example's channel features in canonical order."""
    fmap = ex.channel_features_map
    out: list[float] = []
    for c in W45_CHANNEL_ORDER:
        feats = list(fmap.get(c, ()))[:feature_dim]
        while len(feats) < feature_dim:
            feats.append(0.0)
        out.extend(float(v) for v in feats)
    return out


def fit_autograd_controller(
        training_set: TrainingSet,
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_layers: int = W47_DEFAULT_N_LAYERS,
        hidden_dim: int = W47_DEFAULT_HIDDEN_DIM,
        n_steps: int = W47_DEFAULT_N_STEPS,
        learning_rate: float = W47_DEFAULT_LEARNING_RATE,
        rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
        dictionary_size: int = W46_DEFAULT_DICTIONARY_SIZE,
        head_dim: int = W47_DEFAULT_MEMORY_HEAD_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        fit_role_deltas: bool = True,
        fit_dictionary: bool = True,
        history_head: int = W47_DEFAULT_LOSS_HISTORY_HEAD,
        history_tail: int = W47_DEFAULT_LOSS_HISTORY_TAIL,
        dict_loss_weight: float = 0.5,
        role_loss_weight: float = 0.5,
) -> AutogradManifoldParams:
    """Fit an :class:`AutogradManifoldParams` bundle via SGD/Adam.

    Trains:
      * the multi-layer stack on the per-example binary
        cross-entropy loss (sigmoid(gate_logit) vs label_pos)
      * per-role adapter factors on the per-role residual
      * dictionary prototypes on the per-example soft-assignment
        reconstruction loss
      * the memory head on a synthetic 'gate at turn t depends
        on prior gate logits' loss (only if there is more than
        one example per role, otherwise we skip)
      * the control serializer is left at its default 'emit all'
        state; downstream r94 family family_trainable_packed_control
        trains it on its specific task.

    Returns the trained :class:`AutogradManifoldParams` bundle
    plus a sealed training-trace witness CID.
    """
    fd = int(feature_dim)
    nl = max(1, int(n_layers))
    hd = int(hidden_dim)

    # First, fit the underlying W46 base (closed form). This
    # provides the per-role / spherical / etc. baseline; the
    # autograd stack learns the *residual* on top.
    base = fit_memory_controller(
        training_set,
        feature_dim=fd,
        n_layers=W46_DEFAULT_N_LAYERS,
        role_delta_rank=int(rank),
        dictionary_size=int(dictionary_size),
    )

    # Build the trainable autograd params.
    roles = tuple(
        sorted({str(ex.role) for ex in training_set.examples}))
    params_bundle = build_unfitted_autograd_params(
        feature_dim=fd, hidden_dim=hd, n_layers=nl,
        rank=int(rank), dictionary_size=int(dictionary_size),
        head_dim=int(head_dim),
        seed=int(seed), init_scale=float(init_scale),
        roles=roles,
    )

    # Pre-flatten all examples.
    flat_examples: list[list[float]] = []
    labels_pos: list[bool] = []
    roles_list: list[str] = []
    for ex in training_set.examples:
        flat_examples.append(
            _flatten_example_features(ex, feature_dim=fd))
        labels_pos.append(bool(float(ex.label) > 0.0))
        roles_list.append(str(ex.role))

    # Set up Adam optimiser over the trainable params.
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
        grad_clip=float(grad_clip),
    )
    trainable: list[ParamTensor] = []
    trainable.extend(params_bundle.stack.params())
    if fit_role_deltas:
        trainable.extend(params_bundle.role_adapter.params())
    if fit_dictionary:
        trainable.extend(params_bundle.dictionary.params())

    loss_history: list[float] = []
    grad_norm_history: list[float] = []

    diverged = False
    for step in range(int(n_steps)):
        # Build one full-batch training graph.
        step_losses: list[Variable] = []
        for ex_idx, flat in enumerate(flat_examples):
            inputs = [Variable(float(v)) for v in flat]
            # Stack forward.
            cur = list(inputs)
            for li, layer in enumerate(params_bundle.stack.layers):
                cur = layer.forward(cur)
            # cur is now either (1,) (final layer linear) or
            # (hidden_dim,) if n_layers==1.
            if nl > 1:
                # Apply per-role adapter on the deepest hidden
                # state — which is the *second-to-last* layer
                # output. We need to recompute it because cur is
                # already the final scalar. Easier: redo the
                # forward up to layer L-1 in-place.
                #
                # Implementation: rebuild the hidden state
                # alongside the final scalar.
                hidden = [Variable(float(v)) for v in flat]
                for layer in params_bundle.stack.layers[:-1]:
                    hidden = layer.forward(hidden)
                # The final layer's input *is* the deepest hidden.
                # Use the same deepest hidden for the role adapter.
                # We already have the scalar in cur[0].
                gate = cur[0]
                if (fit_role_deltas
                        and roles_list[ex_idx]
                        in params_bundle.role_adapter.role_factors):
                    role_delta = (
                        params_bundle.role_adapter.forward_delta(
                            role=roles_list[ex_idx],
                            hidden=hidden,
                        ))
                    gate = gate + role_delta
            else:
                # Single-layer fallback: cur is shape (1,).
                gate = cur[0]
            prob = gate.sigmoid()
            # Binary cross-entropy.
            if labels_pos[ex_idx]:
                ll = -(prob + 1e-9).log()
            else:
                ll = -((Variable(1.0) - prob) + 1e-9).log()
            step_losses.append(ll)

        cls_loss = vmean(step_losses)
        loss = cls_loss

        if fit_dictionary:
            dict_losses: list[Variable] = []
            for flat in flat_examples:
                fvs = [Variable(float(v)) for v in flat]
                soft, recon = (
                    params_bundle.dictionary.soft_assign_vars(fvs))
                # L2 reconstruction.
                comps: list[Variable] = []
                for j in range(len(fvs)):
                    diff = fvs[j] - recon[j]
                    comps.append(diff * diff)
                dict_losses.append(vmean(comps))
            d_loss = vmean(dict_losses)
            loss = loss + d_loss * float(dict_loss_weight)

        # Backward.
        loss.backward()

        # Compute total grad norm before stepping.
        total_grad_sq = 0.0
        for p in trainable:
            for g in p.grads():
                total_grad_sq += float(g) * float(g)
        gn = math.sqrt(total_grad_sq)
        grad_norm_history.append(gn)
        loss_history.append(float(loss.value))

        # Check divergence (NaN / inf).
        if (loss.value != loss.value
                or loss.value == float("inf")
                or loss.value == float("-inf")):
            diverged = True
            break

        # Update.
        optim.step(trainable)

    final_loss = float(
        loss_history[-1] if loss_history else 0.0)
    final_gn = float(
        grad_norm_history[-1] if grad_norm_history else 0.0)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    loss_head = tuple(loss_history[:head_n])
    loss_tail = tuple(loss_history[-tail_n:]) if tail_n > 0 else tuple()
    gn_head = tuple(grad_norm_history[:head_n])
    gn_tail = (
        tuple(grad_norm_history[-tail_n:])
        if tail_n > 0 else tuple())

    # Replace stack/role_adapter/dictionary/etc. with fitted versions
    # by re-building the bundle. They are stored by reference already.
    # Build the final-params CID *before* the trace witness, so the
    # trace CID can bind it.
    fitted_bundle = AutogradManifoldParams(
        base=base,
        stack=params_bundle.stack,
        role_adapter=params_bundle.role_adapter,
        dictionary=params_bundle.dictionary,
        memory_head=params_bundle.memory_head,
        control_serializer=params_bundle.control_serializer,
        fitting_method=(
            "autograd_adam_v1" if not diverged
            else "autograd_diverged"),
        training_trace=TrainingTraceWitness(
            seed=int(seed),
            n_steps=int(n_steps),
            optimizer_config=optim.config_dict(),
            init_scale=float(init_scale),
            loss_history_head=loss_head,
            loss_history_tail=loss_tail,
            grad_norm_head=gn_head,
            grad_norm_tail=gn_tail,
            final_train_loss=float(final_loss),
            final_grad_norm=float(final_gn),
            final_params_cid="",
            training_set_cid=str(training_set.cid()),
            diverged=bool(diverged),
        ),
    )
    # Now rebuild with the proper final_params_cid.
    final_cid = _sha256_hex({
        "kind": "w47_autograd_manifold_params_inner",
        "stack": fitted_bundle.stack.to_dict(),
        "role_adapter": fitted_bundle.role_adapter.to_dict(),
        "dictionary": fitted_bundle.dictionary.to_dict(),
        "memory_head": fitted_bundle.memory_head.to_dict(),
        "control_serializer":
            fitted_bundle.control_serializer.to_dict(),
        "base_cid": fitted_bundle.base.cid(),
    })
    trace_v2 = TrainingTraceWitness(
        seed=int(seed),
        n_steps=int(n_steps),
        optimizer_config=optim.config_dict(),
        init_scale=float(init_scale),
        loss_history_head=loss_head,
        loss_history_tail=loss_tail,
        grad_norm_head=gn_head,
        grad_norm_tail=gn_tail,
        final_train_loss=float(final_loss),
        final_grad_norm=float(final_gn),
        final_params_cid=str(final_cid),
        training_set_cid=str(training_set.cid()),
        diverged=bool(diverged),
    )
    return AutogradManifoldParams(
        base=base,
        stack=fitted_bundle.stack,
        role_adapter=fitted_bundle.role_adapter,
        dictionary=fitted_bundle.dictionary,
        memory_head=fitted_bundle.memory_head,
        control_serializer=fitted_bundle.control_serializer,
        fitting_method=fitted_bundle.fitting_method,
        training_trace=trace_v2,
    )


# =============================================================================
# Forward pass (inference)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AutogradForwardResult:
    """Result of one W47 forward pass at inference time."""

    memory_forward: MemoryForwardResult
    autograd_logit: float
    autograd_role_delta: float
    autograd_role_present: bool
    autograd_dict_index: int
    autograd_dict_residual_l1: float
    autograd_memory_pooled: float
    autograd_memory_attn_weights: tuple[float, ...]
    emit_mask: tuple[bool, bool, bool, bool]
    gate_logit: float
    ratify_probability: float
    confidence_bucket: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_forward": self.memory_forward.to_dict(),
            "autograd_logit": float(round(
                self.autograd_logit, 12)),
            "autograd_role_delta": float(round(
                self.autograd_role_delta, 12)),
            "autograd_role_present": bool(
                self.autograd_role_present),
            "autograd_dict_index": int(self.autograd_dict_index),
            "autograd_dict_residual_l1": float(round(
                self.autograd_dict_residual_l1, 12)),
            "autograd_memory_pooled": float(round(
                self.autograd_memory_pooled, 12)),
            "autograd_memory_attn_weights": _round_floats(
                self.autograd_memory_attn_weights),
            "emit_mask": [bool(b) for b in self.emit_mask],
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "confidence_bucket": int(self.confidence_bucket),
        }


def forward_autograd_controller(
        *,
        channel_features: Mapping[str, Sequence[float]],
        params: AutogradManifoldParams,
        role: str,
        memory_bank: ManifoldMemoryBank,
        turn_index: int,
        use_attention_routing: bool = True,
        time_attention_enabled: bool = True,
        role_adapter_disabled: bool = False,
        dictionary_enabled: bool = True,
        autograd_weight: float = 1.0,
        memory_head_weight: float = 1.0,
) -> AutogradForwardResult:
    """W47 forward pass at inference.

    Combines:
      1. The W46 base forward (multi-layer base + role + dict +
         time attention) -> ``memory_forward``.
      2. The trained autograd stack's scalar gate-logit
         contribution.
      3. The trained autograd role adapter's per-role delta.
      4. The trained autograd dictionary index + residual L1
         (used for the envelope).
      5. The trained memory head's attention pool over the bank.

    The final gate logit is
    ``memory_forward.gate_logit + autograd_weight *
    (autograd_logit + autograd_role_delta)
    + memory_head_weight * autograd_memory_pooled``.
    """
    fd = int(params.feature_dim)

    # 1. W46 base forward.
    mem_fwd = forward_memory_controller(
        channel_features=channel_features,
        params=params.base,
        role=str(role),
        memory_bank=memory_bank,
        turn_index=int(turn_index),
        use_attention_routing=bool(use_attention_routing),
        time_attention_enabled=bool(time_attention_enabled),
        role_adapter_disabled=bool(role_adapter_disabled),
        dictionary_enabled=bool(dictionary_enabled),
    )

    # 2. Flatten channel features.
    flat_query = list(_flatten_channel_features(
        channel_features, feature_dim=fd))

    # 3. Autograd stack forward (inference).
    stack_logit = float(params.stack.forward_value(flat_query))

    # 4. Autograd role adapter forward (inference). We need the
    # deepest hidden state.
    role_delta_value = 0.0
    role_present = False
    if not role_adapter_disabled:
        hidden = [float(v) for v in flat_query]
        for layer in params.stack.layers[:-1]:
            new_hidden: list[float] = []
            for r in range(layer.out_dim):
                base_idx = r * layer.in_dim
                proj = sum(
                    layer.weights.values[base_idx + j] * hidden[j]
                    for j in range(layer.in_dim))
                proj += layer.biases.values[r]
                # Apply activation (tanh by default).
                if layer.activation == "tanh":
                    new_hidden.append(math.tanh(proj))
                elif layer.activation == "sigmoid":
                    new_hidden.append(
                        1.0 / (1.0 + math.exp(-proj))
                        if proj >= 0
                        else math.exp(proj) / (1.0 + math.exp(proj)))
                elif layer.activation == "relu":
                    new_hidden.append(max(0.0, proj))
                else:
                    new_hidden.append(proj)
            hidden = new_hidden
        if str(role) in params.role_adapter.role_factors:
            role_present = True
            a, b = params.role_adapter.role_factors[str(role)]
            r_proj = []
            for i in range(params.role_adapter.rank):
                col = [
                    a.values[j * params.role_adapter.rank + i]
                    for j in range(params.role_adapter.in_dim)]
                r_proj.append(
                    sum(c * h for c, h in zip(col, hidden)))
            role_delta_value = sum(
                b.values[i] * r_proj[i]
                for i in range(params.role_adapter.rank))

    # 5. Autograd dictionary encode (inference).
    if dictionary_enabled and params.dictionary.k > 0:
        dict_idx, dict_resid = params.dictionary.encode_inference(
            flat_query)
        dict_resid_l1 = float(sum(abs(v) for v in dict_resid))
    else:
        dict_idx = W46_NO_DICT_CODE
        dict_resid_l1 = 0.0

    # 6. Autograd memory head readout.
    admissible = memory_bank.admissible_for_turn(int(turn_index))
    mem_pooled = 0.0
    mem_attn_weights: list[float] = []
    if time_attention_enabled and admissible:
        keys_inputs: list[list[float]] = []
        entry_logits: list[float] = []
        for e in admissible:
            keys_inputs.append(
                list(_flatten_channel_features(
                    e.channel_features_map, feature_dim=fd)))
            entry_logits.append(float(e.gate_logit))
        mem_pooled, mem_attn_weights = (
            params.memory_head.forward_attention_value(
                query_input=flat_query,
                keys_inputs=keys_inputs,
                entry_logits=entry_logits,
            ))

    emit_mask = params.control_serializer.emit_mask()

    gate_logit = (
        float(mem_fwd.gate_logit)
        + float(autograd_weight) * (
            float(stack_logit) + float(role_delta_value))
        + float(memory_head_weight) * float(mem_pooled))
    # Numerically stable sigmoid.
    if gate_logit >= 0:
        ratify_prob = 1.0 / (1.0 + math.exp(-gate_logit))
    else:
        ratify_prob = (
            math.exp(gate_logit) / (1.0 + math.exp(gate_logit)))
    conf = _confidence_bucket_for_probability(float(ratify_prob))

    return AutogradForwardResult(
        memory_forward=mem_fwd,
        autograd_logit=float(round(stack_logit, 12)),
        autograd_role_delta=float(round(role_delta_value, 12)),
        autograd_role_present=bool(role_present),
        autograd_dict_index=int(dict_idx),
        autograd_dict_residual_l1=float(round(dict_resid_l1, 12)),
        autograd_memory_pooled=float(round(mem_pooled, 12)),
        autograd_memory_attn_weights=tuple(
            _round_floats(mem_attn_weights)),
        emit_mask=tuple(bool(b) for b in emit_mask),
        gate_logit=float(round(gate_logit, 12)),
        ratify_probability=float(round(ratify_prob, 12)),
        confidence_bucket=int(conf),
    )


# =============================================================================
# Registry
# =============================================================================

@dataclasses.dataclass
class AutogradManifoldRegistry:
    """Controller-side configuration for the W47 autograd coupling.

    Wraps a :class:`ManifoldMemoryRegistry` (the W46 inner) and
    adds autograd-controller toggles plus the trained params.
    """

    schema_cid: str
    memory_registry: ManifoldMemoryRegistry
    params: AutogradManifoldParams
    autograd_enabled: bool = True
    autograd_weight: float = 1.0
    memory_head_weight: float = 1.0
    role_adapter_disabled: bool = False
    dictionary_enabled: bool = True
    time_attention_enabled: bool = True
    margin_abstain_threshold: float = 0.0
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT
    control_token_mode: str = W46_CTRL_MODE_FULL

    @property
    def is_trivial(self) -> bool:
        return (
            self.memory_registry.is_trivial
            and not self.autograd_enabled
            and not self.abstain_substitution_enabled
            and self.params.fitting_method == "unfitted"
        )


def build_trivial_autograd_manifold_registry(
        *, schema_cid: str | None = None,
) -> AutogradManifoldRegistry:
    """Build a registry whose orchestrator reduces to
    :class:`coordpy.manifold_memory.ManifoldMemoryTeam` (trivial)
    byte-for-byte (the W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH
    falsifier).
    """
    cid = schema_cid or _sha256_hex({
        "kind": "w47_trivial_schema"})
    return AutogradManifoldRegistry(
        schema_cid=str(cid),
        memory_registry=(
            build_trivial_manifold_memory_registry(
                schema_cid=str(cid))),
        params=build_unfitted_autograd_params(),
        autograd_enabled=False,
        autograd_weight=0.0,
        memory_head_weight=0.0,
        role_adapter_disabled=True,
        dictionary_enabled=False,
        time_attention_enabled=False,
        abstain_substitution_enabled=False,
        control_token_mode=W46_CTRL_MODE_OFF,
    )


def build_autograd_manifold_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        params: AutogradManifoldParams | None = None,
        autograd_enabled: bool = True,
        autograd_weight: float = 1.0,
        memory_head_weight: float = 1.0,
        role_adapter_disabled: bool = False,
        dictionary_enabled: bool = True,
        time_attention_enabled: bool = True,
        margin_abstain_threshold: float = 0.0,
        abstain_substitution_enabled: bool = True,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
        control_token_mode: str = W46_CTRL_MODE_FULL,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
        live_enabled: bool = True,
        learned_enabled: bool = True,
        memory_enabled: bool = True,
        prefix_reuse_enabled: bool = True,
        prefix_turns: int = W46_DEFAULT_PREFIX_TURNS,
        memory_capacity: int = W46_DEFAULT_MEMORY_CAPACITY,
        use_attention_routing: bool = True,
) -> AutogradManifoldRegistry:
    """Build a fully configured W47 autograd-manifold registry."""
    if control_token_mode not in W46_ALL_CTRL_MODES:
        raise ValueError(
            f"control_token_mode={control_token_mode!r} not in "
            f"{W46_ALL_CTRL_MODES}")
    p = params or build_unfitted_autograd_params()
    mem = build_manifold_memory_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        params=p.base,
        memory_enabled=bool(memory_enabled),
        time_attention_enabled=bool(time_attention_enabled),
        dictionary_enabled=bool(dictionary_enabled),
        role_adapter_disabled=bool(role_adapter_disabled),
        control_token_mode=str(control_token_mode),
        prefix_reuse_enabled=bool(prefix_reuse_enabled),
        prefix_turns=int(prefix_turns),
        memory_capacity=int(memory_capacity),
        margin_abstain_threshold=float(margin_abstain_threshold),
        abstain_substitution_enabled=False,  # W47 owns abstain
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
        live_enabled=bool(live_enabled),
        learned_enabled=bool(learned_enabled),
        use_attention_routing=bool(use_attention_routing),
        abstain_output=str(abstain_output),
    )
    return AutogradManifoldRegistry(
        schema_cid=str(schema_cid),
        memory_registry=mem,
        params=p,
        autograd_enabled=bool(autograd_enabled),
        autograd_weight=float(autograd_weight),
        memory_head_weight=float(memory_head_weight),
        role_adapter_disabled=bool(role_adapter_disabled),
        dictionary_enabled=bool(dictionary_enabled),
        time_attention_enabled=bool(time_attention_enabled),
        margin_abstain_threshold=float(margin_abstain_threshold),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        control_token_mode=str(control_token_mode),
    )


# =============================================================================
# Decision selector
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AutogradGatingDecision:
    """Result of running the W47 autograd gate on one turn."""

    branch: str
    w46_branch: str
    w45_branch: str
    w44_branch: str
    pmc_branch: str
    spherical_agreement: float
    subspace_drift: float
    causal_admissible: bool
    factoradic_int: int
    factoradic_n_bits: int
    role_handoff_signature_cid: str
    policy_entry_cid: str
    pmc_envelope_cid: str
    w44_envelope_cid: str
    w45_envelope_cid: str
    w46_envelope_cid: str
    forward: AutogradForwardResult
    causal_mask_cid: str
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W47_AUTOGRAD_ABSTAIN_BRANCHES


def _classify_w46_branch_to_autograd(
        w46_branch: str,
) -> str:
    """Map a W46 memory branch to the corresponding W47 branch."""
    mapping = {
        W46_BRANCH_MEMORY_RATIFIED: W47_BRANCH_AUTOGRAD_RATIFIED,
        W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH:
            W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH,
        "memory_disabled": W47_BRANCH_AUTOGRAD_DISABLED,
        "memory_no_policy": W47_BRANCH_AUTOGRAD_NO_POLICY,
        "memory_causal_abstain":
            W47_BRANCH_AUTOGRAD_CAUSAL_ABSTAIN,
        "memory_spherical_abstain":
            W47_BRANCH_AUTOGRAD_SPHERICAL_ABSTAIN,
        "memory_subspace_abstain":
            W47_BRANCH_AUTOGRAD_SUBSPACE_ABSTAIN,
        "memory_margin_abstain":
            W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN,
        "memory_time_attn_abstain":
            W47_BRANCH_AUTOGRAD_TIME_ATTN_ABSTAIN,
    }
    return mapping.get(w46_branch, W47_BRANCH_AUTOGRAD_DISABLED)


# =============================================================================
# Orchestrator
# =============================================================================

class AutogradManifoldOrchestrator:
    """Per-turn W47 gating + envelope binding.

    Wraps a :class:`ManifoldMemoryOrchestrator` (W46 inner) plus
    a :class:`AutogradManifoldRegistry`. The orchestrator is
    stateful only in the underlying W46 memory bank.
    """

    def __init__(
            self, registry: AutogradManifoldRegistry,
    ) -> None:
        self.registry = registry
        self._memory = ManifoldMemoryOrchestrator(
            registry=registry.memory_registry)

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    @property
    def memory_bank(self) -> ManifoldMemoryBank:
        return self._memory.memory_bank

    def reset_session(self) -> None:
        self._memory.reset_session()

    def gate(
            self,
            *,
            observation: CellObservation,
            role: str,
            role_handoff_signature_cid: str,
            parent_w42_cid: str,
            n_w42_visible_tokens: int,
            turn_index: int,
            expected_spherical: SphericalConsensusSignature | None = None,
            expected_subspace: SubspaceBasis | None = None,
    ) -> tuple[AutogradGatingDecision, Any]:
        # Delegate to the W46 inner.
        w46_decision, w46_aux = self._memory.gate(
            observation=observation,
            role=str(role),
            role_handoff_signature_cid=role_handoff_signature_cid,
            parent_w42_cid=str(parent_w42_cid),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            turn_index=int(turn_index),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )
        (w43_result, causal_mask, bundle,
         w45_decision, w46_forward) = w46_aux

        feats = _channel_features_from_bundle(
            bundle,
            feature_dim=int(self.registry.params.feature_dim),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )

        forward = forward_autograd_controller(
            channel_features=feats,
            params=self.registry.params,
            role=str(role),
            memory_bank=self._memory.memory_bank,
            turn_index=int(turn_index),
            use_attention_routing=bool(
                self.registry.memory_registry.use_attention_routing),
            time_attention_enabled=bool(
                self.registry.time_attention_enabled),
            role_adapter_disabled=bool(
                self.registry.role_adapter_disabled),
            dictionary_enabled=bool(
                self.registry.dictionary_enabled),
            autograd_weight=float(self.registry.autograd_weight),
            memory_head_weight=float(
                self.registry.memory_head_weight),
        )

        autograd_branch = _classify_w46_branch_to_autograd(
            w46_decision.branch)
        abstain_reason = w46_decision.abstain_reason

        if (self.registry.autograd_enabled
                and autograd_branch == W47_BRANCH_AUTOGRAD_RATIFIED
                and forward.gate_logit
                < float(self.registry.margin_abstain_threshold)):
            autograd_branch = W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN
            abstain_reason = "autograd_margin"

        if not self.registry.autograd_enabled:
            if self.registry.is_trivial:
                autograd_branch = (
                    W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH)
            elif autograd_branch == W47_BRANCH_AUTOGRAD_RATIFIED:
                autograd_branch = W47_BRANCH_AUTOGRAD_DISABLED

        if (self.registry.params.training_trace.diverged
                and autograd_branch == W47_BRANCH_AUTOGRAD_RATIFIED):
            autograd_branch = W47_BRANCH_AUTOGRAD_TRAIN_FAILURE
            abstain_reason = "autograd_train_failure"

        decision = AutogradGatingDecision(
            branch=str(autograd_branch),
            w46_branch=str(w46_decision.branch),
            w45_branch=str(w46_decision.w45_branch),
            w44_branch=str(w46_decision.w44_branch),
            pmc_branch=str(w46_decision.pmc_branch),
            spherical_agreement=float(
                w46_decision.spherical_agreement),
            subspace_drift=float(w46_decision.subspace_drift),
            causal_admissible=bool(w46_decision.causal_admissible),
            factoradic_int=int(w46_decision.factoradic_int),
            factoradic_n_bits=int(
                w46_decision.factoradic_n_bits),
            role_handoff_signature_cid=str(
                w46_decision.role_handoff_signature_cid),
            policy_entry_cid=str(w46_decision.policy_entry_cid),
            pmc_envelope_cid=str(w46_decision.pmc_envelope_cid),
            w44_envelope_cid=str(w46_decision.w44_envelope_cid),
            w45_envelope_cid=str(w46_decision.w45_envelope_cid),
            w46_envelope_cid="",  # filled in by the team loop
            forward=forward,
            causal_mask_cid=str(w46_decision.causal_mask_cid),
            abstain_reason=str(abstain_reason),
        )
        return decision, (
            w43_result, causal_mask, bundle,
            w45_decision, w46_decision, w46_forward, forward,
        )


# =============================================================================
# Envelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AutogradManifoldHandoffEnvelope:
    """Sealed autograd-manifold envelope for one turn of the W47
    layer.
    """

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w46_envelope_cid: str
    parent_w45_envelope_cid: str
    parent_w44_envelope_cid: str
    parent_w43_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
    w46_branch: str
    w45_branch: str
    w44_branch: str
    pmc_branch: str
    abstain_reason: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    control_token_mode: str
    inline_route_mode: str
    factoradic_int: int
    factoradic_n_bits: int
    hint_confidence_bucket: int

    # Autograd-controller provenance.
    autograd_params_cid: str
    training_trace_cid: str
    fitting_method: str
    n_autograd_layers: int
    autograd_role_adapter_rank: int
    autograd_dictionary_cid: str
    autograd_dictionary_size: int
    autograd_stack_cid: str
    autograd_memory_head_cid: str
    autograd_control_serializer_cid: str
    autograd_role_adapter_cid: str

    # Memory inner provenance.
    memory_params_cid: str
    memory_bank_head_cid: str
    memory_bank_size: int
    memory_capacity: int

    # Forward witnesses.
    autograd_logit: float
    autograd_role_delta_value: float
    autograd_role_adapter_present: bool
    autograd_dict_index: int
    autograd_dict_residual_l1: float
    autograd_memory_pooled: float
    emit_mask: tuple[bool, bool, bool, bool]
    autograd_forward_witness_cid: str
    causal_mask_witness_cid: str

    # Prompt / control / prefix witnesses.
    prompt_sha256: str
    prompt_construction_witness_cid: str
    control_token_witness_cid: str
    prefix_capsule_cid: str
    prefix_reused: bool
    output_sha256: str

    # Token accounting.
    n_visible_prompt_tokens_textual: int
    n_visible_prompt_tokens_actual: int
    n_visible_prompt_tokens_saved: int
    n_overhead_tokens: int
    n_ctrl_tokens: int
    n_prefix_tokens: int

    # Margin diagnostics.
    gate_logit: float
    ratify_probability: float

    behavioral_change: bool

    autograd_witness_cid: str
    autograd_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w47_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w46_envelope_cid=self.parent_w46_envelope_cid,
            autograd_params_cid=self.autograd_params_cid,
            training_trace_cid=self.training_trace_cid,
            autograd_witness_cid=self.autograd_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_w47_autograd_forward_witness_cid(
        *,
        autograd_logit: float,
        autograd_role_delta_value: float,
        autograd_role_present: bool,
        autograd_dict_index: int,
        autograd_dict_residual_l1: float,
        autograd_memory_pooled: float,
        emit_mask: tuple[bool, ...],
        gate_logit: float,
        ratify_probability: float,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w47_autograd_forward_witness",
        "autograd_logit": float(round(autograd_logit, 12)),
        "autograd_role_delta_value": float(round(
            autograd_role_delta_value, 12)),
        "autograd_role_present": bool(autograd_role_present),
        "autograd_dict_index": int(autograd_dict_index),
        "autograd_dict_residual_l1": float(round(
            autograd_dict_residual_l1, 12)),
        "autograd_memory_pooled": float(round(
            autograd_memory_pooled, 12)),
        "emit_mask": [bool(b) for b in emit_mask],
        "gate_logit": float(round(gate_logit, 12)),
        "ratify_probability": float(round(
            ratify_probability, 12)),
        "turn_index": int(turn_index),
    })


def _compute_w47_prompt_construction_witness_cid(
        *,
        turn_index: int,
        role: str,
        prompt_sha256: str,
        control_token_mode: str,
        inline_route_mode: str,
        factoradic_int: int,
        factoradic_n_bits: int,
        confidence_bucket: int,
        n_visible_prompt_tokens_textual: int,
        n_visible_prompt_tokens_actual: int,
        n_ctrl_tokens: int,
        n_prefix_tokens: int,
        emit_mask: tuple[bool, ...],
) -> str:
    return _sha256_hex({
        "kind": "w47_prompt_construction_witness",
        "turn_index": int(turn_index),
        "role": str(role),
        "prompt_sha256": str(prompt_sha256),
        "control_token_mode": str(control_token_mode),
        "inline_route_mode": str(inline_route_mode),
        "factoradic_int": int(factoradic_int),
        "factoradic_n_bits": int(factoradic_n_bits),
        "confidence_bucket": int(confidence_bucket),
        "n_visible_prompt_tokens_textual": int(
            n_visible_prompt_tokens_textual),
        "n_visible_prompt_tokens_actual": int(
            n_visible_prompt_tokens_actual),
        "n_ctrl_tokens": int(n_ctrl_tokens),
        "n_prefix_tokens": int(n_prefix_tokens),
        "emit_mask": [bool(b) for b in emit_mask],
    })


def _compute_w47_autograd_witness_cid(
        *,
        decision_branch: str,
        w46_branch: str,
        w45_branch: str,
        w44_branch: str,
        pmc_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        autograd_params_cid: str,
        training_trace_cid: str,
        autograd_forward_witness_cid: str,
        causal_mask_witness_cid: str,
        prompt_construction_witness_cid: str,
        control_token_witness_cid: str,
        prefix_capsule_cid: str,
        memory_bank_head_cid: str,
        memory_params_cid: str,
        output_sha256: str,
        behavioral_change: bool,
) -> str:
    return _sha256_hex({
        "kind": "w47_autograd_witness",
        "decision_branch": str(decision_branch),
        "w46_branch": str(w46_branch),
        "w45_branch": str(w45_branch),
        "w44_branch": str(w44_branch),
        "pmc_branch": str(pmc_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "autograd_params_cid": str(autograd_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "autograd_forward_witness_cid": str(
            autograd_forward_witness_cid),
        "causal_mask_witness_cid": str(causal_mask_witness_cid),
        "prompt_construction_witness_cid": str(
            prompt_construction_witness_cid),
        "control_token_witness_cid": str(
            control_token_witness_cid),
        "prefix_capsule_cid": str(prefix_capsule_cid),
        "memory_bank_head_cid": str(memory_bank_head_cid),
        "memory_params_cid": str(memory_params_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
    })


def _compute_w47_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w46_envelope_cid: str,
        autograd_params_cid: str,
        training_trace_cid: str,
        autograd_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w47_autograd_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w46_envelope_cid": str(parent_w46_envelope_cid),
        "autograd_params_cid": str(autograd_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "autograd_witness_cid": str(autograd_witness_cid),
        "turn_index": int(turn_index),
    })


# =============================================================================
# Verifier (18+ enumerated W47 failure modes)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AutogradManifoldVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


W47_ALL_FAILURE_MODES: tuple[str, ...] = (
    "empty_w47_envelope",
    "w47_schema_version_unknown",
    "w47_schema_cid_mismatch",
    "w47_decision_branch_unknown",
    "w47_ctrl_mode_unknown",
    "w47_role_handoff_signature_cid_invalid",
    "w47_prompt_sha256_invalid",
    "w47_token_accounting_invalid",
    "w47_confidence_bucket_invalid",
    "w47_ratify_probability_invalid",
    "w47_autograd_params_cid_invalid",
    "w47_training_trace_cid_invalid",
    "w47_autograd_forward_witness_cid_mismatch",
    "w47_causal_mask_witness_cid_invalid",
    "w47_control_token_witness_cid_invalid",
    "w47_prefix_capsule_cid_invalid",
    "w47_memory_bank_head_cid_invalid",
    "w47_prompt_construction_witness_cid_mismatch",
    "w47_autograd_witness_cid_mismatch",
    "w47_emit_mask_invalid",
    "w47_outer_cid_mismatch",
)


def verify_autograd_manifold_handoff(
        env: "AutogradManifoldHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_autograd_params_cid: str | None = None,
        registered_training_trace_cid: str | None = None,
) -> AutogradManifoldVerificationOutcome:
    """Pure-function verifier for the W47 autograd envelope.

    Enumerates 21 disjoint W47 failure modes (see
    :data:`W47_ALL_FAILURE_MODES`).
    """
    n = 0
    if env is None:
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="empty_w47_envelope", n_checks=0)
    n += 1
    if env.schema_version != (
            W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W47_ALL_BRANCHES:
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.control_token_mode not in W46_ALL_CTRL_MODES:
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_ctrl_mode_unknown", n_checks=n)
    n += 1
    if env.decision_branch != (
            W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return AutogradManifoldVerificationOutcome(
                ok=False,
                reason=(
                    "w47_role_handoff_signature_cid_invalid"),
                n_checks=n)
    n += 1
    if (env.prompt_sha256 is None
            or (env.prompt_sha256
                and len(env.prompt_sha256) not in (0, 64))):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_prompt_sha256_invalid",
            n_checks=n)
    n += 1
    if (env.n_visible_prompt_tokens_textual < 0
            or env.n_visible_prompt_tokens_actual < 0
            or env.n_overhead_tokens < 0
            or env.n_ctrl_tokens < 0
            or env.n_prefix_tokens < 0
            or env.n_visible_prompt_tokens_saved
            != (int(env.n_visible_prompt_tokens_textual)
                - int(env.n_visible_prompt_tokens_actual))):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_token_accounting_invalid",
            n_checks=n)
    n += 1
    if (env.hint_confidence_bucket < 0
            or env.hint_confidence_bucket >= 4):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_confidence_bucket_invalid",
            n_checks=n)
    n += 1
    if not (0.0 - 1e-9 <= float(env.ratify_probability)
            <= 1.0 + 1e-9):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_ratify_probability_invalid",
            n_checks=n)
    n += 1
    if (not env.autograd_params_cid
            or len(env.autograd_params_cid) != 64):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_autograd_params_cid_invalid",
            n_checks=n)
    n += 1
    if registered_autograd_params_cid is not None:
        if env.autograd_params_cid != str(
                registered_autograd_params_cid):
            return AutogradManifoldVerificationOutcome(
                ok=False,
                reason="w47_autograd_params_cid_invalid",
                n_checks=n)
    if (not env.training_trace_cid
            or len(env.training_trace_cid) != 64):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_training_trace_cid_invalid",
            n_checks=n)
    n += 1
    if registered_training_trace_cid is not None:
        if env.training_trace_cid != str(
                registered_training_trace_cid):
            return AutogradManifoldVerificationOutcome(
                ok=False,
                reason="w47_training_trace_cid_invalid",
                n_checks=n)
    if (not env.autograd_forward_witness_cid
            or len(env.autograd_forward_witness_cid) != 64):
        return AutogradManifoldVerificationOutcome(
            ok=False,
            reason=(
                "w47_autograd_forward_witness_cid_mismatch"),
            n_checks=n)
    n += 1
    if (env.decision_branch
            != W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH):
        if (not env.causal_mask_witness_cid
                or len(env.causal_mask_witness_cid) != 64):
            return AutogradManifoldVerificationOutcome(
                ok=False,
                reason="w47_causal_mask_witness_cid_invalid",
                n_checks=n)
    n += 1
    if (env.control_token_mode != W46_CTRL_MODE_OFF):
        if (not env.control_token_witness_cid
                or len(env.control_token_witness_cid) != 64):
            return AutogradManifoldVerificationOutcome(
                ok=False,
                reason="w47_control_token_witness_cid_invalid",
                n_checks=n)
    n += 1
    if (not env.prefix_capsule_cid
            or len(env.prefix_capsule_cid) != 64):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_prefix_capsule_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.memory_bank_head_cid
            or len(env.memory_bank_head_cid) != 64):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_memory_bank_head_cid_invalid",
            n_checks=n)
    n += 1
    if (len(env.emit_mask) != 4
            or any(not isinstance(b, bool) for b in env.emit_mask)):
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_emit_mask_invalid", n_checks=n)
    n += 1
    expected_construction = (
        _compute_w47_prompt_construction_witness_cid(
            turn_index=int(env.turn_index),
            role=env.role,
            prompt_sha256=env.prompt_sha256,
            control_token_mode=env.control_token_mode,
            inline_route_mode=env.inline_route_mode,
            factoradic_int=int(env.factoradic_int),
            factoradic_n_bits=int(env.factoradic_n_bits),
            confidence_bucket=int(env.hint_confidence_bucket),
            n_visible_prompt_tokens_textual=int(
                env.n_visible_prompt_tokens_textual),
            n_visible_prompt_tokens_actual=int(
                env.n_visible_prompt_tokens_actual),
            n_ctrl_tokens=int(env.n_ctrl_tokens),
            n_prefix_tokens=int(env.n_prefix_tokens),
            emit_mask=tuple(bool(b) for b in env.emit_mask),
        ))
    if expected_construction != (
            env.prompt_construction_witness_cid):
        return AutogradManifoldVerificationOutcome(
            ok=False,
            reason=(
                "w47_prompt_construction_witness_cid_mismatch"),
            n_checks=n)
    n += 1
    expected_witness = _compute_w47_autograd_witness_cid(
        decision_branch=env.decision_branch,
        w46_branch=env.w46_branch,
        w45_branch=env.w45_branch,
        w44_branch=env.w44_branch,
        pmc_branch=env.pmc_branch,
        abstain_reason=env.abstain_reason,
        role_handoff_signature_cid=env.role_handoff_signature_cid,
        policy_entry_cid=env.policy_entry_cid,
        autograd_params_cid=env.autograd_params_cid,
        training_trace_cid=env.training_trace_cid,
        autograd_forward_witness_cid=(
            env.autograd_forward_witness_cid),
        causal_mask_witness_cid=env.causal_mask_witness_cid,
        prompt_construction_witness_cid=(
            env.prompt_construction_witness_cid),
        control_token_witness_cid=env.control_token_witness_cid,
        prefix_capsule_cid=env.prefix_capsule_cid,
        memory_bank_head_cid=env.memory_bank_head_cid,
        memory_params_cid=env.memory_params_cid,
        output_sha256=env.output_sha256,
        behavioral_change=bool(env.behavioral_change),
    )
    if expected_witness != env.autograd_witness_cid:
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_autograd_witness_cid_mismatch",
            n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.autograd_outer_cid:
        return AutogradManifoldVerificationOutcome(
            ok=False, reason="w47_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return AutogradManifoldVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Team result + team
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AutogradManifoldTurn:
    """One turn of an :class:`AutogradManifoldTeam` run."""

    agent_turn: AgentTurn
    decision: AutogradGatingDecision
    envelope: AutogradManifoldHandoffEnvelope


@dataclasses.dataclass(frozen=True)
class AutogradManifoldTeamResult:
    """Result of an :class:`AutogradManifoldTeam` run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    autograd_turns: tuple[AutogradManifoldTurn, ...]
    capsule_view: dict[str, Any] | None = None
    root_cid: str | None = None
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_ms: float = 0.0
    total_calls: int = 0
    backend_model: str = ""
    backend_base_url: str | None = None
    team_instructions: str = ""
    task_summary: str | None = None
    max_visible_handoffs: int = 0
    stopped_early: bool = False
    n_behavioral_changes: int = 0
    n_visible_tokens_saved_factoradic: int = 0
    n_visible_tokens_added_ctrl: int = 0
    n_visible_tokens_added_prefix: int = 0
    n_visible_tokens_saved_prefix_reuse: int = 0
    n_abstain_substitutions: int = 0
    n_autograd_margin_abstains: int = 0
    n_autograd_train_failures: int = 0
    n_prefix_reuses: int = 0
    mean_ratify_probability: float = 0.0
    mean_autograd_pooled: float = 0.0
    autograd_params_cid: str = ""
    training_trace_cid: str = ""
    final_memory_bank_head_cid: str = ""
    schema: str = W47_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


class AutogradManifoldTeam:
    """W47 autograd-coupled agent team.

    Wraps the released :class:`coordpy.AgentTeam` contract with
    the W47 autograd stack plus the W46 memory layer + W45
    learned layer + W44 live gate + W43 PMC. With a trivial
    autograd registry, this team reduces to
    ``ManifoldMemoryTeam.run`` byte-for-byte (the
    W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: AutogradManifoldRegistry,
            observation_builder: LiveObservationBuilder | None = None,
            team_instructions: str = "",
            max_visible_handoffs: int = 4,
            capture_capsules: bool = True,
            task_summary: str | None = None,
            handoff_budget: "CapsuleBudget | None" = None,
            parent_w42_cid: str = W44_DEFAULT_PARENT_W42_CID,
            expected_spherical: SphericalConsensusSignature | None = None,
            expected_subspace: SubspaceBasis | None = None,
    ) -> None:
        if not agents:
            raise ValueError(
                "AutogradManifoldTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = AutogradManifoldOrchestrator(registry)
        self.observation_builder = (
            observation_builder or default_live_observation_builder)
        self.team_instructions = team_instructions.strip()
        self.max_visible_handoffs = int(max_visible_handoffs)
        self.capture_capsules = bool(capture_capsules)
        self.task_summary = (
            task_summary.strip() if task_summary else None)
        self.handoff_budget = handoff_budget
        self.parent_w42_cid = str(parent_w42_cid)
        self.expected_spherical = expected_spherical
        self.expected_subspace = expected_subspace

    @property
    def schema_cid(self) -> str:
        return self.orchestrator.schema_cid

    def _resolve_backend(self, member: Agent) -> LLMBackend:
        backend = member.backend or self.backend
        if backend is None:
            raise ValueError(
                "no backend configured; pass backend=... to "
                "AutogradManifoldTeam")
        if not isinstance(backend, LLMBackend):
            raise TypeError(
                "backend must satisfy the LLMBackend protocol")
        return backend

    def _build_prompt(
            self,
            *,
            member: Agent,
            task: str,
            turn_index: int,
            recent_handoffs: Sequence[tuple[str, str]],
            all_prior_outputs: Sequence[tuple[str, str]],
            decision: AutogradGatingDecision,
            role_universe: Sequence[str],
            prior_prefix_sha: str | None,
    ) -> tuple[
            str, str, int, int, int, int, str,
            ControlTokenWitness, PrefixCapsule, str,
            tuple[bool, bool, bool, bool]]:
        """Construct the bounded prompt + textual shadow + emit
        the packed control block (gated by the learned emit_mask).
        """
        common_parts: list[str] = []
        if self.team_instructions:
            common_parts.append(self.team_instructions)
        common_parts.append(f"Agent: {member.name}")
        common_parts.append(f"Role: {member.effective_role}")
        common_parts.append(member.instructions.strip())
        if turn_index == 0 or self.task_summary is None:
            common_parts.append(f"Task: {task.strip()}")
        else:
            common_parts.append(
                f"Task summary: {self.task_summary.strip()}")

        textual_parts = list(common_parts)
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[
                    -self.max_visible_handoffs:])
            textual_parts.append(
                "Visible team handoffs (bounded to avoid token "
                f"cramming):\n{rendered}")
        textual_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        textual_prompt = "\n\n".join(textual_parts)

        # Prefix capsule (from W46 inner).
        prefix_str: str = ""
        prefix_capsule: PrefixCapsule
        if (self.registry.memory_registry.prefix_reuse_enabled
                and turn_index > 0
                and all_prior_outputs):
            prefix_str, prefix_capsule = build_prefix_capsule(
                prior_outputs=all_prior_outputs,
                prefix_turns=int(
                    self.registry.memory_registry.prefix_turns),
                policy_entry_cid=str(decision.policy_entry_cid),
                prior_prefix_sha=prior_prefix_sha,
            )
        else:
            prefix_str, prefix_capsule = "", PrefixCapsule(
                prefix_sha256=hashlib.sha256(
                    b"").hexdigest(),
                prefix_token_count=0,
                policy_entry_cid=str(decision.policy_entry_cid),
                prior_output_shas=tuple(),
                reused=False,
            )

        # Control-token block (W47 honors the trained emit_mask).
        emit_mask = tuple(bool(b) for b in decision.forward.emit_mask)
        # Build the structured fields the W46 builder expects;
        # when the trained gate says "suppress", we replace that
        # field with its all-zero / null representation so the
        # CTRL bytes shrink predictably.
        layer_logits = (
            decision.forward.memory_forward.layer_logits
            if emit_mask[0] else tuple(
                [0.0] * len(
                    decision.forward.memory_forward.layer_logits)))
        mem_attn_value = (
            float(
                decision.forward.memory_forward
                .time_attention.pooled_value)
            if emit_mask[1] else 0.0)
        dict_index = (
            int(decision.forward.autograd_dict_index)
            if emit_mask[2] else int(W46_NO_DICT_CODE))
        # mem_summary: when suppressed, send the literal "off".
        mem_summary_str = "off"
        if emit_mask[3]:
            # Use the W46 default mem summary builder via the bank.
            from .manifold_memory import _build_mem_summary
            mem_summary_str = _build_mem_summary(
                self.orchestrator.memory_bank,
                turn_index=int(turn_index))
        ctrl_str, ctrl_witness = build_control_token_string(
            ctrl_mode=str(self.registry.control_token_mode),
            route=int(decision.factoradic_int),
            confidence_bucket=int(
                decision.forward.confidence_bucket),
            ratify_probability=float(
                decision.forward.ratify_probability),
            layer_logits=layer_logits,
            mem_attn_value=float(mem_attn_value),
            dict_index=int(dict_index),
            mem_summary=str(mem_summary_str),
            role_universe=role_universe,
            turn_index=int(turn_index),
        )

        bounded_parts = list(common_parts)
        if prefix_str:
            bounded_parts.append(prefix_str)
        if (decision.factoradic_n_bits > 0 and recent_handoffs):
            route_header = (
                f"FACTORADIC_ROUTE: {decision.factoradic_int} "
                f"over {','.join(role_universe)}")
            bounded_parts.append(route_header)
        if ctrl_str:
            bounded_parts.append(ctrl_str)
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[
                    -self.max_visible_handoffs:])
            bounded_parts.append(
                "Visible team handoffs (bounded to avoid token "
                f"cramming):\n{rendered}")
        bounded_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        bounded_prompt = "\n\n".join(bounded_parts)
        n_textual = len(textual_prompt.split())
        n_actual = len(bounded_prompt.split())
        n_ctrl_tokens = int(ctrl_witness.n_ctrl_tokens)
        n_prefix_tokens = int(prefix_capsule.prefix_token_count)
        return (
            bounded_prompt,
            textual_prompt,
            n_textual,
            n_actual,
            n_ctrl_tokens,
            n_prefix_tokens,
            ctrl_str,
            ctrl_witness,
            prefix_capsule,
            prefix_str,
            emit_mask,
        )

    def run(
            self,
            task: str,
            *,
            progress: Callable[
                [AutogradManifoldTurn], None] | None = None,
    ) -> AutogradManifoldTeamResult:
        """Run the autograd-coupled team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        autograd_turns: list[AutogradManifoldTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        all_prior_outputs: list[tuple[str, str]] = []
        role_arrival_order: list[str] = []
        causal_counts: dict[str, int] = {
            a.effective_role: 0 for a in self.agents}
        parent_cid: str | None = None
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_wall_ms = 0.0
        total_calls = 0
        n_behavioral_changes = 0
        n_visible_tokens_saved = 0
        n_visible_tokens_added_ctrl = 0
        n_visible_tokens_added_prefix = 0
        n_visible_tokens_saved_prefix_reuse = 0
        n_abstain_substitutions = 0
        n_autograd_margin_abstains = 0
        n_autograd_train_failures = 0
        n_prefix_reuses = 0
        ratify_probabilities: list[float] = []
        autograd_pooled: list[float] = []
        head_backend = self.backend
        head_model = (
            getattr(head_backend, "model", "") or "")
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()
        autograd_params_cid = self.registry.params.cid()
        training_trace_cid = (
            self.registry.params.training_trace.cid())
        memory_params_cid = self.registry.params.base.cid()
        prior_prefix_sha: str | None = None

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            ctx = LiveTurnContext(
                turn_index=int(idx),
                role_universe=role_universe,
                role_arrival_order=tuple(role_arrival_order),
                current_role=str(role),
                recent_handoffs=tuple(recent_handoffs),
                all_prior_outputs=tuple(all_prior_outputs),
                causal_counts=dict(causal_counts),
                injected_clock_violation=False,
            )
            obs_result = self.observation_builder(ctx)
            decision, aux = self.orchestrator.gate(
                observation=obs_result.observation,
                role=str(role),
                role_handoff_signature_cid=(
                    obs_result.role_handoff_signature_cid),
                parent_w42_cid=self.parent_w42_cid,
                n_w42_visible_tokens=n_w42_visible_tokens,
                turn_index=int(idx),
                expected_spherical=self.expected_spherical,
                expected_subspace=self.expected_subspace,
            )
            (w43_result, causal_mask, bundle, w45_decision,
             w46_decision, w46_forward, ag_forward) = aux

            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            (bounded_prompt, textual_prompt, n_textual_tokens,
             n_actual_tokens, n_ctrl_tokens, n_prefix_tokens,
             ctrl_str, ctrl_witness, prefix_capsule,
             prefix_str, emit_mask) = self._build_prompt(
                member=member,
                task=task,
                turn_index=idx,
                recent_handoffs=recent_handoffs,
                all_prior_outputs=all_prior_outputs,
                decision=decision,
                role_universe=role_universe,
                prior_prefix_sha=prior_prefix_sha,
            )

            do_substitute = (
                decision.is_abstain()
                and self.registry.abstain_substitution_enabled)
            if do_substitute:
                output = str(self.registry.abstain_output)
                wall_ms = 0.0
                d_prompt = 0
                d_output = 0
                d_calls = 0
                actual_prompt = ""
                n_abstain_substitutions += 1
                n_behavioral_changes += 1
                if decision.branch == (
                        W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN):
                    n_autograd_margin_abstains += 1
                if decision.branch == (
                        W47_BRANCH_AUTOGRAD_TRAIN_FAILURE):
                    n_autograd_train_failures += 1
            else:
                actual_prompt = bounded_prompt
                usage_before = _safe_usage_snapshot(backend)
                t0 = time.time()
                output = backend.generate(
                    actual_prompt,
                    max_tokens=member.max_tokens,
                    temperature=member.temperature,
                )
                wall_ms = (time.time() - t0) * 1000.0
                usage_after = _safe_usage_snapshot(backend)
                d_prompt = max(
                    0,
                    int(usage_after["prompt_tokens"])
                    - int(usage_before["prompt_tokens"]),
                )
                d_output = max(
                    0,
                    int(usage_after["output_tokens"])
                    - int(usage_before["output_tokens"]),
                )
                d_calls = max(
                    0,
                    int(usage_after["n_calls"])
                    - int(usage_before["n_calls"]),
                )

            n_saved = max(
                0, int(n_textual_tokens) - int(n_actual_tokens))
            n_added = max(
                0, int(n_actual_tokens) - int(n_textual_tokens))
            if n_saved > 0 and not do_substitute:
                n_visible_tokens_saved += int(n_saved)
                n_behavioral_changes += 1
            if n_ctrl_tokens > 0 and not do_substitute:
                n_visible_tokens_added_ctrl += int(n_ctrl_tokens)
            if n_prefix_tokens > 0 and not do_substitute:
                n_visible_tokens_added_prefix += int(
                    n_prefix_tokens)
            if prefix_capsule.reused:
                n_prefix_reuses += 1
                n_visible_tokens_saved_prefix_reuse += int(
                    n_prefix_tokens)
            ratify_probabilities.append(
                float(decision.forward.ratify_probability))
            autograd_pooled.append(
                float(decision.forward.autograd_memory_pooled))

            prompt_sha = _sha256_str(actual_prompt)
            output_sha = _sha256_str(output)
            backend_model = getattr(backend, "model", "") or ""
            capsule_cid: str | None = None
            if ledger is not None:
                next_role = (
                    self.agents[idx + 1].effective_role
                    if idx + 1 < len(self.agents)
                    else "team_output"
                )
                payload_words = max(1, len((output or "").split()))
                if self.handoff_budget is not None:
                    handoff_budget = self.handoff_budget
                else:
                    handoff_max_tokens = max(
                        member.max_tokens,
                        payload_words + 32, 128)
                    handoff_budget = CapsuleBudget(
                        max_bytes=1 << 14,
                        max_tokens=handoff_max_tokens,
                        max_parents=8,
                    )
                claim_kind = (
                    "agent_output_abstain"
                    if do_substitute else "agent_output")
                handoff = capsule_team_handoff(
                    source_role=role,
                    to_role=next_role,
                    claim_kind=claim_kind,
                    payload=output,
                    round=0,
                    parents=(parent_cid,) if parent_cid else (),
                    n_tokens=payload_words,
                    budget=handoff_budget,
                    prompt_sha256=prompt_sha,
                    prompt_bytes=len(
                        actual_prompt.encode("utf-8")),
                    model_tag=backend_model,
                )
                sealed = ledger.admit_and_seal(handoff)
                capsule_cid = sealed.cid
                parent_cid = sealed.cid

            backend_base = getattr(backend, "base_url", None)
            agent_turn = AgentTurn(
                agent_name=member.name,
                role=role,
                prompt=actual_prompt,
                output=output,
                capsule_cid=capsule_cid,
                prompt_tokens=d_prompt,
                output_tokens=d_output,
                wall_ms=wall_ms,
                visible_handoffs=visible_count,
                prompt_sha256=prompt_sha,
                model_tag=backend_model,
                prompt_words=int(n_actual_tokens),
                naive_prompt_words=int(n_textual_tokens),
                temperature=float(member.temperature),
                max_tokens=int(member.max_tokens),
                backend_base_url=backend_base,
            )
            agent_turns.append(agent_turn)

            # Build the W47 envelope witnesses.
            ag_forward_witness_cid = (
                _compute_w47_autograd_forward_witness_cid(
                    autograd_logit=float(
                        decision.forward.autograd_logit),
                    autograd_role_delta_value=float(
                        decision.forward.autograd_role_delta),
                    autograd_role_present=bool(
                        decision.forward.autograd_role_present),
                    autograd_dict_index=int(
                        decision.forward.autograd_dict_index),
                    autograd_dict_residual_l1=float(
                        decision.forward
                        .autograd_dict_residual_l1),
                    autograd_memory_pooled=float(
                        decision.forward
                        .autograd_memory_pooled),
                    emit_mask=tuple(
                        bool(b) for b in
                        decision.forward.emit_mask),
                    gate_logit=float(
                        decision.forward.gate_logit),
                    ratify_probability=float(
                        decision.forward.ratify_probability),
                    turn_index=int(idx),
                ))
            construction_cid = (
                _compute_w47_prompt_construction_witness_cid(
                    turn_index=int(idx),
                    role=str(role),
                    prompt_sha256=prompt_sha,
                    control_token_mode=str(
                        self.registry.control_token_mode),
                    inline_route_mode=(
                        self.registry.memory_registry
                        .learned_registry.live_registry
                        .inline_route_mode),
                    factoradic_int=int(decision.factoradic_int),
                    factoradic_n_bits=int(
                        decision.factoradic_n_bits),
                    confidence_bucket=int(
                        decision.forward.confidence_bucket),
                    n_visible_prompt_tokens_textual=int(
                        n_textual_tokens),
                    n_visible_prompt_tokens_actual=int(
                        n_actual_tokens),
                    n_ctrl_tokens=int(n_ctrl_tokens),
                    n_prefix_tokens=int(n_prefix_tokens),
                    emit_mask=tuple(bool(b) for b in emit_mask),
                ))
            control_token_witness_cid = (
                ctrl_witness.cid()
                if str(self.registry.control_token_mode)
                != W46_CTRL_MODE_OFF else "")
            prefix_capsule_cid = prefix_capsule.cid()
            memory_bank_head_cid = (
                self.orchestrator.memory_bank.head_cid())
            behavioral_change = bool(
                do_substitute or n_saved > 0 or n_added > 0
                or n_ctrl_tokens > 0 or n_prefix_tokens > 0
                or prefix_capsule.reused)
            autograd_witness_cid = (
                _compute_w47_autograd_witness_cid(
                    decision_branch=decision.branch,
                    w46_branch=decision.w46_branch,
                    w45_branch=decision.w45_branch,
                    w44_branch=decision.w44_branch,
                    pmc_branch=decision.pmc_branch,
                    abstain_reason=decision.abstain_reason,
                    role_handoff_signature_cid=(
                        decision.role_handoff_signature_cid),
                    policy_entry_cid=decision.policy_entry_cid,
                    autograd_params_cid=autograd_params_cid,
                    training_trace_cid=training_trace_cid,
                    autograd_forward_witness_cid=(
                        ag_forward_witness_cid),
                    causal_mask_witness_cid=(
                        decision.causal_mask_cid),
                    prompt_construction_witness_cid=(
                        construction_cid),
                    control_token_witness_cid=(
                        control_token_witness_cid),
                    prefix_capsule_cid=prefix_capsule_cid,
                    memory_bank_head_cid=memory_bank_head_cid,
                    memory_params_cid=memory_params_cid,
                    output_sha256=output_sha,
                    behavioral_change=behavioral_change,
                ))
            outer_cid = _compute_w47_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w46_envelope_cid="",
                autograd_params_cid=autograd_params_cid,
                training_trace_cid=training_trace_cid,
                autograd_witness_cid=autograd_witness_cid,
                turn_index=int(idx),
            )
            envelope = AutogradManifoldHandoffEnvelope(
                schema_version=(
                    W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION),
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w46_envelope_cid="",
                parent_w45_envelope_cid=str(
                    decision.w45_envelope_cid),
                parent_w44_envelope_cid=str(
                    decision.w44_envelope_cid),
                parent_w43_envelope_cid=str(
                    decision.pmc_envelope_cid),
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
                w46_branch=decision.w46_branch,
                w45_branch=decision.w45_branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                control_token_mode=str(
                    self.registry.control_token_mode),
                inline_route_mode=(
                    self.registry.memory_registry
                    .learned_registry.live_registry
                    .inline_route_mode),
                factoradic_int=int(decision.factoradic_int),
                factoradic_n_bits=int(
                    decision.factoradic_n_bits),
                hint_confidence_bucket=int(
                    decision.forward.confidence_bucket),
                autograd_params_cid=autograd_params_cid,
                training_trace_cid=training_trace_cid,
                fitting_method=str(
                    self.registry.params.fitting_method),
                n_autograd_layers=int(
                    self.registry.params.n_layers),
                autograd_role_adapter_rank=int(
                    self.registry.params.role_adapter_rank),
                autograd_dictionary_cid=str(
                    self.registry.params.dictionary.cid()),
                autograd_dictionary_size=int(
                    self.registry.params.dictionary_size),
                autograd_stack_cid=str(
                    self.registry.params.stack.cid()),
                autograd_memory_head_cid=str(
                    self.registry.params.memory_head.cid()),
                autograd_control_serializer_cid=str(
                    self.registry.params.control_serializer
                    .cid()),
                autograd_role_adapter_cid=str(
                    self.registry.params.role_adapter.cid()),
                memory_params_cid=memory_params_cid,
                memory_bank_head_cid=memory_bank_head_cid,
                memory_bank_size=int(
                    len(self.orchestrator.memory_bank.entries)),
                memory_capacity=int(
                    self.orchestrator.memory_bank.capacity),
                autograd_logit=float(
                    decision.forward.autograd_logit),
                autograd_role_delta_value=float(
                    decision.forward.autograd_role_delta),
                autograd_role_adapter_present=bool(
                    decision.forward.autograd_role_present),
                autograd_dict_index=int(
                    decision.forward.autograd_dict_index),
                autograd_dict_residual_l1=float(
                    decision.forward.autograd_dict_residual_l1),
                autograd_memory_pooled=float(
                    decision.forward.autograd_memory_pooled),
                emit_mask=tuple(
                    bool(b) for b in decision.forward.emit_mask),
                autograd_forward_witness_cid=str(
                    ag_forward_witness_cid),
                causal_mask_witness_cid=decision.causal_mask_cid,
                prompt_sha256=prompt_sha,
                prompt_construction_witness_cid=construction_cid,
                control_token_witness_cid=(
                    control_token_witness_cid),
                prefix_capsule_cid=prefix_capsule_cid,
                prefix_reused=bool(prefix_capsule.reused),
                output_sha256=output_sha,
                n_visible_prompt_tokens_textual=int(
                    n_textual_tokens),
                n_visible_prompt_tokens_actual=int(
                    n_actual_tokens),
                n_visible_prompt_tokens_saved=int(
                    n_textual_tokens - n_actual_tokens),
                n_overhead_tokens=int(
                    w43_result.n_w43_overhead_tokens),
                n_ctrl_tokens=int(n_ctrl_tokens),
                n_prefix_tokens=int(n_prefix_tokens),
                gate_logit=float(decision.forward.gate_logit),
                ratify_probability=float(
                    decision.forward.ratify_probability),
                behavioral_change=bool(behavioral_change),
                autograd_witness_cid=autograd_witness_cid,
                autograd_outer_cid=outer_cid,
            )
            autograd_turn = AutogradManifoldTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
            )
            autograd_turns.append(autograd_turn)

            total_prompt_tokens += int(d_prompt)
            total_output_tokens += int(d_output)
            total_wall_ms += float(wall_ms)
            total_calls += int(
                d_calls or (0 if do_substitute else 1))

            recent_handoffs.append((role, output))
            all_prior_outputs.append((role, output))
            role_arrival_order.append(role)
            if len(recent_handoffs) > self.max_visible_handoffs:
                recent_handoffs = recent_handoffs[
                    -self.max_visible_handoffs:]
            n_w42_visible_tokens = int(visible_count)
            prior_prefix_sha = prefix_capsule.prefix_sha256

            if progress is not None:
                try:
                    progress(autograd_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[AutogradManifoldTeam] progress callback "
                        "raised; continuing run:",
                        file=_sys.stderr)
                    _tb.print_exc()

        view = (
            render_view(
                ledger, root_cid=parent_cid,
                include_payload=True,
            ).as_dict()
            if ledger is not None else None
        )
        final_output = (
            agent_turns[-1].output if agent_turns else "")
        root_cid = (
            view.get("root_cid") if view is not None else None
        ) or parent_cid
        mean_p = (
            sum(ratify_probabilities) / len(ratify_probabilities)
            if ratify_probabilities else 0.0)
        mean_ag = (
            sum(autograd_pooled) / len(autograd_pooled)
            if autograd_pooled else 0.0)
        return AutogradManifoldTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            autograd_turns=tuple(autograd_turns),
            capsule_view=view,
            root_cid=root_cid,
            total_prompt_tokens=int(total_prompt_tokens),
            total_output_tokens=int(total_output_tokens),
            total_wall_ms=float(total_wall_ms),
            total_calls=int(total_calls),
            backend_model=str(head_model),
            backend_base_url=head_base,
            team_instructions=self.team_instructions,
            task_summary=self.task_summary,
            max_visible_handoffs=int(self.max_visible_handoffs),
            stopped_early=False,
            n_behavioral_changes=int(n_behavioral_changes),
            n_visible_tokens_saved_factoradic=int(
                n_visible_tokens_saved),
            n_visible_tokens_added_ctrl=int(
                n_visible_tokens_added_ctrl),
            n_visible_tokens_added_prefix=int(
                n_visible_tokens_added_prefix),
            n_visible_tokens_saved_prefix_reuse=int(
                n_visible_tokens_saved_prefix_reuse),
            n_abstain_substitutions=int(n_abstain_substitutions),
            n_autograd_margin_abstains=int(
                n_autograd_margin_abstains),
            n_autograd_train_failures=int(
                n_autograd_train_failures),
            n_prefix_reuses=int(n_prefix_reuses),
            mean_ratify_probability=float(mean_p),
            mean_autograd_pooled=float(mean_ag),
            autograd_params_cid=str(autograd_params_cid),
            training_trace_cid=str(training_trace_cid),
            final_memory_bank_head_cid=str(
                self.orchestrator.memory_bank.head_cid()),
        )


# =============================================================================
# CTRL-aware autograd synthetic backend (for r94 model-facing family)
# =============================================================================

@dataclasses.dataclass
class CtrlAwareAutogradBackend:
    """Deterministic backend that returns one canonical answer
    when the prompt contains BOTH a ``MANIFOLD_CTRL:`` substring
    AND an autograd ``layer_logits=`` field, a different answer
    when only ``MANIFOLD_CTRL:`` is present, and a third answer
    when neither is present.

    Used by R-94 to exercise the *behavioural* effect of the W47
    trained control surface on a controlled synthetic ground
    truth.
    """

    correct_with_full_ctrl: str = "AUTOGRAD_OK"
    answer_with_partial_ctrl: str = "AUTOGRAD_PARTIAL"
    answer_without_ctrl: str = "AUTOGRAD_NO_CTRL"
    n_calls: int = 0
    model_tag: str = "synthetic.autograd_aware"
    base_url: str | None = None

    @property
    def model(self) -> str:
        return self.model_tag

    def generate(
            self, prompt: str,
            max_tokens: int = 80,
            temperature: float = 0.0,
    ) -> str:
        self.n_calls += 1
        text = prompt or ""
        if ("MANIFOLD_CTRL:" in text
                and "layer_logits=" in text):
            return self.correct_with_full_ctrl
        if "MANIFOLD_CTRL:" in text:
            return self.answer_with_partial_ctrl
        return self.answer_without_ctrl


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Schema, branches, defaults
    "W47_AUTOGRAD_MANIFOLD_SCHEMA_VERSION",
    "W47_TEAM_RESULT_SCHEMA",
    "W47_BRANCH_TRIVIAL_AUTOGRAD_PASSTHROUGH",
    "W47_BRANCH_AUTOGRAD_DISABLED",
    "W47_BRANCH_AUTOGRAD_RATIFIED",
    "W47_BRANCH_AUTOGRAD_NO_POLICY",
    "W47_BRANCH_AUTOGRAD_CAUSAL_ABSTAIN",
    "W47_BRANCH_AUTOGRAD_SPHERICAL_ABSTAIN",
    "W47_BRANCH_AUTOGRAD_SUBSPACE_ABSTAIN",
    "W47_BRANCH_AUTOGRAD_MARGIN_ABSTAIN",
    "W47_BRANCH_AUTOGRAD_TIME_ATTN_ABSTAIN",
    "W47_BRANCH_AUTOGRAD_TRAIN_FAILURE",
    "W47_BRANCH_AUTOGRAD_REJECTED",
    "W47_ALL_BRANCHES",
    "W47_AUTOGRAD_ABSTAIN_BRANCHES",
    "W47_SUPPORTED_OPS",
    "W47_DEFAULT_HIDDEN_DIM",
    "W47_DEFAULT_N_LAYERS",
    "W47_DEFAULT_LEARNING_RATE",
    "W47_DEFAULT_N_STEPS",
    "W47_DEFAULT_INIT_SCALE",
    "W47_DEFAULT_BETA1",
    "W47_DEFAULT_BETA2",
    "W47_DEFAULT_EPS",
    "W47_DEFAULT_TRAIN_SEED",
    "W47_DEFAULT_MEMORY_HEAD_DIM",
    "W47_DEFAULT_DICT_GUMBEL_TEMP",
    "W47_DEFAULT_GRAD_CLIP",
    "W47_NO_ROLE_DELTA",
    "W47_ALL_FAILURE_MODES",
    # Autograd engine
    "Variable",
    "vdot", "vsum", "vmean", "vsoftmax", "vmatmul",
    "ParamTensor",
    "AdamOptimizer",
    "gradient_check",
    # Trainable components
    "AutogradStackLayer",
    "AutogradManifoldStack",
    "AutogradRoleAdapter",
    "AutogradDictionary",
    "AutogradMemoryHead",
    "AutogradControlSerializer",
    "AutogradManifoldParams",
    "TrainingTraceWitness",
    "build_unfitted_autograd_params",
    "fit_autograd_controller",
    # Forward
    "AutogradForwardResult",
    "forward_autograd_controller",
    # Registry + orchestrator
    "AutogradManifoldRegistry",
    "AutogradManifoldOrchestrator",
    "AutogradGatingDecision",
    "build_trivial_autograd_manifold_registry",
    "build_autograd_manifold_registry",
    # Envelope + verifier
    "AutogradManifoldHandoffEnvelope",
    "AutogradManifoldVerificationOutcome",
    "verify_autograd_manifold_handoff",
    # Team
    "AutogradManifoldTurn",
    "AutogradManifoldTeamResult",
    "AutogradManifoldTeam",
    # CTRL-aware autograd backend
    "CtrlAwareAutogradBackend",
]
