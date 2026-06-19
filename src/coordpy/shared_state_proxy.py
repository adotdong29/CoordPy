"""W48 Shared-State Transformer-Proxy (SSTP) — capsule-native,
autograd-trained, multi-head proxy attention with a pseudo-KV
factor bank, shared base state, reconstruction objective, and
branch/cycle-aware bias on top of W47 AMS, W46 MMC, W45 LMC,
W44 LMCC, and W43 PMC.

W48 is the first capsule-native CoordPy layer where:

  * A **single team-shared base state vector** lives across
    turns and roles, accessed by every role on every turn via
    a per-role low-rank LoRA-style delta.

  * A **pseudo-KV factor bank** of low-rank ``(K, V)`` tuples
    is read and written by a learned, causally-masked
    **multi-head proxy attention block**. The bank reproduces
    the algebraic interface of a transformer KV cache at the
    capsule layer — softmax(Q·K^T/sqrt(d))·V over admissible
    slots — but does NOT touch real KV bytes.

  * A learned **slot-memory write head** decides per turn
    whether the new observation should be appended to the
    pseudo-KV bank, modify an existing slot, or be discarded.

  * A learned **reconstruction decoder** reconstructs the
    previous turn's flat channel feature vector from the
    current shared state + the pseudo-KV read. L1/L2
    reconstruction distances are bound under the envelope.

  * A learned **branch/cycle-aware bias** adds a scalar
    correction for each (branch_id, cycle_id) pair to the
    proxy attention output — the first capsule-layer
    attention bias that explicitly encodes branch/cycle
    structure.

  * A **branch-history compressor** packs the team's branch
    path into a bijective integer header with explicit
    visible-token savings vs textual rendering.

  * A learned **latent control serializer** emits a single
    ``LATENT_CTRL: ...`` line carrying quantised slot indices
    + write-gate bits + branch-bias bits. Bijective from the
    witness CID.

All of W48 reuses the pure-Python reverse-mode autograd engine
from W47 (`Variable` + `AdamOptimizer`) — no NumPy, no
PyTorch, no JAX dependency. The released SDK v3.43 contract
remains byte-for-byte unchanged. W48 ships at
``coordpy.shared_state_proxy`` and is reachable only through
an explicit import.

Honest scope (do-not-overstate)
-------------------------------

W48 does NOT touch transformer-internal hidden state, KV cache
bytes, attention weights, or embeddings. Every parameter of
the shared-state proxy operates over W43 capsule-layer
encodings, the W47 trainable channel features, and the
pseudo-KV factor bank's *capsule-layer* slots. The pseudo-KV
bank reproduces the algebraic interface of a KV cache at the
capsule layer; it does not transplant real KV state from a
transformer's attention layers.

The substrate-blocked W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``) and the W47 carry-
forward ``W47-C-DEEP-TRANSFORMER-COUPLING`` are unchanged.

W48 is strictly additive on top of W47 and the released v3.43
SDK. When the proxy is configured trivially
(``proxy_enabled=False``, ``pseudo_kv_enabled=False``,
``reconstruction_enabled=False``, W47-trivial inner), the W48
orchestrator reduces to ``AutogradManifoldTeam.run`` byte-for-
byte — the W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH falsifier.
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
from .autograd_manifold import (
    AdamOptimizer,
    AutogradManifoldOrchestrator,
    AutogradManifoldRegistry,
    AutogradManifoldParams,
    AutogradManifoldTeam,
    AutogradManifoldTurn,
    ParamTensor,
    TrainingTraceWitness,
    Variable,
    W47_DEFAULT_BETA1,
    W47_DEFAULT_BETA2,
    W47_DEFAULT_EPS,
    W47_DEFAULT_GRAD_CLIP,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_LEARNING_RATE,
    W47_DEFAULT_N_STEPS,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
    build_autograd_manifold_registry,
    build_trivial_autograd_manifold_registry,
    build_unfitted_autograd_params,
    fit_autograd_controller,
    vdot,
    vmatmul,
    vmean,
    vsoftmax,
    vsum,
)
from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .learned_manifold import (
    TrainingExample,
    TrainingSet,
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
    W45_N_CHANNELS,
    _channel_features_from_bundle,
    _softmax,
)
from .live_manifold import (
    LiveObservationBuilder,
    LiveTurnContext,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_DEFAULT_PARENT_W42_CID,
    default_live_observation_builder,
)
from .llm_backend import LLMBackend
from .manifold_memory import (
    ManifoldMemoryBank,
    W46_CTRL_MODE_FULL,
    _flatten_channel_features,
)
from .product_manifold import (
    CellObservation,
    ProductManifoldPolicyEntry,
    SphericalConsensusSignature,
    SubspaceBasis,
)
from .team_coord import capsule_team_handoff


# =============================================================================
# Schema, branches, defaults
# =============================================================================

W48_SHARED_STATE_PROXY_SCHEMA_VERSION: str = (
    "coordpy.shared_state_proxy.v1")
W48_TEAM_RESULT_SCHEMA: str = (
    "coordpy.shared_state_proxy_team_result.v1")

# Decision branches: reuse W47 names + 6 W48-specific.
W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH: str = (
    "shared_state_trivial_passthrough")
W48_BRANCH_PROXY_DISABLED: str = "shared_state_proxy_disabled"
W48_BRANCH_PROXY_RATIFIED: str = "shared_state_proxy_ratified"
W48_BRANCH_PROXY_NO_POLICY: str = "shared_state_proxy_no_policy"
W48_BRANCH_PROXY_MARGIN_ABSTAIN: str = (
    "shared_state_proxy_margin_abstain")
W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN: str = (
    "shared_state_proxy_reconstruction_abstain")
W48_BRANCH_PROXY_BRANCH_BIAS_ABSTAIN: str = (
    "shared_state_proxy_branch_bias_abstain")
W48_BRANCH_PROXY_WRITE_GATE_ABSTAIN: str = (
    "shared_state_proxy_write_gate_abstain")
W48_BRANCH_PROXY_TRAIN_FAILURE: str = (
    "shared_state_proxy_train_failure")
W48_BRANCH_PROXY_REJECTED: str = "shared_state_proxy_rejected"

W48_ALL_BRANCHES: tuple[str, ...] = (
    W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH,
    W48_BRANCH_PROXY_DISABLED,
    W48_BRANCH_PROXY_RATIFIED,
    W48_BRANCH_PROXY_NO_POLICY,
    W48_BRANCH_PROXY_MARGIN_ABSTAIN,
    W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN,
    W48_BRANCH_PROXY_BRANCH_BIAS_ABSTAIN,
    W48_BRANCH_PROXY_WRITE_GATE_ABSTAIN,
    W48_BRANCH_PROXY_TRAIN_FAILURE,
    W48_BRANCH_PROXY_REJECTED,
)

W48_PROXY_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W48_BRANCH_PROXY_MARGIN_ABSTAIN,
    W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN,
    W48_BRANCH_PROXY_BRANCH_BIAS_ABSTAIN,
    W48_BRANCH_PROXY_WRITE_GATE_ABSTAIN,
    W48_BRANCH_PROXY_TRAIN_FAILURE,
})

# Defaults.
W48_DEFAULT_SHARED_STATE_DIM: int = 8
W48_DEFAULT_PSEUDO_KV_SLOTS: int = 6
W48_DEFAULT_FACTOR_DIM: int = 4
W48_DEFAULT_N_HEADS: int = 2
W48_DEFAULT_PROXY_HIDDEN_DIM: int = 16
W48_DEFAULT_PROXY_DEPTH: int = 2
W48_DEFAULT_N_BRANCHES: int = 4
W48_DEFAULT_N_CYCLES: int = 4
W48_DEFAULT_ROLE_STATE_DELTA_RANK: int = 2
W48_DEFAULT_RECON_HIDDEN_DIM: int = 16
W48_DEFAULT_LATENT_CTRL_BITS: int = 6
W48_DEFAULT_BRANCH_HISTORY_MAX_LEN: int = 8
W48_DEFAULT_WRITE_GATE_THRESHOLD: float = 0.5
W48_DEFAULT_RECON_MAX_L1: float = 1.0e9  # disabled by default
W48_DEFAULT_LATENT_CTRL_TAG: str = "LATENT_CTRL"
W48_DEFAULT_SHARED_STATE_TAG: str = "SHARED_STATE_HASH"

# Sentinels.
W48_NO_BRANCH_ID: int = -1
W48_NO_CYCLE_ID: int = -1
W48_NO_PSEUDO_KV_SLOT: int = -1
W48_NO_ROLE_STATE_DELTA: str = "no_role_state_delta"


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


def _l1(values: Sequence[float]) -> float:
    return float(sum(abs(float(v)) for v in values))


def _l2(values: Sequence[float]) -> float:
    return float(
        math.sqrt(sum(float(v) * float(v) for v in values)))


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# =============================================================================
# Shared base state capsule
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateCapsule:
    """Single team-shared base state vector with content-addressed
    CID.

    The vector lives across turns and roles. Every role reads it
    via a per-role rank-r LoRA-style delta; no role overwrites it.
    The CID is the SHA-256 over the rounded values + the
    initialisation seed.
    """

    dim: int
    values: tuple[float, ...]
    seed: int
    init_scale: float

    @classmethod
    def init(
            cls,
            *,
            dim: int = W48_DEFAULT_SHARED_STATE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "SharedStateCapsule":
        rng = _DeterministicLCG(seed=int(seed))
        vs = tuple(
            (rng.next_uniform() * 2.0 - 1.0) * float(init_scale)
            for _ in range(int(dim))
        )
        return cls(
            dim=int(dim), values=vs,
            seed=int(seed), init_scale=float(init_scale),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "values": _round_floats(self.values),
            "seed": int(self.seed),
            "init_scale": float(round(self.init_scale, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_shared_state_capsule",
            "capsule": self.to_dict()})

    def state_hash_short(self) -> str:
        """Short, model-facing hash of the shared state.

        Used as a deterministic, ~12-hex prefix for the
        ``SHARED_STATE_HASH`` prompt token. Stable for a given
        capsule.
        """
        return self.cid()[:12]


# =============================================================================
# Per-role rank-r shared-state delta
# =============================================================================

@dataclasses.dataclass
class RoleSharedStateDelta:
    """Rank-``r`` per-role correction to the shared base state.

    For each role with at least ``rank + 1`` training examples we
    store two factor tensors: ``U`` of shape ``(dim, rank)`` and
    ``V`` of shape ``(rank,)``. The per-role delta applied to the
    base state is ``U V``, an additive ``dim``-vector.
    """

    rank: int
    dim: int
    role_factors: dict[
        str, tuple[ParamTensor, ParamTensor]] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def init(
            cls,
            *,
            roles: Sequence[str],
            rank: int = W48_DEFAULT_ROLE_STATE_DELTA_RANK,
            dim: int = W48_DEFAULT_SHARED_STATE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "RoleSharedStateDelta":
        rng = _DeterministicLCG(seed=int(seed))
        rf: dict[str, tuple[ParamTensor, ParamTensor]] = {}
        for role in sorted(set(str(r) for r in roles)):
            u = ParamTensor(
                shape=(int(dim), int(rank)),
                values=[])
            u.init_seed(
                seed=int(rng.next_uniform() * (1 << 30)),
                scale=float(init_scale))
            v = ParamTensor(
                shape=(int(rank),),
                values=[0.0] * int(rank))
            rf[role] = (u, v)
        return cls(
            rank=int(rank), dim=int(dim), role_factors=rf)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for _, (u, v) in sorted(self.role_factors.items()):
            out.extend([u, v])
        return out

    def forward_value(
            self, *, role: str,
    ) -> tuple[float, ...]:
        """Compute the inference-time additive delta for ``role``.

        Returns the all-zero vector when the role is unknown.
        """
        key = str(role)
        if key not in self.role_factors:
            return tuple(0.0 for _ in range(self.dim))
        u, v = self.role_factors[key]
        # delta[j] = sum_i U[j, i] * V[i].
        delta: list[float] = []
        for j in range(self.dim):
            s = 0.0
            base = j * self.rank
            for i in range(self.rank):
                s += float(u.values[base + i]) * float(v.values[i])
            delta.append(s)
        return tuple(delta)

    def forward_vars(
            self, *, role: str,
    ) -> list[Variable]:
        """Compute the differentiable additive delta for ``role``.

        Returns Variables (one per dim) wired to U/V leaves so
        backward can flow into them.
        """
        key = str(role)
        if key not in self.role_factors:
            return [Variable(0.0) for _ in range(self.dim)]
        u, v = self.role_factors[key]
        u_vars = u.make_vars()
        v_vars = v.make_vars()
        out: list[Variable] = []
        for j in range(self.dim):
            row = [u_vars[j * self.rank + i]
                   for i in range(self.rank)]
            out.append(vdot(row, v_vars))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "dim": int(self.dim),
            "role_factors": {
                str(role): {
                    "U": u.to_dict(), "V": v.to_dict()}
                for role, (u, v) in sorted(
                    self.role_factors.items())
            },
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_role_shared_state_delta",
            "delta": self.to_dict()})


# =============================================================================
# Pseudo-KV factor bank
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PseudoKVSlot:
    """One slot of the pseudo-KV factor bank.

    Each slot carries a key vector, a value vector, and the
    metadata that lets an auditor re-derive the slot from the
    envelope chain.
    """

    slot_index: int
    turn_index: int
    role: str
    key: tuple[float, ...]
    value: tuple[float, ...]
    write_gate_value: float
    source_observation_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": int(self.slot_index),
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "key": _round_floats(self.key),
            "value": _round_floats(self.value),
            "write_gate_value": float(round(
                self.write_gate_value, 12)),
            "source_observation_cid":
                str(self.source_observation_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_pseudo_kv_slot",
            "slot": self.to_dict()})


@dataclasses.dataclass
class PseudoKVBank:
    """Bounded, content-addressed pseudo-KV factor bank.

    The bank reproduces the algebraic interface of a transformer
    KV cache at the capsule layer: each slot stores a key vector
    and a value vector; reads are softmax(Q·K^T/sqrt(d))·V over
    admissible slots. The bank does NOT touch real KV bytes.
    """

    capacity: int
    factor_dim: int
    slots: list[PseudoKVSlot] = dataclasses.field(
        default_factory=list)

    @property
    def size(self) -> int:
        return len(self.slots)

    def head_cid(self) -> str:
        """SHA-256 over the slot CIDs in order."""
        return _sha256_hex({
            "kind": "w48_pseudo_kv_bank_head",
            "capacity": int(self.capacity),
            "factor_dim": int(self.factor_dim),
            "slot_cids": [s.cid() for s in self.slots],
        })

    def admissible_for_turn(
            self, turn_index: int,
    ) -> tuple[PseudoKVSlot, ...]:
        """Strict causal mask: only slots with turn_index < t."""
        t = int(turn_index)
        return tuple(s for s in self.slots if s.turn_index < t)

    def write(self, slot: PseudoKVSlot) -> None:
        """Append a slot (ring buffer)."""
        self.slots.append(slot)
        while len(self.slots) > self.capacity:
            self.slots.pop(0)
            # Re-index slot_index for the remaining entries.
            for i, s in enumerate(self.slots):
                self.slots[i] = dataclasses.replace(
                    s, slot_index=i)

    def reset(self) -> None:
        self.slots = []


# =============================================================================
# Multi-head proxy attention block
# =============================================================================

@dataclasses.dataclass
class ProxyAttentionHead:
    """One head of the multi-head proxy attention block.

    Trainable QKV projections. Each head has its own
    ``(W_Q, W_K, W_V)``; the head dim is ``factor_dim``.
    """

    in_dim: int
    factor_dim: int
    w_query: ParamTensor
    w_key: ParamTensor
    w_value: ParamTensor

    @classmethod
    def init(
            cls,
            *,
            in_dim: int,
            factor_dim: int,
            seed: int,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ProxyAttentionHead":
        rng = _DeterministicLCG(seed=int(seed))
        wq = ParamTensor(
            shape=(int(factor_dim), int(in_dim)), values=[])
        wq.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        wk = ParamTensor(
            shape=(int(factor_dim), int(in_dim)), values=[])
        wk.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        wv = ParamTensor(
            shape=(int(factor_dim), int(in_dim)), values=[])
        wv.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        return cls(
            in_dim=int(in_dim), factor_dim=int(factor_dim),
            w_query=wq, w_key=wk, w_value=wv)

    def params(self) -> list[ParamTensor]:
        return [self.w_query, self.w_key, self.w_value]

    def _project_value(
            self, vec: Sequence[float], w: ParamTensor,
    ) -> list[float]:
        out: list[float] = []
        for r in range(self.factor_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                s += float(w.values[base + j]) * float(vec[j])
            out.append(s)
        return out

    def _project_vars(
            self,
            vec: Sequence[Variable],
            w_vars: Sequence[Variable],
    ) -> list[Variable]:
        rows: list[list[Variable]] = []
        for r in range(self.factor_dim):
            base = r * self.in_dim
            rows.append(list(
                w_vars[base:base + self.in_dim]))
        return vmatmul(rows, list(vec))

    def forward_value(
            self,
            *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
    ) -> tuple[list[float], list[float]]:
        """Inference forward: returns (pooled_value, attn_weights).

        ``pooled_value`` is a ``factor_dim``-vector. When there are
        no slots, returns an all-zero pooled value and an empty
        weight list.
        """
        if not slot_keys:
            return [0.0] * self.factor_dim, []
        q = self._project_value(query_input, self.w_query)
        ks = [self._project_value(k, self.w_key) for k in slot_keys]
        vs = [self._project_value(v, self.w_value) for v in slot_values]
        scale = 1.0 / math.sqrt(max(1.0, float(self.factor_dim)))
        scores = [
            sum(q[r] * k[r] for r in range(self.factor_dim)) * scale
            for k in ks]
        attn = _softmax(scores)
        pooled = [0.0] * self.factor_dim
        for i in range(len(attn)):
            for r in range(self.factor_dim):
                pooled[r] += attn[i] * vs[i][r]
        return pooled, [float(a) for a in attn]

    def forward_vars(
            self,
            *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
    ) -> tuple[list[Variable], list[Variable]]:
        """Training forward with autograd tape."""
        if not slot_keys:
            return [Variable(0.0)] * self.factor_dim, []
        wq_vars = self.w_query.make_vars()
        wk_vars = self.w_key.make_vars()
        wv_vars = self.w_value.make_vars()
        q = self._project_vars(query_input, wq_vars)
        ks = [self._project_vars(k, wk_vars) for k in slot_keys]
        vs = [self._project_vars(v, wv_vars) for v in slot_values]
        scale = 1.0 / math.sqrt(max(1.0, float(self.factor_dim)))
        scores: list[Variable] = []
        for k in ks:
            scores.append(vdot(q, k) * scale)
        attn = vsoftmax(scores)
        pooled: list[Variable] = []
        for r in range(self.factor_dim):
            comps: list[Variable] = []
            for i in range(len(attn)):
                comps.append(attn[i] * vs[i][r])
            pooled.append(vsum(comps))
        return pooled, list(attn)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "w_query": self.w_query.to_dict(),
            "w_key": self.w_key.to_dict(),
            "w_value": self.w_value.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_proxy_attention_head",
            "head": self.to_dict()})


@dataclasses.dataclass
class MultiHeadProxyAttention:
    """Multi-head proxy attention block over the pseudo-KV bank.

    Stacks ``n_heads`` heads in parallel; concatenates their
    pooled outputs; applies a trainable output projection back
    to ``in_dim``.
    """

    in_dim: int
    factor_dim: int
    n_heads: int
    heads: tuple[ProxyAttentionHead, ...]
    w_output: ParamTensor

    @classmethod
    def init(
            cls,
            *,
            in_dim: int,
            factor_dim: int = W48_DEFAULT_FACTOR_DIM,
            n_heads: int = W48_DEFAULT_N_HEADS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "MultiHeadProxyAttention":
        rng = _DeterministicLCG(seed=int(seed))
        heads: list[ProxyAttentionHead] = []
        for h in range(int(n_heads)):
            head_seed = int(rng.next_uniform() * (1 << 30))
            heads.append(ProxyAttentionHead.init(
                in_dim=int(in_dim),
                factor_dim=int(factor_dim),
                seed=int(head_seed),
                init_scale=float(init_scale),
            ))
        concat_dim = int(factor_dim) * int(n_heads)
        wo = ParamTensor(
            shape=(int(in_dim), int(concat_dim)),
            values=[])
        wo.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        return cls(
            in_dim=int(in_dim), factor_dim=int(factor_dim),
            n_heads=int(n_heads), heads=tuple(heads),
            w_output=wo)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for h in self.heads:
            out.extend(h.params())
        out.append(self.w_output)
        return out

    def forward_value(
            self,
            *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
    ) -> tuple[list[float], tuple[tuple[float, ...], ...]]:
        """Inference forward.

        Returns ``(output, per_head_attn_weights)``. ``output`` is
        an ``in_dim``-vector (residual stream output);
        ``per_head_attn_weights`` is a tuple-of-tuples of length
        ``n_heads`` with each element a tuple of per-slot weights.
        """
        if not slot_keys:
            # Returning all-zero output matches a no-op residual.
            return ([0.0] * self.in_dim,
                    tuple(tuple() for _ in range(self.n_heads)))
        head_pools: list[list[float]] = []
        per_head_attns: list[tuple[float, ...]] = []
        for h in self.heads:
            pool, attn = h.forward_value(
                query_input=query_input,
                slot_keys=slot_keys,
                slot_values=slot_values,
            )
            head_pools.append(pool)
            per_head_attns.append(tuple(attn))
        concat = []
        for p in head_pools:
            concat.extend(p)
        out = [0.0] * self.in_dim
        wo = self.w_output.values
        concat_dim = self.factor_dim * self.n_heads
        for r in range(self.in_dim):
            base = r * concat_dim
            s = 0.0
            for j in range(concat_dim):
                s += float(wo[base + j]) * float(concat[j])
            out[r] = s
        return out, tuple(per_head_attns)

    def forward_vars(
            self,
            *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
    ) -> list[Variable]:
        """Training forward; returns the residual-stream output."""
        head_pools: list[list[Variable]] = []
        for h in self.heads:
            pool, _ = h.forward_vars(
                query_input=query_input,
                slot_keys=slot_keys,
                slot_values=slot_values,
            )
            head_pools.append(pool)
        concat: list[Variable] = []
        for p in head_pools:
            concat.extend(p)
        wo_vars = self.w_output.make_vars()
        concat_dim = self.factor_dim * self.n_heads
        out: list[Variable] = []
        for r in range(self.in_dim):
            row = wo_vars[r * concat_dim:(r + 1) * concat_dim]
            out.append(vdot(list(row), concat))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "heads": [h.to_dict() for h in self.heads],
            "w_output": self.w_output.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_multi_head_proxy_attention",
            "attn": self.to_dict()})


# =============================================================================
# Slot-memory write head
# =============================================================================

@dataclasses.dataclass
class SlotMemoryWriteHead:
    """Learned write-gate over the pseudo-KV bank.

    A trainable sigmoid scalar per turn whose value decides
    whether the new observation enters the bank as a new slot.
    Inputs are the current shared state + flat channel features.
    """

    in_dim: int
    w_gate: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "SlotMemoryWriteHead":
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
            "kind": "w48_slot_memory_write_head",
            "head": self.to_dict()})


# =============================================================================
# Reconstruction decoder
# =============================================================================

@dataclasses.dataclass
class ReconstructionDecoder:
    """Trainable decoder that reconstructs the prior-turn flat
    channel feature vector from the current shared state +
    pseudo-KV read.

    Two-layer tanh -> linear stack of shape
    ``(in_dim) -> hidden -> recon_dim``.
    """

    in_dim: int
    hidden_dim: int
    recon_dim: int
    w_hidden: ParamTensor
    b_hidden: ParamTensor
    w_out: ParamTensor
    b_out: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            hidden_dim: int = W48_DEFAULT_RECON_HIDDEN_DIM,
            recon_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ReconstructionDecoder":
        rng = _DeterministicLCG(seed=int(seed))
        wh = ParamTensor(
            shape=(int(hidden_dim), int(in_dim)), values=[])
        wh.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        bh = ParamTensor(
            shape=(int(hidden_dim),),
            values=[0.0] * int(hidden_dim))
        wo = ParamTensor(
            shape=(int(recon_dim), int(hidden_dim)), values=[])
        wo.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        bo = ParamTensor(
            shape=(int(recon_dim),),
            values=[0.0] * int(recon_dim))
        return cls(
            in_dim=int(in_dim), hidden_dim=int(hidden_dim),
            recon_dim=int(recon_dim),
            w_hidden=wh, b_hidden=bh,
            w_out=wo, b_out=bo)

    def params(self) -> list[ParamTensor]:
        return [self.w_hidden, self.b_hidden,
                self.w_out, self.b_out]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> list[float]:
        hidden = [0.0] * self.hidden_dim
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(min(self.in_dim, len(inputs))):
                s += float(self.w_hidden.values[base + j]) \
                    * float(inputs[j])
            s += float(self.b_hidden.values[r])
            hidden[r] = math.tanh(s)
        out = [0.0] * self.recon_dim
        for r in range(self.recon_dim):
            base = r * self.hidden_dim
            s = 0.0
            for j in range(self.hidden_dim):
                s += float(self.w_out.values[base + j]) \
                    * float(hidden[j])
            s += float(self.b_out.values[r])
            out[r] = s
        return out

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> list[Variable]:
        wh_vars = self.w_hidden.make_vars()
        bh_vars = self.b_hidden.make_vars()
        wo_vars = self.w_out.make_vars()
        bo_vars = self.b_out.make_vars()
        # Hidden = tanh(W_h x + b_h).
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(wh_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, list(inputs))
        hidden = [
            (pre_h[i] + bh_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        # Out = W_o hidden + b_o.
        rows_o: list[list[Variable]] = []
        for r in range(self.recon_dim):
            base = r * self.hidden_dim
            rows_o.append(list(wo_vars[base:base + self.hidden_dim]))
        pre_o = vmatmul(rows_o, hidden)
        out = [pre_o[i] + bo_vars[i] for i in range(self.recon_dim)]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "recon_dim": int(self.recon_dim),
            "w_hidden": self.w_hidden.to_dict(),
            "b_hidden": self.b_hidden.to_dict(),
            "w_out": self.w_out.to_dict(),
            "b_out": self.b_out.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_reconstruction_decoder",
            "decoder": self.to_dict()})


# =============================================================================
# Branch / cycle bias
# =============================================================================

@dataclasses.dataclass
class BranchCycleBias:
    """Trainable scalar bias per ``(branch_id, cycle_id)`` pair.

    Stored as a single ``ParamTensor`` of shape
    ``(n_branches, n_cycles)``; the entry at ``(b, c)`` is a
    scalar added to the proxy attention output before the gate.
    """

    n_branches: int
    n_cycles: int
    bias: ParamTensor

    @classmethod
    def init(
            cls, *,
            n_branches: int = W48_DEFAULT_N_BRANCHES,
            n_cycles: int = W48_DEFAULT_N_CYCLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BranchCycleBias":
        b = ParamTensor(
            shape=(int(n_branches), int(n_cycles)),
            values=[])
        b.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(
            n_branches=int(n_branches),
            n_cycles=int(n_cycles), bias=b)

    def params(self) -> list[ParamTensor]:
        return [self.bias]

    def lookup_value(
            self, *, branch_id: int, cycle_id: int,
    ) -> float:
        b = int(branch_id) % max(1, self.n_branches)
        c = int(cycle_id) % max(1, self.n_cycles)
        idx = b * self.n_cycles + c
        if 0 <= idx < len(self.bias.values):
            return float(self.bias.values[idx])
        return 0.0

    def lookup_var(
            self, *, branch_id: int, cycle_id: int,
    ) -> Variable:
        b_vars = self.bias.make_vars()
        b = int(branch_id) % max(1, self.n_branches)
        c = int(cycle_id) % max(1, self.n_cycles)
        idx = b * self.n_cycles + c
        if 0 <= idx < len(b_vars):
            return b_vars[idx]
        return Variable(0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_branches": int(self.n_branches),
            "n_cycles": int(self.n_cycles),
            "bias": self.bias.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_branch_cycle_bias",
            "bias": self.to_dict()})


# =============================================================================
# Latent control serializer
# =============================================================================

@dataclasses.dataclass
class LatentControlSerializer:
    """Trainable gates over a packed ``LATENT_CTRL`` block.

    Each gate is a sigmoid scalar; the block emits one bit per
    above-0.5 gate. The mask + bit count are bound under the
    witness CID so the bytes round-trip exactly.
    """

    n_bits: int
    gates: ParamTensor

    @classmethod
    def init(
            cls, *,
            n_bits: int = W48_DEFAULT_LATENT_CTRL_BITS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
    ) -> "LatentControlSerializer":
        # Default initialisation at logit 0.5 to emit-all-by-
        # default — matches W47's control serializer convention.
        g = ParamTensor(
            shape=(int(n_bits),),
            values=[0.5] * int(n_bits))
        return cls(n_bits=int(n_bits), gates=g)

    def params(self) -> list[ParamTensor]:
        return [self.gates]

    def emit_mask(self) -> tuple[bool, ...]:
        return tuple(
            bool(_stable_sigmoid(v) >= 0.5)
            for v in self.gates.values)

    def forward_loss_vars(
            self, *, target_mask: Sequence[bool],
    ) -> Variable:
        gs = self.gates.make_vars()
        losses: list[Variable] = []
        for i in range(min(self.n_bits, len(target_mask))):
            s = gs[i].sigmoid()
            if bool(target_mask[i]):
                losses.append(-1.0 * (s + 1e-9).log())
            else:
                losses.append(
                    -1.0 * ((Variable(1.0) - s) + 1e-9).log())
        return vmean(losses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bits": int(self.n_bits),
            "gates": self.gates.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_latent_control_serializer",
            "ser": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LatentControlWitness:
    """Sealed witness binding the LATENT_CTRL bytes to the env.

    The auditor re-builds the bytes from the witness fields and
    re-hashes the witness CID; mismatches surface as
    ``w48_latent_control_witness_cid_invalid``.
    """

    ctrl_tag: str
    n_bits: int
    emit_mask: tuple[bool, ...]
    bits_payload: tuple[int, ...]
    shared_state_hash_short: str
    n_ctrl_tokens: int
    ctrl_bytes_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctrl_tag": str(self.ctrl_tag),
            "n_bits": int(self.n_bits),
            "emit_mask": [bool(b) for b in self.emit_mask],
            "bits_payload": [int(b) for b in self.bits_payload],
            "shared_state_hash_short":
                str(self.shared_state_hash_short),
            "n_ctrl_tokens": int(self.n_ctrl_tokens),
            "ctrl_bytes_sha256": str(self.ctrl_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_latent_control_witness",
            "witness": self.to_dict()})


def build_latent_control_string(
        *,
        ctrl_tag: str,
        emit_mask: Sequence[bool],
        bits_payload: Sequence[int],
        shared_state_hash_short: str,
) -> tuple[str, LatentControlWitness]:
    """Build the literal ``LATENT_CTRL: ...`` line + the witness.

    The line carries:
      * ``SHARED_STATE_HASH=<12-hex>``
      * ``mask=<binary mask>``
      * ``bits=<bit string>``
    """
    mask_str = "".join("1" if b else "0" for b in emit_mask)
    bits_str = "".join(
        str(int(b) & 1) for b in bits_payload)
    body = (
        f"{ctrl_tag}: "
        f"SHARED_STATE_HASH={shared_state_hash_short} "
        f"mask={mask_str} bits={bits_str}")
    n_ctrl_tokens = len(body.split())
    ctrl_sha = hashlib.sha256(
        body.encode("utf-8")).hexdigest()
    witness = LatentControlWitness(
        ctrl_tag=str(ctrl_tag),
        n_bits=int(len(emit_mask)),
        emit_mask=tuple(bool(b) for b in emit_mask),
        bits_payload=tuple(int(b) & 1 for b in bits_payload),
        shared_state_hash_short=str(shared_state_hash_short),
        n_ctrl_tokens=int(n_ctrl_tokens),
        ctrl_bytes_sha256=str(ctrl_sha),
    )
    return body, witness


# =============================================================================
# Branch-history compressor
# =============================================================================

@dataclasses.dataclass(frozen=True)
class BranchHistoryWitness:
    """Sealed witness for the branch-history compression."""

    branch_path: tuple[int, ...]
    cycle_path: tuple[int, ...]
    packed_integer: int
    packed_n_bits: int
    textual_tokens: int
    compressed_tokens: int
    compressor_bytes_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_path": [int(b) for b in self.branch_path],
            "cycle_path": [int(c) for c in self.cycle_path],
            "packed_integer": int(self.packed_integer),
            "packed_n_bits": int(self.packed_n_bits),
            "textual_tokens": int(self.textual_tokens),
            "compressed_tokens": int(self.compressed_tokens),
            "compressor_bytes_sha256":
                str(self.compressor_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_branch_history_witness",
            "witness": self.to_dict()})


def compress_branch_history(
        *,
        branch_path: Sequence[int],
        cycle_path: Sequence[int],
        n_branches: int = W48_DEFAULT_N_BRANCHES,
        n_cycles: int = W48_DEFAULT_N_CYCLES,
        max_len: int = W48_DEFAULT_BRANCH_HISTORY_MAX_LEN,
) -> tuple[str, BranchHistoryWitness]:
    """Pack the branch + cycle history into a bijective integer.

    Returns ``(compressed_text, witness)``. The compressed text is
    a single ``BRANCH_HIST: <int> over <base>`` line.

    The packing: interleave ``(branch_id[i], cycle_id[i])`` pairs
    truncated to ``max_len``. Each pair occupies
    ``ceil(log2(n_branches)) + ceil(log2(n_cycles))`` bits.
    """
    bp = tuple(int(b) % max(1, n_branches)
               for b in list(branch_path)[:max_len])
    cp = tuple(int(c) % max(1, n_cycles)
               for c in list(cycle_path)[:max_len])
    n = min(len(bp), len(cp))
    bits_per_branch = max(
        1, int(math.ceil(math.log2(max(2, n_branches)))))
    bits_per_cycle = max(
        1, int(math.ceil(math.log2(max(2, n_cycles)))))
    bits_per_pair = bits_per_branch + bits_per_cycle
    packed = 0
    n_bits = 0
    for i in range(n):
        packed = (packed << bits_per_branch) | bp[i]
        packed = (packed << bits_per_cycle) | cp[i]
        n_bits += bits_per_pair
    text = (
        f"BRANCH_HIST: {int(packed)} "
        f"over {int(n_branches)}x{int(n_cycles)}")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Textual tokens: 'branch=<b>,cycle=<c>' per pair separated.
    textual_n_tokens = max(1, n * 2)  # 'b=#' + 'c=#' per pair
    compressed_n_tokens = len(text.split())
    witness = BranchHistoryWitness(
        branch_path=tuple(bp),
        cycle_path=tuple(cp),
        packed_integer=int(packed),
        packed_n_bits=int(n_bits),
        textual_tokens=int(textual_n_tokens),
        compressed_tokens=int(compressed_n_tokens),
        compressor_bytes_sha256=str(sha),
    )
    return text, witness


def decompress_branch_history(
        *,
        packed_integer: int,
        n_pairs: int,
        n_branches: int = W48_DEFAULT_N_BRANCHES,
        n_cycles: int = W48_DEFAULT_N_CYCLES,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inverse of :func:`compress_branch_history`."""
    bits_per_branch = max(
        1, int(math.ceil(math.log2(max(2, n_branches)))))
    bits_per_cycle = max(
        1, int(math.ceil(math.log2(max(2, n_cycles)))))
    cyc_mask = (1 << bits_per_cycle) - 1
    bra_mask = (1 << bits_per_branch) - 1
    branches: list[int] = []
    cycles: list[int] = []
    p = int(packed_integer)
    for _ in range(int(n_pairs)):
        c = p & cyc_mask
        p >>= bits_per_cycle
        b = p & bra_mask
        p >>= bits_per_branch
        cycles.append(int(c))
        branches.append(int(b))
    branches.reverse()
    cycles.reverse()
    return tuple(branches), tuple(cycles)


# =============================================================================
# Shared-state proxy params bundle + training trace
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateProxyParams:
    """All trained, content-addressed W48 params.

    Wraps the W47 inner ``AutogradManifoldParams`` + W48-specific
    new params.
    """

    inner_w47: AutogradManifoldParams
    shared_state: SharedStateCapsule
    role_state_delta: RoleSharedStateDelta
    proxy_attention: MultiHeadProxyAttention
    write_head: SlotMemoryWriteHead
    reconstruction: ReconstructionDecoder
    branch_cycle_bias: BranchCycleBias
    latent_control: LatentControlSerializer
    fitting_method: str
    training_trace: TrainingTraceWitness

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_w47_cid": str(self.inner_w47.cid()),
            "shared_state": self.shared_state.to_dict(),
            "role_state_delta":
                self.role_state_delta.to_dict(),
            "proxy_attention":
                self.proxy_attention.to_dict(),
            "write_head": self.write_head.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "branch_cycle_bias":
                self.branch_cycle_bias.to_dict(),
            "latent_control": self.latent_control.to_dict(),
            "fitting_method": str(self.fitting_method),
            "training_trace_cid":
                self.training_trace.cid(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_shared_state_proxy_params",
            "params": self.to_dict()})

    @property
    def shared_state_dim(self) -> int:
        return int(self.shared_state.dim)

    @property
    def factor_dim(self) -> int:
        return int(self.proxy_attention.factor_dim)

    @property
    def n_heads(self) -> int:
        return int(self.proxy_attention.n_heads)


def build_unfitted_shared_state_proxy_params(
        *,
        inner_w47: AutogradManifoldParams | None = None,
        roles: Sequence[str] = (),
        shared_state_dim: int = W48_DEFAULT_SHARED_STATE_DIM,
        factor_dim: int = W48_DEFAULT_FACTOR_DIM,
        n_heads: int = W48_DEFAULT_N_HEADS,
        role_state_delta_rank: int = (
            W48_DEFAULT_ROLE_STATE_DELTA_RANK),
        recon_dim: int | None = None,
        recon_hidden_dim: int = W48_DEFAULT_RECON_HIDDEN_DIM,
        n_branches: int = W48_DEFAULT_N_BRANCHES,
        n_cycles: int = W48_DEFAULT_N_CYCLES,
        latent_ctrl_bits: int = W48_DEFAULT_LATENT_CTRL_BITS,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
) -> SharedStateProxyParams:
    """Build a fully-initialised (but untrained) W48 bundle.

    The W47 inner is initialised to its 'unfitted' state by
    default; pass an already-fit ``AutogradManifoldParams`` to
    reuse a trained W47.
    """
    inner = inner_w47 or build_unfitted_autograd_params()
    ss = SharedStateCapsule.init(
        dim=int(shared_state_dim), seed=int(seed),
        init_scale=float(init_scale),
    )
    rsd = RoleSharedStateDelta.init(
        roles=tuple(roles),
        rank=int(role_state_delta_rank),
        dim=int(shared_state_dim),
        seed=int(seed) + 11,
        init_scale=float(init_scale),
    )
    # Proxy attention input dim = shared_state_dim + flat_channels
    # + factor_dim (the prior pseudo-KV read).
    flat_channel_dim = int(W45_N_CHANNELS) * int(feature_dim)
    proxy_in_dim = (
        int(shared_state_dim) + flat_channel_dim
        + int(factor_dim))
    proxy = MultiHeadProxyAttention.init(
        in_dim=int(proxy_in_dim),
        factor_dim=int(factor_dim),
        n_heads=int(n_heads),
        seed=int(seed) + 21,
        init_scale=float(init_scale),
    )
    wh_in_dim = int(shared_state_dim) + flat_channel_dim
    wh = SlotMemoryWriteHead.init(
        in_dim=int(wh_in_dim),
        seed=int(seed) + 31,
        init_scale=float(init_scale),
    )
    rd = int(recon_dim if recon_dim is not None
             else flat_channel_dim)
    recon = ReconstructionDecoder.init(
        in_dim=int(proxy_in_dim),
        hidden_dim=int(recon_hidden_dim),
        recon_dim=int(rd),
        seed=int(seed) + 41,
        init_scale=float(init_scale),
    )
    bcb = BranchCycleBias.init(
        n_branches=int(n_branches),
        n_cycles=int(n_cycles),
        seed=int(seed) + 51,
        init_scale=float(init_scale),
    )
    lcs = LatentControlSerializer.init(
        n_bits=int(latent_ctrl_bits),
        seed=int(seed) + 61,
    )
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
    return SharedStateProxyParams(
        inner_w47=inner,
        shared_state=ss,
        role_state_delta=rsd,
        proxy_attention=proxy,
        write_head=wh,
        reconstruction=recon,
        branch_cycle_bias=bcb,
        latent_control=lcs,
        fitting_method="unfitted",
        training_trace=trace,
    )


# =============================================================================
# Training set helper for W48 proxy regimes
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateExample:
    """One training example for the W48 proxy regimes.

    Carries the W45 channel features (same shape as W45 / W46 /
    W47), plus the branch/cycle ids and a target label.
    """

    role: str
    channel_features: tuple[tuple[str, tuple[float, ...]], ...]
    branch_id: int
    cycle_id: int
    label: float
    target_recon: tuple[float, ...] | None = None
    write_target: float = 1.0  # gold gate value in [0, 1]

    @property
    def channel_features_map(self) -> dict[str, tuple[float, ...]]:
        return {str(c): tuple(v) for c, v in self.channel_features}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "channel_features": [
                [c, list(v)] for c, v in self.channel_features],
            "branch_id": int(self.branch_id),
            "cycle_id": int(self.cycle_id),
            "label": float(self.label),
            "target_recon": (
                list(self.target_recon)
                if self.target_recon is not None else None),
            "write_target": float(self.write_target),
        }


@dataclasses.dataclass(frozen=True)
class SharedStateTrainingSet:
    """A bounded, content-addressed set of training examples for
    the W48 proxy regimes."""

    examples: tuple[SharedStateExample, ...]
    feature_dim: int = W45_DEFAULT_FEATURE_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "examples": [e.to_dict() for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w48_shared_state_training_set",
            "set": self.to_dict()})


def _flatten_example(
        ex: SharedStateExample,
        *,
        feature_dim: int,
) -> list[float]:
    fmap = ex.channel_features_map
    out: list[float] = []
    for c in W45_CHANNEL_ORDER:
        feats = list(fmap.get(c, ()))[:feature_dim]
        while len(feats) < feature_dim:
            feats.append(0.0)
        out.extend(float(v) for v in feats)
    return out


# =============================================================================
# Trainer
# =============================================================================

def fit_shared_state_proxy(
        training_set: SharedStateTrainingSet,
        *,
        inner_w47: AutogradManifoldParams | None = None,
        shared_state_dim: int = W48_DEFAULT_SHARED_STATE_DIM,
        factor_dim: int = W48_DEFAULT_FACTOR_DIM,
        n_heads: int = W48_DEFAULT_N_HEADS,
        role_state_delta_rank: int = (
            W48_DEFAULT_ROLE_STATE_DELTA_RANK),
        recon_hidden_dim: int = W48_DEFAULT_RECON_HIDDEN_DIM,
        n_branches: int = W48_DEFAULT_N_BRANCHES,
        n_cycles: int = W48_DEFAULT_N_CYCLES,
        latent_ctrl_bits: int = W48_DEFAULT_LATENT_CTRL_BITS,
        n_steps: int = W47_DEFAULT_N_STEPS,
        learning_rate: float = W47_DEFAULT_LEARNING_RATE,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        recon_loss_weight: float = 0.5,
        write_gate_loss_weight: float = 0.5,
        branch_bias_loss_weight: float = 0.5,
        history_head: int = 8,
        history_tail: int = 8,
) -> SharedStateProxyParams:
    """Fit the W48 shared-state proxy via Adam SGD.

    Jointly trains:
      * the multi-head proxy attention block on the binary
        cross-entropy gate loss;
      * the role-state delta on the per-role residual;
      * the slot-memory write head on the per-example
        ``write_target`` value (binary cross-entropy);
      * the reconstruction decoder on the per-example
        ``target_recon`` vector via L2;
      * the branch/cycle bias on the per-example
        ``(branch_id, cycle_id)`` regime.

    Returns the trained :class:`SharedStateProxyParams` bundle
    with a sealed training-trace witness CID.
    """
    fd = int(training_set.feature_dim)
    inner = inner_w47 or build_unfitted_autograd_params()
    roles = tuple(
        sorted({str(ex.role) for ex in training_set.examples}))
    params_bundle = build_unfitted_shared_state_proxy_params(
        inner_w47=inner,
        roles=roles,
        shared_state_dim=int(shared_state_dim),
        factor_dim=int(factor_dim),
        n_heads=int(n_heads),
        role_state_delta_rank=int(role_state_delta_rank),
        recon_dim=int(W45_N_CHANNELS) * fd,
        recon_hidden_dim=int(recon_hidden_dim),
        n_branches=int(n_branches),
        n_cycles=int(n_cycles),
        latent_ctrl_bits=int(latent_ctrl_bits),
        feature_dim=fd,
        seed=int(seed),
        init_scale=float(init_scale),
    )

    # Pre-flatten examples + targets.
    flat_examples: list[list[float]] = []
    labels_pos: list[bool] = []
    write_targets: list[float] = []
    recon_targets: list[list[float]] = []
    branch_ids: list[int] = []
    cycle_ids: list[int] = []
    roles_list: list[str] = []
    for ex in training_set.examples:
        flat = _flatten_example(ex, feature_dim=fd)
        flat_examples.append(flat)
        labels_pos.append(bool(float(ex.label) > 0.0))
        write_targets.append(float(ex.write_target))
        if ex.target_recon is not None:
            tr = list(ex.target_recon)
        else:
            tr = list(flat)
        while len(tr) < int(W45_N_CHANNELS) * fd:
            tr.append(0.0)
        recon_targets.append(tr[:int(W45_N_CHANNELS) * fd])
        branch_ids.append(int(ex.branch_id))
        cycle_ids.append(int(ex.cycle_id))
        roles_list.append(str(ex.role))

    # Adam optimiser over W48 trainable params.
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
        grad_clip=float(grad_clip),
    )
    trainable: list[ParamTensor] = []
    trainable.extend(params_bundle.proxy_attention.params())
    trainable.extend(params_bundle.role_state_delta.params())
    trainable.extend(params_bundle.write_head.params())
    trainable.extend(params_bundle.reconstruction.params())
    trainable.extend(params_bundle.branch_cycle_bias.params())

    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False

    # Pre-stage the shared-state values as Variables for forward.
    shared_values = list(params_bundle.shared_state.values)
    flat_channel_dim = int(W45_N_CHANNELS) * fd
    proxy_in_dim = (
        int(shared_state_dim) + flat_channel_dim
        + int(factor_dim))

    # Per-example SGD: one Adam step per example. This avoids
    # the "make_vars overwrites self._vars" issue that would
    # otherwise only pass the last example's gradient to the
    # optimiser. Each step processes a single example end-to-end.
    n_examples = len(flat_examples)
    for step in range(int(n_steps)):
        # Ensure every trainable param has a fresh _vars slot at
        # the start of the step. This means the optimiser will
        # see grad=0.0 for params whose forward path was not
        # exercised this step (rather than IndexError).
        for p in trainable:
            p.make_vars()
        # Cycle through examples in order; mid-step we use the
        # current example.
        ex_idx = step % max(1, n_examples)
        flat = flat_examples[ex_idx]
        role = roles_list[ex_idx]
        # Build the shared-state-with-delta vector.
        delta_vars = params_bundle.role_state_delta.forward_vars(
            role=role)
        shared_state_with_delta: list[Variable] = []
        for j in range(int(shared_state_dim)):
            if j < len(delta_vars):
                shared_state_with_delta.append(
                    Variable(float(shared_values[j]))
                    + delta_vars[j])
            else:
                shared_state_with_delta.append(
                    Variable(float(shared_values[j])))
        flat_vars = [Variable(float(v)) for v in flat]
        pkv_read_vars = [
            Variable(0.0) for _ in range(int(factor_dim))]
        query_vars = (
            shared_state_with_delta + flat_vars + pkv_read_vars)
        slot_keys = [
            [Variable(float(v)) for v in flat][:proxy_in_dim]
            + [Variable(0.0)] * max(
                0, proxy_in_dim - flat_channel_dim
                - int(shared_state_dim))]
        sk = slot_keys[0]
        while len(sk) < proxy_in_dim:
            sk.append(Variable(0.0))
        sk = sk[:proxy_in_dim]
        slot_values = [sk]
        proxy_out_vars = (
            params_bundle.proxy_attention.forward_vars(
                query_input=query_vars,
                slot_keys=[sk],
                slot_values=slot_values,
            ))
        gate_logit = vsum(proxy_out_vars) * (
            1.0 / float(max(1, proxy_in_dim)))
        bcb_var = params_bundle.branch_cycle_bias.lookup_var(
            branch_id=branch_ids[ex_idx],
            cycle_id=cycle_ids[ex_idx])
        gate_logit = gate_logit + bcb_var
        prob = gate_logit.sigmoid()
        if labels_pos[ex_idx]:
            cls_loss = -1.0 * (prob + 1e-9).log()
        else:
            cls_loss = -1.0 * (
                (Variable(1.0) - prob) + 1e-9).log()
        # Write-gate loss with a fresh forward (independent of
        # the gate path so the write head trains on its own
        # signal).
        wh_flat_vars = [Variable(float(v)) for v in flat]
        wh_shared_vars = [
            Variable(float(shared_values[j]))
            for j in range(int(shared_state_dim))]
        wh_input = (wh_shared_vars + wh_flat_vars)
        while len(wh_input) < params_bundle.write_head.in_dim:
            wh_input.append(Variable(0.0))
        wh_input = wh_input[:params_bundle.write_head.in_dim]
        wh_prob = params_bundle.write_head.forward_vars(wh_input)
        wt = float(write_targets[ex_idx])
        if wt > 0.5:
            wh_loss = -1.0 * (wh_prob + 1e-9).log()
        else:
            wh_loss = -1.0 * (
                (Variable(1.0) - wh_prob) + 1e-9).log()
        # Reconstruction L2 loss with its own fresh forward.
        rec_query_vars = (
            list(shared_state_with_delta)
            + [Variable(float(v)) for v in flat]
            + [Variable(0.0) for _ in range(int(factor_dim))])
        while len(rec_query_vars) < params_bundle.reconstruction.in_dim:
            rec_query_vars.append(Variable(0.0))
        rec_query_vars = rec_query_vars[
            :params_bundle.reconstruction.in_dim]
        recon_out = (
            params_bundle.reconstruction.forward_vars(rec_query_vars))
        target = recon_targets[ex_idx]
        recon_terms: list[Variable] = []
        for j in range(len(recon_out)):
            tj = target[j] if j < len(target) else 0.0
            diff = recon_out[j] - Variable(float(tj))
            recon_terms.append(diff * diff)
        recon_loss = vmean(recon_terms)
        loss = (cls_loss
                + write_gate_loss_weight * wh_loss
                + recon_loss_weight * recon_loss
                + branch_bias_loss_weight
                * (bcb_var * bcb_var))
        loss.backward()

        total_grad_sq = 0.0
        for p in trainable:
            for g in p.grads():
                total_grad_sq += float(g) * float(g)
        gn = math.sqrt(total_grad_sq)
        grad_norm_history.append(float(gn))
        loss_history.append(float(loss.value))

        if (loss.value != loss.value
                or loss.value == float("inf")
                or loss.value == float("-inf")):
            diverged = True
            break

        optim.step(trainable)

    final_loss = float(
        loss_history[-1] if loss_history else 0.0)
    final_gn = float(
        grad_norm_history[-1] if grad_norm_history else 0.0)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    loss_head = tuple(loss_history[:head_n])
    loss_tail = (
        tuple(loss_history[-tail_n:]) if tail_n > 0 else tuple())
    gn_head = tuple(grad_norm_history[:head_n])
    gn_tail = (
        tuple(grad_norm_history[-tail_n:])
        if tail_n > 0 else tuple())

    fitted = SharedStateProxyParams(
        inner_w47=inner,
        shared_state=params_bundle.shared_state,
        role_state_delta=params_bundle.role_state_delta,
        proxy_attention=params_bundle.proxy_attention,
        write_head=params_bundle.write_head,
        reconstruction=params_bundle.reconstruction,
        branch_cycle_bias=params_bundle.branch_cycle_bias,
        latent_control=params_bundle.latent_control,
        fitting_method=(
            "shared_state_proxy_adam_v1" if not diverged
            else "shared_state_proxy_diverged"),
        training_trace=TrainingTraceWitness(
            seed=int(seed), n_steps=int(n_steps),
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
    final_cid = _sha256_hex({
        "kind": "w48_shared_state_proxy_params_inner",
        "shared_state": fitted.shared_state.to_dict(),
        "proxy_attention": fitted.proxy_attention.to_dict(),
        "write_head": fitted.write_head.to_dict(),
        "reconstruction": fitted.reconstruction.to_dict(),
        "branch_cycle_bias": fitted.branch_cycle_bias.to_dict(),
        "role_state_delta": fitted.role_state_delta.to_dict(),
        "latent_control": fitted.latent_control.to_dict(),
        "inner_w47_cid": str(fitted.inner_w47.cid()),
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
    return SharedStateProxyParams(
        inner_w47=fitted.inner_w47,
        shared_state=fitted.shared_state,
        role_state_delta=fitted.role_state_delta,
        proxy_attention=fitted.proxy_attention,
        write_head=fitted.write_head,
        reconstruction=fitted.reconstruction,
        branch_cycle_bias=fitted.branch_cycle_bias,
        latent_control=fitted.latent_control,
        fitting_method=fitted.fitting_method,
        training_trace=trace_v2,
    )


# =============================================================================
# Forward pass (inference)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateProxyForwardResult:
    """Result of one W48 forward pass at inference time."""

    shared_state_with_delta: tuple[float, ...]
    role_state_delta_value: tuple[float, ...]
    proxy_output: tuple[float, ...]
    per_head_attn_weights: tuple[tuple[float, ...], ...]
    pseudo_kv_read: tuple[float, ...]
    pseudo_kv_attn_weights: tuple[float, ...]
    write_gate_value: float
    branch_bias_value: float
    reconstruction_output: tuple[float, ...]
    reconstruction_l1: float
    reconstruction_l2: float
    gate_logit: float
    ratify_probability: float
    confidence_bucket: int
    latent_emit_mask: tuple[bool, ...]
    latent_bits_payload: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_state_with_delta": _round_floats(
                self.shared_state_with_delta),
            "role_state_delta_value": _round_floats(
                self.role_state_delta_value),
            "proxy_output": _round_floats(self.proxy_output),
            "per_head_attn_weights": [
                _round_floats(w) for w in
                self.per_head_attn_weights],
            "pseudo_kv_read": _round_floats(self.pseudo_kv_read),
            "pseudo_kv_attn_weights": _round_floats(
                self.pseudo_kv_attn_weights),
            "write_gate_value": float(round(
                self.write_gate_value, 12)),
            "branch_bias_value": float(round(
                self.branch_bias_value, 12)),
            "reconstruction_output": _round_floats(
                self.reconstruction_output),
            "reconstruction_l1": float(round(
                self.reconstruction_l1, 12)),
            "reconstruction_l2": float(round(
                self.reconstruction_l2, 12)),
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "confidence_bucket": int(self.confidence_bucket),
            "latent_emit_mask": [
                bool(b) for b in self.latent_emit_mask],
            "latent_bits_payload": [
                int(b) for b in self.latent_bits_payload],
        }


def _confidence_bucket(p: float) -> int:
    p = max(0.0, min(1.0, float(p)))
    if p < 0.25:
        return 0
    if p < 0.5:
        return 1
    if p < 0.75:
        return 2
    return 3


def forward_shared_state_proxy(
        *,
        channel_features: Mapping[str, Sequence[float]],
        params: SharedStateProxyParams,
        role: str,
        pseudo_kv_bank: PseudoKVBank,
        turn_index: int,
        branch_id: int = 0,
        cycle_id: int = 0,
        proxy_enabled: bool = True,
        pseudo_kv_enabled: bool = True,
        reconstruction_enabled: bool = True,
        target_recon: Sequence[float] | None = None,
        prior_flat_features: Sequence[float] | None = None,
) -> SharedStateProxyForwardResult:
    """W48 forward pass at inference.

    Combines:
      1. shared base state + per-role rank-r delta;
      2. multi-head proxy attention over pseudo-KV bank slots;
      3. reconstruction decoder of the previous turn's flat
         feature vector;
      4. branch/cycle bias;
      5. latent control serializer.
    """
    fd = int(params.inner_w47.feature_dim)
    flat_channel_dim = int(W45_N_CHANNELS) * fd
    flat_query = list(_flatten_channel_features(
        channel_features, feature_dim=fd))
    # 1. Shared state + per-role delta.
    role_delta = list(
        params.role_state_delta.forward_value(role=str(role)))
    shared_with_delta = [
        float(params.shared_state.values[i])
        + (role_delta[i] if i < len(role_delta) else 0.0)
        for i in range(params.shared_state.dim)
    ]
    # 2. Pseudo-KV read over admissible slots.
    pkv_read = [0.0] * params.factor_dim
    pkv_attn: list[float] = []
    proxy_out = [0.0] * params.proxy_attention.in_dim
    per_head_attn: tuple[tuple[float, ...], ...] = tuple(
        tuple() for _ in range(params.n_heads))
    if proxy_enabled and pseudo_kv_enabled:
        admissible = pseudo_kv_bank.admissible_for_turn(
            int(turn_index))
        slot_keys = [list(s.key) for s in admissible]
        slot_values = [list(s.value) for s in admissible]
        # Pad/truncate slot keys to proxy attention's in_dim.
        for i in range(len(slot_keys)):
            while len(slot_keys[i]) < params.proxy_attention.in_dim:
                slot_keys[i].append(0.0)
            slot_keys[i] = slot_keys[i][
                :params.proxy_attention.in_dim]
            while len(slot_values[i]) < params.proxy_attention.in_dim:
                slot_values[i].append(0.0)
            slot_values[i] = slot_values[i][
                :params.proxy_attention.in_dim]
        # Build query input: shared_state + flat_channels + zero
        # pseudo-KV read slot (the first turn's read uses
        # zero-vec).
        prior_read = [0.0] * params.factor_dim
        q = list(shared_with_delta) + list(flat_query) + prior_read
        while len(q) < params.proxy_attention.in_dim:
            q.append(0.0)
        q = q[:params.proxy_attention.in_dim]
        proxy_out, per_head_attn = (
            params.proxy_attention.forward_value(
                query_input=q,
                slot_keys=slot_keys,
                slot_values=slot_values,
            ))
        # Pseudo-KV pooled read (factor-dim): combine attention
        # weighted slot values (first factor_dim coords).
        if per_head_attn and per_head_attn[0]:
            # Use the first head's attn weights as the canonical
            # pseudo-KV attention vector.
            pkv_attn = list(per_head_attn[0])
            for i, w in enumerate(pkv_attn):
                if i < len(slot_values):
                    for r in range(min(
                            params.factor_dim,
                            len(slot_values[i]))):
                        pkv_read[r] += w * slot_values[i][r]

    # 3. Write gate.
    wh_input = list(shared_with_delta) + list(flat_query)
    while len(wh_input) < params.write_head.in_dim:
        wh_input.append(0.0)
    wh_input = wh_input[:params.write_head.in_dim]
    write_gate_value = float(
        params.write_head.forward_value(wh_input))

    # 4. Reconstruction.
    recon_in = list(shared_with_delta) + list(flat_query) + list(
        pkv_read)
    while len(recon_in) < params.reconstruction.in_dim:
        recon_in.append(0.0)
    recon_in = recon_in[:params.reconstruction.in_dim]
    recon_out = (
        list(params.reconstruction.forward_value(recon_in))
        if reconstruction_enabled else [0.0] * (
            params.reconstruction.recon_dim))
    # If target_recon is provided, compute L1/L2; otherwise use the
    # current flat_query as the trivial "no prior turn" baseline.
    target_vec = (
        list(target_recon) if target_recon is not None
        else (list(prior_flat_features)
              if prior_flat_features is not None
              else list(flat_query)))
    while len(target_vec) < params.reconstruction.recon_dim:
        target_vec.append(0.0)
    target_vec = target_vec[:params.reconstruction.recon_dim]
    diffs = [
        float(recon_out[i] - target_vec[i])
        for i in range(min(len(recon_out), len(target_vec)))]
    recon_l1 = _l1(diffs)
    recon_l2 = _l2(diffs)

    # 5. Branch/cycle bias.
    bcb_value = float(params.branch_cycle_bias.lookup_value(
        branch_id=int(branch_id),
        cycle_id=int(cycle_id)))

    # 6. Final gate logit: average of proxy_out + bias + write-gate
    # contribution (scaled so the controller never trivially
    # ratifies).
    proxy_scalar = (
        float(sum(proxy_out) / float(max(1, len(proxy_out))))
        if proxy_out else 0.0)
    gate_logit = (
        proxy_scalar + bcb_value
        + 0.5 * (write_gate_value - 0.5))
    ratify_prob = float(_stable_sigmoid(gate_logit))
    conf = _confidence_bucket(ratify_prob)

    # 7. Latent control emit mask + bits payload.
    emit_mask = tuple(params.latent_control.emit_mask())
    # Derive bits payload deterministically from the proxy state.
    bits_payload = tuple(
        int(1) if i < len(proxy_out) and proxy_out[i] >= 0.0
        else int(0)
        for i in range(params.latent_control.n_bits))

    return SharedStateProxyForwardResult(
        shared_state_with_delta=tuple(
            _round_floats(shared_with_delta)),
        role_state_delta_value=tuple(_round_floats(role_delta)),
        proxy_output=tuple(_round_floats(proxy_out)),
        per_head_attn_weights=tuple(
            tuple(_round_floats(w)) for w in per_head_attn),
        pseudo_kv_read=tuple(_round_floats(pkv_read)),
        pseudo_kv_attn_weights=tuple(_round_floats(pkv_attn)),
        write_gate_value=float(round(write_gate_value, 12)),
        branch_bias_value=float(round(bcb_value, 12)),
        reconstruction_output=tuple(_round_floats(recon_out)),
        reconstruction_l1=float(round(recon_l1, 12)),
        reconstruction_l2=float(round(recon_l2, 12)),
        gate_logit=float(round(gate_logit, 12)),
        ratify_probability=float(round(ratify_prob, 12)),
        confidence_bucket=int(conf),
        latent_emit_mask=tuple(bool(b) for b in emit_mask),
        latent_bits_payload=tuple(
            int(b) for b in bits_payload),
    )


# =============================================================================
# Registry + Orchestrator
# =============================================================================

@dataclasses.dataclass
class SharedStateProxyRegistry:
    """Controller-side configuration for the W48 shared-state
    proxy.
    """

    schema_cid: str
    inner_w47_registry: AutogradManifoldRegistry
    params: SharedStateProxyParams
    proxy_enabled: bool = True
    pseudo_kv_enabled: bool = True
    reconstruction_enabled: bool = True
    branch_bias_enabled: bool = True
    write_head_enabled: bool = True
    latent_control_enabled: bool = True
    pseudo_kv_capacity: int = W48_DEFAULT_PSEUDO_KV_SLOTS
    write_gate_threshold: float = (
        W48_DEFAULT_WRITE_GATE_THRESHOLD)
    margin_abstain_threshold: float = 0.0
    reconstruction_max_l1: float = W48_DEFAULT_RECON_MAX_L1
    branch_bias_min: float = -1e9
    branch_bias_max: float = 1e9
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT
    latent_control_tag: str = W48_DEFAULT_LATENT_CTRL_TAG
    shared_state_tag: str = W48_DEFAULT_SHARED_STATE_TAG
    n_branches: int = W48_DEFAULT_N_BRANCHES
    n_cycles: int = W48_DEFAULT_N_CYCLES

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w47_registry.is_trivial
            and not self.proxy_enabled
            and not self.pseudo_kv_enabled
            and not self.reconstruction_enabled
            and not self.branch_bias_enabled
            and not self.write_head_enabled
            and not self.latent_control_enabled
            and self.params.fitting_method == "unfitted"
        )


def build_trivial_shared_state_proxy_registry(
        *, schema_cid: str | None = None,
) -> SharedStateProxyRegistry:
    """Build a registry whose orchestrator reduces to
    :class:`coordpy.autograd_manifold.AutogradManifoldTeam`
    (trivial) byte-for-byte.
    """
    cid = schema_cid or _sha256_hex({
        "kind": "w48_trivial_schema"})
    inner = build_trivial_autograd_manifold_registry(
        schema_cid=str(cid))
    p = build_unfitted_shared_state_proxy_params(
        inner_w47=inner.params)
    return SharedStateProxyRegistry(
        schema_cid=str(cid),
        inner_w47_registry=inner,
        params=p,
        proxy_enabled=False,
        pseudo_kv_enabled=False,
        reconstruction_enabled=False,
        branch_bias_enabled=False,
        write_head_enabled=False,
        latent_control_enabled=False,
        abstain_substitution_enabled=False,
    )


def build_shared_state_proxy_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        params: SharedStateProxyParams | None = None,
        inner_w47_params: AutogradManifoldParams | None = None,
        proxy_enabled: bool = True,
        pseudo_kv_enabled: bool = True,
        reconstruction_enabled: bool = True,
        branch_bias_enabled: bool = True,
        write_head_enabled: bool = True,
        latent_control_enabled: bool = True,
        pseudo_kv_capacity: int = W48_DEFAULT_PSEUDO_KV_SLOTS,
        write_gate_threshold: float = (
            W48_DEFAULT_WRITE_GATE_THRESHOLD),
        margin_abstain_threshold: float = -10.0,
        reconstruction_max_l1: float = W48_DEFAULT_RECON_MAX_L1,
        abstain_substitution_enabled: bool = True,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
        latent_control_tag: str = W48_DEFAULT_LATENT_CTRL_TAG,
        shared_state_tag: str = W48_DEFAULT_SHARED_STATE_TAG,
        n_branches: int = W48_DEFAULT_N_BRANCHES,
        n_cycles: int = W48_DEFAULT_N_CYCLES,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
        prefix_reuse_enabled: bool = True,
        prefix_turns: int = 2,
        memory_capacity: int = 8,
        use_attention_routing: bool = True,
        control_token_mode: str = W46_CTRL_MODE_FULL,
) -> SharedStateProxyRegistry:
    """Build a fully configured W48 shared-state-proxy registry."""
    inner = build_autograd_manifold_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        params=inner_w47_params,
        autograd_enabled=True,
        margin_abstain_threshold=margin_abstain_threshold,
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
        prefix_reuse_enabled=bool(prefix_reuse_enabled),
        prefix_turns=int(prefix_turns),
        memory_capacity=int(memory_capacity),
        use_attention_routing=bool(use_attention_routing),
        control_token_mode=str(control_token_mode),
        abstain_substitution_enabled=False,  # W48 owns abstain
        abstain_output=str(abstain_output),
    )
    p = params or build_unfitted_shared_state_proxy_params(
        inner_w47=inner.params,
        n_branches=int(n_branches),
        n_cycles=int(n_cycles),
    )
    return SharedStateProxyRegistry(
        schema_cid=str(schema_cid),
        inner_w47_registry=inner,
        params=p,
        proxy_enabled=bool(proxy_enabled),
        pseudo_kv_enabled=bool(pseudo_kv_enabled),
        reconstruction_enabled=bool(reconstruction_enabled),
        branch_bias_enabled=bool(branch_bias_enabled),
        write_head_enabled=bool(write_head_enabled),
        latent_control_enabled=bool(latent_control_enabled),
        pseudo_kv_capacity=int(pseudo_kv_capacity),
        write_gate_threshold=float(write_gate_threshold),
        margin_abstain_threshold=float(margin_abstain_threshold),
        reconstruction_max_l1=float(reconstruction_max_l1),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        latent_control_tag=str(latent_control_tag),
        shared_state_tag=str(shared_state_tag),
        n_branches=int(n_branches),
        n_cycles=int(n_cycles),
    )


@dataclasses.dataclass(frozen=True)
class SharedStateProxyGatingDecision:
    """Result of running the W48 proxy gate on one turn."""

    branch: str
    w47_branch: str
    w46_branch: str
    w45_branch: str
    w44_branch: str
    pmc_branch: str
    role: str
    role_handoff_signature_cid: str
    policy_entry_cid: str
    branch_id: int
    cycle_id: int
    shared_state_capsule_cid: str
    pseudo_kv_bank_head_cid: str
    forward: SharedStateProxyForwardResult
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W48_PROXY_ABSTAIN_BRANCHES


class SharedStateProxyOrchestrator:
    """Per-turn W48 gating + envelope binding.

    Wraps an :class:`AutogradManifoldOrchestrator` (the W47
    inner) plus a :class:`SharedStateProxyRegistry`. Stateful in
    the pseudo-KV bank + the underlying W47 memory bank.
    """

    def __init__(
            self, registry: SharedStateProxyRegistry,
    ) -> None:
        self.registry = registry
        self._inner = AutogradManifoldOrchestrator(
            registry=registry.inner_w47_registry)
        self._pseudo_kv = PseudoKVBank(
            capacity=int(registry.pseudo_kv_capacity),
            factor_dim=int(registry.params.factor_dim),
        )
        self._last_flat_query: tuple[float, ...] | None = None

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    @property
    def pseudo_kv_bank(self) -> PseudoKVBank:
        return self._pseudo_kv

    def reset_session(self) -> None:
        self._inner.reset_session()
        self._pseudo_kv.reset()
        self._last_flat_query = None

    def gate(
            self,
            *,
            observation: CellObservation,
            role: str,
            role_handoff_signature_cid: str,
            parent_w42_cid: str,
            n_w42_visible_tokens: int,
            turn_index: int,
            branch_id: int = 0,
            cycle_id: int = 0,
            expected_spherical: SphericalConsensusSignature | None = None,
            expected_subspace: SubspaceBasis | None = None,
    ) -> tuple[SharedStateProxyGatingDecision, Any]:
        # Delegate to the W47 inner.
        w47_decision, w47_aux = self._inner.gate(
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
         w45_decision, w46_decision, w46_forward, ag_forward) = (
            w47_aux)

        # Build channel features from bundle.
        feats = _channel_features_from_bundle(
            bundle,
            feature_dim=int(
                self.registry.params.inner_w47.feature_dim),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )
        fd = int(self.registry.params.inner_w47.feature_dim)
        flat_query = list(_flatten_channel_features(
            feats, feature_dim=fd))

        # Forward.
        forward = forward_shared_state_proxy(
            channel_features=feats,
            params=self.registry.params,
            role=str(role),
            pseudo_kv_bank=self._pseudo_kv,
            turn_index=int(turn_index),
            branch_id=int(branch_id),
            cycle_id=int(cycle_id),
            proxy_enabled=bool(self.registry.proxy_enabled),
            pseudo_kv_enabled=bool(
                self.registry.pseudo_kv_enabled),
            reconstruction_enabled=bool(
                self.registry.reconstruction_enabled),
            prior_flat_features=(
                self._last_flat_query
                if self._last_flat_query is not None
                else None),
        )

        # Branch selection.
        if self.registry.is_trivial:
            branch = W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH
            abstain_reason = ""
        elif not self.registry.proxy_enabled:
            branch = W48_BRANCH_PROXY_DISABLED
            abstain_reason = ""
        elif self.registry.params.training_trace.diverged:
            branch = W48_BRANCH_PROXY_TRAIN_FAILURE
            abstain_reason = "proxy_train_failure"
        elif (forward.gate_logit
                < float(self.registry.margin_abstain_threshold)):
            branch = W48_BRANCH_PROXY_MARGIN_ABSTAIN
            abstain_reason = "proxy_margin"
        elif (self.registry.reconstruction_enabled
                and forward.reconstruction_l1
                > float(self.registry.reconstruction_max_l1)):
            branch = W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN
            abstain_reason = "proxy_reconstruction"
        elif (self.registry.branch_bias_enabled
                and (forward.branch_bias_value
                     < float(self.registry.branch_bias_min)
                     or forward.branch_bias_value
                     > float(self.registry.branch_bias_max))):
            branch = W48_BRANCH_PROXY_BRANCH_BIAS_ABSTAIN
            abstain_reason = "proxy_branch_bias"
        elif (self.registry.write_head_enabled
                and forward.write_gate_value < 0.0):
            branch = W48_BRANCH_PROXY_WRITE_GATE_ABSTAIN
            abstain_reason = "proxy_write_gate"
        else:
            branch = W48_BRANCH_PROXY_RATIFIED
            abstain_reason = ""

        # Pseudo-KV write decision (post-decision).
        new_pkv_head: str = ""
        if (self.registry.pseudo_kv_enabled
                and self.registry.write_head_enabled
                and forward.write_gate_value
                >= float(self.registry.write_gate_threshold)
                and branch
                in (W48_BRANCH_PROXY_RATIFIED,
                    W48_BRANCH_PROXY_DISABLED)):
            # Build key + value from flat features + shared state.
            ss_with_delta = list(forward.shared_state_with_delta)
            key = list(flat_query)[:self.registry.params.factor_dim]
            while len(key) < self.registry.params.factor_dim:
                key.append(0.0)
            # Value: use proxy_output projected down.
            value = list(forward.proxy_output)[
                :self.registry.params.factor_dim]
            while len(value) < self.registry.params.factor_dim:
                value.append(0.0)
            # If proxy is empty, fall back to shared state.
            if all(abs(v) < 1e-12 for v in value):
                value = ss_with_delta[
                    :self.registry.params.factor_dim]
                while len(value) < self.registry.params.factor_dim:
                    value.append(0.0)
            slot_cid_payload = {
                "kind": "w48_pseudo_kv_slot_source",
                "turn_index": int(turn_index),
                "role": str(role),
                "channel_features_sha": hashlib.sha256(
                    json.dumps(
                        sorted([
                            [c, _round_floats(v)]
                            for c, v in feats.items()],
                            key=lambda x: x[0]),
                        separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
            slot = PseudoKVSlot(
                slot_index=int(len(self._pseudo_kv.slots)),
                turn_index=int(turn_index),
                role=str(role),
                key=tuple(_round_floats(key)),
                value=tuple(_round_floats(value)),
                write_gate_value=float(round(
                    forward.write_gate_value, 12)),
                source_observation_cid=_sha256_hex(
                    slot_cid_payload),
            )
            self._pseudo_kv.write(slot)
        new_pkv_head = self._pseudo_kv.head_cid()

        # Cache the last flat query for the next turn's
        # reconstruction target.
        self._last_flat_query = tuple(flat_query)

        decision = SharedStateProxyGatingDecision(
            branch=str(branch),
            w47_branch=str(w47_decision.branch),
            w46_branch=str(w47_decision.w46_branch),
            w45_branch=str(w47_decision.w45_branch),
            w44_branch=str(w47_decision.w44_branch),
            pmc_branch=str(w47_decision.pmc_branch),
            role=str(role),
            role_handoff_signature_cid=str(
                w47_decision.role_handoff_signature_cid),
            policy_entry_cid=str(w47_decision.policy_entry_cid),
            branch_id=int(branch_id),
            cycle_id=int(cycle_id),
            shared_state_capsule_cid=str(
                self.registry.params.shared_state.cid()),
            pseudo_kv_bank_head_cid=str(new_pkv_head),
            forward=forward,
            abstain_reason=str(abstain_reason),
        )
        return decision, (
            w43_result, causal_mask, bundle, w45_decision,
            w46_decision, w46_forward, ag_forward, w47_decision,
        )


# =============================================================================
# Envelope + verifier
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateProxyHandoffEnvelope:
    """Sealed W48 envelope for one turn."""

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w47_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
    w47_branch: str
    w46_branch: str
    w45_branch: str
    w44_branch: str
    pmc_branch: str
    abstain_reason: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    branch_id: int
    cycle_id: int

    # Param provenance.
    proxy_params_cid: str
    training_trace_cid: str
    fitting_method: str

    shared_state_capsule_cid: str
    shared_state_dim: int
    role_state_delta_cid: str
    role_state_delta_present: bool

    proxy_attention_cid: str
    proxy_n_heads: int
    proxy_factor_dim: int

    write_head_cid: str
    reconstruction_decoder_cid: str
    branch_cycle_bias_cid: str
    latent_control_cid: str

    inner_w47_outer_cid: str
    inner_w47_params_cid: str

    pseudo_kv_bank_head_cid: str
    pseudo_kv_bank_size: int
    pseudo_kv_capacity: int

    # Forward witnesses.
    gate_logit: float
    ratify_probability: float
    confidence_bucket: int
    write_gate_value: float
    branch_bias_value: float
    reconstruction_l1: float
    reconstruction_l2: float
    latent_emit_mask: tuple[bool, ...]
    latent_bits_payload: tuple[int, ...]
    proxy_forward_witness_cid: str

    # Prompt / control / prefix.
    prompt_sha256: str
    prompt_construction_witness_cid: str
    latent_control_witness_cid: str
    branch_history_witness_cid: str
    output_sha256: str

    # Token accounting.
    n_visible_prompt_tokens_textual: int
    n_visible_prompt_tokens_actual: int
    n_visible_prompt_tokens_saved: int
    n_overhead_tokens: int
    n_latent_ctrl_tokens: int
    n_shared_state_header_tokens: int
    n_branch_history_tokens: int
    n_branch_history_tokens_saved: int

    behavioral_change: bool

    proxy_witness_cid: str
    proxy_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w48_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w47_envelope_cid=self.parent_w47_envelope_cid,
            proxy_params_cid=self.proxy_params_cid,
            training_trace_cid=self.training_trace_cid,
            proxy_witness_cid=self.proxy_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_w48_proxy_forward_witness_cid(
        *,
        gate_logit: float,
        ratify_probability: float,
        confidence_bucket: int,
        write_gate_value: float,
        branch_bias_value: float,
        reconstruction_l1: float,
        reconstruction_l2: float,
        latent_emit_mask: tuple[bool, ...],
        latent_bits_payload: tuple[int, ...],
        turn_index: int,
        branch_id: int,
        cycle_id: int,
        role: str,
) -> str:
    return _sha256_hex({
        "kind": "w48_proxy_forward_witness",
        "gate_logit": float(round(gate_logit, 12)),
        "ratify_probability": float(round(
            ratify_probability, 12)),
        "confidence_bucket": int(confidence_bucket),
        "write_gate_value": float(round(write_gate_value, 12)),
        "branch_bias_value": float(round(
            branch_bias_value, 12)),
        "reconstruction_l1": float(round(reconstruction_l1, 12)),
        "reconstruction_l2": float(round(reconstruction_l2, 12)),
        "latent_emit_mask": [bool(b) for b in latent_emit_mask],
        "latent_bits_payload": [
            int(b) for b in latent_bits_payload],
        "turn_index": int(turn_index),
        "branch_id": int(branch_id),
        "cycle_id": int(cycle_id),
        "role": str(role),
    })


def _compute_w48_prompt_construction_witness_cid(
        *,
        turn_index: int,
        role: str,
        prompt_sha256: str,
        n_visible_prompt_tokens_textual: int,
        n_visible_prompt_tokens_actual: int,
        n_latent_ctrl_tokens: int,
        n_shared_state_header_tokens: int,
        n_branch_history_tokens: int,
        shared_state_capsule_cid: str,
        pseudo_kv_bank_head_cid: str,
) -> str:
    return _sha256_hex({
        "kind": "w48_prompt_construction_witness",
        "turn_index": int(turn_index),
        "role": str(role),
        "prompt_sha256": str(prompt_sha256),
        "n_visible_prompt_tokens_textual": int(
            n_visible_prompt_tokens_textual),
        "n_visible_prompt_tokens_actual": int(
            n_visible_prompt_tokens_actual),
        "n_latent_ctrl_tokens": int(n_latent_ctrl_tokens),
        "n_shared_state_header_tokens": int(
            n_shared_state_header_tokens),
        "n_branch_history_tokens": int(n_branch_history_tokens),
        "shared_state_capsule_cid": str(shared_state_capsule_cid),
        "pseudo_kv_bank_head_cid": str(pseudo_kv_bank_head_cid),
    })


def _compute_w48_proxy_witness_cid(
        *,
        decision_branch: str,
        w47_branch: str,
        w46_branch: str,
        w45_branch: str,
        w44_branch: str,
        pmc_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        proxy_params_cid: str,
        training_trace_cid: str,
        proxy_forward_witness_cid: str,
        prompt_construction_witness_cid: str,
        latent_control_witness_cid: str,
        branch_history_witness_cid: str,
        shared_state_capsule_cid: str,
        pseudo_kv_bank_head_cid: str,
        inner_w47_outer_cid: str,
        output_sha256: str,
        behavioral_change: bool,
) -> str:
    return _sha256_hex({
        "kind": "w48_proxy_witness",
        "decision_branch": str(decision_branch),
        "w47_branch": str(w47_branch),
        "w46_branch": str(w46_branch),
        "w45_branch": str(w45_branch),
        "w44_branch": str(w44_branch),
        "pmc_branch": str(pmc_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "proxy_params_cid": str(proxy_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "proxy_forward_witness_cid": str(
            proxy_forward_witness_cid),
        "prompt_construction_witness_cid": str(
            prompt_construction_witness_cid),
        "latent_control_witness_cid": str(
            latent_control_witness_cid),
        "branch_history_witness_cid": str(
            branch_history_witness_cid),
        "shared_state_capsule_cid": str(
            shared_state_capsule_cid),
        "pseudo_kv_bank_head_cid": str(pseudo_kv_bank_head_cid),
        "inner_w47_outer_cid": str(inner_w47_outer_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
    })


def _compute_w48_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w47_envelope_cid: str,
        proxy_params_cid: str,
        training_trace_cid: str,
        proxy_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w48_proxy_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w47_envelope_cid": str(parent_w47_envelope_cid),
        "proxy_params_cid": str(proxy_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "proxy_witness_cid": str(proxy_witness_cid),
        "turn_index": int(turn_index),
    })


@dataclasses.dataclass(frozen=True)
class SharedStateProxyVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


W48_ALL_FAILURE_MODES: tuple[str, ...] = (
    "empty_w48_envelope",
    "w48_schema_version_unknown",
    "w48_schema_cid_mismatch",
    "w48_decision_branch_unknown",
    "w48_role_handoff_signature_cid_invalid",
    "w48_prompt_sha256_invalid",
    "w48_token_accounting_invalid",
    "w48_confidence_bucket_invalid",
    "w48_ratify_probability_invalid",
    "w48_proxy_params_cid_invalid",
    "w48_training_trace_cid_invalid",
    "w48_shared_state_capsule_cid_invalid",
    "w48_pseudo_kv_bank_head_cid_invalid",
    "w48_proxy_attention_cid_invalid",
    "w48_role_state_delta_cid_invalid",
    "w48_write_head_cid_invalid",
    "w48_reconstruction_decoder_cid_invalid",
    "w48_branch_cycle_bias_cid_invalid",
    "w48_latent_control_cid_invalid",
    "w48_proxy_forward_witness_cid_mismatch",
    "w48_prompt_construction_witness_cid_mismatch",
    "w48_latent_control_witness_cid_invalid",
    "w48_branch_history_witness_cid_invalid",
    "w48_proxy_witness_cid_mismatch",
    "w48_emit_mask_invalid",
    "w48_outer_cid_mismatch",
)


def verify_shared_state_proxy_handoff(
        env: "SharedStateProxyHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_proxy_params_cid: str | None = None,
        registered_training_trace_cid: str | None = None,
        registered_shared_state_capsule_cid: str | None = None,
) -> SharedStateProxyVerificationOutcome:
    """Pure-function verifier for the W48 envelope.

    Enumerates 22+ disjoint failure modes (see
    :data:`W48_ALL_FAILURE_MODES`).
    """
    n = 0
    if env is None:
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="empty_w48_envelope", n_checks=0)
    n += 1
    if env.schema_version != (
            W48_SHARED_STATE_PROXY_SCHEMA_VERSION):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W48_ALL_BRANCHES:
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.decision_branch != (
            W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return SharedStateProxyVerificationOutcome(
                ok=False,
                reason=(
                    "w48_role_handoff_signature_cid_invalid"),
                n_checks=n)
    n += 1
    if (env.prompt_sha256 is not None
            and env.prompt_sha256
            and len(env.prompt_sha256) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_prompt_sha256_invalid",
            n_checks=n)
    n += 1
    if (env.n_visible_prompt_tokens_textual < 0
            or env.n_visible_prompt_tokens_actual < 0
            or env.n_overhead_tokens < 0
            or env.n_latent_ctrl_tokens < 0
            or env.n_shared_state_header_tokens < 0
            or env.n_branch_history_tokens < 0
            or env.n_visible_prompt_tokens_saved
            != (int(env.n_visible_prompt_tokens_textual)
                - int(env.n_visible_prompt_tokens_actual))):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_token_accounting_invalid",
            n_checks=n)
    n += 1
    if (env.confidence_bucket < 0
            or env.confidence_bucket >= 4):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_confidence_bucket_invalid",
            n_checks=n)
    n += 1
    if not (0.0 - 1e-9 <= float(env.ratify_probability)
            <= 1.0 + 1e-9):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_ratify_probability_invalid",
            n_checks=n)
    n += 1
    if (not env.proxy_params_cid
            or len(env.proxy_params_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_proxy_params_cid_invalid",
            n_checks=n)
    n += 1
    if registered_proxy_params_cid is not None:
        if env.proxy_params_cid != str(
                registered_proxy_params_cid):
            return SharedStateProxyVerificationOutcome(
                ok=False,
                reason="w48_proxy_params_cid_invalid",
                n_checks=n)
    if (not env.training_trace_cid
            or len(env.training_trace_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_training_trace_cid_invalid",
            n_checks=n)
    n += 1
    if registered_training_trace_cid is not None:
        if env.training_trace_cid != str(
                registered_training_trace_cid):
            return SharedStateProxyVerificationOutcome(
                ok=False,
                reason="w48_training_trace_cid_invalid",
                n_checks=n)
    if (not env.shared_state_capsule_cid
            or len(env.shared_state_capsule_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False,
            reason="w48_shared_state_capsule_cid_invalid",
            n_checks=n)
    n += 1
    if registered_shared_state_capsule_cid is not None:
        if env.shared_state_capsule_cid != str(
                registered_shared_state_capsule_cid):
            return SharedStateProxyVerificationOutcome(
                ok=False,
                reason=(
                    "w48_shared_state_capsule_cid_invalid"),
                n_checks=n)
    if (not env.pseudo_kv_bank_head_cid
            or len(env.pseudo_kv_bank_head_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False,
            reason="w48_pseudo_kv_bank_head_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.proxy_attention_cid
            or len(env.proxy_attention_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_proxy_attention_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.role_state_delta_cid
            or len(env.role_state_delta_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_role_state_delta_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.write_head_cid
            or len(env.write_head_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_write_head_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.reconstruction_decoder_cid
            or len(env.reconstruction_decoder_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False,
            reason="w48_reconstruction_decoder_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.branch_cycle_bias_cid
            or len(env.branch_cycle_bias_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_branch_cycle_bias_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.latent_control_cid
            or len(env.latent_control_cid) != 64):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_latent_control_cid_invalid",
            n_checks=n)
    n += 1
    if (len(env.latent_emit_mask) <= 0
            or any(not isinstance(b, bool)
                   for b in env.latent_emit_mask)):
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_emit_mask_invalid",
            n_checks=n)
    n += 1
    expected_forward = (
        _compute_w48_proxy_forward_witness_cid(
            gate_logit=float(env.gate_logit),
            ratify_probability=float(env.ratify_probability),
            confidence_bucket=int(env.confidence_bucket),
            write_gate_value=float(env.write_gate_value),
            branch_bias_value=float(env.branch_bias_value),
            reconstruction_l1=float(env.reconstruction_l1),
            reconstruction_l2=float(env.reconstruction_l2),
            latent_emit_mask=tuple(
                bool(b) for b in env.latent_emit_mask),
            latent_bits_payload=tuple(
                int(b) for b in env.latent_bits_payload),
            turn_index=int(env.turn_index),
            branch_id=int(env.branch_id),
            cycle_id=int(env.cycle_id),
            role=env.role,
        ))
    if expected_forward != env.proxy_forward_witness_cid:
        return SharedStateProxyVerificationOutcome(
            ok=False,
            reason="w48_proxy_forward_witness_cid_mismatch",
            n_checks=n)
    n += 1
    expected_construction = (
        _compute_w48_prompt_construction_witness_cid(
            turn_index=int(env.turn_index),
            role=env.role,
            prompt_sha256=env.prompt_sha256,
            n_visible_prompt_tokens_textual=int(
                env.n_visible_prompt_tokens_textual),
            n_visible_prompt_tokens_actual=int(
                env.n_visible_prompt_tokens_actual),
            n_latent_ctrl_tokens=int(env.n_latent_ctrl_tokens),
            n_shared_state_header_tokens=int(
                env.n_shared_state_header_tokens),
            n_branch_history_tokens=int(
                env.n_branch_history_tokens),
            shared_state_capsule_cid=env.shared_state_capsule_cid,
            pseudo_kv_bank_head_cid=env.pseudo_kv_bank_head_cid,
        ))
    if expected_construction != (
            env.prompt_construction_witness_cid):
        return SharedStateProxyVerificationOutcome(
            ok=False,
            reason=(
                "w48_prompt_construction_witness_cid_mismatch"),
            n_checks=n)
    n += 1
    expected_witness = _compute_w48_proxy_witness_cid(
        decision_branch=env.decision_branch,
        w47_branch=env.w47_branch,
        w46_branch=env.w46_branch,
        w45_branch=env.w45_branch,
        w44_branch=env.w44_branch,
        pmc_branch=env.pmc_branch,
        abstain_reason=env.abstain_reason,
        role_handoff_signature_cid=env.role_handoff_signature_cid,
        policy_entry_cid=env.policy_entry_cid,
        proxy_params_cid=env.proxy_params_cid,
        training_trace_cid=env.training_trace_cid,
        proxy_forward_witness_cid=env.proxy_forward_witness_cid,
        prompt_construction_witness_cid=(
            env.prompt_construction_witness_cid),
        latent_control_witness_cid=env.latent_control_witness_cid,
        branch_history_witness_cid=env.branch_history_witness_cid,
        shared_state_capsule_cid=env.shared_state_capsule_cid,
        pseudo_kv_bank_head_cid=env.pseudo_kv_bank_head_cid,
        inner_w47_outer_cid=env.inner_w47_outer_cid,
        output_sha256=env.output_sha256,
        behavioral_change=bool(env.behavioral_change),
    )
    if expected_witness != env.proxy_witness_cid:
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_proxy_witness_cid_mismatch",
            n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.proxy_outer_cid:
        return SharedStateProxyVerificationOutcome(
            ok=False, reason="w48_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return SharedStateProxyVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Team result + Team
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedStateProxyTurn:
    """One turn of a :class:`SharedStateProxyTeam` run."""

    agent_turn: AgentTurn
    decision: SharedStateProxyGatingDecision
    envelope: SharedStateProxyHandoffEnvelope


@dataclasses.dataclass(frozen=True)
class SharedStateProxyTeamResult:
    """Result of a :class:`SharedStateProxyTeam` run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    proxy_turns: tuple[SharedStateProxyTurn, ...]
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
    n_behavioral_changes: int = 0
    n_visible_tokens_saved_branch_history: int = 0
    n_visible_tokens_added_latent_ctrl: int = 0
    n_visible_tokens_added_shared_state_header: int = 0
    n_visible_tokens_added_branch_history: int = 0
    n_abstain_substitutions: int = 0
    n_pseudo_kv_writes: int = 0
    n_proxy_margin_abstains: int = 0
    n_proxy_recon_abstains: int = 0
    mean_ratify_probability: float = 0.0
    mean_write_gate: float = 0.0
    mean_reconstruction_l1: float = 0.0
    proxy_params_cid: str = ""
    training_trace_cid: str = ""
    shared_state_capsule_cid: str = ""
    final_pseudo_kv_bank_head_cid: str = ""
    schema: str = W48_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


class SharedStateProxyTeam:
    """W48 shared-state-proxy agent team.

    Wraps the W47 :class:`AutogradManifoldTeam` contract with the
    W48 shared-state + pseudo-KV + reconstruction + branch-bias
    proxy. With a trivial registry, this team reduces to
    ``AutogradManifoldTeam.run`` byte-for-byte (the
    W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: SharedStateProxyRegistry,
            observation_builder: LiveObservationBuilder | None = None,
            team_instructions: str = "",
            max_visible_handoffs: int = 4,
            capture_capsules: bool = True,
            task_summary: str | None = None,
            handoff_budget: "CapsuleBudget | None" = None,
            parent_w42_cid: str = W44_DEFAULT_PARENT_W42_CID,
            expected_spherical: SphericalConsensusSignature | None = None,
            expected_subspace: SubspaceBasis | None = None,
            branch_id_of_turn: Callable[[int], int] | None = None,
            cycle_id_of_turn: Callable[[int], int] | None = None,
    ) -> None:
        if not agents:
            raise ValueError(
                "SharedStateProxyTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = SharedStateProxyOrchestrator(registry)
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
        self.branch_id_of_turn = (
            branch_id_of_turn or (lambda t: 0))
        self.cycle_id_of_turn = (
            cycle_id_of_turn or (lambda t: 0))

    @property
    def schema_cid(self) -> str:
        return self.orchestrator.schema_cid

    def _resolve_backend(self, member: Agent) -> LLMBackend:
        backend = member.backend or self.backend
        if backend is None:
            raise ValueError(
                "no backend configured; pass backend=... to "
                "SharedStateProxyTeam")
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
            decision: SharedStateProxyGatingDecision,
            branch_path: Sequence[int],
            cycle_path: Sequence[int],
            shared_state_short: str,
    ) -> tuple[
            str, str, int, int, int, int, int, int, int,
            LatentControlWitness, BranchHistoryWitness,
            tuple[bool, ...]]:
        """Build the bounded prompt + textual shadow.

        Returns:
          (bounded_prompt, textual_prompt, n_textual, n_actual,
           n_latent_ctrl_tokens, n_shared_state_header_tokens,
           n_branch_history_tokens, n_branch_history_tokens_saved,
           n_visible_tokens_saved,
           latent_ctrl_witness, branch_history_witness,
           emit_mask)
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

        # Textual shadow.
        textual_parts = list(common_parts)
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[
                    -self.max_visible_handoffs:])
            textual_parts.append(
                "Visible team handoffs (bounded to avoid token "
                f"cramming):\n{rendered}")
        # Textual rendering of branch/cycle path (one item per
        # pair).
        if (self.registry.proxy_enabled
                and branch_path):
            textual_branch = ", ".join(
                f"b={int(b)},c={int(c)}"
                for b, c in zip(branch_path, cycle_path))
            textual_parts.append(
                f"Branch history (textual): {textual_branch}")
        textual_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        textual_prompt = "\n\n".join(textual_parts)
        n_textual = len(textual_prompt.split())

        # Bounded prompt with shared-state + branch-history +
        # latent-ctrl headers.
        bounded_parts = list(common_parts)
        emit_mask: tuple[bool, ...] = tuple()
        bits_payload: tuple[int, ...] = tuple()
        n_latent_ctrl_tokens = 0
        latent_ctrl_witness = LatentControlWitness(
            ctrl_tag=str(self.registry.latent_control_tag),
            n_bits=0,
            emit_mask=tuple(),
            bits_payload=tuple(),
            shared_state_hash_short=str(shared_state_short),
            n_ctrl_tokens=0,
            ctrl_bytes_sha256=hashlib.sha256(
                b"").hexdigest(),
        )
        branch_history_witness = BranchHistoryWitness(
            branch_path=tuple(),
            cycle_path=tuple(),
            packed_integer=0,
            packed_n_bits=0,
            textual_tokens=0,
            compressed_tokens=0,
            compressor_bytes_sha256=hashlib.sha256(
                b"").hexdigest(),
        )
        n_shared_state_header_tokens = 0
        n_branch_history_tokens = 0
        n_branch_history_tokens_saved = 0

        # SHARED_STATE_HASH header.
        if (self.registry.proxy_enabled
                and self.registry.shared_state_tag):
            ss_line = (
                f"{self.registry.shared_state_tag}: "
                f"{shared_state_short}")
            bounded_parts.append(ss_line)
            n_shared_state_header_tokens = len(ss_line.split())
        # Branch-history compressor.
        if (self.registry.proxy_enabled
                and branch_path
                and self.registry.branch_bias_enabled):
            bh_text, branch_history_witness = (
                compress_branch_history(
                    branch_path=tuple(branch_path),
                    cycle_path=tuple(cycle_path),
                    n_branches=int(self.registry.n_branches),
                    n_cycles=int(self.registry.n_cycles),
                ))
            bounded_parts.append(bh_text)
            n_branch_history_tokens = int(
                branch_history_witness.compressed_tokens)
            n_branch_history_tokens_saved = max(
                0,
                int(branch_history_witness.textual_tokens)
                - int(branch_history_witness.compressed_tokens))
        # LATENT_CTRL line.
        if (self.registry.proxy_enabled
                and self.registry.latent_control_enabled):
            emit_mask = tuple(decision.forward.latent_emit_mask)
            bits_payload = tuple(decision.forward.latent_bits_payload)
            ctrl_line, latent_ctrl_witness = (
                build_latent_control_string(
                    ctrl_tag=str(
                        self.registry.latent_control_tag),
                    emit_mask=emit_mask,
                    bits_payload=bits_payload,
                    shared_state_hash_short=str(
                        shared_state_short),
                ))
            bounded_parts.append(ctrl_line)
            n_latent_ctrl_tokens = int(
                latent_ctrl_witness.n_ctrl_tokens)

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
        n_actual = len(bounded_prompt.split())
        n_saved = max(0, int(n_textual) - int(n_actual))
        return (
            bounded_prompt,
            textual_prompt,
            int(n_textual),
            int(n_actual),
            int(n_latent_ctrl_tokens),
            int(n_shared_state_header_tokens),
            int(n_branch_history_tokens),
            int(n_branch_history_tokens_saved),
            int(n_saved),
            latent_ctrl_witness,
            branch_history_witness,
            emit_mask,
        )

    def run(
            self, task: str,
            *,
            progress: Callable[
                [SharedStateProxyTurn], None] | None = None,
    ) -> SharedStateProxyTeamResult:
        """Run the W48 proxy-coupled team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        proxy_turns: list[SharedStateProxyTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        role_arrival_order: list[str] = []
        causal_counts: dict[str, int] = {
            a.effective_role: 0 for a in self.agents}
        parent_cid: str | None = None
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_wall_ms = 0.0
        total_calls = 0
        n_behavioral_changes = 0
        n_visible_tokens_saved_bh = 0
        n_visible_tokens_added_latent = 0
        n_visible_tokens_added_ss = 0
        n_visible_tokens_added_bh = 0
        n_abstain_substitutions = 0
        n_pseudo_kv_writes = 0
        n_proxy_margin_abstains = 0
        n_proxy_recon_abstains = 0
        ratify_probabilities: list[float] = []
        write_gates: list[float] = []
        recon_l1s: list[float] = []
        head_backend = self.backend
        head_model = getattr(head_backend, "model", "") or ""
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()
        proxy_params_cid = self.registry.params.cid()
        training_trace_cid = (
            self.registry.params.training_trace.cid())
        shared_state_cid = (
            self.registry.params.shared_state.cid())
        shared_state_short = (
            self.registry.params.shared_state.state_hash_short())

        branch_path: list[int] = []
        cycle_path: list[int] = []
        prior_pkv_size = 0

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            branch_id = int(self.branch_id_of_turn(int(idx)))
            cycle_id = int(self.cycle_id_of_turn(int(idx)))
            ctx = LiveTurnContext(
                turn_index=int(idx),
                role_universe=role_universe,
                role_arrival_order=tuple(role_arrival_order),
                current_role=str(role),
                recent_handoffs=tuple(recent_handoffs),
                all_prior_outputs=tuple(recent_handoffs),
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
                branch_id=int(branch_id),
                cycle_id=int(cycle_id),
                expected_spherical=self.expected_spherical,
                expected_subspace=self.expected_subspace,
            )
            # Detect pseudo-KV write that just happened.
            curr_pkv_size = (
                self.orchestrator.pseudo_kv_bank.size)
            if curr_pkv_size > prior_pkv_size:
                n_pseudo_kv_writes += 1
            prior_pkv_size = curr_pkv_size

            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)

            (bounded_prompt, textual_prompt, n_textual_tokens,
             n_actual_tokens, n_latent_ctrl_tokens,
             n_ss_header_tokens, n_bh_tokens,
             n_bh_tokens_saved, n_saved,
             latent_ctrl_witness, branch_history_witness,
             emit_mask) = self._build_prompt(
                member=member,
                task=task,
                turn_index=idx,
                recent_handoffs=recent_handoffs,
                decision=decision,
                branch_path=tuple(branch_path),
                cycle_path=tuple(cycle_path),
                shared_state_short=str(shared_state_short),
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
                if (decision.branch
                        == W48_BRANCH_PROXY_MARGIN_ABSTAIN):
                    n_proxy_margin_abstains += 1
                if (decision.branch
                        == W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN):
                    n_proxy_recon_abstains += 1
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

            if n_saved > 0 and not do_substitute:
                n_visible_tokens_saved_bh += int(n_bh_tokens_saved)
                if n_saved > 0:
                    n_behavioral_changes += 1
            if n_latent_ctrl_tokens > 0 and not do_substitute:
                n_visible_tokens_added_latent += int(
                    n_latent_ctrl_tokens)
            if n_ss_header_tokens > 0 and not do_substitute:
                n_visible_tokens_added_ss += int(
                    n_ss_header_tokens)
            if n_bh_tokens > 0 and not do_substitute:
                n_visible_tokens_added_bh += int(n_bh_tokens)
            ratify_probabilities.append(
                float(decision.forward.ratify_probability))
            write_gates.append(
                float(decision.forward.write_gate_value))
            recon_l1s.append(
                float(decision.forward.reconstruction_l1))

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

            # Build the W48 envelope witnesses.
            ag_forward_witness_cid = (
                _compute_w48_proxy_forward_witness_cid(
                    gate_logit=float(decision.forward.gate_logit),
                    ratify_probability=float(
                        decision.forward.ratify_probability),
                    confidence_bucket=int(
                        decision.forward.confidence_bucket),
                    write_gate_value=float(
                        decision.forward.write_gate_value),
                    branch_bias_value=float(
                        decision.forward.branch_bias_value),
                    reconstruction_l1=float(
                        decision.forward.reconstruction_l1),
                    reconstruction_l2=float(
                        decision.forward.reconstruction_l2),
                    latent_emit_mask=tuple(
                        bool(b) for b in
                        decision.forward.latent_emit_mask),
                    latent_bits_payload=tuple(
                        int(b) for b in
                        decision.forward.latent_bits_payload),
                    turn_index=int(idx),
                    branch_id=int(branch_id),
                    cycle_id=int(cycle_id),
                    role=str(role),
                ))
            construction_cid = (
                _compute_w48_prompt_construction_witness_cid(
                    turn_index=int(idx),
                    role=str(role),
                    prompt_sha256=prompt_sha,
                    n_visible_prompt_tokens_textual=int(
                        n_textual_tokens),
                    n_visible_prompt_tokens_actual=int(
                        n_actual_tokens),
                    n_latent_ctrl_tokens=int(n_latent_ctrl_tokens),
                    n_shared_state_header_tokens=int(
                        n_ss_header_tokens),
                    n_branch_history_tokens=int(n_bh_tokens),
                    shared_state_capsule_cid=shared_state_cid,
                    pseudo_kv_bank_head_cid=(
                        decision.pseudo_kv_bank_head_cid),
                ))
            latent_witness_cid = latent_ctrl_witness.cid()
            branch_history_cid = branch_history_witness.cid()
            inner_outer = ""  # we let the W47 envelope sit inline

            behavioral_change = bool(
                do_substitute or n_saved > 0
                or n_latent_ctrl_tokens > 0
                or n_ss_header_tokens > 0
                or n_bh_tokens > 0)
            proxy_witness_cid = _compute_w48_proxy_witness_cid(
                decision_branch=decision.branch,
                w47_branch=decision.w47_branch,
                w46_branch=decision.w46_branch,
                w45_branch=decision.w45_branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                proxy_params_cid=proxy_params_cid,
                training_trace_cid=training_trace_cid,
                proxy_forward_witness_cid=(
                    ag_forward_witness_cid),
                prompt_construction_witness_cid=construction_cid,
                latent_control_witness_cid=latent_witness_cid,
                branch_history_witness_cid=branch_history_cid,
                shared_state_capsule_cid=shared_state_cid,
                pseudo_kv_bank_head_cid=(
                    decision.pseudo_kv_bank_head_cid),
                inner_w47_outer_cid=str(inner_outer),
                output_sha256=output_sha,
                behavioral_change=behavioral_change,
            )
            outer_cid = _compute_w48_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w47_envelope_cid="",
                proxy_params_cid=proxy_params_cid,
                training_trace_cid=training_trace_cid,
                proxy_witness_cid=proxy_witness_cid,
                turn_index=int(idx),
            )
            envelope = SharedStateProxyHandoffEnvelope(
                schema_version=(
                    W48_SHARED_STATE_PROXY_SCHEMA_VERSION),
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w47_envelope_cid="",
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
                w47_branch=decision.w47_branch,
                w46_branch=decision.w46_branch,
                w45_branch=decision.w45_branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                branch_id=int(branch_id),
                cycle_id=int(cycle_id),
                proxy_params_cid=proxy_params_cid,
                training_trace_cid=training_trace_cid,
                fitting_method=str(
                    self.registry.params.fitting_method),
                shared_state_capsule_cid=shared_state_cid,
                shared_state_dim=int(
                    self.registry.params.shared_state.dim),
                role_state_delta_cid=str(
                    self.registry.params.role_state_delta.cid()),
                role_state_delta_present=bool(
                    str(role)
                    in self.registry.params.role_state_delta
                    .role_factors),
                proxy_attention_cid=str(
                    self.registry.params.proxy_attention.cid()),
                proxy_n_heads=int(
                    self.registry.params.n_heads),
                proxy_factor_dim=int(
                    self.registry.params.factor_dim),
                write_head_cid=str(
                    self.registry.params.write_head.cid()),
                reconstruction_decoder_cid=str(
                    self.registry.params.reconstruction.cid()),
                branch_cycle_bias_cid=str(
                    self.registry.params.branch_cycle_bias.cid()),
                latent_control_cid=str(
                    self.registry.params.latent_control.cid()),
                inner_w47_outer_cid=str(inner_outer),
                inner_w47_params_cid=str(
                    self.registry.params.inner_w47.cid()),
                pseudo_kv_bank_head_cid=str(
                    decision.pseudo_kv_bank_head_cid),
                pseudo_kv_bank_size=int(
                    self.orchestrator.pseudo_kv_bank.size),
                pseudo_kv_capacity=int(
                    self.orchestrator.pseudo_kv_bank.capacity),
                gate_logit=float(
                    decision.forward.gate_logit),
                ratify_probability=float(
                    decision.forward.ratify_probability),
                confidence_bucket=int(
                    decision.forward.confidence_bucket),
                write_gate_value=float(
                    decision.forward.write_gate_value),
                branch_bias_value=float(
                    decision.forward.branch_bias_value),
                reconstruction_l1=float(
                    decision.forward.reconstruction_l1),
                reconstruction_l2=float(
                    decision.forward.reconstruction_l2),
                latent_emit_mask=tuple(
                    bool(b) for b in
                    decision.forward.latent_emit_mask),
                latent_bits_payload=tuple(
                    int(b) for b in
                    decision.forward.latent_bits_payload),
                proxy_forward_witness_cid=str(
                    ag_forward_witness_cid),
                prompt_sha256=prompt_sha,
                prompt_construction_witness_cid=construction_cid,
                latent_control_witness_cid=str(latent_witness_cid),
                branch_history_witness_cid=str(branch_history_cid),
                output_sha256=output_sha,
                n_visible_prompt_tokens_textual=int(
                    n_textual_tokens),
                n_visible_prompt_tokens_actual=int(
                    n_actual_tokens),
                n_visible_prompt_tokens_saved=int(
                    n_textual_tokens - n_actual_tokens),
                n_overhead_tokens=int(
                    n_latent_ctrl_tokens + n_ss_header_tokens
                    + n_bh_tokens),
                n_latent_ctrl_tokens=int(n_latent_ctrl_tokens),
                n_shared_state_header_tokens=int(
                    n_ss_header_tokens),
                n_branch_history_tokens=int(n_bh_tokens),
                n_branch_history_tokens_saved=int(
                    n_bh_tokens_saved),
                behavioral_change=bool(behavioral_change),
                proxy_witness_cid=str(proxy_witness_cid),
                proxy_outer_cid=str(outer_cid),
            )
            proxy_turn = SharedStateProxyTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
            )
            proxy_turns.append(proxy_turn)

            total_prompt_tokens += int(d_prompt)
            total_output_tokens += int(d_output)
            total_wall_ms += float(wall_ms)
            total_calls += int(
                d_calls or (0 if do_substitute else 1))

            recent_handoffs.append((role, output))
            role_arrival_order.append(role)
            if len(recent_handoffs) > self.max_visible_handoffs:
                recent_handoffs = recent_handoffs[
                    -self.max_visible_handoffs:]
            n_w42_visible_tokens = int(visible_count)
            branch_path.append(int(branch_id))
            cycle_path.append(int(cycle_id))

            if progress is not None:
                try:
                    progress(proxy_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[SharedStateProxyTeam] progress callback "
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
        mean_w = (
            sum(write_gates) / len(write_gates)
            if write_gates else 0.0)
        mean_l1 = (
            sum(recon_l1s) / len(recon_l1s)
            if recon_l1s else 0.0)
        return SharedStateProxyTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            proxy_turns=tuple(proxy_turns),
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
            n_behavioral_changes=int(n_behavioral_changes),
            n_visible_tokens_saved_branch_history=int(
                n_visible_tokens_saved_bh),
            n_visible_tokens_added_latent_ctrl=int(
                n_visible_tokens_added_latent),
            n_visible_tokens_added_shared_state_header=int(
                n_visible_tokens_added_ss),
            n_visible_tokens_added_branch_history=int(
                n_visible_tokens_added_bh),
            n_abstain_substitutions=int(n_abstain_substitutions),
            n_pseudo_kv_writes=int(n_pseudo_kv_writes),
            n_proxy_margin_abstains=int(n_proxy_margin_abstains),
            n_proxy_recon_abstains=int(n_proxy_recon_abstains),
            mean_ratify_probability=float(mean_p),
            mean_write_gate=float(mean_w),
            mean_reconstruction_l1=float(mean_l1),
            proxy_params_cid=str(proxy_params_cid),
            training_trace_cid=str(training_trace_cid),
            shared_state_capsule_cid=str(shared_state_cid),
            final_pseudo_kv_bank_head_cid=str(
                self.orchestrator.pseudo_kv_bank.head_cid()),
        )


# =============================================================================
# Shared-state-aware synthetic backend (for r95 model-facing family)
# =============================================================================

@dataclasses.dataclass
class SharedStateAwareSyntheticBackend:
    """Deterministic backend that answers differently when the
    prompt carries a ``SHARED_STATE_HASH:`` token.

    Used by R-95 to exercise the *behavioural* effect of the W48
    shared-state header on a controlled synthetic ground truth.
    Not a real LLM; the response is keyed only on the substrings.
    """

    correct_with_shared_state: str = "SHARED_STATE_OK"
    answer_without_shared_state: str = "SHARED_STATE_NO"
    n_calls: int = 0
    model_tag: str = "synthetic.shared_state_aware"
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
        if "SHARED_STATE_HASH:" in text:
            return self.correct_with_shared_state
        return self.answer_without_shared_state


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Schema / branches / defaults
    "W48_SHARED_STATE_PROXY_SCHEMA_VERSION",
    "W48_TEAM_RESULT_SCHEMA",
    "W48_BRANCH_TRIVIAL_SHARED_STATE_PASSTHROUGH",
    "W48_BRANCH_PROXY_DISABLED",
    "W48_BRANCH_PROXY_RATIFIED",
    "W48_BRANCH_PROXY_NO_POLICY",
    "W48_BRANCH_PROXY_MARGIN_ABSTAIN",
    "W48_BRANCH_PROXY_RECONSTRUCTION_ABSTAIN",
    "W48_BRANCH_PROXY_BRANCH_BIAS_ABSTAIN",
    "W48_BRANCH_PROXY_WRITE_GATE_ABSTAIN",
    "W48_BRANCH_PROXY_TRAIN_FAILURE",
    "W48_BRANCH_PROXY_REJECTED",
    "W48_ALL_BRANCHES",
    "W48_PROXY_ABSTAIN_BRANCHES",
    "W48_ALL_FAILURE_MODES",
    "W48_DEFAULT_SHARED_STATE_DIM",
    "W48_DEFAULT_PSEUDO_KV_SLOTS",
    "W48_DEFAULT_FACTOR_DIM",
    "W48_DEFAULT_N_HEADS",
    "W48_DEFAULT_PROXY_HIDDEN_DIM",
    "W48_DEFAULT_PROXY_DEPTH",
    "W48_DEFAULT_N_BRANCHES",
    "W48_DEFAULT_N_CYCLES",
    "W48_DEFAULT_ROLE_STATE_DELTA_RANK",
    "W48_DEFAULT_RECON_HIDDEN_DIM",
    "W48_DEFAULT_LATENT_CTRL_BITS",
    "W48_DEFAULT_BRANCH_HISTORY_MAX_LEN",
    "W48_DEFAULT_WRITE_GATE_THRESHOLD",
    "W48_DEFAULT_LATENT_CTRL_TAG",
    "W48_DEFAULT_SHARED_STATE_TAG",
    "W48_NO_BRANCH_ID",
    "W48_NO_CYCLE_ID",
    "W48_NO_PSEUDO_KV_SLOT",
    "W48_NO_ROLE_STATE_DELTA",
    # Components
    "SharedStateCapsule",
    "RoleSharedStateDelta",
    "PseudoKVSlot",
    "PseudoKVBank",
    "ProxyAttentionHead",
    "MultiHeadProxyAttention",
    "SlotMemoryWriteHead",
    "ReconstructionDecoder",
    "BranchCycleBias",
    "LatentControlSerializer",
    "LatentControlWitness",
    "BranchHistoryWitness",
    "build_latent_control_string",
    "compress_branch_history",
    "decompress_branch_history",
    # Params + fitter
    "SharedStateProxyParams",
    "build_unfitted_shared_state_proxy_params",
    "SharedStateExample",
    "SharedStateTrainingSet",
    "fit_shared_state_proxy",
    # Forward
    "SharedStateProxyForwardResult",
    "forward_shared_state_proxy",
    # Registry + orchestrator + envelope + verifier + team
    "SharedStateProxyRegistry",
    "SharedStateProxyOrchestrator",
    "SharedStateProxyGatingDecision",
    "build_trivial_shared_state_proxy_registry",
    "build_shared_state_proxy_registry",
    "SharedStateProxyHandoffEnvelope",
    "SharedStateProxyVerificationOutcome",
    "verify_shared_state_proxy_handoff",
    "SharedStateProxyTurn",
    "SharedStateProxyTeamResult",
    "SharedStateProxyTeam",
    # Synthetic backend for the r95 model-facing family
    "SharedStateAwareSyntheticBackend",
]
