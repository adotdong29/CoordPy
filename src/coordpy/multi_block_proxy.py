"""W49 Multi-Block Cross-Bank Coordination (MBCC) — capsule-
native, autograd-trained, multi-block stacked transformer-proxy
with role-conditioned multi-bank pseudo-KV, learned eviction,
retention head, dictionary-codebook compression, and an evolving
content-addressed shared-latent capsule per turn, layered on top
of W48 SSTP, W47 AMS, W46 MMC, W45 LMC, W44 LMCC, and W43 PMC.

W49 is the first capsule-native CoordPy layer where:

  * A **stacked, residual, multi-block proxy transformer** is
    trained end-to-end (each block = multi-head attention +
    feed-forward + residual + scaling).

  * **Multiple role-conditioned pseudo-KV banks** sit beside a
    shared team bank; reads aggregate over the (role bank, shared
    bank) pair with a learned **bank-mix gate**, writes are
    routed by a learned **bank-router** sigmoid.

  * A **learned eviction policy** scores slots before replacement
    when a bank is at capacity; replaces W48's plain FIFO ring.

  * A **retention head** (separate trainable two-layer head)
    answers a binary "was this fact stored?" question against
    the current shared state + flat channels + multi-bank read.

  * A **dictionary codebook** quantises the latent-control payload
    to its nearest codebook entry. A multi-token packed
    ``LATENT_CTRL_V2`` block carries `code + mask + bits` with
    strictly stronger compression than W48's ``LATENT_CTRL``.

  * A **content-addressed shared-latent capsule per turn** evolves
    deterministically across turns; chain-walk recovers every
    prior latent state from the envelope chain alone.

  * A **CrammingWitness** records structured bits, visible-token
    cost, shared-latent capsule size, and the implied
    structured-bits / visible-token ratio.

All of W49 reuses the pure-Python reverse-mode autograd engine
from W47 (`Variable` + `AdamOptimizer`) — no NumPy, no
PyTorch, no JAX dependency. The released SDK v3.43 contract
remains byte-for-byte unchanged. W49 ships at
``coordpy.multi_block_proxy`` and is reachable only through
an explicit import.

Honest scope (do-not-overstate)
-------------------------------

W49 does NOT touch transformer-internal hidden state, KV cache
bytes, attention weights, or embeddings. Every parameter of the
multi-block proxy operates over W43 capsule-layer encodings, the
W47 trainable channel features, and the (now multi-bank) pseudo-
KV factor banks' *capsule-layer* slots. The pseudo-KV banks
reproduce the algebraic interface of a per-role + shared KV cache
at the capsule layer; they do not transplant real KV state from a
transformer's attention layers.

The substrate-blocked W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``) and the W47/W48 carry-
forwards (``W47-C-DEEP-TRANSFORMER-COUPLING``,
``W48-C-REAL-KV-COUPLED-PROXY``,
``W48-C-MULTI-HOST-SHARED-STATE``) are unchanged.

W49 is strictly additive on top of W48 and the released v3.43
SDK. When the multi-block stack is configured trivially
(``multi_block_enabled=False``, ``multi_bank_enabled=False``,
``retention_enabled=False``, ``dictionary_enabled=False``,
``shared_latent_capsule_enabled=False``, W48-trivial inner), the
W49 orchestrator reduces to ``SharedStateProxyTeam.run`` byte-for-
byte — the W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH falsifier.
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
    AutogradManifoldParams,
    ParamTensor,
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
    build_unfitted_autograd_params,
    vdot,
    vmatmul,
    vmean,
    vsoftmax,
    vsum,
)
from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .learned_manifold import (
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
    W45_N_CHANNELS,
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
from .manifold_memory import _flatten_channel_features
from .product_manifold import (
    CellObservation,
    ProductManifoldPolicyEntry,
    SphericalConsensusSignature,
    SubspaceBasis,
)
from .shared_state_proxy import (
    MultiHeadProxyAttention,
    PseudoKVBank,
    PseudoKVSlot,
    ReconstructionDecoder,
    SharedStateCapsule,
    SharedStateProxyParams,
    SharedStateProxyRegistry,
    SharedStateProxyTeam,
    build_shared_state_proxy_registry,
    build_trivial_shared_state_proxy_registry,
    build_unfitted_shared_state_proxy_params,
    W48_DEFAULT_FACTOR_DIM,
    W48_DEFAULT_N_BRANCHES,
    W48_DEFAULT_N_CYCLES,
    W48_DEFAULT_N_HEADS,
    W48_DEFAULT_PSEUDO_KV_SLOTS,
    W48_DEFAULT_SHARED_STATE_DIM,
)
from .team_coord import capsule_team_handoff


# =============================================================================
# Schema, branches, defaults
# =============================================================================

W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION: str = (
    "coordpy.multi_block_proxy.v1")
W49_TEAM_RESULT_SCHEMA: str = (
    "coordpy.multi_block_proxy_team_result.v1")

W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH: str = (
    "multi_block_trivial_passthrough")
W49_BRANCH_MULTI_BLOCK_DISABLED: str = "multi_block_disabled"
W49_BRANCH_MULTI_BLOCK_RATIFIED: str = "multi_block_ratified"
W49_BRANCH_MULTI_BLOCK_NO_POLICY: str = "multi_block_no_policy"
W49_BRANCH_MULTI_BLOCK_MARGIN_ABSTAIN: str = (
    "multi_block_margin_abstain")
W49_BRANCH_MULTI_BLOCK_RETENTION_ABSTAIN: str = (
    "multi_block_retention_abstain")
W49_BRANCH_MULTI_BLOCK_DICTIONARY_ABSTAIN: str = (
    "multi_block_dictionary_abstain")
W49_BRANCH_MULTI_BLOCK_TRAIN_FAILURE: str = (
    "multi_block_train_failure")
W49_BRANCH_MULTI_BLOCK_REJECTED: str = "multi_block_rejected"

W49_ALL_BRANCHES: tuple[str, ...] = (
    W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH,
    W49_BRANCH_MULTI_BLOCK_DISABLED,
    W49_BRANCH_MULTI_BLOCK_RATIFIED,
    W49_BRANCH_MULTI_BLOCK_NO_POLICY,
    W49_BRANCH_MULTI_BLOCK_MARGIN_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_RETENTION_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_DICTIONARY_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_TRAIN_FAILURE,
    W49_BRANCH_MULTI_BLOCK_REJECTED,
)

W49_MULTI_BLOCK_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W49_BRANCH_MULTI_BLOCK_MARGIN_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_RETENTION_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_DICTIONARY_ABSTAIN,
    W49_BRANCH_MULTI_BLOCK_TRAIN_FAILURE,
})

# Defaults.
W49_DEFAULT_N_BLOCKS: int = 2
W49_DEFAULT_FFN_HIDDEN_DIM: int = 8
W49_DEFAULT_ROLE_BANK_CAPACITY: int = 4
W49_DEFAULT_SHARED_BANK_CAPACITY: int = 6
W49_DEFAULT_RETENTION_HIDDEN_DIM: int = 12
W49_DEFAULT_DICTIONARY_SIZE: int = 8
W49_DEFAULT_DICTIONARY_CODE_BITS: int = 3
W49_DEFAULT_LATENT_CTRL_TAG: str = "LATENT_CTRL_V2"
W49_DEFAULT_SHARED_LATENT_TAG: str = "SHARED_LATENT_HASH"
W49_DEFAULT_BANK_ROUTER_THRESHOLD: float = 0.5
W49_DEFAULT_EVICTION_FLOOR: float = 0.0

W49_NO_ROLE_BANK: str = "no_role_bank"
W49_NO_DICT_CODE: int = -1
W49_NO_SHARED_LATENT: str = "no_shared_latent"


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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# Multi-block proxy transformer stack
# =============================================================================

@dataclasses.dataclass
class FeedForwardBlock:
    """Position-wise tanh feed-forward sub-layer.

    ``out = W2 · tanh(W1 · x + b1) + b2`` of shape ``(in_dim) ->
    hidden_dim -> in_dim``. Trainable.
    """

    in_dim: int
    hidden_dim: int
    w1: ParamTensor
    b1: ParamTensor
    w2: ParamTensor
    b2: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            hidden_dim: int = W49_DEFAULT_FFN_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "FeedForwardBlock":
        rng = _DeterministicLCG(seed=int(seed))
        w1 = ParamTensor(
            shape=(int(hidden_dim), int(in_dim)), values=[])
        w1.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b1 = ParamTensor(
            shape=(int(hidden_dim),),
            values=[0.0] * int(hidden_dim))
        w2 = ParamTensor(
            shape=(int(in_dim), int(hidden_dim)), values=[])
        w2.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2 = ParamTensor(
            shape=(int(in_dim),),
            values=[0.0] * int(in_dim))
        return cls(
            in_dim=int(in_dim), hidden_dim=int(hidden_dim),
            w1=w1, b1=b1, w2=w2, b2=b2)

    def params(self) -> list[ParamTensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> list[float]:
        # Hidden = tanh(W1 x + b1)
        hidden = [0.0] * self.hidden_dim
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w1.values[base + j]) \
                        * float(inputs[j])
            s += float(self.b1.values[r])
            hidden[r] = math.tanh(s)
        # Out = W2 hidden + b2
        out = [0.0] * self.in_dim
        for r in range(self.in_dim):
            base = r * self.hidden_dim
            s = 0.0
            for j in range(self.hidden_dim):
                s += float(self.w2.values[base + j]) \
                    * float(hidden[j])
            s += float(self.b2.values[r])
            out[r] = s
        return out

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> list[Variable]:
        w1_vars = self.w1.make_vars()
        b1_vars = self.b1.make_vars()
        w2_vars = self.w2.make_vars()
        b2_vars = self.b2.make_vars()
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(w1_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, list(inputs))
        hidden = [
            (pre_h[i] + b1_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        rows_o: list[list[Variable]] = []
        for r in range(self.in_dim):
            base = r * self.hidden_dim
            rows_o.append(list(w2_vars[base:base + self.hidden_dim]))
        pre_o = vmatmul(rows_o, hidden)
        return [pre_o[i] + b2_vars[i] for i in range(self.in_dim)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "w1": self.w1.to_dict(),
            "b1": self.b1.to_dict(),
            "w2": self.w2.to_dict(),
            "b2": self.b2.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_feedforward_block",
            "block": self.to_dict()})


@dataclasses.dataclass
class ProxyTransformerBlock:
    """One block of the multi-block proxy transformer.

    Each block = (multi-head attention sub-layer + residual)
    + (feed-forward sub-layer + residual). Trainable.
    """

    in_dim: int
    factor_dim: int
    n_heads: int
    attention: MultiHeadProxyAttention
    ffn: FeedForwardBlock
    residual_scale: ParamTensor  # one scalar per output dim

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            factor_dim: int = W48_DEFAULT_FACTOR_DIM,
            n_heads: int = W48_DEFAULT_N_HEADS,
            ffn_hidden_dim: int = W49_DEFAULT_FFN_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ProxyTransformerBlock":
        rng = _DeterministicLCG(seed=int(seed))
        attn = MultiHeadProxyAttention.init(
            in_dim=int(in_dim), factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale),
        )
        ffn = FeedForwardBlock.init(
            in_dim=int(in_dim),
            hidden_dim=int(ffn_hidden_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale),
        )
        rs = ParamTensor(
            shape=(int(in_dim),),
            values=[1.0] * int(in_dim))
        return cls(
            in_dim=int(in_dim), factor_dim=int(factor_dim),
            n_heads=int(n_heads),
            attention=attn, ffn=ffn,
            residual_scale=rs)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.attention.params())
        out.extend(self.ffn.params())
        out.append(self.residual_scale)
        return out

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
    ) -> list[float]:
        attn_out, _ = self.attention.forward_value(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
        )
        # Residual #1: x' = scale * (attn_out) + x
        scaled = [
            float(self.residual_scale.values[i]) * float(attn_out[i])
            + float(query_input[i] if i < len(query_input) else 0.0)
            for i in range(self.in_dim)
        ]
        # Feed-forward + residual #2.
        ffn_out = self.ffn.forward_value(scaled)
        out = [
            float(ffn_out[i]) + float(scaled[i])
            for i in range(self.in_dim)
        ]
        return out

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
    ) -> list[Variable]:
        attn_vars = self.attention.forward_vars(
            query_input=query_input,
            slot_keys=slot_keys,
            slot_values=slot_values,
        )
        rs_vars = self.residual_scale.make_vars()
        scaled: list[Variable] = []
        qi = list(query_input)
        for i in range(self.in_dim):
            x = qi[i] if i < len(qi) else Variable(0.0)
            scaled.append(rs_vars[i] * attn_vars[i] + x)
        ffn_vars = self.ffn.forward_vars(scaled)
        out: list[Variable] = []
        for i in range(self.in_dim):
            out.append(ffn_vars[i] + scaled[i])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "attention": self.attention.to_dict(),
            "ffn": self.ffn.to_dict(),
            "residual_scale": self.residual_scale.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_proxy_transformer_block",
            "block": self.to_dict()})


@dataclasses.dataclass
class MultiBlockProxyStack:
    """A stack of ``L_p`` proxy transformer blocks.

    All blocks share the same ``in_dim`` and ``factor_dim``; each
    has its own trainable weights.
    """

    n_blocks: int
    in_dim: int
    factor_dim: int
    n_heads: int
    blocks: tuple[ProxyTransformerBlock, ...]

    @classmethod
    def init(
            cls, *,
            n_blocks: int = W49_DEFAULT_N_BLOCKS,
            in_dim: int,
            factor_dim: int = W48_DEFAULT_FACTOR_DIM,
            n_heads: int = W48_DEFAULT_N_HEADS,
            ffn_hidden_dim: int = W49_DEFAULT_FFN_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "MultiBlockProxyStack":
        rng = _DeterministicLCG(seed=int(seed))
        blocks: list[ProxyTransformerBlock] = []
        for _ in range(int(n_blocks)):
            blocks.append(ProxyTransformerBlock.init(
                in_dim=int(in_dim),
                factor_dim=int(factor_dim),
                n_heads=int(n_heads),
                ffn_hidden_dim=int(ffn_hidden_dim),
                seed=int(rng.next_uniform() * (1 << 30)),
                init_scale=float(init_scale),
            ))
        return cls(
            n_blocks=int(n_blocks), in_dim=int(in_dim),
            factor_dim=int(factor_dim), n_heads=int(n_heads),
            blocks=tuple(blocks))

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for b in self.blocks:
            out.extend(b.params())
        return out

    def forward_value(
            self, *,
            query_input: Sequence[float],
            slot_keys: Sequence[Sequence[float]],
            slot_values: Sequence[Sequence[float]],
    ) -> list[float]:
        h = list(query_input)
        for b in self.blocks:
            h = b.forward_value(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
            )
        return h

    def forward_vars(
            self, *,
            query_input: Sequence[Variable],
            slot_keys: Sequence[Sequence[Variable]],
            slot_values: Sequence[Sequence[Variable]],
    ) -> list[Variable]:
        h = list(query_input)
        for b in self.blocks:
            h = b.forward_vars(
                query_input=h,
                slot_keys=slot_keys,
                slot_values=slot_values,
            )
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": int(self.n_blocks),
            "in_dim": int(self.in_dim),
            "factor_dim": int(self.factor_dim),
            "n_heads": int(self.n_heads),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_multi_block_proxy_stack",
            "stack": self.to_dict()})


# =============================================================================
# Multi-bank pseudo-KV
# =============================================================================

@dataclasses.dataclass
class MultiBankPseudoKV:
    """Per-role pseudo-KV banks plus a shared team bank.

    Each role has its own :class:`PseudoKVBank` with bounded
    capacity; a separate :class:`PseudoKVBank` is the team-shared
    bank. Reads aggregate over (role bank, shared bank); writes
    are routed by a learned bank-router.
    """

    role_capacity: int
    shared_capacity: int
    factor_dim: int
    role_banks: dict[str, PseudoKVBank] = dataclasses.field(
        default_factory=dict)
    shared_bank: PseudoKVBank | None = None

    def __post_init__(self) -> None:
        if self.shared_bank is None:
            self.shared_bank = PseudoKVBank(
                capacity=int(self.shared_capacity),
                factor_dim=int(self.factor_dim))

    def get_or_init_role_bank(
            self, role: str,
    ) -> PseudoKVBank:
        key = str(role)
        if key not in self.role_banks:
            self.role_banks[key] = PseudoKVBank(
                capacity=int(self.role_capacity),
                factor_dim=int(self.factor_dim))
        return self.role_banks[key]

    def reset(self) -> None:
        self.role_banks = {}
        self.shared_bank = PseudoKVBank(
            capacity=int(self.shared_capacity),
            factor_dim=int(self.factor_dim))

    def total_size(self) -> int:
        n = self.shared_bank.size if self.shared_bank else 0
        for b in self.role_banks.values():
            n += b.size
        return int(n)

    def head_cid(self) -> str:
        payload = {
            "kind": "w49_multi_bank_pseudo_kv_head",
            "role_capacity": int(self.role_capacity),
            "shared_capacity": int(self.shared_capacity),
            "factor_dim": int(self.factor_dim),
            "shared_bank_head": (
                self.shared_bank.head_cid()
                if self.shared_bank else ""),
            "role_bank_heads": {
                role: bank.head_cid()
                for role, bank in sorted(self.role_banks.items())
            },
        }
        return _sha256_hex(payload)

    def role_bank_head_cid(self, role: str) -> str:
        bank = self.role_banks.get(str(role))
        if bank is None:
            return _sha256_hex({"kind": "w49_empty_role_bank"})
        return bank.head_cid()


@dataclasses.dataclass
class BankRouter:
    """Trainable sigmoid routing scalar over (role bank, shared bank).

    ``route_to_shared = sigmoid(w · features)`` — when above
    threshold, the write goes to the shared bank; otherwise to
    the role bank. (Both is supported when above 0.8.)
    """

    in_dim: int
    w_router: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BankRouter":
        w = ParamTensor(shape=(int(in_dim),), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(in_dim=int(in_dim), w_router=w)

    def params(self) -> list[ParamTensor]:
        return [self.w_router]

    def forward_value(self, inputs: Sequence[float]) -> float:
        s = 0.0
        for i in range(min(self.in_dim, len(inputs))):
            s += float(self.w_router.values[i]) * float(inputs[i])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w_vars = self.w_router.make_vars()
        return vdot(list(w_vars), list(inputs)).sigmoid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "w_router": self.w_router.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_bank_router",
            "router": self.to_dict()})


@dataclasses.dataclass
class BankMixGate:
    """Trainable per-read mix gate over (role read, shared read).

    ``mix_role = sigmoid(w · features)`` weights the role-bank
    pooled value vs the shared-bank pooled value; the mixed
    pseudo-KV read is ``mix_role * role_read +
    (1 - mix_role) * shared_read``.
    """

    in_dim: int
    w_mix: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BankMixGate":
        w = ParamTensor(shape=(int(in_dim),), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(in_dim=int(in_dim), w_mix=w)

    def params(self) -> list[ParamTensor]:
        return [self.w_mix]

    def forward_value(self, inputs: Sequence[float]) -> float:
        s = 0.0
        for i in range(min(self.in_dim, len(inputs))):
            s += float(self.w_mix.values[i]) * float(inputs[i])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w_vars = self.w_mix.make_vars()
        return vdot(list(w_vars), list(inputs)).sigmoid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "w_mix": self.w_mix.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_bank_mix_gate",
            "gate": self.to_dict()})


# =============================================================================
# Learned eviction policy
# =============================================================================

@dataclasses.dataclass
class EvictionPolicy:
    """Trainable sigmoid eviction-score head.

    Given a slot's (age, role_match_flag, write_gate_value)
    features, scores the slot. The slot with the **lowest** score
    is evicted when a bank is at capacity (lowest = least worth
    keeping). Trained via BCE against a synthetic "should I
    evict?" target.
    """

    in_dim: int  # default 3: (age_normalized, role_match, write_gate)
    w_evict: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int = 3,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "EvictionPolicy":
        w = ParamTensor(shape=(int(in_dim),), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(in_dim=int(in_dim), w_evict=w)

    def params(self) -> list[ParamTensor]:
        return [self.w_evict]

    def score_value(self, inputs: Sequence[float]) -> float:
        s = 0.0
        for i in range(min(self.in_dim, len(inputs))):
            s += float(self.w_evict.values[i]) * float(inputs[i])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w_vars = self.w_evict.make_vars()
        return vdot(list(w_vars), list(inputs)).sigmoid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "w_evict": self.w_evict.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_eviction_policy",
            "policy": self.to_dict()})

    def evict_index(
            self, *, bank: PseudoKVBank, current_role: str,
            current_turn: int,
    ) -> int:
        """Index in the bank with the lowest keep-score.

        Returns -1 if the bank is empty (no eviction).
        """
        if not bank.slots:
            return -1
        scores: list[tuple[int, float]] = []
        for i, s in enumerate(bank.slots):
            age = float(max(0, int(current_turn) - int(s.turn_index)))
            age_norm = age / float(
                max(1, int(bank.capacity) + 1))
            role_match = (
                1.0 if str(s.role) == str(current_role) else 0.0)
            wg = float(s.write_gate_value)
            score = self.score_value([age_norm, role_match, wg])
            scores.append((i, score))
        # Lowest score = least worth keeping = evict.
        scores.sort(key=lambda x: x[1])
        return int(scores[0][0])


# =============================================================================
# Retention head
# =============================================================================

@dataclasses.dataclass
class RetentionHead:
    """Trainable two-layer head answering the binary
    "was this fact stored?" question.

    ``q = sigmoid(W2 · tanh(W1 · x + b1) + b2)`` where ``x`` is
    ``(shared_state, flat_channels, multi_bank_read,
    target_fact_hash)``.
    """

    in_dim: int
    hidden_dim: int
    w1: ParamTensor
    b1: ParamTensor
    w2: ParamTensor
    b2: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            hidden_dim: int = W49_DEFAULT_RETENTION_HIDDEN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "RetentionHead":
        rng = _DeterministicLCG(seed=int(seed))
        w1 = ParamTensor(
            shape=(int(hidden_dim), int(in_dim)), values=[])
        w1.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b1 = ParamTensor(
            shape=(int(hidden_dim),),
            values=[0.0] * int(hidden_dim))
        w2 = ParamTensor(
            shape=(int(hidden_dim),), values=[])
        w2.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2 = ParamTensor(shape=(1,), values=[0.0])
        return cls(
            in_dim=int(in_dim), hidden_dim=int(hidden_dim),
            w1=w1, b1=b1, w2=w2, b2=b2)

    def params(self) -> list[ParamTensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> float:
        hidden = [0.0] * self.hidden_dim
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w1.values[base + j]) \
                        * float(inputs[j])
            s += float(self.b1.values[r])
            hidden[r] = math.tanh(s)
        s = 0.0
        for j in range(self.hidden_dim):
            s += float(self.w2.values[j]) * float(hidden[j])
        s += float(self.b2.values[0])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w1_vars = self.w1.make_vars()
        b1_vars = self.b1.make_vars()
        w2_vars = self.w2.make_vars()
        b2_vars = self.b2.make_vars()
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(w1_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, list(inputs))
        hidden = [
            (pre_h[i] + b1_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        s = vdot(list(w2_vars), hidden) + b2_vars[0]
        return s.sigmoid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "w1": self.w1.to_dict(),
            "b1": self.b1.to_dict(),
            "w2": self.w2.to_dict(),
            "b2": self.b2.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_retention_head",
            "head": self.to_dict()})


# =============================================================================
# Dictionary codebook compression
# =============================================================================

@dataclasses.dataclass
class DictionaryCodebook:
    """Trainable K-prototype codebook over the latent-control
    payload vector.

    Encode = argmin_k ||x - C_k||; decode = C_k. Trained via the
    soft-assignment cross-entropy (closest prototype should have
    highest softmax weight on the target index).
    """

    n_codes: int
    code_dim: int
    prototypes: ParamTensor  # shape (n_codes, code_dim)

    @classmethod
    def init(
            cls, *,
            n_codes: int = W49_DEFAULT_DICTIONARY_SIZE,
            code_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DictionaryCodebook":
        p = ParamTensor(
            shape=(int(n_codes), int(code_dim)), values=[])
        p.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(
            n_codes=int(n_codes), code_dim=int(code_dim),
            prototypes=p)

    def params(self) -> list[ParamTensor]:
        return [self.prototypes]

    def code_vector(self, code: int) -> tuple[float, ...]:
        c = int(code) % max(1, self.n_codes)
        base = c * self.code_dim
        return tuple(
            float(self.prototypes.values[base + j])
            for j in range(self.code_dim))

    def encode_value(self, x: Sequence[float]) -> int:
        # argmin_k ||x - C_k||
        best_k = 0
        best_d = float("inf")
        for k in range(self.n_codes):
            base = k * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                xj = float(x[j]) if j < len(x) else 0.0
                diff = float(self.prototypes.values[base + j]) - xj
                d += diff * diff
            if d < best_d:
                best_d = d
                best_k = k
        return int(best_k)

    def encode_soft_vars(
            self, x: Sequence[Variable],
    ) -> tuple[list[Variable], list[Variable]]:
        """Soft-assignment forward.

        Returns ``(softmax_weights_over_codes, distances)``. The
        softmax is taken over negative distances so the closest
        code has the highest weight.
        """
        p_vars = self.prototypes.make_vars()
        neg_dists: list[Variable] = []
        for k in range(self.n_codes):
            base = k * self.code_dim
            terms: list[Variable] = []
            for j in range(self.code_dim):
                if j < len(x):
                    diff = p_vars[base + j] - x[j]
                else:
                    diff = p_vars[base + j]
                terms.append(diff * diff)
            d = vsum(terms)
            neg_dists.append(-1.0 * d)
        weights = vsoftmax(neg_dists)
        return list(weights), neg_dists

    def code_bits(self) -> int:
        return max(1, int(math.ceil(math.log2(max(2, self.n_codes)))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_codes": int(self.n_codes),
            "code_dim": int(self.code_dim),
            "prototypes": self.prototypes.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_dictionary_codebook",
            "codebook": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LatentControlV2Witness:
    """Sealed witness for the W49 packed latent control bytes."""

    ctrl_tag: str
    dictionary_code: int
    code_bits: int
    n_mask_bits: int
    emit_mask: tuple[bool, ...]
    bits_payload: tuple[int, ...]
    shared_latent_hash_short: str
    n_ctrl_tokens: int
    ctrl_bytes_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctrl_tag": str(self.ctrl_tag),
            "dictionary_code": int(self.dictionary_code),
            "code_bits": int(self.code_bits),
            "n_mask_bits": int(self.n_mask_bits),
            "emit_mask": [bool(b) for b in self.emit_mask],
            "bits_payload": [int(b) for b in self.bits_payload],
            "shared_latent_hash_short":
                str(self.shared_latent_hash_short),
            "n_ctrl_tokens": int(self.n_ctrl_tokens),
            "ctrl_bytes_sha256": str(self.ctrl_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_latent_control_v2_witness",
            "witness": self.to_dict()})


def build_latent_control_v2_string(
        *,
        ctrl_tag: str,
        dictionary_code: int,
        code_bits: int,
        emit_mask: Sequence[bool],
        bits_payload: Sequence[int],
        shared_latent_hash_short: str,
) -> tuple[str, LatentControlV2Witness]:
    """Build the literal ``LATENT_CTRL_V2: ...`` line + witness.

    When ``emit_mask`` is empty the ``mask=`` field is omitted;
    when ``bits_payload`` is empty the ``bits=`` field is omitted.
    This keeps the dictionary-code-only form maximally compact.
    """
    mask_str = "".join("1" if b else "0" for b in emit_mask)
    bits_str = "".join(
        str(int(b) & 1) for b in bits_payload)
    parts = [
        f"{ctrl_tag}:",
        f"SHARED_LATENT_HASH={shared_latent_hash_short}",
        f"code={int(dictionary_code)}/{int(code_bits)}b",
    ]
    if emit_mask:
        parts.append(f"mask={mask_str}")
    if bits_payload:
        parts.append(f"bits={bits_str}")
    body = " ".join(parts)
    n_tokens = len(body.split())
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    witness = LatentControlV2Witness(
        ctrl_tag=str(ctrl_tag),
        dictionary_code=int(dictionary_code),
        code_bits=int(code_bits),
        n_mask_bits=int(len(emit_mask)),
        emit_mask=tuple(bool(b) for b in emit_mask),
        bits_payload=tuple(int(b) & 1 for b in bits_payload),
        shared_latent_hash_short=str(shared_latent_hash_short),
        n_ctrl_tokens=int(n_tokens),
        ctrl_bytes_sha256=str(sha),
    )
    return body, witness


# =============================================================================
# Shared-latent capsule (per turn, evolving)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedLatentCapsule:
    """Per-turn evolving shared-latent capsule.

    The latent at turn `t` is the projection of the multi-block
    output of turn `t-1`. CID is content-addressed. Chain-walk
    recovers all prior latent states.
    """

    turn_index: int
    role: str
    dim: int
    values: tuple[float, ...]
    parent_capsule_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "dim": int(self.dim),
            "values": _round_floats(self.values),
            "parent_capsule_cid": str(self.parent_capsule_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_shared_latent_capsule",
            "capsule": self.to_dict()})

    def hash_short(self) -> str:
        return self.cid()[:12]


@dataclasses.dataclass
class SharedLatentProjector:
    """Trainable projection from multi-block output to shared
    latent capsule values.

    ``latent = tanh(W · h + b)``.
    """

    in_dim: int
    out_dim: int
    w_proj: ParamTensor
    b_proj: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            out_dim: int,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "SharedLatentProjector":
        w = ParamTensor(
            shape=(int(out_dim), int(in_dim)), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        b = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            in_dim=int(in_dim), out_dim=int(out_dim),
            w_proj=w, b_proj=b)

    def params(self) -> list[ParamTensor]:
        return [self.w_proj, self.b_proj]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> list[float]:
        out = [0.0] * self.out_dim
        for r in range(self.out_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w_proj.values[base + j]) \
                        * float(inputs[j])
            s += float(self.b_proj.values[r])
            out[r] = math.tanh(s)
        return out

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w_proj.make_vars()
        b_vars = self.b_proj.make_vars()
        rows: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.in_dim
            rows.append(list(w_vars[base:base + self.in_dim]))
        pre = vmatmul(rows, list(inputs))
        return [
            (pre[i] + b_vars[i]).tanh()
            for i in range(self.out_dim)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "out_dim": int(self.out_dim),
            "w_proj": self.w_proj.to_dict(),
            "b_proj": self.b_proj.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_shared_latent_projector",
            "projector": self.to_dict()})


# =============================================================================
# Cramming witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrammingWitness:
    """Per-turn cramming witness.

    Records (i) structured bits (dictionary code + emit mask +
    bits payload), (ii) visible-token cost of the W49 control
    block, (iii) shared-latent capsule size in bytes, and
    (iv) the implied structured-bits / visible-token ratio.
    """

    structured_bits: int
    visible_ctrl_tokens: int
    visible_latent_header_tokens: int
    shared_latent_capsule_bytes: int
    bits_per_visible_token: float
    cramming_bytes_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_bits": int(self.structured_bits),
            "visible_ctrl_tokens": int(self.visible_ctrl_tokens),
            "visible_latent_header_tokens": int(
                self.visible_latent_header_tokens),
            "shared_latent_capsule_bytes": int(
                self.shared_latent_capsule_bytes),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "cramming_bytes_sha256": str(
                self.cramming_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_cramming_witness",
            "witness": self.to_dict()})


def build_cramming_witness(
        *,
        dictionary_code: int,
        code_bits: int,
        emit_mask: Sequence[bool],
        bits_payload: Sequence[int],
        visible_ctrl_tokens: int,
        visible_latent_header_tokens: int,
        shared_latent_capsule_bytes: int,
) -> CrammingWitness:
    structured_bits = (
        int(code_bits)
        + int(len(emit_mask))
        + int(len(bits_payload))
    )
    visible = (
        int(visible_ctrl_tokens)
        + int(visible_latent_header_tokens))
    ratio = (
        float(structured_bits) / float(max(1, visible)))
    blob = json.dumps({
        "dictionary_code": int(dictionary_code),
        "code_bits": int(code_bits),
        "emit_mask": [bool(b) for b in emit_mask],
        "bits_payload": [int(b) for b in bits_payload],
        "visible_ctrl_tokens": int(visible_ctrl_tokens),
        "visible_latent_header_tokens": int(
            visible_latent_header_tokens),
        "shared_latent_capsule_bytes": int(
            shared_latent_capsule_bytes),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    return CrammingWitness(
        structured_bits=int(structured_bits),
        visible_ctrl_tokens=int(visible_ctrl_tokens),
        visible_latent_header_tokens=int(
            visible_latent_header_tokens),
        shared_latent_capsule_bytes=int(
            shared_latent_capsule_bytes),
        bits_per_visible_token=float(ratio),
        cramming_bytes_sha256=str(sha),
    )


# =============================================================================
# W49 parameters bundle + training trace
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiBlockProxyParams:
    """All trained, content-addressed W49 params.

    Wraps the W48 inner ``SharedStateProxyParams`` + W49-specific
    new params.
    """

    inner_w48: SharedStateProxyParams
    multi_block_stack: MultiBlockProxyStack
    bank_router: BankRouter
    bank_mix_gate: BankMixGate
    eviction_policy: EvictionPolicy
    retention_head: RetentionHead
    dictionary: DictionaryCodebook
    shared_latent_projector: SharedLatentProjector
    fitting_method: str
    training_trace_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_w48_cid": str(self.inner_w48.cid()),
            "multi_block_stack": self.multi_block_stack.to_dict(),
            "bank_router": self.bank_router.to_dict(),
            "bank_mix_gate": self.bank_mix_gate.to_dict(),
            "eviction_policy": self.eviction_policy.to_dict(),
            "retention_head": self.retention_head.to_dict(),
            "dictionary": self.dictionary.to_dict(),
            "shared_latent_projector":
                self.shared_latent_projector.to_dict(),
            "fitting_method": str(self.fitting_method),
            "training_trace_cid": str(self.training_trace_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_multi_block_proxy_params",
            "params": self.to_dict()})

    @property
    def n_blocks(self) -> int:
        return int(self.multi_block_stack.n_blocks)

    @property
    def n_codes(self) -> int:
        return int(self.dictionary.n_codes)

    @property
    def shared_latent_dim(self) -> int:
        return int(self.shared_latent_projector.out_dim)


@dataclasses.dataclass(frozen=True)
class MultiBlockTrainingTraceWitness:
    """Sealed training-trace witness for W49 fit."""

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
            "optimizer_config": self.optimizer_config,
            "init_scale": float(round(self.init_scale, 12)),
            "loss_history_head": [
                float(round(v, 12))
                for v in self.loss_history_head],
            "loss_history_tail": [
                float(round(v, 12))
                for v in self.loss_history_tail],
            "grad_norm_head": [
                float(round(v, 12))
                for v in self.grad_norm_head],
            "grad_norm_tail": [
                float(round(v, 12))
                for v in self.grad_norm_tail],
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
            "kind": "w49_multi_block_training_trace_witness",
            "witness": self.to_dict()})


def build_unfitted_multi_block_proxy_params(
        *,
        inner_w48: SharedStateProxyParams | None = None,
        roles: Sequence[str] = (),
        n_blocks: int = W49_DEFAULT_N_BLOCKS,
        ffn_hidden_dim: int = W49_DEFAULT_FFN_HIDDEN_DIM,
        n_codes: int = W49_DEFAULT_DICTIONARY_SIZE,
        retention_hidden_dim: int = W49_DEFAULT_RETENTION_HIDDEN_DIM,
        shared_latent_dim: int = W48_DEFAULT_SHARED_STATE_DIM,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
) -> MultiBlockProxyParams:
    """Build a fully-initialised (but untrained) W49 bundle."""
    inner = inner_w48 or build_unfitted_shared_state_proxy_params(
        roles=tuple(roles),
        feature_dim=int(feature_dim),
        seed=int(seed),
        init_scale=float(init_scale),
    )
    # The multi-block stack reads / writes on the same shape as
    # the W48 proxy attention's in_dim.
    proxy_in_dim = int(inner.proxy_attention.in_dim)
    factor_dim = int(inner.factor_dim)
    n_heads = int(inner.n_heads)
    stack = MultiBlockProxyStack.init(
        n_blocks=int(n_blocks),
        in_dim=int(proxy_in_dim),
        factor_dim=int(factor_dim),
        n_heads=int(n_heads),
        ffn_hidden_dim=int(ffn_hidden_dim),
        seed=int(seed) + 71,
        init_scale=float(init_scale),
    )
    router_in_dim = int(W45_N_CHANNELS) * int(feature_dim) + int(
        inner.shared_state_dim)
    router = BankRouter.init(
        in_dim=int(router_in_dim),
        seed=int(seed) + 81,
        init_scale=float(init_scale),
    )
    mix_gate = BankMixGate.init(
        in_dim=int(router_in_dim),
        seed=int(seed) + 91,
        init_scale=float(init_scale),
    )
    eviction = EvictionPolicy.init(
        in_dim=3,
        seed=int(seed) + 101,
        init_scale=float(init_scale),
    )
    # Retention head input: shared_state + flat_channels +
    # multi_bank_read (factor_dim) + target_hash (factor_dim).
    retention_in_dim = (
        int(inner.shared_state_dim)
        + int(W45_N_CHANNELS) * int(feature_dim)
        + 2 * int(factor_dim)
    )
    retention = RetentionHead.init(
        in_dim=int(retention_in_dim),
        hidden_dim=int(retention_hidden_dim),
        seed=int(seed) + 111,
        init_scale=float(init_scale),
    )
    # Dictionary code dim = factor_dim (the slot value dim).
    dictionary = DictionaryCodebook.init(
        n_codes=int(n_codes),
        code_dim=int(factor_dim),
        seed=int(seed) + 121,
        init_scale=float(init_scale),
    )
    # Shared-latent projector: from multi-block stack output back
    # to shared_latent_dim.
    projector = SharedLatentProjector.init(
        in_dim=int(proxy_in_dim),
        out_dim=int(shared_latent_dim),
        seed=int(seed) + 131,
        init_scale=float(init_scale),
    )
    return MultiBlockProxyParams(
        inner_w48=inner,
        multi_block_stack=stack,
        bank_router=router,
        bank_mix_gate=mix_gate,
        eviction_policy=eviction,
        retention_head=retention,
        dictionary=dictionary,
        shared_latent_projector=projector,
        fitting_method="unfitted",
        training_trace_cid=MultiBlockTrainingTraceWitness(
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
        ).cid(),
    )


# =============================================================================
# Training set + trainer
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiBlockExample:
    """One training example for the W49 multi-block regimes."""

    role: str
    channel_features: tuple[tuple[str, tuple[float, ...]], ...]
    branch_id: int
    cycle_id: int
    label: float
    target_recon: tuple[float, ...] | None = None
    retention_label: float = 1.0
    dictionary_target: int = 0
    eviction_target: float = 0.5
    target_fact_hash: tuple[float, ...] = ()

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
            "retention_label": float(self.retention_label),
            "dictionary_target": int(self.dictionary_target),
            "eviction_target": float(self.eviction_target),
            "target_fact_hash": list(self.target_fact_hash),
        }


@dataclasses.dataclass(frozen=True)
class MultiBlockTrainingSet:
    examples: tuple[MultiBlockExample, ...]
    feature_dim: int = W45_DEFAULT_FEATURE_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "examples": [e.to_dict() for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w49_multi_block_training_set",
            "set": self.to_dict()})


def _flatten_example(
        ex: MultiBlockExample, *, feature_dim: int,
) -> list[float]:
    fmap = ex.channel_features_map
    out: list[float] = []
    for c in W45_CHANNEL_ORDER:
        feats = list(fmap.get(c, ()))[:feature_dim]
        while len(feats) < feature_dim:
            feats.append(0.0)
        out.extend(float(v) for v in feats)
    return out


def fit_multi_block_proxy(
        training_set: MultiBlockTrainingSet,
        *,
        inner_w48: SharedStateProxyParams | None = None,
        n_blocks: int = W49_DEFAULT_N_BLOCKS,
        ffn_hidden_dim: int = W49_DEFAULT_FFN_HIDDEN_DIM,
        n_codes: int = W49_DEFAULT_DICTIONARY_SIZE,
        retention_hidden_dim: int = W49_DEFAULT_RETENTION_HIDDEN_DIM,
        shared_latent_dim: int = W48_DEFAULT_SHARED_STATE_DIM,
        n_steps: int = 60,
        learning_rate: float = W47_DEFAULT_LEARNING_RATE,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        retention_loss_weight: float = 0.5,
        dictionary_loss_weight: float = 0.5,
        eviction_loss_weight: float = 0.3,
        history_head: int = 8,
        history_tail: int = 8,
) -> MultiBlockProxyParams:
    """Fit the W49 multi-block proxy via Adam SGD.

    Joint loss = classifier BCE + retention BCE + dictionary CE
    + eviction BCE.
    """
    fd = int(training_set.feature_dim)
    inner = inner_w48 or build_unfitted_shared_state_proxy_params(
        roles=tuple(
            sorted({str(ex.role)
                    for ex in training_set.examples})),
        feature_dim=fd,
        seed=int(seed),
        init_scale=float(init_scale),
    )
    roles = tuple(
        sorted({str(ex.role) for ex in training_set.examples}))
    params = build_unfitted_multi_block_proxy_params(
        inner_w48=inner, roles=roles,
        n_blocks=int(n_blocks),
        ffn_hidden_dim=int(ffn_hidden_dim),
        n_codes=int(n_codes),
        retention_hidden_dim=int(retention_hidden_dim),
        shared_latent_dim=int(shared_latent_dim),
        feature_dim=fd,
        seed=int(seed),
        init_scale=float(init_scale),
    )

    # Flatten examples.
    flat_examples: list[list[float]] = []
    labels_pos: list[bool] = []
    retention_targets: list[float] = []
    dict_targets: list[int] = []
    eviction_targets: list[float] = []
    fact_hashes: list[list[float]] = []
    roles_list: list[str] = []
    for ex in training_set.examples:
        flat = _flatten_example(ex, feature_dim=fd)
        flat_examples.append(flat)
        labels_pos.append(bool(float(ex.label) > 0.0))
        retention_targets.append(float(ex.retention_label))
        dict_targets.append(int(ex.dictionary_target))
        eviction_targets.append(float(ex.eviction_target))
        fh = list(ex.target_fact_hash)
        while len(fh) < int(inner.factor_dim):
            fh.append(0.0)
        fact_hashes.append(fh[:int(inner.factor_dim)])
        roles_list.append(str(ex.role))

    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable: list[ParamTensor] = []
    trainable.extend(params.multi_block_stack.params())
    trainable.extend(params.bank_router.params())
    trainable.extend(params.bank_mix_gate.params())
    trainable.extend(params.eviction_policy.params())
    trainable.extend(params.retention_head.params())
    trainable.extend(params.dictionary.params())
    trainable.extend(params.shared_latent_projector.params())

    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False

    shared_values = list(inner.shared_state.values)
    flat_channel_dim = int(W45_N_CHANNELS) * fd
    proxy_in_dim = int(inner.proxy_attention.in_dim)
    factor_dim = int(inner.factor_dim)
    n_examples = len(flat_examples)

    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        ex_idx = step % max(1, n_examples)
        flat = flat_examples[ex_idx]
        role = roles_list[ex_idx]
        # Build the input vector to the multi-block stack.
        shared_state_vars = [
            Variable(float(shared_values[j]))
            for j in range(int(inner.shared_state_dim))]
        flat_vars = [Variable(float(v)) for v in flat]
        pkv_read_vars = [
            Variable(0.0) for _ in range(int(factor_dim))]
        query_vars = (
            shared_state_vars + flat_vars + pkv_read_vars)
        while len(query_vars) < proxy_in_dim:
            query_vars.append(Variable(0.0))
        query_vars = query_vars[:proxy_in_dim]
        # Slots: synthetic single-slot derived from current input.
        slot_vec = list(query_vars)
        slot_keys = [slot_vec]
        slot_values = [slot_vec]
        h_vars = params.multi_block_stack.forward_vars(
            query_input=query_vars,
            slot_keys=slot_keys,
            slot_values=slot_values,
        )
        gate_logit = vsum(h_vars) * (1.0 / float(max(1, len(h_vars))))
        prob = gate_logit.sigmoid()
        if labels_pos[ex_idx]:
            cls_loss = -1.0 * (prob + 1e-9).log()
        else:
            cls_loss = -1.0 * (
                (Variable(1.0) - prob) + 1e-9).log()

        # Bank router + mix gate independent forward.
        router_in_vars = (
            [Variable(float(shared_values[j]))
             for j in range(int(inner.shared_state_dim))]
            + [Variable(float(v)) for v in flat])
        # Trim/pad to router input dim.
        while len(router_in_vars) < params.bank_router.in_dim:
            router_in_vars.append(Variable(0.0))
        router_in_vars = router_in_vars[
            :params.bank_router.in_dim]
        router_prob = params.bank_router.forward_vars(
            router_in_vars)
        # Router target: 1 if role is even-indexed (synthetic).
        router_target = 1.0 if (
            roles.index(role) % 2 == 0) else 0.0
        if router_target > 0.5:
            router_loss = -1.0 * (router_prob + 1e-9).log()
        else:
            router_loss = -1.0 * (
                (Variable(1.0) - router_prob) + 1e-9).log()

        # Mix gate target: 1 = prefer role bank, 0 = prefer shared.
        mix_in_vars = (
            [Variable(float(shared_values[j]))
             for j in range(int(inner.shared_state_dim))]
            + [Variable(float(v)) for v in flat])
        while len(mix_in_vars) < params.bank_mix_gate.in_dim:
            mix_in_vars.append(Variable(0.0))
        mix_in_vars = mix_in_vars[
            :params.bank_mix_gate.in_dim]
        mix_prob = params.bank_mix_gate.forward_vars(mix_in_vars)
        # Mix target = the same router pattern but inverted (just
        # to give it independent supervision).
        mix_target = 1.0 - router_target
        if mix_target > 0.5:
            mix_loss = -1.0 * (mix_prob + 1e-9).log()
        else:
            mix_loss = -1.0 * (
                (Variable(1.0) - mix_prob) + 1e-9).log()

        # Eviction loss: trained against a synthetic
        # (age, role_match, write_gate) -> eviction_target signal.
        evict_in_vars = [
            Variable(float(ex_idx) / float(max(1, n_examples))),
            Variable(1.0 if (ex_idx % 2 == 0) else 0.0),
            Variable(0.5),
        ]
        evict_prob = params.eviction_policy.forward_vars(
            evict_in_vars)
        et = float(eviction_targets[ex_idx])
        if et > 0.5:
            evict_loss = -1.0 * (evict_prob + 1e-9).log()
        else:
            evict_loss = -1.0 * (
                (Variable(1.0) - evict_prob) + 1e-9).log()

        # Retention loss.
        retention_in_vars = (
            [Variable(float(shared_values[j]))
             for j in range(int(inner.shared_state_dim))]
            + [Variable(float(v)) for v in flat]
            + [Variable(0.0) for _ in range(int(factor_dim))]  # multi-bank read
            + [Variable(float(v)) for v in fact_hashes[ex_idx]])
        while len(retention_in_vars) < params.retention_head.in_dim:
            retention_in_vars.append(Variable(0.0))
        retention_in_vars = retention_in_vars[
            :params.retention_head.in_dim]
        retention_prob = params.retention_head.forward_vars(
            retention_in_vars)
        rt = float(retention_targets[ex_idx])
        if rt > 0.5:
            retention_loss = -1.0 * (retention_prob + 1e-9).log()
        else:
            retention_loss = -1.0 * (
                (Variable(1.0) - retention_prob) + 1e-9).log()

        # Dictionary cross-entropy loss: pick the multi-block
        # output projected down to ``factor_dim`` and target the
        # ``dictionary_target`` index via soft-assignment.
        proj_input_vars = h_vars[:factor_dim] + [
            Variable(0.0)
            for _ in range(max(0, factor_dim - len(h_vars)))]
        proj_input_vars = proj_input_vars[:factor_dim]
        soft_weights, _ = params.dictionary.encode_soft_vars(
            proj_input_vars)
        # Target index.
        ti = int(dict_targets[ex_idx]) % max(
            1, params.dictionary.n_codes)
        dict_loss = -1.0 * (soft_weights[ti] + 1e-9).log()

        # Shared-latent projector loss: tied to the W48
        # shared state values (L2 to the current shared state,
        # so the latent has a bias toward the base state). This
        # is a regulariser; the projector learns to *evolve*
        # the latent via downstream feedback.
        proj_out_vars = (
            params.shared_latent_projector.forward_vars(h_vars))
        proj_target = [
            float(shared_values[j])
            for j in range(int(inner.shared_state_dim))]
        proj_terms = []
        for j in range(len(proj_out_vars)):
            tj = (proj_target[j]
                  if j < len(proj_target) else 0.0)
            diff = proj_out_vars[j] - Variable(float(tj))
            proj_terms.append(diff * diff)
        proj_loss = vmean(proj_terms)

        loss = (
            cls_loss
            + 0.5 * router_loss
            + 0.5 * mix_loss
            + eviction_loss_weight * evict_loss
            + retention_loss_weight * retention_loss
            + dictionary_loss_weight * dict_loss
            + 0.25 * proj_loss
        )
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

    # Build the trained bundle.
    fitted_no_cid = MultiBlockProxyParams(
        inner_w48=inner,
        multi_block_stack=params.multi_block_stack,
        bank_router=params.bank_router,
        bank_mix_gate=params.bank_mix_gate,
        eviction_policy=params.eviction_policy,
        retention_head=params.retention_head,
        dictionary=params.dictionary,
        shared_latent_projector=params.shared_latent_projector,
        fitting_method=(
            "multi_block_proxy_adam_v1" if not diverged
            else "multi_block_proxy_diverged"),
        training_trace_cid="",
    )
    final_params_cid = _sha256_hex({
        "kind": "w49_multi_block_proxy_params_inner",
        "params_dict": fitted_no_cid.to_dict(),
    })
    trace = MultiBlockTrainingTraceWitness(
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
        final_params_cid=str(final_params_cid),
        training_set_cid=str(training_set.cid()),
        diverged=bool(diverged),
    )
    return MultiBlockProxyParams(
        inner_w48=inner,
        multi_block_stack=params.multi_block_stack,
        bank_router=params.bank_router,
        bank_mix_gate=params.bank_mix_gate,
        eviction_policy=params.eviction_policy,
        retention_head=params.retention_head,
        dictionary=params.dictionary,
        shared_latent_projector=params.shared_latent_projector,
        fitting_method=fitted_no_cid.fitting_method,
        training_trace_cid=str(trace.cid()),
    )


# =============================================================================
# Forward pass (inference)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiBlockProxyForwardResult:
    """One W49 forward pass at inference time."""

    multi_block_output: tuple[float, ...]
    role_bank_read: tuple[float, ...]
    shared_bank_read: tuple[float, ...]
    multi_bank_read: tuple[float, ...]
    bank_router_value: float
    bank_mix_gate_value: float
    retention_prob: float
    dictionary_code: int
    dictionary_code_bits: int
    shared_latent_values: tuple[float, ...]
    gate_logit: float
    ratify_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "multi_block_output": _round_floats(
                self.multi_block_output),
            "role_bank_read": _round_floats(self.role_bank_read),
            "shared_bank_read": _round_floats(
                self.shared_bank_read),
            "multi_bank_read": _round_floats(
                self.multi_bank_read),
            "bank_router_value": float(round(
                self.bank_router_value, 12)),
            "bank_mix_gate_value": float(round(
                self.bank_mix_gate_value, 12)),
            "retention_prob": float(round(
                self.retention_prob, 12)),
            "dictionary_code": int(self.dictionary_code),
            "dictionary_code_bits": int(
                self.dictionary_code_bits),
            "shared_latent_values": _round_floats(
                self.shared_latent_values),
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
        }


def _bank_read(
        bank: PseudoKVBank, *,
        turn_index: int, query: Sequence[float],
        factor_dim: int,
) -> list[float]:
    admissible = bank.admissible_for_turn(int(turn_index))
    if not admissible:
        return [0.0] * int(factor_dim)
    # Dot-product softmax read over admissible slots.
    qs = list(query)[:factor_dim]
    while len(qs) < factor_dim:
        qs.append(0.0)
    scores = []
    for s in admissible:
        sk = list(s.key)[:factor_dim]
        while len(sk) < factor_dim:
            sk.append(0.0)
        scores.append(sum(qs[i] * sk[i] for i in range(factor_dim))
                      / math.sqrt(max(1.0, float(factor_dim))))
    attn = _softmax(scores)
    pooled = [0.0] * factor_dim
    for i, w in enumerate(attn):
        sv = list(admissible[i].value)[:factor_dim]
        while len(sv) < factor_dim:
            sv.append(0.0)
        for r in range(factor_dim):
            pooled[r] += w * sv[r]
    return pooled


def forward_multi_block_proxy(
        *,
        channel_features: Mapping[str, Sequence[float]],
        params: MultiBlockProxyParams,
        role: str,
        multi_bank: MultiBankPseudoKV,
        turn_index: int,
        prev_shared_latent_cid: str = "",
        target_fact_hash: Sequence[float] = (),
        multi_block_enabled: bool = True,
        multi_bank_enabled: bool = True,
        retention_enabled: bool = True,
        dictionary_enabled: bool = True,
        shared_latent_capsule_enabled: bool = True,
) -> tuple[MultiBlockProxyForwardResult, SharedLatentCapsule]:
    """W49 forward pass at inference. Returns the forward result
    + the new shared-latent capsule for this turn."""
    inner = params.inner_w48
    fd = int(inner.inner_w47.feature_dim)
    flat_channel_dim = int(W45_N_CHANNELS) * fd
    flat_query = list(_flatten_channel_features(
        channel_features, feature_dim=fd))
    factor_dim = int(inner.factor_dim)
    shared_state_dim = int(inner.shared_state_dim)
    proxy_in_dim = int(inner.proxy_attention.in_dim)

    # Build the multi-block input.
    shared_values = list(inner.shared_state.values)
    # Per-role delta value (carried from W48).
    role_delta = list(
        inner.role_state_delta.forward_value(role=str(role)))
    shared_with_delta = [
        float(shared_values[i])
        + (role_delta[i] if i < len(role_delta) else 0.0)
        for i in range(shared_state_dim)
    ]

    # Compute multi-bank reads first; the multi-bank read goes
    # into the multi-block input.
    role_bank = multi_bank.get_or_init_role_bank(str(role))
    shared_bank = multi_bank.shared_bank
    role_read = (
        _bank_read(role_bank, turn_index=int(turn_index),
                   query=flat_query, factor_dim=factor_dim)
        if (multi_bank_enabled and role_bank.size > 0)
        else [0.0] * factor_dim)
    shared_read = (
        _bank_read(shared_bank, turn_index=int(turn_index),
                   query=flat_query, factor_dim=factor_dim)
        if (multi_bank_enabled
            and shared_bank is not None and shared_bank.size > 0)
        else [0.0] * factor_dim)
    # Bank mix gate over (role_read, shared_read).
    mix_input = list(shared_with_delta) + list(flat_query)
    while len(mix_input) < params.bank_mix_gate.in_dim:
        mix_input.append(0.0)
    mix_input = mix_input[:params.bank_mix_gate.in_dim]
    bank_mix_value = float(
        params.bank_mix_gate.forward_value(mix_input)
        if multi_bank_enabled else 0.5)
    multi_bank_read = [
        bank_mix_value * role_read[r]
        + (1.0 - bank_mix_value) * shared_read[r]
        for r in range(factor_dim)
    ]

    # Bank router value (recorded but not used in forward gating).
    router_input = list(shared_with_delta) + list(flat_query)
    while len(router_input) < params.bank_router.in_dim:
        router_input.append(0.0)
    router_input = router_input[:params.bank_router.in_dim]
    router_value = float(
        params.bank_router.forward_value(router_input)
        if multi_bank_enabled else 0.5)

    # Build the multi-block input.
    query = list(shared_with_delta) + list(flat_query) + list(
        multi_bank_read)
    while len(query) < proxy_in_dim:
        query.append(0.0)
    query = query[:proxy_in_dim]
    # Synthetic single-slot derived from multi-bank read.
    slot_vec = list(query)
    slot_keys = [slot_vec]
    slot_values = [slot_vec]
    if multi_block_enabled:
        mb_out = params.multi_block_stack.forward_value(
            query_input=query,
            slot_keys=slot_keys,
            slot_values=slot_values,
        )
    else:
        mb_out = [0.0] * proxy_in_dim

    # Retention head.
    retention_input = (
        list(shared_with_delta) + list(flat_query)
        + list(multi_bank_read))
    tfh = list(target_fact_hash)[:factor_dim]
    while len(tfh) < factor_dim:
        tfh.append(0.0)
    retention_input += tfh
    while len(retention_input) < params.retention_head.in_dim:
        retention_input.append(0.0)
    retention_input = retention_input[
        :params.retention_head.in_dim]
    retention_prob_val = float(
        params.retention_head.forward_value(retention_input)
        if retention_enabled else 0.5)

    # Dictionary encode.
    proj_input = list(mb_out)[:factor_dim]
    while len(proj_input) < factor_dim:
        proj_input.append(0.0)
    if dictionary_enabled:
        code = int(params.dictionary.encode_value(proj_input))
    else:
        code = 0
    code_bits = int(params.dictionary.code_bits())

    # Shared-latent capsule (per-turn, evolving).
    if shared_latent_capsule_enabled:
        latent_values = (
            params.shared_latent_projector.forward_value(mb_out))
    else:
        latent_values = [0.0] * (
            params.shared_latent_projector.out_dim)

    new_latent_capsule = SharedLatentCapsule(
        turn_index=int(turn_index),
        role=str(role),
        dim=int(params.shared_latent_projector.out_dim),
        values=tuple(_round_floats(latent_values)),
        parent_capsule_cid=str(prev_shared_latent_cid),
    )

    # Gate logit: mean over multi-block output.
    if mb_out:
        gate_logit = float(sum(mb_out)) / float(max(1, len(mb_out)))
    else:
        gate_logit = 0.0
    ratify_prob = float(_stable_sigmoid(gate_logit))

    fr = MultiBlockProxyForwardResult(
        multi_block_output=tuple(_round_floats(mb_out)),
        role_bank_read=tuple(_round_floats(role_read)),
        shared_bank_read=tuple(_round_floats(shared_read)),
        multi_bank_read=tuple(_round_floats(multi_bank_read)),
        bank_router_value=float(round(router_value, 12)),
        bank_mix_gate_value=float(round(bank_mix_value, 12)),
        retention_prob=float(round(retention_prob_val, 12)),
        dictionary_code=int(code),
        dictionary_code_bits=int(code_bits),
        shared_latent_values=tuple(_round_floats(latent_values)),
        gate_logit=float(round(gate_logit, 12)),
        ratify_probability=float(round(ratify_prob, 12)),
    )
    return fr, new_latent_capsule


# =============================================================================
# Registry + Orchestrator
# =============================================================================

@dataclasses.dataclass
class MultiBlockProxyRegistry:
    """Controller-side configuration for the W49 multi-block proxy."""

    schema_cid: str
    inner_w48_registry: SharedStateProxyRegistry
    params: MultiBlockProxyParams
    multi_block_enabled: bool = True
    multi_bank_enabled: bool = True
    retention_enabled: bool = True
    dictionary_enabled: bool = True
    shared_latent_capsule_enabled: bool = True
    eviction_enabled: bool = True
    role_bank_capacity: int = W49_DEFAULT_ROLE_BANK_CAPACITY
    shared_bank_capacity: int = W49_DEFAULT_SHARED_BANK_CAPACITY
    bank_router_threshold: float = (
        W49_DEFAULT_BANK_ROUTER_THRESHOLD)
    margin_abstain_threshold: float = -10.0
    retention_floor: float = 0.0  # disabled by default
    eviction_floor: float = W49_DEFAULT_EVICTION_FLOOR
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT
    latent_control_tag: str = W49_DEFAULT_LATENT_CTRL_TAG
    shared_latent_tag: str = W49_DEFAULT_SHARED_LATENT_TAG

    @property
    def is_trivial(self) -> bool:
        return (
            self.inner_w48_registry.is_trivial
            and not self.multi_block_enabled
            and not self.multi_bank_enabled
            and not self.retention_enabled
            and not self.dictionary_enabled
            and not self.shared_latent_capsule_enabled
            and not self.eviction_enabled
            and self.params.fitting_method == "unfitted"
        )


def build_trivial_multi_block_proxy_registry(
        *, schema_cid: str | None = None,
) -> MultiBlockProxyRegistry:
    """Build a registry whose orchestrator reduces to
    :class:`SharedStateProxyTeam` (trivial) byte-for-byte."""
    cid = schema_cid or _sha256_hex({
        "kind": "w49_trivial_schema"})
    inner = build_trivial_shared_state_proxy_registry(
        schema_cid=str(cid))
    p = build_unfitted_multi_block_proxy_params(
        inner_w48=inner.params)
    return MultiBlockProxyRegistry(
        schema_cid=str(cid),
        inner_w48_registry=inner,
        params=p,
        multi_block_enabled=False,
        multi_bank_enabled=False,
        retention_enabled=False,
        dictionary_enabled=False,
        shared_latent_capsule_enabled=False,
        eviction_enabled=False,
        abstain_substitution_enabled=False,
    )


def build_multi_block_proxy_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        params: MultiBlockProxyParams | None = None,
        inner_w48_params: SharedStateProxyParams | None = None,
        multi_block_enabled: bool = True,
        multi_bank_enabled: bool = True,
        retention_enabled: bool = True,
        dictionary_enabled: bool = True,
        shared_latent_capsule_enabled: bool = True,
        eviction_enabled: bool = True,
        role_bank_capacity: int = W49_DEFAULT_ROLE_BANK_CAPACITY,
        shared_bank_capacity: int = (
            W49_DEFAULT_SHARED_BANK_CAPACITY),
        bank_router_threshold: float = (
            W49_DEFAULT_BANK_ROUTER_THRESHOLD),
        margin_abstain_threshold: float = -10.0,
        retention_floor: float = 0.0,
        eviction_floor: float = W49_DEFAULT_EVICTION_FLOOR,
        abstain_substitution_enabled: bool = True,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
        latent_control_tag: str = W49_DEFAULT_LATENT_CTRL_TAG,
        shared_latent_tag: str = W49_DEFAULT_SHARED_LATENT_TAG,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
) -> MultiBlockProxyRegistry:
    """Build a fully configured W49 multi-block-proxy registry."""
    inner = build_shared_state_proxy_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        params=inner_w48_params,
        margin_abstain_threshold=margin_abstain_threshold,
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
        abstain_substitution_enabled=False,  # W49 owns abstain
        abstain_output=str(abstain_output),
    )
    p = params or build_unfitted_multi_block_proxy_params(
        inner_w48=inner.params)
    return MultiBlockProxyRegistry(
        schema_cid=str(schema_cid),
        inner_w48_registry=inner,
        params=p,
        multi_block_enabled=bool(multi_block_enabled),
        multi_bank_enabled=bool(multi_bank_enabled),
        retention_enabled=bool(retention_enabled),
        dictionary_enabled=bool(dictionary_enabled),
        shared_latent_capsule_enabled=bool(
            shared_latent_capsule_enabled),
        eviction_enabled=bool(eviction_enabled),
        role_bank_capacity=int(role_bank_capacity),
        shared_bank_capacity=int(shared_bank_capacity),
        bank_router_threshold=float(bank_router_threshold),
        margin_abstain_threshold=float(margin_abstain_threshold),
        retention_floor=float(retention_floor),
        eviction_floor=float(eviction_floor),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        latent_control_tag=str(latent_control_tag),
        shared_latent_tag=str(shared_latent_tag),
    )


@dataclasses.dataclass(frozen=True)
class MultiBlockProxyGatingDecision:
    """Result of running the W49 multi-block gate on one turn."""

    branch: str
    w48_branch: str
    role: str
    role_handoff_signature_cid: str
    policy_entry_cid: str
    branch_id: int
    cycle_id: int
    shared_latent_capsule: SharedLatentCapsule
    multi_bank_head_cid: str
    role_bank_head_cid: str
    shared_bank_head_cid: str
    forward: MultiBlockProxyForwardResult
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W49_MULTI_BLOCK_ABSTAIN_BRANCHES


class MultiBlockProxyOrchestrator:
    """Per-turn W49 gating + envelope binding.

    Wraps a W48 :class:`SharedStateProxyOrchestrator` (via the W48
    inner registry) plus a :class:`MultiBlockProxyRegistry`.
    Stateful in the multi-bank pseudo-KV + the underlying W48
    state.
    """

    def __init__(
            self, registry: MultiBlockProxyRegistry,
    ) -> None:
        self.registry = registry
        from .shared_state_proxy import (
            SharedStateProxyOrchestrator,
        )
        self._inner = SharedStateProxyOrchestrator(
            registry=registry.inner_w48_registry)
        self._multi_bank = MultiBankPseudoKV(
            role_capacity=int(registry.role_bank_capacity),
            shared_capacity=int(registry.shared_bank_capacity),
            factor_dim=int(registry.params.inner_w48.factor_dim),
        )
        self._last_shared_latent_cid: str = ""

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    @property
    def multi_bank(self) -> MultiBankPseudoKV:
        return self._multi_bank

    def reset_session(self) -> None:
        self._inner.reset_session()
        self._multi_bank.reset()
        self._last_shared_latent_cid = ""

    def _write_to_bank(
            self, *, bank: PseudoKVBank, slot: PseudoKVSlot,
            current_role: str, current_turn: int,
    ) -> None:
        """Insert a slot into a bank, evicting if at capacity.

        If at capacity and eviction is enabled, the trained
        eviction policy picks the slot with the lowest keep-score
        to drop; otherwise FIFO is used (W48 behaviour).
        """
        if bank.size < bank.capacity:
            bank.write(slot)
            return
        if not self.registry.eviction_enabled:
            bank.write(slot)
            return
        idx = self.registry.params.eviction_policy.evict_index(
            bank=bank, current_role=str(current_role),
            current_turn=int(current_turn))
        if idx < 0 or idx >= len(bank.slots):
            bank.write(slot)
            return
        bank.slots.pop(int(idx))
        # Re-index remaining.
        for i, s in enumerate(bank.slots):
            bank.slots[i] = dataclasses.replace(s, slot_index=i)
        bank.write(slot)

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
            target_fact_hash: Sequence[float] = (),
            expected_spherical: SphericalConsensusSignature | None = None,
            expected_subspace: SubspaceBasis | None = None,
    ) -> tuple[MultiBlockProxyGatingDecision, Any]:
        # Delegate to the W48 inner.
        w48_decision, w48_aux = self._inner.gate(
            observation=observation,
            role=str(role),
            role_handoff_signature_cid=role_handoff_signature_cid,
            parent_w42_cid=str(parent_w42_cid),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            turn_index=int(turn_index),
            branch_id=int(branch_id),
            cycle_id=int(cycle_id),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )
        (w43_result, causal_mask, bundle,
         w45_decision, w46_decision, w46_forward, ag_forward,
         w47_decision) = w48_aux

        from .learned_manifold import (
            _channel_features_from_bundle,
        )
        feats = _channel_features_from_bundle(
            bundle,
            feature_dim=int(
                self.registry.params.inner_w48.inner_w47.feature_dim),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )

        forward, new_latent_capsule = forward_multi_block_proxy(
            channel_features=feats,
            params=self.registry.params,
            role=str(role),
            multi_bank=self._multi_bank,
            turn_index=int(turn_index),
            prev_shared_latent_cid=str(
                self._last_shared_latent_cid),
            target_fact_hash=target_fact_hash,
            multi_block_enabled=bool(
                self.registry.multi_block_enabled),
            multi_bank_enabled=bool(
                self.registry.multi_bank_enabled),
            retention_enabled=bool(
                self.registry.retention_enabled),
            dictionary_enabled=bool(
                self.registry.dictionary_enabled),
            shared_latent_capsule_enabled=bool(
                self.registry.shared_latent_capsule_enabled),
        )

        # Branch selection.
        if self.registry.is_trivial:
            branch = W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH
            abstain_reason = ""
        elif not self.registry.multi_block_enabled:
            branch = W49_BRANCH_MULTI_BLOCK_DISABLED
            abstain_reason = ""
        elif (forward.gate_logit
                < float(self.registry.margin_abstain_threshold)):
            branch = W49_BRANCH_MULTI_BLOCK_MARGIN_ABSTAIN
            abstain_reason = "multi_block_margin"
        elif (self.registry.retention_enabled
                and forward.retention_prob
                < float(self.registry.retention_floor)):
            branch = W49_BRANCH_MULTI_BLOCK_RETENTION_ABSTAIN
            abstain_reason = "multi_block_retention"
        elif (self.registry.dictionary_enabled
                and forward.dictionary_code < 0):
            branch = W49_BRANCH_MULTI_BLOCK_DICTIONARY_ABSTAIN
            abstain_reason = "multi_block_dictionary"
        else:
            branch = W49_BRANCH_MULTI_BLOCK_RATIFIED
            abstain_reason = ""

        # Multi-bank write decision.
        if (self.registry.multi_bank_enabled
                and branch == W49_BRANCH_MULTI_BLOCK_RATIFIED):
            # Build slot.
            inner_params = self.registry.params.inner_w48
            factor_dim = int(inner_params.factor_dim)
            flat = list(_flatten_channel_features(
                feats, feature_dim=int(
                    inner_params.inner_w47.feature_dim)))
            key = flat[:factor_dim]
            while len(key) < factor_dim:
                key.append(0.0)
            value = list(forward.multi_block_output)[:factor_dim]
            while len(value) < factor_dim:
                value.append(0.0)
            slot = PseudoKVSlot(
                slot_index=0,
                turn_index=int(turn_index),
                role=str(role),
                key=tuple(_round_floats(key)),
                value=tuple(_round_floats(value)),
                write_gate_value=float(round(
                    forward.bank_router_value, 12)),
                source_observation_cid=_sha256_hex({
                    "kind": "w49_multi_bank_slot_source",
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
                }),
            )
            # Routing: write to role bank if router < threshold,
            # shared bank otherwise. If close to 0.5, write to both.
            r_val = float(forward.bank_router_value)
            wrote_to_role = False
            wrote_to_shared = False
            if r_val < float(
                    self.registry.bank_router_threshold):
                # Role bank.
                role_bank = (
                    self._multi_bank.get_or_init_role_bank(
                        str(role)))
                # Re-index slot to be in this bank.
                slot_in_role = dataclasses.replace(
                    slot, slot_index=int(role_bank.size))
                self._write_to_bank(
                    bank=role_bank, slot=slot_in_role,
                    current_role=str(role),
                    current_turn=int(turn_index))
                wrote_to_role = True
            if r_val >= float(
                    self.registry.bank_router_threshold):
                # Shared bank.
                shared_bank = self._multi_bank.shared_bank
                if shared_bank is not None:
                    slot_in_shared = dataclasses.replace(
                        slot, slot_index=int(shared_bank.size))
                    self._write_to_bank(
                        bank=shared_bank, slot=slot_in_shared,
                        current_role=str(role),
                        current_turn=int(turn_index))
                    wrote_to_shared = True

        # Update last shared latent cid.
        if self.registry.shared_latent_capsule_enabled:
            self._last_shared_latent_cid = new_latent_capsule.cid()
        else:
            self._last_shared_latent_cid = ""

        decision = MultiBlockProxyGatingDecision(
            branch=str(branch),
            w48_branch=str(w48_decision.branch),
            role=str(role),
            role_handoff_signature_cid=str(
                w48_decision.role_handoff_signature_cid),
            policy_entry_cid=str(w48_decision.policy_entry_cid),
            branch_id=int(branch_id),
            cycle_id=int(cycle_id),
            shared_latent_capsule=new_latent_capsule,
            multi_bank_head_cid=str(
                self._multi_bank.head_cid()),
            role_bank_head_cid=str(
                self._multi_bank.role_bank_head_cid(str(role))),
            shared_bank_head_cid=str(
                self._multi_bank.shared_bank.head_cid()
                if self._multi_bank.shared_bank else ""),
            forward=forward,
            abstain_reason=str(abstain_reason),
        )
        return decision, (
            w43_result, causal_mask, bundle, w45_decision,
            w46_decision, w46_forward, ag_forward, w47_decision,
            w48_decision,
        )


# =============================================================================
# Envelope + verifier
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiBlockProxyHandoffEnvelope:
    """Sealed W49 envelope for one turn."""

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w48_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
    w48_branch: str
    abstain_reason: str
    role_handoff_signature_cid: str
    policy_entry_cid: str
    branch_id: int
    cycle_id: int

    # Param provenance.
    multi_block_params_cid: str
    multi_block_stack_cid: str
    bank_router_cid: str
    bank_mix_gate_cid: str
    eviction_policy_cid: str
    retention_head_cid: str
    dictionary_cid: str
    shared_latent_projector_cid: str
    training_trace_cid: str
    fitting_method: str
    n_blocks: int

    inner_w48_params_cid: str
    inner_w48_proxy_witness_cid: str

    # Multi-bank state.
    multi_bank_head_cid: str
    role_bank_head_cid: str
    shared_bank_head_cid: str
    role_bank_size: int
    shared_bank_size: int

    # Forward witnesses.
    gate_logit: float
    ratify_probability: float
    bank_router_value: float
    bank_mix_gate_value: float
    retention_prob: float
    dictionary_code: int
    dictionary_code_bits: int
    multi_block_forward_witness_cid: str

    # Shared latent capsule.
    shared_latent_capsule_cid: str
    shared_latent_parent_cid: str
    shared_latent_dim: int

    # Latent control + cramming.
    latent_ctrl_v2_witness_cid: str
    cramming_witness_cid: str

    # Prompt / output.
    prompt_sha256: str
    output_sha256: str

    # Token accounting.
    n_visible_prompt_tokens_textual: int
    n_visible_prompt_tokens_actual: int
    n_visible_prompt_tokens_saved: int
    n_latent_ctrl_v2_tokens: int
    n_shared_latent_header_tokens: int
    n_overhead_tokens: int

    behavioral_change: bool

    multi_block_witness_cid: str
    multi_block_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w49_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w48_envelope_cid=self.parent_w48_envelope_cid,
            multi_block_params_cid=self.multi_block_params_cid,
            training_trace_cid=self.training_trace_cid,
            multi_block_witness_cid=self.multi_block_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_w49_forward_witness_cid(
        *,
        gate_logit: float,
        ratify_probability: float,
        bank_router_value: float,
        bank_mix_gate_value: float,
        retention_prob: float,
        dictionary_code: int,
        dictionary_code_bits: int,
        shared_latent_capsule_cid: str,
        turn_index: int,
        role: str,
) -> str:
    return _sha256_hex({
        "kind": "w49_multi_block_forward_witness",
        "gate_logit": float(round(gate_logit, 12)),
        "ratify_probability": float(round(
            ratify_probability, 12)),
        "bank_router_value": float(round(bank_router_value, 12)),
        "bank_mix_gate_value": float(round(
            bank_mix_gate_value, 12)),
        "retention_prob": float(round(retention_prob, 12)),
        "dictionary_code": int(dictionary_code),
        "dictionary_code_bits": int(dictionary_code_bits),
        "shared_latent_capsule_cid": str(
            shared_latent_capsule_cid),
        "turn_index": int(turn_index),
        "role": str(role),
    })


def _compute_w49_witness_cid(
        *,
        decision_branch: str,
        w48_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        multi_block_params_cid: str,
        training_trace_cid: str,
        multi_block_forward_witness_cid: str,
        shared_latent_capsule_cid: str,
        multi_bank_head_cid: str,
        latent_ctrl_v2_witness_cid: str,
        cramming_witness_cid: str,
        inner_w48_proxy_witness_cid: str,
        output_sha256: str,
        behavioral_change: bool,
        multi_block_stack_cid: str,
        bank_router_cid: str,
        bank_mix_gate_cid: str,
        eviction_policy_cid: str,
        retention_head_cid: str,
        dictionary_cid: str,
        shared_latent_projector_cid: str,
) -> str:
    return _sha256_hex({
        "kind": "w49_multi_block_witness",
        "decision_branch": str(decision_branch),
        "w48_branch": str(w48_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "multi_block_params_cid": str(multi_block_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "multi_block_forward_witness_cid": str(
            multi_block_forward_witness_cid),
        "shared_latent_capsule_cid": str(
            shared_latent_capsule_cid),
        "multi_bank_head_cid": str(multi_bank_head_cid),
        "latent_ctrl_v2_witness_cid": str(
            latent_ctrl_v2_witness_cid),
        "cramming_witness_cid": str(cramming_witness_cid),
        "inner_w48_proxy_witness_cid": str(
            inner_w48_proxy_witness_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
        "multi_block_stack_cid": str(multi_block_stack_cid),
        "bank_router_cid": str(bank_router_cid),
        "bank_mix_gate_cid": str(bank_mix_gate_cid),
        "eviction_policy_cid": str(eviction_policy_cid),
        "retention_head_cid": str(retention_head_cid),
        "dictionary_cid": str(dictionary_cid),
        "shared_latent_projector_cid": str(
            shared_latent_projector_cid),
    })


def _compute_w49_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w48_envelope_cid: str,
        multi_block_params_cid: str,
        training_trace_cid: str,
        multi_block_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w49_multi_block_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w48_envelope_cid": str(parent_w48_envelope_cid),
        "multi_block_params_cid": str(multi_block_params_cid),
        "training_trace_cid": str(training_trace_cid),
        "multi_block_witness_cid": str(multi_block_witness_cid),
        "turn_index": int(turn_index),
    })


W49_ALL_FAILURE_MODES: tuple[str, ...] = (
    "empty_w49_envelope",
    "w49_schema_version_unknown",
    "w49_schema_cid_mismatch",
    "w49_decision_branch_unknown",
    "w49_role_handoff_signature_cid_invalid",
    "w49_multi_block_params_cid_invalid",
    "w49_training_trace_cid_invalid",
    "w49_multi_block_stack_cid_invalid",
    "w49_bank_router_cid_invalid",
    "w49_bank_mix_gate_cid_invalid",
    "w49_eviction_policy_cid_invalid",
    "w49_retention_head_cid_invalid",
    "w49_dictionary_cid_invalid",
    "w49_shared_latent_projector_cid_invalid",
    "w49_shared_latent_capsule_cid_invalid",
    "w49_multi_bank_head_cid_invalid",
    "w49_latent_ctrl_v2_witness_cid_invalid",
    "w49_cramming_witness_cid_invalid",
    "w49_multi_block_forward_witness_cid_mismatch",
    "w49_multi_block_witness_cid_mismatch",
    "w49_outer_cid_mismatch",
    "w49_token_accounting_invalid",
)


@dataclasses.dataclass(frozen=True)
class MultiBlockProxyVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


def verify_multi_block_proxy_handoff(
        env: "MultiBlockProxyHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_multi_block_params_cid: str | None = None,
        registered_training_trace_cid: str | None = None,
) -> MultiBlockProxyVerificationOutcome:
    """Pure-function verifier for the W49 envelope."""
    n = 0
    if env is None:
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="empty_w49_envelope", n_checks=0)
    n += 1
    if env.schema_version != (
            W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W49_ALL_BRANCHES:
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.decision_branch != (
            W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return MultiBlockProxyVerificationOutcome(
                ok=False,
                reason="w49_role_handoff_signature_cid_invalid",
                n_checks=n)
    n += 1
    if (not env.multi_block_params_cid
            or len(env.multi_block_params_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False,
            reason="w49_multi_block_params_cid_invalid",
            n_checks=n)
    n += 1
    if registered_multi_block_params_cid is not None:
        if env.multi_block_params_cid != str(
                registered_multi_block_params_cid):
            return MultiBlockProxyVerificationOutcome(
                ok=False,
                reason="w49_multi_block_params_cid_invalid",
                n_checks=n)
    if (not env.training_trace_cid
            or len(env.training_trace_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_training_trace_cid_invalid",
            n_checks=n)
    n += 1
    if registered_training_trace_cid is not None:
        if env.training_trace_cid != str(
                registered_training_trace_cid):
            return MultiBlockProxyVerificationOutcome(
                ok=False, reason="w49_training_trace_cid_invalid",
                n_checks=n)
    if (not env.multi_block_stack_cid
            or len(env.multi_block_stack_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_multi_block_stack_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.bank_router_cid
            or len(env.bank_router_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_bank_router_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.bank_mix_gate_cid
            or len(env.bank_mix_gate_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_bank_mix_gate_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.eviction_policy_cid
            or len(env.eviction_policy_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_eviction_policy_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.retention_head_cid
            or len(env.retention_head_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_retention_head_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.dictionary_cid
            or len(env.dictionary_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_dictionary_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.shared_latent_projector_cid
            or len(env.shared_latent_projector_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False,
            reason="w49_shared_latent_projector_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.shared_latent_capsule_cid
            or len(env.shared_latent_capsule_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False,
            reason="w49_shared_latent_capsule_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.multi_bank_head_cid
            or len(env.multi_bank_head_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_multi_bank_head_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.latent_ctrl_v2_witness_cid
            or len(env.latent_ctrl_v2_witness_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False,
            reason="w49_latent_ctrl_v2_witness_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.cramming_witness_cid
            or len(env.cramming_witness_cid) != 64):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_cramming_witness_cid_invalid",
            n_checks=n)
    n += 1
    if (env.n_visible_prompt_tokens_textual < 0
            or env.n_visible_prompt_tokens_actual < 0
            or env.n_overhead_tokens < 0
            or env.n_latent_ctrl_v2_tokens < 0
            or env.n_shared_latent_header_tokens < 0
            or env.n_visible_prompt_tokens_saved
            != (int(env.n_visible_prompt_tokens_textual)
                - int(env.n_visible_prompt_tokens_actual))):
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_token_accounting_invalid",
            n_checks=n)
    n += 1
    expected_forward = _compute_w49_forward_witness_cid(
        gate_logit=float(env.gate_logit),
        ratify_probability=float(env.ratify_probability),
        bank_router_value=float(env.bank_router_value),
        bank_mix_gate_value=float(env.bank_mix_gate_value),
        retention_prob=float(env.retention_prob),
        dictionary_code=int(env.dictionary_code),
        dictionary_code_bits=int(env.dictionary_code_bits),
        shared_latent_capsule_cid=env.shared_latent_capsule_cid,
        turn_index=int(env.turn_index),
        role=env.role,
    )
    if expected_forward != env.multi_block_forward_witness_cid:
        return MultiBlockProxyVerificationOutcome(
            ok=False,
            reason="w49_multi_block_forward_witness_cid_mismatch",
            n_checks=n)
    n += 1
    expected_witness = _compute_w49_witness_cid(
        decision_branch=env.decision_branch,
        w48_branch=env.w48_branch,
        abstain_reason=env.abstain_reason,
        role_handoff_signature_cid=env.role_handoff_signature_cid,
        policy_entry_cid=env.policy_entry_cid,
        multi_block_params_cid=env.multi_block_params_cid,
        training_trace_cid=env.training_trace_cid,
        multi_block_forward_witness_cid=(
            env.multi_block_forward_witness_cid),
        shared_latent_capsule_cid=env.shared_latent_capsule_cid,
        multi_bank_head_cid=env.multi_bank_head_cid,
        latent_ctrl_v2_witness_cid=env.latent_ctrl_v2_witness_cid,
        cramming_witness_cid=env.cramming_witness_cid,
        inner_w48_proxy_witness_cid=env.inner_w48_proxy_witness_cid,
        output_sha256=env.output_sha256,
        behavioral_change=bool(env.behavioral_change),
        multi_block_stack_cid=env.multi_block_stack_cid,
        bank_router_cid=env.bank_router_cid,
        bank_mix_gate_cid=env.bank_mix_gate_cid,
        eviction_policy_cid=env.eviction_policy_cid,
        retention_head_cid=env.retention_head_cid,
        dictionary_cid=env.dictionary_cid,
        shared_latent_projector_cid=env.shared_latent_projector_cid,
    )
    if expected_witness != env.multi_block_witness_cid:
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_multi_block_witness_cid_mismatch",
            n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.multi_block_outer_cid:
        return MultiBlockProxyVerificationOutcome(
            ok=False, reason="w49_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return MultiBlockProxyVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Team
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiBlockProxyTurn:
    """One turn of a :class:`MultiBlockProxyTeam` run."""

    agent_turn: AgentTurn
    decision: MultiBlockProxyGatingDecision
    envelope: MultiBlockProxyHandoffEnvelope
    cramming_witness: CrammingWitness


@dataclasses.dataclass(frozen=True)
class MultiBlockProxyTeamResult:
    """Result of a :class:`MultiBlockProxyTeam` run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    multi_block_turns: tuple[MultiBlockProxyTurn, ...]
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
    n_abstain_substitutions: int = 0
    n_role_bank_writes: int = 0
    n_shared_bank_writes: int = 0
    n_dictionary_codes_emitted: int = 0
    mean_retention_prob: float = 0.0
    mean_bank_router_value: float = 0.0
    mean_bank_mix_gate_value: float = 0.0
    mean_bits_per_visible_token: float = 0.0
    n_visible_tokens_added_latent_v2: int = 0
    n_visible_tokens_added_shared_latent: int = 0
    multi_block_params_cid: str = ""
    training_trace_cid: str = ""
    shared_latent_chain_cids: tuple[str, ...] = ()
    final_multi_bank_head_cid: str = ""
    schema: str = W49_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


class MultiBlockProxyTeam:
    """W49 multi-block proxy agent team.

    Wraps the W48 :class:`SharedStateProxyTeam` contract with the
    W49 multi-block stack + multi-bank pseudo-KV + retention head
    + dictionary codebook + shared-latent capsule per turn. With
    a trivial registry, reduces to ``SharedStateProxyTeam.run``
    byte-for-byte (the W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH
    falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: MultiBlockProxyRegistry,
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
            target_fact_hash_of_turn: Callable[
                [int], Sequence[float]] | None = None,
    ) -> None:
        if not agents:
            raise ValueError(
                "MultiBlockProxyTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = MultiBlockProxyOrchestrator(registry)
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
        self.target_fact_hash_of_turn = (
            target_fact_hash_of_turn or (lambda t: ()))

    @property
    def schema_cid(self) -> str:
        return self.orchestrator.schema_cid

    def _resolve_backend(self, member: Agent) -> LLMBackend:
        backend = member.backend or self.backend
        if backend is None:
            raise ValueError(
                "no backend configured; pass backend=... to "
                "MultiBlockProxyTeam")
        if not isinstance(backend, LLMBackend):
            raise TypeError(
                "backend must satisfy the LLMBackend protocol")
        return backend

    def _build_prompt(
            self, *, member: Agent, task: str, turn_index: int,
            recent_handoffs: Sequence[tuple[str, str]],
            decision: MultiBlockProxyGatingDecision,
            shared_latent_short: str,
    ) -> tuple[
            str, str, int, int, int, int, int,
            LatentControlV2Witness, CrammingWitness]:
        """Build the bounded prompt + textual shadow."""
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
        n_textual = len(textual_prompt.split())

        bounded_parts = list(common_parts)
        n_shared_latent_header_tokens = 0
        n_latent_ctrl_v2_tokens = 0
        latent_witness = LatentControlV2Witness(
            ctrl_tag=str(self.registry.latent_control_tag),
            dictionary_code=int(
                decision.forward.dictionary_code),
            code_bits=int(
                decision.forward.dictionary_code_bits),
            n_mask_bits=0,
            emit_mask=tuple(),
            bits_payload=tuple(),
            shared_latent_hash_short=str(shared_latent_short),
            n_ctrl_tokens=0,
            ctrl_bytes_sha256=hashlib.sha256(b"").hexdigest(),
        )
        # SHARED_LATENT_HASH header.
        if (self.registry.shared_latent_capsule_enabled
                and self.registry.shared_latent_tag):
            sl_line = (
                f"{self.registry.shared_latent_tag}: "
                f"{shared_latent_short}")
            bounded_parts.append(sl_line)
            n_shared_latent_header_tokens = len(sl_line.split())

        # LATENT_CTRL_V2 line. The W49 control packs the
        # dictionary code AND a 16-bit emit mask + 16-bit payload
        # derived from the sign of the first 16 multi-block-output
        # coordinates — strictly more structured bits per visible
        # token than the W48 LATENT_CTRL block.
        if (self.registry.multi_block_enabled
                and self.registry.dictionary_enabled):
            mb_out = (
                decision.forward.multi_block_output[:16]
                if decision.forward.multi_block_output
                else tuple([0.0] * 16))
            # Pad to 16 if shorter.
            mb_out = tuple(mb_out) + tuple(
                [0.0] * max(0, 16 - len(mb_out)))
            mask = tuple(v >= 0.0 for v in mb_out)
            bits = tuple(
                int(1) if v >= 0.0 else int(0)
                for v in mb_out)
            ctrl_line, latent_witness = (
                build_latent_control_v2_string(
                    ctrl_tag=str(
                        self.registry.latent_control_tag),
                    dictionary_code=int(
                        decision.forward.dictionary_code),
                    code_bits=int(
                        decision.forward.dictionary_code_bits),
                    emit_mask=mask,
                    bits_payload=bits,
                    shared_latent_hash_short=str(
                        shared_latent_short),
                ))
            bounded_parts.append(ctrl_line)
            n_latent_ctrl_v2_tokens = int(
                latent_witness.n_ctrl_tokens)

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
        # Cramming witness.
        cramming = build_cramming_witness(
            dictionary_code=int(
                decision.forward.dictionary_code),
            code_bits=int(
                decision.forward.dictionary_code_bits),
            emit_mask=latent_witness.emit_mask,
            bits_payload=latent_witness.bits_payload,
            visible_ctrl_tokens=int(n_latent_ctrl_v2_tokens),
            visible_latent_header_tokens=int(
                n_shared_latent_header_tokens),
            shared_latent_capsule_bytes=len(_canonical_bytes(
                decision.shared_latent_capsule.to_dict())),
        )
        return (
            bounded_prompt, textual_prompt,
            int(n_textual), int(n_actual),
            int(n_latent_ctrl_v2_tokens),
            int(n_shared_latent_header_tokens),
            int(n_actual + n_latent_ctrl_v2_tokens
                + n_shared_latent_header_tokens),
            latent_witness, cramming,
        )

    def run(
            self, task: str,
            *,
            progress: Callable[
                [MultiBlockProxyTurn], None] | None = None,
    ) -> MultiBlockProxyTeamResult:
        """Run the W49 multi-block-proxy team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        multi_block_turns: list[MultiBlockProxyTurn] = []
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
        n_abstain_substitutions = 0
        n_dict_emitted = 0
        n_role_bank_writes = 0
        n_shared_bank_writes = 0
        n_visible_tokens_added_latent = 0
        n_visible_tokens_added_sl = 0
        retention_probs: list[float] = []
        router_values: list[float] = []
        mix_values: list[float] = []
        bits_per_token: list[float] = []
        head_backend = self.backend
        head_model = getattr(head_backend, "model", "") or ""
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()
        multi_block_params_cid = self.registry.params.cid()
        training_trace_cid = (
            self.registry.params.training_trace_cid)
        shared_latent_chain: list[str] = []
        prior_role_bank_total = 0
        prior_shared_bank_total = 0

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            branch_id = int(self.branch_id_of_turn(int(idx)))
            cycle_id = int(self.cycle_id_of_turn(int(idx)))
            target_fact = tuple(
                self.target_fact_hash_of_turn(int(idx)))
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
                target_fact_hash=target_fact,
                expected_spherical=self.expected_spherical,
                expected_subspace=self.expected_subspace,
            )
            shared_latent_chain.append(
                decision.shared_latent_capsule.cid())

            # Bank size change accounting.
            curr_role_total = sum(
                b.size for b in
                self.orchestrator.multi_bank.role_banks.values())
            curr_shared_total = (
                self.orchestrator.multi_bank.shared_bank.size
                if self.orchestrator.multi_bank.shared_bank
                else 0)
            if curr_role_total > prior_role_bank_total:
                n_role_bank_writes += 1
            if curr_shared_total > prior_shared_bank_total:
                n_shared_bank_writes += 1
            prior_role_bank_total = curr_role_total
            prior_shared_bank_total = curr_shared_total

            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            shared_latent_short = (
                decision.shared_latent_capsule.hash_short())

            (bounded_prompt, textual_prompt,
             n_textual_tokens, n_actual_tokens,
             n_latent_ctrl_v2_tokens,
             n_shared_latent_header_tokens,
             n_overhead_tokens,
             latent_ctrl_v2_witness, cramming) = (
                self._build_prompt(
                    member=member, task=task,
                    turn_index=idx,
                    recent_handoffs=recent_handoffs,
                    decision=decision,
                    shared_latent_short=str(shared_latent_short),
                ))

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

            n_dict_emitted += (
                1 if (n_latent_ctrl_v2_tokens > 0
                      and not do_substitute) else 0)
            if not do_substitute:
                if n_latent_ctrl_v2_tokens > 0:
                    n_visible_tokens_added_latent += int(
                        n_latent_ctrl_v2_tokens)
                if n_shared_latent_header_tokens > 0:
                    n_visible_tokens_added_sl += int(
                        n_shared_latent_header_tokens)
            retention_probs.append(
                float(decision.forward.retention_prob))
            router_values.append(
                float(decision.forward.bank_router_value))
            mix_values.append(
                float(decision.forward.bank_mix_gate_value))
            bits_per_token.append(
                float(cramming.bits_per_visible_token))

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

            # Build the W49 envelope witnesses.
            forward_witness_cid = (
                _compute_w49_forward_witness_cid(
                    gate_logit=float(decision.forward.gate_logit),
                    ratify_probability=float(
                        decision.forward.ratify_probability),
                    bank_router_value=float(
                        decision.forward.bank_router_value),
                    bank_mix_gate_value=float(
                        decision.forward.bank_mix_gate_value),
                    retention_prob=float(
                        decision.forward.retention_prob),
                    dictionary_code=int(
                        decision.forward.dictionary_code),
                    dictionary_code_bits=int(
                        decision.forward.dictionary_code_bits),
                    shared_latent_capsule_cid=(
                        decision.shared_latent_capsule.cid()),
                    turn_index=int(idx),
                    role=str(role),
                ))
            inner_w48_proxy_witness_cid = ""  # We seal the inner
            #   W48 proxy state via its envelope chain elsewhere;
            #   we record an empty placeholder + bind only the
            #   outer witness here.
            behavioral_change = bool(
                do_substitute
                or n_latent_ctrl_v2_tokens > 0
                or n_shared_latent_header_tokens > 0)
            multi_block_witness_cid = _compute_w49_witness_cid(
                decision_branch=decision.branch,
                w48_branch=decision.w48_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                multi_block_params_cid=multi_block_params_cid,
                training_trace_cid=training_trace_cid,
                multi_block_forward_witness_cid=(
                    forward_witness_cid),
                shared_latent_capsule_cid=(
                    decision.shared_latent_capsule.cid()),
                multi_bank_head_cid=(
                    decision.multi_bank_head_cid),
                latent_ctrl_v2_witness_cid=(
                    latent_ctrl_v2_witness.cid()),
                cramming_witness_cid=cramming.cid(),
                inner_w48_proxy_witness_cid=str(
                    inner_w48_proxy_witness_cid),
                output_sha256=output_sha,
                behavioral_change=behavioral_change,
                multi_block_stack_cid=str(
                    self.registry.params.multi_block_stack.cid()),
                bank_router_cid=str(
                    self.registry.params.bank_router.cid()),
                bank_mix_gate_cid=str(
                    self.registry.params.bank_mix_gate.cid()),
                eviction_policy_cid=str(
                    self.registry.params.eviction_policy.cid()),
                retention_head_cid=str(
                    self.registry.params.retention_head.cid()),
                dictionary_cid=str(
                    self.registry.params.dictionary.cid()),
                shared_latent_projector_cid=str(
                    self.registry.params.shared_latent_projector
                    .cid()),
            )
            outer_cid = _compute_w49_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w48_envelope_cid="",
                multi_block_params_cid=multi_block_params_cid,
                training_trace_cid=training_trace_cid,
                multi_block_witness_cid=multi_block_witness_cid,
                turn_index=int(idx),
            )
            envelope = MultiBlockProxyHandoffEnvelope(
                schema_version=(
                    W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION),
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w48_envelope_cid="",
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
                w48_branch=decision.w48_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                branch_id=int(branch_id),
                cycle_id=int(cycle_id),
                multi_block_params_cid=multi_block_params_cid,
                multi_block_stack_cid=str(
                    self.registry.params.multi_block_stack.cid()),
                bank_router_cid=str(
                    self.registry.params.bank_router.cid()),
                bank_mix_gate_cid=str(
                    self.registry.params.bank_mix_gate.cid()),
                eviction_policy_cid=str(
                    self.registry.params.eviction_policy.cid()),
                retention_head_cid=str(
                    self.registry.params.retention_head.cid()),
                dictionary_cid=str(
                    self.registry.params.dictionary.cid()),
                shared_latent_projector_cid=str(
                    self.registry.params.shared_latent_projector
                    .cid()),
                training_trace_cid=training_trace_cid,
                fitting_method=str(
                    self.registry.params.fitting_method),
                n_blocks=int(
                    self.registry.params.n_blocks),
                inner_w48_params_cid=str(
                    self.registry.params.inner_w48.cid()),
                inner_w48_proxy_witness_cid=str(
                    inner_w48_proxy_witness_cid),
                multi_bank_head_cid=str(
                    decision.multi_bank_head_cid),
                role_bank_head_cid=str(
                    decision.role_bank_head_cid),
                shared_bank_head_cid=str(
                    decision.shared_bank_head_cid),
                role_bank_size=int(
                    self.orchestrator.multi_bank
                    .role_banks.get(role, PseudoKVBank(
                        capacity=0, factor_dim=0)).size),
                shared_bank_size=int(
                    self.orchestrator.multi_bank.shared_bank.size
                    if self.orchestrator.multi_bank.shared_bank
                    else 0),
                gate_logit=float(decision.forward.gate_logit),
                ratify_probability=float(
                    decision.forward.ratify_probability),
                bank_router_value=float(
                    decision.forward.bank_router_value),
                bank_mix_gate_value=float(
                    decision.forward.bank_mix_gate_value),
                retention_prob=float(
                    decision.forward.retention_prob),
                dictionary_code=int(
                    decision.forward.dictionary_code),
                dictionary_code_bits=int(
                    decision.forward.dictionary_code_bits),
                multi_block_forward_witness_cid=str(
                    forward_witness_cid),
                shared_latent_capsule_cid=str(
                    decision.shared_latent_capsule.cid()),
                shared_latent_parent_cid=str(
                    decision.shared_latent_capsule
                    .parent_capsule_cid),
                shared_latent_dim=int(
                    decision.shared_latent_capsule.dim),
                latent_ctrl_v2_witness_cid=str(
                    latent_ctrl_v2_witness.cid()),
                cramming_witness_cid=str(cramming.cid()),
                prompt_sha256=prompt_sha,
                output_sha256=output_sha,
                n_visible_prompt_tokens_textual=int(
                    n_textual_tokens),
                n_visible_prompt_tokens_actual=int(
                    n_actual_tokens),
                n_visible_prompt_tokens_saved=int(
                    n_textual_tokens - n_actual_tokens),
                n_latent_ctrl_v2_tokens=int(
                    n_latent_ctrl_v2_tokens),
                n_shared_latent_header_tokens=int(
                    n_shared_latent_header_tokens),
                n_overhead_tokens=int(
                    n_latent_ctrl_v2_tokens
                    + n_shared_latent_header_tokens),
                behavioral_change=bool(behavioral_change),
                multi_block_witness_cid=str(
                    multi_block_witness_cid),
                multi_block_outer_cid=str(outer_cid),
            )
            multi_block_turn = MultiBlockProxyTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
                cramming_witness=cramming,
            )
            multi_block_turns.append(multi_block_turn)

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

            if progress is not None:
                try:
                    progress(multi_block_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[MultiBlockProxyTeam] progress callback "
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
        mean_r = (
            sum(retention_probs) / len(retention_probs)
            if retention_probs else 0.0)
        mean_router = (
            sum(router_values) / len(router_values)
            if router_values else 0.0)
        mean_mix = (
            sum(mix_values) / len(mix_values)
            if mix_values else 0.0)
        mean_bits = (
            sum(bits_per_token) / len(bits_per_token)
            if bits_per_token else 0.0)
        return MultiBlockProxyTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            multi_block_turns=tuple(multi_block_turns),
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
            n_abstain_substitutions=int(n_abstain_substitutions),
            n_role_bank_writes=int(n_role_bank_writes),
            n_shared_bank_writes=int(n_shared_bank_writes),
            n_dictionary_codes_emitted=int(n_dict_emitted),
            mean_retention_prob=float(mean_r),
            mean_bank_router_value=float(mean_router),
            mean_bank_mix_gate_value=float(mean_mix),
            mean_bits_per_visible_token=float(mean_bits),
            n_visible_tokens_added_latent_v2=int(
                n_visible_tokens_added_latent),
            n_visible_tokens_added_shared_latent=int(
                n_visible_tokens_added_sl),
            multi_block_params_cid=str(multi_block_params_cid),
            training_trace_cid=str(training_trace_cid),
            shared_latent_chain_cids=tuple(shared_latent_chain),
            final_multi_bank_head_cid=str(
                self.orchestrator.multi_bank.head_cid()),
        )


# =============================================================================
# Multi-block-aware synthetic backend
# =============================================================================

@dataclasses.dataclass
class MultiBlockAwareSyntheticBackend:
    """Deterministic backend that answers differently when the
    prompt carries a ``LATENT_CTRL_V2:`` token AND a
    ``SHARED_LATENT_HASH:`` header.

    Used by R-96 / R-97 to exercise the *behavioural* effect of
    the W49 multi-block + shared-latent control surface on a
    controlled synthetic ground truth.
    """

    correct_with_v2: str = "MULTI_BLOCK_OK"
    correct_with_one: str = "MULTI_BLOCK_PARTIAL"
    answer_without: str = "MULTI_BLOCK_NO"
    n_calls: int = 0
    model_tag: str = "synthetic.multi_block_aware"
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
        has_v2 = "LATENT_CTRL_V2:" in text
        has_sl = "SHARED_LATENT_HASH:" in text
        if has_v2 and has_sl:
            return self.correct_with_v2
        if has_v2 or has_sl:
            return self.correct_with_one
        return self.answer_without


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Schema / branches / defaults
    "W49_MULTI_BLOCK_PROXY_SCHEMA_VERSION",
    "W49_TEAM_RESULT_SCHEMA",
    "W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH",
    "W49_BRANCH_MULTI_BLOCK_DISABLED",
    "W49_BRANCH_MULTI_BLOCK_RATIFIED",
    "W49_BRANCH_MULTI_BLOCK_NO_POLICY",
    "W49_BRANCH_MULTI_BLOCK_MARGIN_ABSTAIN",
    "W49_BRANCH_MULTI_BLOCK_RETENTION_ABSTAIN",
    "W49_BRANCH_MULTI_BLOCK_DICTIONARY_ABSTAIN",
    "W49_BRANCH_MULTI_BLOCK_TRAIN_FAILURE",
    "W49_BRANCH_MULTI_BLOCK_REJECTED",
    "W49_ALL_BRANCHES",
    "W49_MULTI_BLOCK_ABSTAIN_BRANCHES",
    "W49_ALL_FAILURE_MODES",
    "W49_DEFAULT_N_BLOCKS",
    "W49_DEFAULT_FFN_HIDDEN_DIM",
    "W49_DEFAULT_ROLE_BANK_CAPACITY",
    "W49_DEFAULT_SHARED_BANK_CAPACITY",
    "W49_DEFAULT_RETENTION_HIDDEN_DIM",
    "W49_DEFAULT_DICTIONARY_SIZE",
    "W49_DEFAULT_DICTIONARY_CODE_BITS",
    "W49_DEFAULT_LATENT_CTRL_TAG",
    "W49_DEFAULT_SHARED_LATENT_TAG",
    "W49_DEFAULT_BANK_ROUTER_THRESHOLD",
    "W49_DEFAULT_EVICTION_FLOOR",
    # Components
    "FeedForwardBlock",
    "ProxyTransformerBlock",
    "MultiBlockProxyStack",
    "MultiBankPseudoKV",
    "BankRouter",
    "BankMixGate",
    "EvictionPolicy",
    "RetentionHead",
    "DictionaryCodebook",
    "LatentControlV2Witness",
    "build_latent_control_v2_string",
    "SharedLatentCapsule",
    "SharedLatentProjector",
    "CrammingWitness",
    "build_cramming_witness",
    # Params + fitter
    "MultiBlockProxyParams",
    "MultiBlockTrainingTraceWitness",
    "build_unfitted_multi_block_proxy_params",
    "MultiBlockExample",
    "MultiBlockTrainingSet",
    "fit_multi_block_proxy",
    # Forward
    "MultiBlockProxyForwardResult",
    "forward_multi_block_proxy",
    # Registry + orchestrator + envelope + verifier + team
    "MultiBlockProxyRegistry",
    "MultiBlockProxyOrchestrator",
    "MultiBlockProxyGatingDecision",
    "build_trivial_multi_block_proxy_registry",
    "build_multi_block_proxy_registry",
    "MultiBlockProxyHandoffEnvelope",
    "MultiBlockProxyVerificationOutcome",
    "verify_multi_block_proxy_handoff",
    "MultiBlockProxyTurn",
    "MultiBlockProxyTeamResult",
    "MultiBlockProxyTeam",
    # Synthetic backend
    "MultiBlockAwareSyntheticBackend",
]
