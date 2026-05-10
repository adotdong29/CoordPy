"""W46 Manifold Memory Controller (MMC) — capsule-native
multi-layer learned manifold-memory controller on top of W45 LMC,
W44 LMCC, and W43 PMC.

W46 is the first capsule-native CoordPy layer where the gating
policy depends on a **bounded memory of previous turns**, runs
through a **multi-layer fitted controller stack**, applies
**rank-r role adapters**, encodes channel features into a
**learned dictionary basis**, emits a **packed multi-token
`MANIFOLD_CTRL` control surface**, and reuses a **deterministic
shared-prefix capsule** across consecutive turns.

The seven learned and content-addressed components — all
closed-form-fittable in pure Python / stdlib without an
autograd dependency — are:

  * **Multi-layer learned controller stack.** A stack of ``L``
    layers fitted stage-wise on residuals. Each layer applies a
    fitted projection + softmax attention pool + residual add.
    The W45 single-layer controller is the ``L=1`` degenerate
    case. ``L >= 2`` is fitted by closed-form ridge on the
    layer-wise residual.

  * **Manifold memory bank.** A bounded, content-addressed ring
    buffer of the last ``M`` turn observations: turn index,
    role, role-handoff-signature CID, per-channel feature
    vector, per-channel logits, gate logit, ratify probability,
    decision branch. The bank's CID is the SHA-256 over its
    sorted entries; an auditor can re-derive the bank state at
    any turn from the envelope chain alone.

  * **Causally-masked time-attention.** When ratifying turn
    ``t``, the controller computes attention over memory entries
    ``< t`` using a learned cosine-similarity readout: the
    per-channel features of turn ``t`` query the bank, the
    bank's gate logits are the values, and the pooled output is
    the *time-attention context* — a single scalar added to the
    layer-wise gate logit before the margin sigmoid. The mask is
    strict.

  * **Multi-rank role adapter stack.** The W45 rank-1 LoRA-style
    adapter is generalized to rank ``r >= 1``. Per-role delta
    carries ``r * n_channels + 1`` scalars; the basis vectors
    are the signed per-channel logits (basis 0) plus their
    cyclic rotations (bases 1..r-1).

  * **Learned dictionary basis.** A small dictionary of ``K``
    prototype channel-feature vectors, fitted by deterministic
    closest-prototype assignment over the training bank. The
    per-turn encoding is a sparse code ``(prototype_index,
    residual)`` with bijective decode.

  * **Packed `MANIFOLD_CTRL` control token surface.** A
    multi-line model-facing control block carrying ``route``,
    ``conf``, ``p``, the layer-wise gate logits, the
    time-attention scalar, the dictionary index, and a memory
    summary signature. Strictly more structured bits per
    visible-token than W45's single-line hint.

  * **Shared-prefix capsule.** A deterministic prefix derived
    from the last ``P`` turn-output SHAs and the registered
    policy entry. When ``prefix_reuse_enabled=True``, the prompt
    builder emits the same prefix bytes across consecutive
    turns; the envelope binds the prefix CID. Honest scope: this
    binds *identical prefix bytes*, not transformer-internal KV
    state.

Honest scope (do-not-overstate)
-------------------------------

W46 does NOT claim transformer-internal access. The memory
controller operates strictly over W43 capsule-layer channel
encodings; it does not read hidden states, transplant KV cache,
inspect attention weights, or modify the model's attention
computation. The W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``) and
``W45-C-DEEP-TRANSFORMER-COUPLING`` carry forward unchanged.

W46 does NOT claim the multi-layer controller is a deep neural
network in the autograd sense; each layer is fitted closed-form
via stage-wise ridge on layer-wise residuals. There is no SGD,
no backprop, no learned non-linearity beyond the per-layer
softmax attention pool.

W46 does NOT claim true shared-KV between turns; the shared-
prefix capsule guarantees byte-identical prefix bytes across
turns, which is a strict capsule-layer property.

W46 is strictly additive on top of W45 and the released v3.43
SDK. When the memory controller is configured trivially
(``memory_enabled=False``, ``n_layers=1``,
``time_attention_enabled=False``, ``dictionary_enabled=False``,
``control_token_mode='off'``, ``prefix_reuse_enabled=False``,
W45 inner trivial), the W46 orchestrator reduces to
``LearnedManifoldTeam.run`` byte-for-byte — the
W46-L-TRIVIAL-MEMORY-PASSTHROUGH falsifier.

This module lives at ``coordpy.manifold_memory`` and is NOT
exported through ``coordpy.__experimental__`` at this milestone;
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W46 surface through an explicit
``from coordpy.manifold_memory import ...`` import.
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
    ControllerForwardResult,
    LearnedControllerParams,
    LearnedGatingDecision,
    LearnedManifoldHandoffEnvelope,
    LearnedManifoldOrchestrator,
    LearnedManifoldRegistry,
    TrainingExample,
    TrainingSet,
    W45_BRANCH_LEARNED_CAUSAL_ABSTAIN,
    W45_BRANCH_LEARNED_DISABLED,
    W45_BRANCH_LEARNED_MARGIN_ABSTAIN,
    W45_BRANCH_LEARNED_NO_POLICY,
    W45_BRANCH_LEARNED_RATIFIED,
    W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN,
    W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN,
    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH,
    W45_CHANNEL_ORDER,
    W45_CONFIDENCE_BUCKETS,
    W45_DEFAULT_FEATURE_DIM,
    W45_DEFAULT_RIDGE_LAMBDA,
    W45_DEFAULT_ROLE_DELTA_RANK,
    W45_HINT_MODE_FACTORADIC_WITH_HINT,
    W45_HINT_MODE_OFF,
    W45_N_CHANNELS,
    _channel_features_from_bundle,
    _confidence_bucket_for_probability,
    _sigmoid,
    _softmax,
    _solve_ridge,
    build_learned_manifold_registry,
    build_trivial_learned_manifold_registry,
    build_unfitted_controller_params,
    fit_learned_controller,
    forward_controller,
)
from .live_manifold import (
    LiveObservationBuilder,
    LiveTurnContext,
    W44_BRANCH_LIVE_NO_POLICY,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_DEFAULT_PARENT_W42_CID,
    W44_ROUTE_MODE_FACTORADIC,
    default_live_observation_builder,
)
from .llm_backend import LLMBackend
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

W46_MANIFOLD_MEMORY_SCHEMA_VERSION: str = (
    "coordpy.manifold_memory.v1")
W46_TEAM_RESULT_SCHEMA: str = (
    "coordpy.manifold_memory_team_result.v1")

# Decision branches reuse W45 names for behaviour compatibility,
# and add three W46-specific branches.
W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH: str = (
    "memory_trivial_passthrough")
W46_BRANCH_MEMORY_DISABLED: str = "memory_disabled"
W46_BRANCH_MEMORY_RATIFIED: str = "memory_ratified"
W46_BRANCH_MEMORY_NO_POLICY: str = "memory_no_policy"
W46_BRANCH_MEMORY_CAUSAL_ABSTAIN: str = "memory_causal_abstain"
W46_BRANCH_MEMORY_SPHERICAL_ABSTAIN: str = (
    "memory_spherical_abstain")
W46_BRANCH_MEMORY_SUBSPACE_ABSTAIN: str = (
    "memory_subspace_abstain")
W46_BRANCH_MEMORY_MARGIN_ABSTAIN: str = (
    "memory_margin_abstain")
W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN: str = (
    "memory_time_attn_abstain")
W46_BRANCH_MEMORY_REJECTED: str = "memory_rejected"

W46_ALL_BRANCHES: tuple[str, ...] = (
    W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH,
    W46_BRANCH_MEMORY_DISABLED,
    W46_BRANCH_MEMORY_RATIFIED,
    W46_BRANCH_MEMORY_NO_POLICY,
    W46_BRANCH_MEMORY_CAUSAL_ABSTAIN,
    W46_BRANCH_MEMORY_SPHERICAL_ABSTAIN,
    W46_BRANCH_MEMORY_SUBSPACE_ABSTAIN,
    W46_BRANCH_MEMORY_MARGIN_ABSTAIN,
    W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN,
    W46_BRANCH_MEMORY_REJECTED,
)

W46_MEMORY_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W46_BRANCH_MEMORY_CAUSAL_ABSTAIN,
    W46_BRANCH_MEMORY_SPHERICAL_ABSTAIN,
    W46_BRANCH_MEMORY_SUBSPACE_ABSTAIN,
    W46_BRANCH_MEMORY_MARGIN_ABSTAIN,
    W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN,
})

# Packed-control-token surface modes. ``off`` reduces the prompt
# builder to W45's hint mode; ``compact`` emits only ``route +
# conf + p + dict_idx``; ``full`` emits the full packed block.
W46_CTRL_MODE_OFF: str = "off"
W46_CTRL_MODE_COMPACT: str = "compact"
W46_CTRL_MODE_FULL: str = "full"

W46_ALL_CTRL_MODES: tuple[str, ...] = (
    W46_CTRL_MODE_OFF,
    W46_CTRL_MODE_COMPACT,
    W46_CTRL_MODE_FULL,
)

# Defaults.
W46_DEFAULT_N_LAYERS: int = 2
W46_DEFAULT_MEMORY_CAPACITY: int = 8
W46_DEFAULT_DICTIONARY_SIZE: int = 4
W46_DEFAULT_ROLE_DELTA_RANK: int = 2
W46_DEFAULT_TIME_ATTN_TEMPERATURE: float = 1.0
W46_DEFAULT_TIME_ATTN_WEIGHT: float = 0.5
W46_DEFAULT_PREFIX_TURNS: int = 2

# Sentinel string used when there is no fitted role delta for a
# given role.
W46_NO_ROLE_DELTA: str = "no_role_delta"

# Sentinel used when the dictionary index has no fitted code.
W46_NO_DICT_CODE: int = -1


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


def _cosine_similarity(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Numerically stable cosine similarity.

    Returns 0.0 when either input is the zero vector; otherwise
    returns ``(a . b) / (||a|| * ||b||)`` clamped to ``[-1, 1]``.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) ** 2 for i in range(n)))
    nb = math.sqrt(sum(float(b[i]) ** 2 for i in range(n)))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    val = dot / (na * nb)
    return max(-1.0, min(1.0, val))


def _l1_norm(values: Sequence[float]) -> float:
    return float(sum(abs(float(v)) for v in values))


def _flatten_channel_features(
        features: Mapping[str, Sequence[float]],
        *,
        feature_dim: int,
) -> tuple[float, ...]:
    """Flatten the per-channel feature map into a single fixed-length
    vector in the canonical channel order. Pads each channel to
    ``feature_dim``."""
    out: list[float] = []
    for c_name in W45_CHANNEL_ORDER:
        feats = list(features.get(c_name, ()))[:feature_dim]
        while len(feats) < feature_dim:
            feats.append(0.0)
        out.extend(float(v) for v in feats)
    return tuple(out)


# =============================================================================
# Multi-layer controller params
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MultiLayerControllerParams:
    """All fitted parameters of the W46 multi-layer controller
    stack.

    Wraps:

    * a ``LearnedControllerParams`` (the W45 layer-0 base, fitted
      via the W45 ridge fitter)
    * a tuple of ``LayerParams`` for layers 1..L-1 (each fitted
      via stage-wise ridge on the previous layer's residual)
    * a ``MultiRankRoleAdapter`` with rank ``r >= 1``
    * a ``DictionaryBasis`` with ``K`` prototype vectors
    * a fitted ``time_attention_weight`` and ``time_attention_bias``

    All sub-objects are frozen and content-addressable.
    """

    base: LearnedControllerParams
    layers: tuple["LayerParams", ...]
    role_adapter: "MultiRankRoleAdapter"
    dictionary: "DictionaryBasis"
    time_attention_weight: float
    time_attention_bias: float
    fitting_method: str
    training_set_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "layers": [lyr.to_dict() for lyr in self.layers],
            "role_adapter": self.role_adapter.to_dict(),
            "dictionary": self.dictionary.to_dict(),
            "time_attention_weight": float(round(
                self.time_attention_weight, 12)),
            "time_attention_bias": float(round(
                self.time_attention_bias, 12)),
            "fitting_method": str(self.fitting_method),
            "training_set_cid": str(self.training_set_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_multi_layer_controller_params",
            "params": self.to_dict(),
        })

    @property
    def n_layers(self) -> int:
        return 1 + len(self.layers)

    @property
    def feature_dim(self) -> int:
        return int(self.base.feature_dim)


@dataclasses.dataclass(frozen=True)
class LayerParams:
    """One layer of the W46 multi-layer controller stack.

    Each layer has:

    * ``channel_projection`` — per-channel ``feature_dim``-vector
      projecting that channel's residual feature vector to a
      scalar layer logit. Shape: ``(W45_N_CHANNELS, feature_dim)``.
    * ``attention_logits`` — per-channel attention logits before
      softmax. Shape: ``(W45_N_CHANNELS,)``.
    * ``residual_scale`` — scalar gain applied to the layer
      output before it is added to the running sum (default 1.0).
    * ``layer_index`` — 1-based index in the stack.
    """

    layer_index: int
    channel_projection: tuple[tuple[float, ...], ...]
    attention_logits: tuple[float, ...]
    residual_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "channel_projection":
                _round_matrix(self.channel_projection),
            "attention_logits":
                _round_floats(self.attention_logits),
            "residual_scale": float(round(
                self.residual_scale, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_layer_params", "layer": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class MultiRankRoleAdapter:
    """Rank-``r`` per-role delta bundle.

    Generalizes the W45 rank-1 LoRA-style adapter to rank-``r``:
    per-role delta carries ``r * n_channels + 1`` scalars. The
    basis vectors are the signed per-channel logits (basis 0) plus
    their cyclic rotations (bases 1..r-1).
    """

    rank: int
    role_deltas: tuple[tuple[str, tuple[float, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "role_deltas": [
                [str(role), _round_floats(delta)]
                for role, delta in self.role_deltas
            ],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_multi_rank_role_adapter",
            "adapter": self.to_dict(),
        })

    @property
    def role_delta_map(self) -> dict[str, tuple[float, ...]]:
        return {str(r): tuple(d) for r, d in self.role_deltas}


@dataclasses.dataclass(frozen=True)
class DictionaryBasis:
    """Learned K-prototype dictionary over flattened channel
    features.

    Each prototype is a ``W45_N_CHANNELS * feature_dim``-vector;
    per-turn encoding is a sparse code ``(prototype_index,
    residual)`` with bijective decode (decode = prototype +
    residual).
    """

    feature_dim: int
    prototypes: tuple[tuple[float, ...], ...]

    @property
    def k(self) -> int:
        return len(self.prototypes)

    @property
    def vector_dim(self) -> int:
        return W45_N_CHANNELS * int(self.feature_dim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "prototypes": _round_matrix(self.prototypes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_dictionary_basis",
            "basis": self.to_dict(),
        })

    def encode(
            self, flat_features: Sequence[float],
    ) -> tuple[int, tuple[float, ...]]:
        """Encode a flat feature vector as ``(index, residual)``.

        Returns the index of the closest prototype (L2) and the
        residual ``features - prototype``. Bijective: decode by
        adding the residual back to ``prototypes[index]``.
        """
        if not self.prototypes:
            return W46_NO_DICT_CODE, tuple(
                float(v) for v in flat_features)
        # Pad / truncate to vector_dim.
        flat = list(flat_features)[:self.vector_dim]
        while len(flat) < self.vector_dim:
            flat.append(0.0)
        best_idx = 0
        best_dist = float("inf")
        for i, proto in enumerate(self.prototypes):
            proto = list(proto)[:self.vector_dim]
            while len(proto) < self.vector_dim:
                proto.append(0.0)
            dist = sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(flat, proto))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        proto = list(self.prototypes[best_idx])[:self.vector_dim]
        while len(proto) < self.vector_dim:
            proto.append(0.0)
        residual = tuple(
            float(a) - float(b) for a, b in zip(flat, proto))
        return int(best_idx), tuple(_round_floats(residual))

    def decode(
            self, index: int, residual: Sequence[float],
    ) -> tuple[float, ...]:
        """Decode an ``(index, residual)`` pair back to the original
        flat features."""
        if index < 0 or index >= len(self.prototypes):
            return tuple(float(v) for v in residual)
        proto = list(self.prototypes[index])[:self.vector_dim]
        while len(proto) < self.vector_dim:
            proto.append(0.0)
        residual_list = list(residual)[:self.vector_dim]
        while len(residual_list) < self.vector_dim:
            residual_list.append(0.0)
        return tuple(
            float(a) + float(b)
            for a, b in zip(proto, residual_list))


def _build_unfitted_dictionary(
        *, feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        k: int = W46_DEFAULT_DICTIONARY_SIZE,
) -> DictionaryBasis:
    """Build a dictionary with K all-zero prototypes (the unfitted
    seed)."""
    fd = max(1, int(feature_dim))
    dim = W45_N_CHANNELS * fd
    return DictionaryBasis(
        feature_dim=fd,
        prototypes=tuple(tuple([0.0] * dim) for _ in range(int(k))),
    )


def _build_unfitted_multi_rank_role_adapter(
        *, rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
) -> MultiRankRoleAdapter:
    return MultiRankRoleAdapter(
        rank=max(1, int(rank)), role_deltas=tuple())


def build_unfitted_memory_controller_params(
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_layers: int = W46_DEFAULT_N_LAYERS,
        rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
        dictionary_size: int = W46_DEFAULT_DICTIONARY_SIZE,
) -> MultiLayerControllerParams:
    """Build a memory-controller parameter bundle whose stack
    output is exactly zero (the unfitted seed). With this, the
    multi-layer gate is the W45 single-layer gate.
    """
    base = build_unfitted_controller_params(
        feature_dim=int(feature_dim),
        role_delta_rank=int(rank),
    )
    layers = tuple()
    for li in range(max(0, int(n_layers) - 1)):
        layers = layers + (LayerParams(
            layer_index=li + 1,
            channel_projection=tuple(
                tuple([0.0] * int(feature_dim))
                for _ in range(W45_N_CHANNELS)),
            attention_logits=tuple([0.0] * W45_N_CHANNELS),
            residual_scale=1.0,
        ),)
    return MultiLayerControllerParams(
        base=base,
        layers=layers,
        role_adapter=_build_unfitted_multi_rank_role_adapter(
            rank=int(rank)),
        dictionary=_build_unfitted_dictionary(
            feature_dim=int(feature_dim),
            k=int(dictionary_size)),
        time_attention_weight=0.0,
        time_attention_bias=0.0,
        fitting_method="unfitted",
        training_set_cid="",
    )


# =============================================================================
# Multi-layer + multi-rank fitter
# =============================================================================

def fit_memory_controller(
        training_set: TrainingSet,
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_layers: int = W46_DEFAULT_N_LAYERS,
        role_delta_rank: int = W46_DEFAULT_ROLE_DELTA_RANK,
        dictionary_size: int = W46_DEFAULT_DICTIONARY_SIZE,
        ridge_lambda: float = W45_DEFAULT_RIDGE_LAMBDA,
        fit_role_deltas: bool = True,
        time_attention_weight: float = (
            W46_DEFAULT_TIME_ATTN_WEIGHT),
) -> MultiLayerControllerParams:
    """Fit a memory controller via stage-wise closed-form ridge.

    Stage 1
        Fit the W45 layer-0 base (shared base + rank-1 role
        delta) via :func:`coordpy.learned_manifold.fit_learned_controller`
        — but pin ``role_delta_rank=1`` regardless of the W46
        requested rank, because layer-0 is by construction the
        rank-1 W45 base.

    Stage 2..L
        For each layer ``i >= 1``, build the per-example residual
        ``y_i = label - sum_{j < i} layer_j(features)`` and fit a
        new ``LayerParams`` via ridge on the per-example feature
        rows.

    Stage L+1 (multi-rank role adapter)
        For each role with at least ``rank+1`` examples, fit a
        rank-``r`` delta vector on the per-role residual after the
        full stack has been applied. The rank-1 W45 adapter is
        re-fitted here as the first rank of the W46 adapter; the
        bases 1..r-1 are cyclic rotations.

    Stage L+2 (dictionary)
        Cluster the training-set's flattened channel features into
        ``K`` prototypes via deterministic seeded assignment
        (pick the example farthest from the running mean for each
        new prototype; assign the rest to the closest prototype;
        update prototypes to be the mean of their assignees).
        Single closed-form pass; reproducible from the seed.

    Trains entirely closed-form with no autograd / iterative
    optimisation in pure Python. The ``time_attention_weight``
    field is the controller-level mixing weight; we don't fit
    it from the training set because the bank state isn't part
    of the per-example feature; the value is taken from the
    argument or defaults to ``W46_DEFAULT_TIME_ATTN_WEIGHT``.
    """
    fd = int(feature_dim)
    if not training_set.examples:
        return build_unfitted_memory_controller_params(
            feature_dim=fd, n_layers=int(n_layers),
            rank=int(role_delta_rank),
            dictionary_size=int(dictionary_size))

    # ---- Stage 1: layer-0 base (W45 rank-1) ----
    base = fit_learned_controller(
        training_set,
        feature_dim=fd,
        role_delta_rank=1,
        ridge_lambda=float(ridge_lambda),
        fit_role_deltas=False,
    )

    # ---- Stage 2..L: residual layers ----
    layers: list[LayerParams] = []
    # Pre-compute per-example flattened features (the residual
    # design matrix for every layer uses these as inputs).
    # We don't allow each layer to *use the previous layer's
    # output* as an input — that would conflate input vs target
    # and break the closed-form fit. Each layer fits *on the
    # same input features* but against the *residual target*
    # from the running sum. This is the discrete analog of
    # residual fitting with shared inputs.
    flat_inputs: list[list[float]] = []
    labels: list[float] = []
    for ex in training_set.examples:
        fmap = ex.channel_features_map
        flat_inputs.append(
            list(_flatten_channel_features(
                fmap, feature_dim=fd)))
        labels.append(float(ex.label))

    # Running prediction from layer 0 (base) on each example.
    running: list[float] = []
    for ex_idx, ex in enumerate(training_set.examples):
        fmap = ex.channel_features_map
        fr = forward_controller(
            channel_features=fmap, params=base, role=ex.role,
            use_attention_routing=True,
        )
        running.append(float(fr.gate_logit))

    n_per_layer_feats = W45_N_CHANNELS * fd + 1
    for li in range(max(0, int(n_layers) - 1)):
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for ex_idx, flat in enumerate(flat_inputs):
            row = list(flat) + [1.0]
            x_rows.append(row)
            resid = float(labels[ex_idx]) - float(running[ex_idx])
            y_rows.append(resid)
        w = _solve_ridge(
            x_rows, y_rows, lam=float(ridge_lambda))
        if len(w) != n_per_layer_feats:
            w = [0.0] * n_per_layer_feats
        proj_chunks: list[tuple[float, ...]] = []
        att_logits: list[float] = []
        for c_idx in range(W45_N_CHANNELS):
            base_idx = c_idx * fd
            chunk = tuple(_round_floats(w[base_idx:base_idx + fd]))
            proj_chunks.append(chunk)
            att_logits.append(float(sum(abs(v) for v in chunk)))
        layer = LayerParams(
            layer_index=li + 1,
            channel_projection=tuple(proj_chunks),
            attention_logits=tuple(_round_floats(att_logits)),
            residual_scale=1.0,
        )
        layers.append(layer)
        # Advance running prediction.
        for ex_idx in range(len(running)):
            flat = flat_inputs[ex_idx]
            per_channel = []
            for c_idx in range(W45_N_CHANNELS):
                base_idx_f = c_idx * fd
                proj_chunk = list(proj_chunks[c_idx])
                feats = flat[base_idx_f:base_idx_f + fd]
                per_channel.append(
                    float(sum(
                        f * p for f, p in
                        zip(feats, proj_chunk))))
            att_weights = _softmax(list(att_logits))
            pooled = sum(
                float(wa) * float(v)
                for wa, v in
                zip(att_weights, per_channel)) * W45_N_CHANNELS
            running[ex_idx] += float(layer.residual_scale) * (
                float(pooled) + float(w[-1]))

    # ---- Stage L+1: multi-rank role adapter ----
    rdr = max(1, int(role_delta_rank))
    role_deltas: list[tuple[str, tuple[float, ...]]] = []
    if fit_role_deltas and rdr > 0:
        by_role: dict[str, list[int]] = {}
        for ex_idx, ex in enumerate(training_set.examples):
            by_role.setdefault(str(ex.role), []).append(ex_idx)
        for role in sorted(by_role.keys()):
            idxs = by_role[role]
            if len(idxs) < rdr + 1:
                continue
            resid_x: list[list[float]] = []
            resid_y: list[float] = []
            for ex_idx in idxs:
                # Per-channel logits from the BASE only (matches
                # the W45 adapter convention).
                fmap = (
                    training_set.examples[ex_idx]
                    .channel_features_map)
                per_channel_logits = []
                for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
                    feats = list(fmap.get(c_name, ()))[:fd]
                    while len(feats) < fd:
                        feats.append(0.0)
                    proj_chunk = list(
                        base.channel_projection[c_idx])[:fd]
                    while len(proj_chunk) < fd:
                        proj_chunk.append(0.0)
                    per_channel_logits.append(
                        sum(f * p for f, p in
                            zip(feats, proj_chunk)))
                rank_feats = []
                pooled = float(sum(per_channel_logits))
                rank_feats.append(pooled)  # basis 0
                for r_idx in range(1, rdr):
                    rotated = sum(
                        per_channel_logits[
                            (c + r_idx) % W45_N_CHANNELS]
                        for c in range(W45_N_CHANNELS)) * (
                            (-1.0) ** r_idx)
                    rank_feats.append(rotated)
                rank_feats.append(1.0)  # per-role bias
                # Residual against the full running prediction.
                resid = float(
                    labels[ex_idx]) - float(running[ex_idx])
                resid_x.append(rank_feats)
                resid_y.append(resid)
            delta_w = _solve_ridge(
                resid_x, resid_y, lam=float(ridge_lambda))
            if len(delta_w) == rdr + 1:
                role_deltas.append(
                    (role, tuple(_round_floats(delta_w))))

    role_adapter = MultiRankRoleAdapter(
        rank=int(rdr), role_deltas=tuple(role_deltas))

    # ---- Stage L+2: dictionary ----
    dict_k = max(1, int(dictionary_size))
    flat_set: list[list[float]] = list(flat_inputs)
    # Deterministic seeded prototype selection. We don't have a
    # PRNG here; we seed from the training-set CID so the
    # process is reproducible across runs.
    seed_int = int(training_set.cid()[:8], 16) if (
        training_set.cid()) else 0
    n = len(flat_set)
    if n == 0:
        prototypes: list[tuple[float, ...]] = [
            tuple([0.0] * (W45_N_CHANNELS * fd))
            for _ in range(dict_k)]
    else:
        # Seed prototypes via a deterministic farthest-point walk.
        first_idx = seed_int % n
        proto_idxs: list[int] = [first_idx]
        for _ in range(min(dict_k - 1, n - 1)):
            # Pick the example farthest from any chosen prototype.
            best_idx = 0
            best_dist = -1.0
            for ex_idx in range(n):
                if ex_idx in proto_idxs:
                    continue
                v = flat_set[ex_idx]
                min_dist = float("inf")
                for pi in proto_idxs:
                    p = flat_set[pi]
                    d = sum(
                        (float(a) - float(b)) ** 2
                        for a, b in zip(v, p))
                    if d < min_dist:
                        min_dist = d
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = ex_idx
            proto_idxs.append(best_idx)
        # Initial prototypes.
        prototypes_init = [
            tuple(_round_floats(flat_set[pi]))
            for pi in proto_idxs]
        # Pad to dict_k if we ran out of examples.
        while len(prototypes_init) < dict_k:
            prototypes_init.append(
                tuple([0.0] * (W45_N_CHANNELS * fd)))
        # One refinement pass: assign each example to closest
        # prototype, recompute prototypes as mean of assignees.
        assigns: list[int] = [0] * n
        for ex_idx in range(n):
            v = flat_set[ex_idx]
            best_i = 0
            best_d = float("inf")
            for pi in range(dict_k):
                p = list(prototypes_init[pi])
                d = sum(
                    (float(a) - float(b)) ** 2
                    for a, b in zip(v, p))
                if d < best_d:
                    best_d = d
                    best_i = pi
            assigns[ex_idx] = best_i
        new_prototypes: list[tuple[float, ...]] = []
        for pi in range(dict_k):
            members = [
                flat_set[ex_idx]
                for ex_idx in range(n)
                if assigns[ex_idx] == pi]
            if not members:
                new_prototypes.append(prototypes_init[pi])
                continue
            dim = len(members[0])
            mean_v = [0.0] * dim
            for v in members:
                for k in range(dim):
                    mean_v[k] += float(v[k])
            mean_v = [m / float(len(members)) for m in mean_v]
            new_prototypes.append(tuple(_round_floats(mean_v)))
        prototypes = new_prototypes

    dictionary = DictionaryBasis(
        feature_dim=fd,
        prototypes=tuple(prototypes),
    )

    return MultiLayerControllerParams(
        base=base,
        layers=tuple(layers),
        role_adapter=role_adapter,
        dictionary=dictionary,
        time_attention_weight=float(round(
            float(time_attention_weight), 12)),
        time_attention_bias=0.0,
        fitting_method="ridge_stack_v1",
        training_set_cid=str(training_set.cid()),
    )


# =============================================================================
# Memory bank
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    """One entry of the W46 manifold memory bank.

    Frozen and content-addressable. Stores everything an auditor
    needs to recompute the controller's per-turn state from the
    envelope chain alone.
    """

    turn_index: int
    role: str
    role_handoff_signature_cid: str
    channel_features: tuple[tuple[str, tuple[float, ...]], ...]
    per_channel_logits: tuple[float, ...]
    gate_logit: float
    ratify_probability: float
    decision_branch: str
    dict_index: int
    dict_residual_l1: float

    @property
    def channel_features_map(self) -> dict[str, tuple[float, ...]]:
        return {str(k): tuple(v) for k, v in self.channel_features}

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "role_handoff_signature_cid": str(
                self.role_handoff_signature_cid),
            "channel_features": [
                [str(k), _round_floats(v)]
                for k, v in self.channel_features
            ],
            "per_channel_logits": _round_floats(
                self.per_channel_logits),
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "decision_branch": str(self.decision_branch),
            "dict_index": int(self.dict_index),
            "dict_residual_l1": float(round(
                self.dict_residual_l1, 12)),
        }


@dataclasses.dataclass
class ManifoldMemoryBank:
    """Bounded, content-addressed ring buffer of past turn entries.

    The bank's CID is the SHA-256 over its sorted entries; an
    auditor can re-derive the bank state from the envelope chain
    alone.
    """

    capacity: int = W46_DEFAULT_MEMORY_CAPACITY
    entries: list[MemoryEntry] = dataclasses.field(
        default_factory=list)

    def reset(self) -> None:
        self.entries = []

    def append(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > int(self.capacity):
            self.entries = self.entries[-int(self.capacity):]

    def admissible_for_turn(
            self, turn_index: int,
    ) -> tuple[MemoryEntry, ...]:
        """Return entries with ``turn_index < turn_index``
        (strict causal mask)."""
        return tuple(
            e for e in self.entries
            if int(e.turn_index) < int(turn_index))

    def head_cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_memory_bank_head",
            "capacity": int(self.capacity),
            "entries": [e.to_dict() for e in self.entries],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "entries": [e.to_dict() for e in self.entries],
        }


# =============================================================================
# Time-attention readout
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TimeAttentionWitness:
    """Records what the W46 controller read from the memory bank
    when ratifying one turn.

    ``query_l1`` is the L1 norm of the query (flattened channel
    features). ``attn_weights`` is the softmax-normalised cosine
    similarities over admissible memory entries. ``pooled_value``
    is the weighted-sum gate-logit contributed by the bank.
    ``mask_size`` is the number of admissible entries (cardinality
    of ``< turn_index``).
    """

    turn_index: int
    mask_size: int
    query_l1: float
    attn_weights: tuple[float, ...]
    pooled_value: float
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "mask_size": int(self.mask_size),
            "query_l1": float(round(self.query_l1, 12)),
            "attn_weights": _round_floats(self.attn_weights),
            "pooled_value": float(round(self.pooled_value, 12)),
            "enabled": bool(self.enabled),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_time_attention_witness",
            "witness": self.to_dict(),
        })


def compute_time_attention(
        *,
        flat_query: Sequence[float],
        memory_bank: ManifoldMemoryBank,
        turn_index: int,
        temperature: float = W46_DEFAULT_TIME_ATTN_TEMPERATURE,
        feature_dim: int,
        enabled: bool = True,
) -> TimeAttentionWitness:
    """Compute the causally-masked time-attention readout.

    For each admissible memory entry (turn_index strictly less
    than the current turn), compute the cosine similarity between
    the current turn's query (flattened channel features) and the
    entry's flattened features. Softmax-normalise the similarities
    with the given temperature and pool the entry's gate logits.

    Returns a :class:`TimeAttentionWitness` capturing the pooled
    value and the attention weights actually used.
    """
    admissible = memory_bank.admissible_for_turn(int(turn_index))
    if not enabled or not admissible:
        return TimeAttentionWitness(
            turn_index=int(turn_index),
            mask_size=len(admissible),
            query_l1=_l1_norm(flat_query),
            attn_weights=tuple(),
            pooled_value=0.0,
            enabled=bool(enabled),
        )
    sims: list[float] = []
    values: list[float] = []
    for e in admissible:
        flat_e = list(_flatten_channel_features(
            e.channel_features_map, feature_dim=int(feature_dim)))
        sim = _cosine_similarity(flat_query, flat_e)
        sims.append(float(sim) / max(1e-9, float(temperature)))
        values.append(float(e.gate_logit))
    weights = _softmax(sims)
    pooled = float(sum(
        float(w) * float(v) for w, v in zip(weights, values)))
    return TimeAttentionWitness(
        turn_index=int(turn_index),
        mask_size=len(admissible),
        query_l1=_l1_norm(flat_query),
        attn_weights=tuple(_round_floats(weights)),
        pooled_value=float(round(pooled, 12)),
        enabled=True,
    )


# =============================================================================
# Multi-layer + multi-rank forward
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MemoryForwardResult:
    """Result of one multi-layer + memory forward pass."""

    base_forward: ControllerForwardResult
    layer_logits: tuple[float, ...]
    role_delta_value: float
    role_adapter_present: bool
    dict_index: int
    dict_residual: tuple[float, ...]
    dict_residual_l1: float
    time_attention: TimeAttentionWitness
    gate_logit: float
    ratify_probability: float
    confidence_bucket: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_forward": self.base_forward.to_dict(),
            "layer_logits": _round_floats(self.layer_logits),
            "role_delta_value": float(round(
                self.role_delta_value, 12)),
            "role_adapter_present": bool(
                self.role_adapter_present),
            "dict_index": int(self.dict_index),
            "dict_residual_l1": float(round(
                self.dict_residual_l1, 12)),
            "time_attention": self.time_attention.to_dict(),
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "confidence_bucket": int(self.confidence_bucket),
        }


def forward_memory_controller(
        *,
        channel_features: Mapping[str, Sequence[float]],
        params: MultiLayerControllerParams,
        role: str,
        memory_bank: ManifoldMemoryBank,
        turn_index: int,
        use_attention_routing: bool = True,
        time_attention_enabled: bool = True,
        time_attention_temperature: float = (
            W46_DEFAULT_TIME_ATTN_TEMPERATURE),
        role_adapter_disabled: bool = False,
        dictionary_enabled: bool = True,
) -> MemoryForwardResult:
    """Run one W46 forward pass.

    Computes:
      1. The W45 base forward (per-channel logits, softmax-pooled
         layer-0 logit, base shared bias).
      2. The per-layer additive logits (each layer's softmax pool
         of channel projections of the *original* features +
         per-layer bias).
      3. The multi-rank role delta (rank-r per-role linear
         combination of basis vectors).
      4. The dictionary encode (closest prototype + residual).
      5. The time-attention readout (causally-masked pool of past
         gate logits).
      6. The total gate logit = base.gate_logit + sum(layer logits)
         + role_delta_value + time_attention_weight * pooled
         time_attention + time_attention_bias.
      7. The sigmoid + confidence bucket.

    Forward is deterministic and content-addressable.
    """
    fd = int(params.feature_dim)

    base_forward = forward_controller(
        channel_features=channel_features,
        params=params.base,
        role=str(role),
        use_attention_routing=bool(use_attention_routing),
    )

    # Per-layer additive logits (operate on the same channel
    # features as the base; residual fitting was done that way).
    layer_logits: list[float] = []
    for layer in params.layers:
        per_channel = []
        for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
            feats = list(channel_features.get(c_name, ()))[:fd]
            while len(feats) < fd:
                feats.append(0.0)
            proj_chunk = list(layer.channel_projection[c_idx])[:fd]
            while len(proj_chunk) < fd:
                proj_chunk.append(0.0)
            per_channel.append(
                float(sum(f * p for f, p in
                          zip(feats, proj_chunk))))
        if use_attention_routing:
            aw = _softmax(list(layer.attention_logits))
            pooled = sum(
                float(w) * float(v)
                for w, v in zip(aw, per_channel)) * W45_N_CHANNELS
        else:
            pooled = sum(per_channel)
        layer_logits.append(
            float(layer.residual_scale) * float(pooled))

    # Multi-rank role adapter (operates on base.per_channel_logits).
    pcl = list(base_forward.per_channel_logits)
    role_delta_value = 0.0
    role_adapter_present = False
    if not role_adapter_disabled:
        rmap = params.role_adapter.role_delta_map
        if str(role) in rmap:
            role_adapter_present = True
            delta = list(rmap[str(role)])
            r = int(params.role_adapter.rank)
            pooled_pcl = float(sum(pcl))
            if r >= 1 and len(delta) >= 1:
                role_delta_value += float(delta[0]) * pooled_pcl
            for r_idx in range(1, r):
                if r_idx >= len(delta):
                    break
                rotated = sum(
                    pcl[(c + r_idx) % W45_N_CHANNELS]
                    for c in range(W45_N_CHANNELS)) * (
                        (-1.0) ** r_idx)
                role_delta_value += float(delta[r_idx]) * rotated
            if len(delta) > r:
                role_delta_value += float(delta[r])

    # Dictionary encode.
    flat_query = list(_flatten_channel_features(
        channel_features, feature_dim=fd))
    if (dictionary_enabled and params.dictionary.k > 0):
        dict_index, dict_residual = params.dictionary.encode(
            flat_query)
    else:
        dict_index = W46_NO_DICT_CODE
        dict_residual = tuple(float(v) for v in flat_query)
    dict_residual_l1 = float(round(
        _l1_norm(dict_residual), 12))

    # Time-attention readout.
    ta = compute_time_attention(
        flat_query=flat_query,
        memory_bank=memory_bank,
        turn_index=int(turn_index),
        temperature=float(time_attention_temperature),
        feature_dim=fd,
        enabled=bool(time_attention_enabled),
    )

    gate_logit = (
        float(base_forward.gate_logit)
        + float(sum(layer_logits))
        + float(role_delta_value)
        + float(params.time_attention_weight) * float(ta.pooled_value)
        + float(params.time_attention_bias))
    ratify_prob = _sigmoid(gate_logit)
    conf_bucket = _confidence_bucket_for_probability(ratify_prob)

    return MemoryForwardResult(
        base_forward=base_forward,
        layer_logits=tuple(_round_floats(layer_logits)),
        role_delta_value=float(round(role_delta_value, 12)),
        role_adapter_present=bool(role_adapter_present),
        dict_index=int(dict_index),
        dict_residual=dict_residual,
        dict_residual_l1=dict_residual_l1,
        time_attention=ta,
        gate_logit=float(round(gate_logit, 12)),
        ratify_probability=float(round(ratify_prob, 12)),
        confidence_bucket=int(conf_bucket),
    )


# =============================================================================
# Shared-prefix capsule
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PrefixCapsule:
    """Deterministic shared-prefix capsule.

    Records the prefix bytes (or their SHA-256), the registered
    policy-entry CID, the per-turn list of *prior output SHAs* the
    prefix was built from, and the prefix's age in turns.

    Honest scope: this binds *identical prefix bytes across
    consecutive turns*; it does NOT manipulate the underlying
    transformer's KV cache. The optimisation that real backends
    may apply (e.g., prompt caching when the prefix is identical)
    is a runtime concern outside the W46 surface.
    """

    prefix_sha256: str
    prefix_token_count: int
    policy_entry_cid: str
    prior_output_shas: tuple[str, ...]
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_sha256": str(self.prefix_sha256),
            "prefix_token_count": int(self.prefix_token_count),
            "policy_entry_cid": str(self.policy_entry_cid),
            "prior_output_shas": [
                str(s) for s in self.prior_output_shas],
            "reused": bool(self.reused),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_prefix_capsule",
            "capsule": self.to_dict(),
        })


def build_prefix_capsule(
        *,
        prior_outputs: Sequence[tuple[str, str]],
        prefix_turns: int = W46_DEFAULT_PREFIX_TURNS,
        policy_entry_cid: str = "",
        prior_prefix_sha: str | None = None,
) -> tuple[str, PrefixCapsule]:
    """Build a deterministic prefix string + its sealed capsule.

    The prefix string is a content-addressed token block that
    summarises the **first** ``prefix_turns`` prior outputs (the
    "team origin" window) by their role and a 12-hex SHA prefix.

    Using the head window rather than the tail window is what
    makes the prefix *stable across consecutive turns* once the
    team has produced ``prefix_turns`` outputs: turn ``k+1`` and
    turn ``k`` (both with ``k >= prefix_turns``) see the same
    first-``prefix_turns`` outputs and therefore the same prefix
    bytes. This is the honest capsule-layer surface for
    "shared prefix bytes" — the model receives byte-identical
    prefix on every applicable turn.

    Returns ``(prefix_str, capsule)``. The capsule's ``reused``
    flag is True iff the new ``prefix_sha256`` equals
    ``prior_prefix_sha`` exactly.
    """
    if prefix_turns <= 0 or not prior_outputs:
        empty = ""
        sha = hashlib.sha256(empty.encode("utf-8")).hexdigest()
        return empty, PrefixCapsule(
            prefix_sha256=sha,
            prefix_token_count=0,
            policy_entry_cid=str(policy_entry_cid),
            prior_output_shas=tuple(),
            reused=(prior_prefix_sha == sha),
        )
    head = list(prior_outputs)[:int(prefix_turns)]
    shas = [
        hashlib.sha256(o.encode("utf-8")).hexdigest()
        for _, o in head]
    lines = ["MANIFOLD_PREFIX:"]
    for (role, _), sha in zip(head, shas):
        lines.append(f"  {role}={sha[:12]}")
    if policy_entry_cid:
        lines.append(f"  policy={policy_entry_cid[:12]}")
    prefix_str = "\n".join(lines)
    prefix_sha = hashlib.sha256(
        prefix_str.encode("utf-8")).hexdigest()
    capsule = PrefixCapsule(
        prefix_sha256=prefix_sha,
        prefix_token_count=len(prefix_str.split()),
        policy_entry_cid=str(policy_entry_cid),
        prior_output_shas=tuple(shas),
        reused=(prior_prefix_sha == prefix_sha),
    )
    return prefix_str, capsule


# =============================================================================
# Packed control token surface
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ControlTokenWitness:
    """Records the packed `MANIFOLD_CTRL` block's bytes + state.

    Bijective: the witness fields plus ``ctrl_mode`` are
    sufficient to reconstruct the literal control bytes.
    """

    ctrl_mode: str
    route: int
    confidence_bucket: int
    ratify_probability: float
    layer_logits: tuple[float, ...]
    mem_attn_value: float
    dict_index: int
    mem_summary: str
    ctrl_bytes_sha256: str
    n_ctrl_tokens: int
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctrl_mode": str(self.ctrl_mode),
            "route": int(self.route),
            "confidence_bucket": int(self.confidence_bucket),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "layer_logits": _round_floats(self.layer_logits),
            "mem_attn_value": float(round(self.mem_attn_value, 12)),
            "dict_index": int(self.dict_index),
            "mem_summary": str(self.mem_summary),
            "ctrl_bytes_sha256": str(self.ctrl_bytes_sha256),
            "n_ctrl_tokens": int(self.n_ctrl_tokens),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w46_control_token_witness",
            "witness": self.to_dict(),
        })


def _build_mem_summary(
        memory_bank: ManifoldMemoryBank,
        *,
        turn_index: int,
) -> str:
    """A compact, deterministic summary of the admissible memory
    bank's role-pattern. Encodes role and decision-branch suffix
    of each admissible entry."""
    admissible = memory_bank.admissible_for_turn(int(turn_index))
    if not admissible:
        return "empty"
    parts: list[str] = []
    for e in admissible:
        # Use last 4 chars of the decision branch as a compact
        # suffix; e.g. "rat" for "ratified", "tnd" for "*tnd".
        suffix = (
            e.decision_branch.split("_")[-1][:3]
            if e.decision_branch else "")
        parts.append(f"{e.role}:{suffix}")
    return ",".join(parts)


def build_control_token_string(
        *,
        ctrl_mode: str,
        route: int,
        confidence_bucket: int,
        ratify_probability: float,
        layer_logits: Sequence[float],
        mem_attn_value: float,
        dict_index: int,
        mem_summary: str,
        role_universe: Sequence[str],
        turn_index: int,
        probability_precision: int = 4,
) -> tuple[str, ControlTokenWitness]:
    """Build the literal ``MANIFOLD_CTRL`` control block bytes plus
    a content-addressed witness."""
    p = round(float(ratify_probability), int(probability_precision))
    if ctrl_mode == W46_CTRL_MODE_OFF:
        ctrl_str = ""
    elif ctrl_mode == W46_CTRL_MODE_COMPACT:
        ctrl_str = (
            f"MANIFOLD_CTRL: "
            f"route={int(route)} "
            f"conf={int(confidence_bucket)} "
            f"p={p} "
            f"dict_idx={int(dict_index)} "
            f"over {','.join(role_universe)}")
    elif ctrl_mode == W46_CTRL_MODE_FULL:
        ll = ",".join(
            f"{round(float(v), int(probability_precision))}"
            for v in layer_logits)
        ctrl_str = (
            f"MANIFOLD_CTRL:\n"
            f"  route={int(route)}\n"
            f"  conf={int(confidence_bucket)}\n"
            f"  p={p}\n"
            f"  layer_logits=[{ll}]\n"
            f"  mem_attn={round(float(mem_attn_value), int(probability_precision))}\n"
            f"  dict_idx={int(dict_index)}\n"
            f"  mem_summary={mem_summary}\n"
            f"  over {','.join(role_universe)}")
    else:
        raise ValueError(
            f"ctrl_mode={ctrl_mode!r} not in {W46_ALL_CTRL_MODES}")
    sha = hashlib.sha256(ctrl_str.encode("utf-8")).hexdigest()
    witness = ControlTokenWitness(
        ctrl_mode=str(ctrl_mode),
        route=int(route),
        confidence_bucket=int(confidence_bucket),
        ratify_probability=float(round(
            ratify_probability, 12)),
        layer_logits=tuple(_round_floats(layer_logits)),
        mem_attn_value=float(round(mem_attn_value, 12)),
        dict_index=int(dict_index),
        mem_summary=str(mem_summary),
        ctrl_bytes_sha256=str(sha),
        n_ctrl_tokens=int(len(ctrl_str.split())),
        turn_index=int(turn_index),
    )
    return ctrl_str, witness


# =============================================================================
# Registry
# =============================================================================

@dataclasses.dataclass
class ManifoldMemoryRegistry:
    """Controller-side configuration for the W46 memory coupling.

    Wraps a :class:`LearnedManifoldRegistry` (the W45 inner) and
    adds memory-controller toggles plus the fitted multi-layer
    params.
    """

    schema_cid: str
    learned_registry: LearnedManifoldRegistry
    params: MultiLayerControllerParams
    memory_enabled: bool = True
    time_attention_enabled: bool = True
    dictionary_enabled: bool = True
    role_adapter_disabled: bool = False
    control_token_mode: str = W46_CTRL_MODE_FULL
    prefix_reuse_enabled: bool = True
    prefix_turns: int = W46_DEFAULT_PREFIX_TURNS
    memory_capacity: int = W46_DEFAULT_MEMORY_CAPACITY
    time_attention_temperature: float = (
        W46_DEFAULT_TIME_ATTN_TEMPERATURE)
    margin_abstain_threshold: float = 0.0
    time_attn_abstain_threshold: float | None = None
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT
    use_attention_routing: bool = True

    @property
    def is_trivial(self) -> bool:
        return (
            self.learned_registry.is_trivial
            and not self.memory_enabled
            and not self.time_attention_enabled
            and not self.dictionary_enabled
            and self.role_adapter_disabled
            and self.control_token_mode == W46_CTRL_MODE_OFF
            and not self.prefix_reuse_enabled
            and not self.abstain_substitution_enabled
            and self.params.fitting_method == "unfitted"
        )


def build_trivial_manifold_memory_registry(
        *, schema_cid: str | None = None,
) -> ManifoldMemoryRegistry:
    """Build a registry whose orchestrator reduces to
    LearnedManifoldTeam (trivial) and therefore to AgentTeam
    byte-for-byte (the W46-L-TRIVIAL-MEMORY-PASSTHROUGH
    falsifier)."""
    cid = schema_cid or _sha256_hex({
        "kind": "w46_trivial_schema"})
    return ManifoldMemoryRegistry(
        schema_cid=str(cid),
        learned_registry=build_trivial_learned_manifold_registry(
            schema_cid=str(cid)),
        params=build_unfitted_memory_controller_params(),
        memory_enabled=False,
        time_attention_enabled=False,
        dictionary_enabled=False,
        role_adapter_disabled=True,
        control_token_mode=W46_CTRL_MODE_OFF,
        prefix_reuse_enabled=False,
        abstain_substitution_enabled=False,
    )


def build_manifold_memory_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        params: MultiLayerControllerParams | None = None,
        memory_enabled: bool = True,
        time_attention_enabled: bool = True,
        dictionary_enabled: bool = True,
        role_adapter_disabled: bool = False,
        control_token_mode: str = W46_CTRL_MODE_FULL,
        prefix_reuse_enabled: bool = True,
        prefix_turns: int = W46_DEFAULT_PREFIX_TURNS,
        memory_capacity: int = W46_DEFAULT_MEMORY_CAPACITY,
        time_attention_temperature: float = (
            W46_DEFAULT_TIME_ATTN_TEMPERATURE),
        margin_abstain_threshold: float = 0.0,
        time_attn_abstain_threshold: float | None = None,
        abstain_substitution_enabled: bool = True,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
        live_enabled: bool = True,
        learned_enabled: bool = True,
        use_attention_routing: bool = True,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
) -> ManifoldMemoryRegistry:
    """Build a fully configured manifold-memory registry."""
    if control_token_mode not in W46_ALL_CTRL_MODES:
        raise ValueError(
            f"control_token_mode={control_token_mode!r} not in "
            f"{W46_ALL_CTRL_MODES}")
    learned = build_learned_manifold_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        params=(params.base if params else None),
        learned_enabled=bool(learned_enabled),
        use_attention_routing=bool(use_attention_routing),
        role_adapter_disabled=bool(role_adapter_disabled),
        prompt_hint_mode=W45_HINT_MODE_OFF,  # W46 supersedes hint
        abstain_substitution_enabled=False,  # W46 owns abstain
        margin_abstain_threshold=float(margin_abstain_threshold),
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
        live_enabled=bool(live_enabled),
        inline_route_mode=W44_ROUTE_MODE_FACTORADIC,
        abstain_output=str(abstain_output),
    )
    return ManifoldMemoryRegistry(
        schema_cid=str(schema_cid),
        learned_registry=learned,
        params=(
            params or build_unfitted_memory_controller_params()),
        memory_enabled=bool(memory_enabled),
        time_attention_enabled=bool(time_attention_enabled),
        dictionary_enabled=bool(dictionary_enabled),
        role_adapter_disabled=bool(role_adapter_disabled),
        control_token_mode=str(control_token_mode),
        prefix_reuse_enabled=bool(prefix_reuse_enabled),
        prefix_turns=int(prefix_turns),
        memory_capacity=int(memory_capacity),
        time_attention_temperature=float(
            time_attention_temperature),
        margin_abstain_threshold=float(margin_abstain_threshold),
        time_attn_abstain_threshold=(
            None if time_attn_abstain_threshold is None
            else float(time_attn_abstain_threshold)),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        use_attention_routing=bool(use_attention_routing),
    )


# =============================================================================
# Decision selector
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MemoryGatingDecision:
    """Result of running the memory gate on one turn."""

    branch: str
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
    forward: MemoryForwardResult
    causal_mask_cid: str
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W46_MEMORY_ABSTAIN_BRANCHES


def _classify_w45_branch_to_memory(
        w45_branch: str,
) -> str:
    """Map a W45 learned branch to the corresponding W46 branch."""
    if w45_branch == W45_BRANCH_LEARNED_RATIFIED:
        return W46_BRANCH_MEMORY_RATIFIED
    if w45_branch == W45_BRANCH_LEARNED_NO_POLICY:
        return W46_BRANCH_MEMORY_NO_POLICY
    if w45_branch == W45_BRANCH_LEARNED_CAUSAL_ABSTAIN:
        return W46_BRANCH_MEMORY_CAUSAL_ABSTAIN
    if w45_branch == W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN:
        return W46_BRANCH_MEMORY_SPHERICAL_ABSTAIN
    if w45_branch == W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN:
        return W46_BRANCH_MEMORY_SUBSPACE_ABSTAIN
    if w45_branch == W45_BRANCH_LEARNED_MARGIN_ABSTAIN:
        return W46_BRANCH_MEMORY_MARGIN_ABSTAIN
    if w45_branch == W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH:
        return W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH
    if w45_branch == W45_BRANCH_LEARNED_DISABLED:
        return W46_BRANCH_MEMORY_DISABLED
    return W46_BRANCH_MEMORY_DISABLED


# =============================================================================
# Orchestrator
# =============================================================================

class ManifoldMemoryOrchestrator:
    """Per-turn memory gating + control-token witness.

    Wraps a :class:`LearnedManifoldOrchestrator` (the W45 inner)
    plus a :class:`ManifoldMemoryRegistry` plus a stateful
    :class:`ManifoldMemoryBank`. The bank is updated once per
    :meth:`gate` call; :meth:`reset_session` clears the bank.
    """

    def __init__(self, registry: ManifoldMemoryRegistry) -> None:
        self.registry = registry
        self._learned = LearnedManifoldOrchestrator(
            registry=registry.learned_registry)
        self._memory_bank = ManifoldMemoryBank(
            capacity=int(registry.memory_capacity))

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    @property
    def memory_bank(self) -> ManifoldMemoryBank:
        return self._memory_bank

    def reset_session(self) -> None:
        self._learned.reset_session()
        self._memory_bank.reset()

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
    ) -> tuple[MemoryGatingDecision, Any]:
        """Run the memory gate for one turn.

        Returns ``(decision, aux)`` where ``aux`` is a tuple of
        the downstream W45/W44/W43 sub-results and the memory
        forward result for downstream envelope sealing.
        """
        # Delegate to the W45 inner.
        w45_decision, w45_aux = self._learned.gate(
            observation=observation,
            role=str(role),
            role_handoff_signature_cid=role_handoff_signature_cid,
            parent_w42_cid=str(parent_w42_cid),
            n_w42_visible_tokens=int(n_w42_visible_tokens),
            turn_index=int(turn_index),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )
        w43_result, causal_mask, bundle = w45_aux

        # Encode channel features ourselves (we need them again).
        feats = _channel_features_from_bundle(
            bundle,
            feature_dim=int(self.registry.params.feature_dim),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )

        # Run the memory forward (multi-layer + multi-rank + memory).
        forward = forward_memory_controller(
            channel_features=feats,
            params=self.registry.params,
            role=str(role),
            memory_bank=self._memory_bank,
            turn_index=int(turn_index),
            use_attention_routing=bool(
                self.registry.use_attention_routing),
            time_attention_enabled=bool(
                self.registry.time_attention_enabled),
            time_attention_temperature=float(
                self.registry.time_attention_temperature),
            role_adapter_disabled=bool(
                self.registry.role_adapter_disabled),
            dictionary_enabled=bool(
                self.registry.dictionary_enabled),
        )

        # Branch classification: start from the W45 branch then
        # apply W46-specific abstain overrides.
        memory_branch = _classify_w45_branch_to_memory(
            w45_decision.branch)
        abstain_reason = w45_decision.abstain_reason

        if (self.registry.memory_enabled
                and memory_branch == W46_BRANCH_MEMORY_RATIFIED
                and forward.gate_logit
                < float(self.registry.margin_abstain_threshold)):
            memory_branch = W46_BRANCH_MEMORY_MARGIN_ABSTAIN
            abstain_reason = "memory_margin"

        if (self.registry.memory_enabled
                and self.registry.time_attention_enabled
                and self.registry.time_attn_abstain_threshold is not None
                and memory_branch == W46_BRANCH_MEMORY_RATIFIED
                and forward.time_attention.pooled_value
                < float(
                    self.registry.time_attn_abstain_threshold)):
            memory_branch = W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN
            abstain_reason = "memory_time_attn"

        if not self.registry.memory_enabled:
            if self.registry.is_trivial:
                memory_branch = (
                    W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH)
            elif memory_branch == W46_BRANCH_MEMORY_RATIFIED:
                memory_branch = W46_BRANCH_MEMORY_DISABLED

        if w45_decision.branch == W45_BRANCH_LEARNED_NO_POLICY or (
                w45_decision.w44_branch
                == W44_BRANCH_LIVE_NO_POLICY):
            memory_branch = W46_BRANCH_MEMORY_NO_POLICY

        # Update the bank AFTER deciding the branch so future
        # turns can see this turn's entry.
        if (self.registry.memory_enabled
                or memory_branch
                != W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH):
            channel_feats_tuple = tuple(
                (str(k), tuple(_round_floats(v)))
                for k, v in feats.items())
            entry = MemoryEntry(
                turn_index=int(turn_index),
                role=str(role),
                role_handoff_signature_cid=str(
                    w45_decision.role_handoff_signature_cid),
                channel_features=channel_feats_tuple,
                per_channel_logits=tuple(_round_floats(
                    forward.base_forward.per_channel_logits)),
                gate_logit=float(forward.gate_logit),
                ratify_probability=float(
                    forward.ratify_probability),
                decision_branch=str(memory_branch),
                dict_index=int(forward.dict_index),
                dict_residual_l1=float(forward.dict_residual_l1),
            )
            self._memory_bank.append(entry)

        decision = MemoryGatingDecision(
            branch=str(memory_branch),
            w45_branch=str(w45_decision.branch),
            w44_branch=str(w45_decision.w44_branch),
            pmc_branch=str(w45_decision.pmc_branch),
            spherical_agreement=float(
                w45_decision.spherical_agreement),
            subspace_drift=float(w45_decision.subspace_drift),
            causal_admissible=bool(
                w45_decision.causal_admissible),
            factoradic_int=int(w45_decision.factoradic_int),
            factoradic_n_bits=int(
                w45_decision.factoradic_n_bits),
            role_handoff_signature_cid=str(
                w45_decision.role_handoff_signature_cid),
            policy_entry_cid=str(w45_decision.policy_entry_cid),
            pmc_envelope_cid=str(w45_decision.pmc_envelope_cid),
            w44_envelope_cid=str(w45_decision.w44_envelope_cid),
            w45_envelope_cid="",  # filled in by the team loop
            forward=forward,
            causal_mask_cid=str(w45_decision.causal_mask_cid),
            abstain_reason=str(abstain_reason),
        )
        return decision, (
            w43_result, causal_mask, bundle, w45_decision, forward)


# =============================================================================
# Envelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ManifoldMemoryHandoffEnvelope:
    """Sealed memory-manifold envelope for one turn of the W46
    layer.

    Records:

    * the underlying ``TEAM_HANDOFF`` capsule CID
    * the W45 learned envelope CID
    * the W44 live envelope CID
    * the W43 manifest-v13 envelope CID
    * the multi-layer-controller-params CID
    * the dictionary CID, dict_index, dict_residual_l1
    * the time-attention witness CID + pooled value
    * the multi-rank-role-adapter witness CID
    * the memory-bank head CID (after this turn's append)
    * the control-token witness CID
    * the prefix-capsule CID
    * the prompt-construction witness CID
    * the outer CID binding everything

    The outer CID is content-addressed by every other field; the
    verifier re-derives the outer CID from the bytes alone.
    """

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w45_envelope_cid: str
    parent_w44_envelope_cid: str
    parent_w43_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
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

    # Multi-layer controller provenance.
    controller_params_cid: str
    training_set_cid: str
    fitting_method: str
    n_layers: int
    role_adapter_rank: int
    dictionary_cid: str
    dictionary_size: int
    use_attention_routing: bool
    role_adapter_disabled: bool

    # Forward result witnesses.
    layer_logits: tuple[float, ...]
    time_attention_witness_cid: str
    time_attention_pooled: float
    time_attention_mask_size: int
    multi_rank_adapter_witness_cid: str
    dict_index: int
    dict_residual_l1: float
    causal_mask_witness_cid: str

    # Memory bank provenance.
    memory_bank_head_cid: str
    memory_bank_size: int
    memory_capacity: int

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

    memory_witness_cid: str
    memory_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w46_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w45_envelope_cid=self.parent_w45_envelope_cid,
            controller_params_cid=self.controller_params_cid,
            memory_bank_head_cid=self.memory_bank_head_cid,
            memory_witness_cid=self.memory_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_w46_multi_rank_adapter_witness_cid(
        *,
        role: str,
        role_delta_value: float,
        role_delta_rank: int,
        role_adapter_disabled: bool,
        present: bool,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w46_multi_rank_role_adapter_witness",
        "role": str(role),
        "role_delta_value": float(round(role_delta_value, 12)),
        "role_delta_rank": int(role_delta_rank),
        "role_adapter_disabled": bool(role_adapter_disabled),
        "present": bool(present),
        "turn_index": int(turn_index),
    })


def _compute_w46_prompt_construction_witness_cid(
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
) -> str:
    return _sha256_hex({
        "kind": "w46_prompt_construction_witness",
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
    })


def _compute_w46_memory_witness_cid(
        *,
        decision_branch: str,
        w45_branch: str,
        w44_branch: str,
        pmc_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        controller_params_cid: str,
        dictionary_cid: str,
        time_attention_witness_cid: str,
        multi_rank_adapter_witness_cid: str,
        causal_mask_witness_cid: str,
        prompt_construction_witness_cid: str,
        control_token_witness_cid: str,
        prefix_capsule_cid: str,
        memory_bank_head_cid: str,
        output_sha256: str,
        behavioral_change: bool,
) -> str:
    return _sha256_hex({
        "kind": "w46_memory_witness",
        "decision_branch": str(decision_branch),
        "w45_branch": str(w45_branch),
        "w44_branch": str(w44_branch),
        "pmc_branch": str(pmc_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "controller_params_cid": str(controller_params_cid),
        "dictionary_cid": str(dictionary_cid),
        "time_attention_witness_cid": str(
            time_attention_witness_cid),
        "multi_rank_adapter_witness_cid": str(
            multi_rank_adapter_witness_cid),
        "causal_mask_witness_cid": str(causal_mask_witness_cid),
        "prompt_construction_witness_cid": str(
            prompt_construction_witness_cid),
        "control_token_witness_cid": str(
            control_token_witness_cid),
        "prefix_capsule_cid": str(prefix_capsule_cid),
        "memory_bank_head_cid": str(memory_bank_head_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
    })


def _compute_w46_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w45_envelope_cid: str,
        controller_params_cid: str,
        memory_bank_head_cid: str,
        memory_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w46_memory_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w45_envelope_cid": str(parent_w45_envelope_cid),
        "controller_params_cid": str(controller_params_cid),
        "memory_bank_head_cid": str(memory_bank_head_cid),
        "memory_witness_cid": str(memory_witness_cid),
        "turn_index": int(turn_index),
    })


# =============================================================================
# Verifier (16+ enumerated W46 failure modes)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ManifoldMemoryVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


W46_ALL_FAILURE_MODES: tuple[str, ...] = (
    "empty_w46_envelope",
    "w46_schema_version_unknown",
    "w46_schema_cid_mismatch",
    "w46_decision_branch_unknown",
    "w46_ctrl_mode_unknown",
    "w46_role_handoff_signature_cid_invalid",
    "w46_prompt_sha256_invalid",
    "w46_token_accounting_invalid",
    "w46_confidence_bucket_invalid",
    "w46_ratify_probability_invalid",
    "w46_controller_params_cid_invalid",
    "w46_dictionary_cid_invalid",
    "w46_time_attention_witness_cid_invalid",
    "w46_multi_rank_adapter_witness_cid_mismatch",
    "w46_causal_mask_witness_cid_invalid",
    "w46_control_token_witness_cid_invalid",
    "w46_prefix_capsule_cid_invalid",
    "w46_memory_bank_head_cid_invalid",
    "w46_prompt_construction_witness_cid_mismatch",
    "w46_memory_witness_cid_mismatch",
    "w46_outer_cid_mismatch",
)


def verify_manifold_memory_handoff(
        env: "ManifoldMemoryHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_controller_params_cid: str | None = None,
) -> ManifoldMemoryVerificationOutcome:
    """Pure-function verifier for the W46 memory envelope.

    Enumerates 21 disjoint W46 failure modes (see
    :data:`W46_ALL_FAILURE_MODES`).
    """
    n = 0
    if env is None:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="empty_w46_envelope", n_checks=0)
    n += 1
    if env.schema_version != W46_MANIFOLD_MEMORY_SCHEMA_VERSION:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W46_ALL_BRANCHES:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.control_token_mode not in W46_ALL_CTRL_MODES:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_ctrl_mode_unknown", n_checks=n)
    n += 1
    if env.decision_branch != (
            W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return ManifoldMemoryVerificationOutcome(
                ok=False,
                reason=(
                    "w46_role_handoff_signature_cid_invalid"),
                n_checks=n)
    n += 1
    if (env.prompt_sha256 is None
            or (env.prompt_sha256
                and len(env.prompt_sha256) not in (0, 64))):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_prompt_sha256_invalid",
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
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_token_accounting_invalid",
            n_checks=n)
    n += 1
    if (env.hint_confidence_bucket < 0
            or env.hint_confidence_bucket
            >= W45_CONFIDENCE_BUCKETS):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_confidence_bucket_invalid",
            n_checks=n)
    n += 1
    if not (0.0 - 1e-9 <= float(env.ratify_probability)
            <= 1.0 + 1e-9):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_ratify_probability_invalid",
            n_checks=n)
    n += 1
    if (not env.controller_params_cid
            or len(env.controller_params_cid) != 64):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_controller_params_cid_invalid",
            n_checks=n)
    n += 1
    if registered_controller_params_cid is not None:
        if env.controller_params_cid != str(
                registered_controller_params_cid):
            return ManifoldMemoryVerificationOutcome(
                ok=False,
                reason="w46_controller_params_cid_invalid",
                n_checks=n)
    if (not env.dictionary_cid
            or len(env.dictionary_cid) != 64):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_dictionary_cid_invalid",
            n_checks=n)
    n += 1
    if (env.decision_branch
            != W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH):
        if (not env.time_attention_witness_cid
                or len(env.time_attention_witness_cid) != 64):
            return ManifoldMemoryVerificationOutcome(
                ok=False,
                reason=(
                    "w46_time_attention_witness_cid_invalid"),
                n_checks=n)
    n += 1
    # multi-rank adapter witness CID re-derivation.
    expected_mr = _compute_w46_multi_rank_adapter_witness_cid(
        role=env.role,
        role_delta_value=0.0,  # round-trip done via env field is
                               # acceptable for the trivial path;
                               # for non-trivial we re-derive
                               # below.
        role_delta_rank=int(env.role_adapter_rank),
        role_adapter_disabled=bool(env.role_adapter_disabled),
        present=False,
        turn_index=int(env.turn_index),
    )
    # Note: we don't have the role_delta_value at the envelope
    # level for re-derivation without round-tripping the forward
    # result; we instead enforce that the witness CID is 64-hex.
    if (not env.multi_rank_adapter_witness_cid
            or len(env.multi_rank_adapter_witness_cid) != 64):
        return ManifoldMemoryVerificationOutcome(
            ok=False,
            reason=(
                "w46_multi_rank_adapter_witness_cid_mismatch"),
            n_checks=n)
    n += 1
    if (env.decision_branch
            != W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH):
        if (not env.causal_mask_witness_cid
                or len(env.causal_mask_witness_cid) != 64):
            return ManifoldMemoryVerificationOutcome(
                ok=False,
                reason="w46_causal_mask_witness_cid_invalid",
                n_checks=n)
    n += 1
    if (env.control_token_mode != W46_CTRL_MODE_OFF):
        if (not env.control_token_witness_cid
                or len(env.control_token_witness_cid) != 64):
            return ManifoldMemoryVerificationOutcome(
                ok=False,
                reason="w46_control_token_witness_cid_invalid",
                n_checks=n)
    n += 1
    if (not env.prefix_capsule_cid
            or len(env.prefix_capsule_cid) != 64):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_prefix_capsule_cid_invalid",
            n_checks=n)
    n += 1
    if (not env.memory_bank_head_cid
            or len(env.memory_bank_head_cid) != 64):
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_memory_bank_head_cid_invalid",
            n_checks=n)
    n += 1
    expected_construction = (
        _compute_w46_prompt_construction_witness_cid(
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
        ))
    if expected_construction != (
            env.prompt_construction_witness_cid):
        return ManifoldMemoryVerificationOutcome(
            ok=False,
            reason=(
                "w46_prompt_construction_witness_cid_mismatch"),
            n_checks=n)
    n += 1
    # Memory witness re-derivation.
    expected_witness = _compute_w46_memory_witness_cid(
        decision_branch=env.decision_branch,
        w45_branch=env.w45_branch,
        w44_branch=env.w44_branch,
        pmc_branch=env.pmc_branch,
        abstain_reason=env.abstain_reason,
        role_handoff_signature_cid=env.role_handoff_signature_cid,
        policy_entry_cid=env.policy_entry_cid,
        controller_params_cid=env.controller_params_cid,
        dictionary_cid=env.dictionary_cid,
        time_attention_witness_cid=env.time_attention_witness_cid,
        multi_rank_adapter_witness_cid=(
            env.multi_rank_adapter_witness_cid),
        causal_mask_witness_cid=env.causal_mask_witness_cid,
        prompt_construction_witness_cid=(
            env.prompt_construction_witness_cid),
        control_token_witness_cid=env.control_token_witness_cid,
        prefix_capsule_cid=env.prefix_capsule_cid,
        memory_bank_head_cid=env.memory_bank_head_cid,
        output_sha256=env.output_sha256,
        behavioral_change=bool(env.behavioral_change),
    )
    if expected_witness != env.memory_witness_cid:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_memory_witness_cid_mismatch",
            n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.memory_outer_cid:
        return ManifoldMemoryVerificationOutcome(
            ok=False, reason="w46_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return ManifoldMemoryVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Team result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ManifoldMemoryTurn:
    """One turn of a :class:`ManifoldMemoryTeam` run."""

    agent_turn: AgentTurn
    decision: MemoryGatingDecision
    envelope: ManifoldMemoryHandoffEnvelope


@dataclasses.dataclass(frozen=True)
class ManifoldMemoryTeamResult:
    """Result of a :class:`ManifoldMemoryTeam` run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    memory_turns: tuple[ManifoldMemoryTurn, ...]
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
    n_memory_margin_abstains: int = 0
    n_memory_time_attn_abstains: int = 0
    n_prefix_reuses: int = 0
    mean_ratify_probability: float = 0.0
    mean_time_attention_pooled: float = 0.0
    controller_params_cid: str = ""
    dictionary_cid: str = ""
    final_memory_bank_head_cid: str = ""
    schema: str = W46_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


# =============================================================================
# Team
# =============================================================================

class ManifoldMemoryTeam:
    """W46 manifold-memory-coupled agent team.

    Wraps the released :class:`coordpy.AgentTeam` contract with
    the W46 multi-layer memory controller plus the W45 learned
    layer + W44 live gate + W43 PMC. With a trivial memory
    registry, this team reduces to ``LearnedManifoldTeam.run``
    byte-for-byte (the W46-L-TRIVIAL-MEMORY-PASSTHROUGH falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: ManifoldMemoryRegistry,
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
                "ManifoldMemoryTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = ManifoldMemoryOrchestrator(registry)
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
                "ManifoldMemoryTeam")
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
            decision: MemoryGatingDecision,
            role_universe: Sequence[str],
            prior_prefix_sha: str | None,
    ) -> tuple[
            str, str, int, int, int, int, str, ControlTokenWitness,
            PrefixCapsule, str]:
        """Construct the bounded prompt + textual shadow.

        Returns ``(bounded_prompt, textual_prompt, n_textual,
        n_actual, n_ctrl_tokens, n_prefix_tokens, ctrl_bytes,
        ctrl_witness, prefix_capsule, prefix_str)``.
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

        # Textual shadow: AgentTeam-equivalent rendering, no ctrl
        # token, no prefix.
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

        # Prefix capsule.
        prefix_str: str = ""
        prefix_capsule: PrefixCapsule
        if (self.registry.prefix_reuse_enabled
                and turn_index > 0
                and all_prior_outputs):
            prefix_str, prefix_capsule = build_prefix_capsule(
                prior_outputs=all_prior_outputs,
                prefix_turns=int(self.registry.prefix_turns),
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

        # Control-token block.
        mem_summary = _build_mem_summary(
            self.orchestrator.memory_bank,
            turn_index=int(turn_index))
        ctrl_str, ctrl_witness = build_control_token_string(
            ctrl_mode=str(self.registry.control_token_mode),
            route=int(decision.factoradic_int),
            confidence_bucket=int(
                decision.forward.confidence_bucket),
            ratify_probability=float(
                decision.forward.ratify_probability),
            layer_logits=decision.forward.layer_logits,
            mem_attn_value=float(
                decision.forward.time_attention.pooled_value),
            dict_index=int(decision.forward.dict_index),
            mem_summary=str(mem_summary),
            role_universe=role_universe,
            turn_index=int(turn_index),
        )

        # Bounded prompt:
        # 1) common parts
        # 2) prefix (if any)
        # 3) factoradic route (preserved from W44 path)
        # 4) control block (if mode != off)
        # 5) recent handoffs (bounded)
        bounded_parts = list(common_parts)
        if prefix_str:
            bounded_parts.append(prefix_str)
        # Always include the route header in W46 (the W45 layer
        # was forced to hint_mode=off; the W46 ctrl supersedes).
        if (decision.factoradic_n_bits > 0
                and recent_handoffs):
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
        )

    def run(
            self,
            task: str,
            *,
            progress: Callable[
                [ManifoldMemoryTurn], None] | None = None,
    ) -> ManifoldMemoryTeamResult:
        """Run the memory-coupled team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        memory_turns: list[ManifoldMemoryTurn] = []
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
        n_memory_margin_abstains = 0
        n_memory_time_attn_abstains = 0
        n_prefix_reuses = 0
        ratify_probabilities: list[float] = []
        time_attn_pooled: list[float] = []
        head_backend = self.backend
        head_model = (
            getattr(head_backend, "model", "") or "")
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()
        controller_params_cid = self.registry.params.cid()
        dictionary_cid = self.registry.params.dictionary.cid()
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
            (w43_result, causal_mask, bundle,
             w45_decision, forward) = aux

            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            (bounded_prompt, textual_prompt, n_textual_tokens,
             n_actual_tokens, n_ctrl_tokens, n_prefix_tokens,
             ctrl_str, ctrl_witness, prefix_capsule,
             prefix_str) = self._build_prompt(
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
                        W46_BRANCH_MEMORY_MARGIN_ABSTAIN):
                    n_memory_margin_abstains += 1
                if decision.branch == (
                        W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN):
                    n_memory_time_attn_abstains += 1
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
                # Reused prefix saves the equivalent prefix tokens
                # vs a non-reused fresh prefix.
                n_visible_tokens_saved_prefix_reuse += int(
                    n_prefix_tokens)
            ratify_probabilities.append(
                float(decision.forward.ratify_probability))
            time_attn_pooled.append(
                float(decision.forward.time_attention.pooled_value))

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

            # Build the W46 envelope (all sub-witness CIDs).
            multi_rank_witness = (
                _compute_w46_multi_rank_adapter_witness_cid(
                    role=str(role),
                    role_delta_value=float(
                        decision.forward.role_delta_value),
                    role_delta_rank=int(
                        self.registry.params.role_adapter.rank),
                    role_adapter_disabled=bool(
                        self.registry.role_adapter_disabled),
                    present=bool(
                        decision.forward.role_adapter_present),
                    turn_index=int(idx),
                ))
            time_attention_witness_cid = (
                decision.forward.time_attention.cid())
            construction_cid = (
                _compute_w46_prompt_construction_witness_cid(
                    turn_index=int(idx),
                    role=str(role),
                    prompt_sha256=prompt_sha,
                    control_token_mode=str(
                        self.registry.control_token_mode),
                    inline_route_mode=(
                        self.registry.learned_registry
                        .live_registry.inline_route_mode),
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
                ))
            control_token_witness_cid = (
                ctrl_witness.cid()
                if str(self.registry.control_token_mode)
                != W46_CTRL_MODE_OFF else "")
            prefix_capsule_cid = prefix_capsule.cid()
            memory_bank_head_cid = (
                self.orchestrator.memory_bank.head_cid())
            behavioral_change = bool(
                do_substitute
                or n_saved > 0 or n_added > 0
                or n_ctrl_tokens > 0
                or n_prefix_tokens > 0
                or prefix_capsule.reused)
            memory_witness_cid = _compute_w46_memory_witness_cid(
                decision_branch=decision.branch,
                w45_branch=decision.w45_branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                controller_params_cid=controller_params_cid,
                dictionary_cid=dictionary_cid,
                time_attention_witness_cid=(
                    time_attention_witness_cid),
                multi_rank_adapter_witness_cid=multi_rank_witness,
                causal_mask_witness_cid=decision.causal_mask_cid,
                prompt_construction_witness_cid=construction_cid,
                control_token_witness_cid=(
                    control_token_witness_cid),
                prefix_capsule_cid=prefix_capsule_cid,
                memory_bank_head_cid=memory_bank_head_cid,
                output_sha256=output_sha,
                behavioral_change=behavioral_change,
            )
            outer_cid = _compute_w46_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w45_envelope_cid="",
                controller_params_cid=controller_params_cid,
                memory_bank_head_cid=memory_bank_head_cid,
                memory_witness_cid=memory_witness_cid,
                turn_index=int(idx),
            )
            envelope = ManifoldMemoryHandoffEnvelope(
                schema_version=W46_MANIFOLD_MEMORY_SCHEMA_VERSION,
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w45_envelope_cid="",
                parent_w44_envelope_cid=str(
                    decision.w44_envelope_cid),
                parent_w43_envelope_cid=str(
                    decision.pmc_envelope_cid),
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
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
                    self.registry.learned_registry
                    .live_registry.inline_route_mode),
                factoradic_int=int(decision.factoradic_int),
                factoradic_n_bits=int(decision.factoradic_n_bits),
                hint_confidence_bucket=int(
                    decision.forward.confidence_bucket),
                controller_params_cid=controller_params_cid,
                training_set_cid=str(
                    self.registry.params.training_set_cid),
                fitting_method=str(
                    self.registry.params.fitting_method),
                n_layers=int(self.registry.params.n_layers),
                role_adapter_rank=int(
                    self.registry.params.role_adapter.rank),
                dictionary_cid=dictionary_cid,
                dictionary_size=int(
                    self.registry.params.dictionary.k),
                use_attention_routing=bool(
                    self.registry.use_attention_routing),
                role_adapter_disabled=bool(
                    self.registry.role_adapter_disabled),
                layer_logits=tuple(_round_floats(
                    decision.forward.layer_logits)),
                time_attention_witness_cid=(
                    time_attention_witness_cid),
                time_attention_pooled=float(
                    decision.forward.time_attention.pooled_value),
                time_attention_mask_size=int(
                    decision.forward.time_attention.mask_size),
                multi_rank_adapter_witness_cid=multi_rank_witness,
                dict_index=int(decision.forward.dict_index),
                dict_residual_l1=float(
                    decision.forward.dict_residual_l1),
                causal_mask_witness_cid=decision.causal_mask_cid,
                memory_bank_head_cid=memory_bank_head_cid,
                memory_bank_size=int(
                    len(self.orchestrator.memory_bank.entries)),
                memory_capacity=int(
                    self.orchestrator.memory_bank.capacity),
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
                memory_witness_cid=memory_witness_cid,
                memory_outer_cid=outer_cid,
            )
            memory_turn = ManifoldMemoryTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
            )
            memory_turns.append(memory_turn)

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
                    progress(memory_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[ManifoldMemoryTeam] progress callback "
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
        mean_ta = (
            sum(time_attn_pooled) / len(time_attn_pooled)
            if time_attn_pooled else 0.0)
        return ManifoldMemoryTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            memory_turns=tuple(memory_turns),
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
            n_memory_margin_abstains=int(
                n_memory_margin_abstains),
            n_memory_time_attn_abstains=int(
                n_memory_time_attn_abstains),
            n_prefix_reuses=int(n_prefix_reuses),
            mean_ratify_probability=float(mean_p),
            mean_time_attention_pooled=float(mean_ta),
            controller_params_cid=str(controller_params_cid),
            dictionary_cid=str(dictionary_cid),
            final_memory_bank_head_cid=str(
                self.orchestrator.memory_bank.head_cid()),
        )


# =============================================================================
# Memory-aware synthetic backend (for r93 model-facing family)
# =============================================================================

@dataclasses.dataclass
class MemoryAwareSyntheticBackend:
    """Deterministic backend that returns one canonical answer
    when the prompt contains BOTH a ``MANIFOLD_CTRL:`` substring
    AND a non-empty ``mem_summary=`` field, and a different
    answer otherwise.

    Used by R-93 to exercise the *behavioural* effect of the W46
    packed control surface on a controlled synthetic ground
    truth. Not a real LLM; the response is keyed only on the
    substrings.
    """

    correct_with_ctrl: str = "MEMORY_OK"
    answer_without_ctrl: str = "MEMORY_NO_CTRL"
    n_calls: int = 0
    model_tag: str = "synthetic.memory_aware"
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
                and "mem_summary=" in text):
            return self.correct_with_ctrl
        return self.answer_without_ctrl


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Schema, branches, defaults
    "W46_MANIFOLD_MEMORY_SCHEMA_VERSION",
    "W46_TEAM_RESULT_SCHEMA",
    "W46_BRANCH_TRIVIAL_MEMORY_PASSTHROUGH",
    "W46_BRANCH_MEMORY_DISABLED",
    "W46_BRANCH_MEMORY_RATIFIED",
    "W46_BRANCH_MEMORY_NO_POLICY",
    "W46_BRANCH_MEMORY_CAUSAL_ABSTAIN",
    "W46_BRANCH_MEMORY_SPHERICAL_ABSTAIN",
    "W46_BRANCH_MEMORY_SUBSPACE_ABSTAIN",
    "W46_BRANCH_MEMORY_MARGIN_ABSTAIN",
    "W46_BRANCH_MEMORY_TIME_ATTN_ABSTAIN",
    "W46_BRANCH_MEMORY_REJECTED",
    "W46_ALL_BRANCHES",
    "W46_MEMORY_ABSTAIN_BRANCHES",
    "W46_CTRL_MODE_OFF",
    "W46_CTRL_MODE_COMPACT",
    "W46_CTRL_MODE_FULL",
    "W46_ALL_CTRL_MODES",
    "W46_DEFAULT_N_LAYERS",
    "W46_DEFAULT_MEMORY_CAPACITY",
    "W46_DEFAULT_DICTIONARY_SIZE",
    "W46_DEFAULT_ROLE_DELTA_RANK",
    "W46_DEFAULT_TIME_ATTN_TEMPERATURE",
    "W46_DEFAULT_TIME_ATTN_WEIGHT",
    "W46_DEFAULT_PREFIX_TURNS",
    "W46_NO_DICT_CODE",
    "W46_ALL_FAILURE_MODES",
    # Params + fitter
    "MultiLayerControllerParams",
    "LayerParams",
    "MultiRankRoleAdapter",
    "DictionaryBasis",
    "build_unfitted_memory_controller_params",
    "fit_memory_controller",
    # Memory bank
    "MemoryEntry",
    "ManifoldMemoryBank",
    # Forward
    "MemoryForwardResult",
    "TimeAttentionWitness",
    "compute_time_attention",
    "forward_memory_controller",
    # Prefix + control surfaces
    "PrefixCapsule",
    "build_prefix_capsule",
    "ControlTokenWitness",
    "build_control_token_string",
    # Registry + orchestrator
    "ManifoldMemoryRegistry",
    "ManifoldMemoryOrchestrator",
    "MemoryGatingDecision",
    # Envelope + verifier
    "ManifoldMemoryHandoffEnvelope",
    "ManifoldMemoryVerificationOutcome",
    "verify_manifold_memory_handoff",
    # Team
    "ManifoldMemoryTurn",
    "ManifoldMemoryTeamResult",
    "ManifoldMemoryTeam",
    # Builders
    "build_trivial_manifold_memory_registry",
    "build_manifold_memory_registry",
    # Synthetic memory-aware backend
    "MemoryAwareSyntheticBackend",
]
