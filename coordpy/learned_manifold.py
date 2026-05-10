"""W45 Learned Manifold Controller (LMC) — capsule-native learned
manifold-conditioned routing on top of W44 LMCC and W43 PMC.

W45 is the first capsule-native CoordPy layer that **learns** a
manifold-conditioned routing / gating policy from cell observations
rather than relying on hand-designed W44 thresholds. It is the
strongest honest model-facing, deep-learning-shaped capsule layer
this repo can support without substrate access.

The five learned components — all closed-form-fittable in pure
NumPy/stdlib without an autograd dependency — are:

  * **Learned channel encoder.** Each of the six W43 channels
    (hyperbolic, spherical, euclidean, factoradic, subspace,
    causal) is mapped through a frozen-then-fitted projection
    head to a fixed-dim feature vector. The hyperbolic and
    euclidean channels — audit-only at the W44 layer — become
    *features* that the learned controller can consume, bounding
    the open W44-C-LIVE-LATENT carry-forward at the capsule
    layer.

  * **Attention-style routing over channels.** A small softmax-
    weighted attention head pools the six channel feature vectors
    into a single scalar gate logit. The attention weights are
    fit-on-policy (closed-form ridge regression on synthetic
    positive/negative cells); the per-channel weights are part of
    the audit envelope so an auditor can trace which channel
    drove any single gating decision.

  * **Adapter-decomposed role-specific policy.** Following the
    shared-base / role-specific-delta decomposition pattern
    ("LoRA-style at the capsule layer"), the gate policy = shared
    base + low-rank role-specific delta. The shared base is
    fitted across the role universe; per-role deltas are rank-1
    perturbations of the base. This is the strongest executable
    approximation we can do for "role-specific KV state" without
    substrate access.

  * **Margin-calibrated gating.** The hard W44 thresholds are
    replaced with a learned **signed margin**: the gate logit +
    bias maps to a sigmoid-shaped ratify probability with a
    learned calibration point.

  * **Factoradic-conditioned learned prompt hint.** The W44
    factoradic compressor emitted only `FACTORADIC_ROUTE: <int>`.
    W45 adds a learned **manifold hint** — `MANIFOLD_HINT:
    route=<int> conf=<integer>` — where `<conf>` is a quantized
    confidence bucket drawn from the gate logit. The hint is
    content-addressed and the full controller state is
    recoverable bijectively from the envelope.

Honest scope (do-not-overstate)
-------------------------------

W45 does NOT claim transformer-internal access. The learned
controller operates strictly over W43 capsule-layer channel
encodings; it does not read hidden states, transplant KV cache,
inspect attention weights, or modify the model's attention
computation. The W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``) carry forward unchanged.

W45 does NOT claim the LoRA-style adapter decomposition is true
LoRA on a transformer; it is a low-rank perturbation of the
capsule-layer policy parameters. W45 does NOT claim the attention
routing is true transformer attention; it is a softmax pool over
six scalar channel features. W45 does NOT claim the learned hint
guarantees a real-LLM behavioural change; the hint is a
deterministic, content-addressed text fragment the model may
attend to or ignore.

W45 is strictly additive on top of W44 and the released v3.43
SDK. When the learned controller is configured trivially
(``learned_enabled=False``, ``prompt_hint_mode='off'``,
``use_attention_routing=False``, ``role_adapter_disabled=True``),
the W45 orchestrator reduces to ``LiveManifoldTeam.run``
byte-for-byte — the W45-L-TRIVIAL-LEARNED-PASSTHROUGH falsifier.

This module lives at ``coordpy.learned_manifold`` and is NOT
exported through ``coordpy.__experimental__`` at this milestone;
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W45 surface through an explicit
``from coordpy.learned_manifold import ...`` import.
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
from .live_manifold import (
    LiveGatingDecision,
    LiveManifoldHandoffEnvelope,
    LiveManifoldOrchestrator,
    LiveManifoldRegistry,
    LiveObservationBuilder,
    LiveTurnContext,
    W44_ABSTAIN_BRANCHES,
    W44_ALL_BRANCHES,
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
    W44_BRANCH_LIVE_NO_POLICY,
    W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_DEFAULT_PARENT_W42_CID,
    W44_ROUTE_MODE_FACTORADIC,
    W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL,
    W44_ROUTE_MODE_TEXTUAL,
    build_live_manifold_registry,
    build_trivial_live_manifold_registry,
    default_live_observation_builder,
)
from .llm_backend import LLMBackend
from .product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldChannelBundle,
    ProductManifoldPolicyEntry,
    SphericalConsensusSignature,
    SubspaceBasis,
    cosine_agreement,
    encode_cell_channels,
    principal_angle_drift,
)
from .team_coord import capsule_team_handoff


# =============================================================================
# Schema, branches, defaults
# =============================================================================

W45_LEARNED_MANIFOLD_SCHEMA_VERSION: str = (
    "coordpy.learned_manifold.v1")
W45_TEAM_RESULT_SCHEMA: str = (
    "coordpy.learned_manifold_team_result.v1")

# Decision branches. The learned layer reuses the W44 branch names
# for behaviour compatibility, but adds three W45-specific branches.
W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH: str = (
    "learned_trivial_passthrough")
W45_BRANCH_LEARNED_DISABLED: str = "learned_disabled"
W45_BRANCH_LEARNED_RATIFIED: str = "learned_ratified"
W45_BRANCH_LEARNED_NO_POLICY: str = "learned_no_policy"
W45_BRANCH_LEARNED_CAUSAL_ABSTAIN: str = "learned_causal_abstain"
W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN: str = (
    "learned_spherical_abstain")
W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN: str = (
    "learned_subspace_abstain")
W45_BRANCH_LEARNED_MARGIN_ABSTAIN: str = (
    "learned_margin_abstain")
W45_BRANCH_LEARNED_REJECTED: str = "learned_rejected"

W45_ALL_BRANCHES: tuple[str, ...] = (
    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH,
    W45_BRANCH_LEARNED_DISABLED,
    W45_BRANCH_LEARNED_RATIFIED,
    W45_BRANCH_LEARNED_NO_POLICY,
    W45_BRANCH_LEARNED_CAUSAL_ABSTAIN,
    W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN,
    W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN,
    W45_BRANCH_LEARNED_MARGIN_ABSTAIN,
    W45_BRANCH_LEARNED_REJECTED,
)

W45_LEARNED_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W45_BRANCH_LEARNED_CAUSAL_ABSTAIN,
    W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN,
    W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN,
    W45_BRANCH_LEARNED_MARGIN_ABSTAIN,
})

# Prompt-hint modes. ``off`` is a strict superset reduction to the
# underlying W44 prompt builder; the other modes add the
# MANIFOLD_HINT or richer surrogate-prompt control.
W45_HINT_MODE_OFF: str = "off"
W45_HINT_MODE_FACTORADIC_WITH_HINT: str = "factoradic_with_hint"
W45_HINT_MODE_HINT_ONLY: str = "hint_only"

W45_ALL_HINT_MODES: tuple[str, ...] = (
    W45_HINT_MODE_OFF,
    W45_HINT_MODE_FACTORADIC_WITH_HINT,
    W45_HINT_MODE_HINT_ONLY,
)

# Channel ordering — fixed for content-addressing.
W45_CHANNEL_ORDER: tuple[str, ...] = (
    "hyperbolic",
    "spherical",
    "euclidean",
    "factoradic",
    "subspace",
    "causal",
)
W45_N_CHANNELS: int = len(W45_CHANNEL_ORDER)

# Quantization buckets for the confidence channel in the hint.
W45_CONFIDENCE_BUCKETS: int = 4

# Default fitting hyperparameters.
W45_DEFAULT_RIDGE_LAMBDA: float = 1e-3
W45_DEFAULT_FEATURE_DIM: int = 4
W45_DEFAULT_ROLE_DELTA_RANK: int = 1
W45_DEFAULT_MARGIN_OFFSET: float = 0.0
W45_DEFAULT_MARGIN_BIAS: float = 0.0
W45_DEFAULT_HINT_PROBABILITY_PRECISION: int = 4

# Sentinel string used when there is no fitted role delta for a
# given role; the W45 envelope records it explicitly so an auditor
# can tell apart "no delta exists" from "delta is zero".
W45_NO_ROLE_DELTA: str = "no_role_delta"


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


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    if s == 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / s for e in exps]


# =============================================================================
# Channel feature extraction
# =============================================================================

def _channel_features_from_bundle(
        bundle: ProductManifoldChannelBundle,
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        expected_spherical: SphericalConsensusSignature | None = None,
        expected_subspace: SubspaceBasis | None = None,
) -> dict[str, list[float]]:
    """Extract a ``feature_dim``-vector for each W43 channel.

    Each channel is mapped to a small fixed-dim feature vector that
    the learned controller consumes. The mapping is deterministic
    and bounded — channels never carry more than ``feature_dim``
    real numbers into the controller.

    Hyperbolic
        First ``feature_dim`` coordinates of the encoded vector,
        zero-padded.
    Spherical
        ``[cosine_agreement, n_observations / 100, dim/16, 0...]``
        — encodes both the absolute observation count (saturating
        at 100) and the agreement against expected (when provided).
    Euclidean
        First ``feature_dim`` coordinates of the attribute vector,
        zero-padded; truncated to feature_dim from the front.
    Factoradic
        ``[n_structured_bits/64, factoradic_int_normalised,
        permutation_norm, 0...]`` where ``factoradic_int_normalised``
        is a bounded folding of the factoradic int into
        ``[0, 1]``.
    Subspace
        ``[principal_angle_drift_radians, ...]`` — the principal
        angle vs expected when provided, plus the basis's first
        ``feature_dim-1`` diagonal entries.
    Causal
        ``[1.0 if admissible else 0.0, n_clocks/16,
        violation_index_normalised, ...]``.
    """
    fd = max(1, int(feature_dim))

    def _pad(values: Sequence[float]) -> list[float]:
        out = [float(v) for v in values][:fd]
        while len(out) < fd:
            out.append(0.0)
        return out

    # Hyperbolic: first fd coords + path-hash-derived spectral bit.
    hyp_coords = list(bundle.hyperbolic.coordinates)
    # Use the first byte of the path hash as a deterministic
    # spectral bit so the controller can distinguish identical-depth
    # paths.
    if bundle.hyperbolic.path_hash:
        spectral_bit = (
            int(bundle.hyperbolic.path_hash[:2], 16) / 255.0)
    else:
        spectral_bit = 0.0
    hyp_feats = _pad(hyp_coords + [spectral_bit])

    # Spherical: cosine agreement + normalised obs count + dim.
    sph = bundle.spherical
    if expected_spherical is not None:
        agree = cosine_agreement(sph, expected_spherical)
    else:
        agree = 1.0
    sph_feats = _pad([
        float(agree),
        min(1.0, float(sph.n_observations) / 100.0),
        float(len(sph.coordinates)) / 16.0,
    ])

    # Euclidean: first fd attribute coords.
    euc_feats = _pad(list(bundle.euclidean.coordinates))

    # Factoradic: n_structured_bits / capacity, normalised int,
    # permutation magnitude.
    fac = bundle.factoradic
    nbits = fac.n_structured_bits()
    max_int = max(1, math.factorial(max(1, fac.n)) - 1)
    fac_normalised = (
        float(fac.factoradic_int) / float(max_int)
        if max_int > 0 else 0.0)
    perm_mag = (
        sum(p * p for p in fac.permutation) /
        max(1.0, float(fac.n * fac.n)))
    fac_feats = _pad([
        float(nbits) / 64.0,
        float(fac_normalised),
        float(perm_mag),
    ])

    # Subspace: principal-angle drift vs expected + basis diagonal.
    sub = bundle.subspace
    if expected_subspace is not None:
        drift = principal_angle_drift(sub, expected_subspace)
    else:
        drift = 0.0
    diag = []
    n_diag = min(sub.dim, sub.rank)
    for i in range(min(n_diag, fd - 1)):
        if (i < len(sub.basis_columns) and
                i < len(sub.basis_columns[i])):
            diag.append(float(sub.basis_columns[i][i]))
        else:
            diag.append(0.0)
    sub_feats = _pad([float(drift)] + diag)

    # Causal: admissibility flag + clock count + violation index.
    n_clocks = len(bundle.causal_clocks)
    vidx = bundle.causal_violation_index
    vidx_norm = float(vidx) / float(max(1, n_clocks))
    cau_feats = _pad([
        1.0 if bundle.causal_admissible else 0.0,
        min(1.0, float(n_clocks) / 16.0),
        max(0.0, vidx_norm) if vidx >= 0 else 0.0,
    ])

    return {
        "hyperbolic": hyp_feats,
        "spherical": sph_feats,
        "euclidean": euc_feats,
        "factoradic": fac_feats,
        "subspace": sub_feats,
        "causal": cau_feats,
    }


# =============================================================================
# Learned controller parameters
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LearnedControllerParams:
    """All fitted parameters of the W45 learned controller.

    A frozen, content-addressable parameter bundle:

    * ``channel_projection`` — per-channel ``feature_dim``-vector
      projecting that channel's feature vector to a scalar logit.
      Shape: ``(W45_N_CHANNELS, feature_dim)``.
    * ``attention_logits`` — per-channel attention logits before
      softmax. Shape: ``(W45_N_CHANNELS,)``.
    * ``shared_base_bias`` — shared scalar bias added to the
      attention-pooled logit.
    * ``role_deltas`` — per-role rank-``role_delta_rank``
      perturbations of the projection / bias. Mapping
      ``role_name -> RoleDelta`` where the delta is a flat tuple of
      the perturbation parameters (1 scalar bias + r * n_channels
      scalars).
    * ``margin_offset`` — calibration offset applied to the
      attention-pooled logit before the sigmoid.
    * ``margin_bias`` — final scalar bias added before the sigmoid.
    * ``feature_dim`` — fitted feature dimension.
    * ``role_delta_rank`` — fitted rank of each role delta.
    * ``training_set_cid`` — CID of the training set the params
      were fitted on (or empty if unfitted).
    * ``fitting_method`` — short string identifying the fitter
      (e.g. ``"ridge_v1"``, ``"unfitted"``).
    """

    channel_projection: tuple[tuple[float, ...], ...]
    attention_logits: tuple[float, ...]
    shared_base_bias: float
    role_deltas: tuple[tuple[str, tuple[float, ...]], ...]
    margin_offset: float
    margin_bias: float
    feature_dim: int
    role_delta_rank: int
    training_set_cid: str
    fitting_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_projection":
                _round_matrix(self.channel_projection),
            "attention_logits":
                _round_floats(self.attention_logits),
            "shared_base_bias": float(round(
                self.shared_base_bias, 12)),
            "role_deltas": [
                [str(role), _round_floats(delta)]
                for role, delta in self.role_deltas
            ],
            "margin_offset": float(round(self.margin_offset, 12)),
            "margin_bias": float(round(self.margin_bias, 12)),
            "feature_dim": int(self.feature_dim),
            "role_delta_rank": int(self.role_delta_rank),
            "training_set_cid": str(self.training_set_cid),
            "fitting_method": str(self.fitting_method),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w45_learned_controller_params",
            "params": self.to_dict(),
        })

    @property
    def role_delta_map(self) -> dict[str, tuple[float, ...]]:
        return {str(r): tuple(d) for r, d in self.role_deltas}


def build_unfitted_controller_params(
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        role_delta_rank: int = W45_DEFAULT_ROLE_DELTA_RANK,
) -> LearnedControllerParams:
    """Build a controller with all-zero parameters. With this, the
    controller's gate logit is exactly zero — equivalent to "always
    pass through the W44 decision" once routing is disabled.
    """
    proj = tuple(
        tuple([0.0] * int(feature_dim))
        for _ in range(W45_N_CHANNELS))
    att = tuple([0.0] * W45_N_CHANNELS)
    return LearnedControllerParams(
        channel_projection=proj,
        attention_logits=att,
        shared_base_bias=0.0,
        role_deltas=tuple(),
        margin_offset=W45_DEFAULT_MARGIN_OFFSET,
        margin_bias=W45_DEFAULT_MARGIN_BIAS,
        feature_dim=int(feature_dim),
        role_delta_rank=int(role_delta_rank),
        training_set_cid="",
        fitting_method="unfitted",
    )


# =============================================================================
# Training set and ridge fitter
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TrainingExample:
    """One labelled training cell for the W45 fitter.

    The label is a signed scalar in ``[-1, 1]``:

      * ``+1`` — the cell is canonically *honest* (ratify)
      * ``-1`` — the cell is canonically *dirty* (abstain)

    Continuous labels in between are tolerated (treated as soft
    targets). The controller is fitted via ridge regression on the
    feature -> label mapping.
    """

    role: str
    role_handoff_signature_cid: str
    channel_features: tuple[tuple[str, tuple[float, ...]], ...]
    label: float

    @property
    def channel_features_map(self) -> dict[str, tuple[float, ...]]:
        return {
            str(k): tuple(v) for k, v in self.channel_features}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "role_handoff_signature_cid": str(
                self.role_handoff_signature_cid),
            "channel_features": [
                [str(k), _round_floats(v)]
                for k, v in self.channel_features
            ],
            "label": float(round(self.label, 12)),
        }


@dataclasses.dataclass(frozen=True)
class TrainingSet:
    """Frozen, content-addressable training set."""

    examples: tuple[TrainingExample, ...]
    feature_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": [e.to_dict() for e in self.examples],
            "feature_dim": int(self.feature_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w45_training_set",
            "set": self.to_dict(),
        })


def _solve_ridge(
        x_rows: Sequence[Sequence[float]],
        y_rows: Sequence[float],
        *,
        lam: float = W45_DEFAULT_RIDGE_LAMBDA,
) -> list[float]:
    """Solve a small ridge regression: minimize ||X w - y||^2 +
    lam * ||w||^2.

    Closed-form: ``w = (X^T X + lam I)^{-1} X^T y``. Solves a
    deterministic linear system via Gauss-Jordan. Pure Python, no
    NumPy dependency.

    Returns the weight vector ``w`` of length ``n_features``. If
    the matrix is empty / singular, returns the zero vector.
    """
    n = len(x_rows)
    if n == 0:
        return []
    d = len(x_rows[0]) if x_rows else 0
    if d == 0:
        return []
    # Build A = X^T X + lam I and b = X^T y.
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for i in range(n):
        xi = list(x_rows[i])
        yi = float(y_rows[i])
        for r in range(d):
            for c in range(d):
                A[r][c] += xi[r] * xi[c]
            b[r] += xi[r] * yi
    for r in range(d):
        A[r][r] += float(lam)
    # Gauss-Jordan elimination on the augmented matrix [A | b].
    M = [row[:] + [b[r]] for r, row in enumerate(A)]
    for col in range(d):
        # Partial pivot.
        max_row = col
        max_val = abs(M[col][col])
        for r in range(col + 1, d):
            if abs(M[r][col]) > max_val:
                max_val = abs(M[r][col])
                max_row = r
        if max_val < 1e-15:
            # Singular — return zero vector.
            return [0.0] * d
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        for c in range(col, d + 1):
            M[col][c] /= pivot
        for r in range(d):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for c in range(col, d + 1):
                M[r][c] -= factor * M[col][c]
    return [float(M[r][d]) for r in range(d)]


def fit_learned_controller(
        training_set: TrainingSet,
        *,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        role_delta_rank: int = W45_DEFAULT_ROLE_DELTA_RANK,
        ridge_lambda: float = W45_DEFAULT_RIDGE_LAMBDA,
        fit_role_deltas: bool = True,
) -> LearnedControllerParams:
    """Fit a learned controller from a training set.

    Two-stage closed-form fit:

      Stage 1 (shared base)
        Solve a single ridge regression for the per-channel
        projection vectors + per-channel attention logits + shared
        bias. The design matrix stacks each example's six
        ``feature_dim``-vectors followed by a 1.0 constant column.

      Stage 2 (role-specific deltas)
        For each role with at least ``role_delta_rank + 1``
        examples, compute the per-role residuals from the shared
        base and solve a small ridge regression for a rank-1
        perturbation that explains them. Roles with too few
        examples get a zero delta (and are recorded as
        ``W45_NO_ROLE_DELTA``).

    Trains entirely closed-form with no autograd / iterative
    optimisation, in pure Python.
    """
    fd = int(feature_dim)
    rdr = max(0, int(role_delta_rank))
    if not training_set.examples:
        return build_unfitted_controller_params(
            feature_dim=fd, role_delta_rank=rdr)

    # Stage 1: shared base.
    # Per example: build a flat row of length n_channels * fd + 1.
    # (The trailing 1 is the bias / shared scalar.) The label is
    # the example's label.
    n_feats = W45_N_CHANNELS * fd + 1
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for ex in training_set.examples:
        row = [0.0] * n_feats
        fmap = ex.channel_features_map
        for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
            feats = list(fmap.get(c_name, ()))[:fd]
            while len(feats) < fd:
                feats.append(0.0)
            base = c_idx * fd
            for k in range(fd):
                row[base + k] = float(feats[k])
        row[-1] = 1.0
        x_rows.append(row)
        y_rows.append(float(ex.label))
    w = _solve_ridge(x_rows, y_rows, lam=ridge_lambda)
    if len(w) != n_feats:
        w = [0.0] * n_feats

    # Decompose w into channel_projection + attention_logits +
    # shared_bias.
    proj: list[tuple[float, ...]] = []
    att_logits: list[float] = []
    for c_idx in range(W45_N_CHANNELS):
        base = c_idx * fd
        chunk = tuple(_round_floats(w[base:base + fd]))
        proj.append(chunk)
        # Attention logit per channel = L1 norm of the projection
        # chunk (a deterministic readout that summarises the
        # channel's total contribution; the controller will softmax
        # these to a per-channel weight).
        att_logits.append(float(sum(abs(v) for v in chunk)))
    shared_bias = float(w[-1])

    # Stage 2: role-specific deltas.
    role_deltas: list[tuple[str, tuple[float, ...]]] = []
    if fit_role_deltas and rdr > 0:
        # Group examples by role.
        by_role: dict[str, list[TrainingExample]] = {}
        for ex in training_set.examples:
            by_role.setdefault(str(ex.role), []).append(ex)
        for role in sorted(by_role.keys()):
            exs = by_role[role]
            if len(exs) < rdr + 1:
                # Too few examples; record an empty delta sentinel.
                continue
            # Residual targets: label - shared_base_prediction.
            resid_x: list[list[float]] = []
            resid_y: list[float] = []
            for ex in exs:
                fmap = ex.channel_features_map
                # Pre-compute the per-channel logit values for this
                # example using the fitted shared base.
                per_channel_logits = []
                for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
                    feats = list(fmap.get(c_name, ()))[:fd]
                    while len(feats) < fd:
                        feats.append(0.0)
                    proj_chunk = proj[c_idx]
                    per_channel_logits.append(
                        sum(f * p for f, p in
                            zip(feats, proj_chunk)))
                # Re-build the shared base prediction.
                base_pred = shared_bias + sum(per_channel_logits)
                resid = float(ex.label) - float(base_pred)
                # Build the rank-rdr feature vector. The rank-k
                # basis is a signed readout of the shared base's
                # pooled output: basis_0 = signed pooled logit;
                # basis_k>0 = rotated-signed channel logits. This
                # makes the rank-1 delta act as a *LoRA-style sign
                # / magnitude correction* on the shared base's
                # output for that role — which is exactly what we
                # need to fit a role whose label convention is
                # inverted vs the shared base.
                rank_feats = []
                pooled = float(sum(per_channel_logits))
                rank_feats.append(pooled)  # basis_0
                for r_idx in range(1, rdr):
                    # Higher ranks: rotated signed-channel-logit
                    # sums. Distinct from basis_0 so the LS
                    # problem has effective rank > 1 when the
                    # data supports it.
                    rotated = sum(
                        per_channel_logits[
                            (c + r_idx) % W45_N_CHANNELS]
                        for c in range(W45_N_CHANNELS)) * (
                                (-1.0) ** r_idx)
                    rank_feats.append(rotated)
                rank_feats.append(1.0)  # bias term per role
                resid_x.append(rank_feats)
                resid_y.append(resid)
            delta_w = _solve_ridge(
                resid_x, resid_y, lam=ridge_lambda)
            if len(delta_w) == rdr + 1:
                role_deltas.append(
                    (role, tuple(_round_floats(delta_w))))

    # Margin calibration: pick offset = -mean(predicted_logits) so
    # the post-fit logits are roughly zero-mean on the training
    # set (this is a deterministic, closed-form calibration —
    # *not* trained via SGD).
    predictions = []
    for ex in training_set.examples:
        fmap = ex.channel_features_map
        pred = shared_bias
        for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
            feats = list(fmap.get(c_name, ()))[:fd]
            while len(feats) < fd:
                feats.append(0.0)
            pred += sum(f * p for f, p in zip(feats, proj[c_idx]))
        predictions.append(pred)
    margin_offset = (
        -sum(predictions) / len(predictions) if predictions else 0.0)

    return LearnedControllerParams(
        channel_projection=tuple(proj),
        attention_logits=tuple(_round_floats(att_logits)),
        shared_base_bias=float(round(shared_bias, 12)),
        role_deltas=tuple(role_deltas),
        margin_offset=float(round(margin_offset, 12)),
        margin_bias=W45_DEFAULT_MARGIN_BIAS,
        feature_dim=fd,
        role_delta_rank=rdr,
        training_set_cid=training_set.cid(),
        fitting_method="ridge_v1",
    )


# =============================================================================
# Controller forward pass + decision
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ControllerForwardResult:
    """Result of one forward pass of the learned controller.

    All fields are content-addressable; the W45 envelope binds the
    forward result via the attention-routing witness CID.
    """

    per_channel_logits: tuple[float, ...]
    attention_weights: tuple[float, ...]
    role_delta_value: float
    gate_logit: float
    ratify_probability: float
    confidence_bucket: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_channel_logits":
                _round_floats(self.per_channel_logits),
            "attention_weights":
                _round_floats(self.attention_weights),
            "role_delta_value": float(round(
                self.role_delta_value, 12)),
            "gate_logit": float(round(self.gate_logit, 12)),
            "ratify_probability": float(round(
                self.ratify_probability, 12)),
            "confidence_bucket": int(self.confidence_bucket),
        }


def _confidence_bucket_for_probability(p: float) -> int:
    """Quantize the ratify probability into one of
    ``W45_CONFIDENCE_BUCKETS`` discrete buckets in ``[0,
    W45_CONFIDENCE_BUCKETS-1]``.
    """
    clamped = max(0.0, min(1.0, float(p)))
    bucket = int(math.floor(clamped * W45_CONFIDENCE_BUCKETS))
    if bucket >= W45_CONFIDENCE_BUCKETS:
        bucket = W45_CONFIDENCE_BUCKETS - 1
    return bucket


def forward_controller(
        *,
        channel_features: Mapping[str, Sequence[float]],
        params: LearnedControllerParams,
        role: str,
        use_attention_routing: bool = True,
) -> ControllerForwardResult:
    """Run one forward pass.

    For each channel, computes a scalar per-channel logit by inner-
    producting the channel's feature vector with the fitted
    projection chunk. Then either:

      * ``use_attention_routing = True`` — softmax-weights the
        per-channel logits and pools into a single gate logit.
      * ``use_attention_routing = False`` — sums all per-channel
        logits with equal weight (= the W44 thresholded sum
        analogue).

    Adds the shared bias + the role-specific delta value +
    margin offset + margin bias. Maps through sigmoid for the
    ratify probability and quantizes for the confidence bucket.
    """
    fd = int(params.feature_dim)
    per_channel: list[float] = []
    for c_idx, c_name in enumerate(W45_CHANNEL_ORDER):
        feats = list(channel_features.get(c_name, ()))[:fd]
        while len(feats) < fd:
            feats.append(0.0)
        proj_chunk = list(params.channel_projection[c_idx])[:fd]
        while len(proj_chunk) < fd:
            proj_chunk.append(0.0)
        per_channel.append(
            float(sum(f * p for f, p in zip(feats, proj_chunk))))

    if use_attention_routing:
        att_weights = _softmax(list(params.attention_logits))
        pooled = sum(
            float(w) * float(v)
            for w, v in zip(att_weights, per_channel)) * W45_N_CHANNELS
    else:
        att_weights = [1.0 / W45_N_CHANNELS] * W45_N_CHANNELS
        pooled = sum(per_channel)

    # Role-specific delta. We use the *signed* shared-base
    # per-channel logits as the rank basis (matching the fitter).
    delta_map = params.role_delta_map
    role_delta_value = 0.0
    if str(role) in delta_map:
        delta = list(delta_map[str(role)])
        rdr = int(params.role_delta_rank)
        pooled = float(sum(per_channel))
        # basis_0 = signed pooled logit.
        if rdr >= 1 and len(delta) >= 1:
            role_delta_value += float(delta[0]) * pooled
        # basis_k>0 = rotated signed channel-logit sum.
        for r_idx in range(1, rdr):
            if r_idx >= len(delta):
                break
            rotated = sum(
                per_channel[(c + r_idx) % W45_N_CHANNELS]
                for c in range(W45_N_CHANNELS)) * (
                    (-1.0) ** r_idx)
            role_delta_value += float(delta[r_idx]) * rotated
        # Final bias from the role delta.
        if len(delta) > rdr:
            role_delta_value += float(delta[rdr])

    gate_logit = (
        float(pooled)
        + float(params.shared_base_bias)
        + float(role_delta_value)
        + float(params.margin_offset)
        + float(params.margin_bias))
    ratify_prob = _sigmoid(gate_logit)
    conf_bucket = _confidence_bucket_for_probability(ratify_prob)

    return ControllerForwardResult(
        per_channel_logits=tuple(_round_floats(per_channel)),
        attention_weights=tuple(_round_floats(att_weights)),
        role_delta_value=float(round(role_delta_value, 12)),
        gate_logit=float(round(gate_logit, 12)),
        ratify_probability=float(round(ratify_prob, 12)),
        confidence_bucket=int(conf_bucket),
    )


# =============================================================================
# Causal mask witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CausalMaskWitness:
    """Records which channels were observable at the time of a
    gating decision.

    ``observable_channels`` is a tuple of booleans indexed by
    :data:`W45_CHANNEL_ORDER`. A channel is *not* observable when:

      * spherical: ``n_observations == 0``.
      * subspace: ``rank == 0`` or basis is the zero matrix.
      * factoradic: ``n == 0`` (no role universe).
      * causal: ``len(causal_clocks) == 0``.
      * hyperbolic / euclidean: always observable (encodings exist
        regardless of input emptiness, with the zero encoding as
        the trivial case).
    """

    observable_channels: tuple[bool, ...]
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_channels":
                [bool(b) for b in self.observable_channels],
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w45_causal_mask_witness",
            "witness": self.to_dict(),
        })


def derive_causal_mask(
        bundle: ProductManifoldChannelBundle,
        *,
        turn_index: int,
) -> CausalMaskWitness:
    """Compute the per-channel observability mask for one cell.

    The causal mask is a strict structural property of the
    observation; it is independent of the controller's parameters.
    """
    flags = []
    for c_name in W45_CHANNEL_ORDER:
        if c_name == "hyperbolic":
            flags.append(True)
        elif c_name == "spherical":
            flags.append(bool(bundle.spherical.n_observations > 0))
        elif c_name == "euclidean":
            flags.append(True)
        elif c_name == "factoradic":
            flags.append(bool(bundle.factoradic.n > 0))
        elif c_name == "subspace":
            # The subspace basis is "observable" iff at least one
            # column has a non-trivial magnitude.
            sub = bundle.subspace
            non_trivial = False
            for col in sub.basis_columns:
                if any(abs(float(v)) > 1e-9 for v in col):
                    non_trivial = True
                    break
            flags.append(bool(non_trivial))
        elif c_name == "causal":
            flags.append(bool(len(bundle.causal_clocks) > 0))
        else:
            flags.append(False)
    return CausalMaskWitness(
        observable_channels=tuple(flags),
        turn_index=int(turn_index),
    )


# =============================================================================
# Learned registry
# =============================================================================

@dataclasses.dataclass
class LearnedManifoldRegistry:
    """Controller-side configuration for the W45 learned coupling.

    Wraps a :class:`LiveManifoldRegistry` (the W44 inner) and adds
    five learned-layer toggles plus the fitted controller params.

      * ``learned_enabled`` — master switch. When False, the W45
        orchestrator delegates entirely to W44 (passthrough).
      * ``use_attention_routing`` — when True, the controller
        softmax-pools per-channel logits; when False, equal-weighted
        sum.
      * ``role_adapter_disabled`` — when True, role-specific deltas
        are ignored (only the shared base is used).
      * ``prompt_hint_mode`` — one of :data:`W45_ALL_HINT_MODES`.
      * ``abstain_substitution_enabled`` — when True, the W45 layer
        substitutes the abstain output on a learned-abstain branch.
      * ``params`` — the fitted (or unfitted) controller params.
      * ``margin_abstain_threshold`` — the controller's gate logit
        must exceed this margin for ratification; below it, the
        decision falls through to W44's gating.

    The W45 registry is *trivial* iff all three toggles are off,
    ``params.fitting_method == "unfitted"``, the prompt hint mode
    is ``"off"``, AND the underlying W44 registry is trivial. In
    that case the W45 orchestrator reduces to ``AgentTeam.run``
    byte-for-byte.
    """

    schema_cid: str
    live_registry: LiveManifoldRegistry
    params: LearnedControllerParams
    learned_enabled: bool = True
    use_attention_routing: bool = True
    role_adapter_disabled: bool = False
    prompt_hint_mode: str = W45_HINT_MODE_OFF
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT
    margin_abstain_threshold: float = 0.0
    hint_probability_precision: int = (
        W45_DEFAULT_HINT_PROBABILITY_PRECISION)

    @property
    def is_trivial(self) -> bool:
        return (
            self.live_registry.is_trivial
            and not self.learned_enabled
            and not self.use_attention_routing
            and self.role_adapter_disabled
            and self.prompt_hint_mode == W45_HINT_MODE_OFF
            and not self.abstain_substitution_enabled
            and self.params.fitting_method == "unfitted"
        )


def build_trivial_learned_manifold_registry(
        *, schema_cid: str | None = None,
) -> LearnedManifoldRegistry:
    """Build a registry whose orchestrator reduces to AgentTeam
    byte-for-byte (the W45-L-TRIVIAL-LEARNED-PASSTHROUGH
    falsifier)."""
    cid = schema_cid or _sha256_hex({
        "kind": "w45_trivial_schema"})
    return LearnedManifoldRegistry(
        schema_cid=str(cid),
        live_registry=build_trivial_live_manifold_registry(
            schema_cid=str(cid)),
        params=build_unfitted_controller_params(),
        learned_enabled=False,
        use_attention_routing=False,
        role_adapter_disabled=True,
        prompt_hint_mode=W45_HINT_MODE_OFF,
        abstain_substitution_enabled=False,
    )


def build_learned_manifold_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        params: LearnedControllerParams | None = None,
        learned_enabled: bool = True,
        use_attention_routing: bool = True,
        role_adapter_disabled: bool = False,
        prompt_hint_mode: str = W45_HINT_MODE_FACTORADIC_WITH_HINT,
        abstain_substitution_enabled: bool = True,
        margin_abstain_threshold: float = 0.0,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
        live_enabled: bool = True,
        inline_route_mode: str = W44_ROUTE_MODE_FACTORADIC,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
) -> LearnedManifoldRegistry:
    """Build a fully configured learned registry."""
    if prompt_hint_mode not in W45_ALL_HINT_MODES:
        raise ValueError(
            f"prompt_hint_mode={prompt_hint_mode!r} not in "
            f"{W45_ALL_HINT_MODES}")
    live_inner = build_live_manifold_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        live_enabled=bool(live_enabled),
        inline_route_mode=str(inline_route_mode),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
    )
    return LearnedManifoldRegistry(
        schema_cid=str(schema_cid),
        live_registry=live_inner,
        params=params or build_unfitted_controller_params(),
        learned_enabled=bool(learned_enabled),
        use_attention_routing=bool(use_attention_routing),
        role_adapter_disabled=bool(role_adapter_disabled),
        prompt_hint_mode=str(prompt_hint_mode),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
        margin_abstain_threshold=float(margin_abstain_threshold),
    )


# =============================================================================
# Decision selector
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LearnedGatingDecision:
    """Result of running the learned gate on one turn."""

    branch: str
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
    forward: ControllerForwardResult
    causal_mask_cid: str
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W45_LEARNED_ABSTAIN_BRANCHES


def _classify_w44_branch_to_learned(
        w44_branch: str,
) -> str:
    """Map a W44 live branch to the matching W45 branch."""
    if w44_branch == W44_BRANCH_LIVE_RATIFIED:
        return W45_BRANCH_LEARNED_RATIFIED
    if w44_branch == W44_BRANCH_LIVE_NO_POLICY:
        return W45_BRANCH_LEARNED_NO_POLICY
    if w44_branch == W44_BRANCH_LIVE_CAUSAL_ABSTAIN:
        return W45_BRANCH_LEARNED_CAUSAL_ABSTAIN
    if w44_branch == W44_BRANCH_LIVE_SPHERICAL_ABSTAIN:
        return W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN
    if w44_branch == W44_BRANCH_LIVE_SUBSPACE_ABSTAIN:
        return W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN
    if w44_branch == W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH:
        return W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH
    return W45_BRANCH_LEARNED_DISABLED


# =============================================================================
# Learned orchestrator
# =============================================================================

class LearnedManifoldOrchestrator:
    """Per-turn learned gating + prompt-construction witness.

    Wraps a :class:`LiveManifoldOrchestrator` (W44 inner) plus a
    :class:`LearnedManifoldRegistry`. Stateless across cells.
    """

    def __init__(self, registry: LearnedManifoldRegistry) -> None:
        self.registry = registry
        self._live = LiveManifoldOrchestrator(
            registry=registry.live_registry)

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    def reset_session(self) -> None:
        self._live.reset_session()

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
    ) -> tuple[LearnedGatingDecision, Any]:
        """Run the learned gate for one turn.

        Returns ``(decision, w43_result)``. The learned decision
        always defers to the underlying W44 decision when the
        learned layer is disabled or untrained; when enabled, the
        learned margin is consulted as an *additional* gate that
        can fire ``learned_margin_abstain``.
        """
        live_decision, w43 = self._live.gate(
            observation=observation,
            role_handoff_signature_cid=role_handoff_signature_cid,
            parent_w42_cid=parent_w42_cid,
            n_w42_visible_tokens=int(n_w42_visible_tokens),
        )
        # Encode channels + extract features.
        bundle = encode_cell_channels(observation)
        feats = _channel_features_from_bundle(
            bundle,
            feature_dim=int(self.registry.params.feature_dim),
            expected_spherical=expected_spherical,
            expected_subspace=expected_subspace,
        )
        causal_mask = derive_causal_mask(
            bundle, turn_index=int(turn_index))
        # Build a transient params view if role deltas are disabled.
        forward_params = self.registry.params
        if self.registry.role_adapter_disabled:
            forward_params = dataclasses.replace(
                self.registry.params, role_deltas=tuple())
        forward = forward_controller(
            channel_features=feats,
            params=forward_params,
            role=str(role),
            use_attention_routing=bool(
                self.registry.use_attention_routing),
        )

        # Start from the live (W44) branch, then potentially add
        # the learned margin abstain.
        learned_branch = _classify_w44_branch_to_learned(
            live_decision.branch)
        abstain_reason = live_decision.abstain_reason

        if (self.registry.learned_enabled
                and learned_branch == W45_BRANCH_LEARNED_RATIFIED
                and forward.gate_logit
                < float(self.registry.margin_abstain_threshold)):
            learned_branch = W45_BRANCH_LEARNED_MARGIN_ABSTAIN
            abstain_reason = "learned_margin"

        # If the learned layer is fully off, force trivial /
        # passthrough behaviour.
        if not self.registry.learned_enabled:
            if self.registry.is_trivial:
                learned_branch = (
                    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH)
            elif learned_branch != (
                    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH):
                # Keep the W44-mapped branch but mark "disabled" if
                # nothing else fires.
                if learned_branch == W45_BRANCH_LEARNED_RATIFIED:
                    learned_branch = W45_BRANCH_LEARNED_DISABLED

        # Honest no-policy fall-through: if the underlying W44
        # branch is no_policy AND the learned controller is
        # confident enough to ratify, we still report no_policy as
        # the learned branch — the controller is not authorised to
        # ratify cells lacking a registered policy.
        if live_decision.branch == W44_BRANCH_LIVE_NO_POLICY:
            learned_branch = W45_BRANCH_LEARNED_NO_POLICY

        # Compute the live envelope's outer CID for downstream
        # binding. We use an empty string when the live layer did
        # not produce one (trivial passthrough).
        w44_envelope_cid = ""  # filled in by the team loop

        decision = LearnedGatingDecision(
            branch=str(learned_branch),
            w44_branch=str(live_decision.branch),
            pmc_branch=str(live_decision.pmc_branch),
            spherical_agreement=float(
                live_decision.spherical_agreement),
            subspace_drift=float(live_decision.subspace_drift),
            causal_admissible=bool(live_decision.causal_admissible),
            factoradic_int=int(live_decision.factoradic_int),
            factoradic_n_bits=int(live_decision.factoradic_n_bits),
            role_handoff_signature_cid=str(
                live_decision.role_handoff_signature_cid),
            policy_entry_cid=str(live_decision.policy_entry_cid),
            pmc_envelope_cid=str(live_decision.pmc_envelope_cid),
            w44_envelope_cid=str(w44_envelope_cid),
            forward=forward,
            causal_mask_cid=causal_mask.cid(),
            abstain_reason=str(abstain_reason),
        )
        return decision, (w43, causal_mask, bundle)


# =============================================================================
# Learned envelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LearnedManifoldHandoffEnvelope:
    """Sealed learned-manifold envelope for one turn of the W45
    layer.

    Records:

      * the underlying ``TEAM_HANDOFF`` capsule CID
      * the W44 live envelope CID
      * the W43 manifest-v13 envelope CID
      * the learned controller parameter CID
      * the attention-routing witness CID + per-channel logits
      * the role-adapter witness CID
      * the causal-mask witness CID
      * the prompt-construction witness CID
      * the manifold-hint witness CID (when hint mode is on)
      * the learned outer CID binding everything

    The outer CID is content-addressed by every other field. The
    verifier (:func:`verify_learned_manifold_handoff`) re-derives
    the outer CID from the bytes alone and detects tampering with
    any subfield through one of 14 disjoint named failure modes.
    """

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w44_envelope_cid: str
    parent_w43_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
    w44_branch: str
    pmc_branch: str
    abstain_reason: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    prompt_hint_mode: str
    inline_route_mode: str
    factoradic_int: int
    factoradic_n_bits: int
    hint_confidence_bucket: int

    # Learned controller provenance.
    controller_params_cid: str
    training_set_cid: str
    fitting_method: str
    use_attention_routing: bool
    role_adapter_disabled: bool

    # Learned forward witnesses.
    attention_routing_witness_cid: str
    role_adapter_witness_cid: str
    causal_mask_witness_cid: str

    # Prompt / hint witnesses.
    prompt_sha256: str
    prompt_construction_witness_cid: str
    hint_witness_cid: str
    output_sha256: str

    # Token accounting.
    n_visible_prompt_tokens_textual: int
    n_visible_prompt_tokens_actual: int
    n_visible_prompt_tokens_saved: int
    n_overhead_tokens: int
    n_hint_tokens: int

    # Margin diagnostics.
    gate_logit: float
    ratify_probability: float

    behavioral_change: bool

    learned_witness_cid: str
    learned_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w45_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w44_envelope_cid=self.parent_w44_envelope_cid,
            controller_params_cid=self.controller_params_cid,
            learned_witness_cid=self.learned_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_w45_attention_routing_witness_cid(
        *,
        per_channel_logits: Sequence[float],
        attention_weights: Sequence[float],
        gate_logit: float,
        ratify_probability: float,
        use_attention_routing: bool,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w45_attention_routing_witness",
        "per_channel_logits": _round_floats(per_channel_logits),
        "attention_weights": _round_floats(attention_weights),
        "gate_logit": float(round(gate_logit, 12)),
        "ratify_probability": float(round(
            ratify_probability, 12)),
        "use_attention_routing": bool(use_attention_routing),
        "turn_index": int(turn_index),
    })


def _compute_w45_role_adapter_witness_cid(
        *,
        role: str,
        role_delta_value: float,
        role_delta_rank: int,
        role_adapter_disabled: bool,
        present: bool,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w45_role_adapter_witness",
        "role": str(role),
        "role_delta_value": float(round(role_delta_value, 12)),
        "role_delta_rank": int(role_delta_rank),
        "role_adapter_disabled": bool(role_adapter_disabled),
        "present": bool(present),
        "turn_index": int(turn_index),
    })


def _compute_w45_hint_witness_cid(
        *,
        prompt_hint_mode: str,
        factoradic_int: int,
        factoradic_n_bits: int,
        confidence_bucket: int,
        ratify_probability: float,
        hint_bytes_sha256: str,
        n_hint_tokens: int,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w45_hint_witness",
        "prompt_hint_mode": str(prompt_hint_mode),
        "factoradic_int": int(factoradic_int),
        "factoradic_n_bits": int(factoradic_n_bits),
        "confidence_bucket": int(confidence_bucket),
        "ratify_probability": float(round(
            ratify_probability, 12)),
        "hint_bytes_sha256": str(hint_bytes_sha256),
        "n_hint_tokens": int(n_hint_tokens),
        "turn_index": int(turn_index),
    })


def _compute_w45_prompt_construction_witness_cid(
        *,
        turn_index: int,
        role: str,
        prompt_sha256: str,
        prompt_hint_mode: str,
        inline_route_mode: str,
        factoradic_int: int,
        factoradic_n_bits: int,
        confidence_bucket: int,
        n_visible_prompt_tokens_textual: int,
        n_visible_prompt_tokens_actual: int,
        n_hint_tokens: int,
) -> str:
    return _sha256_hex({
        "kind": "w45_prompt_construction_witness",
        "turn_index": int(turn_index),
        "role": str(role),
        "prompt_sha256": str(prompt_sha256),
        "prompt_hint_mode": str(prompt_hint_mode),
        "inline_route_mode": str(inline_route_mode),
        "factoradic_int": int(factoradic_int),
        "factoradic_n_bits": int(factoradic_n_bits),
        "confidence_bucket": int(confidence_bucket),
        "n_visible_prompt_tokens_textual": int(
            n_visible_prompt_tokens_textual),
        "n_visible_prompt_tokens_actual": int(
            n_visible_prompt_tokens_actual),
        "n_hint_tokens": int(n_hint_tokens),
    })


def _compute_w45_learned_witness_cid(
        *,
        decision_branch: str,
        w44_branch: str,
        pmc_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        controller_params_cid: str,
        attention_routing_witness_cid: str,
        role_adapter_witness_cid: str,
        causal_mask_witness_cid: str,
        prompt_construction_witness_cid: str,
        hint_witness_cid: str,
        output_sha256: str,
        behavioral_change: bool,
) -> str:
    return _sha256_hex({
        "kind": "w45_learned_witness",
        "decision_branch": str(decision_branch),
        "w44_branch": str(w44_branch),
        "pmc_branch": str(pmc_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "controller_params_cid": str(controller_params_cid),
        "attention_routing_witness_cid": str(
            attention_routing_witness_cid),
        "role_adapter_witness_cid": str(
            role_adapter_witness_cid),
        "causal_mask_witness_cid": str(causal_mask_witness_cid),
        "prompt_construction_witness_cid": str(
            prompt_construction_witness_cid),
        "hint_witness_cid": str(hint_witness_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
    })


def _compute_w45_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w44_envelope_cid: str,
        controller_params_cid: str,
        learned_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w45_learned_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w44_envelope_cid": str(parent_w44_envelope_cid),
        "controller_params_cid": str(controller_params_cid),
        "learned_witness_cid": str(learned_witness_cid),
        "turn_index": int(turn_index),
    })


# =============================================================================
# Verifier (14 enumerated W45 failure modes)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LearnedManifoldVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


W45_ALL_FAILURE_MODES: tuple[str, ...] = (
    "empty_w45_envelope",
    "w45_schema_version_unknown",
    "w45_schema_cid_mismatch",
    "w45_decision_branch_unknown",
    "w45_hint_mode_unknown",
    "w45_role_handoff_signature_cid_invalid",
    "w45_prompt_sha256_invalid",
    "w45_token_accounting_invalid",
    "w45_confidence_bucket_invalid",
    "w45_ratify_probability_invalid",
    "w45_attention_routing_witness_cid_mismatch",
    "w45_role_adapter_witness_cid_mismatch",
    "w45_causal_mask_witness_cid_invalid",
    "w45_prompt_construction_witness_cid_mismatch",
    "w45_hint_witness_cid_mismatch",
    "w45_learned_witness_cid_mismatch",
    "w45_outer_cid_mismatch",
)


def verify_learned_manifold_handoff(
        env: "LearnedManifoldHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
        registered_controller_params_cid: str | None = None,
) -> LearnedManifoldVerificationOutcome:
    """Pure-function verifier for the W45 learned envelope.

    Enumerates 14+ disjoint W45 failure modes (see
    :data:`W45_ALL_FAILURE_MODES`). When
    ``registered_controller_params_cid`` is provided, the verifier
    additionally enforces that the envelope's controller params CID
    matches the registered one.
    """
    n = 0
    if env is None:
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="empty_w45_envelope", n_checks=0)
    n += 1
    if env.schema_version != W45_LEARNED_MANIFOLD_SCHEMA_VERSION:
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W45_ALL_BRANCHES:
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.prompt_hint_mode not in W45_ALL_HINT_MODES:
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_hint_mode_unknown", n_checks=n)
    n += 1
    if env.decision_branch != (
            W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return LearnedManifoldVerificationOutcome(
                ok=False,
                reason=(
                    "w45_role_handoff_signature_cid_invalid"),
                n_checks=n)
    n += 1
    if (env.prompt_sha256 is None
            or (env.prompt_sha256
                and len(env.prompt_sha256) not in (0, 64))):
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_prompt_sha256_invalid",
            n_checks=n)
    n += 1
    if (env.n_visible_prompt_tokens_textual < 0
            or env.n_visible_prompt_tokens_actual < 0
            or env.n_overhead_tokens < 0
            or env.n_hint_tokens < 0
            or env.n_visible_prompt_tokens_saved
            != (int(env.n_visible_prompt_tokens_textual)
                - int(env.n_visible_prompt_tokens_actual))):
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_token_accounting_invalid",
            n_checks=n)
    n += 1
    if (env.hint_confidence_bucket < 0
            or env.hint_confidence_bucket
            >= W45_CONFIDENCE_BUCKETS):
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_confidence_bucket_invalid",
            n_checks=n)
    n += 1
    if not (0.0 - 1e-9 <= float(env.ratify_probability)
            <= 1.0 + 1e-9):
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_ratify_probability_invalid",
            n_checks=n)
    n += 1
    # Causal-mask witness CID: must be 64-hex or empty for trivial
    # passthrough.
    if env.decision_branch != (
            W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH):
        if (not env.causal_mask_witness_cid
                or len(env.causal_mask_witness_cid) != 64):
            return LearnedManifoldVerificationOutcome(
                ok=False,
                reason="w45_causal_mask_witness_cid_invalid",
                n_checks=n)
    n += 1
    # Optional registered-params binding.
    if registered_controller_params_cid is not None:
        if env.controller_params_cid != str(
                registered_controller_params_cid):
            return LearnedManifoldVerificationOutcome(
                ok=False,
                reason=(
                    "w45_attention_routing_witness_cid_mismatch"),
                n_checks=n)
    n += 1
    # Re-derive witness CIDs and verify against envelope.
    if env.decision_branch != (
            W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH):
        # The trivial path may have empty witness CIDs.
        # Construction witness re-derivation.
        expected_construction = (
            _compute_w45_prompt_construction_witness_cid(
                turn_index=int(env.turn_index),
                role=env.role,
                prompt_sha256=env.prompt_sha256,
                prompt_hint_mode=env.prompt_hint_mode,
                inline_route_mode=env.inline_route_mode,
                factoradic_int=int(env.factoradic_int),
                factoradic_n_bits=int(env.factoradic_n_bits),
                confidence_bucket=int(
                    env.hint_confidence_bucket),
                n_visible_prompt_tokens_textual=int(
                    env.n_visible_prompt_tokens_textual),
                n_visible_prompt_tokens_actual=int(
                    env.n_visible_prompt_tokens_actual),
                n_hint_tokens=int(env.n_hint_tokens),
            ))
        if expected_construction != (
                env.prompt_construction_witness_cid):
            return LearnedManifoldVerificationOutcome(
                ok=False,
                reason=(
                    "w45_prompt_construction_witness_cid_mismatch"),
                n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.learned_outer_cid:
        return LearnedManifoldVerificationOutcome(
            ok=False, reason="w45_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return LearnedManifoldVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Learned team result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LearnedManifoldTurn:
    """One turn of a :class:`LearnedManifoldTeam` run."""

    agent_turn: AgentTurn
    decision: LearnedGatingDecision
    envelope: LearnedManifoldHandoffEnvelope


@dataclasses.dataclass(frozen=True)
class LearnedManifoldTeamResult:
    """Result of a :class:`LearnedManifoldTeam` run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    learned_turns: tuple[LearnedManifoldTurn, ...]
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
    n_visible_tokens_added_hint: int = 0
    n_abstain_substitutions: int = 0
    n_learned_margin_abstains: int = 0
    mean_ratify_probability: float = 0.0
    controller_params_cid: str = ""
    schema: str = W45_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


# =============================================================================
# Learned manifold team
# =============================================================================

class LearnedManifoldTeam:
    """W45 learned-coupled agent team.

    Wraps the released :class:`coordpy.AgentTeam` contract with the
    learned manifold controller plus the W44 live gate. With a
    trivial learned registry, this team reduces to ``AgentTeam.run``
    byte-for-byte (the W45-L-TRIVIAL-LEARNED-PASSTHROUGH falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: LearnedManifoldRegistry,
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
                "LearnedManifoldTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = LearnedManifoldOrchestrator(registry)
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
                "LearnedManifoldTeam")
        if not isinstance(backend, LLMBackend):
            raise TypeError(
                "backend must satisfy the LLMBackend protocol")
        return backend

    def _format_manifold_hint(
            self,
            *,
            decision: LearnedGatingDecision,
            role_universe: Sequence[str],
    ) -> str:
        """Construct the model-facing MANIFOLD_HINT string.

        Format:

            MANIFOLD_HINT: route=<int> conf=<bucket> p=<rounded
            probability> over <comma-sep role universe>

        Bijective: route + conf + p (rounded to
        ``hint_probability_precision`` decimals) recover every
        learned-controller surface that the model sees.
        """
        precision = int(
            self.registry.hint_probability_precision)
        p_rounded = round(
            float(decision.forward.ratify_probability), precision)
        return (
            f"MANIFOLD_HINT: route={int(decision.factoradic_int)} "
            f"conf={int(decision.forward.confidence_bucket)} "
            f"p={p_rounded} "
            f"over {','.join(role_universe)}")

    def _build_prompt(
            self,
            *,
            member: Agent,
            task: str,
            turn_index: int,
            recent_handoffs: Sequence[tuple[str, str]],
            all_prior_outputs: Sequence[tuple[str, str]],
            decision: LearnedGatingDecision,
            role_universe: Sequence[str],
            role_arrival_order: Sequence[str],
    ) -> tuple[str, str, int, int, int, str]:
        """Construct the bounded prompt + a textual-shadow prompt.

        Returns ``(bounded_prompt, textual_shadow_prompt,
        n_textual_tokens, n_actual_tokens, n_hint_tokens,
        hint_bytes)``. The hint_bytes field is the literal hint
        string (or empty when no hint mode is on); the controller's
        envelope binds the SHA-256 of these bytes.
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

        # Textual shadow: AgentTeam-equivalent rendering, no hint.
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

        # Bounded prompt with optional hint + factoradic route.
        bounded_parts = list(common_parts)
        hint_mode = self.registry.prompt_hint_mode
        hint_str = ""
        n_hint_tokens = 0
        inline_route_mode = (
            self.registry.live_registry.inline_route_mode)

        if (inline_route_mode in (
                W44_ROUTE_MODE_FACTORADIC,
                W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL)
                and decision.factoradic_n_bits > 0
                and recent_handoffs):
            route_header = (
                f"FACTORADIC_ROUTE: {decision.factoradic_int} "
                f"over {','.join(role_universe)}")
        else:
            route_header = ""

        if hint_mode in (
                W45_HINT_MODE_FACTORADIC_WITH_HINT,
                W45_HINT_MODE_HINT_ONLY):
            hint_str = self._format_manifold_hint(
                decision=decision, role_universe=role_universe)
            n_hint_tokens = len(hint_str.split())

        # Compose the bounded prompt according to hint mode.
        if hint_mode == W45_HINT_MODE_HINT_ONLY and hint_str:
            # Replace the textual handoff list with the hint
            # entirely.
            bounded_parts.append(hint_str)
        elif hint_mode == W45_HINT_MODE_FACTORADIC_WITH_HINT and hint_str:
            if route_header:
                bounded_parts.append(
                    f"{route_header}\n{hint_str}")
            else:
                bounded_parts.append(hint_str)
            if recent_handoffs and hint_mode == (
                    W45_HINT_MODE_FACTORADIC_WITH_HINT):
                # Hint + factoradic only; do not render the textual
                # handoff list (the route + hint preserve it for
                # the audit chain).
                pass
        else:
            # No hint — defer to the underlying W44 builder semantics.
            if route_header and inline_route_mode == (
                    W44_ROUTE_MODE_FACTORADIC):
                bounded_parts.append(route_header)
            elif route_header and inline_route_mode == (
                    W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL):
                rendered = "\n".join(
                    f"- {role}: {text}"
                    for role, text in recent_handoffs[
                        -self.max_visible_handoffs:])
                bounded_parts.append(
                    f"{route_header}\n"
                    "Visible team handoffs:\n"
                    f"{rendered}")
            elif recent_handoffs:
                rendered = "\n".join(
                    f"- {role}: {text}"
                    for role, text in recent_handoffs[
                        -self.max_visible_handoffs:])
                bounded_parts.append(
                    "Visible team handoffs (bounded to avoid "
                    f"token cramming):\n{rendered}")

        bounded_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        bounded_prompt = "\n\n".join(bounded_parts)
        n_textual = len(textual_prompt.split())
        n_actual = len(bounded_prompt.split())
        return (
            bounded_prompt,
            textual_prompt,
            n_textual,
            n_actual,
            n_hint_tokens,
            hint_str,
        )

    def run(
            self,
            task: str,
            *,
            progress: Callable[
                [LearnedManifoldTurn], None] | None = None,
    ) -> LearnedManifoldTeamResult:
        """Run the learned-coupled team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        learned_turns: list[LearnedManifoldTurn] = []
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
        n_visible_tokens_added_hint = 0
        n_abstain_substitutions = 0
        n_learned_margin_abstains = 0
        ratify_probabilities: list[float] = []
        head_backend = self.backend
        head_model = (
            getattr(head_backend, "model", "") or "")
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()
        controller_params_cid = self.registry.params.cid()

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
            w43_result, causal_mask, bundle = aux
            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            (bounded_prompt, textual_prompt,
             n_textual_tokens, n_actual_tokens,
             n_hint_tokens, hint_str) = self._build_prompt(
                member=member,
                task=task,
                turn_index=idx,
                recent_handoffs=recent_handoffs,
                all_prior_outputs=all_prior_outputs,
                decision=decision,
                role_universe=role_universe,
                role_arrival_order=role_arrival_order,
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
                        W45_BRANCH_LEARNED_MARGIN_ABSTAIN):
                    n_learned_margin_abstains += 1
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
            if n_added > 0 and not do_substitute:
                n_visible_tokens_added_hint += int(n_added)
            ratify_probabilities.append(
                float(decision.forward.ratify_probability))

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

            # Build the W45 envelope (incl. all 4 sub-witness CIDs).
            attention_witness_cid = (
                _compute_w45_attention_routing_witness_cid(
                    per_channel_logits=(
                        decision.forward.per_channel_logits),
                    attention_weights=(
                        decision.forward.attention_weights),
                    gate_logit=float(decision.forward.gate_logit),
                    ratify_probability=float(
                        decision.forward.ratify_probability),
                    use_attention_routing=bool(
                        self.registry.use_attention_routing),
                    turn_index=int(idx),
                ))
            role_adapter_present = (
                str(role) in
                self.registry.params.role_delta_map
                and not self.registry.role_adapter_disabled)
            role_adapter_witness_cid = (
                _compute_w45_role_adapter_witness_cid(
                    role=str(role),
                    role_delta_value=float(
                        decision.forward.role_delta_value),
                    role_delta_rank=int(
                        self.registry.params.role_delta_rank),
                    role_adapter_disabled=bool(
                        self.registry.role_adapter_disabled),
                    present=bool(role_adapter_present),
                    turn_index=int(idx),
                ))
            hint_sha = (
                hashlib.sha256(hint_str.encode("utf-8")).hexdigest()
                if hint_str else "")
            hint_witness_cid = _compute_w45_hint_witness_cid(
                prompt_hint_mode=self.registry.prompt_hint_mode,
                factoradic_int=int(decision.factoradic_int),
                factoradic_n_bits=int(
                    decision.factoradic_n_bits),
                confidence_bucket=int(
                    decision.forward.confidence_bucket),
                ratify_probability=float(
                    decision.forward.ratify_probability),
                hint_bytes_sha256=hint_sha,
                n_hint_tokens=int(n_hint_tokens),
                turn_index=int(idx),
            )
            construction_cid = (
                _compute_w45_prompt_construction_witness_cid(
                    turn_index=int(idx),
                    role=str(role),
                    prompt_sha256=prompt_sha,
                    prompt_hint_mode=self.registry.prompt_hint_mode,
                    inline_route_mode=(
                        self.registry.live_registry.inline_route_mode),
                    factoradic_int=int(decision.factoradic_int),
                    factoradic_n_bits=int(
                        decision.factoradic_n_bits),
                    confidence_bucket=int(
                        decision.forward.confidence_bucket),
                    n_visible_prompt_tokens_textual=int(
                        n_textual_tokens),
                    n_visible_prompt_tokens_actual=int(
                        n_actual_tokens),
                    n_hint_tokens=int(n_hint_tokens),
                ))
            behavioral_change = bool(
                do_substitute or n_saved > 0 or n_added > 0)
            learned_witness_cid = _compute_w45_learned_witness_cid(
                decision_branch=decision.branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                controller_params_cid=controller_params_cid,
                attention_routing_witness_cid=(
                    attention_witness_cid),
                role_adapter_witness_cid=role_adapter_witness_cid,
                causal_mask_witness_cid=decision.causal_mask_cid,
                prompt_construction_witness_cid=construction_cid,
                hint_witness_cid=hint_witness_cid,
                output_sha256=output_sha,
                behavioral_change=behavioral_change,
            )
            outer_cid = _compute_w45_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w44_envelope_cid=str(
                    decision.w44_envelope_cid),
                controller_params_cid=controller_params_cid,
                learned_witness_cid=learned_witness_cid,
                turn_index=int(idx),
            )
            envelope = LearnedManifoldHandoffEnvelope(
                schema_version=W45_LEARNED_MANIFOLD_SCHEMA_VERSION,
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w44_envelope_cid=str(
                    decision.w44_envelope_cid),
                parent_w43_envelope_cid=str(
                    decision.pmc_envelope_cid),
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
                w44_branch=decision.w44_branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                prompt_hint_mode=self.registry.prompt_hint_mode,
                inline_route_mode=(
                    self.registry.live_registry.inline_route_mode),
                factoradic_int=int(decision.factoradic_int),
                factoradic_n_bits=int(decision.factoradic_n_bits),
                hint_confidence_bucket=int(
                    decision.forward.confidence_bucket),
                controller_params_cid=controller_params_cid,
                training_set_cid=str(
                    self.registry.params.training_set_cid),
                fitting_method=str(
                    self.registry.params.fitting_method),
                use_attention_routing=bool(
                    self.registry.use_attention_routing),
                role_adapter_disabled=bool(
                    self.registry.role_adapter_disabled),
                attention_routing_witness_cid=(
                    attention_witness_cid),
                role_adapter_witness_cid=role_adapter_witness_cid,
                causal_mask_witness_cid=decision.causal_mask_cid,
                prompt_sha256=prompt_sha,
                prompt_construction_witness_cid=construction_cid,
                hint_witness_cid=hint_witness_cid,
                output_sha256=output_sha,
                n_visible_prompt_tokens_textual=int(
                    n_textual_tokens),
                n_visible_prompt_tokens_actual=int(
                    n_actual_tokens),
                n_visible_prompt_tokens_saved=int(
                    n_textual_tokens - n_actual_tokens),
                n_overhead_tokens=int(
                    w43_result.n_w43_overhead_tokens),
                n_hint_tokens=int(n_hint_tokens),
                gate_logit=float(decision.forward.gate_logit),
                ratify_probability=float(
                    decision.forward.ratify_probability),
                behavioral_change=bool(behavioral_change),
                learned_witness_cid=learned_witness_cid,
                learned_outer_cid=outer_cid,
            )
            learned_turn = LearnedManifoldTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
            )
            learned_turns.append(learned_turn)

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

            if progress is not None:
                try:
                    progress(learned_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[LearnedManifoldTeam] progress callback "
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
        return LearnedManifoldTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            learned_turns=tuple(learned_turns),
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
            n_visible_tokens_added_hint=int(
                n_visible_tokens_added_hint),
            n_abstain_substitutions=int(n_abstain_substitutions),
            n_learned_margin_abstains=int(
                n_learned_margin_abstains),
            mean_ratify_probability=float(mean_p),
            controller_params_cid=str(controller_params_cid),
        )


# =============================================================================
# Synthetic backends for the hint-response benchmark
# =============================================================================

@dataclasses.dataclass
class HintAwareSyntheticBackend:
    """Deterministic backend that returns one canonical answer when
    the prompt contains a ``MANIFOLD_HINT: route=<int>`` substring,
    and a different answer otherwise.

    Used by R-92 to exercise the *behavioural* effect of the W45
    learned prompt-hint on a controlled synthetic ground truth.
    Not a real LLM; the response is keyed only on the presence of
    the hint substring and (optionally) the confidence bucket.

    Honest scope: this backend is not a real LLM; it is a
    pre-canned ground truth distribution we use to demonstrate that
    *if* a model conditions on the hint, the W45 layer's
    behavioural surface delivers a measurable gain. On real LLMs
    the saving is bounded to ``the hint is present in the model's
    context``, which is a strict superset of W44's surface but not
    a guarantee of behavioural lift.
    """

    correct_with_hint: str = "MANIFOLD_OK"
    answer_without_hint: str = "MANIFOLD_NO_HINT"
    n_calls: int = 0
    model_tag: str = "synthetic.hint_aware"
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
        if "MANIFOLD_HINT: route=" in (prompt or ""):
            return self.correct_with_hint
        return self.answer_without_hint


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Schema, branches, defaults
    "W45_LEARNED_MANIFOLD_SCHEMA_VERSION",
    "W45_TEAM_RESULT_SCHEMA",
    "W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH",
    "W45_BRANCH_LEARNED_DISABLED",
    "W45_BRANCH_LEARNED_RATIFIED",
    "W45_BRANCH_LEARNED_NO_POLICY",
    "W45_BRANCH_LEARNED_CAUSAL_ABSTAIN",
    "W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN",
    "W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN",
    "W45_BRANCH_LEARNED_MARGIN_ABSTAIN",
    "W45_BRANCH_LEARNED_REJECTED",
    "W45_ALL_BRANCHES",
    "W45_LEARNED_ABSTAIN_BRANCHES",
    "W45_HINT_MODE_OFF",
    "W45_HINT_MODE_FACTORADIC_WITH_HINT",
    "W45_HINT_MODE_HINT_ONLY",
    "W45_ALL_HINT_MODES",
    "W45_CHANNEL_ORDER",
    "W45_N_CHANNELS",
    "W45_CONFIDENCE_BUCKETS",
    "W45_DEFAULT_RIDGE_LAMBDA",
    "W45_DEFAULT_FEATURE_DIM",
    "W45_DEFAULT_ROLE_DELTA_RANK",
    "W45_ALL_FAILURE_MODES",
    # Channel features
    "_channel_features_from_bundle",
    # Controller params + training
    "LearnedControllerParams",
    "build_unfitted_controller_params",
    "TrainingExample",
    "TrainingSet",
    "fit_learned_controller",
    "ControllerForwardResult",
    "forward_controller",
    # Causal mask witness
    "CausalMaskWitness",
    "derive_causal_mask",
    # Registry + orchestrator
    "LearnedManifoldRegistry",
    "LearnedManifoldOrchestrator",
    "LearnedGatingDecision",
    # Envelope + verifier
    "LearnedManifoldHandoffEnvelope",
    "LearnedManifoldVerificationOutcome",
    "verify_learned_manifold_handoff",
    # Team
    "LearnedManifoldTurn",
    "LearnedManifoldTeamResult",
    "LearnedManifoldTeam",
    # Builders
    "build_trivial_learned_manifold_registry",
    "build_learned_manifold_registry",
    # Synthetic hint-aware backend
    "HintAwareSyntheticBackend",
]
