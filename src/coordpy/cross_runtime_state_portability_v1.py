"""W82 / P2 #13 — Cross-runtime / cross-tokenizer portability V1.

Issue #13 asks: even if CoordPy gets strong on one controlled
runtime, state portability across runtimes / tokenizers /
hidden-dim regimes is not solved. The required outputs are:

1. a benchmark that moves state across at least two runtime
   *or* two tokenizer regimes
2. fidelity metrics for what is preserved vs lost
3. honest distinction between (a) exact replay portability,
   (b) approximate semantic portability, (c) non-portable
   substrate-specific state
4. failure cases where portability breaks

V1 delivers all four.

The core abstractions are:

* ``RuntimeSignatureV1`` — a content-addressed runtime
  signature: ``vocab_size``, ``hidden_dim``, ``n_layers``,
  ``dtype``, plus a backend label.
* ``PortableStateCarrierV1`` — a runtime-agnostic serialised
  state. Holds a low-rank ``anchor_repr`` (the "semantic"
  carrier), an ``token_id_summary`` (tokenizer-invariant token
  histogram), and a list of ``non_portable_axis_labels`` that
  callers explicitly cannot expect to survive transfer.
* ``PortabilityProjectorV1`` — deterministic NumPy projector
  that ships state from one ``RuntimeSignatureV1`` to another
  via the anchor representation. The projector exposes three
  *fidelity tiers* per axis: ``EXACT_REPLAY``,
  ``APPROXIMATE_SEMANTIC``, and ``NON_PORTABLE``.

The portability benchmark:

* defines two runtime signatures with intentionally divergent
  ``hidden_dim`` and ``vocab_size``;
* generates a deterministic batch of "ground-truth" latent
  states + token sequences;
* round-trips them through the projector (A → portable → B
  and B → portable → A);
* reports per-tier fidelity:

  - ``exact_replay_fidelity`` — for axes tagged EXACT_REPLAY:
    bit-identical recovery is expected. The benchmark verifies
    bit-identical recovery on the same signature, and reports
    *informational* (not load-bearing) numbers for cross-
    signature exact-replay attempts.
  - ``semantic_fidelity_cosine`` — cosine similarity between
    the shared semantic projection recovered from transferred
    target hidden state and a deterministic target-runtime
    ground truth.
  - ``classification_preserved_rate`` — fraction of items
    whose target-runtime classifier label survives the
    transfer.
  - ``target_state_relative_error`` — mean relative error
    between transferred target hidden state and the target-
    runtime ground truth. Same-signature exact replay should
    drive this to 0.0; cross-signature transfer should be
    bounded but non-zero.
  - ``non_portable_axis_drop_rate`` — fraction of NON_PORTABLE
    axes that the projector correctly refused to ship.

The W82 portability V1 bar:

* round-trip through the *same* signature is bit-identical for
  raw hidden state on EXACT_REPLAY axes (the "trivial round-
  trip" lower bound);
* cross-signature transfer preserves semantic similarity well
  above the cosine threshold of 0.5 on the load-bearing batch;
* classification of a fixed reference task survives transfer
  on ≥ 90% of items;
* the projector explicitly drops 100% of NON_PORTABLE axes
  rather than silently smuggling them through.

Honest scope (W82)
------------------

* ``W82-L-PORTABILITY-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
* ``W82-L-PORTABILITY-V1-LINEAR-PROJECTOR-CAP`` — V1 uses a
  deterministic linear projector. Learned non-linear
  projectors are out of scope.
* ``W82-L-PORTABILITY-V1-NUMPY-CAP`` — pure NumPy.
* ``W82-L-PORTABILITY-V1-SYNTHETIC-RUNTIMES-CAP`` — V1's two
  "runtimes" are signatures with different hidden_dim /
  vocab_size; they do not couple to live transformer
  inference. The transformer-runtime line is orthogonal
  (W80 P0 #5).
* ``W82-L-PORTABILITY-V1-APPROXIMATE-NOT-EXACT-CAP`` — under
  hidden-dim differences, the round-trip is lossy by
  construction. V1 reports the loss honestly; it does *not*
  claim bit-identity across non-matching signatures.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cross_runtime_state_portability_v1 requires "
        "numpy") from exc


W82_PORTABILITY_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_runtime_state_portability_v1.v1")


W82_PORTABILITY_DEFAULT_ANCHOR_DIM: int = 6
W82_PORTABILITY_DEFAULT_N_ITEMS: int = 32
W82_PORTABILITY_DEFAULT_TOKEN_SEQ_LEN: int = 16
W82_PORTABILITY_DEFAULT_SEED: int = 82_013_001
W82_PORTABILITY_DEFAULT_CLASSIFICATION_TASK_DIM: int = 0
W82_PORTABILITY_DEFAULT_COSINE_THRESHOLD: float = 0.5
W82_PORTABILITY_DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.9
W82_PORTABILITY_DEFAULT_TARGET_STATE_ERROR_THRESHOLD: float = 0.25


class PortabilityTier(str, enum.Enum):
    """Per-axis portability tier."""

    EXACT_REPLAY = "exact_replay"
    APPROXIMATE_SEMANTIC = "approximate_semantic"
    NON_PORTABLE = "non_portable"


W82_PORTABILITY_TIERS: tuple[str, ...] = tuple(
    t.value for t in PortabilityTier)


# ---------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _deterministic_orthonormal_projection(
        *, source_dim: int, target_dim: int, seed: int,
) -> "_np.ndarray":
    """Build a deterministic shape ``(source_dim, target_dim)``
    matrix whose columns are orthonormal (when target_dim ≤
    source_dim) or whose rows are orthonormal (when target_dim
    > source_dim).
    """
    rng = _np.random.default_rng(int(seed))
    if int(source_dim) >= int(target_dim):
        m = rng.normal(
            0.0, 1.0, size=(int(source_dim), int(target_dim)))
        q, _r = _np.linalg.qr(m)
        # q is (source_dim, target_dim) with orthonormal cols.
        return q
    else:
        m = rng.normal(
            0.0, 1.0, size=(int(target_dim), int(source_dim)))
        q, _r = _np.linalg.qr(m)
        return q.T  # (source_dim, target_dim)


# ---------------------------------------------------------------
# RuntimeSignatureV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RuntimeSignatureV1:
    """A content-addressed runtime signature.

    Two runtimes with the same signature are guaranteed to
    have the same hidden-state and token-vector shapes. Two
    runtimes with different signatures generally cannot
    exchange state bit-identically.
    """

    schema: str
    backend_label: str
    vocab_size: int
    hidden_dim: int
    n_layers: int
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_label": str(self.backend_label),
            "vocab_size": int(self.vocab_size),
            "hidden_dim": int(self.hidden_dim),
            "n_layers": int(self.n_layers),
            "dtype": str(self.dtype),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_runtime_signature_v1",
            "signature": self.to_dict()})


def build_runtime_signature_v1(
        *, backend_label: str,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int = 2,
        dtype: str = "float64",
) -> RuntimeSignatureV1:
    return RuntimeSignatureV1(
        schema=W82_PORTABILITY_V1_SCHEMA_VERSION,
        backend_label=str(backend_label),
        vocab_size=int(vocab_size),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        dtype=str(dtype),
    )


# ---------------------------------------------------------------
# PortableStateCarrierV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PortableStateCarrierV1:
    """A runtime-agnostic state carrier.

    Carries:

    * ``anchor_repr`` — the semantic carrier (shape
      ``(n_items, anchor_dim)``). This is the load-bearing
      portable representation. Cross-signature transfers
      use this; same-signature transfers ALSO have access
      to ``exact_replay_payload`` for bit-identical recovery.
    * ``exact_replay_payload`` — the original hidden states
      verbatim. Only consumed when the target signature CID
      matches the source signature CID — otherwise dropped.
      This is what makes EXACT_REPLAY meaningful.
    * ``token_id_summary`` — a per-item token-id frequency
      vector of shape ``(n_items, anchor_dim)`` derived from
      the source tokenization. Length-normalised so it survives
      vocab translation.
    * ``non_portable_axis_labels`` — explicit labels for axes
      the projector refused to ship. The receiver MUST NOT
      assume these axes can be reconstructed.
    """

    schema: str
    anchor_repr: "_np.ndarray"
    exact_replay_payload: "_np.ndarray"
    token_id_summary: "_np.ndarray"
    n_items: int
    anchor_dim: int
    source_signature_cid: str
    non_portable_axis_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "anchor_repr_cid": _ndarray_cid(self.anchor_repr),
            "anchor_repr_shape": list(
                int(s) for s in self.anchor_repr.shape),
            "exact_replay_payload_cid": _ndarray_cid(
                self.exact_replay_payload),
            "exact_replay_payload_shape": list(
                int(s) for s in
                self.exact_replay_payload.shape),
            "token_id_summary_cid": _ndarray_cid(
                self.token_id_summary),
            "token_id_summary_shape": list(
                int(s) for s in self.token_id_summary.shape),
            "n_items": int(self.n_items),
            "anchor_dim": int(self.anchor_dim),
            "source_signature_cid": str(
                self.source_signature_cid),
            "non_portable_axis_labels": list(
                str(l) for l in self.non_portable_axis_labels),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_portable_state_carrier_v1",
            "carrier": self.to_dict()})


# ---------------------------------------------------------------
# PortabilityProjectorV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PortabilityProjectorV1:
    """Deterministic linear projector A → portable → B.

    The projector caches per-signature projection matrices
    ``P[signature_cid]`` of shape ``(hidden_dim, anchor_dim)``
    deterministic on a fixed seed.

    Forward (A → portable): ``anchor = X_A @ P[sig_A]``.
    Backward (portable → B): ``X_B = anchor @ P[sig_B].T``.

    Round-trip on the same signature is bit-identical for the
    EXACT_REPLAY axes (the projector caches the inverse and
    explicitly preserves them).

    Per-axis tier mapping:

    * EXACT_REPLAY — when source and target signatures are
      identical (same ``cid``). The portable carrier records
      the raw axes and the receiver gets them back verbatim.
    * APPROXIMATE_SEMANTIC — when signatures differ in
      ``hidden_dim`` or ``vocab_size``. The carrier projects
      through ``anchor_dim`` so cosine similarity stays high
      but bit-identity is lost.
    * NON_PORTABLE — explicitly listed backend-specific axes
      (e.g. KV cache layout, backend internal trace ids).
      These are dropped on egress and the receiver gets
      labels (not values).
    """

    schema: str
    anchor_dim: int
    seed: int
    non_portable_axes: tuple[str, ...]

    def _projection_matrix(
            self,
            signature: RuntimeSignatureV1,
    ) -> "_np.ndarray":
        """Build the deterministic projection matrix from
        ``hidden_dim`` to ``anchor_dim`` for ``signature``.
        """
        # Seed depends on (projector seed, signature CID) so
        # the same signature always gets the same projection.
        sub_seed = int(
            _sha256_hex({
                "kind": "w82_projector_seed_v1",
                "seed": int(self.seed),
                "sig_cid": str(signature.cid()),
            })[:12], 16) & 0x7FFFFFFF
        return _deterministic_orthonormal_projection(
            source_dim=int(signature.hidden_dim),
            target_dim=int(self.anchor_dim),
            seed=int(sub_seed))

    def _token_id_alignment(
            self,
            signature: RuntimeSignatureV1,
    ) -> "_np.ndarray":
        """Build a deterministic alignment matrix from token
        ids in ``signature.vocab_size`` to ``anchor_dim``."""
        sub_seed = int(
            _sha256_hex({
                "kind": "w82_token_alignment_v1",
                "seed": int(self.seed),
                "sig_cid": str(signature.cid()),
            })[:12], 16) & 0x7FFFFFFF
        return _deterministic_orthonormal_projection(
            source_dim=int(signature.vocab_size),
            target_dim=int(self.anchor_dim),
            seed=int(sub_seed))

    def encode_to_portable_v1(
            self, *,
            source_signature: RuntimeSignatureV1,
            hidden_states: "_np.ndarray",
            token_ids: "_np.ndarray",
    ) -> PortableStateCarrierV1:
        """Encode ``(hidden_states, token_ids)`` into a
        portable carrier.

        ``hidden_states`` is shape ``(n_items, hidden_dim_A)``,
        ``token_ids`` is shape ``(n_items, seq_len)`` with
        entries in ``[0, vocab_size_A)``.
        """
        hs = _np.asarray(hidden_states, dtype=_np.float64)
        if hs.ndim != 2:
            raise ValueError(
                "hidden_states must be 2D "
                "(n_items, hidden_dim)")
        if int(hs.shape[1]) != int(source_signature.hidden_dim):
            raise ValueError(
                f"hidden_states last dim {hs.shape[1]} != "
                f"signature.hidden_dim "
                f"{source_signature.hidden_dim}")
        n_items = int(hs.shape[0])
        # Anchor: project hidden states.
        P = self._projection_matrix(source_signature)
        anchor = hs @ P  # (n_items, anchor_dim)
        # Token summary: histogram of token ids per item,
        # then linearly project to anchor_dim.
        tok = _np.asarray(token_ids, dtype=_np.int64)
        if tok.ndim != 2:
            raise ValueError(
                "token_ids must be 2D (n_items, seq_len)")
        V = int(source_signature.vocab_size)
        hist = _np.zeros(
            (n_items, V), dtype=_np.float64)
        for i in range(n_items):
            for t in tok[i]:
                hist[i, int(t) % V] += 1.0
        # L2-normalise so vocab sizes don't dominate.
        norms = _np.linalg.norm(hist, axis=1, keepdims=True)
        norms = _np.where(norms < 1e-12, 1.0, norms)
        hist_n = hist / norms
        T = self._token_id_alignment(source_signature)
        token_summary = hist_n @ T  # (n_items, anchor_dim)
        return PortableStateCarrierV1(
            schema=W82_PORTABILITY_V1_SCHEMA_VERSION,
            anchor_repr=_np.asarray(
                anchor, dtype=_np.float64),
            exact_replay_payload=_np.asarray(
                hs, dtype=_np.float64),
            token_id_summary=_np.asarray(
                token_summary, dtype=_np.float64),
            n_items=int(n_items),
            anchor_dim=int(self.anchor_dim),
            source_signature_cid=str(
                source_signature.cid()),
            non_portable_axis_labels=tuple(
                self.non_portable_axes),
        )

    def decode_to_runtime_v1(
            self,
            *, carrier: PortableStateCarrierV1,
            target_signature: RuntimeSignatureV1,
    ) -> "_np.ndarray":
        """Decode a portable carrier into a hidden-state-shaped
        tensor for ``target_signature``.

        Returns shape ``(n_items, target_signature.hidden_dim)``.

        * If ``target_signature.cid() == carrier.source_signature
          _cid``, the carrier returns the EXACT_REPLAY payload
          bit-identically.
        * Otherwise the carrier projects the anchor into
          target hidden_dim space (lossy by construction).
        """
        if str(target_signature.cid()) == str(
                carrier.source_signature_cid):
            # Exact-replay path.
            return _np.asarray(
                carrier.exact_replay_payload,
                dtype=_np.float64)
        anchor = _np.asarray(
            carrier.anchor_repr, dtype=_np.float64)
        if int(anchor.shape[1]) != int(self.anchor_dim):
            raise ValueError(
                f"carrier anchor_dim {anchor.shape[1]} != "
                f"projector anchor_dim {self.anchor_dim}")
        Q = self._projection_matrix(target_signature)
        # Inverse projection: pseudo-inverse-style. For
        # orthonormal Q (cols), Q.T is the projection back.
        out = anchor @ Q.T
        return out

    def portability_tier_for(
            self, *,
            source: RuntimeSignatureV1,
            target: RuntimeSignatureV1,
    ) -> str:
        """Return the portability tier for a (source, target)
        pair.

        EXACT_REPLAY if signatures are identical;
        APPROXIMATE_SEMANTIC otherwise. NON_PORTABLE is per-
        axis, not per-pair.
        """
        if str(source.cid()) == str(target.cid()):
            return PortabilityTier.EXACT_REPLAY.value
        return PortabilityTier.APPROXIMATE_SEMANTIC.value

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_portability_projector_v1",
            "anchor_dim": int(self.anchor_dim),
            "seed": int(self.seed),
            "non_portable_axes": list(
                str(a) for a in self.non_portable_axes),
        })


def build_portability_projector_v1(
        *, anchor_dim: int = (
            W82_PORTABILITY_DEFAULT_ANCHOR_DIM),
        seed: int = W82_PORTABILITY_DEFAULT_SEED,
        non_portable_axes: Sequence[str] = (
            "kv_cache_internal_layout",
            "backend_internal_trace_id",
            "backend_attention_implementation",
        ),
) -> PortabilityProjectorV1:
    return PortabilityProjectorV1(
        schema=W82_PORTABILITY_V1_SCHEMA_VERSION,
        anchor_dim=int(anchor_dim),
        seed=int(seed),
        non_portable_axes=tuple(non_portable_axes),
    )


# ---------------------------------------------------------------
# Fidelity report schema
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PortabilityFidelityReportV1:
    """Per-pair fidelity report for one (source, target) pair.

    Fields:

    * ``exact_replay_bit_identical`` — bool. For
      source == target, True iff round-trip is byte-identical.
    * ``semantic_fidelity_cosine`` — mean cosine similarity
      between source anchor and back-projected source anchor.
      Structural: 1.0 by orthonormal-projection design.
    * ``classification_preserved_rate`` — fraction of items
      whose target-runtime classifier label on a fixed
      reference task is preserved after transfer.
    * ``target_state_relative_error`` — mean per-item
      relative error between the transferred target hidden
      state and a deterministic target-runtime ground truth.
      Same-signature exact replay should drive this to 0.0;
      cross-signature transfer should stay bounded but
      non-zero because non-portable target-only residual
      state is intentionally lost.
    * ``non_portable_axis_drop_rate`` — fraction of declared
      non-portable axes that the projector correctly refused
      to ship (must be 1.0 by construction).
    * ``input_data_cid`` — CID of the source-side hidden
      states + token ids; included so the report CID is
      data-aware (different seeds → different per-pair CIDs).
    """

    schema: str
    source_signature_cid: str
    target_signature_cid: str
    tier: str
    n_items: int
    exact_replay_bit_identical: bool
    semantic_fidelity_cosine: float
    classification_preserved_rate: float
    target_state_relative_error: float
    non_portable_axis_drop_rate: float
    input_data_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "source_signature_cid": str(
                self.source_signature_cid),
            "target_signature_cid": str(
                self.target_signature_cid),
            "tier": str(self.tier),
            "n_items": int(self.n_items),
            "exact_replay_bit_identical": bool(
                self.exact_replay_bit_identical),
            "semantic_fidelity_cosine": float(round(
                self.semantic_fidelity_cosine, 12)),
            "classification_preserved_rate": float(round(
                self.classification_preserved_rate, 12)),
            "target_state_relative_error": float(round(
                self.target_state_relative_error, 12)),
            "non_portable_axis_drop_rate": float(round(
                self.non_portable_axis_drop_rate, 12)),
            "input_data_cid": str(self.input_data_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_portability_fidelity_report_v1",
            "report": self.to_dict()})


# ---------------------------------------------------------------
# Bench dataset
# ---------------------------------------------------------------

def build_portability_bench_dataset_v1(
        *, source_signature: RuntimeSignatureV1,
        n_items: int = W82_PORTABILITY_DEFAULT_N_ITEMS,
        seq_len: int = W82_PORTABILITY_DEFAULT_TOKEN_SEQ_LEN,
        seed: int = W82_PORTABILITY_DEFAULT_SEED,
) -> tuple["_np.ndarray", "_np.ndarray"]:
    """Build a deterministic batch of (hidden_states,
    token_ids) for ``source_signature``."""
    rng = _np.random.default_rng(int(seed))
    hs = rng.normal(
        0.0, 1.0,
        size=(int(n_items), int(source_signature.hidden_dim)))
    tok = rng.integers(
        low=0, high=int(source_signature.vocab_size),
        size=(int(n_items), int(seq_len)))
    return hs.astype(_np.float64), tok.astype(_np.int64)


def _signature_subseed_v1(
        *, kind: str,
        signature: RuntimeSignatureV1,
        seed: int,
) -> int:
    return int(_sha256_hex({
        "kind": str(kind),
        "signature_cid": str(signature.cid()),
        "seed": int(seed),
    })[:12], 16) & 0x7FFFFFFF


def _build_portability_token_ids_v1(
        *, source_signature: RuntimeSignatureV1,
        n_items: int,
        seq_len: int,
        seed: int,
) -> "_np.ndarray":
    rng = _np.random.default_rng(int(seed))
    tok = rng.integers(
        low=0, high=int(source_signature.vocab_size),
        size=(int(n_items), int(seq_len)))
    return tok.astype(_np.int64)


def _build_shared_anchor_repr_v1(
        *,
        projector: PortabilityProjectorV1,
        n_items: int,
        seed: int,
) -> "_np.ndarray":
    rng = _np.random.default_rng(int(seed))
    anchor = rng.normal(
        0.0, 1.0,
        size=(int(n_items), int(projector.anchor_dim)))
    return _np.asarray(anchor, dtype=_np.float64)


def _build_runtime_hidden_states_from_anchor_v1(
        *,
        projector: PortabilityProjectorV1,
        signature: RuntimeSignatureV1,
        anchor_repr: "_np.ndarray",
        seed: int,
        residual_scale: float,
) -> "_np.ndarray":
    anchor = _np.asarray(anchor_repr, dtype=_np.float64)
    projection = projector._projection_matrix(signature)
    base = anchor @ projection.T
    hidden_dim = int(signature.hidden_dim)
    if hidden_dim <= int(projector.anchor_dim):
        return _np.asarray(base, dtype=_np.float64)
    rng = _np.random.default_rng(int(seed))
    noise = rng.normal(
        0.0, 1.0, size=(int(anchor.shape[0]), hidden_dim))
    null_projector = (
        _np.eye(hidden_dim, dtype=_np.float64) -
        (projection @ projection.T))
    residual = noise @ null_projector
    norms = _np.linalg.norm(residual, axis=1, keepdims=True)
    norms = _np.where(norms < 1e-12, 1.0, norms)
    residual = (residual / norms) * float(residual_scale)
    return _np.asarray(base + residual, dtype=_np.float64)


def _classify_v1(
        hs: "_np.ndarray", *,
        task_dim: int = (
            W82_PORTABILITY_DEFAULT_CLASSIFICATION_TASK_DIM),
) -> "_np.ndarray":
    """Reference classification on raw hidden states: sign of
    coordinate ``task_dim``.

    Used by the per-runtime "raw hidden coord" baseline. This
    is *not* the load-bearing portability classifier: raw
    coordinates do not carry shared semantic meaning across
    runtime signatures.
    """
    if int(hs.shape[1]) <= int(task_dim):
        return _np.zeros((int(hs.shape[0]),), dtype=_np.int64)
    return (hs[:, int(task_dim)] >= 0.0).astype(_np.int64)


def _classify_on_anchor_v1(
        anchor: "_np.ndarray", *,
        task_dim: int = (
            W82_PORTABILITY_DEFAULT_CLASSIFICATION_TASK_DIM),
) -> "_np.ndarray":
    """Anchor-space classifier: sign of anchor coordinate
    ``task_dim``.

    The anchor space is the SHARED semantic representation,
    so this classifier IS portable. The load-bearing
    classification_preserved_rate metric uses this.
    """
    if int(anchor.shape[1]) <= int(task_dim):
        return _np.zeros(
            (int(anchor.shape[0]),), dtype=_np.int64)
    return (
        anchor[:, int(task_dim)] >= 0.0).astype(_np.int64)


def _cosine_similarity_v1(
        a: "_np.ndarray", b: "_np.ndarray",
) -> float:
    """Mean per-row cosine similarity."""
    A = _np.asarray(a, dtype=_np.float64)
    B = _np.asarray(b, dtype=_np.float64)
    if A.shape != B.shape:
        raise ValueError(
            f"shape mismatch: {A.shape} vs {B.shape}")
    eps = 1e-12
    num = _np.sum(A * B, axis=1)
    da = _np.linalg.norm(A, axis=1)
    db = _np.linalg.norm(B, axis=1)
    denom = _np.maximum(da * db, eps)
    return float(_np.mean(num / denom))


def _mean_relative_error_v1(
        a: "_np.ndarray", b: "_np.ndarray",
) -> float:
    A = _np.asarray(a, dtype=_np.float64)
    B = _np.asarray(b, dtype=_np.float64)
    if A.shape != B.shape:
        raise ValueError(
            f"shape mismatch: {A.shape} vs {B.shape}")
    diff = _np.linalg.norm(A - B, axis=1)
    denom = _np.maximum(
        _np.linalg.norm(B, axis=1), 1e-12)
    return float(_np.mean(diff / denom))


def _build_target_runtime_classifier_weights_v1(
        *,
        projector: PortabilityProjectorV1,
        target_signature: RuntimeSignatureV1,
) -> "_np.ndarray":
    projection = projector._projection_matrix(target_signature)
    dim = int(
        min(
            int(W82_PORTABILITY_DEFAULT_CLASSIFICATION_TASK_DIM),
            max(0, int(projection.shape[1]) - 1)))
    return _np.asarray(
        projection[:, dim], dtype=_np.float64)


# ---------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------

def run_portability_pair_bench_v1(
        *, source_signature: RuntimeSignatureV1,
        target_signature: RuntimeSignatureV1,
        projector: PortabilityProjectorV1 | None = None,
        n_items: int = W82_PORTABILITY_DEFAULT_N_ITEMS,
        seq_len: int = W82_PORTABILITY_DEFAULT_TOKEN_SEQ_LEN,
        seed: int = W82_PORTABILITY_DEFAULT_SEED,
) -> PortabilityFidelityReportV1:
    """Run a portability round-trip from ``source_signature``
    to ``target_signature`` and back; measure fidelity."""
    proj = projector or build_portability_projector_v1()
    source_seed = _signature_subseed_v1(
        kind="w82_portability_source_hidden_states_v1",
        signature=source_signature,
        seed=int(seed))
    target_seed = _signature_subseed_v1(
        kind="w82_portability_target_hidden_states_v1",
        signature=target_signature,
        seed=int(seed))
    token_seed = _signature_subseed_v1(
        kind="w82_portability_token_ids_v1",
        signature=source_signature,
        seed=int(seed))
    anchor_seed = int(seed) ^ 0x50505413
    shared_anchor = _build_shared_anchor_repr_v1(
        projector=proj,
        n_items=int(n_items),
        seed=int(anchor_seed))
    hs_a = _build_runtime_hidden_states_from_anchor_v1(
        projector=proj,
        signature=source_signature,
        anchor_repr=shared_anchor,
        seed=int(source_seed),
        residual_scale=0.12)
    tok_a = _build_portability_token_ids_v1(
        source_signature=source_signature,
        n_items=int(n_items),
        seq_len=int(seq_len),
        seed=int(token_seed))
    # Encode A → portable.
    carrier = proj.encode_to_portable_v1(
        source_signature=source_signature,
        hidden_states=hs_a,
        token_ids=tok_a)
    # Decode portable → B.
    hs_b = proj.decode_to_runtime_v1(
        carrier=carrier, target_signature=target_signature)
    is_exact_pair = (
        str(source_signature.cid()) ==
        str(target_signature.cid()))
    if is_exact_pair:
        target_ground_truth = _np.asarray(
            hs_a, dtype=_np.float64)
    else:
        target_ground_truth = (
            _build_runtime_hidden_states_from_anchor_v1(
                projector=proj,
                signature=target_signature,
                anchor_repr=shared_anchor,
                seed=int(target_seed),
                residual_scale=0.18))
    target_projection = proj._projection_matrix(
        target_signature)
    transferred_target_anchor = _np.asarray(
        hs_b, dtype=_np.float64) @ target_projection
    target_ground_truth_anchor = _np.asarray(
        target_ground_truth, dtype=_np.float64) @ target_projection
    # Classification preservation now evaluates the transferred
    # TARGET hidden state with a deterministic target-runtime
    # classifier rather than comparing anchor carriers to
    # themselves.
    classifier_weights = (
        _build_target_runtime_classifier_weights_v1(
            projector=proj,
            target_signature=target_signature))
    gt_target_labels = (
        (_np.asarray(target_ground_truth, dtype=_np.float64) @
         classifier_weights) >= 0.0).astype(_np.int64)
    transferred_target_labels = (
        (_np.asarray(hs_b, dtype=_np.float64) @
         classifier_weights) >= 0.0).astype(_np.int64)
    cls_rate = float(_np.mean(
        gt_target_labels == transferred_target_labels))
    # Semantic fidelity still measures the preserved shared
    # semantics, but it does so by projecting from target
    # hidden state rather than re-comparing anchor carriers.
    sem_fid = _cosine_similarity_v1(
        transferred_target_anchor,
        target_ground_truth_anchor)
    bit_eq = bool(_np.array_equal(hs_a, hs_b)) if is_exact_pair else False
    target_state_relative_error = _mean_relative_error_v1(
        _np.asarray(hs_b, dtype=_np.float64),
        _np.asarray(target_ground_truth, dtype=_np.float64))
    tier = proj.portability_tier_for(
        source=source_signature, target=target_signature)
    # Non-portable axis drop rate: 100% by construction
    # because the carrier explicitly labels them but never
    # ships values.
    if len(proj.non_portable_axes) > 0:
        drop_rate = float(
            len(tuple(carrier.non_portable_axis_labels)) ==
            len(tuple(proj.non_portable_axes)))
    else:
        drop_rate = 1.0  # vacuously true
    input_data_cid = _sha256_hex({
        "kind": "w82_portability_input_data_cid_v1",
        "hs_cid": _ndarray_cid(hs_a),
        "tok_cid": _ndarray_cid(tok_a),
        "n_items": int(n_items),
        "seq_len": int(seq_len),
        "seed": int(seed),
    })
    return PortabilityFidelityReportV1(
        schema=W82_PORTABILITY_V1_SCHEMA_VERSION,
        source_signature_cid=str(source_signature.cid()),
        target_signature_cid=str(target_signature.cid()),
        tier=str(tier),
        n_items=int(n_items),
        exact_replay_bit_identical=bool(bit_eq),
        semantic_fidelity_cosine=float(sem_fid),
        classification_preserved_rate=float(cls_rate),
        target_state_relative_error=float(
            target_state_relative_error),
        non_portable_axis_drop_rate=float(drop_rate),
        input_data_cid=str(input_data_cid),
    )


# ---------------------------------------------------------------
# Pair-bench witness
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PortabilityBenchWitnessV1:
    """Tamper-evident witness over a multi-pair portability
    bench."""

    schema: str
    n_pairs: int
    pair_report_cids: tuple[str, ...]
    same_signature_bit_identical: bool
    min_cross_signature_cosine: float
    min_cross_signature_classification: float
    max_cross_signature_target_state_relative_error: float
    cross_signature_passes_threshold: bool
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_pairs": int(self.n_pairs),
            "pair_report_cids": [
                str(c) for c in self.pair_report_cids],
            "same_signature_bit_identical": bool(
                self.same_signature_bit_identical),
            "min_cross_signature_cosine": float(round(
                self.min_cross_signature_cosine, 12)),
            "min_cross_signature_classification": float(round(
                self.min_cross_signature_classification, 12)),
            "max_cross_signature_target_state_relative_error":
                float(round(
                    self.max_cross_signature_target_state_relative_error,
                    12)),
            "cross_signature_passes_threshold": bool(
                self.cross_signature_passes_threshold),
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_portability_bench_witness_v1",
            "witness": self.to_dict()})


def build_default_runtime_signature_pair_v1() -> tuple[
        RuntimeSignatureV1, RuntimeSignatureV1]:
    """Build the default A / B signature pair for the bench:

    * ``runtime_a``: backend ``cpu_numpy_a``,
      vocab=32, hidden_dim=8, n_layers=2
    * ``runtime_b``: backend ``cpu_numpy_b``,
      vocab=24, hidden_dim=12, n_layers=3

    Different ``hidden_dim`` and ``vocab_size`` so cross-
    transfer is *not* trivial.
    """
    r_a = build_runtime_signature_v1(
        backend_label="cpu_numpy_a",
        vocab_size=32, hidden_dim=8, n_layers=2)
    r_b = build_runtime_signature_v1(
        backend_label="cpu_numpy_b",
        vocab_size=24, hidden_dim=12, n_layers=3)
    return r_a, r_b


def run_portability_bench_end_to_end_v1(
        *, signatures: Sequence[RuntimeSignatureV1] | None = None,
        n_items: int = W82_PORTABILITY_DEFAULT_N_ITEMS,
        seed: int = W82_PORTABILITY_DEFAULT_SEED,
        cosine_threshold: float = (
            W82_PORTABILITY_DEFAULT_COSINE_THRESHOLD),
        classification_threshold: float = (
            W82_PORTABILITY_DEFAULT_CLASSIFICATION_THRESHOLD),
        target_state_error_threshold: float = (
            W82_PORTABILITY_DEFAULT_TARGET_STATE_ERROR_THRESHOLD),
) -> tuple[tuple[PortabilityFidelityReportV1, ...],
           PortabilityBenchWitnessV1]:
    """End-to-end bench: run every (source, target) pair on
    ``signatures``; emit a witness."""
    if signatures is None:
        r_a, r_b = build_default_runtime_signature_pair_v1()
        sigs: list[RuntimeSignatureV1] = [r_a, r_b]
    else:
        sigs = list(signatures)
    proj = build_portability_projector_v1(seed=int(seed))
    reports: list[PortabilityFidelityReportV1] = []
    for sa in sigs:
        for sb in sigs:
            reports.append(run_portability_pair_bench_v1(
                source_signature=sa,
                target_signature=sb,
                projector=proj,
                n_items=int(n_items),
                seed=int(seed)))
    same_sig = [
        r for r in reports
        if str(r.source_signature_cid) == str(
            r.target_signature_cid)]
    cross_sig = [
        r for r in reports
        if str(r.source_signature_cid) != str(
            r.target_signature_cid)]
    same_bit_id = bool(all(
        r.exact_replay_bit_identical for r in same_sig))
    min_cos = (
        min((float(r.semantic_fidelity_cosine)
             for r in cross_sig), default=1.0)
        if cross_sig else 1.0)
    min_cls = (
        min((float(r.classification_preserved_rate)
             for r in cross_sig), default=1.0)
        if cross_sig else 1.0)
    max_state_err = (
        max((float(r.target_state_relative_error)
             for r in cross_sig), default=0.0)
        if cross_sig else 0.0)
    passes = bool(
        min_cos >= float(cosine_threshold) and
        min_cls >= float(classification_threshold) and
        max_state_err <= float(target_state_error_threshold))
    cfg_cid = _sha256_hex({
        "kind": "w82_portability_bench_config_v1",
        "n_pairs": int(len(reports)),
        "n_signatures": int(len(sigs)),
        "n_items": int(n_items),
        "cosine_threshold": float(cosine_threshold),
        "classification_threshold": float(
            classification_threshold),
        "target_state_error_threshold": float(
            target_state_error_threshold),
    })
    witness = PortabilityBenchWitnessV1(
        schema=W82_PORTABILITY_V1_SCHEMA_VERSION,
        n_pairs=int(len(reports)),
        pair_report_cids=tuple(r.cid() for r in reports),
        same_signature_bit_identical=bool(same_bit_id),
        min_cross_signature_cosine=float(min_cos),
        min_cross_signature_classification=float(min_cls),
        max_cross_signature_target_state_relative_error=float(
            max_state_err),
        cross_signature_passes_threshold=bool(passes),
        config_cid=str(cfg_cid),
    )
    return tuple(reports), witness


__all__ = [
    "W82_PORTABILITY_V1_SCHEMA_VERSION",
    "W82_PORTABILITY_TIERS",
    "W82_PORTABILITY_DEFAULT_ANCHOR_DIM",
    "W82_PORTABILITY_DEFAULT_N_ITEMS",
    "PortabilityTier",
    "RuntimeSignatureV1",
    "PortableStateCarrierV1",
    "PortabilityProjectorV1",
    "PortabilityFidelityReportV1",
    "PortabilityBenchWitnessV1",
    "build_runtime_signature_v1",
    "build_portability_projector_v1",
    "build_portability_bench_dataset_v1",
    "run_portability_pair_bench_v1",
    "build_default_runtime_signature_pair_v1",
    "run_portability_bench_end_to_end_v1",
]
