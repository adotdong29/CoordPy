"""W87 / P3 #46 — Multi-Modal Context Substrate V1.

The W56–W83 capsule schema assumed *text payloads*: prompts are
strings, hidden states are float arrays derived from text token
embeddings, audit chains hash text bytes.  Production multi-agent
teams increasingly handle *multi-modal* context (vision, code,
audio, structured data).

V1 extends the W80 instrumentation contract + the W82+W83 capsule
schemas to carry multi-modal payloads.  This module ships the
shared dataclasses and primitive operations.  Modality-specific
substrate adapters live in:

  * ``coordpy.vision_substrate_v1`` (VLM hidden-state reads)
  * ``coordpy.code_substrate_v1``   (code LM + AST-aware reads)
  * (text reuses ``coordpy.transformers_runtime_v1``)

The composed multi-modal pipeline lives in
``coordpy.composed_multimodal_pipeline_v1``.

Honest scope (W87)
------------------

* ``W87-L-MULTI-MODAL-V1-TEXT-IMAGE-CODE-CAP`` — V1 covers the
  three modalities the issue body's V1 scope names: ``text``,
  ``image``, ``code``.  Audio and tabular are V2.
* ``W87-L-MULTI-MODAL-V1-CONTENT-ADDRESSED-CAP`` — every payload
  has an explicit ``payload_cid`` over the *bytes that travel*
  (the raw bytes for image, the source text for code, the prompt
  text for text); the encoder is identified by its
  ``encoder_cid`` (typically the model name + revision + dtype).
* ``W87-L-MULTI-MODAL-V1-NUMPY-EMBEDDINGS-CAP`` — embeddings are
  carried as numpy arrays.  Cross-modality byte-identity is
  computed at fp64 (the W80 convention).  The per-modality
  *precision floor* tracks where empirical byte-identity actually
  holds for the modality's encoder (fp32 for image encoders,
  fp32 for text encoders, fp32 for code encoders by default —
  see ``ModalityPrecisionFloorV1``).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_modal_payload_v1 requires numpy"
    ) from exc


W87_MULTI_MODAL_V1_SCHEMA_VERSION: str = (
    "coordpy.multi_modal_payload_v1.v1")


class Modality(str, enum.Enum):
    """The closed V1 modality set.  Audio + tabular are V2."""

    TEXT = "text"
    IMAGE = "image"
    CODE = "code"


W87_MODALITIES_V1: tuple[str, ...] = tuple(m.value for m in Modality)


# Per-modality precision floor — *empirical* byte-identity
# tolerance for replay-from-embedding.  Numbers calibrated against
# the canonical adapter for each modality (see the matching
# substrate module).
W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32: dict[str, float] = {
    Modality.TEXT.value:  5e-3,
    Modality.IMAGE.value: 5e-3,
    Modality.CODE.value:  5e-3,
}

# Per-modality precision floor at bf16 (for VLM/code models run
# at bf16; bigger tolerance reflects the wider numerical noise).
W87_MODALITY_PRECISION_FLOOR_DEFAULTS_BF16: dict[str, float] = {
    Modality.TEXT.value:  5e-1,
    Modality.IMAGE.value: 5e-1,
    Modality.CODE.value:  5e-1,
}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes_hex(b: bytes) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


def _ndarray_cid_fp32(arr: "_np.ndarray | None") -> str:
    """Content-address an ndarray at fp32 (the W80 cross-runtime
    convention).  Returns ``"none"`` for None."""
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _ndarray_cid_fp64(arr: "_np.ndarray | None") -> str:
    """Content-address an ndarray at fp64 (audit chain canonical)."""
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _normalise_modality(m: str | Modality) -> str:
    """Coerce / validate a modality string."""
    if isinstance(m, Modality):
        return m.value
    s = str(m)
    if s not in W87_MODALITIES_V1:
        raise ValueError(
            f"unknown modality {m!r}; must be one of "
            f"{W87_MODALITIES_V1}")
    return s


# ---------------------------------------------------------------
# Encoder fingerprint
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EncoderFingerprintV1:
    """Identifies the encoder that produced an embedding.

    The fingerprint is content-addressed via ``cid()`` and travels
    with every multi-modal payload so an auditor can re-derive
    the embedding's provenance.
    """

    schema: str
    modality: str
    encoder_kind: str  # "hf_causal_lm", "hf_vlm", "stub"
    model_name: str
    model_revision: str
    precision_tier: str  # "tier_fp32" / "tier_bf16" / "tier_int8"
    embedding_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "modality": str(self.modality),
            "encoder_kind": str(self.encoder_kind),
            "model_name": str(self.model_name),
            "model_revision": str(self.model_revision),
            "precision_tier": str(self.precision_tier),
            "embedding_dim": int(self.embedding_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_encoder_fingerprint_v1",
            "fingerprint": self.to_dict(),
        })


# ---------------------------------------------------------------
# MultiModalPayloadV1 (the core dataclass)
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MultiModalPayloadV1:
    """A content-addressed multi-modal payload.

    Each payload carries:

      * ``modality`` — one of W87_MODALITIES_V1.
      * ``raw_bytes`` — the bytes that travel (image bytes,
        utf-8 source text, prompt text).
      * ``raw_bytes_cid`` — SHA-256 of ``raw_bytes``.
      * ``embedding`` — numpy fp32 array of shape (n_tokens or
        n_patches, embedding_dim).  ``None`` if the encoder is
        deferred (the payload identifies what to encode but
        hasn't been run through the encoder yet).
      * ``embedding_cid`` — SHA-256 of the fp32 contiguous bytes
        of ``embedding``.  ``"none"`` when ``embedding`` is None.
      * ``encoder`` — the ``EncoderFingerprintV1`` that produced
        the embedding (None when embedding is None).
      * ``extras`` — modality-specific structured metadata
        (e.g. ``{"ast_function_boundaries": [...]}`` for code).
        Values must be JSON-serialisable.

    The Merkle leaf for this payload is ``payload_cid()`` — a
    hash over modality + raw_bytes_cid + embedding_cid +
    encoder_cid + extras.  This is the load-bearing identity
    that the cross-modality Merkle root anchors against.
    """

    schema: str
    modality: str
    raw_bytes: bytes
    raw_bytes_cid: str
    embedding: "_np.ndarray | None"
    embedding_cid: str
    encoder: EncoderFingerprintV1 | None
    extras: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Validate modality.
        if str(self.modality) not in W87_MODALITIES_V1:
            raise ValueError(
                f"unknown modality {self.modality!r}; "
                f"must be one of {W87_MODALITIES_V1}")
        # Validate raw_bytes_cid matches raw_bytes.
        recomputed = _sha256_bytes_hex(bytes(self.raw_bytes))
        if str(self.raw_bytes_cid) != recomputed:
            raise ValueError(
                f"raw_bytes_cid mismatch: declared "
                f"{self.raw_bytes_cid!r} but bytes hash to "
                f"{recomputed!r}")
        # Validate embedding_cid matches embedding.
        if self.embedding is None:
            if str(self.embedding_cid) != "none":
                raise ValueError(
                    f"embedding is None but embedding_cid is "
                    f"{self.embedding_cid!r}")
            if self.encoder is not None:
                raise ValueError(
                    "embedding is None but encoder fingerprint "
                    "is set; either remove the encoder or "
                    "compute the embedding")
        else:
            recomputed_e = _ndarray_cid_fp32(self.embedding)
            if str(self.embedding_cid) != recomputed_e:
                raise ValueError(
                    f"embedding_cid mismatch: declared "
                    f"{self.embedding_cid!r} but array hashes "
                    f"to {recomputed_e!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "modality": str(self.modality),
            "raw_bytes_cid": str(self.raw_bytes_cid),
            "raw_bytes_size": int(len(self.raw_bytes)),
            "embedding_cid": str(self.embedding_cid),
            "embedding_shape": (
                list(int(s) for s in self.embedding.shape)
                if self.embedding is not None else []),
            "encoder_cid": (
                str(self.encoder.cid())
                if self.encoder is not None else "none"),
            "extras": dict(self.extras),
        }

    def payload_cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_multi_modal_payload_v1",
            "payload": self.to_dict(),
        })


def build_multi_modal_payload_v1(
        *, modality: str | Modality,
        raw_bytes: bytes,
        embedding: "_np.ndarray | None" = None,
        encoder: EncoderFingerprintV1 | None = None,
        extras: Mapping[str, Any] | None = None,
) -> MultiModalPayloadV1:
    """Construct a content-addressed multi-modal payload."""
    m = _normalise_modality(modality)
    rb = bytes(raw_bytes)
    rb_cid = _sha256_bytes_hex(rb)
    if embedding is None:
        e_cid = "none"
        enc = None
    else:
        e_cid = _ndarray_cid_fp32(embedding)
        if encoder is None:
            raise ValueError(
                "embedding supplied without encoder fingerprint; "
                "every embedding must identify its encoder")
        enc = encoder
    return MultiModalPayloadV1(
        schema=W87_MULTI_MODAL_V1_SCHEMA_VERSION,
        modality=m,
        raw_bytes=rb,
        raw_bytes_cid=rb_cid,
        embedding=embedding,
        embedding_cid=e_cid,
        encoder=enc,
        extras=dict(extras or {}),
    )


# ---------------------------------------------------------------
# Cross-modality Merkle root
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CrossModalityMerkleRootV1:
    """Merkle root spanning multiple multi-modal payloads.

    The leaves are the per-payload ``payload_cid()`` values, in a
    canonical (modality, payload_index) order.  The root is the
    SHA-256 of the sorted-tuple of leaves under a fixed kind tag.
    This is intentionally *simpler* than the
    W82 ``MerkleHashTreeV1`` — V1 uses a single-level "Merkle
    leaf-flattening" because the inclusion-path machinery for
    multi-modal payloads is V2; the identity property
    (root commits to every payload byte-for-byte) is what V1
    needs.
    """

    schema: str
    per_modality_payload_cids: tuple[tuple[str, str], ...]
    # ^ Each entry is (modality, payload_cid).
    root_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "per_modality_payload_cids": [
                [str(m), str(c)] for m, c in (
                    self.per_modality_payload_cids)
            ],
            "n_modalities": int(len(set(
                m for m, _ in self.per_modality_payload_cids))),
            "n_leaves": int(len(self.per_modality_payload_cids)),
            "root_cid": str(self.root_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_cross_modality_merkle_root_v1",
            "merkle": self.to_dict(),
        })


def build_cross_modality_merkle_root_v1(
        payloads: Sequence[MultiModalPayloadV1],
) -> CrossModalityMerkleRootV1:
    """Build the cross-modality Merkle root anchoring every
    payload's content-address in a single hash."""
    leaves: list[tuple[str, str]] = []
    for p in payloads:
        leaves.append((str(p.modality), str(p.payload_cid())))
    # Canonical sort: (modality, payload_cid) ascending.
    leaves_sorted = sorted(leaves, key=lambda t: (t[0], t[1]))
    root = _sha256_hex({
        "kind": "w87_cross_modality_merkle_root_leaves_v1",
        "leaves": [[m, c] for m, c in leaves_sorted],
    })
    return CrossModalityMerkleRootV1(
        schema=W87_MULTI_MODAL_V1_SCHEMA_VERSION,
        per_modality_payload_cids=tuple(leaves_sorted),
        root_cid=str(root),
    )


# ---------------------------------------------------------------
# Per-modality precision-floor report
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ModalityPrecisionFloorV1:
    """Empirical precision floor for one modality's encoder.

    ``floor_fp32`` is the measured ``max_abs_diff`` between
    embedding bytes computed twice (or computed once and replayed
    from the raw-bytes via the encoder) at fp32.  ``floor_bf16``
    is the same at bf16 (NaN if not tested at bf16).

    ``tolerance_fp32`` / ``tolerance_bf16`` are the CONTRACT
    tolerances; the empirical floor must be <= these for the
    modality to satisfy the per-modality precision-floor DoD.
    """

    schema: str
    modality: str
    encoder_cid: str
    n_samples: int
    floor_fp32: float
    floor_bf16: float
    tolerance_fp32: float
    tolerance_bf16: float
    floor_within_tolerance_fp32: bool
    floor_within_tolerance_bf16: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "modality": str(self.modality),
            "encoder_cid": str(self.encoder_cid),
            "n_samples": int(self.n_samples),
            "floor_fp32": float(self.floor_fp32),
            "floor_bf16": float(self.floor_bf16),
            "tolerance_fp32": float(self.tolerance_fp32),
            "tolerance_bf16": float(self.tolerance_bf16),
            "floor_within_tolerance_fp32": bool(
                self.floor_within_tolerance_fp32),
            "floor_within_tolerance_bf16": bool(
                self.floor_within_tolerance_bf16),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_modality_precision_floor_v1",
            "floor": self.to_dict(),
        })


def measure_modality_precision_floor_fp32_v1(
        *, modality: str | Modality,
        encoder_cid: str,
        embedding_a: "_np.ndarray",
        embedding_b: "_np.ndarray",
        tolerance: float | None = None,
) -> ModalityPrecisionFloorV1:
    """Compare two embeddings (e.g. computed twice or
    computed-then-replayed) and report the max-abs-diff at fp32
    versus the per-modality tolerance."""
    m = _normalise_modality(modality)
    if tolerance is None:
        tolerance = float(
            W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32.get(
                m, 5e-3))
    a = _np.asarray(embedding_a, dtype=_np.float32)
    b = _np.asarray(embedding_b, dtype=_np.float32)
    if a.shape != b.shape:
        raise ValueError(
            f"embedding shape mismatch: {a.shape} != {b.shape}")
    diff = float(_np.max(_np.abs(a - b)))
    return ModalityPrecisionFloorV1(
        schema=W87_MULTI_MODAL_V1_SCHEMA_VERSION,
        modality=m,
        encoder_cid=str(encoder_cid),
        n_samples=int(a.size),
        floor_fp32=float(diff),
        floor_bf16=float("nan"),
        tolerance_fp32=float(tolerance),
        tolerance_bf16=float(
            W87_MODALITY_PRECISION_FLOOR_DEFAULTS_BF16.get(
                m, 5e-1)),
        floor_within_tolerance_fp32=bool(
            float(diff) <= float(tolerance)),
        floor_within_tolerance_bf16=False,
    )


# ---------------------------------------------------------------
# BlockedOnHardwareError
# ---------------------------------------------------------------

class BlockedOnHardwareError(RuntimeError):
    """Raised when a substrate adapter cannot run because the
    required runtime / device / model is not present on this
    host.  Carries a structured message documenting *exactly*
    what is missing so the user can fix it explicitly.

    The W84 ``frontier_capability_probe_v1`` module raises this
    same exception type; W87 reuses the pattern.
    """

    def __init__(
            self, modality: str, missing: Sequence[str],
            *, hint: str = ""):
        super().__init__(
            f"W87 {modality} substrate blocked on hardware: "
            f"missing={sorted(set(missing))}; "
            f"hint={hint!r}")
        self.modality = str(modality)
        self.missing = tuple(sorted(set(str(x) for x in missing)))
        self.hint = str(hint)


__all__ = [
    "W87_MULTI_MODAL_V1_SCHEMA_VERSION",
    "W87_MODALITIES_V1",
    "W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32",
    "W87_MODALITY_PRECISION_FLOOR_DEFAULTS_BF16",
    "Modality",
    "EncoderFingerprintV1",
    "MultiModalPayloadV1",
    "CrossModalityMerkleRootV1",
    "ModalityPrecisionFloorV1",
    "BlockedOnHardwareError",
    "build_multi_modal_payload_v1",
    "build_cross_modality_merkle_root_v1",
    "measure_modality_precision_floor_fp32_v1",
]
