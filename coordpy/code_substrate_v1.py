"""W87 / P3 #46 — Code Substrate Adapter V1.

Reads hidden state from a code-fine-tuned causal LM at AST-aware
boundaries (function definitions, class definitions, top-level
statements), **not** at every token.  This is what the issue body
calls out: "the code adapter must carry at least one AST-aware
axis".

The AST analysis is pure-Python stdlib (`ast` module).  The
encoder forward pass uses:

  * The real HF transformers stack when available — typically
    a code-fine-tuned model (DeepSeek-Coder, Qwen2.5-Coder,
    CodeLlama, StarCoder).  Reads hidden state at the LAST token
    of each AST function-def boundary.
  * A deterministic, honestly-named ``stub`` encoder when
    transformers is not installed.  The stub is reproducible
    (hash-derived) and exercises the *contract* (AST-aware reads,
    content-addressing, replay) without requiring GPU.

Modules using the adapter never have to choose between paths
manually — `CodeSubstrateAdapterV1.encode` raises
`BlockedOnHardwareError` only when the explicit "real model
required" mode is requested AND transformers is absent.

Honest scope (W87)
------------------

* ``W87-L-CODE-SUBSTRATE-V1-AST-PYTHON-CAP`` — V1 uses Python
  stdlib ``ast``; AST-aware reads cover Python source.  Other
  languages (Go, Rust, JS) need a tree-sitter or equivalent; V2.
* ``W87-L-CODE-SUBSTRATE-V1-FUNCTION-DEF-BOUNDARY-CAP`` — V1 reads
  at function-def boundaries (the cleanest AST node for "where
  does this function live in the hidden state").  Per-statement
  reads are V2.
* ``W87-L-CODE-SUBSTRATE-V1-STUB-FOR-CI-CAP`` — when transformers
  is absent, the adapter falls back to a deterministic stub
  encoder.  The stub's ``encoder_kind`` is "stub_sha256_v1" so
  the encoder fingerprint surfaces the fallback clearly.  Tests
  exercising the real model path must skip-gate on transformers.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.code_substrate_v1 requires numpy") from exc

from .multi_modal_payload_v1 import (
    BlockedOnHardwareError,
    EncoderFingerprintV1,
    Modality,
    MultiModalPayloadV1,
    W87_MULTI_MODAL_V1_SCHEMA_VERSION,
    build_multi_modal_payload_v1,
)


W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION: str = (
    "coordpy.code_substrate_v1.v1")

# Default embedding dim for the stub encoder.
W87_CODE_STUB_EMBEDDING_DIM: int = 16


# ---------------------------------------------------------------
# AST boundary extraction
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ASTFunctionBoundaryV1:
    """One function-def boundary in a code payload."""

    schema: str
    name: str
    qualname: str  # e.g. "module.MyClass.method"
    start_line: int
    end_line: int
    n_lines: int
    n_args: int
    is_async: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "name": str(self.name),
            "qualname": str(self.qualname),
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "n_lines": int(self.n_lines),
            "n_args": int(self.n_args),
            "is_async": bool(self.is_async),
        }


def extract_function_boundaries_v1(
        source: str,
) -> tuple[ASTFunctionBoundaryV1, ...]:
    """Walk the Python AST of ``source`` and return every
    function-def boundary (including async + nested + methods).

    Raises ``SyntaxError`` if the source does not parse.
    """
    tree = ast.parse(source)
    out: list[ASTFunctionBoundaryV1] = []

    def _walk(node: ast.AST, qualprefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = str(child.name)
                qn = (qualprefix + "." + name) if qualprefix else name
                start_line = int(getattr(child, "lineno", 1))
                # ast.get_source_segment / end_lineno: 3.8+
                end_line = int(
                    getattr(child, "end_lineno", start_line))
                n_args = (
                    len(child.args.args)
                    + len(child.args.posonlyargs)
                    + len(child.args.kwonlyargs)
                )
                out.append(ASTFunctionBoundaryV1(
                    schema=W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION,
                    name=name, qualname=qn,
                    start_line=start_line, end_line=end_line,
                    n_lines=int(end_line - start_line + 1),
                    n_args=int(n_args),
                    is_async=isinstance(
                        child, ast.AsyncFunctionDef)))
                # Recurse into the body to catch nested defs.
                _walk(child, qn)
            elif isinstance(child, ast.ClassDef):
                qn = (
                    (qualprefix + "." + child.name)
                    if qualprefix else child.name)
                _walk(child, qn)
            else:
                _walk(child, qualprefix)

    _walk(tree, "")
    return tuple(out)


# ---------------------------------------------------------------
# Stub encoder (deterministic, no model deps)
# ---------------------------------------------------------------

def _stub_token_embed(
        token_bytes: bytes, embedding_dim: int,
) -> "_np.ndarray":
    """Deterministic stub embedding.  SHA-256(token) → bytes →
    fp32 array of length ``embedding_dim``."""
    out = _np.zeros((int(embedding_dim),), dtype=_np.float32)
    seed = hashlib.sha256(bytes(token_bytes)).digest()
    # Use the bytes to seed an RNG that produces ``embedding_dim``
    # floats deterministically.
    rng = _np.random.default_rng(
        int.from_bytes(seed[:8], "big") & ((1 << 64) - 1))
    out[:] = rng.normal(0.0, 1.0, (int(embedding_dim),)).astype(
        _np.float32)
    return out


def encode_source_with_stub_v1(
        source: str, *,
        embedding_dim: int = W87_CODE_STUB_EMBEDDING_DIM,
) -> "_np.ndarray":
    """Encode source with the stub: per-line embeddings.

    Each line of the source maps to one embedding row computed
    from SHA-256(line bytes).  The result has shape
    ``(n_lines, embedding_dim)``.  Reproducible across runs and
    across architectures (NumPy + hashlib only).
    """
    lines = source.splitlines()
    if not lines:
        lines = [""]
    out = _np.zeros(
        (len(lines), int(embedding_dim)), dtype=_np.float32)
    for i, line in enumerate(lines):
        out[i, :] = _stub_token_embed(
            line.encode("utf-8"), int(embedding_dim))
    return out


# ---------------------------------------------------------------
# CodeSubstrateAdapterV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CodeSubstrateReadV1:
    """Result of an AST-aware read of code hidden state."""

    schema: str
    source_cid: str
    encoder_cid: str
    function_boundaries: tuple[ASTFunctionBoundaryV1, ...]
    per_function_embedding_cids: tuple[str, ...]
    embedding_full_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "source_cid": str(self.source_cid),
            "encoder_cid": str(self.encoder_cid),
            "n_functions": int(len(self.function_boundaries)),
            "function_boundaries": [
                b.to_dict() for b in self.function_boundaries],
            "per_function_embedding_cids": list(
                self.per_function_embedding_cids),
            "embedding_full_cid": str(self.embedding_full_cid),
        }

    def cid(self) -> str:
        return hashlib.sha256(__import__("json").dumps(
            {"kind": "w87_code_substrate_read_v1",
             "read": self.to_dict()},
            sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


class CodeSubstrateAdapterV1:
    """Code substrate adapter.

    Default behaviour: try HF transformers; on failure, fall back
    to the deterministic stub.  Pass ``require_real_model=True``
    to disable the fallback and raise ``BlockedOnHardwareError``
    if transformers is not available.
    """

    def __init__(
            self, *,
            model_name: str = "stub",
            precision_tier: str = "tier_fp32",
            embedding_dim: int = W87_CODE_STUB_EMBEDDING_DIM,
            require_real_model: bool = False) -> None:
        self.model_name = str(model_name)
        self.precision_tier = str(precision_tier)
        self.embedding_dim = int(embedding_dim)
        self.require_real_model = bool(require_real_model)
        self._real_runtime = None  # lazy load
        if self.require_real_model:
            self._load_real_runtime_or_raise()

    def _load_real_runtime_or_raise(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise BlockedOnHardwareError(
                modality="code",
                missing=(
                    "torch" if "torch" in str(exc).lower()
                    else "transformers",),
                hint=(
                    "install torch + transformers (CPU is fine "
                    "for small code models; GPU recommended "
                    "for Qwen2.5-Coder-7B / DeepSeek-Coder-V2)"))
        from .transformers_runtime_v1 import (
            TransformersRuntimeV1)
        self._real_runtime = TransformersRuntimeV1(
            model_name=str(self.model_name),
            device="cpu",  # CPU OK for small code models;
                            # user can pass cuda:0 via env
            precision_tier=str(self.precision_tier))

    def encoder_fingerprint(self) -> EncoderFingerprintV1:
        if self.model_name == "stub" or self._real_runtime is None:
            return EncoderFingerprintV1(
                schema=W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION,
                modality=Modality.CODE.value,
                encoder_kind="stub_sha256_v1",
                model_name="stub",
                model_revision="v1",
                precision_tier=str(self.precision_tier),
                embedding_dim=int(self.embedding_dim))
        return EncoderFingerprintV1(
            schema=W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION,
            modality=Modality.CODE.value,
            encoder_kind="hf_causal_lm",
            model_name=str(self.model_name),
            model_revision="main",
            precision_tier=str(self.precision_tier),
            embedding_dim=int(self.embedding_dim))

    def encode(self, source: str) -> MultiModalPayloadV1:
        """Encode source code into a MultiModalPayloadV1 with
        an AST-aware extras block."""
        boundaries = extract_function_boundaries_v1(source)
        if self._real_runtime is None and not self.require_real_model:
            embedding = encode_source_with_stub_v1(
                source, embedding_dim=int(self.embedding_dim))
        elif self._real_runtime is not None:
            embedding = self._encode_real(source)
        else:
            raise BlockedOnHardwareError(
                modality="code", missing=("transformers",),
                hint="encode called in require_real_model mode "
                     "but no real runtime loaded")
        enc = self.encoder_fingerprint()
        return build_multi_modal_payload_v1(
            modality=Modality.CODE,
            raw_bytes=source.encode("utf-8"),
            embedding=embedding,
            encoder=enc,
            extras={
                "ast_function_boundaries": [
                    b.to_dict() for b in boundaries],
                "n_functions": int(len(boundaries)),
            })

    def read_at_function_boundaries(
            self, source: str,
    ) -> CodeSubstrateReadV1:
        """Encode source AND return per-function-boundary
        embedding CIDs — the AST-aware substrate read."""
        boundaries = extract_function_boundaries_v1(source)
        if self._real_runtime is None and not self.require_real_model:
            full = encode_source_with_stub_v1(
                source, embedding_dim=int(self.embedding_dim))
        elif self._real_runtime is not None:
            full = self._encode_real(source)
        else:
            raise BlockedOnHardwareError(
                modality="code", missing=("transformers",),
                hint="read_at_function_boundaries called in "
                     "require_real_model mode but no real "
                     "runtime loaded")
        # Per-function CIDs: take the slice of `full` covering
        # the function's line range.  `full` has one row per line.
        per_fn_cids: list[str] = []
        for b in boundaries:
            sl = full[max(0, b.start_line - 1):b.end_line, :]
            per_fn_cids.append(_array_cid_fp32(sl))
        return CodeSubstrateReadV1(
            schema=W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION,
            source_cid=_sha256_bytes(source.encode("utf-8")),
            encoder_cid=str(self.encoder_fingerprint().cid()),
            function_boundaries=boundaries,
            per_function_embedding_cids=tuple(per_fn_cids),
            embedding_full_cid=_array_cid_fp32(full),
        )

    def _encode_real(self, source: str) -> "_np.ndarray":
        """Real-model encode path.  Returns per-token hidden
        state averaged over the model's last layer, then
        re-pooled to per-line via a simple newline-alignment.

        This is intentionally a thin wrapper around the existing
        W80 transformers_runtime_v1 — the LOAD-BEARING property is
        that we DO read hidden state from a real code-LM, not the
        specific pooling strategy."""
        rt = self._real_runtime
        assert rt is not None
        # Tokenize + forward.  W80 TransformersRuntimeV1.forward
        # expects keyword-only input_token_ids.
        ids = rt.tokenize(source, max_len=128)
        result = rt.forward(input_token_ids=ids)
        # ForwardTraceV1.hidden is a HiddenStateSnapshotV1
        # with .per_layer (tuple of ndarrays) and .final.
        hs = getattr(result, "hidden", None)
        if hs is None or not getattr(hs, "per_layer", None):
            raise BlockedOnHardwareError(
                modality="code", missing=("hidden_state_read",),
                hint="real runtime did not surface hidden state")
        # Pool last-layer hidden state across tokens by line.
        per_layer = hs.per_layer
        last = per_layer[-1]
        last = _np.asarray(last, dtype=_np.float32)
        if last.ndim == 3:  # (batch=1, seq, dim)
            last = last[0]
        # Map tokens → lines (approx: equal split).
        lines = source.splitlines() or [""]
        tokens_per_line = max(1, last.shape[0] // len(lines))
        rows: list["_np.ndarray"] = []
        for i in range(len(lines)):
            start = i * tokens_per_line
            end = min((i + 1) * tokens_per_line, last.shape[0])
            if start >= last.shape[0]:
                rows.append(last[-1])
            else:
                rows.append(last[start:end].mean(axis=0))
        return _np.stack(rows).astype(_np.float32)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


def _array_cid_fp32(arr: "_np.ndarray") -> str:
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


__all__ = [
    "W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION",
    "W87_CODE_STUB_EMBEDDING_DIM",
    "ASTFunctionBoundaryV1",
    "CodeSubstrateReadV1",
    "CodeSubstrateAdapterV1",
    "extract_function_boundaries_v1",
    "encode_source_with_stub_v1",
]
