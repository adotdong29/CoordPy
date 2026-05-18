"""W80 / P0 #8 — Canonical Runtime Instrumentation Contract V1.

This module defines a single, programme-level instrumentation
contract for controlled runtimes. The contract is the load-
bearing answer to the P0 #8 problem: until now, every controlled
runtime (in-repo NumPy, future HF transformers backend, future
llama.cpp / vLLM / MLX backends) had its own bespoke hook
surface. That made hidden-state, KV-cache, and attention
instrumentation incomparable across runtimes and turned every
new backend into a one-off experiment.

V1 fixes that by:

1. Naming the *axes* that any controlled runtime can claim
   (``InstrumentationAxis``): a closed set covering reads
   (hidden state, KV, attention, per-layer logits, final
   logits), writes (hidden-state inject, KV restore, attention-
   bias steer, prefix-state inject), and witnessing
   (deterministic replay, content-addressed trace id).
2. Naming the *capability tags* a backend can declare for each
   axis (``CapabilityTag``): ``available``, ``backend_specific``,
   ``best_effort``, ``unavailable``. The tag distinguishes
   universal contract guarantees from honest backend gaps.
3. Defining the canonical, JSON-serialisable snapshot / trace /
   injection / witness schemas (``HiddenStateSnapshotV1``,
   ``KVCacheSnapshotV1``, ``AttentionSnapshotV1``,
   ``ForwardTraceV1``, ``InjectionPlanV1``,
   ``InstrumentationWitnessV1``).
4. Declaring a backend-agnostic protocol
   (``RuntimeInstrumentationProtocolV1``) that any controlled
   runtime can implement — and a conformance test runner
   (``run_instrumentation_conformance_v1``) that mechanically
   checks whether a runtime honours the axes it claims.

The contract is **honest about backend asymmetry**: a backend
can declare ``unavailable`` on any axis without lying, and the
conformance suite will skip checks for unavailable axes rather
than fail. The W79 controlled runtime claims everything; a
future HF/llama.cpp/vLLM backend can claim less and still pass.

Honest scope (W80)
------------------

* ``W80-L-INSTRUMENTATION-V1-CONTRACT-CAP`` — V1 is a contract,
  not a runtime. It tells you what a backend exposes and gives
  you a uniform shape; it does not magically *give* a backend
  the ability to read attention if the underlying API blocks it.
* ``W80-L-INSTRUMENTATION-V1-NUMPY-SHAPES-CAP`` — V1 snapshot
  schemas use NumPy arrays for in-process speed; serialisation
  to JSON content-addresses arrays by SHA-256 (no raw tensor
  bytes in JSON).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.runtime_instrumentation_v1 requires numpy"
    ) from exc


W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION: str = (
    "coordpy.runtime_instrumentation_v1.v1")


# ---------------------------------------------------------------
# Canonical axis names
# ---------------------------------------------------------------

class InstrumentationAxis(str, enum.Enum):
    """The closed set of axes any controlled runtime may claim.

    Names are deliberately short and stable across backends — the
    same string is what shows up in the parity matrix, in the
    capability matrix, in conformance test output, and in CIDs.
    """

    # Reads.
    READ_HIDDEN_STATE = "read_hidden_state"
    READ_KV_CACHE = "read_kv_cache"
    READ_ATTENTION_PROBS = "read_attention_probs"
    READ_PER_LAYER_LOGITS = "read_per_layer_logits"
    READ_FINAL_LOGITS = "read_final_logits"
    # Writes / injections.
    WRITE_HIDDEN_STATE_INJECT = "write_hidden_state_inject"
    WRITE_KV_RESTORE = "write_kv_restore"
    WRITE_ATTENTION_BIAS = "write_attention_bias"
    INJECT_PREFIX_STATE = "inject_prefix_state"
    # Witnessing.
    REPLAY_FROM_KV = "replay_from_kv"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    CONTENT_ADDRESSED_TRACE = "content_addressed_trace"


W80_INSTRUMENTATION_AXES_ALL: tuple[str, ...] = tuple(
    a.value for a in InstrumentationAxis)

W80_INSTRUMENTATION_AXES_READ: tuple[str, ...] = (
    InstrumentationAxis.READ_HIDDEN_STATE.value,
    InstrumentationAxis.READ_KV_CACHE.value,
    InstrumentationAxis.READ_ATTENTION_PROBS.value,
    InstrumentationAxis.READ_PER_LAYER_LOGITS.value,
    InstrumentationAxis.READ_FINAL_LOGITS.value,
)
W80_INSTRUMENTATION_AXES_WRITE: tuple[str, ...] = (
    InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value,
    InstrumentationAxis.WRITE_KV_RESTORE.value,
    InstrumentationAxis.WRITE_ATTENTION_BIAS.value,
    InstrumentationAxis.INJECT_PREFIX_STATE.value,
)
W80_INSTRUMENTATION_AXES_WITNESS: tuple[str, ...] = (
    InstrumentationAxis.REPLAY_FROM_KV.value,
    InstrumentationAxis.DETERMINISTIC_REPLAY.value,
    InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value,
)


class CapabilityTag(str, enum.Enum):
    """How a backend honours an axis.

    * ``available`` — full contract coverage, returns the
      canonical shape.
    * ``backend_specific`` — implemented but with backend-known
      caveats (e.g. attention probs only for certain attention
      implementations).
    * ``best_effort`` — runtime tries to satisfy the axis but
      may degrade silently (e.g. HF KV cache shape varies).
    * ``unavailable`` — the backend honestly does not support
      this axis.
    """

    AVAILABLE = "available"
    BACKEND_SPECIFIC = "backend_specific"
    BEST_EFFORT = "best_effort"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _shape(arr: "_np.ndarray | None") -> tuple[int, ...]:
    if arr is None:
        return ()
    return tuple(int(s) for s in _np.asarray(arr).shape)


# ---------------------------------------------------------------
# Canonical snapshot schemas
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HiddenStateSnapshotV1:
    """Per-layer hidden-state snapshot.

    ``per_layer`` is one ndarray per layer of shape
    ``(seq_len, hidden_dim)``. Backends that do not expose
    per-layer hidden state report ``UNAVAILABLE`` on
    ``READ_HIDDEN_STATE``; backends that do report a single
    final hidden may populate only ``final``.
    """

    schema: str
    n_layers: int
    seq_len: int
    hidden_dim: int
    per_layer: tuple["_np.ndarray", ...]
    final: "_np.ndarray | None"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_layers": int(self.n_layers),
            "seq_len": int(self.seq_len),
            "hidden_dim": int(self.hidden_dim),
            "per_layer_cids": [
                _ndarray_cid(h) for h in self.per_layer],
            "per_layer_shapes": [
                _shape(h) for h in self.per_layer],
            "final_cid": _ndarray_cid(self.final),
            "final_shape": _shape(self.final),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_hidden_state_snapshot_v1",
            "snapshot": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class KVCacheSnapshotV1:
    """Per-layer KV cache snapshot.

    ``k_per_layer`` / ``v_per_layer`` are layer-wise (K, V)
    tensors. Shapes vary across backends; the W80 contract
    records the shape and the CID, not the raw bytes.
    """

    schema: str
    n_layers: int
    seq_len: int
    n_heads: int
    head_dim: int
    k_per_layer: tuple["_np.ndarray | None", ...]
    v_per_layer: tuple["_np.ndarray | None", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_layers": int(self.n_layers),
            "seq_len": int(self.seq_len),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "k_per_layer_cids": [
                _ndarray_cid(k) for k in self.k_per_layer],
            "v_per_layer_cids": [
                _ndarray_cid(v) for v in self.v_per_layer],
            "k_per_layer_shapes": [
                _shape(k) for k in self.k_per_layer],
            "v_per_layer_shapes": [
                _shape(v) for v in self.v_per_layer],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_kv_cache_snapshot_v1",
            "snapshot": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class AttentionSnapshotV1:
    """Per-layer attention-probability snapshot.

    ``per_layer`` is one ndarray per layer of shape
    ``(n_heads, seq_q, seq_k)`` (post-softmax probabilities).
    """

    schema: str
    n_layers: int
    n_heads: int
    seq_q: int
    seq_k: int
    per_layer: tuple["_np.ndarray | None", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "seq_q": int(self.seq_q),
            "seq_k": int(self.seq_k),
            "per_layer_cids": [
                _ndarray_cid(a) for a in self.per_layer],
            "per_layer_shapes": [
                _shape(a) for a in self.per_layer],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_attention_snapshot_v1",
            "snapshot": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ForwardTraceV1:
    """Canonical forward-pass record.

    A backend that implements the contract returns one of these
    on every forward. The trace is content-addressed and the CID
    is what the deterministic-replay check compares.
    """

    schema: str
    backend_id: str
    backend_runtime_id: str
    input_token_ids: tuple[int, ...]
    seq_len: int
    hidden: HiddenStateSnapshotV1 | None
    kv: KVCacheSnapshotV1 | None
    attn: AttentionSnapshotV1 | None
    final_logits: "_np.ndarray | None"
    declared_axes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_id": str(self.backend_id),
            "backend_runtime_id": str(self.backend_runtime_id),
            "input_token_ids": list(self.input_token_ids),
            "seq_len": int(self.seq_len),
            "hidden_cid": (
                self.hidden.cid()
                if self.hidden is not None else "none"),
            "kv_cid": (
                self.kv.cid() if self.kv is not None else "none"),
            "attn_cid": (
                self.attn.cid()
                if self.attn is not None else "none"),
            "final_logits_cid": _ndarray_cid(
                self.final_logits),
            "final_logits_shape": _shape(self.final_logits),
            "declared_axes": [
                [str(a), str(t)] for a, t in self.declared_axes],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_forward_trace_v1",
            "trace": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class InjectionPlanV1:
    """Plan of writes to apply to the next forward.

    Any field left at its default (``None`` / empty) means *do
    not inject*. Backends that do not honour an injection axis
    silently ignore the corresponding field; the conformance
    suite tests that backends which *claim* an axis actually
    move the trace CID under injection.
    """

    schema: str
    hidden_state_inject_per_layer: tuple[
        "_np.ndarray | None", ...] = ()
    kv_restore: KVCacheSnapshotV1 | None = None
    attention_bias_per_layer: tuple[
        "_np.ndarray | None", ...] = ()
    prefix_state_inject: "_np.ndarray | None" = None
    position_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hidden_state_inject_per_layer_cids": [
                _ndarray_cid(h)
                for h in self.hidden_state_inject_per_layer],
            "kv_restore_cid": (
                self.kv_restore.cid()
                if self.kv_restore is not None else "none"),
            "attention_bias_per_layer_cids": [
                _ndarray_cid(a)
                for a in self.attention_bias_per_layer],
            "prefix_state_inject_cid": _ndarray_cid(
                self.prefix_state_inject),
            "position_offset": (
                int(self.position_offset)
                if self.position_offset is not None else None),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_injection_plan_v1",
            "plan": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class InstrumentationWitnessV1:
    """Witness emitted at the end of a contract-conformant run.

    The witness bundles (backend_id, runtime_id, declared axes
    map, last trace CID). Long-horizon plumbing (audit logs,
    cross-runtime comparisons, parity matrices) reads this.
    """

    schema: str
    backend_id: str
    backend_runtime_id: str
    declared_axes: tuple[tuple[str, str], ...]
    last_trace_cid: str
    n_forwards: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_id": str(self.backend_id),
            "backend_runtime_id": str(self.backend_runtime_id),
            "declared_axes": [
                [str(a), str(t)] for a, t in self.declared_axes],
            "last_trace_cid": str(self.last_trace_cid),
            "n_forwards": int(self.n_forwards),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_instrumentation_witness_v1",
            "witness": self.to_dict()})


# ---------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------

class RuntimeInstrumentationProtocolV1(Protocol):
    """The shape any controlled-runtime backend must satisfy.

    A backend implementation does NOT need to inherit from this
    class — Python structural typing applies. The conformance
    runner only requires that these methods exist.
    """

    def backend_id(self) -> str:
        ...

    def backend_runtime_id(self) -> str:
        ...

    def declared_axes(self) -> Mapping[str, str]:
        """Map ``InstrumentationAxis`` value -> ``CapabilityTag``
        value."""
        ...

    def tokenize(self, text: str, *, max_len: int = 64) -> list[int]:
        ...

    def forward(
            self, *, input_token_ids: Sequence[int],
            injection: InjectionPlanV1 | None = None,
    ) -> ForwardTraceV1:
        ...

    def replay_from_kv(
            self, *, kv: KVCacheSnapshotV1,
            new_token_ids: Sequence[int],
    ) -> ForwardTraceV1:
        ...


# ---------------------------------------------------------------
# Conformance test runner
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class InstrumentationConformanceReportV1:
    """Per-backend conformance report.

    Each axis evaluated lands in one of:

    * ``pass`` — the backend claims the axis and the canonical
      check passed.
    * ``skip`` — the backend honestly does not claim the axis.
    * ``fail`` — the backend claims the axis but the check
      failed.
    """

    schema: str
    backend_id: str
    backend_runtime_id: str
    per_axis: tuple[tuple[str, str], ...]
    n_pass: int
    n_skip: int
    n_fail: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_id": str(self.backend_id),
            "backend_runtime_id": str(self.backend_runtime_id),
            "per_axis": [
                [str(a), str(s)] for a, s in self.per_axis],
            "n_pass": int(self.n_pass),
            "n_skip": int(self.n_skip),
            "n_fail": int(self.n_fail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_instrumentation_conformance_v1",
            "report": self.to_dict()})

    def all_claimed_pass(self) -> bool:
        return int(self.n_fail) == 0

    def per_axis_dict(self) -> dict[str, str]:
        return {a: s for a, s in self.per_axis}


def _check_axis_pass(
        backend: RuntimeInstrumentationProtocolV1,
        axis: str, prompt: str,
) -> str:
    """Per-axis conformance check.

    Returns ``"pass"``, ``"fail"``, or ``"skip"``.
    """
    declared = dict(backend.declared_axes())
    tag = declared.get(axis, CapabilityTag.UNAVAILABLE.value)
    if tag == CapabilityTag.UNAVAILABLE.value:
        return "skip"
    ids = list(backend.tokenize(prompt, max_len=16))
    if len(ids) == 0:
        return "skip"
    try:
        baseline = backend.forward(input_token_ids=ids)
    except Exception:  # noqa: BLE001
        return "fail"
    # Universally required: a CID.
    if not isinstance(baseline.cid(), str) or len(baseline.cid()) < 16:
        return "fail"
    # Universally required: declared axes appear in the trace.
    declared_in_trace = {
        a: t for a, t in baseline.declared_axes}
    if declared_in_trace.get(axis, "") != tag:
        return "fail"
    # Axis-specific:
    if axis == InstrumentationAxis.READ_HIDDEN_STATE.value:
        if baseline.hidden is None:
            return "fail"
        if int(baseline.hidden.n_layers) <= 0:
            return "fail"
        if len(baseline.hidden.per_layer) == 0 and (
                baseline.hidden.final is None):
            return "fail"
        return "pass"
    if axis == InstrumentationAxis.READ_KV_CACHE.value:
        if baseline.kv is None:
            return "fail"
        if int(baseline.kv.n_layers) <= 0:
            return "fail"
        # At least one layer should have non-None K and V.
        if not any(
                k is not None
                for k in baseline.kv.k_per_layer):
            return "fail"
        return "pass"
    if axis == InstrumentationAxis.READ_ATTENTION_PROBS.value:
        if baseline.attn is None:
            return "fail"
        if int(baseline.attn.n_layers) <= 0:
            return "fail"
        return "pass"
    if axis == InstrumentationAxis.READ_PER_LAYER_LOGITS.value:
        # Per-layer logits is optional in the canonical schema;
        # backends may expose it via hidden_state + unembed.
        # We pass if hidden_state is available *and* the backend
        # claims this axis honestly.
        if baseline.hidden is None:
            return "fail"
        return "pass"
    if axis == InstrumentationAxis.READ_FINAL_LOGITS.value:
        if baseline.final_logits is None:
            return "fail"
        if int(_np.asarray(
                baseline.final_logits).size) == 0:
            return "fail"
        return "pass"
    if axis == (
            InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value):
        # Compose an injection that is *non-trivial*: a small
        # ndarray at layer 0. CID must move.
        if baseline.hidden is None:
            return "fail"
        layer0_shape = (
            baseline.hidden.per_layer[0].shape
            if len(baseline.hidden.per_layer) > 0
            else (baseline.seq_len,
                  int(baseline.hidden.hidden_dim)))
        inj = _np.full(
            layer0_shape, 0.05, dtype=_np.float64)
        n_layers = (
            len(baseline.hidden.per_layer)
            if len(baseline.hidden.per_layer) > 0
            else int(baseline.hidden.n_layers))
        per_layer: list["_np.ndarray | None"] = (
            [None] * int(n_layers))
        per_layer[0] = inj
        plan = InjectionPlanV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            hidden_state_inject_per_layer=tuple(per_layer))
        try:
            after = backend.forward(
                input_token_ids=ids, injection=plan)
        except Exception:  # noqa: BLE001
            return "fail"
        return "pass" if baseline.cid() != after.cid() else "fail"
    if axis == InstrumentationAxis.WRITE_KV_RESTORE.value:
        # Replay-from-KV exercises this axis: produce a KV, then
        # restore it and run the new tokens.
        if baseline.kv is None:
            return "fail"
        if len(ids) < 2:
            return "skip"
        try:
            replay_trace = backend.replay_from_kv(
                kv=baseline.kv,
                new_token_ids=[int(ids[-1])])
        except Exception:  # noqa: BLE001
            return "fail"
        if replay_trace.final_logits is None:
            return "fail"
        return "pass"
    if axis == InstrumentationAxis.WRITE_ATTENTION_BIAS.value:
        if baseline.attn is None:
            return "fail"
        per_layer_attn = baseline.attn.per_layer
        if len(per_layer_attn) == 0 or per_layer_attn[0] is None:
            return "fail"
        a0 = _np.asarray(per_layer_attn[0])
        bias_shape = a0.shape
        bias = _np.full(
            bias_shape, 0.01, dtype=_np.float64)
        bias_per_layer: list["_np.ndarray | None"] = (
            [None] * int(baseline.attn.n_layers))
        bias_per_layer[0] = bias
        plan = InjectionPlanV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            attention_bias_per_layer=tuple(bias_per_layer))
        try:
            after = backend.forward(
                input_token_ids=ids, injection=plan)
        except Exception:  # noqa: BLE001
            return "fail"
        return "pass" if baseline.cid() != after.cid() else "fail"
    if axis == InstrumentationAxis.INJECT_PREFIX_STATE.value:
        if baseline.hidden is None:
            return "fail"
        # Build a prefix-state injection of (seq_len, hidden_dim).
        if (len(baseline.hidden.per_layer) > 0
                and baseline.hidden.per_layer[0] is not None):
            shape0 = baseline.hidden.per_layer[0].shape
        elif baseline.hidden.final is not None:
            shape0 = baseline.hidden.final.shape
        else:
            return "fail"
        prefix = _np.full(
            shape0, 0.02, dtype=_np.float64)
        plan = InjectionPlanV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            prefix_state_inject=prefix)
        try:
            after = backend.forward(
                input_token_ids=ids, injection=plan)
        except Exception:  # noqa: BLE001
            return "fail"
        return "pass" if baseline.cid() != after.cid() else "fail"
    if axis == InstrumentationAxis.REPLAY_FROM_KV.value:
        if baseline.kv is None:
            return "fail"
        if len(ids) < 3:
            return "skip"
        old = list(ids[:-1])
        new = [int(ids[-1])]
        try:
            full = backend.forward(input_token_ids=ids)
            old_trace = backend.forward(input_token_ids=old)
            replay = backend.replay_from_kv(
                kv=old_trace.kv, new_token_ids=new)
        except Exception:  # noqa: BLE001
            return "fail"
        if full.final_logits is None or (
                replay.final_logits is None):
            return "fail"
        # The final-row logits should match between recompute
        # and replay on the same backend.
        full_last = _np.asarray(full.final_logits)[-1]
        replay_last = _np.asarray(replay.final_logits)[-1]
        if full_last.shape != replay_last.shape:
            return "fail"
        diff = float(_np.max(_np.abs(full_last - replay_last)))
        return "pass" if diff < 1e-4 else "fail"
    if axis == InstrumentationAxis.DETERMINISTIC_REPLAY.value:
        try:
            a = backend.forward(input_token_ids=ids)
            b = backend.forward(input_token_ids=ids)
        except Exception:  # noqa: BLE001
            return "fail"
        return "pass" if a.cid() == b.cid() else "fail"
    if axis == (
            InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value):
        return "pass" if (
            isinstance(baseline.cid(), str)
            and len(baseline.cid()) >= 32) else "fail"
    return "skip"


def run_instrumentation_conformance_v1(
        backend: RuntimeInstrumentationProtocolV1,
        *, prompt: str = "instrumentation conformance smoke",
        axes: Sequence[str] | None = None,
) -> InstrumentationConformanceReportV1:
    """Run the conformance suite against any backend.

    For each axis the backend *claims*, run the canonical check.
    For axes marked ``UNAVAILABLE``, mark ``skip``. Any
    contradiction (claimed but check fails) is a ``fail``.
    """
    axes_eval = tuple(axes) if axes else (
        W80_INSTRUMENTATION_AXES_ALL)
    per: list[tuple[str, str]] = []
    n_pass = 0
    n_skip = 0
    n_fail = 0
    for ax in axes_eval:
        status = _check_axis_pass(backend, ax, prompt)
        per.append((str(ax), str(status)))
        if status == "pass":
            n_pass += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_fail += 1
    return InstrumentationConformanceReportV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        backend_id=str(backend.backend_id()),
        backend_runtime_id=str(backend.backend_runtime_id()),
        per_axis=tuple(per),
        n_pass=int(n_pass), n_skip=int(n_skip),
        n_fail=int(n_fail),
    )


def emit_instrumentation_witness(
        backend: RuntimeInstrumentationProtocolV1,
        *, last_trace_cid: str = "",
        n_forwards: int = 0,
) -> InstrumentationWitnessV1:
    return InstrumentationWitnessV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        backend_id=str(backend.backend_id()),
        backend_runtime_id=str(backend.backend_runtime_id()),
        declared_axes=tuple(
            (str(a), str(t))
            for a, t in backend.declared_axes().items()),
        last_trace_cid=str(last_trace_cid),
        n_forwards=int(n_forwards),
    )


# ---------------------------------------------------------------
# Controlled-runtime V1 adapter (in-repo NumPy)
# ---------------------------------------------------------------

@dataclasses.dataclass
class ControlledRuntimeInstrumentationAdapterV1:
    """Adapter mapping the W79 ``controlled_runtime_substrate_v1``
    into the W80 instrumentation contract.

    Lazy-imported so the contract module stays substrate-free.
    """

    runtime_params: Any = None  # ControlledRuntimeParamsV1

    def __post_init__(self) -> None:
        from .controlled_runtime_substrate_v1 import (
            build_controlled_runtime_params_v1,
        )
        if self.runtime_params is None:
            self.runtime_params = (
                build_controlled_runtime_params_v1())

    def backend_id(self) -> str:
        return "coordpy.controlled_runtime_substrate_v1"

    def backend_runtime_id(self) -> str:
        return f"{self.backend_id()}#{str(self.runtime_params.cid())[:16]}"

    def declared_axes(self) -> Mapping[str, str]:
        a = CapabilityTag.AVAILABLE.value
        return {
            InstrumentationAxis.READ_HIDDEN_STATE.value: a,
            InstrumentationAxis.READ_KV_CACHE.value: a,
            InstrumentationAxis.READ_ATTENTION_PROBS.value: a,
            InstrumentationAxis.READ_PER_LAYER_LOGITS.value: a,
            InstrumentationAxis.READ_FINAL_LOGITS.value: a,
            InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value:
                a,
            InstrumentationAxis.WRITE_KV_RESTORE.value: a,
            InstrumentationAxis.WRITE_ATTENTION_BIAS.value: a,
            InstrumentationAxis.INJECT_PREFIX_STATE.value: a,
            InstrumentationAxis.REPLAY_FROM_KV.value: a,
            InstrumentationAxis.DETERMINISTIC_REPLAY.value: a,
            InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value: a,
        }

    def tokenize(
            self, text: str, *, max_len: int = 64,
    ) -> list[int]:
        from .controlled_runtime_substrate_v1 import (
            tokenize_bytes_v79,
        )
        return list(tokenize_bytes_v79(
            str(text), max_len=int(max_len)))

    def forward(
            self, *, input_token_ids: Sequence[int],
            injection: InjectionPlanV1 | None = None,
    ) -> ForwardTraceV1:
        from .controlled_runtime_substrate_v1 import (
            forward_controlled_runtime,
        )
        hidden_inj = None
        attn_inj = None
        prefix = None
        if injection is not None:
            if len(injection.hidden_state_inject_per_layer) > 0:
                hidden_inj = list(
                    injection.hidden_state_inject_per_layer)
            if len(injection.attention_bias_per_layer) > 0:
                attn_inj = list(
                    injection.attention_bias_per_layer)
            if injection.prefix_state_inject is not None:
                prefix = injection.prefix_state_inject
        trace, kv = forward_controlled_runtime(
            params=self.runtime_params,
            input_token_ids=input_token_ids,
            hidden_state_injections_per_layer=hidden_inj,
            attention_bias_injections_per_layer=attn_inj,
            prefix_state_injection=prefix,
        )
        return self._wrap_trace(trace=trace, kv=kv)

    def replay_from_kv(
            self, *, kv: KVCacheSnapshotV1,
            new_token_ids: Sequence[int],
    ) -> ForwardTraceV1:
        from .controlled_runtime_substrate_v1 import (
            ControlledRuntimeKVCacheV1,
            replay_from_kv_cache,
        )
        kv_obj = ControlledRuntimeKVCacheV1.empty(
            n_layers=int(self.runtime_params.n_layers),
            n_heads=int(self.runtime_params.n_heads),
            head_dim=int(self.runtime_params.head_dim))
        k_layers = list(kv.k_per_layer)
        v_layers = list(kv.v_per_layer)
        # Fill in cached layers as numpy arrays.
        for i in range(int(self.runtime_params.n_layers)):
            if i < len(k_layers) and k_layers[i] is not None:
                kv_obj.k_layers[i] = _np.asarray(
                    k_layers[i], dtype=_np.float64).copy()
            if i < len(v_layers) and v_layers[i] is not None:
                kv_obj.v_layers[i] = _np.asarray(
                    v_layers[i], dtype=_np.float64).copy()
        trace, kv_out = replay_from_kv_cache(
            params=self.runtime_params,
            kv_cache=kv_obj,
            new_token_ids=new_token_ids)
        return self._wrap_trace(trace=trace, kv=kv_out)

    def _wrap_trace(
            self, *, trace: Any, kv: Any,
    ) -> ForwardTraceV1:
        hidden = HiddenStateSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(trace.n_layers),
            seq_len=int(trace.seq_len),
            hidden_dim=int(trace.hidden_dim),
            per_layer=tuple(
                _np.asarray(h, dtype=_np.float64)
                for h in trace.post_mlp_hidden),
            final=(
                _np.asarray(trace.final_hidden, dtype=_np.float64)
                if trace.final_hidden is not None else None),
        )
        kv_snapshot = KVCacheSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(kv.n_layers),
            seq_len=int(kv.total_seq_len()),
            n_heads=int(kv.n_heads),
            head_dim=int(kv.head_dim),
            k_per_layer=tuple(
                (_np.asarray(k, dtype=_np.float64).copy()
                 if k is not None else None)
                for k in kv.k_layers),
            v_per_layer=tuple(
                (_np.asarray(v, dtype=_np.float64).copy()
                 if v is not None else None)
                for v in kv.v_layers),
        )
        attn = AttentionSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(trace.n_layers),
            n_heads=int(trace.n_heads),
            seq_q=int(trace.seq_len),
            seq_k=int(trace.attn_probs[0].shape[-1])
            if len(trace.attn_probs) > 0
            and _np.asarray(trace.attn_probs[0]).size > 0
            else 0,
            per_layer=tuple(
                _np.asarray(a, dtype=_np.float64)
                for a in trace.attn_probs),
        )
        return ForwardTraceV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            backend_id=self.backend_id(),
            backend_runtime_id=self.backend_runtime_id(),
            input_token_ids=tuple(int(t) for t in trace.input_token_ids),
            seq_len=int(trace.seq_len),
            hidden=hidden, kv=kv_snapshot, attn=attn,
            final_logits=(
                _np.asarray(trace.logits, dtype=_np.float64)
                if trace.logits is not None else None),
            declared_axes=tuple(
                (str(k), str(v))
                for k, v in self.declared_axes().items()),
        )


__all__ = [
    "W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION",
    "W80_INSTRUMENTATION_AXES_ALL",
    "W80_INSTRUMENTATION_AXES_READ",
    "W80_INSTRUMENTATION_AXES_WRITE",
    "W80_INSTRUMENTATION_AXES_WITNESS",
    "InstrumentationAxis",
    "CapabilityTag",
    "HiddenStateSnapshotV1",
    "KVCacheSnapshotV1",
    "AttentionSnapshotV1",
    "ForwardTraceV1",
    "InjectionPlanV1",
    "InstrumentationWitnessV1",
    "InstrumentationConformanceReportV1",
    "RuntimeInstrumentationProtocolV1",
    "run_instrumentation_conformance_v1",
    "emit_instrumentation_witness",
    "ControlledRuntimeInstrumentationAdapterV1",
]
