"""W79 M1 — Tiny Transformer Runtime V24.

Strictly extends W78's ``coordpy.tiny_substrate_v23``. V24 keeps
every V23 invariant byte-for-byte under trivial construction and
adds **one** new substrate-load-bearing axis: a per-turn
**replacement-then-restart-after-long-delay trajectory CID**
that unifies the W78 long-horizon-reconstruction trajectory CID
with the recorded W79 replacement-then-restart-after-long-delay
event chain — the carrier the W79 long-horizon-reconstruction
substrate V2 reads from for the W79 regime where the source
event lies before BOTH a replacement and a restart within the
same blackout window.

* **Inherits V23's 23 physical layers**. V24 adds a sixth axis
  layer in the substrate cache (replacement-then-restart-after-
  long-delay axis); the physical transformer depth carries
  forward from V23 byte-for-byte.
* **Per-layer replacement-then-restart-after-long-delay length
  label** — V23's [0..14] extended by 15 =
  ``replacement_then_restart_after_long_delay`` (any layer on
  which a replacement-then-restart event was observed inside a
  long-delay blackout window).
* **Per-layer replacement-then-restart-after-long-delay-
  pressure gate** — substrate-side throttle in [0, 1].

V24 strictly extends V23.

Honest scope (W79)
------------------

* Still NOT a frontier model. Default config inherited from V23.
  ``W79-L-NUMPY-CPU-V24-SUBSTRATE-CAP``.
* V24 does NOT bridge to third-party hosted models.
  ``W79-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward. The controlled-runtime substrate is the W79 attack
  on the wall (see
  ``coordpy.controlled_runtime_substrate_v1``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .tiny_substrate_v23 import (
    TinyV23ForwardTrace, TinyV23KVCache,
    TinyV23SubstrateConfig, TinyV23SubstrateParams,
    W78_DEFAULT_V23_GATE_BIAS, W78_DEFAULT_V23_MAX_ROLES,
    W78_REPAIR_LABELS_V23,
    build_default_tiny_substrate_v23,
    emit_tiny_substrate_v23_forward_witness,
    forward_tiny_substrate_v23,
    record_long_horizon_reconstruction_window_v23,
    tokenize_bytes_v23,
)


W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v24.v1")

W79_TINY_V24_VOCAB_SIZE: int = 260
W79_DEFAULT_V24_MAX_ROLES: int = W78_DEFAULT_V23_MAX_ROLES
W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE_BOOST: (
    float) = 0.86
W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_REPAIR_BOOST: (
    float) = 0.86
W79_DEFAULT_V24_GATE_BIAS: float = W78_DEFAULT_V23_GATE_BIAS
W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_WINDOW_FLOOR: int = 55
W79_DEFAULT_V24_SEED: int = 79123456

# V24 extends W78_REPAIR_LABELS_V23 with a sixteenth primitive.
W79_REPAIR_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY: int = 15
W79_REPAIR_LABELS_V24: tuple[str, ...] = (
    *W78_REPAIR_LABELS_V23,
    "replacement_then_restart_after_long_delay_reconstruction",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def tokenize_bytes_v24(
        text: str, *, max_len: int = 16) -> list[int]:
    return tokenize_bytes_v23(str(text), max_len=int(max_len))


@dataclasses.dataclass(frozen=True)
class W79ReplacementThenRestartAfterLongDelayWindow:
    """A recorded replacement-then-restart-after-long-delay
    window inside the V24 substrate cache."""

    schema: str
    long_delay_blackout_start_turn: int
    replacement_turn: int
    restart_turn: int
    reconstruction_request_turn: int
    role: str
    branch_id: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_replacement_then_restart_after_long_delay_window",
            "schema": str(self.schema),
            "long_delay_blackout_start_turn": int(
                self.long_delay_blackout_start_turn),
            "replacement_turn": int(self.replacement_turn),
            "restart_turn": int(self.restart_turn),
            "reconstruction_request_turn": int(
                self.reconstruction_request_turn),
            "role": str(self.role),
            "branch_id": str(self.branch_id),
        })


@dataclasses.dataclass
class TinyV24KVCache:
    """V24 cache wrapping V23, with the new W79 axis."""

    v23_cache: TinyV23KVCache
    replacement_then_restart_after_long_delay_windows: list[
        W79ReplacementThenRestartAfterLongDelayWindow]
    replacement_then_restart_after_long_delay_trajectory_cid: (
        str)
    replacement_then_restart_after_long_delay_length_per_layer: (
        tuple[int, ...])
    replacement_then_restart_after_long_delay_pressure_gate_per_layer: (
        tuple[float, ...])

    @classmethod
    def empty(
            cls, n_layers: int = 0,
    ) -> "TinyV24KVCache":
        # An empty V24 cache wraps an empty V23 (constructed
        # lazily by the V23 forward path when needed).
        return cls(
            v23_cache=None,  # type: ignore[arg-type]
            replacement_then_restart_after_long_delay_windows=[],
            replacement_then_restart_after_long_delay_trajectory_cid="",
            replacement_then_restart_after_long_delay_length_per_layer=tuple(
                [0] * int(max(0, n_layers))),
            replacement_then_restart_after_long_delay_pressure_gate_per_layer=tuple(
                [0.0] * int(max(0, n_layers))),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_tiny_v24_kv_cache",
            "v23_cache_cid": (
                str(self.v23_cache.cid())
                if self.v23_cache is not None else "none"),
            "n_replacement_then_restart_after_long_delay_windows": int(
                len(
                    self
                    .replacement_then_restart_after_long_delay_windows)),
            "replacement_then_restart_after_long_delay_trajectory_cid": str(
                self
                .replacement_then_restart_after_long_delay_trajectory_cid),
            "replacement_then_restart_after_long_delay_length_per_layer": list(
                self
                .replacement_then_restart_after_long_delay_length_per_layer),
            "replacement_then_restart_after_long_delay_pressure_gate_per_layer": [
                float(round(float(x), 12))
                for x in
                self
                .replacement_then_restart_after_long_delay_pressure_gate_per_layer],
        })


def record_replacement_then_restart_after_long_delay_window_v24(
        cache: TinyV24KVCache,
        *,
        long_delay_blackout_start_turn: int,
        replacement_turn: int,
        restart_turn: int,
        reconstruction_request_turn: int,
        role: str = "planner_w79",
        branch_id: str = "main",
) -> None:
    """Record one W79 replacement-then-restart-after-long-delay
    window into the V24 cache and update its trajectory CID."""
    w = W79ReplacementThenRestartAfterLongDelayWindow(
        schema=W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION,
        long_delay_blackout_start_turn=int(
            long_delay_blackout_start_turn),
        replacement_turn=int(replacement_turn),
        restart_turn=int(restart_turn),
        reconstruction_request_turn=int(
            reconstruction_request_turn),
        role=str(role),
        branch_id=str(branch_id),
    )
    cache.replacement_then_restart_after_long_delay_windows.append(
        w)
    cache.replacement_then_restart_after_long_delay_trajectory_cid = (
        _sha256_hex({
            "kind":
                "w79_replacement_then_restart_after_long_delay_trajectory",
            "previous_trajectory_cid": str(
                cache
                .replacement_then_restart_after_long_delay_trajectory_cid),
            "window_cid": str(w.cid()),
        }))


@dataclasses.dataclass(frozen=True)
class TinyV24ForwardTrace:
    """V24 trace wrapping V23 plus the W79 axis read-back."""

    schema: str
    v23_trace_cid: str
    v24_cache_cid: str
    replacement_then_restart_after_long_delay_l1: int
    replacement_then_restart_after_long_delay_pressure_gate_mean: (
        float)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_tiny_v24_forward_trace",
            "schema": str(self.schema),
            "v23_trace_cid": str(self.v23_trace_cid),
            "v24_cache_cid": str(self.v24_cache_cid),
            "replacement_then_restart_after_long_delay_l1": int(
                self
                .replacement_then_restart_after_long_delay_l1),
            "replacement_then_restart_after_long_delay_pressure_gate_mean": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_gate_mean,
                    12)),
        })


@dataclasses.dataclass(frozen=True)
class TinyV24SubstrateParams:
    """V24 substrate params wrap V23 with one new axis flag."""

    v23: TinyV23SubstrateParams
    expose_replacement_then_restart_after_long_delay_trajectory_cid: bool = (
        True)
    expose_replacement_then_restart_after_long_delay_length_per_layer: bool = (
        True)
    expose_replacement_then_restart_after_long_delay_pressure_gate: bool = (
        True)
    n_layers: int = 23

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION,
            "v23_cid": str(self.v23.cid()),
            "expose_replacement_then_restart_after_long_delay_trajectory_cid": bool(
                self
                .expose_replacement_then_restart_after_long_delay_trajectory_cid),
            "expose_replacement_then_restart_after_long_delay_length_per_layer": bool(
                self
                .expose_replacement_then_restart_after_long_delay_length_per_layer),
            "expose_replacement_then_restart_after_long_delay_pressure_gate": bool(
                self
                .expose_replacement_then_restart_after_long_delay_pressure_gate),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_tiny_v24_substrate_params",
            "params": self.to_dict()})


def build_default_tiny_substrate_v24(
        *, seed: int = W79_DEFAULT_V24_SEED,
) -> TinyV24SubstrateParams:
    v23 = build_default_tiny_substrate_v23(seed=int(seed))
    return TinyV24SubstrateParams(v23=v23)


def forward_tiny_substrate_v24(
        params: TinyV24SubstrateParams,
        token_ids: Sequence[int],
        *,
        v24_kv_cache: TinyV24KVCache | None = None,
        replacement_then_restart_after_long_delay_pressure: float = (
            0.0),
        **v23_kwargs: Any,
) -> tuple[TinyV24ForwardTrace, TinyV24KVCache]:
    """V24 forward — delegates to V23 for the physical work and
    records the W79 axis."""
    if v24_kv_cache is None:
        v24_kv_cache = TinyV24KVCache.empty(
            n_layers=int(params.n_layers))
    v23_trace, v23_cache = forward_tiny_substrate_v23(
        params.v23, token_ids,
        v23_kv_cache=v24_kv_cache.v23_cache,
        **v23_kwargs)
    v24_kv_cache.v23_cache = v23_cache
    L = int(params.n_layers)
    new_lengths = list(
        v24_kv_cache
        .replacement_then_restart_after_long_delay_length_per_layer)
    while len(new_lengths) < L:
        new_lengths.append(0)
    new_gate = list(
        v24_kv_cache
        .replacement_then_restart_after_long_delay_pressure_gate_per_layer)
    while len(new_gate) < L:
        new_gate.append(0.0)
    p = float(max(0.0, min(
        1.0,
        float(
            replacement_then_restart_after_long_delay_pressure))))
    for li in range(L):
        new_lengths[li] += int(p > 0.5)
        # EMA the gate value (alpha 0.30).
        new_gate[li] = (
            0.30 * p + 0.70 * float(new_gate[li]))
    new_cache = TinyV24KVCache(
        v23_cache=v23_cache,
        replacement_then_restart_after_long_delay_windows=list(
            v24_kv_cache
            .replacement_then_restart_after_long_delay_windows),
        replacement_then_restart_after_long_delay_trajectory_cid=(
            v24_kv_cache
            .replacement_then_restart_after_long_delay_trajectory_cid),
        replacement_then_restart_after_long_delay_length_per_layer=tuple(
            int(x) for x in new_lengths),
        replacement_then_restart_after_long_delay_pressure_gate_per_layer=tuple(
            float(x) for x in new_gate),
    )
    l1 = int(sum(new_lengths))
    gate_mean = float(sum(new_gate) / max(1, len(new_gate)))
    trace = TinyV24ForwardTrace(
        schema=W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION,
        v23_trace_cid=str(v23_trace.cid()),
        v24_cache_cid=str(new_cache.cid()),
        replacement_then_restart_after_long_delay_l1=int(l1),
        replacement_then_restart_after_long_delay_pressure_gate_mean=float(
            gate_mean),
    )
    return trace, new_cache


@dataclasses.dataclass(frozen=True)
class TinyV24ForwardWitness:
    schema: str
    params_cid: str
    trace_cid: str
    cache_cid: str
    replacement_then_restart_after_long_delay_l1: int
    replacement_then_restart_after_long_delay_pressure_gate_mean: (
        float)
    replacement_then_restart_after_long_delay_trajectory_cid: str
    n_replacement_then_restart_after_long_delay_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "params_cid": str(self.params_cid),
            "trace_cid": str(self.trace_cid),
            "cache_cid": str(self.cache_cid),
            "replacement_then_restart_after_long_delay_l1": int(
                self
                .replacement_then_restart_after_long_delay_l1),
            "replacement_then_restart_after_long_delay_pressure_gate_mean": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_gate_mean,
                    12)),
            "replacement_then_restart_after_long_delay_trajectory_cid": str(
                self
                .replacement_then_restart_after_long_delay_trajectory_cid),
            "n_replacement_then_restart_after_long_delay_windows": int(
                self
                .n_replacement_then_restart_after_long_delay_windows),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_tiny_v24_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v24_forward_witness(
        params: TinyV24SubstrateParams,
        trace: TinyV24ForwardTrace,
        cache: TinyV24KVCache,
) -> TinyV24ForwardWitness:
    return TinyV24ForwardWitness(
        schema=W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION,
        params_cid=str(params.cid()),
        trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        replacement_then_restart_after_long_delay_l1=int(
            trace
            .replacement_then_restart_after_long_delay_l1),
        replacement_then_restart_after_long_delay_pressure_gate_mean=float(
            trace
            .replacement_then_restart_after_long_delay_pressure_gate_mean),
        replacement_then_restart_after_long_delay_trajectory_cid=str(
            cache
            .replacement_then_restart_after_long_delay_trajectory_cid),
        n_replacement_then_restart_after_long_delay_windows=int(
            len(
                cache
                .replacement_then_restart_after_long_delay_windows)),
    )


__all__ = [
    "W79_TINY_SUBSTRATE_V24_SCHEMA_VERSION",
    "W79_TINY_V24_VOCAB_SIZE",
    "W79_DEFAULT_V24_MAX_ROLES",
    "W79_DEFAULT_V24_GATE_BIAS",
    "W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE_BOOST",
    "W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_REPAIR_BOOST",
    "W79_DEFAULT_V24_REPLACEMENT_THEN_RESTART_WINDOW_FLOOR",
    "W79_DEFAULT_V24_SEED",
    "W79_REPAIR_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY",
    "W79_REPAIR_LABELS_V24",
    "TinyV24KVCache",
    "TinyV24ForwardTrace",
    "TinyV24ForwardWitness",
    "TinyV24SubstrateParams",
    "W79ReplacementThenRestartAfterLongDelayWindow",
    "build_default_tiny_substrate_v24",
    "forward_tiny_substrate_v24",
    "record_replacement_then_restart_after_long_delay_window_v24",
    "emit_tiny_substrate_v24_forward_witness",
    "tokenize_bytes_v24",
]
