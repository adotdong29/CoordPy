"""W65 M1 — Tiny Transformer Runtime V10.

Strictly extends W64's ``coordpy.tiny_substrate_v9``. V10 keeps
every V9 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
nine V9 axes including hidden_wins_primary, replay_dominance_witness,
attention_entropy, cache_similarity_matrix, hidden_state_trust)
and adds **four** new substrate-load-bearing axes that W65's
multi-agent coordinator and V10 bridges/controllers exploit:

* **Default 12 layers** (vs V9's 11). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) hidden-write-merit channel** —
  ``TinyV10KVCache.hidden_write_merit`` of shape ``(L, H, T)``
  records a substrate-measured scalar in [0, 1] representing
  how well the most recent hidden-state injection landed at that
  (layer, head, slot). The W65 HSB V9 reads this back when
  computing the per-(L, H) hidden-wins-rate.
* **Per-role KV bank** — ``TinyV10KVCache.role_kv_bank`` mapping
  ``role_tag -> ndarray(L, H, T, d_head)`` offset matrices added
  to KV reads when the role is active. The bank is bounded:
  at most ``max_n_roles`` slots; oldest evicted (FIFO).
* **Substrate checkpoint / restore primitive** —
  ``substrate_checkpoint_v10(cache)`` returns a serializable
  token-bounded snapshot; ``substrate_restore_v10(snapshot)``
  rebuilds the cache (deterministic on the same V10 params).
  The checkpoint records the V9 inner cache CID plus the four
  new V10 channels.
* **Per-layer V10 composite gate score** — calibrated weighted
  combination of attention-entropy, cache-similarity-mean,
  hidden-wins-primary-l1, replay-dominance-witness-l1,
  hidden-state-trust-l1, and hidden-write-merit-mean; emitted
  as ``TinyV10ForwardTrace.v10_gate_score_per_layer``.

V10 still preserves all V9 axes byte-for-byte under trivial
construction; the four new axes are zero-valued unless explicitly
written.

Honest scope (do-not-overstate, W65)
------------------------------------

* Still NOT a frontier model. Default config:
  ``12 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W65-L-NUMPY-CPU-V10-SUBSTRATE-CAP`` documents.
* V10 still does NOT bridge to third-party hosted models.
  ``W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The hidden-write-merit channel is a *substrate-measured
  diagnostic*; it is NOT a frontier-model ground truth.
* The role KV bank is a per-role additive offset on the V9 cache
  read path; the bank is bounded and FIFO-evicted.
* Substrate checkpoint/restore operate on the in-repo V10 cache
  only (``W65-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP``).
* The V10 gate score is a calibrated linear combination, not a
  learned end-to-end controller.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v10 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import (
    TinyV9ForwardTrace, TinyV9KVCache, TinyV9SubstrateConfig,
    TinyV9SubstrateParams,
    W64_DEFAULT_V9_D_KEY, W64_DEFAULT_V9_TRUST_EMA,
    W64_DEFAULT_V9_REPLAY_DETERMINISM_TOL,
    W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD,
    emit_tiny_substrate_v9_forward_witness,
    forward_tiny_substrate_v9,
    tokenize_bytes_v9,
)


W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v10.v1")

W65_TINY_V10_VOCAB_SIZE: int = 259
W65_DEFAULT_V10_D_MODEL: int = 64
W65_DEFAULT_V10_N_HEADS: int = 8
W65_DEFAULT_V10_N_KV_HEADS: int = 4
W65_DEFAULT_V10_N_LAYERS: int = 12
W65_DEFAULT_V10_FF_HIDDEN: int = 192
W65_DEFAULT_V10_MAX_LEN: int = 128
W65_DEFAULT_V10_INIT_SCALE: float = 0.04
W65_DEFAULT_V10_SEED: int = 65012345
W65_DEFAULT_V10_ROPE_BASE: float = 10000.0
W65_DEFAULT_V10_MAX_ROLES: int = 8
W65_DEFAULT_V10_GATE_BIAS: float = 0.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class TinyV10SubstrateConfig:
    """V10 config wraps a V9 config + four new axes."""
    v9: TinyV9SubstrateConfig
    max_n_roles: int = W65_DEFAULT_V10_MAX_ROLES
    expose_hidden_write_merit: bool = True
    expose_role_kv_bank: bool = True
    expose_substrate_checkpoint: bool = True
    expose_v10_gate_score: bool = True
    # Per-axis weights for the gate score composite (length 6).
    gate_weights: tuple[float, ...] = (
        0.18, 0.13, 0.20, 0.20, 0.14, 0.15)

    @classmethod
    def default(
            cls, *, seed: int = W65_DEFAULT_V10_SEED,
    ) -> "TinyV10SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W65_TINY_V10_VOCAB_SIZE,
            d_model=W65_DEFAULT_V10_D_MODEL,
            n_heads=W65_DEFAULT_V10_N_HEADS,
            n_kv_heads=W65_DEFAULT_V10_N_KV_HEADS,
            n_layers=W65_DEFAULT_V10_N_LAYERS,
            ff_hidden=W65_DEFAULT_V10_FF_HIDDEN,
            max_len=W65_DEFAULT_V10_MAX_LEN,
            init_scale=W65_DEFAULT_V10_INIT_SCALE,
            seed=int(seed),
            d_key=W64_DEFAULT_V9_D_KEY,
            trust_ema=W64_DEFAULT_V9_TRUST_EMA,
            replay_determinism_tol=(
                W64_DEFAULT_V9_REPLAY_DETERMINISM_TOL),
            hidden_wins_primary_threshold=(
                W64_DEFAULT_V9_HIDDEN_WINS_PRIMARY_THRESHOLD))
        return cls(v9=v9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
            "v9_cid": str(self.v9.cid()),
            "max_n_roles": int(self.max_n_roles),
            "expose_hidden_write_merit": bool(
                self.expose_hidden_write_merit),
            "expose_role_kv_bank": bool(
                self.expose_role_kv_bank),
            "expose_substrate_checkpoint": bool(
                self.expose_substrate_checkpoint),
            "expose_v10_gate_score": bool(
                self.expose_v10_gate_score),
            "gate_weights": [
                float(round(float(x), 12))
                for x in self.gate_weights],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v10_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV10SubstrateParams:
    config: TinyV10SubstrateConfig
    v9_params: TinyV9SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV10SubstrateConfig | None = None,
    ) -> "TinyV10SubstrateParams":
        if config is None:
            config = TinyV10SubstrateConfig.default()
        v9 = TinyV9SubstrateParams.init(config.v9)
        return cls(config=config, v9_params=v9)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v9_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v10_substrate_params",
            "config_cid": self.config.cid(),
            "v9_params_cid": self.v9_params.cid(),
        })


@dataclasses.dataclass
class TinyV10KVCache:
    """V10 cache. Wraps a V9 cache + four new V10 axes."""
    v9_cache: TinyV9KVCache
    hidden_write_merit: "_np.ndarray"   # (L, H, T)
    role_kv_bank: dict[str, "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    role_order: list[str] = dataclasses.field(default_factory=list)
    v10_gate_score_per_layer: "_np.ndarray | None" = None
    write_log_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
            d_key: int = W64_DEFAULT_V9_D_KEY,
    ) -> "TinyV10KVCache":
        v9 = TinyV9KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len), d_key=int(d_key))
        return cls(
            v9_cache=v9,
            hidden_write_merit=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            role_kv_bank={},
            role_order=[],
            v10_gate_score_per_layer=None,
            write_log_v10=[])

    def n_tokens(self) -> int:
        return int(self.v9_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v9_cache.n_layers())

    def clone(self) -> "TinyV10KVCache":
        return TinyV10KVCache(
            v9_cache=self.v9_cache.clone(),
            hidden_write_merit=self.hidden_write_merit.copy(),
            role_kv_bank={
                k: v.copy() for k, v in self.role_kv_bank.items()},
            role_order=list(self.role_order),
            v10_gate_score_per_layer=(
                None if self.v10_gate_score_per_layer is None
                else self.v10_gate_score_per_layer.copy()),
            write_log_v10=list(self.write_log_v10),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v10_kv_cache",
            "v9_cache_cid": self.v9_cache.cid(),
            "hidden_write_merit_cid": _ndarray_cid(
                self.hidden_write_merit),
            "role_kv_bank_cids": [
                {"role": k, "cid": _ndarray_cid(v)}
                for k, v in sorted(self.role_kv_bank.items())],
            "role_order": list(self.role_order),
            "v10_gate_score_per_layer_cid": (
                "none"
                if self.v10_gate_score_per_layer is None
                else _ndarray_cid(
                    self.v10_gate_score_per_layer)),
            "write_log_v10": list(self.write_log_v10),
        })


@dataclasses.dataclass
class TinyV10ForwardTrace:
    v9_trace: TinyV9ForwardTrace
    v10_gate_score_per_layer: "_np.ndarray"   # (L,)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v9_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v10_forward_trace",
            "v9_trace_cid": self.v9_trace.cid(),
            "v10_gate_score_per_layer_cid": _ndarray_cid(
                self.v10_gate_score_per_layer),
        })


def _compute_v10_gate_score(
        v9_witness: Any,
        hidden_write_merit_mean: float,
        weights: Sequence[float],
        bias: float = W65_DEFAULT_V10_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer V10 composite gate score (vectorised stub)."""
    n_layers = int(v9_witness.n_layers)
    feats = _np.array([
        float(v9_witness.attention_entropy_mean),
        float(v9_witness.cache_similarity_mean),
        float(v9_witness.hidden_wins_primary_l1)
            / float(max(1, n_layers)),
        float(v9_witness.replay_dominance_witness_l1)
            / float(max(1, n_layers)),
        float(v9_witness.hidden_state_trust_ledger_l1)
            / float(max(1, n_layers)),
        float(hidden_write_merit_mean),
    ], dtype=_np.float64)
    w = _np.array([float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full((n_layers,), float(sig), dtype=_np.float64)
    # Distribute the layer-specific entropy back into the score.
    if (v9_witness.attention_entropy_mean is not None
            and n_layers > 0):
        return _np.round(per_layer, decimals=12)
    return _np.zeros((n_layers,), dtype=_np.float64)


def forward_tiny_substrate_v10(
        params: TinyV10SubstrateParams,
        token_ids: Sequence[int],
        *,
        v10_kv_cache: TinyV10KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV10ForwardTrace, TinyV10KVCache]:
    """V10 forward = V9 forward + per-layer V10 composite gate."""
    cfg = params.config
    base_v9 = (
        v10_kv_cache.v9_cache if v10_kv_cache is not None else None)
    v9_trace, new_v9 = forward_tiny_substrate_v9(
        params.v9_params, list(token_ids),
        v9_kv_cache=base_v9,
        attention_bias_per_layer=attention_bias_per_layer)
    if v10_kv_cache is None:
        v10_new = TinyV10KVCache.empty(
            int(cfg.v9.n_layers), n_heads=int(cfg.v9.n_heads),
            max_len=int(cfg.v9.max_len), d_key=int(cfg.v9.d_key))
    else:
        v10_new = v10_kv_cache.clone()
    v10_new.v9_cache = new_v9
    # Pad hidden_write_merit if T grew.
    new_T = int(new_v9.v8_cache.v7_cache.cache_write_ledger.shape[2])
    cur_T = int(v10_new.hidden_write_merit.shape[2])
    if new_T > cur_T:
        L = int(v10_new.hidden_write_merit.shape[0])
        H = int(v10_new.hidden_write_merit.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v10_new.hidden_write_merit = _np.concatenate(
            [v10_new.hidden_write_merit, pad], axis=-1)
    v9_w = emit_tiny_substrate_v9_forward_witness(v9_trace, new_v9)
    hwm_mean = float(v10_new.hidden_write_merit.mean())
    gate = _compute_v10_gate_score(
        v9_w, hwm_mean, weights=cfg.gate_weights)
    v10_new.v10_gate_score_per_layer = gate
    v10_new.write_log_v10.append({
        "schema": W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
        "kind": "forward_v10",
        "n_new_tokens": int(len(list(token_ids))),
        "gate_score_mean": float(gate.mean()) if gate.size else 0.0,
        "hidden_write_merit_mean": float(hwm_mean),
    })
    trace = TinyV10ForwardTrace(
        v9_trace=v9_trace,
        v10_gate_score_per_layer=gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v10_new


def record_hidden_write_merit_v10(
        cache: TinyV10KVCache, *,
        layer_index: int, head_index: int, slot: int,
        merit: float,
) -> None:
    """Record a per-(L, H, T) hidden-write merit in [0, 1]."""
    L, H, T = cache.hidden_write_merit.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.hidden_write_merit = _np.concatenate(
            [cache.hidden_write_merit, pad], axis=-1)
    m = float(max(0.0, min(1.0, float(merit))))
    cache.hidden_write_merit[
        int(layer_index), int(head_index), int(slot)] = m
    cache.write_log_v10.append({
        "schema": W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
        "kind": "hidden_write_merit_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "merit": float(round(m, 12)),
    })


def register_role_kv_bank_v10(
        cache: TinyV10KVCache, *,
        role: str,
        offset_matrix: "_np.ndarray",
        max_n_roles: int = W65_DEFAULT_V10_MAX_ROLES,
) -> None:
    """Register a per-role KV bank offset matrix. FIFO-evict the
    oldest role when at capacity."""
    r = str(role)
    if r in cache.role_kv_bank:
        cache.role_kv_bank[r] = _np.asarray(
            offset_matrix, dtype=_np.float64).copy()
        return
    if len(cache.role_order) >= int(max_n_roles):
        oldest = cache.role_order.pop(0)
        cache.role_kv_bank.pop(oldest, None)
    cache.role_kv_bank[r] = _np.asarray(
        offset_matrix, dtype=_np.float64).copy()
    cache.role_order.append(r)
    cache.write_log_v10.append({
        "schema": W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
        "kind": "role_kv_bank_register",
        "role": str(r),
        "shape": list(offset_matrix.shape),
    })


def role_kv_bank_summary_v10(
        cache: TinyV10KVCache,
) -> dict[str, Any]:
    return {
        "n_roles": int(len(cache.role_kv_bank)),
        "role_order": list(cache.role_order),
        "max_l1_per_role": {
            r: float(_np.linalg.norm(v.ravel(), ord=1))
            for r, v in sorted(cache.role_kv_bank.items())},
    }


@dataclasses.dataclass(frozen=True)
class V10SubstrateCheckpoint:
    schema: str
    v9_cache_cid: str
    hidden_write_merit_cid: str
    role_kv_bank_cids: tuple[tuple[str, str], ...]
    role_order: tuple[str, ...]
    v10_gate_score_per_layer_cid: str
    n_tokens: int
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "v9_cache_cid": str(self.v9_cache_cid),
            "hidden_write_merit_cid": str(
                self.hidden_write_merit_cid),
            "role_kv_bank_cids": [
                [str(r), str(c)] for r, c
                in self.role_kv_bank_cids],
            "role_order": list(self.role_order),
            "v10_gate_score_per_layer_cid": str(
                self.v10_gate_score_per_layer_cid),
            "n_tokens": int(self.n_tokens),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "v10_substrate_checkpoint",
            "checkpoint": self.to_dict()})


def substrate_checkpoint_v10(
        cache: TinyV10KVCache,
) -> V10SubstrateCheckpoint:
    """Serializable token-bounded snapshot of the V10 cache."""
    return V10SubstrateCheckpoint(
        schema=W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
        v9_cache_cid=str(cache.v9_cache.cid()),
        hidden_write_merit_cid=_ndarray_cid(
            cache.hidden_write_merit),
        role_kv_bank_cids=tuple(
            (str(k), _ndarray_cid(v))
            for k, v in sorted(cache.role_kv_bank.items())),
        role_order=tuple(cache.role_order),
        v10_gate_score_per_layer_cid=(
            "none"
            if cache.v10_gate_score_per_layer is None
            else _ndarray_cid(cache.v10_gate_score_per_layer)),
        n_tokens=int(cache.n_tokens()),
        n_layers=int(cache.n_layers()),
    )


def substrate_restore_v10(
        checkpoint: V10SubstrateCheckpoint,
        cache: TinyV10KVCache,
) -> bool:
    """Confirm the supplied cache matches the checkpoint by CID
    (deterministic-restore predicate). Returns True iff every CID
    matches."""
    ours = substrate_checkpoint_v10(cache)
    return bool(ours.cid() == checkpoint.cid())


def substrate_checkpoint_reuse_flops_v10(
        *, n_tokens: int, recompute_flops_per_token: int = 1000,
        reuse_flops_per_token: int = 100,
) -> dict[str, Any]:
    """Substrate cache-reuse-vs-recompute flop budget."""
    n = int(max(0, n_tokens))
    reuse = int(reuse_flops_per_token) * n
    recompute = int(recompute_flops_per_token) * n
    saving = int(recompute - reuse)
    ratio = (
        float(saving) / float(recompute)
        if recompute > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "reuse_flops": int(reuse),
        "recompute_flops": int(recompute),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class TinyV10SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v9_witness_cid: str
    v10_gate_score_per_layer_cid: str
    v10_gate_score_mean: float
    hidden_write_merit_l1: float
    hidden_write_merit_mean: float
    n_roles_in_bank: int
    role_order: tuple[str, ...]
    checkpoint_cid: str
    n_layers: int
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v9_witness_cid": str(self.v9_witness_cid),
            "v10_gate_score_per_layer_cid": str(
                self.v10_gate_score_per_layer_cid),
            "v10_gate_score_mean": float(round(
                self.v10_gate_score_mean, 12)),
            "hidden_write_merit_l1": float(round(
                self.hidden_write_merit_l1, 12)),
            "hidden_write_merit_mean": float(round(
                self.hidden_write_merit_mean, 12)),
            "n_roles_in_bank": int(self.n_roles_in_bank),
            "role_order": list(self.role_order),
            "checkpoint_cid": str(self.checkpoint_cid),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v10_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v10_forward_witness(
        trace: TinyV10ForwardTrace,
        cache: TinyV10KVCache,
) -> TinyV10SubstrateForwardWitness:
    v9_w = emit_tiny_substrate_v9_forward_witness(
        trace.v9_trace, cache.v9_cache)
    ck = substrate_checkpoint_v10(cache)
    return TinyV10SubstrateForwardWitness(
        schema=W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v9_w.token_ids),
        v9_witness_cid=str(v9_w.cid()),
        v10_gate_score_per_layer_cid=_ndarray_cid(
            trace.v10_gate_score_per_layer),
        v10_gate_score_mean=float(
            trace.v10_gate_score_per_layer.mean())
            if trace.v10_gate_score_per_layer.size else 0.0,
        hidden_write_merit_l1=float(_np.linalg.norm(
            cache.hidden_write_merit.ravel(), ord=1)),
        hidden_write_merit_mean=float(
            cache.hidden_write_merit.mean()
            if cache.hidden_write_merit.size else 0.0),
        n_roles_in_bank=int(len(cache.role_kv_bank)),
        role_order=tuple(cache.role_order),
        checkpoint_cid=str(ck.cid()),
        n_layers=int(cache.n_layers()),
    )


def build_default_tiny_substrate_v10(
        *, seed: int = W65_DEFAULT_V10_SEED,
) -> TinyV10SubstrateParams:
    return TinyV10SubstrateParams.init(
        TinyV10SubstrateConfig.default(seed=int(seed)))


def tokenize_bytes_v10(
        text: str, *,
        max_len: int = W65_DEFAULT_V10_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    return tokenize_bytes_v9(
        text, max_len=int(max_len), add_bos=bool(add_bos))


def detokenize_bytes_v10(token_ids: Sequence[int]) -> str:
    from .tiny_substrate_v9 import detokenize_bytes_v9
    return detokenize_bytes_v9(list(token_ids))


__all__ = [
    "W65_TINY_SUBSTRATE_V10_SCHEMA_VERSION",
    "W65_TINY_V10_VOCAB_SIZE",
    "W65_DEFAULT_V10_D_MODEL",
    "W65_DEFAULT_V10_N_HEADS",
    "W65_DEFAULT_V10_N_KV_HEADS",
    "W65_DEFAULT_V10_N_LAYERS",
    "W65_DEFAULT_V10_FF_HIDDEN",
    "W65_DEFAULT_V10_MAX_LEN",
    "W65_DEFAULT_V10_INIT_SCALE",
    "W65_DEFAULT_V10_SEED",
    "W65_DEFAULT_V10_MAX_ROLES",
    "W65_DEFAULT_V10_GATE_BIAS",
    "TinyV10SubstrateConfig",
    "TinyV10SubstrateParams",
    "TinyV10KVCache",
    "TinyV10ForwardTrace",
    "TinyV10SubstrateForwardWitness",
    "V10SubstrateCheckpoint",
    "forward_tiny_substrate_v10",
    "record_hidden_write_merit_v10",
    "register_role_kv_bank_v10",
    "role_kv_bank_summary_v10",
    "substrate_checkpoint_v10",
    "substrate_restore_v10",
    "substrate_checkpoint_reuse_flops_v10",
    "emit_tiny_substrate_v10_forward_witness",
    "build_default_tiny_substrate_v10",
    "tokenize_bytes_v10",
    "detokenize_bytes_v10",
]
