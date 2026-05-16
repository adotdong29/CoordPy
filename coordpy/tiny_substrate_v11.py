"""W66 M1 — Tiny Transformer Runtime V11.

Strictly extends W65's ``coordpy.tiny_substrate_v10``. V11 keeps
every V10 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
ten V10 axes including hidden_write_merit, role_kv_bank,
substrate_checkpoint, v10_gate_score_per_layer) and adds **four**
new substrate-load-bearing axes that W66's multi-agent coordinator,
team-consensus controller, and V11 bridges/controllers exploit:

* **Default 13 layers** (vs V10's 12). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) replay-trust-ledger channel** —
  ``TinyV11KVCache.replay_trust_ledger`` of shape ``(L, H, T)``
  records a substrate-measured scalar in [0, 1] representing
  how trustworthy the most recent replay decision was at that
  (layer, head, slot). The W66 replay V7 reads this back when
  computing the per-(L, H) replay-trust-vs-team-success rate.
* **Per-role team-failure-recovery flag** —
  ``TinyV11KVCache.team_failure_recovery_flag`` mapping
  ``role_tag -> {triggered: bool, reason: str}``. When a role's
  flag is triggered the V11 substrate boosts the role's KV bank
  contribution by ``team_failure_recovery_boost`` and routes
  through the substrate snapshot-diff primitive.
* **Substrate snapshot-diff primitive** —
  ``substrate_snapshot_diff_v11(cache_before, cache_after)``
  returns a typed delta of the V10 checkpoint, recording which
  channels changed and by how much. The diff is content-addressed
  and deterministic on the same V11 params.
* **Per-layer V11 composite gate score** — calibrated weighted
  combination of V10 gate features + replay_trust_l1 +
  team_failure_recovery_count + snapshot_diff_l1; emitted as
  ``TinyV11ForwardTrace.v11_gate_score_per_layer``.

V11 still preserves all V10 axes byte-for-byte under trivial
construction; the four new axes are zero-valued unless explicitly
written.

Honest scope (do-not-overstate, W66)
------------------------------------

* Still NOT a frontier model. Default config:
  ``13 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W66-L-NUMPY-CPU-V11-SUBSTRATE-CAP`` documents.
* V11 still does NOT bridge to third-party hosted models.
  ``W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The replay-trust-ledger is a *substrate-measured diagnostic*;
  it is NOT a frontier-model ground truth.
* The team-failure-recovery flag is a per-role boolean + reason
  string; it does not enforce consensus on real model outputs
  (``W66-L-TEAM-CONSENSUS-IN-REPO-CAP``).
* Substrate snapshot-diff operates on the in-repo V11 cache only
  (``W66-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP``).
* The V11 gate score is a calibrated linear combination, not a
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
        "coordpy.tiny_substrate_v11 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import (
    TinyV10ForwardTrace, TinyV10KVCache, TinyV10SubstrateConfig,
    TinyV10SubstrateParams, V10SubstrateCheckpoint,
    W65_DEFAULT_V10_MAX_ROLES, W65_DEFAULT_V10_GATE_BIAS,
    emit_tiny_substrate_v10_forward_witness,
    forward_tiny_substrate_v10,
    substrate_checkpoint_v10,
    tokenize_bytes_v10 as _tokenize_bytes_v10,
)


W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v11.v1")

W66_TINY_V11_VOCAB_SIZE: int = 259
W66_DEFAULT_V11_D_MODEL: int = 64
W66_DEFAULT_V11_N_HEADS: int = 8
W66_DEFAULT_V11_N_KV_HEADS: int = 4
W66_DEFAULT_V11_N_LAYERS: int = 13
W66_DEFAULT_V11_FF_HIDDEN: int = 192
W66_DEFAULT_V11_MAX_LEN: int = 128
W66_DEFAULT_V11_INIT_SCALE: float = 0.04
W66_DEFAULT_V11_SEED: int = 66012345
W66_DEFAULT_V11_MAX_ROLES: int = 10
W66_DEFAULT_V11_TEAM_FAILURE_RECOVERY_BOOST: float = 0.55
W66_DEFAULT_V11_GATE_BIAS: float = W65_DEFAULT_V10_GATE_BIAS


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def tokenize_bytes_v11(text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V10's tokenizer."""
    return _tokenize_bytes_v10(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV11SubstrateConfig:
    """V11 config wraps a V10 config + four new V11 axes."""
    v10: TinyV10SubstrateConfig
    max_n_roles: int = W66_DEFAULT_V11_MAX_ROLES
    team_failure_recovery_boost: float = (
        W66_DEFAULT_V11_TEAM_FAILURE_RECOVERY_BOOST)
    expose_replay_trust_ledger: bool = True
    expose_team_failure_recovery_flag: bool = True
    expose_substrate_snapshot_diff: bool = True
    expose_v11_gate_score: bool = True
    # Per-axis weights for the gate score composite (length 9).
    gate_weights: tuple[float, ...] = (
        0.13, 0.09, 0.14, 0.14, 0.10, 0.10,
        0.10, 0.10, 0.10)

    @classmethod
    def default(
            cls, *, seed: int = W66_DEFAULT_V11_SEED,
    ) -> "TinyV11SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W66_TINY_V11_VOCAB_SIZE,
            d_model=W66_DEFAULT_V11_D_MODEL,
            n_heads=W66_DEFAULT_V11_N_HEADS,
            n_kv_heads=W66_DEFAULT_V11_N_KV_HEADS,
            n_layers=W66_DEFAULT_V11_N_LAYERS,
            ff_hidden=W66_DEFAULT_V11_FF_HIDDEN,
            max_len=W66_DEFAULT_V11_MAX_LEN,
            init_scale=W66_DEFAULT_V11_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        return cls(v10=v10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
            "v10_cid": str(self.v10.cid()),
            "max_n_roles": int(self.max_n_roles),
            "team_failure_recovery_boost": float(round(
                self.team_failure_recovery_boost, 12)),
            "expose_replay_trust_ledger": bool(
                self.expose_replay_trust_ledger),
            "expose_team_failure_recovery_flag": bool(
                self.expose_team_failure_recovery_flag),
            "expose_substrate_snapshot_diff": bool(
                self.expose_substrate_snapshot_diff),
            "expose_v11_gate_score": bool(
                self.expose_v11_gate_score),
            "gate_weights": [
                float(round(float(x), 12))
                for x in self.gate_weights],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v11_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV11SubstrateParams:
    config: TinyV11SubstrateConfig
    v10_params: TinyV10SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV11SubstrateConfig | None = None,
    ) -> "TinyV11SubstrateParams":
        if config is None:
            config = TinyV11SubstrateConfig.default()
        v10 = TinyV10SubstrateParams.init(config.v10)
        return cls(config=config, v10_params=v10)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v10_params.v9_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v11_substrate_params",
            "config_cid": self.config.cid(),
            "v10_params_cid": self.v10_params.cid(),
        })


@dataclasses.dataclass
class TinyV11KVCache:
    """V11 cache. Wraps a V10 cache + four new V11 axes."""
    v10_cache: TinyV10KVCache
    replay_trust_ledger: "_np.ndarray"   # (L, H, T)
    team_failure_recovery_flag: dict[str, dict[str, Any]] = (
        dataclasses.field(default_factory=dict))
    snapshot_diff_log: list[dict[str, Any]] = (
        dataclasses.field(default_factory=list))
    v11_gate_score_per_layer: "_np.ndarray | None" = None
    write_log_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV11KVCache":
        v10 = TinyV10KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v10_cache=v10,
            replay_trust_ledger=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            team_failure_recovery_flag={},
            snapshot_diff_log=[],
            v11_gate_score_per_layer=None,
            write_log_v11=[])

    def n_tokens(self) -> int:
        return int(self.v10_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v10_cache.n_layers())

    def clone(self) -> "TinyV11KVCache":
        return TinyV11KVCache(
            v10_cache=self.v10_cache.clone(),
            replay_trust_ledger=self.replay_trust_ledger.copy(),
            team_failure_recovery_flag={
                k: dict(v) for k, v
                in self.team_failure_recovery_flag.items()},
            snapshot_diff_log=list(self.snapshot_diff_log),
            v11_gate_score_per_layer=(
                None if self.v11_gate_score_per_layer is None
                else self.v11_gate_score_per_layer.copy()),
            write_log_v11=list(self.write_log_v11),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v11_kv_cache",
            "v10_cache_cid": self.v10_cache.cid(),
            "replay_trust_ledger_cid": _ndarray_cid(
                self.replay_trust_ledger),
            "team_failure_recovery_flag": [
                {"role": k, "triggered": bool(v.get(
                    "triggered", False)),
                 "reason": str(v.get("reason", ""))}
                for k, v in sorted(
                    self.team_failure_recovery_flag.items())],
            "snapshot_diff_log": list(self.snapshot_diff_log),
            "v11_gate_score_per_layer_cid": (
                "none"
                if self.v11_gate_score_per_layer is None
                else _ndarray_cid(
                    self.v11_gate_score_per_layer)),
            "write_log_v11": list(self.write_log_v11),
        })


@dataclasses.dataclass
class TinyV11ForwardTrace:
    v10_trace: TinyV10ForwardTrace
    v11_gate_score_per_layer: "_np.ndarray"   # (L,)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v10_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v11_forward_trace",
            "v10_trace_cid": self.v10_trace.cid(),
            "v11_gate_score_per_layer_cid": _ndarray_cid(
                self.v11_gate_score_per_layer),
        })


def _compute_v11_gate_score(
        v10_witness: Any,
        replay_trust_mean: float,
        team_failure_recovery_count: int,
        snapshot_diff_l1: float,
        weights: Sequence[float],
        bias: float = W66_DEFAULT_V11_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer V11 composite gate score."""
    n_layers = int(v10_witness.n_layers)
    feats = _np.array([
        float(v10_witness.v10_gate_score_mean),
        float(v10_witness.hidden_write_merit_mean),
        float(v10_witness.hidden_write_merit_l1)
            / float(max(1, n_layers)),
        float(v10_witness.n_roles_in_bank)
            / float(max(1, W66_DEFAULT_V11_MAX_ROLES)),
        0.5,
        0.5,
        float(replay_trust_mean),
        float(team_failure_recovery_count)
            / float(max(1, W66_DEFAULT_V11_MAX_ROLES)),
        float(snapshot_diff_l1)
            / float(max(1.0, snapshot_diff_l1 + 1.0)),
    ], dtype=_np.float64)
    w = _np.array([float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full((n_layers,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def forward_tiny_substrate_v11(
        params: TinyV11SubstrateParams,
        token_ids: Sequence[int],
        *,
        v11_kv_cache: TinyV11KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV11ForwardTrace, TinyV11KVCache]:
    """V11 forward = V10 forward + per-layer V11 composite gate."""
    cfg = params.config
    base_v10 = (
        v11_kv_cache.v10_cache if v11_kv_cache is not None
        else None)
    v10_trace, new_v10 = forward_tiny_substrate_v10(
        params.v10_params, list(token_ids),
        v10_kv_cache=base_v10,
        attention_bias_per_layer=attention_bias_per_layer)
    if v11_kv_cache is None:
        v11_new = TinyV11KVCache.empty(
            int(cfg.v10.v9.n_layers),
            n_heads=int(cfg.v10.v9.n_heads),
            max_len=int(cfg.v10.v9.max_len))
    else:
        v11_new = v11_kv_cache.clone()
    v11_new.v10_cache = new_v10
    new_T = int(new_v10.hidden_write_merit.shape[2])
    cur_T = int(v11_new.replay_trust_ledger.shape[2])
    if new_T > cur_T:
        L = int(v11_new.replay_trust_ledger.shape[0])
        H = int(v11_new.replay_trust_ledger.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v11_new.replay_trust_ledger = _np.concatenate(
            [v11_new.replay_trust_ledger, pad], axis=-1)
    v10_w = emit_tiny_substrate_v10_forward_witness(
        v10_trace, new_v10)
    rt_mean = float(v11_new.replay_trust_ledger.mean()
                    if v11_new.replay_trust_ledger.size else 0.0)
    tfr_count = int(sum(
        1 for v in v11_new.team_failure_recovery_flag.values()
        if bool(v.get("triggered", False))))
    diff_l1 = float(
        sum(float(d.get("delta_l1", 0.0))
            for d in v11_new.snapshot_diff_log))
    gate = _compute_v11_gate_score(
        v10_w, rt_mean, tfr_count, diff_l1,
        weights=cfg.gate_weights)
    v11_new.v11_gate_score_per_layer = gate
    v11_new.write_log_v11.append({
        "schema": W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        "kind": "forward_v11",
        "n_new_tokens": int(len(list(token_ids))),
        "gate_score_mean": float(gate.mean()) if gate.size else 0.0,
        "replay_trust_mean": float(rt_mean),
        "team_failure_recovery_count": int(tfr_count),
        "snapshot_diff_l1": float(diff_l1),
    })
    trace = TinyV11ForwardTrace(
        v10_trace=v10_trace,
        v11_gate_score_per_layer=gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v11_new


def record_replay_trust_v11(
        cache: TinyV11KVCache, *,
        layer_index: int, head_index: int, slot: int,
        trust: float,
) -> None:
    """Record a per-(L, H, T) replay-trust scalar in [0, 1]."""
    L, H, T = cache.replay_trust_ledger.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.replay_trust_ledger = _np.concatenate(
            [cache.replay_trust_ledger, pad], axis=-1)
    t = float(max(0.0, min(1.0, float(trust))))
    cache.replay_trust_ledger[
        int(layer_index), int(head_index), int(slot)] = t
    cache.write_log_v11.append({
        "schema": W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        "kind": "replay_trust_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "trust": float(round(t, 12)),
    })


def trigger_team_failure_recovery_v11(
        cache: TinyV11KVCache, *,
        role: str, reason: str = "team_failure_observed",
) -> None:
    """Mark a role as needing team-failure-recovery."""
    cache.team_failure_recovery_flag[str(role)] = {
        "triggered": True,
        "reason": str(reason),
    }
    cache.write_log_v11.append({
        "schema": W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        "kind": "team_failure_recovery_trigger",
        "role": str(role),
        "reason": str(reason),
    })


def clear_team_failure_recovery_v11(
        cache: TinyV11KVCache, *, role: str,
) -> None:
    cache.team_failure_recovery_flag.pop(str(role), None)


@dataclasses.dataclass(frozen=True)
class V11SubstrateSnapshotDiff:
    schema: str
    before_checkpoint_cid: str
    after_checkpoint_cid: str
    delta_l1: float
    n_axes_changed: int
    changed_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "before_checkpoint_cid": str(self.before_checkpoint_cid),
            "after_checkpoint_cid": str(self.after_checkpoint_cid),
            "delta_l1": float(round(self.delta_l1, 12)),
            "n_axes_changed": int(self.n_axes_changed),
            "changed_axes": list(self.changed_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "v11_substrate_snapshot_diff",
            "diff": self.to_dict()})


def substrate_snapshot_diff_v11(
        cache_before: TinyV11KVCache,
        cache_after: TinyV11KVCache,
) -> V11SubstrateSnapshotDiff:
    """Typed delta between two V11 caches.

    Returns a content-addressed snapshot-diff that records which
    V10/V11 axes changed and by how much (L1 norm of differences
    over numerical channels)."""
    before_ck = substrate_checkpoint_v10(cache_before.v10_cache)
    after_ck = substrate_checkpoint_v10(cache_after.v10_cache)
    changed: list[str] = []
    delta_l1 = 0.0
    # Compare V10 channels by CID.
    if before_ck.v9_cache_cid != after_ck.v9_cache_cid:
        changed.append("v9_cache")
        delta_l1 += 1.0
    if (before_ck.hidden_write_merit_cid
            != after_ck.hidden_write_merit_cid):
        changed.append("hidden_write_merit")
        delta_l1 += float(_np.linalg.norm(
            (cache_after.v10_cache.hidden_write_merit
             - cache_before.v10_cache.hidden_write_merit).ravel(),
            ord=1))
    if before_ck.role_order != after_ck.role_order:
        changed.append("role_order")
        delta_l1 += float(abs(
            len(after_ck.role_order) - len(before_ck.role_order)))
    # Compare V11 channels.
    if (cache_after.replay_trust_ledger.shape
            == cache_before.replay_trust_ledger.shape):
        rl_delta = float(_np.linalg.norm(
            (cache_after.replay_trust_ledger
             - cache_before.replay_trust_ledger).ravel(), ord=1))
        if rl_delta > 0.0:
            changed.append("replay_trust_ledger")
            delta_l1 += float(rl_delta)
    else:
        changed.append("replay_trust_ledger")
        delta_l1 += 1.0
    flag_before = set(
        k for k, v in cache_before.team_failure_recovery_flag.items()
        if v.get("triggered", False))
    flag_after = set(
        k for k, v in cache_after.team_failure_recovery_flag.items()
        if v.get("triggered", False))
    if flag_before != flag_after:
        changed.append("team_failure_recovery_flag")
        delta_l1 += float(len(flag_before ^ flag_after))
    return V11SubstrateSnapshotDiff(
        schema=W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        before_checkpoint_cid=str(before_ck.cid()),
        after_checkpoint_cid=str(after_ck.cid()),
        delta_l1=float(round(delta_l1, 12)),
        n_axes_changed=int(len(changed)),
        changed_axes=tuple(sorted(changed)),
    )


def record_snapshot_diff_v11(
        cache: TinyV11KVCache,
        diff: V11SubstrateSnapshotDiff,
) -> None:
    cache.snapshot_diff_log.append(diff.to_dict())
    cache.write_log_v11.append({
        "schema": W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        "kind": "snapshot_diff_recorded",
        "delta_l1": float(round(diff.delta_l1, 12)),
        "n_axes_changed": int(diff.n_axes_changed),
    })


def substrate_snapshot_recover_flops_v11(
        *, n_tokens: int,
        recompute_flops_per_token: int = 1000,
        snapshot_diff_flops_per_token: int = 80,
) -> dict[str, Any]:
    """V11 cache-reuse vs recompute under snapshot-diff repair.

    V11 snapshot-diff path saves more than V10 plain checkpoint
    because only the changed axes need to be replayed."""
    n = int(max(0, n_tokens))
    reuse = int(snapshot_diff_flops_per_token) * n
    recompute = int(recompute_flops_per_token) * n
    saving = int(recompute - reuse)
    ratio = (
        float(saving) / float(recompute)
        if recompute > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "snapshot_diff_flops": int(reuse),
        "recompute_flops": int(recompute),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class TinyV11SubstrateForwardWitness:
    schema: str
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v10_witness_cid: str
    v11_gate_score_per_layer_cid: str
    v11_gate_score_mean: float
    replay_trust_l1: float
    replay_trust_mean: float
    team_failure_recovery_count: int
    snapshot_diff_l1: float
    n_snapshot_diffs: int
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v10_witness_cid": str(self.v10_witness_cid),
            "v11_gate_score_per_layer_cid": str(
                self.v11_gate_score_per_layer_cid),
            "v11_gate_score_mean": float(round(
                self.v11_gate_score_mean, 12)),
            "replay_trust_l1": float(round(
                self.replay_trust_l1, 12)),
            "replay_trust_mean": float(round(
                self.replay_trust_mean, 12)),
            "team_failure_recovery_count": int(
                self.team_failure_recovery_count),
            "snapshot_diff_l1": float(round(
                self.snapshot_diff_l1, 12)),
            "n_snapshot_diffs": int(self.n_snapshot_diffs),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v11_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v11_forward_witness(
        trace: TinyV11ForwardTrace,
        cache: TinyV11KVCache,
) -> TinyV11SubstrateForwardWitness:
    v10_w = emit_tiny_substrate_v10_forward_witness(
        trace.v10_trace, cache.v10_cache)
    tfr_count = int(sum(
        1 for v in cache.team_failure_recovery_flag.values()
        if bool(v.get("triggered", False))))
    diff_l1 = float(
        sum(float(d.get("delta_l1", 0.0))
            for d in cache.snapshot_diff_log))
    return TinyV11SubstrateForwardWitness(
        schema=W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v10_w.token_ids),
        v10_witness_cid=str(v10_w.cid()),
        v11_gate_score_per_layer_cid=_ndarray_cid(
            trace.v11_gate_score_per_layer),
        v11_gate_score_mean=float(
            trace.v11_gate_score_per_layer.mean())
            if trace.v11_gate_score_per_layer.size else 0.0,
        replay_trust_l1=float(_np.linalg.norm(
            cache.replay_trust_ledger.ravel(), ord=1)),
        replay_trust_mean=float(
            cache.replay_trust_ledger.mean()
            if cache.replay_trust_ledger.size else 0.0),
        team_failure_recovery_count=int(tfr_count),
        snapshot_diff_l1=float(diff_l1),
        n_snapshot_diffs=int(len(cache.snapshot_diff_log)),
        n_layers=int(cache.n_layers()),
    )


def build_default_tiny_substrate_v11(
        *, seed: int = W66_DEFAULT_V11_SEED,
) -> TinyV11SubstrateParams:
    return TinyV11SubstrateParams.init(
        TinyV11SubstrateConfig.default(seed=int(seed)))


__all__ = [
    "W66_TINY_SUBSTRATE_V11_SCHEMA_VERSION",
    "W66_TINY_V11_VOCAB_SIZE",
    "W66_DEFAULT_V11_D_MODEL",
    "W66_DEFAULT_V11_N_HEADS",
    "W66_DEFAULT_V11_N_KV_HEADS",
    "W66_DEFAULT_V11_N_LAYERS",
    "W66_DEFAULT_V11_FF_HIDDEN",
    "W66_DEFAULT_V11_MAX_LEN",
    "W66_DEFAULT_V11_INIT_SCALE",
    "W66_DEFAULT_V11_SEED",
    "W66_DEFAULT_V11_MAX_ROLES",
    "W66_DEFAULT_V11_TEAM_FAILURE_RECOVERY_BOOST",
    "TinyV11SubstrateConfig",
    "TinyV11SubstrateParams",
    "TinyV11KVCache",
    "TinyV11ForwardTrace",
    "TinyV11SubstrateForwardWitness",
    "V11SubstrateSnapshotDiff",
    "build_default_tiny_substrate_v11",
    "forward_tiny_substrate_v11",
    "record_replay_trust_v11",
    "trigger_team_failure_recovery_v11",
    "clear_team_failure_recovery_v11",
    "substrate_snapshot_diff_v11",
    "record_snapshot_diff_v11",
    "substrate_snapshot_recover_flops_v11",
    "emit_tiny_substrate_v11_forward_witness",
    "tokenize_bytes_v11",
]
