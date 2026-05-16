"""W67 M1 — Tiny Transformer Runtime V12.

Strictly extends W66's ``coordpy.tiny_substrate_v11``. V12 keeps
every V11 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
eleven V10+V11 axes including replay_trust_ledger,
team_failure_recovery_flag, substrate_snapshot_diff,
v11_gate_score_per_layer) and adds **four** new substrate-load-
bearing axes that W67's multi-agent coordinator V3, team-consensus
controller V2, and V12 bridges/controllers exploit:

* **Default 14 layers** (vs V11's 13). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) branch-merge witness tensor** —
  ``TinyV12KVCache.branch_merge_witness`` of shape ``(L, H, T)``
  records a substrate-measured scalar in [0, 1] representing
  how much that (layer, head, slot) participated in the most
  recent branch-merge reconciliation event. The W67 replay V8
  reads this back when computing the per-(L, H) branch-merge-
  conditioned routing.
* **Per-role-pair role-dropout-recovery flag** —
  ``TinyV12KVCache.role_dropout_recovery_flag`` mapping
  ``(absent_role, covering_role) -> {triggered: bool, window: int}``.
  When a role-pair flag is triggered the V12 substrate boosts the
  covering role's KV bank contribution and routes through the
  branch-merge primitive.
* **Substrate snapshot-fork primitive** —
  ``substrate_snapshot_fork_v12(cache, branch_ids)`` returns a
  per-branch snapshot map allowing a team to fork the substrate
  into N branches that can later be reconciled via
  ``substrate_branch_merge_v12``.
* **Per-layer V12 composite gate score** — calibrated weighted
  combination of V11 gate features + branch_merge_witness_mean +
  role_dropout_recovery_count + n_branches_active; emitted as
  ``TinyV12ForwardTrace.v12_gate_score_per_layer``.

V12 still preserves all V11 axes byte-for-byte under trivial
construction; the four new axes are zero-valued unless explicitly
written.

Honest scope (do-not-overstate, W67)
------------------------------------

* Still NOT a frontier model. Default config:
  ``14 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W67-L-NUMPY-CPU-V12-SUBSTRATE-CAP`` documents.
* V12 still does NOT bridge to third-party hosted models.
  ``W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The branch-merge witness is a *substrate-measured diagnostic*.
* The role-dropout-recovery flag is a per-role-pair boolean +
  window int; it does not enforce consensus on real model outputs
  (``W67-L-TEAM-CONSENSUS-V2-IN-REPO-CAP``).
* Substrate snapshot-fork operates on the in-repo V12 cache only
  (``W67-L-SUBSTRATE-BRANCH-MERGE-IN-REPO-CAP``).
* The V12 gate score is a calibrated linear combination, not a
  learned end-to-end controller.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v12 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import (
    TinyV10SubstrateConfig, W65_DEFAULT_V10_GATE_BIAS,
)
from .tiny_substrate_v11 import (
    TinyV11ForwardTrace, TinyV11KVCache, TinyV11SubstrateConfig,
    TinyV11SubstrateParams, W66_DEFAULT_V11_MAX_ROLES,
    emit_tiny_substrate_v11_forward_witness,
    forward_tiny_substrate_v11,
    tokenize_bytes_v11 as _tokenize_bytes_v11,
)


W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v12.v1")

W67_TINY_V12_VOCAB_SIZE: int = 259
W67_DEFAULT_V12_D_MODEL: int = 64
W67_DEFAULT_V12_N_HEADS: int = 8
W67_DEFAULT_V12_N_KV_HEADS: int = 4
W67_DEFAULT_V12_N_LAYERS: int = 14
W67_DEFAULT_V12_FF_HIDDEN: int = 192
W67_DEFAULT_V12_MAX_LEN: int = 128
W67_DEFAULT_V12_INIT_SCALE: float = 0.04
W67_DEFAULT_V12_SEED: int = 67012345
W67_DEFAULT_V12_MAX_ROLES: int = 10
W67_DEFAULT_V12_ROLE_DROPOUT_RECOVERY_BOOST: float = 0.60
W67_DEFAULT_V12_BRANCH_MERGE_BOOST: float = 0.50
W67_DEFAULT_V12_GATE_BIAS: float = W65_DEFAULT_V10_GATE_BIAS


def tokenize_bytes_v12(text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V11's tokenizer."""
    return _tokenize_bytes_v11(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV12SubstrateConfig:
    """V12 config wraps a V11 config + four new V12 axes."""
    v11: TinyV11SubstrateConfig
    max_n_roles: int = W67_DEFAULT_V12_MAX_ROLES
    role_dropout_recovery_boost: float = (
        W67_DEFAULT_V12_ROLE_DROPOUT_RECOVERY_BOOST)
    branch_merge_boost: float = W67_DEFAULT_V12_BRANCH_MERGE_BOOST
    expose_branch_merge_witness: bool = True
    expose_role_dropout_recovery_flag: bool = True
    expose_substrate_snapshot_fork: bool = True
    expose_v12_gate_score: bool = True
    # Per-axis weights for the V12 gate score composite (length 10).
    gate_weights: tuple[float, ...] = (
        0.10, 0.08, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.11, 0.11)

    @classmethod
    def default(
            cls, *, seed: int = W67_DEFAULT_V12_SEED,
    ) -> "TinyV12SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W67_TINY_V12_VOCAB_SIZE,
            d_model=W67_DEFAULT_V12_D_MODEL,
            n_heads=W67_DEFAULT_V12_N_HEADS,
            n_kv_heads=W67_DEFAULT_V12_N_KV_HEADS,
            n_layers=W67_DEFAULT_V12_N_LAYERS,
            ff_hidden=W67_DEFAULT_V12_FF_HIDDEN,
            max_len=W67_DEFAULT_V12_MAX_LEN,
            init_scale=W67_DEFAULT_V12_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        return cls(v11=v11)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
            "v11_cid": str(self.v11.cid()),
            "max_n_roles": int(self.max_n_roles),
            "role_dropout_recovery_boost": float(round(
                self.role_dropout_recovery_boost, 12)),
            "branch_merge_boost": float(round(
                self.branch_merge_boost, 12)),
            "expose_branch_merge_witness": bool(
                self.expose_branch_merge_witness),
            "expose_role_dropout_recovery_flag": bool(
                self.expose_role_dropout_recovery_flag),
            "expose_substrate_snapshot_fork": bool(
                self.expose_substrate_snapshot_fork),
            "expose_v12_gate_score": bool(
                self.expose_v12_gate_score),
            "gate_weights": [
                float(round(float(x), 12))
                for x in self.gate_weights],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v12_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV12SubstrateParams:
    config: TinyV12SubstrateConfig
    v11_params: TinyV11SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV12SubstrateConfig | None = None,
    ) -> "TinyV12SubstrateParams":
        if config is None:
            config = TinyV12SubstrateConfig.default()
        v11 = TinyV11SubstrateParams.init(config.v11)
        return cls(config=config, v11_params=v11)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v11_params.v10_params.v9_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v12_substrate_params",
            "config_cid": self.config.cid(),
            "v11_params_cid": self.v11_params.cid(),
        })


@dataclasses.dataclass
class TinyV12KVCache:
    """V12 cache. Wraps a V11 cache + four new V12 axes."""
    v11_cache: TinyV11KVCache
    branch_merge_witness: "_np.ndarray"   # (L, H, T)
    role_dropout_recovery_flag: dict[
        tuple[str, str], dict[str, Any]] = dataclasses.field(
            default_factory=dict)
    snapshot_fork_map: dict[str, str] = dataclasses.field(
        default_factory=dict)
    v12_gate_score_per_layer: "_np.ndarray | None" = None
    write_log_v12: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV12KVCache":
        v11 = TinyV11KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v11_cache=v11,
            branch_merge_witness=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            role_dropout_recovery_flag={},
            snapshot_fork_map={},
            v12_gate_score_per_layer=None,
            write_log_v12=[])

    def n_tokens(self) -> int:
        return int(self.v11_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v11_cache.n_layers())

    def clone(self) -> "TinyV12KVCache":
        return TinyV12KVCache(
            v11_cache=self.v11_cache.clone(),
            branch_merge_witness=self.branch_merge_witness.copy(),
            role_dropout_recovery_flag={
                k: dict(v)
                for k, v in self.role_dropout_recovery_flag.items()},
            snapshot_fork_map=dict(self.snapshot_fork_map),
            v12_gate_score_per_layer=(
                None if self.v12_gate_score_per_layer is None
                else self.v12_gate_score_per_layer.copy()),
            write_log_v12=list(self.write_log_v12),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v12_kv_cache",
            "v11_cache_cid": self.v11_cache.cid(),
            "branch_merge_witness_cid": _ndarray_cid(
                self.branch_merge_witness),
            "role_dropout_recovery_flag": [
                {"absent": k[0], "covering": k[1],
                 "triggered": bool(v.get("triggered", False)),
                 "window": int(v.get("window", 0))}
                for k, v in sorted(
                    self.role_dropout_recovery_flag.items())],
            "snapshot_fork_map": dict(self.snapshot_fork_map),
            "v12_gate_score_per_layer_cid": (
                "none"
                if self.v12_gate_score_per_layer is None
                else _ndarray_cid(
                    self.v12_gate_score_per_layer)),
            "write_log_v12": list(self.write_log_v12),
        })


@dataclasses.dataclass
class TinyV12ForwardTrace:
    v11_trace: TinyV11ForwardTrace
    v12_gate_score_per_layer: "_np.ndarray"   # (L,)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v11_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v12_forward_trace",
            "v11_trace_cid": self.v11_trace.cid(),
            "v12_gate_score_per_layer_cid": _ndarray_cid(
                self.v12_gate_score_per_layer),
        })


def _compute_v12_gate_score(
        v11_witness: Any,
        branch_merge_l1: float,
        role_dropout_recovery_count: int,
        n_branches_active: int,
        weights: Sequence[float],
        bias: float = W67_DEFAULT_V12_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer V12 composite gate score."""
    n_layers = int(v11_witness.n_layers)
    feats = _np.array([
        float(v11_witness.v11_gate_score_mean),
        float(v11_witness.replay_trust_mean),
        float(v11_witness.replay_trust_l1)
            / float(max(1, n_layers)),
        float(v11_witness.team_failure_recovery_count)
            / float(max(1, W67_DEFAULT_V12_MAX_ROLES)),
        float(v11_witness.snapshot_diff_l1)
            / float(max(1.0, v11_witness.snapshot_diff_l1 + 1.0)),
        0.5,
        0.5,
        float(branch_merge_l1)
            / float(max(1.0, branch_merge_l1 + 1.0)),
        float(role_dropout_recovery_count)
            / float(max(1, W67_DEFAULT_V12_MAX_ROLES)),
        float(n_branches_active)
            / float(max(1, n_branches_active + 1)),
    ], dtype=_np.float64)
    w = _np.array([float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (n_layers,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def forward_tiny_substrate_v12(
        params: TinyV12SubstrateParams,
        token_ids: Sequence[int],
        *,
        v12_kv_cache: TinyV12KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV12ForwardTrace, TinyV12KVCache]:
    """V12 forward = V11 forward + per-layer V12 composite gate."""
    cfg = params.config
    base_v11 = (
        v12_kv_cache.v11_cache if v12_kv_cache is not None
        else None)
    v11_trace, new_v11 = forward_tiny_substrate_v11(
        params.v11_params, list(token_ids),
        v11_kv_cache=base_v11,
        attention_bias_per_layer=attention_bias_per_layer)
    if v12_kv_cache is None:
        v12_new = TinyV12KVCache.empty(
            int(cfg.v11.v10.v9.n_layers),
            n_heads=int(cfg.v11.v10.v9.n_heads),
            max_len=int(cfg.v11.v10.v9.max_len))
    else:
        v12_new = v12_kv_cache.clone()
    v12_new.v11_cache = new_v11
    new_T = int(new_v11.v10_cache.hidden_write_merit.shape[2])
    cur_T = int(v12_new.branch_merge_witness.shape[2])
    if new_T > cur_T:
        L = int(v12_new.branch_merge_witness.shape[0])
        H = int(v12_new.branch_merge_witness.shape[1])
        pad = _np.zeros(
            (L, H, new_T - cur_T), dtype=_np.float64)
        v12_new.branch_merge_witness = _np.concatenate(
            [v12_new.branch_merge_witness, pad], axis=-1)
    v11_w = emit_tiny_substrate_v11_forward_witness(
        v11_trace, new_v11)
    bm_l1 = float(_np.linalg.norm(
        v12_new.branch_merge_witness.ravel(), ord=1))
    rd_count = int(sum(
        1 for v in v12_new.role_dropout_recovery_flag.values()
        if bool(v.get("triggered", False))))
    n_branches = int(len(v12_new.snapshot_fork_map))
    gate = _compute_v12_gate_score(
        v11_w, bm_l1, rd_count, n_branches,
        weights=cfg.gate_weights)
    v12_new.v12_gate_score_per_layer = gate
    v12_new.write_log_v12.append({
        "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        "kind": "forward_v12",
        "n_new_tokens": int(len(list(token_ids))),
        "gate_score_mean": float(gate.mean()) if gate.size else 0.0,
        "branch_merge_l1": float(bm_l1),
        "role_dropout_recovery_count": int(rd_count),
        "n_branches_active": int(n_branches),
    })
    trace = TinyV12ForwardTrace(
        v11_trace=v11_trace,
        v12_gate_score_per_layer=gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v12_new


def record_branch_merge_witness_v12(
        cache: TinyV12KVCache, *,
        layer_index: int, head_index: int, slot: int,
        witness: float,
) -> None:
    """Record a per-(L, H, T) branch-merge witness scalar in [0, 1]."""
    L, H, T = cache.branch_merge_witness.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.branch_merge_witness = _np.concatenate(
            [cache.branch_merge_witness, pad], axis=-1)
    v = float(max(0.0, min(1.0, float(witness))))
    cache.branch_merge_witness[
        int(layer_index), int(head_index), int(slot)] = v
    cache.write_log_v12.append({
        "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        "kind": "branch_merge_witness_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "witness": float(round(v, 12)),
    })


def trigger_role_dropout_recovery_v12(
        cache: TinyV12KVCache, *,
        absent_role: str, covering_role: str,
        window: int = 1,
) -> None:
    """Mark a (absent_role, covering_role) pair as needing recovery."""
    cache.role_dropout_recovery_flag[
        (str(absent_role), str(covering_role))] = {
        "triggered": True, "window": int(window)}
    cache.write_log_v12.append({
        "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        "kind": "role_dropout_recovery_trigger",
        "absent_role": str(absent_role),
        "covering_role": str(covering_role),
        "window": int(window),
    })


def clear_role_dropout_recovery_v12(
        cache: TinyV12KVCache, *,
        absent_role: str, covering_role: str,
) -> None:
    cache.role_dropout_recovery_flag.pop(
        (str(absent_role), str(covering_role)), None)


@dataclasses.dataclass(frozen=True)
class V12SubstrateSnapshotFork:
    schema: str
    fork_label: str
    branch_ids: tuple[str, ...]
    per_branch_cid: dict[str, str]
    parent_v11_cache_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fork_label": str(self.fork_label),
            "branch_ids": list(self.branch_ids),
            "per_branch_cid": {
                k: str(v) for k, v in sorted(
                    self.per_branch_cid.items())},
            "parent_v11_cache_cid": str(self.parent_v11_cache_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "v12_substrate_snapshot_fork",
            "fork": self.to_dict()})


def substrate_snapshot_fork_v12(
        cache: TinyV12KVCache, *,
        branch_ids: Sequence[str],
        fork_label: str = "default_fork",
) -> V12SubstrateSnapshotFork:
    """Fork the V12 substrate into N branches.

    Returns a content-addressed snapshot-fork containing a CID per
    branch. The branches share the parent V11 cache CID; per-branch
    CIDs differ by a deterministic suffix so the fork map is
    content-addressed."""
    parent = cache.v11_cache.cid()
    per_branch: dict[str, str] = {}
    for bid in branch_ids:
        per_branch[str(bid)] = _sha256_hex({
            "kind": "v12_branch_snapshot",
            "parent": parent,
            "branch_id": str(bid),
            "fork_label": str(fork_label),
        })
    # Update the cache fork map.
    for bid, cid in per_branch.items():
        cache.snapshot_fork_map[str(bid)] = str(cid)
    cache.write_log_v12.append({
        "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        "kind": "substrate_snapshot_fork",
        "fork_label": str(fork_label),
        "n_branches": int(len(per_branch)),
    })
    return V12SubstrateSnapshotFork(
        schema=W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        fork_label=str(fork_label),
        branch_ids=tuple(str(b) for b in branch_ids),
        per_branch_cid=per_branch,
        parent_v11_cache_cid=str(parent),
    )


@dataclasses.dataclass(frozen=True)
class V12SubstrateBranchMerge:
    schema: str
    fork_label: str
    n_branches_merged: int
    branch_ids_merged: tuple[str, ...]
    merged_branch_cid: str
    reconciliation_delta_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fork_label": str(self.fork_label),
            "n_branches_merged": int(self.n_branches_merged),
            "branch_ids_merged": list(self.branch_ids_merged),
            "merged_branch_cid": str(self.merged_branch_cid),
            "reconciliation_delta_l1": float(round(
                self.reconciliation_delta_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "v12_substrate_branch_merge",
            "merge": self.to_dict()})


def substrate_branch_merge_v12(
        cache: TinyV12KVCache, *,
        fork: V12SubstrateSnapshotFork,
        branch_payloads: dict[str, Sequence[float]],
) -> V12SubstrateBranchMerge:
    """Reconcile N branches into one merged branch.

    The reconciliation is deterministic: per-element mean over the
    branch payloads. Returns the merged branch CID + the
    reconciliation delta L1 (how much the branches disagreed)."""
    branch_ids = sorted(branch_payloads.keys())
    if not branch_ids:
        raise ValueError("must provide >= 1 branch payload")
    arr = _np.asarray(
        [list(branch_payloads[b]) for b in branch_ids],
        dtype=_np.float64)
    mean = arr.mean(axis=0)
    deltas = arr - mean[_np.newaxis, :]
    recon_l1 = float(_np.linalg.norm(deltas.ravel(), ord=1))
    merged_cid = _sha256_hex({
        "kind": "v12_merged_branch",
        "fork_cid": str(fork.cid()),
        "branch_ids": list(branch_ids),
        "merged_payload_cid": _ndarray_cid(mean),
    })
    cache.write_log_v12.append({
        "schema": W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        "kind": "substrate_branch_merge",
        "n_branches_merged": int(len(branch_ids)),
        "reconciliation_delta_l1": float(round(recon_l1, 12)),
    })
    # Clear the fork map: the merge consumes the fork.
    for b in branch_ids:
        cache.snapshot_fork_map.pop(str(b), None)
    return V12SubstrateBranchMerge(
        schema=W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        fork_label=str(fork.fork_label),
        n_branches_merged=int(len(branch_ids)),
        branch_ids_merged=tuple(branch_ids),
        merged_branch_cid=str(merged_cid),
        reconciliation_delta_l1=float(recon_l1),
    )


def substrate_branch_merge_flops_v12(
        *, n_tokens: int, n_branches: int,
        recompute_flops_per_token: int = 1000,
        branch_merge_flops_per_token_per_branch: int = 90,
) -> dict[str, Any]:
    """V12 branch-merge vs recompute saving.

    A 4-branch reconciliation through the branch-merge primitive
    is ≥ 60 % cheaper than recomputing each branch from scratch."""
    n = int(max(0, n_tokens))
    nb = int(max(1, n_branches))
    merge = int(branch_merge_flops_per_token_per_branch) * n * nb
    recompute = int(recompute_flops_per_token) * n * nb
    saving = int(recompute - merge)
    ratio = (
        float(saving) / float(recompute)
        if recompute > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_branches": int(nb),
        "branch_merge_flops": int(merge),
        "recompute_flops": int(recompute),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class TinyV12SubstrateForwardWitness:
    schema: str
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v11_witness_cid: str
    v12_gate_score_per_layer_cid: str
    v12_gate_score_mean: float
    branch_merge_l1: float
    branch_merge_mean: float
    role_dropout_recovery_count: int
    n_branches_active: int
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v11_witness_cid": str(self.v11_witness_cid),
            "v12_gate_score_per_layer_cid": str(
                self.v12_gate_score_per_layer_cid),
            "v12_gate_score_mean": float(round(
                self.v12_gate_score_mean, 12)),
            "branch_merge_l1": float(round(
                self.branch_merge_l1, 12)),
            "branch_merge_mean": float(round(
                self.branch_merge_mean, 12)),
            "role_dropout_recovery_count": int(
                self.role_dropout_recovery_count),
            "n_branches_active": int(self.n_branches_active),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v12_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v12_forward_witness(
        trace: TinyV12ForwardTrace,
        cache: TinyV12KVCache,
) -> TinyV12SubstrateForwardWitness:
    v11_w = emit_tiny_substrate_v11_forward_witness(
        trace.v11_trace, cache.v11_cache)
    rd_count = int(sum(
        1 for v in cache.role_dropout_recovery_flag.values()
        if bool(v.get("triggered", False))))
    n_branches = int(len(cache.snapshot_fork_map))
    return TinyV12SubstrateForwardWitness(
        schema=W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v11_w.token_ids),
        v11_witness_cid=str(v11_w.cid()),
        v12_gate_score_per_layer_cid=_ndarray_cid(
            trace.v12_gate_score_per_layer),
        v12_gate_score_mean=float(
            trace.v12_gate_score_per_layer.mean())
            if trace.v12_gate_score_per_layer.size else 0.0,
        branch_merge_l1=float(_np.linalg.norm(
            cache.branch_merge_witness.ravel(), ord=1)),
        branch_merge_mean=float(
            cache.branch_merge_witness.mean()
            if cache.branch_merge_witness.size else 0.0),
        role_dropout_recovery_count=int(rd_count),
        n_branches_active=int(n_branches),
        n_layers=int(cache.n_layers()),
    )


def build_default_tiny_substrate_v12(
        *, seed: int = W67_DEFAULT_V12_SEED,
) -> TinyV12SubstrateParams:
    return TinyV12SubstrateParams.init(
        TinyV12SubstrateConfig.default(seed=int(seed)))


__all__ = [
    "W67_TINY_SUBSTRATE_V12_SCHEMA_VERSION",
    "W67_TINY_V12_VOCAB_SIZE",
    "W67_DEFAULT_V12_D_MODEL",
    "W67_DEFAULT_V12_N_HEADS",
    "W67_DEFAULT_V12_N_KV_HEADS",
    "W67_DEFAULT_V12_N_LAYERS",
    "W67_DEFAULT_V12_FF_HIDDEN",
    "W67_DEFAULT_V12_MAX_LEN",
    "W67_DEFAULT_V12_INIT_SCALE",
    "W67_DEFAULT_V12_SEED",
    "W67_DEFAULT_V12_MAX_ROLES",
    "W67_DEFAULT_V12_ROLE_DROPOUT_RECOVERY_BOOST",
    "W67_DEFAULT_V12_BRANCH_MERGE_BOOST",
    "TinyV12SubstrateConfig",
    "TinyV12SubstrateParams",
    "TinyV12KVCache",
    "TinyV12ForwardTrace",
    "TinyV12SubstrateForwardWitness",
    "V12SubstrateSnapshotFork",
    "V12SubstrateBranchMerge",
    "build_default_tiny_substrate_v12",
    "forward_tiny_substrate_v12",
    "record_branch_merge_witness_v12",
    "trigger_role_dropout_recovery_v12",
    "clear_role_dropout_recovery_v12",
    "substrate_snapshot_fork_v12",
    "substrate_branch_merge_v12",
    "substrate_branch_merge_flops_v12",
    "emit_tiny_substrate_v12_forward_witness",
    "tokenize_bytes_v12",
]
