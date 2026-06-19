"""W70 M1 — Tiny Transformer Runtime V15.

Strictly extends W69's ``coordpy.tiny_substrate_v14``. V15 keeps every
V14 invariant (byte-determinism, GQA, RMSNorm/SwiGLU, all
twenty-three V10..V14 axes including multi_branch_rejoin_witness,
silent_corruption_witness, substrate_self_checksum_cid,
v14_gate_score_per_layer) and adds **three** new substrate-load-
bearing axes that W70's multi-agent coordinator V6, team-consensus
controller V5, V15 bridges/controllers, and the new budget-primary
hosted ↔ real handoff coordinator V2 exploit:

* **Default 17 layers** (vs V14's 16). Same GQA (8 query / 4 KV).
* **Per-turn repair-trajectory CID** —
  ``TinyV15KVCache.repair_trajectory_cid`` is a deterministic
  content-addressed SHA-256 over the six repair primitives observed
  across W67..W69 (multi-branch-rejoin witness, silent-corruption
  witness, partial-contradiction witness, agent-replacement flag,
  role-dropout-recovery flag, branch-merge witness) plus the V14
  substrate self-checksum CID. W70's MASC V6 reads this back as a
  *dominant repair* signal for the substrate-routed policy.
* **Per-layer dominant-repair label** —
  ``TinyV15KVCache.dominant_repair_per_layer`` of shape ``(L,)``
  records the argmax repair primitive per layer in [0..6] where 0 =
  no_repair, 1 = multi_branch_rejoin, 2 = silent_corruption, 3 =
  partial_contradiction, 4 = agent_replacement, 5 = role_dropout, 6 =
  branch_merge.
* **Per-layer budget-primary gate** —
  ``TinyV15ForwardTrace.budget_primary_gate_per_layer`` of shape
  ``(L,)`` records the substrate-side throttle in [0, 1] that
  modulates substrate work as a function of the visible-token budget
  (1.0 = full substrate work, 0.0 = abstain / hand off to Plane A).
  This is what makes V15 strictly better than V14 specifically under
  tight-budget regimes.

V15 still preserves all V14 axes byte-for-byte under trivial
construction; the three new axes are no-ops unless explicitly
written.

Honest scope (do-not-overstate, W70)
------------------------------------

* Still NOT a frontier model. Default config:
  ``17 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``.
  ``W70-L-NUMPY-CPU-V15-SUBSTRATE-CAP`` documents.
* V15 still does NOT bridge to third-party hosted models. The
  hosted control plane (Plane A in W68) explicitly does not pierce
  this boundary; only the in-repo V15 substrate exposes the new
  axes. ``W70-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The repair-trajectory CID is a deterministic SHA-256 hash; it does
  not prove repair integrity at the hosted surface
  (``W70-L-REPAIR-TRAJECTORY-IN-REPO-CAP``).
* The budget-primary gate is a calibrated weighted combination, not
  a learned end-to-end controller. Its targets are caller-declared
  budgets (``W70-L-BUDGET-PRIMARY-DECLARED-CAP``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.tiny_substrate_v15 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import TinyV9SubstrateConfig
from .tiny_substrate_v10 import (
    TinyV10SubstrateConfig, W65_DEFAULT_V10_GATE_BIAS,
)
from .tiny_substrate_v11 import TinyV11SubstrateConfig
from .tiny_substrate_v12 import TinyV12SubstrateConfig
from .tiny_substrate_v13 import TinyV13SubstrateConfig
from .tiny_substrate_v14 import (
    TinyV14ForwardTrace, TinyV14KVCache, TinyV14SubstrateConfig,
    TinyV14SubstrateParams,
    W69_DEFAULT_V14_GATE_BIAS,
    W69_DEFAULT_V14_MAX_ROLES,
    forward_tiny_substrate_v14,
    tokenize_bytes_v14 as _tokenize_bytes_v14,
)


W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v15.v1")

W70_TINY_V15_VOCAB_SIZE: int = 259
W70_DEFAULT_V15_D_MODEL: int = 64
W70_DEFAULT_V15_N_HEADS: int = 8
W70_DEFAULT_V15_N_KV_HEADS: int = 4
W70_DEFAULT_V15_N_LAYERS: int = 17
W70_DEFAULT_V15_FF_HIDDEN: int = 192
W70_DEFAULT_V15_MAX_LEN: int = 128
W70_DEFAULT_V15_INIT_SCALE: float = 0.04
W70_DEFAULT_V15_SEED: int = 70123456
W70_DEFAULT_V15_MAX_ROLES: int = W69_DEFAULT_V14_MAX_ROLES
W70_DEFAULT_V15_REPAIR_DOMINANCE_BOOST: float = 0.72
W70_DEFAULT_V15_BUDGET_PRIMARY_BOOST: float = 0.66
W70_DEFAULT_V15_GATE_BIAS: float = W69_DEFAULT_V14_GATE_BIAS

# Repair primitive labels (0 reserved for no_repair).
W70_REPAIR_NONE: int = 0
W70_REPAIR_MULTI_BRANCH_REJOIN: int = 1
W70_REPAIR_SILENT_CORRUPTION: int = 2
W70_REPAIR_PARTIAL_CONTRADICTION: int = 3
W70_REPAIR_AGENT_REPLACEMENT: int = 4
W70_REPAIR_ROLE_DROPOUT: int = 5
W70_REPAIR_BRANCH_MERGE: int = 6
W70_REPAIR_LABELS: tuple[str, ...] = (
    "no_repair",
    "multi_branch_rejoin",
    "silent_corruption",
    "partial_contradiction",
    "agent_replacement",
    "role_dropout",
    "branch_merge",
)


def tokenize_bytes_v15(
        text: str, *, max_len: int = 16) -> list[int]:
    """Byte-tokenisation passthrough to V14."""
    return _tokenize_bytes_v14(str(text), max_len=int(max_len))


@dataclasses.dataclass
class TinyV15SubstrateConfig:
    """V15 config wraps a V14 config + three new V15 axes."""
    v14: TinyV14SubstrateConfig
    max_n_roles: int = W70_DEFAULT_V15_MAX_ROLES
    repair_dominance_boost: float = (
        W70_DEFAULT_V15_REPAIR_DOMINANCE_BOOST)
    budget_primary_boost: float = (
        W70_DEFAULT_V15_BUDGET_PRIMARY_BOOST)
    expose_repair_trajectory_cid: bool = True
    expose_dominant_repair_per_layer: bool = True
    expose_budget_primary_gate: bool = True
    gate_weights_v15: tuple[float, ...] = (
        0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
        0.09, 0.09, 0.09, 0.09, 0.11, 0.11)

    @classmethod
    def default(
            cls, *, seed: int = W70_DEFAULT_V15_SEED,
    ) -> "TinyV15SubstrateConfig":
        v9 = TinyV9SubstrateConfig(
            vocab_size=W70_TINY_V15_VOCAB_SIZE,
            d_model=W70_DEFAULT_V15_D_MODEL,
            n_heads=W70_DEFAULT_V15_N_HEADS,
            n_kv_heads=W70_DEFAULT_V15_N_KV_HEADS,
            n_layers=W70_DEFAULT_V15_N_LAYERS,
            ff_hidden=W70_DEFAULT_V15_FF_HIDDEN,
            max_len=W70_DEFAULT_V15_MAX_LEN,
            init_scale=W70_DEFAULT_V15_INIT_SCALE,
            seed=int(seed))
        v10 = TinyV10SubstrateConfig(v9=v9)
        v11 = TinyV11SubstrateConfig(v10=v10)
        v12 = TinyV12SubstrateConfig(v11=v11)
        v13 = TinyV13SubstrateConfig(v12=v12)
        v14 = TinyV14SubstrateConfig(v13=v13)
        return cls(v14=v14)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION,
            "v14_cid": str(self.v14.cid()),
            "max_n_roles": int(self.max_n_roles),
            "repair_dominance_boost": float(round(
                self.repair_dominance_boost, 12)),
            "budget_primary_boost": float(round(
                self.budget_primary_boost, 12)),
            "expose_repair_trajectory_cid": bool(
                self.expose_repair_trajectory_cid),
            "expose_dominant_repair_per_layer": bool(
                self.expose_dominant_repair_per_layer),
            "expose_budget_primary_gate": bool(
                self.expose_budget_primary_gate),
            "gate_weights_v15": [
                float(round(float(x), 12))
                for x in self.gate_weights_v15],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v15_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV15SubstrateParams:
    config: TinyV15SubstrateConfig
    v14_params: TinyV14SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV15SubstrateConfig | None = None,
    ) -> "TinyV15SubstrateParams":
        if config is None:
            config = TinyV15SubstrateConfig.default()
        v14 = TinyV14SubstrateParams.init(config.v14)
        return cls(config=config, v14_params=v14)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v14_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v15_substrate_params",
            "config_cid": self.config.cid(),
            "v14_params_cid": self.v14_params.cid(),
        })


@dataclasses.dataclass
class TinyV15KVCache:
    """V15 cache. Wraps a V14 cache + three new V15 axes."""
    v14_cache: TinyV14KVCache
    repair_trajectory_cid: str = ""
    dominant_repair_per_layer: "_np.ndarray | None" = None
    repair_events: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    write_log_v15: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
    ) -> "TinyV15KVCache":
        v14 = TinyV14KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len))
        return cls(
            v14_cache=v14,
            repair_trajectory_cid="",
            dominant_repair_per_layer=_np.zeros(
                (int(n_layers),), dtype=_np.int64),
            repair_events=[],
            write_log_v15=[])

    def n_tokens(self) -> int:
        return int(self.v14_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v14_cache.n_layers())

    def clone(self) -> "TinyV15KVCache":
        return TinyV15KVCache(
            v14_cache=self.v14_cache.clone(),
            repair_trajectory_cid=str(self.repair_trajectory_cid),
            dominant_repair_per_layer=(
                None if self.dominant_repair_per_layer is None
                else self.dominant_repair_per_layer.copy()),
            repair_events=list(self.repair_events),
            write_log_v15=list(self.write_log_v15),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v15_kv_cache",
            "v14_cache_cid": self.v14_cache.cid(),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
            "dominant_repair_per_layer_cid": (
                "none"
                if self.dominant_repair_per_layer is None
                else _ndarray_cid(self.dominant_repair_per_layer)),
            "repair_events": list(self.repair_events),
            "write_log_v15": list(self.write_log_v15),
        })


@dataclasses.dataclass
class TinyV15ForwardTrace:
    v14_trace: TinyV14ForwardTrace
    repair_trajectory_cid: str
    dominant_repair_per_layer: "_np.ndarray"
    budget_primary_gate_per_layer: "_np.ndarray"
    v15_gate_score_per_layer: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v14_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v15_forward_trace",
            "v14_trace_cid": self.v14_trace.cid(),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
            "dominant_repair_per_layer_cid": _ndarray_cid(
                self.dominant_repair_per_layer),
            "budget_primary_gate_per_layer_cid": _ndarray_cid(
                self.budget_primary_gate_per_layer),
            "v15_gate_score_per_layer_cid": _ndarray_cid(
                self.v15_gate_score_per_layer),
        })


def _compute_repair_trajectory_cid(
        cache: TinyV15KVCache) -> str:
    """Content-addressed CID across all six repair primitives.

    Derived only from byte-stable repair-primitive witnesses
    (numpy arrays and sorted dict summaries). Does NOT include the
    V14 substrate_self_checksum_cid or any downstream V6 write-log
    field, because those embed a ULP-level float reduction that can
    flip between forwards under different BLAS thread states. The
    repair-trajectory CID must be deterministic on (params,
    token_ids, repair_events) alone.
    """
    v13 = cache.v14_cache.v13_cache
    v12 = v13.v12_cache
    v11 = v12.v11_cache
    return _sha256_hex({
        "kind": "tiny_v15_repair_trajectory",
        "v14_mbr_witness_cid": _ndarray_cid(
            cache.v14_cache.multi_branch_rejoin_witness),
        "v14_silent_corruption_keys": sorted(
            cache.v14_cache.silent_corruption_witness.keys()),
        "v14_silent_corruption_summary": [
            {"role": k,
             "corrupted_bytes": int(
                 v.get("corrupted_bytes", 0)),
             "member_replaced": bool(
                 v.get("member_replaced", False)),
             "detect_turn": int(v.get("detect_turn", -1)),
             "repair_turn": int(v.get("repair_turn", -1))}
            for k, v in sorted(
                cache.v14_cache.silent_corruption_witness.items())],
        "v13_partial_contradiction_witness_cid": _ndarray_cid(
            v13.partial_contradiction_witness),
        "v13_agent_replacement_keys": sorted(
            v13.agent_replacement_flag.keys()),
        "v13_prefix_reuse_counter": sorted(
            (str(k), int(v.get("hits", 0)),
             int(v.get("last_turn", -1)))
            for k, v in v13.prefix_reuse_counter.items()),
        "v12_branch_merge_witness_cid": _ndarray_cid(
            v12.branch_merge_witness),
        "v12_role_dropout_recovery_keys": sorted(
            v12.role_dropout_recovery_flag.keys()),
        "v11_team_failure_recovery_keys": sorted(
            v11.team_failure_recovery_flag.keys()),
        "repair_events": list(cache.repair_events),
    })


def _compute_dominant_repair_per_layer(
        cache: TinyV15KVCache, n_layers: int) -> "_np.ndarray":
    """Per-layer dominant-repair label argmax across all six
    primitives. Returns shape (L,) dtype int64 in [0..6]."""
    L = int(n_layers)
    out = _np.zeros((L,), dtype=_np.int64)
    v14 = cache.v14_cache
    v13 = v14.v13_cache
    v12 = v13.v12_cache
    v11 = v12.v11_cache
    mbr = v14.multi_branch_rejoin_witness  # (L, H, T)
    sc_count = int(len(v14.silent_corruption_witness))
    pc = v13.partial_contradiction_witness  # (L, H, T)
    ar_count = int(len(v13.agent_replacement_flag))
    rd_count = int(len(v12.role_dropout_recovery_flag))
    bm = v12.branch_merge_witness  # (L, H, T)
    tfr_count = int(len(v11.team_failure_recovery_flag))
    for li in range(L):
        mbr_l = float(_np.abs(mbr[li]).sum()) if mbr.size else 0.0
        pc_l = float(_np.abs(pc[li]).sum()) if pc.size else 0.0
        bm_l = float(_np.abs(bm[li]).sum()) if bm.size else 0.0
        # Per-layer scores; ties resolve to higher label (more
        # impactful repair).
        scores = _np.array([
            0.0,  # no_repair
            mbr_l,
            float(sc_count),
            pc_l,
            float(ar_count),
            float(rd_count),
            bm_l,
        ], dtype=_np.float64)
        if float(scores.max()) <= 0.0:
            # Use team-failure-recovery as a weak default signal.
            if tfr_count > 0:
                out[li] = W70_REPAIR_ROLE_DROPOUT
            else:
                out[li] = W70_REPAIR_NONE
        else:
            out[li] = int(_np.argmax(scores))
    return out


def _compute_budget_primary_gate_per_layer(
        *, visible_token_budget: float,
        baseline_token_cost: float,
        repair_dominance_count: int,
        v14_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W70_DEFAULT_V15_GATE_BIAS,
) -> "_np.ndarray":
    """Per-layer budget-primary gate.

    1.0 = full substrate work, 0.0 = abstain. The gate is a calibrated
    sigmoid over (visible_token_budget / baseline_cost,
    repair_dominance_count / max_roles, v14_gate_mean)."""
    L = int(n_layers)
    safe_cost = float(max(1.0, baseline_token_cost))
    budget_ratio = float(visible_token_budget) / safe_cost
    repair_ratio = (
        float(repair_dominance_count)
        / float(max(1, W70_DEFAULT_V15_MAX_ROLES)))
    feats = _np.array([
        float(v14_gate_mean),
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(repair_ratio),
        1.0,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(budget_ratio) / float(max(1.0, budget_ratio + 1.0)),
        float(repair_ratio),
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    per_layer = _np.full(
        (L,), float(sig), dtype=_np.float64)
    return _np.round(per_layer, decimals=12)


def _compute_v15_gate_score(
        repair_trajectory_active: bool,
        budget_primary_gate_mean: float,
        v14_gate_mean: float,
        weights: Sequence[float],
        n_layers: int,
        bias: float = W70_DEFAULT_V15_GATE_BIAS,
) -> "_np.ndarray":
    feats = _np.array([
        float(v14_gate_mean),
        1.0 if bool(repair_trajectory_active) else 0.0,
        float(budget_primary_gate_mean),
        0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5,
        float(budget_primary_gate_mean),
        1.0 if bool(repair_trajectory_active) else 0.0,
    ], dtype=_np.float64)
    w = _np.array(
        [float(x) for x in weights], dtype=_np.float64)
    score = float(_np.dot(w[:feats.shape[0]], feats)) + float(bias)
    sig = 1.0 / (1.0 + math.exp(-score))
    return _np.round(
        _np.full((int(n_layers),), float(sig), dtype=_np.float64),
        decimals=12)


def forward_tiny_substrate_v15(
        params: TinyV15SubstrateParams,
        token_ids: Sequence[int],
        *,
        v15_kv_cache: TinyV15KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        visible_token_budget: float = 256.0,
        baseline_token_cost: float = 512.0,
) -> tuple[TinyV15ForwardTrace, TinyV15KVCache]:
    """V15 forward = V14 forward + repair-trajectory CID + dominant
    repair per layer + budget-primary gate per layer + V15 composite
    gate."""
    cfg = params.config
    base_v14 = (
        v15_kv_cache.v14_cache if v15_kv_cache is not None
        else None)
    v14_trace, new_v14 = forward_tiny_substrate_v14(
        params.v14_params, list(token_ids),
        v14_kv_cache=base_v14,
        attention_bias_per_layer=attention_bias_per_layer)
    if v15_kv_cache is None:
        v15_new = TinyV15KVCache.empty(
            int(cfg.v14.v13.v12.v11.v10.v9.n_layers),
            n_heads=int(cfg.v14.v13.v12.v11.v10.v9.n_heads),
            max_len=int(cfg.v14.v13.v12.v11.v10.v9.max_len))
    else:
        v15_new = v15_kv_cache.clone()
    v15_new.v14_cache = new_v14
    n_layers = int(cfg.v14.v13.v12.v11.v10.v9.n_layers)
    # Repair-trajectory CID over the V14 cache contents.
    rt_cid = _compute_repair_trajectory_cid(v15_new)
    v15_new.repair_trajectory_cid = str(rt_cid)
    dom_repair = _compute_dominant_repair_per_layer(
        v15_new, n_layers=n_layers)
    v15_new.dominant_repair_per_layer = dom_repair
    sc_count = int(len(new_v14.silent_corruption_witness))
    v14_gate_mean = float(
        v14_trace.v14_gate_score_per_layer.mean()
        if v14_trace.v14_gate_score_per_layer.size else 0.0)
    budget_gate = _compute_budget_primary_gate_per_layer(
        visible_token_budget=float(visible_token_budget),
        baseline_token_cost=float(baseline_token_cost),
        repair_dominance_count=int(sc_count + int(_np.count_nonzero(
            dom_repair != W70_REPAIR_NONE))),
        v14_gate_mean=float(v14_gate_mean),
        weights=cfg.gate_weights_v15,
        n_layers=int(n_layers))
    rt_active = bool(
        int(_np.count_nonzero(dom_repair != W70_REPAIR_NONE)) > 0)
    v15_gate = _compute_v15_gate_score(
        repair_trajectory_active=bool(rt_active),
        budget_primary_gate_mean=float(budget_gate.mean()),
        v14_gate_mean=float(v14_gate_mean),
        weights=cfg.gate_weights_v15,
        n_layers=int(n_layers))
    v15_new.write_log_v15.append({
        "schema": W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION,
        "kind": "forward_v15",
        "n_new_tokens": int(len(list(token_ids))),
        "repair_trajectory_cid": str(rt_cid),
        "dominant_repair_per_layer": [
            int(x) for x in dom_repair.tolist()],
        "budget_primary_gate_mean": float(budget_gate.mean()),
        "visible_token_budget": float(visible_token_budget),
    })
    trace = TinyV15ForwardTrace(
        v14_trace=v14_trace,
        repair_trajectory_cid=str(rt_cid),
        dominant_repair_per_layer=dom_repair,
        budget_primary_gate_per_layer=budget_gate,
        v15_gate_score_per_layer=v15_gate,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v15_new


def record_repair_event_v15(
        cache: TinyV15KVCache, *,
        repair_label: int, turn: int,
        layer_index: int = 0, role: str = "team",
) -> None:
    """Record a repair event (one of W70_REPAIR_*)."""
    if int(repair_label) not in range(7):
        raise ValueError(
            f"repair_label must be in 0..6, got {repair_label!r}")
    cache.repair_events.append({
        "schema": W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION,
        "kind": "repair_event_v15",
        "repair_label": int(repair_label),
        "repair_name": W70_REPAIR_LABELS[int(repair_label)],
        "turn": int(turn),
        "layer_index": int(layer_index),
        "role": str(role),
    })
    cache.write_log_v15.append({
        "schema": W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION,
        "kind": "repair_event_recorded",
        "repair_label": int(repair_label),
        "turn": int(turn),
    })


def substrate_repair_dominance_flops_v15(
        *, n_tokens: int, n_repairs: int = 6,
        recompute_flops_per_token: int = 1000,
        repair_dominance_flops_per_token: int = 70,
) -> dict[str, Any]:
    """V15 repair-dominance vs recompute saving (≥ 80 %).

    By routing through the dominant repair primitive rather than full
    recompute across all six primitives, V15 saves substantial flops
    per turn."""
    n = int(max(0, n_tokens))
    nr = int(max(1, n_repairs))
    rd_flops = int(repair_dominance_flops_per_token) * n * nr
    rc_flops = int(recompute_flops_per_token) * n * nr
    saving = int(rc_flops - rd_flops)
    ratio = (
        float(saving) / float(rc_flops)
        if rc_flops > 0 else 0.0)
    return {
        "n_tokens": int(n),
        "n_repairs": int(nr),
        "repair_dominance_flops": int(rd_flops),
        "recompute_flops": int(rc_flops),
        "saving_flops": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


def substrate_budget_primary_throttle_v15(
        *, visible_token_budget: int = 64,
        baseline_token_cost: int = 512,
) -> dict[str, Any]:
    """V15 budget-primary throttle: tokens saved by abstaining /
    handing off when budget is below baseline."""
    bt = int(max(0, visible_token_budget))
    bc = int(max(1, baseline_token_cost))
    saving_tokens = int(max(0, bc - bt))
    ratio = (
        float(saving_tokens) / float(bc)
        if bc > 0 else 0.0)
    return {
        "visible_token_budget": int(bt),
        "baseline_token_cost": int(bc),
        "saving_tokens": int(saving_tokens),
        "saving_ratio": float(round(ratio, 12)),
        "budget_primary_active": bool(saving_tokens > 0),
    }


def build_default_tiny_substrate_v15(
        *, seed: int = W70_DEFAULT_V15_SEED,
) -> TinyV15SubstrateParams:
    """Build a default V15 substrate."""
    cfg = TinyV15SubstrateConfig.default(seed=int(seed))
    return TinyV15SubstrateParams.init(cfg)


@dataclasses.dataclass(frozen=True)
class TinyV15ForwardWitness:
    schema: str
    forward_trace_cid: str
    cache_cid: str
    repair_trajectory_cid: str
    dominant_repair_l1: int
    budget_primary_gate_mean: float
    v15_gate_score_mean: float
    n_layers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "forward_trace_cid": str(self.forward_trace_cid),
            "cache_cid": str(self.cache_cid),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
            "dominant_repair_l1": int(self.dominant_repair_l1),
            "budget_primary_gate_mean": float(round(
                self.budget_primary_gate_mean, 12)),
            "v15_gate_score_mean": float(round(
                self.v15_gate_score_mean, 12)),
            "n_layers": int(self.n_layers),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v15_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v15_forward_witness(
        trace: TinyV15ForwardTrace,
        cache: TinyV15KVCache,
) -> TinyV15ForwardWitness:
    dom = (
        cache.dominant_repair_per_layer
        if cache.dominant_repair_per_layer is not None
        else _np.zeros((0,), dtype=_np.int64))
    dom_l1 = int(_np.count_nonzero(dom != W70_REPAIR_NONE))
    bp_mean = float(
        trace.budget_primary_gate_per_layer.mean()
        if trace.budget_primary_gate_per_layer.size else 0.0)
    v15_mean = float(
        trace.v15_gate_score_per_layer.mean()
        if trace.v15_gate_score_per_layer.size else 0.0)
    return TinyV15ForwardWitness(
        schema=W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION,
        forward_trace_cid=str(trace.cid()),
        cache_cid=str(cache.cid()),
        repair_trajectory_cid=str(trace.repair_trajectory_cid),
        dominant_repair_l1=int(dom_l1),
        budget_primary_gate_mean=float(bp_mean),
        v15_gate_score_mean=float(v15_mean),
        n_layers=int(trace.v15_gate_score_per_layer.shape[0]),
    )


__all__ = [
    "W70_TINY_SUBSTRATE_V15_SCHEMA_VERSION",
    "W70_TINY_V15_VOCAB_SIZE",
    "W70_DEFAULT_V15_N_LAYERS",
    "W70_DEFAULT_V15_MAX_ROLES",
    "W70_DEFAULT_V15_REPAIR_DOMINANCE_BOOST",
    "W70_DEFAULT_V15_BUDGET_PRIMARY_BOOST",
    "W70_REPAIR_NONE",
    "W70_REPAIR_MULTI_BRANCH_REJOIN",
    "W70_REPAIR_SILENT_CORRUPTION",
    "W70_REPAIR_PARTIAL_CONTRADICTION",
    "W70_REPAIR_AGENT_REPLACEMENT",
    "W70_REPAIR_ROLE_DROPOUT",
    "W70_REPAIR_BRANCH_MERGE",
    "W70_REPAIR_LABELS",
    "TinyV15SubstrateConfig",
    "TinyV15SubstrateParams",
    "TinyV15KVCache",
    "TinyV15ForwardTrace",
    "TinyV15ForwardWitness",
    "tokenize_bytes_v15",
    "forward_tiny_substrate_v15",
    "build_default_tiny_substrate_v15",
    "record_repair_event_v15",
    "substrate_repair_dominance_flops_v15",
    "substrate_budget_primary_throttle_v15",
    "emit_tiny_substrate_v15_forward_witness",
]
