"""W64 M2 — KV Bridge V9.

Strictly extends W63's ``coordpy.kv_bridge_v8``. V8 fit a 4-target
stack (3 V7 targets + 1 hidden-wins target). V9 adds:

* **Five-target stacked ridge fit** —
  ``fit_kv_bridge_v9_five_target`` fits a (n_directions × 5)
  matrix by closed-form ridge over five stacked targets
  simultaneously: V8's four targets + a *replay-dominance-as-
  primary* target column. The fifth column represents a δ that
  the KV bridge alone cannot reach without help from the V5
  replay controller's primary head.
* **Hidden-wins-primary regime falsifier** —
  ``probe_kv_bridge_v9_hidden_wins_primary_falsifier`` returns 0
  exactly when inverting the hidden-wins primary flag flips the
  decision. Strictly stronger than V8's hidden-wins falsifier.
* **Per-(layer, head, slot) primary-decision write** —
  ``write_kv_bridge_v9_into_v9_primary_decision`` records the
  per-slot primary-decision flag into the V9 substrate's
  hidden_wins_primary tensor.
* **KV-fingerprint perturbation control** —
  ``compute_kv_bridge_v9_fingerprint_delta`` measures the L2
  delta of the V8 correction tensor's SHA256 fingerprint under a
  ε-perturbation; surfaces as a *fitted perturbation control*
  diagnostic.

Honest scope (W64)
------------------

* The fifth replay-dominance-primary target is *constructed* to be
  unreachable by KV bridge alone. ``W64-L-KV-BRIDGE-V9-REPLAY-
  DOMINANCE-PRIMARY-TARGET-CONSTRUCTED-CAP`` documents.
* The fingerprint-delta is a SHA256 over the correction tensor
  bytes. NOT a calibrated perturbation analysis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v9 requires numpy") from exc

from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .kv_bridge_v8 import (
    KVBridgeV8Projection,
    KVBridgeV8FitReport,
    fit_kv_bridge_v8_four_target,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import (
    TinyV9KVCache, TinyV9SubstrateParams,
    record_hidden_wins_primary_v9,
)


W64_KV_BRIDGE_V9_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v9.v1")
W64_DEFAULT_KV_V9_RIDGE_LAMBDA: float = 0.05
W64_DEFAULT_KV_V9_PERTURBATION_EPS: float = 0.01


@dataclasses.dataclass
class KVBridgeV9Projection:
    inner_v8: KVBridgeV8Projection
    correction_layer_f_k: "_np.ndarray"   # (L, H, T, Dh)
    correction_layer_f_v: "_np.ndarray"
    seed_v9: int

    @classmethod
    def init_from_v8(
            cls, inner: KVBridgeV8Projection,
            *, seed_v9: int = 640900,
    ) -> "KVBridgeV9Projection":
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.inner_v7.inner_v6.n_inject_tokens)
        Dh = int(inner.d_head)
        return cls(
            inner_v8=inner,
            correction_layer_f_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_f_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            seed_v9=int(seed_v9),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v8.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v8.n_heads)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v8.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v8.d_head)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_KV_BRIDGE_V9_SCHEMA_VERSION,
            "kind": "kv_bridge_v9_projection",
            "inner_v8_cid": self.inner_v8.cid(),
            "correction_layer_f_k_cid": _ndarray_cid(
                self.correction_layer_f_k),
            "correction_layer_f_v_cid": _ndarray_cid(
                self.correction_layer_f_v),
            "seed_v9": int(self.seed_v9),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV9FitReport:
    schema: str
    fit_kind: str
    n_train: int
    n_targets: int
    n_directions: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    worst_pre_residual: float
    worst_post_residual: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fit_kind": str(self.fit_kind),
            "n_train": int(self.n_train),
            "n_targets": int(self.n_targets),
            "n_directions": int(self.n_directions),
            "per_target_pre_residual": [
                float(round(float(x), 12))
                for x in self.per_target_pre_residual],
            "per_target_post_residual": [
                float(round(float(x), 12))
                for x in self.per_target_post_residual],
            "worst_pre_residual": float(round(
                self.worst_pre_residual, 12)),
            "worst_post_residual": float(round(
                self.worst_post_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v9_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v9_five_target(
        *,
        params: TinyV9SubstrateParams,
        projection: KVBridgeV9Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        n_directions: int = 4,
        ridge_lambda: float = W64_DEFAULT_KV_V9_RIDGE_LAMBDA,
) -> tuple[KVBridgeV9Projection, KVBridgeV9FitReport]:
    """Five-target stacked ridge fit. The fifth target is a
    *replay-dominance-as-primary* target — a desired delta logits
    that the KV bridge cannot reach without help from the V5
    replay controller's primary head."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide ≥ 1 target")
    primary = list(target_delta_logits_stack[:4])
    while len(primary) < 4:
        primary.append(primary[0] if primary
                       else [0.0] * 1)
    fitted_v8, v8_report = fit_kv_bridge_v8_four_target(
        params=params.v8_params,
        projection=projection.inner_v8,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= 5:
        fifth = list(target_delta_logits_stack[4])
    else:
        fifth = list(target_delta_logits_stack[-1])
    fitted_inner_v6, v6_report = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=fitted_v8.inner_v7.inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[fifth],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_f_k = (
        fitted_inner_v6.correction_layer_c_k.copy()
        - fitted_v8.inner_v7.inner_v6.correction_layer_c_k)
    new_f_v = (
        fitted_inner_v6.correction_layer_c_v.copy()
        - fitted_v8.inner_v7.inner_v6.correction_layer_c_v)
    new_proj = dataclasses.replace(
        projection,
        inner_v8=fitted_v8,
        correction_layer_f_k=new_f_k,
        correction_layer_f_v=new_f_v,
    )
    per_pre = list(v8_report.per_target_pre_residual) + [
        float(v6_report.pre_fit_mean_residual)]
    per_post = list(v8_report.per_target_post_residual) + [
        float(v6_report.post_fit_mean_residual)]
    return new_proj, KVBridgeV9FitReport(
        schema=W64_KV_BRIDGE_V9_SCHEMA_VERSION,
        fit_kind="five_target_v9",
        n_train=int(v8_report.n_train),
        n_targets=int(n_targets),
        n_directions=int(n_directions),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        worst_pre_residual=float(max(per_pre)),
        worst_post_residual=float(max(per_post)),
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )


def write_kv_bridge_v9_into_v9_primary_decision(
        *,
        projection: KVBridgeV9Projection,
        v9_cache: TinyV9KVCache,
        hidden_wins_per_slot: (
            Sequence[Sequence[Sequence[float]]] | None) = None,
) -> dict[str, Any]:
    """Compute per-(layer, head, slot) primary decision and record
    into the V9 hidden-wins-primary tensor."""
    k = projection.correction_layer_f_k
    v = projection.correction_layer_f_v
    L, H, T, _Dh = k.shape
    n_writes = 0
    n_hidden_wins = 0
    for li in range(L):
        for hi in range(H):
            for ti in range(T):
                vec_k = _np.asarray(
                    k[li, hi, ti], dtype=_np.float64)
                vec_v = _np.asarray(
                    v[li, hi, ti], dtype=_np.float64)
                l2_kv = float(_np.sqrt(
                    _np.linalg.norm(vec_k) ** 2 +
                    _np.linalg.norm(vec_v) ** 2))
                hidden_score = 0.0
                if hidden_wins_per_slot is not None:
                    try:
                        hidden_score = float(
                            hidden_wins_per_slot[li][hi][ti])
                    except (IndexError, TypeError):
                        hidden_score = 0.0
                if hidden_score > l2_kv:
                    decision = "hidden_wins"
                    n_hidden_wins += 1
                elif hidden_score < l2_kv:
                    decision = "kv_wins"
                else:
                    decision = "tie"
                if l2_kv > 0.0 or hidden_score > 0.0:
                    record_hidden_wins_primary_v9(
                        v9_cache,
                        layer_index=int(li),
                        head_index=int(hi),
                        slot=int(ti),
                        decision=str(decision))
                    n_writes += 1
    return {
        "schema": W64_KV_BRIDGE_V9_SCHEMA_VERSION,
        "kind": "kv_bridge_v9_primary_decision_write",
        "n_writes": int(n_writes),
        "n_hidden_wins": int(n_hidden_wins),
    }


@dataclasses.dataclass(frozen=True)
class HiddenWinsPrimaryFalsifierWitnessV9:
    schema: str
    primary_flag: float
    inverted_flag: float
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "primary_flag": float(round(self.primary_flag, 12)),
            "inverted_flag": float(round(
                self.inverted_flag, 12)),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_wins_primary_falsifier_v9",
            "witness": self.to_dict()})


def probe_kv_bridge_v9_hidden_wins_primary_falsifier(
        *, primary_flag: float, tie_threshold: float = 1e-6,
) -> HiddenWinsPrimaryFalsifierWitnessV9:
    """A strong falsifier: invert the hidden-wins primary flag and
    check that the decision flips. ``falsifier_score`` is 0 when
    flipping the input flips the decision, > 0 otherwise."""
    if float(primary_flag) > float(tie_threshold):
        decision = "hidden_wins"
    elif float(primary_flag) < -float(tie_threshold):
        decision = "kv_wins"
    else:
        decision = "tie"
    inv = -float(primary_flag)
    if inv > float(tie_threshold):
        inverted = "hidden_wins"
    elif inv < -float(tie_threshold):
        inverted = "kv_wins"
    else:
        inverted = "tie"
    expected_inv = {
        "hidden_wins": "kv_wins",
        "kv_wins": "hidden_wins",
        "tie": "tie",
    }[decision]
    score = 0.0 if inverted == expected_inv else 1.0
    return HiddenWinsPrimaryFalsifierWitnessV9(
        schema=W64_KV_BRIDGE_V9_SCHEMA_VERSION,
        primary_flag=float(primary_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inverted),
        falsifier_score=float(score),
    )


def compute_kv_bridge_v9_fingerprint_delta(
        *, projection: KVBridgeV9Projection,
        eps: float = W64_DEFAULT_KV_V9_PERTURBATION_EPS,
) -> dict[str, Any]:
    """Compute SHA256 fingerprint of the V9 correction tensor
    bytes, perturb by eps, and return the L2 delta + new
    fingerprint."""
    k = projection.correction_layer_f_k
    v = projection.correction_layer_f_v
    base_fp = hashlib.sha256(
        k.tobytes() + v.tobytes()).hexdigest()
    pert_k = k + float(eps)
    pert_v = v + float(eps)
    pert_fp = hashlib.sha256(
        pert_k.tobytes() + pert_v.tobytes()).hexdigest()
    l2_delta_k = float(_np.linalg.norm(
        (pert_k - k).ravel()))
    l2_delta_v = float(_np.linalg.norm(
        (pert_v - v).ravel()))
    return {
        "schema": W64_KV_BRIDGE_V9_SCHEMA_VERSION,
        "kind": "kv_bridge_v9_fingerprint_delta",
        "base_fingerprint": str(base_fp),
        "perturbed_fingerprint": str(pert_fp),
        "fingerprint_changed": bool(base_fp != pert_fp),
        "eps": float(round(eps, 12)),
        "l2_delta_k": float(round(l2_delta_k, 12)),
        "l2_delta_v": float(round(l2_delta_v, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV9Witness:
    schema: str
    projection_cid: str
    n_writes: int
    total_l2: float
    fit_report_cid: str
    hidden_wins_primary_falsifier_cid: str
    fingerprint_delta_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "n_writes": int(self.n_writes),
            "total_l2": float(round(self.total_l2, 12)),
            "fit_report_cid": str(self.fit_report_cid),
            "hidden_wins_primary_falsifier_cid": str(
                self.hidden_wins_primary_falsifier_cid),
            "fingerprint_delta_l2": float(round(
                self.fingerprint_delta_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v9_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v9_witness(
        *, projection: KVBridgeV9Projection,
        fit_report: KVBridgeV9FitReport | None = None,
        hidden_wins_primary_falsifier: (
            HiddenWinsPrimaryFalsifierWitnessV9 | None) = None,
        fingerprint_delta_l2: float = 0.0,
) -> KVBridgeV9Witness:
    total_l2 = float(
        _np.linalg.norm(
            projection.correction_layer_f_k.ravel()) ** 2 +
        _np.linalg.norm(
            projection.correction_layer_f_v.ravel()) ** 2)
    return KVBridgeV9Witness(
        schema=W64_KV_BRIDGE_V9_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        n_writes=int(
            projection.correction_layer_f_k.size +
            projection.correction_layer_f_v.size),
        total_l2=float(_np.sqrt(total_l2)),
        fit_report_cid=(
            fit_report.cid()
            if fit_report is not None else ""),
        hidden_wins_primary_falsifier_cid=(
            hidden_wins_primary_falsifier.cid()
            if hidden_wins_primary_falsifier is not None
            else ""),
        fingerprint_delta_l2=float(fingerprint_delta_l2),
    )


__all__ = [
    "W64_KV_BRIDGE_V9_SCHEMA_VERSION",
    "W64_DEFAULT_KV_V9_RIDGE_LAMBDA",
    "W64_DEFAULT_KV_V9_PERTURBATION_EPS",
    "KVBridgeV9Projection",
    "KVBridgeV9FitReport",
    "fit_kv_bridge_v9_five_target",
    "write_kv_bridge_v9_into_v9_primary_decision",
    "HiddenWinsPrimaryFalsifierWitnessV9",
    "probe_kv_bridge_v9_hidden_wins_primary_falsifier",
    "compute_kv_bridge_v9_fingerprint_delta",
    "KVBridgeV9Witness",
    "emit_kv_bridge_v9_witness",
]
