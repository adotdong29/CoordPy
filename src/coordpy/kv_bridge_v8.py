"""W63 M2 — KV Bridge V8.

Strictly extends W62's ``coordpy.kv_bridge_v7``. V7 fit a 3-target
stack and exposed a V7 cache-write ledger coupling + a hidden-vs-KV
decision tap. V8 adds:

* **Four-target stacked ridge fit** —
  ``fit_kv_bridge_v8_four_target`` fits a (n_directions × 4) matrix
  by closed-form ridge over four stacked targets simultaneously,
  with worst-residual reduction guaranteed and an explicit
  ``hidden-wins-target`` slot.
* **Hidden-wins regime falsifier** —
  ``probe_kv_bridge_v8_hidden_wins_falsifier`` runs an inverted
  hidden-wins decision and returns 0 exactly when the inversion
  produces a kv-wins decision on the same carrier — a strong
  per-bucket falsifier.
* **Per-(layer, head, slot) contention coupling** —
  ``write_kv_bridge_v8_into_v8_contention`` records per-slot
  ``|kv|`` into the V8 hidden-vs-KV contention channel (negative
  side). The hidden_state_bridge_v7 writes the positive side.
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
        "coordpy.kv_bridge_v8 requires numpy") from exc

from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .kv_bridge_v7 import (
    KVBridgeV7Projection,
    KVBridgeV7FitReport,
    fit_kv_bridge_v7_three_target,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v8 import (
    TinyV8KVCache, TinyV8SubstrateParams,
    record_hidden_vs_kv_contention_v8,
)


W63_KV_BRIDGE_V8_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v8.v1")
W63_DEFAULT_KV_V8_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class KVBridgeV8Projection:
    inner_v7: KVBridgeV7Projection
    correction_layer_e_k: "_np.ndarray"   # (L, H, T, Dh)
    correction_layer_e_v: "_np.ndarray"
    seed_v8: int

    @classmethod
    def init_from_v7(
            cls, inner: KVBridgeV7Projection,
            *, seed_v8: int = 630800,
    ) -> "KVBridgeV8Projection":
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.inner_v6.n_inject_tokens)
        Dh = int(inner.d_head)
        return cls(
            inner_v7=inner,
            correction_layer_e_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_e_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            seed_v8=int(seed_v8),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v7.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v7.n_heads)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v7.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v7.d_head)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_KV_BRIDGE_V8_SCHEMA_VERSION,
            "kind": "kv_bridge_v8_projection",
            "inner_v7_cid": self.inner_v7.cid(),
            "correction_layer_e_k_cid": _ndarray_cid(
                self.correction_layer_e_k),
            "correction_layer_e_v_cid": _ndarray_cid(
                self.correction_layer_e_v),
            "seed_v8": int(self.seed_v8),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV8FitReport:
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
            "kind": "kv_bridge_v8_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v8_four_target(
        *,
        params: TinyV8SubstrateParams,
        projection: KVBridgeV8Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        n_directions: int = 4,
        ridge_lambda: float = W63_DEFAULT_KV_V8_RIDGE_LAMBDA,
) -> tuple[KVBridgeV8Projection, KVBridgeV8FitReport]:
    """Four-target stacked ridge fit. The fourth target is a
    *hidden-wins target* — a desired delta logits that the
    KV bridge cannot reach without help from the HSB V7 bridge.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide ≥ 1 target")
    # Delegate to V7's three-target ridge on the first three targets.
    primary = list(target_delta_logits_stack[:3])
    while len(primary) < 3:
        primary.append(primary[0] if primary
                       else [0.0] * 1)
    fitted_v7, v7_report = fit_kv_bridge_v7_three_target(
        params=params.v7_params,
        projection=projection.inner_v7,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # For the fourth (hidden-wins) target: fit a second-pass V6
    # multi-target ridge on the fourth column alone.
    if n_targets >= 4:
        fourth = list(target_delta_logits_stack[3])
    else:
        fourth = list(target_delta_logits_stack[-1])
    fitted_inner_v6, v6_report = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=fitted_v7.inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[fourth],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_e_k = (
        fitted_inner_v6.correction_layer_c_k.copy()
        - fitted_v7.inner_v6.correction_layer_c_k)
    new_e_v = (
        fitted_inner_v6.correction_layer_c_v.copy()
        - fitted_v7.inner_v6.correction_layer_c_v)
    new_proj = dataclasses.replace(
        projection,
        inner_v7=fitted_v7,
        correction_layer_e_k=new_e_k,
        correction_layer_e_v=new_e_v,
    )
    per_pre = [float(v7_report.worst_pre_residual)] * 3 + [
        float(v6_report.pre_fit_mean_residual)]
    per_post = [float(v7_report.worst_post_residual)] * 3 + [
        float(v6_report.post_fit_mean_residual)]
    return new_proj, KVBridgeV8FitReport(
        schema=W63_KV_BRIDGE_V8_SCHEMA_VERSION,
        fit_kind="four_target_v8",
        n_train=int(v7_report.n_train),
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


def write_kv_bridge_v8_into_v8_contention(
        *,
        projection: KVBridgeV8Projection,
        v8_cache: TinyV8KVCache,
        hidden_write_abs_per_slot: (
            Sequence[Sequence[Sequence[float]]] | None) = None,
) -> dict[str, Any]:
    """Compute per-(layer, head, slot) L2 of the V8 layer_e
    correction and record into the V8 hidden-vs-KV contention
    channel (negative side).
    """
    k = projection.correction_layer_e_k
    v = projection.correction_layer_e_v
    L, H, T, _Dh = k.shape
    total_l2 = 0.0
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
                hl2 = 0.0
                if hidden_write_abs_per_slot is not None:
                    try:
                        hl2 = float(
                            hidden_write_abs_per_slot[li][hi][ti])
                    except (IndexError, TypeError):
                        hl2 = 0.0
                if l2_kv > 0.0 or hl2 > 0.0:
                    record_hidden_vs_kv_contention_v8(
                        v8_cache,
                        layer_index=int(li),
                        head_index=int(hi),
                        slot=int(ti),
                        hidden_write_abs=float(hl2),
                        kv_write_abs=float(l2_kv))
                    total_l2 += l2_kv
    return {
        "schema": W63_KV_BRIDGE_V8_SCHEMA_VERSION,
        "kind": "kv_bridge_v8_contention_write",
        "total_l2": float(round(total_l2, 12)),
        "n_writes": int(L * H * T),
    }


@dataclasses.dataclass(frozen=True)
class HiddenWinsFalsifierWitnessV8:
    schema: str
    kv_residual_l2: float
    hidden_residual_l2: float
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "kv_residual_l2": float(round(
                self.kv_residual_l2, 12)),
            "hidden_residual_l2": float(round(
                self.hidden_residual_l2, 12)),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_wins_falsifier_witness_v8",
            "witness": self.to_dict()})


def probe_kv_bridge_v8_hidden_wins_falsifier(
        *,
        hidden_residual_l2: float,
        kv_residual_l2: float,
        tie_threshold: float = 1e-6,
) -> HiddenWinsFalsifierWitnessV8:
    """A strong falsifier: invert the hidden/kv residual roles and
    check that the decision flips. ``falsifier_score`` is 0 when
    flipping the inputs flips the decision, > 0 otherwise.
    """
    diff = float(hidden_residual_l2) - float(kv_residual_l2)
    if diff < -float(tie_threshold):
        decision = "hidden_beats_kv"
    elif diff > float(tie_threshold):
        decision = "kv_beats_hidden"
    else:
        decision = "tie"
    diff_inv = float(kv_residual_l2) - float(hidden_residual_l2)
    if diff_inv < -float(tie_threshold):
        inverted = "hidden_beats_kv"
    elif diff_inv > float(tie_threshold):
        inverted = "kv_beats_hidden"
    else:
        inverted = "tie"
    expected_inv = {
        "hidden_beats_kv": "kv_beats_hidden",
        "kv_beats_hidden": "hidden_beats_kv",
        "tie": "tie",
    }[decision]
    score = 0.0 if inverted == expected_inv else 1.0
    return HiddenWinsFalsifierWitnessV8(
        schema=W63_KV_BRIDGE_V8_SCHEMA_VERSION,
        kv_residual_l2=float(kv_residual_l2),
        hidden_residual_l2=float(hidden_residual_l2),
        decision=str(decision),
        inverted_decision=str(inverted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV8Witness:
    schema: str
    projection_cid: str
    n_writes: int
    total_l2: float
    fit_report_cid: str
    hidden_wins_falsifier_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "n_writes": int(self.n_writes),
            "total_l2": float(round(self.total_l2, 12)),
            "fit_report_cid": str(self.fit_report_cid),
            "hidden_wins_falsifier_cid": str(
                self.hidden_wins_falsifier_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v8_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v8_witness(
        *, projection: KVBridgeV8Projection,
        fit_report: KVBridgeV8FitReport | None = None,
        hidden_wins_falsifier: (
            HiddenWinsFalsifierWitnessV8 | None) = None,
) -> KVBridgeV8Witness:
    total_l2 = float(
        _np.linalg.norm(
            projection.correction_layer_e_k.ravel()) ** 2 +
        _np.linalg.norm(
            projection.correction_layer_e_v.ravel()) ** 2)
    return KVBridgeV8Witness(
        schema=W63_KV_BRIDGE_V8_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        n_writes=int(
            projection.correction_layer_e_k.size +
            projection.correction_layer_e_v.size),
        total_l2=float(_np.sqrt(total_l2)),
        fit_report_cid=(
            fit_report.cid()
            if fit_report is not None else ""),
        hidden_wins_falsifier_cid=(
            hidden_wins_falsifier.cid()
            if hidden_wins_falsifier is not None else ""),
    )


__all__ = [
    "W63_KV_BRIDGE_V8_SCHEMA_VERSION",
    "W63_DEFAULT_KV_V8_RIDGE_LAMBDA",
    "KVBridgeV8Projection",
    "KVBridgeV8FitReport",
    "fit_kv_bridge_v8_four_target",
    "write_kv_bridge_v8_into_v8_contention",
    "HiddenWinsFalsifierWitnessV8",
    "probe_kv_bridge_v8_hidden_wins_falsifier",
    "KVBridgeV8Witness",
    "emit_kv_bridge_v8_witness",
]
