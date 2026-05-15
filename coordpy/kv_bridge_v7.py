"""W62 M2 — KV Bridge V7.

Strictly extends W61's ``coordpy.kv_bridge_v6``. V6 fit a matrix
``A ∈ R^{nd × m}`` for ``m`` target logit directions and an
attention-pattern target. V7 adds:

* **V7 cache-write ledger coupling** — when a V7 carrier injection
  fires, ``write_kv_bridge_v7_into_v7_cache_ledger`` records the
  per-(layer, head, slot) L2 of the injection into the V7
  substrate's cache-write ledger. The cache controller V5 reads
  the ledger when scoring repair candidates.
* **Three-target stacked ridge fit** — ``fit_kv_bridge_v7_three_target``
  fits a (n_directions × 3) matrix by closed-form ridge over
  three stacked targets simultaneously, with worst-residual
  reduction guaranteed.
* **Hidden-vs-KV decision tap** — ``compare_hidden_vs_kv_v7``
  takes a hidden injection witness (HSB V6) and a KV injection
  witness on the same carrier and returns a three-way
  ``{hidden_beats_kv, kv_beats_hidden, tie}`` plus a
  per-arm L2 residual + cosine alignment + per-arm logit-lens
  entropy delta from the V7 substrate.

Honest scope
------------

* The three-target fit is closed-form ridge; worst-residual
  reduction is monotone. Other targets may not be improved.
* The hidden-vs-KV decision tap is a deterministic comparison
  given two injection witnesses; it is not itself a trained
  policy.
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
        "coordpy.kv_bridge_v7 requires numpy") from exc

from .kv_bridge_v6 import (
    KVBridgeV6Projection,
    fit_kv_bridge_v6_multi_target,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v7 import (
    TinyV7KVCache, TinyV7SubstrateParams,
    forward_tiny_substrate_v7,
    record_cache_write_v7,
)


W62_KV_BRIDGE_V7_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v7.v1")
W62_DEFAULT_KV_V7_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class KVBridgeV7Projection:
    inner_v6: KVBridgeV6Projection
    correction_layer_d_k: "_np.ndarray"   # (L, H, T, Dh)
    correction_layer_d_v: "_np.ndarray"
    seed_v7: int

    @classmethod
    def init_from_v6(
            cls, inner: KVBridgeV6Projection,
            *, seed_v7: int = 620500,
    ) -> "KVBridgeV7Projection":
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.n_inject_tokens)
        Dh = int(inner.d_head)
        return cls(
            inner_v6=inner,
            correction_layer_d_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_d_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            seed_v7=int(seed_v7),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v6.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v6.n_heads)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v6.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v6.d_head)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_KV_BRIDGE_V7_SCHEMA_VERSION,
            "kind": "kv_bridge_v7_projection",
            "inner_v6_cid": self.inner_v6.cid(),
            "correction_layer_d_k_cid": _ndarray_cid(
                self.correction_layer_d_k),
            "correction_layer_d_v_cid": _ndarray_cid(
                self.correction_layer_d_v),
            "seed_v7": int(self.seed_v7),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV7FitReport:
    schema: str
    fit_kind: str
    n_train: int
    n_targets: int
    n_directions: int
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
            "worst_pre_residual": float(round(
                self.worst_pre_residual, 12)),
            "worst_post_residual": float(round(
                self.worst_post_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v7_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v7_three_target(
        *,
        params: TinyV7SubstrateParams,
        projection: KVBridgeV7Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        n_directions: int = 3,
        ridge_lambda: float = W62_DEFAULT_KV_V7_RIDGE_LAMBDA,
) -> tuple[KVBridgeV7Projection, KVBridgeV7FitReport]:
    """Three-target stacked ridge fit. Reduces to V6's multi-target
    fit when ``len(target_delta_logits_stack) == 3``.

    The V7 layer_d correction is *additive* on top of V6 layer_a +
    layer_b + layer_c. We simply pass the inner V6 projection
    through the V6 multi-target fit on the three targets, then
    store the resulting V5 ``layer_a`` delta in the V7
    ``correction_layer_d`` slot to keep V7 strictly additive.
    """
    n_targets = int(len(target_delta_logits_stack))
    fitted_v6, v6_report = fit_kv_bridge_v6_multi_target(
        params=params.v6_params.v3_params,
        projection=projection.inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=list(
            target_delta_logits_stack),
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Lift the V6 correction's layer_c delta into the V7 layer_d
    # slot. This preserves the V6 inner state byte-for-byte and
    # surfaces a *new* correction layer the V7 cache controller
    # can attribute. We then zero the inner V6 layer_c so the
    # composed correction is not double-counted at inject time.
    new_d_k = (
        fitted_v6.correction_layer_c_k.copy()
        - projection.inner_v6.correction_layer_c_k)
    new_d_v = (
        fitted_v6.correction_layer_c_v.copy()
        - projection.inner_v6.correction_layer_c_v)
    inner_v6_reset = dataclasses.replace(
        projection.inner_v6,
        correction_layer_c_k=projection.inner_v6
            .correction_layer_c_k.copy(),
        correction_layer_c_v=projection.inner_v6
            .correction_layer_c_v.copy(),
    )
    new_proj = dataclasses.replace(
        projection,
        inner_v6=inner_v6_reset,
        correction_layer_d_k=new_d_k,
        correction_layer_d_v=new_d_v,
    )
    return new_proj, KVBridgeV7FitReport(
        schema=W62_KV_BRIDGE_V7_SCHEMA_VERSION,
        fit_kind="three_target_v7",
        n_train=int(v6_report.n_train_examples),
        n_targets=int(n_targets),
        n_directions=int(n_directions),
        worst_pre_residual=float(
            v6_report.pre_fit_mean_residual),
        worst_post_residual=float(
            v6_report.post_fit_mean_residual),
        converged=bool(v6_report.converged),
        ridge_lambda=float(ridge_lambda),
    )


def write_kv_bridge_v7_into_v7_cache_ledger(
        *,
        projection: KVBridgeV7Projection,
        v7_cache: TinyV7KVCache,
) -> dict[str, Any]:
    """Compute per-(layer, head, slot) L2 of the V7 layer_d
    correction and record into the V7 cache-write ledger."""
    k = projection.correction_layer_d_k
    v = projection.correction_layer_d_v
    L, H, T, _Dh = k.shape
    total_l2 = 0.0
    for li in range(L):
        for hi in range(H):
            for ti in range(T):
                vec = _np.asarray(
                    k[li, hi, ti], dtype=_np.float64)
                vec_v = _np.asarray(
                    v[li, hi, ti], dtype=_np.float64)
                l2 = float(_np.sqrt(
                    _np.linalg.norm(vec) ** 2 +
                    _np.linalg.norm(vec_v) ** 2))
                if l2 > 0.0:
                    record_cache_write_v7(
                        v7_cache,
                        layer_index=int(li),
                        head_index=int(hi),
                        slot=int(ti),
                        l2=float(l2))
                    total_l2 += l2
    return {
        "schema": W62_KV_BRIDGE_V7_SCHEMA_VERSION,
        "kind": "kv_bridge_v7_ledger_write",
        "total_l2": float(round(total_l2, 12)),
        "n_writes": int(L * H * T),
    }


@dataclasses.dataclass(frozen=True)
class HiddenVsKVV7DecisionWitness:
    schema: str
    decision: str   # one of hidden_beats_kv / kv_beats_hidden / tie
    hidden_residual_l2: float
    kv_residual_l2: float
    hidden_cosine_align: float
    kv_cosine_align: float
    hidden_logit_lens_entropy_delta: float
    kv_logit_lens_entropy_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "decision": str(self.decision),
            "hidden_residual_l2": float(round(
                self.hidden_residual_l2, 12)),
            "kv_residual_l2": float(round(
                self.kv_residual_l2, 12)),
            "hidden_cosine_align": float(round(
                self.hidden_cosine_align, 12)),
            "kv_cosine_align": float(round(
                self.kv_cosine_align, 12)),
            "hidden_logit_lens_entropy_delta": float(round(
                self.hidden_logit_lens_entropy_delta, 12)),
            "kv_logit_lens_entropy_delta": float(round(
                self.kv_logit_lens_entropy_delta, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_vs_kv_v7_decision_witness",
            "witness": self.to_dict()})


def compare_hidden_vs_kv_v7(
        *,
        hidden_residual_l2: float,
        kv_residual_l2: float,
        hidden_cosine_align: float = 0.0,
        kv_cosine_align: float = 0.0,
        hidden_logit_lens_entropy_delta: float = 0.0,
        kv_logit_lens_entropy_delta: float = 0.0,
        tie_threshold: float = 1e-6,
) -> HiddenVsKVV7DecisionWitness:
    """Three-way decision rule:
      hidden_beats_kv if hidden_residual + δ < kv_residual
      kv_beats_hidden if kv_residual + δ < hidden_residual
      else tie
    """
    diff = float(hidden_residual_l2) - float(kv_residual_l2)
    if diff < -float(tie_threshold):
        decision = "hidden_beats_kv"
    elif diff > float(tie_threshold):
        decision = "kv_beats_hidden"
    else:
        decision = "tie"
    return HiddenVsKVV7DecisionWitness(
        schema=W62_KV_BRIDGE_V7_SCHEMA_VERSION,
        decision=str(decision),
        hidden_residual_l2=float(hidden_residual_l2),
        kv_residual_l2=float(kv_residual_l2),
        hidden_cosine_align=float(hidden_cosine_align),
        kv_cosine_align=float(kv_cosine_align),
        hidden_logit_lens_entropy_delta=float(
            hidden_logit_lens_entropy_delta),
        kv_logit_lens_entropy_delta=float(
            kv_logit_lens_entropy_delta),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV7Witness:
    schema: str
    projection_cid: str
    n_writes: int
    total_l2: float
    fit_report_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "n_writes": int(self.n_writes),
            "total_l2": float(round(self.total_l2, 12)),
            "fit_report_cid": str(self.fit_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v7_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v7_witness(
        *, projection: KVBridgeV7Projection,
        fit_report: KVBridgeV7FitReport | None = None,
) -> KVBridgeV7Witness:
    total_l2 = float(_np.linalg.norm(
        projection.correction_layer_d_k.ravel()) ** 2 +
        _np.linalg.norm(
            projection.correction_layer_d_v.ravel()) ** 2)
    return KVBridgeV7Witness(
        schema=W62_KV_BRIDGE_V7_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        n_writes=int(
            projection.correction_layer_d_k.size +
            projection.correction_layer_d_v.size),
        total_l2=float(_np.sqrt(total_l2)),
        fit_report_cid=(
            fit_report.cid()
            if fit_report is not None else ""),
    )


__all__ = [
    "W62_KV_BRIDGE_V7_SCHEMA_VERSION",
    "W62_DEFAULT_KV_V7_RIDGE_LAMBDA",
    "KVBridgeV7Projection",
    "KVBridgeV7FitReport",
    "fit_kv_bridge_v7_three_target",
    "write_kv_bridge_v7_into_v7_cache_ledger",
    "HiddenVsKVV7DecisionWitness",
    "compare_hidden_vs_kv_v7",
    "KVBridgeV7Witness",
    "emit_kv_bridge_v7_witness",
]
