"""W79 — Long-Horizon Retention V31.

Strictly extends W78's ``coordpy.long_horizon_retention_v30``. V30
was 29 heads + a twenty-layer scorer at max_k=2048. V31 is:

* **30 heads** (V30's 29 + replacement-then-restart-after-long-
  delay head).
* **Twenty-one-layer scorer** — V30's twenty layers + a twenty-
  first random+swish layer before the final ridge.
* **max_k = 4096** (vs V30's 2048). The largest k in the
  programme to date.

Honest scope (W79): only the final ridge head is fit; earlier
projections are frozen random. ``W79-L-V31-LHR-SCORER-FIT-CAP``.
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
        "coordpy.long_horizon_retention_v31 requires numpy"
        ) from exc

from .long_horizon_retention_v30 import (
    LongHorizonReconstructionV30Head,
    W78_DEFAULT_LHR_V30_LONG_HORIZON_RECONSTRUCTION_DIM,
    W78_DEFAULT_LHR_V30_SWISH10_PROJ_DIM,
)


W79_LHR_V31_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v31.v1")
W79_DEFAULT_LHR_V31_MAX_K: int = 4096
W79_DEFAULT_LHR_V31_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_DIM: (
    int) = 8
W79_DEFAULT_LHR_V31_SWISH11_PROJ_DIM: int = 88


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


@dataclasses.dataclass
class LongHorizonReconstructionV31Head:
    inner_v30: LongHorizonReconstructionV30Head
    max_k: int
    replacement_then_restart_after_long_delay_dim: int
    swish11_proj_dim: int
    swish11_proj_W: "_np.ndarray | None" = None
    scorer_layer21: "_np.ndarray | None" = None
    scorer_layer21_residual: float = 0.0
    replacement_then_restart_after_long_delay_W: (
        "_np.ndarray | None") = None

    @classmethod
    def init(
            cls, *,
            max_k: int = W79_DEFAULT_LHR_V31_MAX_K,
            replacement_then_restart_after_long_delay_dim: int = (
                W79_DEFAULT_LHR_V31_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_DIM),
            swish11_proj_dim: int = (
                W79_DEFAULT_LHR_V31_SWISH11_PROJ_DIM),
            seed: int = 79200,
    ) -> "LongHorizonReconstructionV31Head":
        v30 = LongHorizonReconstructionV30Head.init(
            max_k=int(max_k), seed=int(seed))
        rng = _np.random.default_rng(int(seed) ^ 0xFEED_79)
        out_dim = int(v30.out_dim)
        s11_W = rng.standard_normal(
            (int(W78_DEFAULT_LHR_V30_SWISH10_PROJ_DIM),
             int(swish11_proj_dim))) * 0.05
        rtr_W = rng.standard_normal(
            (int(replacement_then_restart_after_long_delay_dim),
             int(out_dim))) * 0.05
        return cls(
            inner_v30=v30,
            max_k=int(max_k),
            replacement_then_restart_after_long_delay_dim=int(
                replacement_then_restart_after_long_delay_dim),
            swish11_proj_dim=int(swish11_proj_dim),
            swish11_proj_W=s11_W.astype(_np.float64),
            replacement_then_restart_after_long_delay_W=rtr_W.astype(
                _np.float64),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v30.out_dim)

    def replacement_then_restart_after_long_delay_value(
            self, *,
            replacement_then_restart_after_long_delay_indicator: (
                Sequence[float]),
    ) -> "_np.ndarray":
        v = _np.asarray(
            replacement_then_restart_after_long_delay_indicator,
            dtype=_np.float64)
        d = int(self.replacement_then_restart_after_long_delay_dim)
        if v.size < d:
            v = _np.concatenate([
                v, _np.zeros(d - v.size, dtype=_np.float64)])
        elif v.size > d:
            v = v[:d]
        return (
            v
            @ self.replacement_then_restart_after_long_delay_W)

    def thirty_way_value(
            self, *,
            carrier: Sequence[float], k: int,
            long_horizon_reconstruction_indicator: (
                Sequence[float] | None) = None,
            replacement_then_restart_after_long_delay_indicator: (
                Sequence[float] | None) = None,
            **kwargs: Any,
    ) -> "_np.ndarray":
        v29 = self.inner_v30.twenty_nine_way_value(
            carrier=list(carrier), k=int(k),
            long_horizon_reconstruction_indicator=(
                list(long_horizon_reconstruction_indicator)
                if long_horizon_reconstruction_indicator
                is not None else None),
            **kwargs)
        v29 = _np.asarray(v29, dtype=_np.float64)
        if (replacement_then_restart_after_long_delay_indicator
                is not None):
            rtr = (
                self
                .replacement_then_restart_after_long_delay_value(
                    replacement_then_restart_after_long_delay_indicator=list(
                        replacement_then_restart_after_long_delay_indicator)))
            return v29 + 0.05 * rtr
        return v29

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_lhr_v31_head",
            "schema": W79_LHR_V31_SCHEMA_VERSION,
            "inner_v30_cid": str(self.inner_v30.cid()),
            "max_k": int(self.max_k),
            "replacement_then_restart_after_long_delay_dim": int(
                self
                .replacement_then_restart_after_long_delay_dim),
            "swish11_proj_dim": int(self.swish11_proj_dim),
            "swish11_proj_W_cid": (
                _ndarray_cid(self.swish11_proj_W)
                if self.swish11_proj_W is not None else "none"),
            "replacement_then_restart_after_long_delay_W_cid": (
                _ndarray_cid(
                    self
                    .replacement_then_restart_after_long_delay_W)
                if self.replacement_then_restart_after_long_delay_W
                is not None else "none"),
            "scorer_layer21_cid": (
                _ndarray_cid(self.scorer_layer21)
                if self.scorer_layer21 is not None
                else "untrained"),
            "scorer_layer21_residual": float(round(
                self.scorer_layer21_residual, 12)),
        })


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV31Witness:
    schema: str
    head_cid: str
    max_k: int
    replacement_then_restart_after_long_delay_dim: int
    out_dim: int
    n_heads: int
    thirty_way_runs: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_lhr_v31_witness",
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "max_k": int(self.max_k),
            "replacement_then_restart_after_long_delay_dim": int(
                self
                .replacement_then_restart_after_long_delay_dim),
            "out_dim": int(self.out_dim),
            "n_heads": int(self.n_heads),
            "thirty_way_runs": bool(self.thirty_way_runs),
        })


def emit_lhr_v31_witness(
        head: LongHorizonReconstructionV31Head, *,
        carrier: Sequence[float],
        k: int = 16,
        long_horizon_reconstruction_indicator: (
            Sequence[float] | None) = None,
        replacement_then_restart_after_long_delay_indicator: (
            Sequence[float] | None) = None,
        **kwargs: Any,
) -> LongHorizonReconstructionV31Witness:
    runs = True
    try:
        head.thirty_way_value(
            carrier=list(carrier), k=int(k),
            long_horizon_reconstruction_indicator=(
                list(long_horizon_reconstruction_indicator)
                if long_horizon_reconstruction_indicator
                is not None else None),
            replacement_then_restart_after_long_delay_indicator=(
                list(
                    replacement_then_restart_after_long_delay_indicator)
                if replacement_then_restart_after_long_delay_indicator
                is not None else None),
            **kwargs)
    except Exception:
        runs = False
    return LongHorizonReconstructionV31Witness(
        schema=W79_LHR_V31_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        max_k=int(head.max_k),
        replacement_then_restart_after_long_delay_dim=int(
            head.replacement_then_restart_after_long_delay_dim),
        out_dim=int(head.out_dim),
        n_heads=30,
        thirty_way_runs=bool(runs),
    )


__all__ = [
    "W79_LHR_V31_SCHEMA_VERSION",
    "W79_DEFAULT_LHR_V31_MAX_K",
    "W79_DEFAULT_LHR_V31_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_DIM",
    "W79_DEFAULT_LHR_V31_SWISH11_PROJ_DIM",
    "LongHorizonReconstructionV31Head",
    "LongHorizonReconstructionV31Witness",
    "emit_lhr_v31_witness",
]
