"""W79 M4 — Replay Controller V20.

Strictly extends W78's ``coordpy.replay_controller_v19``. V19 had
26 regimes and a 16-label long-horizon-reconstruction-aware
routing head. V20 introduces **one new** regime and a new
**replacement-then-restart-after-long-delay-aware routing head**.

* ``replacement_then_restart_after_long_delay_regime`` — long-
  delay blackout (~25..85 %) with a replacement at ~40 % and a
  restart at ~70 %, then a reconstruction request at ~92 %.
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
        "coordpy.replay_controller_v20 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v19 import (
    ReplayControllerV19,
    W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS,
    W78_REPLAY_REGIMES_V19,
)


W79_REPLAY_CONTROLLER_V20_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v20.v1")

W79_REPLAY_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY: str = (
    "replacement_then_restart_after_long_delay_regime")
W79_REPLAY_REGIMES_V20_NEW: tuple[str, ...] = (
    W79_REPLAY_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY,
)
W79_REPLAY_REGIMES_V20: tuple[str, ...] = (
    *W78_REPLAY_REGIMES_V19,
    *W79_REPLAY_REGIMES_V20_NEW,
)
W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_ROUTING_LABEL: str = (
    "replacement_then_restart_after_long_delay_route")
W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS: tuple[
        str, ...] = (
    *W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS,
    W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_ROUTING_LABEL,
)


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


@dataclasses.dataclass
class ReplayControllerV20:
    inner_v19: ReplayControllerV19
    per_role_per_regime_heads_v20: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    replacement_then_restart_after_long_delay_aware_routing_head: (
        "_np.ndarray | None") = None

    @classmethod
    def init(cls) -> "ReplayControllerV20":
        return cls(
            inner_v19=ReplayControllerV19.init())

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_replay_controller_v20",
            "schema": W79_REPLAY_CONTROLLER_V20_SCHEMA_VERSION,
            "inner_v19_cid": str(self.inner_v19.cid()),
            "per_role_per_regime_heads_v20_cids": {
                f"{k[0]}::{k[1]}": _ndarray_cid(v)
                for k, v in sorted(
                    self.per_role_per_regime_heads_v20.items())},
            "replacement_then_restart_after_long_delay_aware_routing_head_cid": _ndarray_cid(
                self
                .replacement_then_restart_after_long_delay_aware_routing_head),
        })


def fit_replay_controller_v20_per_role(
        *,
        controller: ReplayControllerV20,
        role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = 0.10,
) -> tuple[ReplayControllerV20, dict[str, Any]]:
    """Fit a 5-dim feature ridge per regime for the V20 head."""
    new_heads = dict(
        controller.per_role_per_regime_heads_v20)
    fitted_count = 0
    for regime, cands in train_candidates_per_regime.items():
        decs = list(
            train_decisions_per_regime.get(regime, []))
        if not cands or not decs:
            continue
        X = _np.asarray(
            [[float(c.flop_reuse),
              float(c.flop_recompute),
              float(c.drift_l2_reuse),
              float(c.drift_l2_recompute),
              float(c.n_corruption_flags)]
             for c in cands], dtype=_np.float64)
        y = _np.asarray(
            [1.0 if d == "choose_reuse" else 0.0 for d in decs],
            dtype=_np.float64)
        lam = max(float(ridge_lambda), 1e-9)
        A = X.T @ X + lam * _np.eye(
            X.shape[1], dtype=_np.float64)
        b = X.T @ y
        try:
            theta = _np.linalg.solve(A, b)
        except Exception:
            theta = _np.zeros(
                (X.shape[1],), dtype=_np.float64)
        new_heads[(str(role), str(regime))] = theta.astype(
            _np.float64).copy()
        fitted_count += 1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v20=new_heads)
    return fitted, {
        "role": str(role),
        "n_regimes_fit": int(fitted_count),
    }


def fit_replay_v20_replacement_then_restart_after_long_delay_aware_routing_head(
        *,
        controller: ReplayControllerV20,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = 0.10,
) -> tuple[ReplayControllerV20, dict[str, Any]]:
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.shape[0] == 0:
        return controller, {"n": 0}
    n_labels = int(
        len(W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS))
    Y = _np.zeros((X.shape[0], n_labels), dtype=_np.float64)
    label_to_idx = {
        l: i for i, l in enumerate(
            W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS)}
    for i, lab in enumerate(train_routing_labels):
        if str(lab) in label_to_idx:
            Y[i, label_to_idx[str(lab)]] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(
        X.shape[1], dtype=_np.float64)
    B = X.T @ Y
    try:
        theta = _np.linalg.solve(A, B).T
    except Exception:
        theta = _np.zeros(
            (n_labels, X.shape[1]), dtype=_np.float64)
    fitted = dataclasses.replace(
        controller,
        replacement_then_restart_after_long_delay_aware_routing_head=theta.astype(
            _np.float64))
    return fitted, {
        "n_labels": int(n_labels),
        "n": int(X.shape[0]),
    }


@dataclasses.dataclass(frozen=True)
class ReplayControllerV20Witness:
    schema: str
    controller_cid: str
    n_regimes: int
    n_routing_labels: int
    routing_head_present: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_replay_controller_v20_witness",
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_regimes": int(self.n_regimes),
            "n_routing_labels": int(self.n_routing_labels),
            "routing_head_present": bool(
                self.routing_head_present),
        })


def emit_replay_controller_v20_witness(
        controller: ReplayControllerV20,
) -> ReplayControllerV20Witness:
    return ReplayControllerV20Witness(
        schema=W79_REPLAY_CONTROLLER_V20_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_regimes=int(len(W79_REPLAY_REGIMES_V20)),
        n_routing_labels=int(
            len(W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS)),
        routing_head_present=bool(
            controller
            .replacement_then_restart_after_long_delay_aware_routing_head
            is not None),
    )


__all__ = [
    "W79_REPLAY_CONTROLLER_V20_SCHEMA_VERSION",
    "W79_REPLAY_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY",
    "W79_REPLAY_REGIMES_V20",
    "W79_REPLAY_REGIMES_V20_NEW",
    "W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_ROUTING_LABEL",
    "W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS",
    "ReplayControllerV20",
    "ReplayControllerV20Witness",
    "fit_replay_controller_v20_per_role",
    "fit_replay_v20_replacement_then_restart_after_long_delay_aware_routing_head",
    "emit_replay_controller_v20_witness",
]
