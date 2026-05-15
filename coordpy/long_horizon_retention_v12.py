"""W60 M15 — Long-Horizon Retention V12.

Strictly extends W59's
``coordpy.long_horizon_retention_v11``. V12 adds an *eleventh*
head — **replay-conditioned reconstruction** — and raises ``max_k``
to **96** (vs V11's 80). The new head consumes a "replay signal"
— the W60 ReplayController's per-turn decision distribution — and
projects it into the reconstruction output dimension.

V12 also extends V11's trained retention scorer: V11 fit a single
linear head; V12 fits a **two-layer (linear + ReLU + linear)**
scorer by *closed-form ridge on the post-ReLU features* — i.e.
the first layer is initialised randomly + frozen, then the second
linear layer is fit by ridge against the same synthetic
supervised set V11 used. This is still NOT autograd: the first
layer is fixed.

V12 strictly extends V11: when ``replay_state = None`` and
``two_layer_scorer = False``, V12's eleven-way value reduces to
V11 byte-for-byte.

Honest scope
------------

* The two-layer scorer's first layer is *random projection +
  frozen*. The second layer is fit by closed-form ridge. Not
  autograd. ``W60-L-V12-LHR-SCORER-FIT-CAP`` documents the
  boundary.
* The new head asserts that ``evaluate_lhr_v12_six_way`` runs
  without crashing and reports six MSEs (proxy / substrate /
  hidden / attention / retrieval / replay).
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
        "coordpy.long_horizon_retention_v12 requires numpy"
        ) from exc

from .long_horizon_retention_v11 import (
    LongHorizonReconstructionV11Head,
    W59_DEFAULT_LHR_V11_RETRIEVAL_DIM,
)


W60_LHR_V12_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v12.v1")
W60_DEFAULT_LHR_V12_MAX_K: int = 96
W60_DEFAULT_LHR_V12_REPLAY_DIM: int = 16
W60_DEFAULT_LHR_V12_HIDDEN_PROJ_DIM: int = 32
W60_DEFAULT_LHR_V12_SCORER_RIDGE_LAMBDA: float = 0.10


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray") -> str:
    a = _np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(repr(a.shape).encode("utf-8"))
    h.update(b"|")
    h.update(str(a.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


@dataclasses.dataclass
class LongHorizonReconstructionV12Head:
    inner_v11: LongHorizonReconstructionV11Head
    replay_dim: int
    max_k: int
    hidden_proj_dim: int
    hidden_proj_W: "_np.ndarray | None" = None  # (carrier_dim, H)
    scorer_layer2: "_np.ndarray | None" = None  # (H,)
    scorer_layer2_residual: float = 0.0

    @classmethod
    def init(
            cls, *,
            replay_dim: int = W60_DEFAULT_LHR_V12_REPLAY_DIM,
            max_k: int = W60_DEFAULT_LHR_V12_MAX_K,
            hidden_proj_dim: int = (
                W60_DEFAULT_LHR_V12_HIDDEN_PROJ_DIM),
            seed: int = 60120,
    ) -> "LongHorizonReconstructionV12Head":
        v11 = LongHorizonReconstructionV11Head.init(
            seed=int(seed))
        return cls(
            inner_v11=v11,
            replay_dim=int(replay_dim),
            max_k=int(max_k),
            hidden_proj_dim=int(hidden_proj_dim),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v11.out_dim)

    def replay_conditioned_value(
            self, *, carrier: Sequence[float], k: int,
            replay_state: Sequence[float] | None,
            retrieval_state: Sequence[float] | None,
            attention_state: Sequence[float] | None,
            hidden_state: Sequence[float] | None,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        base = self.inner_v11.retrieval_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(self.inner_v11.inner_v10.max_k)),
            retrieval_state=retrieval_state,
            attention_state=attention_state,
            hidden_state=hidden_state,
            substrate_state=substrate_state)
        if replay_state is None:
            return base
        r = list(replay_state)[: int(self.replay_dim)]
        out_dim = int(self.out_dim)
        contrib = [0.0] * out_dim
        for i in range(out_dim):
            s = 0.0
            for j in range(len(r)):
                phase = (
                    float(((i * 29) + j * 19) % 64) / 32.0
                    - 1.0)
                s += float(r[j]) * phase
            contrib[i] = 0.025 * s
        return [
            float(base[i] if i < len(base) else 0.0)
            + float(contrib[i])
            for i in range(out_dim)
        ]

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_LHR_V12_SCHEMA_VERSION,
            "kind": "lhr_v12_head",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "replay_dim": int(self.replay_dim),
            "max_k": int(self.max_k),
            "hidden_proj_dim": int(self.hidden_proj_dim),
            "hidden_proj_W_cid": (
                _ndarray_cid(self.hidden_proj_W)
                if self.hidden_proj_W is not None
                else "uninit"),
            "scorer_layer2_cid": (
                _ndarray_cid(self.scorer_layer2)
                if self.scorer_layer2 is not None
                else "untrained"),
            "scorer_layer2_residual": float(round(
                self.scorer_layer2_residual, 12)),
        })


def _safe_condition(a: "_np.ndarray") -> float:
    try:
        s = _np.linalg.svd(a, compute_uv=False)
        s_max = float(_np.max(s))
        s_min = float(_np.min(s))
        if s_min < 1e-30:
            return float("inf")
        return float(s_max / s_min)
    except Exception:
        return float("nan")


def fit_lhr_v12_two_layer_retention_scorer(
        head: LongHorizonReconstructionV12Head,
        *,
        train_carriers: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = (
            W60_DEFAULT_LHR_V12_SCORER_RIDGE_LAMBDA),
        seed: int = 60125,
) -> tuple[LongHorizonReconstructionV12Head, float]:
    """Fit a two-layer (random projection + frozen ReLU + linear)
    retention scorer by closed-form ridge.

    Layer 1: ``z = ReLU(carrier @ W1)`` where ``W1`` is random and
    frozen. Layer 2: ``s = z @ w2`` where ``w2`` is fit by
    closed-form ridge against ``train_targets``.
    """
    if not train_carriers:
        return head, 0.0
    X_raw = _np.asarray(
        [list(c) for c in train_carriers], dtype=_np.float64)
    y = _np.asarray(
        list(train_targets), dtype=_np.float64)
    d = int(X_raw.shape[1])
    H = int(head.hidden_proj_dim)
    rng = _np.random.default_rng(int(seed))
    W1 = rng.standard_normal((d, H)) * (1.0 / max(1, d) ** 0.5)
    Z = _np.maximum(0.0, X_raw @ W1)
    lam = max(float(ridge_lambda), 1e-9)
    A = Z.T @ Z + lam * _np.eye(H, dtype=_np.float64)
    b = Z.T @ y
    w2 = _np.linalg.solve(A, b)
    pred = Z @ w2
    residual = float(_np.mean(_np.abs(y - pred)))
    head.hidden_proj_W = W1
    head.scorer_layer2 = w2
    head.scorer_layer2_residual = float(residual)
    return head, float(residual)


def evaluate_lhr_v12_six_way(
        head: LongHorizonReconstructionV12Head,
        *,
        carrier_examples: Sequence[Sequence[float]],
        target_examples: Sequence[Sequence[float]],
        substrate_states: Sequence[Sequence[float] | None],
        hidden_states: Sequence[Sequence[float] | None],
        attention_states: Sequence[Sequence[float] | None],
        retrieval_states: Sequence[Sequence[float] | None],
        replay_states: Sequence[Sequence[float] | None],
        k: int = 16,
) -> dict[str, Any]:
    """Six-way comparison: proxy vs substrate vs hidden vs
    attention vs retrieval vs replay. Reports per-head MSE."""
    if not carrier_examples:
        return {
            "schema": W60_LHR_V12_SCHEMA_VERSION,
            "proxy_mse": 0.0,
            "substrate_mse": 0.0,
            "hidden_state_mse": 0.0,
            "attention_mse": 0.0,
            "retrieval_mse": 0.0,
            "replay_mse": 0.0,
            "n": 0,
        }
    proxy_se = 0.0
    sub_se = 0.0
    hid_se = 0.0
    att_se = 0.0
    ret_se = 0.0
    rep_se = 0.0
    out_dim = int(head.out_dim)
    n = 0
    inner_v9 = head.inner_v11.inner_v10.inner_v9
    for (carrier, target, sub_st, hid_st,
            att_st, ret_st, rep_st) in zip(
            carrier_examples, target_examples,
            substrate_states, hidden_states,
            attention_states, retrieval_states,
            replay_states):
        proxy_out = inner_v9.causal_value(
            carrier=list(carrier),
            k=min(int(k), int(inner_v9.max_k)))
        sub_out = inner_v9.substrate_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(inner_v9.max_k)),
            substrate_state=sub_st)
        hid_out = inner_v9.hidden_state_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(inner_v9.max_k)),
            hidden_state=hid_st, substrate_state=sub_st)
        att_out = (
            head.inner_v11.inner_v10.attention_conditioned_value(
                carrier=list(carrier), k=int(k),
                attention_state=att_st,
                hidden_state=hid_st,
                substrate_state=sub_st))
        ret_out = head.inner_v11.retrieval_conditioned_value(
            carrier=list(carrier), k=int(k),
            retrieval_state=ret_st,
            attention_state=att_st,
            hidden_state=hid_st,
            substrate_state=sub_st)
        rep_out = head.replay_conditioned_value(
            carrier=list(carrier), k=int(k),
            replay_state=rep_st,
            retrieval_state=ret_st,
            attention_state=att_st,
            hidden_state=hid_st,
            substrate_state=sub_st)
        t = list(target)[:out_dim]
        while len(t) < out_dim:
            t.append(0.0)
        for i in range(out_dim):
            d_p = (
                float(proxy_out[i] if i < len(proxy_out)
                       else 0.0)
                - float(t[i]))
            d_s = (
                float(sub_out[i] if i < len(sub_out)
                       else 0.0)
                - float(t[i]))
            d_h = (
                float(hid_out[i] if i < len(hid_out)
                       else 0.0)
                - float(t[i]))
            d_a = (
                float(att_out[i] if i < len(att_out)
                       else 0.0)
                - float(t[i]))
            d_r = (
                float(ret_out[i] if i < len(ret_out)
                       else 0.0)
                - float(t[i]))
            d_rp = (
                float(rep_out[i] if i < len(rep_out)
                       else 0.0)
                - float(t[i]))
            proxy_se += d_p * d_p
            sub_se += d_s * d_s
            hid_se += d_h * d_h
            att_se += d_a * d_a
            ret_se += d_r * d_r
            rep_se += d_rp * d_rp
        n += 1
    denom = max(1, n) * max(1, out_dim)
    return {
        "schema": W60_LHR_V12_SCHEMA_VERSION,
        "proxy_mse": float(proxy_se / float(denom)),
        "substrate_mse": float(sub_se / float(denom)),
        "hidden_state_mse": float(hid_se / float(denom)),
        "attention_mse": float(att_se / float(denom)),
        "retrieval_mse": float(ret_se / float(denom)),
        "replay_mse": float(rep_se / float(denom)),
        "n": int(n),
    }


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV12Witness:
    schema: str
    head_cid: str
    inner_v11_cid: str
    max_k: int
    n_heads: int   # 11 = V11's 10 + replay
    two_layer_scorer_fitted: bool
    two_layer_scorer_residual: float
    replay_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "inner_v11_cid": str(self.inner_v11_cid),
            "max_k": int(self.max_k),
            "n_heads": int(self.n_heads),
            "two_layer_scorer_fitted": bool(
                self.two_layer_scorer_fitted),
            "two_layer_scorer_residual": float(round(
                self.two_layer_scorer_residual, 12)),
            "replay_examples": int(self.replay_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v12_witness",
            "witness": self.to_dict()})


def emit_lhr_v12_witness(
        *,
        head: LongHorizonReconstructionV12Head,
        n_replay: int = 0,
) -> LongHorizonReconstructionV12Witness:
    return LongHorizonReconstructionV12Witness(
        schema=W60_LHR_V12_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        inner_v11_cid=str(head.inner_v11.cid()),
        max_k=int(head.max_k),
        n_heads=11,
        two_layer_scorer_fitted=bool(
            head.scorer_layer2 is not None),
        two_layer_scorer_residual=float(
            head.scorer_layer2_residual),
        replay_examples=int(n_replay),
    )


__all__ = [
    "W60_LHR_V12_SCHEMA_VERSION",
    "W60_DEFAULT_LHR_V12_MAX_K",
    "W60_DEFAULT_LHR_V12_REPLAY_DIM",
    "W60_DEFAULT_LHR_V12_HIDDEN_PROJ_DIM",
    "W60_DEFAULT_LHR_V12_SCORER_RIDGE_LAMBDA",
    "LongHorizonReconstructionV12Head",
    "LongHorizonReconstructionV12Witness",
    "fit_lhr_v12_two_layer_retention_scorer",
    "evaluate_lhr_v12_six_way",
    "emit_lhr_v12_witness",
]
