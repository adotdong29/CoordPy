"""W59 M12 — Long-Horizon Retention V11.

Strictly extends W58's
``coordpy.long_horizon_retention_v10.LongHorizonReconstructionV10Head``.
V11 adds a *tenth* head — **retrieval-conditioned reconstruction**
— and raises ``max_k`` to **80** (vs V10's 72). The new head
consumes a "retrieval signal" — the cache-controller-V2 retrieval
score field at the corresponding turn — and projects it into the
reconstruction output dimension.

V11 also introduces a **trained retention scorer head**: a
single linear layer ``s = W_scorer x`` fitted by closed-form
least squares on a small synthetic supervised set. This is the
first reconstruction layer in the programme whose parameters are
*fit* (not just initialised). ``W59-L-V11-LHR-SCORER-FIT-CAP``
documents the boundary: a linear ridge fit, not a deep network,
not autograd.

V11 strictly extends V10: when ``retrieval_state = None``, V11's
``retrieval_conditioned_value`` reduces to V10 byte-for-byte.

Honest scope
------------

* The V11 head is *partly trained*: the retention scorer is fit
  by closed-form ridge. The reconstruction heads are still
  initialised but not trained end-to-end.
* The H-bar asserts ``evaluate_lhr_v11_five_way`` runs without
  crashing and reports five MSEs (proxy / substrate / hidden /
  attention / retrieval). The *quality* claim that the new head
  beats older heads on retrieval-aligned targets is *constructive
  only* (synthetic targets explicitly projected via the retrieval
  head by definition).
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
        "coordpy.long_horizon_retention_v11 requires numpy"
        ) from exc

from .long_horizon_retention_v10 import (
    LongHorizonReconstructionV10Head,
    W58_DEFAULT_LHR_V10_ATTENTION_DIM,
    W58_DEFAULT_LHR_V10_HIDDEN_DIM,
    W58_DEFAULT_LHR_V10_MAX_K,
)


W59_LHR_V11_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v11.v1")
W59_DEFAULT_LHR_V11_MAX_K: int = 80
W59_DEFAULT_LHR_V11_RETRIEVAL_DIM: int = 32
W59_DEFAULT_LHR_V11_SCORER_RIDGE_LAMBDA: float = 0.10


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
class LongHorizonReconstructionV11Head:
    inner_v10: LongHorizonReconstructionV10Head
    retrieval_dim: int
    max_k: int
    # Trained retention scorer: a single linear (carrier_dim,)
    # ridge-fit head, fit on synthetic supervised data when
    # ``fit_retention_scorer`` is called. None until fitted.
    retention_scorer: "_np.ndarray | None" = None
    retention_scorer_ridge_lambda: float = (
        W59_DEFAULT_LHR_V11_SCORER_RIDGE_LAMBDA)
    retention_scorer_fit_residual: float = 0.0

    @classmethod
    def init(
            cls, *,
            hidden_dim: int = W58_DEFAULT_LHR_V10_HIDDEN_DIM,
            attention_dim: int = (
                W58_DEFAULT_LHR_V10_ATTENTION_DIM),
            retrieval_dim: int = (
                W59_DEFAULT_LHR_V11_RETRIEVAL_DIM),
            max_k: int = W59_DEFAULT_LHR_V11_MAX_K,
            seed: int = 59120,
    ) -> "LongHorizonReconstructionV11Head":
        v10 = LongHorizonReconstructionV10Head.init(
            hidden_dim=int(hidden_dim),
            attention_dim=int(attention_dim),
            max_k=int(W58_DEFAULT_LHR_V10_MAX_K),
            seed=int(seed))
        return cls(
            inner_v10=v10,
            retrieval_dim=int(retrieval_dim),
            max_k=int(max_k),
        )

    @property
    def out_dim(self) -> int:
        return int(self.inner_v10.out_dim)

    def retrieval_conditioned_value(
            self, *, carrier: Sequence[float], k: int,
            retrieval_state: Sequence[float] | None,
            attention_state: Sequence[float] | None,
            hidden_state: Sequence[float] | None,
            substrate_state: Sequence[float] | None,
    ) -> list[float]:
        base = self.inner_v10.attention_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(self.inner_v10.max_k)),
            attention_state=attention_state,
            hidden_state=hidden_state,
            substrate_state=substrate_state)
        if retrieval_state is None:
            return base
        r = list(retrieval_state)[: int(self.retrieval_dim)]
        out_dim = int(self.out_dim)
        contrib = [0.0] * out_dim
        for i in range(out_dim):
            s = 0.0
            for j in range(len(r)):
                phase = (
                    float(((i * 23) + j * 17) % 64) / 32.0
                    - 1.0)
                s += float(r[j]) * phase
            contrib[i] = 0.03 * s
        return [
            float(base[i] if i < len(base) else 0.0)
            + float(contrib[i])
            for i in range(out_dim)
        ]

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_LHR_V11_SCHEMA_VERSION,
            "kind": "lhr_v11_head",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "retrieval_dim": int(self.retrieval_dim),
            "max_k": int(self.max_k),
            "retention_scorer_cid": (
                _ndarray_cid(self.retention_scorer)
                if self.retention_scorer is not None
                else "untrained"),
            "retention_scorer_ridge_lambda": float(round(
                self.retention_scorer_ridge_lambda, 12)),
            "retention_scorer_fit_residual": float(round(
                self.retention_scorer_fit_residual, 12)),
        })


def fit_lhr_v11_retention_scorer(
        head: LongHorizonReconstructionV11Head,
        *,
        train_carriers: Sequence[Sequence[float]],
        train_targets: Sequence[float],
        ridge_lambda: float = (
            W59_DEFAULT_LHR_V11_SCORER_RIDGE_LAMBDA),
) -> tuple[LongHorizonReconstructionV11Head, float]:
    """Fit a single linear retention scorer by closed-form ridge.

    The scorer maps a carrier to a scalar retention score. Used
    by the W59 cache controller V2 / consensus controller V5 for
    routing.
    """
    if not train_carriers:
        return head, 0.0
    X = _np.asarray(
        [list(c) for c in train_carriers], dtype=_np.float64)
    y = _np.asarray(
        list(train_targets), dtype=_np.float64)
    d = X.shape[1]
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(d, dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    pred = X @ w
    residual = float(_np.mean(_np.abs(y - pred)))
    head.retention_scorer = w
    head.retention_scorer_ridge_lambda = float(ridge_lambda)
    head.retention_scorer_fit_residual = float(residual)
    return head, float(residual)


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV11Witness:
    schema: str
    head_cid: str
    inner_v10_cid: str
    max_k: int
    n_heads: int  # 10 = V10's 9 + retrieval
    retention_scorer_fitted: bool
    retention_scorer_residual: float
    causal_examples: int
    substrate_examples: int
    hidden_state_examples: int
    attention_examples: int
    retrieval_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "head_cid": str(self.head_cid),
            "inner_v10_cid": str(self.inner_v10_cid),
            "max_k": int(self.max_k),
            "n_heads": int(self.n_heads),
            "retention_scorer_fitted": bool(
                self.retention_scorer_fitted),
            "retention_scorer_residual": float(round(
                self.retention_scorer_residual, 12)),
            "causal_examples": int(self.causal_examples),
            "substrate_examples": int(self.substrate_examples),
            "hidden_state_examples": int(
                self.hidden_state_examples),
            "attention_examples": int(self.attention_examples),
            "retrieval_examples": int(self.retrieval_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "lhr_v11_witness",
            "witness": self.to_dict()})


def emit_lhr_v11_witness(
        *,
        head: LongHorizonReconstructionV11Head,
        n_causal: int = 0,
        n_substrate: int = 0,
        n_hidden: int = 0,
        n_attention: int = 0,
        n_retrieval: int = 0,
) -> LongHorizonReconstructionV11Witness:
    return LongHorizonReconstructionV11Witness(
        schema=W59_LHR_V11_SCHEMA_VERSION,
        head_cid=str(head.cid()),
        inner_v10_cid=str(head.inner_v10.cid()),
        max_k=int(head.max_k),
        n_heads=10,
        retention_scorer_fitted=bool(
            head.retention_scorer is not None),
        retention_scorer_residual=float(
            head.retention_scorer_fit_residual),
        causal_examples=int(n_causal),
        substrate_examples=int(n_substrate),
        hidden_state_examples=int(n_hidden),
        attention_examples=int(n_attention),
        retrieval_examples=int(n_retrieval),
    )


def evaluate_lhr_v11_five_way(
        head: LongHorizonReconstructionV11Head,
        *,
        carrier_examples: Sequence[Sequence[float]],
        target_examples: Sequence[Sequence[float]],
        substrate_states: Sequence[Sequence[float] | None],
        hidden_states: Sequence[Sequence[float] | None],
        attention_states: Sequence[Sequence[float] | None],
        retrieval_states: Sequence[Sequence[float] | None],
        k: int = 16,
) -> dict[str, Any]:
    """Five-way comparison: proxy vs substrate vs hidden vs
    attention vs retrieval. Reports per-head MSE."""
    if not carrier_examples:
        return {
            "schema": W59_LHR_V11_SCHEMA_VERSION,
            "proxy_mse": 0.0,
            "substrate_mse": 0.0,
            "hidden_state_mse": 0.0,
            "attention_mse": 0.0,
            "retrieval_mse": 0.0,
            "n": 0,
        }
    proxy_se = 0.0
    sub_se = 0.0
    hid_se = 0.0
    att_se = 0.0
    ret_se = 0.0
    out_dim = int(head.out_dim)
    n = 0
    for carrier, target, sub_st, hid_st, att_st, ret_st in zip(
            carrier_examples, target_examples,
            substrate_states, hidden_states,
            attention_states, retrieval_states):
        proxy_out = head.inner_v10.inner_v9.causal_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v10.inner_v9.max_k)))
        sub_out = head.inner_v10.inner_v9.substrate_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v10.inner_v9.max_k)),
            substrate_state=sub_st)
        hid_out = head.inner_v10.inner_v9.hidden_state_conditioned_value(
            carrier=list(carrier),
            k=min(int(k), int(head.inner_v10.inner_v9.max_k)),
            hidden_state=hid_st, substrate_state=sub_st)
        att_out = head.inner_v10.attention_conditioned_value(
            carrier=list(carrier), k=int(k),
            attention_state=att_st,
            hidden_state=hid_st,
            substrate_state=sub_st)
        ret_out = head.retrieval_conditioned_value(
            carrier=list(carrier), k=int(k),
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
            proxy_se += d_p * d_p
            sub_se += d_s * d_s
            hid_se += d_h * d_h
            att_se += d_a * d_a
            ret_se += d_r * d_r
        n += 1
    denom = max(1, n) * max(1, out_dim)
    return {
        "schema": W59_LHR_V11_SCHEMA_VERSION,
        "proxy_mse": float(proxy_se / float(denom)),
        "substrate_mse": float(sub_se / float(denom)),
        "hidden_state_mse": float(hid_se / float(denom)),
        "attention_mse": float(att_se / float(denom)),
        "retrieval_mse": float(ret_se / float(denom)),
        "n": int(n),
    }


__all__ = [
    "W59_LHR_V11_SCHEMA_VERSION",
    "W59_DEFAULT_LHR_V11_MAX_K",
    "W59_DEFAULT_LHR_V11_RETRIEVAL_DIM",
    "W59_DEFAULT_LHR_V11_SCORER_RIDGE_LAMBDA",
    "LongHorizonReconstructionV11Head",
    "LongHorizonReconstructionV11Witness",
    "fit_lhr_v11_retention_scorer",
    "emit_lhr_v11_witness",
    "evaluate_lhr_v11_five_way",
]
