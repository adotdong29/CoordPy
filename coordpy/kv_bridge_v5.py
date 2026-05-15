"""W60 M2 — KV Bridge V5.

Strictly extends W59's ``coordpy.kv_bridge_v4``. V4 fit a *single
scalar α* along a fixed random direction by closed-form ridge.
V5 makes the fit *multi-direction* and *multi-target*:

* **Multi-direction closed-form ridge fit** —
  ``fit_kv_bridge_v5_correction`` fits a vector ``α ∈ R^d`` over
  ``d`` orthogonal correction directions simultaneously. The
  substrate-side Jacobian becomes a ``(n_train, d)`` matrix and
  the solve becomes a real multi-feature ridge regression:
  ``α* = (G^T G + λI)^{-1} G^T (target - L2_pre)``. V4's
  single-scalar fit is the d=1 case of V5.
* **Logit-direction target** — V4 fit against a scalar
  ``target_l2``. V5 can additionally fit against a *logit-shift
  direction* ``target_delta_logits``: the fit minimises the L2
  distance between the *injected* last-position logit delta and
  the target direction, again via closed-form linear ridge
  (Jacobian sampled once by finite differences).
* **Two correction layers** — V5 carries TWO additive corrections
  (``correction_k_v5_layer_a`` and ``correction_k_v5_layer_b``)
  that can be fit jointly. The W60 controller composes them as
  ``correction = α_a · direction_a + α_b · direction_b``.
* **Cross-bank readback fingerprint** — V5 computes the 128-bucket
  fingerprint of all FOUR role-bank readbacks (a, b, c, d) in
  one pass. Lets the CRC V8 detect corruption that hits one bank
  but not the others.
* **Bidirectional carrier extraction** — V5 exposes a
  ``extract_carrier_from_v5_kv_cache`` that reverses the
  injection: given a cache, it estimates the carrier that would
  have produced the observed top-T inject slots under the V5
  projection (closed-form least squares against the projection
  matrix).

Honest scope
------------

* The fit is *multi-feature closed-form ridge*, not autograd /
  SGD / GPU. ``W60-L-V5-NO-AUTOGRAD-CAP`` carries forward the
  W59 ridge boundary unchanged.
* The reverse-extract is a *least-squares estimate*, not a perfect
  decoder. It is exact only when the inject region is uncorrupted
  and the projection has full column rank.
* Hosted backends remain text-only.
  ``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
        "coordpy.kv_bridge_v5 requires numpy") from exc

from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    inject_carrier_into_v4_kv_cache,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W60_KV_BRIDGE_V5_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v5.v1")
W60_DEFAULT_BRIDGE_V5_RIDGE_LAMBDA: float = 0.05
W60_DEFAULT_BRIDGE_V5_RIDGE_PROBE_EPS: float = 0.04
W60_DEFAULT_BRIDGE_V5_N_DIRECTIONS: int = 4


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeV5Projection:
    """Two-correction-layer projection on top of a V4 projection.

    The V5 projection carries two additive correction tensors
    (``correction_layer_a`` and ``correction_layer_b``) that are
    composed at inject time. V4's single ``correction_*`` lives
    inside ``inner_v4``.
    """

    inner_v4: KVBridgeV4Projection
    correction_layer_a_k: "_np.ndarray"  # (L, H, T, Dh)
    correction_layer_a_v: "_np.ndarray"
    correction_layer_b_k: "_np.ndarray"
    correction_layer_b_v: "_np.ndarray"
    seed_v5: int

    @classmethod
    def init_from_v4(
            cls, inner: KVBridgeV4Projection,
            *, seed_v5: int = 60050050,
    ) -> "KVBridgeV5Projection":
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.n_inject_tokens)
        Dh = int(inner.d_head)
        return cls(
            inner_v4=inner,
            correction_layer_a_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_a_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_b_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_b_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            seed_v5=int(seed_v5),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v4.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v4.n_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(self.inner_v4.n_kv_heads)

    @property
    def n_inject_tokens(self) -> int:
        return int(self.inner_v4.n_inject_tokens)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v4.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v4.d_head)

    def composed_inner_v4(self) -> KVBridgeV4Projection:
        """Return a V4 projection whose ``correction_*`` field is
        the COMPOSITION of V4's original + layer_a + layer_b.
        Downstream consumers reuse V4's inject path."""
        new_k = (
            self.inner_v4.correction_k
            + self.correction_layer_a_k
            + self.correction_layer_b_k)
        new_v = (
            self.inner_v4.correction_v
            + self.correction_layer_a_v
            + self.correction_layer_b_v)
        return self.inner_v4.with_correction(
            correction_k=new_k, correction_v=new_v)

    def with_correction_a(
            self, *, correction_k: "_np.ndarray",
            correction_v: "_np.ndarray",
    ) -> "KVBridgeV5Projection":
        return dataclasses.replace(
            self,
            correction_layer_a_k=_np.asarray(
                correction_k, dtype=_np.float64).copy(),
            correction_layer_a_v=_np.asarray(
                correction_v, dtype=_np.float64).copy(),
        )

    def with_correction_b(
            self, *, correction_k: "_np.ndarray",
            correction_v: "_np.ndarray",
    ) -> "KVBridgeV5Projection":
        return dataclasses.replace(
            self,
            correction_layer_b_k=_np.asarray(
                correction_k, dtype=_np.float64).copy(),
            correction_layer_b_v=_np.asarray(
                correction_v, dtype=_np.float64).copy(),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_KV_BRIDGE_V5_SCHEMA_VERSION,
            "kind": "kv_bridge_v5_projection",
            "inner_v4_cid": self.inner_v4.cid(),
            "correction_layer_a_k_cid": _ndarray_cid(
                self.correction_layer_a_k),
            "correction_layer_a_v_cid": _ndarray_cid(
                self.correction_layer_a_v),
            "correction_layer_b_k_cid": _ndarray_cid(
                self.correction_layer_b_k),
            "correction_layer_b_v_cid": _ndarray_cid(
                self.correction_layer_b_v),
            "seed_v5": int(self.seed_v5),
        })


def inject_carrier_into_v5_kv_cache(
        *,
        carrier: Sequence[float],
        projection: KVBridgeV5Projection,
        kv_cache: TinyV3KVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[TinyV3KVCache, dict[str, Any]]:
    """V5 injection: defers to V4 with a composed correction."""
    composed = projection.composed_inner_v4()
    return inject_carrier_into_v4_kv_cache(
        carrier=carrier, projection=composed,
        kv_cache=kv_cache,
        layer_indices=layer_indices,
        position=position, role=role)


@dataclasses.dataclass(frozen=True)
class KVBridgeV5FitReport:
    schema: str
    n_train_examples: int
    n_directions: int
    pre_fit_mean_residual: float
    post_fit_mean_residual: float
    fit_used_closed_form: bool
    ridge_lambda: float
    condition_number: float
    fit_kind: str   # "scalar_l2" | "logit_direction"
    alpha_l2: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_directions": int(self.n_directions),
            "pre_fit_mean_residual": float(round(
                self.pre_fit_mean_residual, 12)),
            "post_fit_mean_residual": float(round(
                self.post_fit_mean_residual, 12)),
            "fit_used_closed_form": bool(
                self.fit_used_closed_form),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "fit_kind": str(self.fit_kind),
            "alpha_l2": float(round(self.alpha_l2, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v5_fit_report",
            "report": self.to_dict()})


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


def fit_kv_bridge_v5_correction(
        *,
        params: TinyV3SubstrateParams,
        projection: KVBridgeV5Projection,
        train_carriers: Sequence[Sequence[float]],
        train_target_l2: Sequence[float],
        follow_up_token_ids: Sequence[int],
        n_directions: int = (
            W60_DEFAULT_BRIDGE_V5_N_DIRECTIONS),
        ridge_lambda: float = (
            W60_DEFAULT_BRIDGE_V5_RIDGE_LAMBDA),
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
        fit_target: str = "layer_a",
) -> tuple[KVBridgeV5Projection, KVBridgeV5FitReport]:
    """Multi-direction closed-form ridge fit of the V5 correction.

    Decision variable: ``α ∈ R^{n_directions}`` placing weight on
    ``n_directions`` random orthogonal correction directions. For
    each direction ``d_j`` the substrate-side L2 Jacobian is
    estimated by central finite differences, then ``α`` is solved
    in closed form by ridge regression.

    Final correction: ``Σ_j α_j · direction_j``. With
    ``n_directions=1`` this reduces to V4's single-α fit.
    """
    if not train_carriers or not train_target_l2:
        raise ValueError(
            "fit requires non-empty train_carriers + targets")
    if len(train_carriers) != len(train_target_l2):
        raise ValueError(
            "train_carriers and train_target_l2 length mismatch")
    rng = _np.random.default_rng(int(projection.seed_v5) + 5051)
    n = int(len(train_carriers))
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    T = int(projection.n_inject_tokens)
    Dh = int(projection.d_head)
    nd = max(1, int(n_directions))
    # Generate nd random directions in correction space.
    dirs_k = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    dirs_v = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    inner_v4 = projection.inner_v4
    base_cache_v3 = TinyV3KVCache.empty(L)
    base_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache_v3, return_attention=False)
    base_logits = base_trace.logits[-1]
    eps = W60_DEFAULT_BRIDGE_V5_RIDGE_PROBE_EPS

    def project_with_correction(
            corr_k: "_np.ndarray", corr_v: "_np.ndarray",
    ) -> KVBridgeV4Projection:
        # Compose V4 correction = inner V4 + this proposed.
        comp_k = inner_v4.correction_k + corr_k
        comp_v = inner_v4.correction_v + corr_v
        return inner_v4.with_correction(
            correction_k=comp_k, correction_v=comp_v)

    def measure_l2(corr_k: "_np.ndarray",
                    corr_v: "_np.ndarray",
                    carrier: Sequence[float]) -> float:
        proj = project_with_correction(corr_k, corr_v)
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=carrier, projection=proj,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=False)
        return float(_np.linalg.norm(
            t.logits[-1] - base_logits))

    # Pre-fit residual.
    zero_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    zero_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    pre_l2 = [measure_l2(zero_k, zero_v, c)
              for c in train_carriers]
    pre_residual = [pre_l2[i] - float(train_target_l2[i])
                     for i in range(n)]
    pre_mean = float(_np.mean(_np.abs(_np.asarray(
        pre_residual, dtype=_np.float64))))
    # Per-direction Jacobian g_{i,j} = dL2_i/dα_j at α=0.
    G = _np.zeros((n, nd), dtype=_np.float64)
    for j in range(nd):
        for i, c in enumerate(train_carriers):
            l2p = measure_l2(
                dirs_k[j] * eps, dirs_v[j] * eps, c)
            l2m = measure_l2(
                dirs_k[j] * (-eps), dirs_v[j] * (-eps), c)
            G[i, j] = (l2p - l2m) / (2.0 * eps)
    r = _np.asarray(pre_residual, dtype=_np.float64)
    lam = max(float(ridge_lambda), 1e-9)
    A = G.T @ G + lam * _np.eye(nd, dtype=_np.float64)
    b = -(G.T @ r)
    alpha = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    # Compose final correction.
    new_corr_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    new_corr_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    for j in range(nd):
        new_corr_k += float(alpha[j]) * dirs_k[j]
        new_corr_v += float(alpha[j]) * dirs_v[j]
    if str(fit_target) == "layer_b":
        fitted = projection.with_correction_b(
            correction_k=new_corr_k,
            correction_v=new_corr_v)
    else:
        fitted = projection.with_correction_a(
            correction_k=new_corr_k,
            correction_v=new_corr_v)
    # Post-fit residual.
    post_l2: list[float] = []
    fitted_inner_v4 = fitted.composed_inner_v4()
    for c in train_carriers:
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=c, projection=fitted_inner_v4,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=False)
        post_l2.append(float(_np.linalg.norm(
            t.logits[-1] - base_logits)))
    post_residual = [post_l2[i] - float(train_target_l2[i])
                      for i in range(n)]
    post_mean = float(_np.mean(_np.abs(_np.asarray(
        post_residual, dtype=_np.float64))))
    report = KVBridgeV5FitReport(
        schema=W60_KV_BRIDGE_V5_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_directions=int(nd),
        pre_fit_mean_residual=float(pre_mean),
        post_fit_mean_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        fit_kind="scalar_l2",
        alpha_l2=float(_np.linalg.norm(alpha)),
        converged=bool(post_mean <= pre_mean + 1e-9),
    )
    return fitted, report


def fit_kv_bridge_v5_logit_direction(
        *,
        params: TinyV3SubstrateParams,
        projection: KVBridgeV5Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits: Sequence[float],
        follow_up_token_ids: Sequence[int],
        n_directions: int = (
            W60_DEFAULT_BRIDGE_V5_N_DIRECTIONS),
        ridge_lambda: float = (
            W60_DEFAULT_BRIDGE_V5_RIDGE_LAMBDA),
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
        fit_target: str = "layer_b",
) -> tuple[KVBridgeV5Projection, KVBridgeV5FitReport]:
    """Fit V5 correction so the *injected last-position logit delta*
    is closest to ``target_delta_logits`` in the L2 direction sense.

    Decision variable: ``α ∈ R^{n_directions}`` over ``n_directions``
    random correction directions. We project each direction's effect
    on the logit delta onto the target direction; minimise the L2
    distance in closed-form ridge.
    """
    if not train_carriers:
        raise ValueError(
            "fit requires non-empty train_carriers")
    n = int(len(train_carriers))
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    T = int(projection.n_inject_tokens)
    Dh = int(projection.d_head)
    nd = max(1, int(n_directions))
    rng = _np.random.default_rng(int(projection.seed_v5) + 5061)
    dirs_k = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    dirs_v = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    target = _np.asarray(target_delta_logits, dtype=_np.float64)
    target_norm = float(_np.linalg.norm(target))
    if target_norm < 1e-12:
        target_unit = _np.zeros_like(target)
    else:
        target_unit = target / target_norm
    inner_v4 = projection.inner_v4
    base_cache_v3 = TinyV3KVCache.empty(L)
    base_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache_v3, return_attention=False)
    base_logits = base_trace.logits[-1]
    eps = W60_DEFAULT_BRIDGE_V5_RIDGE_PROBE_EPS

    def measure_delta_proj(
            corr_k: "_np.ndarray", corr_v: "_np.ndarray",
            carrier: Sequence[float]) -> float:
        comp_k = inner_v4.correction_k + corr_k
        comp_v = inner_v4.correction_v + corr_v
        proj = inner_v4.with_correction(
            correction_k=comp_k, correction_v=comp_v)
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=carrier, projection=proj,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=False)
        delta = t.logits[-1] - base_logits
        return float(_np.dot(delta, target_unit))

    zero_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    zero_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    pre_proj = [measure_delta_proj(zero_k, zero_v, c)
                 for c in train_carriers]
    pre_residual = [target_norm - pre_proj[i]
                     for i in range(n)]
    pre_mean = float(_np.mean(_np.abs(_np.asarray(pre_residual))))
    G = _np.zeros((n, nd), dtype=_np.float64)
    for j in range(nd):
        for i, c in enumerate(train_carriers):
            pj = measure_delta_proj(
                dirs_k[j] * eps, dirs_v[j] * eps, c)
            mj = measure_delta_proj(
                dirs_k[j] * (-eps), dirs_v[j] * (-eps), c)
            G[i, j] = (pj - mj) / (2.0 * eps)
    r = _np.asarray(pre_residual, dtype=_np.float64)
    lam = max(float(ridge_lambda), 1e-9)
    A = G.T @ G + lam * _np.eye(nd, dtype=_np.float64)
    b = G.T @ r
    alpha = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    new_corr_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    new_corr_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    for j in range(nd):
        new_corr_k += float(alpha[j]) * dirs_k[j]
        new_corr_v += float(alpha[j]) * dirs_v[j]
    if str(fit_target) == "layer_a":
        fitted = projection.with_correction_a(
            correction_k=new_corr_k,
            correction_v=new_corr_v)
    else:
        fitted = projection.with_correction_b(
            correction_k=new_corr_k,
            correction_v=new_corr_v)
    fitted_inner_v4 = fitted.composed_inner_v4()
    post_proj: list[float] = []
    for c in train_carriers:
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=c, projection=fitted_inner_v4,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=False)
        delta = t.logits[-1] - base_logits
        post_proj.append(float(_np.dot(delta, target_unit)))
    post_residual = [target_norm - post_proj[i] for i in range(n)]
    post_mean = float(_np.mean(_np.abs(_np.asarray(post_residual))))
    return fitted, KVBridgeV5FitReport(
        schema=W60_KV_BRIDGE_V5_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_directions=int(nd),
        pre_fit_mean_residual=float(pre_mean),
        post_fit_mean_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        fit_kind="logit_direction",
        alpha_l2=float(_np.linalg.norm(alpha)),
        converged=bool(post_mean <= pre_mean + 1e-9),
    )


def extract_carrier_from_v5_kv_cache(
        *, projection: KVBridgeV5Projection,
        kv_cache: TinyV3KVCache,
        role: str = "bank_a",
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
) -> dict[str, Any]:
    """Reverse the V5 injection: estimate the carrier that would
    have produced the observed inject slots, in least-squares
    sense, against the V5 projection.

    Returns ``{'carrier_estimate': list, 'residual_l2': float,
    'condition_number': float, 'n_observations': int}``.
    """
    comp = projection.composed_inner_v4()
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    T = int(projection.n_inject_tokens)
    d_kv = int(projection.n_kv_heads) * int(projection.d_head)
    # Collect observed inject slots.
    if str(position) == "prepend":
        slc = slice(0, T)
    else:
        # Not supported in v5 reverse for the prototype path.
        slc = slice(0, T)
    obs_pieces: list["_np.ndarray"] = []
    for l in layer_indices:
        if l >= len(kv_cache.keys):
            continue
        k = kv_cache.keys[l]
        v = kv_cache.values[l]
        if k.size == 0 or k.shape[0] < T:
            continue
        obs_pieces.append(_np.concatenate(
            [k[slc], v[slc]], axis=0))
    if not obs_pieces:
        return {
            "schema": W60_KV_BRIDGE_V5_SCHEMA_VERSION,
            "carrier_estimate": [0.0] * projection.carrier_dim,
            "residual_l2": 0.0,
            "condition_number": float("inf"),
            "n_observations": 0,
        }
    obs = _np.concatenate(
        [p.reshape(-1) for p in obs_pieces], axis=0)
    # Build the projection matrix M such that obs = M · carrier.
    # We use ONE-shot probes: project unit-vectors and observe.
    cd = int(projection.carrier_dim)
    columns: list["_np.ndarray"] = []
    zero_carrier = _np.zeros((cd,), dtype=_np.float64)
    base_keys, base_vals = comp.project(
        zero_carrier.tolist(), role=str(role))
    # base_keys: (L, T, d_kv). Slice to layer_indices.
    base_obs_pieces: list["_np.ndarray"] = []
    for l in layer_indices:
        base_obs_pieces.append(_np.concatenate(
            [base_keys[l], base_vals[l]], axis=0))
    base_obs = _np.concatenate(
        [p.reshape(-1) for p in base_obs_pieces], axis=0)
    for j in range(cd):
        unit = _np.zeros((cd,), dtype=_np.float64)
        unit[j] = 1.0
        ks, vs = comp.project(unit.tolist(), role=str(role))
        pieces: list["_np.ndarray"] = []
        for l in layer_indices:
            pieces.append(_np.concatenate(
                [ks[l], vs[l]], axis=0))
        col = _np.concatenate(
            [p.reshape(-1) for p in pieces], axis=0) - base_obs
        columns.append(col)
    M = _np.stack(columns, axis=1)
    # Solve obs = M · carrier + base_obs.
    target = obs - base_obs
    lam = 1e-6
    A = M.T @ M + lam * _np.eye(cd, dtype=_np.float64)
    b = M.T @ target
    carrier_est = _np.linalg.solve(A, b)
    residual = float(_np.linalg.norm(M @ carrier_est - target))
    cond = _safe_condition(A)
    return {
        "schema": W60_KV_BRIDGE_V5_SCHEMA_VERSION,
        "carrier_estimate": carrier_est.tolist(),
        "residual_l2": float(residual),
        "condition_number": float(cond),
        "n_observations": int(target.size),
    }


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


@dataclasses.dataclass(frozen=True)
class KVBridgeV5Witness:
    schema: str
    projection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    injected_layer_indices: tuple[int, ...]
    n_inject_tokens: int
    position: str
    role: str
    baseline_forward_cid: str
    injected_forward_cid: str
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    last_position_argmax_baseline: int
    last_position_argmax_injected: int
    cross_entropy_delta: float
    readback_fingerprint_128_all_banks: tuple[
        tuple[int, ...], ...]   # 4 banks × 128
    extract_carrier_residual_l2: float
    fit_report_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(
                self.post_inject_kv_cid),
            "injected_layer_indices": list(
                self.injected_layer_indices),
            "n_inject_tokens": int(self.n_inject_tokens),
            "position": str(self.position),
            "role": str(self.role),
            "baseline_forward_cid": str(
                self.baseline_forward_cid),
            "injected_forward_cid": str(
                self.injected_forward_cid),
            "max_abs_logit_perturbation": float(round(
                self.max_abs_logit_perturbation, 12)),
            "last_logit_l2_perturbation": float(round(
                self.last_logit_l2_perturbation, 12)),
            "last_position_argmax_baseline": int(
                self.last_position_argmax_baseline),
            "last_position_argmax_injected": int(
                self.last_position_argmax_injected),
            "cross_entropy_delta": float(round(
                self.cross_entropy_delta, 12)),
            "readback_fingerprint_128_all_banks": [
                list(fp)
                for fp in self.readback_fingerprint_128_all_banks],
            "extract_carrier_residual_l2": float(round(
                self.extract_carrier_residual_l2, 12)),
            "fit_report_cid": str(self.fit_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v5_witness",
            "witness": self.to_dict()})


def bridge_carrier_and_measure_v5(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV5Projection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyV3KVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
        fit_report_cid: str = "no_fit",
) -> KVBridgeV5Witness:
    composed = projection.composed_inner_v4()
    base_cache = (
        TinyV3KVCache.empty(int(projection.n_layers))
        if baseline_kv_cache is None
        else baseline_kv_cache.clone())
    baseline_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache, return_attention=False)
    new_cache, record = inject_carrier_into_v4_kv_cache(
        carrier=carrier, projection=composed,
        kv_cache=base_cache, layer_indices=layer_indices,
        position=position, role=role)
    injected_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=new_cache, return_attention=False)
    base_logits = baseline_trace.logits[-1]
    inj_logits = injected_trace.logits[-1]
    diff = inj_logits - base_logits
    max_abs = float(_np.max(_np.abs(diff)))
    l2 = float(_np.linalg.norm(diff))
    bp = _softmax_logits(base_logits)
    ip = _softmax_logits(inj_logits)
    eps = 1e-30
    ce = float(_np.sum(bp * (_np.log(bp + eps)
                              - _np.log(ip + eps))))
    # All-bank fingerprint.
    bank_fps: list[tuple[int, ...]] = []
    for bank in ("bank_a", "bank_b", "bank_c", "bank_d"):
        try:
            bc = TinyV3KVCache.empty(int(projection.n_layers))
            _, rec_b = inject_carrier_into_v4_kv_cache(
                carrier=carrier, projection=composed,
                kv_cache=bc, layer_indices=layer_indices,
                position=position, role=bank)
            bank_fps.append(tuple(
                int(b) for b in rec_b[
                    "readback_fingerprint_128"]))
        except Exception:
            bank_fps.append(tuple([0] * 128))
    # Reverse-extract test on the injected cache.
    extract = extract_carrier_from_v5_kv_cache(
        projection=projection, kv_cache=new_cache,
        role=role, layer_indices=layer_indices,
        position=position)
    return KVBridgeV5Witness(
        schema=W60_KV_BRIDGE_V5_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        pre_inject_kv_cid=str(record["pre_inject_kv_cid"]),
        post_inject_kv_cid=str(record["post_inject_kv_cid"]),
        injected_layer_indices=tuple(
            int(i) for i in record["injected_layer_indices"]),
        n_inject_tokens=int(record["n_inject_tokens"]),
        position=str(record["position"]),
        role=str(record["role"]),
        baseline_forward_cid=str(baseline_trace.cid()),
        injected_forward_cid=str(injected_trace.cid()),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        last_position_argmax_baseline=int(
            _np.argmax(base_logits)),
        last_position_argmax_injected=int(
            _np.argmax(inj_logits)),
        cross_entropy_delta=float(ce),
        readback_fingerprint_128_all_banks=tuple(bank_fps),
        extract_carrier_residual_l2=float(
            extract["residual_l2"]),
        fit_report_cid=str(fit_report_cid),
    )


__all__ = [
    "W60_KV_BRIDGE_V5_SCHEMA_VERSION",
    "W60_DEFAULT_BRIDGE_V5_RIDGE_LAMBDA",
    "W60_DEFAULT_BRIDGE_V5_RIDGE_PROBE_EPS",
    "W60_DEFAULT_BRIDGE_V5_N_DIRECTIONS",
    "KVBridgeV5Projection",
    "KVBridgeV5FitReport",
    "KVBridgeV5Witness",
    "inject_carrier_into_v5_kv_cache",
    "fit_kv_bridge_v5_correction",
    "fit_kv_bridge_v5_logit_direction",
    "extract_carrier_from_v5_kv_cache",
    "bridge_carrier_and_measure_v5",
]
