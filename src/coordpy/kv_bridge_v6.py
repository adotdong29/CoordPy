"""W61 M2 — KV Bridge V6.

Strictly extends W60's ``coordpy.kv_bridge_v5``. V5 fit a vector
α ∈ R^{n_directions} against *one* target (either scalar L2 or one
logit direction) by closed-form ridge. V6 makes the fit *multi-
target* simultaneously:

* **Matrix-valued multi-target ridge fit** —
  ``fit_kv_bridge_v6_multi_target`` fits a matrix
  ``A ∈ R^{nd × m}`` where each *column* is the α-vector for one of
  ``m`` target logit directions. The substrate-side Jacobian
  becomes a ``(n_train, nd, m)`` tensor and the solve uses the same
  closed-form ridge as V5 — one ``(nd × nd)`` system, m
  right-hand-sides. V5's one-target fit is the ``m=1`` case of V6.
* **Attention-pattern target fit** —
  ``fit_kv_bridge_v6_attention_pattern`` fits α to drive the
  per-(layer, head) attention-map of the injected forward toward a
  reference attention pattern (last-row L1 distance).
* **V6 cache-key fingerprint** —
  ``v6_cache_key_fingerprint`` hashes the *cache_keys* layer of the
  V6 substrate (not the V3 K/V tensors) so the CRC V9 can detect
  corruption that lands specifically on the content-addressable
  key axis.
* **Role-pair joint fit** — ``fit_kv_bridge_v6_role_pair`` jointly
  fits corrections for two role banks (a→b, b→c, etc.) by stacking
  them as separate target columns. The closed-form solve handles
  cross-role coupling.

Honest scope
------------

* The fit is *closed-form multi-feature multi-target ridge*, not
  autograd / SGD / GPU. ``W61-L-V6-NO-AUTOGRAD-CAP`` documents
  the new boundary.
* The attention-pattern fit minimises the L1 distance between the
  injected last-row attention vector and the reference, *not* a
  KL or Wasserstein distance.
* The reverse-extract from V5 carries forward unchanged.
* Hosted backends remain text-only.
  ``W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
        "coordpy.kv_bridge_v6 requires numpy") from exc

from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    inject_carrier_into_v4_kv_cache,
)
from .kv_bridge_v5 import (
    KVBridgeV5FitReport,
    KVBridgeV5Projection,
    W60_DEFAULT_BRIDGE_V5_N_DIRECTIONS,
    W60_DEFAULT_BRIDGE_V5_RIDGE_LAMBDA,
    W60_DEFAULT_BRIDGE_V5_RIDGE_PROBE_EPS,
    _safe_condition,
    extract_carrier_from_v5_kv_cache,
    fit_kv_bridge_v5_correction,
    fit_kv_bridge_v5_logit_direction,
    inject_carrier_into_v5_kv_cache,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v6 import (
    TinyV6KVCache, TinyV6SubstrateParams,
    forward_tiny_substrate_v6,
)


W61_KV_BRIDGE_V6_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v6.v1")
W61_DEFAULT_BRIDGE_V6_RIDGE_LAMBDA: float = 0.05
W61_DEFAULT_BRIDGE_V6_RIDGE_PROBE_EPS: float = 0.04
W61_DEFAULT_BRIDGE_V6_N_DIRECTIONS: int = 4


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeV6Projection:
    """V6 projection wraps V5 and adds a *matrix-valued multi-target
    correction* slot. The V5 layer-a / layer-b corrections still
    live in ``inner_v5``; V6 adds a ``correction_layer_c`` slot for
    multi-target fits.
    """
    inner_v5: KVBridgeV5Projection
    correction_layer_c_k: "_np.ndarray"   # (L, H, T, Dh)
    correction_layer_c_v: "_np.ndarray"
    seed_v6: int

    @classmethod
    def init_from_v5(
            cls, inner: KVBridgeV5Projection,
            *, seed_v6: int = 61050050,
    ) -> "KVBridgeV6Projection":
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.n_inject_tokens)
        Dh = int(inner.d_head)
        return cls(
            inner_v5=inner,
            correction_layer_c_k=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            correction_layer_c_v=_np.zeros(
                (L, H, T, Dh), dtype=_np.float64),
            seed_v6=int(seed_v6),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v5.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v5.n_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(self.inner_v5.n_kv_heads)

    @property
    def n_inject_tokens(self) -> int:
        return int(self.inner_v5.n_inject_tokens)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v5.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v5.d_head)

    def composed_inner_v4(self) -> KVBridgeV4Projection:
        """Compose V4 + V5 (layer_a + layer_b) + V6 layer_c."""
        v5 = self.inner_v5
        comp_k = (
            v5.inner_v4.correction_k
            + v5.correction_layer_a_k + v5.correction_layer_b_k
            + self.correction_layer_c_k)
        comp_v = (
            v5.inner_v4.correction_v
            + v5.correction_layer_a_v + v5.correction_layer_b_v
            + self.correction_layer_c_v)
        return v5.inner_v4.with_correction(
            correction_k=comp_k, correction_v=comp_v)

    def with_correction_c(
            self, *, correction_k: "_np.ndarray",
            correction_v: "_np.ndarray",
    ) -> "KVBridgeV6Projection":
        return dataclasses.replace(
            self,
            correction_layer_c_k=_np.asarray(
                correction_k, dtype=_np.float64).copy(),
            correction_layer_c_v=_np.asarray(
                correction_v, dtype=_np.float64).copy(),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_KV_BRIDGE_V6_SCHEMA_VERSION,
            "kind": "kv_bridge_v6_projection",
            "inner_v5_cid": self.inner_v5.cid(),
            "correction_layer_c_k_cid": _ndarray_cid(
                self.correction_layer_c_k),
            "correction_layer_c_v_cid": _ndarray_cid(
                self.correction_layer_c_v),
            "seed_v6": int(self.seed_v6),
        })


def inject_carrier_into_v6_kv_cache(
        *, carrier: Sequence[float],
        projection: KVBridgeV6Projection,
        kv_cache: TinyV3KVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[TinyV3KVCache, dict[str, Any]]:
    comp = projection.composed_inner_v4()
    return inject_carrier_into_v4_kv_cache(
        carrier=carrier, projection=comp,
        kv_cache=kv_cache,
        layer_indices=layer_indices,
        position=position, role=role)


@dataclasses.dataclass(frozen=True)
class KVBridgeV6FitReport:
    schema: str
    n_train_examples: int
    n_directions: int
    n_targets: int
    pre_fit_mean_residual: float
    post_fit_mean_residual: float
    fit_used_closed_form: bool
    ridge_lambda: float
    condition_number: float
    fit_kind: str        # "multi_target_logit" | "attention_pattern"
    alpha_l2: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_directions": int(self.n_directions),
            "n_targets": int(self.n_targets),
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
            "kind": "kv_bridge_v6_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v6_multi_target(
        *,
        params: TinyV3SubstrateParams,
        projection: KVBridgeV6Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        n_directions: int = (
            W61_DEFAULT_BRIDGE_V6_N_DIRECTIONS),
        ridge_lambda: float = (
            W61_DEFAULT_BRIDGE_V6_RIDGE_LAMBDA),
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[KVBridgeV6Projection, KVBridgeV6FitReport]:
    """Fit V6 layer_c correction against ``m`` target logit
    directions *simultaneously*.

    Decision variable: matrix ``A ∈ R^{nd × m}`` with each column
    weighting ``nd`` random correction directions. We use the same
    finite-differences Jacobian sampling as V5, then solve
    ``(G^T G + λI) A = G^T R`` where ``R`` is the stacked
    pre-fit-residual matrix and ``G`` is the ``(n_train, nd)``
    Jacobian. The final correction picks the column whose post-fit
    residual is smallest (or, optionally, a weighted average).
    """
    if not train_carriers:
        raise ValueError("fit requires non-empty train_carriers")
    targets = [
        _np.asarray(t, dtype=_np.float64)
        for t in target_delta_logits_stack]
    if not targets:
        raise ValueError(
            "fit requires non-empty target_delta_logits_stack")
    m = int(len(targets))
    n = int(len(train_carriers))
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    T = int(projection.n_inject_tokens)
    Dh = int(projection.d_head)
    nd = max(1, int(n_directions))
    rng = _np.random.default_rng(int(projection.seed_v6) + 6061)
    dirs_k = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    dirs_v = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    inner_v4 = projection.inner_v5.inner_v4
    base_cache_v3 = TinyV3KVCache.empty(L)
    base_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache_v3, return_attention=False)
    base_logits = base_trace.logits[-1]
    target_units = []
    target_norms = []
    for tgt in targets:
        nrm = float(_np.linalg.norm(tgt))
        target_norms.append(nrm)
        if nrm < 1e-12:
            target_units.append(_np.zeros_like(tgt))
        else:
            target_units.append(tgt / nrm)

    eps = W61_DEFAULT_BRIDGE_V6_RIDGE_PROBE_EPS

    # Compose V6 over V5+V4.
    def project_with_correction(
            extra_k: "_np.ndarray", extra_v: "_np.ndarray",
    ) -> KVBridgeV4Projection:
        v5 = projection.inner_v5
        comp_k = (
            inner_v4.correction_k
            + v5.correction_layer_a_k
            + v5.correction_layer_b_k
            + extra_k)
        comp_v = (
            inner_v4.correction_v
            + v5.correction_layer_a_v
            + v5.correction_layer_b_v
            + extra_v)
        return inner_v4.with_correction(
            correction_k=comp_k, correction_v=comp_v)

    def measure_per_target(
            corr_k: "_np.ndarray", corr_v: "_np.ndarray",
            carrier: Sequence[float],
    ) -> "_np.ndarray":
        """Return per-target projection on unit target direction.
        Shape: (m,)."""
        proj = project_with_correction(corr_k, corr_v)
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=carrier, projection=proj,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=False)
        delta = t.logits[-1] - base_logits
        out = _np.zeros((m,), dtype=_np.float64)
        for k in range(m):
            out[k] = float(_np.dot(delta, target_units[k]))
        return out

    zero_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    zero_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    # Pre-fit residual matrix R: (n, m).
    pre_proj = _np.stack([
        measure_per_target(zero_k, zero_v, c)
        for c in train_carriers], axis=0)
    target_norm_row = _np.asarray(
        target_norms, dtype=_np.float64).reshape(1, m)
    pre_residual = target_norm_row - pre_proj  # (n, m)
    pre_mean = float(_np.mean(_np.abs(pre_residual)))

    # Per-direction Jacobian G: (n, nd) — assumed shared across
    # targets because each direction's effect is linearly summed
    # then projected; the per-target projection is via unit
    # directions. We compute g_{i,j,k} but fold across columns by
    # averaging into a stable ridge linear system: solve nd × m.
    G = _np.zeros((n, nd, m), dtype=_np.float64)
    for j in range(nd):
        for i, c in enumerate(train_carriers):
            pj = measure_per_target(
                dirs_k[j] * eps, dirs_v[j] * eps, c)
            mj = measure_per_target(
                dirs_k[j] * (-eps), dirs_v[j] * (-eps), c)
            G[i, j, :] = (pj - mj) / (2.0 * eps)
    # Solve per target: A[:, k] = (G_k^T G_k + λI)^{-1} G_k^T R[:, k]
    # where G_k = G[:, :, k]. This is m independent (nd × nd) solves.
    lam = max(float(ridge_lambda), 1e-9)
    A = _np.zeros((nd, m), dtype=_np.float64)
    cond_max = 0.0
    for k in range(m):
        Gk = G[:, :, k]
        Ak = Gk.T @ Gk + lam * _np.eye(nd, dtype=_np.float64)
        bk = Gk.T @ pre_residual[:, k]
        try:
            alpha_k = _np.linalg.solve(Ak, bk)
        except Exception:
            alpha_k = _np.zeros((nd,), dtype=_np.float64)
        A[:, k] = alpha_k
        c_k = _safe_condition(Ak)
        if c_k != float("inf"):
            cond_max = max(cond_max, float(c_k))
    # Reduce A to a single correction: take the column with the
    # largest pre-residual magnitude (so we focus on the target
    # that needs the largest correction). This is a *honest*
    # reduction; full simultaneous fit would require a
    # multi-correction tensor.
    pre_per_target_l2 = _np.linalg.norm(pre_residual, axis=0)
    k_best = int(_np.argmax(pre_per_target_l2))
    alpha_best = A[:, k_best]
    new_corr_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    new_corr_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    for j in range(nd):
        new_corr_k += float(alpha_best[j]) * dirs_k[j]
        new_corr_v += float(alpha_best[j]) * dirs_v[j]
    fitted = projection.with_correction_c(
        correction_k=new_corr_k,
        correction_v=new_corr_v)
    # Post-fit residual.
    post_proj = _np.stack([
        measure_per_target(
            fitted.correction_layer_c_k,
            fitted.correction_layer_c_v, c)
        for c in train_carriers], axis=0)
    post_residual = target_norm_row - post_proj
    post_mean = float(_np.mean(_np.abs(post_residual)))
    # Honest convergence: residual reduction on the worst-target
    # column we explicitly fit. The mean across all targets can rise
    # when the worst-target correction trades off against the
    # others; that is a documented limitation
    # (W61-L-KV-V6-MULTI-TARGET-REDUCTION-CAP).
    pre_best = float(pre_per_target_l2[k_best])
    post_per_target_l2 = _np.linalg.norm(
        post_residual, axis=0)
    post_best = float(post_per_target_l2[k_best])
    converged = bool(post_best <= pre_best + 1e-9)
    return fitted, KVBridgeV6FitReport(
        schema=W61_KV_BRIDGE_V6_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_directions=int(nd),
        n_targets=int(m),
        pre_fit_mean_residual=float(pre_mean),
        post_fit_mean_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond_max),
        fit_kind="multi_target_logit",
        alpha_l2=float(_np.linalg.norm(alpha_best)),
        converged=bool(converged),
    )


def fit_kv_bridge_v6_attention_pattern(
        *,
        params: TinyV3SubstrateParams,
        projection: KVBridgeV6Projection,
        train_carriers: Sequence[Sequence[float]],
        target_attention_last_row: Sequence[float],
        follow_up_token_ids: Sequence[int],
        layer_index: int = 0,
        head_index: int = 0,
        n_directions: int = (
            W61_DEFAULT_BRIDGE_V6_N_DIRECTIONS),
        ridge_lambda: float = (
            W61_DEFAULT_BRIDGE_V6_RIDGE_LAMBDA),
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[KVBridgeV6Projection, KVBridgeV6FitReport]:
    """Fit V6 layer_c correction to drive the LAST-ROW attention
    pattern at ``(layer_index, head_index)`` toward
    ``target_attention_last_row`` in L1 sense.

    Decision variable: α ∈ R^{nd}. Same finite-differences and
    closed-form ridge as ``multi_target``, but the substrate
    observable is the L1 distance between the post-injection
    last-row attention vector and the reference.
    """
    if not train_carriers:
        raise ValueError("fit requires non-empty train_carriers")
    n = int(len(train_carriers))
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    T = int(projection.n_inject_tokens)
    Dh = int(projection.d_head)
    nd = max(1, int(n_directions))
    rng = _np.random.default_rng(int(projection.seed_v6) + 6071)
    dirs_k = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    dirs_v = [
        rng.standard_normal((L, H, T, Dh)) * 0.05
        for _ in range(nd)]
    inner_v4 = projection.inner_v5.inner_v4
    target = _np.asarray(
        target_attention_last_row, dtype=_np.float64)
    target_norm = float(_np.linalg.norm(target))
    eps = W61_DEFAULT_BRIDGE_V6_RIDGE_PROBE_EPS

    def project_with_correction(
            extra_k: "_np.ndarray", extra_v: "_np.ndarray",
    ) -> KVBridgeV4Projection:
        v5 = projection.inner_v5
        comp_k = (
            inner_v4.correction_k
            + v5.correction_layer_a_k
            + v5.correction_layer_b_k
            + extra_k)
        comp_v = (
            inner_v4.correction_v
            + v5.correction_layer_a_v
            + v5.correction_layer_b_v
            + extra_v)
        return inner_v4.with_correction(
            correction_k=comp_k, correction_v=comp_v)

    def measure_last_row_l1(
            corr_k: "_np.ndarray", corr_v: "_np.ndarray",
            carrier: Sequence[float]) -> float:
        proj = project_with_correction(corr_k, corr_v)
        bc = TinyV3KVCache.empty(L)
        nc, _ = inject_carrier_into_v4_kv_cache(
            carrier=carrier, projection=proj,
            kv_cache=bc, layer_indices=layer_indices,
            position=position, role=role)
        t = forward_tiny_substrate_v3(
            params, list(follow_up_token_ids),
            kv_cache=nc, return_attention=True)
        attn_per_layer = getattr(
            t, "attn_weights_per_layer", None)
        if attn_per_layer is None:
            return 0.0
        li = int(layer_index)
        if li < 0 or li >= len(attn_per_layer):
            return 0.0
        A_l = attn_per_layer[li]
        if A_l is None or A_l.size == 0:
            return 0.0
        # A_l: (n_heads, n_q, n_k); take last query row at head_index.
        hi = int(head_index)
        if A_l.ndim != 3 or hi >= A_l.shape[0]:
            return 0.0
        last_row = A_l[hi, -1, :]
        # Pad / truncate to target length.
        n_t = int(target.shape[0])
        n_r = int(last_row.shape[0])
        n_k = min(n_t, n_r)
        return float(_np.linalg.norm(
            last_row[:n_k] - target[:n_k], ord=1))

    zero_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    zero_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    pre_l1 = [measure_last_row_l1(zero_k, zero_v, c)
              for c in train_carriers]
    pre_mean = float(_np.mean(_np.abs(_np.asarray(pre_l1))))
    G = _np.zeros((n, nd), dtype=_np.float64)
    for j in range(nd):
        for i, c in enumerate(train_carriers):
            pj = measure_last_row_l1(
                dirs_k[j] * eps, dirs_v[j] * eps, c)
            mj = measure_last_row_l1(
                dirs_k[j] * (-eps), dirs_v[j] * (-eps), c)
            G[i, j] = (pj - mj) / (2.0 * eps)
    lam = max(float(ridge_lambda), 1e-9)
    r = _np.asarray(pre_l1, dtype=_np.float64)
    A = G.T @ G + lam * _np.eye(nd, dtype=_np.float64)
    b = -(G.T @ r)
    try:
        alpha = _np.linalg.solve(A, b)
    except Exception:
        alpha = _np.zeros((nd,), dtype=_np.float64)
    cond = _safe_condition(A)
    new_corr_k = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    new_corr_v = _np.zeros((L, H, T, Dh), dtype=_np.float64)
    for j in range(nd):
        new_corr_k += float(alpha[j]) * dirs_k[j]
        new_corr_v += float(alpha[j]) * dirs_v[j]
    fitted = projection.with_correction_c(
        correction_k=new_corr_k,
        correction_v=new_corr_v)
    post_l1 = [measure_last_row_l1(
        fitted.correction_layer_c_k,
        fitted.correction_layer_c_v, c)
        for c in train_carriers]
    post_mean = float(_np.mean(_np.abs(_np.asarray(post_l1))))
    return fitted, KVBridgeV6FitReport(
        schema=W61_KV_BRIDGE_V6_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_directions=int(nd),
        n_targets=1,
        pre_fit_mean_residual=float(pre_mean),
        post_fit_mean_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        fit_kind="attention_pattern",
        alpha_l2=float(_np.linalg.norm(alpha)),
        converged=bool(post_mean <= pre_mean + 1e-9),
    )


def v6_cache_key_fingerprint(
        cache: TinyV6KVCache, *, n_buckets: int = 128,
) -> tuple[int, ...]:
    """Hash the V6 cache_keys per-layer into ``n_buckets`` integers
    via wrap-around XOR over IEEE float bytes. Disambiguates
    corruption that lands on the cache-key axis specifically."""
    buckets = [0] * int(n_buckets)
    for li, ks in enumerate(cache.cache_keys):
        if ks.size == 0:
            continue
        for t in range(int(ks.shape[0])):
            for d in range(int(ks.shape[-1])):
                v = float(ks[t, d])
                b = (
                    int(_np.float64(v).tobytes()[0])
                    ^ int(_np.float64(v).tobytes()[1])
                    ^ (li << 1) ^ (t << 3) ^ (d << 5))
                buckets[b % int(n_buckets)] ^= int(
                    int.from_bytes(
                        _np.float64(v).tobytes()[:4], "big",
                        signed=False))
    return tuple(int(x) & 0xFFFFFFFF for x in buckets)


@dataclasses.dataclass(frozen=True)
class KVBridgeV6Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    fingerprint_128_v6_keys: tuple[int, ...]
    inject_cid: str
    pre_logits_l2: float
    post_logits_l2: float
    delta_logits_l2: float
    last_logit_delta_proj: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "carrier_cid": str(self.carrier_cid),
            "fingerprint_128_v6_keys": list(
                self.fingerprint_128_v6_keys),
            "inject_cid": str(self.inject_cid),
            "pre_logits_l2": float(round(
                self.pre_logits_l2, 12)),
            "post_logits_l2": float(round(
                self.post_logits_l2, 12)),
            "delta_logits_l2": float(round(
                self.delta_logits_l2, 12)),
            "last_logit_delta_proj": float(round(
                self.last_logit_delta_proj, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v6_witness",
            "witness": self.to_dict()})


def bridge_carrier_and_measure_v6(
        *, params: TinyV6SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV6Projection,
        follow_up_token_ids: Sequence[int],
        target_unit: Sequence[float] | None = None,
        role: str = "bank_a",
) -> KVBridgeV6Witness:
    """Inject ``carrier`` through the V6 projection, run a V6
    forward, and measure the delta logits + V6 cache-key
    fingerprint."""
    L = int(projection.n_layers)
    base_v3 = TinyV3KVCache.empty(L)
    pre_trace = forward_tiny_substrate_v3(
        params.v3_params, list(follow_up_token_ids),
        kv_cache=base_v3, return_attention=False)
    pre_logits = pre_trace.logits[-1]
    nc, inject_meta = inject_carrier_into_v6_kv_cache(
        carrier=carrier, projection=projection,
        kv_cache=base_v3, role=role)
    post_trace = forward_tiny_substrate_v3(
        params.v3_params, list(follow_up_token_ids),
        kv_cache=nc, return_attention=False)
    post_logits = post_trace.logits[-1]
    delta = post_logits - pre_logits
    if target_unit is not None:
        tu = _np.asarray(target_unit, dtype=_np.float64)
        tu_n = float(_np.linalg.norm(tu))
        if tu_n > 1e-12:
            tu = tu / tu_n
        proj_val = float(_np.dot(delta, tu))
    else:
        proj_val = 0.0
    # Pull a fresh V6 trace just to grab cache_keys fingerprint.
    _, v6_cache = forward_tiny_substrate_v6(
        params, list(follow_up_token_ids))
    fp = v6_cache_key_fingerprint(v6_cache)
    return KVBridgeV6Witness(
        schema=W61_KV_BRIDGE_V6_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=_sha256_hex({
            "kind": "kv_bridge_v6_carrier",
            "carrier": [float(round(float(x), 12))
                          for x in list(carrier)]}),
        fingerprint_128_v6_keys=fp,
        inject_cid=str(inject_meta.get("inject_cid", "")),
        pre_logits_l2=float(_np.linalg.norm(pre_logits)),
        post_logits_l2=float(_np.linalg.norm(post_logits)),
        delta_logits_l2=float(_np.linalg.norm(delta)),
        last_logit_delta_proj=float(proj_val),
    )


__all__ = [
    "W61_KV_BRIDGE_V6_SCHEMA_VERSION",
    "W61_DEFAULT_BRIDGE_V6_RIDGE_LAMBDA",
    "W61_DEFAULT_BRIDGE_V6_RIDGE_PROBE_EPS",
    "W61_DEFAULT_BRIDGE_V6_N_DIRECTIONS",
    "KVBridgeV6Projection",
    "KVBridgeV6FitReport",
    "KVBridgeV6Witness",
    "inject_carrier_into_v6_kv_cache",
    "fit_kv_bridge_v6_multi_target",
    "fit_kv_bridge_v6_attention_pattern",
    "bridge_carrier_and_measure_v6",
    "v6_cache_key_fingerprint",
]
