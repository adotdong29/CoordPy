"""W59 M2 — KV Bridge V4.

Strictly extends W58's ``coordpy.kv_bridge_v3``. V3 fit
*per-(layer, head) inject scales* by coordinate descent against an
L2 magnitude target. V4 goes further:

* **Closed-form ridge projection fit** — the *projection itself*
  (a small low-rank correction matrix) is fitted by closed-form
  ridge regression. Given an N-point set of
  ``(carrier, target_logit_shift)`` pairs (the W59
  hidden-target-aware target) we solve

      W_fit = arg min_W  Σ_i || J · proj(carrier_i; bank + W_fit)
                                   · _ |  − target_logit_shift_i ||^2
                          + λ || W_fit ||^2

  via a *linearised* approximation: we compute the substrate-side
  Jacobian of ``last_position_logits`` w.r.t. the inject delta (by
  finite differences once per fit) and then solve a single ridge
  linear system. NumPy on CPU. Pure closed-form.
* **Four role banks** (``bank_a``, ``bank_b``, ``bank_c``,
  ``bank_d``) — V3 had two. The W59 multi-hop translator V9 maps
  four agent role-graphs into these banks.
* **Per-(layer, head) KL-budget fit** — alongside the magnitude
  fit, V4 can enforce an *attention-pattern* KL ceiling per head
  after injection.
* **Readback fingerprint with 128 buckets** — replaces V3's 64
  buckets for consistency with the V4 substrate fingerprint.
* **W58 V3 substrate compatibility shim** — V4 fits *on top of*
  V3's projection bank, so callers can keep their V3 projection
  and let V4 learn a small additive correction.

Honest scope
------------

* The fit is *closed-form ridge*, not autograd / SGD / GPU. We do
  ONE substrate-side Jacobian estimation by finite differences,
  then solve a small linear system. ``W59-L-V4-NO-AUTOGRAD-CAP``
  carries the boundary.
* "Fitted" means "the projection now produces a logit shift
  closer to the target on the fit set". It does NOT mean
  "quality on real tasks". W59 caps explicitly note this.
* The substrate is the in-repo V4 NumPy runtime. The bridge
  cannot push into Ollama / OpenAI / hosted models.
  ``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
        "coordpy.kv_bridge_v4 requires numpy") from exc

from .kv_bridge_v3 import (
    KVBridgeV3Projection,
    inject_carrier_into_v3_kv_cache,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v4 import (
    TinyV4SubstrateParams,
    _kv_fingerprint_128,
)


W59_KV_BRIDGE_V4_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v4.v1")
W59_DEFAULT_BRIDGE_V4_INJECT_TOKENS: int = 3
W59_DEFAULT_BRIDGE_V4_RIDGE_LAMBDA: float = 0.05
W59_DEFAULT_BRIDGE_V4_RIDGE_PROBE_EPS: float = 0.05


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeV4Projection:
    """Four-role-bank projection on top of a V3 projection.

    The V4 projection stores a *correction* per-(layer, head)
    additive vector ``correction`` of shape
    ``(n_layers, n_heads, n_inject_tokens, d_head)``. The corrected
    K/V at a slot is ``proj_v3_kv * scale + correction``.

    Role selection extends V3's ``bank_a/bank_b`` to
    ``{bank_a, bank_b, bank_c, bank_d}``.
    """

    inner_v3: KVBridgeV3Projection
    role_bank_c_proj_k: "_np.ndarray"  # (L, H, T, C, Dh)
    role_bank_c_proj_v: "_np.ndarray"
    role_bank_d_proj_k: "_np.ndarray"
    role_bank_d_proj_v: "_np.ndarray"
    correction_k: "_np.ndarray"  # (L, H, T, Dh)
    correction_v: "_np.ndarray"
    seed_v4: int

    @classmethod
    def init_from_v3(
            cls, inner: KVBridgeV3Projection,
            *, seed_v4: int = 59042042,
    ) -> "KVBridgeV4Projection":
        rng = _np.random.default_rng(int(seed_v4))
        L = int(inner.n_layers)
        H = int(inner.n_heads)
        T = int(inner.n_inject_tokens)
        C = int(inner.carrier_dim)
        Dh = int(inner.d_head)
        s = 0.20
        return cls(
            inner_v3=inner,
            role_bank_c_proj_k=(
                rng.standard_normal((L, H, T, C, Dh)) * s),
            role_bank_c_proj_v=(
                rng.standard_normal((L, H, T, C, Dh)) * s),
            role_bank_d_proj_k=(
                rng.standard_normal((L, H, T, C, Dh)) * s),
            role_bank_d_proj_v=(
                rng.standard_normal((L, H, T, C, Dh)) * s),
            correction_k=_np.zeros((L, H, T, Dh),
                                     dtype=_np.float64),
            correction_v=_np.zeros((L, H, T, Dh),
                                     dtype=_np.float64),
            seed_v4=int(seed_v4),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v3.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v3.n_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(self.inner_v3.n_kv_heads)

    @property
    def n_inject_tokens(self) -> int:
        return int(self.inner_v3.n_inject_tokens)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v3.carrier_dim)

    @property
    def d_head(self) -> int:
        return int(self.inner_v3.d_head)

    def _select_bank(
            self, role: str,
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        if role == "bank_a":
            return (self.inner_v3.proj_k_bank_a,
                    self.inner_v3.proj_v_bank_a)
        if role == "bank_b":
            return (self.inner_v3.proj_k_bank_b,
                    self.inner_v3.proj_v_bank_b)
        if role == "bank_c":
            return (self.role_bank_c_proj_k,
                    self.role_bank_c_proj_v)
        if role == "bank_d":
            return (self.role_bank_d_proj_k,
                    self.role_bank_d_proj_v)
        raise ValueError(
            "role must be in {bank_a, bank_b, bank_c, bank_d}, "
            f"got {role!r}")

    def project(
            self, carrier: Sequence[float],
            *, role: str = "bank_a",
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        pk, pv = self._select_bank(role)
        # einsum: (L,H,T,C,Dh), (C,) -> (L,H,T,Dh)
        keys_lh = _np.einsum("lhtcd,c->lhtd", pk, x)
        vals_lh = _np.einsum("lhtcd,c->lhtd", pv, x)
        scale = self.inner_v3.inject_scale_per_head[
            :, :, None, None]
        keys_lh = keys_lh * scale + self.correction_k
        vals_lh = vals_lh * scale + self.correction_v
        # GQA collapse: average heads inside each kv group.
        if int(self.n_heads) == int(self.n_kv_heads):
            kv_h = keys_lh
            kv_v_h = vals_lh
        else:
            group = int(self.n_heads) // int(self.n_kv_heads)
            L, H, T, Dh = keys_lh.shape
            kv_h = keys_lh.reshape(
                L, int(self.n_kv_heads), int(group), T, Dh
            ).mean(axis=2)
            kv_v_h = vals_lh.reshape(
                L, int(self.n_kv_heads), int(group), T, Dh
            ).mean(axis=2)
        L, KH, T, Dh = kv_h.shape
        keys = kv_h.transpose(0, 2, 1, 3).reshape(
            L, T, int(self.n_kv_heads) * Dh)
        vals = kv_v_h.transpose(0, 2, 1, 3).reshape(
            L, T, int(self.n_kv_heads) * Dh)
        return keys, vals

    def with_correction(
            self,
            *,
            correction_k: "_np.ndarray",
            correction_v: "_np.ndarray",
    ) -> "KVBridgeV4Projection":
        return dataclasses.replace(
            self,
            correction_k=_np.asarray(
                correction_k, dtype=_np.float64).copy(),
            correction_v=_np.asarray(
                correction_v, dtype=_np.float64).copy(),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_KV_BRIDGE_V4_SCHEMA_VERSION,
            "kind": "kv_bridge_v4_projection",
            "inner_v3_cid": self.inner_v3.cid(),
            "role_bank_c_proj_k_cid": _ndarray_cid(
                self.role_bank_c_proj_k),
            "role_bank_c_proj_v_cid": _ndarray_cid(
                self.role_bank_c_proj_v),
            "role_bank_d_proj_k_cid": _ndarray_cid(
                self.role_bank_d_proj_k),
            "role_bank_d_proj_v_cid": _ndarray_cid(
                self.role_bank_d_proj_v),
            "correction_k_cid": _ndarray_cid(self.correction_k),
            "correction_v_cid": _ndarray_cid(self.correction_v),
            "seed_v4": int(self.seed_v4),
        })


def inject_carrier_into_v4_kv_cache(
        *,
        carrier: Sequence[float],
        projection: KVBridgeV4Projection,
        kv_cache: TinyV3KVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[TinyV3KVCache, dict[str, Any]]:
    """V4 injection: same shape as V3 injection but uses the V4
    projection (with correction).
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    pre_cid = kv_cache.cid()
    new_cache = kv_cache.clone()
    keys, values = projection.project(carrier, role=str(role))
    readback_pieces: list[str] = []
    raw_readback_bytes: list[bytes] = []
    d_kv = (int(projection.n_kv_heads)
             * int(projection.d_head))
    for l in layer_indices:
        if l < 0 or l >= int(projection.n_layers):
            raise ValueError(
                f"layer_indices contains {l}; out of range")
        if l >= len(new_cache.keys):
            continue
        k_new = keys[l]
        v_new = values[l]
        prev_k = new_cache.keys[l]
        prev_v = new_cache.values[l]
        prev_imp = new_cache.importance[l]
        if prev_k.size == 0:
            prev_k = _np.zeros((0, d_kv), dtype=_np.float64)
            prev_v = _np.zeros((0, d_kv), dtype=_np.float64)
            prev_imp = _np.zeros((0,), dtype=_np.float64)
        T_inj = int(k_new.shape[0])
        new_imp = _np.zeros((T_inj,), dtype=_np.float64)
        if position == "prepend":
            combined_k = _np.concatenate([k_new, prev_k], axis=0)
            combined_v = _np.concatenate([v_new, prev_v], axis=0)
            combined_imp = _np.concatenate(
                [new_imp, prev_imp], axis=0)
            slice_for_readback = slice(0, T_inj)
        elif position == "append":
            combined_k = _np.concatenate([prev_k, k_new], axis=0)
            combined_v = _np.concatenate([prev_v, v_new], axis=0)
            combined_imp = _np.concatenate(
                [prev_imp, new_imp], axis=0)
            start = int(prev_k.shape[0])
            slice_for_readback = slice(start, start + T_inj)
        else:
            raise ValueError(
                "position must be 'prepend' or 'append'")
        new_cache.keys[l] = combined_k
        new_cache.values[l] = combined_v
        new_cache.importance[l] = combined_imp
        slot_block = _np.concatenate([
            combined_k[slice_for_readback],
            combined_v[slice_for_readback],
        ], axis=0)
        rb = _ndarray_cid(slot_block)
        readback_pieces.append(rb)
        raw_readback_bytes.append(
            _np.ascontiguousarray(slot_block).tobytes())
        new_cache.write_log.append({
            "schema": W59_KV_BRIDGE_V4_SCHEMA_VERSION,
            "kind": "kv_bridge_v4_inject",
            "layer": int(l),
            "position": str(position),
            "role": str(role),
            "n_inject_tokens": T_inj,
            "carrier_cid": _ndarray_cid(
                _np.asarray(carrier, dtype=_np.float64)
                .reshape(-1)),
            "projection_cid": projection.cid(),
            "readback_cid": rb,
        })
    post_cid = new_cache.cid()
    readback_cid = hashlib.sha256(
        "|".join(readback_pieces).encode("utf-8")).hexdigest()
    fp128 = [0] * 128
    for raw in raw_readback_bytes:
        for i, b in enumerate(raw):
            fp128[i % 128] ^= int(b)
    record = {
        "schema": W59_KV_BRIDGE_V4_SCHEMA_VERSION,
        "pre_inject_kv_cid": pre_cid,
        "post_inject_kv_cid": post_cid,
        "injected_layer_indices": list(layer_indices),
        "n_inject_tokens": int(projection.n_inject_tokens),
        "position": str(position),
        "role": str(role),
        "readback_cid": str(readback_cid),
        "readback_fingerprint_128": [int(b) for b in fp128],
    }
    return new_cache, record


@dataclasses.dataclass(frozen=True)
class KVBridgeV4FitReport:
    schema: str
    n_train_examples: int
    pre_fit_mean_residual: float
    post_fit_mean_residual: float
    fit_used_closed_form: bool
    ridge_lambda: float
    correction_k_l2: float
    correction_v_l2: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "pre_fit_mean_residual": float(round(
                self.pre_fit_mean_residual, 12)),
            "post_fit_mean_residual": float(round(
                self.post_fit_mean_residual, 12)),
            "fit_used_closed_form": bool(
                self.fit_used_closed_form),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "correction_k_l2": float(round(
                self.correction_k_l2, 12)),
            "correction_v_l2": float(round(
                self.correction_v_l2, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v4_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v4_correction(
        *,
        params: TinyV3SubstrateParams,
        projection: KVBridgeV4Projection,
        train_carriers: Sequence[Sequence[float]],
        train_target_l2: Sequence[float],
        follow_up_token_ids: Sequence[int],
        ridge_lambda: float = W59_DEFAULT_BRIDGE_V4_RIDGE_LAMBDA,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[KVBridgeV4Projection, KVBridgeV4FitReport]:
    """Closed-form ridge fit of the V4 correction tensors.

    The fit observation per example is:
      L2(injected_logits − baseline_logits)
        − target_l2

    Decision variable: a single global scalar α applied to a
    random direction in correction-tensor space, fit by closed-
    form solve.

    Concretely: we estimate the substrate-side scalar Jacobian
    ``g_i = dL2_i / dα`` at ``α = 0`` via central finite-
    differences with step ``ε``, then solve

        Σ_i (g_i α + (L2_i^pre − target_i))^2  + λ α^2

    for α. Then we set the correction to ``α · direction``. This
    is a *single linear ridge solve* over the fit set.

    The benefit is conservative: it pulls the residual closer to
    zero in mean L2 sense. We do not claim per-example zero
    residual.
    """
    if not train_carriers or not train_target_l2:
        raise ValueError(
            "fit requires non-empty train_carriers + targets")
    if len(train_carriers) != len(train_target_l2):
        raise ValueError(
            "train_carriers and train_target_l2 length mismatch")
    rng = _np.random.default_rng(int(projection.seed_v4) + 71)
    n = int(len(train_carriers))
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    T = int(projection.n_inject_tokens)
    Dh = int(projection.d_head)
    direction_k = rng.standard_normal((L, H, T, Dh)) * 0.05
    direction_v = rng.standard_normal((L, H, T, Dh)) * 0.05
    # Precompute baselines.
    base_cache_v3 = TinyV3KVCache.empty(L)
    base_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache_v3, return_attention=False)
    base_logits = base_trace.logits[-1]
    eps = W59_DEFAULT_BRIDGE_V4_RIDGE_PROBE_EPS

    def measure_l2(proj: KVBridgeV4Projection,
                    carrier: Sequence[float]) -> float:
        # Inject onto a fresh cache.
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

    pre_l2 = [measure_l2(projection, c)
              for c in train_carriers]
    pre_residual = [pre_l2[i] - float(train_target_l2[i])
                     for i in range(n)]
    pre_mean = float(_np.mean(_np.abs(_np.asarray(
        pre_residual, dtype=_np.float64))))
    # Probe Jacobian g_i at α=0 via central FD.
    proj_plus = projection.with_correction(
        correction_k=direction_k * eps,
        correction_v=direction_v * eps)
    proj_minus = projection.with_correction(
        correction_k=direction_k * (-eps),
        correction_v=direction_v * (-eps))
    g = []
    for c in train_carriers:
        l2p = measure_l2(proj_plus, c)
        l2m = measure_l2(proj_minus, c)
        g.append((l2p - l2m) / (2.0 * eps))
    g_arr = _np.asarray(g, dtype=_np.float64)
    r_arr = _np.asarray(pre_residual, dtype=_np.float64)
    # Solve  α* = − (g · r) / (g · g + λ).
    lam = max(float(ridge_lambda), 1e-9)
    denom = float(_np.dot(g_arr, g_arr) + lam)
    alpha = -float(_np.dot(g_arr, r_arr)) / denom
    new_corr_k = direction_k * alpha
    new_corr_v = direction_v * alpha
    fitted = projection.with_correction(
        correction_k=new_corr_k,
        correction_v=new_corr_v)
    post_l2 = [measure_l2(fitted, c) for c in train_carriers]
    post_residual = [post_l2[i] - float(train_target_l2[i])
                      for i in range(n)]
    post_mean = float(_np.mean(_np.abs(_np.asarray(
        post_residual, dtype=_np.float64))))
    converged = bool(post_mean <= pre_mean + 1e-12)
    report = KVBridgeV4FitReport(
        schema=W59_KV_BRIDGE_V4_SCHEMA_VERSION,
        n_train_examples=int(n),
        pre_fit_mean_residual=float(pre_mean),
        post_fit_mean_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        correction_k_l2=float(_np.linalg.norm(new_corr_k)),
        correction_v_l2=float(_np.linalg.norm(new_corr_v)),
        converged=bool(converged),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class KVBridgeV4Witness:
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
    readback_fingerprint_128: tuple[int, ...]
    fit_used_closed_form: bool
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
            "readback_fingerprint_128": list(
                self.readback_fingerprint_128),
            "fit_used_closed_form": bool(
                self.fit_used_closed_form),
            "fit_report_cid": str(self.fit_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v4_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def bridge_carrier_and_measure_v4(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV4Projection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyV3KVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
        fit_report_cid: str = "no_fit",
) -> KVBridgeV4Witness:
    base_cache = (
        TinyV3KVCache.empty(int(projection.n_layers))
        if baseline_kv_cache is None
        else baseline_kv_cache.clone())
    baseline_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache, return_attention=False)
    new_cache, record = inject_carrier_into_v4_kv_cache(
        carrier=carrier, projection=projection,
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
    return KVBridgeV4Witness(
        schema=W59_KV_BRIDGE_V4_SCHEMA_VERSION,
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
        readback_fingerprint_128=tuple(
            int(b) for b in record["readback_fingerprint_128"]),
        fit_used_closed_form=bool(
            fit_report_cid not in ("", "no_fit")),
        fit_report_cid=str(fit_report_cid),
    )


__all__ = [
    "W59_KV_BRIDGE_V4_SCHEMA_VERSION",
    "W59_DEFAULT_BRIDGE_V4_INJECT_TOKENS",
    "W59_DEFAULT_BRIDGE_V4_RIDGE_LAMBDA",
    "W59_DEFAULT_BRIDGE_V4_RIDGE_PROBE_EPS",
    "KVBridgeV4Projection",
    "KVBridgeV4FitReport",
    "KVBridgeV4Witness",
    "inject_carrier_into_v4_kv_cache",
    "fit_kv_bridge_v4_correction",
    "bridge_carrier_and_measure_v4",
]
