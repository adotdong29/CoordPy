"""W60 M3 — Hidden-State Bridge V4.

Strictly extends W59's ``coordpy.hidden_state_bridge_v3``. V3 fit a
*single global α* against a target logit-shift direction. V4 makes
the fit **multi-layer** AND **multi-head**:

* **Per-(layer, head) closed-form ridge fit** —
  ``fit_hsb_v4_per_head_target`` solves a closed-form ridge problem
  whose decision variable is the per-(layer, head) scale tensor.
  Substrate-side: each per-head probe perturbs ``per_head_scale``
  by a small ε on a single (layer, head) entry and measures the
  resulting ``Δlogit · target_unit`` projection. The Jacobian is
  ``(n_train, L*H)`` and the solve is a single
  ``(L*H, L*H)`` ridge linear system. With L=2, H=8 this is a
  16-dim feature solve — easily tractable in NumPy.
* **Recovery path** — given a hidden-state perturbation that
  drives the substrate's logits *away* from the desired direction,
  ``recover_hsb_v4_inject`` fits a *counter-perturbation* by the
  same closed-form ridge that pushes the logit shift back toward
  the target. Used by the W60 corruption path.
* **KV-vs-Hidden head-to-head harness** —
  ``compare_hidden_vs_kv_injection_v4`` runs the same carrier
  through both a hidden-state V4 fit and a KV bridge V4 fit on
  the same target logit direction, and reports per-arm residual,
  cosine alignment, and ``argmax_preserved``. The
  W60 R-125 H-bar uses this for the ``hidden_beats_kv`` /
  ``kv_beats_hidden`` falsifiable claim.

V4 strictly extends V3: with ``per_head_scale_v4 = ones`` and
the recovery path disabled, V4's ``bridge_hidden_state_and_measure_v4``
reduces to V3 byte-for-byte.

Honest scope
------------

* Multi-layer multi-head fit is still **closed-form ridge**, not
  autograd / SGD. ``W60-L-V5-NO-AUTOGRAD-CAP`` carries forward
  the W59 ridge boundary unchanged.
* The KV-vs-Hidden harness measures *which arm gets closer to a
  target logit shift on this specific carrier set* — it does NOT
  prove general superiority of one arm over the other.
* Hosted backends remain text-only at the HTTP surface.
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
        "coordpy.hidden_state_bridge_v4 requires numpy"
        ) from exc

from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    forward_with_hidden_state_injection_v2,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
    fit_hsb_v3_target_logit_shift,
    bridge_hidden_state_and_measure_v3,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    bridge_carrier_and_measure_v4,
    fit_kv_bridge_v4_correction,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)


W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v4.v1")
W60_DEFAULT_HSB4_PROJECTION_SEED: int = 60043043
W60_DEFAULT_HSB4_FIT_PROBE_EPS: float = 0.04
W60_DEFAULT_HSB4_FIT_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class HiddenStateBridgeV4Projection:
    """Wraps a V3 projection. V4 stores a *per-(layer, head)*
    correction tensor in ``per_head_scale_v4`` — V3 had a per-head
    *uniform* scale tensor of ones, V4 fits the entries directly."""
    inner_v3: HiddenStateBridgeV3Projection
    per_head_scale_v4: "_np.ndarray"   # (L, H)
    seed_v4: int

    @classmethod
    def init_from_v3(
            cls, inner: HiddenStateBridgeV3Projection,
            *, seed_v4: int = W60_DEFAULT_HSB4_PROJECTION_SEED,
    ) -> "HiddenStateBridgeV4Projection":
        # V4 inherits V3's per-head scale; V4 starts as ones so it
        # reduces to V3 on no-fit paths.
        per_head = _np.asarray(inner.per_head_scale, dtype=_np.float64).copy()
        return cls(
            inner_v3=inner,
            per_head_scale_v4=per_head,
            seed_v4=int(seed_v4),
        )

    @property
    def target_layers(self) -> tuple[int, ...]:
        return tuple(self.inner_v3.target_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v3.n_heads)

    def with_per_head_scale_v4(
            self, per_head: "_np.ndarray",
    ) -> "HiddenStateBridgeV4Projection":
        """Apply a new per-(layer, head) scale tensor to V4. Also
        propagates into V3 so the project_all_layers path picks it
        up via V3's existing per-head reshape."""
        new_per_head = _np.asarray(per_head, dtype=_np.float64).copy()
        new_inner = self.inner_v3.with_per_head_scale(new_per_head)
        return dataclasses.replace(
            self, inner_v3=new_inner,
            per_head_scale_v4=new_per_head)

    def project_all_layers(
            self, carrier: Sequence[float],
    ) -> dict[int, "_np.ndarray"]:
        """Defer to V3."""
        return self.inner_v3.project_all_layers(carrier)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION,
            "kind": "hidden_state_bridge_v4_projection",
            "inner_v3_cid": self.inner_v3.cid(),
            "per_head_scale_v4_cid": _ndarray_cid(
                self.per_head_scale_v4),
            "seed_v4": int(self.seed_v4),
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


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV4FitReport:
    schema: str
    n_train_examples: int
    n_layers_fitted: int
    n_heads_fitted: int
    pre_fit_residual: float
    post_fit_residual: float
    fit_used_closed_form: bool
    ridge_lambda: float
    condition_number: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_layers_fitted": int(self.n_layers_fitted),
            "n_heads_fitted": int(self.n_heads_fitted),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "fit_used_closed_form": bool(
                self.fit_used_closed_form),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v4_fit_report",
            "report": self.to_dict()})


def _measure_logit_proj(
        params: TinyV3SubstrateParams,
        proj: HiddenStateBridgeV4Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_unit: "_np.ndarray",
) -> float:
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    base_logits = base["last_position_logits"]
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=proj.project_all_layers(carrier))
    delta = inj["last_position_logits"] - base_logits
    return float(_np.dot(delta, target_unit))


def fit_hsb_v4_per_head_target(
        *,
        params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV4Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits: "_np.ndarray",
        token_ids: Sequence[int],
        ridge_lambda: float = (
            W60_DEFAULT_HSB4_FIT_RIDGE_LAMBDA),
        probe_eps: float = W60_DEFAULT_HSB4_FIT_PROBE_EPS,
) -> tuple[HiddenStateBridgeV4Projection,
            HiddenStateBridgeV4FitReport]:
    """Per-(layer, head) closed-form ridge fit.

    Decision variable: ``δ ∈ R^{L*H}`` representing per-(layer,
    head) ADDITIVE perturbations to the projection's
    ``per_head_scale_v4`` tensor. The substrate-side Jacobian
    G_{i,(l,h)} = ∂(Δlogit · target_unit) / ∂δ_{l,h} is estimated
    by central FD. The solve is a single closed-form ridge.
    """
    if not train_carriers:
        raise ValueError(
            "fit requires non-empty train_carriers")
    target = _np.asarray(target_delta_logits, dtype=_np.float64)
    target_norm = float(_np.linalg.norm(target))
    if target_norm < 1e-12:
        target_unit = _np.zeros_like(target)
    else:
        target_unit = target / target_norm
    L = int(projection.inner_v3.inner_v2.n_target_layers)
    H = int(projection.n_heads)
    LH = L * H
    n = int(len(train_carriers))
    eps = float(probe_eps)
    base_per_head = projection.per_head_scale_v4.copy()
    pre_proj = [
        _measure_logit_proj(
            params, projection, c, token_ids, target_unit)
        for c in train_carriers]
    pre_residual = [target_norm - p for p in pre_proj]
    pre_mean = float(_np.mean(_np.abs(_np.asarray(pre_residual))))
    G = _np.zeros((n, LH), dtype=_np.float64)
    for li in range(L):
        for hi in range(H):
            j = li * H + hi
            ph_p = base_per_head.copy()
            ph_p[li, hi] = float(ph_p[li, hi]) + eps
            proj_p = projection.with_per_head_scale_v4(ph_p)
            ph_m = base_per_head.copy()
            ph_m[li, hi] = float(ph_m[li, hi]) - eps
            proj_m = projection.with_per_head_scale_v4(ph_m)
            for i, c in enumerate(train_carriers):
                pp = _measure_logit_proj(
                    params, proj_p, c, token_ids, target_unit)
                mm = _measure_logit_proj(
                    params, proj_m, c, token_ids, target_unit)
                G[i, j] = (pp - mm) / (2.0 * eps)
    r = _np.asarray(pre_residual, dtype=_np.float64)
    lam = max(float(ridge_lambda), 1e-9)
    A = G.T @ G + lam * _np.eye(LH, dtype=_np.float64)
    b = G.T @ r
    delta = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    new_per_head = base_per_head + delta.reshape(L, H)
    fitted = projection.with_per_head_scale_v4(new_per_head)
    post_proj = [
        _measure_logit_proj(
            params, fitted, c, token_ids, target_unit)
        for c in train_carriers]
    post_residual = [target_norm - p for p in post_proj]
    post_mean = float(_np.mean(_np.abs(_np.asarray(post_residual))))
    return fitted, HiddenStateBridgeV4FitReport(
        schema=W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_layers_fitted=int(L),
        n_heads_fitted=int(H),
        pre_fit_residual=float(pre_mean),
        post_fit_residual=float(post_mean),
        fit_used_closed_form=True,
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        converged=bool(post_mean <= pre_mean + 1e-9),
    )


def recover_hsb_v4_inject(
        *,
        params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV4Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: "_np.ndarray",
        adversarial_per_head: "_np.ndarray",
        ridge_lambda: float = (
            W60_DEFAULT_HSB4_FIT_RIDGE_LAMBDA),
) -> tuple[HiddenStateBridgeV4Projection,
            HiddenStateBridgeV4FitReport]:
    """Recover from an adversarial per-(layer, head) perturbation.

    Apply ``adversarial_per_head`` to ``per_head_scale_v4``, then
    fit a counter-perturbation via the same closed-form ridge so
    the output logit shift returns toward ``target_delta_logits``.
    """
    adv = _np.asarray(adversarial_per_head, dtype=_np.float64)
    adv_proj = projection.with_per_head_scale_v4(
        projection.per_head_scale_v4 + adv)
    return fit_hsb_v4_per_head_target(
        params=params, projection=adv_proj,
        train_carriers=[list(carrier)],
        target_delta_logits=target_delta_logits,
        token_ids=token_ids,
        ridge_lambda=float(ridge_lambda))


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV4Witness:
    schema: str
    projection_cid: str
    target_layers: tuple[int, ...]
    n_heads: int
    per_head_l1: float
    last_logit_l2_perturbation: float
    target_alignment_cosine: float
    target_alignment_l2_residual: float
    fit_used_closed_form: bool
    fit_report_cid: str
    n_train_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "target_layers": list(self.target_layers),
            "n_heads": int(self.n_heads),
            "per_head_l1": float(round(self.per_head_l1, 12)),
            "last_logit_l2_perturbation": float(round(
                self.last_logit_l2_perturbation, 12)),
            "target_alignment_cosine": float(round(
                self.target_alignment_cosine, 12)),
            "target_alignment_l2_residual": float(round(
                self.target_alignment_l2_residual, 12)),
            "fit_used_closed_form": bool(
                self.fit_used_closed_form),
            "fit_report_cid": str(self.fit_report_cid),
            "n_train_examples": int(self.n_train_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v4_witness",
            "witness": self.to_dict()})


def _cosine_vec(a: "_np.ndarray", b: "_np.ndarray") -> float:
    da = float(_np.linalg.norm(a))
    db = float(_np.linalg.norm(b))
    if da < 1e-30 or db < 1e-30:
        return 0.0
    return float(_np.dot(a, b) / (da * db))


def bridge_hidden_state_and_measure_v4(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV4Projection,
        token_ids: Sequence[int],
        target_delta_logits: "_np.ndarray | None" = None,
        train_carriers: Sequence[Sequence[float]] | None = None,
        fit_report_cid: str = "no_fit",
) -> HiddenStateBridgeV4Witness:
    """V4 measurement: optionally fit per-(layer, head) scale, then
    measure last-position logit perturbation."""
    cur = projection
    fit_used = False
    if (target_delta_logits is not None
            and train_carriers is not None
            and len(train_carriers) > 0):
        cur, _ = fit_hsb_v4_per_head_target(
            params=params, projection=projection,
            train_carriers=train_carriers,
            target_delta_logits=target_delta_logits,
            token_ids=token_ids)
        fit_used = True
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=cur.project_all_layers(carrier))
    diff = (inj["last_position_logits"]
             - base["last_position_logits"])
    l2 = float(_np.linalg.norm(diff))
    target_cos = 0.0
    target_l2_res = 0.0
    if target_delta_logits is not None:
        t = _np.asarray(target_delta_logits, dtype=_np.float64)
        if t.size > diff.size:
            t = t[:diff.size]
        elif t.size < diff.size:
            t = _np.concatenate(
                [t, _np.zeros(diff.size - t.size,
                              dtype=_np.float64)])
        target_cos = _cosine_vec(diff, t)
        target_l2_res = float(_np.linalg.norm(diff - t))
    return HiddenStateBridgeV4Witness(
        schema=W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION,
        projection_cid=str(cur.cid()),
        target_layers=cur.target_layers,
        n_heads=int(cur.n_heads),
        per_head_l1=float(_np.sum(_np.abs(
            cur.per_head_scale_v4 - 1.0))),
        last_logit_l2_perturbation=float(l2),
        target_alignment_cosine=float(target_cos),
        target_alignment_l2_residual=float(target_l2_res),
        fit_used_closed_form=bool(fit_used),
        fit_report_cid=str(fit_report_cid),
        n_train_examples=int(
            len(train_carriers) if train_carriers else 0),
    )


@dataclasses.dataclass(frozen=True)
class HiddenVsKVCompareResult:
    schema: str
    target_norm: float
    hidden_l2_residual: float
    kv_l2_residual: float
    hidden_cosine: float
    kv_cosine: float
    hidden_argmax_preserved: bool
    kv_argmax_preserved: bool
    hidden_beats_kv: bool
    kv_beats_hidden: bool
    tie: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "target_norm": float(round(self.target_norm, 12)),
            "hidden_l2_residual": float(round(
                self.hidden_l2_residual, 12)),
            "kv_l2_residual": float(round(
                self.kv_l2_residual, 12)),
            "hidden_cosine": float(round(self.hidden_cosine, 12)),
            "kv_cosine": float(round(self.kv_cosine, 12)),
            "hidden_argmax_preserved": bool(
                self.hidden_argmax_preserved),
            "kv_argmax_preserved": bool(
                self.kv_argmax_preserved),
            "hidden_beats_kv": bool(self.hidden_beats_kv),
            "kv_beats_hidden": bool(self.kv_beats_hidden),
            "tie": bool(self.tie),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_vs_kv_compare_v4",
            "result": self.to_dict()})


def compare_hidden_vs_kv_injection_v4(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: "_np.ndarray",
        hsb_v4_projection: HiddenStateBridgeV4Projection,
        kv_v4_projection: KVBridgeV4Projection,
        train_carriers: Sequence[Sequence[float]] | None = None,
) -> HiddenVsKVCompareResult:
    """Run the same carrier through hidden-state V4 (per-head fit)
    and KV bridge V4 (closed-form ridge α fit) on the same target
    logit direction. Reports per-arm L2 residual, cosine alignment,
    argmax preserved, and the two falsifiable claims:

      * hidden_beats_kv: hidden_l2_residual < kv_l2_residual
      * kv_beats_hidden: kv_l2_residual < hidden_l2_residual
    """
    if train_carriers is None:
        train_carriers = [list(carrier)]
    target = _np.asarray(target_delta_logits, dtype=_np.float64)
    target_norm = float(_np.linalg.norm(target))
    if target_norm < 1e-12:
        target_unit = _np.zeros_like(target)
    else:
        target_unit = target / target_norm
    # Hidden arm.
    hsb_witness = bridge_hidden_state_and_measure_v4(
        params=params, carrier=carrier,
        projection=hsb_v4_projection, token_ids=token_ids,
        target_delta_logits=target,
        train_carriers=list(train_carriers))
    # Re-measure for argmax.
    hsb_proj_fit = hsb_v4_projection
    if train_carriers and len(train_carriers) > 0:
        hsb_proj_fit, _ = fit_hsb_v4_per_head_target(
            params=params, projection=hsb_v4_projection,
            train_carriers=list(train_carriers),
            target_delta_logits=target,
            token_ids=token_ids)
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    inj_h = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=hsb_proj_fit.project_all_layers(carrier))
    base_logits = base["last_position_logits"]
    h_logits = inj_h["last_position_logits"]
    h_diff = h_logits - base_logits
    h_argmax_preserved = (
        int(_np.argmax(base_logits))
        == int(_np.argmax(h_logits)))
    # KV arm: fit KV bridge V4 against the same target by L2 length
    # (V4's API takes target_l2 lengths). We use ``target_norm`` as
    # the L2 target so the magnitudes are comparable.
    fitted_kv, _ = fit_kv_bridge_v4_correction(
        params=params, projection=kv_v4_projection,
        train_carriers=list(train_carriers),
        train_target_l2=[target_norm] * len(train_carriers),
        follow_up_token_ids=list(token_ids))
    wkv = bridge_carrier_and_measure_v4(
        params=params, carrier=list(carrier),
        projection=fitted_kv,
        follow_up_token_ids=list(token_ids))
    # Reconstruct the KV diff vector from its measured L2 and the
    # baseline argmax, but for the per-arm test we want the actual
    # logit delta. Recompute baseline and inject:
    from .kv_bridge_v4 import inject_carrier_into_v4_kv_cache
    from .tiny_substrate_v3 import (
        TinyV3KVCache,
        forward_tiny_substrate_v3,
    )
    base_cache = TinyV3KVCache.empty(
        int(kv_v4_projection.n_layers))
    base_kv = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=base_cache, return_attention=False)
    nc, _ = inject_carrier_into_v4_kv_cache(
        carrier=list(carrier), projection=fitted_kv,
        kv_cache=base_cache)
    inj_kv = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=nc, return_attention=False)
    kv_logits = inj_kv.logits[-1]
    base_kv_logits = base_kv.logits[-1]
    kv_diff = kv_logits - base_kv_logits
    kv_argmax_preserved = (
        int(_np.argmax(base_kv_logits))
        == int(_np.argmax(kv_logits)))
    # Per-arm residuals.
    hidden_residual = float(_np.linalg.norm(h_diff - target))
    kv_residual = float(_np.linalg.norm(kv_diff - target))
    hidden_cos = _cosine_vec(h_diff, target)
    kv_cos = _cosine_vec(kv_diff, target)
    hb = bool(hidden_residual < kv_residual - 1e-9)
    kb = bool(kv_residual < hidden_residual - 1e-9)
    return HiddenVsKVCompareResult(
        schema=W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION,
        target_norm=float(target_norm),
        hidden_l2_residual=float(hidden_residual),
        kv_l2_residual=float(kv_residual),
        hidden_cosine=float(hidden_cos),
        kv_cosine=float(kv_cos),
        hidden_argmax_preserved=bool(h_argmax_preserved),
        kv_argmax_preserved=bool(kv_argmax_preserved),
        hidden_beats_kv=bool(hb),
        kv_beats_hidden=bool(kb),
        tie=bool(not hb and not kb),
    )


__all__ = [
    "W60_HIDDEN_STATE_BRIDGE_V4_SCHEMA_VERSION",
    "W60_DEFAULT_HSB4_PROJECTION_SEED",
    "W60_DEFAULT_HSB4_FIT_PROBE_EPS",
    "W60_DEFAULT_HSB4_FIT_RIDGE_LAMBDA",
    "HiddenStateBridgeV4Projection",
    "HiddenStateBridgeV4FitReport",
    "HiddenStateBridgeV4Witness",
    "HiddenVsKVCompareResult",
    "fit_hsb_v4_per_head_target",
    "recover_hsb_v4_inject",
    "bridge_hidden_state_and_measure_v4",
    "compare_hidden_vs_kv_injection_v4",
]
