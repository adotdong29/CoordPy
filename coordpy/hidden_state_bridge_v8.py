"""W64 M3 — Hidden-State Bridge V8.

Strictly extends W63's ``coordpy.hidden_state_bridge_v7``. V7 fit a
4-target stack with hidden-wins target. V8 adds:

* **Five-target stacked ridge fit** —
  ``fit_hsb_v8_five_target`` is V7's four-target fit with an added
  *hidden-wins-primary target* column. The fifth target represents
  a δ that requires the hidden bridge to be the *primary* path
  (i.e., the hidden injection must dominate AND set the V9
  hidden_wins_primary tensor).
* **V9 hidden-state-trust coupling** —
  ``write_hsb_v8_into_v9_hidden_state_trust`` records per-(layer,
  head) hidden-state decisions into the V9 substrate's
  hidden_state_trust_ledger.
* **Recovery audit V4** — ``recover_hsb_v8_inject_v4`` is V7's
  recovery path with a *three-stage* recovery margin (post-V6 +
  post-V5 + post-V4 basin width).
* **Hidden-wins primary margin** —
  ``compute_hsb_v8_hidden_wins_primary_margin`` returns a positive
  scalar when the hidden injection's residual is strictly less
  than BOTH the KV residual AND the prefix-replay residual.

Honest scope (W64)
------------------

* All fits delegate to V6/V5's closed-form ridge. No new gradient
  descent. ``W64-L-V8-HSB-NO-AUTOGRAD-CAP`` documents.
* The hidden-wins-primary target is *constructed* — engineered
  such that only the hidden bridge can win as primary. Not a
  measured in-the-wild target.
* The three-stage recovery audit measures basin widths only; it
  does NOT prove robust hidden recovery on real models.
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
        "coordpy.hidden_state_bridge_v8 requires numpy"
        ) from exc

from .hidden_state_bridge_v5 import (
    fit_hsb_v5_multi_target,
    recover_hsb_v5_inject,
)
from .hidden_state_bridge_v7 import (
    HiddenStateBridgeV7Projection,
    HiddenStateBridgeV7FitReport,
    fit_hsb_v7_four_target,
    recover_hsb_v7_inject_v3,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v9 import (
    TinyV9KVCache,
    record_hidden_state_trust_decision_v9,
)


W64_HSB_V8_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v8.v1")
W64_DEFAULT_HSB_V8_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV8Projection:
    inner_v7: HiddenStateBridgeV7Projection
    seed_v8: int

    @classmethod
    def init_from_v7(
            cls, inner: HiddenStateBridgeV7Projection,
            *, seed_v8: int = 640800,
    ) -> "HiddenStateBridgeV8Projection":
        return cls(inner_v7=inner, seed_v8=int(seed_v8))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v7.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v7.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v7.n_heads)

    @property
    def n_positions(self) -> int:
        return int(self.inner_v7.n_positions)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_HSB_V8_SCHEMA_VERSION,
            "kind": "hsb_v8_projection",
            "inner_v7_cid": self.inner_v7.cid(),
            "seed_v8": int(self.seed_v8),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV8FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    hidden_wins_primary_target_index: int
    hidden_wins_primary_pre: float
    hidden_wins_primary_post: float
    worst_index: int
    worst_pre: float
    worst_post: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_targets": int(self.n_targets),
            "per_target_pre_residual": [
                float(round(float(x), 12))
                for x in self.per_target_pre_residual],
            "per_target_post_residual": [
                float(round(float(x), 12))
                for x in self.per_target_post_residual],
            "hidden_wins_primary_target_index": int(
                self.hidden_wins_primary_target_index),
            "hidden_wins_primary_pre": float(round(
                self.hidden_wins_primary_pre, 12)),
            "hidden_wins_primary_post": float(round(
                self.hidden_wins_primary_post, 12)),
            "worst_index": int(self.worst_index),
            "worst_pre": float(round(self.worst_pre, 12)),
            "worst_post": float(round(self.worst_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v8_fit_report",
            "report": self.to_dict()})


def fit_hsb_v8_five_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV8Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        hidden_wins_primary_target_index: int = 4,
        ridge_lambda: float = W64_DEFAULT_HSB_V8_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV8Projection,
            HiddenStateBridgeV8FitReport]:
    """Five-target stacked ridge fit. Delegates to V7 for the first
    four targets, then a second V5 fit on the hidden-wins-primary
    target."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide ≥ 1 target")
    primary = list(target_delta_logits_stack[:4])
    while len(primary) < 4:
        primary.append(primary[0] if primary
                       else [0.0] * 1)
    fitted_v7, v7_report = fit_hsb_v7_four_target(
        params=params, projection=projection.inner_v7,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(hidden_wins_primary_target_index) + 1:
        hwp_target = list(
            target_delta_logits_stack[
                int(hidden_wins_primary_target_index)])
    else:
        hwp_target = list(target_delta_logits_stack[-1])
    fitted_v5, v5_report = fit_hsb_v5_multi_target(
        params=params, projection=fitted_v7.inner_v6.inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[hwp_target],
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        fitted_v7.inner_v6, inner_v5=fitted_v5)
    new_inner_v7 = dataclasses.replace(
        fitted_v7, inner_v6=new_inner_v6)
    new_proj = dataclasses.replace(
        projection, inner_v7=new_inner_v7)
    per_pre_list = list(v7_report.per_target_pre_residual)[
        :4] + [float(v5_report.pre_fit_residual)]
    per_post_list = list(v7_report.per_target_post_residual)[
        :4] + [float(v5_report.post_fit_residual)]
    worst_idx = int(_np.argmax(per_pre_list))
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre_list, per_post_list)))
    report = HiddenStateBridgeV8FitReport(
        schema=W64_HSB_V8_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre_list),
        per_target_post_residual=tuple(per_post_list),
        hidden_wins_primary_target_index=int(
            hidden_wins_primary_target_index),
        hidden_wins_primary_pre=float(v5_report.pre_fit_residual),
        hidden_wins_primary_post=float(
            v5_report.post_fit_residual),
        worst_index=int(worst_idx),
        worst_pre=float(per_pre_list[worst_idx]),
        worst_post=float(per_post_list[worst_idx]),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def recover_hsb_v8_inject_v4(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV8Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: Sequence[float],
        adversarial_per_head_pos: "_np.ndarray",
) -> tuple[HiddenStateBridgeV8Projection, dict[str, Any]]:
    """Three-stage recovery audit. Stage 1 = V7 recovery (which
    runs V6 + V5 inner). Stage 2 = re-fit the V5 inner on the
    recovered carrier. Stage 3 = measure post-recovery basin
    width over a wider perturbation."""
    new_v7, audit_v7 = recover_hsb_v7_inject_v3(
        params=params, projection=projection.inner_v7,
        carrier=list(carrier), token_ids=list(token_ids),
        target_delta_logits=list(target_delta_logits),
        adversarial_per_head_pos=adversarial_per_head_pos)
    rec_v5, v5_rep3 = recover_hsb_v5_inject(
        params=params,
        projection=new_v7.inner_v6.inner_v5,
        carrier=list(carrier), token_ids=list(token_ids),
        target_delta_logits=list(target_delta_logits),
        adversarial_per_head_pos=adversarial_per_head_pos * 1.5)
    new_inner_v6 = dataclasses.replace(
        new_v7.inner_v6, inner_v5=rec_v5)
    new_inner_v7 = dataclasses.replace(
        new_v7, inner_v6=new_inner_v6)
    new_proj = dataclasses.replace(
        projection, inner_v7=new_inner_v7)
    basin_width_v4 = float(
        v5_rep3.pre_fit_residual - v5_rep3.post_fit_residual)
    audit = {
        "schema": W64_HSB_V8_SCHEMA_VERSION,
        "kind": "hsb_v8_recovery_audit_v4",
        "stage1_2_audit": dict(audit_v7),
        "stage3_basin_width": float(round(basin_width_v4, 12)),
        "stage3_pre": float(round(
            v5_rep3.pre_fit_residual, 12)),
        "stage3_post": float(round(
            v5_rep3.post_fit_residual, 12)),
        "three_stage_recovered": bool(
            audit_v7.get("two_stage_recovered", False)
            and basin_width_v4 >= -1e-9),
    }
    return new_proj, audit


def write_hsb_v8_into_v9_hidden_state_trust(
        *, decisions_per_layer_head: dict[tuple[int, int], str],
        v9_cache: TinyV9KVCache,
        trust_ema: float = 0.5,
) -> dict[str, Any]:
    """Apply hidden-state decisions to the V9 hidden-state-trust
    ledger."""
    n_writes = 0
    for (li, hi), dec in decisions_per_layer_head.items():
        record_hidden_state_trust_decision_v9(
            v9_cache,
            layer_index=int(li), head_index=int(hi),
            decision=str(dec),
            trust_ema=float(trust_ema))
        n_writes += 1
    return {
        "schema": W64_HSB_V8_SCHEMA_VERSION,
        "kind": "hsb_v8_hidden_state_trust_write",
        "n_writes": int(n_writes),
    }


def compute_hsb_v8_hidden_wins_primary_margin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
) -> float:
    """Returns min(kv, prefix) - hidden. Positive ⇒ hidden wins
    primary; negative ⇒ either kv or prefix beats hidden."""
    challenger = float(min(
        float(kv_residual_l2), float(prefix_residual_l2)))
    return float(challenger) - float(hidden_residual_l2)


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV8Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    hidden_wins_primary_margin: float
    hidden_state_trust_ledger_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "hidden_wins_primary_margin": float(round(
                self.hidden_wins_primary_margin, 12)),
            "hidden_state_trust_ledger_l1": float(round(
                self.hidden_state_trust_ledger_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v8_witness",
            "witness": self.to_dict()})


def emit_hsb_v8_witness(
        *, projection: HiddenStateBridgeV8Projection,
        fit_report: HiddenStateBridgeV8FitReport | None = None,
        hidden_wins_primary_margin: float = 0.0,
        hidden_state_trust_ledger_l1: float = 0.0,
) -> HiddenStateBridgeV8Witness:
    return HiddenStateBridgeV8Witness(
        schema=W64_HSB_V8_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        hidden_wins_primary_margin=float(
            hidden_wins_primary_margin),
        hidden_state_trust_ledger_l1=float(
            hidden_state_trust_ledger_l1),
    )


__all__ = [
    "W64_HSB_V8_SCHEMA_VERSION",
    "W64_DEFAULT_HSB_V8_RIDGE_LAMBDA",
    "HiddenStateBridgeV8Projection",
    "HiddenStateBridgeV8FitReport",
    "fit_hsb_v8_five_target",
    "recover_hsb_v8_inject_v4",
    "write_hsb_v8_into_v9_hidden_state_trust",
    "compute_hsb_v8_hidden_wins_primary_margin",
    "HiddenStateBridgeV8Witness",
    "emit_hsb_v8_witness",
]
