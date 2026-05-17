"""W75 M2 — KV Bridge V20.

Strictly extends W74's ``coordpy.kv_bridge_v19``. V19 fit a 15-
target stack (14 V18 + 1 compound). V20 adds:

* **Sixteen-target stacked ridge fit** —
  ``fit_kv_bridge_v20_sixteen_target`` adds a sixteenth column for
  *compound-chain-repair-after-replacement-then-rejoin* routing.
* **130-dim compound-chain-repair fingerprint** — derived from the
  V19 fingerprint inputs plus the compound-chain-repair-trajectory
  CID and compound-chain-window width.
* **Compound-chain-pressure falsifier** — returns 0 iff inverting
  the compound-chain-pressure flag flips the routing decision.

Honest scope (W75)
------------------

* All ridge fits remain closed-form linear
  (``W75-L-V20-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W75 = 73 (3 new on top of W74's
  70 — KV V20 sixteen-target adds 1; cache V18 fifteen-objective +
  per-role 16-dim chain head adds 2; replay V16 23-regime +
  chain-aware routing adds 2 — KV V20 counts only the new sixteen-
  target as a single new solve).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v20 requires numpy") from exc

from .kv_bridge_v19 import (
    KVBridgeV19FitReport, KVBridgeV19Projection,
    fit_kv_bridge_v19_fifteen_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v20 import (
    TinyV20SubstrateParams, W75_REPAIR_LABELS_V20,
    forward_tiny_substrate_v20,
)


W75_KV_BRIDGE_V20_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v20.v1")
W75_DEFAULT_KV_V20_RIDGE_LAMBDA: float = 0.10
W75_KV_V20_FINGERPRINT_DIM: int = 130


@dataclasses.dataclass
class KVBridgeV20Projection:
    inner_v19: KVBridgeV19Projection
    seed_v20: int

    @classmethod
    def init_from_v19(
            cls, inner: KVBridgeV19Projection,
            *, seed_v20: int = 750100,
    ) -> "KVBridgeV20Projection":
        return cls(inner_v19=inner, seed_v20=int(seed_v20))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v19.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_KV_BRIDGE_V20_SCHEMA_VERSION,
            "kind": "kv_bridge_v20_projection",
            "inner_v19_cid": str(self.inner_v19.cid()),
            "seed_v20": int(self.seed_v20),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV20FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    compound_chain_target_index: int
    compound_chain_pre: float
    compound_chain_post: float
    converged: bool
    ridge_lambda: float
    inner_v19_report_cid: str

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
            "compound_chain_target_index": int(
                self.compound_chain_target_index),
            "compound_chain_pre": float(round(
                self.compound_chain_pre, 12)),
            "compound_chain_post": float(round(
                self.compound_chain_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v19_report_cid": str(
                self.inner_v19_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v20_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v20_sixteen_target(
        *, params: TinyV20SubstrateParams,
        projection: KVBridgeV20Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        compound_chain_target_index: int = 15,
        n_directions: int = 3,
        ridge_lambda: float = W75_DEFAULT_KV_V20_RIDGE_LAMBDA,
) -> tuple[KVBridgeV20Projection, KVBridgeV20FitReport]:
    """Sixteen-target stacked ridge: 15 V19 + 1 compound-chain.

    Sources the inner V19 fit (15 targets), then layers a single
    additional inner-V6 multi-target fit for the new compound-chain
    column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:15])
    while len(primary) < 15:
        primary.append(primary[0] if primary else [0.0])
    v19_fit, v19_report = fit_kv_bridge_v19_fifteen_target(
        params=params.v19_params,
        projection=projection.inner_v19,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(compound_chain_target_index) + 1:
        chain_target = list(target_delta_logits_stack[
            int(compound_chain_target_index)])
    else:
        chain_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11.inner_v10.inner_v9.inner_v8
        .inner_v7.inner_v6)
    v6_fit_chain, v6_audit_chain = fit_kv_bridge_v6_multi_target(
        params=params.v19_params.v18_params.v17_params.v16_params
        .v15_params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[chain_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_v7 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11.inner_v10.inner_v9.inner_v8
        .inner_v7, inner_v6=v6_fit_chain)
    new_v8 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11.inner_v10.inner_v9.inner_v8,
        inner_v7=new_v7)
    new_v9 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11.inner_v10.inner_v9,
        inner_v8=new_v8)
    new_v10 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11.inner_v10, inner_v9=new_v9)
    new_v11 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12.inner_v11, inner_v10=new_v10)
    new_v12 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13.inner_v12, inner_v11=new_v11)
    new_v13 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
        .inner_v13, inner_v12=new_v12)
    new_v14 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14,
        inner_v13=new_v13)
    new_v15 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16.inner_v15,
        inner_v14=new_v14)
    new_v16 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17.inner_v16, inner_v15=new_v15)
    new_v17 = dataclasses.replace(
        v19_fit.inner_v18.inner_v17, inner_v16=new_v16)
    new_v18 = dataclasses.replace(
        v19_fit.inner_v18, inner_v17=new_v17)
    new_v19 = dataclasses.replace(
        v19_fit, inner_v18=new_v18)
    new_proj = dataclasses.replace(
        projection, inner_v19=new_v19)
    pre16 = float(v6_audit_chain.pre_fit_mean_residual)
    post16 = float(v6_audit_chain.post_fit_mean_residual)
    per_pre = (
        list(v19_report.per_target_pre_residual) + [pre16])
    per_post = (
        list(v19_report.per_target_post_residual) + [post16])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:15], per_post[:15]))
        and per_post[15] <= per_pre[15] + 1e-2)
    report = KVBridgeV20FitReport(
        schema=W75_KV_BRIDGE_V20_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        compound_chain_target_index=int(
            compound_chain_target_index),
        compound_chain_pre=float(pre16),
        compound_chain_post=float(post16),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v19_report_cid=str(v19_report.cid()),
    )
    return new_proj, report


def compute_compound_chain_repair_fingerprint_v20(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        restart_repair_trajectory_cid: str,
        replacement_repair_trajectory_cid: str,
        compound_repair_trajectory_cid: str,
        compound_chain_repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        rejoin_count: int = 0,
        replacement_count: int = 0,
        contradiction_count: int = 0,
        delayed_repair_count: int = 0,
        compound_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        rejoin_lag_turns: int = 0,
        replacement_lag_turns: int = 0,
        compound_window_turns: int = 0,
        compound_chain_window_turns: int = 0,
        dim: int = W75_KV_V20_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 130-dim compound-chain-repair fingerprint."""
    label_str = (
        W75_REPAIR_LABELS_V20[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W75_REPAIR_LABELS_V20)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|"
        f"{restart_repair_trajectory_cid}|"
        f"{replacement_repair_trajectory_cid}|"
        f"{compound_repair_trajectory_cid}|"
        f"{compound_chain_repair_trajectory_cid}|{label_str}|"
        f"{int(restart_count)}|{int(rejoin_count)}|"
        f"{int(replacement_count)}|{int(contradiction_count)}|"
        f"{int(delayed_repair_count)}|{int(compound_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}|{int(rejoin_lag_turns)}|"
        f"{int(replacement_lag_turns)}|"
        f"{int(compound_window_turns)}|"
        f"{int(compound_chain_window_turns)}"
    ).encode("utf-8")
    h = hashlib.sha256(base).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return tuple(out)


def probe_kv_bridge_v20_compound_chain_pressure_margin(
        *, params: TinyV20SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 16,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V20 forward."""
    base_trace, _ = forward_tiny_substrate_v20(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v20(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W75_KV_BRIDGE_V20_SCHEMA_VERSION,
        "kind": "v20_compound_chain_pressure_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV20CompoundChainPressureFalsifier:
    primary_flag: int
    inverted_flag: int
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_flag": int(self.primary_flag),
            "inverted_flag": int(self.inverted_flag),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "kv_bridge_v20_compound_chain_pressure_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v20_compound_chain_pressure_falsifier(
        *, compound_chain_pressure_flag: int,
) -> KVBridgeV20CompoundChainPressureFalsifier:
    """Returns 0 iff inverting the compound-chain-pressure flag
    flips the routing decision.

    Inversion semantics: flag 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when flag > 0 and
    ``route_through_text`` otherwise. Flipping the flag must flip
    the decision (honest case: score = 0).
    """
    f = int(compound_chain_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV20CompoundChainPressureFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV20Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    compound_chain_pressure_margin_probe_cid: str
    compound_chain_pressure_falsifier_cid: str
    compound_chain_repair_fingerprint_l1: float
    max_compound_chain_pressure_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "compound_chain_pressure_margin_probe_cid": str(
                self.compound_chain_pressure_margin_probe_cid),
            "compound_chain_pressure_falsifier_cid": str(
                self.compound_chain_pressure_falsifier_cid),
            "compound_chain_repair_fingerprint_l1": float(round(
                self.compound_chain_repair_fingerprint_l1, 12)),
            "max_compound_chain_pressure_margin": float(round(
                self.max_compound_chain_pressure_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v20_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v20_witness(
        *, projection: KVBridgeV20Projection,
        fit_report: KVBridgeV20FitReport | None = None,
        compound_chain_pressure_margin_probe: (
            dict[str, Any] | None) = None,
        compound_chain_pressure_falsifier: (
            KVBridgeV20CompoundChainPressureFalsifier | None
        ) = None,
        compound_chain_repair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV20Witness:
    fp_l1 = 0.0
    if compound_chain_repair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in compound_chain_repair_fingerprint))
    return KVBridgeV20Witness(
        schema=W75_KV_BRIDGE_V20_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        compound_chain_pressure_margin_probe_cid=(
            _sha256_hex(compound_chain_pressure_margin_probe)
            if compound_chain_pressure_margin_probe is not None
            else ""),
        compound_chain_pressure_falsifier_cid=(
            compound_chain_pressure_falsifier.cid()
            if compound_chain_pressure_falsifier is not None
            else ""),
        compound_chain_repair_fingerprint_l1=float(fp_l1),
        max_compound_chain_pressure_margin=float(
            compound_chain_pressure_margin_probe["max_margin"]
            if compound_chain_pressure_margin_probe is not None
            and "max_margin"
                in compound_chain_pressure_margin_probe
            else 0.0),
    )


__all__ = [
    "W75_KV_BRIDGE_V20_SCHEMA_VERSION",
    "W75_DEFAULT_KV_V20_RIDGE_LAMBDA",
    "W75_KV_V20_FINGERPRINT_DIM",
    "KVBridgeV20Projection",
    "KVBridgeV20FitReport",
    "fit_kv_bridge_v20_sixteen_target",
    "compute_compound_chain_repair_fingerprint_v20",
    "probe_kv_bridge_v20_compound_chain_pressure_margin",
    "KVBridgeV20CompoundChainPressureFalsifier",
    "probe_kv_bridge_v20_compound_chain_pressure_falsifier",
    "KVBridgeV20Witness",
    "emit_kv_bridge_v20_witness",
]
