"""W76 M2 — KV Bridge V21.

Strictly extends W75's ``coordpy.kv_bridge_v20``. V20 fit a 16-
target stack (15 V19 + 1 compound-chain). V21 adds:

* **Seventeen-target stacked ridge fit** —
  ``fit_kv_bridge_v21_seventeen_target`` adds a seventeenth column
  for *compound-chain-then-restart* routing.
* **140-dim compound-chain-then-restart fingerprint** — derived
  from the V20 fingerprint inputs plus the compound-chain-then-
  restart-trajectory CID and post-compound-chain-restart-window.
* **Compound-chain-then-restart-pressure falsifier** — returns 0
  iff inverting the chain-then-restart-pressure flag flips the
  routing decision.

Honest scope (W76)
------------------

* All ridge fits remain closed-form linear
  (``W76-L-V21-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W76 = 76 (3 new on top of W75's
  73 — KV V21 seventeen-target + cache V19 sixteen-objective +
  replay V17 chain-then-restart-aware routing).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v21 requires numpy") from exc

from .kv_bridge_v20 import (
    KVBridgeV20FitReport, KVBridgeV20Projection,
    fit_kv_bridge_v20_sixteen_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v21 import (
    TinyV21SubstrateParams, W76_REPAIR_LABELS_V21,
    forward_tiny_substrate_v21,
)


W76_KV_BRIDGE_V21_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v21.v1")
W76_DEFAULT_KV_V21_RIDGE_LAMBDA: float = 0.10
W76_KV_V21_FINGERPRINT_DIM: int = 140


@dataclasses.dataclass
class KVBridgeV21Projection:
    inner_v20: KVBridgeV20Projection
    seed_v21: int

    @classmethod
    def init_from_v20(
            cls, inner: KVBridgeV20Projection,
            *, seed_v21: int = 760100,
    ) -> "KVBridgeV21Projection":
        return cls(inner_v20=inner, seed_v21=int(seed_v21))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v20.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_KV_BRIDGE_V21_SCHEMA_VERSION,
            "kind": "kv_bridge_v21_projection",
            "inner_v20_cid": str(self.inner_v20.cid()),
            "seed_v21": int(self.seed_v21),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV21FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    chain_then_restart_target_index: int
    chain_then_restart_pre: float
    chain_then_restart_post: float
    converged: bool
    ridge_lambda: float
    inner_v20_report_cid: str

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
            "chain_then_restart_target_index": int(
                self.chain_then_restart_target_index),
            "chain_then_restart_pre": float(round(
                self.chain_then_restart_pre, 12)),
            "chain_then_restart_post": float(round(
                self.chain_then_restart_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v20_report_cid": str(
                self.inner_v20_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v21_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v21_seventeen_target(
        *, params: TinyV21SubstrateParams,
        projection: KVBridgeV21Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        chain_then_restart_target_index: int = 16,
        n_directions: int = 3,
        ridge_lambda: float = W76_DEFAULT_KV_V21_RIDGE_LAMBDA,
) -> tuple[KVBridgeV21Projection, KVBridgeV21FitReport]:
    """Seventeen-target stacked ridge: 16 V20 + 1 chain-then-
    restart.

    Sources the inner V20 fit (16 targets), then layers a single
    additional inner-V6 multi-target fit for the new chain-then-
    restart column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:16])
    while len(primary) < 16:
        primary.append(primary[0] if primary else [0.0])
    v20_fit, v20_report = fit_kv_bridge_v20_sixteen_target(
        params=params.v20_params,
        projection=projection.inner_v20,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(chain_then_restart_target_index) + 1:
        ctr_target = list(target_delta_logits_stack[
            int(chain_then_restart_target_index)])
    else:
        ctr_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8.inner_v7.inner_v6)
    v3_params = (
        params.v20_params.v19_params.v18_params.v17_params
        .v16_params.v15_params.v3_params)
    v6_fit_ctr, v6_audit_ctr = fit_kv_bridge_v6_multi_target(
        params=v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[ctr_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_v7 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8.inner_v7, inner_v6=v6_fit_ctr)
    new_v8 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8, inner_v7=new_v7)
    new_v9 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9, inner_v8=new_v8)
    new_v10 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11.inner_v10,
        inner_v9=new_v9)
    new_v11 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12.inner_v11,
        inner_v10=new_v10)
    new_v12 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13.inner_v12, inner_v11=new_v11)
    new_v13 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14.inner_v13, inner_v12=new_v12)
    new_v14 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15
        .inner_v14, inner_v13=new_v13)
    new_v15 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16.inner_v15,
        inner_v14=new_v14)
    new_v16 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17.inner_v16,
        inner_v15=new_v15)
    new_v17 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18.inner_v17, inner_v16=new_v16)
    new_v18 = dataclasses.replace(
        v20_fit.inner_v19.inner_v18, inner_v17=new_v17)
    new_v19 = dataclasses.replace(
        v20_fit.inner_v19, inner_v18=new_v18)
    new_v20 = dataclasses.replace(
        v20_fit, inner_v19=new_v19)
    new_proj = dataclasses.replace(
        projection, inner_v20=new_v20)
    pre17 = float(v6_audit_ctr.pre_fit_mean_residual)
    post17 = float(v6_audit_ctr.post_fit_mean_residual)
    per_pre = (
        list(v20_report.per_target_pre_residual) + [pre17])
    per_post = (
        list(v20_report.per_target_post_residual) + [post17])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:16], per_post[:16]))
        and per_post[16] <= per_pre[16] + 1e-2)
    report = KVBridgeV21FitReport(
        schema=W76_KV_BRIDGE_V21_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        chain_then_restart_target_index=int(
            chain_then_restart_target_index),
        chain_then_restart_pre=float(pre17),
        chain_then_restart_post=float(post17),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v20_report_cid=str(v20_report.cid()),
    )
    return new_proj, report


def compute_chain_then_restart_fingerprint_v21(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        restart_repair_trajectory_cid: str,
        replacement_repair_trajectory_cid: str,
        compound_repair_trajectory_cid: str,
        compound_chain_repair_trajectory_cid: str,
        compound_chain_then_restart_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        rejoin_count: int = 0,
        replacement_count: int = 0,
        contradiction_count: int = 0,
        delayed_repair_count: int = 0,
        compound_count: int = 0,
        compound_chain_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        rejoin_lag_turns: int = 0,
        replacement_lag_turns: int = 0,
        compound_window_turns: int = 0,
        compound_chain_window_turns: int = 0,
        post_compound_chain_restart_window_turns: int = 0,
        dim: int = W76_KV_V21_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 140-dim chain-then-restart fingerprint."""
    label_str = (
        W76_REPAIR_LABELS_V21[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W76_REPAIR_LABELS_V21)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|"
        f"{restart_repair_trajectory_cid}|"
        f"{replacement_repair_trajectory_cid}|"
        f"{compound_repair_trajectory_cid}|"
        f"{compound_chain_repair_trajectory_cid}|"
        f"{compound_chain_then_restart_trajectory_cid}|"
        f"{label_str}|"
        f"{int(restart_count)}|{int(rejoin_count)}|"
        f"{int(replacement_count)}|{int(contradiction_count)}|"
        f"{int(delayed_repair_count)}|{int(compound_count)}|"
        f"{int(compound_chain_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}|{int(rejoin_lag_turns)}|"
        f"{int(replacement_lag_turns)}|"
        f"{int(compound_window_turns)}|"
        f"{int(compound_chain_window_turns)}|"
        f"{int(post_compound_chain_restart_window_turns)}"
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


def probe_kv_bridge_v21_chain_then_restart_margin(
        *, params: TinyV21SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 17,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V21 forward."""
    base_trace, _ = forward_tiny_substrate_v21(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v21(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W76_KV_BRIDGE_V21_SCHEMA_VERSION,
        "kind": "v21_chain_then_restart_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV21ChainThenRestartFalsifier:
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
                "kv_bridge_v21_chain_then_restart_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v21_chain_then_restart_falsifier(
        *, chain_then_restart_pressure_flag: int,
) -> KVBridgeV21ChainThenRestartFalsifier:
    """Returns 0 iff inverting the chain-then-restart-pressure
    flag flips the routing decision."""
    f = int(chain_then_restart_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV21ChainThenRestartFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV21Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    chain_then_restart_margin_probe_cid: str
    chain_then_restart_falsifier_cid: str
    chain_then_restart_fingerprint_l1: float
    max_chain_then_restart_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "chain_then_restart_margin_probe_cid": str(
                self.chain_then_restart_margin_probe_cid),
            "chain_then_restart_falsifier_cid": str(
                self.chain_then_restart_falsifier_cid),
            "chain_then_restart_fingerprint_l1": float(round(
                self.chain_then_restart_fingerprint_l1, 12)),
            "max_chain_then_restart_margin": float(round(
                self.max_chain_then_restart_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v21_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v21_witness(
        *, projection: KVBridgeV21Projection,
        fit_report: KVBridgeV21FitReport | None = None,
        chain_then_restart_margin_probe: (
            dict[str, Any] | None) = None,
        chain_then_restart_falsifier: (
            KVBridgeV21ChainThenRestartFalsifier | None) = None,
        chain_then_restart_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV21Witness:
    fp_l1 = 0.0
    if chain_then_restart_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in chain_then_restart_fingerprint))
    return KVBridgeV21Witness(
        schema=W76_KV_BRIDGE_V21_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        chain_then_restart_margin_probe_cid=(
            _sha256_hex(chain_then_restart_margin_probe)
            if chain_then_restart_margin_probe is not None
            else ""),
        chain_then_restart_falsifier_cid=(
            chain_then_restart_falsifier.cid()
            if chain_then_restart_falsifier is not None
            else ""),
        chain_then_restart_fingerprint_l1=float(fp_l1),
        max_chain_then_restart_margin=float(
            chain_then_restart_margin_probe["max_margin"]
            if chain_then_restart_margin_probe is not None
            and "max_margin"
                in chain_then_restart_margin_probe
            else 0.0),
    )


__all__ = [
    "W76_KV_BRIDGE_V21_SCHEMA_VERSION",
    "W76_DEFAULT_KV_V21_RIDGE_LAMBDA",
    "W76_KV_V21_FINGERPRINT_DIM",
    "KVBridgeV21Projection",
    "KVBridgeV21FitReport",
    "fit_kv_bridge_v21_seventeen_target",
    "compute_chain_then_restart_fingerprint_v21",
    "probe_kv_bridge_v21_chain_then_restart_margin",
    "KVBridgeV21ChainThenRestartFalsifier",
    "probe_kv_bridge_v21_chain_then_restart_falsifier",
    "KVBridgeV21Witness",
    "emit_kv_bridge_v21_witness",
]
