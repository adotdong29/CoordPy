"""W72 R-172 benchmark family — Handoff V4 + Falsifier + Limitation.

The W72 falsifier-and-limitation-reproduction family + Plane A↔B
rejoin-aware handoff family.

H760..H773 cell families (14 H-bars):

* H760   Handoff V4 envelope is content-addressed
* H761   Handoff V4 text-only routes to plane A
* H762   Handoff V4 substrate-only routes to plane B
* H763   Handoff V4 rejoin_pressure ≥ floor + substrate_trust ≥
         floor promotes to real_substrate_only with
         rejoin_alignment = 1.0
* H764   Handoff V4 restart_repair_trajectory + rejoin_lag > 0
         falls back to delayed_rejoin_after_restart_fallback
* H765   Handoff V4 cross-plane savings ≥ 78 %
* H766   Handoff V4 delayed-rejoin falsifier honest=0, dishonest=1
* H767   Boundary V5 falsifier honest=0 / dishonest=1
* H768   KV V17 rejoin-pressure falsifier honest = 0
* H769   MASC V8 compound regime
         (delayed_rejoin_after_restart_under_budget) ≥ 50 %
         strict-beat
* H770   W72 substrate is in-repo NumPy (limitation reproduction)
* H771   W72 hosted control plane V5 does NOT pierce wall
         (limitation reproduction; 28 blocked-axes)
* H772   No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H773   Frontier substrate access still blocked (V5 frontier-
         blocked axes set non-empty and unchanged from W70)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy import SDK_VERSION, __version__
from coordpy.hosted_real_handoff_coordinator import HandoffRequest
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4, HostedRealHandoffCoordinatorV4,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
    hosted_real_handoff_v4_rejoin_aware_savings,
    probe_hosted_real_handoff_v4_delayed_rejoin_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v5 import (
    W72_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v5,
    probe_hosted_real_substrate_boundary_v5_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.kv_bridge_v17 import (
    probe_kv_bridge_v17_rejoin_pressure_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v8 import (
    MultiAgentSubstrateCoordinatorV8,
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
from coordpy.tiny_substrate_v17 import (
    W72_DEFAULT_V17_N_LAYERS, W72_TINY_V17_VOCAB_SIZE,
)


R172_SCHEMA_VERSION: str = "coordpy.r172_benchmark.v1"


def _req_v4(*, rc: str, rejoin_pressure: float = 0.0,
            restart_repair_trajectory_cid: str = "",
            rejoin_lag_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            restart_pressure: float = 0.0,
            dominant_repair_label: int = 0,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV4:
    return HandoffRequestV4(
        inner_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=str(rc),
                    needs_text_only=bool(needs_text_only),
                    needs_substrate_state_access=bool(
                        needs_substrate_state_access)),
                visible_token_budget=int(visible_token_budget),
                baseline_token_cost=int(baseline_token_cost),
                dominant_repair_label=int(
                    dominant_repair_label)),
            restart_pressure=float(restart_pressure),
            delayed_repair_trajectory_cid="",
            delay_turns=0,
            expected_substrate_trust=0.7),
        rejoin_pressure=float(rejoin_pressure),
        restart_repair_trajectory_cid=str(
            restart_repair_trajectory_cid),
        rejoin_lag_turns=int(rejoin_lag_turns),
        expected_substrate_trust_v4=0.7)


def family_h760_handoff_v4_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinatorV4()
    c2 = HostedRealHandoffCoordinatorV4()
    req = _req_v4(rc=f"h760_{int(seed)}")
    e1 = c1.decide_v4(
        req_v4=req, substrate_self_checksum_cid="x")
    e2 = c2.decide_v4(
        req_v4=req, substrate_self_checksum_cid="x")
    return {
        "schema": R172_SCHEMA_VERSION,
        "name":
            "h760_handoff_v4_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h761_handoff_v4_text_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV4()
    e = c.decide_v4(
        req_v4=_req_v4(rc=f"h761_{int(seed)}"))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h761_handoff_v4_text_only_route",
        "passed": bool(
            str(e.decision_v4)
            == W69_HANDOFF_DECISION_HOSTED_ONLY),
    }


def family_h762_handoff_v4_substrate_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV4()
    e = c.decide_v4(
        req_v4=_req_v4(
            rc=f"h762_{int(seed)}",
            needs_text_only=False,
            needs_substrate_state_access=True,
            restart_pressure=0.3,
            dominant_repair_label=1,
            visible_token_budget=128))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h762_handoff_v4_substrate_only_route",
        "passed": bool(
            str(e.decision_v4)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY),
    }


def family_h763_handoff_v4_rejoin_promotion(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV4()
    e = c.decide_v4(
        req_v4=_req_v4(
            rc=f"h763_{int(seed)}",
            rejoin_pressure=0.8))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h763_handoff_v4_rejoin_promotion",
        "passed": bool(
            str(e.decision_v4)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.rejoin_alignment == 1.0),
    }


def family_h764_handoff_v4_delayed_rejoin_fallback(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV4()
    e = c.decide_v4(
        req_v4=_req_v4(
            rc=f"h764_{int(seed)}",
            restart_repair_trajectory_cid="rrt_xyz",
            rejoin_lag_turns=4))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h764_handoff_v4_delayed_rejoin_fallback",
        "passed": bool(
            str(e.decision_v4)
            == W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK
            and e.delayed_rejoin_fallback_active
            and e.delayed_rejoin_alignment == 1.0),
    }


def family_h765_handoff_v4_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_v4_rejoin_aware_savings(
        n_turns=12)
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h765_handoff_v4_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.78),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h766_handoff_v4_delayed_rejoin_falsifier(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV4()
    e = c.decide_v4(
        req_v4=_req_v4(rc=f"h766_{int(seed)}"))
    honest = (
        probe_hosted_real_handoff_v4_delayed_rejoin_falsifier(
            envelope_v4=e, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v4_delayed_rejoin_falsifier(
            envelope_v4=e, claim_satisfied=True))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name":
            "h766_handoff_v4_delayed_rejoin_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h767_boundary_v5_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v5()
    honest = probe_hosted_real_substrate_boundary_v5_falsifier(
        boundary=b,
        claimed_axis="restart_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v5_falsifier(
            boundary=b,
            claimed_axis="restart_repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h767_boundary_v5_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h768_kv_v17_rejoin_pressure_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v17_rejoin_pressure_falsifier(
        rejoin_pressure_flag=1)
    return {
        "schema": R172_SCHEMA_VERSION,
        "name":
            "h768_kv_v17_rejoin_pressure_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h769_compound_delayed_rejoin_after_restart_under_budget(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV8()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=(
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET))
    return {
        "schema": R172_SCHEMA_VERSION,
        "name":
            "h769_compound_delayed_rejoin_after_restart_under_budget",
        "passed": bool(
            agg.v17_beats_v16_rate >= 0.5
            and agg.tsc_v17_beats_tsc_v16_rate >= 0.5),
        "v17_beats_v16": float(agg.v17_beats_v16_rate),
        "tsc_v17_beats_tsc_v16": float(
            agg.tsc_v17_beats_tsc_v16_rate),
    }


def family_h770_w72_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V17 substrate is 19-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h770_w72_substrate_is_numpy_cpu",
        "passed": bool(
            W72_DEFAULT_V17_N_LAYERS == 19
            and W72_TINY_V17_VOCAB_SIZE == 259),
    }


def family_h771_w72_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted V5
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v5()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v5_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h771_w72_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


def family_h772_no_version_bump_invariant(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h772_no_version_bump_invariant",
        "passed": bool(
            str(__version__) == "0.5.20"
            and str(SDK_VERSION) == "coordpy.sdk.v3.43"),
    }


def family_h773_frontier_still_blocked(
        seed: int) -> dict[str, Any]:
    """Frontier-blocked axes set is non-empty and equal to W70's
    set; W72 does not unblock anything at the frontier."""
    return {
        "schema": R172_SCHEMA_VERSION,
        "name": "h773_frontier_still_blocked",
        "passed": bool(
            len(W72_FRONTIER_BLOCKED_AXES) >= 1
            and (
                tuple(W72_FRONTIER_BLOCKED_AXES)
                == tuple(W70_FRONTIER_BLOCKED_AXES))),
    }


_R172_FAMILIES: tuple[Any, ...] = (
    family_h760_handoff_v4_envelope_content_addressed,
    family_h761_handoff_v4_text_only_route,
    family_h762_handoff_v4_substrate_only_route,
    family_h763_handoff_v4_rejoin_promotion,
    family_h764_handoff_v4_delayed_rejoin_fallback,
    family_h765_handoff_v4_cross_plane_savings,
    family_h766_handoff_v4_delayed_rejoin_falsifier,
    family_h767_boundary_v5_falsifier,
    family_h768_kv_v17_rejoin_pressure_falsifier,
    family_h769_compound_delayed_rejoin_after_restart_under_budget,
    family_h770_w72_substrate_is_numpy_cpu,
    family_h771_w72_hosted_does_not_pierce_wall,
    family_h772_no_version_bump_invariant,
    family_h773_frontier_still_blocked,
)


def run_r172(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R172_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R172_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R172_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R172_SCHEMA_VERSION", "run_r172"]
