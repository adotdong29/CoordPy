"""W71 R-168 benchmark family — Handoff V3 + Falsifier + Limitation.

The W71 falsifier-and-limitation-reproduction family + Plane A↔B
restart-aware handoff family.

H600..H613 cell families (14 H-bars):

* H600   Handoff V3 envelope is content-addressed
* H601   Handoff V3 text-only routes to plane A
* H602   Handoff V3 substrate-only routes to plane B
* H603   Handoff V3 restart_pressure ≥ floor + substrate_trust ≥
         floor promotes to real_substrate_only with
         restart_alignment = 1.0
* H604   Handoff V3 delayed_repair_trajectory + delay > 0 falls
         back to delayed_repair_fallback
* H605   Handoff V3 cross-plane savings ≥ 70 %
* H606   Handoff V3 delayed-repair falsifier honest=0, dishonest=1
* H607   Boundary V4 falsifier honest=0 / dishonest=1
* H608   KV V16 restart-dominance falsifier honest = 0
* H609   MASC V7 compound regime
         (delayed_repair_after_restart) ≥ 50 % strict-beat
* H610   W71 substrate is in-repo NumPy (limitation reproduction)
* H611   W71 hosted control plane V4 does NOT pierce wall
         (limitation reproduction; carries forward 25 blocked-axes)
* H612   No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H613   Frontier substrate access still blocked (V4 frontier-
         blocked axes set non-empty and unchanged from W70)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy import SDK_VERSION, __version__
from coordpy.hosted_real_handoff_coordinator import HandoffRequest
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3, HostedRealHandoffCoordinatorV3,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    hosted_real_handoff_v3_restart_aware_savings,
    probe_hosted_real_handoff_v3_delayed_repair_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v4 import (
    W71_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v4,
    probe_hosted_real_substrate_boundary_v4_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.kv_bridge_v16 import (
    probe_kv_bridge_v16_restart_dominance_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v7 import (
    MultiAgentSubstrateCoordinatorV7,
    W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
)
from coordpy.tiny_substrate_v16 import (
    W71_DEFAULT_V16_N_LAYERS, W71_TINY_V16_VOCAB_SIZE,
)


R168_SCHEMA_VERSION: str = "coordpy.r168_benchmark.v1"


def family_h600_handoff_v3_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinatorV3()
    c2 = HostedRealHandoffCoordinatorV3()
    req = HandoffRequestV3(
        inner_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h600_{int(seed)}",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=0),
        restart_pressure=0.0,
        delayed_repair_trajectory_cid="",
        delay_turns=0,
        expected_substrate_trust=0.7)
    e1 = c1.decide_v3(
        req_v3=req, substrate_self_checksum_cid="x")
    e2 = c2.decide_v3(
        req_v3=req, substrate_self_checksum_cid="x")
    return {
        "schema": R168_SCHEMA_VERSION,
        "name":
            "h600_handoff_v3_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h601_handoff_v3_text_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV3()
    e = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=f"h601_{int(seed)}",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.0,
            delayed_repair_trajectory_cid="",
            delay_turns=0,
            expected_substrate_trust=0.7))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h601_handoff_v3_text_only_route",
        "passed": bool(
            str(e.decision_v3)
            == W69_HANDOFF_DECISION_HOSTED_ONLY),
    }


def family_h602_handoff_v3_substrate_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV3()
    e = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=f"h602_{int(seed)}",
                    needs_text_only=False,
                    needs_substrate_state_access=True),
                visible_token_budget=128,
                baseline_token_cost=512,
                dominant_repair_label=1),
            restart_pressure=0.3,
            expected_substrate_trust=0.7))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h602_handoff_v3_substrate_only_route",
        "passed": bool(
            str(e.decision_v3)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY),
    }


def family_h603_handoff_v3_restart_promotion(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV3()
    e = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=f"h603_{int(seed)}",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.8,
            expected_substrate_trust=0.7))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h603_handoff_v3_restart_promotion",
        "passed": bool(
            str(e.decision_v3)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.restart_alignment == 1.0),
    }


def family_h604_handoff_v3_delayed_repair_fallback(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV3()
    e = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=f"h604_{int(seed)}",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.0,
            delayed_repair_trajectory_cid="drt_xyz",
            delay_turns=3,
            expected_substrate_trust=0.7))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h604_handoff_v3_delayed_repair_fallback",
        "passed": bool(
            str(e.decision_v3)
            == W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK
            and e.delayed_repair_fallback_active
            and e.delayed_repair_alignment == 1.0),
    }


def family_h605_handoff_v3_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_v3_restart_aware_savings(
        n_turns=12)
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h605_handoff_v3_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.70),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h606_handoff_v3_delayed_repair_falsifier(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV3()
    e = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid=f"h606_{int(seed)}",
                    needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.0,
            delayed_repair_trajectory_cid="",
            delay_turns=0,
            expected_substrate_trust=0.7))
    honest = (
        probe_hosted_real_handoff_v3_delayed_repair_falsifier(
            envelope_v3=e, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v3_delayed_repair_falsifier(
            envelope_v3=e, claim_satisfied=True))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name":
            "h606_handoff_v3_delayed_repair_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h607_boundary_v4_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v4()
    honest = probe_hosted_real_substrate_boundary_v4_falsifier(
        boundary=b,
        claimed_axis="delayed_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v4_falsifier(
            boundary=b,
            claimed_axis="delayed_repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h607_boundary_v4_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h608_kv_v16_restart_dominance_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v16_restart_dominance_falsifier(
        restart_dominance_flag=1)
    return {
        "schema": R168_SCHEMA_VERSION,
        "name":
            "h608_kv_v16_restart_dominance_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h609_compound_delayed_repair_after_restart(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV7()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=(
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART))
    return {
        "schema": R168_SCHEMA_VERSION,
        "name":
            "h609_compound_delayed_repair_after_restart",
        "passed": bool(
            agg.v16_beats_v15_rate >= 0.5
            and agg.tsc_v16_beats_tsc_v15_rate >= 0.5),
        "v16_beats_v15": float(agg.v16_beats_v15_rate),
        "tsc_v16_beats_tsc_v15": float(
            agg.tsc_v16_beats_tsc_v15_rate),
    }


def family_h610_w71_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V16 substrate is 18-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h610_w71_substrate_is_numpy_cpu",
        "passed": bool(
            W71_DEFAULT_V16_N_LAYERS == 18
            and W71_TINY_V16_VOCAB_SIZE == 259),
    }


def family_h611_w71_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted V4
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v4()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v4_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h611_w71_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


def family_h612_no_version_bump_invariant(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h612_no_version_bump_invariant",
        "passed": bool(
            str(__version__) == "0.5.20"
            and str(SDK_VERSION) == "coordpy.sdk.v3.43"),
    }


def family_h613_frontier_still_blocked(
        seed: int) -> dict[str, Any]:
    """Frontier-blocked axes set is non-empty and equal to W70's
    set; W71 does not unblock anything at the frontier."""
    return {
        "schema": R168_SCHEMA_VERSION,
        "name": "h613_frontier_still_blocked",
        "passed": bool(
            len(W71_FRONTIER_BLOCKED_AXES) >= 1
            and (
                tuple(W71_FRONTIER_BLOCKED_AXES)
                == tuple(W70_FRONTIER_BLOCKED_AXES))),
    }


_R168_FAMILIES: tuple[Any, ...] = (
    family_h600_handoff_v3_envelope_content_addressed,
    family_h601_handoff_v3_text_only_route,
    family_h602_handoff_v3_substrate_only_route,
    family_h603_handoff_v3_restart_promotion,
    family_h604_handoff_v3_delayed_repair_fallback,
    family_h605_handoff_v3_cross_plane_savings,
    family_h606_handoff_v3_delayed_repair_falsifier,
    family_h607_boundary_v4_falsifier,
    family_h608_kv_v16_restart_dominance_falsifier,
    family_h609_compound_delayed_repair_after_restart,
    family_h610_w71_substrate_is_numpy_cpu,
    family_h611_w71_hosted_does_not_pierce_wall,
    family_h612_no_version_bump_invariant,
    family_h613_frontier_still_blocked,
)


def run_r168(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R168_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R168_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R168_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R168_SCHEMA_VERSION", "run_r168"]
