"""W70 R-164 benchmark family — Handoff V2 + Falsifier + Limitation.

The W70 falsifier-and-limitation-reproduction family + Plane A↔B
budget-primary handoff family.

H520..H531 cell families (12 H-bars):

* H520   Handoff V2 envelope is content-addressed
* H521   Handoff V2 text-only routes to plane A (low-budget=>BP)
* H522   Handoff V2 substrate-only routes to plane B
* H523   Handoff V2 both-required+tight-budget routes to B
* H524   Handoff V2 dominant_repair_label > 0 + hosted_only request
         is promoted to real_substrate_only (repair-dominance
         alignment = 1.0)
* H525   Handoff V2 cross-plane savings ≥ 50 %
* H526   Handoff V2 repair-dominance falsifier honest = 0, dishonest = 1
* H527   Boundary V3 falsifier honest = 0 / dishonest = 1
* H528   KV V15 repair-dominance falsifier reproduces (0 on honest)
* H529   MASC V6 compound regime (contradiction_then_rejoin_under_
         budget) ≥ 50 % strict-beat
* H530   W70 substrate is in-repo NumPy (limitation reproduction)
* H531   W70 hosted control plane V3 does NOT pierce wall (limitation
         reproduction; carries forward 22 blocked-axes)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_real_handoff_coordinator import HandoffRequest
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2, HostedRealHandoffCoordinatorV2,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    hosted_real_handoff_v2_budget_primary_savings,
    probe_hosted_real_handoff_v2_repair_dominance_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    build_default_hosted_real_substrate_boundary_v3,
    probe_hosted_real_substrate_boundary_v3_falsifier,
)
from coordpy.kv_bridge_v15 import (
    probe_kv_bridge_v15_repair_dominance_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v6 import (
    MultiAgentSubstrateCoordinatorV6,
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
)
from coordpy.tiny_substrate_v15 import (
    W70_DEFAULT_V15_N_LAYERS, W70_TINY_V15_VOCAB_SIZE,
)


R164_SCHEMA_VERSION: str = "coordpy.r164_benchmark.v1"


def family_h520_handoff_v2_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinatorV2()
    c2 = HostedRealHandoffCoordinatorV2()
    req = HandoffRequestV2(
        inner_v1=HandoffRequest(
            request_cid=f"h520_{int(seed)}",
            needs_text_only=True,
            needs_substrate_state_access=False),
        visible_token_budget=256,
        baseline_token_cost=512,
        dominant_repair_label=0)
    e1 = c1.decide_v2(
        req_v2=req, substrate_self_checksum_cid="x")
    e2 = c2.decide_v2(
        req_v2=req, substrate_self_checksum_cid="x")
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h520_handoff_v2_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h521_handoff_v2_text_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV2()
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h521_{int(seed)}",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=0))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h521_handoff_v2_text_only_route",
        "passed": bool(
            str(e.decision_v2) == W69_HANDOFF_DECISION_HOSTED_ONLY
            and e.plane_v2 == "A"),
    }


def family_h522_handoff_v2_substrate_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV2()
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h522_{int(seed)}",
                needs_text_only=False,
                needs_substrate_state_access=True),
            visible_token_budget=128,
            baseline_token_cost=512,
            dominant_repair_label=1))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h522_handoff_v2_substrate_only_route",
        "passed": bool(
            str(e.decision_v2)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.plane_v2 == "B"),
    }


def family_h523_handoff_v2_tight_budget_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV2()
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h523_{int(seed)}",
                needs_text_only=True,
                needs_substrate_state_access=True),
            visible_token_budget=64,
            baseline_token_cost=512,
            dominant_repair_label=1))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h523_handoff_v2_tight_budget_route",
        "passed": bool(
            str(e.decision_v2)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.plane_v2 == "B"),
    }


def family_h524_handoff_v2_repair_dominance_promotion(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV2()
    # Text-only request + dominant_repair_label=1 → V2 promotes to
    # real_substrate_only (repair-dominance alignment).
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h524_{int(seed)}",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=1))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h524_handoff_v2_repair_dominance_promotion",
        "passed": bool(
            str(e.decision_v2)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.repair_dominance_alignment == 1.0),
    }


def family_h525_handoff_v2_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_v2_budget_primary_savings(
        n_turns=12)
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h525_handoff_v2_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.5),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h526_handoff_v2_repair_dominance_falsifier(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV2()
    # Construct a hosted_only envelope explicitly.
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid=f"h526_{int(seed)}",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=0))
    honest = (
        probe_hosted_real_handoff_v2_repair_dominance_falsifier(
            envelope_v2=e, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v2_repair_dominance_falsifier(
            envelope_v2=e, claim_satisfied=True))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h526_handoff_v2_repair_dominance_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h527_boundary_v3_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v3()
    honest = probe_hosted_real_substrate_boundary_v3_falsifier(
        boundary=b,
        claimed_axis="repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v3_falsifier(
            boundary=b,
            claimed_axis="repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h527_boundary_v3_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h528_kv_v15_repair_dominance_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v15_repair_dominance_falsifier(
        dominant_repair_label=1)
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h528_kv_v15_repair_dominance_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h529_compound_contradiction_then_rejoin(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV6()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=(
            W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET))
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h529_compound_contradiction_then_rejoin",
        "passed": bool(
            agg.v15_beats_v14_rate >= 0.5
            and agg.tsc_v15_beats_tsc_v14_rate >= 0.5),
        "v15_beats_v14": float(agg.v15_beats_v14_rate),
        "tsc_v15_beats_tsc_v14": float(
            agg.tsc_v15_beats_tsc_v14_rate),
    }


def family_h530_w70_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V15 substrate is 17-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h530_w70_substrate_is_numpy_cpu",
        "passed": bool(
            W70_DEFAULT_V15_N_LAYERS == 17
            and W70_TINY_V15_VOCAB_SIZE == 259),
    }


def family_h531_w70_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted V3
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v3()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v3_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R164_SCHEMA_VERSION,
        "name": "h531_w70_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


_R164_FAMILIES: tuple[Any, ...] = (
    family_h520_handoff_v2_envelope_content_addressed,
    family_h521_handoff_v2_text_only_route,
    family_h522_handoff_v2_substrate_only_route,
    family_h523_handoff_v2_tight_budget_route,
    family_h524_handoff_v2_repair_dominance_promotion,
    family_h525_handoff_v2_cross_plane_savings,
    family_h526_handoff_v2_repair_dominance_falsifier,
    family_h527_boundary_v3_falsifier,
    family_h528_kv_v15_repair_dominance_falsifier,
    family_h529_compound_contradiction_then_rejoin,
    family_h530_w70_substrate_is_numpy_cpu,
    family_h531_w70_hosted_does_not_pierce_wall,
)


def run_r164(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R164_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R164_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R164_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R164_SCHEMA_VERSION", "run_r164"]
