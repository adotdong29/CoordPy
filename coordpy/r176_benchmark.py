"""W73 R-176 benchmark family — Handoff V5 + Falsifier + Limitation.

The W73 falsifier-and-limitation-reproduction family + Plane A↔B
replacement-aware handoff family.

H860..H873 cell families (14 H-bars):

* H860   Handoff V5 envelope is content-addressed
* H861   Handoff V5 text-only routes to plane A
* H862   Handoff V5 substrate-only routes to plane B
* H863   Handoff V5 replacement_pressure ≥ floor + substrate_trust
         ≥ floor promotes to real_substrate_only with
         replacement_alignment = 1.0
* H864   Handoff V5 replacement_repair_trajectory + replacement_lag
         > 0 falls back to replacement_after_contradiction_then_rejoin_fallback
* H865   Handoff V5 cross-plane savings ≥ 80 %
* H866   Handoff V5 replacement falsifier honest=0, dishonest=1
* H867   Boundary V6 falsifier honest=0 / dishonest=1
* H868   KV V18 replacement-pressure falsifier honest = 0
* H869   MASC V9 compound regime
         (replacement_after_contradiction_then_rejoin) ≥ 50 %
         strict-beat
* H870   W73 substrate is in-repo NumPy (limitation reproduction)
* H871   W73 hosted control plane V6 does NOT pierce wall
         (limitation reproduction; 31 blocked-axes)
* H872   No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H873   Frontier substrate access still blocked (V6 frontier-
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
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5, HostedRealHandoffCoordinatorV5,
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
    hosted_real_handoff_v5_replacement_aware_savings,
    probe_hosted_real_handoff_v5_replacement_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v6 import (
    W73_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v6,
    probe_hosted_real_substrate_boundary_v6_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.kv_bridge_v18 import (
    probe_kv_bridge_v18_replacement_pressure_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v9 import (
    MultiAgentSubstrateCoordinatorV9,
    W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
)
from coordpy.tiny_substrate_v18 import (
    W73_DEFAULT_V18_N_LAYERS, W73_TINY_V18_VOCAB_SIZE,
)


R176_SCHEMA_VERSION: str = "coordpy.r176_benchmark.v1"


def _req_v5(*, rc: str, replacement_pressure: float = 0.0,
            replacement_repair_trajectory_cid: str = "",
            replacement_lag_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            restart_pressure: float = 0.0,
            rejoin_pressure: float = 0.0,
            dominant_repair_label: int = 0,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV5:
    return HandoffRequestV5(
        inner_v4=HandoffRequestV4(
            inner_v3=HandoffRequestV3(
                inner_v2=HandoffRequestV2(
                    inner_v1=HandoffRequest(
                        request_cid=str(rc),
                        needs_text_only=bool(needs_text_only),
                        needs_substrate_state_access=bool(
                            needs_substrate_state_access)),
                    visible_token_budget=int(
                        visible_token_budget),
                    baseline_token_cost=int(
                        baseline_token_cost),
                    dominant_repair_label=int(
                        dominant_repair_label)),
                restart_pressure=float(restart_pressure),
                delayed_repair_trajectory_cid="",
                delay_turns=0,
                expected_substrate_trust=0.7),
            rejoin_pressure=float(rejoin_pressure),
            restart_repair_trajectory_cid="",
            rejoin_lag_turns=0,
            expected_substrate_trust_v4=0.7),
        replacement_pressure=float(replacement_pressure),
        replacement_repair_trajectory_cid=str(
            replacement_repair_trajectory_cid),
        replacement_lag_turns=int(replacement_lag_turns),
        expected_substrate_trust_v5=0.7)


def family_h860_handoff_v5_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinatorV5()
    c2 = HostedRealHandoffCoordinatorV5()
    req = _req_v5(rc=f"h860_{int(seed)}")
    e1 = c1.decide_v5(
        req_v5=req, substrate_self_checksum_cid="x")
    e2 = c2.decide_v5(
        req_v5=req, substrate_self_checksum_cid="x")
    return {
        "schema": R176_SCHEMA_VERSION,
        "name":
            "h860_handoff_v5_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h861_handoff_v5_text_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV5()
    e = c.decide_v5(
        req_v5=_req_v5(rc=f"h861_{int(seed)}"))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h861_handoff_v5_text_only_route",
        "passed": bool(
            str(e.decision_v5)
            == W69_HANDOFF_DECISION_HOSTED_ONLY),
    }


def family_h862_handoff_v5_substrate_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV5()
    e = c.decide_v5(
        req_v5=_req_v5(
            rc=f"h862_{int(seed)}",
            needs_text_only=False,
            needs_substrate_state_access=True,
            restart_pressure=0.3,
            dominant_repair_label=1,
            visible_token_budget=128))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h862_handoff_v5_substrate_only_route",
        "passed": bool(
            str(e.decision_v5)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY),
    }


def family_h863_handoff_v5_replacement_promotion(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV5()
    e = c.decide_v5(
        req_v5=_req_v5(
            rc=f"h863_{int(seed)}",
            replacement_pressure=0.8))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h863_handoff_v5_replacement_promotion",
        "passed": bool(
            str(e.decision_v5)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.replacement_alignment == 1.0),
    }


def family_h864_handoff_v5_replacement_fallback(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV5()
    e = c.decide_v5(
        req_v5=_req_v5(
            rc=f"h864_{int(seed)}",
            replacement_repair_trajectory_cid="rep_xyz",
            replacement_lag_turns=5))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h864_handoff_v5_replacement_fallback",
        "passed": bool(
            str(e.decision_v5)
            == W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK
            and e.replacement_after_ctr_fallback_active
            and e.replacement_after_ctr_alignment == 1.0),
    }


def family_h865_handoff_v5_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_v5_replacement_aware_savings(
        n_turns=12)
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h865_handoff_v5_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.80),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h866_handoff_v5_replacement_falsifier(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV5()
    e = c.decide_v5(
        req_v5=_req_v5(rc=f"h866_{int(seed)}"))
    honest = (
        probe_hosted_real_handoff_v5_replacement_falsifier(
            envelope_v5=e, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v5_replacement_falsifier(
            envelope_v5=e, claim_satisfied=True))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name":
            "h866_handoff_v5_replacement_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h867_boundary_v6_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v6()
    honest = probe_hosted_real_substrate_boundary_v6_falsifier(
        boundary=b,
        claimed_axis="replacement_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v6_falsifier(
            boundary=b,
            claimed_axis="replacement_repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h867_boundary_v6_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h868_kv_v18_replacement_pressure_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v18_replacement_pressure_falsifier(
        replacement_pressure_flag=1)
    return {
        "schema": R176_SCHEMA_VERSION,
        "name":
            "h868_kv_v18_replacement_pressure_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h869_compound_replacement_after_ctr(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV9()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    return {
        "schema": R176_SCHEMA_VERSION,
        "name":
            "h869_compound_replacement_after_ctr",
        "passed": bool(
            agg.v18_beats_v17_rate >= 0.5
            and agg.tsc_v18_beats_tsc_v17_rate >= 0.5),
        "v18_beats_v17": float(agg.v18_beats_v17_rate),
        "tsc_v18_beats_tsc_v17": float(
            agg.tsc_v18_beats_tsc_v17_rate),
    }


def family_h870_w73_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V18 substrate is 20-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h870_w73_substrate_is_numpy_cpu",
        "passed": bool(
            W73_DEFAULT_V18_N_LAYERS == 20
            and W73_TINY_V18_VOCAB_SIZE == 259),
    }


def family_h871_w73_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted V6
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v6()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v6_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h871_w73_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


def family_h872_no_version_bump_invariant(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h872_no_version_bump_invariant",
        "passed": bool(
            str(__version__) == "0.5.20"
            and str(SDK_VERSION) == "coordpy.sdk.v3.43"),
    }


def family_h873_frontier_still_blocked(
        seed: int) -> dict[str, Any]:
    """Frontier-blocked axes set is non-empty and equal to W70's
    set; W73 does not unblock anything at the frontier."""
    return {
        "schema": R176_SCHEMA_VERSION,
        "name": "h873_frontier_still_blocked",
        "passed": bool(
            len(W73_FRONTIER_BLOCKED_AXES) >= 1
            and (
                tuple(W73_FRONTIER_BLOCKED_AXES)
                == tuple(W70_FRONTIER_BLOCKED_AXES))),
    }


_R176_FAMILIES: tuple[Any, ...] = (
    family_h860_handoff_v5_envelope_content_addressed,
    family_h861_handoff_v5_text_only_route,
    family_h862_handoff_v5_substrate_only_route,
    family_h863_handoff_v5_replacement_promotion,
    family_h864_handoff_v5_replacement_fallback,
    family_h865_handoff_v5_cross_plane_savings,
    family_h866_handoff_v5_replacement_falsifier,
    family_h867_boundary_v6_falsifier,
    family_h868_kv_v18_replacement_pressure_falsifier,
    family_h869_compound_replacement_after_ctr,
    family_h870_w73_substrate_is_numpy_cpu,
    family_h871_w73_hosted_does_not_pierce_wall,
    family_h872_no_version_bump_invariant,
    family_h873_frontier_still_blocked,
)


def run_r176(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R176_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R176_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R176_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R176_SCHEMA_VERSION", "run_r176"]
