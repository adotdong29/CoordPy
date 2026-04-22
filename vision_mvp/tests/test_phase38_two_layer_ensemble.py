"""Phase 38 Part A — two-layer ensemble composition tests."""

from __future__ import annotations

import pytest

from vision_mvp.core.extractor_adversary import (
    DropGoldClaimExtractor, NarrativeSecondaryExtractor,
    UnionClaimExtractor, build_union_extractor,
)
from vision_mvp.core.two_layer_ensemble import (
    ALL_PATH_MODES, PATH_MODE_DUAL_AGREE, PATH_MODE_UNION_ROOT,
    PATH_MODE_VERIFIED, PathUnionCausalityExtractor,
    TwoLayerDefense,
)
from vision_mvp.core.reply_noise import (
    CAUSALITY_DOWNSTREAM_PREFIX, CAUSALITY_INDEPENDENT_ROOT,
    CAUSALITY_UNCERTAIN,
)
from vision_mvp.tasks.contested_incident import build_contested_bank
from vision_mvp.tasks.incident_triage import extract_claims_for_role


# =============================================================================
# PathUnionCausalityExtractor
# =============================================================================


def _mk_extractor(resp):
    """Build a constant causality extractor."""
    def _f(scenario, role, kind, payload):
        return resp
    return _f


def test_path_union_dual_agree_both_ir():
    p = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    s = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_DUAL_AGREE)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_INDEPENDENT_ROOT
    assert combiner.stats.n_agree == 1


def test_path_union_dual_agree_disagree_emits_uncertain():
    p = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    s = _mk_extractor(CAUSALITY_DOWNSTREAM_PREFIX + "X")
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_DUAL_AGREE)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_UNCERTAIN
    assert combiner.stats.n_disagree == 1


def test_path_union_union_root_one_ir_wins():
    # Simulates adversarial drop: primary has been corrupted to
    # UNCERTAIN, secondary emits IR. UNION_ROOT recovers IR.
    p = _mk_extractor(CAUSALITY_UNCERTAIN)
    s = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_UNION_ROOT)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_INDEPENDENT_ROOT
    assert combiner.stats.n_secondary_used == 1


def test_path_union_union_root_contradictory_emits_uncertain():
    # Primary says IR, secondary says DS on same candidate:
    # conservative UNCERTAIN.
    p = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    s = _mk_extractor(CAUSALITY_DOWNSTREAM_PREFIX + "Y")
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_UNION_ROOT)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_UNCERTAIN


def test_path_union_verified_primary_accepted_on_match():
    p = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    s = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_VERIFIED)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_INDEPENDENT_ROOT


def test_path_union_verified_rejects_mismatched_ir():
    # Primary biased to IR, secondary disagrees → UNCERTAIN.
    p = _mk_extractor(CAUSALITY_INDEPENDENT_ROOT)
    s = _mk_extractor(CAUSALITY_DOWNSTREAM_PREFIX + "Z")
    combiner = PathUnionCausalityExtractor(
        primary=p, secondary=s, mode=PATH_MODE_VERIFIED)
    out = combiner(None, "role", "kind", "payload")
    assert out == CAUSALITY_UNCERTAIN


def test_path_union_all_modes_registered():
    for m in ALL_PATH_MODES:
        combiner = PathUnionCausalityExtractor(
            primary=_mk_extractor(CAUSALITY_UNCERTAIN),
            secondary=_mk_extractor(CAUSALITY_UNCERTAIN),
            mode=m)
        out = combiner(None, "r", "k", "p")
        assert out == CAUSALITY_UNCERTAIN


def test_path_union_unknown_mode_raises():
    with pytest.raises(ValueError):
        PathUnionCausalityExtractor(
            primary=_mk_extractor(CAUSALITY_UNCERTAIN),
            secondary=_mk_extractor(CAUSALITY_UNCERTAIN),
            mode="badmode")


# =============================================================================
# Extractor adversary + narrative
# =============================================================================


def test_drop_gold_claim_extractor_removes_target():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    scen = next(s for s in bank
                 if s.scenario_id
                 == "contested_deadlock_vs_shadow_cron")
    adv = DropGoldClaimExtractor(
        target_role="db_admin",
        target_kind="DEADLOCK_SUSPECTED", budget=1)
    evs = list(scen.per_role_events["db_admin"])
    kinds = [c[0] for c in adv("db_admin", evs, scen.base)]
    assert "DEADLOCK_SUSPECTED" not in kinds
    assert adv.n_dropped == 1


def test_narrative_catches_dropped_deadlock():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    scen = next(s for s in bank
                 if s.scenario_id
                 == "contested_deadlock_vs_shadow_cron")
    narr = NarrativeSecondaryExtractor()
    evs = list(scen.per_role_events["db_admin"])
    kinds = [c[0] for c in narr("db_admin", evs, scen.base)]
    assert "DEADLOCK_SUSPECTED" in kinds


def test_narrative_does_not_emit_on_distractor_only_role():
    # auditor has no events; narrative should emit nothing.
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    scen = bank[0]
    narr = NarrativeSecondaryExtractor()
    claims = narr("auditor", scen.per_role_events.get(
        "auditor", ()), scen.base)
    assert claims == []


def test_union_extractor_recovers_dropped_gold():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    scen = next(s for s in bank
                 if s.scenario_id
                 == "contested_tls_vs_disk_shadow")
    union, adv, narr = build_union_extractor(
        target_role="network", target_kind="TLS_EXPIRED",
        drop_budget=1)
    evs = list(scen.per_role_events["network"])
    kinds = [c[0] for c in union("network", evs, scen.base)]
    assert "TLS_EXPIRED" in kinds
    # The adversary still dropped on the primary, but the
    # narrative caught it.
    assert adv.n_dropped >= 1


# =============================================================================
# TwoLayerDefense record
# =============================================================================


def test_two_layer_defense_record_serialises():
    d = TwoLayerDefense(
        label="ext=union_narr;reply=dual_agree",
        n_layer1_active=1, n_layer2_active=1,
        path_union_mode=None)
    out = d.as_dict()
    assert out["label"] == "ext=union_narr;reply=dual_agree"
    assert out["n_layer1_active"] == 1
    assert out["n_layer2_active"] == 1
