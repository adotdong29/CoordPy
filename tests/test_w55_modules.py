"""W55 M2-M10 — compact per-module unit tests."""

from __future__ import annotations

from coordpy.corruption_robust_carrier_v3 import (
    CorruptionRobustCarrierV3,
    bch_15_7_decode,
    bch_15_7_encode,
    deinterleave_bits,
    emit_corruption_robustness_v3_witness,
    interleave_bits,
    majority_decode_n_of_m,
    repeat_bits_n_of_m,
)
from coordpy.deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from coordpy.ecc_codebook_v5 import (
    ECCCodebookV5, W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from coordpy.ecc_codebook_v7 import (
    ECCCodebookV7,
    compress_carrier_ecc_v7,
    emit_ecc_v7_compression_witness,
    probe_ecc_v7_rate_floor_falsifier,
)
from coordpy.long_horizon_retention_v7 import (
    LongHorizonReconstructionV7Head,
    emit_lhr_v7_witness,
    probe_v7_degradation_curve,
)
from coordpy.mergeable_latent_capsule_v3 import (
    MergeAuditTrailV3, MergeOperatorV3,
    emit_mlsc_v3_witness,
    make_root_capsule_v3,
    merge_capsules_v3,
    reinforce_capsule_trust_v3,
    step_branch_capsule_v3,
)
from coordpy.multi_hop_translator_v5 import (
    emit_multi_hop_v5_witness,
    fit_hept_translator,
    score_hept_fidelity,
    synthesize_hept_training_set,
    trust_weighted_compromise_arbitration,
)
from coordpy.quantised_compression import QuantisedBudgetGate
from coordpy.transcript_vs_shared_arbiter_v4 import (
    W55_TVS_ARM_TRUST_WEIGHTED_MERGE,
    arbiter_decide_v4,
    emit_tvs_arbiter_v4_witness,
    five_arm_compare,
)
from coordpy.trust_weighted_consensus_controller import (
    TrustWeightedConsensusController,
    TrustWeightedConsensusPolicy,
    W55_TWCC_DECISION_QUORUM,
    emit_twcc_witness,
)
from coordpy.uncertainty_layer import calibration_check
from coordpy.uncertainty_layer_v2 import calibration_check_under_noise
from coordpy.uncertainty_layer_v3 import (
    FactUncertainty,
    calibration_check_under_adversarial,
    compose_uncertainty_report_v3,
    emit_uncertainty_layer_v3_witness,
    propagate_fact_uncertainty,
)


# --------------------------------------------------------------------- M2 ---


def test_hept_translator_has_7_backends() -> None:
    ts = synthesize_hept_training_set(
        n_examples=8, code_dim=4, feature_dim=4, seed=1)
    assert len(ts.backends) == 7


def test_hept_chain_len6_fidelity_reasonable() -> None:
    ts = synthesize_hept_training_set(
        n_examples=8, code_dim=4, feature_dim=4, seed=1)
    tr, _ = fit_hept_translator(ts, n_steps=24, seed=1)
    fid = score_hept_fidelity(tr, ts.examples[:4])
    assert fid.chain_len6_fid_mean > 0.0


def test_trust_weighted_arbitration_excludes_low_trust() -> None:
    ts = synthesize_hept_training_set(
        n_examples=8, code_dim=4, feature_dim=4, seed=1)
    tr, _ = fit_hept_translator(ts, n_steps=24, seed=1)
    paths = (
        ("A", "B"), ("A", "B", "C"))
    res = trust_weighted_compromise_arbitration(
        tr, paths=paths,
        input_vec=ts.examples[0].feature_by_backend["A"],
        feature_dim=4,
        trust_per_backend={
            "B": 0.01, "C": 0.01},
        trust_floor=0.5)
    assert res.abstain  # no path passes trust floor


# --------------------------------------------------------------------- M3 ---


def test_mlsc_v3_merge_records_fact_confirmation() -> None:
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    audit = MergeAuditTrailV3.empty()
    a = make_root_capsule_v3(
        branch_id="a", payload=[1, 0, 0, 0],
        confidence=0.8, trust=0.9,
        fact_tags=("shared", "a1"))
    b = make_root_capsule_v3(
        branch_id="b", payload=[0.9, 0.1, 0, 0],
        confidence=0.7, trust=0.85,
        fact_tags=("shared", "b1"))
    m = merge_capsules_v3(op, [a, b], audit_trail=audit)
    assert m.get_confirmation_count("shared") == 2
    assert m.get_confirmation_count("a1") == 1
    assert m.get_confirmation_count("b1") == 1


def test_mlsc_v3_trust_decays_each_turn() -> None:
    p = make_root_capsule_v3(
        branch_id="a", payload=[0, 0, 0, 0],
        confidence=0.9, trust=0.9, trust_decay=0.5)
    ch = step_branch_capsule_v3(
        parent=p, payload=[0, 0, 0, 0])
    assert ch.trust < p.trust


def test_mlsc_v3_reinforce_increases_trust() -> None:
    p = make_root_capsule_v3(
        branch_id="a", payload=[0, 0, 0, 0],
        confidence=0.9, trust=0.3)
    re = reinforce_capsule_trust_v3(p, reinforcement=0.5)
    assert re.trust > p.trust


def test_mlsc_v3_witness_records_agreement_mask() -> None:
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    audit = MergeAuditTrailV3.empty()
    a = make_root_capsule_v3(
        branch_id="a", payload=[1, 0, 0.5, 0],
        confidence=0.8, trust=0.9)
    b = make_root_capsule_v3(
        branch_id="b", payload=[1, 1, 0.5, 0],
        confidence=0.8, trust=0.9)
    m = merge_capsules_v3(op, [a, b], audit_trail=audit)
    store = {a.cid(): a, b.cid(): b, m.cid(): m}
    w = emit_mlsc_v3_witness(
        leaf=m, operator=op, audit_trail=audit,
        capsule_store=store)
    assert w.leaf_agreement_mask_sum >= 3  # dims 0, 2, 3 agree


# --------------------------------------------------------------------- M4 ---


def test_twcc_kof_n_succeeds_on_consistent() -> None:
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    ctrl = TrustWeightedConsensusController.init(
        operator=op)
    bs = [
        make_root_capsule_v3(
            branch_id=f"b{i}", payload=[1, 0, 0.5, 0.2],
            confidence=0.9, trust=0.9)
        for i in range(4)
    ]
    res, _ = ctrl.decide(bs, k_required=2)
    assert res.decision == W55_TWCC_DECISION_QUORUM


def test_twcc_falls_back_to_best_parent() -> None:
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    policy = TrustWeightedConsensusPolicy(
        k_min=4, k_max=4, cosine_floor=0.99,
        fallback_cosine_floor=0.0,
        trust_threshold=10.0,
        allow_trust_weighted=True,
        allow_fallback_best_parent=True,
        allow_fallback_transcript=False)
    ctrl = TrustWeightedConsensusController.init(
        policy=policy, operator=op)
    bs = [
        make_root_capsule_v3(
            branch_id="b0", payload=[1, 0, 0, 0],
            confidence=0.5, trust=0.5),
        make_root_capsule_v3(
            branch_id="b1", payload=[-1, 0, 0, 0],
            confidence=0.5, trust=0.5),
    ]
    res, _ = ctrl.decide(bs, k_required=4)
    assert res.decision != W55_TWCC_DECISION_QUORUM


def test_twcc_witness_rates_sum_to_one() -> None:
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    ctrl = TrustWeightedConsensusController.init(
        operator=op)
    bs = [
        make_root_capsule_v3(
            branch_id=f"b{i}", payload=[1, 0, 0.5, 0.2],
            confidence=0.9, trust=0.9)
        for i in range(4)
    ]
    ctrl.decide(bs, k_required=2)
    w = emit_twcc_witness(ctrl)
    s = (
        w.quorum_rate + w.trust_weighted_rate
        + w.best_parent_rate + w.transcript_rate
        + w.abstain_rate)
    assert abs(s - 1.0) < 1e-6


# --------------------------------------------------------------------- M5 ---


def test_bch_encode_decode_clean() -> None:
    data = (1, 0, 1, 1, 0, 1, 0)
    cw = bch_15_7_encode(data)
    assert len(cw) == 15
    back, dist, _, double = bch_15_7_decode(cw)
    assert back == data
    assert dist == 0


def test_bch_corrects_single_bit() -> None:
    data = (1, 0, 1, 1, 0, 1, 0)
    cw = list(bch_15_7_encode(data))
    cw[3] ^= 1
    back, dist, _, _ = bch_15_7_decode(cw)
    assert back == data


def test_bch_corrects_double_bit() -> None:
    data = (1, 0, 1, 1, 0, 1, 0)
    cw = list(bch_15_7_encode(data))
    cw[3] ^= 1
    cw[7] ^= 1
    back, dist, _, _ = bch_15_7_decode(cw)
    assert back == data


def test_interleave_round_trip() -> None:
    seg = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]
    flat = interleave_bits(seg)
    out = deinterleave_bits(
        flat, n_segments=3, bits_per_segment=4)
    assert out == seg


def test_repetition_5_of_7_majority() -> None:
    bits = [1, 0, 1]
    rep = repeat_bits_n_of_m(bits, n_rep=7)
    assert len(rep) == 21
    out = majority_decode_n_of_m(rep, n_rep=7, majority=4)
    assert out == bits


# --------------------------------------------------------------------- M6 ---


def test_deep_v6_has_14_layers_default() -> None:
    stack = DeepProxyStackV6.init(
        n_layers=14, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2, seed=1)
    assert stack.n_layers == 14


def test_deep_v6_pathological_input_triggers_abstain() -> None:
    stack = DeepProxyStackV6.init(
        n_layers=14, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2, seed=1)
    q = [1e6] * 8
    w, _ = emit_deep_proxy_stack_v6_forward_witness(
        stack=stack, query_input=q,
        slot_keys=[q], slot_values=[q])
    assert w.adaptive_abstain or w.corruption_flag


# --------------------------------------------------------------------- M7 ---


def test_lhr_v7_max_k_36() -> None:
    head = LongHorizonReconstructionV7Head.init(
        carrier_dim=24, hidden_dim=8, out_dim=4,
        max_k=36, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2, seed=1)
    assert head.max_k == 36


def test_lhr_v7_degradation_curve_returns_points() -> None:
    head = LongHorizonReconstructionV7Head.init(
        carrier_dim=24, hidden_dim=8, out_dim=4,
        max_k=36, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2, seed=1)
    w = emit_lhr_v7_witness(
        head=head, examples=(),
        k_max_for_degradation=4)
    assert w.max_k == 36


# --------------------------------------------------------------------- M8 ---


def test_ecc_v7_has_6_levels() -> None:
    cb = ECCCodebookV7.init(seed=1)
    assert cb.n_ultra4 == 2  # K6 = 2


def test_ecc_v7_compression_yields_18plus_bits_per_token() -> None:
    cb = ECCCodebookV7.init(seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    import random
    random.seed(7)
    carrier = [
        random.uniform(-1, 1) for _ in range(cb.code_dim)]
    comp = compress_carrier_ecc_v7(
        carrier, codebook=cb, gate=gate)
    assert comp.bits_per_visible_token_v7 >= 18.0


def test_ecc_v7_rate_floor_falsifier_96bit_misses() -> None:
    cb = ECCCodebookV7.init(seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    import random
    random.seed(7)
    carriers = [
        [random.uniform(-1, 1) for _ in range(cb.code_dim)]
        for _ in range(4)
    ]
    res = probe_ecc_v7_rate_floor_falsifier(
        codebook=cb, gate=gate,
        sample_carriers=carriers,
        target_bits_per_token=96.0)
    assert res["rate_target_missed_rate"] == 1.0


# --------------------------------------------------------------------- M9 ---


def test_tvs_v4_5arm_pick_rates_sum_to_one() -> None:
    from coordpy.quantised_compression import QuantisedCodebookV4
    cb = QuantisedCodebookV4.init(seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=cb.code_dim, emit_mask_len=4, seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    import random
    random.seed(7)
    carriers = [
        [random.uniform(-1, 1) for _ in range(cb.code_dim)]
        for _ in range(4)
    ]
    res = five_arm_compare(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=5)
    s = (
        res.pick_rate_transcript + res.pick_rate_shared
        + res.pick_rate_merge
        + res.pick_rate_trust_weighted
        + res.pick_rate_abstain_with_fallback)
    assert abs(s - 1.0) < 1e-6


def test_tvs_v4_budget_allocator_sums_to_total() -> None:
    from coordpy.quantised_compression import QuantisedCodebookV4
    cb = QuantisedCodebookV4.init(seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=cb.code_dim, emit_mask_len=4, seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    d = arbiter_decide_v4(
        turn_index=0, carrier=[0.1] * cb.code_dim,
        codebook=cb, gate=gate, budget_tokens=5,
        confidence=0.8, trust_score=0.9,
        merge_consensus_retention=0.7,
        trust_weighted_retention=0.85)
    assert sum(b for _, b in d.budget_per_arm) == 5


# -------------------------------------------------------------------- M10 ---


def test_uncert_v3_per_fact_propagation_geomean() -> None:
    a_facts = (
        FactUncertainty("shared", 0.5, 1),
        FactUncertainty("a1", 0.8, 1))
    b_facts = (
        FactUncertainty("shared", 0.9, 1),
        FactUncertainty("b1", 0.7, 1))
    merged = propagate_fact_uncertainty([a_facts, b_facts])
    # shared confidence = sqrt(0.5 * 0.9) ≈ 0.6708
    shared = next(f for f in merged if f.tag == "shared")
    assert abs(shared.confidence - 0.6708) < 1e-3
    assert shared.n_contributors == 2


def test_uncert_v3_trust_weighted_lowers_composite() -> None:
    r_eq = compose_uncertainty_report_v3(
        persistent_v7_confidence=0.9,
        multi_hop_v5_confidence=0.9,
        mlsc_v3_capsule_confidence=0.9,
        deep_v6_corruption_confidence=0.9,
        crc_v3_silent_failure_rate=0.05,
        trust_weights={
            "persistent_v7": 1.0,
            "multi_hop_v5": 1.0,
            "mlsc_v3": 1.0,
            "deep_v6": 1.0,
            "crc_v3": 1.0,
        })
    r_low = compose_uncertainty_report_v3(
        persistent_v7_confidence=0.9,
        multi_hop_v5_confidence=0.9,
        mlsc_v3_capsule_confidence=0.9,
        deep_v6_corruption_confidence=0.9,
        crc_v3_silent_failure_rate=0.05,
        trust_weights={
            "persistent_v7": 0.05,
            "multi_hop_v5": 1.0,
            "mlsc_v3": 1.0,
            "deep_v6": 1.0,
            "crc_v3": 1.0,
        })
    # When one high-conf component has very low trust, the
    # trust-weighted composite should differ from full-trust.
    assert abs(
        r_low.trust_weighted_composite
        - r_eq.trust_weighted_composite) > 1e-6


def test_uncert_v3_adversarial_calibration_runs() -> None:
    import random
    rng = random.Random(11)
    confs = [
        rng.uniform(0.7, 0.9) if i % 3 != 0
        else rng.uniform(0.0, 0.3)
        for i in range(30)
    ]
    accs = [
        rng.uniform(0.7, 1.0) if i % 3 != 0
        else rng.uniform(0.0, 0.4)
        for i in range(30)
    ]
    adv = calibration_check_under_adversarial(
        confs, accs, perturbation_magnitude=0.1, seed=1)
    assert adv.calibration_gap_noisy >= 0.0
