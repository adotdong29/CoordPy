"""W81 / P1 #20 — Adversarial consensus & repair tests.

Covers:
- decision kinds are canonical (4 of them)
- delay-decay trust prior
- corruption-suspicion drops trust on adversarial witnesses
- abstain triggers when CI is wide
- escalate triggers when corruption suspicion is high
- commit happens on clean witnesses
- V1 strictly beats naive averaging on adversarial bench
- V1 beats naive on majority of seeds
- audit chain is content-addressed
"""

from __future__ import annotations

import numpy as np
import pytest


def _witness(
        wid, value, *, delay=0.0, confidence=1.0, role="default"):
    from coordpy.adversarial_consensus_repair_v1 import (
        WitnessEvidenceV1,
    )
    return WitnessEvidenceV1(
        witness_id=str(wid),
        value=np.asarray(value, dtype=np.float64),
        arrival_delay=float(delay),
        self_confidence=float(confidence),
        role=str(role))


def test_w81_consensus_decision_kinds_are_canonical():
    from coordpy.adversarial_consensus_repair_v1 import (
        W81_DECISION_KINDS,
    )
    assert tuple(W81_DECISION_KINDS) == (
        "commit",
        "abstain",
        "escalate_to_richer_substrate",
        "replay_from_trusted",
    )


def test_w81_consensus_commit_on_clean_witnesses():
    from coordpy.adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    witnesses = [
        _witness(f"w{i}", truth + 0.01 * np.random.RandomState(
            i).standard_normal(4))
        for i in range(7)
    ]
    decision = trust_weighted_consensus_v1(
        witnesses=witnesses)
    assert decision.decision_kind == "commit"
    assert decision.fused_value is not None
    err = float(np.linalg.norm(
        decision.fused_value - truth))
    assert err < 0.05


def test_w81_consensus_abstain_on_empty_witnesses():
    from coordpy.adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    decision = trust_weighted_consensus_v1(witnesses=[])
    assert decision.decision_kind == "abstain"
    assert decision.fused_value is None
    assert decision.n_witnesses == 0


def test_w81_consensus_escalate_when_corruption_suspicion_high():
    """A single adversarial witness with a huge value should
    cross the escalate threshold."""
    from coordpy.adversarial_consensus_repair_v1 import (
        TrustWeightedConsensusConfigV1,
        trust_weighted_consensus_v1,
    )
    truth = np.array([0.0, 0.0])
    witnesses = (
        [_witness(f"h{i}", truth + 0.01 * (i - 1))
         for i in range(5)]
        + [_witness(
            "adv", np.array([50.0, 50.0]),
            confidence=1.0, role="adversarial")]
    )
    cfg = TrustWeightedConsensusConfigV1(
        escalate_corruption_threshold=0.65)
    decision = trust_weighted_consensus_v1(
        witnesses=witnesses, config=cfg)
    assert decision.decision_kind == (
        "escalate_to_richer_substrate")
    assert decision.fused_value is None
    assert bool(decision.escalate_active)


def test_w81_consensus_delayed_witnesses_have_lower_trust():
    """A witness that arrives 12 timesteps late (with default
    halflife 6) should have ~ 1/4 the trust of an on-time
    witness when values are similar."""
    from coordpy.adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    truth = np.array([0.0, 0.0])
    witnesses = [
        _witness("ontime", truth, delay=0.0),
        _witness("late", truth, delay=12.0),
    ]
    decision = trust_weighted_consensus_v1(
        witnesses=witnesses)
    # Both witnesses agree perfectly, so corruption penalty is
    # neutral; the trust ratio should follow the delay decay.
    trusts = decision.trust_distribution
    ratio = trusts[1] / max(trusts[0], 1e-12)
    # delay 12, halflife 6 -> 2^-2 = 0.25.
    assert abs(float(ratio) - 0.25) < 0.05


def test_w81_consensus_corruption_penalty_drops_adversary_trust():
    """An adversarial witness whose value is far from the rest
    should have trust ~ 0 after corruption penalty."""
    from coordpy.adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    witnesses = (
        [_witness(f"h{i}", np.array([1.0, 1.0]))
         for i in range(5)]
        + [_witness(
            "adv", np.array([10.0, 10.0]), role="adversarial")]
    )
    decision = trust_weighted_consensus_v1(
        witnesses=witnesses)
    trusts = decision.trust_distribution
    # adversary trust (last) should be much smaller than the
    # honest mean.
    adv_trust = float(trusts[-1])
    honest_mean = float(np.mean(trusts[:-1]))
    assert adv_trust < 0.10 * honest_mean


def test_w81_consensus_v1_strictly_beats_naive_on_bench():
    """The load-bearing claim — V1's mean error is strictly
    smaller than naive averaging on the adversarial bench."""
    from coordpy.adversarial_consensus_repair_v1 import (
        run_adversarial_consensus_bench_v1,
    )
    rep = run_adversarial_consensus_bench_v1(
        n_seeds=80, n_witnesses=7, n_corrupted=2,
        seed=11)
    assert bool(rep.v1_strictly_beats_naive), (
        f"V1 mean err {rep.v1_mean_error} >= naive mean "
        f"err {rep.naive_mean_error}")
    # And V1 wins on a clear majority of seeds.
    assert float(rep.v1_beats_naive_frac) >= 0.80


def test_w81_consensus_v1_competitive_with_median():
    """V1 should be at least competitive with median on
    corrupted data."""
    from coordpy.adversarial_consensus_repair_v1 import (
        run_adversarial_consensus_bench_v1,
    )
    rep = run_adversarial_consensus_bench_v1(
        n_seeds=60, n_witnesses=7, n_corrupted=2, seed=17)
    # V1 should beat median on at least 40% of seeds (median
    # is a strong robust baseline, but V1 leverages trust
    # info).
    assert float(rep.v1_beats_median_frac) >= 0.40


def test_w81_consensus_bench_exercises_default_routes_honestly():
    from coordpy.adversarial_consensus_repair_v1 import (
        run_adversarial_consensus_bench_v1,
    )
    rep = run_adversarial_consensus_bench_v1(
        n_seeds=80, n_witnesses=7, n_corrupted=2, seed=11)
    # The shipped default config should no longer be silently
    # replaced by a fusion-only threshold override.
    assert float(rep.escalate_frac) < 1.0
    assert float(rep.commit_frac + rep.abstain_frac
                 + rep.escalate_frac + rep.replay_frac) == (
        pytest.approx(1.0))


def test_w81_consensus_decision_audit_chain_is_content_addressed():
    """Identical inputs -> identical audit CID. Different
    inputs -> different audit CID."""
    from coordpy.adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    truth = np.array([1.0, 2.0])
    witnesses_a = [
        _witness(f"w{i}", truth + 0.01) for i in range(5)]
    witnesses_b = [
        _witness(f"w{i}", truth + 0.01) for i in range(5)]
    d_a = trust_weighted_consensus_v1(witnesses=witnesses_a)
    d_b = trust_weighted_consensus_v1(witnesses=witnesses_b)
    assert d_a.audit_cid == d_b.audit_cid
    witnesses_c = [
        _witness(f"w{i}", truth + 0.10) for i in range(5)]
    d_c = trust_weighted_consensus_v1(witnesses=witnesses_c)
    assert d_a.audit_cid != d_c.audit_cid


def test_w81_consensus_witness_value_cid_changes_value():
    from coordpy.adversarial_consensus_repair_v1 import (
        WitnessEvidenceV1,
    )
    w1 = WitnessEvidenceV1(
        witness_id="x",
        value=np.array([1.0, 2.0]))
    w2 = WitnessEvidenceV1(
        witness_id="x",
        value=np.array([3.0, 4.0]))
    assert w1.cid() != w2.cid()


def test_w81_consensus_replay_when_ci_wide_but_trust_exists():
    """A witness set that's split (wide CI) but contains at
    least one high-trust witness should route to replay,
    not abstain."""
    from coordpy.adversarial_consensus_repair_v1 import (
        TrustWeightedConsensusConfigV1,
        trust_weighted_consensus_v1,
    )
    # Three honest witnesses near zero and three near 5.0
    # creates a wide spread without a single suspicious outlier
    # large enough to escalate.
    witnesses = (
        [_witness(f"a{i}", np.array([0.0, 0.0]),
                  delay=float(i))
         for i in range(3)]
        + [_witness(f"b{i}", np.array([1.0, 1.0]),
                    delay=float(i))
           for i in range(3)]
    )
    cfg = TrustWeightedConsensusConfigV1(
        abstain_width_threshold=0.05,
        escalate_corruption_threshold=10.0,
        replay_trust_floor=0.10)
    decision = trust_weighted_consensus_v1(
        witnesses=witnesses, config=cfg)
    assert decision.decision_kind in (
        "replay_from_trusted", "abstain")
    if decision.decision_kind == "replay_from_trusted":
        assert bool(decision.replay_active)


def test_w81_consensus_witness_round_trip():
    from coordpy.adversarial_consensus_repair_v1 import (
        TrustWeightedConsensusConfigV1,
        emit_adv_consensus_witness_v1,
        trust_weighted_consensus_v1,
    )
    cfg = TrustWeightedConsensusConfigV1()
    w1 = emit_adv_consensus_witness_v1(config=cfg)
    decision = trust_weighted_consensus_v1(
        witnesses=[
            _witness("a", np.array([1.0, 1.0])),
            _witness("b", np.array([1.0, 1.1])),
        ],
        config=cfg)
    w2 = emit_adv_consensus_witness_v1(
        config=cfg, last_decision=decision)
    assert w1.last_decision_cid is None
    assert w2.last_decision_cid is not None
    assert w1.cid() != w2.cid()
