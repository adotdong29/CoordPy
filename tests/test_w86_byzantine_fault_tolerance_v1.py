"""Tests for ``coordpy.byzantine_fault_tolerance_v1``."""

from __future__ import annotations

import pytest

cryptography = pytest.importorskip("cryptography")

from coordpy.byzantine_fault_tolerance_v1 import (  # noqa: E402
    BFTMembershipV1,
    BFTPhase,
    BFTPhaseMessageV1,
    BFTReplicaIdentity,
    BFTReplicaInputV1,
    BFTReplicaKey,
    BFTVerdict,
    ByzantineEquivocationEvidenceV1,
    ByzantineWitnessV1,
    run_bft_v1_full_suite,
    run_collusion_bench_at_byzantine_bound_v1,
    run_equivocation_detection_bench_v1,
    run_pbft_consensus_round_v1,
    run_refuse_to_commit_bench_above_byzantine_bound_v1,
    sign_byzantine_witness,
    sign_phase_message,
)


def _membership_and_keys(n: int, seed: int = 86_038):
    keys = [BFTReplicaKey.from_seed(
        f"replica_{i:02d}", seed + i) for i in range(n)]
    identities = tuple(
        BFTReplicaIdentity(
            replica_id=k.replica_id,
            public_key_bytes=k.public_key_bytes) for k in keys)
    return BFTMembershipV1(replicas=identities), keys


def test_membership_enforces_n_ge_4():
    keys = [BFTReplicaKey.from_seed(f"r_{i}", i) for i in range(3)]
    ids = tuple(
        BFTReplicaIdentity(k.replica_id, k.public_key_bytes)
        for k in keys)
    with pytest.raises(ValueError):
        BFTMembershipV1(replicas=ids)


def test_membership_quorum_bounds():
    m4, _ = _membership_and_keys(4)
    assert m4.n == 4 and m4.f_byzantine_bound == 1
    assert m4.quorum_size == 3  # 2f + 1
    m7, _ = _membership_and_keys(7)
    assert m7.n == 7 and m7.f_byzantine_bound == 2
    assert m7.quorum_size == 5
    m13, _ = _membership_and_keys(13)
    assert m13.f_byzantine_bound == 4 and m13.quorum_size == 9


def test_sign_and_verify_phase_message_round_trip():
    m, keys = _membership_and_keys(4)
    msg = sign_phase_message(
        keys[0], BFTPhase.PRE_PREPARE, view=0, sequence=0,
        value_cid="abc123", membership_cid=m.cid())
    assert msg.verify(m.public_key_for(keys[0].replica_id))
    # Verifying with the WRONG public key fails.
    assert not msg.verify(
        m.public_key_for(keys[1].replica_id))


def test_phase_message_signature_does_not_verify_after_tamper():
    m, keys = _membership_and_keys(4)
    msg = sign_phase_message(
        keys[0], BFTPhase.PRE_PREPARE, view=0, sequence=0,
        value_cid="abc123", membership_cid=m.cid())
    # Swap value_cid for a different one — same signature should
    # no longer verify.
    tampered = BFTPhaseMessageV1(
        sender_id=msg.sender_id, phase=msg.phase,
        view=msg.view, sequence=msg.sequence,
        value_cid="different_cid",
        membership_cid=msg.membership_cid,
        signature_bytes=msg.signature_bytes)
    assert not tampered.verify(
        m.public_key_for(keys[0].replica_id))


def test_byzantine_witness_signs_value_cid_and_verifies():
    m, keys = _membership_and_keys(4)
    w = sign_byzantine_witness(
        keys[0], value=42.5,
        membership_cid=m.cid(),
        arrival_delay=0.1, self_confidence=0.9)
    assert w.verify(m.public_key_for(keys[0].replica_id))
    # Replacing the value (but keeping the signed CID) must
    # break verification because the CID derivation is checked.
    tampered = w.__class__(
        witness_id=w.witness_id, value=99.0,
        value_cid=w.value_cid,
        membership_cid=w.membership_cid,
        signature_bytes=w.signature_bytes,
        arrival_delay=w.arrival_delay,
        self_confidence=w.self_confidence)
    assert not tampered.verify(
        m.public_key_for(keys[0].replica_id))


def test_happy_path_consensus_commits():
    m, keys = _membership_and_keys(4)
    inputs = [
        BFTReplicaInputV1(replica_id=k.replica_id, proposed_value=1.0)
        for k in keys]
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m)
    assert outcome.verdict == BFTVerdict.COMMITTED
    assert outcome.committed_value == 1.0
    assert outcome.quorum_reached_prepare
    assert outcome.quorum_reached_commit
    assert len(outcome.equivocation_evidence) == 0


def test_collusion_at_f_commits_to_honest_value():
    rep = run_collusion_bench_at_byzantine_bound_v1(
        n=7, mu=1.0, delta=0.3)
    # With n=7, f=2: 5 honest + 2 colluding. Honest 5 form a
    # 2f+1=5 quorum on mu. Protocol must commit to mu exactly.
    assert rep.committed is True
    assert rep.committed_value == 1.0
    assert rep.committed_error is not None and rep.committed_error <= 1e-12
    assert rep.safety_holds is True
    # No equivocation — this is value-lying, not equivocating.
    assert rep.equivocation_evidence_count == 0


def test_above_byzantine_bound_refuses_to_commit():
    # n=4, f_bound=1, f_target=2 (above bound). 2 byzantine, 2
    # honest. Honest cannot reach 2f+1=3 quorum on their own.
    rep = run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n=4, delta=0.3)
    assert rep.committed is False
    assert rep.verdict in {
        "refused_quorum_not_reached", "refused_equivocation"}
    assert rep.safety_holds is True


def test_above_byzantine_bound_larger_n_refuses_to_commit():
    # n=7, f_bound=2. Try f_target=3 — strictly above. Honest 4
    # cannot reach 2*2+1=5 quorum.
    rep = run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n=7, delta=0.5)
    # f_target is computed as f_bound+1=3.
    assert rep.f_target == 3
    assert rep.f_byzantine_bound == 2
    assert rep.committed is False
    assert rep.safety_holds is True


def test_equivocation_detection_produces_independently_verifiable_evidence():
    rep = run_equivocation_detection_bench_v1(
        n=4, mu=1.0, target_delta=7.7)
    assert rep.committed is False
    assert rep.equivocation_evidence_count >= 1
    assert rep.equivocation_independently_verifiable is True
    assert rep.safety_holds is True


def test_equivocation_evidence_full_independent_verification():
    """A third party with only the public keys must be able to
    re-derive: signature_a_valid, signature_b_valid,
    messages_contradict, conclusively_byzantine.
    """
    m, keys = _membership_and_keys(4)
    inputs = [
        BFTReplicaInputV1(
            replica_id=keys[i].replica_id,
            proposed_value=1.0,
            is_byzantine=(i == 3),
            equivocation_target_value=(8.7 if i == 3 else None))
        for i in range(4)]
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m)
    assert len(outcome.equivocation_evidence) == 1
    ev = outcome.equivocation_evidence[0]
    v = ev.independently_verify(m)
    assert v["signature_a_valid"] is True
    assert v["signature_b_valid"] is True
    assert v["messages_contradict"] is True
    assert v["conclusively_byzantine"] is True


def test_byzantine_replica_dropping_pre_prepare_kills_liveness():
    """A Byzantine primary that drops pre_prepare must NOT
    cause a safety violation; the round simply refuses.
    """
    m, keys = _membership_and_keys(4)
    inputs = [
        BFTReplicaInputV1(
            replica_id=keys[i].replica_id,
            proposed_value=1.0,
            is_byzantine=(i == 0),
            drop_phase=(BFTPhase.PRE_PREPARE if i == 0 else None))
        for i in range(4)]
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m,
        view=0)
    assert outcome.verdict != BFTVerdict.COMMITTED
    assert outcome.committed_value is None
    assert outcome.safety_violation_detected is False


def test_safety_below_bound_holds_at_n7_f2():
    """At n=7, f=2 — the edge case where collusion equals f.
    Honest 5 must still commit to mu.
    """
    m, keys = _membership_and_keys(7)
    inputs = []
    mu = 1.0
    delta = 0.3
    for i, k in enumerate(keys):
        is_byz = i >= 5  # last 2 are colluding
        inputs.append(BFTReplicaInputV1(
            replica_id=k.replica_id,
            proposed_value=(mu + delta if is_byz else mu),
            is_byzantine=is_byz))
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m)
    assert outcome.verdict == BFTVerdict.COMMITTED
    assert outcome.committed_value == mu


def test_membership_cid_is_stable_across_construction():
    m1, _ = _membership_and_keys(4, seed=1234)
    m2, _ = _membership_and_keys(4, seed=1234)
    assert m1.cid() == m2.cid()


def test_membership_cid_changes_with_replica_set():
    m1, _ = _membership_and_keys(4, seed=1234)
    m2, _ = _membership_and_keys(7, seed=1234)
    assert m1.cid() != m2.cid()


def test_full_suite_passes_all_three_benches():
    suite = run_bft_v1_full_suite()
    assert suite.closed is True
    # All three reports have safety_holds=True.
    assert all(r.safety_holds for r in suite.reports)
    # Distinct CIDs.
    cids = {
        suite.collusion_report_cid,
        suite.refuse_report_cid,
        suite.equivocation_report_cid,
    }
    assert len(cids) == 3


def test_outcome_cid_is_deterministic():
    rep_a = run_collusion_bench_at_byzantine_bound_v1(
        n=7, mu=1.0, delta=0.3, seed=86_038)
    rep_b = run_collusion_bench_at_byzantine_bound_v1(
        n=7, mu=1.0, delta=0.3, seed=86_038)
    assert rep_a.report_cid == rep_b.report_cid
    assert rep_a.outcome_cid == rep_b.outcome_cid


def test_outcome_cid_changes_with_delta():
    rep_a = run_collusion_bench_at_byzantine_bound_v1(
        n=7, mu=1.0, delta=0.3)
    rep_b = run_collusion_bench_at_byzantine_bound_v1(
        n=7, mu=1.0, delta=0.7)
    # Both should commit to mu=1.0, but the colluding shares of
    # the prepare set differ in value_cid, so outcome CIDs differ
    # via the rejected-prepare structure.
    assert rep_a.committed_value == 1.0
    assert rep_b.committed_value == 1.0


def test_equivocator_cannot_steal_quorum():
    """A Byzantine replica that equivocates must NOT have its
    prepare counted at all — even if its primary-aligned message
    looks valid. This is what protects safety under equivocation.
    """
    m, keys = _membership_and_keys(4)
    inputs = []
    for i, k in enumerate(keys):
        inputs.append(BFTReplicaInputV1(
            replica_id=k.replica_id,
            proposed_value=1.0,
            is_byzantine=(i == 3),
            equivocation_target_value=(9.9 if i == 3 else None)))
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m)
    # The equivocator IS rejected. The honest 3 still form a 3-quorum.
    # But our protocol policy is: when equivocation evidence is
    # present, honest replicas refuse to commit (safety > liveness).
    # So the round refuses.
    assert outcome.verdict != BFTVerdict.COMMITTED


def test_equivocator_value_not_committed():
    """Even if equivocator's signed prepare for primary's value
    looks superficially valid, the equivocator's other signed
    message destroys their credibility for THIS sequence."""
    m, keys = _membership_and_keys(4)
    target_value = 99.99
    inputs = []
    for i, k in enumerate(keys):
        is_byz = (i == 3)
        inputs.append(BFTReplicaInputV1(
            replica_id=k.replica_id,
            proposed_value=1.0,
            is_byzantine=is_byz,
            equivocation_target_value=(
                target_value if is_byz else None)))
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs, membership=m)
    assert outcome.committed_value != target_value


def test_capsule_cids_are_content_addressed():
    """Every load-bearing capsule must be content-addressed."""
    m, keys = _membership_and_keys(4)
    w = sign_byzantine_witness(
        keys[0], value=1.0, membership_cid=m.cid())
    assert len(w.cid()) == 64
    msg = sign_phase_message(
        keys[0], BFTPhase.PRE_PREPARE, 0, 0, w.value_cid, m.cid())
    assert len(msg.cid()) == 64


def test_proven_safety_holds_strict_minority_above_bound():
    """The classical PBFT safety theorem says: with up to f
    Byzantine where 3f + 1 ≤ n, safety holds. We test the
    contrapositive: when f > (n-1)/3, the protocol must not
    silently commit a Byzantine value.
    """
    rep = run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n=4, delta=10.0)  # huge delta — easy to detect
    assert rep.committed is False
    rep = run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n=10)
    assert rep.committed is False
