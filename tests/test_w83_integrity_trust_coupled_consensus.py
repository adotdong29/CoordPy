"""W83 — integrity-trust-coupled consensus tests."""

from __future__ import annotations


def test_w83_itc_decision_kinds_and_config_cid():
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        IntegrityTrustCoupledConsensusConfigV1,
        W83_DEFAULT_INTEGRITY_TRUST_PENALTIES,
    )
    cfg = IntegrityTrustCoupledConsensusConfigV1()
    assert len(cfg.cid()) == 64
    # Verdict penalty table covers all five verdicts.
    assert len(W83_DEFAULT_INTEGRITY_TRUST_PENALTIES) == 5
    assert (W83_DEFAULT_INTEGRITY_TRUST_PENALTIES["ok"]
            > W83_DEFAULT_INTEGRITY_TRUST_PENALTIES["corrupt"])
    assert (
        W83_DEFAULT_INTEGRITY_TRUST_PENALTIES["bad_signature"]
        <= 0.1)


def test_w83_itc_consensus_hard_drops_bad_signature():
    import numpy as np
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        IntegrityTrustCoupledConsensusConfigV1,
        IntegrityVerifiedWitnessEvidenceV1,
        integrity_trust_coupled_consensus_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    mu = np.array([1.0, 2.0, 3.0])
    witnesses = [
        IntegrityVerifiedWitnessEvidenceV1(
            witness_id="ok1", value=mu + 0.01,
            integrity_verdict=IntegrityVerdict.OK.value),
        IntegrityVerifiedWitnessEvidenceV1(
            witness_id="ok2", value=mu - 0.01,
            integrity_verdict=IntegrityVerdict.OK.value),
        IntegrityVerifiedWitnessEvidenceV1(
            witness_id="bad", value=mu + 0.05,
            integrity_verdict=(
                IntegrityVerdict.BAD_SIGNATURE.value)),
    ]
    out = integrity_trust_coupled_consensus_v1(
        witnesses=witnesses,
        config=IntegrityTrustCoupledConsensusConfigV1())
    # The bad witness was dropped.
    assert int(out.integrity_witnesses_dropped) == 1
    # Trust distribution length matches the original witness
    # count (with zero entries for dropped).
    assert len(out.integrity_adjusted_trust) == 3
    # The trust of the bad witness is 0 after hard-drop.
    assert float(out.integrity_adjusted_trust[2]) == 0.0


def test_w83_itc_consensus_beats_w81_on_mean_error():
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        run_integrity_trust_coupled_bench_v1,
    )
    rep = run_integrity_trust_coupled_bench_v1(
        n_seeds=60, n_witnesses=7,
        n_stealth_tampered=2, n_obvious_corrupt=1)
    assert bool(rep.w83_beats_w81_on_error), rep.to_dict()


def test_w83_itc_consensus_refuses_to_commit_under_many_tampered():
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        run_integrity_trust_coupled_bench_v1,
    )
    rep = run_integrity_trust_coupled_bench_v1(
        n_seeds=60, n_witnesses=7,
        n_stealth_tampered=2, n_obvious_corrupt=1)
    assert bool(
        rep.w83_refuses_to_commit_under_many_tampered), (
            rep.to_dict())


def test_w83_itc_consensus_audit_chain_content_addressed():
    import numpy as np
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        IntegrityTrustCoupledConsensusConfigV1,
        IntegrityVerifiedWitnessEvidenceV1,
        integrity_trust_coupled_consensus_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    mu = np.array([0.0, 1.0])
    cfg = IntegrityTrustCoupledConsensusConfigV1()
    ws_a = [
        IntegrityVerifiedWitnessEvidenceV1(
            witness_id=f"w{i}", value=mu + 0.01 * i,
            integrity_verdict=IntegrityVerdict.OK.value)
        for i in range(3)]
    ws_b = ws_a  # same witnesses
    a = integrity_trust_coupled_consensus_v1(
        witnesses=ws_a, config=cfg)
    b = integrity_trust_coupled_consensus_v1(
        witnesses=ws_b, config=cfg)
    assert str(a.integrity_audit_cid) == str(
        b.integrity_audit_cid)
    # Changing one witness value should change the audit CID.
    ws_c = list(ws_a)
    ws_c[0] = IntegrityVerifiedWitnessEvidenceV1(
        witness_id="w0", value=mu + 99.0,
        integrity_verdict=IntegrityVerdict.OK.value)
    c = integrity_trust_coupled_consensus_v1(
        witnesses=ws_c, config=cfg)
    assert str(c.integrity_audit_cid) != str(
        a.integrity_audit_cid)


def test_w83_itc_consensus_witness_emitted():
    from coordpy.integrity_trust_coupled_consensus_v1 import (
        IntegrityTrustCoupledConsensusConfigV1,
        emit_integrity_trust_coupled_consensus_witness_v1,
        run_integrity_trust_coupled_bench_v1,
    )
    rep = run_integrity_trust_coupled_bench_v1(n_seeds=20)
    cfg = IntegrityTrustCoupledConsensusConfigV1()
    w = emit_integrity_trust_coupled_consensus_witness_v1(
        config=cfg, bench=rep)
    assert len(w.cid()) == 64
