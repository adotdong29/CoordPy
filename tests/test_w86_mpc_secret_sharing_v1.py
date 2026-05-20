"""Tests for ``coordpy.mpc_secret_sharing_v1``."""

from __future__ import annotations

import pytest

from coordpy.mpc_secret_sharing_v1 import (
    CrossOrgMPCBenchReportV1,
    MPCAverageOutcomeV1,
    PedersenParamsV1,
    SchnorrProofV1,
    SecretShareCapsuleV1,
    ShamirSchemeV1,
    ThresholdReconstructorV1,
    W86_MPC_V1_PRIME,
    default_pedersen_params_v1,
    make_schnorr_proof_v1,
    pedersen_commit_v1,
    run_cross_org_mpc_bench_v1,
    run_mpc_average_v1,
    split_secret_v1,
    verify_schnorr_proof_v1,
)


def test_shamir_round_trip_basic():
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3, n_shares=5)
    shares = split_secret_v1(42, scheme)
    rec = ThresholdReconstructorV1(scheme=scheme)
    # Any 3 shares recover.
    assert rec.reconstruct(list(shares)[:3]) == 42
    assert rec.reconstruct(list(shares)[1:4]) == 42
    assert rec.reconstruct(list(shares)[2:5]) == 42


def test_shamir_below_threshold_does_not_recover():
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3, n_shares=5)
    shares = split_secret_v1(42, scheme)
    rec = ThresholdReconstructorV1(scheme=scheme)
    with pytest.raises(ValueError):
        rec.reconstruct(list(shares)[:2])  # only 2 of 3 needed


def test_shamir_secret_out_of_range_rejected():
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3, n_shares=5)
    with pytest.raises(ValueError):
        split_secret_v1(W86_MPC_V1_PRIME + 1, scheme)


def test_share_cid_64_chars():
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=2, n_shares=4)
    shares = split_secret_v1(7, scheme)
    for s in shares:
        assert len(s.cid()) == 64


def test_pedersen_commit_binding():
    params = default_pedersen_params_v1()
    c1 = pedersen_commit_v1(params, 10, 11)
    c2 = pedersen_commit_v1(params, 10, 11)
    assert c1 == c2
    c3 = pedersen_commit_v1(params, 10, 12)
    assert c1 != c3


def test_schnorr_proof_round_trip():
    params = default_pedersen_params_v1()
    proof = make_schnorr_proof_v1(params, m=42, r=99)
    assert verify_schnorr_proof_v1(params, proof) is True


def test_schnorr_proof_forged_commitment_rejected():
    params = default_pedersen_params_v1()
    proof = make_schnorr_proof_v1(params, m=42, r=99)
    import dataclasses
    forged = dataclasses.replace(
        proof,
        commitment_c=(proof.commitment_c + 1) % params.prime)
    assert verify_schnorr_proof_v1(params, forged) is False


def test_schnorr_proof_forged_t_rejected():
    params = default_pedersen_params_v1()
    proof = make_schnorr_proof_v1(params, m=42, r=99)
    import dataclasses
    forged = dataclasses.replace(
        proof, t=(proof.t + 1) % params.prime)
    assert verify_schnorr_proof_v1(params, forged) is False


def test_schnorr_proof_wrong_pedersen_params_rejected():
    p1 = default_pedersen_params_v1()
    # Build a "different" params with the same prime but
    # different h.
    p2 = PedersenParamsV1(
        prime=p1.prime, g=p1.g, h=(p1.h + 1) % p1.prime)
    proof = make_schnorr_proof_v1(p1, m=42, r=99)
    # Verifying under p2's params (different CID) refuses.
    assert verify_schnorr_proof_v1(p2, proof) is False


def test_mpc_average_recovers_correct_sum():
    secrets_list = [10, 20, 30, 40, 50]
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3,
        n_shares=len(secrets_list))
    out = run_mpc_average_v1(
        party_secrets=secrets_list, scheme=scheme)
    assert out.secret_sum == sum(secrets_list)
    assert out.average_value == sum(secrets_list) / len(secrets_list)
    assert out.all_share_proofs_valid is True
    assert out.forged_share_rejected is True


def test_cross_org_bench_meets_dod():
    rep = run_cross_org_mpc_bench_v1()
    assert rep.sum_matches is True
    assert rep.no_cleartext_secrets_crossed_orgs is True
    assert rep.drop_out_test_works is True
    assert rep.all_proofs_valid is True
    assert rep.forged_share_rejected is True
    assert rep.insufficient_shares_recovers_nothing is True
    assert rep.n_orgs == 2


def test_cross_org_bench_report_deterministic():
    r1 = run_cross_org_mpc_bench_v1(seed=86_040)
    r2 = run_cross_org_mpc_bench_v1(seed=86_040)
    assert r1.report_cid == r2.report_cid


def test_shamir_scheme_cid_stable():
    s1 = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3, n_shares=5)
    s2 = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3, n_shares=5)
    assert s1.cid() == s2.cid()


def test_threshold_must_be_at_least_two():
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=1, n_shares=3)
    with pytest.raises(ValueError):
        split_secret_v1(7, scheme)


def test_drop_out_works_when_threshold_less_than_n():
    """k < n: with `n - 1` shares + threshold k, reconstructable.
    """
    secrets_list = [1, 2, 3, 4, 5]
    scheme = ShamirSchemeV1(
        prime=W86_MPC_V1_PRIME, threshold=3,
        n_shares=len(secrets_list))
    shares = split_secret_v1(15, scheme)  # use sum = 15
    rec = ThresholdReconstructorV1(scheme=scheme)
    # Drop the last share; reconstruct from 4 of 5.
    assert rec.reconstruct(list(shares)[:-1]) == 15
    assert rec.reconstruct(list(shares)[:3]) == 15
