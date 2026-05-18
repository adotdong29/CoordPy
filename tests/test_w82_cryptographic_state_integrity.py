"""W82 / P2 #18 — Cryptographic state integrity tests.

Covers:
- snapshot is content-addressed
- payload_cid is independently recomputable from payload_bytes
- Merkle tree root_cid is content-addressed and reproducible
- HMAC signature verifies for keyed snapshots
- bad HMAC is detected
- corruption (byte flip) is detected by re-hashing payload
- provenance violation (wrong parent_cid) is detected
- rollback anchor restores to a snapshot under integrity check
- branch-merge witness verifies for legitimate merges
- branch-merge witness rejects unsafe merges (parent mismatch
  or unknown CID)
- end-to-end corruption-rollback-merge bench passes
"""

from __future__ import annotations


def test_w82_integrity_snapshot_content_addressed():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
    )
    a = build_state_snapshot_v1(
        snapshot_id="s0", parent_cid="root",
        payload_bytes=b"hello",
        timestamp_ns=1_000_000_000,
        hmac_key=b"k")
    b = build_state_snapshot_v1(
        snapshot_id="s0", parent_cid="root",
        payload_bytes=b"hello",
        timestamp_ns=1_000_000_000,
        hmac_key=b"k")
    c = build_state_snapshot_v1(
        snapshot_id="s0", parent_cid="root",
        payload_bytes=b"hello!",
        timestamp_ns=1_000_000_000,
        hmac_key=b"k")
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_integrity_payload_cid_recomputable():
    import hashlib
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
    )
    s = build_state_snapshot_v1(
        snapshot_id="x", parent_cid="root",
        payload_bytes=b"payload",
        timestamp_ns=1,
        hmac_key=None)
    assert s.payload_cid == hashlib.sha256(
        b"payload").hexdigest()
    assert s.re_hash_payload() == s.payload_cid


def test_w82_integrity_merkle_root_cid_reproducible():
    from coordpy.cryptographic_state_integrity_v1 import (
        MerkleHashTreeV1,
    )
    cids = tuple(f"cid_{i}" for i in range(7))
    t_a = MerkleHashTreeV1.from_snapshot_cids(cids)
    t_b = MerkleHashTreeV1.from_snapshot_cids(cids)
    assert t_a.root_cid == t_b.root_cid
    assert t_a.cid() == t_b.cid()


def test_w82_integrity_merkle_inclusion_path_size_log_n():
    from coordpy.cryptographic_state_integrity_v1 import (
        MerkleHashTreeV1,
        W82_INTEGRITY_DEFAULT_MERKLE_FANOUT,
    )
    cids = tuple(f"cid_{i}" for i in range(64))
    t = MerkleHashTreeV1.from_snapshot_cids(cids, fanout=4)
    path = t.inclusion_path(0)
    # 64 leaves, fanout 4 -> levels: 64, 16, 4, 1 → path
    # length is 3 (one node per non-root level)
    assert len(path) == 3


def test_w82_integrity_hmac_signature_verifies_on_clean_snapshot():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
        verify_snapshot_integrity_v1,
        IntegrityVerdict,
    )
    s = build_state_snapshot_v1(
        snapshot_id="x", parent_cid="root",
        payload_bytes=b"x" * 64,
        timestamp_ns=10,
        hmac_key=b"k")
    rep = verify_snapshot_integrity_v1(
        snapshot=s, chain_root_cid="root",
        expected_parent_cid="root",
        hmac_key=b"k")
    assert rep.verdict == IntegrityVerdict.OK.value


def test_w82_integrity_bad_hmac_key_detected():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
        verify_snapshot_integrity_v1,
        IntegrityVerdict,
    )
    s = build_state_snapshot_v1(
        snapshot_id="x", parent_cid="root",
        payload_bytes=b"x" * 64,
        timestamp_ns=10,
        hmac_key=b"key_a")
    rep = verify_snapshot_integrity_v1(
        snapshot=s, chain_root_cid="root",
        expected_parent_cid="root",
        hmac_key=b"key_b")  # wrong key
    assert rep.verdict == IntegrityVerdict.BAD_SIGNATURE.value


def test_w82_integrity_silent_corruption_detected():
    """Flip one byte of payload_bytes; verifier must catch it."""
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
        simulate_silent_corruption_v1,
        verify_snapshot_integrity_v1,
        IntegrityVerdict,
    )
    s = build_state_snapshot_v1(
        snapshot_id="x", parent_cid="root",
        payload_bytes=b"hello world",
        timestamp_ns=1,
        hmac_key=None)
    bad = simulate_silent_corruption_v1(s)
    rep = verify_snapshot_integrity_v1(
        snapshot=bad, chain_root_cid="root",
        expected_parent_cid="root")
    assert rep.verdict == IntegrityVerdict.CORRUPT.value


def test_w82_integrity_provenance_violation_detected():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
        verify_snapshot_integrity_v1,
        IntegrityVerdict,
    )
    s = build_state_snapshot_v1(
        snapshot_id="x", parent_cid="wrong_parent",
        payload_bytes=b"data",
        timestamp_ns=1,
        hmac_key=None)
    rep = verify_snapshot_integrity_v1(
        snapshot=s, chain_root_cid="root",
        expected_parent_cid="real_parent")
    assert rep.verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_clean_chain_verifies_end_to_end():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        verify_chain_integrity_v1,
        IntegrityVerdict,
    )
    chain = build_clean_snapshot_chain_v1(
        n=12, hmac_key=b"test-key")
    reports = verify_chain_integrity_v1(
        snapshots=chain, hmac_key=b"test-key")
    for r in reports:
        assert r.verdict == IntegrityVerdict.OK.value


def test_w82_integrity_rollback_anchor_round_trips():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        build_rollback_anchor_v1,
        execute_rollback_v1,
        MerkleHashTreeV1,
        IntegrityVerdict,
    )
    chain = build_clean_snapshot_chain_v1(n=6)
    tree = MerkleHashTreeV1.from_snapshot_cids(
        [s.cid() for s in chain])
    anchor = build_rollback_anchor_v1(
        label="anchor_3", snapshot=chain[3],
        chain_root_cid=str(tree.root_cid),
        created_at_ns=99)
    snap, rep = execute_rollback_v1(
        anchor=anchor, snapshots=chain)
    assert snap is not None
    assert snap.cid() == chain[3].cid()
    assert rep.verdict == IntegrityVerdict.OK.value


def test_w82_integrity_rollback_to_missing_anchor_fails():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        RollbackAnchorV1,
        execute_rollback_v1,
        W82_INTEGRITY_V1_SCHEMA_VERSION,
        IntegrityVerdict,
    )
    chain = build_clean_snapshot_chain_v1(n=6)
    anchor = RollbackAnchorV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        label="missing", snapshot_cid="0" * 64,
        chain_root_cid="root", created_at_ns=99)
    snap, rep = execute_rollback_v1(
        anchor=anchor, snapshots=chain)
    assert snap is None
    assert rep.verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_branch_merge_witness_clean():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        build_state_snapshot_v1,
        build_branch_merge_witness_v1,
        verify_branch_merge_witness_v1,
        IntegrityVerdict,
    )
    base = build_clean_snapshot_chain_v1(n=3)
    common = base[2]
    a_tip = build_state_snapshot_v1(
        snapshot_id="a", parent_cid=common.cid(),
        payload_bytes=b"branch a", timestamp_ns=10)
    b_tip = build_state_snapshot_v1(
        snapshot_id="b", parent_cid=common.cid(),
        payload_bytes=b"branch b", timestamp_ns=11)
    merged = build_state_snapshot_v1(
        snapshot_id="m", parent_cid=a_tip.cid(),
        additional_parent_cids=(b_tip.cid(),),
        payload_bytes=(
            b'{\"merge_parent_cids\":['
            + f'\"{a_tip.cid()}\",\"{b_tip.cid()}\"'.encode("utf-8")
            + b'],\"kind\":\"merged\"}'),
        timestamp_ns=12)
    all_snaps = list(base) + [a_tip, b_tip, merged]
    w = build_branch_merge_witness_v1(
        branch_a_tip=a_tip, branch_b_tip=b_tip,
        common_ancestor=common, merged_snapshot=merged,
        all_snapshots=all_snaps, n_conflicts_resolved=2)
    assert w.integrity_verdict == IntegrityVerdict.OK.value
    rep = verify_branch_merge_witness_v1(
        witness=w, snapshots=all_snaps)
    assert rep.verdict == IntegrityVerdict.OK.value


def test_w82_integrity_branch_merge_witness_rejects_single_parent_child():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        build_state_snapshot_v1,
        build_branch_merge_witness_v1,
        verify_branch_merge_witness_v1,
        IntegrityVerdict,
    )
    base = build_clean_snapshot_chain_v1(n=3)
    common = base[2]
    a_tip = build_state_snapshot_v1(
        snapshot_id="a", parent_cid=common.cid(),
        payload_bytes=b"branch a", timestamp_ns=10)
    b_tip = build_state_snapshot_v1(
        snapshot_id="b", parent_cid=common.cid(),
        payload_bytes=b"branch b", timestamp_ns=11)
    merged = build_state_snapshot_v1(
        snapshot_id="m", parent_cid=a_tip.cid(),
        payload_bytes=b"merged-without-proof", timestamp_ns=12)
    all_snaps = list(base) + [a_tip, b_tip, merged]
    w = build_branch_merge_witness_v1(
        branch_a_tip=a_tip, branch_b_tip=b_tip,
        common_ancestor=common, merged_snapshot=merged,
        all_snapshots=all_snaps, n_conflicts_resolved=0)
    assert w.integrity_verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)
    rep = verify_branch_merge_witness_v1(
        witness=w, snapshots=all_snaps)
    assert rep.verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_unsafe_merge_witness_rejected():
    """A merge whose merged snapshot points at a parent that
    is neither of the two branch tips must be flagged."""
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        build_state_snapshot_v1,
        build_branch_merge_witness_v1,
        IntegrityVerdict,
    )
    base = build_clean_snapshot_chain_v1(n=3)
    common = base[2]
    a_tip = build_state_snapshot_v1(
        snapshot_id="a", parent_cid=common.cid(),
        payload_bytes=b"a", timestamp_ns=10)
    b_tip = build_state_snapshot_v1(
        snapshot_id="b", parent_cid=common.cid(),
        payload_bytes=b"b", timestamp_ns=11)
    bad_merged = build_state_snapshot_v1(
        snapshot_id="bad", parent_cid="ghost_parent",
        payload_bytes=b"bad", timestamp_ns=12)
    all_snaps = list(base) + [a_tip, b_tip, bad_merged]
    w = build_branch_merge_witness_v1(
        branch_a_tip=a_tip, branch_b_tip=b_tip,
        common_ancestor=common, merged_snapshot=bad_merged,
        all_snapshots=all_snaps, n_conflicts_resolved=0)
    assert w.integrity_verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_branch_merge_witness_rejects_unknown_cids():
    from coordpy.cryptographic_state_integrity_v1 import (
        build_clean_snapshot_chain_v1,
        build_state_snapshot_v1,
        build_branch_merge_witness_v1,
        verify_branch_merge_witness_v1,
        IntegrityVerdict,
        W82_INTEGRITY_V1_SCHEMA_VERSION,
        BranchMergeWitnessV1,
        MerkleHashTreeV1,
    )
    base = build_clean_snapshot_chain_v1(n=3)
    # Verifier given a snapshot list that doesn't include the
    # cited CIDs.
    fake_witness = BranchMergeWitnessV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        branch_a_tip_cid="x" * 64,
        branch_b_tip_cid="y" * 64,
        common_ancestor_cid="z" * 64,
        merged_snapshot_cid="w" * 64,
        merge_tree_root_cid=str(
            MerkleHashTreeV1.from_snapshot_cids(
                [s.cid() for s in base]).root_cid),
        n_conflicts_resolved=0,
        integrity_verdict=IntegrityVerdict.OK.value)
    rep = verify_branch_merge_witness_v1(
        witness=fake_witness, snapshots=base)
    assert rep.verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_end_to_end_corruption_rollback_merge():
    """The load-bearing W82 P2 #18 bench."""
    from coordpy.cryptographic_state_integrity_v1 import (
        run_corruption_rollback_merge_bench_v1,
        IntegrityVerdict,
    )
    rep = run_corruption_rollback_merge_bench_v1()
    # Clean chain is all OK
    assert all(
        v == IntegrityVerdict.OK.value
        for v in rep.chain_clean_verdicts)
    # Corruption was detected
    assert rep.corruption_detected is True
    # Rollback to pre-corruption anchor succeeded
    assert rep.rollback_after_corruption_ok is True
    # Clean merge OK
    assert rep.merge_clean_verdict == IntegrityVerdict.OK.value
    # Unsafe merge flagged
    assert rep.merge_unsafe_verdict == (
        IntegrityVerdict.PROVENANCE_VIOLATION.value)


def test_w82_integrity_bench_deterministic():
    from coordpy.cryptographic_state_integrity_v1 import (
        run_corruption_rollback_merge_bench_v1,
    )
    a = run_corruption_rollback_merge_bench_v1(
        n=4, seed=7, corrupt_index=2)
    b = run_corruption_rollback_merge_bench_v1(
        n=4, seed=7, corrupt_index=2)
    assert a.cid() == b.cid()


def test_w82_integrity_bench_cid_changes_with_seed():
    from coordpy.cryptographic_state_integrity_v1 import (
        run_corruption_rollback_merge_bench_v1,
    )
    a = run_corruption_rollback_merge_bench_v1(seed=1)
    b = run_corruption_rollback_merge_bench_v1(seed=2)
    assert a.cid() != b.cid()


def test_w82_integrity_unsigned_snapshot_verifier_returns_unsigned():
    """If a verifier is invoked with an HMAC key but the
    snapshot is unsigned, the verdict is UNSIGNED (informational)."""
    from coordpy.cryptographic_state_integrity_v1 import (
        build_state_snapshot_v1,
        verify_snapshot_integrity_v1,
        IntegrityVerdict,
    )
    s = build_state_snapshot_v1(
        snapshot_id="u", parent_cid="root",
        payload_bytes=b"unsigned",
        timestamp_ns=1,
        hmac_key=None)  # explicitly unsigned
    rep = verify_snapshot_integrity_v1(
        snapshot=s, chain_root_cid="root",
        expected_parent_cid="root",
        hmac_key=b"key_provided")
    assert rep.verdict == IntegrityVerdict.UNSIGNED.value


def test_w82_integrity_all_five_verdicts_enumerate():
    from coordpy.cryptographic_state_integrity_v1 import (
        W82_INTEGRITY_VERDICTS,
        IntegrityVerdict,
    )
    assert set(W82_INTEGRITY_VERDICTS) == {
        IntegrityVerdict.OK.value,
        IntegrityVerdict.CORRUPT.value,
        IntegrityVerdict.PROVENANCE_VIOLATION.value,
        IntegrityVerdict.BAD_SIGNATURE.value,
        IntegrityVerdict.UNSIGNED.value,
    }


def test_w82_integrity_aware_actions_enumerate():
    from coordpy.cryptographic_state_integrity_v1 import (
        W82_INTEGRITY_AWARE_ACTIONS,
        IntegrityAwareAction,
    )
    assert set(W82_INTEGRITY_AWARE_ACTIONS) == {
        IntegrityAwareAction.COMMIT.value,
        IntegrityAwareAction.ABSTAIN.value,
        IntegrityAwareAction.ROLLBACK.value,
        IntegrityAwareAction.ESCALATE.value,
    }


def test_w82_integrity_aware_fallback_decision_maps_each_verdict():
    """Closes P2 #18 'integrity-aware abstain/fallback
    decisions' candidate direction: each integrity verdict
    maps to a recommended action."""
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
        IntegrityAwareAction,
        integrity_aware_fallback_decision_v1,
    )
    # OK → COMMIT
    assert integrity_aware_fallback_decision_v1(
        IntegrityVerdict.OK.value
    ).recommended_action == IntegrityAwareAction.COMMIT.value
    # CORRUPT → ROLLBACK
    assert integrity_aware_fallback_decision_v1(
        IntegrityVerdict.CORRUPT.value
    ).recommended_action == IntegrityAwareAction.ROLLBACK.value
    # PROVENANCE_VIOLATION → ABSTAIN
    assert integrity_aware_fallback_decision_v1(
        IntegrityVerdict.PROVENANCE_VIOLATION.value
    ).recommended_action == IntegrityAwareAction.ABSTAIN.value
    # BAD_SIGNATURE → ESCALATE
    assert integrity_aware_fallback_decision_v1(
        IntegrityVerdict.BAD_SIGNATURE.value
    ).recommended_action == IntegrityAwareAction.ESCALATE.value
    # UNSIGNED → ABSTAIN
    assert integrity_aware_fallback_decision_v1(
        IntegrityVerdict.UNSIGNED.value
    ).recommended_action == IntegrityAwareAction.ABSTAIN.value


def test_w82_integrity_aware_fallback_decision_content_addressed():
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
        integrity_aware_fallback_decision_v1,
    )
    a = integrity_aware_fallback_decision_v1(
        IntegrityVerdict.CORRUPT.value)
    b = integrity_aware_fallback_decision_v1(
        IntegrityVerdict.CORRUPT.value)
    c = integrity_aware_fallback_decision_v1(
        IntegrityVerdict.OK.value)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_integrity_aware_fallback_unknown_verdict_fails_safe():
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityAwareAction,
        integrity_aware_fallback_decision_v1,
    )
    d = integrity_aware_fallback_decision_v1("not_a_real_verdict")
    # Fail safe: unknown verdicts → ABSTAIN
    assert d.recommended_action == (
        IntegrityAwareAction.ABSTAIN.value)
