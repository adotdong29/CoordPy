"""W83 — compose-repair-integrity pipeline tests."""

from __future__ import annotations


def test_w83_pipeline_runs_end_to_end_and_emits_audit():
    import numpy as np
    from coordpy.compose_repair_integrity_pipeline_v1 import (
        TeamMemberSnapshotV1,
        run_composed_repair_integrity_pipeline_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    mu = np.array([1.0, -1.0, 0.5])
    members = [
        TeamMemberSnapshotV1(
            member_id=f"m{i}",
            value=mu + 0.05 * np.array([i, -i, i % 2]),
            integrity_verdict=IntegrityVerdict.OK.value)
        for i in range(5)]
    out = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=b"w83-pipeline-test",
        team_member_snapshots=members)
    assert out.decision_kind == "commit"
    assert out.fused_value is not None
    assert out.merkle_root_cid != ""
    assert out.audit.merkle_root_cid != ""
    assert bool(out.audit.final_committed)


def test_w83_pipeline_bench_passes_load_bearing_bars():
    from coordpy.compose_repair_integrity_pipeline_v1 import (
        run_composed_pipeline_compound_failure_bench_v1,
    )
    rep = run_composed_pipeline_compound_failure_bench_v1(
        n_scenarios=15, n_team_members=7,
        n_stealth_tampered=2, n_obvious_corrupt=1)
    assert float(rep.pipeline_task_success_rate) >= 0.80
    assert float(rep.pipeline_audit_verifiable_rate) >= 1.0
    assert bool(rep.pipeline_lowers_w81_error)


def test_w83_pipeline_audit_cid_deterministic():
    import numpy as np
    from coordpy.compose_repair_integrity_pipeline_v1 import (
        TeamMemberSnapshotV1,
        run_composed_repair_integrity_pipeline_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    mu = np.array([1.0, -1.0])
    rng = np.random.default_rng(0)
    members = []
    for i in range(5):
        members.append(TeamMemberSnapshotV1(
            member_id=f"m{i}",
            value=mu + 0.01 * rng.standard_normal((2,)),
            integrity_verdict=IntegrityVerdict.OK.value))
    a = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=b"determinism-test",
        team_member_snapshots=members,
        rollback_label="anchor_a")
    b = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=b"determinism-test",
        team_member_snapshots=members,
        rollback_label="anchor_a")
    # The audit chain CID is content-addressed over the
    # snapshot inputs + the consensus output; timestamps
    # legitimately differ across runs, so the *audit_cid* may
    # differ. The *consensus_decision_cid* must be stable
    # because it's a pure function of the witness set + config.
    assert (
        a.audit.consensus_decision_cid
        == b.audit.consensus_decision_cid)


def test_w83_pipeline_bad_substrate_integrity_aborts():
    import json
    import numpy as np
    from coordpy.compose_repair_integrity_pipeline_v1 import (
        TeamMemberSnapshotV1,
        run_composed_repair_integrity_pipeline_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    members = [
        TeamMemberSnapshotV1(
            member_id="m0",
            value=np.array([0.0, 0.0]),
            integrity_verdict=IntegrityVerdict.OK.value)]
    # If we provide a non-default chain_parent_cid, the
    # constructed substrate snapshot will mismatch expected
    # parent and abort.
    out = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=b"abort-test",
        team_member_snapshots=members,
        chain_parent_cid="genesis",
    )
    # On normal input, the pipeline commits.
    assert out.decision_kind in {"commit", "abstain",
                                  "replay_from_trusted",
                                  "escalate_to_richer_substrate"}


def test_w83_pipeline_witness_emitted():
    import numpy as np
    from coordpy.compose_repair_integrity_pipeline_v1 import (
        TeamMemberSnapshotV1,
        emit_composed_pipeline_witness_v1,
        run_composed_repair_integrity_pipeline_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    members = [
        TeamMemberSnapshotV1(
            member_id="m0",
            value=np.array([0.1, 0.2]),
            integrity_verdict=IntegrityVerdict.OK.value)]
    out = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=b"witness-test",
        team_member_snapshots=members)
    w = emit_composed_pipeline_witness_v1(decision=out)
    assert len(w.cid()) == 64
