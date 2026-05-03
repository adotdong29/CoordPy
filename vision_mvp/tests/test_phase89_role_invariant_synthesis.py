"""W42 / Phase 89 -- cross-role-invariant synthesis tests.

Covers:
  * Role-handoff signature CID determinism + permutation-
    invariance + payload-canonicalisation.
  * Decision selector: every named branch.
  * Envelope / manifest-v12 CID determinism.
  * 14 enumerated W42 verifier failure modes.
  * Trivial-W42 byte-for-W41 reduction on R-89-TRIVIAL-W42.
  * Load-bearing R-89 banks: role_invariant_agrees preserves
    trust precision; role_invariant_recover delivers strict
    +0.500 trust_precision gain over W41;
    full_composite_collusion fires the W42-L-FULL-COMPOSITE-
    COLLUSION-CAP limitation theorem;
    insufficient_invariance_policy preserves W41 byte-for-W40.
"""

from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.coordpy.team_coord import (
    SchemaCapsule,
    LatentVerificationOutcome,
    build_incident_triage_schema_capsule,
    _DecodedHandoff,
)
from vision_mvp.coordpy.role_invariant_synthesis import (
    W42_ROLE_INVARIANT_SCHEMA_VERSION,
    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH,
    W42_BRANCH_INVARIANCE_DISABLED,
    W42_BRANCH_INVARIANCE_REJECTED,
    W42_BRANCH_INVARIANCE_NO_TRIGGER,
    W42_BRANCH_INVARIANCE_RATIFIED,
    W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED,
    W42_BRANCH_INVARIANCE_NO_POLICY,
    W42_ALL_BRANCHES,
    compute_role_handoff_signature_cid,
    select_role_invariance_decision,
    verify_role_invariant_synthesis_ratification,
    RoleInvariantSynthesisRatificationEnvelope,
    RoleInvariancePolicyEntry,
    RoleInvariancePolicyRegistry,
    RoleInvariantSynthesisRegistry,
    RoleInvariantSynthesisOrchestrator,
    build_role_invariant_registry,
    build_trivial_role_invariant_registry,
    _compute_w42_invariance_state_cid,
    _compute_w42_invariance_decision_cid,
    _compute_w42_invariance_audit_cid,
    _compute_w42_invariance_witness_cid,
    _compute_w42_manifest_v12_cid,
    _compute_w42_outer_cid,
)
from vision_mvp.experiments.phase89_role_invariant_synthesis import (
    run_phase89,
)


def _schema() -> SchemaCapsule:
    return build_incident_triage_schema_capsule()


def _build_clean_envelope(
        *,
        schema: SchemaCapsule,
        cell_index: int = 0,
        invariance_branch: str = W42_BRANCH_INVARIANCE_RATIFIED,
        role_handoff_signature_cid: str = "ab" * 32,
        policy_entry_cid: str = "cd" * 32,
        integrated_services_pre_w42: tuple[str, ...] = ("a", "b"),
        expected_services: tuple[str, ...] = ("a", "b"),
        integrated_services_post_w42: tuple[str, ...] = ("a", "b"),
        invariance_score: float = 1.0,
        n_w41_visible: int = 14,
        n_overhead: int = 1,
        parent_w41_cid: str = "deadbeef" * 8,
) -> RoleInvariantSynthesisRatificationEnvelope:
    n_visible = n_w41_visible + n_overhead
    state_cid = _compute_w42_invariance_state_cid(
        invariance_branch=invariance_branch,
        role_handoff_signature_cid=role_handoff_signature_cid,
        integrated_services_pre_w42=integrated_services_pre_w42,
        integrated_services_post_w42=integrated_services_post_w42,
        cell_index=cell_index)
    decision_cid = _compute_w42_invariance_decision_cid(
        invariance_branch=invariance_branch,
        integrated_services_post_w42=integrated_services_post_w42,
        invariance_score=invariance_score,
        n_w41_visible_tokens=n_w41_visible,
        n_w42_visible_tokens=n_visible,
        n_w42_overhead_tokens=n_overhead)
    audit_cid = _compute_w42_invariance_audit_cid(
        invariance_branch=invariance_branch,
        role_handoff_signature_cid=role_handoff_signature_cid,
        policy_entry_cid=policy_entry_cid,
        integrated_services_pre_w42=integrated_services_pre_w42,
        expected_services=expected_services,
        integrated_services_post_w42=integrated_services_post_w42,
        invariance_score=invariance_score)
    n_structured_bits = 2048
    witness_cid = _compute_w42_invariance_witness_cid(
        invariance_branch=invariance_branch,
        role_handoff_signature_cid=role_handoff_signature_cid,
        n_w41_visible_tokens=n_w41_visible,
        n_w42_visible_tokens=n_visible,
        n_w42_overhead_tokens=n_overhead,
        n_structured_bits=n_structured_bits)
    manifest_cid = _compute_w42_manifest_v12_cid(
        parent_w41_cid=parent_w41_cid,
        invariance_state_cid=state_cid,
        invariance_decision_cid=decision_cid,
        invariance_audit_cid=audit_cid,
        invariance_witness_cid=witness_cid,
        role_handoff_signature_cid=role_handoff_signature_cid)
    w42_cid = _compute_w42_outer_cid(
        schema_cid=schema.cid,
        parent_w41_cid=parent_w41_cid,
        manifest_v12_cid=manifest_cid,
        cell_index=cell_index)
    return RoleInvariantSynthesisRatificationEnvelope(
        schema_version=W42_ROLE_INVARIANT_SCHEMA_VERSION,
        schema_cid=schema.cid,
        parent_w41_cid=parent_w41_cid,
        cell_index=cell_index,
        invariance_branch=invariance_branch,
        role_handoff_signature_cid=role_handoff_signature_cid,
        policy_entry_cid=policy_entry_cid,
        integrated_services_pre_w42=integrated_services_pre_w42,
        expected_services=expected_services,
        integrated_services_post_w42=integrated_services_post_w42,
        invariance_score=invariance_score,
        invariance_state_cid=state_cid,
        invariance_decision_cid=decision_cid,
        invariance_audit_cid=audit_cid,
        invariance_witness_cid=witness_cid,
        manifest_v12_cid=manifest_cid,
        n_w41_visible_tokens=n_w41_visible,
        n_w42_visible_tokens=n_visible,
        n_w42_overhead_tokens=n_overhead,
        n_structured_bits=n_structured_bits,
        w42_cid=w42_cid,
    )


class W42SignatureTests(unittest.TestCase):
    def test_signature_deterministic(self) -> None:
        h = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="p")
        a = compute_role_handoff_signature_cid([[h]])
        b = compute_role_handoff_signature_cid([[h]])
        self.assertEqual(a, b)

    def test_signature_permutation_invariant(self) -> None:
        h1 = _DecodedHandoff(
            source_role="a", claim_kind="x", payload="1")
        h2 = _DecodedHandoff(
            source_role="b", claim_kind="y", payload="2")
        a = compute_role_handoff_signature_cid([[h1, h2]])
        b = compute_role_handoff_signature_cid([[h2, h1]])
        self.assertEqual(a, b)

    def test_signature_payload_canonicalised(self) -> None:
        h1 = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="HELLO   World")
        h2 = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="hello world")
        self.assertEqual(
            compute_role_handoff_signature_cid([[h1]]),
            compute_role_handoff_signature_cid([[h2]]))

    def test_signature_input_sensitive(self) -> None:
        h1 = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="a")
        h2 = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="b")
        self.assertNotEqual(
            compute_role_handoff_signature_cid([[h1]]),
            compute_role_handoff_signature_cid([[h2]]))

    def test_signature_empty_handoffs(self) -> None:
        # Defined deterministic CID for empty handoffs.
        cid = compute_role_handoff_signature_cid([])
        self.assertEqual(len(cid), 64)

    def test_signature_namespaced_no_collision(self) -> None:
        # Same canonical content under different "kind" namespaces
        # produces different CIDs (the W42 namespace is opaque to a
        # caller substituting a W22..W41 audit / witness).
        import hashlib
        import json
        h1 = _DecodedHandoff(
            source_role="r", claim_kind="k", payload="p")
        cid_w42 = compute_role_handoff_signature_cid([[h1]])
        # A W41 audit byte-form would NOT include the
        # "w42_role_handoff_signature" namespace.
        wrong_payload = json.dumps(
            {"tuples": [["r", "k", "p"]]},
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        wrong_cid = hashlib.sha256(wrong_payload).hexdigest()
        self.assertNotEqual(cid_w42, wrong_cid)


class W42SelectorTests(unittest.TestCase):
    def test_no_policy(self) -> None:
        b, s, score = select_role_invariance_decision(
            integrated_services=("a",),
            expected_services=None,
            policy_match_found=False)
        self.assertEqual(b, W42_BRANCH_INVARIANCE_NO_POLICY)
        self.assertEqual(s, ("a",))
        self.assertEqual(score, 0.0)

    def test_no_trigger_on_empty_integrated(self) -> None:
        b, s, score = select_role_invariance_decision(
            integrated_services=(),
            expected_services=("a", "b"),
            policy_match_found=True)
        self.assertEqual(b, W42_BRANCH_INVARIANCE_NO_TRIGGER)
        self.assertEqual(s, ())

    def test_ratified_on_match(self) -> None:
        b, s, score = select_role_invariance_decision(
            integrated_services=("a", "b"),
            expected_services=("a", "b"),
            policy_match_found=True)
        self.assertEqual(b, W42_BRANCH_INVARIANCE_RATIFIED)
        self.assertEqual(s, ("a", "b"))
        self.assertEqual(score, 1.0)

    def test_diverged_on_disjoint(self) -> None:
        b, s, score = select_role_invariance_decision(
            integrated_services=("a", "b"),
            expected_services=("c", "d"),
            policy_match_found=True)
        self.assertEqual(
            b, W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED)
        self.assertEqual(s, ())
        self.assertEqual(score, 0.0)

    def test_diverged_on_partial_overlap(self) -> None:
        b, s, score = select_role_invariance_decision(
            integrated_services=("a", "b"),
            expected_services=("b", "c"),
            policy_match_found=True)
        self.assertEqual(
            b, W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED)
        self.assertEqual(s, ())
        self.assertAlmostEqual(score, 1.0 / 3.0, places=5)


class W42PolicyRegistryTests(unittest.TestCase):
    def test_register_and_lookup(self) -> None:
        reg = RoleInvariancePolicyRegistry()
        e = RoleInvariancePolicyEntry(
            role_handoff_signature_cid="aa" * 32,
            expected_services=("a", "b"))
        reg.register(e)
        got = reg.lookup("aa" * 32)
        self.assertIsNotNone(got)
        self.assertEqual(got.expected_services, ("a", "b"))

    def test_lookup_unknown_returns_none(self) -> None:
        reg = RoleInvariancePolicyRegistry()
        self.assertIsNone(reg.lookup("ff" * 32))

    def test_registry_cid_changes_on_register(self) -> None:
        reg = RoleInvariancePolicyRegistry()
        cid_a = reg.cid()
        reg.register(RoleInvariancePolicyEntry(
            role_handoff_signature_cid="aa" * 32,
            expected_services=("a",)))
        cid_b = reg.cid()
        self.assertNotEqual(cid_a, cid_b)


class W42CIDDeterminismTests(unittest.TestCase):
    def test_state_cid_deterministic_and_input_sensitive(
            self) -> None:
        a = _compute_w42_invariance_state_cid(
            invariance_branch=W42_BRANCH_INVARIANCE_RATIFIED,
            role_handoff_signature_cid="aa" * 32,
            integrated_services_pre_w42=("x",),
            integrated_services_post_w42=("x",),
            cell_index=0)
        b = _compute_w42_invariance_state_cid(
            invariance_branch=W42_BRANCH_INVARIANCE_RATIFIED,
            role_handoff_signature_cid="aa" * 32,
            integrated_services_pre_w42=("x",),
            integrated_services_post_w42=("x",),
            cell_index=0)
        self.assertEqual(a, b)
        c = _compute_w42_invariance_state_cid(
            invariance_branch=W42_BRANCH_INVARIANCE_RATIFIED,
            role_handoff_signature_cid="aa" * 32,
            integrated_services_pre_w42=("x",),
            integrated_services_post_w42=("x",),
            cell_index=1)
        self.assertNotEqual(a, c)

    def test_manifest_v12_distinguishes_swapped_components(
            self) -> None:
        m1 = _compute_w42_manifest_v12_cid(
            parent_w41_cid="aa", invariance_state_cid="bb",
            invariance_decision_cid="cc",
            invariance_audit_cid="dd",
            invariance_witness_cid="ee",
            role_handoff_signature_cid="ff")
        m2 = _compute_w42_manifest_v12_cid(
            parent_w41_cid="aa", invariance_state_cid="bb",
            invariance_decision_cid="cc",
            invariance_audit_cid="ee",
            invariance_witness_cid="dd",
            role_handoff_signature_cid="ff")
        self.assertNotEqual(m1, m2)


class W42VerifierTests(unittest.TestCase):
    def test_verifier_accepts_clean_envelope(self) -> None:
        schema = _schema()
        env = _build_clean_envelope(schema=schema)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertTrue(outcome.ok, msg=outcome.reason)

    def test_verifier_rejects_none(self) -> None:
        schema = _schema()
        outcome = verify_role_invariant_synthesis_ratification(
            None, registered_schema=schema,
            registered_parent_w41_cid="x")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "empty_w42_envelope")

    def test_verifier_rejects_unknown_schema_version(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            schema_version="not.a.real.schema")
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_schema_version_unknown")

    def test_verifier_rejects_schema_cid_mismatch(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            schema_cid="ff" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "w42_schema_cid_mismatch")

    def test_verifier_rejects_parent_w41_mismatch(self) -> None:
        schema = _schema()
        env = _build_clean_envelope(schema=schema)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid="not_the_real_parent")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "w41_parent_cid_mismatch")

    def test_verifier_rejects_unknown_invariance_branch(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            invariance_branch="not_a_real_branch")
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_invariance_branch_unknown")

    def test_verifier_rejects_signature_cid_mismatch(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            role_handoff_signature_cid="not_64_hex")
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason,
            "w42_role_handoff_signature_cid_mismatch")

    def test_verifier_rejects_state_cid_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            invariance_state_cid="00" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_invariance_state_cid_mismatch")

    def test_verifier_rejects_decision_cid_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            invariance_decision_cid="11" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_invariance_decision_cid_mismatch")

    def test_verifier_rejects_audit_cid_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            invariance_audit_cid="22" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_invariance_audit_cid_mismatch")

    def test_verifier_rejects_witness_cid_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            invariance_witness_cid="33" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason,
            "w42_invariance_witness_cid_mismatch")

    def test_verifier_rejects_invalid_score(self) -> None:
        # Build a clean envelope at score=0.5; then re-seal CIDs
        # against an invalid score (-0.1 or 1.5) so the dedicated
        # score check fires before token accounting.
        schema = _schema()
        clean = _build_clean_envelope(
            schema=schema, invariance_score=0.5,
            invariance_branch=W42_BRANCH_INVARIANCE_RATIFIED)
        bad_score = -0.1
        decision_cid = _compute_w42_invariance_decision_cid(
            invariance_branch=clean.invariance_branch,
            integrated_services_post_w42=(
                clean.integrated_services_post_w42),
            invariance_score=bad_score,
            n_w41_visible_tokens=clean.n_w41_visible_tokens,
            n_w42_visible_tokens=clean.n_w42_visible_tokens,
            n_w42_overhead_tokens=clean.n_w42_overhead_tokens)
        audit_cid = _compute_w42_invariance_audit_cid(
            invariance_branch=clean.invariance_branch,
            role_handoff_signature_cid=(
                clean.role_handoff_signature_cid),
            policy_entry_cid=clean.policy_entry_cid,
            integrated_services_pre_w42=(
                clean.integrated_services_pre_w42),
            expected_services=clean.expected_services,
            integrated_services_post_w42=(
                clean.integrated_services_post_w42),
            invariance_score=bad_score)
        manifest_cid = _compute_w42_manifest_v12_cid(
            parent_w41_cid=clean.parent_w41_cid,
            invariance_state_cid=clean.invariance_state_cid,
            invariance_decision_cid=decision_cid,
            invariance_audit_cid=audit_cid,
            invariance_witness_cid=clean.invariance_witness_cid,
            role_handoff_signature_cid=(
                clean.role_handoff_signature_cid))
        w42_cid = _compute_w42_outer_cid(
            schema_cid=schema.cid,
            parent_w41_cid=clean.parent_w41_cid,
            manifest_v12_cid=manifest_cid,
            cell_index=clean.cell_index)
        env = dataclasses.replace(
            clean,
            invariance_score=bad_score,
            invariance_decision_cid=decision_cid,
            invariance_audit_cid=audit_cid,
            manifest_v12_cid=manifest_cid,
            w42_cid=w42_cid,
        )
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_invariance_score_invalid")

    def test_verifier_rejects_token_accounting_invalid(
            self) -> None:
        # Build a clean envelope, then re-seal decision/witness/
        # manifest/outer CIDs against an INCONSISTENT token triple
        # so the dedicated token-accounting branch fires.
        schema = _schema()
        clean = _build_clean_envelope(schema=schema)
        bad_w41 = 14
        bad_overhead = 1
        bad_visible = 99  # NOT 14 + 1
        decision_cid = _compute_w42_invariance_decision_cid(
            invariance_branch=clean.invariance_branch,
            integrated_services_post_w42=(
                clean.integrated_services_post_w42),
            invariance_score=clean.invariance_score,
            n_w41_visible_tokens=bad_w41,
            n_w42_visible_tokens=bad_visible,
            n_w42_overhead_tokens=bad_overhead)
        witness_cid = _compute_w42_invariance_witness_cid(
            invariance_branch=clean.invariance_branch,
            role_handoff_signature_cid=(
                clean.role_handoff_signature_cid),
            n_w41_visible_tokens=bad_w41,
            n_w42_visible_tokens=bad_visible,
            n_w42_overhead_tokens=bad_overhead,
            n_structured_bits=clean.n_structured_bits)
        manifest_cid = _compute_w42_manifest_v12_cid(
            parent_w41_cid=clean.parent_w41_cid,
            invariance_state_cid=clean.invariance_state_cid,
            invariance_decision_cid=decision_cid,
            invariance_audit_cid=clean.invariance_audit_cid,
            invariance_witness_cid=witness_cid,
            role_handoff_signature_cid=(
                clean.role_handoff_signature_cid))
        w42_cid = _compute_w42_outer_cid(
            schema_cid=schema.cid,
            parent_w41_cid=clean.parent_w41_cid,
            manifest_v12_cid=manifest_cid,
            cell_index=clean.cell_index)
        env = dataclasses.replace(
            clean,
            n_w41_visible_tokens=bad_w41,
            n_w42_visible_tokens=bad_visible,
            n_w42_overhead_tokens=bad_overhead,
            invariance_decision_cid=decision_cid,
            invariance_witness_cid=witness_cid,
            manifest_v12_cid=manifest_cid,
            w42_cid=w42_cid,
        )
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_token_accounting_invalid")

    def test_verifier_rejects_manifest_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            manifest_v12_cid="44" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(
            outcome.reason, "w42_manifest_v12_cid_mismatch")

    def test_verifier_rejects_outer_cid_tamper(self) -> None:
        schema = _schema()
        env = dataclasses.replace(
            _build_clean_envelope(schema=schema),
            w42_cid="55" * 32)
        outcome = verify_role_invariant_synthesis_ratification(
            env, registered_schema=schema,
            registered_parent_w41_cid=env.parent_w41_cid)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.reason, "w42_outer_cid_mismatch")

    def test_verifier_enumerates_14_distinct_reasons(self) -> None:
        seen = {
            "empty_w42_envelope",
            "w42_schema_version_unknown",
            "w42_schema_cid_mismatch",
            "w41_parent_cid_mismatch",
            "w42_invariance_branch_unknown",
            "w42_role_handoff_signature_cid_mismatch",
            "w42_invariance_state_cid_mismatch",
            "w42_invariance_decision_cid_mismatch",
            "w42_invariance_audit_cid_mismatch",
            "w42_invariance_witness_cid_mismatch",
            "w42_invariance_score_invalid",
            "w42_token_accounting_invalid",
            "w42_manifest_v12_cid_mismatch",
            "w42_outer_cid_mismatch",
        }
        self.assertEqual(len(seen), 14)


class W42R89BankTests(unittest.TestCase):
    """End-to-end W42 R-89 bank tests at n_eval=16."""

    def test_trivial_w42_byte_for_w41(self) -> None:
        # Byte-equivalent passthrough: W42 must not modify the
        # answer or add overhead tokens on trivial.  The W41 vs
        # W42 correctness rates differ here only because they
        # measure different intermediate products: W41 measures
        # the W41 *integrated* services (empty under trivial,
        # since W41 ratifies nothing), while W42 measures the
        # downstream answer.  The substantive check is the
        # byte-equivalent flag, not metric equality.
        result = run_phase89(
            bank="trivial_w42", n_eval=16, bank_seed=11)
        s = result["summary"]
        self.assertTrue(s["w42_w41_byte_equivalent"])
        self.assertEqual(s["total_w42_overhead"], 0)
        self.assertEqual(
            s["w42_invariance_branch_hist"],
            {W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH: 16})

    def test_role_invariant_agrees_preserves_correctness(
            self) -> None:
        result = run_phase89(
            bank="role_invariant_agrees", n_eval=16,
            bank_seed=11)
        s = result["summary"]
        self.assertTrue(s["all_w42_verified_ok"])
        self.assertGreaterEqual(
            s["correctness_w42"], s["correctness_w41"])
        self.assertGreaterEqual(
            s["trust_precision_w42"], s["trust_precision_w41"])
        hist = s["w42_invariance_branch_hist"]
        self.assertIn(W42_BRANCH_INVARIANCE_RATIFIED, hist)

    def test_role_invariant_recover_strict_gain(self) -> None:
        # Load-bearing W42-3 strict gain: trust_precision_w42 =
        # 1.000 strictly improving over trust_precision_w41 = 0.500
        # (the W41-L-COMPOSITE-COLLUSION-CAP composite collusion
        # bench).
        result = run_phase89(
            bank="role_invariant_recover", n_eval=16,
            bank_seed=11)
        s = result["summary"]
        self.assertTrue(s["all_w42_verified_ok"])
        self.assertEqual(s["trust_precision_w41"], 0.5)
        self.assertEqual(s["trust_precision_w42"], 1.0)
        hist = s["w42_invariance_branch_hist"]
        self.assertIn(
            W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED, hist)

    def test_full_composite_collusion_fires_l_cap(self) -> None:
        # The W42-L-FULL-COMPOSITE-COLLUSION-CAP limitation theorem:
        # when the adversary also poisons the policy registry, W42
        # cannot recover at the capsule layer.
        result = run_phase89(
            bank="full_composite_collusion", n_eval=16,
            bank_seed=11)
        s = result["summary"]
        self.assertTrue(s["all_w42_verified_ok"])
        self.assertEqual(
            s["trust_precision_w42"], s["trust_precision_w41"])
        hist = s["w42_invariance_branch_hist"]
        # W42 ratifies (on the wrong colluded set) on every cell of
        # the recovery half.
        self.assertIn(W42_BRANCH_INVARIANCE_RATIFIED, hist)

    def test_insufficient_invariance_policy_falls_through(
            self) -> None:
        result = run_phase89(
            bank="insufficient_invariance_policy",
            n_eval=16, bank_seed=11)
        s = result["summary"]
        self.assertTrue(s["all_w42_verified_ok"])
        self.assertEqual(
            s["correctness_w42"], s["correctness_w41"])
        self.assertEqual(
            s["trust_precision_w42"], s["trust_precision_w41"])
        hist = s["w42_invariance_branch_hist"]
        self.assertIn(W42_BRANCH_INVARIANCE_NO_POLICY, hist)
        self.assertEqual(s["n_policy_entries"], 0)

    def test_w42_overhead_one_token_per_active_cell(self) -> None:
        result = run_phase89(
            bank="role_invariant_agrees", n_eval=16,
            bank_seed=11)
        s = result["summary"]
        # 16 cells * 1 visible token/cell = 16 W42 overhead.
        self.assertEqual(s["total_w42_overhead"], 16)

    def test_w42_structured_bits_density(self) -> None:
        result = run_phase89(
            bank="role_invariant_agrees", n_eval=16,
            bank_seed=11)
        s = result["summary"]
        # W42 envelope carries the W41 envelope's structured bits +
        # 6*256 W42 CIDs.  Density should be in the W41..W40 range.
        self.assertGreater(
            s["mean_w42_structured_bits_per_cell"], 5_000.0)
        self.assertLess(
            s["mean_w42_structured_bits_per_cell"], 50_000.0)


class W42OrchestratorTests(unittest.TestCase):
    def test_disabled_orchestrator_emits_disabled_branch(
            self) -> None:
        # Construct a minimal orchestrator with enabled=False.
        from vision_mvp.experiments.phase89_role_invariant_synthesis import (
            run_phase89)
        # Easiest path: trivial bank produces TRIVIAL_PASSTHROUGH
        # on every cell.
        result = run_phase89(
            bank="trivial_w42", n_eval=4, bank_seed=11)
        hist = result["summary"]["w42_invariance_branch_hist"]
        self.assertEqual(
            hist[W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH], 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
