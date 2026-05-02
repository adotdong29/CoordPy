"""Phase 81 — W34 Live-aware multi-anchor adjudication tests.

Covers:

  * Multi-anchor consensus reference (intersection-of-anchors).
  * Response-feature signature byte-stability + first-token-class
    + length-bucket discrimination.
  * Host-aware EWMA decay (closed-form multiplicative).
  * Verifier failure modes (14 enumerated, disjoint from W22..W33).
  * Registry factories (trivial / non-trivial).
  * Byte-for-W33 invariant (W34-Λ-trivial-multi-anchor).
  * Falsifiers (no-anchor-disagreement, frozen-host-decay,
    mis-feature-signature).
  * H6 main load-bearing claim on R-81-DOUBLE-ANCHOR-COMPROMISE.
  * Manifest-v4 tamper detection.
  * Live oracle attestation.
"""
from __future__ import annotations

import dataclasses
import unittest

from vision_mvp.wevra.team_coord import (
    LiveAwareMultiAnchorOrchestrator,
    LiveAwareMultiAnchorRegistry,
    LiveAwareMultiAnchorRatificationEnvelope,
    LiveOracleAttestation,
    HostRegistration,
    W34LiveAwareResult,
    verify_live_aware_multi_anchor_ratification,
    derive_multi_anchor_consensus_reference,
    compute_response_feature_signature,
    apply_host_decay,
    build_trivial_live_aware_registry,
    build_live_aware_registry,
    W34_LIVE_AWARE_SCHEMA_VERSION,
    W34_DEFAULT_ANCHOR_QUORUM_MIN,
    W34_DEFAULT_HOST_DECAY_FACTOR,
    W34_BRANCH_LIVE_AWARE_RESOLVED,
    W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH,
    W34_BRANCH_LIVE_AWARE_REJECTED,
    W34_BRANCH_MULTI_ANCHOR_CONSENSUS,
    W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
    W34_BRANCH_HOST_DECAY_FIRED,
    W34_ALL_BRANCHES,
    W21OracleProbe,
    LatentVerificationOutcome,
    _compute_live_attestation_cid,
    _compute_multi_anchor_cid,
    _compute_w34_manifest_v4_cid,
    _compute_w34_outer_cid,
    _compute_host_topology_cid,
)
from vision_mvp.experiments.phase81_live_aware_multi_anchor import (
    _stable_schema_capsule,
    _five_named_w34_tampers,
    run_phase81,
    run_phase81_seed_sweep,
    run_phase81_manifest_v4_tamper_sweep,
    run_phase81_response_feature_signature_byte_stability,
)


# ---------------------------------------------------------------------------
# Multi-anchor consensus reference
# ---------------------------------------------------------------------------


def _probe(oracle_id: str, top_set: tuple[str, ...],
           abstained: bool = False) -> W21OracleProbe:
    return W21OracleProbe(
        oracle_id=str(oracle_id),
        role_label=str(oracle_id),
        trust_prior=1.0,
        payload="",
        payload_tokens=(),
        per_tag_count={t: 1 for t in top_set},
        top_set=tuple(top_set),
        abstained=bool(abstained),
        n_outside_tokens=0,
    )


class W34MultiAnchorConsensusTests(unittest.TestCase):

    def test_two_anchors_full_agreement(self) -> None:
        probes = [_probe("a", ("gold",)), _probe("b", ("gold",))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=2))
        self.assertEqual(("gold",), top)
        self.assertEqual(2, n_agree)
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_CONSENSUS, branch)

    def test_two_anchors_disagreement_no_consensus(self) -> None:
        probes = [_probe("a", ("gold",)), _probe("b", ("decoy",))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=2))
        self.assertEqual((), top)
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
                         branch)

    def test_three_anchors_partial_overlap(self) -> None:
        probes = [
            _probe("a", ("gold", "x")),
            _probe("b", ("gold", "y")),
            _probe("c", ("gold", "z"))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=3))
        # Intersection is {"gold"} only.
        self.assertEqual(("gold",), top)
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_CONSENSUS, branch)

    def test_anchor_abstaining_below_quorum_fails(self) -> None:
        probes = [_probe("a", ("gold",), abstained=True),
                  _probe("b", ("gold",))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=2))
        # Only one non-abstaining; below quorum_min=2.
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
                         branch)

    def test_intersection_empty_no_consensus(self) -> None:
        probes = [_probe("a", ("x",)), _probe("b", ("y",))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=2))
        self.assertEqual((), top)
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
                         branch)

    def test_quorum_min_one_single_probe(self) -> None:
        probes = [_probe("a", ("gold",))]
        top, n_agree, branch = (
            derive_multi_anchor_consensus_reference(
                anchor_probes=probes, anchor_quorum_min=1))
        self.assertEqual(("gold",), top)
        self.assertEqual(W34_BRANCH_MULTI_ANCHOR_CONSENSUS, branch)


# ---------------------------------------------------------------------------
# Response-feature signature
# ---------------------------------------------------------------------------


class W34ResponseFeatureSignatureTests(unittest.TestCase):

    def test_byte_stable_across_runs(self) -> None:
        text = "hello world 42"
        s1 = compute_response_feature_signature(response_text=text)
        s2 = compute_response_feature_signature(response_text=text)
        self.assertEqual(s1, s2)

    def test_signature_length_is_16_hex(self) -> None:
        s = compute_response_feature_signature(
            response_text="abc")
        self.assertEqual(16, len(s))
        int(s, 16)  # must parse as hex

    def test_empty_response_signature(self) -> None:
        s = compute_response_feature_signature(response_text="")
        self.assertEqual(16, len(s))

    def test_first_token_class_distinguishes(self) -> None:
        s_digit = compute_response_feature_signature(
            response_text="42 things")
        s_alpha = compute_response_feature_signature(
            response_text="forty-two things")
        # Different first_token_class ⇒ different signatures.
        self.assertNotEqual(s_digit, s_alpha)

    def test_length_bucket_distinguishes_distant_lengths(self) -> None:
        s_short = compute_response_feature_signature(
            response_text="hi")
        s_long = compute_response_feature_signature(
            response_text="x" * 500)
        self.assertNotEqual(s_short, s_long)

    def test_collision_within_bucket_acknowledged(self) -> None:
        # Both same length-bucket ("17..64"), same first_token_class
        # ("alpha"), but DIFFERENT structural_hash ⇒ different
        # signatures.  This documents that collision is in principle
        # possible only if structural_hash also collides — which is
        # SHA-256-prefix bound.
        s1 = compute_response_feature_signature(
            response_text="deadlock service detected")
        s2 = compute_response_feature_signature(
            response_text="deadlock service rejected")
        self.assertNotEqual(s1, s2)


# ---------------------------------------------------------------------------
# Host-aware EWMA decay
# ---------------------------------------------------------------------------


class W34HostDecayTests(unittest.TestCase):

    def test_no_decay_when_healthy(self) -> None:
        self.assertEqual(0.7, apply_host_decay(
            prev_ewma=0.7, host_decay_factor=0.5,
            host_unhealthy=False))

    def test_decay_fires_when_unhealthy(self) -> None:
        self.assertAlmostEqual(0.35, apply_host_decay(
            prev_ewma=0.7, host_decay_factor=0.5,
            host_unhealthy=True))

    def test_factor_one_no_decay(self) -> None:
        # The W34-Λ-frozen-host-decay falsifier.
        self.assertEqual(0.7, apply_host_decay(
            prev_ewma=0.7, host_decay_factor=1.0,
            host_unhealthy=True))

    def test_clamp_to_lower_bound(self) -> None:
        # factor < 0.5 is clamped to 0.5.
        self.assertAlmostEqual(0.40, apply_host_decay(
            prev_ewma=0.8, host_decay_factor=0.1,
            host_unhealthy=True))

    def test_bounded_to_unit_interval(self) -> None:
        # Decayed result is bounded in [0, 1].
        self.assertEqual(0.0, apply_host_decay(
            prev_ewma=0.0, host_decay_factor=0.5,
            host_unhealthy=True))


# ---------------------------------------------------------------------------
# Live oracle attestation
# ---------------------------------------------------------------------------


class W34LiveOracleAttestationTests(unittest.TestCase):

    def test_attestation_cid_recompute_byte_stable(self) -> None:
        att = LiveOracleAttestation(
            oracle_id="ollama_qwen35:35b",
            host_id="192.168.12.191",
            model_id="qwen3.5:35b",
            response_feature_signature=
                compute_response_feature_signature(
                    response_text="dijkstra"),
            latency_ms_bucket="10k..60k",
            preflight_ok=True,
        )
        self.assertEqual(att.attestation_cid,
                         att.recompute_attestation_cid())

    def test_attestation_dict_round_trip(self) -> None:
        att = LiveOracleAttestation(
            oracle_id="o", host_id="h", model_id="m",
            response_feature_signature="0123456789abcdef",
            latency_ms_bucket="0..1k", preflight_ok=False,
        )
        d = att.as_dict()
        self.assertEqual(d["oracle_id"], "o")
        self.assertEqual(d["host_id"], "h")
        self.assertEqual(d["preflight_ok"], False)


# ---------------------------------------------------------------------------
# Registry factories
# ---------------------------------------------------------------------------


class W34RegistryFactoryTests(unittest.TestCase):

    def test_trivial_registry_is_trivial(self) -> None:
        schema = _stable_schema_capsule()
        reg = build_trivial_live_aware_registry(schema=schema)
        self.assertTrue(reg.is_trivial)
        self.assertFalse(reg.has_wire_required_layer)

    def test_non_trivial_registry_not_trivial(self) -> None:
        schema = _stable_schema_capsule()
        reg = build_live_aware_registry(
            schema=schema,
            inner_w33_registry=None,
            multi_anchor_quorum_min=2,
            live_attestation_disabled=False,
            manifest_v4_disabled=False,
            host_decay_factor=0.5,
            registered_hosts={},
        )
        self.assertFalse(reg.is_trivial)
        self.assertTrue(reg.has_wire_required_layer)

    def test_host_topology_cid_byte_stable(self) -> None:
        schema = _stable_schema_capsule()
        h1 = HostRegistration(
            host_id="localhost", model_id="gemma2:9b",
            base_url="http://localhost:11434",
            timeout_ms_bucket=30_000, preflight_ok=True)
        h2 = HostRegistration(
            host_id="192.168.12.191", model_id="qwen3.5:35b",
            base_url="http://192.168.12.191:11434",
            timeout_ms_bucket=240_000, preflight_ok=True)
        reg = build_live_aware_registry(
            schema=schema, inner_w33_registry=None,
            registered_hosts={
                h1.host_id: h1, h2.host_id: h2,
            })
        cid1 = reg.host_topology_cid
        cid2 = reg.host_topology_cid
        self.assertEqual(cid1, cid2)
        self.assertEqual(64, len(cid1))


# ---------------------------------------------------------------------------
# Manifest-v4 cross-component tamper detection (H3)
# ---------------------------------------------------------------------------


class W34ManifestV4TamperTests(unittest.TestCase):

    def test_400_of_400_tampers_rejected(self) -> None:
        result = run_phase81_manifest_v4_tamper_sweep(
            bank="manifest_v4_tamper", n_eval=16,
            seeds=(11, 17, 23, 29, 31), T_decoder=None)
        self.assertEqual(400, result["n_tamper_attempts_total"])
        self.assertEqual(400, result["n_tamper_rejected_total"])
        self.assertEqual(1.0, result["reject_rate_total"])


# ---------------------------------------------------------------------------
# Verifier failure modes (14 enumerated, disjoint from W22..W33)
# ---------------------------------------------------------------------------


def _build_clean_envelope(
        cell_index: int = 0,
) -> tuple[LiveAwareMultiAnchorRatificationEnvelope, dict]:
    schema = _stable_schema_capsule()
    anchor_ids = ("service_graph", "change_history")
    parent_w33_cid = "ab" * 32
    multi_anchor_cid = _compute_multi_anchor_cid(
        anchor_oracle_ids=anchor_ids,
        anchor_quorum_min=2,
        consensus_branch=W34_BRANCH_MULTI_ANCHOR_CONSENSUS,
        consensus_top_set=("gold",),
        n_anchors_agreeing=2,
    )
    live_attestation_cid = _compute_live_attestation_cid(
        attestations=())
    host_topology_cid = _compute_host_topology_cid(
        registered_hosts={})
    manifest_v4_cid = _compute_w34_manifest_v4_cid(
        parent_w33_cid=parent_w33_cid,
        live_attestation_cid=live_attestation_cid,
        multi_anchor_cid=multi_anchor_cid,
        host_topology_cid=host_topology_cid,
    )
    env = LiveAwareMultiAnchorRatificationEnvelope(
        schema_version=W34_LIVE_AWARE_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        parent_w33_cid=parent_w33_cid,
        anchor_oracle_ids=anchor_ids,
        anchor_quorum_min=2,
        multi_anchor_consensus_top_set=("gold",),
        multi_anchor_branch=W34_BRANCH_MULTI_ANCHOR_CONSENSUS,
        n_anchors_agreeing=2,
        multi_anchor_cid=multi_anchor_cid,
        live_attestations=(),
        live_attestation_cid=live_attestation_cid,
        live_attestation_disabled=True,
        host_topology_cid=host_topology_cid,
        manifest_v4_cid=manifest_v4_cid,
        cell_index=int(cell_index),
        wire_required=True,
    )
    registry_kwargs = dict(
        registered_schema=schema,
        registered_parent_w33_cid=parent_w33_cid,
        registered_anchor_oracle_ids=frozenset(anchor_ids),
        registered_anchor_quorum_min=2,
        registered_host_topology_cid=host_topology_cid,
    )
    return env, registry_kwargs


class W34VerifierFailureModeTests(unittest.TestCase):

    def test_clean_envelope_passes(self) -> None:
        env, kwargs = _build_clean_envelope()
        outcome = verify_live_aware_multi_anchor_ratification(
            env, **kwargs)
        self.assertTrue(outcome.ok)

    def test_empty_envelope(self) -> None:
        _env, kwargs = _build_clean_envelope()
        outcome = verify_live_aware_multi_anchor_ratification(
            None, **kwargs)
        self.assertEqual("empty_w34_envelope", outcome.reason)

    def test_schema_version_unknown(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env,
                                   schema_version="bogus.version.v0",
                                   w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_schema_version_unknown",
                         outcome.reason)

    def test_schema_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env, schema_cid="bad",
                                   w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_schema_cid_mismatch", outcome.reason)

    def test_w33_parent_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env, parent_w33_cid="bad",
                                   w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w33_parent_cid_mismatch", outcome.reason)

    def test_anchor_oracle_set_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(
            env, anchor_oracle_ids=("only_one",),
            anchor_quorum_min=1, w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_anchor_oracle_set_mismatch",
                         outcome.reason)

    def test_anchor_quorum_min_too_large(self) -> None:
        env, kwargs = _build_clean_envelope()
        # quorum_min=5 > len(anchors)=2
        bad = dataclasses.replace(
            env, anchor_quorum_min=5, w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_anchor_quorum_min_out_of_range",
                         outcome.reason)

    def test_anchor_quorum_min_zero(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env, anchor_quorum_min=0,
                                   w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_anchor_quorum_min_out_of_range",
                         outcome.reason)

    def test_multi_anchor_branch_unknown(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env,
                                   multi_anchor_branch="bogus",
                                   w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_multi_anchor_branch_unknown",
                         outcome.reason)

    def test_multi_anchor_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(
            env, multi_anchor_cid="bad" * 21,
            w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_multi_anchor_cid_mismatch",
                         outcome.reason)

    def test_live_attestation_signature_invalid_length(self) -> None:
        env, kwargs = _build_clean_envelope()
        # Inject a bad attestation with wrong-length signature.
        bad_att = LiveOracleAttestation(
            oracle_id="o", host_id="h", model_id="m",
            response_feature_signature="bad",  # length 3, not 16
            latency_ms_bucket="0..1k", preflight_ok=True,
        )
        new_live_cid = _compute_live_attestation_cid(
            attestations=(bad_att,))
        new_manifest = _compute_w34_manifest_v4_cid(
            parent_w33_cid=env.parent_w33_cid,
            live_attestation_cid=new_live_cid,
            multi_anchor_cid=env.multi_anchor_cid,
            host_topology_cid=env.host_topology_cid,
        )
        bad = dataclasses.replace(
            env, live_attestations=(bad_att,),
            live_attestation_cid=new_live_cid,
            manifest_v4_cid=new_manifest,
            w34_cid="",
        )
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_live_attestation_signature_invalid",
                         outcome.reason)

    def test_live_attestation_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        # Add an attestation but keep OLD live_attestation_cid.
        good_att = LiveOracleAttestation(
            oracle_id="o", host_id="h", model_id="m",
            response_feature_signature=(
                compute_response_feature_signature(
                    response_text="x")),
            latency_ms_bucket="0..1k", preflight_ok=True,
        )
        bad = dataclasses.replace(
            env, live_attestations=(good_att,),
            live_attestation_cid=env.live_attestation_cid,  # OLD
            w34_cid=env.w34_cid,
        )
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_live_attestation_cid_mismatch",
                         outcome.reason)

    def test_host_topology_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(
            env, host_topology_cid="00" * 32,
            w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_host_topology_cid_mismatch",
                         outcome.reason)

    def test_attestation_oracle_unregistered(self) -> None:
        env, kwargs = _build_clean_envelope()
        # Empty oracle_id ⇒ unregistered.
        bad_att = LiveOracleAttestation(
            oracle_id="", host_id="h", model_id="m",
            response_feature_signature=(
                "0123456789abcdef"),
            latency_ms_bucket="0..1k", preflight_ok=True,
        )
        new_live_cid = _compute_live_attestation_cid(
            attestations=(bad_att,))
        new_manifest = _compute_w34_manifest_v4_cid(
            parent_w33_cid=env.parent_w33_cid,
            live_attestation_cid=new_live_cid,
            multi_anchor_cid=env.multi_anchor_cid,
            host_topology_cid=env.host_topology_cid,
        )
        bad = dataclasses.replace(
            env, live_attestations=(bad_att,),
            live_attestation_cid=new_live_cid,
            manifest_v4_cid=new_manifest,
            w34_cid="",
        )
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_attestation_oracle_unregistered",
                         outcome.reason)

    def test_manifest_v4_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(
            env, manifest_v4_cid="00" * 32,
            w34_cid=env.w34_cid)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_manifest_v4_cid_mismatch",
                         outcome.reason)

    def test_outer_cid_mismatch(self) -> None:
        env, kwargs = _build_clean_envelope()
        bad = dataclasses.replace(env, w34_cid="00" * 32)
        outcome = verify_live_aware_multi_anchor_ratification(
            bad, **kwargs)
        self.assertEqual("w34_outer_cid_mismatch", outcome.reason)


# ---------------------------------------------------------------------------
# H6 main load-bearing claim — R-81-DOUBLE-ANCHOR-COMPROMISE
# ---------------------------------------------------------------------------


class W34DoubleAnchorCompromiseTests(unittest.TestCase):

    def test_h6_strict_trust_precision_gain_5_seeds(self) -> None:
        # Pre-committed: Δ_trust_precision_w34_w33 ≥ +0.10
        # AND min_trust_precision_w34 = 1.000 AND no correctness
        # regression AND ≤ 2 token/cell overhead across 5/5 seeds.
        result = run_phase81_seed_sweep(
            bank="double_anchor_compromise", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            multi_anchor_quorum_min=2,
            manifest_v4_disabled=False,
            anchor_oracle_ids=("service_graph", "change_history"),
        )
        self.assertGreaterEqual(
            result["min_delta_trust_precision_w34_w33"], 0.10,
            f"Δ_trust_prec floor below H6 bar: {result}")
        self.assertEqual(
            1.0, result["min_trust_precision_w34"],
            f"trust_prec floor below 1.000: {result}")
        self.assertGreaterEqual(
            result["min_delta_correctness_w34_w33"], 0.0,
            f"correctness regression: {result}")
        self.assertLessEqual(
            result["max_overhead_w34_per_cell"], 2,
            f"overhead exceeds 2 token/cell: {result}")


# ---------------------------------------------------------------------------
# Falsifiers
# ---------------------------------------------------------------------------


class W34FalsifierTests(unittest.TestCase):

    def test_trivial_multi_anchor_byte_for_w33(self) -> None:
        # W34-Λ-trivial-multi-anchor: degenerates to W33 byte-for-byte.
        result = run_phase81_seed_sweep(
            bank="trivial_w34", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            multi_anchor_quorum_min=1,
            live_attestation_disabled=True,
            manifest_v4_disabled=True,
            host_decay_factor=1.0,
        )
        self.assertTrue(
            result["all_byte_equivalent_w34_w33"],
            f"H2 byte-for-W33 anchor failed: {result}")

    def test_no_anchor_disagreement_no_lift(self) -> None:
        # W34-Λ-no-anchor-disagreement: all anchors agree throughout
        # ⇒ multi_anchor consensus is the same as single-anchor;
        # W34 ties W33 on correctness.
        result = run_phase81_seed_sweep(
            bank="no_anchor_disagreement", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            multi_anchor_quorum_min=2,
            manifest_v4_disabled=False,
            anchor_oracle_ids=("service_graph", "change_history"),
        )
        self.assertEqual(
            0.0, result["min_delta_trust_precision_w34_w33"])
        self.assertEqual(
            0.0, result["min_delta_correctness_w34_w33"])

    def test_frozen_host_decay_no_decay(self) -> None:
        # W34-Λ-frozen-host-decay: host_decay_factor=1.0 ⇒
        # decay never fires.
        result = run_phase81_seed_sweep(
            bank="frozen_host_decay", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            multi_anchor_quorum_min=2,
            manifest_v4_disabled=False,
            host_decay_factor=1.0,
        )
        # trust precision delta = 0 because decay never fires;
        # hosts are healthy in this regime.
        self.assertEqual(
            0.0, result["min_delta_trust_precision_w34_w33"])

    def test_mis_feature_signature_collision_no_regression(
            self) -> None:
        # W34-Λ-mis-feature-signature: even if responses collide on
        # the feature signature, W34 does not regress vs W33 because
        # the signature is part of the audited ENVELOPE, not the
        # routing decision.  We confirm by running the
        # mis_feature_signature bank with trivial knobs (no live
        # attestations) and verifying W34 ties W33.
        result = run_phase81_seed_sweep(
            bank="mis_feature_signature", n_eval=16,
            seeds=(11, 17, 23, 29, 31),
            multi_anchor_quorum_min=2,
            manifest_v4_disabled=False,
            anchor_oracle_ids=("service_graph", "change_history"),
        )
        # On the all-honest mis_feature regime, W34 ties W33 on
        # correctness (trust precision = 1.0 across 5 seeds; no
        # regression).
        self.assertGreaterEqual(
            result["min_delta_correctness_w34_w33"], 0.0)


# ---------------------------------------------------------------------------
# Response-feature signature byte-stability (H8)
# ---------------------------------------------------------------------------


class W34ResponseFeatureSignatureByteStabilityTests(unittest.TestCase):

    def test_h8_byte_stable_10_fixtures_3_runs_each(self) -> None:
        result = (
            run_phase81_response_feature_signature_byte_stability())
        self.assertTrue(
            result["all_byte_stable"],
            f"H8 byte-stability failed: {result}")
        self.assertEqual(10, result["n_fixtures"])


# ---------------------------------------------------------------------------
# Full envelope round-trip
# ---------------------------------------------------------------------------


class W34EnvelopeRoundTripTests(unittest.TestCase):

    def test_envelope_recompute_byte_stable(self) -> None:
        env, _ = _build_clean_envelope()
        self.assertEqual(env.w34_cid, env.recompute_w34_cid())

    def test_envelope_to_canonical_bytes_byte_stable(self) -> None:
        env, _ = _build_clean_envelope()
        self.assertEqual(env.to_canonical_bytes(),
                         env.to_canonical_bytes())

    def test_envelope_decoder_text_format(self) -> None:
        env, _ = _build_clean_envelope()
        text = env.to_decoder_text()
        self.assertTrue(text.startswith("<w34_ref:"))
        self.assertTrue(text.endswith(">"))


if __name__ == "__main__":
    unittest.main()
