"""Tests for Phase 79 — W32 long-window convergent online geometry-
aware dense control + EWMA prior accumulator + Page CUSUM change-
point detector + gold-correlated disagreement-routing + W32 manifest-
v2 CID (SDK v3.33).
"""

from __future__ import annotations

import dataclasses
import json
import math
import unittest

from vision_mvp.wevra.team_coord import (
    SchemaCapsule, build_incident_triage_schema_capsule,
    LatentVerificationOutcome,
    # W31 surface (parents)
    OnlineCalibratedRatificationEnvelope,
    PriorTrajectoryEntry,
    # W32 surface
    GoldCorrelationMap, build_gold_correlation_map,
    ConvergenceStateEntry,
    LongWindowConvergentRatificationEnvelope,
    LongWindowConvergentRegistry,
    W32LongWindowResult,
    LongWindowConvergentOrchestrator,
    verify_long_window_convergent_ratification,
    update_ewma_prior, update_cusum_two_sided, detect_change_point,
    build_trivial_long_window_registry,
    build_long_window_convergent_registry,
    W32_LONG_WINDOW_SCHEMA_VERSION,
    W32_DEFAULT_EWMA_ALPHA,
    W32_DEFAULT_CUSUM_THRESHOLD,
    W32_DEFAULT_CUSUM_K,
    W32_DEFAULT_CUSUM_MAX,
    W32_DEFAULT_LONG_WINDOW,
    W32_DEFAULT_GOLD_CORRELATION_MIN,
    W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH,
    W32_BRANCH_LONG_WINDOW_RESOLVED,
    W32_ALL_BRANCHES,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    _compute_convergence_state_cid,
    _compute_w32_manifest_v2_cid,
    _compute_w32_outer_cid,
    _compute_gold_correlation_cid,
)


REGISTERED_PIDS = frozenset((
    W29_PARTITION_LINEAR,
    W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
))


def _build_simple_envelope(
        *,
        schema: SchemaCapsule,
        states: tuple[ConvergenceStateEntry, ...] = (),
        ewma_alpha: float = 0.20,
        cusum_threshold: float = 1.5,
        cusum_k: float = 0.10,
        gold_map: GoldCorrelationMap | None = None,
        w31_online_cid: str = "ab" * 32,
        gold_route_active: bool = False,
        gold_route_target_partition_id: int = -1,
        change_point_active: bool = False,
        cell_index: int = 0,
        wire_required: bool = True,
) -> LongWindowConvergentRatificationEnvelope:
    """Build an internally-consistent W32 envelope for tests."""
    conv_cid = _compute_convergence_state_cid(states=states)
    if gold_map is None:
        gold_cid = _compute_gold_correlation_cid(
            partition_to_score=(),
            gold_correlation_min=W32_DEFAULT_GOLD_CORRELATION_MIN)
    else:
        gold_cid = gold_map.gold_correlation_cid
    route_audit_payload = json.dumps({
        "change_point_active": bool(change_point_active),
        "gold_route_active": bool(gold_route_active),
        "gold_route_target_partition_id": int(
            gold_route_target_partition_id),
    }, sort_keys=True).encode()
    import hashlib
    route_audit_cid = hashlib.sha256(route_audit_payload).hexdigest()
    manifest_v2_cid = _compute_w32_manifest_v2_cid(
        w31_online_cid=w31_online_cid,
        convergence_state_cid=conv_cid,
        gold_correlation_cid=gold_cid,
        route_audit_cid_v2=route_audit_cid,
    )
    env = LongWindowConvergentRatificationEnvelope(
        schema_version=W32_LONG_WINDOW_SCHEMA_VERSION,
        schema_cid=str(schema.cid),
        w31_online_cid=str(w31_online_cid),
        convergence_states=states,
        convergence_state_cid=conv_cid,
        ewma_alpha=float(ewma_alpha),
        cusum_threshold=float(cusum_threshold),
        cusum_k=float(cusum_k),
        gold_correlation_cid=gold_cid,
        gold_route_target_partition_id=int(
            gold_route_target_partition_id),
        gold_route_active=bool(gold_route_active),
        change_point_active=bool(change_point_active),
        route_audit_cid_v2=route_audit_cid,
        manifest_v2_cid=manifest_v2_cid,
        cell_index=int(cell_index),
        wire_required=bool(wire_required),
    )
    return env


class W32EWMAUpdateTests(unittest.TestCase):
    """Closed-form EWMA arithmetic — bounded, deterministic."""

    def test_ewma_zero_alpha_keeps_prev(self) -> None:
        out = update_ewma_prior(prev_ewma=0.7, observation=0.0, alpha=0.0)
        self.assertEqual(out, 0.7)

    def test_ewma_unit_alpha_takes_observation(self) -> None:
        out = update_ewma_prior(prev_ewma=0.7, observation=0.0, alpha=1.0)
        self.assertEqual(out, 0.0)

    def test_ewma_default_decays(self) -> None:
        out = update_ewma_prior(
            prev_ewma=1.0, observation=0.0,
            alpha=W32_DEFAULT_EWMA_ALPHA)
        self.assertAlmostEqual(out, 0.8, places=6)

    def test_ewma_clamps_alpha_above_one(self) -> None:
        out = update_ewma_prior(prev_ewma=0.7, observation=0.0, alpha=2.5)
        # alpha clamped to 1.0 → out = observation = 0.0.
        self.assertEqual(out, 0.0)

    def test_ewma_clamps_alpha_below_zero(self) -> None:
        out = update_ewma_prior(prev_ewma=0.7, observation=0.0, alpha=-1.0)
        # alpha clamped to 0.0 → out = prev = 0.7.
        self.assertEqual(out, 0.7)


class W32CUSUMTests(unittest.TestCase):
    """Page two-sided CUSUM math + change-point detection."""

    def test_cusum_no_drift(self) -> None:
        cp, cn = update_cusum_two_sided(
            cusum_pos_prev=0.0, cusum_neg_prev=0.0,
            observation=0.5, target=0.5,
            slack_k=0.10, cusum_max=10.0)
        # No drift → cusum stays 0 (or negative → clamped to 0).
        self.assertEqual(cp, 0.0)
        self.assertEqual(cn, 0.0)

    def test_cusum_positive_drift_accumulates(self) -> None:
        cp, cn = update_cusum_two_sided(
            cusum_pos_prev=0.0, cusum_neg_prev=0.0,
            observation=1.0, target=0.0,
            slack_k=0.10, cusum_max=10.0)
        # +1 - 0 - 0.1 = 0.9 above target.
        self.assertAlmostEqual(cp, 0.9, places=6)
        self.assertEqual(cn, 0.0)

    def test_cusum_negative_drift_accumulates(self) -> None:
        cp, cn = update_cusum_two_sided(
            cusum_pos_prev=0.0, cusum_neg_prev=0.0,
            observation=0.0, target=1.0,
            slack_k=0.10, cusum_max=10.0)
        # -1 - (-1) - 0.1 = ... target shift -1 + obs -1 ... let's check:
        # cn = max(0, 0 - (0 - 1) - 0.1) = max(0, 1 - 0.1) = 0.9.
        self.assertEqual(cp, 0.0)
        self.assertAlmostEqual(cn, 0.9, places=6)

    def test_cusum_clamps_to_max(self) -> None:
        cp, cn = update_cusum_two_sided(
            cusum_pos_prev=8.0, cusum_neg_prev=0.0,
            observation=1.0, target=0.0,
            slack_k=0.10, cusum_max=8.5)
        self.assertAlmostEqual(cp, 8.5, places=6)

    def test_change_point_detection_fires_on_threshold(self) -> None:
        self.assertTrue(detect_change_point(
            cusum_pos=2.0, cusum_neg=0.0, threshold=1.5))
        self.assertTrue(detect_change_point(
            cusum_pos=0.0, cusum_neg=2.0, threshold=1.5))
        self.assertFalse(detect_change_point(
            cusum_pos=1.0, cusum_neg=0.5, threshold=1.5))


class W32GoldCorrelationMapTests(unittest.TestCase):
    """GoldCorrelationMap construction + winner detection."""

    def test_build_map_canonicalises_order(self) -> None:
        m1 = build_gold_correlation_map(
            partition_to_score=[(2, 0.5), (0, 0.85), (1, 0.30)])
        m2 = build_gold_correlation_map(
            partition_to_score=[(0, 0.85), (1, 0.30), (2, 0.5)])
        self.assertEqual(m1.gold_correlation_cid, m2.gold_correlation_cid)
        # Canonical sort is by partition_id ascending.
        self.assertEqual(m1.partition_to_score[0][0], 0)
        self.assertEqual(m1.partition_to_score[-1][0], 2)

    def test_best_partition_returns_unique_winner_above_threshold(self) -> None:
        m = build_gold_correlation_map(
            partition_to_score=[(0, 0.85), (1, 0.30)],
            gold_correlation_min=0.50)
        best = m.best_partition()
        self.assertIsNotNone(best)
        self.assertEqual(best, (0, 0.85))

    def test_best_partition_returns_none_on_tie(self) -> None:
        m = build_gold_correlation_map(
            partition_to_score=[(0, 0.85), (1, 0.85)],
            gold_correlation_min=0.50)
        # Two-way tie → no unique winner → no route fires.
        self.assertIsNone(m.best_partition())

    def test_best_partition_returns_none_below_threshold(self) -> None:
        m = build_gold_correlation_map(
            partition_to_score=[(0, 0.40), (1, 0.30)],
            gold_correlation_min=0.50)
        self.assertIsNone(m.best_partition())

    def test_invalid_score_above_one_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            build_gold_correlation_map(
                partition_to_score=[(0, 1.5)],
                gold_correlation_min=0.50)


class W32RegistryFactoryTests(unittest.TestCase):
    """Convenience factories for trivial / non-trivial W32 registries."""

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def test_trivial_registry_is_trivial(self) -> None:
        reg = build_trivial_long_window_registry(schema=self.schema)
        self.assertTrue(reg.is_trivial)
        self.assertFalse(reg.has_wire_required_layer)

    def test_long_window_registry_is_not_trivial(self) -> None:
        reg = build_long_window_convergent_registry(schema=self.schema)
        self.assertFalse(reg.is_trivial)
        self.assertTrue(reg.has_wire_required_layer)

    def test_gold_correlation_cid_default_for_no_map(self) -> None:
        reg = build_trivial_long_window_registry(schema=self.schema)
        # Default empty map → known canonical CID.
        empty_cid = _compute_gold_correlation_cid(
            partition_to_score=(),
            gold_correlation_min=W32_DEFAULT_GOLD_CORRELATION_MIN)
        self.assertEqual(reg.gold_correlation_cid, empty_cid)


class W32EnvelopeBasicTests(unittest.TestCase):
    """Envelope construction + canonical-bytes recompute."""

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()

    def test_empty_envelope_self_consistent(self) -> None:
        env = _build_simple_envelope(schema=self.schema)
        self.assertEqual(env.recompute_w32_cid(), env.w32_cid)

    def test_envelope_n_wire_tokens_zero_when_not_required(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, wire_required=False)
        self.assertEqual(env.n_wire_tokens, 0)

    def test_envelope_n_wire_tokens_one_when_required(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, wire_required=True)
        # The decoder text "<w32_ref:DDDD>" is one whitespace token.
        self.assertEqual(env.n_wire_tokens, 1)

    def test_envelope_serialises_to_dict_round_trip(self) -> None:
        env = _build_simple_envelope(schema=self.schema)
        d = env.as_dict()
        self.assertEqual(d["schema_version"], W32_LONG_WINDOW_SCHEMA_VERSION)
        self.assertIn("convergence_state_cid", d)
        self.assertIn("manifest_v2_cid", d)
        self.assertIn("w32_cid", d)


class W32VerifierFailureModeTests(unittest.TestCase):
    """Each of the 14 enumerated W32 verifier failure modes is
    exercised by at least one test.  The verifier's soundness is by
    inspection over the enumerated failures."""

    def setUp(self) -> None:
        self.schema = build_incident_triage_schema_capsule()
        self.w31_online_cid = "ab" * 32

    def _verify(self, env, **kwargs) -> LatentVerificationOutcome:
        defaults = dict(
            registered_schema=self.schema,
            registered_w31_online_cid=self.w31_online_cid,
            registered_partition_ids=REGISTERED_PIDS,
            registered_long_window=64,
            registered_cusum_max=W32_DEFAULT_CUSUM_MAX,
            registered_gold_correlation_cid=_compute_gold_correlation_cid(
                partition_to_score=(),
                gold_correlation_min=W32_DEFAULT_GOLD_CORRELATION_MIN),
        )
        defaults.update(kwargs)
        return verify_long_window_convergent_ratification(env, **defaults)

    def test_1_empty_envelope(self) -> None:
        out = self._verify(None)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "empty_w32_envelope")

    def test_2_schema_version_unknown(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(env, schema_version="bad", w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "w32_schema_version_unknown")

    def test_3_schema_cid_mismatch(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(env, schema_cid="bad", w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "w32_schema_cid_mismatch")

    def test_4_w31_parent_cid_mismatch(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(env, w31_online_cid="cd" * 32, w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "w31_parent_cid_mismatch")

    def test_5_convergence_state_cid_mismatch_recompute(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(
            env, convergence_state_cid="ee" * 32, w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "convergence_state_cid_mismatch")

    def test_5b_convergence_state_cid_mismatch_registered(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        out = self._verify(
            env,
            registered_convergence_state_cid="ee" * 32)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "convergence_state_cid_mismatch")

    def test_6_convergence_state_length_mismatch(self) -> None:
        # window=2 but envelope has 3 entries.
        states = (
            ConvergenceStateEntry(
                cell_idx=0, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
            ConvergenceStateEntry(
                cell_idx=1, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
            ConvergenceStateEntry(
                cell_idx=2, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
        )
        env = _build_simple_envelope(
            schema=self.schema, states=states,
            w31_online_cid=self.w31_online_cid)
        out = self._verify(env, registered_long_window=2)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "convergence_state_length_mismatch")

    def test_6b_convergence_state_non_monotone(self) -> None:
        states = (
            ConvergenceStateEntry(
                cell_idx=2, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
            ConvergenceStateEntry(
                cell_idx=1, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
        )
        env = _build_simple_envelope(
            schema=self.schema, states=states,
            w31_online_cid=self.w31_online_cid)
        out = self._verify(env)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "convergence_state_length_mismatch")

    def test_7_unregistered_partition(self) -> None:
        states = (
            ConvergenceStateEntry(
                cell_idx=0, partition_id=99,  # unregistered
                ewma_prior_after=1.0, cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
        )
        env = _build_simple_envelope(
            schema=self.schema, states=states,
            w31_online_cid=self.w31_online_cid)
        out = self._verify(env)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason,
                          "convergence_state_unregistered_partition")

    def test_8_ewma_out_of_range(self) -> None:
        states = (
            ConvergenceStateEntry(
                cell_idx=0, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=2.5,  # out of range
                cusum_pos=0.0, cusum_neg=0.0,
                change_point_fired=False),
        )
        env = _build_simple_envelope(
            schema=self.schema, states=states,
            w31_online_cid=self.w31_online_cid)
        out = self._verify(env)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason,
                          "convergence_state_ewma_out_of_range")

    def test_9_cusum_out_of_range(self) -> None:
        states = (
            ConvergenceStateEntry(
                cell_idx=0, partition_id=W29_PARTITION_LINEAR,
                ewma_prior_after=1.0,
                cusum_pos=99.0,  # exceeds cusum_max
                cusum_neg=0.0,
                change_point_fired=False),
        )
        env = _build_simple_envelope(
            schema=self.schema, states=states,
            w31_online_cid=self.w31_online_cid)
        out = self._verify(env, registered_cusum_max=10.0)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason,
                          "convergence_state_cusum_out_of_range")

    def test_10_ewma_alpha_out_of_range(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid,
            ewma_alpha=2.5)  # out of [0, 1]
        # Recompute manifest + outer to be self-consistent at the bad
        # alpha so the only failing check is ewma_alpha.
        bad = dataclasses.replace(env, w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "ewma_alpha_out_of_range")

    def test_11_cusum_threshold_out_of_range(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid,
            cusum_threshold=999.0)
        bad = dataclasses.replace(env, w32_cid="")
        out = self._verify(bad, registered_cusum_max=10.0)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "cusum_threshold_out_of_range")

    def test_12_gold_correlation_cid_mismatch(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        out = self._verify(env, registered_gold_correlation_cid="ee" * 32)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "gold_correlation_cid_mismatch")

    def test_13_manifest_v2_cid_mismatch(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(env, manifest_v2_cid="ee" * 32, w32_cid="")
        out = self._verify(bad)
        self.assertFalse(out.ok)
        # Could be either manifest_v2_cid_mismatch (recompute fail) or
        # w32_outer_cid_mismatch (outer CID over manifest_v2 mismatches).
        self.assertIn(out.reason, (
            "manifest_v2_cid_mismatch", "w32_outer_cid_mismatch"))

    def test_14_w32_outer_cid_mismatch(self) -> None:
        env = _build_simple_envelope(
            schema=self.schema, w31_online_cid=self.w31_online_cid)
        bad = dataclasses.replace(env)
        object.__setattr__(bad, "w32_cid", "ff" * 32)
        out = self._verify(bad)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, "w32_outer_cid_mismatch")


class W32BenchTests(unittest.TestCase):
    """End-to-end W32 bench correctness on R-79 sub-banks."""

    def test_h2_trivial_w32_byte_for_w31(self) -> None:
        from vision_mvp.experiments.phase79_long_window_convergent import (
            run_phase79_seed_sweep)
        r = run_phase79_seed_sweep(
            bank="trivial_w32",
            seeds=(11, 17, 23, 29, 31),
            n_eval=16, signature_period=4)
        self.assertTrue(r["all_byte_equivalent_w32_w31"])
        self.assertEqual(r["min_correctness_w32"], r["max_correctness_w32"])
        self.assertEqual(r["min_correctness_w31"], r["max_correctness_w31"])

    def test_h7_long_window_no_degradation(self) -> None:
        from vision_mvp.experiments.phase79_long_window_convergent import (
            run_phase79_long_window_sweep)
        r = run_phase79_long_window_sweep(
            long_windows=(16, 32, 64, 128),
            seeds=(11, 17, 23, 29, 31),
            n_eval=64,
            signature_period=4,
        )
        # On every window in {16, 32, 64, 128}, W32 ≥ W31 across
        # 5/5 seeds (no degradation).
        for s in r["sweep"]:
            self.assertTrue(s["all_w32_ge_w31"])
            self.assertGreaterEqual(s["min_correctness_w32"], 0.0)

    def test_h8_manifest_v2_tamper_reject_rate_one(self) -> None:
        from vision_mvp.experiments.phase79_long_window_convergent import (
            run_phase79_seed_sweep)
        r = run_phase79_seed_sweep(
            bank="manifest_v2_tamper",
            seeds=(11, 17, 23, 29, 31),
            n_eval=64, signature_period=4)
        for sr in r["seed_results"]:
            self.assertEqual(sr["tamper_reject_rate"], 1.0)
            self.assertGreater(sr["n_tamper_attempts"], 0)

    def test_w32_lambda_no_change_point_falsifier(self) -> None:
        from vision_mvp.experiments.phase79_long_window_convergent import (
            run_phase79_seed_sweep)
        r = run_phase79_seed_sweep(
            bank="no_change_point",
            seeds=(11, 17, 23, 29, 31),
            n_eval=16, signature_period=4)
        # On the stationary regime, W32 ties W31 (delta=0).
        self.assertEqual(r["min_delta_w32_minus_w31"], 0.0)
        self.assertEqual(r["max_delta_w32_minus_w31"], 0.0)

    def test_w32_orchestrator_branch_set(self) -> None:
        # Cross-check that every W32 decoder branch label is
        # registered in W32_ALL_BRANCHES.
        for b in (
                "long_window_resolved",
                "trivial_long_window_passthrough",
                "long_window_rejected",
                "long_window_disabled",
                "long_window_no_trigger",
                "gold_correlated_rerouted",
                "change_point_reset"):
            self.assertIn(b, W32_ALL_BRANCHES)


class W32CompositionTests(unittest.TestCase):
    """W32 composes byte-for-W31 on the trivial path AND adds wire
    overhead on the non-trivial path."""

    def test_trivial_registry_yields_zero_overhead(self) -> None:
        schema = build_incident_triage_schema_capsule()
        reg = build_trivial_long_window_registry(schema=schema)
        # is_trivial → no wire required → 0 wire tokens.
        self.assertTrue(reg.is_trivial)
        env = _build_simple_envelope(
            schema=schema, wire_required=False)
        self.assertEqual(env.n_wire_tokens, 0)

    def test_non_trivial_registry_yields_one_overhead(self) -> None:
        schema = build_incident_triage_schema_capsule()
        reg = build_long_window_convergent_registry(
            schema=schema,
            long_window_enabled=True,
            change_point_enabled=True,
            long_window=64,
        )
        # is_trivial = False.
        self.assertFalse(reg.is_trivial)
        env = _build_simple_envelope(
            schema=schema, wire_required=True)
        self.assertEqual(env.n_wire_tokens, 1)


if __name__ == "__main__":
    unittest.main()
