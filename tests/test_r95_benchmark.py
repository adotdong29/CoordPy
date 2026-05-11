"""Tests for the R-95 benchmark family — H1..H14 hypotheses.

Each test runs one R-95 family on a small seed set and asserts
the pre-committed success bar of the corresponding hypothesis
from ``docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md``.
"""

from __future__ import annotations

from coordpy.r95_benchmark import (
    run_family,
    family_branch_cycle_bias,
    family_branch_history_compression,
    family_latent_control_round_trip,
    family_multi_head_specialisation,
    family_proxy_distribution_cap,
    family_proxy_envelope_verifier,
    family_proxy_falsifier,
    family_pseudo_kv_reuse,
    family_reconstruction_objective,
    family_replay_determinism,
    family_shared_state_aware_backend,
    family_shared_state_cid_stability,
    family_trivial_shared_state_passthrough,
    family_write_gate_selectivity,
)


SEEDS = (0, 1, 2)


# =============================================================================
# H1
# =============================================================================

class TestH1TrivialPassthrough:
    def test_passthrough_ok_across_all_arms(self):
        comp = run_family(
            "r95_trivial_shared_state_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w43_closed_form",
                "w44_live_coupled", "w45_learned_coupled",
                "w46_memory_coupled", "w47_autograd",
                "w48_shared_state"):
            a = comp.get(arm)
            assert a is not None, arm
            assert a.mean == 1.0, (arm, a.values)


# =============================================================================
# H2
# =============================================================================

class TestH2SharedStateCIDStability:
    def test_shared_state_cid_stable_across_turns(self):
        comp = run_family(
            "r95_shared_state_cid_stability", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0


# =============================================================================
# H3
# =============================================================================

class TestH3PseudoKVReuse:
    def test_pseudo_kv_reuse_beats_w47_zero(self):
        comp = run_family("r95_pseudo_kv_reuse", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        w47 = comp.get("w47_autograd")
        assert w48 is not None and w47 is not None
        assert w48.mean >= 0.5
        assert w48.mean - w47.mean >= 0.5


# =============================================================================
# H4
# =============================================================================

class TestH4MultiHeadSpecialisation:
    def test_multi_head_diversity_positive(self):
        comp = run_family(
            "r95_multi_head_specialisation", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        w47 = comp.get("w47_autograd")
        assert w48 is not None and w47 is not None
        assert w48.mean > 0.0
        assert w47.mean == 0.0


# =============================================================================
# H5
# =============================================================================

class TestH5ReconstructionObjective:
    def test_reconstruction_beats_baseline(self):
        comp = run_family(
            "r95_reconstruction_objective", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0


# =============================================================================
# H6
# =============================================================================

class TestH6BranchCycleBias:
    def test_branch_split_acc_at_least_0_9(self):
        comp = run_family("r95_branch_cycle_bias", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.min >= 0.9


# =============================================================================
# H7
# =============================================================================

class TestH7WriteGateSelectivity:
    def test_selectivity_positive(self):
        comp = run_family(
            "r95_write_gate_selectivity", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        w47 = comp.get("w47_autograd")
        assert w48 is not None
        assert w48.mean > 0.30
        # W47 has no write gate trained on this regime.
        assert w47.mean == 0.0


# =============================================================================
# H8
# =============================================================================

class TestH8LatentControlRoundTrip:
    def test_latent_control_round_trip_ok(self):
        comp = run_family(
            "r95_latent_control_round_trip", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0


# =============================================================================
# H9
# =============================================================================

class TestH9BranchHistoryCompression:
    def test_save_ratio_at_least_0_5(self):
        comp = run_family(
            "r95_branch_history_compression", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.min >= 0.5


# =============================================================================
# H10
# =============================================================================

class TestH10ReplayDeterminism:
    def test_replay_determinism_ok(self):
        comp = run_family(
            "r95_replay_determinism", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0


# =============================================================================
# H11
# =============================================================================

class TestH11ProxyEnvelopeVerifier:
    def test_verifier_soundness_ok(self):
        comp = run_family(
            "r95_proxy_envelope_verifier", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0


# =============================================================================
# H12
# =============================================================================

class TestH12ProxyDistributionCap:
    def test_limitation_reproduces_low_protect_rate(self):
        # Acceptance: mean downstream_protect_rate ≤ 0.9 — the
        # W48 mechanism is not consistently protective under
        # adversarial training distribution. This honestly
        # reproduces W48-L-PROXY-DISTRIBUTION-CAP.
        comp = run_family(
            "r95_proxy_distribution_cap", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean <= 0.9


# =============================================================================
# H13
# =============================================================================

class TestH13SharedStateAwareBackend:
    def test_w48_strictly_beats_w47(self):
        comp = run_family(
            "r95_shared_state_aware_backend", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        w47 = comp.get("w47_autograd")
        assert w48 is not None and w47 is not None
        assert w48.mean >= 0.9
        assert w48.mean - w47.mean >= 0.9


# =============================================================================
# H14
# =============================================================================

class TestH14SDKByteIdentityPreserved:
    def test_sdk_byte_identity_preserved(self):
        comp = run_family("r95_proxy_falsifier", seeds=SEEDS)
        w48 = comp.get("w48_shared_state")
        assert w48 is not None
        assert w48.mean == 1.0
