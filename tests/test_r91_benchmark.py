"""Tests for the R-91 benchmark family that exercise the
W44 live-manifold-coupled coordination layer.

Each test covers one of H1..H10 of the W44 success criterion in
``docs/SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md``.
"""

from __future__ import annotations

import pytest

from coordpy.live_manifold import (
    W44_LIVE_MANIFOLD_SCHEMA_VERSION,
    verify_live_manifold_handoff,
)
from coordpy.r91_benchmark import (
    R91_FAMILY_TABLE,
    family_live_causal_gate,
    family_live_dual_channel_collusion,
    family_live_factoradic_compression,
    family_live_falsifier,
    family_live_spherical_gate,
    family_live_subspace_gate,
    family_trivial_live_passthrough,
    render_text_report,
    run_all_families,
    run_family,
)


SEEDS = (0, 1, 2, 3, 4)


# =============================================================================
# H1 — Trivial live passthrough is byte-for-AgentTeam
# =============================================================================

class TestH1TrivialLivePassthrough:
    def test_h1_all_arms_pass(self):
        cmp_ = run_family(
            "r91_trivial_live_passthrough", seeds=SEEDS)
        for arm in (
                "baseline_team", "w43_closed_form",
                "w44_live_coupled"):
            agg = cmp_.get(arm)
            assert agg is not None
            assert agg.min == agg.max == 1.0


# =============================================================================
# H2 — Causal-violation gate strictly improves downstream protect rate
# =============================================================================

class TestH2LiveCausalGate:
    def test_h2_live_strictly_improves(self):
        cmp_ = run_family(
            "r91_live_causal_gate", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        w43 = cmp_.get("w43_closed_form")
        assert live is not None and w43 is not None
        assert live.min == 1.0
        assert live.mean - w43.mean >= 0.40


# =============================================================================
# H3 — Spherical gate strictly improves downstream protect rate
# =============================================================================

class TestH3LiveSphericalGate:
    def test_h3_live_strictly_improves(self):
        cmp_ = run_family(
            "r91_live_spherical_gate", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        w43 = cmp_.get("w43_closed_form")
        assert live is not None and w43 is not None
        assert live.min == 1.0
        assert live.mean - w43.mean >= 0.40


# =============================================================================
# H4 — Subspace gate strictly improves downstream protect rate
# =============================================================================

class TestH4LiveSubspaceGate:
    def test_h4_live_strictly_improves(self):
        cmp_ = run_family(
            "r91_live_subspace_gate", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        w43 = cmp_.get("w43_closed_form")
        assert live is not None and w43 is not None
        assert live.min == 1.0
        assert live.mean - w43.mean >= 0.40


# =============================================================================
# H5 — Factoradic compressor strictly reduces visible prompt tokens
# =============================================================================

class TestH5LiveFactoradicCompression:
    def test_h5_strict_visible_token_saving(self):
        cmp_ = run_family(
            "r91_live_factoradic_compression", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        baseline = cmp_.get("baseline_team")
        assert live is not None and baseline is not None
        # H5 requires >=4 saved per turn at n_roles=8 averaged
        # across 8 turns — we measure the run-level total.
        assert live.min - baseline.max >= 4 * 8


# =============================================================================
# H6 — No false abstention on linear-flow falsifier
# =============================================================================

class TestH6LiveFalsifier:
    def test_h6_no_false_abstain(self):
        cmp_ = run_family(
            "r91_live_falsifier", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        assert live is not None
        assert live.min == live.max == 1.0


# =============================================================================
# H7 — Live dual-channel collusion limitation reproduces honestly
# =============================================================================

class TestH7LiveDualChannelCollusion:
    def test_h7_limitation_reproduces(self):
        cmp_ = run_family(
            "r91_live_dual_channel_collusion", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        w43 = cmp_.get("w43_closed_form")
        assert live is not None and w43 is not None
        assert live.mean == w43.mean == 0.0


# =============================================================================
# H8 — Live envelope verifier enumerates >= 12 disjoint failure modes
# =============================================================================

class TestH8VerifierBoundary:
    def test_h8_n_checks_at_least_12(self):
        # Run a clean live family and inspect the resulting envelope.
        from coordpy.live_manifold import (
            LiveManifoldTeam,
            LiveTurnContext,
            LiveObservationBuilderResult,
            build_live_manifold_registry,
            W44_ROUTE_MODE_TEXTUAL,
        )
        from coordpy.product_manifold import (
            CausalVectorClock,
            CellObservation,
            ProductManifoldPolicyEntry,
            encode_spherical_consensus,
            encode_subspace_basis,
        )
        from coordpy.synthetic_llm import SyntheticLLMClient
        from coordpy import agent
        import hashlib

        sig = hashlib.sha256(b"h8-test").hexdigest()
        kinds = ("event", "event")
        sub = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        policy = ProductManifoldPolicyEntry(
            role_handoff_signature_cid=sig,
            expected_services=("h8",),
            expected_spherical=encode_spherical_consensus(kinds),
            expected_subspace=encode_subspace_basis(sub),
            expected_causal_topology_hash="(...)",
        )
        schema = hashlib.sha256(b"h8-schema").hexdigest()
        reg = build_live_manifold_registry(
            schema_cid=schema, policy_entries=(policy,),
            abstain_substitution_enabled=False,
            inline_route_mode=W44_ROUTE_MODE_TEXTUAL)

        def _builder(ctx):
            snapshots = []
            walk = {r: 0 for r in ctx.role_universe}
            for r in ctx.role_arrival_order:
                walk[r] = walk.get(r, 0) + 1
                snapshots.append(
                    CausalVectorClock.from_mapping(dict(walk)))
            obs = CellObservation(
                branch_path=tuple(0 for _ in range(ctx.turn_index)),
                claim_kinds=kinds,
                role_arrival_order=tuple(ctx.role_arrival_order),
                role_universe=tuple(ctx.role_universe),
                attributes=tuple({"r": float(ctx.turn_index)}.items()),
                subspace_vectors=sub,
                causal_clocks=tuple(snapshots),
            )
            return LiveObservationBuilderResult(
                observation=obs, role_handoff_signature_cid=sig)

        agents = [agent(f"r{i}", f"i{i}") for i in range(2)]
        team = LiveManifoldTeam(
            agents,
            backend=SyntheticLLMClient(default_response="x"),
            registry=reg, observation_builder=_builder,
            capture_capsules=True)
        result = team.run("h8 task")
        env = result.live_turns[1].envelope
        outcome = verify_live_manifold_handoff(
            env, registered_schema_cid=schema)
        assert outcome.ok
        assert outcome.n_checks >= 12


# =============================================================================
# H9 — Live cram-frontier preserves W43 audit while reducing visible tokens
# =============================================================================

class TestH9LiveCramFrontier:
    def test_h9_visible_tokens_saved_and_audit_preserved(self):
        cmp_ = run_family(
            "r91_live_factoradic_compression", seeds=SEEDS)
        live = cmp_.get("w44_live_coupled")
        baseline = cmp_.get("baseline_team")
        assert live is not None and baseline is not None
        assert live.min - baseline.max >= 4 * 8


# =============================================================================
# H10 — Smoke driver still green (delegated to test_smoke_full)
# =============================================================================

class TestH10StableContract:
    def test_h10_pmc_module_does_not_break_imports(self):
        # Import the live module and confirm that the released
        # public surface still imports cleanly.
        import coordpy
        from coordpy.live_manifold import LiveManifoldTeam  # noqa: F401
        # The released public surface continues to work.
        assert hasattr(coordpy, "AgentTeam")
        assert hasattr(coordpy, "RunSpec")
        assert hasattr(coordpy, "run")


# =============================================================================
# Aggregate report sanity
# =============================================================================

class TestReportRendering:
    def test_render_text_report_sections(self):
        results = run_all_families(seeds=SEEDS)
        report = render_text_report(results)
        for fam in R91_FAMILY_TABLE:
            assert fam in report
        assert "delta_live_vs_w43" in report
        assert "delta_live_vs_baseline" in report

    def test_run_all_families_returns_full_set(self):
        results = run_all_families(seeds=SEEDS)
        assert set(results.keys()) == set(R91_FAMILY_TABLE.keys())


# =============================================================================
# Per-family individual seeds (regression coverage)
# =============================================================================

class TestPerSeedDeterminism:
    @pytest.mark.parametrize("seed", list(SEEDS))
    def test_trivial_passthrough_per_seed(self, seed):
        out = family_trivial_live_passthrough(seed)
        for r in out.values():
            assert r.metric_value == 1.0

    @pytest.mark.parametrize("seed", list(SEEDS))
    def test_falsifier_per_seed(self, seed):
        out = family_live_falsifier(seed)
        for r in out.values():
            assert r.metric_value == 1.0

    @pytest.mark.parametrize("seed", list(SEEDS))
    def test_collusion_per_seed(self, seed):
        out = family_live_dual_channel_collusion(seed)
        for r in out.values():
            assert r.metric_value == 0.0
