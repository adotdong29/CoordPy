"""W55 smoke driver — runs a 4-turn W55Team scenario and prints
the outer-CID chain
w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54 → w55.

Pure-Python only. Uses a deterministic synthetic backend so the
output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w55_team import (
    W55Team,
    build_w55_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w55.smoke",
        default_response="W55_OK")
    agents = [
        Agent(name="alpha", instructions="explore",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="verify",
              role="r1", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="charlie", instructions="commit",
              role="r2", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="delta", instructions="seal",
              role="r3", backend=backend,
              temperature=0.0, max_tokens=24),
    ]
    reg = build_w55_registry(
        schema_cid="w55_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W55Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run(
        "W55 smoke task: traverse four roles with V7 "
        "triple-skip persistent state + hept translator V5 "
        "(7-backend) + MLSC V3 trust-decay capsules + TWCC "
        "5-stage controller + L=14 deep stack V6 + ECC V7 "
        "K1xK2xK3xK4xK5xK6 + BCH(15,7) compression + V7 "
        "reconstruction (cross-cycle) + CRC V3 double-bit "
        "correction + bit-interleave + transcript-vs-shared "
        "arbiter V4 (5-arm) + uncertainty layer V3 + "
        "disagreement algebra")
    print("=" * 72)
    print(
        f"W55 smoke driver — "
        f"{len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    print(
        f"final_output:                  "
        f"{result.final_output!r}")
    print(
        f"w54_outer_cid:                 "
        f"{result.w54_outer_cid}")
    print(
        f"w55_params_cid:                "
        f"{result.w55_params_cid}")
    print(
        f"w55_outer_cid:                 "
        f"{result.w55_outer_cid}")
    print(
        f"composite_confidence_v3_mean:  "
        f"{result.composite_confidence_mean_v3:.4f}")
    print(
        f"trust_weighted_composite_mean: "
        f"{result.trust_weighted_composite_mean:.4f}")
    print(
        f"twcc_quorum_rate:              "
        f"{result.twcc_quorum_rate:.4f}")
    print(
        f"persistent_v7_chain (n):       "
        f"{len(result.persistent_v7_state_cids)}")
    print(
        f"mlsc_v3_capsule_chain (n):     "
        f"{len(result.mlsc_v3_capsule_cids)}")
    print()
    print("Per-turn W55 witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(
            f"    PV7:      "
            f"{b.persistent_v7_witness_cid[:32]}")
        print(
            f"    HEPT:     "
            f"{b.hept_translator_witness_cid[:32]}")
        print(
            f"    MLSC v3:  "
            f"{b.mlsc_v3_witness_cid[:32]}")
        print(
            f"    TWCC:     "
            f"{b.twcc_witness_cid[:32]}")
        print(
            f"    DSV6:     "
            f"{b.deep_stack_v6_witness_cid[:32]}")
        print(
            f"    ECC V7:   "
            f"{b.ecc_v7_compression_witness_cid[:32]}")
        print(
            f"    LHR V7:   "
            f"{b.long_horizon_v7_witness_cid[:32]}")
        print(
            f"    CRC V3:   "
            f"{b.crc_v3_witness_cid[:32]}")
        print(
            f"    TVS v4:   "
            f"{b.tvs_arbiter_v4_witness_cid[:32]}")
        print(
            f"    Uncert v3:"
            f"{b.uncertainty_v3_witness_cid[:32]}")
        print(
            f"    Algebra:  "
            f"{b.disagreement_algebra_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(
        "  w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54 → w55")
    print(
        f"  → w54_outer_cid: "
        f"{result.w54_outer_cid[:32]}")
    print(
        f"  → w55_outer_cid: "
        f"{result.w55_outer_cid[:32]}")
    print()
    print(
        "Done. No version bump (coordpy stays at 0.5.20). "
        "Reachable via explicit `from coordpy.w55_team "
        "import W55Team`.")


if __name__ == "__main__":
    main()
