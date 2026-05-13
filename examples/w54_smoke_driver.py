"""W54 smoke driver — runs a 4-turn W54Team scenario and prints
the outer-CID chain w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54.

Pure-Python only. Uses a deterministic synthetic backend so the
output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w54_team import (
    W54Team,
    build_w54_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w54.smoke",
        default_response="W54_OK")
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
    reg = build_w54_registry(
        schema_cid="w54_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W54Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run(
        "W54 smoke task: traverse four roles with V6 "
        "persistent state + hex translator V4 + MLSC V2 "
        "merge + consensus controller + L=12 deep stack V5 "
        "+ ECC V6 K1xK2xK3xK4xK5 + Hamming compression + V6 "
        "reconstruction + CRC V2 single-bit correction + "
        "transcript-vs-shared arbiter V3 + "
        "uncertainty layer V2")
    print("=" * 72)
    print(
        f"W54 smoke driver — "
        f"{len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    print(
        f"final_output:                  "
        f"{result.final_output!r}")
    print(
        f"w53_outer_cid:                 "
        f"{result.w53_outer_cid}")
    print(
        f"w54_params_cid:                "
        f"{result.w54_params_cid}")
    print(
        f"w54_outer_cid:                 "
        f"{result.w54_outer_cid}")
    print(
        f"composite_confidence_v2_mean:  "
        f"{result.composite_confidence_mean_v2:.4f}")
    print(
        f"arbiter_pick_rate_merge_mean:  "
        f"{result.arbiter_pick_rate_merge_mean:.4f}")
    print(
        f"persistent_v6_chain (n):       "
        f"{len(result.persistent_v6_state_cids)}")
    print(
        f"mlsc_v2_capsule_chain (n):     "
        f"{len(result.mlsc_v2_capsule_cids)}")
    print()
    print("Per-turn W54 witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(
            f"    PV6:      "
            f"{b.persistent_v6_witness_cid[:32]}")
        print(
            f"    HEX:      "
            f"{b.hex_translator_witness_cid[:32]}")
        print(
            f"    MLSC v2:  "
            f"{b.mlsc_v2_witness_cid[:32]}")
        print(
            f"    Consens:  "
            f"{b.consensus_controller_witness_cid[:32]}")
        print(
            f"    DSV5:     "
            f"{b.deep_stack_v5_witness_cid[:32]}")
        print(
            f"    ECC V6:   "
            f"{b.ecc_v6_compression_witness_cid[:32]}")
        print(
            f"    LHR V6:   "
            f"{b.long_horizon_v6_witness_cid[:32]}")
        print(
            f"    CRC V2:   "
            f"{b.crc_v2_witness_cid[:32]}")
        print(
            f"    TVS v3:   "
            f"{b.tvs_arbiter_v3_witness_cid[:32]}")
        print(
            f"    Uncert v2:"
            f"{b.uncertainty_v2_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(
        "  w47_outer → w48_proxy_outer → "
        "w49_multi_block_outer")
    print(
        "  → w50_outer → w51_outer → w52_outer → "
        "w53_outer → w54_outer")
    print(
        f"  → w53_outer_cid: "
        f"{result.w53_outer_cid[:32]}")
    print(
        f"  → w54_outer_cid: "
        f"{result.w54_outer_cid[:32]}")
    print()
    print(
        "Done. No version bump (coordpy stays at 0.5.20). "
        "Reachable via explicit `from coordpy.w54_team "
        "import W54Team`.")


if __name__ == "__main__":
    main()
