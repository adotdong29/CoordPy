"""W53 smoke driver — runs a 4-turn W53Team scenario and prints
the outer-CID chain w47 → w48 → w49 → w50 → w51 → w52 → w53.

Pure-Python only. Uses a deterministic synthetic backend so the
output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w53_team import (
    W53Team,
    build_w53_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w53.smoke",
        default_response="W53_OK")
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
    reg = build_w53_registry(
        schema_cid="w53_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W53Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run(
        "W53 smoke task: traverse four roles with V5 "
        "persistent state + quint translator V3 + MLSC "
        "merge + L=10 deep stack V4 + ECC K1xK2xK3xK4 "
        "compression + V5 reconstruction + BMM V3 "
        "consensus + corruption-robust carrier + "
        "transcript-vs-shared arbiter V2 + "
        "uncertainty layer")
    print("=" * 72)
    print(
        f"W53 smoke driver — "
        f"{len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    print(
        f"final_output:                  "
        f"{result.final_output!r}")
    print(
        f"w52_outer_cid:                 "
        f"{result.w52_outer_cid}")
    print(
        f"w53_params_cid:                "
        f"{result.w53_params_cid}")
    print(
        f"w53_outer_cid:                 "
        f"{result.w53_outer_cid}")
    print(
        f"composite_confidence_mean:     "
        f"{result.composite_confidence_mean:.4f}")
    print(
        f"arbiter_pick_rate_shared_mean: "
        f"{result.arbiter_pick_rate_shared_mean:.4f}")
    print(
        f"persistent_v5_chain (n):       "
        f"{len(result.persistent_v5_state_cids)}")
    print(
        f"mlsc_capsule_chain (n):        "
        f"{len(result.mlsc_capsule_cids)}")
    print()
    print("Per-turn W53 witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(
            f"    PV5:      "
            f"{b.persistent_v5_witness_cid[:32]}")
        print(
            f"    Quint:    "
            f"{b.quint_translator_witness_cid[:32]}")
        print(
            f"    MLSC:     "
            f"{b.mlsc_witness_cid[:32]}")
        print(
            f"    DSV4:     "
            f"{b.deep_stack_v4_forward_witness_cid[:32]}")
        print(
            f"    ECC:      "
            f"{b.ecc_compression_witness_cid[:32]}")
        print(
            f"    LHRV5:    "
            f"{b.long_horizon_v5_witness_cid[:32]}")
        print(
            f"    BMMV3:    "
            f"{b.branch_merge_memory_v3_witness_cid[:32]}")
        print(
            f"    CRC:      "
            f"{b.corruption_robust_carrier_witness_cid[:32]}")
        print(
            f"    TVSv2:    "
            f"{b.transcript_vs_shared_arbiter_v2_witness_cid[:32]}")
        print(
            f"    Uncert:   "
            f"{b.uncertainty_layer_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(
        "  w47_outer → w48_proxy_outer → "
        "w49_multi_block_outer")
    print(
        "  → w50_outer → w51_outer → w52_outer → "
        "w53_outer")
    print(
        f"  → w52_outer_cid: "
        f"{result.w52_outer_cid[:32]}")
    print(
        f"  → w53_outer_cid: "
        f"{result.w53_outer_cid[:32]}")
    print()
    print(
        "Done. No version bump (coordpy stays at 0.5.20). "
        "Reachable via explicit `from coordpy.w53_team "
        "import W53Team`.")


if __name__ == "__main__":
    main()
