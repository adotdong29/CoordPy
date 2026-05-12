"""W52 smoke driver — runs a 4-turn W52Team scenario and
prints the outer-CID chain w47 → w48 → w49 → w50 → w51 → w52.

Pure-Python only. Uses a deterministic synthetic backend so the
output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w52_team import (
    W52Team,
    build_w52_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w52.smoke",
        default_response="W52_OK")
    agents = [
        Agent(name="alpha", instructions="explore", role="r0",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="verify", role="r1",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="charlie", instructions="commit", role="r2",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="delta", instructions="seal", role="r3",
              backend=backend, temperature=0.0, max_tokens=24),
    ]
    reg = build_w52_registry(
        schema_cid="w52_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run(
        "W52 smoke task: traverse four roles with stacked "
        "persistent V4 + multi-hop quad translator + L=8 deep "
        "stack V3 + quantised K1xK2xK3 compression + V4 "
        "reconstruction + BCM V2 + role-graph transfer + "
        "transcript-vs-shared-state comparator")
    print("=" * 72)
    print(f"W52 smoke driver — {len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    print(f"final_output:                  {result.final_output!r}")
    print(f"w51_outer_cid:                 {result.w51_outer_cid}")
    print(f"w52_params_cid:                {result.w52_params_cid}")
    print(f"w52_outer_cid:                 {result.w52_outer_cid}")
    print(f"multi_hop_anchor_status:       {result.multi_hop_anchor_status}")
    print(f"persistent_v4_chain (n):       "
          f"{len(result.persistent_v4_state_cids)}")
    print()
    print("Per-turn W52 witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(f"    PV4:      {b.persistent_v4_witness_cid[:32]}")
        print(f"    MH:       {b.multi_hop_witness_cid[:32]}")
        print(f"    DSV3:     {b.deep_stack_v3_forward_witness_cid[:32]}")
        print(f"    QuantC:   {b.quantised_compression_witness_cid[:32]}")
        print(f"    CramV4:   {b.cramming_witness_v4_cid[:32]}")
        print(f"    LHRV4:    {b.long_horizon_v4_witness_cid[:32]}")
        print(f"    BCMV2:    {b.branch_cycle_memory_v2_witness_cid[:32]}")
        print(f"    RG:       {b.role_graph_witness_cid[:32]}")
        print(f"    TVS:      {b.transcript_vs_shared_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(f"  w47_outer → w48_proxy_outer → w49_multi_block_outer")
    print(f"  → w50_outer → w51_outer → w52_outer")
    print(f"  → w51_outer_cid: {result.w51_outer_cid[:32]}")
    print(f"  → w52_outer_cid: {result.w52_outer_cid[:32]}")
    print()
    print("Done. No version bump (coordpy stays at 0.5.20). "
          "Reachable via explicit `from coordpy.w52_team import "
          "W52Team`.")


if __name__ == "__main__":
    main()
