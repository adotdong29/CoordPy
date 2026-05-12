"""W51 smoke driver — runs a 4-turn W51Team scenario and
prints the outer-CID chain w47 → w48 → w49 → w50 → w51.

Pure-Python only. Uses a deterministic synthetic backend so
the output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w51_team import (
    W51Team,
    build_w51_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w51.smoke",
        default_response="W51_OK")
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
    reg = build_w51_registry(
        schema_cid="w51_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run("W51 smoke task: traverse four roles "
                       "with persistent state + triple-backend "
                       "+ deep-stack-V2 + hierarchical "
                       "compression + long-horizon retention "
                       "+ branch/cycle memory")
    print("=" * 72)
    print(f"W51 smoke driver — {len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    print(f"final_output:                  {result.final_output!r}")
    print(f"w50_outer_cid:                 {result.w50_outer_cid}")
    print(f"w51_params_cid:                {result.w51_params_cid}")
    print(f"w51_outer_cid:                 {result.w51_outer_cid}")
    print(f"triple_anchor_status:          {result.triple_anchor_status}")
    print(f"persistent_state_chain (n):    "
          f"{len(result.persistent_state_cids)}")
    print()
    print("Per-turn W51 witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(f"    PS:       {b.persistent_state_witness_cid[:32]}")
        print(f"    TB:       {b.triple_backend_witness_cid[:32]}")
        print(f"    Deep V2:  {b.deep_stack_v2_forward_witness_cid[:32]}")
        print(f"    HierC:    {b.hierarchical_compression_witness_cid[:32]}")
        print(f"    CramV3:   {b.cramming_witness_v3_cid[:32]}")
        print(f"    LHR:      {b.long_horizon_reconstruction_witness_cid[:32]}")
        print(f"    BCM:      {b.branch_cycle_memory_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(f"  w47_outer → w48_proxy_outer → w49_multi_block_outer")
    print(f"  → w50_outer → w51_outer")
    print(f"  → w50_outer_cid: {result.w50_outer_cid[:32]}")
    print(f"  → w51_outer_cid: {result.w51_outer_cid[:32]}")
    print()
    print("Done. No version bump (coordpy stays at 0.5.20). "
          "Reachable via explicit `from coordpy.w51_team import "
          "W51Team`.")


if __name__ == "__main__":
    main()
