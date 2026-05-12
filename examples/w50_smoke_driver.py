"""W50 smoke driver — runs a 4-turn W50Team scenario and prints
the outer-CID chain w47 → w48 → w49 → w50.

Pure-Python only. Uses a deterministic synthetic backend so the
output is reproducible across runs.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w50_team import (
    W50Team,
    build_w50_registry,
)


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w50.smoke",
        default_response="W50_OK")
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
    reg = build_w50_registry(
        schema_cid="w50_smoke_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    result = team.run("W50 smoke task: traverse four roles")
    print("=" * 60)
    print(f"W50 smoke driver — {len(result.turn_witness_bundles)} turns")
    print("=" * 60)
    print(f"final_output:           {result.final_output!r}")
    print(f"w49_root_cid:           {result.w49_root_cid}")
    print(f"w50_params_cid:         {result.w50_params_cid}")
    print(f"w50_outer_cid:          {result.w50_outer_cid}")
    print(f"anchor_status:          {result.anchor_status}")
    print(f"carrier_chain (n):      {len(result.carrier_chain_cids)}")
    print()
    print("Per-turn witness bundle CIDs:")
    for i, b in enumerate(result.turn_witness_bundles):
        print(f"  turn {i}:")
        print(f"    CB:       {b.cross_backend_witness_cid[:32]}")
        print(f"    Deep:     {b.deep_proxy_forward_witness_cid[:32]}")
        print(f"    AC:       {b.adaptive_compression_witness_cid[:32]}")
        print(f"    Cram:     {b.cramming_witness_v2_cid[:32]}")
        print(f"    XBT:      {b.cross_bank_transfer_witness_cid[:32]}")
        print(f"    SLCV2:    {b.shared_latent_carrier_witness_cid[:32]}")
        print(f"    RecV2:    {b.reconstruction_v2_witness_cid[:32]}")
    print()
    print("Envelope chain:")
    print(f"  w47_outer → w48_proxy_outer → w49_multi_block_outer")
    print(f"  → w49_root_cid: {result.w49_root_cid[:32]}")
    print(f"  → w50_outer:    {result.w50_outer_cid[:32]}")
    print()
    print("Done. No version bump (coordpy stays at 0.5.20). "
          "Reachable via explicit `from coordpy.w50_team import "
          "W50Team`.")


if __name__ == "__main__":
    main()
