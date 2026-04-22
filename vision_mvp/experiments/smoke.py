"""Smoke test — one run of each protocol at small N."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.consensus import ConsensusTask
from vision_mvp.protocols.naive import run_naive
from vision_mvp.protocols.gossip import run_gossip
from vision_mvp.protocols.manifold_only import run_manifold_only
from vision_mvp.protocols.full_stack import run_full


def smoke(n: int = 20, d: int = 16, noise: float = 1.0, intrinsic_rank: int | None = None):
    if intrinsic_rank is None:
        import math
        intrinsic_rank = max(2, math.ceil(math.log2(max(n, 2))))
    task = ConsensusTask(n_agents=n, dim=d, noise=noise, seed=42,
                         intrinsic_rank=intrinsic_rank)
    task.generate()

    results = []
    for fn, name in [
        (run_naive, "naive"),
        (run_gossip, "gossip"),
        (run_manifold_only, "manifold"),
        (run_full, "full"),
    ]:
        # Re-generate so agents start with identical observations
        task.generate()
        r = fn(task)
        results.append(r)
        print(f"\n{name}:")
        print(f"  rounds: {r.rounds}")
        for k, v in r.bus_summary.items():
            print(f"  {k}: {v:.2f}")
        for k, v in r.task_metrics.items():
            print(f"  {k}: {v:.4f}")
    return results


if __name__ == "__main__":
    smoke()
