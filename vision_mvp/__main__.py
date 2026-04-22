"""CLI entry point:

    python3 -m vision_mvp <command> [options]

Commands
--------
  demo            Run a minimal CASRRouter demo at N=200 and print stats.
  scale SWEEP     Re-run the scaling sweep (= experiments/scaling.py).
  phase N         Re-run experiments for phase N ∈ {1..5}.
  test            Run the test suite.
  info            Print version + configuration info.
"""

from __future__ import annotations
import argparse
import math
import sys
import subprocess


def cmd_demo(args):
    import numpy as np
    from vision_mvp import CASRRouter

    N, d, rank = args.n, args.d, args.rank or max(2, math.ceil(math.log2(args.n)))
    rng = np.random.default_rng(args.seed)
    # Synthetic low-rank truth
    basis_raw = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(basis_raw)
    z = rng.standard_normal(rank)
    truth = Q @ z

    router = CASRRouter(
        n_agents=N, state_dim=d, task_rank=rank,
        observation_noise=1.0, seed=args.seed,
    )

    for t in range(args.rounds):
        obs = truth[None, :] + rng.standard_normal((N, d))
        router.step(obs)

    est = router.estimates
    err = float(np.linalg.norm(est - truth, axis=1).mean()
                / (np.linalg.norm(truth) + 1e-8))
    print(f"\nCASRRouter demo  —  N={N}, d={d}, rank={rank}, rounds={args.rounds}")
    print(f"  final tracking error: {err:.4f}")
    for k, v in router.stats.items():
        print(f"  {k}: {v}")


def cmd_scale(args):
    from .experiments.scaling import sweep, aggregate, format_table
    rows = sweep(n_values=args.n_values, dim=args.d, seeds=args.seeds,
                 skip_naive_above=args.skip_naive_above)
    agg = aggregate(rows)
    print(format_table(agg))


def cmd_phase(args):
    # Delegate to module entry points — these are already argparse-driven.
    modname = {
        1: "vision_mvp.experiments.scaling",
        2: "vision_mvp.experiments.phase2",
        3: "vision_mvp.experiments.phase3",
        4: "vision_mvp.experiments.phase4",
        5: "vision_mvp.experiments.phase5_llm",
    }[args.phase_num]
    subprocess.run([sys.executable, "-u", "-m", modname] + args.rest, check=False)


def cmd_test(args):
    subprocess.run([sys.executable, "-m", "unittest", "discover",
                    "-s", "vision_mvp/tests", "-v"], check=False)


def cmd_info(args):
    from vision_mvp import __version__
    print(f"vision_mvp {__version__}")
    print(f"Python     {sys.version}")
    import numpy; print(f"numpy      {numpy.__version__}")


def main():
    ap = argparse.ArgumentParser(prog="python -m vision_mvp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="run a quick CASRRouter demo")
    p_demo.add_argument("--n", type=int, default=200)
    p_demo.add_argument("--d", type=int, default=32)
    p_demo.add_argument("--rank", type=int, default=None)
    p_demo.add_argument("--rounds", type=int, default=20)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.set_defaults(func=cmd_demo)

    p_scale = sub.add_parser("scale", help="run the Phase-1 scaling sweep")
    p_scale.add_argument("--n-values", type=int, nargs="+",
                         default=[10, 50, 200, 1000])
    p_scale.add_argument("--d", type=int, default=64)
    p_scale.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p_scale.add_argument("--skip-naive-above", type=int, default=500)
    p_scale.set_defaults(func=cmd_scale)

    p_phase = sub.add_parser("phase", help="run a phase's experiments")
    p_phase.add_argument("phase_num", type=int, choices=[1, 2, 3, 4, 5])
    p_phase.add_argument("rest", nargs=argparse.REMAINDER)
    p_phase.set_defaults(func=cmd_phase)

    p_test = sub.add_parser("test", help="run the test suite")
    p_test.set_defaults(func=cmd_test)

    p_info = sub.add_parser("info", help="print version info")
    p_info.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
