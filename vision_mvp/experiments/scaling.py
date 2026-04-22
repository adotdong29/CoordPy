"""Scaling experiment — sweep N across protocols.

For each N ∈ {10, 50, 200, 1000} and multiple seeds, run every protocol on
a consensus task and record:
  - Total tokens (system-wide communication)
  - Peak per-agent context (the CASR metric)
  - Task accuracy, agreement error

Output: CSV-like table and a tidy markdown summary.
"""

from __future__ import annotations
import sys, os, math, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from vision_mvp.tasks.consensus import ConsensusTask
from vision_mvp.protocols.naive import run_naive
from vision_mvp.protocols.gossip import run_gossip
from vision_mvp.protocols.manifold_only import run_manifold_only
from vision_mvp.protocols.full_stack import run_full


PROTOCOLS = [
    ("naive", run_naive),
    ("gossip", run_gossip),
    ("manifold", run_manifold_only),
    ("full", run_full),
]


def sweep(n_values: list[int] | None = None,
          dim: int = 64,
          noise: float = 1.0,
          seeds: list[int] | None = None,
          skip_naive_above: int = 500,
          rounds_full: int = 2) -> list[dict]:
    if n_values is None:
        n_values = [10, 50, 200, 1000]
    if seeds is None:
        seeds = [1, 2, 3]

    all_rows: list[dict] = []
    for n in n_values:
        for s in seeds:
            intrinsic_rank = max(2, math.ceil(math.log2(max(n, 2))))
            task = ConsensusTask(n_agents=n, dim=dim, noise=noise, seed=s,
                                 intrinsic_rank=intrinsic_rank)
            task.generate()

            for name, fn in PROTOCOLS:
                # Skip naive at very large N — it's quadratic and would take forever
                if name == "naive" and n > skip_naive_above:
                    continue
                task.generate()  # reset task state
                t0 = time.time()
                if name == "full":
                    r = fn(task, rounds=rounds_full)
                else:
                    r = fn(task)
                wall = time.time() - t0
                row = r.as_row()
                row["seed"] = s
                row["rank"] = intrinsic_rank
                row["wall_sec"] = round(wall, 4)
                all_rows.append(row)
    return all_rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Average numeric columns across seeds per (protocol, N)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["protocol"], r["N"])
        groups.setdefault(key, []).append(r)
    out = []
    for (protocol, n), group in groups.items():
        agg = {"protocol": protocol, "N": n, "seeds": len(group),
               "d": group[0]["d"], "rank": group[0]["rank"]}
        for k in ("total_tokens", "n_messages", "peak_agent_context",
                 "mean_agent_context", "rounds", "mean_accuracy_error",
                 "max_accuracy_error", "agreement_error", "oracle_error",
                 "wall_sec"):
            vals = [r[k] for r in group]
            agg[k] = sum(vals) / len(vals)
        out.append(agg)
    out.sort(key=lambda x: (x["N"], x["protocol"]))
    return out


def format_table(rows: list[dict]) -> str:
    cols = ["N", "protocol", "peak_agent_context", "total_tokens",
            "rounds", "mean_accuracy_error", "agreement_error", "oracle_error"]
    widths = {c: max(len(c), 14) for c in cols}
    lines = []
    lines.append(" | ".join(c.rjust(widths[c]) for c in cols))
    lines.append("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}".rjust(widths[c]))
            else:
                cells.append(str(v).rjust(widths[c]))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="+",
                    default=[10, 50, 200, 1000])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--skip-naive-above", type=int, default=500)
    args = ap.parse_args()

    rows = sweep(n_values=args.n_values, dim=args.dim, noise=args.noise,
                 seeds=args.seeds, skip_naive_above=args.skip_naive_above)
    agg = aggregate(rows)
    table = format_table(agg)
    print(table)

    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps({"rows": rows, "aggregate": agg}, indent=2))
        print(f"\nWrote raw+aggregate to {args.out}")
