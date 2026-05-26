#!/usr/bin/env python3
"""W102 — arsenal-mining extension (cross-bench HumanEval+ + MBPP+ V2).

Extends the W101 arsenal-mining report with two additional
cluster surfaces:

1. **HumanEval+** — re-executes every persisted W88 70B HumanEval
   candidate response against the V2 HumanEval+ `check()` block
   (the EvalPlus hardened test surface).  Produces a per-(seed,
   task_id, arm) cluster surface analogous to the W101 humaneval
   block.

2. **MBPP+ V2** — re-executes every persisted W91 5-seed 70B MBPP
   candidate response against the V2 MBPP+ extra `test` program
   (the EvalPlus hardened iteration loop).  Produces a per-(seed,
   task_id, arm) cluster surface analogous to the W101 mbpp
   block.

These cluster surfaces are the empirical cross-bench priors the
W102 HumanEval+ + MBPP+ V2 preflights consume.  They also feed
the COO-14 code-side slice-selection helper with HumanEval+ +
MBPP+ V2-grade slice proposals.

NIM-free: re-uses the on-disk prompt+response strings from the
W88 + W91 calls.jsonl sidecars.

Output:
``results/w102/arsenal_mining/<RUN>/mining_report.json`` — extends
the W101 report shape with `humaneval_plus` and `mbpp_plus_v2`
blocks alongside the W101 `humaneval` and `mbpp` blocks.

Usage::

    python scripts/run_w102_arsenal_mining.py
    python scripts/run_w102_arsenal_mining.py --skip-humaneval-plus
    python scripts/run_w102_arsenal_mining.py --skip-mbpp-plus
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_plus_executor_v1 import (  # noqa: E402
    run_humaneval_plus_executor_v1,
)
from coordpy.humaneval_plus_loader_v1 import (  # noqa: E402
    load_humaneval_plus_corpus_v1,
)
from coordpy.humaneval_real_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1 as humaneval_extract,
    select_humaneval_subset_v1,
)
from coordpy.mbpp_plus_executor_v2 import (  # noqa: E402
    run_mbpp_plus_executor_v2,
)
from coordpy.mbpp_plus_loader_v2 import (  # noqa: E402
    load_mbpp_plus_v2_corpus,
)
from coordpy.mbpp_reflexion_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1 as mbpp_extract,
    load_mbpp_corpus_v1,
    select_mbpp_subset_v1,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def _aggregate(
        bench_kind: str,
        per_seed_arm: dict[
            int, dict[str, dict[str, bool]]],
) -> dict[str, Any]:
    """Same cluster aggregator as the W101 arsenal-mining
    script (cluster partition + mechanism-load-bearing
    estimate)."""
    per_seed: dict[int, dict[str, Any]] = {}
    agg_a1_only: list[str] = []
    agg_b_only: list[str] = []
    agg_shared_wins: list[str] = []
    agg_shared_fails: list[str] = []
    a1_wins = 0
    b_wins = 0
    for seed in sorted(per_seed_arm.keys()):
        a0 = per_seed_arm[seed].get("A0", {})
        a1 = per_seed_arm[seed].get("A1", {})
        b = per_seed_arm[seed].get("B", {})
        tids = sorted(set(a0) | set(a1) | set(b))
        a0_wins_n = sum(
            1 for t in tids if a0.get(t, False))
        a1_wins_n = sum(
            1 for t in tids if a1.get(t, False))
        b_wins_n = sum(
            1 for t in tids if b.get(t, False))
        a1_wins += a1_wins_n
        b_wins += b_wins_n
        a1_only = [
            t for t in tids
            if a1.get(t, False) and not b.get(t, False)]
        b_only = [
            t for t in tids
            if b.get(t, False) and not a1.get(t, False)]
        shared_wins = [
            t for t in tids
            if a1.get(t, False) and b.get(t, False)]
        shared_fails = [
            t for t in tids
            if not a1.get(t, False) and not b.get(t, False)]
        per_seed[seed] = {
            "n_problems": int(len(tids)),
            "a0_pass_rate": float(
                a0_wins_n / max(len(tids), 1)),
            "a1_pass_rate": float(
                a1_wins_n / max(len(tids), 1)),
            "b_pass_rate": float(
                b_wins_n / max(len(tids), 1)),
            "b_minus_a1_pp": float(round(
                (b_wins_n - a1_wins_n) / max(
                    len(tids), 1) * 100.0, 4)),
            "a1_only_wins": a1_only,
            "b_only_wins": b_only,
            "shared_wins": shared_wins,
            "shared_fails": shared_fails,
            "n_a1_only_wins": int(len(a1_only)),
            "n_b_only_wins": int(len(b_only)),
            "n_shared_wins": int(len(shared_wins)),
            "n_shared_fails": int(len(shared_fails)),
        }
        agg_a1_only.extend(
            f"{seed}:{t}" for t in a1_only)
        agg_b_only.extend(
            f"{seed}:{t}" for t in b_only)
        agg_shared_wins.extend(
            f"{seed}:{t}" for t in shared_wins)
        agg_shared_fails.extend(
            f"{seed}:{t}" for t in shared_fails)
    mech_lb = float(
        len(agg_b_only) / max(b_wins, 1))
    return {
        "bench_kind": bench_kind,
        "n_seeds": int(len(per_seed_arm)),
        "n_problems_per_seed": (
            int(per_seed[sorted(per_seed)[0]]["n_problems"])
            if per_seed else 0),
        "per_seed": per_seed,
        "aggregate": {
            "n_a1_only_wins": int(len(agg_a1_only)),
            "n_b_only_wins": int(len(agg_b_only)),
            "n_shared_wins": int(len(agg_shared_wins)),
            "n_shared_fails": int(len(agg_shared_fails)),
            "a1_only_wins": agg_a1_only,
            "b_only_wins": agg_b_only,
            "shared_wins": agg_shared_wins,
            "shared_fails": agg_shared_fails,
        },
        "mechanism_load_bearing_estimate": {
            "fraction_b_wins_from_reflexion_rescue": float(
                round(mech_lb, 4)),
            "n_b_wins_total": int(b_wins),
            "n_b_only_rescues": int(len(agg_b_only)),
        },
    }


def _mine_humaneval_plus(
        sidecar_path: Path,
        seeds: list[int],
        n_problems: int,
        per_call_writer,
) -> dict[str, Any]:
    """Re-execute W88 70B HumanEval candidates against the
    HumanEval+ check() block."""
    print(f"  [humaneval_plus] sidecar = {sidecar_path}")
    if not sidecar_path.exists():
        return {"skipped": True,
                "reason": "no W88 sidecar"}
    # The W88 HumanEval bench used base-HumanEval problems; we
    # must re-derive the per-seed subset against the BASE
    # HumanEval corpus (so the (seed, problem) ↔ candidate
    # mapping is identical to W88), then look up each problem's
    # HumanEval+ entry via task_id.
    from coordpy.humaneval_real_bench_v1 import (
        load_humaneval_corpus_v1)
    base_corpus = load_humaneval_corpus_v1()
    plus_corpus = load_humaneval_plus_corpus_v1()
    plus_by_tid = {p.task_id: p for p in plus_corpus}
    sub_by_seed: dict[int, list[Any]] = {}
    for s in seeds:
        subset = select_humaneval_subset_v1(
            corpus=base_corpus, n_problems=n_problems, seed=s)
        sub_by_seed[s] = list(subset)
    records = _read_jsonl(sidecar_path)
    print(
        f"  [humaneval_plus] sidecar calls = {len(records)}")
    K = 5
    calls_per_problem = 1 + K + K
    expected = n_problems * calls_per_problem * len(seeds)
    if len(records) != expected:
        print(
            f"  [humaneval_plus] WARN sidecar has "
            f"{len(records)} calls; expected {expected}")
    per_arm: dict[
        tuple[int, str, str], list[tuple[int, bool]]] = (
            defaultdict(list))
    rec_iter = iter(records)
    for seed in seeds:
        subset = sub_by_seed[seed]
        for p_idx, base_problem in enumerate(subset):
            plus_problem = plus_by_tid.get(
                base_problem.task_id)
            # A0 block: 1 call
            try:
                rec = next(rec_iter)
            except StopIteration:
                break
            text = str(rec.get("response_text", ""))
            if plus_problem is not None:
                code = humaneval_extract(
                    response_text=text,
                    prompt=plus_problem.prompt,
                    entry_point=plus_problem.entry_point)
                exe = run_humaneval_plus_executor_v1(
                    problem=plus_problem,
                    candidate_code=code)
                per_arm[(seed, base_problem.task_id,
                         "A0")].append(
                    (0, bool(exe.passed)))
                per_call_writer({
                    "bench": "humaneval_plus",
                    "seed": int(seed),
                    "task_id": str(base_problem.task_id),
                    "arm": "A0",
                    "call_idx": 0,
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
            # A1 block: K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                if plus_problem is None:
                    continue
                code = humaneval_extract(
                    response_text=text,
                    prompt=plus_problem.prompt,
                    entry_point=plus_problem.entry_point)
                exe = run_humaneval_plus_executor_v1(
                    problem=plus_problem,
                    candidate_code=code)
                per_arm[(seed, base_problem.task_id,
                         "A1")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "humaneval_plus",
                    "seed": int(seed),
                    "task_id": str(base_problem.task_id),
                    "arm": "A1",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
            # B block: K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                if plus_problem is None:
                    continue
                code = humaneval_extract(
                    response_text=text,
                    prompt=plus_problem.prompt,
                    entry_point=plus_problem.entry_point)
                exe = run_humaneval_plus_executor_v1(
                    problem=plus_problem,
                    candidate_code=code)
                per_arm[(seed, base_problem.task_id,
                         "B")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "humaneval_plus",
                    "seed": int(seed),
                    "task_id": str(base_problem.task_id),
                    "arm": "B",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
    per_seed_arm: dict[
        int, dict[str, dict[str, bool]]] = (
            defaultdict(lambda: defaultdict(dict)))
    for (seed, tid, arm), calls in per_arm.items():
        calls_sorted = sorted(calls, key=lambda x: x[0])
        if arm == "A0":
            passed = (
                bool(calls_sorted[0][1])
                if calls_sorted else False)
        elif arm in ("A1", "B"):
            passed = any(p for _, p in calls_sorted)
        else:
            passed = False
        per_seed_arm[seed][arm][tid] = bool(passed)
    return _aggregate("humaneval_plus", per_seed_arm)


def _mine_mbpp_plus_v2(
        sidecar_path: Path,
        seeds: list[int],
        n_problems: int,
        per_call_writer,
) -> dict[str, Any]:
    """Re-execute W91 5-seed 70B MBPP candidates against the
    MBPP+ V2 extra `test` program."""
    print(f"  [mbpp_plus_v2] sidecar = {sidecar_path}")
    if not sidecar_path.exists():
        return {"skipped": True,
                "reason": "no W91 sidecar"}
    base_corpus = load_mbpp_corpus_v1()
    plus_corpus = load_mbpp_plus_v2_corpus()
    plus_by_tid = {
        # MBPP+ uses `Mbpp/<n>` task_id; base MBPP uses integer
        # task_id.  Build both lookup paths.
        str(p.task_id): p for p in plus_corpus}
    sub_by_seed: dict[int, list[Any]] = {}
    for s in seeds:
        subset = select_mbpp_subset_v1(
            corpus=base_corpus, n_problems=n_problems, seed=s)
        sub_by_seed[s] = list(subset)
    records = _read_jsonl(sidecar_path)
    print(
        f"  [mbpp_plus_v2] sidecar calls = {len(records)}")
    K = 5
    per_arm: dict[
        tuple[int, str, str], list[tuple[int, bool]]] = (
            defaultdict(list))
    rec_iter = iter(records)
    for seed in seeds:
        subset = sub_by_seed[seed]
        for p_idx, base_problem in enumerate(subset):
            plus_problem = plus_by_tid.get(
                f"Mbpp/{int(base_problem.task_id)}")
            # A0
            try:
                rec = next(rec_iter)
            except StopIteration:
                break
            text = str(rec.get("response_text", ""))
            tid_str = str(base_problem.task_id)
            if plus_problem is not None:
                code = mbpp_extract(
                    response_text=text,
                    entry_point=plus_problem.entry_point)
                exe = run_mbpp_plus_executor_v2(
                    problem=plus_problem,
                    candidate_code=code,
                    mode="base_and_plus")
                per_arm[(seed, tid_str, "A0")].append(
                    (0, bool(exe.passed)))
                per_call_writer({
                    "bench": "mbpp_plus_v2",
                    "seed": int(seed),
                    "task_id": str(tid_str),
                    "arm": "A0",
                    "call_idx": 0,
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
            # A1 K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                if plus_problem is None:
                    continue
                code = mbpp_extract(
                    response_text=text,
                    entry_point=plus_problem.entry_point)
                exe = run_mbpp_plus_executor_v2(
                    problem=plus_problem,
                    candidate_code=code,
                    mode="base_and_plus")
                per_arm[(seed, tid_str, "A1")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "mbpp_plus_v2",
                    "seed": int(seed),
                    "task_id": str(tid_str),
                    "arm": "A1",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
            # B K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                if plus_problem is None:
                    continue
                code = mbpp_extract(
                    response_text=text,
                    entry_point=plus_problem.entry_point)
                exe = run_mbpp_plus_executor_v2(
                    problem=plus_problem,
                    candidate_code=code,
                    mode="base_and_plus")
                per_arm[(seed, tid_str, "B")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "mbpp_plus_v2",
                    "seed": int(seed),
                    "task_id": str(tid_str),
                    "arm": "B",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
    per_seed_arm: dict[
        int, dict[str, dict[str, bool]]] = (
            defaultdict(lambda: defaultdict(dict)))
    for (seed, tid, arm), calls in per_arm.items():
        calls_sorted = sorted(calls, key=lambda x: x[0])
        if arm == "A0":
            passed = (
                bool(calls_sorted[0][1])
                if calls_sorted else False)
        elif arm in ("A1", "B"):
            passed = any(p for _, p in calls_sorted)
        else:
            passed = False
        per_seed_arm[seed][arm][tid] = bool(passed)
    return _aggregate("mbpp_plus_v2", per_seed_arm)


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W102 — arsenal-mining extension (HumanEval+ + MBPP+ V2)"))
    ap.add_argument(
        "--humaneval-sidecar",
        default=str(
            ROOT / "results" / "w88" / "humaneval_reflexion"
            / "w88_nim_meta_llama-3.3-70b-instruct"
              "_20260522T222541Z"
            / "humaneval_reflexion_calls.jsonl"))
    ap.add_argument(
        "--mbpp-sidecar",
        default=str(
            ROOT / "results" / "w91" / "mbpp_reflexion_5seeds"
            / "w90_mbpp_nim_meta_llama-3.3-70b-instruct"
              "_20260523T141809Z"
            / "mbpp_reflexion_calls.jsonl"))
    ap.add_argument(
        "--skip-humaneval-plus", action="store_true")
    ap.add_argument(
        "--skip-mbpp-plus", action="store_true")
    ap.add_argument(
        "--include-w101-mining",
        action="store_true",
        help=(
            "Also include the W101 mining (humaneval + mbpp "
            "blocks) in the W102 report shape for a single "
            "combined artifact"))
    ap.add_argument(
        "--w101-mining-report",
        default=str(
            ROOT / "results" / "w101" / "arsenal_mining"
            / "latest_run.txt"),
        help="W101 latest_run.txt or mining_report.json")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w102" / "arsenal_mining"),
        help="Output root")
    args = ap.parse_args()

    run_id = _dt.datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir) / f"w102_arsenal_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_call_path = out_dir / "per_call_outcomes.jsonl"
    per_call_f = open(per_call_path, "w")

    def per_call_writer(rec: dict[str, Any]) -> None:
        per_call_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")

    report: dict[str, Any] = {
        "schema": "coordpy.w102_arsenal_mining_v1",
        "started_at_utc": run_id,
        "humaneval": None,        # W101 blocks (carried)
        "mbpp": None,             # W101 blocks (carried)
        "humaneval_plus": None,   # NEW W102
        "mbpp_plus_v2": None,     # NEW W102
    }
    if args.include_w101_mining:
        try:
            w101_ptr = Path(args.w101_mining_report)
            if (w101_ptr.is_file()
                    and w101_ptr.name == "latest_run.txt"):
                pointer = w101_ptr.read_text().strip()
                w101_mining = (
                    w101_ptr.parent / pointer
                    / "mining_report.json")
            else:
                w101_mining = w101_ptr
            w101 = json.loads(w101_mining.read_text())
            report["humaneval"] = w101.get("humaneval")
            report["mbpp"] = w101.get("mbpp")
            print(
                f"  [W101 carry] {w101_mining}: "
                "humaneval/mbpp blocks carried")
        except Exception as e:  # noqa: BLE001
            print(
                f"  [W101 carry] WARN failed to load: {e}")
    if not args.skip_humaneval_plus:
        print("[humaneval_plus] mining ...")
        report["humaneval_plus"] = _mine_humaneval_plus(
            Path(args.humaneval_sidecar),
            seeds=[88_028_001, 88_028_002, 88_028_003],
            n_problems=30,
            per_call_writer=per_call_writer)
    if not args.skip_mbpp_plus:
        print("[mbpp_plus_v2] mining ...")
        report["mbpp_plus_v2"] = _mine_mbpp_plus_v2(
            Path(args.mbpp_sidecar),
            seeds=[90_001, 90_002, 90_003, 90_004, 90_005],
            n_problems=30,
            per_call_writer=per_call_writer)
    per_call_f.close()
    rep_path = out_dir / "mining_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    pp_path = out_dir / "per_problem_outcomes.json"
    pp_payload: dict[str, Any] = {
        k: ({"per_seed": v.get("per_seed"),
             "aggregate": v.get("aggregate")}
            if isinstance(v, dict) else v)
        for k, v in report.items()
        if k in (
            "humaneval", "mbpp",
            "humaneval_plus", "mbpp_plus_v2")}
    with open(pp_path, "w") as f:
        json.dump(pp_payload, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
