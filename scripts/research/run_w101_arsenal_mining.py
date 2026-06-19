#!/usr/bin/env python3
"""W101 — arsenal mining via offline sidecar re-execution.

Reads existing W88 70B HumanEval reflexion sidecar +
W91 5-seed MBPP reflexion sidecar and re-executes every
persisted candidate response against the same canonical executor
the original bench used.  Output is a per-(seed, task_id, arm,
call_idx) -> passed table plus failure-cluster aggregates the
W101 runbook + preflight can mine.

NIM-free: re-uses the on-disk prompt+response strings from the
calls.jsonl sidecars.  No new model calls.  Costs only local
CPU time (one subprocess executor per call, ~1-3 s typical).

Outputs:

* ``results/w101/arsenal_mining/<RUN_ID>/per_call_outcomes.jsonl``
  — one line per re-executed call with passed/wall_ms/stderr_tail
* ``results/w101/arsenal_mining/<RUN_ID>/per_problem_outcomes.json``
  — aggregated per-(seed, task_id, arm) verdict + per-cluster
  membership (a1_only_wins / b_only_wins / shared_wins /
  shared_fails)
* ``results/w101/arsenal_mining/<RUN_ID>/mining_report.json``
  — top-level summary the W101 preflight + runbook reads

Usage::

    python scripts/run_w101_arsenal_mining.py
    python scripts/run_w101_arsenal_mining.py --skip-humaneval
    python scripts/run_w101_arsenal_mining.py --skip-mbpp
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_real_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1 as humaneval_extract,
    load_humaneval_corpus_v1,
    run_humaneval_executor_v1,
)
from coordpy.mbpp_reflexion_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1 as mbpp_extract,
    load_mbpp_corpus_v1,
    run_mbpp_executor_v1,
    select_mbpp_subset_v1,
)
from coordpy.humaneval_real_bench_v1 import (  # noqa: E402
    select_humaneval_subset_v1,
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


def _humaneval_subset_for_run(
        seeds: list[int], n_problems: int) -> dict[
            int, list[Any]]:
    """Reconstruct the deterministic per-seed subset the W88/W89
    bench used."""
    corpus = load_humaneval_corpus_v1()
    out: dict[int, list[Any]] = {}
    for s in seeds:
        subset = select_humaneval_subset_v1(
            corpus=corpus, n_problems=n_problems, seed=s)
        out[s] = list(subset)
    return out


def _mbpp_subset_for_run(
        seeds: list[int], n_problems: int) -> dict[
            int, list[Any]]:
    corpus = load_mbpp_corpus_v1()
    out: dict[int, list[Any]] = {}
    for s in seeds:
        subset = select_mbpp_subset_v1(
            corpus=corpus, n_problems=n_problems, seed=s)
        out[s] = list(subset)
    return out


def _humaneval_role_to_arm(role: str, call_idx: int) -> str:
    """Map the W88/W89 reflexion call role to its arm.

    A0 = role 'solver' at call_idx 0
    A1 = role 'sample' at call_idx 0..K-1
    B  = role 'initial' (call 0) or 'reflexion' (call 1..K-1)
    """
    if role == "solver":
        return "A0"
    if role == "sample":
        return "A1"
    if role in ("initial", "reflexion"):
        return "B"
    return "?"


def _mbpp_role_to_arm(role: str, call_idx: int) -> str:
    return _humaneval_role_to_arm(role, call_idx)


def _mine_humaneval_run(
        run_dir: Path,
        out_dir: Path,
        seeds: list[int],
        n_problems: int,
        per_call_writer,
) -> dict[str, Any]:
    """Re-execute every persisted W88/W89 HumanEval candidate
    response against its canonical problem.  Returns aggregated
    per-(seed, task_id, arm) outcomes."""
    print(f"  humaneval run_dir = {run_dir}")
    sidecar = run_dir / "humaneval_reflexion_calls.jsonl"
    if not sidecar.exists():
        sidecar = run_dir / "humaneval_real_calls.jsonl"
    if not sidecar.exists():
        print(f"  SKIP — no calls.jsonl at {run_dir}")
        return {
            "skipped": True,
            "reason": "no calls.jsonl"}
    sub_by_seed = _humaneval_subset_for_run(seeds, n_problems)
    task_idx_to_id: dict[int, dict[int, str]] = {}
    task_id_to_problem: dict[int, dict[str, Any]] = {}
    for s in seeds:
        task_idx_to_id[s] = {}
        for i, p in enumerate(sub_by_seed[s]):
            task_idx_to_id[s][i] = p.task_id
            task_id_to_problem[id(p)] = {"problem": p}
    # Build map task_id -> problem for executor lookups.
    tid_to_p: dict[str, Any] = {}
    for s in seeds:
        for p in sub_by_seed[s]:
            tid_to_p[p.task_id] = p
    # Now stream through the sidecar.  Each line has a
    # seed-tagged record; we re-execute each candidate and emit
    # a per_call_outcome record.
    records = _read_jsonl(sidecar)
    print(f"  sidecar calls = {len(records)}")
    # Per-(seed, task_id, arm) -> list[(call_idx, passed_bool)]
    per_arm: dict[
        tuple[int, str, str], list[tuple[int, bool]]] = (
            defaultdict(list))
    # We can't recover (seed, task_id) directly from the W88
    # sidecar (it only records prompt/response + model_id).  But
    # we CAN walk the sidecar in order: the bench writes calls
    # in deterministic per-(seed, problem, arm) order with K
    # samples per A1 and K samples per B and 1 sample per A0.
    # Per-problem call block layout:
    #   A0:   1 call  (T=0.0, role='solver')
    #   A1:   K calls (T=0.7, role='sample')
    #   B:    K calls (T=0.7, role='initial'/'reflexion')
    # Total per problem = 1 + K + K = 11 (K=5).
    K = 5
    calls_per_problem = 1 + K + K
    n_problems_per_seed = n_problems
    expected = (
        n_problems_per_seed * calls_per_problem * len(seeds))
    if len(records) != expected:
        print(
            f"  WARN sidecar has {len(records)} calls; "
            f"expected {expected} for "
            f"{len(seeds)} seeds × {n_problems_per_seed} "
            f"problems × {calls_per_problem} calls/problem")
    rec_iter = iter(records)
    for seed in seeds:
        subset = sub_by_seed[seed]
        for p_idx, problem in enumerate(subset):
            # A0 block: 1 call
            try:
                a0_rec = next(rec_iter)
            except StopIteration:
                break
            text = str(a0_rec.get("response_text", ""))
            code = humaneval_extract(
                response_text=text,
                prompt=problem.prompt,
                entry_point=problem.entry_point)
            exe = run_humaneval_executor_v1(
                problem=problem, candidate_code=code)
            per_arm[(seed, problem.task_id, "A0")].append(
                (0, bool(exe.passed)))
            per_call_writer({
                "bench": "humaneval",
                "seed": int(seed),
                "task_id": str(problem.task_id),
                "arm": "A0",
                "call_idx": 0,
                "passed": bool(exe.passed),
                "exec_wall_ms": int(exe.wall_ms),
                "returncode": int(exe.returncode),
                "stderr_tail": str(exe.stderr_tail)[:240],
            })
            # A1 block: K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                code = humaneval_extract(
                    response_text=text,
                    prompt=problem.prompt,
                    entry_point=problem.entry_point)
                exe = run_humaneval_executor_v1(
                    problem=problem, candidate_code=code)
                per_arm[(seed, problem.task_id, "A1")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "humaneval",
                    "seed": int(seed),
                    "task_id": str(problem.task_id),
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
                code = humaneval_extract(
                    response_text=text,
                    prompt=problem.prompt,
                    entry_point=problem.entry_point)
                exe = run_humaneval_executor_v1(
                    problem=problem, candidate_code=code)
                per_arm[(seed, problem.task_id, "B")].append(
                    (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "humaneval",
                    "seed": int(seed),
                    "task_id": str(problem.task_id),
                    "arm": "B",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(
                        exe.stderr_tail)[:240],
                })
    # Aggregate per-(seed, task_id) by arm:
    #  A0: pass = call 0 passed.
    #  A1: pass = first call with passed=True (literature
    #       "first-pass-among-K"); else FAIL.
    #  B:  pass = any call with passed=True (literature
    #       "ship first PASS by attempt index"); else FAIL.
    per_seed_arm: dict[int, dict[str, dict[str, bool]]] = (
        defaultdict(lambda: defaultdict(dict)))
    for (seed, tid, arm), calls in per_arm.items():
        calls_sorted = sorted(calls, key=lambda x: x[0])
        if arm == "A0":
            passed = bool(calls_sorted[0][1]) if calls_sorted else False
        elif arm in ("A1", "B"):
            passed = any(p for _, p in calls_sorted)
        else:
            passed = False
        per_seed_arm[seed][arm][tid] = bool(passed)
    return _aggregate_failure_clusters(
        bench_kind="humaneval",
        per_seed_arm=per_seed_arm)


def _mine_mbpp_run(
        run_dir: Path,
        out_dir: Path,
        seeds: list[int],
        n_problems: int,
        per_call_writer,
) -> dict[str, Any]:
    print(f"  mbpp run_dir = {run_dir}")
    sidecar = run_dir / "mbpp_reflexion_calls.jsonl"
    if not sidecar.exists():
        print(f"  SKIP — no calls.jsonl at {run_dir}")
        return {"skipped": True, "reason": "no calls.jsonl"}
    sub_by_seed = _mbpp_subset_for_run(seeds, n_problems)
    records = _read_jsonl(sidecar)
    print(f"  sidecar calls = {len(records)}")
    K = 5
    calls_per_problem = 1 + K + K
    expected = n_problems * calls_per_problem * len(seeds)
    if len(records) != expected:
        print(
            f"  WARN sidecar has {len(records)} calls; "
            f"expected {expected}")
    per_arm: dict[
        tuple[int, int, str], list[tuple[int, bool]]] = (
            defaultdict(list))
    rec_iter = iter(records)
    for seed in seeds:
        subset = sub_by_seed[seed]
        for p_idx, problem in enumerate(subset):
            # A0
            try:
                rec = next(rec_iter)
            except StopIteration:
                break
            text = str(rec.get("response_text", ""))
            code = mbpp_extract(
                response_text=text,
                entry_point=problem.entry_point)
            exe = run_mbpp_executor_v1(
                problem=problem, candidate_code=code)
            per_arm[(seed, problem.task_id, "A0")].append(
                (0, bool(exe.passed)))
            per_call_writer({
                "bench": "mbpp",
                "seed": int(seed),
                "task_id": int(problem.task_id),
                "arm": "A0",
                "call_idx": 0,
                "passed": bool(exe.passed),
                "n_pass": int(exe.n_assertions_passed),
                "n_total": int(exe.n_assertions_total),
                "exec_wall_ms": int(exe.wall_ms),
                "returncode": int(exe.returncode),
                "stderr_tail": str(exe.stderr_tail)[:240],
            })
            # A1 K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                code = mbpp_extract(
                    response_text=text,
                    entry_point=problem.entry_point)
                exe = run_mbpp_executor_v1(
                    problem=problem, candidate_code=code)
                per_arm[
                    (seed, problem.task_id, "A1")].append(
                        (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "mbpp",
                    "seed": int(seed),
                    "task_id": int(problem.task_id),
                    "arm": "A1",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "n_pass": int(exe.n_assertions_passed),
                    "n_total": int(exe.n_assertions_total),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(exe.stderr_tail)[:240],
                })
            # B K calls
            for k in range(K):
                rec = next(rec_iter)
                text = str(rec.get("response_text", ""))
                code = mbpp_extract(
                    response_text=text,
                    entry_point=problem.entry_point)
                exe = run_mbpp_executor_v1(
                    problem=problem, candidate_code=code)
                per_arm[
                    (seed, problem.task_id, "B")].append(
                        (k, bool(exe.passed)))
                per_call_writer({
                    "bench": "mbpp",
                    "seed": int(seed),
                    "task_id": int(problem.task_id),
                    "arm": "B",
                    "call_idx": int(k),
                    "passed": bool(exe.passed),
                    "n_pass": int(exe.n_assertions_passed),
                    "n_total": int(exe.n_assertions_total),
                    "exec_wall_ms": int(exe.wall_ms),
                    "returncode": int(exe.returncode),
                    "stderr_tail": str(exe.stderr_tail)[:240],
                })
    per_seed_arm: dict[int, dict[str, dict[str, bool]]] = (
        defaultdict(lambda: defaultdict(dict)))
    for (seed, tid, arm), calls in per_arm.items():
        calls_sorted = sorted(calls, key=lambda x: x[0])
        if arm == "A0":
            passed = bool(calls_sorted[0][1]) if calls_sorted else False
        elif arm in ("A1", "B"):
            passed = any(p for _, p in calls_sorted)
        else:
            passed = False
        per_seed_arm[seed][arm][str(tid)] = bool(passed)
    return _aggregate_failure_clusters(
        bench_kind="mbpp",
        per_seed_arm=per_seed_arm)


def _aggregate_failure_clusters(
        bench_kind: str,
        per_seed_arm: dict[
            int, dict[str, dict[str, bool]]],
) -> dict[str, Any]:
    """Compute the W93-shape per-problem cluster surface.

    Output: {
       "bench_kind": ...,
       "per_seed": {seed: {a0_pass_rate, a1_pass_rate,
                            b_pass_rate, b_minus_a1_pp,
                            a1_only_wins:[...], b_only_wins:[...],
                            shared_wins:[...], shared_fails:[...],
                            b_rescues_a1_fails:[...],
                            a1_rescues_b_fails:[...]}},
       "aggregate": {sum across seeds},
       "mechanism_load_bearing_estimate": <fraction>,
    }
    """
    per_seed: dict[int, dict[str, Any]] = {}
    agg_a1_only: list[str] = []
    agg_b_only: list[str] = []
    agg_shared_wins: list[str] = []
    agg_shared_fails: list[str] = []
    a1_calls = 0
    b_calls = 0
    a1_wins = 0
    b_wins = 0
    n_problems_total = 0
    for seed in sorted(per_seed_arm.keys()):
        a0 = per_seed_arm[seed].get("A0", {})
        a1 = per_seed_arm[seed].get("A1", {})
        b = per_seed_arm[seed].get("B", {})
        tids = sorted(set(a0) | set(a1) | set(b))
        n_problems_total += len(tids)
        a0_wins_n = sum(
            1 for t in tids if a0.get(t, False))
        a1_wins_n = sum(
            1 for t in tids if a1.get(t, False))
        b_wins_n = sum(
            1 for t in tids if b.get(t, False))
        a1_calls += len(tids)
        b_calls += len(tids)
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
    # Mechanism-load-bearing estimate (on the code-line):
    # fraction of B-wins that came via reflexion-on-A1-failure,
    # i.e., B-only-wins / total-B-wins.
    total_b_wins = b_wins
    b_only_wins = len(agg_b_only)
    mech_lb = float(
        b_only_wins / max(total_b_wins, 1))
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
            "n_b_wins_total": int(total_b_wins),
            "n_b_only_rescues": int(b_only_wins),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W101 — arsenal mining via offline sidecar "
        "re-execution"))
    ap.add_argument(
        "--humaneval-run-dir",
        default=str(
            ROOT / "results" / "w88" / "humaneval_reflexion"
            / "w88_nim_meta_llama-3.3-70b-instruct"
              "_20260522T222541Z"),
        help="W89 70B HumanEval reflexion run dir")
    ap.add_argument(
        "--mbpp-run-dir",
        default=str(
            ROOT / "results" / "w91" / "mbpp_reflexion_5seeds"
            / "w90_mbpp_nim_meta_llama-3.3-70b-instruct"
              "_20260523T141809Z"),
        help="W91 5-seed 70B MBPP reflexion run dir")
    ap.add_argument(
        "--skip-humaneval", action="store_true")
    ap.add_argument(
        "--skip-mbpp", action="store_true")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w101" / "arsenal_mining"),
        help="Output root")
    args = ap.parse_args()

    run_id = _dt.datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"w101_arsenal_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_call_path = out_dir / "per_call_outcomes.jsonl"
    per_call_f = open(per_call_path, "w")

    def per_call_writer(rec: dict[str, Any]) -> None:
        per_call_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")

    report: dict[str, Any] = {
        "schema": "coordpy.w101_arsenal_mining_v1",
        "started_at_utc": run_id,
        "humaneval": None,
        "mbpp": None,
    }
    if not args.skip_humaneval:
        print("[humaneval] mining ...")
        report["humaneval"] = _mine_humaneval_run(
            run_dir=Path(args.humaneval_run_dir),
            out_dir=out_dir,
            seeds=[88_028_001, 88_028_002, 88_028_003],
            n_problems=30,
            per_call_writer=per_call_writer)
    if not args.skip_mbpp:
        print("[mbpp] mining ...")
        report["mbpp"] = _mine_mbpp_run(
            run_dir=Path(args.mbpp_run_dir),
            out_dir=out_dir,
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
        if k in ("humaneval", "mbpp")}
    with open(pp_path, "w") as f:
        json.dump(pp_payload, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
