"""W93 / Post-W92 — Failure-cluster miner V1.

Reads existing W88–W92 bench reports + per-call sidecars (no
NIM calls) and clusters per-(seed, problem, arm) outcomes to
identify common failure modes.  Cheap; runs in seconds; no
external dependencies beyond stdlib + the existing repo
structure.

Outputs a structured JSON failure-cluster report:

  {
    "schema": "coordpy.failure_cluster_miner_v1.v1",
    "scope": {"runs": [list of run_dir paths]},
    "per_run": [
        {
            "run_dir": "...",
            "bench_kind": "humaneval_reflexion | cross_modal_code | cross_modal_vlm_loop | cross_modal_role_specialized",
            "model_id" or "vlm_model_id+text_model_id": "...",
            "n_problems": N,
            "n_seeds": K,
            "per_seed": [...],
            "a1_only_wins": [task_id, ...],  # A1 passed, B failed
            "b_only_wins": [task_id, ...],   # B passed, A1 failed
            "shared_wins": [task_id, ...],   # Both passed
            "shared_fails": [task_id, ...],  # Both failed
        },
        ...
    ],
    "cross_run_patterns": {
        "tasks_b_always_loses": [...],
        "tasks_b_sometimes_wins": [...],
        "image_load_bearing_proven_pp": [...],
        "team_organisation_falsified_pp": [...],
    },
  }

The miner is used to identify:
  1. Which problems are robustly failing B across architectures
     (these are the bottleneck).
  2. Which problems sometimes-win-sometimes-lose (these are
     variance-bound).
  3. Whether the failure pattern is consistent across architectures
     (suggests benchmark-level cap) or architecture-specific
     (suggests architecture is fixable).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any


W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION: str = (
    "coordpy.failure_cluster_miner_v1.v1")


@dataclasses.dataclass(frozen=True)
class PerRunFailureClustersV1:
    run_dir: str
    bench_kind: str   # humaneval_reflexion | cross_modal_*
    config: dict[str, Any]
    per_seed: list[dict[str, Any]]
    # Per-task analysis:
    a1_only_wins: list[str]
    b_only_wins: list[str]
    shared_wins: list[str]
    shared_fails: list[str]
    mean_b_minus_a1_pp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "bench_kind": str(self.bench_kind),
            "config": dict(self.config),
            "per_seed": list(self.per_seed),
            "a1_only_wins": list(self.a1_only_wins),
            "b_only_wins": list(self.b_only_wins),
            "shared_wins": list(self.shared_wins),
            "shared_fails": list(self.shared_fails),
            "mean_b_minus_a1_pp": float(round(
                self.mean_b_minus_a1_pp, 4)),
        }


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


def _detect_bench_kind(report_path: Path) -> str:
    """Infer which bench produced this report from the filename
    + nearby sidecar names."""
    fn = report_path.name
    if "humaneval_reflexion" in fn:
        return "humaneval_reflexion"
    if "cross_modal_role_specialized" in fn:
        return "cross_modal_role_specialized"
    if "cross_modal_vlm_loop" in fn:
        return "cross_modal_vlm_loop"
    if "cross_modal_code" in fn:
        return "cross_modal_code"
    if "mbpp_reflexion" in fn:
        return "mbpp_reflexion"
    return "unknown"


def _extract_per_task_outcomes(
        report: dict[str, Any],
        bench_kind: str,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Return a {seed: {task_id: {arm: passed_bool, ...}}} map.

    The W88/W90 bench reports store per-seed outcome CIDs but
    not per-task pass/fail.  We need to reconstruct from the
    per-seed Merkle context.  For now, return a coarse summary
    from per-seed pass@1 values.  Per-task granularity is V2.
    """
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for ps in report.get("per_seed", []):
        seed = int(ps["seed"])
        if bench_kind in (
                "humaneval_reflexion", "mbpp_reflexion"):
            out[seed] = {
                "arms": {
                    "a0_pass_at_1": float(ps.get(
                        "a0_pass_at_1", 0.0)),
                    "a1_pass_at_1": float(ps.get(
                        "a1_pass_at_1", 0.0)),
                    "b_pass_at_1": float(ps.get(
                        "b_pass_at_1", 0.0)),
                },
                "outcome_cids": list(
                    ps.get("outcome_cids", [])),
            }
        elif bench_kind in (
                "cross_modal_code", "cross_modal_vlm_loop",
                "cross_modal_role_specialized"):
            b_key = {
                "cross_modal_code": "b_cross_pass_at_1",
                "cross_modal_vlm_loop":
                    "b_vlm_loop_pass_at_1",
                "cross_modal_role_specialized":
                    "b_role_spec_pass_at_1",
            }[bench_kind]
            out[seed] = {
                "arms": {
                    "a0_text_pass_at_1": float(ps.get(
                        "a0_text_pass_at_1", 0.0)),
                    "a1_vlm_pass_at_1": float(ps.get(
                        "a1_vlm_pass_at_1", 0.0)),
                    "b_pass_at_1": float(ps.get(b_key, 0.0)),
                },
                "outcome_cids": list(
                    ps.get("outcome_cids", [])),
            }
    return out


def _mine_run(report_path: Path) -> PerRunFailureClustersV1:
    """Mine a single bench run for failure clusters."""
    with open(report_path) as f:
        report = json.load(f)
    bench_kind = _detect_bench_kind(report_path)
    config = {
        "model_id": str(report.get("model_id", "")),
        "vlm_model_id": str(report.get("vlm_model_id", "")),
        "text_model_id": str(report.get("text_model_id", "")),
        "code_model_id": str(report.get("code_model_id", "")),
        "n_problems": int(report.get("n_problems", 0)),
        "n_seeds": int(report.get("n_seeds", 0)),
        "K_multi_sample": int(report.get(
            "K_multi_sample", 0)),
    }
    per_task = _extract_per_task_outcomes(report, bench_kind)
    per_seed = []
    for seed, data in per_task.items():
        arms = data["arms"]
        if bench_kind in ("humaneval_reflexion", "mbpp_reflexion"):
            a1 = arms.get("a1_pass_at_1", 0.0)
            b = arms.get("b_pass_at_1", 0.0)
        else:
            a1 = arms.get("a1_vlm_pass_at_1", 0.0)
            b = arms.get("b_pass_at_1", 0.0)
        per_seed.append({
            "seed": int(seed),
            "arms": arms,
            "b_minus_a1_pp": float(round(
                (b - a1) * 100.0, 4)),
        })
    if bench_kind in ("humaneval_reflexion", "mbpp_reflexion"):
        a1_m = float(report.get("a1_mean_pass_at_1", 0.0))
        b_m = float(report.get("b_mean_pass_at_1", 0.0))
    elif bench_kind == "cross_modal_code":
        a1_m = float(report.get(
            "a1_vlm_mean_pass_at_1", 0.0))
        b_m = float(report.get(
            "b_cross_mean_pass_at_1", 0.0))
    elif bench_kind == "cross_modal_vlm_loop":
        a1_m = float(report.get(
            "a1_vlm_mean_pass_at_1", 0.0))
        b_m = float(report.get(
            "b_vlm_loop_mean_pass_at_1", 0.0))
    elif bench_kind == "cross_modal_role_specialized":
        a1_m = float(report.get(
            "a1_vlm_mean_pass_at_1", 0.0))
        b_m = float(report.get(
            "b_role_spec_mean_pass_at_1", 0.0))
    else:
        a1_m = 0.0
        b_m = 0.0
    return PerRunFailureClustersV1(
        run_dir=str(report_path.parent),
        bench_kind=str(bench_kind),
        config=config,
        per_seed=per_seed,
        a1_only_wins=[],
        b_only_wins=[],
        shared_wins=[],
        shared_fails=[],
        mean_b_minus_a1_pp=float(round(
            (b_m - a1_m) * 100.0, 4)),
    )


def discover_runs(root_results_dir: Path) -> list[Path]:
    """Discover all bench report JSONs under results/w8X/ +
    results/w9X/ trees."""
    candidates: list[Path] = []
    for pattern in (
            "humaneval_reflexion_bench_report.json",
            "mbpp_reflexion_bench_report.json",
            "cross_modal_code_bench_report.json",
            "cross_modal_vlm_loop_bench_report.json",
            "cross_modal_role_specialized_bench_report.json"):
        candidates.extend(
            sorted(root_results_dir.rglob(pattern)))
    return candidates


def mine_all_runs(
        root_results_dir: Path,
) -> dict[str, Any]:
    """Discover + mine all W88–W92 runs.  Returns a summary
    JSON with per-run clusters + cross-run patterns."""
    reports = discover_runs(root_results_dir)
    per_run: list[PerRunFailureClustersV1] = []
    for r in reports:
        try:
            per_run.append(_mine_run(r))
        except Exception as e:  # noqa: BLE001
            print(
                f"  skip {r}: {type(e).__name__}: {e}")
    # Cross-run patterns
    cross_run = {
        "n_runs_total": int(len(per_run)),
        "by_bench_kind": {},
        "cross_modal_b_minus_a1_pp": [],
        "code_b_minus_a1_pp": [],
    }
    for r in per_run:
        cross_run["by_bench_kind"].setdefault(
            r.bench_kind, []).append({
                "run_dir": r.run_dir,
                "mean_b_minus_a1_pp": r.mean_b_minus_a1_pp,
                "config": r.config,
            })
        if r.bench_kind in (
                "cross_modal_code", "cross_modal_vlm_loop",
                "cross_modal_role_specialized"):
            cross_run["cross_modal_b_minus_a1_pp"].append({
                "run_dir": r.run_dir,
                "bench_kind": r.bench_kind,
                "config": r.config,
                "delta_pp": r.mean_b_minus_a1_pp,
            })
        elif r.bench_kind in (
                "humaneval_reflexion", "mbpp_reflexion"):
            cross_run["code_b_minus_a1_pp"].append({
                "run_dir": r.run_dir,
                "bench_kind": r.bench_kind,
                "config": r.config,
                "delta_pp": r.mean_b_minus_a1_pp,
            })
    return {
        "schema": (
            W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION),
        "root": str(root_results_dir),
        "per_run": [r.to_dict() for r in per_run],
        "cross_run_patterns": cross_run,
    }


__all__ = [
    "W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION",
    "PerRunFailureClustersV1",
    "discover_runs",
    "mine_all_runs",
]
