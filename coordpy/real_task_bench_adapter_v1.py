"""W84 / P0 #28 — Real-task multi-agent bench adapter (stub).

Issue #28 asks for one ``RealTaskBenchAdapterV1`` against a
published real-task benchmark (SWE-bench / GAIA / MLE-bench /
OSWorld) with a strict head-to-head improvement on a published
metric.

This module ships the **adapter shape** (the contract); it does
NOT close the issue. Closing requires:

1. A real model that can produce code generation / tool-use
   outputs at SWE-bench-Verified scale (blocked on #25).
2. The stock harness for the chosen bench (we ship a stub that
   reads SWE-bench-Lite JSONL via the existing
   ``coordpy._internal.tasks.swe_bench_bridge``).
3. Multiple seeds + audit chain (we ship the audit-chain
   capsule).

Anti-cheat:

* The adapter refuses to run the harness loop unless a real
  model client is provided.
* The adapter emits a ``RealTaskBenchPlanV1`` capsule per task
  (no execution) so a future GPU host can re-run the bench
  faithfully.
* The Merkle root over committed task outcomes is re-verifiable
  from disk.

Honest scope
------------

* ``W84-L-REAL-TASK-BENCH-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
* ``W84-L-REAL-TASK-BENCH-V1-PLAN-ONLY-CAP`` — V1 emits plan
  capsules; full task execution + head-to-head requires a real
  model.
* ``W84-L-REAL-TASK-BENCH-V1-SWE-BENCH-LITE-ONLY-CAP`` — V1
  ingests SWE-bench-Lite-style JSONL (the format ``coordpy-
  import`` already understands). Other bench formats are V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.real_task_bench_adapter_v1.v1")


class RealTaskBenchBlockedOnModelError(RuntimeError):
    """Raised when the real-task bench cannot run because no
    model client is available."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class RealTaskBenchPlanV1:
    """Per-task plan capsule emitted by the adapter (no exec).

    A future GPU host with a real model client can consume these
    capsules and run the bench against the stock harness.
    """

    schema: str
    bench_name: str
    task_id: str
    task_prompt_cid: str
    task_metadata_cid: str
    composed_pipeline_config_cid: str
    expected_metric_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "bench_name": str(self.bench_name),
            "task_id": str(self.task_id),
            "task_prompt_cid": str(self.task_prompt_cid),
            "task_metadata_cid": str(self.task_metadata_cid),
            "composed_pipeline_config_cid": str(
                self.composed_pipeline_config_cid),
            "expected_metric_name": str(
                self.expected_metric_name),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_real_task_bench_plan_v1",
            "plan": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class RealTaskBenchPlanChainV1:
    """A chain of per-task plan capsules, Merkle-rooted."""

    schema: str
    bench_name: str
    n_tasks: int
    plan_cids: tuple[str, ...]
    chain_root_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "bench_name": str(self.bench_name),
            "n_tasks": int(self.n_tasks),
            "chain_root_cid": str(self.chain_root_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_real_task_bench_plan_chain_v1",
            "chain": self.to_dict(),
            "plan_cids": list(self.plan_cids),
        })


@dataclasses.dataclass(frozen=True)
class RealTaskBenchAdapterV1:
    """The adapter contract.

    V1 ingests SWE-bench-Lite-style JSONL (the same format
    ``coordpy-import`` already understands). Each row becomes a
    ``RealTaskBenchPlanV1`` capsule; the chain ships as a
    ``RealTaskBenchPlanChainV1`` capsule.
    """

    schema: str
    bench_name: str = "swe_bench_lite"

    def plan_from_swe_lite_jsonl(
            self, *, jsonl_path: str,
            composed_pipeline_config_cid: str,
            expected_metric_name: str = (
                "resolved_rate_at_1"),
    ) -> RealTaskBenchPlanChainV1:
        """Ingest a SWE-bench-Lite JSONL and emit a plan chain."""
        plans: list[RealTaskBenchPlanV1] = []
        with open(str(jsonl_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = str(row.get("instance_id",
                                      row.get("task_id", "")))
                if not task_id:
                    continue
                problem = str(row.get("problem_statement",
                                      row.get("question",
                                              "")))
                meta = {
                    k: v for k, v in row.items()
                    if k not in {"problem_statement", "question"}
                }
                plan = RealTaskBenchPlanV1(
                    schema=(
                        W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION),
                    bench_name=str(self.bench_name),
                    task_id=str(task_id),
                    task_prompt_cid=hashlib.sha256(
                        problem.encode("utf-8")).hexdigest(),
                    task_metadata_cid=_sha256_hex({
                        "kind": "swe_task_metadata",
                        "row": meta}),
                    composed_pipeline_config_cid=str(
                        composed_pipeline_config_cid),
                    expected_metric_name=str(
                        expected_metric_name),
                )
                plans.append(plan)
        chain_root = _sha256_hex({
            "kind":
                "w84_real_task_bench_plan_chain_merkle_root",
            "bench_name": str(self.bench_name),
            "plan_cids": [str(p.cid()) for p in plans],
        })
        return RealTaskBenchPlanChainV1(
            schema=W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION,
            bench_name=str(self.bench_name),
            n_tasks=int(len(plans)),
            plan_cids=tuple(str(p.cid()) for p in plans),
            chain_root_cid=str(chain_root),
        )

    def run_harness(
            self, *,
            jsonl_path: str,
            model_client: Any | None = None,
            composed_pipeline_config_cid: str = "",
    ) -> RealTaskBenchPlanChainV1:
        """Run the bench end-to-end.

        Without a model client, refuses with a structured
        ``RealTaskBenchBlockedOnModelError``.
        """
        if model_client is None:
            raise RealTaskBenchBlockedOnModelError(
                "real-task bench requires a model client; "
                "ingestion-only path: use "
                "``plan_from_swe_lite_jsonl`` instead. "
                f"jsonl_path={jsonl_path!r}")
        # The execution path is wired here for a future host
        # with a real model client. Intentionally not stubbed
        # with a synthetic model: that would be exactly the
        # anti-cheat the issue warns about.
        raise NotImplementedError(
            "model_client provided; the W84 V1 adapter ships "
            "the plan-only contract. Full execution is V2 work "
            "and lives on a GPU host with a real frontier "
            "model.")


__all__ = [
    "W84_REAL_TASK_BENCH_V1_SCHEMA_VERSION",
    "RealTaskBenchBlockedOnModelError",
    "RealTaskBenchPlanV1",
    "RealTaskBenchPlanChainV1",
    "RealTaskBenchAdapterV1",
]
