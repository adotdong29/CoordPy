"""W99 — RealWorldQA bench V5 (D2-B5 candidate, switch baseline).

Question-type-routed switch.  W99 candidate B5.

What B5 is
----------
B5 is a **same-budget switch baseline** that routes each
RealWorldQA problem to one of two existing arms based on the
deterministic question-type parser
(``detect_question_type_v2``):

    multi_choice_letter           ->  W97 D2-B0 (free-text
                                       extraction + reflexion;
                                       same as bench_v1's B arm)
    yes_no | numeric | short_text ->  A1 K=5 (unified VLM)

The budget per problem is K=5 byte-exact on the B arm regardless
of the routing decision (D2-B0 = 1 VLM reader + 4 text solver =
5 calls; A1 = 5 VLM calls).  Same VLM model on every arm.  Same
executor (``evaluate_realworldqa_answer_v1``).  No oracle.

What B5 is NOT
--------------
* B5 is **not** a frontier mechanism.  It is a *switch baseline*
  that bounds how much "team" superiority is achievable by
  routing alone vs. by a structurally new mechanism.  The
  frontier audit explicitly demoted it to baseline-only.
* B5 is **not** bounded-context / compaction / prose-summary.
  It does not modify the underlying arm prompts.
* B5 is **not** an oracle.  The question-type classifier uses
  only surface regex features (it does NOT consult the gold
  answer).
* B5 is **not** a verifier — it commits to the routed arm's
  output verbatim.  No final-turn reconciliation.

Why we run it
-------------
W97 + W98 sidecars show that on the 96_504_002 / 30-problem
slice:

* D2-B0 (free-text extraction + reflexion) PASSes 18 / 18
  multi-choice problems and FAILs 5 / 6 yes/no problems (the
  W97 unique-A1-rescue cluster).
* A1 K=5 PASSes 12 / 12 yes/no + numeric + short_text problems
  and FAILs 3 multi-choice problems (the W97 unique-B-rescue
  cluster).

A NIM-free oracle simulation of B5 on the W97 sidecars predicts
**B5 = 30 / 30 = 100 %** on this slice (multi-choice routed to
D2-B0 which owns it; yes/no/numeric/short_text routed to A1
which owns them).  B5 − A1 (NIM-free) = +10.00 pp.

Running B5 live is the cleanest cheap discriminator of:

* whether the W97 + W98 per-question outcomes are stable enough
  under temperature 0.7 NIM sampling that the prediction holds
  empirically;
* what the maximum achievable margin is by *routing alone*
  (an upper bound on any team mechanism that doesn't actually
  do more than route per question).

If B5 cleanly beats +5 pp Phase 2, this proves the per-question
ceiling is high enough that the W95-B0-family cap is a routing
problem, not an architecture problem.  It does NOT imply
multi-agent context superiority — it is an oracle-bounded
ceiling for the routing class.

Honest scope (W99 B5)
---------------------

* ``W99-L-REALWORLDQA-BENCH-V5-SWITCH-BASELINE-CAP`` — V5 is a
  switch baseline, not a substrate or trust mechanism.  It does
  not claim multi-agent context superiority.
* ``W99-L-REALWORLDQA-BENCH-V5-QUESTION-TYPE-PARSER-CAP`` —
  the routing depends on a deterministic regex parser; mistakes
  there propagate to wrong routing.  The W98 preflight AddrP6
  measured the parser at 29/30 = 96.7 % on the W97 slice.
* ``W99-L-REALWORLDQA-BENCH-V5-K5-EXACT-CAP`` — both routes use
  exactly K=5 model calls; same anti-cheat budget as A1 / B1
  / B2 / B4.
* ``W99-L-REALWORLDQA-BENCH-V5-NIM-DEPENDENT-CAP`` — V5 drives
  through caller-provided text / VLM clients.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .realworldqa_executor_v1 import (
    RealWorldQAExecutorResultV1,
    W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_realworldqa_answer_v1,
)
from .realworldqa_loader_v1 import (
    RealWorldQAProblemV1,
    W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION,
)
from .realworldqa_bench_v1 import (
    _run_a0_text as _v1_run_a0_text,
    _run_a1_vlm as _v1_run_a1_vlm,
    _run_b_vlm_team as _v1_run_b_vlm_team,
)
from .realworldqa_bench_v2 import (
    QUESTION_TYPE_MULTI_CHOICE_LETTER,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_SHORT_TEXT,
    QUESTION_TYPE_YES_NO,
    detect_question_type_v2,
    extract_candidate_answer_v1,
)


W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_bench_v5.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Routing rule
# ---------------------------------------------------------------

ROUTE_VLM_TEAM_B0: str = "vlm_team_b0"      # multi_choice_letter
ROUTE_A1_VLM_K5: str = "a1_vlm_k5"          # everything else


def b5_route_for_question(question: str) -> str:
    """Deterministic NIM-free routing decision.

    multi_choice_letter -> D2-B0 (W97 bench V1's B arm)
    yes_no | numeric | short_text -> A1 K=5
    """
    qt = detect_question_type_v2(question)
    if qt == QUESTION_TYPE_MULTI_CHOICE_LETTER:
        return ROUTE_VLM_TEAM_B0
    return ROUTE_A1_VLM_K5


# ---------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealWorldQAV5RouteDecisionCapsule:
    schema: str
    seed: int
    pid: str
    question_type: str
    route: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "pid": str(self.pid),
            "question_type": str(self.question_type),
            "route": str(self.route),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w99_realworldqa_v5_route_decision",
            **self.to_dict()})


@dataclasses.dataclass(frozen=True)
class RealWorldQAV5ArmOutcomeCapsule:
    schema: str
    seed: int
    pid: str
    arm_id: str
    question_type: str
    route: str
    final_passed: bool
    final_prediction_cid: str
    final_executor_rule: str
    n_model_calls: int
    total_wall_ms: int
    route_capsule_cid: str
    underlying_outcome_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "pid": str(self.pid),
            "arm_id": str(self.arm_id),
            "question_type": str(self.question_type),
            "route": str(self.route),
            "final_passed": bool(self.final_passed),
            "final_prediction_cid": str(
                self.final_prediction_cid),
            "final_executor_rule": str(
                self.final_executor_rule),
            "n_model_calls": int(self.n_model_calls),
            "total_wall_ms": int(self.total_wall_ms),
            "route_capsule_cid": str(self.route_capsule_cid),
            "underlying_outcome_cid": str(
                self.underlying_outcome_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w99_realworldqa_v5_arm_outcome",
            **self.to_dict()})


def _executor_result_cid(
        result: RealWorldQAExecutorResultV1) -> str:
    return _sha256_hex({
        "kind": "w99_realworldqa_v5_executor_result",
        **result.to_dict()})


# ---------------------------------------------------------------
# Gen fn signatures
# ---------------------------------------------------------------

_TextGenFn = Callable[[str, int, float], tuple[str, int]]
_VlmGenFn = Callable[
    [str, "bytes | None", int, float], tuple[str, int]]


# ---------------------------------------------------------------
# Per-arm runners (A0 / A1 = V1 verbatim; B = router over V1's B
# and V1's A1)
# ---------------------------------------------------------------

def _run_a0_text(
        *, seed: int, p: RealWorldQAProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[Any, RealWorldQAExecutorResultV1]:
    """Defers to V1's A0 implementation byte-exact."""
    return _v1_run_a0_text(
        seed=int(seed), p=p, text_gen=text_gen,
        max_tokens=int(max_tokens))


def _run_a1_vlm(
        *, seed: int, p: RealWorldQAProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        max_tokens: int,
) -> tuple[Any, list[RealWorldQAExecutorResultV1]]:
    """Defers to V1's A1 K=5 implementation byte-exact."""
    return _v1_run_a1_vlm(
        seed=int(seed), p=p, K=int(K),
        temperature=float(temperature), vlm_gen=vlm_gen,
        max_tokens=int(max_tokens))


def _run_b_routed_switch(
        *, seed: int, p: RealWorldQAProblemV1, K: int,
        temperature: float, vlm_gen: _VlmGenFn,
        text_gen: _TextGenFn, max_tokens: int,
) -> tuple[RealWorldQAV5ArmOutcomeCapsule,
           list[RealWorldQAExecutorResultV1],
           str, str, RealWorldQAV5RouteDecisionCapsule]:
    """W99 B5 (D2-B5): deterministic NIM-free routing —
    multi-choice -> D2-B0 (W97 V1's B arm); else -> A1 K=5.
    Same K=5 byte-exact budget on either route.
    """
    qt = detect_question_type_v2(p.question)
    route = b5_route_for_question(p.question)
    route_capsule = RealWorldQAV5RouteDecisionCapsule(
        schema=W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        question_type=str(qt), route=str(route))
    if route == ROUTE_VLM_TEAM_B0:
        v1_out, exes = _v1_run_b_vlm_team(
            seed=int(seed), p=p, K=int(K),
            temperature=float(temperature),
            vlm_gen=vlm_gen, text_gen=text_gen,
            max_tokens=int(max_tokens))
    else:
        v1_out, exes = _v1_run_a1_vlm(
            seed=int(seed), p=p, K=int(K),
            temperature=float(temperature),
            vlm_gen=vlm_gen,
            max_tokens=int(max_tokens))
    out = RealWorldQAV5ArmOutcomeCapsule(
        schema=W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
        seed=int(seed), pid=str(p.pid),
        arm_id="B_routed_switch",
        question_type=str(qt),
        route=str(route),
        final_passed=bool(v1_out.final_passed),
        final_prediction_cid=str(v1_out.final_prediction_cid),
        final_executor_rule=str(v1_out.final_executor_rule),
        n_model_calls=int(v1_out.n_model_calls),
        total_wall_ms=int(v1_out.total_wall_ms),
        route_capsule_cid=str(route_capsule.cid()),
        underlying_outcome_cid=str(v1_out.cid()))
    return out, exes, qt, route, route_capsule


# ---------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealWorldQAV5SeedReport:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_routed_switch_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_routed_switch_total_wall_ms: int
    n_route_vlm_team_b0: int
    n_route_a1_vlm_k5: int
    per_problem_outcomes: tuple[dict[str, Any], ...]
    outcome_cids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_text_pass_at_1": float(round(
                self.a0_text_pass_at_1, 6)),
            "a1_vlm_pass_at_1": float(round(
                self.a1_vlm_pass_at_1, 6)),
            "b_routed_switch_pass_at_1": float(round(
                self.b_routed_switch_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_routed_switch_total_wall_ms": int(
                self.b_routed_switch_total_wall_ms),
            "n_route_vlm_team_b0": int(
                self.n_route_vlm_team_b0),
            "n_route_a1_vlm_k5": int(
                self.n_route_a1_vlm_k5),
            "per_problem_outcomes": [
                dict(po) for po in self.per_problem_outcomes],
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class RealWorldQAV5BenchReport:
    schema: str
    vlm_model_id: str
    text_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    corpus_parquet_shard_sha256: tuple[str, ...]
    corpus_merkle_root: str
    per_seed: tuple[RealWorldQAV5SeedReport, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_routed_switch_mean_pass_at_1: float
    b_beats_a0_text_per_seed: tuple[bool, ...]
    b_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_mean_strictly_beats_a0_text_mean: bool
    b_mean_strictly_beats_a1_vlm_mean: bool
    b_mean_minus_a0_text_mean_pp: float
    b_mean_minus_a1_vlm_mean_pp: float
    n_b_ge_a1_problems_per_seed: tuple[int, ...]
    question_type_distribution: dict[str, int]
    route_distribution: dict[str, int]
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vlm_model_id": str(self.vlm_model_id),
            "text_model_id": str(self.text_model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "corpus_parquet_shard_sha256": list(
                self.corpus_parquet_shard_sha256),
            "corpus_merkle_root": str(self.corpus_merkle_root),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_text_mean_pass_at_1": float(round(
                self.a0_text_mean_pass_at_1, 6)),
            "a1_vlm_mean_pass_at_1": float(round(
                self.a1_vlm_mean_pass_at_1, 6)),
            "b_routed_switch_mean_pass_at_1": float(round(
                self.b_routed_switch_mean_pass_at_1, 6)),
            "b_beats_a0_text_per_seed": list(
                self.b_beats_a0_text_per_seed),
            "b_beats_a1_vlm_per_seed": list(
                self.b_beats_a1_vlm_per_seed),
            "b_mean_strictly_beats_a0_text_mean": bool(
                self.b_mean_strictly_beats_a0_text_mean),
            "b_mean_strictly_beats_a1_vlm_mean": bool(
                self.b_mean_strictly_beats_a1_vlm_mean),
            "b_mean_minus_a0_text_mean_pp": float(round(
                self.b_mean_minus_a0_text_mean_pp, 4)),
            "b_mean_minus_a1_vlm_mean_pp": float(round(
                self.b_mean_minus_a1_vlm_mean_pp, 4)),
            "n_b_ge_a1_problems_per_seed": list(
                self.n_b_ge_a1_problems_per_seed),
            "question_type_distribution": dict(
                self.question_type_distribution),
            "route_distribution": dict(
                self.route_distribution),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class RealWorldQAV5BenchConfig:
    schema: str = W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (96_504_002,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 384


def run_realworldqa_bench_v5(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        text_model_id: str,
        corpus: Sequence[RealWorldQAProblemV1],
        corpus_parquet_shard_sha256: tuple[str, ...],
        corpus_merkle_root: str,
        config: RealWorldQAV5BenchConfig | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
        sidecar_writer: (
            Callable[[dict[str, Any]], None] | None) = None,
) -> RealWorldQAV5BenchReport:
    from .realworldqa_loader_v1 import (
        select_realworldqa_subset_v1)

    cfg = config or RealWorldQAV5BenchConfig()
    per_seed: list[RealWorldQAV5SeedReport] = []
    all_outcome_cids: list[str] = []
    qt_counter: dict[str, int] = {}
    route_counter: dict[str, int] = {}
    for seed in cfg.seeds:
        subset = select_realworldqa_subset_v1(
            seed=int(seed),
            n_problems=int(cfg.n_problems),
            corpus=tuple(corpus))
        if len(subset) < int(cfg.n_problems):
            raise RuntimeError(
                "corpus has only "
                f"{len(subset)} problems for seed {seed}; "
                f"need {cfg.n_problems}")
        a0_outs: list = []
        a1_outs: list = []
        b_outs: list[RealWorldQAV5ArmOutcomeCapsule] = []
        per_problem_outcomes: list[dict[str, Any]] = []
        n_b0 = 0
        n_a1 = 0
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(p.pid))
            a0_out, _ = _run_a0_text(
                seed=int(seed), p=p,
                text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a0_outs.append(a0_out)
            a1_out, _ = _run_a1_vlm(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                max_tokens=int(cfg.max_tokens_per_call))
            a1_outs.append(a1_out)
            b_out, _, qt, route, route_capsule = (
                _run_b_routed_switch(
                    seed=int(seed), p=p,
                    K=int(cfg.K_multi_sample),
                    temperature=float(cfg.sampling_temperature),
                    vlm_gen=vlm_gen, text_gen=text_gen,
                    max_tokens=int(cfg.max_tokens_per_call)))
            b_outs.append(b_out)
            qt_counter[qt] = qt_counter.get(qt, 0) + 1
            route_counter[route] = route_counter.get(
                route, 0) + 1
            if route == ROUTE_VLM_TEAM_B0:
                n_b0 += 1
            else:
                n_a1 += 1
            per_problem_outcomes.append({
                "pid": str(p.pid),
                "question": str(p.question),
                "gold_answer": str(p.answer),
                "question_type": str(qt),
                "route": str(route),
                "a0_text_passed": bool(a0_out.final_passed),
                "a1_vlm_passed": bool(a1_out.final_passed),
                "b_routed_switch_passed": bool(
                    b_out.final_passed),
                "a0_outcome_cid": str(a0_out.cid()),
                "a1_outcome_cid": str(a1_out.cid()),
                "b_outcome_cid": str(b_out.cid()),
                "b_route_capsule_cid": str(route_capsule.cid()),
            })
            if sidecar_writer is not None:
                sidecar_writer({
                    "kind": (
                        "w99_realworldqa_v5_per_problem_outcome"),
                    **per_problem_outcomes[-1],
                })
        n = float(len(a0_outs))
        a0_acc = sum(
            1 for o in a0_outs if o.final_passed) / n
        a1_acc = sum(
            1 for o in a1_outs if o.final_passed) / n
        b_acc = sum(
            1 for o in b_outs if o.final_passed) / n
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w99_realworldqa_v5_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(RealWorldQAV5SeedReport(
            schema=W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_routed_switch_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_routed_switch_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            n_route_vlm_team_b0=int(n_b0),
            n_route_a1_vlm_k5=int(n_a1),
            per_problem_outcomes=tuple(per_problem_outcomes),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle)))
        all_outcome_cids.extend(outcome_cids)
    nseeds = float(len(per_seed))
    a0_mean = sum(
        s.a0_text_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(
        s.a1_vlm_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(
        s.b_routed_switch_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w99_realworldqa_v5_bench_merkle_root",
        "vlm_model_id": str(vlm_model_id),
        "text_model_id": str(text_model_id),
        "corpus_parquet_shard_sha256": list(
            corpus_parquet_shard_sha256),
        "corpus_merkle_root": str(corpus_merkle_root),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
        "n_problems": int(cfg.n_problems),
        "K": int(cfg.K_multi_sample),
    })
    n_b_ge_a1_per_seed: list[int] = []
    for s in per_seed:
        n_bg = 0
        for po in s.per_problem_outcomes:
            if (bool(po["b_routed_switch_passed"])
                    >= bool(po["a1_vlm_passed"])):
                n_bg += 1
        n_b_ge_a1_per_seed.append(int(n_bg))
    report = RealWorldQAV5BenchReport(
        schema=W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        text_model_id=str(text_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        corpus_parquet_shard_sha256=tuple(
            corpus_parquet_shard_sha256),
        corpus_merkle_root=str(corpus_merkle_root),
        per_seed=tuple(per_seed),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_routed_switch_mean_pass_at_1=float(b_mean),
        b_beats_a0_text_per_seed=tuple(
            s.b_routed_switch_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_beats_a1_vlm_per_seed=tuple(
            s.b_routed_switch_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_mean_strictly_beats_a0_text_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_vlm_mean=bool(b_mean > a1_mean),
        b_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        n_b_ge_a1_problems_per_seed=tuple(n_b_ge_a1_per_seed),
        question_type_distribution=dict(qt_counter),
        route_distribution=dict(route_counter),
        bench_merkle_root=str(bench_merkle))
    return report


__all__ = [
    "W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION",
    "ROUTE_VLM_TEAM_B0",
    "ROUTE_A1_VLM_K5",
    "b5_route_for_question",
    "RealWorldQAV5RouteDecisionCapsule",
    "RealWorldQAV5ArmOutcomeCapsule",
    "RealWorldQAV5SeedReport",
    "RealWorldQAV5BenchReport",
    "RealWorldQAV5BenchConfig",
    "run_realworldqa_bench_v5",
]
