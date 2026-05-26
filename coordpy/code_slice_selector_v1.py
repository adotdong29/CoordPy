"""W102 / COO-14 — code-side slice-selection + candidate-ranking
helper V1.

Delivers the four-item COO-14 Definition of Done verbatim:

1. **Rank candidate directions cheaply** using committed W88–W93
   + W101 evidence.
2. **Select failure-cluster slices for pilots** subject to a
   cheap-pilot budget.
3. **Answer "what exact problems should the next cheap pilot
   attack?"** with a justification per problem.
4. **Feed output into runbooks** before expensive runs are
   approved.

Consumes the W101 / W102 arsenal-mining report JSON shape
(`mining_report.json` produced by
`scripts/run_w101_arsenal_mining.py` and its W102 extension).

Stand-alone module: no NIM, no expensive bench, no model loading.
Unit-tested.

Honest scope (W102)
-------------------

* ``W102-L-CODE-SLICE-SELECTOR-V1-MINING-DEPENDENT-CAP`` — the
  selector reads pre-computed arsenal-mining reports.  It does
  NOT itself re-execute candidate responses; that lives in
  `scripts/run_w10X_arsenal_mining.py`.
* ``W102-L-CODE-SLICE-SELECTOR-V1-CHEAP-PILOT-SCOPE-CAP`` — the
  default cheap-pilot budget is 30 problems × 11 calls/problem =
  330 NIM calls (matches W89 / W91 / W101 / W102 cheap-pilot
  convention).  Larger budgets are explicit-parameter only.
* ``W102-L-CODE-SLICE-SELECTOR-V1-ANTI-PATTERN-GUARD-CAP`` —
  refuses to propose a slice for any candidate whose name
  contains a forbidden anti-pattern token (`bounded_window`,
  `compaction`, etc.).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


W102_CODE_SLICE_SELECTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.code_slice_selector_v1.v1")


FORBIDDEN_TOKENS: tuple[str, ...] = (
    "bounded_window",
    "compaction",
    "context_compaction",
    "prose_summary",
    "context_pruning",
    "summarizer",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class BenchCandidateRanking:
    """One row of the candidate-direction ranking table."""

    bench: str
    rescue_fraction: float       # B-only wins / total B wins
    hard_cluster_size: int       # shared_fails
    mean_b_minus_a1_pp: float    # mean per-seed margin
    per_seed_margin_std_pp: float  # variance / spread
    n_problems_per_seed: int
    n_seeds: int
    composite_score: float       # see compute_composite_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench": str(self.bench),
            "rescue_fraction": float(round(
                self.rescue_fraction, 4)),
            "hard_cluster_size": int(self.hard_cluster_size),
            "mean_b_minus_a1_pp": float(round(
                self.mean_b_minus_a1_pp, 4)),
            "per_seed_margin_std_pp": float(round(
                self.per_seed_margin_std_pp, 4)),
            "n_problems_per_seed": int(self.n_problems_per_seed),
            "n_seeds": int(self.n_seeds),
            "composite_score": float(round(
                self.composite_score, 4)),
        }


@dataclasses.dataclass(frozen=True)
class SliceProposalEntry:
    """One problem proposed for the next cheap pilot."""

    bench: str
    seed: int
    task_id: str
    cluster: str    # b_only_wins | shared_fails | a1_only_wins
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench": str(self.bench),
            "seed": int(self.seed),
            "task_id": str(self.task_id),
            "cluster": str(self.cluster),
            "justification": str(self.justification),
        }


@dataclasses.dataclass(frozen=True)
class SliceProposal:
    """The W102 helper-lane output the runbook reads."""

    schema: str
    bench: str
    n_problems: int
    cheap_pilot_budget_nim_calls: int
    proposal: tuple[SliceProposalEntry, ...]
    rationale_summary: str
    proposal_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "bench": str(self.bench),
            "n_problems": int(self.n_problems),
            "cheap_pilot_budget_nim_calls": int(
                self.cheap_pilot_budget_nim_calls),
            "proposal": [
                e.to_dict() for e in self.proposal],
            "rationale_summary": str(self.rationale_summary),
            "proposal_cid": str(self.proposal_cid),
        }


def _bench_block(mining: dict[str, Any], bench: str) -> dict[str, Any] | None:
    block = mining.get(bench)
    return block if isinstance(block, dict) else None


def _per_seed_margins(block: dict[str, Any]) -> list[float]:
    per_seed = block.get("per_seed") or {}
    out: list[float] = []
    for v in per_seed.values():
        if isinstance(v, dict) and "b_minus_a1_pp" in v:
            out.append(float(v["b_minus_a1_pp"]))
    return out


def compute_composite_score(
        *,
        rescue_fraction: float,
        hard_cluster_size: int,
        mean_b_minus_a1_pp: float,
        per_seed_margin_std_pp: float,
        n_problems_per_seed: int,
) -> float:
    """A simple monotonic composite ordering.  Higher score ⇒
    more attractive candidate direction:

    * Reward rescue_fraction (mechanism is load-bearing).
    * Reward mean_b_minus_a1_pp (margin already positive).
    * Reward per_seed_margin_std_pp at modest levels (some
      variance ⇒ surface to attack), penalise extreme variance.
    * Reward hard_cluster_size as a *bench-quality* signal
      (larger hard cluster ⇒ more headroom; surgical
      benchmarks like MBPP+ should rank higher).
    """
    # Normalise hard-cluster size by per-seed problem count so
    # different bench sizes are comparable.
    if n_problems_per_seed > 0:
        hard_fraction = float(
            hard_cluster_size / n_problems_per_seed)
    else:
        hard_fraction = 0.0
    # Squashed variance term (saturating; rewards 0 – 8 pp std).
    var_term = min(per_seed_margin_std_pp / 10.0, 1.0)
    score = (
        2.0 * float(rescue_fraction)
        + 0.10 * float(mean_b_minus_a1_pp)
        + 0.30 * float(hard_fraction)
        + 0.20 * float(var_term))
    return float(score)


def rank_candidate_benches(
        *, mining: dict[str, Any],
        benches: Iterable[str] = ("humaneval", "mbpp"),
) -> list[BenchCandidateRanking]:
    """Rank each bench by its composite score using the per-bench
    cluster surface in the arsenal-mining report."""
    out: list[BenchCandidateRanking] = []
    for bench in benches:
        block = _bench_block(mining, bench)
        if block is None:
            continue
        agg = block.get("aggregate") or {}
        mlb = block.get(
            "mechanism_load_bearing_estimate") or {}
        n_b_only = int(agg.get("n_b_only_wins", 0))
        n_b_wins_total = int(mlb.get("n_b_wins_total", 0))
        rescue = (
            float(n_b_only / n_b_wins_total)
            if n_b_wins_total > 0 else 0.0)
        n_shared_fails = int(agg.get("n_shared_fails", 0))
        margins = _per_seed_margins(block)
        if margins:
            mean_m = float(sum(margins) / len(margins))
            std_m = float(
                (sum((m - mean_m) ** 2 for m in margins)
                 / max(len(margins), 1)) ** 0.5)
        else:
            mean_m = 0.0
            std_m = 0.0
        n_problems_per_seed = int(
            block.get("n_problems_per_seed", 0))
        n_seeds = int(block.get("n_seeds", 0))
        score = compute_composite_score(
            rescue_fraction=rescue,
            hard_cluster_size=n_shared_fails,
            mean_b_minus_a1_pp=mean_m,
            per_seed_margin_std_pp=std_m,
            n_problems_per_seed=n_problems_per_seed)
        out.append(BenchCandidateRanking(
            bench=str(bench),
            rescue_fraction=float(rescue),
            hard_cluster_size=int(n_shared_fails),
            mean_b_minus_a1_pp=float(mean_m),
            per_seed_margin_std_pp=float(std_m),
            n_problems_per_seed=int(n_problems_per_seed),
            n_seeds=int(n_seeds),
            composite_score=float(score),
        ))
    return sorted(
        out, key=lambda r: r.composite_score, reverse=True)


def _parse_seeded_tid(s: str) -> tuple[int, str]:
    """Parse `<seed>:<task_id>` produced by the arsenal-mining
    aggregator."""
    s = str(s)
    if ":" not in s:
        return -1, s
    head, tail = s.split(":", 1)
    try:
        return int(head), str(tail)
    except ValueError:
        return -1, s


def propose_cheap_pilot_slice(
        *, mining: dict[str, Any], bench: str,
        n_problems: int = 30,
        bench_module_name: str | None = None,
) -> SliceProposal:
    """Produce a slice proposal for the next cheap pilot on
    ``bench``.

    Selection priority (deterministic):
      1. Unique-B-rescue problems (b_only_wins) — direct evidence
         of mechanism load-bearingness.
      2. Hard-cluster problems (shared_fails) — bench-stress
         surface.
      3. Unique-A1-only problems (a1_only_wins) — anti-mechanism
         calibration.
      4. Top-up from shared_wins until n_problems reached.

    Anti-pattern guard: if ``bench_module_name`` is provided AND
    contains any forbidden token (e.g., `bounded_window`), the
    selector REFUSES to produce a proposal."""
    if bench_module_name is not None:
        for tok in FORBIDDEN_TOKENS:
            if tok in str(bench_module_name):
                raise ValueError(
                    "code_slice_selector_v1 refuses to propose "
                    f"a slice for bench_module_name containing "
                    f"forbidden anti-pattern token {tok!r}.")
    block = _bench_block(mining, bench)
    if block is None:
        raise ValueError(
            f"bench {bench!r} not present in mining report")
    agg = block.get("aggregate") or {}
    b_only = list(agg.get("b_only_wins") or [])
    shared_fails = list(agg.get("shared_fails") or [])
    a1_only = list(agg.get("a1_only_wins") or [])
    shared_wins = list(agg.get("shared_wins") or [])
    proposals: list[SliceProposalEntry] = []
    seen: set[tuple[int, str]] = set()
    for tag, cluster, just_template in (
            (b_only, "b_only_wins",
             "unique-B-rescue: reflexion mechanism rescued this "
             "in the historical bench"),
            (shared_fails, "shared_fails",
             "hard-cluster: neither A1 nor B passed historically;"
             " mechanism stress surface"),
            (a1_only, "a1_only_wins",
             "A1-only-win: anti-mechanism calibration "
             "(reflexion lost ground here)"),
            (shared_wins, "shared_wins",
             "top-up from shared_wins to fill slice")):
        for s in tag:
            if len(proposals) >= n_problems:
                break
            seed, tid = _parse_seeded_tid(s)
            key = (int(seed), str(tid))
            if key in seen:
                continue
            seen.add(key)
            proposals.append(SliceProposalEntry(
                bench=str(bench),
                seed=int(seed),
                task_id=str(tid),
                cluster=str(cluster),
                justification=str(just_template),
            ))
        if len(proposals) >= n_problems:
            break
    cluster_counts: dict[str, int] = {}
    for e in proposals:
        cluster_counts[e.cluster] = (
            cluster_counts.get(e.cluster, 0) + 1)
    rationale = (
        f"Proposed {len(proposals)} problems from {bench}; "
        f"cluster distribution = " + ", ".join(
            f"{c}={n}"
            for c, n in sorted(cluster_counts.items())) + ". "
        "Priority: b_only_wins (mechanism rescue) → "
        "shared_fails (bench stress) → a1_only_wins "
        "(anti-mechanism calibration) → shared_wins (top-up).")
    proposal = SliceProposal(
        schema=W102_CODE_SLICE_SELECTOR_V1_SCHEMA_VERSION,
        bench=str(bench),
        n_problems=int(len(proposals)),
        cheap_pilot_budget_nim_calls=int(
            len(proposals) * 11),
        proposal=tuple(proposals),
        rationale_summary=str(rationale),
        proposal_cid="",
    )
    cid = _sha256_hex(proposal.to_dict())
    return dataclasses.replace(proposal, proposal_cid=str(cid))


def format_slice_proposal_markdown(
        proposal: SliceProposal) -> str:
    """Serialise a slice proposal into a runbook-ready Markdown
    table that W103+ runbooks can `include` verbatim."""
    lines = [
        f"# Cheap-pilot slice proposal — {proposal.bench}",
        "",
        f"* n_problems: {proposal.n_problems}",
        f"* approximate NIM budget: "
        f"{proposal.cheap_pilot_budget_nim_calls} calls "
        "(at K=5, 11 calls/problem)",
        f"* proposal CID: `{proposal.proposal_cid}`",
        f"* rationale: {proposal.rationale_summary}",
        "",
        "| # | Seed | task_id | Cluster | Justification |",
        "|---|---|---|---|---|",
    ]
    for i, e in enumerate(proposal.proposal):
        lines.append(
            f"| {i+1} | {e.seed} | {e.task_id} | "
            f"{e.cluster} | {e.justification} |")
    return "\n".join(lines) + "\n"


def format_ranking_markdown(
        rankings: list[BenchCandidateRanking]) -> str:
    """Serialise a candidate-direction ranking into a runbook-
    ready Markdown table."""
    lines = [
        "# Code-side candidate-direction ranking",
        "",
        "| Rank | Bench | rescue_fraction | hard_cluster_size | "
        "mean_B-A1_pp | per_seed_std_pp | composite_score |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rankings):
        lines.append(
            f"| {i+1} | {r.bench} | "
            f"{r.rescue_fraction*100:.2f}% | "
            f"{r.hard_cluster_size} | "
            f"{r.mean_b_minus_a1_pp:+.2f} | "
            f"{r.per_seed_margin_std_pp:.2f} | "
            f"{r.composite_score:.4f} |")
    return "\n".join(lines) + "\n"


def load_mining_report(path: Path | str) -> dict[str, Any]:
    """Load a W101 / W102 arsenal-mining report JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"mining report not found at {p}")
    return json.loads(p.read_text())


__all__ = [
    "W102_CODE_SLICE_SELECTOR_V1_SCHEMA_VERSION",
    "FORBIDDEN_TOKENS",
    "BenchCandidateRanking",
    "SliceProposalEntry",
    "SliceProposal",
    "compute_composite_score",
    "rank_candidate_benches",
    "propose_cheap_pilot_slice",
    "format_slice_proposal_markdown",
    "format_ranking_markdown",
    "load_mining_report",
]
