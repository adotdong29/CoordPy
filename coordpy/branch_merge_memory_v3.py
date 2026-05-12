"""W53 M7 — Branch Merge Memory V3 (consensus + audit + abstain).

Extends W52 ``BranchCycleMemoryV2Head`` with:

* **consensus pages** keyed by ``(branch_pair, cycle)`` that hold
  cross-branch consensus reads, populated when ≥ K-of-N joint
  pages return cosine-similar values
* **abstain semantics** when no consensus is reached: read
  returns a sentinel rather than a noisy average
* **audit trail extension** that records every consensus
  population with parent page CIDs

Pure-Python only — wraps W52 BCM V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .branch_cycle_memory_v2 import (
    BranchCycleMemoryV2Head,
    MergeAuditEntry as V2MergeAuditEntry,
    W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
    W52_DEFAULT_BCM_V2_PAGE_SLOTS,
)
from .branch_cycle_memory import (
    PageSlot, StoragePage,
    W51_DEFAULT_BCM_FACTOR_DIM,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_BMM_V3_SCHEMA_VERSION: str = (
    "coordpy.branch_merge_memory_v3.v1")

W53_DEFAULT_BMM_V3_N_CONSENSUS_PAGES: int = 4
W53_DEFAULT_BMM_V3_K_REQUIRED: int = 2
W53_DEFAULT_BMM_V3_COSINE_FLOOR: float = 0.5
W53_BMM_V3_NO_CONSENSUS: tuple[float, ...] = ()


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# Consensus audit entry
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusAuditEntry:
    consensus_page_index: int
    branch_pair_indices: tuple[int, ...]
    cycle_index: int
    cosine_floor: float
    n_quorum: int
    fact_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_page_index": int(
                self.consensus_page_index),
            "branch_pair_indices": list(
                self.branch_pair_indices),
            "cycle_index": int(self.cycle_index),
            "cosine_floor": float(round(
                self.cosine_floor, 12)),
            "n_quorum": int(self.n_quorum),
            "fact_tags": list(self.fact_tags),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_bmm_v3_consensus_entry",
            "entry": self.to_dict()})


# =============================================================================
# BranchMergeMemoryV3Head
# =============================================================================


@dataclasses.dataclass
class BranchMergeMemoryV3Head:
    """V3 BMM head with consensus pages + abstain semantics."""

    inner_v2: BranchCycleMemoryV2Head
    n_consensus_pages: int
    consensus_pages: list[StoragePage]
    k_required: int
    cosine_floor: float
    consensus_audit: list[ConsensusAuditEntry]

    @classmethod
    def init(
            cls, *,
            factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages: int = (
                W51_DEFAULT_BCM_N_BRANCH_PAGES),
            n_cycle_pages: int = (
                W51_DEFAULT_BCM_N_CYCLE_PAGES),
            page_capacity: int = (
                W52_DEFAULT_BCM_V2_PAGE_SLOTS),
            n_joint_pages: int = (
                W52_DEFAULT_BCM_V2_N_JOINT_PAGES),
            n_consensus_pages: int = (
                W53_DEFAULT_BMM_V3_N_CONSENSUS_PAGES),
            k_required: int = W53_DEFAULT_BMM_V3_K_REQUIRED,
            cosine_floor: float = (
                W53_DEFAULT_BMM_V3_COSINE_FLOOR),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BranchMergeMemoryV3Head":
        rng = _DeterministicLCG(seed=int(seed))
        inner = BranchCycleMemoryV2Head.init(
            factor_dim=int(factor_dim),
            n_branch_pages=int(n_branch_pages),
            n_cycle_pages=int(n_cycle_pages),
            page_capacity=int(page_capacity),
            n_joint_pages=int(n_joint_pages),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        consensus_pages = [
            StoragePage(
                page_index=i,
                page_kind="consensus",
                capacity=int(page_capacity),
                factor_dim=int(factor_dim))
            for i in range(int(n_consensus_pages))
        ]
        return cls(
            inner_v2=inner,
            n_consensus_pages=int(n_consensus_pages),
            consensus_pages=consensus_pages,
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            consensus_audit=[])

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v2.params())

    @property
    def factor_dim(self) -> int:
        return int(self.inner_v2.factor_dim)

    @property
    def n_joint_pages(self) -> int:
        return int(self.inner_v2.n_joint_pages)

    def write_to_inner_joint(
            self, *,
            branch_index: int,
            cycle_index: int,
            key: Sequence[float],
            value: Sequence[float],
            fact_tag: str = "",
    ) -> None:
        self.inner_v2.write_to_joint(
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            key=key, value=value, fact_tag=str(fact_tag))

    def maybe_populate_consensus(
            self, *,
            cycle_index: int,
            query: Sequence[float],
    ) -> ConsensusAuditEntry | None:
        """Vote across joint pages: if ≥ K share cosine ≥ floor,
        store the consensus payload in a consensus page."""
        # Collect joint reads.
        reads: list[tuple[int, list[float]]] = []
        for i in range(int(self.n_joint_pages)):
            v = self.inner_v2.joint_pages[i].read_value(query)
            reads.append((int(i), list(v)))
        # Compute pairwise cosines.
        n = len(reads)
        if n < int(self.k_required):
            return None
        # Greedy clique by central-cosine.
        cos = [[1.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = _cosine(reads[i][1], reads[j][1])
                cos[i][j] = float(c)
                cos[j][i] = float(c)
        centrality = [
            sum(cos[i][j] for j in range(n) if j != i)
            for i in range(n)
        ]
        order = sorted(
            range(n), key=lambda i: -centrality[i])
        selected: list[int] = []
        for i in order:
            ok = True
            for s in selected:
                if cos[i][s] < float(self.cosine_floor):
                    ok = False
                    break
            if ok:
                selected.append(i)
        if len(selected) < int(self.k_required):
            return None
        # Average the selected reads into a consensus payload.
        sd = int(self.factor_dim)
        consensus = [0.0] * sd
        for s in selected:
            v = reads[s][1]
            for j in range(sd):
                consensus[j] += float(
                    v[j] if j < len(v) else 0.0) / float(
                        len(selected))
        # Write to the consensus page indexed by cycle.
        idx = int(cycle_index) % max(
            1, int(self.n_consensus_pages))
        page = self.consensus_pages[idx]
        slot = PageSlot(
            slot_index=int(len(page.slots)),
            factor_dim=sd,
            key=tuple(_round_floats(query)),
            value=tuple(_round_floats(consensus)),
            fact_tag=f"consensus:k={self.k_required}")
        # Eviction: drop lowest-importance if full.
        if len(page.slots) >= int(page.capacity):
            importances = [
                float(sum(float(v) * float(v)
                          for v in s.value))
                for s in page.slots
            ]
            min_idx = importances.index(min(importances))
            page.slots.pop(min_idx)
            for i, s in enumerate(page.slots):
                page.slots[i] = dataclasses.replace(
                    s, slot_index=i)
        page.write(slot)
        entry = ConsensusAuditEntry(
            consensus_page_index=int(idx),
            branch_pair_indices=tuple(
                int(reads[s][0]) for s in selected),
            cycle_index=int(cycle_index),
            cosine_floor=float(self.cosine_floor),
            n_quorum=int(len(selected)),
            fact_tags=(f"k={self.k_required}",
                       f"n={len(selected)}"))
        self.consensus_audit.append(entry)
        return entry

    def read_consensus(
            self, *,
            cycle_index: int,
            query: Sequence[float],
    ) -> tuple[list[float], bool]:
        """Read from consensus page; return (payload, abstain).

        ``abstain=True`` when no consensus has been populated for
        this cycle index (i.e. ``maybe_populate_consensus`` never
        returned an entry).
        """
        idx = int(cycle_index) % max(
            1, int(self.n_consensus_pages))
        page = self.consensus_pages[idx]
        if not page.slots:
            return [0.0] * int(self.factor_dim), True
        return list(page.read_value(query)), False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W53_BMM_V3_SCHEMA_VERSION),
            "inner_v2": self.inner_v2.to_dict(),
            "n_consensus_pages": int(
                self.n_consensus_pages),
            "k_required": int(self.k_required),
            "cosine_floor": float(round(
                self.cosine_floor, 12)),
            "consensus_pages_n_slots": [
                int(len(p.slots))
                for p in self.consensus_pages],
            "consensus_audit_count": int(
                len(self.consensus_audit)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_bmm_v3_head",
            "head": self.to_dict()})


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BranchMergeMemoryV3Witness:
    head_cid: str
    inner_v2_cid: str
    n_consensus_pages: int
    k_required: int
    cosine_floor: float
    consensus_audit_count: int
    consensus_recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "inner_v2_cid": str(self.inner_v2_cid),
            "n_consensus_pages": int(
                self.n_consensus_pages),
            "k_required": int(self.k_required),
            "cosine_floor": float(round(
                self.cosine_floor, 12)),
            "consensus_audit_count": int(
                self.consensus_audit_count),
            "consensus_recall": float(round(
                self.consensus_recall, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_bmm_v3_witness",
            "witness": self.to_dict()})


def emit_bmm_v3_witness(
        *,
        head: BranchMergeMemoryV3Head,
        consensus_recall: float = 0.0,
) -> BranchMergeMemoryV3Witness:
    return BranchMergeMemoryV3Witness(
        head_cid=str(head.cid()),
        inner_v2_cid=str(head.inner_v2.cid()),
        n_consensus_pages=int(head.n_consensus_pages),
        k_required=int(head.k_required),
        cosine_floor=float(head.cosine_floor),
        consensus_audit_count=int(
            len(head.consensus_audit)),
        consensus_recall=float(consensus_recall),
    )


# =============================================================================
# Verifier
# =============================================================================

W53_BMM_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_bmm_v3_head_cid_mismatch",
    "w53_bmm_v3_inner_v2_cid_mismatch",
    "w53_bmm_v3_consensus_pages_mismatch",
    "w53_bmm_v3_k_required_mismatch",
    "w53_bmm_v3_consensus_recall_below_floor",
)


def verify_bmm_v3_witness(
        witness: BranchMergeMemoryV3Witness,
        *,
        expected_head_cid: str | None = None,
        expected_inner_v2_cid: str | None = None,
        expected_n_consensus_pages: int | None = None,
        expected_k_required: int | None = None,
        min_consensus_recall: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != str(expected_head_cid)):
        failures.append("w53_bmm_v3_head_cid_mismatch")
    if (expected_inner_v2_cid is not None
            and witness.inner_v2_cid
            != str(expected_inner_v2_cid)):
        failures.append("w53_bmm_v3_inner_v2_cid_mismatch")
    if (expected_n_consensus_pages is not None
            and witness.n_consensus_pages
            != int(expected_n_consensus_pages)):
        failures.append(
            "w53_bmm_v3_consensus_pages_mismatch")
    if (expected_k_required is not None
            and witness.k_required
            != int(expected_k_required)):
        failures.append("w53_bmm_v3_k_required_mismatch")
    if (min_consensus_recall is not None
            and witness.consensus_recall
            < float(min_consensus_recall)):
        failures.append(
            "w53_bmm_v3_consensus_recall_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_consensus_recall(
        head: BranchMergeMemoryV3Head,
        *,
        n_branches: int = 4,
        n_consistent: int = 3,
        factor_dim: int | None = None,
        seed: int = 0,
) -> float:
    """Insert a consistent payload across ``n_consistent``
    distinct joint pages plus one outlier, then measure
    consensus recall.

    Spreads writes across joint pages by using cycle_index = i,
    branch_index = 0 — the W52 ``_joint_index`` formula
    ``(branch * n_cycle_pages + cycle) % n_joint_pages`` then
    maps each ``i`` to a distinct page.
    """
    sd = int(
        factor_dim if factor_dim is not None
        else head.factor_dim)
    rng = _DeterministicLCG(seed=int(seed))
    n_succ = 0
    n_total = 0
    n_consensus_cycles = 4
    for trial in range(n_consensus_cycles):
        # Reset all joint + consensus pages.
        for p in head.consensus_pages:
            p.reset()
        for p in head.inner_v2.joint_pages:
            p.reset()
        # Build a clean target payload.
        target = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(sd)
        ]
        # Write target to n_consistent distinct joint pages.
        for b in range(int(n_consistent)):
            noisy = [
                float(target[j])
                + 0.02 * float(
                    rng.next_uniform() * 2.0 - 1.0)
                for j in range(sd)
            ]
            head.write_to_inner_joint(
                branch_index=0,
                cycle_index=int(b),
                key=target, value=noisy,
                fact_tag=f"consistent_{b}")
        # Outlier: opposite sign at a distinct page.
        outlier = [-float(t) for t in target]
        head.write_to_inner_joint(
            branch_index=0,
            cycle_index=int(n_consistent),
            key=target, value=outlier,
            fact_tag="outlier")
        # Populate consensus.
        entry = head.maybe_populate_consensus(
            cycle_index=int(trial), query=target)
        n_total += 1
        if entry is None:
            continue
        payload, abstain = head.read_consensus(
            cycle_index=int(trial), query=target)
        if not abstain and _cosine(
                payload, target) > 0.5:
            n_succ += 1
    return float(n_succ) / float(max(1, n_total))


__all__ = [
    "W53_BMM_V3_SCHEMA_VERSION",
    "W53_DEFAULT_BMM_V3_N_CONSENSUS_PAGES",
    "W53_DEFAULT_BMM_V3_K_REQUIRED",
    "W53_DEFAULT_BMM_V3_COSINE_FLOOR",
    "W53_BMM_V3_VERIFIER_FAILURE_MODES",
    "ConsensusAuditEntry",
    "BranchMergeMemoryV3Head",
    "BranchMergeMemoryV3Witness",
    "emit_bmm_v3_witness",
    "verify_bmm_v3_witness",
    "evaluate_consensus_recall",
]
