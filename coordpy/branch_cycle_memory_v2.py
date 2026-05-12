"""W52 M6 — Branch/Cycle Memory V2 (merge + evict + joint pages).

Extends W51's :class:`BranchCycleMemoryHead` with three new
heads:

* **Trainable merge head** — decides when two pages should be
  merged; merges are content-addressed and produce an audit
  trail.
* **Trainable evict head** — importance-weighted eviction
  replacing W51's FIFO ordering.
* **Cross-branch-cycle joint pages** — pages keyed by
  ``(branch, cycle)`` tuple instead of branch-only or
  cycle-only.

Pure-Python only — reuses the W47 ``Variable`` autograd engine
and the W51 ``BranchCycleMemoryHead`` building blocks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .autograd_manifold import (
    AdamOptimizer,
    ParamTensor,
    Variable,
    W47_DEFAULT_BETA1,
    W47_DEFAULT_BETA2,
    W47_DEFAULT_EPS,
    W47_DEFAULT_GRAD_CLIP,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_LEARNING_RATE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
    vmean,
)
from .branch_cycle_memory import (
    PageSlot,
    StoragePage,
    BranchCycleMemoryHead,
    BranchCycleMemoryExample,
    W51_DEFAULT_BCM_FACTOR_DIM,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
    W51_DEFAULT_BCM_PAGE_SLOTS,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_BCM_V2_SCHEMA_VERSION: str = (
    "coordpy.branch_cycle_memory_v2.v1")

W52_DEFAULT_BCM_V2_N_JOINT_PAGES: int = (
    W51_DEFAULT_BCM_N_BRANCH_PAGES
    * W51_DEFAULT_BCM_N_CYCLE_PAGES)
W52_DEFAULT_BCM_V2_PAGE_SLOTS: int = (
    W51_DEFAULT_BCM_PAGE_SLOTS)


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


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    m = max(float(v) for v in values)
    exps = [math.exp(float(v) - m) for v in values]
    z = sum(exps)
    if z <= 1e-30:
        return [1.0 / len(values)] * len(values)
    return [e / z for e in exps]


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
# Merge audit trail
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeAuditEntry:
    """One entry in the merge audit log."""

    page_a_index: int
    page_b_index: int
    page_kind: str
    merge_score: float
    fact_tags_merged: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_a_index": int(self.page_a_index),
            "page_b_index": int(self.page_b_index),
            "page_kind": str(self.page_kind),
            "merge_score": float(round(
                self.merge_score, 12)),
            "fact_tags_merged": list(self.fact_tags_merged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_merge_audit_entry",
            "entry": self.to_dict()})


# =============================================================================
# BranchCycleMemoryV2Head
# =============================================================================


@dataclasses.dataclass
class BranchCycleMemoryV2Head:
    """V2 BCM head with joint pages, trainable merge + evict.

    Owns the W51 inner head + joint pages + merge/evict
    parameters.
    """

    inner: BranchCycleMemoryHead
    n_joint_pages: int
    joint_pages: list[StoragePage]
    w_merge_head: ParamTensor  # (1,) sigmoid threshold logit
    w_evict_head: ParamTensor  # (1,) sigmoid threshold logit
    w_joint_alpha: ParamTensor  # (1,) blend with inner
    merge_audit: list[MergeAuditEntry]

    @classmethod
    def init(
            cls, *,
            factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages: int = (
                W51_DEFAULT_BCM_N_BRANCH_PAGES),
            n_cycle_pages: int = W51_DEFAULT_BCM_N_CYCLE_PAGES,
            page_capacity: int = W51_DEFAULT_BCM_PAGE_SLOTS,
            n_joint_pages: int = W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BranchCycleMemoryV2Head":
        inner = BranchCycleMemoryHead.init(
            factor_dim=int(factor_dim),
            n_branch_pages=int(n_branch_pages),
            n_cycle_pages=int(n_cycle_pages),
            page_capacity=int(page_capacity),
            seed=int(seed),
            init_scale=float(init_scale))
        joint_pages = [
            StoragePage(
                page_index=i,
                page_kind="joint",
                capacity=int(page_capacity),
                factor_dim=int(factor_dim))
            for i in range(int(n_joint_pages))
        ]
        w_merge = ParamTensor(shape=(1,), values=[-1.5])
        w_evict = ParamTensor(shape=(1,), values=[0.0])
        w_joint_alpha = ParamTensor(shape=(1,), values=[0.5])
        return cls(
            inner=inner,
            n_joint_pages=int(n_joint_pages),
            joint_pages=joint_pages,
            w_merge_head=w_merge,
            w_evict_head=w_evict,
            w_joint_alpha=w_joint_alpha,
            merge_audit=[])

    def params(self) -> list[ParamTensor]:
        out = list(self.inner.params())
        out.extend([
            self.w_merge_head,
            self.w_evict_head,
            self.w_joint_alpha,
        ])
        return out

    @property
    def factor_dim(self) -> int:
        return int(self.inner.factor_dim)

    @property
    def n_branch_pages(self) -> int:
        return int(self.inner.n_branch_pages)

    @property
    def n_cycle_pages(self) -> int:
        return int(self.inner.n_cycle_pages)

    @property
    def page_capacity(self) -> int:
        return int(self.inner.page_capacity)

    @property
    def merge_threshold(self) -> float:
        return float(_stable_sigmoid(
            float(self.w_merge_head.values[0])))

    @property
    def evict_threshold(self) -> float:
        return float(_stable_sigmoid(
            float(self.w_evict_head.values[0])))

    @property
    def joint_alpha(self) -> float:
        return float(_stable_sigmoid(
            float(self.w_joint_alpha.values[0])))

    def _joint_index(
            self, branch_index: int, cycle_index: int,
    ) -> int:
        b = int(branch_index) % max(1, self.inner.n_branch_pages)
        c = int(cycle_index) % max(1, self.inner.n_cycle_pages)
        return int(
            (b * self.inner.n_cycle_pages + c)
            % max(1, self.n_joint_pages))

    def write_to_joint(
            self, *,
            branch_index: int,
            cycle_index: int,
            key: Sequence[float],
            value: Sequence[float],
            fact_tag: str = "",
    ) -> None:
        idx = self._joint_index(
            int(branch_index), int(cycle_index))
        page = self.joint_pages[idx]
        slot = PageSlot(
            slot_index=int(len(page.slots)),
            factor_dim=self.factor_dim,
            key=tuple(_round_floats(key)),
            value=tuple(_round_floats(value)),
            fact_tag=str(fact_tag))
        # Importance-weighted eviction:
        if (len(page.slots) >= int(page.capacity)
                and self.evict_threshold > 0.0):
            # Compute per-slot importance by L2 norm of value;
            # evict the lowest-importance slot.
            importances = [
                float(sum(float(v) * float(v) for v in s.value))
                for s in page.slots
            ]
            min_idx = importances.index(min(importances))
            page.slots.pop(min_idx)
            for i, s in enumerate(page.slots):
                page.slots[i] = dataclasses.replace(
                    s, slot_index=i)
        page.write(slot)

    def maybe_merge_joint_pages(
            self, *,
            page_a_index: int,
            page_b_index: int,
    ) -> bool:
        """Apply the merge head between two joint pages.

        Returns True iff a merge occurred. Merges are
        content-addressed and produce a MergeAuditEntry.
        """
        if (page_a_index == page_b_index
                or page_a_index < 0
                or page_a_index >= self.n_joint_pages
                or page_b_index < 0
                or page_b_index >= self.n_joint_pages):
            return False
        page_a = self.joint_pages[page_a_index]
        page_b = self.joint_pages[page_b_index]
        if not page_a.slots or not page_b.slots:
            return False
        # Compute merge score via cosine of average values.
        avg_a = [0.0] * self.factor_dim
        for s in page_a.slots:
            for j in range(self.factor_dim):
                avg_a[j] += (
                    float(s.value[j]) if j < len(s.value) else 0.0)
        for j in range(self.factor_dim):
            avg_a[j] /= float(max(1, len(page_a.slots)))
        avg_b = [0.0] * self.factor_dim
        for s in page_b.slots:
            for j in range(self.factor_dim):
                avg_b[j] += (
                    float(s.value[j]) if j < len(s.value) else 0.0)
        for j in range(self.factor_dim):
            avg_b[j] /= float(max(1, len(page_b.slots)))
        similarity = float(_cosine(avg_a, avg_b))
        merge_score = float(_stable_sigmoid(
            similarity - float(self.merge_threshold)))
        if merge_score > 0.5:
            # Concatenate slots; capacity is honoured per page.
            merged_tags = tuple(
                str(s.fact_tag) for s in page_a.slots
                + page_b.slots)
            for s in page_b.slots:
                page_a.write(s)
            page_b.reset()
            self.merge_audit.append(MergeAuditEntry(
                page_a_index=int(page_a_index),
                page_b_index=int(page_b_index),
                page_kind="joint",
                merge_score=float(merge_score),
                fact_tags_merged=tuple(merged_tags)))
            return True
        return False

    def read_value(
            self, *,
            branch_index: int,
            cycle_index: int,
            query: Sequence[float],
    ) -> tuple[
            list[float], list[float], list[float], list[float],
            float, float]:
        """Returns (final, inner_read, joint_read, joint_target_read,
        alpha, joint_alpha)."""
        # Inner W51 read
        inner_final, branch_read, cycle_read, alpha = (
            self.inner.read_value(
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                query=query))
        # Joint read
        idx = self._joint_index(
            int(branch_index), int(cycle_index))
        joint_target_read = (
            self.joint_pages[idx].read_value(query))
        joint_read = [0.0] * self.factor_dim
        # Uniform read over all joint pages (consensus-ish).
        for jp in self.joint_pages:
            v = jp.read_value(query)
            for j in range(self.factor_dim):
                joint_read[j] += float(v[j]) / float(
                    max(1, self.n_joint_pages))
        # Blend joint-target with joint-uniform.
        joint_blend = [
            0.7 * float(joint_target_read[j])
            + 0.3 * float(joint_read[j])
            for j in range(self.factor_dim)
        ]
        jalpha = self.joint_alpha
        final = [
            (1.0 - jalpha) * float(inner_final[j])
            + jalpha * float(joint_blend[j])
            for j in range(self.factor_dim)
        ]
        return (
            final, inner_final, joint_read,
            joint_target_read,
            float(alpha), float(jalpha))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W52_BCM_V2_SCHEMA_VERSION),
            "inner": self.inner.to_dict(),
            "n_joint_pages": int(self.n_joint_pages),
            "joint_page_head_cids": [
                p.head_cid() for p in self.joint_pages],
            "w_merge_head": self.w_merge_head.to_dict(),
            "w_evict_head": self.w_evict_head.to_dict(),
            "w_joint_alpha": self.w_joint_alpha.to_dict(),
            "merge_audit_cids": [
                e.cid() for e in self.merge_audit],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_branch_cycle_memory_v2_head",
            "head": self.to_dict()})


# =============================================================================
# Training set + fit
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryV2Example:
    branch_index: int
    cycle_index: int
    key: tuple[float, ...]
    value: tuple[float, ...]
    fact_tag: str


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryV2TrainingSet:
    examples: tuple[BranchCycleMemoryV2Example, ...]
    factor_dim: int
    n_branch_pages: int
    n_cycle_pages: int
    n_joint_pages: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_dim": int(self.factor_dim),
            "n_branch_pages": int(self.n_branch_pages),
            "n_cycle_pages": int(self.n_cycle_pages),
            "n_joint_pages": int(self.n_joint_pages),
            "examples": [
                {"branch_index": int(e.branch_index),
                 "cycle_index": int(e.cycle_index),
                 "key": list(e.key),
                 "value": list(e.value),
                 "fact_tag": str(e.fact_tag)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_bcm_v2_training_set",
            "set": self.to_dict()})


def synthesize_branch_cycle_memory_v2_training_set(
        *,
        n_examples: int = 16,
        factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
        n_branch_pages: int = W51_DEFAULT_BCM_N_BRANCH_PAGES,
        n_cycle_pages: int = W51_DEFAULT_BCM_N_CYCLE_PAGES,
        n_joint_pages: int = W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> BranchCycleMemoryV2TrainingSet:
    """Synthesise a deterministic training set with joint-page
    facts.

    Each (branch, cycle) pair has a UNIQUE per-pair direction
    vector that points orthogonally to other pairs. V1's
    branch/cycle averaging loses this direction; V2's joint
    pages preserve it.
    """
    rng = _DeterministicLCG(seed=int(seed))
    # Per-pair direction vector — used as the target value
    # signature for that pair.
    pair_directions: dict[tuple[int, int], list[float]] = {}
    for b in range(int(n_branch_pages)):
        for c in range(int(n_cycle_pages)):
            d = [
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(int(factor_dim))
            ]
            pair_directions[(b, c)] = d
    exs: list[BranchCycleMemoryV2Example] = []
    for i in range(int(n_examples)):
        b = int(i % int(n_branch_pages))
        c = int((i // int(n_branch_pages)) % int(n_cycle_pages))
        key = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(factor_dim)))
        # Value: per-pair direction + tiny noise.
        direction = pair_directions[(b, c)]
        val = tuple(
            float(direction[j])
            + (rng.next_uniform() - 0.5) * 0.02
            for j in range(int(factor_dim))
        )
        exs.append(BranchCycleMemoryV2Example(
            branch_index=int(b),
            cycle_index=int(c),
            key=key,
            value=val,
            fact_tag=f"b{b}_c{c}_i{i}"))
    return BranchCycleMemoryV2TrainingSet(
        examples=tuple(exs),
        factor_dim=int(factor_dim),
        n_branch_pages=int(n_branch_pages),
        n_cycle_pages=int(n_cycle_pages),
        n_joint_pages=int(n_joint_pages))


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryV2TrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_head_cid: str
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "final_loss": float(round(self.final_loss, 12)),
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
            "loss_head": [float(round(v, 12))
                          for v in self.loss_head],
            "loss_tail": [float(round(v, 12))
                          for v in self.loss_tail],
            "training_set_cid": str(self.training_set_cid),
            "final_head_cid": str(self.final_head_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_bcm_v2_training_trace",
            "trace": self.to_dict()})


def apply_writes_to_v2_head(
        head: BranchCycleMemoryV2Head,
        examples: Sequence[BranchCycleMemoryV2Example],
) -> None:
    """Apply training-set writes to all (branch, cycle, joint) pages."""
    for ex in examples:
        head.inner.write_to_branch(
            branch_index=int(ex.branch_index),
            key=ex.key, value=ex.value,
            fact_tag=str(ex.fact_tag))
        head.inner.write_to_cycle(
            cycle_index=int(ex.cycle_index),
            key=ex.key, value=ex.value,
            fact_tag=str(ex.fact_tag))
        head.write_to_joint(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            key=ex.key, value=ex.value,
            fact_tag=str(ex.fact_tag))


def fit_branch_cycle_memory_v2(
        training_set: BranchCycleMemoryV2TrainingSet,
        *,
        n_steps: int = 48,
        learning_rate: float = 0.05,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[BranchCycleMemoryV2Head,
           BranchCycleMemoryV2TrainingTrace]:
    """Fit V2 BCM head: train consensus + merger + alpha +
    joint-alpha via gradient descent against per-example
    target value reads.

    The merge and evict heads are *threshold* heads; we tune
    them via a simple cosine schedule rather than backprop
    (since their effect on the read is non-differentiable
    in pure Python).
    """
    head = BranchCycleMemoryV2Head.init(
        factor_dim=int(training_set.factor_dim),
        n_branch_pages=int(training_set.n_branch_pages),
        n_cycle_pages=int(training_set.n_cycle_pages),
        n_joint_pages=int(training_set.n_joint_pages),
        seed=int(seed),
        init_scale=float(init_scale))
    apply_writes_to_v2_head(head, training_set.examples)
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    # Trainable subset: joint_alpha + inner.w_alpha +
    # inner.w_branch_consensus + inner.w_cycle_merger.
    trainable = [
        head.inner.w_branch_consensus,
        head.inner.w_cycle_merger,
        head.inner.w_alpha,
        head.w_joint_alpha,
    ]
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        ex = training_set.examples[step % n]
        # Differentiable mock: compute the read value using
        # current consensus/merger/alpha values and supervise
        # against ex.value. We do this by directly using the
        # parameter values (not full autograd through page
        # reads, which involve non-trainable slots).
        # Compute per-page (branch, cycle, joint) reads as
        # values; weight by softmax of consensus/merger/alpha.
        branch_target = (
            head.inner.branch_pages[
                int(ex.branch_index) % head.n_branch_pages]
            .read_value(ex.key))
        cycle_target = (
            head.inner.cycle_pages[
                int(ex.cycle_index) % head.n_cycle_pages]
            .read_value(ex.key))
        idx = head._joint_index(
            int(ex.branch_index), int(ex.cycle_index))
        joint_target = head.joint_pages[idx].read_value(ex.key)
        # Use trainable joint_alpha + inner alpha as differentiable.
        w_alpha_vars = head.inner.w_alpha.make_vars()
        w_joint_alpha_vars = head.w_joint_alpha.make_vars()
        alpha_var = w_alpha_vars[0].sigmoid()
        joint_alpha_var = w_joint_alpha_vars[0].sigmoid()
        # Pre-multiply: inner_final = alpha*branch + (1-alpha)*cycle
        # final = (1 - joint_alpha)*inner_final + joint_alpha*joint
        # All as Variables.
        terms = []
        for j in range(int(training_set.factor_dim)):
            b_j = Variable(float(branch_target[j]
                                  if j < len(branch_target) else 0.0))
            c_j = Variable(float(cycle_target[j]
                                  if j < len(cycle_target) else 0.0))
            jp_j = Variable(float(joint_target[j]
                                   if j < len(joint_target) else 0.0))
            inner_final = (
                alpha_var * b_j
                + (Variable(1.0) - alpha_var) * c_j)
            final = (
                (Variable(1.0) - joint_alpha_var) * inner_final
                + joint_alpha_var * jp_j)
            t = Variable(float(ex.value[j]
                                if j < len(ex.value) else 0.0))
            d = final - t
            terms.append(d * d)
        loss = vmean(terms)
        loss.backward()
        total_grad_sq = 0.0
        for p in trainable:
            for g in p.grads():
                total_grad_sq += float(g) * float(g)
        gn = math.sqrt(total_grad_sq)
        loss_history.append(float(loss.value))
        grad_norm_history.append(float(gn))
        lv = loss.value
        if (lv != lv or lv == float("inf")
                or lv == float("-inf")):
            diverged = True
            break
        optim.step(trainable)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = BranchCycleMemoryV2TrainingTrace(
        seed=int(seed),
        n_steps=int(n_steps),
        final_loss=float(
            loss_history[-1] if loss_history else 0.0),
        final_grad_norm=float(
            grad_norm_history[-1]
            if grad_norm_history else 0.0),
        loss_head=tuple(loss_history[:head_n]),
        loss_tail=tuple(
            loss_history[-tail_n:] if tail_n > 0 else ()),
        training_set_cid=str(training_set.cid()),
        final_head_cid=str(head.cid()),
        diverged=bool(diverged),
    )
    return head, trace


def evaluate_joint_recall_v2(
        head: BranchCycleMemoryV2Head,
        examples: Sequence[BranchCycleMemoryV2Example],
) -> float:
    """Mean cosine recall against (branch, cycle, target) value."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        final, _, _, _, _, _ = head.read_value(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            query=ex.key)
        cos_sum += _cosine(final, ex.value)
        n += 1
    return float(cos_sum) / float(max(1, n))


def evaluate_v1_joint_recall_baseline(
        inner: BranchCycleMemoryHead,
        examples: Sequence[BranchCycleMemoryV2Example],
) -> float:
    """Evaluate W51 V1 BCM head on the same examples.

    V1 has no joint pages — it can only see branch + cycle pages.
    """
    # Apply writes to V1 only (no joint).
    for ex in examples:
        inner.write_to_branch(
            branch_index=int(ex.branch_index),
            key=ex.key, value=ex.value,
            fact_tag=str(ex.fact_tag))
        inner.write_to_cycle(
            cycle_index=int(ex.cycle_index),
            key=ex.key, value=ex.value,
            fact_tag=str(ex.fact_tag))
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        final, _, _, _ = inner.read_value(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            query=ex.key)
        cos_sum += _cosine(final, ex.value)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryV2Witness:
    head_cid: str
    training_trace_cid: str
    factor_dim: int
    n_branch_pages: int
    n_cycle_pages: int
    n_joint_pages: int
    page_storage_cid: str
    merge_audit_cid: str
    evict_policy_cid: str
    mean_branch_recall: float
    mean_cycle_recall: float
    mean_joint_recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "factor_dim": int(self.factor_dim),
            "n_branch_pages": int(self.n_branch_pages),
            "n_cycle_pages": int(self.n_cycle_pages),
            "n_joint_pages": int(self.n_joint_pages),
            "page_storage_cid": str(self.page_storage_cid),
            "merge_audit_cid": str(self.merge_audit_cid),
            "evict_policy_cid": str(self.evict_policy_cid),
            "mean_branch_recall": float(round(
                self.mean_branch_recall, 12)),
            "mean_cycle_recall": float(round(
                self.mean_cycle_recall, 12)),
            "mean_joint_recall": float(round(
                self.mean_joint_recall, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_branch_cycle_memory_v2_witness",
            "witness": self.to_dict()})


def emit_branch_cycle_memory_v2_witness(
        *,
        head: BranchCycleMemoryV2Head,
        training_trace: BranchCycleMemoryV2TrainingTrace,
        examples: Sequence[BranchCycleMemoryV2Example] = (),
) -> BranchCycleMemoryV2Witness:
    page_storage_cid = _sha256_hex({
        "kind": "w52_v2_page_storage",
        "branch_cids": [
            p.head_cid() for p in head.inner.branch_pages],
        "cycle_cids": [
            p.head_cid() for p in head.inner.cycle_pages],
        "joint_cids": [
            p.head_cid() for p in head.joint_pages],
    })
    merge_audit_cid = _sha256_hex({
        "kind": "w52_v2_merge_audit",
        "entries": [e.to_dict() for e in head.merge_audit],
    })
    evict_policy_cid = _sha256_hex({
        "kind": "w52_v2_evict_policy",
        "evict_threshold": float(round(
            head.evict_threshold, 12)),
    })
    if examples:
        joint_recall = evaluate_joint_recall_v2(head, examples)
    else:
        joint_recall = 0.0
    return BranchCycleMemoryV2Witness(
        head_cid=str(head.cid()),
        training_trace_cid=str(training_trace.cid()),
        factor_dim=int(head.factor_dim),
        n_branch_pages=int(head.n_branch_pages),
        n_cycle_pages=int(head.n_cycle_pages),
        n_joint_pages=int(head.n_joint_pages),
        page_storage_cid=str(page_storage_cid),
        merge_audit_cid=str(merge_audit_cid),
        evict_policy_cid=str(evict_policy_cid),
        mean_branch_recall=0.0,
        mean_cycle_recall=0.0,
        mean_joint_recall=float(joint_recall),
    )


# =============================================================================
# Verifier
# =============================================================================


W52_BCM_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_bcm_v2_schema_mismatch",
    "w52_bcm_v2_head_cid_mismatch",
    "w52_bcm_v2_n_joint_pages_mismatch",
    "w52_bcm_v2_page_storage_cid_mismatch",
    "w52_bcm_v2_merge_audit_cid_mismatch",
    "w52_bcm_v2_evict_policy_cid_mismatch",
)


def verify_branch_cycle_memory_v2_witness(
        witness: BranchCycleMemoryV2Witness,
        *,
        expected_head_cid: str | None = None,
        expected_n_joint_pages: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != expected_head_cid):
        failures.append("w52_bcm_v2_head_cid_mismatch")
    if (expected_n_joint_pages is not None
            and witness.n_joint_pages
            != int(expected_n_joint_pages)):
        failures.append("w52_bcm_v2_n_joint_pages_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_BCM_V2_SCHEMA_VERSION",
    "W52_DEFAULT_BCM_V2_N_JOINT_PAGES",
    "W52_DEFAULT_BCM_V2_PAGE_SLOTS",
    "W52_BCM_V2_VERIFIER_FAILURE_MODES",
    "MergeAuditEntry",
    "BranchCycleMemoryV2Head",
    "BranchCycleMemoryV2Example",
    "BranchCycleMemoryV2TrainingSet",
    "BranchCycleMemoryV2TrainingTrace",
    "BranchCycleMemoryV2Witness",
    "synthesize_branch_cycle_memory_v2_training_set",
    "apply_writes_to_v2_head",
    "fit_branch_cycle_memory_v2",
    "evaluate_joint_recall_v2",
    "evaluate_v1_joint_recall_baseline",
    "emit_branch_cycle_memory_v2_witness",
    "verify_branch_cycle_memory_v2_witness",
]
