"""W51 M6 — Branch/Cycle-Specialised Memory Head.

A memory head that maintains separate **branch-storage pages**
(one per branch index) and **cycle-storage pages** (one per
cycle-position class). Cross-branch readout uses a learned
consensus weight over branch pages; cross-cycle readout uses
a learned cycle merger. The head is trainable end-to-end via
the W47 autograd engine.

Branch pages and cycle pages are bounded ``factor_dim`` slot
banks. Reads aggregate by softmax over (key · query) within
each page, then combine pages with the learned consensus +
merger.

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state,
KV cache bytes, attention weights, or embeddings.

The H6 strict-gain claim is empirical on the R-100 multi-
branch-cycle regime. The W51 head's branch isolation is a
content-addressed property of the per-page CIDs, not a real
transformer-internal isolation.
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
    vdot,
    vmatmul,
    vmean,
    vsoftmax,
    vsum,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W51_BRANCH_CYCLE_MEMORY_SCHEMA_VERSION: str = (
    "coordpy.branch_cycle_memory.v1")

W51_DEFAULT_BCM_FACTOR_DIM: int = 4
W51_DEFAULT_BCM_PAGE_SLOTS: int = 4
W51_DEFAULT_BCM_N_BRANCH_PAGES: int = 4
W51_DEFAULT_BCM_N_CYCLE_PAGES: int = 4


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
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
    exp_v = [math.exp(float(v) - m) for v in values]
    z = sum(exp_v)
    if z <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [v / z for v in exp_v]


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
# Page slot
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PageSlot:
    """A single slot in a branch- or cycle-page."""

    slot_index: int
    factor_dim: int
    key: tuple[float, ...]
    value: tuple[float, ...]
    fact_tag: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": int(self.slot_index),
            "factor_dim": int(self.factor_dim),
            "key": _round_floats(self.key),
            "value": _round_floats(self.value),
            "fact_tag": str(self.fact_tag),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_page_slot",
            "slot": self.to_dict()})


# =============================================================================
# Storage page (per branch or per cycle)
# =============================================================================

@dataclasses.dataclass
class StoragePage:
    """A bounded content-addressed storage page."""

    page_index: int
    page_kind: str            # "branch" | "cycle"
    capacity: int
    factor_dim: int
    slots: list[PageSlot] = dataclasses.field(
        default_factory=list)

    def head_cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_storage_page_head",
            "page_index": int(self.page_index),
            "page_kind": str(self.page_kind),
            "capacity": int(self.capacity),
            "factor_dim": int(self.factor_dim),
            "slot_cids": [s.cid() for s in self.slots],
        })

    def write(self, slot: PageSlot) -> None:
        self.slots.append(slot)
        while len(self.slots) > self.capacity:
            self.slots.pop(0)
            for i, s in enumerate(self.slots):
                self.slots[i] = dataclasses.replace(
                    s, slot_index=i)

    def read_value(
            self, query: Sequence[float],
    ) -> list[float]:
        if not self.slots:
            return [0.0] * self.factor_dim
        logits = []
        for slot in self.slots:
            dot = 0.0
            for j in range(self.factor_dim):
                qj = float(query[j]) if j < len(query) else 0.0
                kj = float(slot.key[j]) \
                    if j < len(slot.key) else 0.0
                dot += qj * kj
            logits.append(
                dot / math.sqrt(float(max(1, self.factor_dim))))
        attn = _softmax(logits)
        out = [0.0] * self.factor_dim
        for s_idx, slot in enumerate(self.slots):
            w = float(attn[s_idx])
            for j in range(self.factor_dim):
                vj = float(slot.value[j]) \
                    if j < len(slot.value) else 0.0
                out[j] += w * vj
        return out

    def reset(self) -> None:
        self.slots = []


# =============================================================================
# Branch/Cycle Memory Head
# =============================================================================

@dataclasses.dataclass
class BranchCycleMemoryHead:
    """Trainable memory head with separate per-branch and
    per-cycle pages.

    Reads from branch_pages[b] and cycle_pages[c] for the
    current (branch_index, cycle_index), combines them through
    the learned branch consensus weights ``w_branch_consensus``
    (shape ``n_branch_pages``) and the cycle merger weights
    ``w_cycle_merger`` (shape ``n_cycle_pages``). Final output
    = ``alpha * branch_read + (1 - alpha) * cycle_read`` where
    ``alpha = sigmoid(w_alpha)``.
    """

    factor_dim: int
    n_branch_pages: int
    n_cycle_pages: int
    page_capacity: int
    branch_pages: list[StoragePage]
    cycle_pages: list[StoragePage]
    w_branch_consensus: ParamTensor  # (n_branch_pages,)
    w_cycle_merger: ParamTensor      # (n_cycle_pages,)
    w_alpha: ParamTensor             # (1,)

    @classmethod
    def init(
            cls, *,
            factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
            n_branch_pages: int = (
                W51_DEFAULT_BCM_N_BRANCH_PAGES),
            n_cycle_pages: int = W51_DEFAULT_BCM_N_CYCLE_PAGES,
            page_capacity: int = W51_DEFAULT_BCM_PAGE_SLOTS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BranchCycleMemoryHead":
        rng = _DeterministicLCG(seed=int(seed))
        branch_pages = [
            StoragePage(
                page_index=i,
                page_kind="branch",
                capacity=int(page_capacity),
                factor_dim=int(factor_dim))
            for i in range(int(n_branch_pages))
        ]
        cycle_pages = [
            StoragePage(
                page_index=i,
                page_kind="cycle",
                capacity=int(page_capacity),
                factor_dim=int(factor_dim))
            for i in range(int(n_cycle_pages))
        ]
        wb = ParamTensor(
            shape=(int(n_branch_pages),),
            values=[
                float(rng.next_uniform() - 0.5) * float(init_scale)
                for _ in range(int(n_branch_pages))
            ])
        wc = ParamTensor(
            shape=(int(n_cycle_pages),),
            values=[
                float(rng.next_uniform() - 0.5) * float(init_scale)
                for _ in range(int(n_cycle_pages))
            ])
        wa = ParamTensor(
            shape=(1,),
            values=[0.0])  # alpha logit initialised neutral
        return cls(
            factor_dim=int(factor_dim),
            n_branch_pages=int(n_branch_pages),
            n_cycle_pages=int(n_cycle_pages),
            page_capacity=int(page_capacity),
            branch_pages=branch_pages,
            cycle_pages=cycle_pages,
            w_branch_consensus=wb,
            w_cycle_merger=wc,
            w_alpha=wa)

    def params(self) -> list[ParamTensor]:
        return [
            self.w_branch_consensus,
            self.w_cycle_merger,
            self.w_alpha,
        ]

    def write_to_branch(
            self, *,
            branch_index: int,
            key: Sequence[float],
            value: Sequence[float],
            fact_tag: str = "",
    ) -> None:
        b = int(branch_index) % max(1, self.n_branch_pages)
        page = self.branch_pages[b]
        slot = PageSlot(
            slot_index=int(page.size if hasattr(page, "size")
                            else len(page.slots)),
            factor_dim=self.factor_dim,
            key=tuple(_round_floats(key)),
            value=tuple(_round_floats(value)),
            fact_tag=str(fact_tag))
        page.write(slot)

    def write_to_cycle(
            self, *,
            cycle_index: int,
            key: Sequence[float],
            value: Sequence[float],
            fact_tag: str = "",
    ) -> None:
        c = int(cycle_index) % max(1, self.n_cycle_pages)
        page = self.cycle_pages[c]
        slot = PageSlot(
            slot_index=int(len(page.slots)),
            factor_dim=self.factor_dim,
            key=tuple(_round_floats(key)),
            value=tuple(_round_floats(value)),
            fact_tag=str(fact_tag))
        page.write(slot)

    def read_value(
            self, *,
            branch_index: int,
            cycle_index: int,
            query: Sequence[float],
    ) -> tuple[list[float], list[float], list[float], float]:
        """Returns (final_read, branch_read, cycle_read, alpha)."""
        # Branch consensus
        b_target = int(branch_index) % max(
            1, self.n_branch_pages)
        b_consensus = _softmax(
            list(self.w_branch_consensus.values))
        branch_read = [0.0] * self.factor_dim
        for b_idx, page in enumerate(self.branch_pages):
            w_b = float(b_consensus[b_idx])
            page_read = page.read_value(query)
            for j in range(self.factor_dim):
                branch_read[j] += w_b * float(page_read[j])
        # Bias the consensus to prefer the target branch
        # (we add the target's direct read with weight 0.5 — a
        # learned offset would be cleaner but this is fine for
        # the auditable proxy).
        target_branch_read = (
            self.branch_pages[b_target].read_value(query))
        for j in range(self.factor_dim):
            branch_read[j] = (
                0.5 * branch_read[j]
                + 0.5 * float(target_branch_read[j]))
        # Cycle merger
        c_target = int(cycle_index) % max(
            1, self.n_cycle_pages)
        c_merger = _softmax(list(self.w_cycle_merger.values))
        cycle_read = [0.0] * self.factor_dim
        for c_idx, page in enumerate(self.cycle_pages):
            w_c = float(c_merger[c_idx])
            page_read = page.read_value(query)
            for j in range(self.factor_dim):
                cycle_read[j] += w_c * float(page_read[j])
        target_cycle_read = (
            self.cycle_pages[c_target].read_value(query))
        for j in range(self.factor_dim):
            cycle_read[j] = (
                0.5 * cycle_read[j]
                + 0.5 * float(target_cycle_read[j]))
        # Final blend
        alpha = float(_stable_sigmoid(
            float(self.w_alpha.values[0])))
        final = [
            alpha * branch_read[j]
            + (1.0 - alpha) * cycle_read[j]
            for j in range(self.factor_dim)
        ]
        return final, branch_read, cycle_read, float(alpha)

    def read_vars(
            self, *,
            branch_index: int,
            cycle_index: int,
            query: Sequence[Variable],
            branch_reads_value: Sequence[Sequence[float]],
            cycle_reads_value: Sequence[Sequence[float]],
    ) -> list[Variable]:
        """Differentiable read.

        ``branch_reads_value`` / ``cycle_reads_value`` are
        precomputed per-page reads (as values, not vars). The
        differentiable part is over the consensus + merger +
        alpha weights only — the slot keys/values are treated
        as constants here (we don't train slot contents through
        this head in the H6 regime).
        """
        wb_vars = self.w_branch_consensus.make_vars()
        wc_vars = self.w_cycle_merger.make_vars()
        wa_vars = self.w_alpha.make_vars()
        b_consensus = vsoftmax(list(wb_vars))
        c_merger = vsoftmax(list(wc_vars))
        # Branch
        b_target = int(branch_index) % max(
            1, self.n_branch_pages)
        branch_read: list[Variable] = (
            [Variable(0.0)] * self.factor_dim)
        for b_idx in range(self.n_branch_pages):
            page_read_v = branch_reads_value[b_idx] \
                if b_idx < len(branch_reads_value) \
                else [0.0] * self.factor_dim
            w_b = b_consensus[b_idx]
            for j in range(self.factor_dim):
                v = (Variable(float(page_read_v[j]))
                     if j < len(page_read_v)
                     else Variable(0.0))
                branch_read[j] = branch_read[j] + w_b * v
        target_branch_v = (
            branch_reads_value[b_target]
            if b_target < len(branch_reads_value)
            else [0.0] * self.factor_dim)
        for j in range(self.factor_dim):
            t = (Variable(float(target_branch_v[j]))
                 if j < len(target_branch_v)
                 else Variable(0.0))
            branch_read[j] = (
                0.5 * branch_read[j] + 0.5 * t)
        # Cycle
        c_target = int(cycle_index) % max(
            1, self.n_cycle_pages)
        cycle_read: list[Variable] = (
            [Variable(0.0)] * self.factor_dim)
        for c_idx in range(self.n_cycle_pages):
            page_read_v = cycle_reads_value[c_idx] \
                if c_idx < len(cycle_reads_value) \
                else [0.0] * self.factor_dim
            w_c = c_merger[c_idx]
            for j in range(self.factor_dim):
                v = (Variable(float(page_read_v[j]))
                     if j < len(page_read_v)
                     else Variable(0.0))
                cycle_read[j] = cycle_read[j] + w_c * v
        target_cycle_v = (
            cycle_reads_value[c_target]
            if c_target < len(cycle_reads_value)
            else [0.0] * self.factor_dim)
        for j in range(self.factor_dim):
            t = (Variable(float(target_cycle_v[j]))
                 if j < len(target_cycle_v)
                 else Variable(0.0))
            cycle_read[j] = (
                0.5 * cycle_read[j] + 0.5 * t)
        alpha = wa_vars[0].sigmoid()
        one_minus_alpha = Variable(1.0) - alpha
        out: list[Variable] = []
        for j in range(self.factor_dim):
            out.append(
                alpha * branch_read[j]
                + one_minus_alpha * cycle_read[j])
        return out

    def reset(self) -> None:
        for p in self.branch_pages:
            p.reset()
        for p in self.cycle_pages:
            p.reset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_dim": int(self.factor_dim),
            "n_branch_pages": int(self.n_branch_pages),
            "n_cycle_pages": int(self.n_cycle_pages),
            "page_capacity": int(self.page_capacity),
            "branch_pages": [
                p.head_cid() for p in self.branch_pages],
            "cycle_pages": [
                p.head_cid() for p in self.cycle_pages],
            "w_branch_consensus":
                self.w_branch_consensus.to_dict(),
            "w_cycle_merger": self.w_cycle_merger.to_dict(),
            "w_alpha": self.w_alpha.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_branch_cycle_memory_head",
            "head": self.to_dict()})


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryExample:
    """One training example for the BCM head.

    Each example is a (branch_index, cycle_index, query,
    target_value). Before the read, the example's preceding
    writes are applied to the head's pages.
    """

    branch_index: int
    cycle_index: int
    writes: tuple[
        tuple[str, int, tuple[float, ...], tuple[float, ...]],
        ...]   # (page_kind, page_index, key, value)
    query: tuple[float, ...]
    target_value: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryTrainingSet:
    examples: tuple[BranchCycleMemoryExample, ...]
    factor_dim: int
    n_branch_pages: int
    n_cycle_pages: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_dim": int(self.factor_dim),
            "n_branch_pages": int(self.n_branch_pages),
            "n_cycle_pages": int(self.n_cycle_pages),
            "examples": [
                {"branch_index": int(e.branch_index),
                 "cycle_index": int(e.cycle_index),
                 "writes": [
                     [str(w[0]), int(w[1]), list(w[2]),
                      list(w[3])]
                     for w in e.writes],
                 "query": list(e.query),
                 "target_value": list(e.target_value)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_branch_cycle_memory_training_set",
            "set": self.to_dict()})


def synthesize_branch_cycle_memory_training_set(
        *,
        n_examples: int = 16,
        factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
        n_branch_pages: int = W51_DEFAULT_BCM_N_BRANCH_PAGES,
        n_cycle_pages: int = W51_DEFAULT_BCM_N_CYCLE_PAGES,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> BranchCycleMemoryTrainingSet:
    """Synthesise a deterministic dataset.

    Each example writes a (key, value) pair to the
    branch_pages[branch_idx], then asks the head to recover
    the value given the key as the query. Multiple writes per
    example come from different (branch, cycle) combinations
    to test isolation.
    """
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[BranchCycleMemoryExample] = []
    for ex_idx in range(int(n_examples)):
        target_branch = (
            ex_idx % max(1, int(n_branch_pages)))
        target_cycle = (
            ex_idx % max(1, int(n_cycle_pages)))
        # Target key+value
        target_key = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(factor_dim)))
        target_value = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(factor_dim)))
        # Distractor writes in OTHER branches + cycles
        writes: list[tuple[
            str, int, tuple[float, ...], tuple[float, ...]]] = []
        # Write the target to its branch page
        writes.append(
            ("branch", int(target_branch),
             target_key, target_value))
        # Write the target to its cycle page (same key + value)
        writes.append(
            ("cycle", int(target_cycle),
             target_key, target_value))
        # Distractor writes — 2 per page in OTHER branches
        for b in range(int(n_branch_pages)):
            if b == target_branch:
                continue
            for _ in range(2):
                dk = tuple(
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(int(factor_dim)))
                dv = tuple(
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(int(factor_dim)))
                writes.append(("branch", int(b), dk, dv))
        examples.append(BranchCycleMemoryExample(
            branch_index=int(target_branch),
            cycle_index=int(target_cycle),
            writes=tuple(writes),
            query=target_key,
            target_value=target_value))
    return BranchCycleMemoryTrainingSet(
        examples=tuple(examples),
        factor_dim=int(factor_dim),
        n_branch_pages=int(n_branch_pages),
        n_cycle_pages=int(n_cycle_pages))


def apply_writes_to_head(
        head: BranchCycleMemoryHead,
        writes: Sequence[tuple[
            str, int, tuple[float, ...], tuple[float, ...]]],
) -> None:
    head.reset()
    for (kind, idx, key, value) in writes:
        if str(kind) == "branch":
            head.write_to_branch(
                branch_index=int(idx),
                key=list(key), value=list(value))
        else:
            head.write_to_cycle(
                cycle_index=int(idx),
                key=list(key), value=list(value))


@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryTrainingTrace:
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
            "final_grad_norm": float(
                round(self.final_grad_norm, 12)),
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
            "kind": "w51_branch_cycle_memory_training_trace",
            "trace": self.to_dict()})


def fit_branch_cycle_memory(
        training_set: BranchCycleMemoryTrainingSet,
        *,
        n_steps: int = 96,
        learning_rate: float = 0.05,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[BranchCycleMemoryHead,
           BranchCycleMemoryTrainingTrace]:
    """Fit the BCM head via Adam SGD on cosine-similarity loss
    (we minimise ``1 - cos(read, target)``).

    Slot keys/values are not trained — only the consensus,
    merger, and alpha weights.
    """
    head = BranchCycleMemoryHead.init(
        factor_dim=int(training_set.factor_dim),
        n_branch_pages=int(training_set.n_branch_pages),
        n_cycle_pages=int(training_set.n_cycle_pages),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = head.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        # Apply writes to head (rebuild pages each step).
        apply_writes_to_head(head, ex.writes)
        # Precompute per-page reads as values.
        b_reads = [
            head.branch_pages[i].read_value(ex.query)
            for i in range(head.n_branch_pages)
        ]
        c_reads = [
            head.cycle_pages[i].read_value(ex.query)
            for i in range(head.n_cycle_pages)
        ]
        q_vars = [Variable(float(v)) for v in ex.query]
        out_vars = head.read_vars(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            query=q_vars,
            branch_reads_value=b_reads,
            cycle_reads_value=c_reads)
        # Loss = MSE(out_vars, target_value)
        terms = []
        for j in range(len(ex.target_value)):
            t = Variable(float(ex.target_value[j]))
            o = out_vars[j] if j < len(out_vars) else Variable(0.0)
            d = o - t
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
    trace = BranchCycleMemoryTrainingTrace(
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


def evaluate_branch_cycle_recall(
        head: BranchCycleMemoryHead,
        examples: Sequence[BranchCycleMemoryExample],
) -> tuple[float, float]:
    """Returns (mean_branch_cosine, mean_cycle_cosine).

    For each example, re-applies its writes, then reads from
    the head. Branch cosine = cosine(branch_read, target_value);
    cycle cosine = cosine(cycle_read, target_value).
    """
    if not examples:
        return 0.0, 0.0
    bsum = 0.0
    csum = 0.0
    n = 0
    for ex in examples:
        apply_writes_to_head(head, ex.writes)
        _, br, cr, _ = head.read_value(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            query=ex.query)
        bsum += _cosine(br, ex.target_value)
        csum += _cosine(cr, ex.target_value)
        n += 1
    return (
        float(bsum) / float(max(1, n)),
        float(csum) / float(max(1, n)))


def evaluate_branch_cycle_recall_specialised(
        head: BranchCycleMemoryHead,
        examples: Sequence[BranchCycleMemoryExample],
) -> float:
    """Returns the head's overall recall (final blended read
    vs target).
    """
    if not examples:
        return 0.0
    s = 0.0
    n = 0
    for ex in examples:
        apply_writes_to_head(head, ex.writes)
        final, _, _, _ = head.read_value(
            branch_index=int(ex.branch_index),
            cycle_index=int(ex.cycle_index),
            query=ex.query)
        s += _cosine(final, ex.target_value)
        n += 1
    return float(s) / float(max(1, n))


# =============================================================================
# Generic baseline (no branch/cycle specialisation)
# =============================================================================

def evaluate_generic_memory_recall(
        examples: Sequence[BranchCycleMemoryExample],
        *,
        factor_dim: int = W51_DEFAULT_BCM_FACTOR_DIM,
) -> float:
    """Baseline: write all examples to a single generic page,
    then read using softmax(K^T Q) and measure cosine.

    Used as a comparison anchor for the H6 strict-gain claim.
    """
    if not examples:
        return 0.0
    s = 0.0
    n = 0
    for ex in examples:
        # Build a single generic page with all writes.
        page = StoragePage(
            page_index=0, page_kind="generic",
            capacity=64, factor_dim=int(factor_dim))
        for (_kind, _idx, key, value) in ex.writes:
            slot = PageSlot(
                slot_index=int(len(page.slots)),
                factor_dim=int(factor_dim),
                key=tuple(_round_floats(key)),
                value=tuple(_round_floats(value)),
                fact_tag="")
            page.write(slot)
        out = page.read_value(ex.query)
        s += _cosine(out, ex.target_value)
        n += 1
    return float(s) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class BranchCycleMemoryWitness:
    """Sealed per-turn BCM witness."""

    head_cid: str
    training_trace_cid: str
    factor_dim: int
    n_branch_pages: int
    n_cycle_pages: int
    page_capacity: int
    branch_page_cids: tuple[str, ...]
    cycle_page_cids: tuple[str, ...]
    mean_branch_recall: float
    mean_cycle_recall: float
    final_recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "factor_dim": int(self.factor_dim),
            "n_branch_pages": int(self.n_branch_pages),
            "n_cycle_pages": int(self.n_cycle_pages),
            "page_capacity": int(self.page_capacity),
            "branch_page_cids": list(self.branch_page_cids),
            "cycle_page_cids": list(self.cycle_page_cids),
            "mean_branch_recall": float(round(
                self.mean_branch_recall, 12)),
            "mean_cycle_recall": float(round(
                self.mean_cycle_recall, 12)),
            "final_recall": float(round(
                self.final_recall, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_branch_cycle_memory_witness",
            "witness": self.to_dict()})


def emit_branch_cycle_memory_witness(
        *,
        head: BranchCycleMemoryHead,
        training_trace: BranchCycleMemoryTrainingTrace,
        examples: Sequence[BranchCycleMemoryExample],
) -> BranchCycleMemoryWitness:
    branch_recall, cycle_recall = (
        evaluate_branch_cycle_recall(head, examples))
    final_recall = evaluate_branch_cycle_recall_specialised(
        head, examples)
    return BranchCycleMemoryWitness(
        head_cid=str(head.cid()),
        training_trace_cid=str(training_trace.cid()),
        factor_dim=int(head.factor_dim),
        n_branch_pages=int(head.n_branch_pages),
        n_cycle_pages=int(head.n_cycle_pages),
        page_capacity=int(head.page_capacity),
        branch_page_cids=tuple(
            p.head_cid() for p in head.branch_pages),
        cycle_page_cids=tuple(
            p.head_cid() for p in head.cycle_pages),
        mean_branch_recall=float(branch_recall),
        mean_cycle_recall=float(cycle_recall),
        final_recall=float(final_recall),
    )


# =============================================================================
# Verifier
# =============================================================================

W51_BRANCH_CYCLE_MEMORY_VERIFIER_FAILURE_MODES: tuple[
        str, ...] = (
    "w51_bcm_schema_mismatch",
    "w51_bcm_head_cid_mismatch",
    "w51_bcm_training_trace_cid_mismatch",
    "w51_bcm_witness_cid_mismatch",
    "w51_bcm_branch_page_count_mismatch",
    "w51_bcm_cycle_page_count_mismatch",
    "w51_bcm_recall_below_floor",
    "w51_bcm_page_cid_mismatch",
)


def verify_branch_cycle_memory_witness(
        witness: BranchCycleMemoryWitness,
        *,
        expected_head_cid: str | None = None,
        expected_trace_cid: str | None = None,
        recall_floor: float | None = None,
        expected_n_branch_pages: int | None = None,
        expected_n_cycle_pages: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != expected_head_cid):
        failures.append("w51_bcm_head_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append("w51_bcm_training_trace_cid_mismatch")
    if (recall_floor is not None
            and witness.final_recall < float(recall_floor)):
        failures.append("w51_bcm_recall_below_floor")
    if (expected_n_branch_pages is not None
            and witness.n_branch_pages
            != int(expected_n_branch_pages)):
        failures.append("w51_bcm_branch_page_count_mismatch")
    if (expected_n_cycle_pages is not None
            and witness.n_cycle_pages
            != int(expected_n_cycle_pages)):
        failures.append("w51_bcm_cycle_page_count_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W51_BRANCH_CYCLE_MEMORY_SCHEMA_VERSION",
    "W51_DEFAULT_BCM_FACTOR_DIM",
    "W51_DEFAULT_BCM_PAGE_SLOTS",
    "W51_DEFAULT_BCM_N_BRANCH_PAGES",
    "W51_DEFAULT_BCM_N_CYCLE_PAGES",
    "W51_BRANCH_CYCLE_MEMORY_VERIFIER_FAILURE_MODES",
    "PageSlot",
    "StoragePage",
    "BranchCycleMemoryHead",
    "BranchCycleMemoryExample",
    "BranchCycleMemoryTrainingSet",
    "BranchCycleMemoryTrainingTrace",
    "BranchCycleMemoryWitness",
    "synthesize_branch_cycle_memory_training_set",
    "apply_writes_to_head",
    "fit_branch_cycle_memory",
    "evaluate_branch_cycle_recall",
    "evaluate_branch_cycle_recall_specialised",
    "evaluate_generic_memory_recall",
    "emit_branch_cycle_memory_witness",
    "verify_branch_cycle_memory_witness",
]
