"""W52 M7 — Role-Graph Conditioned Cross-Role Transfer.

A new capsule-native module: cross-role transfer conditioned
on a **role graph** — a DAG of per-edge mixers indexed by
``(src_role, dst_role)``. Each edge carries a learned linear
projection + bias.

Useful when team roles have *direction-dependent* communication
patterns (e.g. planner → researcher is different from
researcher → planner).

Pure-Python only — reuses the W47 ``Variable`` autograd engine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

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
    vmatmul,
    vmean,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_ROLE_GRAPH_SCHEMA_VERSION: str = (
    "coordpy.role_graph_transfer.v1")

W52_DEFAULT_RG_STATE_DIM: int = 8


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
# RoleGraphEdge
# =============================================================================


@dataclasses.dataclass
class RoleGraphEdge:
    """A (src, dst) directional edge with learned projection."""

    src_role: str
    dst_role: str
    state_dim: int
    w: ParamTensor  # (state_dim, state_dim)
    b: ParamTensor  # (state_dim,)

    @classmethod
    def init(
            cls, *,
            src_role: str, dst_role: str,
            state_dim: int = W52_DEFAULT_RG_STATE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "RoleGraphEdge":
        rng = _DeterministicLCG(seed=int(seed))
        w = ParamTensor(
            shape=(int(state_dim), int(state_dim)),
            values=[])
        # Identity-init + small noise.
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 1.0
        for k in range(len(vals)):
            vals[k] += (rng.next_uniform() - 0.5) * 0.05
        w.values = vals
        b = ParamTensor(
            shape=(int(state_dim),),
            values=[0.0] * int(state_dim))
        return cls(
            src_role=str(src_role),
            dst_role=str(dst_role),
            state_dim=int(state_dim),
            w=w, b=b)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b]

    def apply_value(
            self, x: Sequence[float],
    ) -> list[float]:
        sd = self.state_dim
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                xj = float(x[j]) if j < len(x) else 0.0
                s += float(self.w.values[i * sd + j]) * xj
            s += float(self.b.values[i])
            out[i] = s
        return out

    def apply_vars(
            self, x: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        sd = self.state_dim
        rows: list[list[Variable]] = []
        for i in range(sd):
            rows.append(list(w_vars[i * sd:i * sd + sd]))
        pre = vmatmul(rows, list(x))
        return [pre[i] + b_vars[i] for i in range(sd)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_role": str(self.src_role),
            "dst_role": str(self.dst_role),
            "state_dim": int(self.state_dim),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_graph_edge",
            "edge": self.to_dict()})


# =============================================================================
# RoleGraphMixer
# =============================================================================


@dataclasses.dataclass
class RoleGraphMixer:
    """A DAG of per-edge mixers across a role universe."""

    role_universe: tuple[str, ...]
    state_dim: int
    edges: dict[tuple[str, str], RoleGraphEdge]

    @classmethod
    def init(
            cls, *,
            role_universe: Sequence[str] = (
                "r0", "r1", "r2", "r3"),
            state_dim: int = W52_DEFAULT_RG_STATE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            include_self_loops: bool = True,
    ) -> "RoleGraphMixer":
        roles = tuple(sorted({str(r) for r in role_universe}))
        edges: dict[tuple[str, str], RoleGraphEdge] = {}
        rng = _DeterministicLCG(seed=int(seed))
        for src in roles:
            for dst in roles:
                if (not include_self_loops
                        and src == dst):
                    continue
                e = RoleGraphEdge.init(
                    src_role=src, dst_role=dst,
                    state_dim=int(state_dim),
                    seed=int(rng.next_uniform() * (1 << 30)),
                    init_scale=float(init_scale))
                edges[(src, dst)] = e
        return cls(
            role_universe=roles,
            state_dim=int(state_dim),
            edges=edges)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for key in sorted(self.edges.keys()):
            out.extend(self.edges[key].params())
        return out

    def get(
            self, src: str, dst: str,
    ) -> RoleGraphEdge | None:
        return self.edges.get((str(src), str(dst)))

    def project_value(
            self, *,
            src_role: str, dst_role: str,
            x: Sequence[float],
    ) -> list[float]:
        e = self.get(src_role, dst_role)
        if e is None:
            return list(x)[:self.state_dim]
        return e.apply_value(x)

    def equal_weight_project_value(
            self, x: Sequence[float],
    ) -> list[float]:
        """Equal-weight baseline: average over all edges.

        This is the W51 ``CrossRoleMixer``-style behaviour
        when blend coefficients are uniform.
        """
        if not self.edges:
            return [0.0] * self.state_dim
        out = [0.0] * self.state_dim
        n = float(len(self.edges))
        for e in self.edges.values():
            v = e.apply_value(x)
            for i in range(self.state_dim):
                out[i] += float(v[i]) / n
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W52_ROLE_GRAPH_SCHEMA_VERSION),
            "role_universe": list(self.role_universe),
            "state_dim": int(self.state_dim),
            "edges": [
                self.edges[key].to_dict()
                for key in sorted(self.edges.keys())
            ],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_graph_mixer",
            "mixer": self.to_dict()})


def build_unfitted_role_graph_mixer(
        *,
        role_universe: Sequence[str] = (
            "r0", "r1", "r2", "r3"),
        state_dim: int = W52_DEFAULT_RG_STATE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> RoleGraphMixer:
    return RoleGraphMixer.init(
        role_universe=role_universe,
        state_dim=int(state_dim),
        seed=int(seed))


# =============================================================================
# Training set + fit
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RoleGraphExample:
    src_role: str
    dst_role: str
    src_state: tuple[float, ...]
    dst_target: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class RoleGraphTrainingSet:
    examples: tuple[RoleGraphExample, ...]
    role_universe: tuple[str, ...]
    state_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_universe": list(self.role_universe),
            "state_dim": int(self.state_dim),
            "examples": [
                {"src_role": str(e.src_role),
                 "dst_role": str(e.dst_role),
                 "src_state": list(e.src_state),
                 "dst_target": list(e.dst_target)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_graph_training_set",
            "set": self.to_dict()})


def synthesize_role_graph_training_set(
        *,
        role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
        state_dim: int = W52_DEFAULT_RG_STATE_DIM,
        n_examples_per_edge: int = 6,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> RoleGraphTrainingSet:
    """Synthesise a training set with per-edge direction-dependent
    target maps.

    For each ordered (src, dst) pair, a unique linear map
    ``M_(src,dst)`` is sampled. The target = ``M(src_state)``.
    """
    rng = _DeterministicLCG(seed=int(seed))
    roles = tuple(sorted({str(r) for r in role_universe}))
    # Per-edge map.
    edge_maps: dict[tuple[str, str], list[float]] = {}
    for src in roles:
        for dst in roles:
            m = [
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(int(state_dim) * int(state_dim))
            ]
            edge_maps[(src, dst)] = m
    examples: list[RoleGraphExample] = []
    for src in roles:
        for dst in roles:
            mp = edge_maps[(src, dst)]
            for _ in range(int(n_examples_per_edge)):
                s = tuple(
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(int(state_dim)))
                t = [0.0] * int(state_dim)
                for i in range(int(state_dim)):
                    for j in range(int(state_dim)):
                        t[i] += (
                            float(mp[i * int(state_dim) + j])
                            * float(s[j]))
                examples.append(RoleGraphExample(
                    src_role=src, dst_role=dst,
                    src_state=s,
                    dst_target=tuple(t)))
    return RoleGraphTrainingSet(
        examples=tuple(examples),
        role_universe=roles,
        state_dim=int(state_dim))


@dataclasses.dataclass(frozen=True)
class RoleGraphTrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_mixer_cid: str
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
            "final_mixer_cid": str(self.final_mixer_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_graph_training_trace",
            "trace": self.to_dict()})


def fit_role_graph_mixer(
        training_set: RoleGraphTrainingSet,
        *,
        n_steps: int = 144,
        learning_rate: float = 0.05,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[RoleGraphMixer, RoleGraphTrainingTrace]:
    """Fit per-edge projections via Adam SGD on MSE."""
    mixer = RoleGraphMixer.init(
        role_universe=training_set.role_universe,
        state_dim=int(training_set.state_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    params = mixer.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in params:
            p.make_vars()
        ex = training_set.examples[step % n]
        edge = mixer.get(ex.src_role, ex.dst_role)
        if edge is None:
            continue
        x_vars = [Variable(float(v)) for v in ex.src_state]
        pred = edge.apply_vars(x_vars)
        terms = []
        for j in range(len(ex.dst_target)):
            t = Variable(float(ex.dst_target[j]))
            o = pred[j] if j < len(pred) else Variable(0.0)
            d = o - t
            terms.append(d * d)
        loss = vmean(terms)
        loss.backward()
        total_grad_sq = 0.0
        for p in params:
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
        optim.step(params)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = RoleGraphTrainingTrace(
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
        final_mixer_cid=str(mixer.cid()),
        diverged=bool(diverged),
    )
    return mixer, trace


def evaluate_role_graph_accuracy(
        mixer: RoleGraphMixer,
        examples: Sequence[RoleGraphExample],
) -> float:
    """Mean cosine accuracy of per-edge transfer."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        pred = mixer.project_value(
            src_role=ex.src_role,
            dst_role=ex.dst_role,
            x=ex.src_state)
        cos_sum += _cosine(pred, ex.dst_target)
        n += 1
    return float(cos_sum) / float(max(1, n))


def evaluate_equal_weight_accuracy(
        mixer: RoleGraphMixer,
        examples: Sequence[RoleGraphExample],
) -> float:
    """Equal-weight baseline accuracy (averages all edges)."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        pred = mixer.equal_weight_project_value(ex.src_state)
        cos_sum += _cosine(pred, ex.dst_target)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RoleGraphWitness:
    mixer_cid: str
    training_trace_cid: str
    role_universe: tuple[str, ...]
    state_dim: int
    n_edges: int
    per_edge_cids: tuple[str, ...]
    mean_role_graph_accuracy: float
    mean_equal_weight_accuracy: float
    gain_over_equal_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixer_cid": str(self.mixer_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "role_universe": list(self.role_universe),
            "state_dim": int(self.state_dim),
            "n_edges": int(self.n_edges),
            "per_edge_cids": list(self.per_edge_cids),
            "mean_role_graph_accuracy": float(round(
                self.mean_role_graph_accuracy, 12)),
            "mean_equal_weight_accuracy": float(round(
                self.mean_equal_weight_accuracy, 12)),
            "gain_over_equal_weight": float(round(
                self.gain_over_equal_weight, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_role_graph_witness",
            "witness": self.to_dict()})


def emit_role_graph_witness(
        *,
        mixer: RoleGraphMixer,
        training_trace: RoleGraphTrainingTrace,
        examples: Sequence[RoleGraphExample] = (),
) -> RoleGraphWitness:
    rg_acc = (
        evaluate_role_graph_accuracy(mixer, examples)
        if examples else 0.0)
    ew_acc = (
        evaluate_equal_weight_accuracy(mixer, examples)
        if examples else 0.0)
    per_edge_cids = tuple(
        mixer.edges[key].cid()
        for key in sorted(mixer.edges.keys()))
    return RoleGraphWitness(
        mixer_cid=str(mixer.cid()),
        training_trace_cid=str(training_trace.cid()),
        role_universe=tuple(mixer.role_universe),
        state_dim=int(mixer.state_dim),
        n_edges=int(len(mixer.edges)),
        per_edge_cids=per_edge_cids,
        mean_role_graph_accuracy=float(rg_acc),
        mean_equal_weight_accuracy=float(ew_acc),
        gain_over_equal_weight=float(rg_acc - ew_acc),
    )


# =============================================================================
# Forge for the distribution-cap falsifier
# =============================================================================


def forge_role_graph_training_set(
        original: RoleGraphTrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> RoleGraphTrainingSet:
    """Scramble per-edge targets adversarially."""
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[RoleGraphExample] = []
    for ex in original.examples:
        tgt = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.dst_target)))
        forged.append(RoleGraphExample(
            src_role=ex.src_role,
            dst_role=ex.dst_role,
            src_state=ex.src_state,
            dst_target=tgt))
    return RoleGraphTrainingSet(
        examples=tuple(forged),
        role_universe=original.role_universe,
        state_dim=original.state_dim)


# =============================================================================
# Verifier
# =============================================================================


W52_ROLE_GRAPH_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_role_graph_schema_mismatch",
    "w52_role_graph_mixer_cid_mismatch",
    "w52_role_graph_training_trace_cid_mismatch",
    "w52_role_graph_edge_count_mismatch",
    "w52_role_graph_role_universe_mismatch",
)


def verify_role_graph_witness(
        witness: RoleGraphWitness,
        *,
        expected_mixer_cid: str | None = None,
        expected_n_edges: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_mixer_cid is not None
            and witness.mixer_cid != expected_mixer_cid):
        failures.append("w52_role_graph_mixer_cid_mismatch")
    if (expected_n_edges is not None
            and witness.n_edges != int(expected_n_edges)):
        failures.append("w52_role_graph_edge_count_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_ROLE_GRAPH_SCHEMA_VERSION",
    "W52_DEFAULT_RG_STATE_DIM",
    "W52_ROLE_GRAPH_VERIFIER_FAILURE_MODES",
    "RoleGraphEdge",
    "RoleGraphMixer",
    "RoleGraphExample",
    "RoleGraphTrainingSet",
    "RoleGraphTrainingTrace",
    "RoleGraphWitness",
    "build_unfitted_role_graph_mixer",
    "synthesize_role_graph_training_set",
    "fit_role_graph_mixer",
    "evaluate_role_graph_accuracy",
    "evaluate_equal_weight_accuracy",
    "emit_role_graph_witness",
    "forge_role_graph_training_set",
    "verify_role_graph_witness",
]
