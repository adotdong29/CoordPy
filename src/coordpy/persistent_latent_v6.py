"""W54 M1 — Persistent Latent State V6 (4-layer stack + dual skip-link
   + disagreement-tagged merge head).

Extends W53 V5 with:

* **4 layers** (vs V5's 3): adds one more GRU layer above the V5
  top layer. The new top layer reads from the V5 top state and
  also receives the running EMA of past carriers as a second
  skip-link.
* **Dual persistent skip-link** — instead of V5's single skip
  (turn-0 carrier), V6 maintains BOTH the turn-0 anchor AND a
  running EMA of past carriers, applying both at every step.
  The fixed anchor preserves the initial signal; the EMA gives
  the upper layers continual access to a recency-biased summary.
* **Disagreement-tagged state merge head** — when merging two
  states, V6 emits the merged state *plus* a per-dim disagreement
  vector (|s_a - s_b|) so the audit trail records how far apart
  the parents were.
* **`max_chain_walk_depth = 64`** (vs V5's 32) — V6 can walk back
  through 64 turns of chain history.

V6 inherits V5's pure-Python autograd surface. It does NOT touch
transformer-internal state.

W54-L-V6-DOES-NOT-TOUCH-SUBSTRATE: this module operates over
capsule-layer signals exclusively.
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
from .persistent_latent_v5 import (
    PersistentLatentStateV5,
    PersistentLatentStateV5Chain,
    V5StackedCell,
    W53_DEFAULT_V5_INPUT_DIM,
    W53_DEFAULT_V5_N_LAYERS,
    W53_DEFAULT_V5_STATE_DIM,
    W53_V5_NO_PARENT_STATE,
    _round_floats,
    _stable_sigmoid,
    step_persistent_state_v5,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_PERSISTENT_V6_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v6.v1")

W54_DEFAULT_V6_STATE_DIM: int = W53_DEFAULT_V5_STATE_DIM
W54_DEFAULT_V6_INPUT_DIM: int = W53_DEFAULT_V5_INPUT_DIM
W54_DEFAULT_V6_N_LAYERS: int = 4
W54_DEFAULT_V6_MAX_CHAIN_WALK_DEPTH: int = 64
W54_DEFAULT_V6_EMA_ALPHA: float = 0.5
W54_V6_NO_PARENT_STATE: str = "no_parent_v6_state"


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
# V6 stacked cell (composes V5 + extra layer + EMA carrier)
# =============================================================================


@dataclasses.dataclass
class V6StackedCell:
    """V6 cell: an inner V5 cell plus a top GRU layer and an EMA skip."""

    inner_v5: V5StackedCell
    w_z_top: ParamTensor
    b_z_top: ParamTensor
    w_h_top: ParamTensor
    b_h_top: ParamTensor
    w_ema_skip: ParamTensor
    state_dim: int
    ema_alpha: float

    @classmethod
    def init(
            cls, *,
            state_dim: int = W54_DEFAULT_V6_STATE_DIM,
            input_dim: int = W54_DEFAULT_V6_INPUT_DIM,
            n_layers: int = W54_DEFAULT_V6_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            ema_alpha: float = W54_DEFAULT_V6_EMA_ALPHA,
    ) -> "V6StackedCell":
        inner = V5StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W53_DEFAULT_V5_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 41)
        cat_d = int(state_dim) + int(state_dim)
        w_z = ParamTensor(
            shape=(int(state_dim), int(cat_d)), values=[])
        w_z.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b_z = ParamTensor(
            shape=(int(state_dim),),
            values=[-1.0] * int(state_dim))
        w_h = ParamTensor(
            shape=(int(state_dim), int(cat_d)), values=[])
        w_h.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b_h = ParamTensor(
            shape=(int(state_dim),),
            values=[0.0] * int(state_dim))
        w_ema = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.15
        w_ema.values = vals
        return cls(
            inner_v5=inner,
            w_z_top=w_z, b_z_top=b_z,
            w_h_top=w_h, b_h_top=b_h,
            w_ema_skip=w_ema,
            state_dim=int(state_dim),
            ema_alpha=float(max(0.0, min(1.0, float(ema_alpha)))))

    def _ema_project_value(
            self,
            ema_input: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                sj = float(
                    ema_input[j]
                    if j < len(ema_input) else 0.0)
                s += float(
                    self.w_ema_skip.values[i * sd + j]) * sj
            out[i] = s
        return out

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            anchor_skip: Sequence[float] | None = None,
            ema_skip: Sequence[float] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Compute V6 layered state for one step.

        Layers 0..n_inner-1 are V5 with anchor_skip applied at layer 1.
        Layer n_inner is the new V6 top layer reading
        (state_below + ema_proj).
        """
        v5_layers, v5_gates = self.inner_v5.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v5.n_layers],
            input_x=input_x,
            skip_input=anchor_skip)
        sd = int(self.state_dim)
        top_below = (
            list(v5_layers[-1]) if v5_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v5.n_layers])
            if self.inner_v5.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        proj = (
            self._ema_project_value(ema_skip)
            if ema_skip is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(proj[i] if i < len(proj) else 0.0)
            for i in range(sd)
        ]
        cat = [
            float(prev_top[i] if i < len(prev_top) else 0.0)
            for i in range(sd)
        ] + [
            float(layer_input[j] if j < len(layer_input) else 0.0)
            for j in range(sd)
        ]
        cat_d = 2 * sd
        wz = self.w_z_top.values
        bz = self.b_z_top.values
        wh = self.w_h_top.values
        bh = self.b_h_top.values
        z = [0.0] * sd
        h = [0.0] * sd
        for r in range(sd):
            base = r * cat_d
            sz = 0.0
            sh = 0.0
            for j in range(cat_d):
                cj = float(cat[j])
                sz += float(wz[base + j]) * cj
                sh += float(wh[base + j]) * cj
            sz += float(bz[r])
            sh += float(bh[r])
            z[r] = float(_stable_sigmoid(sz))
            h[r] = math.tanh(sh)
        top_next = [
            (1.0 - z[i]) * float(
                prev_top[i] if i < len(prev_top) else 0.0)
            + z[i] * h[i]
            for i in range(sd)
        ]
        next_layers = list(v5_layers) + [top_next]
        gates = list(v5_gates) + [z]
        return next_layers, gates

    def merge_states_value(
            self,
            state_a: Sequence[float],
            state_b: Sequence[float],
    ) -> tuple[list[float], list[float], list[float]]:
        """Return (merged_state, alpha_per_dim, disagreement_per_dim)."""
        merged, alpha = self.inner_v5.merge_states_value(
            state_a, state_b)
        disagreement = [
            float(abs(float(state_a[i] if i < len(state_a) else 0.0)
                       - float(state_b[i]
                               if i < len(state_b) else 0.0)))
            for i in range(int(self.state_dim))
        ]
        return merged, alpha, disagreement

    @property
    def n_layers(self) -> int:
        return int(self.inner_v5.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_PERSISTENT_V6_SCHEMA_VERSION),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "state_dim": int(self.state_dim),
            "ema_alpha": float(round(self.ema_alpha, 12)),
            "w_z_top": self.w_z_top.to_dict(),
            "b_z_top": self.b_z_top.to_dict(),
            "w_h_top": self.w_h_top.to_dict(),
            "b_h_top": self.b_h_top.to_dict(),
            "w_ema_skip": self.w_ema_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_v6_stacked_cell",
            "cell": self.to_dict()})


# =============================================================================
# V6 PersistentLatentState
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV6:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    ema_carrier: tuple[float, ...]
    anchor_carrier: tuple[float, ...]
    parent_state_cid: str
    second_parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    ema_skip_cid: str
    update_gate_l1_sum: float
    disagreement_l1_sum: float
    is_merge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "branch_id": str(self.branch_id),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "layer_states": [
                list(_round_floats(s))
                for s in self.layer_states],
            "ema_carrier": list(_round_floats(
                self.ema_carrier)),
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "parent_state_cid": str(self.parent_state_cid),
            "second_parent_state_cid": str(
                self.second_parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "ema_skip_cid": str(self.ema_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "disagreement_l1_sum": float(round(
                self.disagreement_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_v6_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV6Chain:
    states: dict[str, PersistentLatentStateV6]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV6Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV6) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV6 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W54_DEFAULT_V6_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV6]:
        out: list[PersistentLatentStateV6] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            parent = self.get(cur.parent_state_cid)
            if parent is None or parent.cid() in seen:
                break
            cur = parent
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_v6_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())
            ],
        })


# =============================================================================
# Step helper
# =============================================================================


def step_persistent_state_v6(
        *,
        cell: V6StackedCell,
        prev_state: PersistentLatentStateV6 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
) -> PersistentLatentStateV6:
    """Advance V6 by one step.

    Uses `anchor_skip` (turn-0 carrier) and `ema_skip` (running EMA
    of past carriers, maintained in the state itself).
    """
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_ema = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W54_V6_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_ema = list(prev_state.ema_carrier)
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    # EMA update: ema_t = alpha * carrier + (1-alpha) * ema_{t-1}.
    alpha = float(cell.ema_alpha)
    ema_next = [
        alpha * float(
            carrier_values[i]
            if i < len(carrier_values) else 0.0)
        + (1.0 - alpha) * float(
            prev_ema[i] if i < len(prev_ema) else 0.0)
        for i in range(sd)
    ]
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        ema_skip=ema_next)
    anchor_cid = _sha256_hex({
        "kind": "w54_v6_anchor_skip",
        "values": _round_floats(anchor),
    })
    ema_cid = _sha256_hex({
        "kind": "w54_v6_ema_skip",
        "values": _round_floats(ema_next),
        "turn_index": int(turn_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g))
            for layer_z in gates for g in layer_z))
    return PersistentLatentStateV6(
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in next_layers),
        ema_carrier=tuple(_round_floats(ema_next)),
        anchor_carrier=tuple(_round_floats(anchor)),
        parent_state_cid=str(parent_cid),
        second_parent_state_cid="",
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        ema_skip_cid=str(ema_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
        disagreement_l1_sum=0.0,
        is_merge=False,
    )


def merge_persistent_states_v6(
        *,
        cell: V6StackedCell,
        state_a: PersistentLatentStateV6,
        state_b: PersistentLatentStateV6,
        merged_branch_id: str,
        turn_index: int | None = None,
        role: str = "",
) -> tuple[PersistentLatentStateV6, list[list[float]]]:
    """Merge two V6 states; return merged state + per-layer disagreement."""
    if state_a.state_dim != state_b.state_dim:
        raise ValueError(
            "merge_persistent_states_v6: state_dim mismatch")
    sd = int(state_a.state_dim)
    n_layers = max(state_a.n_layers, state_b.n_layers)
    merged_layers: list[list[float]] = []
    disagreement_layers: list[list[float]] = []
    for layer in range(n_layers):
        sa = (
            list(state_a.layer_states[layer])
            if layer < len(state_a.layer_states)
            else [0.0] * sd)
        sb = (
            list(state_b.layer_states[layer])
            if layer < len(state_b.layer_states)
            else [0.0] * sd)
        m, _, d = cell.merge_states_value(sa, sb)
        merged_layers.append(m)
        disagreement_layers.append(d)
    merged_ema, _, _ = cell.merge_states_value(
        list(state_a.ema_carrier),
        list(state_b.ema_carrier))
    anchor = list(state_a.anchor_carrier)
    ti = (
        int(turn_index) if turn_index is not None
        else max(int(state_a.turn_index),
                  int(state_b.turn_index)) + 1)
    rl = str(role) if role else str(state_a.role)
    anchor_cid = _sha256_hex({
        "kind": "w54_v6_merge_anchor",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    ema_cid = _sha256_hex({
        "kind": "w54_v6_merge_ema",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    disagreement_total = float(sum(
        float(v) for d in disagreement_layers for v in d))
    merged_state = PersistentLatentStateV6(
        turn_index=int(ti),
        role=str(rl),
        branch_id=str(merged_branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in merged_layers),
        ema_carrier=tuple(_round_floats(merged_ema)),
        anchor_carrier=tuple(_round_floats(anchor)),
        parent_state_cid=str(state_a.cid()),
        second_parent_state_cid=str(state_b.cid()),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        ema_skip_cid=str(ema_cid),
        update_gate_l1_sum=0.0,
        disagreement_l1_sum=float(round(
            disagreement_total, 12)),
        is_merge=True,
    )
    return merged_state, disagreement_layers


# =============================================================================
# Long-horizon recall evaluator
# =============================================================================


def evaluate_v6_long_horizon_recall(
        cell: V6StackedCell,
        sequences: Sequence[Sequence[Sequence[float]]],
        targets: Sequence[Sequence[float]],
) -> float:
    """Score V6 cosine recall over a list of carrier sequences.

    Each sequence walks the cell forward turn-by-turn; we compare
    the top state vs `target`. Returns the mean cosine.
    """
    if not sequences:
        return 0.0
    cos_sum = 0.0
    n = 0
    for seq, tgt in zip(sequences, targets):
        sd = int(cell.state_dim)
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)
        ]
        anchor = (
            list(seq[0])[:sd] if len(seq) > 0
            else [0.0] * sd)
        while len(anchor) < sd:
            anchor.append(0.0)
        ema = [0.0] * sd
        for x in seq:
            ema = [
                cell.ema_alpha * float(
                    x[i] if i < len(x) else 0.0)
                + (1.0 - cell.ema_alpha) * float(
                    ema[i] if i < len(ema) else 0.0)
                for i in range(sd)
            ]
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=x,
                anchor_skip=anchor,
                ema_skip=ema)
        top = (
            layer_states[-1] if layer_states else [0.0] * sd)
        cos_sum += _cosine(top, tgt)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV6Witness:
    state_cid: str
    parent_state_cid: str
    second_parent_state_cid: str
    role: str
    branch_id: str
    turn_index: int
    state_dim: int
    n_layers: int
    cell_cid: str
    anchor_skip_cid: str
    ema_skip_cid: str
    chain_walk_depth: int
    chain_cid: str
    update_gate_l1_sum: float
    disagreement_l1_sum: float
    is_merge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_cid": str(self.state_cid),
            "parent_state_cid": str(self.parent_state_cid),
            "second_parent_state_cid": str(
                self.second_parent_state_cid),
            "role": str(self.role),
            "branch_id": str(self.branch_id),
            "turn_index": int(self.turn_index),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "ema_skip_cid": str(self.ema_skip_cid),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "disagreement_l1_sum": float(round(
                self.disagreement_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_v6_witness",
            "witness": self.to_dict()})


def emit_persistent_v6_witness(
        *,
        state: PersistentLatentStateV6,
        cell: V6StackedCell,
        chain: PersistentLatentStateV6Chain,
        max_walk_depth: int = (
            W54_DEFAULT_V6_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV6Witness:
    walk = chain.walk_from(
        state.cid(), max_depth=int(max_walk_depth))
    return PersistentLatentStateV6Witness(
        state_cid=str(state.cid()),
        parent_state_cid=str(state.parent_state_cid),
        second_parent_state_cid=str(
            state.second_parent_state_cid),
        role=str(state.role),
        branch_id=str(state.branch_id),
        turn_index=int(state.turn_index),
        state_dim=int(state.state_dim),
        n_layers=int(state.n_layers),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(state.anchor_skip_cid),
        ema_skip_cid=str(state.ema_skip_cid),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
        update_gate_l1_sum=float(state.update_gate_l1_sum),
        disagreement_l1_sum=float(state.disagreement_l1_sum),
        is_merge=bool(state.is_merge),
    )


# =============================================================================
# Verifier + compromise helper
# =============================================================================

W54_V6_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_v6_schema_mismatch",
    "w54_v6_state_cid_mismatch",
    "w54_v6_cell_cid_mismatch",
    "w54_v6_chain_walk_depth_below_floor",
    "w54_v6_update_gate_pathology",
    "w54_v6_n_layers_mismatch",
    "w54_v6_anchor_skip_cid_mismatch",
    "w54_v6_ema_skip_cid_mismatch",
    "w54_v6_merge_state_missing_second_parent",
    "w54_v6_disagreement_l1_negative",
)


def verify_persistent_v6_witness(
        witness: PersistentLatentStateV6Witness,
        *,
        expected_state_cid: str | None = None,
        expected_cell_cid: str | None = None,
        expected_n_layers: int | None = None,
        min_chain_walk_depth: int | None = None,
        max_gate_l1_pathology: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_state_cid is not None
            and witness.state_cid != expected_state_cid):
        failures.append("w54_v6_state_cid_mismatch")
    if (expected_cell_cid is not None
            and witness.cell_cid != expected_cell_cid):
        failures.append("w54_v6_cell_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w54_v6_n_layers_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w54_v6_chain_walk_depth_below_floor")
    if (max_gate_l1_pathology is not None
            and witness.update_gate_l1_sum
            > float(max_gate_l1_pathology)):
        failures.append("w54_v6_update_gate_pathology")
    if (witness.is_merge
            and not witness.second_parent_state_cid):
        failures.append(
            "w54_v6_merge_state_missing_second_parent")
    if witness.disagreement_l1_sum < 0.0:
        failures.append("w54_v6_disagreement_l1_negative")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


def fit_persistent_v6(
        *,
        state_dim: int = W54_DEFAULT_V6_STATE_DIM,
        input_dim: int = W54_DEFAULT_V6_INPUT_DIM,
        n_layers: int = W54_DEFAULT_V6_N_LAYERS,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        n_sequences: int = 6,
        sequence_length: int = 16,
        n_steps: int = 64,
        learning_rate: float = 0.04,
        truncate_bptt: int = 4,
        distractor_window: tuple[int, int] | None = None,
        distractor_magnitude: float = 0.5,
) -> tuple["V6StackedCell", Any]:
    """Fit the inner V5 cell on a V5 training set, then wrap
    in a V6 cell. The V6 outer layer + EMA skip-link are
    initialised but not separately trained (the V5 inner is the
    learning-heavy core).
    """
    from .persistent_latent_v5 import (
        fit_persistent_v5,
        synthesize_v5_training_set,
    )
    ts = synthesize_v5_training_set(
        n_sequences=int(n_sequences),
        sequence_length=int(sequence_length),
        state_dim=int(state_dim),
        input_dim=int(input_dim),
        seed=int(seed),
        distractor_window=distractor_window,
        distractor_magnitude=float(distractor_magnitude))
    v5_cell, trace = fit_persistent_v5(
        ts,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        seed=int(seed),
        truncate_bptt=int(truncate_bptt),
        n_layers=max(W53_DEFAULT_V5_N_LAYERS,
                      int(n_layers) - 1))
    # Build a fresh V6 cell with the trained V5 cell substituted.
    v6 = V6StackedCell.init(
        state_dim=int(state_dim),
        input_dim=int(input_dim),
        n_layers=int(n_layers),
        seed=int(seed))
    v6.inner_v5 = v5_cell
    return v6, trace


def forge_v6_carrier_sequences(
        sequences: Sequence[Sequence[Sequence[float]]],
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> list[list[list[float]]]:
    """Forge carriers by replacing the signal turn with random noise."""
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[list[list[float]]] = []
    for seq in sequences:
        new_seq: list[list[float]] = []
        for t, x in enumerate(seq):
            if t == 0:
                new_seq.append([
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(len(x))
                ])
            else:
                new_seq.append(list(x))
        forged.append(new_seq)
    return forged


__all__ = [
    "W54_PERSISTENT_V6_SCHEMA_VERSION",
    "W54_DEFAULT_V6_STATE_DIM",
    "W54_DEFAULT_V6_INPUT_DIM",
    "W54_DEFAULT_V6_N_LAYERS",
    "W54_DEFAULT_V6_MAX_CHAIN_WALK_DEPTH",
    "W54_DEFAULT_V6_EMA_ALPHA",
    "W54_V6_NO_PARENT_STATE",
    "W54_V6_VERIFIER_FAILURE_MODES",
    "V6StackedCell",
    "PersistentLatentStateV6",
    "PersistentLatentStateV6Chain",
    "PersistentLatentStateV6Witness",
    "step_persistent_state_v6",
    "merge_persistent_states_v6",
    "evaluate_v6_long_horizon_recall",
    "emit_persistent_v6_witness",
    "verify_persistent_v6_witness",
    "fit_persistent_v6",
    "forge_v6_carrier_sequences",
]
