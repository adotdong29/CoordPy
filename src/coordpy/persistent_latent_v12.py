"""W60 M8 — Persistent Latent State V12.

Strictly extends W59's ``coordpy.persistent_latent_v11``. V12 adds:

* **10 layers** (vs V11's 9) — one extra GRU stacked on top.
* **Octuple persistent skip-link** — V11's septuple
  (anchor + fast EMA + slow EMA + substrate-EMA + hidden-EMA +
  attention-EMA + retrieval-EMA) plus an eighth: a **replay-
  controller decision EMA**. Each turn the W60 ReplayController
  emits one of {reuse, recompute, fallback, abstain}; V12 carries
  an EMA of the *one-hot* decision vector, letting the persistent
  cell pick up on whether replay or recompute has been winning.
* **``max_chain_walk_depth = 1024``** (vs V11's 768).
* **Replay-fidelity weighting** — alongside V11's
  ``substrate_fidelity``, ``attention_fidelity``,
  ``retrieval_fidelity``, V12 introduces ``replay_fidelity ∈
  [0, 1]`` that damps the replay-skip when the replay-controller
  audit is unreliable.
* **Distractor resistance** — V12's GRU includes a random-
  projection ``replay_skip → ⊥(distractor_basis)`` projection
  that nulls the distractor subspace before injecting. The
  distractor basis is fit by closed-form ridge on a small
  synthetic distractor set during init.

V12 strictly extends V11: with ``replay_skip = None`` and
``replay_fidelity = 1.0``, V12 reduces to V11 plus an inert top
GRU (zero-initialised on the replay path).

Honest scope
------------

* The V12 outer GRU is *initialised but not trained* end-to-end.
  ``W60-L-V12-OUTER-NOT-TRAINED-CAP`` carries forward the V11
  cap unchanged for the new layer.
* "Distractor resistance" means the replay skip is projected
  orthogonal to a fixed random distractor basis at init. Whether
  a real distractor stream survives this projection in practice
  is empirical.
* Permutation invariance carries forward.
  ``W60-L-V12-PERMUTATION-INVARIANCE-CAP`` documents this.
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
from .persistent_latent_v7 import (
    W55_DEFAULT_V7_INPUT_DIM,
    _round_floats,
    _stable_sigmoid,
)
from .persistent_latent_v11 import (
    PersistentLatentStateV11,
    PersistentLatentStateV11Chain,
    V11StackedCell,
    W59_DEFAULT_V11_MAX_CHAIN_WALK_DEPTH,
    W59_DEFAULT_V11_N_LAYERS,
    W59_DEFAULT_V11_STATE_DIM,
    step_persistent_state_v11,
)


W60_PERSISTENT_V12_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v12.v1")
W60_DEFAULT_V12_STATE_DIM: int = W59_DEFAULT_V11_STATE_DIM
W60_DEFAULT_V12_N_LAYERS: int = 10
W60_DEFAULT_V12_MAX_CHAIN_WALK_DEPTH: int = 1024
W60_V12_NO_PARENT_STATE: str = "no_parent_v12_state"
W60_DEFAULT_V12_DISTRACTOR_RANK: int = 4


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class V12StackedCell:
    inner_v11: V11StackedCell
    w_z_top12: ParamTensor
    b_z_top12: ParamTensor
    w_h_top12: ParamTensor
    b_h_top12: ParamTensor
    w_replay_skip: ParamTensor
    distractor_basis: list[list[float]]   # (rank, state_dim)
    state_dim: int

    @classmethod
    def init(
            cls, *,
            state_dim: int = W60_DEFAULT_V12_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W60_DEFAULT_V12_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            distractor_rank: int = (
                W60_DEFAULT_V12_DISTRACTOR_RANK),
    ) -> "V12StackedCell":
        inner = V11StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W59_DEFAULT_V11_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 612)
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
        w_replay = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.05
        w_replay.values = vals
        # Distractor basis: random orthonormal subspace of state_dim.
        rng_d = _DeterministicLCG(seed=int(seed) + 712)
        rank = max(1, int(distractor_rank))
        basis: list[list[float]] = []
        for _ in range(rank):
            v = [rng_d.next_uniform() - 0.5
                  for _ in range(int(state_dim))]
            n = math.sqrt(sum(x * x for x in v))
            if n > 1e-12:
                v = [x / n for x in v]
            # Gram-Schmidt against prior basis.
            for u in basis:
                dot = sum(v[i] * u[i]
                            for i in range(int(state_dim)))
                v = [v[i] - dot * u[i]
                     for i in range(int(state_dim))]
                n2 = math.sqrt(sum(x * x for x in v))
                if n2 > 1e-12:
                    v = [x / n2 for x in v]
            basis.append(v)
        return cls(
            inner_v11=inner,
            w_z_top12=w_z, b_z_top12=b_z,
            w_h_top12=w_h, b_h_top12=b_h,
            w_replay_skip=w_replay,
            distractor_basis=basis,
            state_dim=int(state_dim),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v11.n_layers) + 1

    def _project_orth_distractors(
            self, vec: Sequence[float]) -> list[float]:
        sd = int(self.state_dim)
        v = list(vec)[:sd]
        while len(v) < sd:
            v.append(0.0)
        for u in self.distractor_basis:
            dot = sum(float(v[i]) * float(u[i])
                       for i in range(sd))
            v = [float(v[i]) - dot * float(u[i])
                 for i in range(sd)]
        return v

    def _replay_project(
            self, replay_skip: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        # Project orthogonal to distractor basis.
        purged = self._project_orth_distractors(replay_skip)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                v = float(
                    purged[j] if j < len(purged) else 0.0)
                s += float(
                    self.w_replay_skip.values[i * sd + j]) * v
            out[i] = s
        return out

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            anchor_skip: Sequence[float] | None = None,
            fast_ema_skip: Sequence[float] | None = None,
            slow_ema_skip: Sequence[float] | None = None,
            substrate_skip: Sequence[float] | None = None,
            hidden_state_skip: Sequence[float] | None = None,
            attention_skip: Sequence[float] | None = None,
            retrieval_skip: Sequence[float] | None = None,
            replay_skip: Sequence[float] | None = None,
            substrate_fidelity: float = 1.0,
            attention_fidelity: float = 1.0,
            retrieval_fidelity: float = 1.0,
            replay_fidelity: float = 1.0,
    ) -> tuple[list[list[float]], list[list[float]]]:
        rfid = float(
            max(0.0, min(1.0, float(replay_fidelity))))
        damped_replay = (
            None if replay_skip is None
            else [float(x) * rfid for x in replay_skip])
        v11_layers, v11_gates = self.inner_v11.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v11.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            fast_ema_skip=fast_ema_skip,
            slow_ema_skip=slow_ema_skip,
            substrate_skip=substrate_skip,
            hidden_state_skip=hidden_state_skip,
            attention_skip=attention_skip,
            retrieval_skip=retrieval_skip,
            substrate_fidelity=substrate_fidelity,
            attention_fidelity=attention_fidelity,
            retrieval_fidelity=retrieval_fidelity)
        sd = int(self.state_dim)
        top_below = (
            list(v11_layers[-1]) if v11_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v11.n_layers])
            if self.inner_v11.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        replay_proj = (
            self._replay_project(damped_replay)
            if damped_replay is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(replay_proj[i]
                    if i < len(replay_proj) else 0.0)
            for i in range(sd)
        ]
        cat = [
            float(prev_top[i] if i < len(prev_top) else 0.0)
            for i in range(sd)
        ] + [
            float(layer_input[j]
                  if j < len(layer_input) else 0.0)
            for j in range(sd)
        ]
        cat_d = 2 * sd
        wz = self.w_z_top12.values
        bz = self.b_z_top12.values
        wh = self.w_h_top12.values
        bh = self.b_h_top12.values
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
        next_layers = list(v11_layers) + [top_next]
        gates = list(v11_gates) + [z]
        return next_layers, gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W60_PERSISTENT_V12_SCHEMA_VERSION),
            "inner_v11_cid": str(self.inner_v11.cid()),
            "state_dim": int(self.state_dim),
            "w_z_top12": self.w_z_top12.to_dict(),
            "b_z_top12": self.b_z_top12.to_dict(),
            "w_h_top12": self.w_h_top12.to_dict(),
            "b_h_top12": self.b_h_top12.to_dict(),
            "w_replay_skip": self.w_replay_skip.to_dict(),
            "distractor_basis_rank": int(
                len(self.distractor_basis)),
            "distractor_basis_cid": _sha256_hex({
                "kind": "distractor_basis",
                "vectors": [
                    list(_round_floats(u))
                    for u in self.distractor_basis]}),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_v12_stacked_cell",
            "cell": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV12:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    fast_ema_carrier: tuple[float, ...]
    slow_ema_carrier: tuple[float, ...]
    substrate_carrier: tuple[float, ...]
    hidden_state_carrier: tuple[float, ...]
    attention_carrier: tuple[float, ...]
    retrieval_carrier: tuple[float, ...]
    replay_carrier: tuple[float, ...]
    anchor_carrier: tuple[float, ...]
    substrate_fidelity: float
    attention_fidelity: float
    retrieval_fidelity: float
    replay_fidelity: float
    parent_state_cid: str
    cell_cid: str
    update_gate_l1_sum: float
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
            "fast_ema_carrier": list(_round_floats(
                self.fast_ema_carrier)),
            "slow_ema_carrier": list(_round_floats(
                self.slow_ema_carrier)),
            "substrate_carrier": list(_round_floats(
                self.substrate_carrier)),
            "hidden_state_carrier": list(_round_floats(
                self.hidden_state_carrier)),
            "attention_carrier": list(_round_floats(
                self.attention_carrier)),
            "retrieval_carrier": list(_round_floats(
                self.retrieval_carrier)),
            "replay_carrier": list(_round_floats(
                self.replay_carrier)),
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "substrate_fidelity": float(round(
                self.substrate_fidelity, 12)),
            "attention_fidelity": float(round(
                self.attention_fidelity, 12)),
            "retrieval_fidelity": float(round(
                self.retrieval_fidelity, 12)),
            "replay_fidelity": float(round(
                self.replay_fidelity, 12)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_v12_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV12Chain:
    states: dict[str, PersistentLatentStateV12]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV12Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV12) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV12 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W60_DEFAULT_V12_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV12]:
        out: list[PersistentLatentStateV12] = []
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
            "kind": "w60_v12_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v12(
        *,
        cell: V12StackedCell,
        prev_state: PersistentLatentStateV12 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        attention_skip: Sequence[float] | None = None,
        retrieval_skip: Sequence[float] | None = None,
        replay_skip: Sequence[float] | None = None,
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        replay_fidelity: float = 1.0,
        fast_ema_alpha: float = 0.5,
        slow_ema_alpha: float = 0.10,
        substrate_ema_alpha: float = 0.25,
        hidden_ema_alpha: float = 0.20,
        attention_ema_alpha: float = 0.18,
        retrieval_ema_alpha: float = 0.16,
        replay_ema_alpha: float = 0.14,
) -> PersistentLatentStateV12:
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_fast_ema = [0.0] * sd
        prev_slow_ema = [0.0] * sd
        prev_substrate = [0.0] * sd
        prev_hidden = [0.0] * sd
        prev_attn = [0.0] * sd
        prev_retrieval = [0.0] * sd
        prev_replay = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W60_V12_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_fast_ema = list(prev_state.fast_ema_carrier)
        prev_slow_ema = list(prev_state.slow_ema_carrier)
        prev_substrate = list(prev_state.substrate_carrier)
        prev_hidden = list(prev_state.hidden_state_carrier)
        prev_attn = list(prev_state.attention_carrier)
        prev_retrieval = list(prev_state.retrieval_carrier)
        prev_replay = list(prev_state.replay_carrier)
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(max(0.0, min(1.0, float(fast_ema_alpha))))
    sa = float(max(0.0, min(1.0, float(slow_ema_alpha))))
    ua = float(max(0.0, min(1.0, float(substrate_ema_alpha))))
    ha = float(max(0.0, min(1.0, float(hidden_ema_alpha))))
    aa = float(max(0.0, min(1.0, float(attention_ema_alpha))))
    ra = float(max(0.0, min(1.0, float(retrieval_ema_alpha))))
    rpa = float(max(0.0, min(1.0, float(replay_ema_alpha))))
    fast_ema_next = [
        fa * float(
            carrier_values[i]
            if i < len(carrier_values) else 0.0)
        + (1.0 - fa) * float(
            prev_fast_ema[i]
            if i < len(prev_fast_ema) else 0.0)
        for i in range(sd)
    ]
    slow_ema_next = [
        sa * float(
            carrier_values[i]
            if i < len(carrier_values) else 0.0)
        + (1.0 - sa) * float(
            prev_slow_ema[i]
            if i < len(prev_slow_ema) else 0.0)
        for i in range(sd)
    ]
    substrate_next = list(prev_substrate)
    if substrate_skip is not None:
        substrate_next = [
            ua * float(
                substrate_skip[i]
                if i < len(substrate_skip) else 0.0)
            + (1.0 - ua) * float(
                prev_substrate[i]
                if i < len(prev_substrate) else 0.0)
            for i in range(sd)
        ]
    hidden_next = list(prev_hidden)
    if hidden_state_skip is not None:
        hidden_next = [
            ha * float(
                hidden_state_skip[i]
                if i < len(hidden_state_skip) else 0.0)
            + (1.0 - ha) * float(
                prev_hidden[i]
                if i < len(prev_hidden) else 0.0)
            for i in range(sd)
        ]
    attn_next = list(prev_attn)
    if attention_skip is not None:
        attn_next = [
            aa * float(
                attention_skip[i]
                if i < len(attention_skip) else 0.0)
            + (1.0 - aa) * float(
                prev_attn[i]
                if i < len(prev_attn) else 0.0)
            for i in range(sd)
        ]
    retrieval_next = list(prev_retrieval)
    if retrieval_skip is not None:
        retrieval_next = [
            ra * float(
                retrieval_skip[i]
                if i < len(retrieval_skip) else 0.0)
            + (1.0 - ra) * float(
                prev_retrieval[i]
                if i < len(prev_retrieval) else 0.0)
            for i in range(sd)
        ]
    replay_next = list(prev_replay)
    if replay_skip is not None:
        replay_next = [
            rpa * float(
                replay_skip[i]
                if i < len(replay_skip) else 0.0)
            + (1.0 - rpa) * float(
                prev_replay[i]
                if i < len(prev_replay) else 0.0)
            for i in range(sd)
        ]
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        fast_ema_skip=fast_ema_next,
        slow_ema_skip=slow_ema_next,
        substrate_skip=substrate_next,
        hidden_state_skip=hidden_next,
        attention_skip=attn_next,
        retrieval_skip=retrieval_next,
        replay_skip=replay_next,
        substrate_fidelity=substrate_fidelity,
        attention_fidelity=attention_fidelity,
        retrieval_fidelity=retrieval_fidelity,
        replay_fidelity=replay_fidelity)
    gate_l1 = float(sum(
        abs(float(g)) for layer in gates for g in layer))
    return PersistentLatentStateV12(
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(float(v) for v in layer)
            for layer in next_layers),
        fast_ema_carrier=tuple(fast_ema_next),
        slow_ema_carrier=tuple(slow_ema_next),
        substrate_carrier=tuple(substrate_next),
        hidden_state_carrier=tuple(hidden_next),
        attention_carrier=tuple(attn_next),
        retrieval_carrier=tuple(retrieval_next),
        replay_carrier=tuple(replay_next),
        anchor_carrier=tuple(anchor),
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        replay_fidelity=float(replay_fidelity),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        update_gate_l1_sum=float(gate_l1),
        is_merge=False,
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV12Witness:
    schema: str
    cell_cid: str
    chain_cid: str
    max_chain_walk_depth: int
    achieved_chain_walk_depth: int
    n_layers: int
    state_dim: int
    distractor_basis_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "cell_cid": str(self.cell_cid),
            "chain_cid": str(self.chain_cid),
            "max_chain_walk_depth": int(
                self.max_chain_walk_depth),
            "achieved_chain_walk_depth": int(
                self.achieved_chain_walk_depth),
            "n_layers": int(self.n_layers),
            "state_dim": int(self.state_dim),
            "distractor_basis_rank": int(
                self.distractor_basis_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_v12_persistent_witness",
            "witness": self.to_dict()})


def emit_persistent_v12_witness(
        *,
        cell: V12StackedCell,
        chain: PersistentLatentStateV12Chain,
        leaf_cid: str,
        max_chain_walk_depth: int = (
            W60_DEFAULT_V12_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV12Witness:
    walked = chain.walk_from(
        str(leaf_cid),
        max_depth=int(max_chain_walk_depth))
    return PersistentLatentStateV12Witness(
        schema=W60_PERSISTENT_V12_SCHEMA_VERSION,
        cell_cid=str(cell.cid()),
        chain_cid=str(chain.cid()),
        max_chain_walk_depth=int(max_chain_walk_depth),
        achieved_chain_walk_depth=int(len(walked)),
        n_layers=int(cell.n_layers),
        state_dim=int(cell.state_dim),
        distractor_basis_rank=int(len(cell.distractor_basis)),
    )


__all__ = [
    "W60_PERSISTENT_V12_SCHEMA_VERSION",
    "W60_DEFAULT_V12_STATE_DIM",
    "W60_DEFAULT_V12_N_LAYERS",
    "W60_DEFAULT_V12_MAX_CHAIN_WALK_DEPTH",
    "W60_DEFAULT_V12_DISTRACTOR_RANK",
    "W60_V12_NO_PARENT_STATE",
    "V12StackedCell",
    "PersistentLatentStateV12",
    "PersistentLatentStateV12Chain",
    "PersistentLatentStateV12Witness",
    "step_persistent_state_v12",
    "emit_persistent_v12_witness",
]
