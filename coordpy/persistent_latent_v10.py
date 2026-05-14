"""W58 M7 — Persistent Latent State V10.

Strictly extends W57's ``coordpy.persistent_latent_v9``. V10 adds:

* **8 layers** (vs V9's 7) — one extra GRU stacked above V9.
* **Sextuple persistent skip-link** — V9's quintuple skip
  (anchor + fast EMA + slow EMA + substrate-EMA + hidden-EMA)
  plus a sixth: an **attention-pattern-conditioned EMA**. The
  W58 attention-steering bridge V2 produces a per-layer attention
  KL signal; V10 carries an EMA of that signal as its sixth
  skip.
* **``max_chain_walk_depth = 512``** (vs V9's 384).
* **Attention-fidelity weighting** — alongside V9's
  ``substrate_fidelity``, V10 introduces ``attention_fidelity`` in
  ``[0, 1]`` that damps the attention-skip when the attention
  signal is unreliable.

V10 strictly extends V9: with ``attention_skip = None`` and
``attention_fidelity = 1.0``, V10 reduces to V9 plus an inert
top GRU (zero-initialised on the attention path).

Honest scope
------------

* The V10 outer GRU + attention-skip linear are *initialised but
  not trained* end-to-end. ``W58-L-V10-OUTER-NOT-TRAINED-CAP``
  carries forward the V9 cap unchanged.
* Permutation invariance of V9 carriers carries forward. EMA
  carriers smooth out sequence order. ``W58-L-V10-PERMUTATION-
  INVARIANCE-CAP`` documents this.
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
from .persistent_latent_v9 import (
    PersistentLatentStateV9,
    PersistentLatentStateV9Chain,
    V9StackedCell,
    W57_DEFAULT_V9_N_LAYERS,
    W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH,
    W57_DEFAULT_V9_STATE_DIM,
    W57_V9_NO_PARENT_STATE,
)
from .persistent_latent_v7 import (
    W55_DEFAULT_V7_INPUT_DIM,
    _round_floats,
    _stable_sigmoid,
)


W58_PERSISTENT_V10_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v10.v1")
W58_DEFAULT_V10_STATE_DIM: int = W57_DEFAULT_V9_STATE_DIM
W58_DEFAULT_V10_N_LAYERS: int = 8
W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH: int = 512
W58_V10_NO_PARENT_STATE: str = "no_parent_v10_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class V10StackedCell:
    inner_v9: V9StackedCell
    w_z_top10: ParamTensor
    b_z_top10: ParamTensor
    w_h_top10: ParamTensor
    b_h_top10: ParamTensor
    w_attn_skip: ParamTensor
    state_dim: int

    @classmethod
    def init(
            cls, *,
            state_dim: int = W58_DEFAULT_V10_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W58_DEFAULT_V10_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V10StackedCell":
        inner = V9StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W57_DEFAULT_V9_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 311)
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
        w_attn = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        # Diagonal init so attention skip starts informative.
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.08
        w_attn.values = vals
        return cls(
            inner_v9=inner,
            w_z_top10=w_z, b_z_top10=b_z,
            w_h_top10=w_h, b_h_top10=b_h,
            w_attn_skip=w_attn,
            state_dim=int(state_dim),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v9.n_layers) + 1

    def _attn_project(
            self, attn_skip: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                v = float(
                    attn_skip[j]
                    if j < len(attn_skip) else 0.0)
                s += float(
                    self.w_attn_skip.values[i * sd + j]) * v
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
            substrate_fidelity: float = 1.0,
            attention_fidelity: float = 1.0,
    ) -> tuple[list[list[float]], list[list[float]]]:
        afid = float(
            max(0.0, min(1.0, float(attention_fidelity))))
        damped_attn = (
            None if attention_skip is None
            else [float(x) * afid for x in attention_skip])
        v9_layers, v9_gates = self.inner_v9.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v9.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            fast_ema_skip=fast_ema_skip,
            slow_ema_skip=slow_ema_skip,
            substrate_skip=substrate_skip,
            hidden_state_skip=hidden_state_skip,
            substrate_fidelity=substrate_fidelity)
        sd = int(self.state_dim)
        top_below = (
            list(v9_layers[-1]) if v9_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v9.n_layers])
            if self.inner_v9.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        attn_proj = (
            self._attn_project(damped_attn)
            if damped_attn is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(attn_proj[i]
                    if i < len(attn_proj) else 0.0)
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
        wz = self.w_z_top10.values
        bz = self.b_z_top10.values
        wh = self.w_h_top10.values
        bh = self.b_h_top10.values
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
        next_layers = list(v9_layers) + [top_next]
        gates = list(v9_gates) + [z]
        return next_layers, gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W58_PERSISTENT_V10_SCHEMA_VERSION),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "state_dim": int(self.state_dim),
            "w_z_top10": self.w_z_top10.to_dict(),
            "b_z_top10": self.b_z_top10.to_dict(),
            "w_h_top10": self.w_h_top10.to_dict(),
            "b_h_top10": self.b_h_top10.to_dict(),
            "w_attn_skip": self.w_attn_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_v10_stacked_cell",
            "cell": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV10:
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
    anchor_carrier: tuple[float, ...]
    substrate_fidelity: float
    attention_fidelity: float
    parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    substrate_skip_cid: str
    hidden_skip_cid: str
    attention_skip_cid: str
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
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "substrate_fidelity": float(round(
                self.substrate_fidelity, 12)),
            "attention_fidelity": float(round(
                self.attention_fidelity, 12)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
            "hidden_skip_cid": str(self.hidden_skip_cid),
            "attention_skip_cid": str(self.attention_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_v10_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV10Chain:
    states: dict[str, PersistentLatentStateV10]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV10Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV10) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV10 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV10]:
        out: list[PersistentLatentStateV10] = []
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
            "kind": "w58_v10_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v10(
        *,
        cell: V10StackedCell,
        prev_state: PersistentLatentStateV10 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        attention_skip: Sequence[float] | None = None,
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        fast_ema_alpha: float = 0.5,
        slow_ema_alpha: float = 0.10,
        substrate_ema_alpha: float = 0.25,
        hidden_ema_alpha: float = 0.20,
        attention_ema_alpha: float = 0.18,
) -> PersistentLatentStateV10:
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_fast_ema = [0.0] * sd
        prev_slow_ema = [0.0] * sd
        prev_substrate = [0.0] * sd
        prev_hidden = [0.0] * sd
        prev_attn = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W58_V10_NO_PARENT_STATE
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
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(max(0.0, min(1.0, float(fast_ema_alpha))))
    sa = float(max(0.0, min(1.0, float(slow_ema_alpha))))
    ua = float(max(0.0, min(1.0, float(substrate_ema_alpha))))
    ha = float(max(0.0, min(1.0, float(hidden_ema_alpha))))
    aa = float(max(0.0, min(1.0, float(attention_ema_alpha))))
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
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        fast_ema_skip=fast_ema_next,
        slow_ema_skip=slow_ema_next,
        substrate_skip=substrate_next,
        hidden_state_skip=hidden_next,
        attention_skip=attn_next,
        substrate_fidelity=substrate_fidelity,
        attention_fidelity=attention_fidelity)
    anchor_cid = _sha256_hex({
        "kind": "w58_v10_anchor_skip",
        "values": _round_floats(anchor)})
    fast_ema_cid = _sha256_hex({
        "kind": "w58_v10_fast_ema_skip",
        "values": _round_floats(fast_ema_next)})
    slow_ema_cid = _sha256_hex({
        "kind": "w58_v10_slow_ema_skip",
        "values": _round_floats(slow_ema_next)})
    substrate_cid = _sha256_hex({
        "kind": "w58_v10_substrate_skip",
        "values": _round_floats(substrate_next)})
    hidden_cid = _sha256_hex({
        "kind": "w58_v10_hidden_skip",
        "values": _round_floats(hidden_next)})
    attn_cid = _sha256_hex({
        "kind": "w58_v10_attention_skip",
        "values": _round_floats(attn_next)})
    gate_l1 = float(sum(
        abs(float(g)) for layer in gates for g in layer))
    return PersistentLatentStateV10(
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
        anchor_carrier=tuple(anchor),
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        substrate_skip_cid=str(substrate_cid),
        hidden_skip_cid=str(hidden_cid),
        attention_skip_cid=str(attn_cid),
        update_gate_l1_sum=float(gate_l1),
        is_merge=False,
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV10Witness:
    schema: str
    cell_cid: str
    chain_cid: str
    max_chain_walk_depth: int
    achieved_chain_walk_depth: int
    n_layers: int
    state_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "cell_cid": str(self.cell_cid),
            "chain_cid": str(self.chain_cid),
            "max_chain_walk_depth": int(self.max_chain_walk_depth),
            "achieved_chain_walk_depth": int(
                self.achieved_chain_walk_depth),
            "n_layers": int(self.n_layers),
            "state_dim": int(self.state_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_v10_persistent_witness",
            "witness": self.to_dict()})


def emit_persistent_v10_witness(
        *,
        cell: V10StackedCell,
        chain: PersistentLatentStateV10Chain,
        leaf_cid: str,
        max_chain_walk_depth: int = (
            W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV10Witness:
    walked = chain.walk_from(
        str(leaf_cid),
        max_depth=int(max_chain_walk_depth))
    return PersistentLatentStateV10Witness(
        schema=W58_PERSISTENT_V10_SCHEMA_VERSION,
        cell_cid=str(cell.cid()),
        chain_cid=str(chain.cid()),
        max_chain_walk_depth=int(max_chain_walk_depth),
        achieved_chain_walk_depth=int(len(walked)),
        n_layers=int(cell.n_layers),
        state_dim=int(cell.state_dim),
    )


__all__ = [
    "W58_PERSISTENT_V10_SCHEMA_VERSION",
    "W58_DEFAULT_V10_STATE_DIM",
    "W58_DEFAULT_V10_N_LAYERS",
    "W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH",
    "W58_V10_NO_PARENT_STATE",
    "V10StackedCell",
    "PersistentLatentStateV10",
    "PersistentLatentStateV10Chain",
    "PersistentLatentStateV10Witness",
    "step_persistent_state_v10",
    "emit_persistent_v10_witness",
]
