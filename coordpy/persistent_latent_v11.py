"""W59 M7 — Persistent Latent State V11.

Strictly extends W58's ``coordpy.persistent_latent_v10``. V11 adds:

* **9 layers** (vs V10's 8) — one extra GRU stacked above V10.
* **Septuple persistent skip-link** — V10's sextuple
  (anchor + fast EMA + slow EMA + substrate-EMA + hidden-EMA +
  attention-EMA) plus a seventh: a **cache-retrieval-conditioned
  EMA**. The W59 cache controller V2 emits a *retrieval score*
  per turn (the bilinear ``q^T M h_t`` field); V11 carries an
  EMA of that signal.
* **``max_chain_walk_depth = 768``** (vs V10's 512).
* **Retrieval-fidelity weighting** — alongside V10's
  ``substrate_fidelity`` and ``attention_fidelity``, V11
  introduces ``retrieval_fidelity`` ∈ [0,1] that damps the
  retrieval-skip when the retrieval signal is unreliable.

V11 strictly extends V10: with ``retrieval_skip = None`` and
``retrieval_fidelity = 1.0``, V11 reduces to V10 plus an inert
top GRU (zero-initialised on the retrieval path).

Honest scope
------------

* The V11 outer GRU + retrieval-skip projection are *initialised
  but not trained* end-to-end. ``W59-L-V11-OUTER-NOT-TRAINED-CAP``
  carries forward the V10 cap unchanged for the new layer.
* Permutation invariance of V10 carriers carries forward. EMA
  carriers smooth out sequence order. ``W59-L-V11-PERMUTATION-
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
from .persistent_latent_v7 import (
    W55_DEFAULT_V7_INPUT_DIM,
    _round_floats,
    _stable_sigmoid,
)
from .persistent_latent_v10 import (
    PersistentLatentStateV10,
    PersistentLatentStateV10Chain,
    V10StackedCell,
    W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH,
    W58_DEFAULT_V10_N_LAYERS,
    W58_DEFAULT_V10_STATE_DIM,
    W58_V10_NO_PARENT_STATE,
    step_persistent_state_v10,
)


W59_PERSISTENT_V11_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v11.v1")
W59_DEFAULT_V11_STATE_DIM: int = W58_DEFAULT_V10_STATE_DIM
W59_DEFAULT_V11_N_LAYERS: int = 9
W59_DEFAULT_V11_MAX_CHAIN_WALK_DEPTH: int = 768
W59_V11_NO_PARENT_STATE: str = "no_parent_v11_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class V11StackedCell:
    inner_v10: V10StackedCell
    w_z_top11: ParamTensor
    b_z_top11: ParamTensor
    w_h_top11: ParamTensor
    b_h_top11: ParamTensor
    w_retrieval_skip: ParamTensor
    state_dim: int

    @classmethod
    def init(
            cls, *,
            state_dim: int = W59_DEFAULT_V11_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W59_DEFAULT_V11_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V11StackedCell":
        inner = V10StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W58_DEFAULT_V10_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 511)
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
        w_ret = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        # Diagonal init so retrieval skip starts informative.
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.07
        w_ret.values = vals
        return cls(
            inner_v10=inner,
            w_z_top11=w_z, b_z_top11=b_z,
            w_h_top11=w_h, b_h_top11=b_h,
            w_retrieval_skip=w_ret,
            state_dim=int(state_dim),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v10.n_layers) + 1

    def _retrieval_project(
            self, retrieval_skip: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                v = float(
                    retrieval_skip[j]
                    if j < len(retrieval_skip) else 0.0)
                s += float(
                    self.w_retrieval_skip.values[i * sd + j]) * v
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
            substrate_fidelity: float = 1.0,
            attention_fidelity: float = 1.0,
            retrieval_fidelity: float = 1.0,
    ) -> tuple[list[list[float]], list[list[float]]]:
        rfid = float(
            max(0.0, min(1.0, float(retrieval_fidelity))))
        damped_ret = (
            None if retrieval_skip is None
            else [float(x) * rfid for x in retrieval_skip])
        v10_layers, v10_gates = self.inner_v10.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v10.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            fast_ema_skip=fast_ema_skip,
            slow_ema_skip=slow_ema_skip,
            substrate_skip=substrate_skip,
            hidden_state_skip=hidden_state_skip,
            attention_skip=attention_skip,
            substrate_fidelity=substrate_fidelity,
            attention_fidelity=attention_fidelity)
        sd = int(self.state_dim)
        top_below = (
            list(v10_layers[-1]) if v10_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v10.n_layers])
            if self.inner_v10.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        ret_proj = (
            self._retrieval_project(damped_ret)
            if damped_ret is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(ret_proj[i]
                    if i < len(ret_proj) else 0.0)
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
        wz = self.w_z_top11.values
        bz = self.b_z_top11.values
        wh = self.w_h_top11.values
        bh = self.b_h_top11.values
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
        next_layers = list(v10_layers) + [top_next]
        gates = list(v10_gates) + [z]
        return next_layers, gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W59_PERSISTENT_V11_SCHEMA_VERSION),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "state_dim": int(self.state_dim),
            "w_z_top11": self.w_z_top11.to_dict(),
            "b_z_top11": self.b_z_top11.to_dict(),
            "w_h_top11": self.w_h_top11.to_dict(),
            "b_h_top11": self.b_h_top11.to_dict(),
            "w_retrieval_skip": self.w_retrieval_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_v11_stacked_cell",
            "cell": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV11:
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
    anchor_carrier: tuple[float, ...]
    substrate_fidelity: float
    attention_fidelity: float
    retrieval_fidelity: float
    parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    substrate_skip_cid: str
    hidden_skip_cid: str
    attention_skip_cid: str
    retrieval_skip_cid: str
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
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "substrate_fidelity": float(round(
                self.substrate_fidelity, 12)),
            "attention_fidelity": float(round(
                self.attention_fidelity, 12)),
            "retrieval_fidelity": float(round(
                self.retrieval_fidelity, 12)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
            "hidden_skip_cid": str(self.hidden_skip_cid),
            "attention_skip_cid": str(self.attention_skip_cid),
            "retrieval_skip_cid": str(self.retrieval_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_v11_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV11Chain:
    states: dict[str, PersistentLatentStateV11]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV11Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV11) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV11 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W59_DEFAULT_V11_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV11]:
        out: list[PersistentLatentStateV11] = []
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
            "kind": "w59_v11_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v11(
        *,
        cell: V11StackedCell,
        prev_state: PersistentLatentStateV11 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        attention_skip: Sequence[float] | None = None,
        retrieval_skip: Sequence[float] | None = None,
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        fast_ema_alpha: float = 0.5,
        slow_ema_alpha: float = 0.10,
        substrate_ema_alpha: float = 0.25,
        hidden_ema_alpha: float = 0.20,
        attention_ema_alpha: float = 0.18,
        retrieval_ema_alpha: float = 0.16,
) -> PersistentLatentStateV11:
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
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W59_V11_NO_PARENT_STATE
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
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(max(0.0, min(1.0, float(fast_ema_alpha))))
    sa = float(max(0.0, min(1.0, float(slow_ema_alpha))))
    ua = float(max(0.0, min(1.0, float(substrate_ema_alpha))))
    ha = float(max(0.0, min(1.0, float(hidden_ema_alpha))))
    aa = float(max(0.0, min(1.0, float(attention_ema_alpha))))
    ra = float(max(0.0, min(1.0, float(retrieval_ema_alpha))))
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
        substrate_fidelity=substrate_fidelity,
        attention_fidelity=attention_fidelity,
        retrieval_fidelity=retrieval_fidelity)
    anchor_cid = _sha256_hex({
        "kind": "w59_v11_anchor_skip",
        "values": _round_floats(anchor)})
    fast_ema_cid = _sha256_hex({
        "kind": "w59_v11_fast_ema_skip",
        "values": _round_floats(fast_ema_next)})
    slow_ema_cid = _sha256_hex({
        "kind": "w59_v11_slow_ema_skip",
        "values": _round_floats(slow_ema_next)})
    substrate_cid = _sha256_hex({
        "kind": "w59_v11_substrate_skip",
        "values": _round_floats(substrate_next)})
    hidden_cid = _sha256_hex({
        "kind": "w59_v11_hidden_skip",
        "values": _round_floats(hidden_next)})
    attn_cid = _sha256_hex({
        "kind": "w59_v11_attention_skip",
        "values": _round_floats(attn_next)})
    retrieval_cid = _sha256_hex({
        "kind": "w59_v11_retrieval_skip",
        "values": _round_floats(retrieval_next)})
    gate_l1 = float(sum(
        abs(float(g)) for layer in gates for g in layer))
    return PersistentLatentStateV11(
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
        anchor_carrier=tuple(anchor),
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        substrate_skip_cid=str(substrate_cid),
        hidden_skip_cid=str(hidden_cid),
        attention_skip_cid=str(attn_cid),
        retrieval_skip_cid=str(retrieval_cid),
        update_gate_l1_sum=float(gate_l1),
        is_merge=False,
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV11Witness:
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
            "max_chain_walk_depth": int(
                self.max_chain_walk_depth),
            "achieved_chain_walk_depth": int(
                self.achieved_chain_walk_depth),
            "n_layers": int(self.n_layers),
            "state_dim": int(self.state_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_v11_persistent_witness",
            "witness": self.to_dict()})


def emit_persistent_v11_witness(
        *,
        cell: V11StackedCell,
        chain: PersistentLatentStateV11Chain,
        leaf_cid: str,
        max_chain_walk_depth: int = (
            W59_DEFAULT_V11_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV11Witness:
    walked = chain.walk_from(
        str(leaf_cid),
        max_depth=int(max_chain_walk_depth))
    return PersistentLatentStateV11Witness(
        schema=W59_PERSISTENT_V11_SCHEMA_VERSION,
        cell_cid=str(cell.cid()),
        chain_cid=str(chain.cid()),
        max_chain_walk_depth=int(max_chain_walk_depth),
        achieved_chain_walk_depth=int(len(walked)),
        n_layers=int(cell.n_layers),
        state_dim=int(cell.state_dim),
    )


__all__ = [
    "W59_PERSISTENT_V11_SCHEMA_VERSION",
    "W59_DEFAULT_V11_STATE_DIM",
    "W59_DEFAULT_V11_N_LAYERS",
    "W59_DEFAULT_V11_MAX_CHAIN_WALK_DEPTH",
    "W59_V11_NO_PARENT_STATE",
    "V11StackedCell",
    "PersistentLatentStateV11",
    "PersistentLatentStateV11Chain",
    "PersistentLatentStateV11Witness",
    "step_persistent_state_v11",
    "emit_persistent_v11_witness",
]
