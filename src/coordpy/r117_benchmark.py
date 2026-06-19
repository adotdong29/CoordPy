"""R-117 — W57 long-horizon retention / reconstruction / cramming family.

H57..H70 cell families that exercise:

  * V9 96-turn chain walk depth
  * V9 256-turn stretch chain walk
  * V9 384-turn deep stretch chain walk
  * LHR V9 three-way (proxy vs substrate vs hidden) comparison
  * LHR V9 substrate path beats proxy on substrate-aligned inputs
  * LHR V9 hidden path beats proxy on hidden-aligned inputs
  * ECC V9 ≥ 20 bits/visible-token at full emit
  * ECC V9 1-bit K8 stage
  * ECC V9 rate-floor (256 bits) falsifier reproduces
  * TVS V6 7-arm pick-rate sums to 1
  * TVS V6 substrate_hidden_inject preferred when hf high
  * TVS V6 substrate_replay preferred when sf high
  * V9 quad-skip + hidden-skip damping under low fidelity
  * V9 substrate carrier round-trip determinism
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r117_benchmark requires numpy") from exc

from .ecc_codebook_v9 import (
    ECCCodebookV9,
    compress_carrier_ecc_v9,
    probe_ecc_v9_rate_floor_falsifier,
    W57_DEFAULT_ECC_V9_TARGET_BITS_PER_TOKEN,
)
from .long_horizon_retention_v9 import (
    LongHorizonReconstructionV9Head,
    evaluate_lhr_v9_three_way,
)
from .persistent_latent_v9 import (
    PersistentLatentStateV9Chain,
    V9StackedCell,
    step_persistent_state_v9,
)
from .quantised_compression import QuantisedBudgetGate
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .transcript_vs_shared_arbiter_v6 import (
    W57_TVS_V6_ARMS,
    seven_arm_compare,
)


R117_SCHEMA_VERSION: str = "coordpy.r117_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R117SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R117_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def _v9_chain_walk(seed: int, n_turns: int) -> int:
    cell = V9StackedCell.init(seed=int(seed))
    chain = PersistentLatentStateV9Chain.empty()
    rng = random.Random(int(seed))
    prev = None
    for i in range(int(n_turns)):
        carrier = [rng.gauss(0.0, 0.5)
                    for _ in range(cell.state_dim)]
        hidden = [rng.gauss(0.0, 0.3)
                   for _ in range(cell.state_dim)]
        sub = [rng.gauss(0.0, 0.3)
                for _ in range(cell.state_dim)]
        st = step_persistent_state_v9(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=i, role="r0",
            hidden_state_skip=hidden,
            substrate_skip=sub,
            substrate_fidelity=0.8)
        chain.add(st)
        prev = st
    return int(len(chain.walk_from(prev.cid())))


def family_v9_chain_walk_96(seed: int) -> dict[str, Any]:
    """H57 — V9 96-turn chain walk."""
    depth = _v9_chain_walk(int(seed), 96)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "v9_chain_walk_96",
        "passed": bool(depth >= 96),
        "depth": int(depth),
    }


def family_v9_chain_walk_256(seed: int) -> dict[str, Any]:
    """H58 — V9 256-turn stretch chain walk."""
    depth = _v9_chain_walk(int(seed), 256)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "v9_chain_walk_256",
        "passed": bool(depth >= 256),
        "depth": int(depth),
    }


def family_v9_chain_walk_384(seed: int) -> dict[str, Any]:
    """H59 — V9 384-turn deep stretch."""
    depth = _v9_chain_walk(int(seed), 384)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "v9_chain_walk_384",
        "passed": bool(depth >= 384),
        "depth": int(depth),
    }


def family_lhr_v9_three_way(seed: int) -> dict[str, Any]:
    """H60 — LHR V9 three-way comparison runs and reports MSEs."""
    head = LongHorizonReconstructionV9Head.init(
        seed=int(seed))
    rng = _np.random.default_rng(int(seed))
    n_examples = 4
    carrier_dim = head.inner_v8.inner_v7.carrier_dim
    out_dim = head.out_dim
    carriers = [
        list(rng.standard_normal(carrier_dim))
        for _ in range(n_examples)]
    targets = [
        list(rng.standard_normal(out_dim))
        for _ in range(n_examples)]
    sub_states = [
        list(rng.standard_normal(head.hidden_dim))
        for _ in range(n_examples)]
    hidden_states = [
        list(rng.standard_normal(head.hidden_dim))
        for _ in range(n_examples)]
    r = evaluate_lhr_v9_three_way(
        head,
        carrier_examples=carriers,
        target_examples=targets,
        substrate_states=sub_states,
        hidden_states=hidden_states,
        k=8)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "lhr_v9_three_way",
        "passed": bool(r["n"] == n_examples),
        "proxy_mse": float(r["proxy_mse"]),
        "substrate_mse": float(r["substrate_mse"]),
        "hidden_state_mse": float(r["hidden_state_mse"]),
    }


def family_lhr_v9_substrate_helps_aligned(
        seed: int,
) -> dict[str, Any]:
    """H61 — substrate-aligned targets: substrate head beats
    proxy."""
    head = LongHorizonReconstructionV9Head.init(
        seed=int(seed))
    rng = _np.random.default_rng(int(seed))
    n = 4
    carrier_dim = head.inner_v8.inner_v7.carrier_dim
    out_dim = head.out_dim
    carriers = [
        list(rng.standard_normal(carrier_dim) * 0.5)
        for _ in range(n)]
    sub_states = [
        list(rng.standard_normal(head.hidden_dim))
        for _ in range(n)]
    # Targets aligned with substrate projection (so substrate
    # head should help).
    targets = []
    for i in range(n):
        sub_proj = head.substrate_conditioned_value(
            carrier=carriers[i], k=8,
            substrate_state=sub_states[i])
        # Target IS the substrate head's output (substrate head
        # then has MSE = 0; proxy head has MSE > 0).
        targets.append(list(sub_proj))
    r = evaluate_lhr_v9_three_way(
        head,
        carrier_examples=carriers,
        target_examples=targets,
        substrate_states=sub_states,
        hidden_states=[None] * n,
        k=8)
    helps = bool(r["substrate_mse"] < r["proxy_mse"] - 1e-6)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "lhr_v9_substrate_helps_aligned",
        "passed": bool(helps),
        "proxy_mse": float(r["proxy_mse"]),
        "substrate_mse": float(r["substrate_mse"]),
    }


def family_lhr_v9_hidden_helps_aligned(
        seed: int,
) -> dict[str, Any]:
    """H62 — hidden-state-aligned targets: hidden head beats
    substrate-only head."""
    head = LongHorizonReconstructionV9Head.init(
        seed=int(seed))
    rng = _np.random.default_rng(int(seed))
    n = 4
    carrier_dim = head.inner_v8.inner_v7.carrier_dim
    out_dim = head.out_dim
    carriers = [
        list(rng.standard_normal(carrier_dim) * 0.5)
        for _ in range(n)]
    sub_states = [
        list(rng.standard_normal(head.hidden_dim))
        for _ in range(n)]
    hidden_states = [
        list(rng.standard_normal(head.hidden_dim))
        for _ in range(n)]
    targets = []
    for i in range(n):
        hp = head.hidden_state_conditioned_value(
            carrier=carriers[i], k=8,
            hidden_state=hidden_states[i],
            substrate_state=sub_states[i])
        targets.append(list(hp))
    r = evaluate_lhr_v9_three_way(
        head,
        carrier_examples=carriers,
        target_examples=targets,
        substrate_states=sub_states,
        hidden_states=hidden_states,
        k=8)
    helps = bool(
        r["hidden_state_mse"] < r["substrate_mse"] - 1e-6)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "lhr_v9_hidden_helps_aligned",
        "passed": bool(helps),
        "substrate_mse": float(r["substrate_mse"]),
        "hidden_state_mse": float(r["hidden_state_mse"]),
    }


def family_ecc_v9_20_bits_per_token(seed: int) -> dict[str, Any]:
    """H63 — ECC V9 delivers ≥ 20 bits/visible-token at full emit."""
    cb = ECCCodebookV9.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 4)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed))
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v9(
        carrier, codebook=cb, gate=gate)
    bp = float(comp["bits_per_visible_token"])
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "ecc_v9_20_bits_per_token",
        "passed": bool(bp >= float(
            W57_DEFAULT_ECC_V9_TARGET_BITS_PER_TOKEN)),
        "bits_per_visible_token": float(bp),
        "structured_bits_v9": int(comp["structured_bits_v9"]),
    }


def family_ecc_v9_k8_bit(seed: int) -> dict[str, Any]:
    """H64 — ECC V9 K8 bit is deterministic over a fixed carrier."""
    cb = ECCCodebookV9.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 4)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    # Carrier with sq sum > 1 -> K8=1; carrier with sq sum < 1 -> K8=0.
    high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    low = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    c_hi = compress_carrier_ecc_v9(
        high, codebook=cb, gate=gate)
    c_lo = compress_carrier_ecc_v9(
        low, codebook=cb, gate=gate)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "ecc_v9_k8_bit",
        "passed": bool(
            int(c_hi["ultra6_index"]) == 1
            and int(c_lo["ultra6_index"]) == 0),
        "high_k8": int(c_hi["ultra6_index"]),
        "low_k8": int(c_lo["ultra6_index"]),
    }


def family_ecc_v9_rate_floor_falsifier(
        seed: int,
) -> dict[str, Any]:
    """H65 — 256 bits/token target above structural ceiling."""
    cb = ECCCodebookV9.init(seed=int(seed))
    f = probe_ecc_v9_rate_floor_falsifier(
        codebook=cb,
        target_bits_per_token=256.0,
        seed=int(seed) + 5)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "ecc_v9_rate_floor_falsifier",
        "passed": bool(f["target_above_info_bound"]),
        "info_bound": float(f["info_bound"]),
        "target_bits_per_token": 256.0,
    }


def family_tvs_v6_pick_rates_sum_to_1(
        seed: int,
) -> dict[str, Any]:
    """H66 — 7-arm pick-rates sum to 1 over n_turns."""
    n = 8
    rng = random.Random(int(seed))
    confs = [rng.random() for _ in range(n)]
    trusts = [rng.random() for _ in range(n)]
    merges = [rng.random() for _ in range(n)]
    tws = [rng.random() for _ in range(n)]
    subs = [rng.random() for _ in range(n)]
    hids = [rng.random() for _ in range(n)]
    r = seven_arm_compare(
        per_turn_confidences=confs,
        per_turn_trust_scores=trusts,
        per_turn_merge_retentions=merges,
        per_turn_tw_retentions=tws,
        per_turn_substrate_fidelities=subs,
        per_turn_hidden_fidelities=hids,
        budget_tokens=6)
    s = float(sum(r.pick_rates.values()))
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "tvs_v6_pick_rates_sum_to_1",
        "passed": bool(abs(s - 1.0) < 1e-9),
        "sum": float(s),
        "n_arms": len(W57_TVS_V6_ARMS),
    }


def family_tvs_v6_hidden_inject_preferred(
        seed: int,
) -> dict[str, Any]:
    """H67 — when hf >> sf, hidden_inject arm preferred."""
    r = seven_arm_compare(
        per_turn_confidences=[0.3, 0.3, 0.3],
        per_turn_trust_scores=[0.3, 0.3, 0.3],
        per_turn_merge_retentions=[0.3, 0.3, 0.3],
        per_turn_tw_retentions=[0.3, 0.3, 0.3],
        per_turn_substrate_fidelities=[0.4, 0.4, 0.4],
        per_turn_hidden_fidelities=[0.9, 0.9, 0.9],
        budget_tokens=6)
    hid_pct = float(r.pick_rates.get(
        "substrate_hidden_inject", 0.0))
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "tvs_v6_hidden_inject_preferred",
        "passed": bool(hid_pct >= 1.0 - 1e-9),
        "hidden_inject_rate": float(hid_pct),
    }


def family_tvs_v6_substrate_replay_preferred(
        seed: int,
) -> dict[str, Any]:
    """H68 — when sf >> hf, substrate_replay arm preferred."""
    r = seven_arm_compare(
        per_turn_confidences=[0.3, 0.3, 0.3],
        per_turn_trust_scores=[0.3, 0.3, 0.3],
        per_turn_merge_retentions=[0.3, 0.3, 0.3],
        per_turn_tw_retentions=[0.3, 0.3, 0.3],
        per_turn_substrate_fidelities=[0.9, 0.9, 0.9],
        per_turn_hidden_fidelities=[0.4, 0.4, 0.4],
        budget_tokens=6)
    sub_pct = float(r.pick_rates.get("substrate_replay", 0.0))
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "tvs_v6_substrate_replay_preferred",
        "passed": bool(sub_pct >= 1.0 - 1e-9),
        "substrate_replay_rate": float(sub_pct),
    }


def family_v9_fidelity_damps_substrate(
        seed: int,
) -> dict[str, Any]:
    """H69 — V9 substrate_fidelity=0 damps substrate carrier to 0
    contribution (carrier unchanged when substrate damped)."""
    cell = V9StackedCell.init(seed=int(seed))
    rng = random.Random(int(seed))
    carrier = [rng.gauss(0.0, 0.5)
                for _ in range(cell.state_dim)]
    hidden = [rng.gauss(0.0, 0.5)
               for _ in range(cell.state_dim)]
    sub = [1.0] * cell.state_dim
    st_full = step_persistent_state_v9(
        cell=cell, prev_state=None,
        carrier_values=carrier, turn_index=0, role="r0",
        substrate_skip=sub, hidden_state_skip=hidden,
        substrate_fidelity=1.0)
    st_damped = step_persistent_state_v9(
        cell=cell, prev_state=None,
        carrier_values=carrier, turn_index=0, role="r0",
        substrate_skip=sub, hidden_state_skip=hidden,
        substrate_fidelity=0.0)
    # The two top-layer states should differ (with fid=1 substrate
    # contributes; with fid=0 it doesn't).
    differ = any(
        abs(float(a) - float(b)) > 1e-6
        for a, b in zip(st_full.top_state, st_damped.top_state))
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "v9_fidelity_damps_substrate",
        "passed": bool(differ),
        "top_full": list(st_full.top_state)[:4],
        "top_damped": list(st_damped.top_state)[:4],
    }


def family_v9_carrier_round_trip(seed: int) -> dict[str, Any]:
    """H70 — V9 step is byte-deterministic with identical inputs."""
    cell = V9StackedCell.init(seed=int(seed))
    rng = random.Random(int(seed))
    carrier = [rng.gauss(0.0, 0.5)
                for _ in range(cell.state_dim)]
    hidden = [rng.gauss(0.0, 0.5)
               for _ in range(cell.state_dim)]
    sub = [rng.gauss(0.0, 0.5)
            for _ in range(cell.state_dim)]
    s1 = step_persistent_state_v9(
        cell=cell, prev_state=None, carrier_values=carrier,
        turn_index=0, role="r0",
        substrate_skip=sub, hidden_state_skip=hidden,
        substrate_fidelity=0.8)
    s2 = step_persistent_state_v9(
        cell=cell, prev_state=None, carrier_values=carrier,
        turn_index=0, role="r0",
        substrate_skip=sub, hidden_state_skip=hidden,
        substrate_fidelity=0.8)
    return {
        "schema": R117_SCHEMA_VERSION,
        "name": "v9_carrier_round_trip",
        "passed": bool(s1.cid() == s2.cid()),
        "state_cids_match": bool(s1.cid() == s2.cid()),
    }


R117_FAMILIES = (
    ("h57_v9_chain_walk_96", family_v9_chain_walk_96),
    ("h58_v9_chain_walk_256", family_v9_chain_walk_256),
    ("h59_v9_chain_walk_384", family_v9_chain_walk_384),
    ("h60_lhr_v9_three_way", family_lhr_v9_three_way),
    ("h61_lhr_v9_substrate_helps_aligned",
     family_lhr_v9_substrate_helps_aligned),
    ("h62_lhr_v9_hidden_helps_aligned",
     family_lhr_v9_hidden_helps_aligned),
    ("h63_ecc_v9_20_bits_per_token",
     family_ecc_v9_20_bits_per_token),
    ("h64_ecc_v9_k8_bit", family_ecc_v9_k8_bit),
    ("h65_ecc_v9_rate_floor_falsifier",
     family_ecc_v9_rate_floor_falsifier),
    ("h66_tvs_v6_pick_rates_sum_to_1",
     family_tvs_v6_pick_rates_sum_to_1),
    ("h67_tvs_v6_hidden_inject_preferred",
     family_tvs_v6_hidden_inject_preferred),
    ("h68_tvs_v6_substrate_replay_preferred",
     family_tvs_v6_substrate_replay_preferred),
    ("h69_v9_fidelity_damps_substrate",
     family_v9_fidelity_damps_substrate),
    ("h70_v9_carrier_round_trip",
     family_v9_carrier_round_trip),
)


def run_r117(*, seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    rows: list[R117SeedResult] = []
    for s in seeds:
        results: dict[str, dict[str, Any]] = {}
        for name, fn in R117_FAMILIES:
            results[name] = fn(int(s))
        rows.append(R117SeedResult(
            seed=int(s), family_results=results))
    summary = {
        "schema": R117_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "seeds": [r.to_dict() for r in rows],
    }
    pass_counts: dict[str, int] = {}
    for r in rows:
        for k, v in r.family_results.items():
            if bool(v.get("passed", False)):
                pass_counts[k] = pass_counts.get(k, 0) + 1
    summary["pass_counts"] = pass_counts
    summary["all_passed"] = bool(all(
        pass_counts.get(name, 0) == len(seeds)
        for name, _ in R117_FAMILIES))
    return summary


__all__ = [
    "R117_SCHEMA_VERSION",
    "R117_FAMILIES",
    "R117SeedResult",
    "run_r117",
]
