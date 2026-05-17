"""W75 tests — tiny_substrate_v20."""

from __future__ import annotations

from coordpy.tiny_substrate_v16 import record_restart_event_v16
from coordpy.tiny_substrate_v17 import record_rejoin_event_v17
from coordpy.tiny_substrate_v18 import (
    record_replacement_event_v18,
)
from coordpy.tiny_substrate_v19 import (
    record_compound_failure_window_v19,
    record_delayed_repair_event_v19,
)
from coordpy.tiny_substrate_v20 import (
    W75_DEFAULT_V20_N_LAYERS, W75_REPAIR_COMPOUND_CHAIN_REPAIR,
    W75_REPAIR_LABELS_V20,
    build_default_tiny_substrate_v20,
    emit_tiny_substrate_v20_forward_witness,
    forward_tiny_substrate_v20,
    record_compound_chain_window_v20,
    substrate_compound_chain_pressure_throttle_v20,
    substrate_compound_chain_repair_dominance_flops_v20,
    tokenize_bytes_v20,
)


def test_v20_substrate_has_22_layers() -> None:
    p = build_default_tiny_substrate_v20()
    ids = tokenize_bytes_v20("w75", max_len=4)
    trace, _ = forward_tiny_substrate_v20(p, ids)
    assert trace.v20_gate_score_per_layer.shape[0] == 22
    assert W75_DEFAULT_V20_N_LAYERS == 22


def test_v20_repair_labels_include_chain() -> None:
    assert len(W75_REPAIR_LABELS_V20) == 12
    assert W75_REPAIR_LABELS_V20[
        W75_REPAIR_COMPOUND_CHAIN_REPAIR] == (
        "compound_repair_after_replacement_then_rejoin")


def test_v20_compound_chain_cid_is_content_addressed() -> None:
    p = build_default_tiny_substrate_v20()
    ids = tokenize_bytes_v20("w75-chain-cid", max_len=12)
    _, cache_a = forward_tiny_substrate_v20(p, ids)
    _, cache_b = forward_tiny_substrate_v20(p, ids)
    # Same setup, same CID.
    assert (
        str(cache_a.compound_chain_repair_trajectory_cid)
        == str(cache_b.compound_chain_repair_trajectory_cid))
    # Record events and re-run — CID changes.
    record_compound_chain_window_v20(
        cache_a, replacement_turn=2, delayed_repair_turn=5,
        rejoin_turn=10, compound_chain_window_turns=8,
        role="r", branch_id="b")
    _, cache_a2 = forward_tiny_substrate_v20(
        p, ids, v20_kv_cache=cache_a)
    assert (
        str(cache_a2.compound_chain_repair_trajectory_cid)
        != str(cache_b.compound_chain_repair_trajectory_cid))


def test_v20_chain_label_fires_under_compound_chain() -> None:
    p = build_default_tiny_substrate_v20()
    ids = tokenize_bytes_v20("w75-chain-label", max_len=12)
    _, cache = forward_tiny_substrate_v20(p, ids)
    v19 = cache.v19_cache
    v18 = v19.v18_cache
    record_restart_event_v16(
        v18.v17_cache.v16_cache, turn=1,
        restart_kind="r", role="p")
    record_rejoin_event_v17(
        v18.v17_cache, turn=2, rejoin_kind="rj",
        branch_id="b", role="p")
    record_replacement_event_v18(
        v18, turn=3, replacement_kind="rep",
        role="p", new_role="p2")
    record_delayed_repair_event_v19(
        v19, turn=4, delayed_kind="dr", role="p")
    record_compound_failure_window_v19(
        v19, delayed_repair_turn=4, replacement_turn=3,
        rejoin_turn=10, compound_window_turns=8, role="p",
        branch_id="b")
    record_compound_chain_window_v20(
        cache, replacement_turn=3, delayed_repair_turn=4,
        rejoin_turn=10, compound_chain_window_turns=8, role="p",
        branch_id="b")
    trace, cache = forward_tiny_substrate_v20(
        p, ids, v20_kv_cache=cache,
        compound_chain_pressure=0.8)
    w = emit_tiny_substrate_v20_forward_witness(trace, cache)
    assert int(w.compound_chain_repair_l1) >= 1


def test_v20_chain_repair_dominance_flops_saves() -> None:
    r = substrate_compound_chain_repair_dominance_flops_v20(
        n_tokens=128, n_repairs=11)
    assert r["saving_ratio"] >= 0.9
    assert r["recompute_flops"] > r["chain_dominance_flops"]


def test_v20_chain_pressure_throttle_saves_tokens() -> None:
    r = substrate_compound_chain_pressure_throttle_v20(
        visible_token_budget=32, baseline_token_cost=512,
        compound_chain_window_turns=4)
    assert r["compound_chain_pressure_active"]
    assert r["saving_tokens"] > 0
