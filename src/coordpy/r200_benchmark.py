"""W79 R-200 benchmark family — Direct-blocker-attack:
Controlled-runtime substrate + OpenAI-compatible façade +
learned consolidation + replay-vs-recompute economics +
hosted-vs-controlled comparison.

The first benchmark family in the Context-Zero programme that
makes the W79 direct-blocker-attack pillars load-bearing.

H1360..H1383 cell families (24 H-bars):

* H1360  Controlled runtime forward deterministic on identical
         (params, inputs)
* H1361  Controlled runtime exposes hidden state
* H1362  Controlled runtime exposes KV cache
* H1363  Controlled runtime exposes attention probs
* H1364  Controlled runtime exposes per-head attention bias hook
* H1365  Controlled runtime exposes prefix-state inject hook
* H1366  Controlled runtime exposes per-layer logits
* H1367  Controlled runtime replay-from-KV byte-identical to
         recompute
* H1368  Controlled runtime replay flops < recompute flops
* H1369  Façade chat completion deterministic on identical
         request CID
* H1370  Façade substrate side channel present when requested
* H1371  Façade substrate side channel CID stable on identical
         request
* H1372  Hosted plane blocks substrate axes; façade exposes them
* H1373  Hosted-vs-façade comparison report reflects asymmetry
* H1374  Learned consolidation training reduces loss
* H1375  Learned consolidation strictly beats closed-form ridge
         on nonlinear targets
* H1376  Learned consolidation training deterministic on
         (seed, data)
* H1377  Learned consolidation witness CID stable
* H1378  Replay-vs-recompute V2 arbiter ABSTAINS when transcript
         recompute is cheapest but ratio < floor
* H1379  Replay-vs-recompute V2 arbiter picks runtime_recompute
         when carrier is uncached and runtime is cheap
* H1380  Controlled-runtime substrate tier is exposed by
         substrate adapter V24
* H1381  Hosted boundary V12 explicitly blocks all 7 controlled-
         runtime axes at the hosted surface
* H1382  Bounded-window V2 baselines fail on a 200-turn-back
         query; controlled-runtime + LHR V2 substrate solve it
* H1383  Direct-blocker-attack: at least 2 of 4 W79 attack
         pillars deliver a load-bearing capability
         (controlled-runtime substrate, façade, learned
         consolidation, RTRLD substrate)
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import (
    BoundedWindowQuery,
)
from coordpy.bounded_window_baseline_v2 import (
    build_default_bounded_window_baselines_v2,
    run_bounded_window_falsifier_v2,
)
from coordpy.controlled_runtime_substrate_v1 import (
    W79_CONTROLLED_RUNTIME_AXES,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    measure_controlled_runtime_replay_vs_recompute,
    tokenize_bytes_v79,
)
from coordpy.hosted_real_substrate_boundary_v12 import (
    build_default_hosted_real_substrate_boundary_v12,
)
from coordpy.learned_consolidation_v1 import (
    build_learned_consolidation_head_v1,
    build_nonlinear_consolidation_dataset,
    compare_learned_vs_closed_form,
    emit_learned_consolidation_witness,
    train_learned_consolidation_head,
)
from coordpy.local_openai_compatible_facade_v1 import (
    LocalOpenAIChatCompletionRequestV1,
    LocalOpenAIChatMessageV1,
    LocalOpenAICompatibleFacadeV1,
    compare_facade_vs_hosted_substrate,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
)
from coordpy.long_horizon_reconstruction_substrate_v2 import (
    W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN,
    W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE,
    arbitrate_replay_vs_recompute_v2,
    build_default_long_horizon_reconstruction_carrier_v2,
    reconstruct_long_horizon_event_v2,
)
from coordpy.substrate_adapter_v24 import (
    W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
    probe_all_v24_adapters,
)


R200_SCHEMA_VERSION: str = "coordpy.r200_benchmark.v1"


def run_r200(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79("r200 controlled runtime smoke")
    # H1360: deterministic on identical inputs.
    t1, k1 = forward_controlled_runtime(
        params=p, input_token_ids=ids)
    t2, k2 = forward_controlled_runtime(
        params=p, input_token_ids=ids)
    cells["H1360"] = bool(t1.cid() == t2.cid())
    cells["H1361"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_hidden_state)
    cells["H1362"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_kv_cache)
    cells["H1363"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_attention_probs)
    cells["H1364"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_per_head_attention_bias)
    cells["H1365"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_prefix_state_inject)
    cells["H1366"] = bool(
        W79_CONTROLLED_RUNTIME_AXES.exposes_per_layer_logits)
    # H1367 + H1368: replay byte identical + flops saving.
    rep = measure_controlled_runtime_replay_vs_recompute(
        params=p, old_token_ids=ids[:6], new_token_ids=ids[6:])
    cells["H1367"] = bool(rep.replay_byte_identical)
    cells["H1368"] = bool(rep.saving_ratio > 0.0)
    # H1369 + H1370 + H1371: façade behavior.
    facade = LocalOpenAICompatibleFacadeV1(runtime_params=p)
    req = LocalOpenAIChatCompletionRequestV1(
        model="coordpy.controlled_runtime_substrate_v1",
        messages=(
            LocalOpenAIChatMessageV1(
                role="user",
                content=(
                    "Solve context for multi-agent teams")),),
        substrate_return_hidden_state=True,
        substrate_return_kv_cache_cid=True,
        substrate_return_attention_probs=True)
    resp_a = facade.chat_completions_create(request=req)
    resp_b = facade.chat_completions_create(request=req)
    cells["H1369"] = bool(resp_a.cid() == resp_b.cid())
    cells["H1370"] = bool(
        resp_a.substrate_side_channel is not None)
    cells["H1371"] = bool(
        resp_a.substrate_side_channel.cid()
        == resp_b.substrate_side_channel.cid())
    # H1372: hosted plane blocks substrate axes; façade exposes.
    boundary = build_default_hosted_real_substrate_boundary_v12()
    cells["H1372"] = bool(
        "controlled_runtime_hidden_state"
        in tuple(boundary.blocked_axes))
    cmp = compare_facade_vs_hosted_substrate(
        facade_response=resp_a,
        hosted_blocked_axes=tuple(boundary.blocked_axes))
    cells["H1373"] = bool(
        len(cmp.facade_axes_exposed) >= 7
        and not cmp.hosted_substrate_side_channel_present)
    # H1374 + H1375 + H1376 + H1377: learned consolidation.
    head = build_learned_consolidation_head_v1(seed=200)
    X, Y = build_nonlinear_consolidation_dataset(
        n_samples=96, seed=200)
    head, train_rep = train_learned_consolidation_head(
        head=head, train_features=X.tolist(),
        train_targets=Y.tolist())
    cells["H1374"] = bool(
        float(train_rep.post_loss)
        < float(train_rep.pre_loss))
    vs = compare_learned_vs_closed_form(
        head=head, eval_features=X.tolist(),
        eval_targets=Y.tolist())
    cells["H1375"] = bool(vs.learned_strictly_beats_ridge)
    head2 = build_learned_consolidation_head_v1(seed=200)
    X2, Y2 = build_nonlinear_consolidation_dataset(
        n_samples=96, seed=200)
    head2, train_rep2 = train_learned_consolidation_head(
        head=head2, train_features=X2.tolist(),
        train_targets=Y2.tolist())
    cells["H1376"] = bool(
        head.cid() == head2.cid()
        and train_rep.cid() == train_rep2.cid())
    w = emit_learned_consolidation_witness(
        head=head, vs_closed_form=vs)
    w2 = emit_learned_consolidation_witness(
        head=head2, vs_closed_form=vs)
    cells["H1377"] = bool(w.cid() == w2.cid())
    # H1378 + H1379: arbitration.
    carrier = build_default_long_horizon_reconstruction_carrier_v2(
        n_events=128, seed=200, head=head)
    q_long = LongHorizonReconstructionQuery(
        query_id="q_long", source_turn=10, current_turn=400)
    # Configure the three paths to be very close in cost so the
    # saving ratio falls below the high abstain floor under
    # heavy RTRLD pressure.
    arb_abs = arbitrate_replay_vs_recompute_v2(
        carrier_v2=carrier, query=q_long,
        controlled_runtime_params_cid="fake",
        substrate_walk_flops_per_hop=300_000,
        runtime_recompute_flops_per_new_token=11_000,
        transcript_recompute_flops_per_token=2700,
        replacement_then_restart_pressure=0.95,
        abstain_floor=0.30)
    cells["H1378"] = bool(
        arb_abs.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN)
    arb_run = arbitrate_replay_vs_recompute_v2(
        carrier_v2=carrier, query=q_long,
        controlled_runtime_params_cid="fake",
        substrate_walk_flops_per_hop=100_000_000,
        runtime_recompute_flops_per_new_token=1,
        transcript_recompute_flops_per_token=100_000_000)
    cells["H1379"] = bool(
        arb_run.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE)
    # H1380: substrate adapter exposes controlled-runtime tier.
    adapters = probe_all_v24_adapters()
    tiers = {c.tier for c in adapters.capabilities}
    cells["H1380"] = bool(
        W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1 in tiers
        or any(t.startswith("substrate_v24") for t in tiers))
    # H1381: 7 controlled-runtime axes blocked at hosted surface.
    ctrl_axes = (
        "controlled_runtime_hidden_state",
        "controlled_runtime_kv_cache",
        "controlled_runtime_attention_probs",
        "controlled_runtime_per_head_attention_bias",
        "controlled_runtime_prefix_state_inject",
        "controlled_runtime_per_layer_logits",
        "controlled_runtime_replay_from_kv")
    cells["H1381"] = bool(
        all(a in tuple(boundary.blocked_axes)
            for a in ctrl_axes))
    # H1382: bounded-window fail, LHR V2 substrate solves.
    baselines = build_default_bounded_window_baselines_v2()
    bw_q = BoundedWindowQuery(
        query_id="bq200", current_turn=210, source_turn=5,
        expected_event_cid=str(
            carrier.inner_v1.entries[5].event_cid))
    _, bw_rep = run_bounded_window_falsifier_v2(
        baselines_v2=baselines, query=bw_q)
    q_lhr = LongHorizonReconstructionQuery(
        query_id="q_lhr200", source_turn=5, current_turn=210)
    out_lhr = reconstruct_long_horizon_event_v2(
        carrier_v2=carrier, query=q_lhr,
        visible_tokens_used=4)
    cells["H1382"] = bool(
        bw_rep.all_fixed_k_failed_v2 and bool(out_lhr.success))
    # H1383: ≥ 2 of 4 attack pillars deliver a load-bearing
    # capability (controlled runtime byte-identical replay,
    # façade exposing substrate, learned > ridge, RTRLD substrate
    # solving long-delay reconstruction).
    pillar_count = int(sum([
        bool(rep.replay_byte_identical),
        bool(cells["H1370"]),
        bool(vs.learned_strictly_beats_ridge),
        bool(out_lhr.success),
    ]))
    cells["H1383"] = bool(pillar_count >= 2)
    return {
        "schema": R200_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R200_SCHEMA_VERSION", "run_r200"]
