"""Phase 17 — task generality probe.

Phase 14/15/16 all used ProtocolKit-style tasks where the coordination
surface is underspecified dict-key naming. Phase 17 swaps both ends of
that surface:

  * Task: `numeric_ledger` — 12 agents coordinating on numerical
    conventions (rounding, scale, NaN, overflow, signed encoding).
  * Event trigger: `core/behavior_trigger.py` — disagreement is measured
    by running probe batteries through each pair's drafts, not by
    comparing dict-key sets.

Same 3-leg structure (full / casr / ablation), same pre-registered
claims C1–C5 as Phase 14, plus the generality-specific C7:

  C7: in the CASR leg, per-pair test pass correlates with the
      behavior-trigger's post-refinement convention agreement; in the
      ablation leg, it does not. Operational test:
          mean(per-pair pass fraction) in CASR > mean in ABLATION + 0.20

If C1–C5 hold at comparable levels to Phase 14 but with the new task
and trigger, the CASR mechanism is not ProtocolKit-specific.

Usage:
  python3 -m vision_mvp.experiments.phase17_generality \\
      --out vision_mvp/results_phase17.json
"""

from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.code_harness import (
    extract_code, function_is_defined, run_sandboxed,
)
from vision_mvp.core.causal_footprint import (
    footprint_from_call_graph, random_footprint,
)
from vision_mvp.core.casr_router import CASRRouter, RouterMessage
from vision_mvp.core.token_meter import count_tokens
# Behavior-based trigger (alternative to Phase-14's dict-key Jaccard).
from vision_mvp.core.trigger import Trigger, behavior_probe_trigger
from vision_mvp.tasks.numeric_ledger import (
    FUNCTION_SPECS, SPEC_ORDER, CALL_GRAPH,
    TEST_RUNNER_SRC, compose_module, score_tests, agent_prompt,
    PAIR_TESTS, INTEGRATOR_TESTS,
)


def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------- Tier derivation (same as Phase 14) --------------------

def compute_tiers(call_graph: dict[str, list[str]]) -> dict[str, int]:
    memo: dict[str, int] = {}

    def depth(node: str, seen: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in seen:
            return 0
        callees = call_graph.get(node, [])
        if not callees:
            memo[node] = 0
            return 0
        d = 1 + max(depth(c, seen | {node}) for c in callees)
        memo[node] = d
        return d

    return {n: depth(n, set()) for n in call_graph}


# -------------------- Round 1 -----------------------------------------------

def run_round1(client: LLMClient, max_retries: int = 2) -> dict:
    log("\n" + "=" * 78)
    log(f"[round 1] independent drafts (all {len(SPEC_ORDER)} agents)")
    log("=" * 78)

    drafts: dict[str, str] = {}
    tokens: dict[str, dict] = {}
    acceptance: dict[str, dict] = {}

    for specialty in SPEC_ORDER:
        target_name = FUNCTION_SPECS[specialty]["name"]
        log(f"  - {specialty}: drafting ...")
        prompt = agent_prompt(specialty)
        prompt_toks = count_tokens(prompt)

        accepted = None
        out_tokens = 0
        n_attempts = 0
        for attempt in range(max_retries + 1):
            n_attempts += 1
            t0 = time.time()
            out = client.generate(prompt, max_tokens=500, temperature=0.2)
            wall = time.time() - t0
            out_tokens = count_tokens(out)
            code = extract_code(out)
            if code and function_is_defined(code, target_name):
                accepted = code
                log(f"    accepted ({wall:.1f}s, prompt={prompt_toks}t, "
                    f"completion={out_tokens}t)")
                break
            log(f"    attempt {attempt+1} failed")

        drafts[specialty] = accepted or ""
        tokens[specialty] = {"prompt": prompt_toks, "completion": out_tokens}
        acceptance[specialty] = {
            "accepted": accepted is not None, "attempts": n_attempts,
        }

    return {"drafts": drafts, "tokens": tokens, "acceptance": acceptance}


# -------------------- Round 2 ------------------------------------------------

def round2_prompt(specialty: str, own_draft: str,
                  bulletin: dict[str, str]) -> str:
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 12-person Python team "
        f"building NumericLedger — small typed primitives where pairs of "
        f"agents must agree on hidden NUMERICAL CONVENTIONS (rounding, "
        f"scale, NaN, overflow, signed encoding).",
        f"Your function: `{s['name']}` with signature {s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Your own ROUND-1 draft:\n```python\n{own_draft or '# (failed)'}\n```",
    ]
    if bulletin:
        parts.append("FROZEN teammates' drafts (these will NOT change):")
        for dep, src in bulletin.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(
                f"  - `{dep_name}` (from {dep}):\n```python\n{src[:500]}\n```"
            )
    parts.append(
        f"Output ONLY the final version of `{s['name']}` in a "
        f"```python code fence. MIRROR the teammates' numerical "
        f"conventions EXACTLY. No commentary. No re-defining teammates' "
        f"functions."
    )
    return "\n\n".join(parts)


def run_round2_topological(
    client: LLMClient,
    round1_drafts: dict[str, str],
    router: CASRRouter,
    tiers: dict[str, int],
    event_threshold: float,
    max_retries: int = 2,
    force_refine: bool = False,
    trigger: Trigger | None = None,
) -> dict:
    if trigger is None:
        trigger = behavior_probe_trigger()
    final: dict[str, str] = dict(round1_drafts)
    tokens: dict[str, dict] = {}
    routing_stats: dict[str, dict] = {}
    trigger_info: dict[str, dict] = {}
    acceptance: dict[str, dict] = {}

    max_tier = max(tiers.values())

    for tier in range(max_tier + 1):
        tier_members = [s for s in SPEC_ORDER if tiers[s] == tier]
        log(f"\n  -- tier {tier}: {tier_members} --")
        if tier == 0:
            for s in tier_members:
                log(f"    [{s}] FROZEN (tier 0)")
                tokens[s] = {"prompt": 0, "completion": 0}
                routing_stats[s] = {
                    "delivered": 0, "dropped": 0,
                    "delivered_tokens": 0, "dropped_tokens": 0,
                }
                trigger_info[s] = {"tier": 0, "refined": False,
                                    "reason": "tier0_frozen"}
                acceptance[s] = {"accepted": bool(final[s]), "attempts": 0}
            continue

        pool: list[RouterMessage] = []
        for src in SPEC_ORDER:
            if tiers[src] < tier and final[src]:
                pool.append(RouterMessage(
                    source_id=src, payload=final[src],
                    tokens=count_tokens(final[src]),
                ))

        for specialty in tier_members:
            target_name = FUNCTION_SPECS[specialty]["name"]
            delivered, stats = router.route(pool, recipient_id=specialty)
            bulletin = {m.source_id: m.payload for m in delivered}

            own = round1_drafts.get(specialty, "")
            trig = trigger.should_refine(own, list(bulletin.values()),
                                          threshold=event_threshold)
            effective_refine = trig.refine or force_refine

            log(f"    [{specialty}] routed={stats.delivered}  "
                f"disagreement={trig.score:.2f}  "
                f"{'REFINE' if effective_refine else 'SKIP'}"
                f"{' (forced)' if (force_refine and not trig.refine) else ''}")

            if not effective_refine:
                final[specialty] = own
                tokens[specialty] = {"prompt": 0, "completion": 0}
                routing_stats[specialty] = {
                    "delivered": stats.delivered,
                    "dropped": stats.dropped,
                    "delivered_tokens": stats.delivered_tokens,
                    "dropped_tokens": stats.dropped_tokens,
                }
                trigger_info[specialty] = {
                    "tier": tier, "refined": False,
                    "score": trig.score, "threshold": event_threshold,
                    "reason": "below_threshold",
                }
                acceptance[specialty] = {"accepted": bool(own), "attempts": 0}
                continue

            prompt = round2_prompt(specialty, own, bulletin)
            prompt_toks = count_tokens(prompt)
            accepted = None
            out_tokens = 0
            n_attempts = 0
            for attempt in range(max_retries + 1):
                n_attempts += 1
                t0 = time.time()
                out = client.generate(prompt, max_tokens=500, temperature=0.2)
                wall = time.time() - t0
                out_tokens = count_tokens(out)
                code = extract_code(out)
                if code and function_is_defined(code, target_name):
                    accepted = code
                    log(f"      LLM accepted ({wall:.1f}s, "
                        f"prompt={prompt_toks}t, completion={out_tokens}t)")
                    break
                log(f"      attempt {attempt+1} failed")

            if accepted is None:
                accepted = own
                log(f"      fell back to round-1")

            final[specialty] = accepted
            tokens[specialty] = {"prompt": prompt_toks, "completion": out_tokens}
            routing_stats[specialty] = {
                "delivered": stats.delivered,
                "dropped": stats.dropped,
                "delivered_tokens": stats.delivered_tokens,
                "dropped_tokens": stats.dropped_tokens,
            }
            trigger_info[specialty] = {
                "tier": tier, "refined": True,
                "score": trig.score, "threshold": event_threshold,
                "reason": "above_threshold",
            }
            acceptance[specialty] = {
                "accepted": accepted is not own, "attempts": n_attempts,
            }

    return {
        "drafts": final,
        "tokens": tokens,
        "routing_stats": routing_stats,
        "trigger_info": trigger_info,
        "acceptance": acceptance,
    }


# -------------------- Scoring + footprints ---------------------------------

def score_drafts(drafts: dict[str, str], label: str) -> dict:
    module_src = compose_module(drafts)
    log(f"\n[score:{label}] module {len(module_src)} chars")
    result = run_sandboxed(module_src, TEST_RUNNER_SRC, timeout_s=30)
    score = score_tests(result.per_test)
    log(f"[score:{label}] {score['n_passed']}/{score['n_total']} passed "
        f"(weighted {score['weighted_score']})")
    return {
        "weighted_score": score["weighted_score"],
        "n_passed": score["n_passed"],
        "n_total": score["n_total"],
        "passed_tests": score["passed_tests"],
        "failed_tests": score["failed_tests"],
        "per_test": score["per_test"],
        "syntax_error": result.syntax_error,
        "timed_out": result.timed_out,
        "stderr": result.stderr[:500],
        "module_src": module_src,
    }


def build_causal_footprints() -> dict:
    return {s: footprint_from_call_graph(CALL_GRAPH, s) for s in SPEC_ORDER}


def build_random_footprints(reference_footprints: dict, seed: int) -> dict:
    out = {}
    for i, spec in enumerate(SPEC_ORDER):
        out[spec] = random_footprint(
            list(SPEC_ORDER),
            size=len(reference_footprints[spec]),
            seed=seed + i,
        )
    return out


def summarize_tokens(round1: dict, round2: dict) -> dict:
    total_prompt = 0
    total_completion = 0
    for spec in SPEC_ORDER:
        total_prompt += round1["tokens"][spec]["prompt"]
        total_completion += round1["tokens"][spec]["completion"]
        total_prompt += round2["tokens"][spec]["prompt"]
        total_completion += round2["tokens"][spec]["completion"]
    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }


# -------------------- C7: per-pair pass-fraction ---------------------------

def per_pair_pass_fractions(per_test: dict[str, bool]) -> dict[str, float]:
    """For each pair, compute the fraction of its 3 tests that passed."""
    out = {}
    for pair, tests in PAIR_TESTS.items():
        passed = sum(1 for t in tests if per_test.get(t) is True)
        out[pair] = passed / len(tests)
    return out


def mean_pair_pass_fraction(per_test: dict[str, bool]) -> float:
    fractions = per_pair_pass_fractions(per_test)
    return sum(fractions.values()) / len(fractions)


# -------------------- Main ---------------------------------------------------

def run(model: str, out_path: str | None, max_retries: int,
        ablation_seed: int, event_threshold: float,
        force_refine: bool = False,
        trigger: Trigger | None = None,
        round1: dict | None = None) -> dict:
    client = LLMClient(model=model)
    if trigger is None:
        trigger = behavior_probe_trigger()
    log(f"Phase 17 — task generality probe  (numeric_ledger, N={len(SPEC_ORDER)})")
    log(f"Model: {model}")
    log(f"Trigger: {trigger.name}")
    log(f"Event trigger threshold: {event_threshold}")
    log(f"Force-refine all tier-1 agents: {force_refine}")

    tiers = compute_tiers(CALL_GRAPH)
    log(f"Tiers: {tiers}")

    if round1 is None:
        round1 = run_round1(client, max_retries=max_retries)
    else:
        log("[round 1] using pre-generated round-1 (paired A/B mode)")
    round1_score = score_drafts(round1["drafts"], "round1_no_coord")

    causal_fps = build_causal_footprints()
    random_fps = build_random_footprints(causal_fps, seed=ablation_seed)

    legs = {}
    for mode, fps in [
        ("full", causal_fps),
        ("casr", causal_fps),
        ("ablation", random_fps),
    ]:
        log("\n" + "=" * 78)
        log(f"[round 2 / {mode}] topological + behavior-triggered")
        log("=" * 78)
        router = CASRRouter(mode=mode, footprints=fps)
        r2 = run_round2_topological(
            client, round1["drafts"], router, tiers,
            event_threshold=event_threshold,
            max_retries=max_retries,
            force_refine=force_refine,
            trigger=trigger,
        )
        leg_score = score_drafts(r2["drafts"], f"round2_{mode}")
        token_totals = summarize_tokens(round1, r2)
        n_refined = sum(1 for t in r2["trigger_info"].values() if t.get("refined"))
        n_skipped = sum(
            1 for s, t in r2["trigger_info"].items()
            if (not t.get("refined")) and tiers[s] > 0
        )
        legs[mode] = {
            "round2": r2,
            "score": leg_score,
            "token_totals": token_totals,
            "n_refined": n_refined,
            "n_skipped": n_skipped,
            "per_pair_pass": per_pair_pass_fractions(leg_score["per_test"]),
            "mean_pair_pass": mean_pair_pass_fraction(leg_score["per_test"]),
        }
        log(f"  → {mode}: {n_refined} refined, {n_skipped} skipped")
        log(f"     per-pair pass fractions: {legs[mode]['per_pair_pass']}")

        # Crash-resilient partial save
        if out_path:
            with open(out_path, "w") as f:
                json.dump({
                    "round1_score": round1_score,
                    "tiers": tiers,
                    "legs": {k: {
                        "weighted_score": v["score"]["weighted_score"],
                        "n_passed": v["score"]["n_passed"],
                        "n_total": v["score"]["n_total"],
                        "passed_tests": v["score"]["passed_tests"],
                        "failed_tests": v["score"]["failed_tests"],
                        "per_test": v["score"]["per_test"],
                        "total_prompt_tokens": v["token_totals"]["total_prompt_tokens"],
                        "total_completion_tokens": v["token_totals"]["total_completion_tokens"],
                        "n_refined": v["n_refined"],
                        "n_skipped": v["n_skipped"],
                        "per_pair_pass": v["per_pair_pass"],
                        "mean_pair_pass": v["mean_pair_pass"],
                        "trigger_info": v["round2"]["trigger_info"],
                        "module_src": v["score"]["module_src"],
                    } for k, v in legs.items()},
                    "partial": True,
                }, f, indent=2, default=str)

    # Verdict
    log("\n" + "=" * 78)
    log("THESIS VERDICT  (Phase 17: generality)")
    log("=" * 78)

    full_tok = legs["full"]["token_totals"]["total_prompt_tokens"]
    casr_tok = legs["casr"]["token_totals"]["total_prompt_tokens"]
    abl_tok = legs["ablation"]["token_totals"]["total_prompt_tokens"]

    full_sc = legs["full"]["score"]["weighted_score"]
    casr_sc = legs["casr"]["score"]["weighted_score"]
    abl_sc = legs["ablation"]["score"]["weighted_score"]
    r1_sc = round1_score["weighted_score"]

    c1 = casr_tok < full_tok
    c2 = casr_sc >= full_sc - 0.05
    c3 = casr_sc > abl_sc + 0.05
    c4 = full_sc >= r1_sc
    c5 = legs["casr"]["n_skipped"] > 0

    # C7: per-pair pass fraction — CASR should do strictly better than
    # ablation on pair tests specifically (convention-propagation proof).
    casr_pair = legs["casr"]["mean_pair_pass"]
    abl_pair = legs["ablation"]["mean_pair_pass"]
    c7 = casr_pair > abl_pair + 0.20
    c7_delta = casr_pair - abl_pair

    log(f"  round-1 only:          score={r1_sc:.3f} "
        f"({round1_score['n_passed']}/25)")
    log(f"  full:     {full_tok:>6} tok   score={full_sc:.3f} "
        f"refined={legs['full']['n_refined']} skipped={legs['full']['n_skipped']}")
    log(f"  casr:     {casr_tok:>6} tok   score={casr_sc:.3f} "
        f"refined={legs['casr']['n_refined']} skipped={legs['casr']['n_skipped']}")
    log(f"  ablation: {abl_tok:>6} tok   score={abl_sc:.3f} "
        f"refined={legs['ablation']['n_refined']} skipped={legs['ablation']['n_skipped']}")
    log("")
    log(f"  C1 casr<full tok:         {c1}  "
        f"({(1 - casr_tok/max(full_tok,1))*100:.1f}% reduction)")
    log(f"  C2 casr>=full-0.05:       {c2}  (delta={casr_sc-full_sc:+.3f})")
    log(f"  C3 casr>ablation+0.05:    {c3}  (delta={casr_sc-abl_sc:+.3f})")
    log(f"  C4 full>=round1:          {c4}  (delta={full_sc-r1_sc:+.3f})")
    log(f"  C5 trigger fired skips:   {c5}  ({legs['casr']['n_skipped']} skips)")
    log(f"  C7 per-pair pass gap:     {c7}  "
        f"(casr pair={casr_pair:.3f} vs ablation pair={abl_pair:.3f}, "
        f"delta={c7_delta:+.3f}; gate ≥ +0.20)")
    log("")
    pre_reg = all([c1, c2, c3, c4, c5])
    log(f"  Pre-registered thesis {'HOLDS' if pre_reg else 'PARTIAL or FAILS'}; "
        f"generality claim C7 {'HOLDS' if c7 else 'FAILS'}.")
    log(f"  Overall: {'CASR IS GENERAL' if (pre_reg and c7) else 'needs deeper analysis'}")

    final = {
        "round1_score": round1_score,
        "tiers": tiers,
        "event_threshold": event_threshold,
        "trigger": trigger.name,
        "trigger_type": "behavior-fingerprint",
        "legs": {k: {
            "weighted_score": v["score"]["weighted_score"],
            "n_passed": v["score"]["n_passed"],
            "n_total": v["score"]["n_total"],
            "passed_tests": v["score"]["passed_tests"],
            "failed_tests": v["score"]["failed_tests"],
            "per_test": v["score"]["per_test"],
            "total_prompt_tokens": v["token_totals"]["total_prompt_tokens"],
            "total_completion_tokens": v["token_totals"]["total_completion_tokens"],
            "n_refined": v["n_refined"],
            "n_skipped": v["n_skipped"],
            "per_pair_pass": v["per_pair_pass"],
            "mean_pair_pass": v["mean_pair_pass"],
            "trigger_info": v["round2"]["trigger_info"],
            "routing_stats": v["round2"]["routing_stats"],
            "acceptance": v["round2"]["acceptance"],
            "module_src": v["score"]["module_src"],
        } for k, v in legs.items()},
        "thesis": {
            "C1_casr_fewer_tokens": c1,
            "C2_casr_quality_within_5pct": c2,
            "C3_casr_beats_random": c3,
            "C4_no_protocol_regression": c4,
            "C5_event_trigger_fires": c5,
            "C7_convention_propagation_beats_random_by_20pp": c7,
            "c7_delta": c7_delta,
            "all_core_hold": pre_reg,
            "generality_claim_holds": c7,
        },
        "model": model,
        "ablation_seed": ablation_seed,
        "partial": False,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        log(f"\nWrote {out_path}")

    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--out", default="vision_mvp/results_phase17.json")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--ablation-seed", type=int, default=42)
    ap.add_argument("--event-threshold", type=float, default=0.34)
    ap.add_argument("--force-refine", action="store_true",
                     help="bypass the event trigger: every tier-1 agent "
                          "refines. Isolates CASR routing from trigger.")
    args = ap.parse_args()
    run(
        model=args.model, out_path=args.out,
        max_retries=args.retries, ablation_seed=args.ablation_seed,
        event_threshold=args.event_threshold,
        force_refine=args.force_refine,
    )


if __name__ == "__main__":
    main()
