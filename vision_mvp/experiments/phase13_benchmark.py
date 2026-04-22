"""Phase 13 — 3-leg CASR benchmark on the ProtocolKit co-design task.

Rerun of Phase 12 with a task where round-2 coordination actually matters.
Phase 12 had a ceiling problem: each agent's round-1 draft was independently
sufficient, so no leg could be distinguished from any other. Phase 13 fixes
this by deliberately UNDERSPECIFYING pairwise dict schemas — producers invent
conventions, consumers must read them. Round-1 agents guess; only CASR
routing reliably connects the right producer to the right consumer.

Task: 12 LLM agents each write one function of ProtocolKit. 25 automated
tests pass only if producer-consumer pairs agree on schema.

Protocol:
  Round 1 — every agent drafts independently (no bulletin, same across all legs).
  Round 2 — every agent sees a "coordination bulletin" with drafts from other
            agents. The bulletin's *contents* differ per leg:

    full      — bulletin = all 11 other agents' drafts                    [baseline]
    casr      — bulletin = only agents in this recipient's causal         [CASR]
                footprint (call-graph neighbors, forward and reverse)
    ablation  — bulletin = random subset of same cardinality as CASR      [control]

Measure per leg:
  - total prompt tokens   (what each agent had to consume)
  - total completion tok  (what each agent produced)
  - 31-test weighted score on composed module
  - per-agent per-round tokens
  - how many messages the router delivered vs dropped

Falsifiable claims (each decides 'yes' or 'no' empirically):
  C1: casr prompt tokens < full prompt tokens          (should drop substantially)
  C2: casr weighted score >= full - 0.05               (no significant quality drop)
  C3: casr weighted score > ablation + 0.05            (causal structure matters)

If C1 fails, the routing didn't actually route.
If C2 fails, the routing dropped useful information.
If C3 fails, any k-subset works equally well — the causal selection is not what's
            doing the work (thesis is weakened).

Usage:
    python -m vision_mvp.experiments.phase13_benchmark
    python -m vision_mvp.experiments.phase13_benchmark --out vision_mvp/results_phase13.json
"""

from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.code_harness import (
    extract_code, function_is_defined, run_sandboxed,
)
from vision_mvp.core.causal_footprint import (
    footprint_from_call_graph, random_footprint,
)
from vision_mvp.core.casr_router import CASRRouter, RouterMessage
from vision_mvp.core.token_meter import count_tokens
from vision_mvp.tasks.protocol_codesign import (
    FUNCTION_SPECS, SPEC_ORDER, CALL_GRAPH,
    TEST_RUNNER_SRC, compose_module, score_tests, agent_prompt,
)


def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------- Round 1: independent drafts (shared across legs) -------

def run_round1(client: LLMClient, max_retries: int = 2) -> dict:
    """Every agent drafts independently. Returns a dict with drafts, tokens,
    acceptance info."""
    log("\n" + "=" * 78)
    log("[round 1] independent drafts (shared across all legs)")
    log("=" * 78)

    drafts: dict[str, str] = {}
    tokens: dict[str, dict] = {}
    acceptance: dict[str, dict] = {}

    for specialty in SPEC_ORDER:
        target_name = FUNCTION_SPECS[specialty]["name"]
        log(f"  - {specialty}: drafting ...")
        prompt = agent_prompt(specialty)  # no bulletin
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
                log(f"    accepted attempt {attempt+1}  ({wall:.1f}s, "
                    f"prompt={prompt_toks}t, completion={out_tokens}t)")
                break
            log(f"    attempt {attempt+1} failed — "
                f"{'no code' if not code else 'missing fn name'}")

        drafts[specialty] = accepted or ""
        tokens[specialty] = {"prompt": prompt_toks, "completion": out_tokens}
        acceptance[specialty] = {
            "accepted": accepted is not None,
            "attempts": n_attempts,
        }

    return {
        "drafts": drafts,
        "tokens": tokens,
        "acceptance": acceptance,
    }


# -------------------- Round 2: coordinated refinement per leg --------------

def round2_prompt(specialty: str, own_draft: str,
                  bulletin: dict[str, str]) -> str:
    """Prompt for round 2: agent sees its own round-1 draft + a filtered
    bulletin of teammates' drafts, and produces a (possibly-refined) final."""
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 12-person Python team "
        f"building ProtocolKit — small typed primitives where pairs of "
        f"agents must agree on private dict schemas.",
        f"Your function: `{s['name']}` with signature {s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Your own ROUND-1 draft:\n```python\n{own_draft or '# (failed to draft)'}\n```",
    ]
    if bulletin:
        parts.append(
            f"COORDINATION BULLETIN — drafts from {len(bulletin)} teammate(s). "
            f"Your function may call some of these; use the EXACT names and "
            f"signatures shown:"
        )
        for dep, src in bulletin.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(
                f"  - `{dep_name}` (from {dep}):\n"
                f"```python\n{src[:500]}\n```"
            )
    else:
        parts.append(
            "COORDINATION BULLETIN: (empty — no teammates' drafts routed to you)"
        )
    parts.append(
        f"Output ONLY the final refined version of `{s['name']}` in a "
        f"```python code fence. No commentary. No re-defining teammates' "
        f"functions. Match their signatures as shown."
    )
    return "\n\n".join(parts)


def run_round2(client: LLMClient, round1_drafts: dict[str, str],
               router: CASRRouter, max_retries: int = 2) -> dict:
    """Each agent gets a filtered bulletin + refines their draft."""
    # Build the full message pool once (round-1 drafts as RouterMessages)
    pool: list[RouterMessage] = []
    for src_spec, src_code in round1_drafts.items():
        if src_code:  # only route non-empty drafts
            pool.append(RouterMessage(
                source_id=src_spec,
                payload=src_code,
                tokens=count_tokens(src_code),
            ))

    drafts: dict[str, str] = {}
    tokens: dict[str, dict] = {}
    acceptance: dict[str, dict] = {}
    routing_stats: dict[str, dict] = {}

    for specialty in SPEC_ORDER:
        target_name = FUNCTION_SPECS[specialty]["name"]
        # Route
        delivered, stats = router.route(pool, recipient_id=specialty)
        bulletin = {m.source_id: m.payload for m in delivered}
        log(f"  - {specialty}: router delivered {stats.delivered}/"
            f"{stats.delivered + stats.dropped} messages "
            f"({stats.delivered_tokens}t in, {stats.dropped_tokens}t dropped)")

        prompt = round2_prompt(specialty, round1_drafts.get(specialty, ""),
                                bulletin)
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
                log(f"      accepted attempt {attempt+1}  ({wall:.1f}s, "
                    f"prompt={prompt_toks}t, completion={out_tokens}t)")
                break
            log(f"      attempt {attempt+1} failed — "
                f"{'no code' if not code else 'missing fn name'}")

        # Fall back to round-1 draft if round-2 fails entirely
        if accepted is None:
            accepted = round1_drafts.get(specialty, "")
            log(f"      fell back to round-1 draft")

        drafts[specialty] = accepted
        tokens[specialty] = {"prompt": prompt_toks, "completion": out_tokens}
        acceptance[specialty] = {
            "accepted": bool(accepted),
            "attempts": n_attempts,
            "fell_back_to_round1": accepted == round1_drafts.get(specialty, "") and n_attempts > 0,
        }
        routing_stats[specialty] = {
            "delivered": stats.delivered,
            "dropped": stats.dropped,
            "delivered_tokens": stats.delivered_tokens,
            "dropped_tokens": stats.dropped_tokens,
        }

    return {
        "drafts": drafts,
        "tokens": tokens,
        "acceptance": acceptance,
        "routing_stats": routing_stats,
    }


# -------------------- Score a set of drafts ---------------------------------

def score_drafts(drafts: dict[str, str], label: str) -> dict:
    module_src = compose_module(drafts)
    log(f"\n[score:{label}] composing module ({len(module_src)} chars)")
    result = run_sandboxed(module_src, TEST_RUNNER_SRC, timeout_s=30)
    score = score_tests(result.per_test)
    log(f"[score:{label}] {score['n_passed']}/{score['n_total']} passed "
        f"(weighted {score['weighted_score']})")
    if result.syntax_error:
        log(f"[score:{label}] SYNTAX ERROR in composed module")
    if result.timed_out:
        log(f"[score:{label}] timed out")
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


# -------------------- Footprint construction --------------------------------

def build_causal_footprints() -> dict:
    """Build a CausalFootprint for each specialty from CALL_GRAPH."""
    return {
        spec: footprint_from_call_graph(CALL_GRAPH, spec)
        for spec in SPEC_ORDER
    }


def build_random_footprints(reference_footprints: dict, seed: int) -> dict:
    """Build per-agent random footprints with same size as the CASR footprint
    for that agent. Ensures a fair 'same volume, different selection' ablation.
    """
    out = {}
    for i, spec in enumerate(SPEC_ORDER):
        size = len(reference_footprints[spec])
        out[spec] = random_footprint(
            list(SPEC_ORDER), size=size, seed=seed + i,
        )
    return out


# -------------------- Per-leg summary ---------------------------------------

def summarize_tokens(round1: dict, round2: dict) -> dict:
    """Sum tokens across both rounds."""
    total_prompt = 0
    total_completion = 0
    for spec in SPEC_ORDER:
        total_prompt += round1["tokens"][spec]["prompt"]
        total_prompt += round2["tokens"][spec]["prompt"]
        total_completion += round1["tokens"][spec]["completion"]
        total_completion += round2["tokens"][spec]["completion"]
    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }


# -------------------- Main ---------------------------------------------------

def run(model: str, out_path: str | None, max_retries: int, ablation_seed: int) -> dict:
    client = LLMClient(model=model)
    log(f"Phase 12 — 3-leg CASR benchmark")
    log(f"Model: {model}")
    log(f"Agents: {len(SPEC_ORDER)}")
    log(f"Tests: 25 (weighted score in [0, 1])")

    # ---- Round 1 (shared) ------------------------------------------------
    round1 = run_round1(client, max_retries=max_retries)

    # Score round-1 as a "no-coordination" baseline
    round1_score = score_drafts(round1["drafts"], "round1_no_coord")

    # ---- Footprints ------------------------------------------------------
    causal_fps = build_causal_footprints()
    random_fps = build_random_footprints(causal_fps, seed=ablation_seed)
    fp_sizes = {s: len(causal_fps[s]) for s in SPEC_ORDER}
    log(f"\n[footprints] avg size = {sum(fp_sizes.values())/len(fp_sizes):.1f}, "
        f"max = {max(fp_sizes.values())}, per-agent: {fp_sizes}")

    # ---- Round 2 x 3 legs ------------------------------------------------
    legs = {}
    for mode, fps in [
        ("full", causal_fps),        # fps unused for "full"
        ("casr", causal_fps),
        ("ablation", random_fps),
    ]:
        log("\n" + "=" * 78)
        log(f"[round 2 / {mode}] coordinated refinement")
        log("=" * 78)
        router = CASRRouter(mode=mode, footprints=fps)
        r2 = run_round2(client, round1["drafts"], router, max_retries=max_retries)
        leg_score = score_drafts(r2["drafts"], f"round2_{mode}")
        token_totals = summarize_tokens(round1, r2)
        legs[mode] = {
            "round2": r2,
            "score": leg_score,
            "token_totals": token_totals,
        }

        # Write partial results after each leg (crash-resilient)
        if out_path:
            partial = {
                "round1_score": round1_score,
                "round1_tokens_per_agent": round1["tokens"],
                "round1_acceptance": round1["acceptance"],
                "footprint_sizes": fp_sizes,
                "legs": {k: {
                    "weighted_score": v["score"]["weighted_score"],
                    "n_passed": v["score"]["n_passed"],
                    "n_total": v["score"]["n_total"],
                    "passed_tests": v["score"]["passed_tests"],
                    "failed_tests": v["score"]["failed_tests"],
                    "per_test": v["score"]["per_test"],
                    "total_prompt_tokens": v["token_totals"]["total_prompt_tokens"],
                    "total_completion_tokens": v["token_totals"]["total_completion_tokens"],
                    "routing_stats": v["round2"]["routing_stats"],
                    "round2_tokens_per_agent": v["round2"]["tokens"],
                    "round2_acceptance": v["round2"]["acceptance"],
                    "module_src": v["score"]["module_src"],
                    "stderr": v["score"]["stderr"],
                } for k, v in legs.items()},
                "partial": True,
            }
            with open(out_path, "w") as f:
                json.dump(partial, f, indent=2, default=str)
            log(f"[saved partial] {out_path}")

    # ---- Thesis verdict --------------------------------------------------
    log("\n" + "=" * 78)
    log("THESIS VERDICT")
    log("=" * 78)

    full_tok = legs["full"]["token_totals"]["total_prompt_tokens"]
    casr_tok = legs["casr"]["token_totals"]["total_prompt_tokens"]
    abl_tok = legs["ablation"]["token_totals"]["total_prompt_tokens"]

    full_sc = legs["full"]["score"]["weighted_score"]
    casr_sc = legs["casr"]["score"]["weighted_score"]
    abl_sc = legs["ablation"]["score"]["weighted_score"]

    c1 = casr_tok < full_tok
    c2 = casr_sc >= full_sc - 0.05
    c3 = casr_sc > abl_sc + 0.05

    log(f"  full:     {full_tok:>7} prompt tok   score={full_sc:.3f} ({legs['full']['score']['n_passed']}/25)")
    log(f"  casr:     {casr_tok:>7} prompt tok   score={casr_sc:.3f} ({legs['casr']['score']['n_passed']}/25)")
    log(f"  ablation: {abl_tok:>7} prompt tok   score={abl_sc:.3f} ({legs['ablation']['score']['n_passed']}/25)")
    log(f"  round-1 only (no coord): score={round1_score['weighted_score']:.3f} "
        f"({round1_score['n_passed']}/25)")
    log("")
    reduction = (full_tok - casr_tok) / max(full_tok, 1)
    log(f"  C1 (casr_tok < full_tok):            {c1}  (reduction = {reduction*100:.1f}%)")
    log(f"  C2 (casr_sc >= full_sc - 0.05):      {c2}  (delta = {casr_sc - full_sc:+.3f})")
    log(f"  C3 (casr_sc > ablation_sc + 0.05):   {c3}  (delta = {casr_sc - abl_sc:+.3f})")
    log("")
    all_hold = c1 and c2 and c3
    log(f"  Thesis {'HOLDS' if all_hold else 'FAILS'} on this run.")

    final = {
        "round1_score": round1_score,
        "round1_tokens_per_agent": round1["tokens"],
        "round1_acceptance": round1["acceptance"],
        "footprint_sizes": fp_sizes,
        "legs": {k: {
            "weighted_score": v["score"]["weighted_score"],
            "n_passed": v["score"]["n_passed"],
            "n_total": v["score"]["n_total"],
            "passed_tests": v["score"]["passed_tests"],
            "failed_tests": v["score"]["failed_tests"],
            "per_test": v["score"]["per_test"],
            "total_prompt_tokens": v["token_totals"]["total_prompt_tokens"],
            "total_completion_tokens": v["token_totals"]["total_completion_tokens"],
            "routing_stats": v["round2"]["routing_stats"],
            "round2_tokens_per_agent": v["round2"]["tokens"],
            "round2_acceptance": v["round2"]["acceptance"],
            "module_src": v["score"]["module_src"],
            "stderr": v["score"]["stderr"],
        } for k, v in legs.items()},
        "thesis": {
            "C1_casr_fewer_tokens_than_full": c1,
            "C2_casr_quality_within_5pct_of_full": c2,
            "C3_casr_beats_random_by_5pct": c3,
            "all_hold": all_hold,
            "prompt_token_reduction_pct": round(reduction * 100, 2),
        },
        "llm_total_calls": client.stats.n_generate_calls,
        "llm_total_tokens_meter": client.stats.total_tokens(),
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
    ap.add_argument("--out", default="vision_mvp/results_phase12.json")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--ablation-seed", type=int, default=42)
    args = ap.parse_args()
    run(model=args.model, out_path=args.out,
        max_retries=args.retries, ablation_seed=args.ablation_seed)


if __name__ == "__main__":
    main()
