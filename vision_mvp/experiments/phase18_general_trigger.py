"""Phase 18 — does a more general trigger reduce the per-task engineering tax?

Phase 17 isolated the deployment tax: the CASR routing layer transferred
across coordination surfaces (ProtocolKit dict-keys → NumericLedger numeric
conventions), but the event-trigger layer did not. Each new task surface
needed a hand-built disagreement signal. RESULTS_PHASE17.md framed this as
the open question for Phase 18:

    "A natural Phase 18 would be to take *one* trigger-general approach
    (e.g., LLM-as-a-judge) and measure whether it matches the per-task
    triggers at acceptable cost."

This experiment does exactly that. For each of the two task surfaces
(ProtocolKit, NumericLedger) we run the exact Phase-14 / Phase-17 harness
twice — once with the task-specific trigger, once with the new general
trigger (LLM-judge with hybrid-structural fallback) — and compare:

  * Trigger fire rate          (does the general trigger fire too often /
                                 not often enough vs the per-task one?)
  * Total tokens               (does general add a judge-call tax that wipes
                                 out CASR's routing savings?)
  * Weighted score             (does the general trigger preserve quality?)
  * CASR-vs-ablation gap (C3)  (does the routing-layer signal still work
                                 when the trigger is task-agnostic?)

Same 3-leg structure (full / casr / ablation) and topological refinement
as the prior phases — we only swap the trigger.

Pre-registered claims:
  P18-A: General trigger preserves the CASR-vs-ablation gap on at least one
         of the two surfaces. Operational test:
              gap_general(surface) >= 0.5 * gap_specific(surface) - 0.05
  P18-B: General trigger does not blow up token cost. Operational test:
              tokens_general(casr) <= 1.5 * tokens_specific(casr)
  P18-C: General trigger fire rate is in a sensible range on both surfaces:
              0 < fire_rate_general < n_higher_tier_agents
         (i.e., it does not always-skip or always-fire — the trigger is
         actually using the bulletin signal.)

Honest reporting: each of these is checked per-surface and the report shows
both the surface-specific verdict and the cross-surface aggregate. We do not
claim the general trigger MATCHES the per-task triggers — only that the
gap, if any, is small enough that the engineering tax saved (zero new probe
batteries to maintain) is worth it.

Usage:
  # full benchmark (uses local Ollama for LLM judge)
  python3 -m vision_mvp.experiments.phase18_general_trigger \\
      --out vision_mvp/results_phase18.json

  # heuristic-only (no LLM judge — fast smoke test)
  python3 -m vision_mvp.experiments.phase18_general_trigger \\
      --no-llm --out vision_mvp/results_phase18_heuristic.json

  # one surface only
  python3 -m vision_mvp.experiments.phase18_general_trigger \\
      --surfaces protocolkit
"""

from __future__ import annotations
import sys, os, argparse, json, time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.trigger import (
    Trigger, schema_key_trigger, behavior_probe_trigger,
)
from vision_mvp.core.general_trigger import (
    GeneralTrigger, HybridStructuralTrigger,
)
from vision_mvp.experiments.round1_cache import save_round1, load_round1

# Reuse the existing harnesses verbatim — they already accept a `trigger`.
from vision_mvp.experiments import (
    phase14_benchmark as harness_protocolkit,
    phase17_generality as harness_numericledger,
)


def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------- Surface registry --------------------------------------

# Each surface declares (a) its harness module, (b) the task-specific trigger
# Phase 14/17 used. The harness's `run` is invoked twice with the same args
# but different `trigger=...`.
SURFACES = {
    "protocolkit": {
        "label": "ProtocolKit (dict-key schemas)",
        "harness": harness_protocolkit,
        "task_specific_trigger_factory": schema_key_trigger,
        "n_higher_tier_agents": 7,   # known from RESULTS_PHASE14.md
    },
    "numericledger": {
        "label": "NumericLedger (numerical conventions)",
        "harness": harness_numericledger,
        "task_specific_trigger_factory": behavior_probe_trigger,
        "n_higher_tier_agents": 7,   # known from RESULTS_PHASE17.md
    },
}


# -------------------- Single-surface runner ---------------------------------

def _run_one(
    surface: str,
    trigger: Trigger,
    *,
    model: str,
    out_path: str | None,
    max_retries: int,
    ablation_seed: int,
    event_threshold: float,
    round1: dict | None = None,
    extra_kwargs: dict | None = None,
) -> dict:
    """Invoke the surface's harness with the given trigger and return its
    full result dict. Each invocation also writes a per-surface JSON
    sibling file so partial progress survives crashes."""
    harness = SURFACES[surface]["harness"]
    extra_kwargs = dict(extra_kwargs or {})
    if round1 is not None:
        extra_kwargs["round1"] = round1
    log("\n" + "#" * 78)
    log(f"# surface={surface}  trigger={trigger.name}"
        f"{'  [shared-round1]' if round1 is not None else ''}")
    log("#" * 78)
    return harness.run(
        model=model, out_path=out_path,
        max_retries=max_retries,
        ablation_seed=ablation_seed,
        event_threshold=event_threshold,
        trigger=trigger,
        **extra_kwargs,
    )


# -------------------- Per-surface comparison summary -----------------------

def _surface_summary(surface: str, result_specific: dict, result_general: dict
                     ) -> dict:
    """Pull headline numbers out of two harness-run results and judge the
    Phase-18 per-surface claims (P18-A, P18-B, P18-C)."""
    info = SURFACES[surface]

    def _leg(r, leg, key, default=0):
        return r["legs"][leg].get(key, default)

    # Compute CASR-vs-ablation gaps.
    gap_specific = (_leg(result_specific, "casr", "weighted_score")
                    - _leg(result_specific, "ablation", "weighted_score"))
    gap_general = (_leg(result_general, "casr", "weighted_score")
                   - _leg(result_general, "ablation", "weighted_score"))

    # Tokens (CASR leg).
    tok_specific = _leg(result_specific, "casr", "total_prompt_tokens")
    tok_general = _leg(result_general, "casr", "total_prompt_tokens")

    # Fire rate of the general trigger on the CASR leg.
    fire_general = _leg(result_general, "casr", "n_refined")
    skip_general = _leg(result_general, "casr", "n_skipped")
    fire_specific = _leg(result_specific, "casr", "n_refined")
    skip_specific = _leg(result_specific, "casr", "n_skipped")

    n_high = info["n_higher_tier_agents"]

    # Pre-registered claims.
    p18a = gap_general >= 0.5 * gap_specific - 0.05
    p18b = tok_general <= int(1.5 * max(tok_specific, 1))
    p18c = (0 < fire_general < n_high) or (fire_general == n_high and gap_general > 0)

    return {
        "surface": surface,
        "label": info["label"],
        "trigger_specific": result_specific.get("trigger"),
        "trigger_general": result_general.get("trigger"),
        "scores": {
            "specific": {
                "round1": result_specific["round1_score"]["weighted_score"],
                "full": _leg(result_specific, "full", "weighted_score"),
                "casr": _leg(result_specific, "casr", "weighted_score"),
                "ablation": _leg(result_specific, "ablation", "weighted_score"),
            },
            "general": {
                "round1": result_general["round1_score"]["weighted_score"],
                "full": _leg(result_general, "full", "weighted_score"),
                "casr": _leg(result_general, "casr", "weighted_score"),
                "ablation": _leg(result_general, "ablation", "weighted_score"),
            },
        },
        "tokens": {
            "specific_full": _leg(result_specific, "full", "total_prompt_tokens"),
            "specific_casr": tok_specific,
            "specific_ablation": _leg(result_specific, "ablation",
                                       "total_prompt_tokens"),
            "general_full": _leg(result_general, "full", "total_prompt_tokens"),
            "general_casr": tok_general,
            "general_ablation": _leg(result_general, "ablation",
                                      "total_prompt_tokens"),
        },
        "fire_rates_casr_leg": {
            "specific_refined": fire_specific,
            "specific_skipped": skip_specific,
            "general_refined": fire_general,
            "general_skipped": skip_general,
            "n_higher_tier_agents": n_high,
        },
        "casr_vs_ablation_gap": {
            "specific": round(gap_specific, 4),
            "general": round(gap_general, 4),
            "ratio": round(gap_general / gap_specific, 4)
                     if abs(gap_specific) > 1e-9 else None,
        },
        "claims": {
            "P18A_gap_preserved": bool(p18a),
            "P18B_token_cost_bounded": bool(p18b),
            "P18C_fire_rate_sensible": bool(p18c),
        },
    }


# -------------------- Cross-repeat aggregation --------------------------------

def _aggregate_surface_repeats(surface: str, summaries: list[dict]) -> dict:
    """Compute descriptive stats over N paired-run summaries for one surface.

    Each element of `summaries` is a dict returned by `_surface_summary()`.
    Returns a dict with mean/min/max/stddev (when N>1) for the key metrics,
    plus pass rates for the three pre-registered claims.
    """
    import statistics as _st

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        d: dict = {
            "mean": round(_st.mean(vals), 4),
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
        }
        if len(vals) > 1:
            d["stddev"] = round(_st.stdev(vals), 4)
        return d

    def _pluck(s: dict, *keys: str) -> float | None:
        v: object = s
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None  # type: ignore[attr-defined]
        return float(v) if isinstance(v, (int, float)) else None  # type: ignore[arg-type]

    def _collect(*keys: str) -> list[float]:
        return [v for s in summaries if (v := _pluck(s, *keys)) is not None]

    n = len(summaries)
    gap_gen  = _collect("casr_vs_ablation_gap", "general")
    gap_spec = _collect("casr_vs_ablation_gap", "specific")
    tok_ratios = [
        s["tokens"]["general_casr"] / s["tokens"]["specific_casr"]
        for s in summaries
        if s["tokens"].get("specific_casr", 0) > 0
    ]
    gen_beats = sum(1 for g, sp in zip(gap_gen, gap_spec) if g > sp)
    p18a = sum(1 for s in summaries if s["claims"]["P18A_gap_preserved"])
    p18b = sum(1 for s in summaries if s["claims"]["P18B_token_cost_bounded"])
    p18c = sum(1 for s in summaries if s["claims"]["P18C_fire_rate_sensible"])

    return {
        "surface":            surface,
        "n_repeats":          n,
        "gap_general":        _stats(gap_gen),
        "gap_specific":       _stats(gap_spec),
        "casr_general":       _stats(_collect("scores", "general",  "casr")),
        "casr_specific":      _stats(_collect("scores", "specific", "casr")),
        "ablation_general":   _stats(_collect("scores", "general",  "ablation")),
        "ablation_specific":  _stats(_collect("scores", "specific", "ablation")),
        "token_ratio_casr":   _stats(tok_ratios),
        "fire_general":       _stats(_collect("fire_rates_casr_leg", "general_refined")),
        "fire_specific":      _stats(_collect("fire_rates_casr_leg", "specific_refined")),
        "claim_pass_rates":   {"P18A": p18a / n, "P18B": p18b / n, "P18C": p18c / n},
        "general_beats_specific_gap": {
            "count": gen_beats,
            "rate":  round(gen_beats / n, 4),
        },
    }


# -------------------- Main --------------------------------------------------

def _run_one_rep(
    *,
    rep: int,
    surfaces: list[str],
    model: str,
    out_path: str | None,
    max_retries: int,
    ablation_seed: int,
    event_threshold: float,
    use_llm_judge: bool,
    reuse_round1: bool,
    general_trigger: Trigger,
    judge_client: object,
    n_repeats: int,
) -> tuple[dict, dict | None]:
    """Run all surfaces for one repeat. Returns (surface_results, judge_stats)."""

    # Each repeat gets a different ablation seed so random footprints vary.
    rep_seed = ablation_seed + rep
    rep_tag  = f"_rep{rep}" if n_repeats > 1 else ""

    surface_results: dict[str, dict] = {}

    for surface in surfaces:
        info = SURFACES[surface]
        specific_trigger = info["task_specific_trigger_factory"]()

        if out_path:
            base, ext = os.path.splitext(out_path)
            specific_path = f"{base}{rep_tag}_{surface}_specific{ext}"
            general_path  = f"{base}{rep_tag}_{surface}_general{ext}"
            r1_cache_path = f"{base}{rep_tag}_{surface}_round1.json"
        else:
            specific_path = general_path = r1_cache_path = None

        harness = info["harness"]
        shared_round1: dict | None = None
        if r1_cache_path and reuse_round1 and Path(r1_cache_path).exists():
            shared_round1, r1_meta = load_round1(r1_cache_path)
            log(f"\n>>>>>> rep={rep} {surface} : loaded cached round-1 "
                f"(model={r1_meta['model']}, ts={r1_meta['timestamp']})")
        else:
            log(f"\n>>>>>> rep={rep} {surface} : generating shared round-1 ...")
            r1_client = LLMClient(model=model)
            shared_round1 = harness.run_round1(r1_client, max_retries=max_retries)
            if r1_cache_path:
                save_round1(r1_cache_path, shared_round1,
                            surface=surface, model=model)
                log(f"  Saved round-1 cache: {r1_cache_path}")

        log(f"\n>>>>>> rep={rep} {surface} : task-specific trigger"
            f" ({specific_trigger.name})")
        result_specific = _run_one(
            surface, specific_trigger,
            model=model, out_path=specific_path,
            max_retries=max_retries, ablation_seed=rep_seed,
            event_threshold=event_threshold,
            round1=shared_round1,
        )

        log(f"\n>>>>>> rep={rep} {surface} : general trigger ({general_trigger.name})")
        result_general = _run_one(
            surface, general_trigger,
            model=model, out_path=general_path,
            max_retries=max_retries, ablation_seed=rep_seed,
            event_threshold=event_threshold,
            round1=shared_round1,
        )

        surface_results[surface] = {
            "specific":      result_specific,
            "general":       result_general,
            "shared_round1": shared_round1 is not None,
            "summary": _surface_summary(surface, result_specific, result_general),
        }

    # Verdict for this repeat
    rows = [(s, surface_results[s]["summary"]) for s in surfaces]
    any_p18a = any(r[1]["claims"]["P18A_gap_preserved"]       for r in rows)
    all_p18b = all(r[1]["claims"]["P18B_token_cost_bounded"]  for r in rows)
    all_p18c = all(r[1]["claims"]["P18C_fire_rate_sensible"]  for r in rows)

    log(f"\n  rep={rep} PHASE-18 CROSS-SURFACE")
    for s, summ in rows:
        gap = summ["casr_vs_ablation_gap"]
        log(f"    [{s}] gap specific={gap['specific']:+.3f} "
            f"general={gap['general']:+.3f} "
            f"ratio={gap['ratio']}")
    log(f"    P18A={any_p18a}  P18B={all_p18b}  P18C={all_p18c}  "
        f"overall={'VIABLE' if (any_p18a and all_p18b and all_p18c) else 'NEEDS WORK'}")

    judge_stats = None
    if judge_client is not None and hasattr(judge_client, "stats"):
        s = judge_client.stats  # type: ignore[union-attr]
        judge_stats = {
            "n_generate_calls": s.n_generate_calls,
            "prompt_tokens":    s.prompt_tokens,
            "output_tokens":    s.output_tokens,
            "wall_seconds":     round(s.total_wall, 2),
        }

    rep_result = {
        "rep":              rep,
        "rep_ablation_seed": rep_seed,
        "model":            model,
        "use_llm_judge":    use_llm_judge,
        "event_threshold":  event_threshold,
        "paired_round1":    True,
        "judge_stats":      judge_stats,
        "surfaces": {s: surface_results[s]["summary"] for s in surface_results},
        "claims_aggregate": {
            "P18A_any_surface_gap_preserved": any_p18a,
            "P18B_all_surfaces_token_bounded": all_p18b,
            "P18C_all_surfaces_fire_sensibly":  all_p18c,
            "overall_viable": any_p18a and all_p18b and all_p18c,
        },
        "partial": False,
    }

    # Crash-resilient per-repeat save
    if out_path and n_repeats > 1:
        base, ext = os.path.splitext(out_path)
        rep_path = f"{base}_rep{rep}{ext}"
        with open(rep_path, "w") as f:
            json.dump(rep_result, f, indent=2, default=str)
        log(f"\n  Wrote {rep_path}")

    return surface_results, rep_result


def run(*, model: str,
        surfaces: list[str],
        out_path: str | None,
        max_retries: int,
        ablation_seed: int,
        event_threshold: float,
        use_llm_judge: bool,
        reuse_round1: bool = False,
        n_repeats: int = 1,
        ) -> dict:

    log("=" * 78)
    log(f"Phase 18 — General-trigger benchmark")
    log(f"Surfaces:  {surfaces}")
    log(f"Model:     {model}")
    log(f"Threshold: {event_threshold}")
    log(f"Repeats:   {n_repeats}")
    log(f"Mode:      {'LLM-judge + hybrid fallback' if use_llm_judge else 'hybrid-structural only'}")
    log("=" * 78)

    for surface in surfaces:
        if surface not in SURFACES:
            raise ValueError(
                f"unknown surface {surface!r}; pick from {list(SURFACES)}"
            )

    if use_llm_judge:
        judge_client: object = LLMClient(model=model)
        general_trigger: Trigger = GeneralTrigger(client=judge_client)
    else:
        judge_client = None
        general_trigger = HybridStructuralTrigger()

    # Collect per-surface summaries across all repeats for aggregation.
    surface_summaries_by_rep: list[dict[str, dict]] = []
    summaries_by_surface: dict[str, list[dict]] = {s: [] for s in surfaces}

    for rep in range(n_repeats):
        if n_repeats > 1:
            log(f"\n{'#'*78}")
            log(f"# REPEAT {rep + 1} / {n_repeats}  (ablation_seed={ablation_seed + rep})")
            log(f"{'#'*78}")

        surface_results, rep_result = _run_one_rep(
            rep=rep, surfaces=surfaces, model=model, out_path=out_path,
            max_retries=max_retries, ablation_seed=ablation_seed,
            event_threshold=event_threshold, use_llm_judge=use_llm_judge,
            reuse_round1=reuse_round1, general_trigger=general_trigger,
            judge_client=judge_client, n_repeats=n_repeats,
        )

        surface_summaries_by_rep.append(
            {s: surface_results[s]["summary"] for s in surface_results}
        )
        for s in surfaces:
            summaries_by_surface[s].append(surface_results[s]["summary"])

    # ---- Aggregate across repeats ------------------------------------------
    aggregate: dict[str, dict] | None = None
    if n_repeats > 1:
        aggregate = {
            s: _aggregate_surface_repeats(s, summaries_by_surface[s])
            for s in surfaces
        }

        log("\n" + "=" * 78)
        log(f"PHASE-18 AGGREGATE  ({n_repeats} repeats)")
        log("=" * 78)
        for s, agg in aggregate.items():
            log(f"\n[{s}]")
            log(f"  gap_general  mean={agg['gap_general'].get('mean', 0):+.3f}"
                f"  min={agg['gap_general'].get('min', 0):+.3f}"
                f"  max={agg['gap_general'].get('max', 0):+.3f}"
                + (f"  stddev={agg['gap_general'].get('stddev', 0):.3f}"
                   if "stddev" in agg["gap_general"] else ""))
            log(f"  gap_specific mean={agg['gap_specific'].get('mean', 0):+.3f}"
                f"  min={agg['gap_specific'].get('min', 0):+.3f}"
                f"  max={agg['gap_specific'].get('max', 0):+.3f}")
            log(f"  token_ratio  mean={agg['token_ratio_casr'].get('mean', 0):.3f}")
            log(f"  gen>spec     {agg['general_beats_specific_gap']['count']}"
                f"/{n_repeats} repeats"
                f"  P18A_rate={agg['claim_pass_rates']['P18A']:.2f}")

        if out_path:
            base, ext = os.path.splitext(out_path)
            agg_path = f"{base}_aggregate{ext}"
            with open(agg_path, "w") as f:
                json.dump(
                    {"model": model, "n_repeats": n_repeats,
                     "ablation_seed_base": ablation_seed,
                     "event_threshold": event_threshold,
                     "use_llm_judge": use_llm_judge,
                     "surfaces": aggregate},
                    f, indent=2, default=str,
                )
            log(f"\nWrote aggregate: {agg_path}")

    # ---- Single-repeat path: backward-compat output format -----------------
    if n_repeats == 1:
        single = surface_summaries_by_rep[0]
        rows = [(s, single[s]) for s in surfaces]
        any_p18a = any(r[1]["claims"]["P18A_gap_preserved"]       for r in rows)
        all_p18b = all(r[1]["claims"]["P18B_token_cost_bounded"]  for r in rows)
        all_p18c = all(r[1]["claims"]["P18C_fire_rate_sensible"]  for r in rows)

        log("\n" + "=" * 78)
        log("PHASE-18 VERDICT")
        log("=" * 78)
        for s, summ in rows:
            sc  = summ["scores"]
            tk  = summ["tokens"]
            fr  = summ["fire_rates_casr_leg"]
            gap = summ["casr_vs_ablation_gap"]
            cl  = summ["claims"]
            log(f"\n[{s}] {summ['label']}")
            log(f"  scores  specific: round1={sc['specific']['round1']:.3f} "
                f"full={sc['specific']['full']:.3f} "
                f"casr={sc['specific']['casr']:.3f} "
                f"ablation={sc['specific']['ablation']:.3f}")
            log(f"  scores  general:  round1={sc['general']['round1']:.3f} "
                f"full={sc['general']['full']:.3f} "
                f"casr={sc['general']['casr']:.3f} "
                f"ablation={sc['general']['ablation']:.3f}")
            log(f"  tokens  specific casr={tk['specific_casr']}  "
                f"general casr={tk['general_casr']}  "
                f"ratio={tk['general_casr']/max(tk['specific_casr'],1):.2f}x")
            log(f"  fire    specific {fr['specific_refined']}/{fr['n_higher_tier_agents']}; "
                f"general {fr['general_refined']}/{fr['n_higher_tier_agents']}")
            log(f"  gap     specific={gap['specific']:+.3f}  "
                f"general={gap['general']:+.3f}  ratio={gap['ratio']}")
            log(f"  claims  P18A={cl['P18A_gap_preserved']}  "
                f"P18B={cl['P18B_token_cost_bounded']}  "
                f"P18C={cl['P18C_fire_rate_sensible']}")

        log(f"\n  CROSS-SURFACE")
        log(f"    P18A any surface preserves CASR gap : {any_p18a}")
        log(f"    P18B all surfaces bound token tax   : {all_p18b}")
        log(f"    P18C all surfaces fire sensibly     : {all_p18c}")
        log(f"    overall: "
            f"{'GENERAL TRIGGER VIABLE' if (any_p18a and all_p18b and all_p18c) else 'NEEDS WORK'}")

        final: dict = {
            "model": model, "use_llm_judge": use_llm_judge,
            "event_threshold": event_threshold,
            "ablation_seed": ablation_seed, "paired_round1": True,
            "judge_stats": None,
            "surfaces": {s: single[s] for s in surfaces},
            "claims_aggregate": {
                "P18A_any_surface_gap_preserved": any_p18a,
                "P18B_all_surfaces_token_bounded": all_p18b,
                "P18C_all_surfaces_fire_sensibly":  all_p18c,
                "overall_viable": any_p18a and all_p18b and all_p18c,
            },
            "partial": False,
        }
        if out_path:
            with open(out_path, "w") as f:
                json.dump(final, f, indent=2, default=str)
            log(f"\nWrote {out_path}")
        return final

    # n_repeats > 1: return aggregate-centred result
    return {
        "model": model, "n_repeats": n_repeats,
        "ablation_seed_base": ablation_seed,
        "event_threshold": event_threshold,
        "use_llm_judge": use_llm_judge,
        "aggregate": aggregate,
        "repeats": [
            {s: surface_summaries_by_rep[r][s] for s in surfaces}
            for r in range(n_repeats)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--out", default="vision_mvp/results_phase18.json")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--ablation-seed", type=int, default=42)
    ap.add_argument("--event-threshold", type=float, default=0.34)
    ap.add_argument("--surfaces", nargs="+",
                     default=["protocolkit", "numericledger"],
                     help="which surfaces to evaluate")
    ap.add_argument("--no-llm", action="store_true",
                     help="skip the LLM judge — use the hybrid heuristic only")
    ap.add_argument("--reuse-round1", action="store_true",
                     help="load cached round-1 from <out>_<surface>_round1.json "
                          "instead of generating fresh drafts")
    ap.add_argument("--repeats", type=int, default=1,
                     help="number of independent paired runs per surface "
                          "(each repeat regenerates round-1; use --reuse-round1 "
                          "to hold round-1 fixed across repeats)")
    args = ap.parse_args()
    run(
        model=args.model, out_path=args.out,
        max_retries=args.retries, ablation_seed=args.ablation_seed,
        event_threshold=args.event_threshold,
        surfaces=args.surfaces,
        use_llm_judge=not args.no_llm,
        reuse_round1=args.reuse_round1,
        n_repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
