"""Phase 44 — Raw-text residue driver and refined semantic taxonomy
report.

Phase 43 shipped the nine-label Phase-43 semantic taxonomy and the
analysis-only ``phase43_frontier_headroom`` driver. That driver
consumed Phase-42-shape artifacts which did **not** preserve raw
LLM output, so the classifier worked on a sentinel ``proposed_patch``
for every non-parse / non-match cell — collapsing several refined
buckets into ``SEM_WRONG_EDIT_SITE`` (§ D.7 of ``RESULTS_PHASE43.md``).

Phase 44 closes that gap with two concrete pieces:

  1. **Raw-capture sweep path**, driven through the new
     ``swe_raw_capture.make_capturing_generator`` wrapper. The driver
     runs the same ``phase42_parser_sweep``-shape experiment but writes
     a companion JSON artifact containing one row per measurement with
     the raw LLM text, the parser outcome dict, the parsed
     substitutions, the applied substitutions (when the matcher
     accepted them), and the SHA-256 of the resulting patched source.
     The driver can target the ASPEN cluster just like the Phase-42
     driver (``--ollama-url``).
  2. **Refined-taxonomy analysis**. The driver also operates in a
     pure analysis mode (``--analyse-only``) that consumes a parent
     Phase-42 artifact + its raw-capture companion and emits a
     Phase-44 report JSON: pass@1 table, parser-compliance summary,
     semantic counter under **both** the Phase-43 coarse classifier
     (backwards-compatible) and the Phase-44 refined classifier. The
     delta table — which coarse labels split into which refined
     labels — is the Phase-44 headline.

Architectural discipline
------------------------

  * The Phase-39..43 bridge, parser, matcher, substrate, and sandbox
    paths are *untouched*. Raw capture is a driver-side wrapper that
    sits above the existing ``llm_patch_generator`` factory.
  * The Phase-43 ``classify_semantic_outcome`` is preserved byte-for-
    byte. ``classify_semantic_outcome_v2`` subsumes it on matched
    inputs (Theorem P44-2) but never *replaces* the v1 classifier;
    downstream reports include both histograms so a reviewer can
    verify the refinement is a strict partition.

Reproducible runs
-----------------

    # 1. Cluster mac1 — full 57-instance bank @ qwen2.5-coder:14b,
    #    raw capture on. This is the Phase-44 headline coder cell.
    python3 -m vision_mvp.experiments.phase44_semantic_residue \\
        --mode real --model qwen2.5-coder:14b \\
        --ollama-url http://192.168.12.191:11434 \\
        --sandbox subprocess \\
        --parser-modes strict robust --apply-modes strict \\
        --n-distractors 6 --think default --max-tokens 400 \\
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \\
        --out-parent vision_mvp/results_phase44_parser_14b_coder.json \\
        --out-capture vision_mvp/results_phase44_capture_14b_coder.json

    # 2. Cluster mac2 — secondary comparison. On the 2-mac cluster
    #    we run qwen3.5:35b (the Phase-43 frontier datapoint) with
    #    raw capture, so its 2/57 residue can be refined.
    python3 -m vision_mvp.experiments.phase44_semantic_residue \\
        --mode real --model qwen3.5:35b \\
        --ollama-url http://192.168.12.248:11434 \\
        --sandbox subprocess \\
        --parser-modes strict robust --apply-modes strict \\
        --n-distractors 6 --think off --max-tokens 600 \\
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \\
        --out-parent vision_mvp/results_phase44_parser_35b_moe.json \\
        --out-capture vision_mvp/results_phase44_capture_35b_moe.json

    # 3. Pure analysis: consume both artifacts + bank and emit the
    #    refined-taxonomy report.
    python3 -m vision_mvp.experiments.phase44_semantic_residue \\
        --analyse-only \\
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \\
        --artifacts \\
           vision_mvp/results_phase44_parser_14b_coder.json \\
           vision_mvp/results_phase44_parser_35b_moe.json \\
        --captures \\
           vision_mvp/results_phase44_capture_14b_coder.json \\
           vision_mvp/results_phase44_capture_35b_moe.json \\
        --out vision_mvp/results_phase44_refined_summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_APPLY_MODES, ALL_SWE_STRATEGIES, APPLY_MODE_STRICT,
    ParserComplianceCounter, apply_patch, build_synthetic_event_log,
    deterministic_oracle_generator, load_jsonl_bank,
    parse_unified_diff,
)
from vision_mvp.tasks.swe_patch_parser import (
    ALL_PARSER_MODES, PARSER_ROBUST, PARSER_STRICT, PARSER_UNIFIED,
    parse_patch_block,
)
from vision_mvp.tasks.swe_raw_capture import (
    RawCaptureStore, make_capturing_generator,
)
from vision_mvp.tasks.swe_sandbox import (
    SubprocessSandbox, run_swe_loop_sandboxed, select_sandbox,
)
from vision_mvp.tasks.swe_semantic_taxonomy import (
    ALL_REFINED_LABELS, ALL_SEMANTIC_LABELS, ALL_SEMANTIC_LABELS_V2,
    REFINEMENT_MAP, SEM_OK, SEM_PARSE_FAIL, SEM_WRONG_EDIT_SITE,
    SemanticCounter, classify_semantic_outcome,
    classify_semantic_outcome_v2,
)


_DEFAULT_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl")


# =============================================================================
# Cluster-side path — run the sweep with raw capture on
# =============================================================================


def _make_raw_capture_generator(*,
                                   mode: str,
                                   model: str,
                                   ollama_url: str | None,
                                   timeout: float,
                                   parser_mode: str,
                                   apply_mode: str,
                                   n_distractors: int,
                                   prompt_style: str,
                                   think,
                                   max_tokens: int,
                                   store: RawCaptureStore,
                                   counter: ParserComplianceCounter,
                                   shared_llm_client=None,
                                   shared_raw_cache: dict | None = None,
                                   ):
    """Compose the Phase-44 capture-aware generator.

    ``mode`` in ``{"mock", "real"}``. On ``mock``, the oracle
    generator is used and the raw-text field is empty (the
    oracle returns a proposed patch directly, without going
    through an LLM). The oracle path is still captured so a
    mock sweep produces a valid Phase-44 artifact for
    regression testing.
    """
    if mode == "mock":
        # Wrap the oracle — no LLM call. The capturing generator
        # still records the proposed patch (equal to gold) and
        # assigns an "ok" parse outcome.
        return make_capturing_generator(
            deterministic_oracle_generator,
            store=store,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=n_distractors,
            llm_call=None,
            parser_counter=counter,
            prompt_style=prompt_style,
            unified_diff_parser=parse_unified_diff,
            shared_raw_cache=shared_raw_cache,
        ), None

    if shared_llm_client is not None:
        client = shared_llm_client
    else:
        from vision_mvp.core.llm_client import LLMClient
        think_arg: bool | None
        if think == "on":
            think_arg = True
        elif think == "off":
            think_arg = False
        else:
            think_arg = None
        client = LLMClient(model=model, timeout=timeout,
                            base_url=ollama_url, think=think_arg)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_tokens,
                                temperature=0.0)

    gen = make_capturing_generator(
        base_gen=None,  # unused in shape (2); llm_call drives it
        store=store,
        parser_mode=parser_mode,
        apply_mode=apply_mode,
        n_distractors=n_distractors,
        llm_call=_call,
        parser_counter=counter,
        prompt_style=prompt_style,
        unified_diff_parser=parse_unified_diff,
        shared_raw_cache=shared_raw_cache,
    )
    return gen, client


def _run_sweep(args) -> int:
    """Phase-42-shape sweep with raw capture."""
    jsonl_path = args.jsonl
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"--jsonl {jsonl_path} not found")

    sandbox = select_sandbox(args.sandbox,
                               docker_image=args.docker_image)
    print(f"[phase44] sandbox={sandbox.name()} (asked={args.sandbox})",
          flush=True)
    if args.sandbox == "docker" and not sandbox.is_available():
        print("[phase44] WARNING: docker unavailable; using subprocess.",
              flush=True)
        sandbox = SubprocessSandbox()

    # Ensure strict parser runs first so the attribution table is
    # well-defined.
    valid_parser = set(ALL_PARSER_MODES) | {"none"}
    for pm in args.parser_modes:
        if pm not in valid_parser:
            raise SystemExit(
                f"--parser-modes: unknown {pm!r}; valid: {sorted(valid_parser)}")
    if PARSER_STRICT in args.parser_modes:
        args.parser_modes = [PARSER_STRICT] + [
            p for p in args.parser_modes if p != PARSER_STRICT]
    for m in args.apply_modes:
        if m not in ALL_APPLY_MODES:
            raise SystemExit(
                f"--apply-modes: unknown {m!r}; valid: {sorted(ALL_APPLY_MODES)}")

    store = RawCaptureStore(meta={
        "model": args.model if args.mode == "real" else "<oracle>",
        "ollama_url": args.ollama_url or "localhost",
        "mode": args.mode,
        "jsonl": os.path.basename(jsonl_path),
    })

    counters: dict[tuple[str, int], ParserComplianceCounter] = {}
    cells: list[dict] = []
    overall_start = time.time()

    # One shared LLMClient for every cell → stats aggregate, VRAM
    # stays hot. One shared raw-response cache keyed to
    # (instance, strat_proxy, nd, prompt_style) so parser-mode cells
    # reuse the same LLM output bytes (Phase 42 discipline extended
    # through Phase 44).
    _shared_client = None
    if args.mode == "real":
        from vision_mvp.core.llm_client import LLMClient
        think_arg: bool | None
        if args.think == "on":
            think_arg = True
        elif args.think == "off":
            think_arg = False
        else:
            think_arg = None
        _shared_client = LLMClient(
            model=args.model, timeout=args.llm_timeout,
            base_url=args.ollama_url, think=think_arg)
    shared_raw_cache: dict = {}

    for parser_mode in args.parser_modes:
        for apply_mode in args.apply_modes:
            for n_distractors in args.n_distractors:
                tasks, repo_files = load_jsonl_bank(
                    jsonl_path,
                    hidden_event_log_factory=(
                        lambda t, k=n_distractors:
                            build_synthetic_event_log(t, k)),
                    limit=args.n_instances,
                )
                counter = counters.setdefault(
                    (parser_mode, n_distractors),
                    ParserComplianceCounter())
                cell_label = (
                    f"parser={parser_mode} apply={apply_mode} "
                    f"nd={n_distractors}")
                print(f"\n[phase44] {cell_label} "
                      f"jsonl={os.path.basename(jsonl_path)} "
                      f"n_instances={len(tasks)}", flush=True)
                t0 = time.time()

                gen, client = _make_raw_capture_generator(
                    mode=args.mode, model=args.model,
                    ollama_url=args.ollama_url,
                    timeout=args.llm_timeout,
                    parser_mode=parser_mode, apply_mode=apply_mode,
                    n_distractors=n_distractors,
                    prompt_style=args.prompt_style,
                    think=args.think, max_tokens=args.max_tokens,
                    store=store, counter=counter,
                    shared_llm_client=_shared_client,
                    shared_raw_cache=shared_raw_cache,
                )

                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=gen,
                    sandbox=sandbox,
                    strategies=tuple(args.strategies),
                    timeout_s=args.timeout_s,
                    apply_mode=apply_mode)
                pooled = rep.pooled_summary()
                wall = time.time() - t0
                print(f"  [{cell_label}] pooled:")
                for strat, p in pooled.items():
                    print(f"    {strat:>12} pass@1={p['pass_at_1']:.3f} "
                          f"apply={p['patch_applied_rate']:.3f} "
                          f"tok≈{p['mean_patch_gen_prompt_tokens_approx']:.1f} "
                          f"n={p['n']}")
                # Fan out capture rows with measurement data.
                store.annotate_from_report(
                    rep, parser_mode=parser_mode,
                    apply_mode=apply_mode,
                    n_distractors=n_distractors,
                    repo_files=repo_files)
                cells.append({
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": n_distractors,
                    "report": rep.as_dict(),
                    "parser_compliance": counter.as_dict(),
                    "cell_wall_s": round(wall, 2),
                })

    overall_wall = time.time() - overall_start
    parent = {
        "config": {**vars(args),
                    "schema": "phase44.parent.v1"},
        "sandbox": sandbox.name(),
        "jsonl_path": jsonl_path,
        "cells": cells,
        "wall_seconds": round(overall_wall, 2),
    }
    if _shared_client is not None and hasattr(_shared_client, "stats"):
        parent["llm_client_stats"] = {
            "prompt_tokens": _shared_client.stats.prompt_tokens,
            "output_tokens": _shared_client.stats.output_tokens,
            "n_generate_calls": _shared_client.stats.n_generate_calls,
            "total_wall": round(_shared_client.stats.total_wall, 2),
        }
    if args.out_parent:
        with open(args.out_parent, "w", encoding="utf-8") as fh:
            json.dump(parent, fh, indent=2, default=str)
        print(f"\nWrote parent artifact: {args.out_parent}")
    if args.out_capture:
        store.write(args.out_capture)
        print(f"Wrote raw capture:     {args.out_capture}  "
              f"(n_records={len(store.records)})")
    return 0


# =============================================================================
# Analysis-only path — refined-taxonomy report
# =============================================================================


def _load_bank_index(jsonl_path: str) -> dict:
    tasks, repo_files = load_jsonl_bank(jsonl_path)
    idx: dict[str, dict] = {}
    for t in tasks:
        src = repo_files.get(t.buggy_file_relpath, "")
        idx[t.instance_id] = {
            "buggy_source": src,
            "gold_patch": tuple(t.gold_patch),
            "buggy_file_relpath": t.buggy_file_relpath,
        }
    return idx


def _refined_counter_for_model(*,
                                 parent: dict,
                                 capture: RawCaptureStore,
                                 bank_index: dict,
                                 ) -> dict:
    """Build two semantic counters per (parser_mode, apply_mode, nd)
    cell — one under the Phase-43 coarse classifier (sentinel path,
    for backwards compatibility with the Phase-43 summary), and one
    under the Phase-44 refined classifier using real raw-capture
    bytes.
    """
    # Index capture by (iid, strat, pm, am, nd).
    cap_idx: dict[tuple, dict] = {}
    for r in capture.records:
        cap_idx[(r.instance_id, r.strategy, r.parser_mode,
                   r.apply_mode, r.n_distractors)] = r

    per_cell: dict[str, dict] = {}
    for cell in parent.get("cells", []):
        pm = cell["parser_mode"]
        am = cell["apply_mode"]
        nd = int(cell["n_distractors"])
        label = f"parser={pm}/nd={nd}/apply={am}"
        coarse = SemanticCounter()
        refined = SemanticCounter()
        partition_audit: dict[str, dict[str, int]] = {}

        for m in cell["report"]["measurements"]:
            iid = m["instance_id"]
            strat = m["strategy"]
            info = bank_index.get(iid)
            if info is None:
                continue
            # Phase-43 coarse (sentinel) classification.
            rat = m.get("rationale", "") or ""
            if rat.startswith("parse_failed"):
                proposed_sent: tuple = ()
            else:
                proposed_sent = (("__sentinel__", "__sentinel__"),)
            c_label = classify_semantic_outcome(
                buggy_source=info["buggy_source"],
                gold_patch=info["gold_patch"],
                proposed_patch=proposed_sent,
                error_kind=m["error_kind"],
                test_passed=bool(m["test_passed"]),
                error_detail=(m.get("error_detail") or ""),
            )
            coarse.record(c_label, strategy=strat)

            # Phase-44 refined classification (raw-bytes path).
            rec = cap_idx.get((iid, strat, pm, am, nd))
            if rec is None:
                # No raw-capture for this measurement — record
                # the coarse label in the refined counter too (no
                # refinement possible).
                refined.record(c_label, strategy=strat)
                continue
            r_label = classify_semantic_outcome_v2(
                buggy_source=info["buggy_source"],
                gold_patch=info["gold_patch"],
                proposed_patch=rec.proposed_patch,
                applied_patch=rec.applied_patch,
                patched_source=None,
                error_kind=rec.error_kind,
                test_passed=rec.test_passed,
                error_detail=(m.get("error_detail") or ""),
            )
            refined.record(r_label, strategy=strat)

            # Track partition: coarse → refined.
            cmap = partition_audit.setdefault(c_label, {})
            cmap[r_label] = cmap.get(r_label, 0) + 1

        per_cell[label] = {
            "coarse_taxonomy": coarse.as_dict(),
            "refined_taxonomy": refined.as_dict(),
            "coarse_to_refined_partition": {
                c: dict(sorted(d.items()))
                for (c, d) in sorted(partition_audit.items())
            },
        }
    return per_cell


def _pass_rate_matrix(parent: dict) -> dict:
    out: dict[str, dict[str, float]] = {}
    for cell in parent.get("cells", []):
        label = (f"parser={cell['parser_mode']}/"
                 f"nd={cell['n_distractors']}/"
                 f"apply={cell['apply_mode']}")
        per_strat: dict[str, list] = {}
        for m in cell["report"]["measurements"]:
            per_strat.setdefault(m["strategy"], []).append(
                bool(m["test_passed"]))
        row = {}
        for strat in ("naive", "routing", "substrate"):
            xs = per_strat.get(strat, [])
            row[strat] = round(sum(xs) / max(1, len(xs)), 4)
        out[label] = row
    return out


def _analyse(args) -> int:
    bank_index = _load_bank_index(args.jsonl)
    print(f"[phase44] loaded bank: {len(bank_index)} instances")
    if len(args.artifacts) != len(args.captures):
        raise SystemExit(
            "--artifacts and --captures must be pairwise aligned "
            "(same length, same order).")

    models: dict[str, dict] = {}
    for (ap_path, cap_path) in zip(args.artifacts, args.captures):
        if not os.path.exists(ap_path):
            print(f"[phase44] SKIP missing artifact: {ap_path}")
            continue
        if not os.path.exists(cap_path):
            print(f"[phase44] SKIP missing capture:  {cap_path}")
            continue
        with open(ap_path, "r", encoding="utf-8") as fh:
            parent = json.load(fh)
        capture = RawCaptureStore.read(cap_path)
        cfg = parent.get("config", {})
        model = cfg.get("model") or capture.meta.get("model") or "?"
        ollama = cfg.get("ollama_url") or capture.meta.get("ollama_url") or "?"
        key = f"{model}@{ollama}::{os.path.basename(ap_path)}"

        pr = _pass_rate_matrix(parent)
        per_cell = _refined_counter_for_model(
            parent=parent, capture=capture, bank_index=bank_index)
        models[key] = {
            "artifact": ap_path,
            "capture":  cap_path,
            "model":    model,
            "ollama":   ollama,
            "pass_rates": pr,
            "taxonomy": per_cell,
            "n_capture_records": len(capture.records),
        }

    canonical = "parser=robust/nd=6/apply=strict"
    print()
    print("=" * 72)
    print(f"PHASE 44 — refined residue @ {canonical}")
    print("=" * 72)
    cross = []
    for (k, info) in models.items():
        pr = info["pass_rates"].get(canonical, {})
        tax = info["taxonomy"].get(canonical, {}) or {}
        refined = tax.get("refined_taxonomy", {})
        coarse = tax.get("coarse_taxonomy", {})
        partition = tax.get("coarse_to_refined_partition", {})
        print(f"\n  {info['model']}  @  {info['ollama']}")
        print(f"    pass@1: naive={pr.get('naive',0):.3f}  "
              f"routing={pr.get('routing',0):.3f}  "
              f"substrate={pr.get('substrate',0):.3f}  "
              f"S-N gap={100*abs(pr.get('substrate',0)-pr.get('naive',0)):.1f}pp")
        print(f"    pooled coarse failure mix  : "
              f"{coarse.get('pooled_failure_mix', {})}")
        print(f"    pooled refined failure mix : "
              f"{refined.get('pooled_failure_mix', {})}")
        print(f"    coarse → refined partition : {partition}")
        cross.append({
            "model": info["model"],
            "ollama": info["ollama"],
            "pass_rate_substrate": pr.get("substrate", 0.0),
            "pass_rate_naive": pr.get("naive", 0.0),
            "substrate_vs_naive_gap_pp": round(
                100 * abs(pr.get("substrate", 0.0) -
                            pr.get("naive", 0.0)), 1),
            "coarse_failure_mix": coarse.get("pooled_failure_mix", {}),
            "refined_failure_mix": refined.get("pooled_failure_mix", {}),
            "coarse_to_refined_partition": partition,
        })

    payload = {
        "bank_jsonl": args.jsonl,
        "bank_n_instances": len(bank_index),
        "canonical_cell": canonical,
        "models": models,
        "cross_model_canonical": cross,
        "schema": "phase44.summary.v1",
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true",
                      help=("Skip the LLM sweep path and run the analysis "
                             "stage on pre-existing Phase-44 artifact + "
                             "raw-capture pairs."))
    ap.add_argument("--mode", choices=("mock", "real"), default="mock")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--jsonl", default=_DEFAULT_JSONL)
    ap.add_argument("--n-distractors", nargs="+", type=int, default=[6])
    ap.add_argument("--n-instances", type=int, default=None)
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_SWE_STRATEGIES))
    ap.add_argument("--apply-modes", nargs="+", default=["strict"])
    ap.add_argument("--parser-modes", nargs="+",
                      default=["strict", "robust"])
    ap.add_argument("--prompt-style", default="block",
                      choices=("block", "unified_diff"))
    ap.add_argument("--sandbox",
                      choices=("auto", "in_process", "subprocess", "docker"),
                      default="subprocess")
    ap.add_argument("--docker-image", default="python:3.11-slim")
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--think", choices=("on", "off", "default"),
                      default="default")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out-parent", default=None)
    ap.add_argument("--out-capture", default=None)
    # Analysis flags
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--captures", nargs="*", default=[])
    ap.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.analyse_only:
        if not args.artifacts:
            raise SystemExit(
                "--analyse-only requires --artifacts and --captures")
        return _analyse(args)
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
