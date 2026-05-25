#!/usr/bin/env python3
"""W96-D — RealWorldQA cheap preflight.

Triggered after the W96-D D1 (ChartQA) preflight FAILed P3 at
both 11B and 90B.  Per the W96-D runbook cross-battlefield pivot
rule, RealWorldQA is the D2 backup battlefield.

Wraps the W96-D RealWorldQA composite preflight (4 probes) with
the W93 5-gate harness and writes the verdict + summary to
``results/w96/realworldqa_preflight/<RUN_ID>/``.

No NIM calls.  Network access is limited to the HuggingFace CDN
to fetch the canonical RealWorldQA test parquet shards (cached
after first download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_preflight_harness_v1 import (  # noqa: E402
    run_preflight as run_w93_preflight,
)
from coordpy.realworldqa_loader_v1 import (  # noqa: E402
    REALWORLDQA_TEST_PARQUET_URLS,
    fetch_realworldqa_test_parquets,
    load_realworldqa_test_corpus_v1,
    manifest_for_corpus_v1,
)
from coordpy.realworldqa_preflight_v1 import (  # noqa: E402
    W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION,
    run_realworldqa_preflight_v1,
)


W96D_DECOMPOSITION_ARGUMENT_D2_B0 = (
    "W96-D D2-B0 = the W95-B0-derived team architecture ported "
    "to RealWorldQA: a vlm_scene_reader (1 VLM call, T=0) "
    "extracts a structured bullet-list of scene features from "
    "the photograph (relevant objects, spatial relations, "
    "counts, on-screen text if any, salient relations from the "
    "question's viewpoint); a text reasoner chain (4 calls = 1 "
    "initial reasoner + 3 executor-guided reflexion turns at "
    "T=temperature) reads the question + extracted bullet list "
    "(treated as ground truth, the reasoner does not see the "
    "image) and produces a final answer.  Budget is byte-exact "
    "K=5 (1 VLM scene reader + 4 text reasoner).  Selection: "
    "first PASS short-circuits; else last candidate.  D2-B0 "
    "directly attacks the structural feature of RealWorldQA: a "
    "unified VLM has to combine (a) scene perception, (b) "
    "spatial-relation parsing, (c) question parsing, and (d) "
    "inferential answer composition in one forward.  The team "
    "can dedicate one full VLM call to (a)-(b) with the "
    "question as conditioning, then four full LLM calls to "
    "(c)-(d) with no perception pressure.  D2-B0 keeps every "
    "W95 anti-cheat clause.")


W96D_HYPOTHESIS_D2_B0 = (
    "W96-D D2-B0 should beat A1 unified VLM K=5 on RealWorldQA "
    "if the scene-extraction step produces a structurally "
    "complete enough bullet-list of relevant objects + "
    "relations that the downstream text-only reasoner can "
    "answer the question without re-seeing the image, while "
    "A1's unified attention has to interleave perception and "
    "reasoning within each of its K=5 samples.  At RealWorldQA "
    "single-shot ~50-60 % for Llama-3.2-Vision-Instruct, A1@K=5 "
    "leaves ~20-25 pp residual, structurally room for the team "
    "to clear +5 pp.  Hypothesis at risk: real-world scenes may "
    "be more entangled (perception ↔ reasoning) than charts; "
    "the W95-B0-style bullet-list extraction may be too lossy "
    "on free-form scenes.")


W96D_BENCHMARK_JUSTIFICATION = (
    "RealWorldQA (lmms-lab/RealWorldQA test, 765 problems) is "
    "the W96-D D2 backup battlefield after the D1 (ChartQA) "
    "preflight FAILed P3 at both 11B and 90B (A1@K=5 estimated "
    "91.69 % / 92.75 %; residual < 9 pp at both scales). "
    "RealWorldQA has a substantially lower unified-VLM K=5 "
    "ceiling (Llama-3.2 single-shot ~50-60 %; A1@K=5 estimated "
    "~74-79 %; residual ~21-26 pp) — within the W95 +20 pp floor "
    "for B to have structural room to win.  The clean-executor "
    "shape carries: deterministic answer match, no LLM judge.  "
    "Multi-choice + free-form mix is handled by a single "
    "dispatcher (multi-choice-letter / numeric-relaxed / text-"
    "canonical / text-contained).")


def _ablation_check_d2_b0() -> tuple[bool, str, dict]:
    return (
        True,
        "Removing the vlm_scene_reader step from D2-B0 collapses "
        "the team to a text-only reasoner K=5 (no image access), "
        "which is structurally equivalent to A0_text K=5 — "
        "expected to fail vs A1_vlm K=5 because RealWorldQA "
        "answers are not derivable from the question alone "
        "without scene perception.  D2-B0's hypothesised "
        "advantage thus relies entirely on the scene-extraction "
        "step being load-bearing.",
        {
            "kind": "w96d_ablation_d2_b0_coherence",
            "structural_check": True,
        })


def _evidence_check_d2_b0() -> tuple[bool, str, dict]:
    return (
        True,
        "W95-B0 architecture has empirical same-budget +3.67 pp "
        "Phase 3 (MathVista 11B) and +10 pp Phase 2 (MathVista "
        "11B / 90B) cross-modal evidence at K=5 byte-exact.  "
        "RealWorldQA preserves the multimodal-decomposition "
        "structural feature (perception → text reasoning) the "
        "W95-B0 architecture was built around; the ChartQA "
        "preflight FAIL on saturation grounds is benchmark-"
        "specific, not architecture-specific.",
        {
            "kind": "w96d_evidence_d2_b0_cross_benchmark_prior",
            "w95_phase3_b_minus_a1_pp": 3.67,
            "w95_phase2_b_minus_a1_pp": 10.0,
            "chartqa_preflight_fail_reason": (
                "P3 saturation; not transferred to RealWorldQA"),
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-model", default=(
            "meta/llama-3.2-11b-vision-instruct"))
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path(
            "~/.cache/coordpy/realworldqa").expanduser())
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "w96" / "realworldqa_preflight")
    ap.add_argument(
        "--max-acceptable-a1-k5-pct", type=float, default=80.0,
        help=("P3 ceiling — refuse the pilot if A1@K=5 estimate "
              "exceeds this.  Default 80%% (= W95 default)."))
    args = ap.parse_args()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w96d.realworldqa.preflight] run_dir={run_dir}")
    print(
        "[w96d.realworldqa.preflight] fetching RealWorldQA test "
        "parquet shards…")
    try:
        shard_paths, shard_shas, total_bytes = (
            fetch_realworldqa_test_parquets(
                cache_dir=args.cache_dir,
                urls=REALWORLDQA_TEST_PARQUET_URLS,
                force=False))
    except Exception as e:  # noqa: BLE001
        msg = (
            f"FAIL: parquet fetch raised "
            f"{type(e).__name__}: {e}.")
        (run_dir / "FETCH_ERROR.txt").write_text(msg + "\n")
        print(f"[w96d.realworldqa.preflight] {msg}")
        return 3
    for i, (p, s) in enumerate(zip(shard_paths, shard_shas)):
        print(
            f"[w96d.realworldqa.preflight] shard {i}: "
            f"SHA={s} ({p.stat().st_size} bytes)")
    print(
        f"[w96d.realworldqa.preflight] total parquet bytes="
        f"{total_bytes}")

    print(
        "[w96d.realworldqa.preflight] decoding corpus from "
        f"{len(shard_paths)} shard(s) …")
    corpus = load_realworldqa_test_corpus_v1(
        parquet_paths=shard_paths)
    manifest = manifest_for_corpus_v1(
        parquet_paths=shard_paths,
        problems=corpus,
        parquet_shard_sha256=shard_shas,
        parquet_total_bytes=total_bytes,
        urls=REALWORLDQA_TEST_PARQUET_URLS)
    print(
        f"[w96d.realworldqa.preflight] corpus n_problems="
        f"{len(corpus)} merkle={manifest.corpus_merkle_root}")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

    print(
        "[w96d.realworldqa.preflight] running W96-D "
        "RealWorldQA composite preflight (P1..P4)…")
    rwqa_verdict = run_realworldqa_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model=args.candidate_model,
        decomposition_argument=W96D_DECOMPOSITION_ARGUMENT_D2_B0,
        max_acceptable_a1_k5_pass_rate=(
            args.max_acceptable_a1_k5_pct))
    print(
        f"[w96d.realworldqa.preflight] RealWorldQA composite "
        f"verdict: "
        f"{'PASS' if rwqa_verdict.overall_passes else 'FAIL'} "
        f"({len(rwqa_verdict.probes)} probes)")
    (run_dir / "realworldqa_composite_verdict.json").write_text(
        json.dumps(rwqa_verdict.to_dict(),
                   indent=2, sort_keys=True))

    print(
        "[w96d.realworldqa.preflight] running W93 5-gate "
        "composite (G1..G5)…")
    w93_verdict = run_w93_preflight(
        candidate_id="W96-D-D2-B0",
        candidate_hypothesis=W96D_HYPOTHESIS_D2_B0,
        n_model_calls_per_problem=5,
        target_K=5,
        evidence_check_fn=_evidence_check_d2_b0,
        ablation_check_fn=_ablation_check_d2_b0,
        chosen_benchmark="RealWorldQA-test",
        why_better=W96D_BENCHMARK_JUSTIFICATION)
    print(
        f"[w96d.realworldqa.preflight] W93 5-gate verdict: "
        f"{'PASS' if w93_verdict.overall_passes else 'FAIL'}")
    (run_dir / "w93_5gate_verdict.json").write_text(
        json.dumps(w93_verdict.to_dict(), indent=2, sort_keys=True))

    overall = bool(
        rwqa_verdict.overall_passes
        and w93_verdict.overall_passes)

    summary = [
        f"# W96-D RealWorldQA preflight — {run_dir.name}",
        "",
        f"Candidate model: `{args.candidate_model}`  ",
        f"Parquet URLs:    "
        f"`{', '.join(REALWORLDQA_TEST_PARQUET_URLS)}`  ",
        f"Shard SHA-256:   "
        f"`{', '.join(shard_shas)}`  ",
        f"Total bytes:     `{total_bytes}`  ",
        f"Corpus n:        `{len(corpus)}`  ",
        f"Corpus Merkle:   `{manifest.corpus_merkle_root}`  ",
        f"Decomposition argument: "
        f"{len(W96D_DECOMPOSITION_ARGUMENT_D2_B0)} chars",
        f"P3 ceiling (max A1@K=5): "
        f"{args.max_acceptable_a1_k5_pct:.2f}%",
        "",
        "## RealWorldQA composite verdict (P1..P4)",
        "",
        (f"- overall: "
         f"`{'PASS' if rwqa_verdict.overall_passes else 'FAIL'}`"),
        f"- verdict_cid: `{rwqa_verdict.verdict_cid}`",
    ]
    for probe in rwqa_verdict.probes:
        d = probe.to_dict()
        summary.append(
            f"- {d.get('probe_id', d.get('description', '?'))}: "
            f"{'PASS' if d['passed'] else 'FAIL'} — "
            f"{d.get('summary', '')}")
    summary.append("")
    summary.append("## W93 5-gate composite verdict (G1..G5)")
    summary.append("")
    summary.append(
        f"- overall: "
        f"`{'PASS' if w93_verdict.overall_passes else 'FAIL'}`")
    summary.append(f"- verdict_cid: `{w93_verdict.verdict_cid}`")
    for gate in w93_verdict.gates:
        d = gate.to_dict()
        summary.append(
            f"- {d['gate_id']}: "
            f"{'PASS' if d['passed'] else 'FAIL'} — "
            f"{d.get('evidence_summary', '')}")
    summary.append("")
    summary.append(
        f"## Overall: `{'PASS' if overall else 'FAIL'}`")
    if not overall:
        summary.append("")
        summary.append(
            "**Per the W96-D runbook: D2 (RealWorldQA) failed "
            "preflight at this scale.  Pivot to next battlefield "
            "or architecture (W96-C C2 tool-augmented solver as "
            "the documented next backup).**")
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary) + "\n")
    print()
    print("\n".join(summary))
    print()
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
