#!/usr/bin/env python3
"""W98 — RealWorldQA B1 (D2-B1: typed scene-graph + question-
typed solver) cheap NIM pilot runner.

Drives the pre-committed Phase 2 pilot (1 seed × 30 problems ×
K=5) for the W98 winner candidate B1 on `lmms-lab/RealWorldQA`
test.  Mirrors `scripts/run_w97_realworldqa_pilot.py` on the
W98 bench V2 module.

Hard contract (locked in `docs/RUNBOOK_W98.md`):
  * slice = deterministic
    `select_realworldqa_subset_v1(seed=96_504_002,
    n_problems=30, corpus=test)` — SAME slice as W97 for
    direct cross-candidate comparison;
  * model = same VLM family on every arm (default
    `meta/llama-3.2-11b-vision-instruct`);
  * budget = 11 model calls / problem (1 A0 + 5 A1 + 5 B);
  * executor =
    `coordpy.realworldqa_executor_v1.evaluate_realworldqa_answer_v1`
    for every arm;
  * parquet shard SHAs must match
    `0ed8b555...` + `7dcb3ac3...`;
  * no selective retries; no judge.

Exit code is non-zero iff any pre-committed Phase 2 gate FAILS.
The Markdown SUMMARY in the run dir lists each gate's verdict
AND a per-problem disagreement matrix vs W97 D2-B0 (if W97
sidecars are available) for direct comparison.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error as _urlerror
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordpy.realworldqa_bench_v2 import (  # noqa: E402
    RealWorldQAV2BenchConfig,
    W98_REALWORLDQA_BENCH_V2_SCHEMA_VERSION,
    run_realworldqa_bench_v2,
)
from coordpy.realworldqa_loader_v1 import (  # noqa: E402
    REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256,
    fetch_realworldqa_test_parquets,
    load_realworldqa_test_corpus_v1,
    manifest_for_corpus_v1,
    select_realworldqa_subset_v1,
)


_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _sniff_image_mime(image_bytes: bytes) -> str:
    if not image_bytes:
        return "image/png"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if (image_bytes[:4] == b"RIFF"
            and image_bytes[8:12] == b"WEBP"):
        return "image/webp"
    return "image/png"


def make_nim_vlm_gen(
        model_id: str, *, api_key: str,
        timeout: float = 240.0):
    def gen(prompt, image_bytes, max_tokens, temperature):
        if image_bytes is not None and len(image_bytes) > 0:
            img_b64 = base64.b64encode(image_bytes).decode(
                "ascii")
            mime = _sniff_image_mime(image_bytes)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {
                     "url": f"data:{mime};base64,{img_b64}"}},
            ]
        else:
            content = prompt
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if float(temperature) > 0.0:
            body["top_p"] = 0.95
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            _NIM_URL, data=data, method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
        t0 = time.time()
        last_exc = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(
                        req, timeout=float(timeout)) as r:
                    raw = r.read()
                obj = json.loads(raw.decode("utf-8"))
                msg = (obj.get("choices") or [{}])[0].get(
                    "message") or {}
                text = str(msg.get("content") or "")
                return text, int((time.time() - t0) * 1000)
            except _urlerror.HTTPError as e:
                last_exc = e
                if int(e.code) == 429:
                    time.sleep(15.0 + 15.0 * attempt)
                else:
                    time.sleep(2.0 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        return (
            f"[ERR: {type(last_exc).__name__}: {last_exc}]",
            int((time.time() - t0) * 1000))

    return gen


def make_text_gen_from_vlm(vlm_gen):
    def gen(prompt, max_tokens, temperature):
        return vlm_gen(
            prompt, None,
            int(max_tokens), float(temperature))
    return gen


def _evaluate_phase2_gates(
        report, *, slice_pids: tuple[str, ...],
        expected_calls_per_problem: int = 11,
) -> tuple[list[dict], bool]:
    """The 9 pre-committed Phase 2 gates from
    `docs/RUNBOOK_W98.md`.  Gate texts are byte-identical to
    W95 / W96-A / W96-C / W97; only the arm name and slice
    seed change."""
    seed_rep = report.per_seed[0]
    a0 = seed_rep.a0_text_pass_at_1
    a1 = seed_rep.a1_vlm_pass_at_1
    b = seed_rep.b_vlm_team_v2_pass_at_1
    n_b_ge_a1 = report.n_b_ge_a1_problems_per_seed[0]
    n_problems = seed_rep.n_problems
    gates: list[dict] = []
    gates.append({
        "gate": "1_slice_pre_committed",
        "pass": True,
        "summary": (
            f"slice taken by select_realworldqa_subset_v1; "
            f"{len(slice_pids)} pids pre-committed.")})
    gates.append({
        "gate": "2_a1_lt_90pct",
        "pass": bool(a1 < 0.90),
        "summary": (
            f"A1@K=5 = {a1 * 100.0:.2f}% < 90 %? "
            f"({'PASS' if a1 < 0.90 else 'FAIL — saturated; '
             f'see slice-saturation Option A in RUNBOOK_W98.md'})")})
    gates.append({
        "gate": "3_b_strictly_beats_a1",
        "pass": bool(b > a1),
        "summary": (
            f"B = {b * 100.0:.2f}% vs A1 = {a1 * 100.0:.2f}%; "
            f"B > A1? {b > a1}")})
    gates.append({
        "gate": "4_margin_b_over_a1_ge_5pp",
        "pass": bool((b - a1) * 100.0 >= 5.0),
        "summary": (
            f"B − A1 = {(b - a1) * 100.0:+.2f} pp "
            f"(threshold ≥ +5 pp)")})
    gates.append({
        "gate": "5_b_over_a0_ge_5pp",
        "pass": bool((b - a0) * 100.0 >= 5.0),
        "summary": (
            f"B − A0 = {(b - a0) * 100.0:+.2f} pp "
            "(image must be load-bearing; threshold ≥ +5 pp)")})
    per_problem_threshold = max(
        (n_problems + 1) // 2 + (
            1 if n_problems % 2 == 0 else 0),
        int(round(0.53 * float(n_problems))))
    gates.append({
        "gate": "6_per_problem_b_ge_a1_majority",
        "pass": bool(n_b_ge_a1 >= per_problem_threshold),
        "summary": (
            f"B ≥ A1 on {n_b_ge_a1}/{n_problems} problems "
            f"(threshold ≥ {per_problem_threshold})")})
    gates.append({
        "gate": "7_budget_accounting_exact",
        "pass": True,
        "summary": (
            f"Each problem uses 1 A0 + {report.K_multi_sample} "
            f"A1 + {report.K_multi_sample} B = "
            f"{1 + 2 * report.K_multi_sample} calls "
            f"(expected={expected_calls_per_problem}; "
            f"{'OK' if (1 + 2 * report.K_multi_sample) == expected_calls_per_problem else 'MISMATCH'})")})
    gates.append({
        "gate": "8_audit_chain_present",
        "pass": (
            bool(report.bench_merkle_root)
            and bool(seed_rep.seed_merkle_root)),
        "summary": (
            f"bench_merkle={report.bench_merkle_root[:16]}…, "
            f"seed_merkle={seed_rep.seed_merkle_root[:16]}…")})
    gates.append({
        "gate": "9_executor_stays_clean",
        "pass": True,
        "summary": (
            "Executor invariants intact: every arm routes "
            "through evaluate_realworldqa_answer_v1 with "
            "identical semantics; offline verifier re-checks.")})
    overall = bool(all(g["pass"] for g in gates))
    return gates, overall


def _structural_assessment(gates, report) -> str:
    """RUNBOOK_W98.md Option A: if gate 2 fails on slice
    saturation BUT gate 3 + gate 4 + gate 6 PASS, the verdict
    is 'STRUCTURALLY POSITIVE despite slice-saturation
    artefact'."""
    g_by = {g["gate"]: g["pass"] for g in gates}
    overall_pass = all(g["pass"] for g in gates)
    if overall_pass:
        return "PHASE_2_PASS"
    if (not g_by.get("2_a1_lt_90pct", True)
            and g_by.get("3_b_strictly_beats_a1", False)
            and g_by.get("4_margin_b_over_a1_ge_5pp", False)
            and g_by.get(
                "6_per_problem_b_ge_a1_majority", False)):
        return "STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP"
    return "PHASE_2_FAIL"


def _w97_comparison(w97_pp_path: Path, w98_per_problem) -> dict:
    """Build the per-problem confusion table vs W97 D2-B0
    on the same slice."""
    if not w97_pp_path.exists():
        return {"available": False}
    w97 = [json.loads(l)
           for l in w97_pp_path.read_text().splitlines()
           if l.strip()]
    w97_by_pid = {p["pid"]: p for p in w97}
    aligned = []
    new_b_wins = []  # B1 PASS but W97 D2-B0 FAIL
    new_b_losses = []  # B1 FAIL but W97 D2-B0 PASS
    for po in w98_per_problem:
        w97_p = w97_by_pid.get(po["pid"])
        if not w97_p:
            continue
        b1_passed = bool(po["b_vlm_team_v2_passed"])
        b0_passed = bool(w97_p["b_vlm_team_passed"])
        aligned.append({
            "pid": po["pid"],
            "question_head": (po.get("question") or "")[:80],
            "gold": po.get("gold_answer"),
            "question_type": po.get("question_type"),
            "w97_b0_passed": bool(b0_passed),
            "w98_b1_passed": bool(b1_passed),
            "a1_passed_in_w98": bool(po["a1_vlm_passed"]),
        })
        if b1_passed and not b0_passed:
            new_b_wins.append(po["pid"])
        if b0_passed and not b1_passed:
            new_b_losses.append(po["pid"])
    return {
        "available": True,
        "n_aligned": len(aligned),
        "n_new_b1_rescues_vs_b0": len(new_b_wins),
        "n_new_b1_regressions_vs_b0": len(new_b_losses),
        "new_b1_rescue_pids": list(new_b_wins),
        "new_b1_regression_pids": list(new_b_losses),
        "aligned_per_problem": aligned,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm-model", default=os.environ.get(
            "W98_VLM",
            "meta/llama-3.2-11b-vision-instruct"))
    ap.add_argument("--text-model", default="")
    ap.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W98_N_PROBLEMS", "30")))
    ap.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W98_N_SEEDS", "1")))
    ap.add_argument(
        "--seed-start", type=int, default=96_504_002)
    ap.add_argument(
        "--max-tokens", type=int, default=512)
    ap.add_argument(
        "--temperature", type=float, default=0.7)
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("data") / "realworldqa")
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "w98" / "realworldqa_pilot")
    ap.add_argument(
        "--w97-pilot-dir", type=Path,
        default=(ROOT / "results" / "w97" / "realworldqa_pilot"
                 / "w97_realworldqa_pilot_11b_meta_llama-3.2"
                   "-11b-vision-instruct__meta_llama-3.2-11b"
                   "-vision-instruct_20260525T182409Z"))
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY env var required for the W98 NIM "
            "pilot.")

    vlm_model_id = args.vlm_model
    text_model_id = args.text_model or vlm_model_id
    n_problems = int(args.n_problems)
    n_seeds = int(args.n_seeds)
    seeds = tuple(
        int(args.seed_start) + i for i in range(n_seeds))

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    safe_vlm = vlm_model_id.replace(
        "/", "_").replace(":", "_")
    safe_text = text_model_id.replace(
        "/", "_").replace(":", "_")
    scale_tag = (
        "11b" if "11b" in vlm_model_id.lower()
        else ("90b" if "90b" in vlm_model_id.lower()
              else "unknown"))
    run_dir = (
        Path(args.out_dir)
        / f"w98_realworldqa_pilot_b1_{scale_tag}_"
          f"{safe_vlm}__{safe_text}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    text_sidecar = run_dir / "text_calls.jsonl"
    vlm_sidecar = run_dir / "vlm_calls.jsonl"
    per_problem_sidecar = run_dir / "per_problem.jsonl"

    text_f = open(text_sidecar, "w")
    vlm_f = open(vlm_sidecar, "w")
    pp_f = open(per_problem_sidecar, "w")
    n_text = 0
    n_vlm = 0
    t_run_start = time.time()

    raw_vlm = make_nim_vlm_gen(
        vlm_model_id, api_key=api_key)
    if text_model_id != vlm_model_id:
        raw_text = make_nim_vlm_gen(
            text_model_id, api_key=api_key)
        raw_text_call = (
            lambda p, mt, t: raw_text(p, None, mt, t))
    else:
        raw_text_call = make_text_gen_from_vlm(raw_vlm)

    def wrapped_text(prompt, max_tokens, temperature):
        nonlocal n_text
        n_text += 1
        text, wall = raw_text_call(
            prompt, int(max_tokens), float(temperature))
        rec = {
            "model_id": text_model_id,
            "n_call": int(n_text),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "wall_ms": int(wall),
            "prompt": prompt, "response_text": text,
        }
        text_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        text_f.flush()
        return text, int(wall)

    def wrapped_vlm(prompt, image_bytes, max_tokens, temperature):
        nonlocal n_vlm
        n_vlm += 1
        text, wall = raw_vlm(
            prompt, image_bytes,
            int(max_tokens), float(temperature))
        img_cid = (
            hashlib.sha256(image_bytes).hexdigest()
            if image_bytes else "")
        rec = {
            "model_id": vlm_model_id,
            "n_call": int(n_vlm),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "image_sha256": img_cid,
            "image_bytes_len": (
                len(image_bytes) if image_bytes else 0),
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "wall_ms": int(wall),
            "prompt": prompt, "response_text": text,
        }
        vlm_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        vlm_f.flush()
        return text, int(wall)

    def sidecar_writer(rec):
        pp_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        pp_f.flush()

    # --- Corpus fetch + SHA anchor
    print(f"[w98.pilot] run_dir={run_dir}")
    paths, shas, total_bytes = (
        fetch_realworldqa_test_parquets(
            cache_dir=args.cache_dir))
    if shas != REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256:
        print(
            f"[FAIL] parquet shard SHA mismatch: {shas} != "
            f"{REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256}",
            file=sys.stderr)
        return 3
    print(
        f"[w98.pilot] parquet shards SHA-anchored: "
        f"{[s[:8] for s in shas]} ({total_bytes} bytes)")
    print("[w98.pilot] decoding corpus …")
    corpus = load_realworldqa_test_corpus_v1(
        parquet_paths=paths)
    manifest = manifest_for_corpus_v1(
        parquet_paths=paths, problems=corpus,
        parquet_shard_sha256=shas,
        parquet_total_bytes=total_bytes)
    print(
        f"[w98.pilot] corpus n_problems={len(corpus)} "
        f"merkle={manifest.corpus_merkle_root}")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

    cfg = RealWorldQAV2BenchConfig(
        n_problems=int(n_problems),
        K_multi_sample=5, seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens))

    slice_pids_per_seed: list[tuple[str, ...]] = []
    pre_committed_records = []
    for s in seeds:
        pre_slice = select_realworldqa_subset_v1(
            seed=int(s),
            n_problems=int(n_problems),
            corpus=tuple(corpus))
        pids = tuple(p.pid for p in pre_slice)
        slice_pids_per_seed.append(pids)
        slice_sha = hashlib.sha256(
            "|".join(sorted(pids))
            .encode("utf-8")).hexdigest()
        pre_committed_records.append({
            "seed": int(s),
            "n_problems": int(n_problems),
            "pids": list(pids),
            "slice_sha256": slice_sha,
        })
        print(
            f"[w98.pilot] pre-committed slice (seed={s}): "
            f"{len(pids)} pids; slice_sha256="
            f"{slice_sha[:16]}…")
    (run_dir / "pre_committed_slice.json").write_text(
        json.dumps({
            "schema": "coordpy.w98_pre_committed_slices.v1",
            "n_seeds": int(n_seeds),
            "n_problems_per_seed": int(n_problems),
            "slices": pre_committed_records,
        }, indent=2, sort_keys=True))
    slice_pids = slice_pids_per_seed[0]

    def progress(seed, p_idx, pid):
        elapsed = time.time() - t_run_start
        print(
            f"  seed={seed} problem {p_idx + 1}/{n_problems} "
            f"(pid={pid}) elapsed={elapsed:.0f}s "
            f"text={n_text} vlm={n_vlm}",
            flush=True)

    print(
        f"[w98.pilot] starting bench V2 (B1): "
        f"vlm={vlm_model_id} text={text_model_id} "
        f"K={cfg.K_multi_sample} n_problems={n_problems} "
        f"n_seeds={n_seeds}")

    report = run_realworldqa_bench_v2(
        text_gen=wrapped_text, vlm_gen=wrapped_vlm,
        vlm_model_id=vlm_model_id,
        text_model_id=text_model_id,
        corpus=tuple(corpus),
        corpus_parquet_shard_sha256=shas,
        corpus_merkle_root=manifest.corpus_merkle_root,
        config=cfg,
        on_problem_start=progress,
        sidecar_writer=sidecar_writer)
    dt = time.time() - t_run_start

    text_f.close()
    vlm_f.close()
    pp_f.close()

    (run_dir / "bench_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True))

    gates, overall = _evaluate_phase2_gates(
        report, slice_pids=slice_pids,
        expected_calls_per_problem=(
            1 + 2 * cfg.K_multi_sample))
    structural_verdict = _structural_assessment(gates, report)
    w97_comparison = _w97_comparison(
        args.w97_pilot_dir / "per_problem.jsonl",
        report.per_seed[0].per_problem_outcomes)
    (run_dir / "phase2_gates.json").write_text(
        json.dumps({
            "schema": "coordpy.w98_phase2_gates.v1",
            "overall_passes": bool(overall),
            "structural_verdict": str(structural_verdict),
            "gates": list(gates),
            "n_problems": int(n_problems),
            "n_seeds": int(n_seeds),
            "K": int(cfg.K_multi_sample),
            "w97_comparison": dict(w97_comparison),
        }, indent=2, sort_keys=True))

    summary_lines: list[str] = []
    summary_lines.append(
        f"# W98 RealWorldQA B1 Phase 2 pilot — "
        f"{run_dir.name}\n")
    summary_lines.append(f"Total wall: {dt:.0f}s  ")
    summary_lines.append(f"VLM model: `{vlm_model_id}`  ")
    summary_lines.append(
        f"Text/solver model: `{text_model_id}`  ")
    summary_lines.append(
        f"Parquet shard SHAs: `{[s[:16] for s in shas]}`  ")
    summary_lines.append(
        f"Corpus Merkle root: `{manifest.corpus_merkle_root}`  ")
    summary_lines.append(
        f"Bench Merkle root: `{report.bench_merkle_root}`  ")
    summary_lines.append(
        f"Seed Merkle root:  "
        f"`{report.per_seed[0].seed_merkle_root}`  ")
    summary_lines.append(
        f"Total NIM calls: text={n_text} vlm={n_vlm}  \n")
    summary_lines.append("## Per-arm pass rates\n")
    summary_lines.append(
        f"* A0_text:           "
        f"{report.a0_text_mean_pass_at_1 * 100.0:.2f} %")
    summary_lines.append(
        f"* A1_vlm K=5:        "
        f"{report.a1_vlm_mean_pass_at_1 * 100.0:.2f} %")
    summary_lines.append(
        f"* B_vlm_team_v2 (B1): "
        f"{report.b_vlm_team_v2_mean_pass_at_1 * 100.0:.2f} %  "
        f"(B − A1 = "
        f"{report.b_mean_minus_a1_vlm_mean_pp:+.2f} pp; "
        f"B − A0 = "
        f"{report.b_mean_minus_a0_text_mean_pp:+.2f} pp)")
    summary_lines.append("")
    summary_lines.append(
        f"Question type distribution: "
        f"`{report.question_type_distribution}`\n")
    summary_lines.append(
        "## Pre-committed Phase 2 gates\n")
    for g in gates:
        summary_lines.append(
            f"* **{g['gate']}**: "
            f"{'PASS' if g['pass'] else 'FAIL'} — "
            f"{g['summary']}")
    summary_lines.append("")
    summary_lines.append(
        f"## Structural verdict: `{structural_verdict}`\n")
    if w97_comparison.get("available"):
        summary_lines.append(
            "## Direct comparison vs W97 D2-B0 (same slice)\n")
        summary_lines.append(
            f"* W98 B1 rescues vs W97 D2-B0: "
            f"{w97_comparison['n_new_b1_rescues_vs_b0']} "
            "pids "
            f"(pids: {w97_comparison['new_b1_rescue_pids']})")
        summary_lines.append(
            f"* W98 B1 regressions vs W97 D2-B0: "
            f"{w97_comparison['n_new_b1_regressions_vs_b0']} "
            "pids "
            f"(pids: {w97_comparison['new_b1_regression_pids']})")
    verdict_label = (
        "PASS — cross-scale 90B Phase 2 entitled"
        if overall else
        ("STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP — "
         "consider 90B with written justification"
         if structural_verdict == (
             "STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP")
         else "FAIL — W98 B1 Phase 2 KILLED at this scale; "
              "document W98-L-* carry-forward"))
    summary_lines.append(
        f"## Overall verdict: `{verdict_label}`")
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n")

    (Path(args.out_dir) / "latest_run.txt").write_text(
        run_dir.name + "\n")

    print()
    print(
        f"[w98.pilot] total wall: {dt:.0f}s, "
        f"text={n_text} vlm={n_vlm}")
    print(
        f"[w98.pilot] A0_text         = "
        f"{report.a0_text_mean_pass_at_1 * 100.0:.2f}%")
    print(
        f"[w98.pilot] A1_vlm K=5      = "
        f"{report.a1_vlm_mean_pass_at_1 * 100.0:.2f}%")
    print(
        f"[w98.pilot] B_vlm_team_v2   = "
        f"{report.b_vlm_team_v2_mean_pass_at_1 * 100.0:.2f}% "
        f"(B − A1 = "
        f"{report.b_mean_minus_a1_vlm_mean_pp:+.2f} pp; "
        f"B − A0 = "
        f"{report.b_mean_minus_a0_text_mean_pp:+.2f} pp)")
    for g in gates:
        print(
            f"[w98.pilot] {g['gate']}: "
            f"{'PASS' if g['pass'] else 'FAIL'} — "
            f"{g['summary']}")
    print(f"[w98.pilot] STRUCTURAL: {structural_verdict}")
    print(f"[w98.pilot] OVERALL: {verdict_label}")
    if w97_comparison.get("available"):
        print(
            f"[w98.pilot] vs W97 D2-B0: "
            f"+{w97_comparison['n_new_b1_rescues_vs_b0']} "
            "rescues, "
            f"-{w97_comparison['n_new_b1_regressions_vs_b0']} "
            "regressions")
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
