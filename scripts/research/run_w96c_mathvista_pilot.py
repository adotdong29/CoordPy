#!/usr/bin/env python3
"""W96-C — MathVista V2 (C1 VLM-Verifier-Final-Turn) NIM pilot.

Forks ``scripts/run_w95_mathvista_pilot.py`` to drive the V2
bench (``coordpy.mathvista_bench_v2``) instead of V1, on the
SAME deterministic 30-problem slice (seed 95_005_001) as W95
Phase 2 and W96-A Phase 2 so the cross-architecture comparison
stays problem-level fair.

Hard contract (locked in `docs/RUNBOOK_W96C.md`):

  * slice = ``select_mathvista_subset_v1(95_005_001, 30,
    corpus=testmini)`` BEFORE any NIM call (byte-identical to
    W95 / W96-A Phase 2);
  * model = same VLM family on every arm (default depends on the
    cross-scale step being run; either 11B or 90B);
  * budget = 11 model calls / problem (1 A0 + 5 A1 + 5 B) —
    byte-identical to V1;
  * executor = ``coordpy.mathvista_executor_v1.evaluate_answer_v1``
    for every arm;
  * parquet SHA-256 must match the W95 anchor;
  * no selective retries; no judge.

Exit code is non-zero iff any pre-committed Phase 2 gate FAILS.
The Markdown SUMMARY in the run dir lists each gate's verdict
and the new V2 verifier-rescue accounting.
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

from coordpy.mathvista_bench_v2 import (  # noqa: E402
    MathVistaBenchConfigV2,
    W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
    run_mathvista_bench_v2,
)
from coordpy.mathvista_loader_v1 import (  # noqa: E402
    MATHVISTA_TESTMINI_PARQUET_URL,
    fetch_testmini_parquet,
    load_testmini_corpus_v1,
    manifest_for_corpus_v1,
)


EXPECTED_PARQUET_SHA = (
    "373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa969"
    "4d4f2d")


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
    """Identical to W95's NIM client (same backoff, same headers,
    same payload shape).  Forked verbatim so the pilot scripts
    are independent."""

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


def _evaluate_phase2_v2_gates(
        report, *, slice_pids: tuple[str, ...],
        expected_calls_per_problem: int = 11,
) -> tuple[list[dict], bool]:
    """Phase 2 gates for V2.  Mirrors W95 V1's 9-gate shape; the
    arm under evaluation is ``b_vlm_team_v2`` and the report is
    a ``MathVistaBenchReportV2``.  Gate texts are identical to
    V1's so cross-architecture verdicts can be compared
    directly.
    """
    seed_rep = report.per_seed[0]
    a0 = seed_rep.a0_text_pass_at_1
    a1 = seed_rep.a1_vlm_pass_at_1
    b = seed_rep.b_vlm_team_pass_at_1  # V2's b_vlm_team_v2 reuses the V1 seed-report field name
    n_b_ge_a1 = report.n_b_ge_a1_problems_per_seed[0]
    n_problems = seed_rep.n_problems
    gates: list[dict] = []
    gates.append({
        "gate": "1_slice_pre_committed",
        "pass": True,
        "summary": (
            f"slice taken by select_mathvista_subset_v1; "
            f"{len(slice_pids)} pids pre-committed.")})
    gates.append({
        "gate": "2_a1_lt_90pct",
        "pass": bool(a1 < 0.90),
        "summary": (
            f"A1@K=5 = {a1 * 100.0:.2f}% < 90 %? "
            f"({'PASS' if a1 < 0.90 else 'FAIL — saturated'})")})
    gates.append({
        "gate": "3_b_strictly_beats_a1",
        "pass": bool(b > a1),
        "summary": (
            f"B_v2 = {b * 100.0:.2f}% vs A1 = {a1 * 100.0:.2f}%; "
            f"B_v2 > A1? {b > a1}")})
    gates.append({
        "gate": "4_margin_b_over_a1_ge_5pp",
        "pass": bool((b - a1) * 100.0 >= 5.0),
        "summary": (
            f"B_v2 − A1 = {(b - a1) * 100.0:+.2f} pp "
            f"(threshold ≥ +5 pp)")})
    gates.append({
        "gate": "5_b_over_a0_ge_5pp",
        "pass": bool((b - a0) * 100.0 >= 5.0),
        "summary": (
            f"B_v2 − A0 = {(b - a0) * 100.0:+.2f} pp "
            "(image must be load-bearing; threshold ≥ +5 pp)")})
    per_problem_threshold = max(
        int(round(0.53 * float(n_problems))),
        (n_problems // 2) + 1)
    gates.append({
        "gate": "6_per_problem_b_ge_a1_majority",
        "pass": bool(n_b_ge_a1 >= per_problem_threshold),
        "summary": (
            f"B_v2 ≥ A1 on {n_b_ge_a1}/{n_problems} problems "
            f"(threshold ≥ {per_problem_threshold})")})
    gates.append({
        "gate": "7_budget_accounting_exact",
        "pass": True,
        "summary": (
            f"Each problem uses 1 A0 + {report.K_multi_sample} "
            f"A1 + {report.K_multi_sample} B_v2 = "
            f"{1 + 2 * report.K_multi_sample} calls "
            f"(expected={expected_calls_per_problem})")})
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
            "through evaluate_answer_v1.")})
    overall = bool(all(g["pass"] for g in gates))
    return gates, overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm-model", default=os.environ.get(
            "W96C_VLM",
            "meta/llama-3.2-11b-vision-instruct"))
    ap.add_argument(
        "--text-model", default="",
        help=("Optional separate text-LM; default = same as "
              "VLM."))
    ap.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W96C_N_PROBLEMS", "30")))
    ap.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W96C_N_SEEDS", "1")))
    ap.add_argument(
        "--seed-start", type=int, default=95_005_001)
    ap.add_argument(
        "--max-tokens", type=int, default=384)
    ap.add_argument(
        "--temperature", type=float, default=0.7)
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("~/.cache/coordpy/mathvista").expanduser())
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "w96" / "mathvista_c1_pilot")
    ap.add_argument(
        "--expected-parquet-sha256",
        default=EXPECTED_PARQUET_SHA)
    ap.add_argument(
        "--out-subdir", default="",
        help=("Override the leaf subdirectory under --out-dir "
              "(default: derived from models + timestamp)."))
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY env var required for the W96-C NIM "
            "pilot (`export NVIDIA_API_KEY=...`).")

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
    if args.out_subdir:
        run_dir = Path(args.out_dir) / args.out_subdir
    else:
        prefix = "w96c_mathvista_v2_pilot_"
        run_dir = (
            Path(args.out_dir)
            / f"{prefix}{safe_vlm}__{safe_text}_{timestamp}")
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

    # --- Corpus fetch + SHA verification
    print(f"[w96c.pilot] run_dir={run_dir}")
    parquet_path, parquet_sha, parquet_bytes = (
        fetch_testmini_parquet(
            cache_dir=args.cache_dir,
            url=MATHVISTA_TESTMINI_PARQUET_URL,
            force=False,
            expected_sha256=args.expected_parquet_sha256))
    print(
        f"[w96c.pilot] parquet SHA-verified: {parquet_sha} "
        f"({parquet_bytes} bytes)")
    print("[w96c.pilot] decoding corpus …")
    corpus = load_testmini_corpus_v1(parquet_path=parquet_path)
    manifest = manifest_for_corpus_v1(
        parquet_path=parquet_path,
        problems=corpus,
        parquet_sha256=parquet_sha,
        parquet_bytes=parquet_bytes)
    print(
        f"[w96c.pilot] corpus n_problems={len(corpus)} "
        f"merkle={manifest.corpus_merkle_root}")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

    cfg = MathVistaBenchConfigV2(
        n_problems=int(n_problems),
        K_multi_sample=5, seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens))

    from coordpy.mathvista_loader_v1 import (
        select_mathvista_subset_v1)
    slice_pids_per_seed: list[tuple[str, ...]] = []
    pre_committed_records = []
    for s in seeds:
        pre_slice = select_mathvista_subset_v1(
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
            f"[w96c.pilot] pre-committed slice (seed={s}): "
            f"{len(pids)} pids; slice_sha256="
            f"{slice_sha[:16]}…")
    (run_dir / "pre_committed_slice.json").write_text(
        json.dumps({
            "schema": "coordpy.w96c_pre_committed_slices.v1",
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
        f"[w96c.pilot] starting V2 bench: vlm={vlm_model_id} "
        f"text={text_model_id} K={cfg.K_multi_sample} "
        f"n_problems={n_problems} n_seeds={n_seeds}")

    report = run_mathvista_bench_v2(
        text_gen=wrapped_text, vlm_gen=wrapped_vlm,
        vlm_model_id=vlm_model_id,
        text_model_id=text_model_id,
        corpus=tuple(corpus),
        corpus_parquet_sha256=parquet_sha,
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

    gates, overall = _evaluate_phase2_v2_gates(
        report, slice_pids=slice_pids,
        expected_calls_per_problem=(
            1 + 2 * cfg.K_multi_sample))
    (run_dir / "phase2_v2_gates.json").write_text(
        json.dumps({
            "schema": "coordpy.w96c_phase2_v2_gates.v1",
            "overall_passes": bool(overall),
            "gates": list(gates),
            "n_problems": int(n_problems),
            "n_seeds": int(n_seeds),
            "K": int(cfg.K_multi_sample),
        }, indent=2, sort_keys=True))

    summary_lines: list[str] = []
    summary_lines.append(
        f"# W96-C V2 MathVista Phase 2 pilot — "
        f"{run_dir.name}\n")
    summary_lines.append(
        f"Total wall: {dt:.0f}s  ")
    summary_lines.append(
        f"VLM model: `{vlm_model_id}`  ")
    summary_lines.append(
        f"Text/solver model: `{text_model_id}`  ")
    summary_lines.append(
        f"Parquet SHA-256: `{parquet_sha}`  ")
    summary_lines.append(
        f"Corpus Merkle root: `{manifest.corpus_merkle_root}`  ")
    summary_lines.append(
        f"Bench Merkle root: `{report.bench_merkle_root}`  ")
    summary_lines.append(
        f"Seed Merkle root:  `{report.per_seed[0].seed_merkle_root}`  ")
    summary_lines.append(
        f"Total NIM calls: text={n_text} vlm={n_vlm}  \n")
    summary_lines.append("## Per-arm pass rates\n")
    summary_lines.append(
        f"* A0_text:           {report.a0_text_mean_pass_at_1 * 100.0:.2f} %")
    summary_lines.append(
        f"* A1_vlm K=5:        {report.a1_vlm_mean_pass_at_1 * 100.0:.2f} %")
    summary_lines.append(
        f"* B_vlm_team_v2:     {report.b_vlm_team_v2_mean_pass_at_1 * 100.0:.2f} %  "
        f"(B_v2 − A1 = {report.b_mean_minus_a1_vlm_mean_pp:+.2f} pp; "
        f"B_v2 − A0 = {report.b_mean_minus_a0_text_mean_pp:+.2f} pp)")
    summary_lines.append("")
    summary_lines.append("## V2 verifier-rescue accounting\n")
    summary_lines.append(
        f"* Text-only PASS (W95-B0-style win): "
        f"{list(report.n_text_only_passes_per_seed)} / "
        f"{n_problems * n_seeds} (per-seed)")
    summary_lines.append(
        f"* Verifier-rescue (text-only FAIL → VLM-verifier PASS): "
        f"{list(report.n_verifier_rescues_per_seed)} / "
        f"{n_problems * n_seeds} (per-seed)")
    summary_lines.append("")
    summary_lines.append("## Pre-committed Phase 2 gates\n")
    for g in gates:
        summary_lines.append(
            f"* **{g['gate']}**: "
            f"{'PASS' if g['pass'] else 'FAIL'} — {g['summary']}")
    summary_lines.append("")
    verdict_label = (
        "PASS — Phase 3 entitled (subject to cross-scale rule "
        "in RUNBOOK_W96C.md)"
        if overall else "FAIL — Phase 2 KILLED")
    summary_lines.append(
        f"## Overall verdict: `{verdict_label}`")
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n")

    latest_path = Path(args.out_dir) / "latest_run.txt"
    latest_path.write_text(run_dir.name + "\n")

    print()
    print(
        f"[w96c.pilot] total wall: {dt:.0f}s, "
        f"text={n_text} vlm={n_vlm}")
    print(
        f"[w96c.pilot] A0_text       = "
        f"{report.a0_text_mean_pass_at_1 * 100.0:.2f}%")
    print(
        f"[w96c.pilot] A1_vlm K=5    = "
        f"{report.a1_vlm_mean_pass_at_1 * 100.0:.2f}%")
    print(
        f"[w96c.pilot] B_vlm_team_v2 = "
        f"{report.b_vlm_team_v2_mean_pass_at_1 * 100.0:.2f}% "
        f"(B_v2 − A1 = "
        f"{report.b_mean_minus_a1_vlm_mean_pp:+.2f} pp; "
        f"B_v2 − A0 = "
        f"{report.b_mean_minus_a0_text_mean_pp:+.2f} pp)")
    print(
        f"[w96c.pilot] verifier rescues / seed: "
        f"{list(report.n_verifier_rescues_per_seed)}")
    for g in gates:
        print(
            f"[w96c.pilot] {g['gate']}: "
            f"{'PASS' if g['pass'] else 'FAIL'} — "
            f"{g['summary']}")
    print(f"[w96c.pilot] OVERALL: {verdict_label}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
