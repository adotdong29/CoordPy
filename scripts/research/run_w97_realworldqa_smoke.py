#!/usr/bin/env python3
"""W97 — RealWorldQA D2-B0 NIM smoke test.

Two-step smoke check before launching the W97 Phase 2 cheap
pilot:

  1. One 1-token POST against the candidate VLM model returns
     HTTP 200 with a non-empty completion (and the same for the
     text-only path).
  2. One 1-problem end-to-end dry-run of the D2-B0 wiring (1
     A0 + 5 A1 + 5 B = 11 NIM calls) at 11B (default) or 90B.

Hard contract (locked in `docs/RUNBOOK_W97.md`):

  * model = same VLM family on every arm (default
    `meta/llama-3.2-11b-vision-instruct`).
  * budget = 11 model calls / problem on the dry-run.
  * executor = `coordpy.realworldqa_executor_v1.evaluate_realworldqa_answer_v1`.
  * parquet shards SHA-anchored at start.
  * no selective retries; no judge.

Exit code is non-zero if the smoke fails or if any sidecar /
audit-chain artifact is missing.
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

from coordpy.realworldqa_bench_v1 import (  # noqa: E402
    RealWorldQABenchConfigV1,
    W97_REALWORLDQA_BENCH_V1_SCHEMA_VERSION,
    run_realworldqa_bench_v1,
)
from coordpy.realworldqa_loader_v1 import (  # noqa: E402
    REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256,
    REALWORLDQA_TEST_PARQUET_URLS,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-model", required=False,
        default="meta/llama-3.2-11b-vision-instruct")
    parser.add_argument(
        "--cache-dir", required=False,
        default=str(Path("data") / "realworldqa"))
    parser.add_argument(
        "--max-tokens", required=False, type=int, default=384)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY") or ""
    if not api_key:
        print("[FAIL] NVIDIA_API_KEY not set", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    scale_tag = (
        "11b" if "11b" in args.candidate_model.lower()
        else ("90b" if "90b" in args.candidate_model.lower()
              else "unknown"))
    run_dir = (
        ROOT / "results" / "w97"
        / f"realworldqa_smoke_{scale_tag}" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = run_dir / "calls.jsonl"

    # Step 1 — corpus fetch + SHA anchor (no NIM yet).
    print("[1/4] Fetching RealWorldQA test parquet shards...")
    t0 = time.time()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    paths, shas, total_bytes = (
        fetch_realworldqa_test_parquets(cache_dir=cache_dir))
    expected = REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256
    if shas != expected:
        print(
            f"[FAIL] parquet SHA mismatch: {shas} vs {expected}",
            file=sys.stderr)
        return 3
    corpus = load_realworldqa_test_corpus_v1(
        parquet_paths=paths)
    manifest = manifest_for_corpus_v1(
        parquet_paths=paths, problems=corpus,
        parquet_shard_sha256=shas,
        parquet_total_bytes=total_bytes)
    print(
        f"  -> n_problems={len(corpus)}; "
        f"parquet bytes={total_bytes}; "
        f"merkle={manifest.corpus_merkle_root[:16]}...")

    # Step 2 — 1-token smoke (VLM + text)
    print("[2/4] 1-token NIM smoke calls...")
    vlm_gen = make_nim_vlm_gen(
        args.candidate_model, api_key=api_key)
    text_gen = make_text_gen_from_vlm(vlm_gen)
    smoke_text_vlm, t_vlm = vlm_gen(
        "Reply with a single token: ok", b"", 8, 0.0)
    smoke_text_text, t_text = text_gen(
        "Reply with a single token: ok", 8, 0.0)
    smoke_ok = (
        bool(smoke_text_vlm.strip())
        and not smoke_text_vlm.startswith("[ERR")
        and bool(smoke_text_text.strip())
        and not smoke_text_text.startswith("[ERR"))
    print(
        f"  -> vlm={smoke_text_vlm[:32]!r} "
        f"text={smoke_text_text[:32]!r} ok={smoke_ok}")

    # Step 3 — 1-problem dry-run (D2-B0).
    print("[3/4] 1-problem dry-run on D2-B0 wiring...")
    cfg = RealWorldQABenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(96_504_002,),
        sampling_temperature=0.7,
        max_tokens_per_call=int(args.max_tokens))

    def write_sidecar(record):
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a") as fh:
            fh.write(json.dumps(
                record, sort_keys=True,
                separators=(",", ":")) + "\n")

    try:
        report = run_realworldqa_bench_v1(
            text_gen=text_gen,
            vlm_gen=vlm_gen,
            vlm_model_id=args.candidate_model,
            text_model_id=args.candidate_model,
            corpus=corpus,
            corpus_parquet_shard_sha256=shas,
            corpus_merkle_root=manifest.corpus_merkle_root,
            config=cfg,
            sidecar_writer=write_sidecar)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] dry-run failed: {exc}",
              file=sys.stderr)
        return 4
    seed_rep = report.per_seed[0]
    print(
        f"  -> a0={seed_rep.a0_text_pass_at_1 * 100:.1f}%  "
        f"a1={seed_rep.a1_vlm_pass_at_1 * 100:.1f}%  "
        f"b={seed_rep.b_vlm_team_pass_at_1 * 100:.1f}%")

    # Step 4 — write smoke verdict
    print("[4/4] Writing smoke verdict...")
    dt = time.time() - t0
    verdict = {
        "schema": "coordpy.w97_realworldqa_smoke.v1",
        "run_id": run_id,
        "candidate_model": args.candidate_model,
        "smoke_ok": bool(smoke_ok),
        "smoke_vlm_text_head": str(smoke_text_vlm[:96]),
        "smoke_text_text_head": str(smoke_text_text[:96]),
        "smoke_vlm_wall_ms": int(t_vlm),
        "smoke_text_wall_ms": int(t_text),
        "dry_run_a0": float(seed_rep.a0_text_pass_at_1),
        "dry_run_a1": float(seed_rep.a1_vlm_pass_at_1),
        "dry_run_b": float(seed_rep.b_vlm_team_pass_at_1),
        "n_problems_in_dry_run": int(seed_rep.n_problems),
        "n_model_calls_in_dry_run": int(
            seed_rep.n_problems * (1 + 5 + 5)),
        "parquet_shard_sha256": list(shas),
        "parquet_total_bytes": int(total_bytes),
        "corpus_merkle_root": str(
            manifest.corpus_merkle_root),
        "bench_merkle_root": str(report.bench_merkle_root),
        "seed_merkle_root": str(
            report.per_seed[0].seed_merkle_root),
        "schema_bench": (
            W97_REALWORLDQA_BENCH_V1_SCHEMA_VERSION),
        "wall_s": float(dt),
    }
    (run_dir / "verdict.json").write_text(
        json.dumps(verdict, sort_keys=True, indent=2))
    summary = []
    summary.append(
        f"# W97 RealWorldQA smoke — {run_dir.name}\n")
    summary.append(
        f"Total wall: {dt:.0f}s  ")
    summary.append(
        f"Candidate model: `{args.candidate_model}`  ")
    summary.append(
        f"Parquet shard SHAs: {[s[:8] for s in shas]}  ")
    summary.append(
        f"Corpus Merkle: `{manifest.corpus_merkle_root}`  ")
    summary.append(
        f"Bench Merkle: `{report.bench_merkle_root}`  \n")
    summary.append("## Verdicts\n")
    summary.append(
        f"* 1-token VLM smoke: "
        f"`{smoke_text_vlm[:64]!r}` (ok={smoke_ok})")
    summary.append(
        f"* 1-token text smoke: `{smoke_text_text[:64]!r}`")
    summary.append(
        f"* 1-problem A0 = {seed_rep.a0_text_pass_at_1 * 100:.0f}%")
    summary.append(
        f"* 1-problem A1@K=5 = {seed_rep.a1_vlm_pass_at_1 * 100:.0f}%")
    summary.append(
        f"* 1-problem B (D2-B0) = "
        f"{seed_rep.b_vlm_team_pass_at_1 * 100:.0f}%")
    summary.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(summary))
    (ROOT / "results" / "w97" / "latest_run.txt").write_text(
        f"smoke_{scale_tag}/{run_id}\n")

    print(f"Smoke run: {run_dir}")
    return 0 if smoke_ok else 1


if __name__ == "__main__":
    sys.exit(main())
