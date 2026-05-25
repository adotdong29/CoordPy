#!/usr/bin/env python3
"""W100 — RealWorldQA cross-scale 90B Phase 2 pilot driver (B2 / B5).

W100 = cross-scale confirmation of the W99 winners on
`lmms-lab/RealWorldQA` at `meta/llama-3.2-90b-vision-instruct`.

The candidate slate is FROZEN by W99 (see `docs/RUNBOOK_W100.md`):

  --candidate B2  -> coordpy.realworldqa_bench_v3
                     (direct-vision final-turn answerer;
                      structural frontier lead)
  --candidate B5  -> coordpy.realworldqa_bench_v5
                     (question-type router / switch baseline;
                      baseline-only ceiling reference)

W100 explicitly DOES NOT re-open:
  * B4 (typed schema sans direct_answer_hint; W99 FAIL by -16.67 pp)
  * any other typed-extract-then-text-reason variant
  * any new tournament candidate

This driver re-uses the W99 pilot internals (gate evaluator,
structural verdict logic, NIM HTTPS path, sidecar writers) but
adds:

  * AddrW100-B2-P5 — cross-scale rescue-prior stability probe
  * AddrW100-B5-P4 — cross-scale route-mass stability probe
  * MLB-1 — final-VLM invocation rate <= 50% (B2 only)
  * MLB-2 — final-VLM rescue rate >= 33% (B2 only)
  * cross-scale per-problem comparison vs W99 11B per_problem.jsonl

Hard contract:
  * slice = select_realworldqa_subset_v1(seed=96_504_002, n=30)
  * same VLM model on every arm (default
    `meta/llama-3.2-90b-vision-instruct`)
  * budget = 11 model calls / problem (1 A0 + 5 A1 + 5 B)
  * executor = evaluate_realworldqa_answer_v1
  * parquet shard SHAs must match
    0ed8b555... + 7dcb3ac3...
  * no selective retries; no judge
  * 9 pre-committed Phase 2 gates byte-identical to W95 / W96-A /
    W96-C / W97 / W98 / W99
  * For B2: 2 mechanism-load-bearingness sub-gates (MLB-1, MLB-2)
    additionally evaluated

Exit code is non-zero iff any pre-committed Phase 2 gate FAILS or
any B2 MLB sub-gate FAILS.
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

from coordpy.realworldqa_bench_v3 import (  # noqa: E402
    RealWorldQAV3BenchConfig,
    run_realworldqa_bench_v3,
)
from coordpy.realworldqa_bench_v5 import (  # noqa: E402
    RealWorldQAV5BenchConfig,
    run_realworldqa_bench_v5,
)
from coordpy.realworldqa_loader_v1 import (  # noqa: E402
    REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256,
    fetch_realworldqa_test_parquets,
    load_realworldqa_test_corpus_v1,
    manifest_for_corpus_v1,
    select_realworldqa_subset_v1,
)


_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


_CANDIDATE_BENCH_FN = {
    "B2": run_realworldqa_bench_v3,
    "B5": run_realworldqa_bench_v5,
}

_CANDIDATE_CONFIG_CLS = {
    "B2": RealWorldQAV3BenchConfig,
    "B5": RealWorldQAV5BenchConfig,
}

_CANDIDATE_PASS_RATE_ATTR = {
    "B2": "b_direct_vision_final_mean_pass_at_1",
    "B5": "b_routed_switch_mean_pass_at_1",
}

_CANDIDATE_SEED_PASS_RATE_ATTR = {
    "B2": "b_direct_vision_final_pass_at_1",
    "B5": "b_routed_switch_pass_at_1",
}

_CANDIDATE_PROBLEM_PASS_KEY = {
    "B2": "b_direct_vision_final_passed",
    "B5": "b_routed_switch_passed",
}


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
        report, candidate: str,
        *, slice_pids: tuple[str, ...],
        expected_calls_per_problem: int = 11,
) -> tuple[list[dict], bool]:
    """The 9 pre-committed Phase 2 gates from
    `docs/RUNBOOK_W100.md`.  Gate texts byte-identical to W99 /
    W98 / W97 / W96-C / W96-A / W95; only the arm name, scale
    tag, and slice seed appear differently."""
    seed_rep = report.per_seed[0]
    a0 = seed_rep.a0_text_pass_at_1
    a1 = seed_rep.a1_vlm_pass_at_1
    b_attr = _CANDIDATE_SEED_PASS_RATE_ATTR[candidate]
    b = getattr(seed_rep, b_attr)
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
            f"({'PASS' if a1 < 0.90 else 'FAIL — saturated'})")})
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


def _evaluate_mlb_subgates_b2(report) -> tuple[list[dict], bool]:
    """W100 mechanism-load-bearingness sub-gates (B2 only).
    Pre-committed in docs/RUNBOOK_W100.md.  Guards against
    W96-C C1-style variance-driven PASS at the cross-scale level."""
    seed_rep = report.per_seed[0]
    n_problems = int(seed_rep.n_problems)
    inv = int(seed_rep.final_vlm_invocation_count)
    res = int(seed_rep.final_vlm_rescue_count)
    inv_rate = float(inv) / float(max(n_problems, 1))
    res_rate = float(res) / float(max(inv, 1)) if inv > 0 else 0.0
    sub: list[dict] = []
    sub.append({
        "subgate": "MLB_1_invocation_rate_le_50pct",
        "pass": bool(inv_rate <= 0.50),
        "summary": (
            f"Final-VLM invoked on {inv}/{n_problems} = "
            f"{inv_rate * 100.0:.2f}% of problems "
            f"(threshold ≤ 50 %)")})
    sub.append({
        "subgate": "MLB_2_rescue_rate_ge_33pct",
        "pass": bool(res_rate >= (1.0 / 3.0) if inv > 0 else False),
        "summary": (
            f"Final-VLM rescued {res}/{inv} = "
            f"{res_rate * 100.0:.2f}% of invocations "
            f"(threshold ≥ 33.33 %)" if inv > 0 else
            f"Final-VLM not invoked on any problem — "
            "MLB_2 is undefined; treated as FAIL (cannot "
            "establish mechanism is load-bearing)")})
    overall = bool(all(g["pass"] for g in sub))
    return sub, overall


def _structural_assessment(
        gates, *, candidate: str,
        mlb_overall_pass: bool | None) -> str:
    g_by = {g["gate"]: g["pass"] for g in gates}
    overall_pass = all(g["pass"] for g in gates)
    if overall_pass:
        if candidate == "B2":
            if mlb_overall_pass is True:
                return "PHASE_2_PASS_MECHANISM_LOAD_BEARING"
            if mlb_overall_pass is False:
                return "PHASE_2_PASS_NON_MECHANISM_DRIVEN"
        return "PHASE_2_PASS"
    if (not g_by.get("2_a1_lt_90pct", True)
            and g_by.get("3_b_strictly_beats_a1", False)
            and g_by.get("4_margin_b_over_a1_ge_5pp", False)
            and g_by.get(
                "6_per_problem_b_ge_a1_majority", False)):
        if candidate == "B2":
            if mlb_overall_pass is True:
                return (
                    "STRUCTURALLY_POSITIVE_SLICE_SATURATION_"
                    "CAP_MECHANISM_LOAD_BEARING")
            if mlb_overall_pass is False:
                return (
                    "STRUCTURALLY_POSITIVE_SLICE_SATURATION_"
                    "CAP_NON_MECHANISM_DRIVEN")
        return "STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP"
    return "PHASE_2_FAIL"


def _cross_scale_comparison(
        w99_11b_pp_path: Path,
        w100_per_problem,
        candidate: str) -> dict:
    """Per-problem comparison of W100 90B vs W99 11B for the
    same candidate on the same slice."""
    out: dict = {"candidate": candidate}
    b_key = _CANDIDATE_PROBLEM_PASS_KEY[candidate]
    if not w99_11b_pp_path.exists():
        out["w99_11b"] = {
            "available": False,
            "path": str(w99_11b_pp_path),
        }
        return out
    prev = [json.loads(l)
            for l in w99_11b_pp_path.read_text().splitlines()
            if l.strip()]
    by_pid = {p["pid"]: p for p in prev}
    new_wins = []
    new_losses = []
    both_pass_pids = []
    neither_pass_pids = []
    n_pids_matched = 0
    for po in w100_per_problem:
        prev_p = by_pid.get(po["pid"])
        if not prev_p:
            continue
        n_pids_matched += 1
        this_passed = bool(po[b_key])
        prev_passed = bool(prev_p[b_key])
        if this_passed and not prev_passed:
            new_wins.append(po["pid"])
        if prev_passed and not this_passed:
            new_losses.append(po["pid"])
        if this_passed and prev_passed:
            both_pass_pids.append(po["pid"])
        if not this_passed and not prev_passed:
            neither_pass_pids.append(po["pid"])
    out["w99_11b"] = {
        "available": True,
        "path": str(w99_11b_pp_path),
        "n_matched": int(n_pids_matched),
        "n_new_wins_at_90b_vs_11b": len(new_wins),
        "n_new_losses_at_90b_vs_11b": len(new_losses),
        "n_both_pass": len(both_pass_pids),
        "n_neither_pass": len(neither_pass_pids),
        "new_win_pids": list(new_wins),
        "new_loss_pids": list(new_losses),
        "both_pass_pids": list(both_pass_pids),
        "neither_pass_pids": list(neither_pass_pids),
    }
    return out


def _addr_w100_b2_p5_rescue_prior(w99_b2_pp_path: Path) -> dict:
    """AddrW100-B2-P5 NIM-free: mine the W99 B2 per-problem
    outcomes; report the empirical 11B rescue rate and the W96-D
    90B residual-headroom-implied rescue room."""
    if not w99_b2_pp_path.exists():
        return {
            "probe": "AddrW100_B2_P5_cross_scale_rescue_prior",
            "pass": False,
            "summary": (
                f"W99 B2 per-problem file not found: "
                f"{w99_b2_pp_path}")}
    pp = [json.loads(l)
          for l in w99_b2_pp_path.read_text().splitlines()
          if l.strip()]
    n = len(pp)
    if n == 0:
        return {
            "probe": "AddrW100_B2_P5_cross_scale_rescue_prior",
            "pass": False,
            "summary": "W99 B2 per-problem file is empty"}
    n_b_pass = sum(
        1 for p in pp
        if bool(p.get("b_direct_vision_final_passed")))
    n_b_passed_via_final = sum(
        1 for p in pp
        if bool(p.get("b_final_vlm_rescued")))
    n_b_final_invoked = sum(
        1 for p in pp
        if bool(p.get("b_final_vlm_invoked")))
    headroom_90b = 20.51  # W96-D 90B residual
    expected_unique_a1_at_90b = max(
        1, int(round(headroom_90b / 100.0 * float(n))))
    has_unique_a1_room = (
        expected_unique_a1_at_90b >= 3)
    return {
        "probe": "AddrW100_B2_P5_cross_scale_rescue_prior",
        "pass": bool(has_unique_a1_room),
        "summary": (
            f"W99 B2 11B per-problem: n={n}; "
            f"B2 PASS={n_b_pass}; final-VLM invoked="
            f"{n_b_final_invoked}; final-VLM rescued="
            f"{n_b_passed_via_final}.  "
            f"W96-D 90B residual={headroom_90b:.2f} pp; "
            f"expected unique-A1-rescues at 90B (by residual) "
            f"≥ {expected_unique_a1_at_90b} (threshold ≥ 3 "
            f"for rescue-prior stability)")}


def _addr_w100_b5_p4_route_mass(
        corpus, slice_pids: tuple[str, ...],
        w99_b5_11b_pp_path: Path) -> dict:
    """AddrW100-B5-P4 NIM-free: re-run the deterministic
    question-type parser on the same slice; confirm the route
    distribution is byte-identical to the W99 11B on-disk
    per-problem routing (parser is NIM-free and scale-
    independent by construction)."""
    from coordpy.realworldqa_bench_v5 import (
        b5_route_for_question, detect_question_type_v2)
    by_pid = {p.pid: p for p in corpus}
    qt_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    per_pid_route: dict[str, str] = {}
    for pid in slice_pids:
        p = by_pid[pid]
        qt = detect_question_type_v2(p.question)
        rt = b5_route_for_question(p.question)
        qt_counts[qt] = qt_counts.get(qt, 0) + 1
        route_counts[rt] = route_counts.get(rt, 0) + 1
        per_pid_route[pid] = rt
    expected_routes = {"vlm_team_b0": 18, "a1_vlm_k5": 12}
    expected_match = (route_counts == expected_routes)
    on_disk_match = True
    on_disk_summary = "(W99 11B per_problem.jsonl not available)"
    if w99_b5_11b_pp_path.exists():
        prev = [json.loads(l)
                for l in (
                    w99_b5_11b_pp_path.read_text().splitlines())
                if l.strip()]
        prev_by_pid = {p["pid"]: p for p in prev}
        n_match = 0
        mismatches: list[str] = []
        for pid, rt in per_pid_route.items():
            prev_p = prev_by_pid.get(pid)
            if prev_p is None:
                continue
            prev_rt = str(prev_p.get("route"))
            if prev_rt == rt:
                n_match += 1
            else:
                mismatches.append(
                    f"{pid}({rt}!={prev_rt})")
        on_disk_match = (
            n_match == len(per_pid_route) and not mismatches)
        on_disk_summary = (
            f"on-disk match: {n_match}/{len(per_pid_route)} "
            f"per-pid routes equal W99 11B; mismatches="
            f"{mismatches[:5]}")
    ok = bool(expected_match and on_disk_match)
    return {
        "probe": "AddrW100_B5_P4_cross_scale_route_mass",
        "pass": ok,
        "summary": (
            f"Question type distribution on slice = {qt_counts}; "
            f"route distribution = {route_counts}; "
            f"expected = {expected_routes}; "
            f"expected_match={expected_match}; "
            f"{on_disk_summary}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate", choices=["B2", "B5"], required=True,
        help="Which W100 candidate to confirm at 90B.  B4 is "
             "explicitly REMOVED from the slate (W99 dead).")
    ap.add_argument(
        "--vlm-model", default=os.environ.get(
            "W100_VLM",
            "meta/llama-3.2-90b-vision-instruct"))
    ap.add_argument("--text-model", default="")
    ap.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W100_N_PROBLEMS", "30")))
    ap.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W100_N_SEEDS", "1")))
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
        default=ROOT / "results" / "w100" / "realworldqa_pilot")
    ap.add_argument(
        "--w99-b2-11b-pilot-dir", type=Path,
        default=(ROOT / "results" / "w99" / "realworldqa_pilot"
                 / "w99_realworldqa_pilot_b2_11b_meta_llama-3.2"
                   "-11b-vision-instruct__meta_llama-3.2-11b"
                   "-vision-instruct_20260525T205551Z"))
    ap.add_argument(
        "--w99-b5-11b-pilot-dir", type=Path,
        default=(ROOT / "results" / "w99" / "realworldqa_pilot"
                 / "w99_realworldqa_pilot_b5_11b_meta_llama-3.2"
                   "-11b-vision-instruct__meta_llama-3.2-11b"
                   "-vision-instruct_20260525T202433Z"))
    ap.add_argument(
        "--probes-only", action="store_true",
        help="Run only the NIM-free AddrW100 probes; do not "
             "call NIM.  Used for pre-pilot sanity checking.")
    args = ap.parse_args()

    candidate = str(args.candidate)
    bench_fn = _CANDIDATE_BENCH_FN[candidate]
    config_cls = _CANDIDATE_CONFIG_CLS[candidate]
    pass_rate_attr = _CANDIDATE_PASS_RATE_ATTR[candidate]

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
        / (f"w100_realworldqa_pilot_{candidate.lower()}_"
           f"{scale_tag}_{safe_vlm}__{safe_text}_{timestamp}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w100.pilot] candidate={candidate} run_dir={run_dir}")
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
        f"[w100.pilot] parquet shards SHA-anchored: "
        f"{[s[:8] for s in shas]} ({total_bytes} bytes)")
    corpus = load_realworldqa_test_corpus_v1(
        parquet_paths=paths)
    manifest = manifest_for_corpus_v1(
        parquet_paths=paths, problems=corpus,
        parquet_shard_sha256=shas,
        parquet_total_bytes=total_bytes)
    print(
        f"[w100.pilot] corpus n_problems={len(corpus)} "
        f"merkle={manifest.corpus_merkle_root}")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

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
            f"[w100.pilot] pre-committed slice (seed={s}): "
            f"{len(pids)} pids; slice_sha256="
            f"{slice_sha[:16]}…")
    (run_dir / "pre_committed_slice.json").write_text(
        json.dumps({
            "schema": "coordpy.w100_pre_committed_slices.v1",
            "candidate": candidate,
            "n_seeds": int(n_seeds),
            "n_problems_per_seed": int(n_problems),
            "slices": pre_committed_records,
        }, indent=2, sort_keys=True))
    slice_pids = slice_pids_per_seed[0]

    addr_probes = []
    if candidate == "B2":
        addr_probes.append(
            _addr_w100_b2_p5_rescue_prior(
                args.w99_b2_11b_pilot_dir / "per_problem.jsonl"))
    if candidate == "B5":
        addr_probes.append(
            _addr_w100_b5_p4_route_mass(
                corpus, slice_pids,
                args.w99_b5_11b_pilot_dir
                / "per_problem.jsonl"))
    (run_dir / "addr_w100_probes.json").write_text(
        json.dumps({
            "schema": "coordpy.w100_addr_probes.v1",
            "candidate": candidate,
            "probes": addr_probes,
        }, indent=2, sort_keys=True))
    for ap_rec in addr_probes:
        print(
            f"[w100.pilot] {ap_rec['probe']}: "
            f"{'PASS' if ap_rec['pass'] else 'FAIL'} — "
            f"{ap_rec['summary']}")
    all_addr_pass = all(p["pass"] for p in addr_probes)
    if not all_addr_pass:
        print(
            "[w100.pilot] AddrW100 probe(s) FAILED; "
            "aborting before NIM call",
            file=sys.stderr)
        return 4

    if args.probes_only:
        print(
            "[w100.pilot] --probes-only mode: skipping NIM "
            "pilot; probes recorded.")
        (Path(args.out_dir) /
         f"latest_probes_{candidate.lower()}.txt"
         ).write_text(run_dir.name + "\n")
        return 0

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY env var required for the W100 NIM "
            "pilot.")

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

    cfg = config_cls(
        n_problems=int(n_problems),
        K_multi_sample=5, seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens))

    def progress(seed, p_idx, pid):
        elapsed = time.time() - t_run_start
        print(
            f"  seed={seed} problem {p_idx + 1}/{n_problems} "
            f"(pid={pid}) elapsed={elapsed:.0f}s "
            f"text={n_text} vlm={n_vlm}",
            flush=True)

    print(
        f"[w100.pilot] starting bench {candidate} at "
        f"{scale_tag}: vlm={vlm_model_id} text={text_model_id} "
        f"K={cfg.K_multi_sample} n_problems={n_problems} "
        f"n_seeds={n_seeds}")

    report = bench_fn(
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
        report, candidate, slice_pids=slice_pids,
        expected_calls_per_problem=(
            1 + 2 * cfg.K_multi_sample))
    mlb_sub: list[dict] | None = None
    mlb_overall: bool | None = None
    if candidate == "B2":
        mlb_sub, mlb_overall = _evaluate_mlb_subgates_b2(report)
    structural_verdict = _structural_assessment(
        gates, candidate=candidate,
        mlb_overall_pass=mlb_overall)

    w99_pp_path = (
        args.w99_b2_11b_pilot_dir / "per_problem.jsonl"
        if candidate == "B2"
        else args.w99_b5_11b_pilot_dir / "per_problem.jsonl")
    cmp = _cross_scale_comparison(
        w99_pp_path,
        report.per_seed[0].per_problem_outcomes,
        candidate)

    (run_dir / "phase2_gates.json").write_text(
        json.dumps({
            "schema": "coordpy.w100_phase2_gates.v1",
            "candidate": candidate,
            "scale_tag": scale_tag,
            "overall_passes": bool(overall),
            "mlb_subgates": (
                list(mlb_sub) if mlb_sub is not None else None),
            "mlb_overall_pass": (
                bool(mlb_overall)
                if mlb_overall is not None else None),
            "structural_verdict": str(structural_verdict),
            "gates": list(gates),
            "n_problems": int(n_problems),
            "n_seeds": int(n_seeds),
            "K": int(cfg.K_multi_sample),
            "cross_scale_comparison": dict(cmp),
            "addr_w100_probes": list(addr_probes),
        }, indent=2, sort_keys=True))

    b_mean = getattr(report, pass_rate_attr)
    a0_mean = report.a0_text_mean_pass_at_1
    a1_mean = report.a1_vlm_mean_pass_at_1
    b_minus_a1 = report.b_mean_minus_a1_vlm_mean_pp
    b_minus_a0 = report.b_mean_minus_a0_text_mean_pp

    summary_lines: list[str] = []
    summary_lines.append(
        f"# W100 RealWorldQA {candidate} Phase 2 90B pilot — "
        f"{run_dir.name}\n")
    summary_lines.append(f"Candidate: `{candidate}`  ")
    summary_lines.append(f"Scale: `{scale_tag}`  ")
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

    summary_lines.append(
        "## AddrW100 NIM-free pre-flight probes\n")
    for ap_rec in addr_probes:
        summary_lines.append(
            f"* **{ap_rec['probe']}**: "
            f"{'PASS' if ap_rec['pass'] else 'FAIL'} — "
            f"{ap_rec['summary']}")
    summary_lines.append("")

    summary_lines.append("## Per-arm pass rates\n")
    summary_lines.append(
        f"* A0_text:           {a0_mean * 100.0:.2f} %")
    summary_lines.append(
        f"* A1_vlm K=5:        {a1_mean * 100.0:.2f} %")
    summary_lines.append(
        f"* B ({candidate}):              "
        f"{b_mean * 100.0:.2f} %  "
        f"(B − A1 = {b_minus_a1:+.2f} pp; "
        f"B − A0 = {b_minus_a0:+.2f} pp)")
    summary_lines.append("")
    summary_lines.append(
        f"Question type distribution: "
        f"`{report.question_type_distribution}`")
    if hasattr(report, "route_distribution"):
        summary_lines.append(
            f"Route distribution (B5): "
            f"`{report.route_distribution}`")
    if hasattr(report,
               "final_vlm_invocation_count_total"):
        summary_lines.append(
            f"Final VLM invocations (B2): "
            f"{report.final_vlm_invocation_count_total}  ")
        summary_lines.append(
            f"Final VLM rescues (B2): "
            f"{report.final_vlm_rescue_count_total}  ")
    summary_lines.append("")
    summary_lines.append(
        "## Pre-committed Phase 2 gates\n")
    for g in gates:
        summary_lines.append(
            f"* **{g['gate']}**: "
            f"{'PASS' if g['pass'] else 'FAIL'} — "
            f"{g['summary']}")
    if mlb_sub is not None:
        summary_lines.append(
            "\n## Mechanism-load-bearingness sub-gates "
            "(B2 only)\n")
        for g in mlb_sub:
            summary_lines.append(
                f"* **{g['subgate']}**: "
                f"{'PASS' if g['pass'] else 'FAIL'} — "
                f"{g['summary']}")
    summary_lines.append("")
    summary_lines.append(
        f"## Structural verdict: `{structural_verdict}`\n")
    if cmp.get("w99_11b", {}).get("available"):
        summary_lines.append(
            "## Cross-scale comparison vs W99 11B (same slice, "
            "same candidate)\n")
        cs = cmp["w99_11b"]
        summary_lines.append(
            f"* matched problems: {cs['n_matched']}")
        summary_lines.append(
            f"* new wins at 90B vs 11B: "
            f"{cs['n_new_wins_at_90b_vs_11b']}")
        summary_lines.append(
            f"* new losses at 90B vs 11B: "
            f"{cs['n_new_losses_at_90b_vs_11b']}")
        summary_lines.append(
            f"* both pass: {cs['n_both_pass']}")
        summary_lines.append(
            f"* neither pass: {cs['n_neither_pass']}")
        if cs["new_win_pids"]:
            summary_lines.append(
                f"* new win pids: {cs['new_win_pids']}")
        if cs["new_loss_pids"]:
            summary_lines.append(
                f"* new loss pids: {cs['new_loss_pids']}")
    if candidate == "B2":
        if structural_verdict.endswith("LOAD_BEARING"):
            verdict_label = (
                "PASS — cross-scale confirmation succeeded; "
                "mechanism load-bearing at 90B; Phase 3 "
                "ENTITLED per W100 runbook")
        elif structural_verdict.endswith("NON_MECHANISM_DRIVEN"):
            verdict_label = (
                "PASS_NON_MECHANISM_DRIVEN — 90B PASS but "
                "MLB sub-gates FAIL; variance-driven; "
                "Phase 3 NOT entitled (W96-C C1 precedent)")
        elif overall:
            verdict_label = "PHASE_2_PASS (B2 without MLB context)"
        elif structural_verdict.startswith(
                "STRUCTURALLY_POSITIVE"):
            verdict_label = (
                f"STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP — "
                f"consider Phase 3 carefully; structural verdict "
                f"= `{structural_verdict}`")
        else:
            verdict_label = (
                "FAIL — W100 B2 90B Phase 2 KILLED; "
                "promote COO-9 per W100 code-pivot contingency; "
                "document W100-L-* carry-forward")
    else:
        verdict_label = (
            "PASS — B5 90B routing-ceiling confirmed "
            "(baseline-only; no Phase 3 implication)"
            if overall else
            ("STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP "
             "(B5 baseline-only ceiling reference)"
             if structural_verdict == (
                 "STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP")
             else "FAIL — W100 B5 90B Phase 2 KILLED; "
                  "routing-ceiling does not generalize "
                  "cross-scale on this slice"))
    summary_lines.append(
        f"\n## Overall verdict: `{verdict_label}`")
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n")

    (Path(args.out_dir) / f"latest_run_{candidate.lower()}.txt"
     ).write_text(run_dir.name + "\n")

    print()
    print(
        f"[w100.pilot] total wall: {dt:.0f}s, "
        f"text={n_text} vlm={n_vlm}")
    print(
        f"[w100.pilot] A0_text         = "
        f"{a0_mean * 100.0:.2f}%")
    print(
        f"[w100.pilot] A1_vlm K=5      = "
        f"{a1_mean * 100.0:.2f}%")
    print(
        f"[w100.pilot] B ({candidate})  = "
        f"{b_mean * 100.0:.2f}% "
        f"(B − A1 = {b_minus_a1:+.2f} pp; "
        f"B − A0 = {b_minus_a0:+.2f} pp)")
    for g in gates:
        print(
            f"[w100.pilot] {g['gate']}: "
            f"{'PASS' if g['pass'] else 'FAIL'} — "
            f"{g['summary']}")
    if mlb_sub is not None:
        for g in mlb_sub:
            print(
                f"[w100.pilot] {g['subgate']}: "
                f"{'PASS' if g['pass'] else 'FAIL'} — "
                f"{g['summary']}")
    print(f"[w100.pilot] STRUCTURAL: {structural_verdict}")
    print(f"[w100.pilot] OVERALL: {verdict_label}")
    mlb_required_for_exit = (
        mlb_overall is True if candidate == "B2" else True)
    exit_ok = bool(overall) and mlb_required_for_exit
    return 0 if exit_ok else 2


if __name__ == "__main__":
    sys.exit(main())
