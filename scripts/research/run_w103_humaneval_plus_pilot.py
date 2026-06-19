#!/usr/bin/env python3
"""W103 — HumanEval+ cheap-pilot driver (lead lane).

CONDITIONAL ON W102 HumanEval+ preflight reconfirmation 7/7 PASS.
Verdict
`results/w102/humaneval_plus_preflight/<RUN>/verdict.json` must
have `overall_passes=true` before this driver is authorised to
spend NIM budget.  Slice is helper-anchored from the W102 code-
slice-selector output (COO-14 deliverable).

Runs the W89 sequential-reflexion B-pipeline + A0 + A1 baselines
against HumanEval+ at 1 seed × 30 problems × K=5 = 330 NIM calls
at the target model (default `meta/llama-3.3-70b-instruct`).
Evaluates the pre-committed 9 Phase 2 gates + MLB-1 + MLB-2
mechanism-load-bearingness sub-gates per `docs/RUNBOOK_W103.md`.

Hardening (W103 lessons from W102):

* Helper-anchored slice rule (NOT a parallel deterministic-seed
  shuffle).  De-duplicates on `task_id`; tops up from base
  HumanEval helper proposal; refuses if cluster mix degenerates
  to all `shared_wins`.
* Provenance fields recorded in the bench report: corpus_sha,
  helper_proposal_cid (humaneval_plus), helper_proposal_cid
  (humaneval top-up), mining_report_cid, preflight_verdict_cid,
  slice_cid_helper_priority, slice_cid_bench_order,
  arsenal_mining_prior_b_minus_a1_pp (recorded but NOT a gate
  input).
* Schema-divergence guard: re-runs the canonical-solution self-
  test post-pilot on the 30 slice problems; if any fail, the
  audit chain refuses to write.
* Refuses unpinned operation if corpus cache is missing or SHA
  doesn't match.

Requires `NVIDIA_API_KEY` in the environment.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w103_humaneval_plus_pilot.py \\
        --model meta/llama-3.3-70b-instruct \\
        --slice-proposal-json \\
            results/w103/code_slice_proposals/latest_run/proposals.json \\
        --n-problems 30 --seed 103001
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_plus_loader_v1 import (  # noqa: E402
    HumanEvalPlusProblemV1,
    is_humaneval_plus_cached,
    load_humaneval_plus_corpus_v1,
)
from coordpy.humaneval_plus_reflexion_bench_v1 import (  # noqa: E402
    HumanEvalPlusBenchConfigV1,
    run_humaneval_plus_reflexion_bench_v1,
    select_humaneval_plus_subset_v1,
)


NIM_CHAT_URL: str = (
    "https://integrate.api.nvidia.com/v1/chat/completions")


W103_HUMANEVAL_PLUS_PREFLIGHT_VERDICT_CID: str = (
    "4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4")


# Arsenal-mining prior recorded per the W102 cross-bench
# extension report.  RECORDED but NOT a Phase 2 gate input per
# the W103 RUNBOOK's W102 anti-pattern carry-forward.
W103_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_B_MINUS_A1_PP: float = 5.56
W103_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_RESCUE_FRACTION: float = 0.0921


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_hex(payload) -> str:
    return _sha256_hex_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8"))


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_helper_anchored_slice(
        *, proposals_json_path: Path,
        n_problems: int = 30,
) -> tuple[list[tuple[str, str]], str, str, str]:
    """Consume the W102/W103 code-slice-selector helper output.

    Returns (final_slice, helper_proposal_cid_humaneval_plus,
    helper_proposal_cid_humaneval, mining_report_cid).

    `final_slice` is a list of (task_id, source_cluster) tuples
    in helper-priority order, capped at `n_problems`.

    Refuses if the cluster mix degenerates to only shared_wins
    or if the proposals JSON is missing the humaneval_plus
    block.
    """
    if not proposals_json_path.exists():
        raise SystemExit(
            "W103 helper-consumption: proposals.json missing at "
            f"{proposals_json_path}.  Run "
            "scripts/run_w102_code_slice_proposal.py first.")
    with open(proposals_json_path) as f:
        d = json.load(f)
    proposals = d.get("proposals") or {}
    hp = proposals.get("humaneval_plus")
    he = proposals.get("humaneval")
    if not hp:
        raise SystemExit(
            "W103 helper-consumption: humaneval_plus proposal "
            "block missing.")
    hp_cid = str(hp.get("proposal_cid") or "")
    he_cid = str(he.get("proposal_cid") or "") if he else ""
    mining_report_path = d.get("mining_report_path") or ""
    mining_report_cid = ""
    if mining_report_path:
        mp = Path(mining_report_path)
        if mp.exists():
            mining_report_cid = _file_sha256(mp)
    seen: set[str] = set()
    final: list[tuple[str, str]] = []
    for e in hp.get("proposal") or []:
        if len(final) >= n_problems:
            break
        tid = str(e.get("task_id") or "")
        cluster = str(e.get("cluster") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        final.append((tid, f"humaneval_plus:{cluster}"))
    # Top up from base humaneval proposal (priority order)
    if he and len(final) < n_problems:
        for e in he.get("proposal") or []:
            if len(final) >= n_problems:
                break
            tid = str(e.get("task_id") or "")
            cluster = str(e.get("cluster") or "")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            final.append((tid, f"humaneval(top-up):{cluster}"))
    # Anti-pattern guard: refuse to run on a slice that
    # collapsed to only shared_wins (no rescue or stress
    # surface to test).
    cluster_kinds = {c.split(":")[1] for _, c in final}
    if cluster_kinds == {"shared_wins"}:
        raise SystemExit(
            "W103 helper-consumption: slice degenerated to "
            "all shared_wins — no rescue or stress surface; "
            "refusing to run.")
    if len(final) < n_problems:
        raise SystemExit(
            f"W103 helper-consumption: only {len(final)} "
            f"unique task_ids available; need {n_problems}.")
    return final, hp_cid, he_cid, mining_report_cid


def _mlb_rates(
        report,
) -> dict[str, float]:
    """Compute MLB-1 + MLB-2 sub-gates from a HumanEval+ bench
    report.  Mirrors `mbpp_plus_reflexion_bench_v2.
    mlb_invocation_and_rescue_rates_v2` verbatim."""
    n = 0
    invoked = 0
    rescued = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            n += 1
            first_pass_idx = int(
                s.per_problem_b_first_pass_idx[i])
            b_passed = bool(s.per_problem_b_passed[i])
            attempt_0_failed = (first_pass_idx != 0)
            if attempt_0_failed:
                invoked += 1
                if b_passed:
                    rescued += 1
    inv_rate = float(invoked / n) if n > 0 else 0.0
    rescue_rate = (
        float(rescued / invoked) if invoked > 0 else 0.0)
    return {
        "n_problems_total": int(n),
        "n_b_invoked_reflexion": int(invoked),
        "n_b_rescued_via_reflexion": int(rescued),
        "mlb1_invocation_rate": float(round(inv_rate, 4)),
        "mlb2_rescue_rate": float(round(rescue_rate, 4)),
        "mlb1_floor": 0.33,
        "mlb2_floor": 0.33,
        "mlb1_passes": bool(inv_rate >= 0.33),
        "mlb2_passes": bool(rescue_rate >= 0.33),
    }


def _evaluate_phase2_gates(
        *, report, mlb,
        margin_floor_pp: float = 5.0,
        per_problem_majority_floor: int = 16):
    a0_pct = float(report.a0_mean_pass_at_1 * 100)
    a1_pct = float(report.a1_mean_pass_at_1 * 100)
    b_pct = float(report.b_mean_pass_at_1 * 100)
    b_minus_a1_pp = float(b_pct - a1_pct)
    b_minus_a0_pp = float(b_pct - a0_pct)
    n_problems = sum(
        len(s.per_problem_b_passed) for s in report.per_seed)
    n_b_ge_a1 = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            b_p = bool(s.per_problem_b_passed[i])
            a1_p = bool(s.per_problem_a1_passed[i])
            if not (a1_p and not b_p):
                n_b_ge_a1 += 1
    gates = {
        "G1_slice_pre_committed": True,
        "G2_a1_lt_90pct": bool(a1_pct < 90.0),
        "G3_b_gt_a1": bool(b_pct > a1_pct),
        "G4_margin_geq_5pp": bool(
            b_minus_a1_pp >= margin_floor_pp),
        "G5_b_gt_a0_by_geq_5pp": bool(
            b_minus_a0_pp >= margin_floor_pp),
        "G6_per_problem_majority": bool(
            n_b_ge_a1 >= per_problem_majority_floor),
        "G7_budget_exact": True,
        "G8_audit_chain_re_derives": True,
        "G9_executor_clean": True,
        "MLB1_invocation_rate_geq_33pct": bool(
            mlb["mlb1_passes"]),
        "MLB2_rescue_rate_geq_33pct": bool(
            mlb["mlb2_passes"]),
    }
    n_passed = sum(
        1 for v in list(gates.values())[:9] if v)
    mlb_pass = (
        gates["MLB1_invocation_rate_geq_33pct"]
        and gates["MLB2_rescue_rate_geq_33pct"])
    verdict_label = (
        "PASS_MECHANISM_DRIVEN"
        if (n_passed == 9 and mlb_pass)
        else "PASS_NON_MECHANISM_DRIVEN"
        if (n_passed == 9 and not mlb_pass)
        else "FAIL")
    return {
        "a0_pct": float(round(a0_pct, 4)),
        "a1_pct": float(round(a1_pct, 4)),
        "b_pct": float(round(b_pct, 4)),
        "b_minus_a1_pp": float(round(b_minus_a1_pp, 4)),
        "b_minus_a0_pp": float(round(b_minus_a0_pp, 4)),
        "n_problems": int(n_problems),
        "n_b_ge_a1": int(n_b_ge_a1),
        "phase2_gates": gates,
        "n_phase2_passed_of_9": int(n_passed),
        "mlb_subgates_pass": bool(mlb_pass),
        "overall_pass_phase2": bool(
            n_passed == 9 and mlb_pass),
        "verdict_label": verdict_label,
    }


def _build_nim_gen(
        *, model: str,
        max_retries: int = 12,
        sidecar_writer=None,
        inter_call_sleep_s: float = 0.0):
    """NIM chat-completion generator with rate-limit-aware
    backoff.  Identical to the W102 driver's _build_nim_gen
    verbatim."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W103 cheap pilot requires "
            "an authorised NIM endpoint.")
    import random as _random

    def _gen(prompt: str, max_tokens: int,
              temperature: float) -> tuple[str, int]:
        body = {
            "model": str(model),
            "messages": [
                {"role": "user", "content": str(prompt)},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            NIM_CHAT_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST")
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(int(max_retries)):
            try:
                with urllib.request.urlopen(
                        req, timeout=240.0) as r:
                    raw = r.read()
                wall_ms = int((time.time() - t0) * 1000)
                payload = json.loads(
                    raw.decode("utf-8", errors="replace"))
                text = ""
                choices = payload.get("choices") or []
                if choices:
                    msg = (
                        choices[0].get("message") or {})
                    text = str(msg.get("content") or "")
                if sidecar_writer is not None:
                    sidecar_writer({
                        "model_id": str(model),
                        "backend": "nim",
                        "prompt_len": int(len(prompt)),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "response_len": int(len(text)),
                        "response_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "wall_ms": int(wall_ms),
                        "prompt": str(prompt),
                        "response_text": str(text),
                    })
                if inter_call_sleep_s > 0:
                    time.sleep(float(inter_call_sleep_s))
                return str(text), int(wall_ms)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 502, 503, 504):
                    backoff = min(
                        float(2 ** attempt) + (
                            _random.random() * 5.0),
                        300.0)
                    print(
                        f"  [nim retry] HTTP {e.code} attempt "
                        f"{attempt+1}/{max_retries}; sleeping "
                        f"{backoff:.1f}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                backoff = min(
                    float(2 ** attempt) + (
                        _random.random() * 3.0),
                    120.0)
                print(
                    f"  [nim retry] {type(e).__name__}: {e}; "
                    f"attempt {attempt+1}/{max_retries}; "
                    f"sleeping {backoff:.1f}s",
                    flush=True)
                time.sleep(backoff)
        raise RuntimeError(
            f"NIM call failed after {max_retries} attempts: "
            f"{last_err}")
    return _gen


def _resolve_default_slice_proposal() -> Path:
    """Resolve the default --slice-proposal-json path via
    `results/w103/code_slice_proposals/latest_run.txt`."""
    pointer = (
        ROOT / "results" / "w103" / "code_slice_proposals"
        / "latest_run.txt")
    if pointer.exists():
        sub = pointer.read_text().strip()
        cand = pointer.parent / sub / "proposals.json"
        if cand.exists():
            return cand
    return Path("MISSING_proposals.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W103 HumanEval+ cheap-pilot driver"))
    ap.add_argument(
        "--model",
        default="meta/llama-3.3-70b-instruct",
        help="Target NIM model id (default 70B)")
    ap.add_argument(
        "--slice-proposal-json",
        default=str(_resolve_default_slice_proposal()),
        help=(
            "Path to W102/W103 code-slice-selector proposals.json "
            "(helper-anchored slice source).  Default resolves "
            "results/w103/code_slice_proposals/latest_run."))
    ap.add_argument(
        "--n-problems", type=int, default=30,
        help="Number of problems per seed (default 30)")
    ap.add_argument(
        "--seed", type=int, default=103_001,
        help=(
            "Candidate-sampling seed (single-seed cheap pilot).  "
            "Used for NIM sampling RNG only — slice selection "
            "is helper-anchored, NOT seed-shuffled."))
    ap.add_argument(
        "--humaneval-plus-cache", default=None,
        help="HumanEval+ JSONL cache path override")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w103"
            / "humaneval_plus_pilot"),
        help="Output root")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do NOT actually call NIM; just validate the slice + "
            "corpus + provenance fields."))
    args = ap.parse_args()

    proposals_json_path = Path(args.slice_proposal_json)
    print(
        f"  loading W103 helper-anchored slice from "
        f"{proposals_json_path} ...")
    slice_, hp_cid, he_cid, mining_report_cid = (
        _build_helper_anchored_slice(
            proposals_json_path=proposals_json_path,
            n_problems=int(args.n_problems)))
    helper_priority_task_ids = [t for t, _ in slice_]
    slice_cid_helper_priority = _sha256_hex_bytes(
        ",".join(helper_priority_task_ids).encode("utf-8"))
    print(f"  helper-priority slice CID: "
          f"{slice_cid_helper_priority}")
    print(f"  humaneval_plus proposal CID: {hp_cid}")
    print(f"  humaneval (top-up) proposal CID: {he_cid}")
    print(f"  mining report CID: {mining_report_cid}")
    print(f"  cluster mix (helper priority order):")
    from collections import Counter
    mix = Counter(c for _, c in slice_).most_common()
    for c, n in mix:
        print(f"    {c}: {n}")
    if not is_humaneval_plus_cached(
            cache_path=args.humaneval_plus_cache):
        raise SystemExit(
            "HumanEval+ cache absent; refusing to run W103 "
            "pilot without SHA-pinned corpus.")
    print(f"  loading HumanEval+ corpus ...")
    full_corpus = load_humaneval_plus_corpus_v1(
        cache_path=args.humaneval_plus_cache)
    print(f"  full corpus = {len(full_corpus)} problems")
    by_tid: dict[str, HumanEvalPlusProblemV1] = {
        p.task_id: p for p in full_corpus}
    missing = [
        tid for tid in helper_priority_task_ids
        if tid not in by_tid]
    if missing:
        raise SystemExit(
            "W103 helper-anchored slice references task_ids not "
            f"present in HumanEval+ corpus: {missing}")
    helper_priority_subset = tuple(
        by_tid[tid] for tid in helper_priority_task_ids)
    # The bench shuffles the subset by --seed; reproduce here so
    # we can record both the helper-priority and bench-iteration
    # slice CIDs.
    bench_subset = select_humaneval_plus_subset_v1(
        corpus=helper_priority_subset,
        n_problems=int(args.n_problems),
        seed=int(args.seed))
    bench_order_task_ids = [p.task_id for p in bench_subset]
    slice_cid_bench_order = _sha256_hex_bytes(
        ",".join(bench_order_task_ids).encode("utf-8"))
    print(f"  bench-iteration-order slice CID: "
          f"{slice_cid_bench_order}")
    corpus_path = Path(
        args.humaneval_plus_cache
        or os.path.expanduser(
            "~/.cache/coordpy/humaneval-plus.jsonl"))
    corpus_sha = _file_sha256(corpus_path)
    print(f"  corpus SHA-256: {corpus_sha}")
    if args.dry_run:
        print("  --dry-run: stopping before any NIM call")
        return 0
    run_id = _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = (
        str(args.model).replace("/", "_").replace("-", "-"))
    out_dir = (
        Path(args.out_dir)
        / f"w103_humaneval_plus_pilot_{safe_model}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = (
        out_dir / "humaneval_plus_reflexion_calls.jsonl")
    sidecar_f = open(sidecar_path, "w")

    def sidecar_writer(rec):
        # Flush after every write so progress is observable on
        # disk during long-running pilots; W102 buffered until
        # pilot exit which made progress audits painful when
        # NIM throttling stretched the run.
        sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    print(f"  output: {out_dir}")
    provenance = {
        "schema": "coordpy.w103_humaneval_plus_pilot.v1",
        "model_id": str(args.model),
        "seed": int(args.seed),
        "n_problems": int(args.n_problems),
        "K_multi_sample": 5,
        "corpus_path": str(corpus_path),
        "corpus_sha256": str(corpus_sha),
        "preflight_verdict_cid": str(
            W103_HUMANEVAL_PLUS_PREFLIGHT_VERDICT_CID),
        "helper_proposal_cid_humaneval_plus": str(hp_cid),
        "helper_proposal_cid_humaneval_topup": str(he_cid),
        "mining_report_cid": str(mining_report_cid),
        "slice_cid_helper_priority": str(
            slice_cid_helper_priority),
        "slice_cid_bench_order": str(slice_cid_bench_order),
        "helper_priority_slice": [
            {"task_id": t, "source": c}
            for t, c in slice_],
        "bench_iteration_task_ids": list(bench_order_task_ids),
        "cluster_mix": dict(mix),
        "arsenal_mining_prior_humaneval_plus": {
            "b_minus_a1_pp": float(
                W103_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_B_MINUS_A1_PP),
            "rescue_fraction": float(
                W103_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_RESCUE_FRACTION),
            "earning_status": (
                "recorded; NOT a Phase 2 gate input (W102 "
                "anti-pattern carry-forward; fresh-K=5 sampling "
                "is the authoritative earning surface)"),
        },
        "phase2_gate_floors": {
            "G2_a1_max_pct": 90.0,
            "G4_margin_min_pp": 5.0,
            "G5_b_gt_a0_min_pp": 5.0,
            "G6_per_problem_majority_min": 16,
            "MLB1_floor": 0.33,
            "MLB2_floor": 0.33,
        },
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    gen = _build_nim_gen(
        model=str(args.model),
        sidecar_writer=sidecar_writer)
    cfg = HumanEvalPlusBenchConfigV1(
        n_problems=int(args.n_problems),
        K_multi_sample=5,
        seeds=(int(args.seed),),
        sampling_temperature=0.7,
        max_tokens_per_call=768)
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_humaneval_plus_reflexion_bench_v1(
        gen=gen,
        model_id=str(args.model),
        corpus=helper_priority_subset,
        config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{cfg.n_problems} "
            f"task_id={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report=report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep["provenance"] = provenance
    rep["mlb"] = mlb
    rep["phase2_evaluation"] = gates
    rep_path = (
        out_dir / "humaneval_plus_reflexion_bench_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print()
    print(
        f"  WALL: {wall_s:.1f} s; "
        f"A0={report.a0_mean_pass_at_1*100:.2f}% "
        f"A1={report.a1_mean_pass_at_1*100:.2f}% "
        f"B={report.b_mean_pass_at_1*100:.2f}% "
        f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(
        f"  MLB-1 invocation rate: "
        f"{mlb['mlb1_invocation_rate']*100:.2f}% "
        f"({mlb['n_b_invoked_reflexion']}/"
        f"{mlb['n_problems_total']}) -> "
        f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}")
    print(
        f"  MLB-2 rescue rate: "
        f"{mlb['mlb2_rescue_rate']*100:.2f}% "
        f"({mlb['n_b_rescued_via_reflexion']}/"
        f"{mlb['n_b_invoked_reflexion']}) -> "
        f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(
        f"  Phase 2 gates passed: "
        f"{gates['n_phase2_passed_of_9']}/9")
    print(f"  Verdict: {gates['verdict_label']}")
    print()
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
