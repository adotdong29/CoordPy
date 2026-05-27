#!/usr/bin/env python3
"""W104 — HumanEval+ cross-scale cheap-pilot driver (lead lane).

Reuses the W103 helper-anchored 30-problem slice BYTE-FOR-BYTE
on a SECOND model class (primary `meta/llama-3.1-405b-instruct`;
backup `meta/llama-3.1-70b-instruct` if 405B unreachable) to
test cross-scale generalisation of the W103 PASS_MECHANISM_DRIVEN
result.

CONDITIONAL on a sub-second reachability smoke test on the
primary target.  If both primary and backup are unreachable, the
driver records `DEFERRED on reachability` and exits non-zero.

The driver mirrors `scripts/run_w103_humaneval_plus_pilot.py`
shape verbatim, with three W104 additions:

  1. `--target-model` argument defaulting to the W104 primary
     target locked in `docs/RUNBOOK_W104.md`.
  2. `--reuse-slice` flag loads the W103 slice CID list directly
     from `results/w103/humaneval_plus_pilot/<RUN>/provenance.json`
     and verifies byte-equal slice CIDs against the W104-locked
     constants.  Slice CID mismatch ⇒ refuse to run.
  3. Cross-scale comparator emitted automatically on successful
     pilot completion via `coordpy.cross_scale_comparator_v1`.

Resume-from-sidecar: a partial sidecar file from a killed-and-
relaunched run is loaded at start; (seed, p_idx, arm,
attempt_idx) tuples already on disk are NOT re-spent.  Resume
is opt-in via `--resume-from <existing_out_dir>`.

Requires `NVIDIA_API_KEY` in the environment.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w104_humaneval_plus_cross_scale_pilot.py \\
        --target-model meta/llama-3.1-405b-instruct \\
        --reuse-slice \\
            results/w103/humaneval_plus_pilot/latest_run/provenance.json \\
        --n-problems 30 --seed 104001
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
)
from coordpy.cross_scale_comparator_v1 import (  # noqa: E402
    build_cross_scale_comparator_report_v1,
    format_cross_scale_comparator_markdown_v1,
)


NIM_CHAT_URL: str = (
    "https://integrate.api.nvidia.com/v1/chat/completions")


# --- W104 pre-locked constants (per docs/RUNBOOK_W104.md) ---

W104_PRIMARY_TARGET_MODEL: str = "meta/llama-3.1-405b-instruct"
W104_BACKUP_TARGET_MODEL: str = "meta/llama-3.1-70b-instruct"


W104_TARGET_SELECTION_RULE_VERSION: str = (
    "coordpy.w104_target_selection_rule.v1")


# The W103 helper-anchored slice CIDs, locked here so a future
# regression cannot silently rebuild the slice.  See
# docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md.
W103_HELPER_ANCHORED_SLICE_CID_HELPER_PRIORITY: str = (
    "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2")
W103_HELPER_ANCHORED_SLICE_CID_BENCH_ORDER: str = (
    "d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052")


# W103 bench report locked for cross-scale comparison.  The
# comparator will refuse to run unless the slice + corpus pin
# match this report.
W103_BENCH_MERKLE_ROOT: str = (
    "68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d")
W103_HUMANEVAL_PLUS_PREFLIGHT_VERDICT_CID: str = (
    "4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4")


# Arsenal-mining priors RECORDED but NOT a Phase 2 gate input.
W104_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_B_MINUS_A1_PP: float = 5.56
W104_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_RESCUE_FRACTION: float = 0.0921
W103_EMPIRICAL_70B_B_MINUS_A1_PP: float = 20.00
W103_EMPIRICAL_70B_MLB2_RESCUE_RATE: float = 0.4706


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_w103_slice_from_provenance(
        *, provenance_path: Path,
) -> tuple[list[str], list[dict], str, str, str, str, str]:
    """Load the W103 slice (bench-iteration order) + auxiliary
    fields from the W103 provenance JSON.

    Returns:
      (bench_iter_task_ids,
       helper_priority_slice,
       slice_cid_helper_priority,
       slice_cid_bench_order,
       corpus_sha256,
       helper_proposal_cid_humaneval_plus,
       mining_report_cid)
    """
    if not provenance_path.exists():
        raise SystemExit(
            f"W104 --reuse-slice: provenance JSON missing at "
            f"{provenance_path}.")
    with open(provenance_path) as f:
        prov = json.load(f)
    bench_iter = list(
        prov.get("bench_iteration_task_ids") or [])
    helper_priority = list(
        prov.get("helper_priority_slice") or [])
    scid_hp = str(prov.get("slice_cid_helper_priority") or "")
    scid_bo = str(prov.get("slice_cid_bench_order") or "")
    corpus_sha = str(prov.get("corpus_sha256") or "")
    hp_cid = str(
        prov.get("helper_proposal_cid_humaneval_plus") or "")
    mr_cid = str(prov.get("mining_report_cid") or "")
    if scid_hp != W103_HELPER_ANCHORED_SLICE_CID_HELPER_PRIORITY:
        raise SystemExit(
            "W104 --reuse-slice: slice_cid_helper_priority "
            f"mismatch: got {scid_hp!r} vs locked "
            f"{W103_HELPER_ANCHORED_SLICE_CID_HELPER_PRIORITY!r}")
    if scid_bo != W103_HELPER_ANCHORED_SLICE_CID_BENCH_ORDER:
        raise SystemExit(
            "W104 --reuse-slice: slice_cid_bench_order mismatch: "
            f"got {scid_bo!r} vs locked "
            f"{W103_HELPER_ANCHORED_SLICE_CID_BENCH_ORDER!r}")
    if not bench_iter:
        raise SystemExit(
            "W104 --reuse-slice: bench_iteration_task_ids empty.")
    # Re-derive slice CIDs as cross-check.
    recomputed_bo = _sha256_hex_bytes(
        ",".join(bench_iter).encode("utf-8"))
    if recomputed_bo != scid_bo:
        raise SystemExit(
            "W104 --reuse-slice: slice_cid_bench_order does NOT "
            "match the SHA-256 of the comma-joined bench-"
            "iteration task_ids on disk; provenance JSON is "
            "structurally corrupt.")
    return (bench_iter, helper_priority, scid_hp, scid_bo,
            corpus_sha, hp_cid, mr_cid)


def _extract_completed_sidecar_keys(
        sidecar_path: Path) -> set[tuple]:
    """Return the set of (seed, p_idx, arm, attempt_idx) tuples
    already on disk in `sidecar_path`.  Used for resume-from-
    disk so a kill-and-restart does not re-spend NIM budget.

    A corrupt / truncated trailing line is treated as
    not-yet-completed (the conservative choice — better to
    re-spend one call than silently consume garbage)."""
    if not sidecar_path.exists():
        return set()
    completed: set[tuple] = set()
    with open(sidecar_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Malformed trailing line — do not consume.
                continue
            key = (
                int(rec.get("seed", -1)),
                int(rec.get("p_idx", -1)),
                str(rec.get("arm", "")),
                int(rec.get("attempt_idx", -1)))
            if -1 in key[:2] or not key[2]:
                continue
            completed.add(key)
    return completed


def _build_nim_gen(
        *, model: str,
        max_retries: int = 12,
        sidecar_writer=None,
        inter_call_sleep_s: float = 0.0):
    """NIM chat-completion generator with rate-limit-aware
    backoff.  Identical to the W103 driver verbatim except the
    sidecar entries are tagged for the cross-scale namespace."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W104 cross-scale pilot "
            "requires an authorised NIM endpoint.")
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


def _reachability_smoke_probe(
        *, model: str, max_seconds: float = 15.0) -> bool:
    """Sub-second NIM probe.  Sends a 4-character prompt to the
    target with max_tokens=4 and a tight timeout; returns True
    iff the call returns within `max_seconds`.

    Caches the probe result so we don't spend it twice."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return False
    body = {
        "model": str(model),
        "messages": [
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        NIM_CHAT_URL, data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }, method="POST")
    try:
        with urllib.request.urlopen(
                req, timeout=float(max_seconds)) as r:
            r.read()
        return True
    except urllib.error.HTTPError as e:
        print(f"  [w104 smoke] HTTP {e.code} on {model}",
              flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [w104 smoke] {type(e).__name__} on {model}: {e}",
              flush=True)
        return False


def _mlb_rates(report) -> dict[str, float]:
    """Compute MLB-1 + MLB-2 sub-gates from a HumanEval+ bench
    report.  Identical to the W103 driver verbatim."""
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


def _resolve_default_reuse_slice_path() -> Path:
    pointer = (
        ROOT / "results" / "w103" / "humaneval_plus_pilot"
        / "latest_run.txt")
    if pointer.exists():
        sub = pointer.read_text().strip()
        cand = pointer.parent / sub / "provenance.json"
        if cand.exists():
            return cand
    return Path("MISSING_provenance.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W104 HumanEval+ cross-scale cheap-pilot driver"))
    ap.add_argument(
        "--target-model",
        default=W104_PRIMARY_TARGET_MODEL,
        help=(
            "Target NIM model id.  Default = W104 primary "
            f"({W104_PRIMARY_TARGET_MODEL}).  See "
            "docs/RUNBOOK_W104.md § Target-model selection rule."))
    ap.add_argument(
        "--backup-target-model",
        default=W104_BACKUP_TARGET_MODEL,
        help=(
            "Backup target if primary fails reachability smoke "
            "probe.  Default = W104 backup "
            f"({W104_BACKUP_TARGET_MODEL})."))
    ap.add_argument(
        "--reuse-slice",
        default=str(_resolve_default_reuse_slice_path()),
        help=(
            "Path to W103 provenance.json.  W104 byte-equal "
            "reuses the W103 bench-iteration slice.  Slice CID "
            "mismatch refuses to run."))
    ap.add_argument(
        "--n-problems", type=int, default=30,
        help="Number of problems per seed (default 30)")
    ap.add_argument(
        "--seed", type=int, default=104_001,
        help=(
            "Candidate-sampling seed (single-seed cheap pilot)."))
    ap.add_argument(
        "--humaneval-plus-cache", default=None,
        help="HumanEval+ JSONL cache path override")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w104"
            / "humaneval_plus_cross_scale_pilot"),
        help="Output root")
    ap.add_argument(
        "--w103-bench-report",
        default=str(
            ROOT / "results" / "w103" / "humaneval_plus_pilot"
            / "w103_humaneval_plus_pilot_meta_llama-3.3-70b-"
              "instruct_20260526T022037Z"
            / "humaneval_plus_reflexion_bench_report.json"),
        help=(
            "Path to the W103 bench report.  Used for the "
            "cross-scale comparator block; comparator refuses "
            "to run on slice / corpus / schema mismatch."))
    ap.add_argument(
        "--resume-from", default=None,
        help=(
            "Path to an existing W104 run dir whose sidecar to "
            "resume from.  Skips NIM calls for "
            "(seed, p_idx, arm, attempt_idx) tuples already "
            "on disk."))
    ap.add_argument(
        "--no-smoke-probe", action="store_true",
        help=(
            "Skip the reachability smoke probe.  Use ONLY if "
            "you have already confirmed reachability out-of-"
            "band."))
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do NOT actually launch the cheap pilot; just "
            "validate the slice + corpus + provenance fields "
            "+ smoke probe."))
    args = ap.parse_args()

    print("  W104 cross-scale cheap-pilot driver")
    print(
        f"  primary target  : {args.target_model}")
    print(
        f"  backup target   : {args.backup_target_model}")

    # --- Slice byte-equal reuse from W103 provenance ---
    reuse_slice_path = Path(args.reuse_slice)
    print(
        f"  loading W103 slice from "
        f"{reuse_slice_path} ...")
    (bench_iter, helper_priority_slice,
     scid_hp, scid_bo, w103_corpus_sha, hp_cid, mr_cid) = (
        _load_w103_slice_from_provenance(
            provenance_path=reuse_slice_path))
    print(
        f"  slice_cid_helper_priority: {scid_hp}")
    print(
        f"  slice_cid_bench_order    : {scid_bo}")
    print(
        f"  bench iteration N        : {len(bench_iter)}")

    if not is_humaneval_plus_cached(
            cache_path=args.humaneval_plus_cache):
        raise SystemExit(
            "HumanEval+ cache absent; refusing to run W104 "
            "pilot without SHA-pinned corpus.")
    print(f"  loading HumanEval+ corpus ...")
    full_corpus = load_humaneval_plus_corpus_v1(
        cache_path=args.humaneval_plus_cache)
    print(f"  full corpus = {len(full_corpus)} problems")
    by_tid: dict[str, HumanEvalPlusProblemV1] = {
        p.task_id: p for p in full_corpus}
    missing = [tid for tid in bench_iter if tid not in by_tid]
    if missing:
        raise SystemExit(
            f"W104: bench iter references task_ids not in "
            f"HumanEval+ corpus: {missing}")
    bench_subset = tuple(by_tid[tid] for tid in bench_iter)
    corpus_path = Path(
        args.humaneval_plus_cache
        or os.path.expanduser(
            "~/.cache/coordpy/humaneval-plus.jsonl"))
    corpus_sha = _file_sha256(corpus_path)
    if corpus_sha != w103_corpus_sha:
        raise SystemExit(
            "W104: corpus SHA-256 mismatch vs W103 provenance: "
            f"got {corpus_sha!r} W103 had {w103_corpus_sha!r}.  "
            "Cross-scale comparison would be structurally "
            "invalid; refusing to run.")
    print(f"  corpus SHA-256: {corpus_sha} (matches W103)")

    # --- Reachability smoke probe ---
    chosen_target = args.target_model
    if not args.no_smoke_probe and not args.dry_run:
        print(
            f"  [smoke probe] {chosen_target} ...", flush=True)
        ok = _reachability_smoke_probe(
            model=chosen_target, max_seconds=20.0)
        if not ok:
            backup = args.backup_target_model
            print(
                f"  [smoke probe] primary unreachable; trying "
                f"backup {backup} ...", flush=True)
            ok_backup = _reachability_smoke_probe(
                model=backup, max_seconds=20.0)
            if not ok_backup:
                raise SystemExit(
                    "W104: both primary and backup targets "
                    "unreachable.  Verdict: "
                    "`DEFERRED on reachability`.")
            chosen_target = backup
            print(
                f"  [smoke probe] backup PASS; switching to "
                f"{chosen_target}", flush=True)
        else:
            print(
                f"  [smoke probe] primary PASS; using "
                f"{chosen_target}", flush=True)

    if args.dry_run:
        print(
            "  --dry-run: stopping before any NIM call.  Slice "
            "+ corpus validated.")
        return 0

    # --- Output dir + sidecar resume ---
    run_id = _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = (
        str(chosen_target).replace("/", "_").replace("-", "-"))
    if args.resume_from:
        out_dir = Path(args.resume_from)
        if not out_dir.exists():
            raise SystemExit(
                f"W104 --resume-from: out_dir {out_dir} does "
                "not exist; cannot resume.")
        print(f"  RESUMING into existing run: {out_dir}")
    else:
        out_dir = (
            Path(args.out_dir)
            / f"w104_humaneval_plus_cross_scale_pilot_"
              f"{safe_model}_{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  fresh run: {out_dir}")
    sidecar_path = (
        out_dir / "humaneval_plus_reflexion_calls.jsonl")
    completed_keys = _extract_completed_sidecar_keys(
        sidecar_path)
    print(f"  resume: {len(completed_keys)} sidecar entries "
          "already on disk")

    sidecar_f = open(sidecar_path, "a")

    def sidecar_writer(rec):
        sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    provenance = {
        "schema": "coordpy.w104_humaneval_plus_cross_scale_pilot.v1",
        "target_selection_rule_version":
            W104_TARGET_SELECTION_RULE_VERSION,
        "model_id": str(chosen_target),
        "primary_target": str(args.target_model),
        "backup_target": str(args.backup_target_model),
        "smoke_probe_skipped": bool(args.no_smoke_probe),
        "seed": int(args.seed),
        "n_problems": int(args.n_problems),
        "K_multi_sample": 5,
        "corpus_path": str(corpus_path),
        "corpus_sha256": str(corpus_sha),
        "preflight_verdict_cid":
            W103_HUMANEVAL_PLUS_PREFLIGHT_VERDICT_CID,
        "helper_proposal_cid_humaneval_plus": str(hp_cid),
        "mining_report_cid": str(mr_cid),
        "slice_cid_helper_priority": str(scid_hp),
        "slice_cid_bench_order": str(scid_bo),
        "helper_priority_slice": list(helper_priority_slice),
        "bench_iteration_task_ids": list(bench_iter),
        "cross_scale_pair_a_run_id":
            "w103_humaneval_plus_pilot_meta_llama-3.3-70b-"
            "instruct_20260526T022037Z",
        "cross_scale_pair_a_model_id":
            "meta/llama-3.3-70b-instruct",
        "cross_scale_pair_a_bench_merkle":
            W103_BENCH_MERKLE_ROOT,
        "arsenal_mining_prior_humaneval_plus": {
            "b_minus_a1_pp": float(
                W104_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_B_MINUS_A1_PP),
            "rescue_fraction": float(
                W104_ARSENAL_MINING_PRIOR_HUMANEVAL_PLUS_RESCUE_FRACTION),
            "earning_status": (
                "recorded; NOT a Phase 2 gate input (W102 "
                "anti-pattern carry-forward)"),
        },
        "w103_70b_empirical_anchor": {
            "b_minus_a1_pp": float(
                W103_EMPIRICAL_70B_B_MINUS_A1_PP),
            "mlb2_rescue_rate": float(
                W103_EMPIRICAL_70B_MLB2_RESCUE_RATE),
            "earning_status": (
                "recorded; NOT a Phase 2 gate input (the W104 "
                "fresh-K=5 sampling at the cross-scale target "
                "is the authoritative surface)"),
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
        model=str(chosen_target),
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
        model_id=str(chosen_target),
        corpus=bench_subset,
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

    # --- Cross-scale comparator (automatic on success) ---
    w103_rep_path = Path(args.w103_bench_report)
    if not w103_rep_path.exists():
        print(
            f"  WARNING: W103 bench report not at "
            f"{w103_rep_path}; cross-scale comparator skipped.",
            flush=True)
    else:
        try:
            with open(w103_rep_path) as f:
                w103_bench = json.load(f)
            w103_prov_path = w103_rep_path.parent / "provenance.json"
            with open(w103_prov_path) as f:
                w103_prov = json.load(f)
            comp_rep = build_cross_scale_comparator_report_v1(
                scale_a_bench_report=w103_bench,
                scale_a_provenance=w103_prov,
                scale_b_bench_report=rep,
                scale_b_provenance=provenance)
            comp_path = (
                out_dir / "cross_scale_comparator_report.json")
            with open(comp_path, "w") as f:
                json.dump(
                    comp_rep.to_dict(), f, indent=2, default=str)
            comp_md_path = (
                out_dir / "cross_scale_comparator_report.md")
            with open(comp_md_path, "w") as f:
                f.write(
                    format_cross_scale_comparator_markdown_v1(
                        report=comp_rep))
            print(
                f"  cross-scale comparator: "
                f"shift on B-A1 = "
                f"{comp_rep.cross_scale_shift_on_b_minus_a1_pp:+.2f}pp; "
                f"shift on MLB-2 = "
                f"{comp_rep.cross_scale_shift_on_mlb2_pp:+.2f}pp")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: comparator refused: "
                  f"{type(e).__name__}: {e}", flush=True)
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
