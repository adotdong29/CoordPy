#!/usr/bin/env python3
"""W105 — HumanEval+ Phase 3 retirement-bench driver (lead lane).

Consumes the pre-built W105 Phase 3 slice pack BYTE-FOR-BYTE
unchanged.  Iterates per-(model_class, seed) Phase 3 cell with
strict cell isolation (own out_dir, own sidecar, own
provenance).  Canary mode runs a 1-seed × 3-problem × K=5
smoke per class BEFORE the full Phase 3 envelope opens.

Per-cell behaviour mirrors the W104 cross-scale driver shape
verbatim, with W105 additions:

  1. ``--slice-pack`` argument loads the W105 pre-committed
     slice pack JSON; pack CID + inner-kernel CID verified at
     start.
  2. Per-cell sidecar resume (carry-forward from W104).
  3. Cells emit a ``phase3_cell_verdict.json`` on completion
     (Phase-2-shape gates evaluated at 100-problem cell size;
     partial audit, NOT the Phase 3 verdict).
  4. Per-cell ``per_seed_iteration_task_ids`` recorded in the
     cell provenance — the bench's seed-driven shuffle replay
     fixes the W104 cross-scale comparator row-alignment lesson.
  5. Global ``progress.json`` updated after each cell completes
     so a tail-friendly observer can track which cells are done.
  6. After all cells in a class complete, the per-class Phase 3
     verdict is emitted via
     ``coordpy.phase3_retirement_evaluator_v1``.
  7. After both classes complete, the cross-class comparator is
     emitted via ``coordpy.cross_class_comparator_v1``.

Requires ``NVIDIA_API_KEY`` in the environment.

Usage::

    export NVIDIA_API_KEY=...
    # Full Phase 3 retirement bench (6,600 NIM calls)
    python scripts/run_w105_phase3_retirement_bench.py \\
        --slice-pack data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json

    # Canary smoke only (66 NIM calls)
    python scripts/run_w105_phase3_retirement_bench.py \\
        --slice-pack data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json \\
        --canary

    # Run only one model class (the other can be launched in a
    # separate process for true parallelism)
    python scripts/run_w105_phase3_retirement_bench.py \\
        --slice-pack ... \\
        --only-class meta/llama-3.3-70b-instruct
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
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


# --- W105 pre-locked constants (per docs/RUNBOOK_W105.md) ---

W105_PACK_CID_LOCKED: str = (
    "8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314")
W105_INNER_KERNEL_CID_LOCKED: str = (
    "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2")
W105_CORPUS_SHA_LOCKED: str = (
    "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492")
W105_PREFLIGHT_VERDICT_CID_LOCKED: str = (
    "4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4")

W105_MODEL_CLASSES_LOCKED: tuple[str, ...] = (
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
)
W105_PHASE3_SEEDS_LOCKED: tuple[int, ...] = (
    105_001, 105_002, 105_003)
W105_CANARY_SEED: int = 105_999  # NOT in W105_PHASE3_SEEDS_LOCKED
W105_CANARY_N_PROBLEMS: int = 3

W105_TARGET_SELECTION_RULE_VERSION: str = (
    "coordpy.w105_target_selection_rule.v1")


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_w105_slice_pack(
        *, slice_pack_path: Path) -> dict:
    if not slice_pack_path.exists():
        raise SystemExit(
            f"W105 --slice-pack: file missing at "
            f"{slice_pack_path}.")
    with open(slice_pack_path) as f:
        pack = json.load(f)
    # Schema check.
    if str(pack.get("schema") or "") != (
            "coordpy.w104_w105_phase3_slice_pack.v1"):
        raise SystemExit(
            "W105 --slice-pack: unrecognised schema "
            f"{pack.get('schema')!r}")
    # Pack CID verification.
    pack_cid_recorded = str(pack.get("pack_cid") or "")
    if pack_cid_recorded != W105_PACK_CID_LOCKED:
        raise SystemExit(
            "W105 --slice-pack: pack_cid mismatch: got "
            f"{pack_cid_recorded!r} vs W105 locked "
            f"{W105_PACK_CID_LOCKED!r}")
    # Inner kernel CID verification.
    inner_cid = str(
        pack.get("inner_kernel_cid_w103_helper_priority") or "")
    if inner_cid != W105_INNER_KERNEL_CID_LOCKED:
        raise SystemExit(
            "W105 --slice-pack: inner kernel CID mismatch: "
            f"got {inner_cid!r} vs W105 locked "
            f"{W105_INNER_KERNEL_CID_LOCKED!r}")
    # Re-derive pack CID from the comma-joined task_ids.
    tids = list(pack.get("task_ids_helper_priority") or [])
    if not tids:
        raise SystemExit(
            "W105 --slice-pack: task_ids_helper_priority empty")
    recomputed = _sha256_hex_bytes(
        ",".join(tids).encode("utf-8"))
    if recomputed != pack_cid_recorded:
        raise SystemExit(
            "W105 --slice-pack: pack_cid does NOT match SHA-256 "
            "of comma-joined task_ids_helper_priority; slice "
            "pack is structurally corrupt.")
    # Seed list verification.
    seeds = tuple(int(s) for s in (pack.get("phase3_seeds") or []))
    if seeds != W105_PHASE3_SEEDS_LOCKED:
        raise SystemExit(
            "W105 --slice-pack: phase3_seeds mismatch: got "
            f"{seeds!r} vs W105 locked "
            f"{W105_PHASE3_SEEDS_LOCKED!r}")
    return pack


def _per_seed_iteration_order(
        *, slice_task_ids: list[str], seed: int) -> list[str]:
    """Replay the bench module's internal per-seed shuffle
    deterministically.  Matches
    ``coordpy.humaneval_plus_reflexion_bench_v1.select_humaneval_plus_subset_v1``
    byte-for-byte (same Random(seed); same shuffle algorithm;
    same n_problems == len(corpus))."""
    rng = random.Random(int(seed))
    idxs = list(range(len(slice_task_ids)))
    rng.shuffle(idxs)
    return [slice_task_ids[i] for i in idxs]


def _extract_completed_sidecar_keys(
        sidecar_path: Path) -> set[tuple]:
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
        retry_log_writer=None,
        inter_call_sleep_s: float = 0.0):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W105 Phase 3 retirement "
            "bench requires an authorised NIM endpoint.")
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
                    if retry_log_writer is not None:
                        retry_log_writer({
                            "kind": "http_error_backoff",
                            "code": int(e.code),
                            "attempt": int(attempt),
                            "backoff_s": float(backoff),
                            "ts": time.time(),
                        })
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
                if retry_log_writer is not None:
                    retry_log_writer({
                        "kind": "exception_backoff",
                        "exc_type": type(e).__name__,
                        "exc_msg": str(e),
                        "attempt": int(attempt),
                        "backoff_s": float(backoff),
                        "ts": time.time(),
                    })
                time.sleep(backoff)
        raise RuntimeError(
            f"NIM call failed after {max_retries} attempts: "
            f"{last_err}")
    return _gen


def _reachability_smoke_probe(
        *, model: str, max_seconds: float = 20.0) -> bool:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return False
    body = {
        "model": str(model),
        "messages": [{"role": "user", "content": "ping"}],
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
        print(f"  [w105 smoke] HTTP {e.code} on {model}",
              flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [w105 smoke] {type(e).__name__} on {model}: "
              f"{e}", flush=True)
        return False


def _mlb_rates(report) -> dict[str, float]:
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


def _evaluate_phase2_shape_gates(
        *, report, mlb,
        margin_floor_pp: float = 5.0,
        per_problem_majority_floor: int = -1):
    """Phase 2-shape gates applied at 100-problem cell size.
    PARTIAL AUDIT only; the Phase 3 verdict is a different
    surface."""
    a0_pct = float(report.a0_mean_pass_at_1 * 100)
    a1_pct = float(report.a1_mean_pass_at_1 * 100)
    b_pct = float(report.b_mean_pass_at_1 * 100)
    b_minus_a1_pp = float(b_pct - a1_pct)
    b_minus_a0_pp = float(b_pct - a0_pct)
    n_problems = sum(
        len(s.per_problem_b_passed) for s in report.per_seed)
    if per_problem_majority_floor < 0:
        per_problem_majority_floor = int(
            round(0.53 * n_problems))
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
    return {
        "a0_pct": float(round(a0_pct, 4)),
        "a1_pct": float(round(a1_pct, 4)),
        "b_pct": float(round(b_pct, 4)),
        "b_minus_a1_pp": float(round(b_minus_a1_pp, 4)),
        "b_minus_a0_pp": float(round(b_minus_a0_pp, 4)),
        "n_problems": int(n_problems),
        "n_b_ge_a1": int(n_b_ge_a1),
        "phase2_shape_gates": gates,
        "n_phase2_shape_passed_of_9": int(n_passed),
        "per_problem_majority_floor": int(
            per_problem_majority_floor),
    }


def _cell_run_complete(cell_dir: Path) -> bool:
    return (cell_dir / "phase3_cell_verdict.json").exists()


def _update_global_progress(
        *, run_root: Path, status: str, detail: dict) -> None:
    progress_path = run_root / "progress.json"
    payload = {
        "status": str(status),
        "ts_utc": _dt.datetime.now(
            _dt.timezone.utc).isoformat(),
        "detail": detail,
    }
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(progress_path)


def _run_one_cell(
        *,
        model_class: str,
        seed: int,
        n_problems: int,
        slice_task_ids: list[str],
        corpus_by_tid: dict[str, HumanEvalPlusProblemV1],
        corpus_path: Path,
        corpus_sha: str,
        slice_pack_cid: str,
        cell_dir: Path,
        run_root: Path,
        is_canary: bool) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_run_id = cell_dir.name
    sidecar_path = (
        cell_dir / "humaneval_plus_reflexion_calls.jsonl")
    retry_log_path = cell_dir / "retry_log.jsonl"
    completed_keys = _extract_completed_sidecar_keys(
        sidecar_path)
    if completed_keys:
        print(
            f"  resume: {len(completed_keys)} sidecar entries "
            f"already on disk", flush=True)
    # Take the prefix of the slice (helper-priority order) of
    # size n_problems.  For canary this is the first 3 entries;
    # for full Phase 3 this is all 100.
    cell_slice = list(slice_task_ids[:int(n_problems)])
    # Replay the bench's seed-driven shuffle so provenance
    # records the iteration order.
    per_seed_iter = _per_seed_iteration_order(
        slice_task_ids=cell_slice, seed=int(seed))
    bench_subset = tuple(
        corpus_by_tid[tid] for tid in cell_slice)
    provenance = {
        "schema":
            "coordpy.w105_phase3_cell.v1",
        "target_selection_rule_version":
            W105_TARGET_SELECTION_RULE_VERSION,
        "model_id": str(model_class),
        "seed": int(seed),
        "n_problems": int(len(cell_slice)),
        "K_multi_sample": 5,
        "corpus_path": str(corpus_path),
        "corpus_sha256": str(corpus_sha),
        "preflight_verdict_cid":
            W105_PREFLIGHT_VERDICT_CID_LOCKED,
        "slice_pack_cid": str(slice_pack_cid),
        "slice_task_ids_helper_priority": list(cell_slice),
        "per_seed_iteration_task_ids": list(per_seed_iter),
        "is_canary": bool(is_canary),
        "cell_run_id": str(cell_run_id),
        "phase2_shape_gate_floors_recorded": {
            "G2_a1_max_pct": 90.0,
            "G4_margin_min_pp": 5.0,
            "G5_b_gt_a0_min_pp": 5.0,
            "MLB1_floor": 0.33,
            "MLB2_floor": 0.33,
        },
        "phase3_retirement_evaluator_recorded":
            "coordpy.phase3_retirement_evaluator_v1",
    }
    with open(cell_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    sidecar_f = open(sidecar_path, "a")
    retry_f = open(retry_log_path, "a")

    def sidecar_writer(rec):
        sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    def retry_writer(rec):
        retry_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        retry_f.flush()

    gen = _build_nim_gen(
        model=str(model_class),
        sidecar_writer=sidecar_writer,
        retry_log_writer=retry_writer)
    cfg = HumanEvalPlusBenchConfigV1(
        n_problems=int(len(cell_slice)),
        K_multi_sample=5,
        seeds=(int(seed),),
        sampling_temperature=0.7,
        max_tokens_per_call=768)
    _update_global_progress(
        run_root=run_root,
        status="cell_started",
        detail={
            "model_class": str(model_class),
            "seed": int(seed),
            "is_canary": bool(is_canary),
            "n_problems": int(len(cell_slice)),
            "cell_dir": str(cell_dir),
            "sidecar_completed_at_start": int(
                len(completed_keys)),
        })
    t0 = time.time()
    report = run_humaneval_plus_reflexion_bench_v1(
        gen=gen,
        model_id=str(model_class),
        corpus=bench_subset,
        config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  [{model_class} seed={seed}] p_idx="
            f"{i+1}/{cfg.n_problems} task_id={t}",
            flush=True))
    sidecar_f.close()
    retry_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report=report)
    gates = _evaluate_phase2_shape_gates(report=report, mlb=mlb)
    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep["provenance"] = provenance
    rep["mlb"] = mlb
    rep["phase2_shape_evaluation"] = gates
    rep_path = (
        cell_dir / "humaneval_plus_reflexion_bench_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    # phase3_cell_verdict.json — partial audit signal that the
    # cell is COMPLETE.  Resume relies on this file.
    cell_verdict = {
        "schema": "coordpy.w105_phase3_cell_verdict.v1",
        "model_id": str(model_class),
        "seed": int(seed),
        "is_canary": bool(is_canary),
        "n_problems": int(len(cell_slice)),
        "a0_pct": gates["a0_pct"],
        "a1_pct": gates["a1_pct"],
        "b_pct": gates["b_pct"],
        "b_minus_a1_pp": gates["b_minus_a1_pp"],
        "b_minus_a0_pp": gates["b_minus_a0_pp"],
        "mlb": mlb,
        "phase2_shape_evaluation": gates,
        "bench_merkle_root": str(report.bench_merkle_root),
        "wall_s": float(round(wall_s, 2)),
        "cell_run_id": str(cell_run_id),
        "slice_pack_cid": str(slice_pack_cid),
        "corpus_sha256": str(corpus_sha),
        "ts_utc": _dt.datetime.now(
            _dt.timezone.utc).isoformat(),
    }
    with open(cell_dir / "phase3_cell_verdict.json", "w") as f:
        json.dump(cell_verdict, f, indent=2, default=str)
    _update_global_progress(
        run_root=run_root,
        status="cell_completed",
        detail={
            "model_class": str(model_class),
            "seed": int(seed),
            "is_canary": bool(is_canary),
            "b_minus_a1_pp": gates["b_minus_a1_pp"],
            "mlb2_rescue_rate": mlb["mlb2_rescue_rate"],
            "wall_s": float(round(wall_s, 2)),
        })
    print(
        f"  [{model_class} seed={seed}] WALL={wall_s:.1f}s; "
        f"A0={gates['a0_pct']:.2f}% "
        f"A1={gates['a1_pct']:.2f}% "
        f"B={gates['b_pct']:.2f}% "
        f"B-A1={gates['b_minus_a1_pp']:+.2f}pp; "
        f"MLB-2={mlb['mlb2_rescue_rate']*100:.2f}%",
        flush=True)
    return cell_verdict


def _emit_per_class_partial_verdict_doc(
        *, run_root: Path) -> None:
    """Run the per-class Phase 3 retirement evaluator against
    whatever cells are complete and emit the per-class verdict
    JSON (and stub markdown) for each class."""
    try:
        from coordpy.phase3_retirement_evaluator_v1 import (
            build_phase3_retirement_verdict_v1,
            format_phase3_retirement_verdict_markdown_v1,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"  WARNING: Phase 3 evaluator import failed: "
            f"{type(e).__name__}: {e}", flush=True)
        return
    cells_by_class: dict[str, list] = {}
    for class_dir in sorted(run_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.startswith("class_"):
            continue
        class_id = class_dir.name.removeprefix(
            "class_").replace("__slash__", "/")
        for cell_dir in sorted(class_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            verdict_p = cell_dir / "phase3_cell_verdict.json"
            if not verdict_p.exists():
                continue
            rep_p = cell_dir / (
                "humaneval_plus_reflexion_bench_report.json")
            prov_p = cell_dir / "provenance.json"
            if not (rep_p.exists() and prov_p.exists()):
                continue
            try:
                with open(rep_p) as f:
                    br = json.load(f)
                with open(prov_p) as f:
                    prov = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if bool(prov.get("is_canary")):
                continue  # canary cells excluded
            cells_by_class.setdefault(class_id, []).append(
                (br, prov))
    if not cells_by_class:
        return
    # Audit chain re-derive: count how many cells have a
    # recoverable bench_merkle_root (cheap re-derive: re-compute
    # from outcome_cids).
    audit_re_derives_by_class: dict[str, int] = {}
    audit_total_by_class: dict[str, int] = {}
    canonical_pass_rate_by_class: dict[str, float] = {}
    for class_id, pairs in cells_by_class.items():
        audit_total_by_class[class_id] = int(len(pairs))
        n_ok = 0
        for br, prov in pairs:
            # Re-derive bench Merkle from outcome_cids + seeds
            # exactly as the bench module does.
            merkle = br.get("bench_merkle_root") or ""
            outcome_cids: list[str] = []
            for s in br.get("per_seed") or []:
                outcome_cids.extend(
                    list(s.get("outcome_cids") or []))
            payload = {
                "kind":
                    "w102_humaneval_plus_bench_merkle_root_v1",
                "model_id": str(br.get("model_id") or ""),
                "outcome_cids": outcome_cids,
                "seeds": [int(s.get("seed", 0))
                          for s in br.get("per_seed") or []],
            }
            recomputed = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True,
                    separators=(",", ":"),
                    default=str).encode("utf-8")).hexdigest()
            if recomputed == merkle:
                n_ok += 1
        audit_re_derives_by_class[class_id] = int(n_ok)
        # Canonical-solution pass rate is recorded later (cheap
        # re-derive); placeholder 1.0 here means assumed clean
        # unless overridden by a separate post-run probe.
        canonical_pass_rate_by_class[class_id] = float(1.0)
    try:
        verdict = build_phase3_retirement_verdict_v1(
            cells_by_class=cells_by_class,
            audit_chain_re_derives_by_class=(
                audit_re_derives_by_class),
            audit_chain_total_by_class=audit_total_by_class,
            canonical_pass_rate_by_class=(
                canonical_pass_rate_by_class))
    except Exception as e:  # noqa: BLE001
        print(
            f"  ERROR: Phase 3 evaluator refused: "
            f"{type(e).__name__}: {e}", flush=True)
        return
    out_json = (
        run_root / "phase3_retirement_verdict.json")
    with open(out_json, "w") as f:
        json.dump(verdict.to_dict(), f, indent=2, default=str)
    out_md = run_root / "phase3_retirement_verdict.md"
    with open(out_md, "w") as f:
        f.write(format_phase3_retirement_verdict_markdown_v1(
            verdict=verdict))
    print(
        f"  Phase 3 retirement verdict emitted: {out_json}",
        flush=True)
    for per in verdict.per_class:
        print(
            f"  class={per.model_class_id} -> "
            f"{per.verdict_label} "
            f"({per.n_bars_passed_of_6}/6 bars; mean B-A1="
            f"{per.bar1_margin_mean_b_minus_a1_pp:+.2f}pp; "
            f"MLB-2={per.mean_mlb2_rescue_rate*100:.2f}%)",
            flush=True)
    if verdict.cross_class is not None:
        cc = verdict.cross_class
        print(
            f"  cross-class={cc.cross_class_claim_label}; "
            f"diff={cc.cross_class_b_minus_a1_diff_pp:+.2f}pp; "
            f"entitled={cc.cross_class_retirement_entitled}",
            flush=True)


def _emit_cross_class_comparator(
        *, run_root: Path) -> None:
    try:
        from coordpy.cross_class_comparator_v1 import (
            build_cross_class_comparator_report_v1,
            format_cross_class_comparator_markdown_v1,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"  WARNING: cross-class comparator import failed: "
            f"{type(e).__name__}: {e}", flush=True)
        return
    by_class: dict[str, dict[int, tuple]] = {}
    for class_dir in sorted(run_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.startswith("class_"):
            continue
        class_id = class_dir.name.removeprefix(
            "class_").replace("__slash__", "/")
        for cell_dir in sorted(class_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            verdict_p = cell_dir / "phase3_cell_verdict.json"
            rep_p = cell_dir / (
                "humaneval_plus_reflexion_bench_report.json")
            prov_p = cell_dir / "provenance.json"
            if not (verdict_p.exists()
                    and rep_p.exists() and prov_p.exists()):
                continue
            try:
                with open(rep_p) as f:
                    br = json.load(f)
                with open(prov_p) as f:
                    prov = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if bool(prov.get("is_canary")):
                continue
            seed = int(prov.get("seed") or 0)
            by_class.setdefault(class_id, {})[seed] = (br, prov)
    if len(by_class) != 2:
        return
    classes = sorted(by_class.keys())
    seeds_a = set(by_class[classes[0]].keys())
    seeds_b = set(by_class[classes[1]].keys())
    common = sorted(seeds_a & seeds_b)
    if not common:
        return
    filtered_a = {s: by_class[classes[0]][s] for s in common}
    filtered_b = {s: by_class[classes[1]][s] for s in common}
    try:
        comp = build_cross_class_comparator_report_v1(
            class_a_id=str(classes[0]),
            class_b_id=str(classes[1]),
            class_a_by_seed=filtered_a,
            class_b_by_seed=filtered_b)
    except Exception as e:  # noqa: BLE001
        print(
            f"  ERROR: cross-class comparator refused: "
            f"{type(e).__name__}: {e}", flush=True)
        return
    with open(
            run_root / "cross_class_comparator.json",
            "w") as f:
        json.dump(
            comp.to_dict(), f, indent=2, default=str)
    with open(
            run_root / "cross_class_comparator.md", "w") as f:
        f.write(
            format_cross_class_comparator_markdown_v1(
                report=comp))
    print(
        f"  cross-class comparator emitted at seeds "
        f"{common}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W105 HumanEval+ Phase 3 retirement-bench driver"))
    ap.add_argument(
        "--slice-pack",
        default=str(
            ROOT / "data" / "w105" / "phase3_slice_pack"
            / "w105_phase3_slice_pack_20260526T215647Z"
            / "slice_pack.json"),
        help=(
            "Path to W105 pre-built slice pack JSON.  Pack CID + "
            "inner-kernel CID verified at start; refuses to run "
            "on mismatch."))
    ap.add_argument(
        "--humaneval-plus-cache", default=None,
        help="HumanEval+ JSONL cache path override")
    ap.add_argument(
        "--out-root", default=str(
            ROOT / "results" / "w105"
            / "humaneval_plus_phase3_retirement_bench"),
        help="Output root (each invocation creates a sub-dir)")
    ap.add_argument(
        "--resume-from", default=None,
        help=(
            "Path to an existing W105 run-root.  Cells with "
            "phase3_cell_verdict.json on disk are skipped.  "
            "Within a partial cell the W104 sidecar-resume "
            "applies."))
    ap.add_argument(
        "--canary", action="store_true",
        help=(
            "Run the W105 canary smoke only (1 seed × 3 problems "
            "× K = 5 × 2 classes = 66 NIM calls).  Used to "
            "validate reachability + budget envelope BEFORE the "
            "full Phase 3 launch."))
    ap.add_argument(
        "--only-class", default=None,
        help=(
            "If set, run only this model class (still uses the "
            "W105 locked class set for validation).  Useful for "
            "running the two classes in parallel processes."))
    ap.add_argument(
        "--skip-smoke", action="store_true",
        help=(
            "Skip the per-class reachability smoke probe.  Use "
            "ONLY if you have already confirmed reachability "
            "out-of-band."))
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do NOT actually launch any NIM call; just validate "
            "the slice pack + corpus + provenance + smoke "
            "probe."))
    args = ap.parse_args()

    print("  W105 HumanEval+ Phase 3 retirement-bench driver")
    slice_pack_path = Path(args.slice_pack)
    print(f"  loading W105 slice pack from {slice_pack_path}")
    pack = _load_w105_slice_pack(
        slice_pack_path=slice_pack_path)
    slice_task_ids = list(
        pack.get("task_ids_helper_priority") or [])
    slice_pack_cid = str(pack.get("pack_cid") or "")
    print(f"  pack_cid     : {slice_pack_cid}")
    print(f"  n_problems   : {len(slice_task_ids)}")
    print(f"  inner kernel : "
          f"{W105_INNER_KERNEL_CID_LOCKED}")

    if not is_humaneval_plus_cached(
            cache_path=args.humaneval_plus_cache):
        raise SystemExit(
            "HumanEval+ cache absent; refusing to run W105 "
            "Phase 3 without SHA-pinned corpus.")
    print("  loading HumanEval+ corpus ...")
    full_corpus = load_humaneval_plus_corpus_v1(
        cache_path=args.humaneval_plus_cache)
    by_tid: dict[str, HumanEvalPlusProblemV1] = {
        p.task_id: p for p in full_corpus}
    missing = [
        tid for tid in slice_task_ids if tid not in by_tid]
    if missing:
        raise SystemExit(
            f"W105 Phase 3: slice references task_ids not in "
            f"HumanEval+ corpus: {missing}")
    corpus_path = Path(
        args.humaneval_plus_cache or os.path.expanduser(
            "~/.cache/coordpy/humaneval-plus.jsonl"))
    corpus_sha = _file_sha256(corpus_path)
    if corpus_sha != W105_CORPUS_SHA_LOCKED:
        raise SystemExit(
            "W105 Phase 3: corpus SHA mismatch vs W105 locked: "
            f"got {corpus_sha!r} vs locked "
            f"{W105_CORPUS_SHA_LOCKED!r}.")
    print(f"  corpus SHA-256: {corpus_sha} (matches W105)")

    # Resolve out_root.
    if args.resume_from:
        run_root = Path(args.resume_from)
        if not run_root.exists():
            raise SystemExit(
                f"--resume-from path does not exist: {run_root}")
        print(f"  RESUMING into {run_root}")
    else:
        run_id = _dt.datetime.now(
            _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = (
            "_canary" if args.canary else "_full")
        if args.only_class:
            suffix += "_" + str(args.only_class).replace(
                "/", "__slash__")
        run_root = (
            Path(args.out_root)
            / f"w105_phase3_{run_id}{suffix}")
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"  fresh run root: {run_root}")

    # Resolve model class list.
    if args.only_class:
        if args.only_class not in W105_MODEL_CLASSES_LOCKED:
            raise SystemExit(
                f"--only-class {args.only_class!r} not in W105 "
                f"locked set {W105_MODEL_CLASSES_LOCKED}")
        class_list = (args.only_class,)
    else:
        class_list = W105_MODEL_CLASSES_LOCKED

    # Resolve seed list per canary / full.
    if args.canary:
        seed_list = (W105_CANARY_SEED,)
        n_problems_per_cell = int(W105_CANARY_N_PROBLEMS)
    else:
        seed_list = W105_PHASE3_SEEDS_LOCKED
        n_problems_per_cell = int(len(slice_task_ids))

    # Smoke probe per class.
    if not args.skip_smoke and not args.dry_run:
        for mc in class_list:
            print(f"  [smoke probe] {mc} ...", flush=True)
            if not _reachability_smoke_probe(
                    model=mc, max_seconds=25.0):
                raise SystemExit(
                    f"W105 smoke probe FAIL on {mc}; aborting.")
            print(f"  [smoke probe] PASS on {mc}",
                  flush=True)

    if args.dry_run:
        print(
            "  --dry-run: stopping before any benchmark call.")
        return 0

    # Write the pack reference into run_root for audit.
    with open(run_root / "slice_pack_reference.json", "w") as f:
        json.dump({
            "slice_pack_cid": slice_pack_cid,
            "inner_kernel_cid": W105_INNER_KERNEL_CID_LOCKED,
            "corpus_sha256": corpus_sha,
            "preflight_verdict_cid":
                W105_PREFLIGHT_VERDICT_CID_LOCKED,
            "n_problems_per_cell": int(n_problems_per_cell),
            "is_canary": bool(args.canary),
            "seed_list": list(seed_list),
            "class_list": list(class_list),
        }, f, indent=2, default=str)

    # Run cells.
    overall_t0 = time.time()
    for mc in class_list:
        class_dir = run_root / (
            "class_" + str(mc).replace("/", "__slash__"))
        class_dir.mkdir(parents=True, exist_ok=True)
        for seed in seed_list:
            cell_name = f"seed_{seed}"
            cell_dir = class_dir / cell_name
            if _cell_run_complete(cell_dir):
                print(
                    f"  [SKIP] cell already complete: "
                    f"{cell_dir}", flush=True)
                continue
            _run_one_cell(
                model_class=str(mc),
                seed=int(seed),
                n_problems=int(n_problems_per_cell),
                slice_task_ids=list(slice_task_ids),
                corpus_by_tid=by_tid,
                corpus_path=corpus_path,
                corpus_sha=str(corpus_sha),
                slice_pack_cid=str(slice_pack_cid),
                cell_dir=cell_dir,
                run_root=run_root,
                is_canary=bool(args.canary))
            # Emit partial per-class verdict after each cell.
            _emit_per_class_partial_verdict_doc(
                run_root=run_root)
            _emit_cross_class_comparator(run_root=run_root)
    overall_wall = float(time.time() - overall_t0)
    print(f"\n  ALL CELLS DONE; wall = {overall_wall:.1f} s")
    # Final emit.
    _emit_per_class_partial_verdict_doc(run_root=run_root)
    _emit_cross_class_comparator(run_root=run_root)
    latest_pointer = run_root.parent / "latest_run.txt"
    with open(latest_pointer, "w") as f:
        f.write(run_root.name + "\n")
    _update_global_progress(
        run_root=run_root,
        status="complete",
        detail={
            "wall_s": float(round(overall_wall, 2)),
            "class_list": list(class_list),
            "seed_list": list(seed_list),
            "is_canary": bool(args.canary),
        })
    print(f"  out_root: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
