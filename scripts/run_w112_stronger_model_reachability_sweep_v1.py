#!/usr/bin/env python3
"""W112-α — stronger-model resistant-code reachability/availability sweep.

Executes the ``docs/RUNBOOK_W112.md`` § 1α target-selection rule (locked BEFORE
this probe). Honest reachability pass over stronger-than-70B code-capable NIM
targets — does NOT assume 405B exists just because it is the standing extension.

Each probe is a sub-second ``max_tokens=4`` chat-completion (the W105 probe body)
=> effectively $0. We also GET the live ``/v1/models`` catalogue so the sweep is
grounded in what NIM actually serves at run time, not a hardcoded guess.

Eligibility (RUNBOOK § 1α): a target is ELIGIBLE iff
  C-S1 reachable (HTTP 200)
  C-S2 strictly stronger than meta/llama-3.3-70b-instruct on code (scale or pass@1)
  C-S3 same-budget comparable: PLAIN single-completion path; EXCLUDES reasoning
       models that emit long/hidden CoT by default (budget non-comparable)
  C-S4 honest plain code-gen path (no tool/agent scaffold)
Ranking: same-FAMILY larger Llama instruct > cross-arch strictly-larger
NON-reasoning instruct. Pick the strongest eligible target; else Lane α CLOSED.

The C-S2/C-S3 metadata below encode KNOWN model facts (scale, reasoning-by-
default); C-S1 is measured live. No NIM call earns an expensive pilot here — this
sweep only selects the target and records the gate. The earn gate is § 1α-earn.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w112_stronger_model_reachability_sweep_v1.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NIM_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
BASELINE_70B = "meta/llama-3.3-70b-instruct"

# Curated candidate set (RUNBOOK § 1α probe order: 405B first). Metadata encode
# KNOWN facts; reachability (C-S1) is measured live. `strictly_stronger` and
# `reasoning_by_default` are the C-S2 / C-S3 judgments with rationale.
# rank_tier encodes the LOCKED § 1α ranking rule (lower = preferred):
#   1 = Llama-FAMILY larger instruct, non-reasoning (cleanest cross-generation-UP
#       from the Llama-3.3-70B baseline; matches the W104 cross-generation precedent)
#   2 = cross-architecture strictly-larger NON-reasoning instruct (eligible, but
#       cross-vendor confound) — code-specialized first within the tier
#   3 = reasoning-by-default => C-S3 EXCLUDE (budget non-comparable)
#   9 = NOT strictly stronger (context only)
CANDIDATES: tuple[dict, ...] = (
    {"model": "meta/llama-3.1-405b-instruct", "family": "llama",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 1, "note": "standing extension; same-family larger (405B>70B)"},
    {"model": "meta/llama-4-maverick-17b-128e-instruct", "family": "llama4",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 1, "note": "Llama-family cross-generation-UP (3.3->4); frontier "
     "MoE 400B total/17B active; >3.3-70B on code; cleanest lineage probe"},
    # --- cross-architecture strictly-larger non-reasoning (tier 2) ---
    {"model": "qwen/qwen3-coder-480b-a35b-instruct", "family": "qwen",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "STRONGEST reachable CODE-specialized frontier "
     "(480B/35B active); cross-vendor confound => W113 escalation/cross-check"},
    {"model": "mistralai/mistral-large-3-675b-instruct-2512", "family": "mistral",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "frontier 675B instruct non-reasoning; cross-vendor"},
    {"model": "nvidia/nemotron-4-340b-instruct", "family": "nemotron",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "340B instruct non-reasoning; cross-vendor"},
    {"model": "deepseek-ai/deepseek-v4-pro", "family": "deepseek",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "DeepSeek V4-pro (V-series chat, non-reasoning); strong code"},
    {"model": "mistralai/mistral-small-4-119b-2603", "family": "mistral",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "119B non-reasoning instruct; > 70B scale"},
    {"model": "mistralai/mistral-large-2-instruct", "family": "mistral",
     "strictly_stronger": True, "reasoning_by_default": False,
     "rank_tier": 2, "note": "123B dense non-reasoning instruct; code-capable"},
    # --- reasoning-by-default => C-S3 EXCLUDE (tier 3) ---
    {"model": "qwen/qwen3.5-397b-a17b", "family": "qwen",
     "strictly_stronger": True, "reasoning_by_default": True,
     "rank_tier": 3, "note": "397B frontier BUT qwen3.5 thinking-mode => C-S3 risk"},
    {"model": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "family": "nemotron",
     "strictly_stronger": True, "reasoning_by_default": True,
     "rank_tier": 3, "note": "253B but reasoning-by-default => C-S3 EXCLUDE"},
    {"model": "deepseek-ai/deepseek-r1", "family": "deepseek",
     "strictly_stronger": True, "reasoning_by_default": True,
     "rank_tier": 3, "note": "671B MoE strong code BUT reasoning => C-S3 EXCLUDE"},
    # --- NOT strictly stronger (tier 9; context only) ---
    {"model": "nvidia/llama-3.1-nemotron-70b-instruct", "family": "nemotron",
     "strictly_stronger": False, "reasoning_by_default": False,
     "rank_tier": 9, "note": "same 70B scale => NOT strictly stronger (context)"},
    {"model": "meta/codellama-70b", "family": "llama",
     "strictly_stronger": False, "reasoning_by_default": False,
     "rank_tier": 9, "note": "code-specialized but 70B => NOT strictly stronger"},
)


def _probe_chat(*, model: str, api_key: str, max_seconds: float) -> dict:
    body = {"model": str(model),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4, "temperature": 0.0, "stream": False}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        NIM_CHAT_URL, data=data, headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=float(max_seconds)) as r:
            raw = r.read()
        return {"status": "reachable", "http_status": 200,
                "wall_ms": int((time.time() - t0) * 1000),
                "response_len": int(len(raw))}
    except urllib.error.HTTPError as e:
        return {"status": "http_error", "http_status": int(e.code),
                "reason": str(e.reason)[:200],
                "wall_ms": int((time.time() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"status": "exception", "exc_type": type(e).__name__,
                "exc_msg": str(e)[:200],
                "wall_ms": int((time.time() - t0) * 1000)}


def _get_models(*, api_key: str, max_seconds: float) -> dict:
    req = urllib.request.Request(
        NIM_MODELS_URL, headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=float(max_seconds)) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
        ids = sorted(str(m.get("id")) for m in (payload.get("data") or []))
        return {"ok": True, "n_models": len(ids), "model_ids": ids}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "exc_type": type(e).__name__, "exc_msg": str(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description="W112 stronger-model reachability sweep")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--out-root",
                    default=str(ROOT / "results" / "w112" / "stronger_model_reachability"))
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY not set; W112 reachability sweep needs NIM.")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"sweep_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  GET /v1/models catalogue ...")
    catalogue = _get_models(api_key=api_key, max_seconds=args.max_seconds)
    if catalogue.get("ok"):
        print(f"    catalogue: {catalogue['n_models']} models served")
    else:
        print(f"    catalogue GET failed: {catalogue}")
    served = set(catalogue.get("model_ids") or [])

    results = []
    for cand in CANDIDATES:
        model = cand["model"]
        print(f"  probe {model} ...", flush=True)
        probe = _probe_chat(model=model, api_key=api_key,
                            max_seconds=args.max_seconds)
        reachable = (probe.get("status") == "reachable"
                     and int(probe.get("http_status") or 0) == 200)
        # Eligibility per RUNBOOK § 1α.
        c_s1 = bool(reachable)
        c_s2 = bool(cand["strictly_stronger"])
        c_s3 = not bool(cand["reasoning_by_default"])
        c_s4 = True  # plain code-gen path assumed for all instruct chat models
        eligible = bool(c_s1 and c_s2 and c_s3 and c_s4)
        rec = {**cand, "probe": probe, "reachable": reachable,
               "in_catalogue": (model in served) if served else None,
               "C_S1_reachable": c_s1, "C_S2_strictly_stronger": c_s2,
               "C_S3_same_budget_comparable_nonreasoning": c_s3,
               "C_S4_honest_code_path": c_s4, "eligible": eligible}
        results.append(rec)
        verdict = ("ELIGIBLE" if eligible else
                   ("reachable-but-ineligible" if reachable else "unreachable"))
        print(f"    -> {probe.get('status')} "
              f"(http={probe.get('http_status')}, {probe.get('wall_ms')}ms) "
              f"=> {verdict}", flush=True)

    eligible = [r for r in results if r["eligible"]]
    eligible.sort(key=lambda r: (r["rank_tier"], r["model"]))
    selected = eligible[0] if eligible else None

    decision = {
        "schema": "coordpy.w112_stronger_model_reachability_sweep.v1",
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "baseline_70b": BASELINE_70B,
        "catalogue_ok": bool(catalogue.get("ok")),
        "catalogue_n_models": catalogue.get("n_models"),
        "n_candidates_probed": len(results),
        "n_reachable": sum(1 for r in results if r["reachable"]),
        "n_eligible": len(eligible),
        "selected_target": (selected["model"] if selected else None),
        "lane_alpha_gate": ("OPEN" if selected else "CLOSED"),
        "lane_alpha_rationale": (
            f"Eligible stronger target selected: {selected['model']} "
            f"({selected['note']}); proceed to § 1α-earn canary."
            if selected else
            "NO stronger-than-70B code-capable, same-budget-comparable "
            "(non-reasoning) target is reachable on NIM; Lane α CLOSED. Combined "
            "with the Lane β $0 kill, the bounded-claim fallback (§ 6) is the "
            "honest ceiling."),
        "results": results,
        "catalogue_model_ids": catalogue.get("model_ids"),
    }
    decision["decision_cid"] = hashlib.sha256(
        json.dumps({k: v for k, v in decision.items() if k != "ts_utc"},
                   sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()
    (out_dir / "sweep_decision.json").write_text(
        json.dumps(decision, indent=2, default=str))
    (Path(args.out_root) / "latest_run.txt").write_text(out_dir.name + "\n")

    print("\n=== W112 STRONGER-MODEL REACHABILITY SWEEP ===")
    print(f"  catalogue: {decision['catalogue_n_models']} models; "
          f"probed {decision['n_candidates_probed']}; "
          f"reachable {decision['n_reachable']}; eligible {decision['n_eligible']}")
    print(f"  Lane α gate: {decision['lane_alpha_gate']}")
    print(f"  selected target: {decision['selected_target']}")
    print(f"  rationale: {decision['lane_alpha_rationale']}")
    print(f"  decision_cid: {decision['decision_cid']}")
    print(f"  out: {out_dir / 'sweep_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
