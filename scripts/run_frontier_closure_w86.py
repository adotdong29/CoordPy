#!/usr/bin/env python3
"""W86 frontier closure driver.

Runs the end-to-end frontier closure on a real GPU host:

* Loads a 7B+ open-weight LLM (default
  ``meta-llama/Llama-3.1-8B-Instruct``) under the W80
  ``TransformersRuntimeV1`` contract in bf16 on cuda:0.
* Runs the W80 instrumentation conformance suite (≥ 10/12 axes
  expected).
* Runs the W83 ``hidden_state_intercept_bench_v1`` at frontier
  scale.
* Runs the W86 live composed-learned-memory training (#26
  closure).
* Runs the W86 long-context hidden-state intercept bench at
  ≥ 32 k input tokens (#27 last bar).

Writes a single content-addressed
``frontier_closure_report.json`` plus per-step sidecar JSON
files into ``--out-dir`` (default
``results/w86/<UTC ISO>/``). The Vertex AI Colab Enterprise
notebook uploads ``--out-dir`` to GCS so a third party can
re-verify the entire run from disk.

This script is intentionally self-contained — the only repo
dependency is ``coordpy`` (and its existing optional
``transformers``+``torch`` deps).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _save_json(path: Path, obj: Any) -> str:
    payload = _canonical_bytes(obj)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


W86_DEFAULT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"


def _build_prompts(seed: int, n_train: int, n_eval: int) -> tuple[
        list[str], list[str]]:
    """Build a deterministic prompt corpus for live training.

    Held-out disjointness is enforced downstream by the dataset
    capsule, which hashes per-prompt CIDs and refuses to construct
    a dataset where train and eval CIDs overlap.
    """
    import random
    rng = random.Random(int(seed))
    pool: list[str] = []
    n_total = int(n_train) + int(n_eval) + 64
    topics = [
        "the cat sat on the mat and watched the rain through the window for hours",
        "in the year 2026 a small research collective tried to solve context for multi-agent teams",
        "transformer models compose hidden states through residual streams that accumulate over layers",
        "open-weight models like llama and mistral and qwen made frontier research reproducible",
        "the long horizon credit assignment problem requires slot memory and explicit recurrence",
        "content-addressed traces let a third party re-verify every forward without re-running it",
        "the substrate hooks read hidden state at every layer and write back through the residual stream",
        "a needle in a haystack benchmark places a unique fact at a configurable position deep in a long prompt",
        "byte-identical replay from kv cache requires fp32 arithmetic and deterministic kernel scheduling",
        "the composed learned memory module routes writes through a softmax over K slot banks and reads via attention",
    ]
    for i in range(n_total):
        topic = topics[rng.randrange(len(topics))]
        tag = rng.randrange(100_000, 999_999)
        pool.append(f"[seq-{tag}] {topic} (variant {i}).")
    # Shuffle once for fair train/eval split.
    rng.shuffle(pool)
    train = pool[:int(n_train)]
    eval_ = pool[int(n_train):int(n_train) + int(n_eval)]
    return train, eval_


def _detect_torch_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": str(sys.version),
        "platform": str(platform.platform()),
    }
    try:
        import torch  # type: ignore
        env["torch_version"] = str(torch.__version__)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_count"] = int(
                torch.cuda.device_count())
            env["cuda_device_name"] = str(
                torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            env["cuda_device_total_mem_gb"] = float(round(
                props.total_memory / (1024 ** 3), 3))
            env["cuda_device_capability"] = (
                f"{props.major}.{props.minor}")
    except Exception as exc:  # noqa: BLE001
        env["torch_import_error"] = (
            f"{type(exc).__name__}: {exc}")
    try:
        import transformers  # type: ignore
        env["transformers_version"] = str(transformers.__version__)
    except Exception as exc:  # noqa: BLE001
        env["transformers_import_error"] = (
            f"{type(exc).__name__}: {exc}")
    return env


def _run_25_substrate_coupling(
        *, runtime: Any, model_name: str, out_dir: Path,
) -> dict[str, Any]:
    """#25 — frontier-scale substrate coupling.

    Runs the W80 conformance suite + the W83 hidden-state
    intercept bench at frontier scale.
    """
    from coordpy.runtime_instrumentation_v1 import (
        run_instrumentation_conformance_v1,
    )
    from coordpy.hidden_state_intercept_bench_v1 import (
        run_hidden_state_intercept_bench_v1,
    )

    out: dict[str, Any] = {
        "issue": 25,
        "title": (
            "P0 Frontier-Scale Live Substrate Coupling"),
    }

    t0 = time.time()
    # W80 conformance suite.
    conf = run_instrumentation_conformance_v1(
        backend=runtime,
        prompt=(
            "frontier-scale substrate coupling instrumentation "
            "conformance smoke"))
    out["conformance"] = {
        "backend_id": str(conf.backend_id),
        "backend_runtime_id": str(conf.backend_runtime_id),
        "n_pass": int(conf.n_pass),
        "n_skip": int(conf.n_skip),
        "n_fail": int(conf.n_fail),
        "per_axis": [
            {"axis": str(a), "status": str(s)}
            for a, s in conf.per_axis],
    }
    out["wall_seconds_conformance"] = float(
        round(time.time() - t0, 6))

    # W83 hidden-state intercept bench at frontier scale.
    t0 = time.time()
    # Use a substantive prompt for frontier intercept; default
    # ~16 token short prompt is honestly small but the intercept
    # bar is the trace-CID move, which is independent of prompt
    # length.
    hib = run_hidden_state_intercept_bench_v1(
        model_name=str(model_name),
        prompt=(
            "Context Zero is the research programme that "
            "delivered the W80 instrumentation contract and "
            "the W83 composed substrate; this is a frontier "
            "substrate coupling check."),
        prompt_max_len=24,
        n_continuation_tokens=4,
        inject_layer=8,
        inject_magnitude=1.0,
    )
    out["hidden_state_intercept_bench"] = hib.to_dict()
    out["wall_seconds_hidden_state_intercept"] = float(
        round(time.time() - t0, 6))

    # Replay-from-KV at the model's native precision tier.
    t0 = time.time()
    ids = runtime.tokenize(
        "Frontier replay-from-KV byte-identity check at the "
        "W80 contract level on a real 7B+ open-weight model.",
        max_len=24)
    if len(ids) > 4:
        old_ids = ids[:-2]
        new_ids = ids[-2:]
        meas = runtime.measure_replay_vs_recompute(
            old_token_ids=old_ids, new_token_ids=new_ids)
        out["replay_from_kv"] = dict(meas)
    out["wall_seconds_replay"] = float(
        round(time.time() - t0, 6))

    # W83 load-bearing claim reproduction at frontier:
    # the hidden_state_intercept_bench result IS one of the
    # W83 load-bearing claims (the live runtime substrate
    # intercept advance). The bench passing on a 7B+ model is
    # the reproduction of that claim at frontier scale.
    out["w83_load_bearing_claim_reproduced"] = bool(
        out["hidden_state_intercept_bench"][
            "hidden_state_intercept_moves_cid"])
    out["report_cid"] = _sha256_hex({
        "kind": "w86_25_substrate_coupling_report",
        "out": {
            k: v for k, v in out.items()
            if k != "report_cid"},
    })

    _save_json(out_dir / "25_substrate_coupling.json", out)
    return out


def _run_26_live_learned_memory(
        *, runtime: Any, model_name: str, precision_tier: str,
        device: str, out_dir: Path, seed: int,
        n_train: int, n_eval: int,
) -> dict[str, Any]:
    """#26 — live LLM training of composed learned memory."""
    from coordpy.live_composed_memory_training_v1 import (
        train_composed_learned_memory_on_live_hidden_states_v1,
        W86_LIVE_CM_DEFAULT_LAYER,
    )

    prompts_train, prompts_eval = _build_prompts(
        seed=int(seed),
        n_train=int(n_train),
        n_eval=int(n_eval),
    )

    out: dict[str, Any] = {
        "issue": 26,
        "title": (
            "P0 Live LLM Training of Composed Learned Memory"),
    }
    t0 = time.time()
    try:
        report, live_w, syn_w = (
            train_composed_learned_memory_on_live_hidden_states_v1(
                prompts_train=prompts_train,
                prompts_eval=prompts_eval,
                model_name=str(model_name),
                device=str(device),
                precision_tier=str(precision_tier),
                layer_index=int(W86_LIVE_CM_DEFAULT_LAYER),
                seed=int(seed),
                runtime=runtime,
            ))
        out["train_report"] = report.to_dict()
        out["live_training_witness"] = {
            "schema": str(live_w.schema),
            "seed": int(live_w.seed),
            "optimiser_config_cid": str(
                live_w.optimiser_config_cid),
            "loss_curve_cid": str(live_w.loss_curve_cid),
            "fitted_module_cid": str(
                live_w.fitted_module_cid),
            "dataset_cid": str(live_w.dataset_cid),
        }
        out["synthetic_training_witness"] = {
            "schema": str(syn_w.schema),
            "seed": int(syn_w.seed),
            "optimiser_config_cid": str(
                syn_w.optimiser_config_cid),
            "loss_curve_cid": str(syn_w.loss_curve_cid),
            "fitted_module_cid": str(
                syn_w.fitted_module_cid),
            "dataset_cid": str(syn_w.dataset_cid),
        }
        out["live_strictly_beats_synthetic"] = bool(
            report.live_strictly_beats_synthetic_on_holdout)
    except Exception as exc:  # noqa: BLE001
        out["error"] = (
            f"{type(exc).__name__}: {str(exc)[:200]}")
        out["live_strictly_beats_synthetic"] = False
    out["wall_seconds"] = float(round(time.time() - t0, 6))
    out["report_cid"] = _sha256_hex({
        "kind": "w86_26_live_learned_memory_report",
        "out": {
            k: v for k, v in out.items()
            if k != "report_cid"},
    })
    _save_json(out_dir / "26_live_learned_memory.json", out)
    return out


def _run_27_long_context_intercept(
        *, runtime: Any, model_name: str, precision_tier: str,
        device: str, out_dir: Path, seed: int,
        horizon_tokens: int,
) -> dict[str, Any]:
    """#27 — live long-context hidden-state intercept at ≥ 32k."""
    from coordpy.long_context_intercept_bench_v1 import (
        run_long_context_intercept_bench_v1,
    )

    out: dict[str, Any] = {
        "issue": 27,
        "title": (
            "P0 Long-Context Live Evaluation (≥ 32k tokens) — "
            "hidden-state intercept axis"),
    }
    t0 = time.time()
    # IMPORTANT: this bench uses a SECOND runtime instance with
    # skinny_trace=True; the main runtime (with full hidden-state
    # capture) would OOM at 32 k tokens on the L4. The bench
    # creates its own runtime internally when ``runtime=None``.
    horizons = (int(horizon_tokens),)
    try:
        rep = run_long_context_intercept_bench_v1(
            model_name=str(model_name),
            device=str(device),
            precision_tier=str(precision_tier),
            horizons=horizons,
            inject_layer=16,
            inject_magnitude=1.0,
            seed=int(seed),
        )
        out["bench"] = rep.to_dict()
        out["intercept_moves_cid_at_min_32k"] = bool(
            rep.intercept_moves_cid_at_min_32k)
    except Exception as exc:  # noqa: BLE001
        out["error"] = (
            f"{type(exc).__name__}: {str(exc)[:200]}")
        out["intercept_moves_cid_at_min_32k"] = False
    out["wall_seconds"] = float(round(time.time() - t0, 6))
    out["report_cid"] = _sha256_hex({
        "kind": "w86_27_long_context_intercept_report",
        "out": {
            k: v for k, v in out.items()
            if k != "report_cid"},
    })
    _save_json(
        out_dir / "27_long_context_intercept.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir", required=False, default=None,
        help="output directory (default results/w86/<UTC>)")
    p.add_argument(
        "--model-name", default=W86_DEFAULT_MODEL,
        help=("HF model id (default Llama-3.1-8B-Instruct; "
              "gated, requires HF token)"))
    p.add_argument(
        "--device", default="cuda:0",
        help="torch device for the runtime")
    p.add_argument(
        "--precision-tier", default="tier_bf16",
        help=("W84 precision tier: fp32 / bf16 / fp16 / int8 "
              "(default bf16)"))
    p.add_argument(
        "--seed", type=int, default=86_001_001)
    p.add_argument(
        "--n-train-prompts", type=int, default=40)
    p.add_argument(
        "--n-eval-prompts", type=int, default=10)
    p.add_argument(
        "--horizon-tokens", type=int, default=32_768)
    p.add_argument(
        "--skip-25", action="store_true")
    p.add_argument(
        "--skip-26", action="store_true")
    p.add_argument(
        "--skip-27", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "results") / "w86" / dt.datetime.now(
            tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {
        "schema": "coordpy.w86.frontier_closure_report.v1",
        "run_started_utc": (
            dt.datetime.now(tz=dt.timezone.utc).isoformat()),
        "model_name": str(args.model_name),
        "device": str(args.device),
        "precision_tier": str(args.precision_tier),
        "seed": int(args.seed),
        "horizon_tokens": int(args.horizon_tokens),
        "n_train_prompts": int(args.n_train_prompts),
        "n_eval_prompts": int(args.n_eval_prompts),
        "env": _detect_torch_env(),
        "out_dir": str(out_dir),
    }
    print(f"[w86] out_dir = {out_dir}", flush=True)
    print(
        f"[w86] env = {json.dumps(overall['env'], indent=0)}",
        flush=True)

    # Construct ONE main runtime (full-trace mode) for #25 and
    # #26. The #27 bench creates its own skinny-trace runtime
    # internally.
    runtime: Any = None
    try:
        from coordpy.transformers_runtime_v1 import (
            TransformersRuntimeV1,
        )
        print(
            "[w86] loading "
            f"{args.model_name} on {args.device} at "
            f"{args.precision_tier}...", flush=True)
        t_load_0 = time.time()
        runtime = TransformersRuntimeV1(
            model_name=str(args.model_name),
            device=str(args.device),
            precision_tier=str(args.precision_tier),
        )
        overall["model_load_wall_seconds"] = float(
            round(time.time() - t_load_0, 6))
        overall["runtime_backend_id"] = str(
            runtime.backend_id())
        overall["runtime_backend_runtime_id"] = str(
            runtime.backend_runtime_id())
        overall["model_n_layers"] = int(runtime.n_layers)
        overall["model_n_heads"] = int(runtime.n_heads)
        overall["model_hidden_dim"] = int(runtime.hidden_dim)
        print(
            f"[w86] loaded: n_layers={runtime.n_layers}, "
            f"n_heads={runtime.n_heads}, "
            f"hidden_dim={runtime.hidden_dim}", flush=True)
    except Exception as exc:  # noqa: BLE001
        overall["model_load_error"] = (
            f"{type(exc).__name__}: {str(exc)[:300]}")
        print(
            f"[w86] FAILED to load model: {exc}",
            file=sys.stderr, flush=True)
        # Save and exit honestly; never fake a positive result.
        # The report_cid is computed even on failure so the
        # failure manifest is itself auditable.
        overall["run_finished_utc"] = (
            dt.datetime.now(tz=dt.timezone.utc).isoformat())
        overall["report_cid"] = _sha256_hex({
            "kind": "w86_frontier_closure_report_v1",
            "report": {
                k: v for k, v in overall.items()
                if k != "report_cid"},
        })
        _save_json(out_dir / "frontier_closure_report.json",
                   overall)
        return 1

    # #25 substrate coupling.
    if not args.skip_25:
        print("[w86] running #25 substrate coupling...",
              flush=True)
        try:
            overall["closure_25"] = (
                _run_25_substrate_coupling(
                    runtime=runtime,
                    model_name=str(args.model_name),
                    out_dir=out_dir,
                ))
        except Exception as exc:  # noqa: BLE001
            overall["closure_25_error"] = (
                f"{type(exc).__name__}: {str(exc)[:300]}")
    else:
        overall["closure_25"] = {"skipped": True}

    # #26 live learned memory.
    if not args.skip_26:
        print("[w86] running #26 live learned memory...",
              flush=True)
        try:
            overall["closure_26"] = (
                _run_26_live_learned_memory(
                    runtime=runtime,
                    model_name=str(args.model_name),
                    precision_tier=str(args.precision_tier),
                    device=str(args.device),
                    out_dir=out_dir,
                    seed=int(args.seed),
                    n_train=int(args.n_train_prompts),
                    n_eval=int(args.n_eval_prompts),
                ))
        except Exception as exc:  # noqa: BLE001
            overall["closure_26_error"] = (
                f"{type(exc).__name__}: {str(exc)[:300]}")
    else:
        overall["closure_26"] = {"skipped": True}

    # #27 long-context hidden-state intercept.
    if not args.skip_27:
        # Free the main runtime BEFORE the long-context bench so
        # its skinny-trace runtime gets fresh VRAM headroom for
        # the 32 k forward.
        del runtime
        try:
            import torch  # type: ignore
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        print("[w86] running #27 long-context intercept "
              f"at {args.horizon_tokens} tokens...", flush=True)
        try:
            overall["closure_27"] = (
                _run_27_long_context_intercept(
                    runtime=None,
                    model_name=str(args.model_name),
                    precision_tier=str(args.precision_tier),
                    device=str(args.device),
                    out_dir=out_dir,
                    seed=int(args.seed),
                    horizon_tokens=int(args.horizon_tokens),
                ))
        except Exception as exc:  # noqa: BLE001
            overall["closure_27_error"] = (
                f"{type(exc).__name__}: {str(exc)[:300]}")
    else:
        overall["closure_27"] = {"skipped": True}

    overall["run_finished_utc"] = (
        dt.datetime.now(tz=dt.timezone.utc).isoformat())
    overall_cid = _sha256_hex({
        "kind": "w86_frontier_closure_report_v1",
        "report": {
            k: v for k, v in overall.items()
            if k != "report_cid"},
    })
    overall["report_cid"] = str(overall_cid)
    json_path = out_dir / "frontier_closure_report.json"
    _save_json(json_path, overall)
    print(f"[w86] DONE — report CID={overall_cid}",
          flush=True)
    print(f"[w86] report = {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
