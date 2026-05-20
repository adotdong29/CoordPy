#!/usr/bin/env python3
"""W86 MoE substrate closure driver (#31).

Runs ``run_moe_substrate_closure_bench_v1`` against a real
open-weight MoE model on a CUDA host. Writes the
content-addressed bench report to ``--out-dir``.

Default model: ``allenai/OLMoE-1B-7B-0924-Instruct`` (7 B
params total, 1.3 B active, 8 experts top-2) — fits easily on
A100-40GB at bf16.

Usage::

    python scripts/run_w86_moe_substrate_closure.py \\
        --out-dir results/w86/moe/<TS> \\
        --model-name allenai/OLMoE-1B-7B-0924-Instruct \\
        --device cuda:0 \\
        --precision-tier tier_bf16

The script is designed to be the body of one Colab cell; the
Colab notebook calls it via ``!python …``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _env_probe() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": str(sys.version),
        "platform": str(platform.platform()),
    }
    try:
        import torch  # type: ignore
        env["torch_version"] = str(torch.__version__)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["cuda_device_name"] = str(p.name)
            env["cuda_device_total_mem_gb"] = float(round(
                p.total_memory / (1024 ** 3), 3))
            env["cuda_device_capability"] = (
                f"{p.major}.{p.minor}")
    except Exception as exc:  # noqa: BLE001
        env["torch_import_error"] = (
            f"{type(exc).__name__}: {exc}")
    try:
        import transformers  # type: ignore
        env["transformers_version"] = str(
            transformers.__version__)
    except Exception as exc:  # noqa: BLE001
        env["transformers_import_error"] = (
            f"{type(exc).__name__}: {exc}")
    return env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=False, default=None)
    p.add_argument(
        "--model-name",
        default="allenai/OLMoE-1B-7B-0924-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision-tier", default="tier_bf16")
    p.add_argument(
        "--prompt",
        default=(
            "Context Zero is the research programme that "
            "ships a real substrate contract for "
            "mixture-of-experts transformer routing."))
    p.add_argument(
        "--prompt-max-len", type=int, default=24)
    p.add_argument(
        "--n-continuation-tokens", type=int, default=4)
    p.add_argument("--inject-layer", type=int, default=4)
    p.add_argument(
        "--inject-magnitude", type=float, default=1.0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "results") / "w86" / "moe" / dt.datetime.now(
            tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _env_probe()
    overall: dict[str, Any] = {
        "schema": "coordpy.w86_moe_substrate_closure_v1",
        "run_started_utc": dt.datetime.now(
            tz=dt.timezone.utc).isoformat(),
        "model_name": str(args.model_name),
        "device": str(args.device),
        "precision_tier": str(args.precision_tier),
        "prompt": str(args.prompt),
        "prompt_max_len": int(args.prompt_max_len),
        "n_continuation_tokens": int(
            args.n_continuation_tokens),
        "inject_layer": int(args.inject_layer),
        "inject_magnitude": float(args.inject_magnitude),
        "env": env,
        "out_dir": str(out_dir),
    }
    print(f"[w86-moe] out_dir = {out_dir}", flush=True)
    print(
        "[w86-moe] env = "
        f"{json.dumps(env, indent=0)}",
        flush=True)

    # Probe first.
    try:
        from coordpy.moe_runtime_substrate_v1 import (
            probe_moe_capability_v1,
            run_moe_substrate_closure_bench_v1,
        )
    except Exception as exc:  # noqa: BLE001
        overall["import_error"] = (
            f"{type(exc).__name__}: {exc}")
        out_dir.joinpath(
            "moe_substrate_closure_report.json").write_text(
            json.dumps(overall, indent=2),
            encoding="utf-8")
        print(
            f"[w86-moe] FAILED to import: {exc}",
            file=sys.stderr, flush=True)
        return 1

    print("[w86-moe] probing capability...", flush=True)
    probe = probe_moe_capability_v1(
        model_name=str(args.model_name))
    overall["capability_probe"] = probe.to_dict()
    print(
        "[w86-moe] probe.model_is_moe="
        f"{probe.model_is_moe} n_experts={probe.n_experts} "
        f"top_k={probe.top_k}", flush=True)

    if not probe.transformers_available:
        overall["closure_31_error"] = (
            "transformers/torch not available")
        out_dir.joinpath(
            "moe_substrate_closure_report.json").write_text(
            json.dumps(overall, indent=2),
            encoding="utf-8")
        return 1
    if not probe.model_is_moe:
        overall["closure_31_error"] = (
            f"model {args.model_name} is not an MoE per "
            "config; AutoConfig did not report "
            "num_experts/top_k")
        out_dir.joinpath(
            "moe_substrate_closure_report.json").write_text(
            json.dumps(overall, indent=2),
            encoding="utf-8")
        return 1

    print(
        f"[w86-moe] loading {args.model_name} on "
        f"{args.device} at {args.precision_tier}...",
        flush=True)
    try:
        rep = run_moe_substrate_closure_bench_v1(
            model_name=str(args.model_name),
            device=str(args.device),
            precision_tier=str(args.precision_tier),
            prompt=str(args.prompt),
            prompt_max_len=int(args.prompt_max_len),
            n_continuation_tokens=int(
                args.n_continuation_tokens),
            inject_layer=int(args.inject_layer),
            inject_magnitude=float(args.inject_magnitude),
        )
        overall["closure_31"] = rep.to_dict()
        overall["bench_cid"] = str(rep.cid())
    except Exception as exc:  # noqa: BLE001
        import traceback
        overall["closure_31_error"] = (
            f"{type(exc).__name__}: {str(exc)[:300]}")
        overall["closure_31_traceback"] = traceback.format_exc()
        print(
            f"[w86-moe] FAILED in bench: {exc}",
            file=sys.stderr, flush=True)

    overall["run_finished_utc"] = dt.datetime.now(
        tz=dt.timezone.utc).isoformat()
    out_path = out_dir / "moe_substrate_closure_report.json"
    out_path.write_text(
        json.dumps(overall, indent=2), encoding="utf-8")
    print(
        f"[w86-moe] DONE — report written to {out_path}",
        flush=True)
    rd = overall.get("closure_31", {})
    if rd:
        for k in (
                "forward_routing_captured",
                "replay_with_routing_matches_forward_floor",
                "moe_routing_is_load_bearing",
                "routing_deterministic_across_two_forwards",
                "hidden_state_intercept_on_moe_block_moves_cid",
                "n_moe_layers", "n_experts", "top_k",
                "n_layers_routing_captured",
                "tier_tolerance",
                "max_abs_diff_with_routing_vs_forward_last_logits",
                "max_abs_diff_without_routing_vs_forward_last_logits",
                "max_abs_diff_force_random_vs_forward_last_logits",
                "wall_clock_seconds", "bench_cid"):
            if k in rd:
                print(f"[w86-moe] {k}: {rd[k]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
