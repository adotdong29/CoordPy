"""Cross-model parser-boundary — REAL LLM measurements (SDK v3.6,
W3-C6 promoted from synthetic to real on the smaller model class).

Scope.
------

The synthetic cross-model parser-boundary study
(``parser_boundary_cross_model``) shows that the parser's
PARSE_OUTCOME failure-kind closed vocabulary has resolving power
TVD up to 1.000 across a calibrated synthetic distribution
library. The "Future work" section of
``papers/wevra_capsule_native_runtime.md § 9`` flagged a
*real* cross-LLM study (e.g. ``gemma2:9b`` vs ``qwen2.5:7b``) as
the natural next move. SDK v3.6 ships the integration boundary
(LLM backend abstraction; see ``vision_mvp.wevra.llm_backend``)
that makes this trivial to layer on, and this experiment ships
the first real cross-LLM measurement on the *available* hardware
(two 36 GB Apple Silicon Macs running Ollama under the existing
ASPEN cluster harness).

Honest scope.
-------------

This experiment is REAL but LIMITED. It runs against:

  * ``qwen2.5:14b-32k``  (14.8B parameters, dense, Q4_K_M)
  * ``qwen3.5:35b``      (36B parameters, MoE, Q4_K_M, ``think=False``)

Both run on a *single* Apple Silicon Mac (Mac 1) under Ollama;
neither is itself sharded across the two Macs. The MLX
distributed two-Mac path that SDK v3.6 ships the *integration
boundary* for is the natural next step: a 70B-class model in 4-bit
sharded across two 36 GB Macs (4 GB margin per host, ~40 GB total
weights). Once that endpoint is up, this experiment runs
unchanged with ``llm_backend=MLXDistributedBackend(...)``.

What this experiment measures.
------------------------------

Per (model, parser_mode) cell:

  * the PARSE_OUTCOME failure-kind histogram drawn from
    ``swe_patch_parser.ALL_PARSE_KINDS``;
  * the recovery-label histogram;
  * ok-rate and parser-mode-conditional shift TVD.

Per pair of models, holding parser_mode fixed:

  * the cross-model PARSE_OUTCOME failure-kind TVD.

This isolates the *model-capacity* axis of parser-boundary
behaviour from the *prompt / parser* axis. The honest empirical
question is: **does a stronger model materially reduce
parser-boundary instability (more "ok" outcomes) and shrink the
strict→robust shift?**

Output shape.
-------------

A JSON-serialisable dict with the same shape as
``parser_boundary_cross_model.run_cross_model_study`` so consumers
can switch between synthetic and real with one flag::

    {
      "schema": "wevra.parser_boundary_real.v1",
      "n_instances": <int>,
      "endpoint": <str>,
      "model_tags": [...],
      "parser_modes": ["strict", "robust"],
      "distributions": {
        ("qwen2.5:14b-32k", "strict"): {
          "failure_kind": {"ok": 5, "unclosed_new": 1, ...},
          "recovery": {...},
          "ok_rate": 0.83,
        },
        ...
      },
      "tvd_pairwise": {("qwen2.5:14b-32k", "qwen3.5:35b", "strict"): 0.33},
      "parser_mode_shift": {"qwen2.5:14b-32k": {"strict_to_robust_tvd": 0.10}},
      "wall_seconds": <float>,
      "calls": <int>,
      "claim": <str>,
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from vision_mvp.tasks.swe_patch_parser import (
    ALL_PARSE_KINDS, ALL_RECOVERY_LABELS, parse_patch_block,
)
from vision_mvp.tasks.swe_bench_bridge import (
    load_jsonl_bank, build_synthetic_event_log, parse_unified_diff,
    build_patch_generator_prompt,
)


SCHEMA = "wevra.parser_boundary_real.v1"

# The bundled bank used by ``local_smoke`` / ``bundled_57`` profiles.
DEFAULT_JSONL = "vision_mvp/tasks/data/swe_lite_style_bank.jsonl"

DEFAULT_PARSER_MODES: tuple[str, ...] = ("strict", "robust")

# The (model_tag, ollama_kwargs) pairs we sweep over. Names match
# the Ollama cluster's loaded models on Mac 1 as of 2026-04-26
# (see ``aspen_cluster_config.local.json``).
DEFAULT_MODELS: tuple[dict[str, Any], ...] = (
    {"tag": "qwen2.5:14b-32k", "think": None},
    # qwen3 reasoning models default to ``think=True`` which produces
    # only thinking-trace output up to ``num_predict`` and an empty
    # ``response`` field. Disable for parser-boundary measurement.
    {"tag": "qwen3.5:35b", "think": False},
)

DEFAULT_ENDPOINT = "http://192.168.12.191:11434"


def _gather_tasks(jsonl_path: str, n_instances: "int | None"):
    """Load the bank with synthetic event logs (matches the standard
    sweep harness)."""
    tasks, repo_files = load_jsonl_bank(
        jsonl_path,
        hidden_event_log_factory=(
            lambda t, k=6: build_synthetic_event_log(t, k)),
        limit=n_instances,
    )
    return tasks, repo_files


def _normalise(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def _tvd(p: dict[str, int], q: dict[str, int]) -> float:
    """Total Variation Distance over two unnormalised count dicts.
    Range [0, 1].
    """
    pn = _normalise(p)
    qn = _normalise(q)
    keys = set(pn) | set(qn)
    return 0.5 * sum(abs(pn.get(k, 0.0) - qn.get(k, 0.0))
                     for k in keys)


def _call_one(client, prompt: str, max_tokens: int) -> tuple[str, float]:
    """One LLM call. Returns (response_text, elapsed_seconds)."""
    t0 = time.time()
    text = client.generate(
        prompt, max_tokens=max_tokens, temperature=0.0)
    return text, time.time() - t0


def run_real_cross_model_study(*,
                                 jsonl_path: str = DEFAULT_JSONL,
                                 endpoint: str = DEFAULT_ENDPOINT,
                                 models: "tuple[dict[str, Any], ...]" = DEFAULT_MODELS,
                                 parser_modes: tuple[str, ...] = DEFAULT_PARSER_MODES,
                                 n_instances: "int | None" = 6,
                                 max_tokens: int = 320,
                                 timeout: float = 240.0,
                                 ) -> dict[str, Any]:
    """Sweep (model, parser_mode) over the bundled bank using the
    real Ollama endpoint and return a structured result dict.

    Generation: each (model, instance) pair makes ONE LLM call; the
    response text is then parsed under each parser mode. This
    keeps the LLM-call cost independent of the parser axis (the
    parser is deterministic CPU work given the response bytes) so
    a 2-model × 2-mode sweep over N instances costs 2 × N calls,
    not 4 × N.
    """
    from vision_mvp.core.llm_client import LLMClient
    tasks, _repo_files = _gather_tasks(jsonl_path, n_instances)
    # Per (model, instance_id) → response text. ONE call per cell.
    responses: dict[tuple[str, str], str] = {}
    elapsed_per_call: list[tuple[str, str, float, int, int]] = []
    t_run0 = time.time()
    for spec in models:
        tag = spec["tag"]
        client = LLMClient(
            model=tag, base_url=endpoint, timeout=timeout,
            think=spec.get("think"),
        )
        for task in tasks:
            prompt = build_patch_generator_prompt(
                task=task,
                ctx={"hunk_header": "(none)"},
                buggy_source="(buggy source intentionally omitted "
                              "from research harness)",
                issue_summary=task.problem_statement or "",
                prompt_style="block",
            )
            text, dt = _call_one(client, prompt, max_tokens)
            responses[(tag, task.instance_id)] = text
            elapsed_per_call.append((tag, task.instance_id, round(dt, 3),
                                       len(prompt), len(text)))
    # Now compute parser distributions per (model, parser_mode).
    distributions: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in models:
        tag = spec["tag"]
        for pm in parser_modes:
            counts_kind: dict[str, int] = {k: 0 for k in ALL_PARSE_KINDS}
            counts_recovery: dict[str, int] = {
                (r or "<none>"): 0 for r in ALL_RECOVERY_LABELS}
            n_ok = 0
            n_total = 0
            for task in tasks:
                text = responses.get((tag, task.instance_id), "")
                outcome = parse_patch_block(
                    text, mode=pm,
                    unified_diff_parser=parse_unified_diff)
                n_total += 1
                if outcome.ok:
                    n_ok += 1
                kind = outcome.failure_kind or "<unknown>"
                counts_kind[kind] = counts_kind.get(kind, 0) + 1
                rec = outcome.recovery or "<none>"
                counts_recovery[rec] = counts_recovery.get(rec, 0) + 1
            distributions[(tag, pm)] = {
                "n_total": n_total,
                "n_ok": n_ok,
                "ok_rate": (n_ok / n_total) if n_total else 0.0,
                "failure_kind": counts_kind,
                "recovery": counts_recovery,
            }
    # Pairwise cross-model TVD per parser_mode.
    tvd_pairwise: dict[tuple[str, str, str], float] = {}
    tags = [s["tag"] for s in models]
    for pm in parser_modes:
        for i, ti in enumerate(tags):
            for j in range(i + 1, len(tags)):
                tj = tags[j]
                tvd_pairwise[(ti, tj, pm)] = _tvd(
                    distributions[(ti, pm)]["failure_kind"],
                    distributions[(tj, pm)]["failure_kind"])
    # Parser-mode-conditional shift per model.
    parser_mode_shift: dict[str, dict[str, float]] = {}
    if "strict" in parser_modes and "robust" in parser_modes:
        for tag in tags:
            tvd = _tvd(
                distributions[(tag, "strict")]["failure_kind"],
                distributions[(tag, "robust")]["failure_kind"])
            parser_mode_shift[tag] = {"strict_to_robust_tvd": tvd}
    wall_s = time.time() - t_run0
    max_cross_tvd = (max(tvd_pairwise.values())
                      if tvd_pairwise else 0.0)
    max_mode_shift = (
        max(d["strict_to_robust_tvd"] for d in parser_mode_shift.values())
        if parser_mode_shift else 0.0)
    # Strongest-model ok-rate uplift (informal): the gap on
    # ok-rate (parser_mode=strict) between the largest model and
    # the smallest model.
    if tags and "strict" in parser_modes:
        smallest = distributions[(tags[0], "strict")]
        largest = distributions[(tags[-1], "strict")]
        ok_uplift_strict = largest["ok_rate"] - smallest["ok_rate"]
    else:
        ok_uplift_strict = 0.0
    claim = (
        f"REAL cross-LLM measurement on {len(tasks)} instances: "
        f"max cross-model PARSE_OUTCOME failure-kind TVD = "
        f"{max_cross_tvd:.3f}; max strict→robust parser-mode shift "
        f"= {max_mode_shift:.3f}; strongest-model ok-rate uplift "
        f"on strict parser = {ok_uplift_strict:+.3f}."
    )
    return {
        "schema": SCHEMA,
        "n_instances": len(tasks),
        "endpoint": endpoint,
        "model_tags": tags,
        "parser_modes": list(parser_modes),
        "distributions": distributions,
        "tvd_pairwise": tvd_pairwise,
        "parser_mode_shift": parser_mode_shift,
        "max_cross_tvd": max_cross_tvd,
        "max_parser_mode_shift": max_mode_shift,
        "ok_rate_uplift_strict": ok_uplift_strict,
        "wall_seconds": round(wall_s, 2),
        "n_llm_calls": len(elapsed_per_call),
        "elapsed_per_call": elapsed_per_call,
        "claim": claim,
    }


def as_json_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Project the ``run_real_cross_model_study`` output into a
    JSON-safe dict (stringifying tuple keys)."""
    out = dict(result)
    out["distributions"] = {
        f"{k[0]}|{k[1]}": v
        for k, v in result["distributions"].items()
    }
    out["tvd_pairwise"] = {
        f"{k[0]}|{k[1]}|{k[2]}": v
        for k, v in result["tvd_pairwise"].items()
    }
    return out


def render_summary(result: dict[str, Any]) -> str:
    """Compact human-readable summary (matches the synthetic
    experiment's ``render_summary_table`` layout)."""
    lines: list[str] = []
    lines.append(f"{result['schema']}  endpoint={result['endpoint']}")
    lines.append(f"n_instances={result['n_instances']}  "
                 f"n_llm_calls={result['n_llm_calls']}  "
                 f"wall_seconds={result['wall_seconds']}")
    lines.append("")
    lines.append("Per-(model, parser_mode) failure-kind distribution:")
    fmt = "  {:<22}  {:<8}  ok_rate={:>5.2f}  topkinds={}"
    for (tag, pm), dist in result["distributions"].items():
        topkinds = sorted(
            ((k, v) for k, v in dist["failure_kind"].items() if v),
            key=lambda kv: -kv[1])[:3]
        lines.append(fmt.format(
            tag, pm, dist["ok_rate"],
            ", ".join(f"{k}={v}" for k, v in topkinds)))
    lines.append("")
    lines.append("Cross-model TVD (failure_kind, holding parser_mode fixed):")
    for (a, b, pm), v in result["tvd_pairwise"].items():
        lines.append(f"  {a}  vs  {b}  ({pm:6s})  TVD = {v:.3f}")
    lines.append("")
    lines.append("Parser-mode shift (strict → robust, per model):")
    for tag, d in result["parser_mode_shift"].items():
        lines.append(f"  {tag:<22}  {d['strict_to_robust_tvd']:.3f}")
    lines.append("")
    lines.append("Headline metrics:")
    lines.append(f"  max cross-model TVD          = {result['max_cross_tvd']:.3f}")
    lines.append(f"  max strict→robust shift TVD  = {result['max_parser_mode_shift']:.3f}")
    lines.append(f"  ok-rate uplift (strict)      = "
                 f"{result['ok_rate_uplift_strict']:+.3f}")
    lines.append("")
    lines.append("Claim: " + result["claim"])
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=DEFAULT_JSONL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--n-instances", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--out", default=None,
                   help="Write JSON result to this path; default: stdout")
    args = p.parse_args(argv)
    result = run_real_cross_model_study(
        jsonl_path=args.jsonl,
        endpoint=args.endpoint,
        n_instances=args.n_instances,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(render_summary(result), file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(as_json_dict(result), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
