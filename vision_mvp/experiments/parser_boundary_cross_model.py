"""Cross-model parser-boundary experiment (SDK v3.4 — W3-C4).

Question.
---------

Conjecture **W3-C4** (SDK v3.3) says: the parser's failure-kind
distribution (drawn from ``swe_patch_parser.ALL_PARSE_KINDS``)
should be *stable* across LLM tags — that is, two different LLMs
producing the same logical patches should drive the parser into
proportional regions of its failure taxonomy.

The strict reading of W3-C4 is empirical and dependent on a
particular LLM ensemble that is too expensive to sweep regularly.
This experiment exercises a *necessary, sharper* version that is
both reproducible from a published seed and runnable in CI:

> *Conditional on a small library of synthetic LLM-output
> distributions, each calibrated to drive the parser into a
> distinct failure region, the PARSE_OUTCOME failure-kind
> distribution is **NOT stable across distributions** — it shifts
> by Total Variation Distance > 0.5 between any two non-identical
> distributions, and parser-mode choice (`strict` vs `robust`)
> moves the distribution by a smaller but still measurable
> amount on the same distribution.*

The empirical claim has two parts:

  1. **Cross-distribution variance** is large. Different
     "model" outputs land in different parts of the failure
     taxonomy with TVD ≫ 0; the closed-vocabulary
     ``failure_kind`` is a faithful axis of attribution.
  2. **Parser-mode-conditional shift** is non-zero. Robust mode
     converts a fraction of `unclosed_new` into
     `ok+recovery=closed_at_eos`, fenced-only into
     `ok+recovery=fenced_code_heuristic`, etc. The shift is
     measurable.

Honest scope.
-------------

This is **not** a real cross-model study. The "models" are
canned distributions (see ``synthetic_llm.SYNTHETIC_MODEL_PROFILES``)
calibrated to exercise distinct regions of the parser taxonomy.
The honest claim of this experiment is: *the parser-boundary
attribution layer (PARSE_OUTCOME closed vocabulary + recovery
labels + parser-mode choice) is sharp enough to make
distribution shifts in LLM output measurable as distribution
shifts in PARSE_OUTCOME, on synthetic distributions calibrated
to span the failure taxonomy.* That is W3-39 in empirical form,
not W3-C4 in its full strict form.

A real cross-model study (e.g. ``gemma2:9b`` vs ``qwen2.5:7b``)
is straightforward to layer on top: substitute real
``LLMClient``-driven cells via ``mode="real"`` instead of
``mode="synthetic"``. The harness here is intentionally
zero-cost so it can run in CI on every commit.

Output shape.
-------------

A nested dict::

    {
      "schema": "wevra.parser_boundary.v1",
      "n_instances": <int>,
      "model_tags": [...],
      "parser_modes": [...],
      "distributions": {
        ("synthetic.clean", "strict"): {
          "failure_kind": {"ok": 8, ...}, "ok_rate": 1.0,
          "recovery": {"": 8, ...},
        },
        ...
      },
      "tvd_pairwise": {
        ("synthetic.clean", "synthetic.unclosed", "strict"): 1.0,
        ...
      },
      "parser_mode_shift": {
        "synthetic.unclosed": {"strict_to_robust_tvd": 1.0},
        ...
      },
      "claim": "<one-line empirical claim>",
    }

The dict is JSON-serialisable (after stringifying tuple keys).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from vision_mvp.tasks.swe_patch_parser import (
    ALL_PARSE_KINDS, ALL_RECOVERY_LABELS, parse_patch_block,
)
from vision_mvp.tasks.swe_bench_bridge import (
    load_jsonl_bank, build_synthetic_event_log, parse_unified_diff,
)
from vision_mvp.wevra.synthetic_llm import (
    SYNTHETIC_MODEL_PROFILES, make_synthetic_response_fn,
)


SCHEMA = "wevra.parser_boundary.v1"

# The bundled bank used by ``local_smoke`` / ``bundled_57`` profiles.
DEFAULT_JSONL = ("vision_mvp/tasks/data/swe_lite_style_bank.jsonl")

# A short, fixed list of parser modes we cross-product over.
# ``"none"`` is included so the closed regex baseline (Phase 41)
# is on the same axis.
DEFAULT_PARSER_MODES: tuple[str, ...] = ("strict", "robust")

# A reasonably broad slice of synthetic distributions. ``mixed`` is
# the only one with intra-distribution variance; the others map a
# single failure shape over every instance.
DEFAULT_MODEL_TAGS: tuple[str, ...] = (
    "synthetic.clean",
    "synthetic.unclosed",
    "synthetic.prose",
    "synthetic.empty",
    "synthetic.fenced",
    "synthetic.multi_block",
    "synthetic.mixed",
)


def _gold_patches_from_bank(jsonl_path: str,
                              n_instances: int | None,
                              ) -> dict[str, tuple[str, str]]:
    """Load (instance_id → (old, new)) pairs from a JSONL bank."""
    tasks, _repo_files = load_jsonl_bank(
        jsonl_path,
        hidden_event_log_factory=(
            lambda t, k=6: build_synthetic_event_log(t, k)),
        limit=n_instances,
    )
    out: dict[str, tuple[str, str]] = {}
    for task in tasks:
        if task.gold_patch:
            out[task.instance_id] = task.gold_patch[0]
    return out


def _failure_kind_distribution(
        gold_patches: dict[str, tuple[str, str]],
        model_tag: str,
        parser_mode: str,
        ) -> dict[str, Any]:
    """Run one (model_tag, parser_mode) pair through the parser
    over every instance in ``gold_patches`` and collect the
    PARSE_OUTCOME structured outcomes."""
    fn = make_synthetic_response_fn(model_tag, gold_patches)
    counts_kind: dict[str, int] = {k: 0 for k in ALL_PARSE_KINDS}
    counts_recovery: dict[str, int] = {
        r or "<none>": 0 for r in ALL_RECOVERY_LABELS}
    n_ok = 0
    n_total = 0
    for instance_id in sorted(gold_patches):
        text = fn("(prompt unused)", instance_id)
        if parser_mode == "none":
            # The closed Phase-41 regex doesn't have a parse_patch_block
            # entry — skip the parser-failure layer.
            continue
        outcome = parse_patch_block(
            text, mode=parser_mode,
            unified_diff_parser=parse_unified_diff)
        n_total += 1
        if outcome.ok:
            n_ok += 1
        kind = outcome.failure_kind or "<unknown>"
        counts_kind[kind] = counts_kind.get(kind, 0) + 1
        rec = outcome.recovery or "<none>"
        counts_recovery[rec] = counts_recovery.get(rec, 0) + 1
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "ok_rate": (n_ok / n_total) if n_total else 0.0,
        "failure_kind": counts_kind,
        "recovery": counts_recovery,
    }


def _normalise(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def _tvd(p: dict[str, int], q: dict[str, int]) -> float:
    """Total Variation Distance between two unnormalised
    multinomial counts (i.e. \\sum_k 0.5 |p_k - q_k| after
    normalisation). Returns a value in [0, 1]."""
    pn = _normalise(p)
    qn = _normalise(q)
    keys = set(pn) | set(qn)
    return 0.5 * sum(abs(pn.get(k, 0.0) - qn.get(k, 0.0))
                       for k in keys)


def run_cross_model_study(*,
                            jsonl_path: str = DEFAULT_JSONL,
                            n_instances: int | None = None,
                            model_tags: tuple[str, ...] = (
                                DEFAULT_MODEL_TAGS),
                            parser_modes: tuple[str, ...] = (
                                DEFAULT_PARSER_MODES),
                            ) -> dict[str, Any]:
    """Sweep ``(model_tag, parser_mode)`` over the bundled bank
    and return a structured result dict.

    The output is JSON-serialisable (tuple keys are stringified
    in the JSON projection ``as_json_dict``).
    """
    gold = _gold_patches_from_bank(jsonl_path, n_instances)
    distributions: dict[tuple[str, str], dict[str, Any]] = {}
    for tag in model_tags:
        if tag not in SYNTHETIC_MODEL_PROFILES:
            raise KeyError(
                f"unknown synthetic model_tag {tag!r}; "
                f"valid: {sorted(SYNTHETIC_MODEL_PROFILES)}")
        for pm in parser_modes:
            distributions[(tag, pm)] = _failure_kind_distribution(
                gold, tag, pm)
    # Pairwise Total Variation Distance over failure_kind, holding
    # parser_mode fixed. Reports cross-distribution variance.
    tvd_pairwise: dict[tuple[str, str, str], float] = {}
    for pm in parser_modes:
        for i, ti in enumerate(model_tags):
            for j, tj in enumerate(model_tags):
                if j <= i:
                    continue
                tvd_pairwise[(ti, tj, pm)] = _tvd(
                    distributions[(ti, pm)]["failure_kind"],
                    distributions[(tj, pm)]["failure_kind"])
    # Parser-mode-conditional shift: TVD between (tag, strict) and
    # (tag, robust) for each tag.
    parser_mode_shift: dict[str, dict[str, float]] = {}
    if "strict" in parser_modes and "robust" in parser_modes:
        for tag in model_tags:
            tvd = _tvd(
                distributions[(tag, "strict")]["failure_kind"],
                distributions[(tag, "robust")]["failure_kind"])
            parser_mode_shift[tag] = {
                "strict_to_robust_tvd": tvd}
    # Empirical claim — minted from the data, not asserted.
    max_cross_tvd = (max(tvd_pairwise.values())
                      if tvd_pairwise else 0.0)
    max_mode_shift = (
        max(d["strict_to_robust_tvd"]
             for d in parser_mode_shift.values())
        if parser_mode_shift else 0.0)
    claim = (
        f"Cross-distribution failure-kind TVD up to {max_cross_tvd:.3f}; "
        f"strict→robust parser-mode shift up to "
        f"{max_mode_shift:.3f}. The PARSE_OUTCOME closed "
        f"vocabulary distinguishes synthetic distributions and "
        f"the parser-mode choice measurably moves their "
        f"failure-kind footprint.")
    return {
        "schema": SCHEMA,
        "n_instances": len(gold),
        "model_tags": list(model_tags),
        "parser_modes": list(parser_modes),
        "distributions": distributions,
        "tvd_pairwise": tvd_pairwise,
        "parser_mode_shift": parser_mode_shift,
        "max_cross_tvd": max_cross_tvd,
        "max_parser_mode_shift": max_mode_shift,
        "claim": claim,
    }


def as_json_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Project ``run_cross_model_study``'s output into a
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


def render_summary_table(result: dict[str, Any]) -> str:
    """Human-readable summary table for printout."""
    lines: list[str] = []
    lines.append(f"{result['schema']}  n_instances={result['n_instances']}")
    lines.append("")
    lines.append("Per-(model_tag, parser_mode) failure-kind distribution:")
    lines.append("")
    fmt = "  {:<26}  {:<8}  ok_rate={:>5.2f}  topkinds={}"
    for (tag, pm), dist in result["distributions"].items():
        topkinds = sorted(
            ((k, v) for k, v in dist["failure_kind"].items() if v),
            key=lambda kv: -kv[1])[:3]
        lines.append(fmt.format(
            tag, pm, dist["ok_rate"],
            ", ".join(f"{k}={v}" for k, v in topkinds)))
    lines.append("")
    lines.append(f"Max cross-distribution TVD: "
                  f"{result['max_cross_tvd']:.3f}")
    lines.append(f"Max parser-mode shift TVD: "
                  f"{result['max_parser_mode_shift']:.3f}")
    lines.append("")
    lines.append("Claim: " + result["claim"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--n-instances", type=int, default=None)
    ap.add_argument("--model-tags", nargs="+",
                     default=list(DEFAULT_MODEL_TAGS))
    ap.add_argument("--parser-modes", nargs="+",
                     default=list(DEFAULT_PARSER_MODES))
    ap.add_argument("--out-json", default=None,
                     help="optional path to write the JSON result")
    args = ap.parse_args(argv)
    result = run_cross_model_study(
        jsonl_path=args.jsonl,
        n_instances=args.n_instances,
        model_tags=tuple(args.model_tags),
        parser_modes=tuple(args.parser_modes))
    print(render_summary_table(result))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(as_json_dict(result), fh, indent=2,
                       default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
