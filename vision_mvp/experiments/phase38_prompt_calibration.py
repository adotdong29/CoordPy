"""Phase 38 Part C — prompt-engineering calibration study.

Measures whether the Phase-37 ``sem_root_as_symptom`` bias
(Theorem P37-1; real LLMs over-emit ``DOWNSTREAM_SYMPTOM``)
is prompt-shaped. Runs the five Phase-38 prompt variants
(``core/prompt_variants``) through a replier + calibration
pipeline and reports per-variant calibration rates and
downstream task accuracy.

Two execution modes:

  * ``--mode mock``  (default) — uses a
    ``BiasShiftMockReplier`` that *simulates* an LLM whose
    bias toward DS shifts by a fixed delta per prompt
    variant. The delta table is ship-grade seed-stable and
    reproduces sub-second; it is not a prediction of a real
    LLM's behaviour, just a lightweight scaffold to make
    the Phase-38 experiment frame runnable and the
    substrate code path exercised.
  * ``--mode real``  — uses a real Ollama model under each
    variant; writes per-variant calibration reports. Same
    shape as Phase-37's real-LLM calibration driver, one
    row per (model, variant).

The point is to (a) ship the experiment frame — prompt
variants, calibration wrapper, report shape — so the real-LLM
sweep becomes a parameter change, and (b) characterise the
substrate invariant: regardless of whether any variant
actually shifts the real-LLM bias, the substrate's bounded
typed-reply contract holds on every variant.

Reproducible commands:

    # Mock sweep (sub-second; deterministic).
    python3 -m vision_mvp.experiments.phase38_prompt_calibration \\
        --mode mock --seeds 35 36 --distractor-counts 4 6 \\
        --out vision_mvp/results_phase38_prompt_calibration_mock.json

    # Real-LLM sweep (requires Ollama).
    python3 -m vision_mvp.experiments.phase38_prompt_calibration \\
        --mode real --models qwen2.5:0.5b --seeds 35 \\
        --distractor-counts 4 \\
        --out vision_mvp/results_phase38_prompt_calibration_real.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMThreadReplier,
)
from vision_mvp.core.prompt_variants import (
    ALL_PROMPT_VARIANTS, PROMPT_VARIANT_DEFAULT,
    PromptVariantReport,
    build_thread_reply_prompt_variant,
)
from vision_mvp.core.reply_calibration import (
    CalibratingReplier, ReplyCalibrationReport,
    causality_extractor_from_calibrating_replier,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_ADAPTIVE_SUB, STRATEGY_DYNAMIC,
    STRATEGY_STATIC_HANDOFF, MockContestedAuditor,
    build_contested_bank, infer_causality_hypothesis,
    run_contested_loop,
)


# =============================================================================
# Variant-aware LLMThreadReplier wrapper
# =============================================================================


@dataclass
class VariantLLMThreadReplier:
    """Wrap an ``LLMThreadReplier`` so a named prompt-variant
    builder is used in place of ``build_thread_reply_prompt``.

    The wrapped replier's parse / stats pathway is unchanged —
    the only surgical change is which prompt is sent to the
    underlying ``llm_call``. Output shape is identical to
    ``LLMThreadReplier.__call__``.
    """

    inner: LLMThreadReplier
    variant: str = PROMPT_VARIANT_DEFAULT

    @property
    def config(self) -> LLMReplyConfig:
        return self.inner.config

    @property
    def stats(self):
        return self.inner.stats

    def __call__(self, scenario, role: str, kind: str,
                 payload: str, other_candidates=(),
                 role_events=None):
        cfg = self.inner.config
        prompt = build_thread_reply_prompt_variant(
            variant=self.variant, role=role,
            candidate_role=role, candidate_kind=kind,
            candidate_payload=payload,
            other_candidates=other_candidates,
            role_events=role_events, cfg=cfg)
        self.inner.stats.n_calls += 1
        self.inner.stats.total_prompt_chars += len(prompt)
        reply_text = self.inner.llm_call(prompt)
        self.inner.stats.total_reply_chars += len(reply_text)
        # Reuse the inner's parse contract.
        from vision_mvp.core.llm_thread_replier import (
            parse_llm_reply_json, _REPLY_JSON_RE,
        )
        reply_kind, witness, well_formed = parse_llm_reply_json(
            reply_text, cfg)
        if well_formed:
            self.inner.stats.n_well_formed += 1
        else:
            if _REPLY_JSON_RE.search(reply_text):
                self.inner.stats.n_out_of_vocab += 1
            else:
                self.inner.stats.n_malformed += 1
        return reply_kind, witness, well_formed


# =============================================================================
# BiasShiftMockReplier — deterministic variant-responsive LLM stand-in
# =============================================================================


@dataclass
class BiasShiftMockReplier:
    """Deterministic mock whose bias shifts by variant.

    The Phase-37 baseline bias (``sem_root_as_symptom = 0.50``,
    ``sem_uncertain_as_symptom = 0.40``, ``correct = 0.10``)
    is encoded as the ``default`` variant. Other variants
    apply a fixed per-variant shift. The shifts are chosen so
    the experiment frame exercises the full signed range of
    possible outcomes (some variants reduce bias, some preserve
    it, one amplifies it) — the point is to validate the
    measurement pipeline, not to claim predictive power.

    Outputs parseable JSON on every call with the bias
    distribution encoded by a deterministic hash of (role,
    kind, payload, call_index, variant).
    """

    variant: str = PROMPT_VARIANT_DEFAULT
    _calls: int = 0

    # Per-variant bias table: (p_ir_on_ir, p_ir_on_ds, p_ir_on_unc,
    # p_ds_on_ir, p_ds_on_ds, p_ds_on_unc, p_unc_on_ir,
    # p_unc_on_ds, p_unc_on_unc). The default row reproduces
    # Phase-37's 0.5b/7b calibration roughly; other variants
    # target shifts that a thoughtful prompt might plausibly
    # produce.
    _variant_table = {
        PROMPT_VARIANT_DEFAULT: {
            "IR": (0.10, 0.50, 0.40),
            "DS": (0.00, 0.10, 0.90),
            "UNC": (0.00, 0.40, 0.60),
        },
        "contrastive": {
            # contrastive reduces DS-default on IR
            "IR": (0.70, 0.20, 0.10),
            "DS": (0.00, 0.70, 0.30),
            "UNC": (0.10, 0.10, 0.80),
        },
        "few_shot": {
            # few_shot anchors IR examples; moderate gain
            "IR": (0.50, 0.40, 0.10),
            "DS": (0.10, 0.50, 0.40),
            "UNC": (0.10, 0.30, 0.60),
        },
        "rubric": {
            # rubric: stepped reasoning, strong IR lift
            "IR": (0.80, 0.10, 0.10),
            "DS": (0.05, 0.65, 0.30),
            "UNC": (0.10, 0.05, 0.85),
        },
        "forced_order": {
            # forced_order: two-step tag forces distinct
            # answers but without rubric; mild gain
            "IR": (0.40, 0.40, 0.20),
            "DS": (0.10, 0.40, 0.50),
            "UNC": (0.10, 0.20, 0.70),
        },
    }

    def _oracle_from_prompt(self, prompt: str
                              ) -> tuple[str, str, str]:
        m = re.search(
            r"YOUR CLAIM:\s*\[([\w_]+)/([\w_]+)\]\s*(.+)",
            prompt)
        if not m:
            return "", "", ""
        return m.group(1), m.group(2), m.group(3)

    def _true_class(self, role: str, kind: str,
                     payload: str) -> str:
        # Mirror of Phase-36 ScenarioAwareMockReplier's
        # _oracle_kind_from_payload, inlined so this module does
        # not depend on the Phase-36 driver.
        if role == "db_admin" and kind == "DEADLOCK_SUSPECTED":
            return "IR"
        if role == "db_admin" and kind == "POOL_EXHAUSTION":
            return "DS"
        if role == "sysadmin" and kind == "CRON_OVERRUN":
            if "service=app" in payload:
                return "IR"
            if "service=archival" in payload:
                return "UNC"
            return "DS"
        if role == "sysadmin" and kind == "OOM_KILL":
            if "service=batch" in payload:
                return "UNC"
            return "IR"
        if role == "sysadmin" and kind == "DISK_FILL_CRITICAL":
            if "fs=/var/archive" in payload:
                return "UNC"
            return "IR"
        if role == "network" and kind == "TLS_EXPIRED":
            if "service=mail" in payload:
                return "UNC"
            return "IR"
        if role == "network" and kind == "DNS_MISROUTE":
            return "IR"
        if role == "network" and kind == "FW_BLOCK_SURGE":
            return "DS"
        if role == "monitor":
            return "DS"
        return "UNC"

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        role, kind, payload = self._oracle_from_prompt(prompt)
        if not role:
            return ('{"reply_kind": "UNCERTAIN", '
                    '"witness": ""}')
        true_cls = self._true_class(role, kind, payload)
        row = self._variant_table.get(
            self.variant, self._variant_table[
                PROMPT_VARIANT_DEFAULT])[true_cls]
        # Deterministic pick: draw a pseudo-uniform from a hash.
        h = hash((self.variant, role, kind, payload,
                   self._calls)) & 0xFFFF
        u = h / 0xFFFF
        # Row = (p_ir, p_ds, p_unc). Partition [0, 1] into the
        # three buckets.
        p_ir, p_ds, _p_unc = row
        if u < p_ir:
            emitted = REPLY_INDEPENDENT_ROOT
        elif u < p_ir + p_ds:
            emitted = REPLY_DOWNSTREAM_SYMPTOM
        else:
            emitted = REPLY_UNCERTAIN
        # Witness: a short slice of the payload.
        short = " ".join(payload.split()[:6])
        return ('{"reply_kind": "' + emitted + '", '
                '"witness": "' + short + '"}')


# =============================================================================
# Runner
# =============================================================================


def _build_variant_pipeline(variant: str,
                              *,
                              mode: str = "mock",
                              model: str | None = None,
                              timeout: float = 180.0,
                              ) -> tuple[VariantLLMThreadReplier,
                                         CalibratingReplier,
                                         ReplyCalibrationReport,
                                         Callable]:
    """Build the Phase-38 prompt-variant pipeline for one
    variant.

    Returns (variant_replier, calibrating_wrapper, report,
    causality_extractor).
    """
    if mode == "mock":
        stub = BiasShiftMockReplier(variant=variant)
        inner = LLMThreadReplier(
            llm_call=stub,
            config=LLMReplyConfig(witness_token_cap=12),
            cache={})
    else:
        from vision_mvp.core.llm_client import LLMClient
        client = LLMClient(model=model, timeout=timeout)

        def _call(prompt: str) -> str:
            return client.generate(prompt, max_tokens=60,
                                    temperature=0.0)
        inner = LLMThreadReplier(
            llm_call=_call,
            config=LLMReplyConfig(witness_token_cap=12),
            cache={})
    variant_rep = VariantLLMThreadReplier(
        inner=inner, variant=variant)
    report = ReplyCalibrationReport()
    wrapper = CalibratingReplier(
        inner=inner,  # CalibratingReplier calls inner; we swap
        oracle=infer_causality_hypothesis, report=report)
    # Replace wrapper.inner with variant-aware replier so the
    # calibration wrapper sees variant-aware prompts.
    wrapper.inner = variant_rep  # type: ignore[assignment]
    extractor = causality_extractor_from_calibrating_replier(
        wrapper)
    return variant_rep, wrapper, report, extractor


def run_variant(variant: str,
                  seeds: list[int],
                  distractor_counts: list[int],
                  *,
                  mode: str = "mock",
                  model: str | None = None,
                  timeout: float = 180.0,
                  strategies: tuple[str, ...] = (
                      STRATEGY_DYNAMIC,
                      STRATEGY_ADAPTIVE_SUB,
                      STRATEGY_STATIC_HANDOFF),
                  ) -> dict:
    auditor = MockContestedAuditor()
    _replier, _wrapper, report, extractor = \
        _build_variant_pipeline(
            variant, mode=mode, model=model,
            timeout=timeout)
    t0 = time.time()
    pooled_by_strat: dict[str, list[dict]] = {}
    for k in distractor_counts:
        for seed in seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for strat in strategies:
                rep = run_contested_loop(
                    bank, auditor, strategies=(strat,),
                    seed=seed,
                    max_events_in_prompt=200,
                    inbox_capacity=32,
                    causality_extractor=extractor)
                pooled_by_strat.setdefault(strat, []).append(
                    rep.pooled().get(strat, {}))
    wall = time.time() - t0
    means: dict[str, dict] = {}
    for strat, rows in pooled_by_strat.items():
        if not rows:
            continue
        n = len(rows)
        mean_acc = sum(r.get("accuracy_full", 0.0)
                        for r in rows) / max(1, n)
        mean_cont = sum(r.get("contested_accuracy_full", 0.0)
                         for r in rows) / max(1, n)
        means[strat] = {
            "n_cells": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "contested_accuracy_full_mean": round(mean_cont, 4),
        }
    return {
        "variant": variant,
        "mode": mode,
        "model": model,
        "calibration_rates": report.rates(),
        "pooled_mean_over_cells": means,
        "wall_seconds": round(wall, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mock", "real"),
                      default="mock")
    ap.add_argument("--models", nargs="+",
                      default=["qwen2.5:0.5b"])
    ap.add_argument("--variants", nargs="+",
                      default=list(ALL_PROMPT_VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int,
                      default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[4, 6])
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB,
                                STRATEGY_STATIC_HANDOFF])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    models = args.models if args.mode == "real" else [None]

    for model in models:
        for variant in args.variants:
            print(f"\n[phase38-C] mode={args.mode} "
                  f"model={model!r} variant={variant}",
                  flush=True)
            try:
                row = run_variant(
                    variant=variant,
                    seeds=args.seeds,
                    distractor_counts=args.distractor_counts,
                    mode=args.mode, model=model,
                    timeout=args.timeout,
                    strategies=tuple(args.strategies))
            except Exception as ex:
                row = {"variant": variant, "model": model,
                        "mode": args.mode, "error": str(ex)}
            rows.append(row)
            if "calibration_rates" in row:
                rates = row["calibration_rates"]
                print(f"  n_calls={rates['n_calls']} "
                      f"correct={rates['correct_rate']:.3f} "
                      f"sem_wrong={rates['semantic_wrong_rate']:.3f}")
                for strat, p in row[
                        "pooled_mean_over_cells"].items():
                    print(f"    {strat:>20}  "
                          f"acc={p['accuracy_full_mean']:.3f}  "
                          f"contest={p['contested_accuracy_full_mean']:.3f}")

    wall = time.time() - t0

    # Summary table.
    print()
    print("=" * 82)
    print("PHASE 38 PART C — prompt-variant calibration (pooled)")
    print("=" * 82)
    print(f"{'variant':16} {'model':20} {'correct':>10} "
          f"{'sem_wrong':>10} {'dyn_acc':>10} {'dyn_ctst':>10}")
    for r in rows:
        if "calibration_rates" not in r:
            continue
        rates = r["calibration_rates"]
        dyn = r["pooled_mean_over_cells"].get(
            STRATEGY_DYNAMIC, {})
        model_str = str(r.get("model", ""))
        print(f"{r['variant']:16} {model_str:20} "
              f"{rates['correct_rate']:10.3f} "
              f"{rates['semantic_wrong_rate']:10.3f} "
              f"{dyn.get('accuracy_full_mean', 0):10.3f} "
              f"{dyn.get('contested_accuracy_full_mean', 0):10.3f}")

    payload = {
        "config": vars(args),
        "rows": rows,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
