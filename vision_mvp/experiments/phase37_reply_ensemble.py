"""Phase 37 Part B — reply-axis ensemble defenses.

Conjecture C36-7: a *redundant-reply* protocol with robust
aggregation can recover the Phase-36 adversarial-reply collapse
(Theorem P36-2) in the same way Phase-34's ``UnionExtractor``
recovered the adversarial-extractor collapse.

This driver measures three ensemble patterns against the single
Phase-36 LLMThreadReplier path on the contested bank under:

  1. **Clean baseline** — no noise applied.
  2. **Synthetic malformed_prob = 0.5** — half of replies are
     plain prose instead of JSON (stresses syntactic-noise
     recovery).
  3. **Synthetic mislabel** via
     ``core/reply_noise.ReplyNoiseConfig(mislabel_prob=0.5)``
     (stresses semantic-noise recovery — the dominant Phase-37
     failure mode).
  4. **Adversarial drop_root (budget=1)** — the worst-case
     Phase-36 reply-axis attack.

Primitives compared (all behind ``core/reply_ensemble``):

  * ``single``          — Phase-36 ``LLMThreadReplier`` alone
    (scenario-aware mock as the LLM). Baseline.
  * ``dual_agree``      — two independent mock repliers with
    different deterministic seeds (implemented as two
    scenario-aware mocks that use the same payload rule — so
    they agree on clean inputs and disagree under noise). AND-
    gated: emits the agreed reply_kind or UNCERTAIN otherwise.
  * ``primary_fallback`` — primary LLMThreadReplier with a
    deterministic scenario-aware fallback on parse failure.
  * ``verified``        — LLMThreadReplier + deterministic
    verifier (``verifier_from_payload_classifier`` over the
    Phase-36 ``ScenarioAwareMockReplier`` classifier).

The headline compares:

  * Accuracy per (mode × noise-cell).
  * False-positive (CONFLICT) and false-negative (NO_CONSENSUS)
    rates.
  * Token overhead per ensemble mode (roughly 2× calls for
    dual/verified vs 1× for single / primary_fallback when
    primary is well-formed).

Reproducible commands:

    python3 -m vision_mvp.experiments.phase37_reply_ensemble \\
        --seeds 35 36 --distractor-counts 6 \\
        --out vision_mvp/results_phase37_reply_ensemble.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMReplierStats, LLMThreadReplier,
    causality_extractor_from_replier,
)
from vision_mvp.core.reply_ensemble import (
    ALL_ENSEMBLE_MODES, EnsembleReplier, EnsembleStats,
    MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK, MODE_VERIFIED,
    causality_extractor_from_ensemble,
    verifier_from_payload_classifier,
)
from vision_mvp.core.reply_noise import (
    AdversarialReplyConfig, ReplyNoiseConfig,
    ReplyCorruptionReport, adversarial_reply_extractor,
    ADVERSARIAL_REPLY_MODE_DROP_ROOT,
    noisy_causality_extractor,
)
from vision_mvp.experiments.phase36_llm_replies import (
    ScenarioAwareMockReplier,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_ADAPTIVE_SUB, STRATEGY_DYNAMIC,
    STRATEGY_STATIC_HANDOFF, MockContestedAuditor,
    build_contested_bank, run_contested_loop,
    infer_causality_hypothesis,
)


def _build_scenario_aware_replier(malformed_prob: float = 0.0,
                                    out_of_vocab_prob: float = 0.0,
                                    ) -> LLMThreadReplier:
    stub = ScenarioAwareMockReplier(
        malformed_prob=malformed_prob,
        out_of_vocab_prob=out_of_vocab_prob)
    return LLMThreadReplier(
        llm_call=stub,
        config=LLMReplyConfig(witness_token_cap=12),
        cache={},
    )


# -----------------------------------------------------------------------------
# Biased-primary stub — models a real-LLM systematic wrong-kind emission.
#
# Motivated by the Phase-37 Part A real-LLM calibration: qwen2.5:0.5b emits
# ``DOWNSTREAM_SYMPTOM`` on ~ 50 % of the Phase-35 contested calls (including
# on actual INDEPENDENT_ROOT candidates). A related failure mode is the
# mirror — a model that over-eagerly emits ``INDEPENDENT_ROOT`` on every
# candidate, which under the Phase-35 rule produces CONFLICT on every
# contested scenario and collapses the dynamic strategy to 0.333.
# -----------------------------------------------------------------------------


@dataclass
class BiasedIRMockStub:
    """Stub LLM that always emits ``INDEPENDENT_ROOT`` for any
    parseable prompt. Used to model the over-eager semantic bias
    — the reply-axis analogue of Phase-34's over-eager extractor.
    """

    _calls: int = 0

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        # Extract the candidate tokens for a minimal witness.
        m = re.search(r"YOUR CLAIM:\s*\[([\w_]+)/([\w_]+)\]",
                       prompt)
        witness = (m.group(2) if m else "evidence")
        return ('{"reply_kind": "' + REPLY_INDEPENDENT_ROOT + '", '
                '"witness": "' + witness + '"}')


def _build_biased_ir_replier() -> LLMThreadReplier:
    return LLMThreadReplier(
        llm_call=BiasedIRMockStub(),
        config=LLMReplyConfig(witness_token_cap=12),
        cache={},
    )


def _payload_classifier_from_mock() -> Callable[[str, str, str], str]:
    """A ``(role, kind, payload)`` -> ``reply_kind`` classifier
    that mirrors ``ScenarioAwareMockReplier._oracle_kind_from_payload``
    without an LLM in the loop. Used by ``verified`` mode as a
    deterministic cross-check against a chatty LLM.
    """
    classifier_impl = ScenarioAwareMockReplier()

    def _cls(role: str, kind: str, payload: str) -> str:
        return classifier_impl._oracle_kind_from_payload(
            role, kind, payload)

    return _cls


_ENSEMBLE_MODE_DESCRIPTIONS = {
    MODE_DUAL_AGREE: "two mock repliers; AND-gate on agreement",
    MODE_PRIMARY_FALLBACK: "chatty primary + scenario-aware fallback",
    MODE_VERIFIED: "chatty primary + deterministic payload verifier",
}


# =============================================================================
# Build a noisy / adversarial wrapper around a *replier's extractor* so we
# match Phase-36 semantics — noise is applied on the extractor output.
# =============================================================================


def _wrap_extractor(
        base_extractor: Callable,
        noise_spec: dict,
        seed: int,
        ) -> tuple[Callable, ReplyCorruptionReport]:
    rep = ReplyCorruptionReport()
    if noise_spec.get("adversarial"):
        mode = noise_spec["adversarial"]
        adv = AdversarialReplyConfig(
            target_mode=mode,
            budget=noise_spec.get("budget", 1))
        return (adversarial_reply_extractor(
                    base_extractor, adv, report=rep),
                rep)
    cfg = ReplyNoiseConfig(
        drop_prob=noise_spec.get("drop_prob", 0.0),
        mislabel_prob=noise_spec.get("mislabel_prob", 0.0),
        seed=seed)
    return noisy_causality_extractor(
        base_extractor, cfg, report=rep), rep


# =============================================================================
# Make an extractor for a given (mode, mock_malformed_prob) cell
# =============================================================================


def _make_primary(biased_primary: bool, mock_malformed_prob: float
                    ) -> LLMThreadReplier:
    if biased_primary:
        return _build_biased_ir_replier()
    return _build_scenario_aware_replier(
        malformed_prob=mock_malformed_prob)


def _make_ensemble_extractor(mode: str,
                               mock_malformed_prob: float,
                               biased_primary: bool = False,
                               ) -> tuple[Callable, EnsembleReplier | None,
                                          LLMThreadReplier | None]:
    """Return ``(extractor, ensemble_or_None, primary_replier_or_None)``.

    The extractor is a ``CausalityExtractor`` (shape
    ``(scenario, role, kind, payload) -> str``) with synthetic
    noise wired in at the ensemble input. Ensemble-mode extractors
    wire through the ensemble; ``single`` wires through the
    LLMThreadReplier directly.

    If ``biased_primary`` is True, the primary replier always
    emits ``INDEPENDENT_ROOT`` regardless of the scenario — the
    adversarial reply-generation bias. Secondary (if any) is a
    *clean* scenario-aware mock so we can measure ensemble
    recovery.
    """
    primary = _make_primary(biased_primary, mock_malformed_prob)
    if mode == "single":
        ext = causality_extractor_from_replier(primary)
        return ext, None, primary
    if mode == MODE_DUAL_AGREE:
        # Secondary is a clean scenario-aware mock — models the
        # safety-path of a second-opinion replier that is more
        # reliable than the biased primary. Under a biased
        # primary, dual_agree emits only where both agree — i.e.
        # exactly on the correct answers the clean secondary
        # delivered.
        secondary = _build_scenario_aware_replier(
            malformed_prob=mock_malformed_prob)
        ensemble = EnsembleReplier(
            mode=MODE_DUAL_AGREE,
            primary=primary,
            secondary=secondary,
        )
    elif mode == MODE_PRIMARY_FALLBACK:
        clean_stub = ScenarioAwareMockReplier(
            malformed_prob=0.0, out_of_vocab_prob=0.0)
        fallback = LLMThreadReplier(
            llm_call=clean_stub,
            config=LLMReplyConfig(witness_token_cap=12),
            cache={},
        )
        ensemble = EnsembleReplier(
            mode=MODE_PRIMARY_FALLBACK,
            primary=primary,
            secondary=fallback,
        )
    elif mode == MODE_VERIFIED:
        classifier = _payload_classifier_from_mock()
        verifier = verifier_from_payload_classifier(classifier)
        ensemble = EnsembleReplier(
            mode=MODE_VERIFIED, primary=primary,
            verifier=verifier)
    else:
        raise ValueError(f"unknown ensemble mode {mode!r}")
    return causality_extractor_from_ensemble(ensemble), \
        ensemble, primary


# =============================================================================
# Main sweep
# =============================================================================


MODE_KEYS = ("single", MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK,
              MODE_VERIFIED)


def _noise_cells() -> list[tuple[str, dict]]:
    return [
        ("clean", {}),
        ("synth_malformed_0.5", {"_mock_malformed_prob": 0.5}),
        ("synth_mislabel_0.5", {"mislabel_prob": 0.5}),
        ("adv_drop_root", {"adversarial":
                              ADVERSARIAL_REPLY_MODE_DROP_ROOT,
                           "budget": 1}),
        # Biased-primary cell: primary always emits
        # INDEPENDENT_ROOT regardless of scenario. Models a
        # real-LLM over-eager-IR bias.
        ("biased_primary_ir", {"_biased_primary": True}),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--modes", nargs="+", default=list(MODE_KEYS))
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB,
                                STRATEGY_STATIC_HANDOFF])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor = MockContestedAuditor()

    cells = _noise_cells()
    t0 = time.time()
    per_cell: list[dict] = []
    print(f"\n[phase37-B] modes={args.modes}  "
          f"noise_cells={[c[0] for c in cells]}", flush=True)

    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for (noise_name, noise_spec) in cells:
                for mode in args.modes:
                    mock_mal = noise_spec.get(
                        "_mock_malformed_prob", 0.0)
                    biased = noise_spec.get(
                        "_biased_primary", False)
                    noise_spec_for_wrap = {
                        kk: vv for kk, vv in noise_spec.items()
                        if not kk.startswith("_")
                    }
                    pooled_accum: dict[str, dict] = {}
                    ensemble_final = None
                    primary_final = None
                    rep_rep_final = ReplyCorruptionReport()
                    for strat in args.strategies:
                        # Build a fresh extractor + noise wrapper
                        # per strategy so stateful wrappers
                        # (adversarial budgets, deterministic RNG)
                        # are not depleted by the previous strategy.
                        ext_i, ensemble_i, primary_i = \
                            _make_ensemble_extractor(
                                mode,
                                mock_malformed_prob=mock_mal,
                                biased_primary=biased)
                        if noise_spec_for_wrap:
                            ext_i, rep_rep_i = _wrap_extractor(
                                ext_i, noise_spec_for_wrap,
                                seed=seed)
                        else:
                            rep_rep_i = ReplyCorruptionReport()
                        rep_single = run_contested_loop(
                            bank, auditor, strategies=(strat,),
                            seed=seed,
                            max_events_in_prompt=200,
                            inbox_capacity=32,
                            causality_extractor=ext_i,
                        )
                        pooled_accum.update(rep_single.pooled())
                        if ensemble_i is not None:
                            ensemble_final = ensemble_i
                        if primary_i is not None:
                            primary_final = primary_i
                        rep_rep_final = rep_rep_i
                    ensemble = ensemble_final
                    primary = primary_final
                    rep_rep = rep_rep_final
                    print(f"\n[phase37-B] k={k} seed={seed} "
                          f"noise={noise_name:18} mode={mode:18}",
                          flush=True)
                    for s in args.strategies:
                        p = pooled_accum.get(s, {})
                        if not p:
                            continue
                        print(
                            f"    {s:>14}  "
                            f"acc_full={p['accuracy_full']:.3f}  "
                            f"contest={p['contested_accuracy_full']:.3f}  "
                            f"fhist={p['failure_hist']}", flush=True)
                    ensemble_stats = (ensemble.stats.as_dict()
                                        if ensemble is not None else None)
                    per_cell.append({
                        "distractors_per_role": k,
                        "seed": seed,
                        "noise_cell": noise_name,
                        "noise_spec": noise_spec,
                        "mode": mode,
                        "ensemble_stats": ensemble_stats,
                        "primary_stats": (primary.stats.as_dict()
                                            if primary is not None
                                            else None),
                        "reply_corruption":
                            rep_rep.as_dict(),
                        "pooled": pooled_accum,
                    })

    wall = time.time() - t0
    print(f"\n[phase37-B] overall wall = {wall:.1f}s")

    # Summary: per (noise_cell, mode) pooled across seeds/k.
    key_tuples: dict[tuple[str, str], list[dict]] = {}
    for row in per_cell:
        kt = (row["noise_cell"], row["mode"])
        key_tuples.setdefault(kt, []).append(row)
    pooled_out: list[dict] = []
    print()
    print("=" * 78)
    print("PHASE 37 PART B — POOLED (noise_cell, mode)")
    print("=" * 78)
    for kt in sorted(key_tuples.keys()):
        rows = key_tuples[kt]
        # Per-strategy means across rows.
        per_strat_means: dict[str, dict] = {}
        for s in args.strategies:
            accs = [r["pooled"].get(s, {}).get("accuracy_full", 0.0)
                     for r in rows if s in r["pooled"]]
            conts = [r["pooled"].get(s, {}).get(
                        "contested_accuracy_full", 0.0)
                     for r in rows if s in r["pooled"]]
            if accs:
                per_strat_means[s] = {
                    "n": len(accs),
                    "accuracy_full_mean": round(
                        sum(accs) / len(accs), 4),
                    "contested_accuracy_full_mean": round(
                        sum(conts) / max(1, len(conts)), 4),
                }
        print(f"  noise={kt[0]:18} mode={kt[1]:18}")
        for s, p in per_strat_means.items():
            print(f"    {s:>16}  acc={p['accuracy_full_mean']:.3f}  "
                  f"contest={p['contested_accuracy_full_mean']:.3f}  "
                  f"(n={p['n']})")
        pooled_out.append({
            "noise_cell": kt[0], "mode": kt[1],
            "per_strategy_means": per_strat_means,
        })

    payload = {
        "config": vars(args),
        "noise_cells": [c[0] for c in cells],
        "per_cell": per_cell,
        "pooled_by_cell_and_mode": pooled_out,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
