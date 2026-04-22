"""Phase 36 Part B — LLM-driven thread replies.

Drives ``vision_mvp.core.llm_thread_replier.LLMThreadReplier`` on
the Phase-35 contested bank and compares three reply paths:

  * ``deterministic_typed``   — the Phase-35 baseline:
    ``infer_causality_hypothesis`` is the oracle, precision /
    recall 1.0.
  * ``llm_typed_mock``        — a scenario-aware deterministic
    mock LLM replier (``ScenarioAwareMockReplier``) that mimics
    the oracle on a payload-pattern basis; optionally emits a
    fraction of malformed / out-of-vocab replies.
  * ``llm_typed_real``        — a real Ollama LLM replier
    (``LLMClient.generate``). Runs only if ``--model`` is set
    and ``--mock`` is off.

Phase-36 Part B headline (mock bank, seed=35, k=6, scenario-
aware mock at malformed_prob=0):

  * deterministic_typed   contested_acc = 1.00
  * llm_typed_mock        contested_acc = 1.00
  * llm_typed_mock (m=0.5) contested_acc ≈ 0.50 — degrades
                            gracefully (parser fallback UNCERTAIN
                            → NO_CONSENSUS → decoder falls back
                            to static priority).

Reproducible commands:

    # Mock sweep.
    python3 -m vision_mvp.experiments.phase36_llm_replies \\
        --mock --seeds 35 36 \\
        --malformed-probs 0.0 0.1 0.25 0.5 \\
        --out vision_mvp/results_phase36_llm_replies_mock.json

    # Real-LLM spot check (qwen2.5:0.5b replies only; auditor is
    # still the deterministic mock).
    python3 -m vision_mvp.experiments.phase36_llm_replies \\
        --real-replier qwen2.5:0.5b --seeds 35 \\
        --out vision_mvp/results_phase36_llm_replies_0p5b.json

The auditor is always ``MockContestedAuditor`` — Part B isolates
the *reply* fidelity, not the final auditor prompt synthesis.
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

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.llm_thread_replier import (
    DEFAULT_REPLY_KINDS, LLMReplyConfig, LLMReplierStats,
    LLMThreadReplier, DeterministicMockReplier,
    causality_extractor_from_replier,
)
from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_STATIC_HANDOFF, STRATEGY_DYNAMIC,
    STRATEGY_ADAPTIVE_SUB, MockContestedAuditor,
    build_contested_bank, run_contested_loop,
    infer_causality_hypothesis,
)


# Payload-level markers that let a scenario-aware mock replier
# mimic the oracle from the thread-reply prompt alone (without
# access to the scenario's internal causality map).
#
# The rules mirror ``infer_causality_hypothesis`` on the six-
# scenario bank. Every clause is one small regex on the candidate
# payload string.
@dataclass
class ScenarioAwareMockReplier:
    """A deterministic LLM-stand-in that inspects the candidate
    payload (present in the thread-reply prompt) and emits the
    oracle reply_kind. Adds optional malformed / out-of-vocab
    emissions on a hash-determined subset for Phase-36 Part B.

    Matches ``DeterministicMockReplier`` signature so it is a
    drop-in for ``LLMThreadReplier.llm_call``.
    """

    malformed_prob: float = 0.0
    out_of_vocab_prob: float = 0.0
    _calls: int = 0

    def _oracle_kind_from_payload(
            self, role: str, kind: str, payload: str) -> str:
        """Mirror of the Phase-35 causality map for the six-scenario
        bank. Every rule encodes one `(role, kind, payload-pattern)`
        clause derived from the scenario source in
        ``tasks/contested_incident``.
        """
        # Rule 1: db_admin/DEADLOCK_SUSPECTED is always INDEPENDENT.
        if role == "db_admin" and kind == "DEADLOCK_SUSPECTED":
            return REPLY_INDEPENDENT_ROOT
        # Rule 2: db_admin/POOL_EXHAUSTION is always DOWNSTREAM.
        if role == "db_admin" and kind == "POOL_EXHAUSTION":
            return REPLY_DOWNSTREAM_SYMPTOM
        # Rule 3: sysadmin/CRON_OVERRUN — context-dependent:
        #   service=app → INDEPENDENT_ROOT (cron_vs_oom_shadow),
        #   service=archival → UNCERTAIN
        #     (deadlock_vs_shadow_cron),
        #   service=api → DOWNSTREAM_OF:DISK_FILL (concordant).
        if role == "sysadmin" and kind == "CRON_OVERRUN":
            if "service=app" in payload:
                return REPLY_INDEPENDENT_ROOT
            if "service=archival" in payload:
                return REPLY_UNCERTAIN
            return REPLY_DOWNSTREAM_SYMPTOM
        # Rule 4: sysadmin/OOM_KILL — service=batch → UNCERTAIN;
        # otherwise INDEPENDENT_ROOT.
        if role == "sysadmin" and kind == "OOM_KILL":
            if "service=batch" in payload:
                return REPLY_UNCERTAIN
            return REPLY_INDEPENDENT_ROOT
        # Rule 5: sysadmin/DISK_FILL_CRITICAL —
        #   fs=/var/archive → UNCERTAIN (tls_vs_disk_shadow),
        #   otherwise INDEPENDENT_ROOT (concordant).
        if role == "sysadmin" and kind == "DISK_FILL_CRITICAL":
            if "fs=/var/archive" in payload:
                return REPLY_UNCERTAIN
            return REPLY_INDEPENDENT_ROOT
        # Rule 6: network/TLS_EXPIRED —
        #   service=mail → UNCERTAIN (shadow in dns_vs_tls),
        #   otherwise INDEPENDENT_ROOT (tls_vs_disk).
        if role == "network" and kind == "TLS_EXPIRED":
            if "service=mail" in payload:
                return REPLY_UNCERTAIN
            return REPLY_INDEPENDENT_ROOT
        # Rule 7: network/DNS_MISROUTE — always INDEPENDENT_ROOT.
        if role == "network" and kind == "DNS_MISROUTE":
            return REPLY_INDEPENDENT_ROOT
        # Rule 8: network/FW_BLOCK_SURGE → DOWNSTREAM.
        if role == "network" and kind == "FW_BLOCK_SURGE":
            return REPLY_DOWNSTREAM_SYMPTOM
        # Rule 9: monitor/ERROR_RATE_SPIKE and LATENCY_SPIKE →
        # DOWNSTREAM.
        if role == "monitor":
            return REPLY_DOWNSTREAM_SYMPTOM
        return REPLY_UNCERTAIN

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        m = re.search(
            r"YOUR CLAIM:\s*\[([\w_]+)/([\w_]+)\]\s*(.+)", prompt)
        if not m:
            return '{"reply_kind": "UNCERTAIN", "witness": ""}'
        role = m.group(1)
        kind = m.group(2)
        payload = m.group(3).strip()
        reply_kind = self._oracle_kind_from_payload(
            role, kind, payload)

        # Malformed pass — produce plain prose on a
        # hash-determined subset.
        if self.malformed_prob > 0:
            h = hash(("malformed", role, kind, self._calls,
                       payload)) & 0xFFFF
            if h / 0xFFFF < self.malformed_prob:
                return (f"I think the {kind} is probably a root "
                        f"cause, but I'm not sure.")

        # Out-of-vocab pass.
        if self.out_of_vocab_prob > 0:
            h = hash(("oov", role, kind, self._calls,
                       payload)) & 0xFFFF
            if h / 0xFFFF < self.out_of_vocab_prob:
                return ('{"reply_kind": "NOT_IN_VOCAB", '
                        '"witness": "some evidence"}')

        short_witness = " ".join(payload.split()[:6])
        return ('{"reply_kind": "' + reply_kind + '", '
                '"witness": "' + short_witness + '"}')


def _build_replier(mode: str, real_model: str | None,
                    malformed_prob: float,
                    out_of_vocab_prob: float,
                    ) -> tuple[Callable, LLMThreadReplier | None]:
    """Return ``(extractor, replier_or_None)`` for the given mode."""
    if mode == "deterministic_typed":
        return infer_causality_hypothesis, None
    if mode == "llm_typed_mock":
        stub = ScenarioAwareMockReplier(
            malformed_prob=malformed_prob,
            out_of_vocab_prob=out_of_vocab_prob)
        replier = LLMThreadReplier(
            llm_call=stub,
            config=LLMReplyConfig(witness_token_cap=12),
            cache={},
        )
        return causality_extractor_from_replier(replier), replier
    if mode == "llm_typed_real":
        if not real_model:
            raise ValueError("llm_typed_real requires --real-replier")
        client = LLMClient(model=real_model, timeout=300.0)

        def _call(prompt: str) -> str:
            return client.generate(prompt, max_tokens=60,
                                     temperature=0.0)
        replier = LLMThreadReplier(
            llm_call=_call,
            config=LLMReplyConfig(witness_token_cap=12),
            cache={},
        )
        # Attach the client so the driver can read stats.
        replier._real_client = client  # type: ignore[attr-defined]
        return causality_extractor_from_replier(replier), replier
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                      help="use scenario-aware mock replier")
    ap.add_argument("--real-replier", default=None,
                      help="model name for a real Ollama replier "
                           "(e.g. qwen2.5:0.5b)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[35])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--malformed-probs", nargs="+", type=float,
                      default=[0.0])
    ap.add_argument("--oov-probs", nargs="+", type=float,
                      default=[0.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor = MockContestedAuditor()

    modes: list[tuple[str, float, float]] = []
    modes.append(("deterministic_typed", 0.0, 0.0))
    if args.mock or not args.real_replier:
        for mp in args.malformed_probs:
            for oo in args.oov_probs:
                modes.append(("llm_typed_mock", mp, oo))
    if args.real_replier:
        modes.append(("llm_typed_real", 0.0, 0.0))

    t0 = time.time()
    per_config: list[dict] = []
    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for (mode, mp, oo) in modes:
                extractor, replier = _build_replier(
                    mode, args.real_replier, mp, oo)
                strats = (STRATEGY_DYNAMIC, STRATEGY_ADAPTIVE_SUB,
                          STRATEGY_STATIC_HANDOFF)
                pooled_accum: dict[str, dict] = {}
                for strat in strats:
                    # Re-build the replier per strategy so cache
                    # and counters are isolated.
                    if replier is not None:
                        replier.stats = LLMReplierStats()
                        replier.cache = {}
                    rep_single = run_contested_loop(
                        bank, auditor, strategies=(strat,),
                        seed=seed,
                        max_events_in_prompt=200,
                        inbox_capacity=32,
                        causality_extractor=extractor,
                    )
                    pooled_accum.update(rep_single.pooled())
                print(f"\n[phase36-B] k={k} seed={seed} "
                      f"mode={mode} mp={mp} oo={oo}", flush=True)
                for s in strats:
                    p = pooled_accum.get(s, {})
                    if not p:
                        continue
                    print(
                        f"    {s:>18}  "
                        f"acc_full={p['accuracy_full']:.3f}  "
                        f"contest={p['contested_accuracy_full']:.3f}  "
                        f"tok={p['mean_prompt_tokens']:.0f}  "
                        f"fhist={p['failure_hist']}",
                        flush=True)
                replier_stats = (replier.stats.as_dict()
                                   if replier is not None else None)
                if replier is not None and \
                        hasattr(replier, "_real_client"):
                    cli = replier._real_client
                    replier_stats["real_client"] = {  # type: ignore[index]
                        "model": cli.model,
                        "prompt_tokens": cli.stats.prompt_tokens,
                        "output_tokens": cli.stats.output_tokens,
                        "n_calls": cli.stats.n_generate_calls,
                        "total_wall": round(
                            cli.stats.total_wall, 2),
                    }
                per_config.append({
                    "distractors_per_role": k,
                    "seed": seed,
                    "mode": mode,
                    "malformed_prob": mp,
                    "out_of_vocab_prob": oo,
                    "replier_stats": replier_stats,
                    "pooled": pooled_accum,
                })

    wall = time.time() - t0
    print(f"\n[phase36-B] overall wall = {wall:.1f}s")

    payload = {
        "config": vars(args),
        "per_config": per_config,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
