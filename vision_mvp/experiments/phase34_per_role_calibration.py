"""Phase 34 Part A — per-role extractor-noise calibration benchmark.

Phase 33 showed that a pooled i.i.d. Bernoulli noise model
(``core/extractor_noise``) approximates the real 0.5b-LLM extractor's
compliance-domain noise within ~10 pp on the pooled quadruple. The
same Phase-33 data also revealed *strongly heterogeneous* per-role
noise: on compliance the 0.5b has drop_rate = 0.50 on legal but
drop_rate = 1.00 on finance. Phase 33 § B.5 flagged this as Conjecture
C33-3: pooled i.i.d. hides structure the substrate's decoder may
actually depend on.

This experiment is the Phase 34 Part A instrument. It:

  1. Calibrates an LLM extractor (real Ollama call or the
     ``DeterministicMockExtractorLLM``) on each of the three non-code
     domains (incident, compliance, security) over the full scenario
     bank.
  2. Reports per-role (δ̂, ε̂, μ̂, π̂) — not just pooled.
  3. Maps each role to the closest Phase-32 synthetic grid point and
     records which role is the *limiting* one (highest drop).
  4. Replays the Phase-32 substrate sweep with a
     ``PerRoleNoiseConfig`` fit from the measured per-role profile,
     and compares the per-role-replay accuracy to the pooled-replay
     accuracy. The gap between them is the empirical measure of
     *how much pooled calibration hides*.

Reproducible commands:

    # Deterministic mock (no network, seconds of wall).
    python3 -m vision_mvp.experiments.phase34_per_role_calibration --mode mock \\
        --out vision_mvp/results_phase34_per_role_calibration_mock.json

    # Real 0.5b Ollama, compliance domain only.
    python3 -m vision_mvp.experiments.phase34_per_role_calibration --mode real \\
        --model qwen2.5:0.5b --domains compliance \\
        --out vision_mvp/results_phase34_per_role_calibration_0p5b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_calibration import (
    calibrate_extractor, per_role_audit_summary,
)
from vision_mvp.core.extractor_noise import (
    NoiseConfig, PerRoleNoiseConfig,
    build_from_audit_per_role, noisy_extractor,
    per_role_noisy_extractor,
)
from vision_mvp.experiments.phase33_llm_extractor import (
    DOMAIN_ADAPTERS, _build_llm_extractor,
)


def _replay_substrate(adapter, extractor_fn, distractors, seed):
    """Run the substrate under ``extractor_fn`` and return the pooled
    substrate-strategy stats from the domain's loop."""
    bank = adapter["build_bank"](seed=seed,
                                   distractors_per_role=distractors)
    mock_aud = adapter["mock_auditor"]()
    loop = adapter["run_loop"]
    rep = loop(bank, mock_aud, strategies=("substrate",),
                seed=seed, extractor=extractor_fn)
    return rep.pooled()["substrate"]


def run_one_domain(adapter, mode: str, model: str, k: int, seed: int,
                    sweep_pooled: dict | None) -> dict:
    # 1. Build the LLM extractor and run calibration.
    extractor = _build_llm_extractor(adapter, mode, model, None)
    audit = calibrate_extractor(
        extractor, adapter["build_bank"](seed=seed,
                                          distractors_per_role=k),
        adapter["role_events"],
        adapter["causal_ids"],
        adapter["gold_chain"],
        roles_to_probe=list(adapter["producer_roles"]),
        extractor_label=f"{mode}:{model}",
    )

    # 2. Substrate accuracy under the real / mock LLM extractor.
    real_pooled = _replay_substrate(
        adapter, extractor, k, seed)

    # 3. Per-role synthetic match.
    domain = adapter["name"]
    if sweep_pooled is not None:
        sweep_domain = ("incident" if domain == "incident"
                        else "compliance" if domain == "compliance"
                        else "compliance")  # security maps to compliance
    else:
        sweep_domain = None
    summary = per_role_audit_summary(
        audit, sweep_pooled, domain=sweep_domain)

    # 4. Pooled-replay accuracy — run substrate under a *single*
    # NoiseConfig fit from the pooled quadruple on the regex baseline.
    baseline = adapter["baseline_extractor"]
    pooled_cfg = NoiseConfig(
        drop_prob=round(audit.drop_rate, 4),
        spurious_prob=round(audit.spurious_per_event, 4),
        mislabel_prob=round(audit.mislabel_rate, 4),
        payload_corrupt_prob=round(audit.payload_corrupt_rate, 4),
        seed=seed,
    )
    pooled_ext = noisy_extractor(
        baseline, adapter["known_kinds"], pooled_cfg)
    pooled_replay = _replay_substrate(adapter, pooled_ext, k, seed)

    # 5. Per-role-replay accuracy — each role gets its own
    # NoiseConfig from the audit.
    per_role_cfg = build_from_audit_per_role(
        audit_by_role=audit.by_role, seed=seed)
    per_role_ext = per_role_noisy_extractor(
        baseline, adapter["known_kinds"], per_role_cfg)
    per_role_replay = _replay_substrate(adapter, per_role_ext, k, seed)

    # Gap: |acc(real) - acc(pooled_replay)| vs
    # |acc(real) - acc(per_role_replay)|. A smaller gap on the per-role
    # replay is the empirical evidence that per-role fit is *closer* to
    # the real LLM's substrate effect than the pooled one.
    real_acc = float(real_pooled.get("accuracy_full", 0.0))
    pooled_acc = float(pooled_replay.get("accuracy_full", 0.0))
    per_role_acc = float(per_role_replay.get("accuracy_full", 0.0))

    return {
        "domain": domain,
        "mode": mode, "model": model,
        "distractors_per_role": k, "seed": seed,
        "audit": audit.as_dict(),
        "substrate_measured_real": {
            "accuracy_full": real_pooled.get("accuracy_full"),
            "mean_handoff_recall": real_pooled.get("mean_handoff_recall"),
            "mean_handoff_precision":
                real_pooled.get("mean_handoff_precision"),
            "mean_prompt_tokens": real_pooled.get("mean_prompt_tokens"),
            "failure_hist": real_pooled.get("failure_hist"),
        },
        "pooled_replay": {
            "config": pooled_cfg.as_dict(),
            "accuracy_full": pooled_acc,
            "mean_handoff_recall":
                pooled_replay.get("mean_handoff_recall"),
            "mean_handoff_precision":
                pooled_replay.get("mean_handoff_precision"),
            "mean_prompt_tokens":
                pooled_replay.get("mean_prompt_tokens"),
            "failure_hist": pooled_replay.get("failure_hist"),
        },
        "per_role_replay": {
            "config": per_role_cfg.as_dict(),
            "accuracy_full": per_role_acc,
            "mean_handoff_recall":
                per_role_replay.get("mean_handoff_recall"),
            "mean_handoff_precision":
                per_role_replay.get("mean_handoff_precision"),
            "mean_prompt_tokens":
                per_role_replay.get("mean_prompt_tokens"),
            "failure_hist": per_role_replay.get("failure_hist"),
        },
        "replay_gap": {
            "real_vs_pooled": round(real_acc - pooled_acc, 4),
            "real_vs_per_role": round(real_acc - per_role_acc, 4),
            "pooled_vs_per_role": round(pooled_acc - per_role_acc, 4),
        },
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mock",
                     choices=["mock", "real"])
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--domains", nargs="+",
                     default=["incident", "compliance", "security"])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                     default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[34])
    ap.add_argument("--sweep-path",
                     default="vision_mvp/results_phase32_noise_sweep.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep_pooled = None
    if args.sweep_path and os.path.exists(args.sweep_path):
        with open(args.sweep_path) as f:
            sweep_pooled = json.load(f).get("pooled", {})
        print(f"[phase34/A] loaded {len(sweep_pooled)} sweep points "
              f"from {args.sweep_path}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    for domain in args.domains:
        if domain not in DOMAIN_ADAPTERS:
            raise SystemExit(f"unknown domain {domain!r}")
        adapter = DOMAIN_ADAPTERS[domain]()
        print(f"\n[phase34/A] domain={domain} mode={args.mode}",
              flush=True)
        for k in args.distractor_counts:
            for seed in args.seeds:
                row = run_one_domain(adapter, args.mode, args.model,
                                      k, seed, sweep_pooled)
                rows.append(row)
                het = row["summary"]["heterogeneity"]
                limiting = row["summary"].get("role_limited_by")
                gap = row["replay_gap"]
                print(f"  k={k} seed={seed}  "
                      f"pooled_drop={row['summary']['pooled']['rates']['drop_rate']}  "
                      f"max_spread={het['max_spread_any_axis']}  "
                      f"limiting_role={limiting}  "
                      f"real_vs_pooled={gap['real_vs_pooled']:+.3f}  "
                      f"real_vs_per_role={gap['real_vs_per_role']:+.3f}",
                      flush=True)

    wall = time.time() - t0
    print(f"\n[phase34/A] wall = {wall:.1f}s", flush=True)

    payload = {
        "config": {
            "mode": args.mode, "model": args.model,
            "domains": args.domains,
            "distractor_counts": args.distractor_counts,
            "seeds": args.seeds,
            "sweep_path": args.sweep_path,
        },
        "rows": rows,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
