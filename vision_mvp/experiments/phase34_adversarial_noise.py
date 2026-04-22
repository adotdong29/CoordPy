"""Phase 34 Part B — adversarial extractor-noise sweep.

Phase 32's noise model (``core/extractor_noise.NoiseConfig``) injects
i.i.d. Bernoulli damage per emission. That is the right *synthetic*
stress test, but the production threat model is different: a flaky
LLM or a partial outage does not independently coin-flip each claim
— it silently drops specific load-bearing claims, or an attacker
selectively spoils the highest-priority role's output, or a prompt-
injection escalates a severity verdict.

Phase 34 adds ``adversarial_extractor`` (in the same module) which:

  * Drops *load-bearing* claims — those whose ``(role, kind)`` is in
    the scenario's gold causal chain — up to a budget. This is the
    Phase 34 direct analogue of "targeted drop of load-bearing claims"
    the ROADMAP has been asking for since Phase 32.
  * Silences specific roles (one-extractor-instance-out).
  * Injects severity escalations on max-ordinal decoders.

This experiment sweeps the adversary over all three non-code domains
and compares against an i.i.d. baseline at a *matched nominal noise
budget* (the i.i.d. drop_prob is set equal to the adversarial drop
budget divided by the per-scenario causal chain length).

Reproducible command:

    python3 -m vision_mvp.experiments.phase34_adversarial_noise \\
        --domains incident compliance security \\
        --out vision_mvp/results_phase34_adversarial_noise.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_noise import (
    AdversarialConfig, NoiseConfig,
    ADVERSARIAL_MODE_COMBINED, ADVERSARIAL_MODE_LOAD_BEARING_DROP,
    ADVERSARIAL_MODE_ROLE_SILENCING,
    ADVERSARIAL_MODE_SEVERITY_ESCALATION,
    adversarial_extractor, noisy_extractor,
)
from vision_mvp.experiments.phase33_llm_extractor import DOMAIN_ADAPTERS


def _causal_chain_length(adapter, bank) -> float:
    """Mean causal-chain length across scenarios (the R* proxy)."""
    lengths = []
    for s in bank:
        chain = adapter["gold_chain"](s)
        lengths.append(len(chain))
    return statistics.mean(lengths) if lengths else 0.0


def _run_substrate(adapter, extractor, bank, seed):
    mock_aud = adapter["mock_auditor"]()
    loop = adapter["run_loop"]
    rep = loop(bank, mock_aud, strategies=("substrate",),
                seed=seed, extractor=extractor)
    return rep.pooled()["substrate"]


def _severity_escalation_kinds(adapter) -> tuple[str, ...]:
    """Pick a high-severity kind per domain for the escalation pass."""
    name = adapter["name"]
    if name == "security":
        # MALWARE_DETECTED is ir_engineer's highest-severity kind.
        return ("MALWARE_DETECTED", "DATA_STAGING",
                "PERSISTENCE_INSTALLED")
    if name == "compliance":
        # Blocking-verdict kinds escalate approved → blocked.
        return ("DPA_MISSING", "ENCRYPTION_AT_REST_MISSING",
                "CROSS_BORDER_UNAUTHORIZED")
    # Incident — triggering a high-priority root cause kind.
    return ("DISK_FILL_CRITICAL", "OOM_KILL", "TLS_EXPIRED")


def _target_role_for_silencing(adapter) -> str:
    """Pick a single role to silence (the one whose silence is most
    damaging — i.e. owning the most claim kinds in the causal chain).
    Simple heuristic: pick the first producer role."""
    return adapter["producer_roles"][0]


def _priority_order_for_load_bearing(adapter) -> tuple[str, ...]:
    """Kinds we'd prefer the adversary drop first — higher priority
    first. Matches each domain's decoder priority."""
    name = adapter["name"]
    if name == "security":
        return (
            "MALWARE_DETECTED", "DATA_STAGING", "PERSISTENCE_INSTALLED",
            "REGULATED_DATA_EXPOSED", "PHISHING_DETECTED",
            "CROSS_TENANT_LEAK", "LATERAL_MOVEMENT",
            "TTP_ATTRIBUTED", "SUPPLY_CHAIN_IOC",
            "AUTH_SPIKE", "BRUTE_FORCE",
            "PII_AT_RISK", "IOC_KNOWN_BAD_IP",
        )
    if name == "compliance":
        return (
            "DPA_MISSING", "CROSS_BORDER_UNAUTHORIZED",
            "ENCRYPTION_AT_REST_MISSING", "PII_CATEGORY_UNDISCLOSED",
            "LIABILITY_CAP_MISSING",
            "AUTO_RENEWAL_UNFAVOURABLE", "TERMINATION_RESTRICTIVE",
            "PENTEST_STALE", "INCIDENT_SLA_INADEQUATE",
            "RETENTION_UNCAPPED", "BUDGET_THRESHOLD_BREACH",
            "PAYMENT_TERMS_AGGRESSIVE", "SSO_NOT_SUPPORTED",
        )
    # incident
    return (
        "DISK_FILL_CRITICAL", "OOM_KILL", "TLS_EXPIRED",
        "DNS_MISROUTE", "POOL_EXHAUSTION", "DEADLOCK_SUSPECTED",
        "SLOW_QUERY_OBSERVED", "CRON_OVERRUN",
        "ERROR_RATE_SPIKE", "LATENCY_SPIKE", "FW_BLOCK_SURGE",
    )


def run_one_domain(adapter, k: int, seed: int,
                    drop_budgets: list[int]) -> dict:
    bank = adapter["build_bank"](seed=seed,
                                   distractors_per_role=k)
    baseline = adapter["baseline_extractor"]
    mean_R_star = _causal_chain_length(adapter, bank)

    # Identity baseline (no noise).
    identity = _run_substrate(adapter, baseline, bank, seed)

    rows: list[dict] = []

    # 1. Load-bearing drop vs matched i.i.d. at multiple budgets.
    priority = _priority_order_for_load_bearing(adapter)
    for budget in drop_budgets:
        # Adversarial — targets load-bearing claims.
        adv_cfg = AdversarialConfig(
            target_mode=ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            drop_budget=budget,
            priority_order=priority, seed=seed)
        adv_ext = adversarial_extractor(
            baseline, adapter["known_kinds"], adv_cfg)
        adv_stats = _run_substrate(adapter, adv_ext, bank, seed)

        # Matched i.i.d. — drop_prob chosen so expected drops ≈ budget.
        # On a causal chain of length R*, i.i.d. drop_prob = budget/R*
        # produces the same *expected* number of load-bearing drops,
        # but spread over all roles rather than concentrated on the
        # highest-priority ones.
        matched_drop = min(1.0, budget / max(1.0, mean_R_star))
        iid_cfg = NoiseConfig(drop_prob=matched_drop, seed=seed)
        iid_ext = noisy_extractor(
            baseline, adapter["known_kinds"], iid_cfg)
        iid_stats = _run_substrate(adapter, iid_ext, bank, seed)

        rows.append({
            "experiment": "load_bearing_drop_vs_iid",
            "drop_budget": budget,
            "matched_iid_drop_prob": round(matched_drop, 4),
            "mean_R_star": round(mean_R_star, 4),
            "adversarial": {
                "config": adv_cfg.as_dict(),
                "accuracy_full": adv_stats.get("accuracy_full"),
                "mean_handoff_recall":
                    adv_stats.get("mean_handoff_recall"),
                "mean_handoff_precision":
                    adv_stats.get("mean_handoff_precision"),
                "mean_prompt_tokens":
                    adv_stats.get("mean_prompt_tokens"),
                "failure_hist": adv_stats.get("failure_hist"),
            },
            "iid_matched": {
                "config": iid_cfg.as_dict(),
                "accuracy_full": iid_stats.get("accuracy_full"),
                "mean_handoff_recall":
                    iid_stats.get("mean_handoff_recall"),
                "mean_handoff_precision":
                    iid_stats.get("mean_handoff_precision"),
                "mean_prompt_tokens":
                    iid_stats.get("mean_prompt_tokens"),
                "failure_hist": iid_stats.get("failure_hist"),
            },
            "accuracy_gap_iid_minus_adv": round(
                float(iid_stats.get("accuracy_full") or 0.0)
                - float(adv_stats.get("accuracy_full") or 0.0), 4),
        })

    # 2. Role silencing — one-extractor-outage.
    silent_role = _target_role_for_silencing(adapter)
    adv_cfg = AdversarialConfig(
        target_mode=ADVERSARIAL_MODE_ROLE_SILENCING,
        target_roles=(silent_role,), seed=seed)
    adv_ext = adversarial_extractor(
        baseline, adapter["known_kinds"], adv_cfg)
    adv_stats = _run_substrate(adapter, adv_ext, bank, seed)
    rows.append({
        "experiment": "role_silencing",
        "silenced_role": silent_role,
        "adversarial": {
            "config": adv_cfg.as_dict(),
            "accuracy_full": adv_stats.get("accuracy_full"),
            "mean_handoff_recall":
                adv_stats.get("mean_handoff_recall"),
            "mean_handoff_precision":
                adv_stats.get("mean_handoff_precision"),
            "mean_prompt_tokens":
                adv_stats.get("mean_prompt_tokens"),
            "failure_hist": adv_stats.get("failure_hist"),
        },
    })

    # 3. Severity escalation — useful on max-ordinal decoders; still
    # run on all domains to keep the sweep table stable, but expect
    # non-trivial effect on security only.
    escal_kinds = _severity_escalation_kinds(adapter)
    adv_cfg = AdversarialConfig(
        target_mode=ADVERSARIAL_MODE_SEVERITY_ESCALATION,
        escalation_kinds=escal_kinds, seed=seed)
    adv_ext = adversarial_extractor(
        baseline, adapter["known_kinds"], adv_cfg)
    adv_stats = _run_substrate(adapter, adv_ext, bank, seed)
    rows.append({
        "experiment": "severity_escalation",
        "escalation_kinds": list(escal_kinds),
        "adversarial": {
            "config": adv_cfg.as_dict(),
            "accuracy_full": adv_stats.get("accuracy_full"),
            "mean_handoff_recall":
                adv_stats.get("mean_handoff_recall"),
            "mean_handoff_precision":
                adv_stats.get("mean_handoff_precision"),
            "mean_prompt_tokens":
                adv_stats.get("mean_prompt_tokens"),
            "failure_hist": adv_stats.get("failure_hist"),
        },
    })

    return {
        "domain": adapter["name"],
        "k": k, "seed": seed,
        "mean_R_star": round(mean_R_star, 4),
        "identity": {
            "accuracy_full": identity.get("accuracy_full"),
            "mean_handoff_recall": identity.get("mean_handoff_recall"),
            "mean_handoff_precision":
                identity.get("mean_handoff_precision"),
            "mean_prompt_tokens": identity.get("mean_prompt_tokens"),
            "failure_hist": identity.get("failure_hist"),
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                     default=["incident", "compliance", "security"])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                     default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[34, 35])
    ap.add_argument("--drop-budgets", nargs="+", type=int,
                     default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    domain_rows: list[dict] = []
    for domain in args.domains:
        if domain not in DOMAIN_ADAPTERS:
            raise SystemExit(f"unknown domain {domain!r}")
        adapter = DOMAIN_ADAPTERS[domain]()
        print(f"\n[phase34/B] domain={domain}", flush=True)
        for k in args.distractor_counts:
            for seed in args.seeds:
                row = run_one_domain(adapter, k, seed,
                                      args.drop_budgets)
                domain_rows.append(row)
                for r in row["rows"]:
                    if r["experiment"] == "load_bearing_drop_vs_iid":
                        adv_acc = r["adversarial"]["accuracy_full"]
                        iid_acc = r["iid_matched"]["accuracy_full"]
                        print(f"  seed={seed}  budget={r['drop_budget']}  "
                              f"matched_iid_drop={r['matched_iid_drop_prob']}  "
                              f"adv_acc={adv_acc:.2f}  "
                              f"iid_acc={iid_acc:.2f}  "
                              f"gap_iid-adv={r['accuracy_gap_iid_minus_adv']:+.2f}",
                              flush=True)
                    elif r["experiment"] == "role_silencing":
                        print(f"  seed={seed}  silenced={r['silenced_role']}  "
                              f"adv_acc={r['adversarial']['accuracy_full']:.2f}",
                              flush=True)
                    elif r["experiment"] == "severity_escalation":
                        print(f"  seed={seed}  escalate "
                              f"adv_acc="
                              f"{r['adversarial']['accuracy_full']:.2f}",
                              flush=True)

    wall = time.time() - t0
    print(f"\n[phase34/B] wall = {wall:.1f}s", flush=True)

    # Pool across seeds/k per (domain, experiment, budget) for the
    # headline table.
    pooled: dict[str, dict] = {}
    for dr in domain_rows:
        for r in dr["rows"]:
            if r["experiment"] == "load_bearing_drop_vs_iid":
                key = (f"{dr['domain']}_drop_budget_{r['drop_budget']}")
                slot = pooled.setdefault(key, {
                    "domain": dr["domain"],
                    "experiment": r["experiment"],
                    "drop_budget": r["drop_budget"],
                    "n": 0, "adv_acc_sum": 0.0,
                    "iid_acc_sum": 0.0,
                    "iid_matched_drop": r["matched_iid_drop_prob"],
                    "mean_R_star": dr["mean_R_star"],
                })
                slot["n"] += 1
                slot["adv_acc_sum"] += float(
                    r["adversarial"]["accuracy_full"] or 0.0)
                slot["iid_acc_sum"] += float(
                    r["iid_matched"]["accuracy_full"] or 0.0)
            elif r["experiment"] == "role_silencing":
                key = f"{dr['domain']}_silence_{r['silenced_role']}"
                slot = pooled.setdefault(key, {
                    "domain": dr["domain"],
                    "experiment": r["experiment"],
                    "silenced_role": r["silenced_role"],
                    "n": 0, "adv_acc_sum": 0.0,
                })
                slot["n"] += 1
                slot["adv_acc_sum"] += float(
                    r["adversarial"]["accuracy_full"] or 0.0)
            elif r["experiment"] == "severity_escalation":
                key = f"{dr['domain']}_severity_escalation"
                slot = pooled.setdefault(key, {
                    "domain": dr["domain"],
                    "experiment": r["experiment"],
                    "n": 0, "adv_acc_sum": 0.0,
                })
                slot["n"] += 1
                slot["adv_acc_sum"] += float(
                    r["adversarial"]["accuracy_full"] or 0.0)
    for slot in pooled.values():
        n = max(1, slot["n"])
        slot["adv_acc_mean"] = round(slot["adv_acc_sum"] / n, 4)
        slot.pop("adv_acc_sum", None)
        if "iid_acc_sum" in slot:
            slot["iid_acc_mean"] = round(slot["iid_acc_sum"] / n, 4)
            slot.pop("iid_acc_sum", None)
            slot["accuracy_gap_iid_minus_adv"] = round(
                slot["iid_acc_mean"] - slot["adv_acc_mean"], 4)

    print()
    print("=" * 100)
    print("PHASE 34 / PART B POOLED — adversarial vs i.i.d. at matched budget")
    print("=" * 100)
    for key in sorted(pooled.keys()):
        s = pooled[key]
        exp = s.get("experiment", "?")
        if exp == "load_bearing_drop_vs_iid":
            print(f"  {s['domain']:>10}  budget={s['drop_budget']}  "
                  f"matched_drop={s['iid_matched_drop']:.2f}  "
                  f"adv={s['adv_acc_mean']:.2f}  "
                  f"iid={s['iid_acc_mean']:.2f}  "
                  f"gap(iid-adv)={s['accuracy_gap_iid_minus_adv']:+.2f}")
        elif exp == "role_silencing":
            print(f"  {s['domain']:>10}  silence {s['silenced_role']}  "
                  f"adv={s['adv_acc_mean']:.2f}")
        elif exp == "severity_escalation":
            print(f"  {s['domain']:>10}  escalate  "
                  f"adv={s['adv_acc_mean']:.2f}")

    payload = {
        "config": {
            "domains": args.domains,
            "distractor_counts": args.distractor_counts,
            "seeds": args.seeds,
            "drop_budgets": args.drop_budgets,
        },
        "rows": domain_rows,
        "pooled": pooled,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
