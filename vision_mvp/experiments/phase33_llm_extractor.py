"""Phase 33 Part A + B — LLM-driven extractor benchmark + real-noise
calibration.

Runs the typed-handoff substrate with an *LLM-driven* extractor
(``core/llm_extractor``) on all three non-code domains, measures the
empirical noise profile (δ̂ drop-rate, ε̂ spurious-per-event, μ̂
mislabel-rate, π̂ payload-corrupt-rate), and compares the measured
profile against the Phase-32 synthetic noise sweep
(``results_phase32_noise_sweep.json``) to answer:

  * **Part A** — does the substrate's bounded-context +
    correctness-preservation pattern survive when the extractor is
    a real LLM (not a perfect regex) ?
  * **Part B** — does the Phase-32 i.i.d. Bernoulli noise model
    approximate the LLM extractor's empirical degradation curve?

Both answers are *per-domain*. The experiment emits one JSON artifact
covering all three domains. The same harness can run in three modes:

  1. ``--mode mock`` — a deterministic mock LLM (no network I/O),
     useful for tests and CI. Produces an audit that the Phase-32
     sweep should trivially match (near-zero noise).
  2. ``--mode mock-noisy`` — a deterministic mock LLM with configured
     drop / spurious noise, used to exercise the calibration pipeline
     at controlled noise levels without paying for real LLM calls.
  3. ``--mode real`` — a real Ollama call per (role, scenario). This
     is the Phase-33 headline experiment; it costs ~50-200 LLM calls
     depending on the number of domains × roles × scenarios × seeds.

Reproducible commands:

    # Deterministic mock — no network, seconds of wall.
    python3 -m vision_mvp.experiments.phase33_llm_extractor --mode mock \\
        --out vision_mvp/results_phase33_llm_extractor_mock.json

    # Real Ollama, both domains, 1 seed.
    python3 -m vision_mvp.experiments.phase33_llm_extractor --mode real \\
        --model qwen2.5-coder:7b \\
        --sweep-path vision_mvp/results_phase32_noise_sweep.json \\
        --out vision_mvp/results_phase33_llm_extractor_7b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_calibration import (
    ExtractorAudit, calibrate_extractor, closest_synthetic_config,
    compare_to_synthetic_curve,
)
from vision_mvp.core.extractor_noise import (
    compliance_review_known_kinds, incident_triage_known_kinds,
)
from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.llm_extractor import (
    DeterministicCache, DeterministicMockExtractorLLM, LLMExtractor,
    LLMExtractorConfig,
)


# =============================================================================
# Domain adapters — how to plug each non-code domain into the
# calibration harness. The adapter carries everything the extractor
# and calibrator need: known-kinds map, scenario builder, role list,
# event-getter, causal-ids getter, gold-chain getter, and a handoff-
# protocol runner that injects a custom extractor.
# =============================================================================


def _incident_adapter():
    from vision_mvp.tasks.incident_triage import (
        MockIncidentAuditor, ROLE_AUDITOR, ROLE_MONITOR, ROLE_DB_ADMIN,
        ROLE_NETWORK, ROLE_SYSADMIN, build_scenario_bank,
        extract_claims_for_role, run_incident_loop,
    )
    return {
        "name": "incident",
        "known_kinds": incident_triage_known_kinds(),
        "build_bank": build_scenario_bank,
        "producer_roles": (ROLE_MONITOR, ROLE_DB_ADMIN,
                             ROLE_SYSADMIN, ROLE_NETWORK),
        "aggregator_role": ROLE_AUDITOR,
        "role_events": lambda s, r: list(
            s.per_role_events.get(r, ())),
        "causal_ids": lambda s, r: [ev.event_id
                                      for ev in s.per_role_events.get(
                                          r, ())
                                      if ev.is_causal],
        "gold_chain": lambda s: s.causal_chain,
        "mock_auditor": MockIncidentAuditor,
        "run_loop": run_incident_loop,
        "baseline_extractor": extract_claims_for_role,
        "keyword_map": {
            # monitor
            "error_rate=0.": "ERROR_RATE_SPIKE",
            "uptime_pct=4": "ERROR_RATE_SPIKE",
            "p95_ms=": "LATENCY_SPIKE",
            # db_admin
            "mean_ms=4": "SLOW_QUERY_OBSERVED",
            "active=200/200": "POOL_EXHAUSTION",
            "reconnect_attempts=": "POOL_EXHAUSTION",
            "deadlock": "DEADLOCK_SUSPECTED",
            "space left on device": "SLOW_QUERY_OBSERVED",
            # sysadmin
            "used=99": "DISK_FILL_CRITICAL",
            "exit=137": "CRON_OVERRUN",
            "duration_s=5400": "CRON_OVERRUN",
            "oom_kill": "OOM_KILL",
            # network
            "reason=expired": "TLS_EXPIRED",
            "action=drop": "FW_BLOCK_SURGE",
            "servfail": "DNS_MISROUTE",
        },
    }


def _compliance_adapter():
    from vision_mvp.tasks.compliance_review import (
        MockComplianceAuditor, ROLE_COMPLIANCE, ROLE_FINANCE,
        ROLE_LEGAL, ROLE_PRIVACY, ROLE_SECURITY,
        build_scenario_bank, extract_claims_for_role,
        run_compliance_loop,
    )
    return {
        "name": "compliance",
        "known_kinds": compliance_review_known_kinds(),
        "build_bank": build_scenario_bank,
        "producer_roles": (ROLE_LEGAL, ROLE_SECURITY,
                             ROLE_PRIVACY, ROLE_FINANCE),
        "aggregator_role": ROLE_COMPLIANCE,
        "role_events": lambda s, r: list(
            s.per_role_docs.get(r, ())),
        "causal_ids": lambda s, r: [d.doc_id
                                      for d in s.per_role_docs.get(
                                          r, ())
                                      if d.is_causal],
        "gold_chain": lambda s: s.causal_chain,
        "mock_auditor": MockComplianceAuditor,
        "run_loop": run_compliance_loop,
        "baseline_extractor": extract_claims_for_role,
        "keyword_map": {
            # legal
            "liability clause limits=none": "LIABILITY_CAP_MISSING",
            "liability limits=none": "LIABILITY_CAP_MISSING",
            "auto_renewal term=36mo": "AUTO_RENEWAL_UNFAVOURABLE",
            "termination notice=": "TERMINATION_RESTRICTIVE",
            # security
            "encryption_at_rest=no": "ENCRYPTION_AT_REST_MISSING",
            "sso=no": "SSO_NOT_SUPPORTED",
            "pentest_report vintage_days=9": "PENTEST_STALE",
            "incident_sla_hours=7": "INCIDENT_SLA_INADEQUATE",
            # privacy
            "data_processing_agreement=missing": "DPA_MISSING",
            "cross_border_transfer=yes": "CROSS_BORDER_UNAUTHORIZED",
            "retention=unbounded": "RETENTION_UNCAPPED",
            "pii_category=": "PII_CATEGORY_UNDISCLOSED",
            # finance
            "proposed_spend amount_usd=42": "BUDGET_THRESHOLD_BREACH",
            "net_days=7 prepay_pct=50": "PAYMENT_TERMS_AGGRESSIVE",
        },
    }


def _security_adapter():
    from vision_mvp.tasks.security_escalation import (
        MockSecurityAuditor, ROLE_CISO, ROLE_SOC_ANALYST,
        ROLE_IR_ENGINEER, ROLE_THREAT_INTEL, ROLE_DATA_STEWARD,
        build_scenario_bank, extract_claims_for_role,
        run_security_loop, CLAIM_AUTH_SPIKE,
        CLAIM_PHISHING_DETECTED, CLAIM_LATERAL_MOVEMENT,
        CLAIM_BRUTE_FORCE, CLAIM_PERSISTENCE_INSTALLED,
        CLAIM_DATA_STAGING, CLAIM_MALWARE_DETECTED,
        CLAIM_PRIV_ESCALATION,
        CLAIM_IOC_KNOWN_BAD_IP, CLAIM_IOC_MALICIOUS_DOMAIN,
        CLAIM_TTP_ATTRIBUTED, CLAIM_SUPPLY_CHAIN_IOC,
        CLAIM_REGULATED_DATA_EXPOSED, CLAIM_PII_AT_RISK,
        CLAIM_CROSS_TENANT_LEAK,
    )
    return {
        "name": "security",
        "known_kinds": {
            ROLE_SOC_ANALYST: (CLAIM_AUTH_SPIKE,
                                CLAIM_PHISHING_DETECTED,
                                CLAIM_LATERAL_MOVEMENT,
                                CLAIM_BRUTE_FORCE),
            ROLE_IR_ENGINEER: (CLAIM_PERSISTENCE_INSTALLED,
                                CLAIM_DATA_STAGING,
                                CLAIM_MALWARE_DETECTED,
                                CLAIM_PRIV_ESCALATION),
            ROLE_THREAT_INTEL: (CLAIM_IOC_KNOWN_BAD_IP,
                                 CLAIM_IOC_MALICIOUS_DOMAIN,
                                 CLAIM_TTP_ATTRIBUTED,
                                 CLAIM_SUPPLY_CHAIN_IOC),
            ROLE_DATA_STEWARD: (CLAIM_REGULATED_DATA_EXPOSED,
                                 CLAIM_PII_AT_RISK,
                                 CLAIM_CROSS_TENANT_LEAK),
        },
        "build_bank": build_scenario_bank,
        "producer_roles": (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER,
                             ROLE_THREAT_INTEL, ROLE_DATA_STEWARD),
        "aggregator_role": ROLE_CISO,
        "role_events": lambda s, r: list(
            s.per_role_events.get(r, ())),
        "causal_ids": lambda s, r: [ev.event_id
                                      for ev in s.per_role_events.get(
                                          r, ())
                                      if ev.is_causal],
        "gold_chain": lambda s: s.causal_chain,
        "mock_auditor": MockSecurityAuditor,
        "run_loop": run_security_loop,
        "baseline_extractor": extract_claims_for_role,
        "keyword_map": {
            # SOC
            "credential_harvest": "PHISHING_DETECTED",
            "rule=phishing": "PHISHING_DETECTED",
            "login_burst": "AUTH_SPIKE",
            "off_hours_access": "AUTH_SPIKE",
            "brute_force": "BRUTE_FORCE",
            "smb_share_enum": "LATERAL_MOVEMENT",
            "abnormal_outbound": "LATERAL_MOVEMENT",
            # IR
            "persistence registry_key=hk": "PERSISTENCE_INSTALLED",
            "malware_detection": "MALWARE_DETECTED",
            "/tmp/exfil": "DATA_STAGING",
            "rsync ": "DATA_STAGING",
            # Threat intel
            "type=supply_chain": "SUPPLY_CHAIN_IOC",
            "type=hash hit=known-bad": "TTP_ATTRIBUTED",
            "ioc=ip:": "IOC_KNOWN_BAD_IP",
            "ioc=domain:": "IOC_MALICIOUS_DOMAIN",
            # Data steward
            "class=regulated_pii": "REGULATED_DATA_EXPOSED",
            "regulation=": "PII_AT_RISK",
            "class=cross_tenant": "CROSS_TENANT_LEAK",
        },
    }


DOMAIN_ADAPTERS = {
    "incident": _incident_adapter,
    "compliance": _compliance_adapter,
    "security": _security_adapter,
}


# =============================================================================
# Build a domain-specific LLM extractor
# =============================================================================


def _build_llm_extractor(adapter: dict, mode: str,
                          model: str, noise_cfg: dict | None = None,
                          ) -> LLMExtractor:
    cache = DeterministicCache(model=f"{mode}:{model}")
    cfg = LLMExtractorConfig()
    if mode == "real":
        client = LLMClient(model=model, timeout=300.0)

        def _call(prompt: str) -> str:
            return client.generate(
                prompt, max_tokens=cfg.max_tokens,
                temperature=cfg.temperature)
        extractor = LLMExtractor(
            llm_call=_call,
            known_kinds_by_role=adapter["known_kinds"],
            config=cfg, cache=cache)
        extractor._client_stats = client.stats  # type: ignore
        return extractor
    elif mode == "mock":
        mock = DeterministicMockExtractorLLM(
            keyword_to_kind=adapter["keyword_map"])
        return LLMExtractor(
            llm_call=mock,
            known_kinds_by_role=adapter["known_kinds"],
            config=cfg, cache=cache)
    elif mode == "mock-noisy":
        drop_prob = (noise_cfg or {}).get("drop_prob", 0.15)
        spurious_kind = (noise_cfg or {}).get("spurious_kind")
        mock = DeterministicMockExtractorLLM(
            keyword_to_kind=adapter["keyword_map"],
            drop_prob=drop_prob,
            spurious_body="noisy spurious claim",
            spurious_kind=spurious_kind)
        return LLMExtractor(
            llm_call=mock,
            known_kinds_by_role=adapter["known_kinds"],
            config=cfg, cache=cache)
    raise ValueError(f"unknown mode {mode!r}")


# =============================================================================
# Domain runner — calibration + substrate accuracy + compare to sweep
# =============================================================================


def run_one_domain(adapter: dict, mode: str, model: str,
                    distractors: int, seed: int,
                    sweep_pooled: dict | None,
                    noise_cfg: dict | None = None,
                    ) -> dict:
    bank = adapter["build_bank"](seed=seed,
                                   distractors_per_role=distractors)
    extractor = _build_llm_extractor(adapter, mode, model, noise_cfg)

    # Calibration — per-role / per-scenario emissions vs gold chain.
    audit = calibrate_extractor(
        extractor, bank,
        adapter["role_events"],
        adapter["causal_ids"],
        adapter["gold_chain"],
        roles_to_probe=list(adapter["producer_roles"]),
        extractor_label=f"{mode}:{model}")

    # Substrate accuracy under the *same* LLM extractor.
    mock_aud = adapter["mock_auditor"]()
    loop = adapter["run_loop"]
    rep = loop(bank, mock_aud,
                strategies=("substrate",),
                seed=seed,
                extractor=extractor)
    pooled = rep.pooled()["substrate"]
    measured_acc = pooled["accuracy_full"]
    measured_recall = pooled["mean_handoff_recall"]
    measured_prec = pooled.get("mean_handoff_precision", 1.0)
    measured_tokens = pooled["mean_prompt_tokens"]
    failure_hist = pooled["failure_hist"]

    match_report = None
    if sweep_pooled is not None:
        match_report = compare_to_synthetic_curve(
            audit, sweep_pooled,
            real_measured_accuracy=measured_acc,
            real_measured_recall=measured_recall,
            real_measured_precision=measured_prec,
            domain="incident" if adapter["name"] == "incident"
            else "compliance")
        # Security maps to compliance regime (strict decoder on flags);
        # no first-class security sweep yet.

    return {
        "domain": adapter["name"],
        "mode": mode,
        "model": model,
        "distractors_per_role": distractors,
        "seed": seed,
        "audit": audit.as_dict(),
        "substrate_measured": {
            "accuracy_full": measured_acc,
            "mean_handoff_recall": measured_recall,
            "mean_handoff_precision": measured_prec,
            "mean_prompt_tokens": measured_tokens,
            "failure_hist": failure_hist,
        },
        "extractor_stats": extractor.stats.as_dict(),
        "synthetic_comparison": match_report,
    }


# =============================================================================
# Driver
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mock",
                     choices=["mock", "mock-noisy", "real"])
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--domains", nargs="+",
                     default=["incident", "compliance", "security"])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                     default=[20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[33])
    ap.add_argument("--sweep-path", default=None,
                     help="Path to results_phase32_noise_sweep.json "
                          "for real-vs-synthetic comparison.")
    ap.add_argument("--noise-drop", type=float, default=0.15,
                     help="For mock-noisy mode: per-claim drop prob.")
    ap.add_argument("--noise-spurious-kind", default=None,
                     help="For mock-noisy mode: spurious kind "
                          "(must be outside known-kinds to be filtered).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep_pooled = None
    if args.sweep_path and os.path.exists(args.sweep_path):
        with open(args.sweep_path) as f:
            sweep = json.load(f)
        sweep_pooled = sweep.get("pooled", {})
        print(f"[phase33] loaded {len(sweep_pooled)} sweep points from "
              f"{args.sweep_path}", flush=True)

    overall_start = time.time()
    rows: list[dict] = []
    for domain in args.domains:
        if domain not in DOMAIN_ADAPTERS:
            raise SystemExit(f"unknown domain {domain!r}")
        adapter = DOMAIN_ADAPTERS[domain]()
        print(f"\n[phase33] domain={domain} mode={args.mode} "
              f"model={args.model}", flush=True)
        for k in args.distractor_counts:
            for seed in args.seeds:
                row = run_one_domain(
                    adapter, args.mode, args.model, k, seed,
                    sweep_pooled,
                    noise_cfg={
                        "drop_prob": args.noise_drop,
                        "spurious_kind": args.noise_spurious_kind,
                    })
                audit = row["audit"]
                sm = row["substrate_measured"]
                print(f"  k={k} seed={seed}  δ̂={audit['drop_rate']}  "
                      f"μ̂={audit['mislabel_rate']}  "
                      f"ε̂={audit['spurious_per_event']}  "
                      f"π̂={audit['payload_corrupt_rate']}   "
                      f"acc={sm['accuracy_full']}  "
                      f"rec={sm['mean_handoff_recall']}  "
                      f"prec={sm['mean_handoff_precision']:.3f}  "
                      f"tok={sm['mean_prompt_tokens']:.0f}",
                      flush=True)
                if row.get("synthetic_comparison"):
                    sc = row["synthetic_comparison"]
                    if sc.get("verdict"):
                        print(f"     synthetic verdict={sc['verdict']}  "
                              f"residual={sc.get('residual')}",
                              flush=True)
                rows.append(row)

    overall = time.time() - overall_start
    print(f"\n[phase33] overall wall = {overall:.1f}s", flush=True)

    payload = {
        "config": {
            "mode": args.mode, "model": args.model,
            "domains": args.domains,
            "distractor_counts": args.distractor_counts,
            "seeds": args.seeds,
            "sweep_path": args.sweep_path,
            "noise_drop": args.noise_drop,
            "noise_spurious_kind": args.noise_spurious_kind,
        },
        "rows": rows,
        "wall_seconds": round(overall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
