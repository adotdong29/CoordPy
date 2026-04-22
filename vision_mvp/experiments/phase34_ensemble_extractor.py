"""Phase 34 Part C — first meaningful regex + LLM ensemble result.

Phase 33 Conjecture C33-4 stated the informal union-bound on two
noisy extractors: ``δ_u ≤ δ_r · δ_l``, ``ε_u ≤ ε_r + ε_l``. Phase 33
couldn't *measure* it because the existing benchmarks all have regex
recall = 1 on the causal events by construction — the LLM output was
strictly dominated by the regex output and the ensemble was never
worth building.

Phase 34 Part C constructs the smallest *honest* test of the union
bound: a compliance scenario bank where half of the scenarios have
the Phase-32 canonical witness (regex catches, LLM narrative triggers
miss) and half have a free-form narrative witness (regex cannot parse
it; the narrative-keyword LLM extractor catches). A scenario's
causal chain is *either* canonical *or* narrative — never both —
so each extractor has coverage < 1 on the pooled bank and the
ensemble is the only path to coverage = 1.

Under this shape:

  * regex-only has ``recall ≤ 0.5`` (misses every narrative scenario).
  * LLM-narrative alone has ``recall ≤ 0.5`` (misses every canonical
    scenario — narrative triggers don't match canonical phrasings).
  * the union should approach ``recall ≈ 1`` without inflating
    spurious ε beyond the sum ``ε_r + ε_l`` (Conjecture C33-4).

Reproducible command:

    python3 -m vision_mvp.experiments.phase34_ensemble_extractor \\
        --seeds 34 35 --distractor-counts 6 20 \\
        --out vision_mvp/results_phase34_ensemble_extractor.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.ensemble_extractor import union_of
from vision_mvp.core.extractor_calibration import calibrate_extractor
from vision_mvp.core.extractor_noise import (
    compliance_review_known_kinds,
)
from vision_mvp.tasks.compliance_review import (
    ALL_ROLES, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
    CLAIM_CROSS_BORDER_UNAUTHORIZED, CLAIM_DPA_MISSING,
    CLAIM_ENCRYPTION_AT_REST_MISSING,
    CLAIM_LIABILITY_CAP_MISSING, CLAIM_PAYMENT_TERMS_AGGRESSIVE,
    DOC_CONTRACT_CLAUSE, DOC_PRIVACY_INVENTORY,
    DOC_SECURITY_QA, DOC_FINANCE_LINEITEM,
    MockComplianceAuditor, ROLE_LEGAL, ROLE_PRIVACY,
    ROLE_SECURITY, ROLE_FINANCE,
    STRATEGY_SUBSTRATE, VendorDoc, VendorScenario,
    build_scenario_bank,
    extract_claims_for_role as regex_extract_claims_for_role,
    run_compliance_loop,
)


# =============================================================================
# Narrative-only scenario construction — regex genuinely can't see
# these causal docs. The narrative phrasings do not contain any of the
# Phase-32 regex's canonical markers (``liability.*limits=none``,
# ``data_processing_agreement=missing``, ``encryption_at_rest=no``,
# ``cross_border_transfer=yes.*sccs=no``, etc.).
# =============================================================================


# Narrative causal replacement per scenario: (doc_type, role,
# narrative body, kind emitted on match). The narrative has NO
# regex-triggering substring; it conveys the same semantic fact.
_NARRATIVE_REPLACEMENT: dict[str, tuple[str, str, str, str]] = {
    "missing_dpa": (
        DOC_PRIVACY_INVENTORY, ROLE_PRIVACY,
        ("there is currently no signed data handling contract in "
         "place for this vendor and procurement of one has been "
         "stalled for three quarters running"),
        CLAIM_DPA_MISSING),
    "uncapped_liability": (
        DOC_CONTRACT_CLAUSE, ROLE_LEGAL,
        ("the vendor shall not be held responsible for any category "
         "of damages of any kind or amount and no ceiling of any "
         "sort is named anywhere in the master services agreement"),
        CLAIM_LIABILITY_CAP_MISSING),
    "weak_encryption": (
        DOC_SECURITY_QA, ROLE_SECURITY,
        ("the vendor stores all tenant data on disk in its original "
         "cleartext form and applies no protection of any kind while "
         "the data is at rest"),
        CLAIM_ENCRYPTION_AT_REST_MISSING),
    "cross_border_transfer_unauthorized": (
        DOC_PRIVACY_INVENTORY, ROLE_PRIVACY,
        ("personally identifiable information moves on a daily basis "
         "from our european facility to a location in a country the "
         "regulators consider a third country and we have no "
         "standard contract provisions covering that move"),
        CLAIM_CROSS_BORDER_UNAUTHORIZED),
    "budget_threshold_breach": (
        DOC_FINANCE_LINEITEM, ROLE_FINANCE,
        ("the payment schedule calls for one half of the total "
         "contract value to be prepaid within seven calendar days of "
         "signature which we view as aggressive for a newly "
         "onboarded vendor"),
        CLAIM_PAYMENT_TERMS_AGGRESSIVE),
}


def _narrative_replacement_for(scenario: VendorScenario
                                 ) -> VendorScenario:
    """Replace the canonical causal doc of scenario's first chain entry
    with a narrative doc; drop the original causal doc (so the regex
    cannot see it). Keeps the *other* causal doc (if any) canonical so
    regex still has something to anchor on — the second required claim
    may still be canonical.

    The result is a scenario where at least one load-bearing claim
    requires narrative extraction to recover.
    """
    if scenario.scenario_id not in _NARRATIVE_REPLACEMENT:
        return scenario
    (doc_type, role, body, kind) = _NARRATIVE_REPLACEMENT[
        scenario.scenario_id]
    # Identify the original canonical doc for this (role, kind) in
    # the causal chain; its evids point to it.
    replaced_chain_entry = None
    for ent in scenario.causal_chain:
        if ent[0] == role and ent[1] == kind:
            replaced_chain_entry = ent
            break
    if replaced_chain_entry is None:
        return scenario
    replaced_doc_id = replaced_chain_entry[3][0]

    # Build the narrative doc with the same doc_id so the scenario's
    # other chain entries / evids stay consistent; the narrative swap
    # is a *content* swap on the existing doc id.
    existing_docs = list(scenario.per_role_docs.get(role, ()))
    new_docs: list[VendorDoc] = []
    for d in existing_docs:
        if d.doc_id == replaced_doc_id:
            new_docs.append(VendorDoc(
                doc_id=replaced_doc_id, doc_type=doc_type,
                origin_role=role, body=body,
                tags=("narrative",), is_causal=True))
        else:
            new_docs.append(d)
    new_role_docs = dict(scenario.per_role_docs)
    new_role_docs[role] = tuple(new_docs)
    new_chain = tuple(
        (r, k, (body if (r == role and k == kind) else p), evs)
        for (r, k, p, evs) in scenario.causal_chain
    )
    return replace(scenario, per_role_docs=new_role_docs,
                    causal_chain=new_chain,
                    scenario_id=scenario.scenario_id + "_narrative")


def build_mixed_bank(seed: int, distractors_per_role: int
                      ) -> list[VendorScenario]:
    """A bank with the Phase-32 canonical scenarios PLUS narrative-
    replacement variants. The narrative half has the first causal doc
    replaced with a narrative phrasing the regex cannot parse; the
    canonical half is unchanged. The result is 10 scenarios (5 canon
    + 5 narrative)."""
    canonical = build_scenario_bank(
        seed=seed, distractors_per_role=distractors_per_role)
    narrative = [_narrative_replacement_for(s) for s in canonical]
    return canonical + narrative


# =============================================================================
# LLM-style narrative extractor — keyword-match on narrative phrasings.
# The triggers are chosen so they do NOT appear in canonical docs (so
# LLM alone misses canonical scenarios). This is the *complementary*
# layer for the ensemble.
# =============================================================================


_LLM_NARRATIVE_TRIGGERS: tuple[tuple[str, str, str], ...] = (
    (ROLE_PRIVACY, "no signed data handling contract",
     CLAIM_DPA_MISSING),
    (ROLE_PRIVACY, "procurement of one has been stalled",
     CLAIM_DPA_MISSING),
    (ROLE_LEGAL, "not be held responsible",
     CLAIM_LIABILITY_CAP_MISSING),
    (ROLE_LEGAL, "no ceiling of any", CLAIM_LIABILITY_CAP_MISSING),
    (ROLE_SECURITY, "cleartext form",
     CLAIM_ENCRYPTION_AT_REST_MISSING),
    (ROLE_SECURITY, "applies no protection",
     CLAIM_ENCRYPTION_AT_REST_MISSING),
    (ROLE_PRIVACY, "considers a third country",
     CLAIM_CROSS_BORDER_UNAUTHORIZED),
    (ROLE_PRIVACY, "no standard contract provisions",
     CLAIM_CROSS_BORDER_UNAUTHORIZED),
    (ROLE_FINANCE, "to be prepaid within seven calendar days",
     CLAIM_PAYMENT_TERMS_AGGRESSIVE),
    (ROLE_FINANCE, "aggressive for a newly",
     CLAIM_PAYMENT_TERMS_AGGRESSIVE),
)


def llm_narrative_extractor(role, docs, scenario):
    out: list[tuple[str, str, tuple[int, ...]]] = []
    body_triggers = [(trig, kind)
                      for (r, trig, kind) in _LLM_NARRATIVE_TRIGGERS
                      if r == role]
    if not body_triggers:
        return out
    for d in docs:
        body_lc = (d.body or "").lower()
        for (trig, kind) in body_triggers:
            if trig in body_lc:
                out.append((kind, d.body, (d.doc_id,)))
                break
    return out


# =============================================================================
# Runner
# =============================================================================


def run_flavor(bank, extractor, seed=34):
    aud = MockComplianceAuditor()
    rep = run_compliance_loop(
        bank, aud, strategies=(STRATEGY_SUBSTRATE,),
        seed=seed, extractor=extractor)
    return rep.pooled()[STRATEGY_SUBSTRATE]


def _adapter_for_mixed():
    return {
        "role_events": lambda s, r: list(s.per_role_docs.get(r, ())),
        "causal_ids": lambda s, r: [d.doc_id
                                      for d in s.per_role_docs.get(
                                          r, ())
                                      if d.is_causal],
        "gold_chain": lambda s: s.causal_chain,
        "producer_roles": (ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY,
                             ROLE_FINANCE),
    }


def calibrate(extractor, bank, seed=34, label="unlabeled"):
    ad = _adapter_for_mixed()
    audit = calibrate_extractor(
        extractor, bank,
        ad["role_events"], ad["causal_ids"], ad["gold_chain"],
        roles_to_probe=list(ad["producer_roles"]),
        extractor_label=label,
    )
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                     default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[34, 35])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()

    rows: list[dict] = []
    union_stats_accum = {"n_primary_only": 0,
                         "n_secondary_only": 0,
                         "n_shared": 0, "n_calls": 0}
    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_mixed_bank(
                seed=seed, distractors_per_role=k)
            regex_p = run_flavor(
                bank, regex_extract_claims_for_role, seed=seed)
            llm_p = run_flavor(bank, llm_narrative_extractor,
                                 seed=seed)
            union = union_of(
                regex_extract_claims_for_role,
                llm_narrative_extractor,
                label_primary="regex", label_secondary="llm")
            union_p = run_flavor(bank, union, seed=seed)

            regex_audit = calibrate(
                regex_extract_claims_for_role, bank,
                seed=seed, label="regex")
            llm_audit = calibrate(
                llm_narrative_extractor, bank, seed=seed, label="llm")
            union_audit_ext = union_of(
                regex_extract_claims_for_role,
                llm_narrative_extractor,
                label_primary="regex", label_secondary="llm")
            union_audit = calibrate(
                union_audit_ext, bank, seed=seed, label="ensemble")
            for k_, v_ in union.stats.items():
                union_stats_accum[k_] += v_

            rows.append({
                "k": k, "seed": seed,
                "n_scenarios": len(bank),
                "regex": {
                    "audit": regex_audit.as_dict(),
                    "substrate": {
                        "accuracy_full": regex_p["accuracy_full"],
                        "mean_handoff_recall":
                            regex_p["mean_handoff_recall"],
                        "mean_handoff_precision":
                            regex_p.get("mean_handoff_precision"),
                        "mean_prompt_tokens":
                            regex_p["mean_prompt_tokens"],
                        "failure_hist": regex_p["failure_hist"],
                    },
                },
                "llm": {
                    "audit": llm_audit.as_dict(),
                    "substrate": {
                        "accuracy_full": llm_p["accuracy_full"],
                        "mean_handoff_recall":
                            llm_p["mean_handoff_recall"],
                        "mean_handoff_precision":
                            llm_p.get("mean_handoff_precision"),
                        "mean_prompt_tokens":
                            llm_p["mean_prompt_tokens"],
                        "failure_hist": llm_p["failure_hist"],
                    },
                },
                "ensemble": {
                    "audit": union_audit.as_dict(),
                    "substrate": {
                        "accuracy_full": union_p["accuracy_full"],
                        "mean_handoff_recall":
                            union_p["mean_handoff_recall"],
                        "mean_handoff_precision":
                            union_p.get("mean_handoff_precision"),
                        "mean_prompt_tokens":
                            union_p["mean_prompt_tokens"],
                        "failure_hist": union_p["failure_hist"],
                    },
                },
                "ensemble_stats": {
                    "n_primary_only": union.stats["n_primary_only"],
                    "n_secondary_only":
                        union.stats["n_secondary_only"],
                    "n_shared": union.stats["n_shared"],
                    "n_calls": union.stats["n_calls"],
                },
            })

            print(f"\n  k={k} seed={seed}  (10 scenarios: 5 "
                  f"canonical + 5 narrative)")
            print(f"            regex:     "
                  f"acc={regex_p['accuracy_full']:.2f}  "
                  f"rec={regex_p['mean_handoff_recall']:.2f}  "
                  f"prec={regex_p.get('mean_handoff_precision', 1.0):.2f}  "
                  f"drop={regex_audit.drop_rate:.2f}  "
                  f"spur/ev={regex_audit.spurious_per_event:.2f}")
            print(f"            llm-narr:  "
                  f"acc={llm_p['accuracy_full']:.2f}  "
                  f"rec={llm_p['mean_handoff_recall']:.2f}  "
                  f"prec={llm_p.get('mean_handoff_precision', 1.0):.2f}  "
                  f"drop={llm_audit.drop_rate:.2f}  "
                  f"spur/ev={llm_audit.spurious_per_event:.2f}")
            print(f"            ENSEMBLE:  "
                  f"acc={union_p['accuracy_full']:.2f}  "
                  f"rec={union_p['mean_handoff_recall']:.2f}  "
                  f"prec={union_p.get('mean_handoff_precision', 1.0):.2f}  "
                  f"drop={union_audit.drop_rate:.2f}  "
                  f"spur/ev={union_audit.spurious_per_event:.2f}",
                  flush=True)

    wall = time.time() - t0

    def _mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    pooled_summary = {
        flavor: {
            "accuracy_full": round(_mean(
                [r[flavor]["substrate"]["accuracy_full"]
                 for r in rows]), 4),
            "drop_rate": round(_mean(
                [r[flavor]["audit"]["drop_rate"] for r in rows]), 4),
            "spurious_per_event": round(_mean(
                [r[flavor]["audit"]["spurious_per_event"]
                 for r in rows]), 4),
            "mean_handoff_recall": round(_mean(
                [r[flavor]["substrate"]["mean_handoff_recall"]
                 for r in rows]), 4),
            "mean_handoff_precision": round(_mean(
                [r[flavor]["substrate"]["mean_handoff_precision"]
                 for r in rows]), 4),
        }
        for flavor in ("regex", "llm", "ensemble")
    }
    pooled_summary["union_stats"] = union_stats_accum

    print()
    print("=" * 100)
    print("PHASE 34 / PART C POOLED — regex / llm / ensemble on "
          "compliance mixed (5 canonical + 5 narrative)")
    print("=" * 100)
    print("                      acc        drop   spur/ev   recall   "
          "prec")
    for flavor in ("regex", "llm", "ensemble"):
        s = pooled_summary[flavor]
        print(f"  {flavor:>10}  "
              f"acc={s['accuracy_full']:>4.2f}   "
              f"drop={s['drop_rate']:>4.2f}   "
              f"spur/ev={s['spurious_per_event']:>4.2f}   "
              f"rec={s['mean_handoff_recall']:>4.2f}   "
              f"prec={s['mean_handoff_precision']:>4.2f}")

    dr = pooled_summary["regex"]["drop_rate"]
    dl = pooled_summary["llm"]["drop_rate"]
    du = pooled_summary["ensemble"]["drop_rate"]
    er = pooled_summary["regex"]["spurious_per_event"]
    el = pooled_summary["llm"]["spurious_per_event"]
    eu = pooled_summary["ensemble"]["spurious_per_event"]
    predicted_drop = dr * dl
    predicted_spur = er + el
    print()
    print("  C33-4 bound: δ_union ≤ δ_r · δ_l ?")
    print(f"           measured δ_u = {du:.3f}  "
          f"predicted = δ_r·δ_l = {dr:.3f}·{dl:.3f} = {predicted_drop:.3f}"
          f"  → bound {'SATISFIED' if du <= predicted_drop + 1e-6 else 'VIOLATED'}")
    print("  C33-4 bound: ε_union ≤ ε_r + ε_l ?")
    print(f"           measured ε_u = {eu:.3f}  "
          f"predicted = ε_r+ε_l = {er:.3f}+{el:.3f} = {predicted_spur:.3f}"
          f"  → bound {'SATISFIED' if eu <= predicted_spur + 1e-6 else 'VIOLATED'}")
    print()
    print(f"  wall = {wall:.1f}s")

    payload = {
        "config": {
            "seeds": args.seeds,
            "distractor_counts": args.distractor_counts,
        },
        "rows": rows,
        "pooled_summary": pooled_summary,
        "c33_4_check": {
            "delta_r": dr, "delta_l": dl, "delta_u": du,
            "delta_r_times_l": predicted_drop,
            "delta_bound_satisfied":
                du <= predicted_drop + 1e-6,
            "epsilon_r": er, "epsilon_l": el, "epsilon_u": eu,
            "epsilon_r_plus_l": predicted_spur,
            "epsilon_bound_satisfied":
                eu <= predicted_spur + 1e-6,
        },
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
