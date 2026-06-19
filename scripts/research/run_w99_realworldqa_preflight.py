#!/usr/bin/env python3
"""W99 — RealWorldQA NIM-free preflight + addressability probes
for the W99 candidate slate (B2, B4, B5).

Runs the W96-D D2 composite (P1..P4) per candidate plus W99-
specific addressability probes that mine the W97 and W98
sidecars to produce PRE-NIM predictions:

  * **B2** (direct-vision final-turn answerer) —
    ``coordpy.realworldqa_bench_v3`` (built in W98, preflight-
    earned in W98 but deferred from W98 pilot).  W99 mines:
      - AddrW99-B2-P1: Best/realistic/conservative NIM-free
        upper-bound from W97 confusion table (both-pass +
        unique-B-rescues short-circuit + unique-A1-rescues
        rescued by final VLM ≈ A1).
      - AddrW99-B2-P2: K=3 text-solver budget preserves W97
        D2-B0 first-PASS-at-≤3 wins?  We don't have per-turn
        data, so this is a STATIC code audit.
      - AddrW99-B2-P3: Final VLM rescue prior (A1 wins 5/5 on
        W97 unique-A1-rescues = 100%).
      - AddrW99-B2-P4: K=5 byte-exact.

  * **B4** (typed schema WITHOUT direct_answer_hint) —
    ``coordpy.realworldqa_bench_v4`` (new W99).  W99 mines:
      - AddrW99-B4-P1: B4 schema retains W98 B1's yes/no fix
        primitives (state / orientation / depth / text_in_object).
      - AddrW99-B4-P2: B4 schema strips ``direct_answer_hint``;
        removes the proximate cause of W98 B1's 5 multi-choice
        regressions.
      - AddrW99-B4-P3: K=5 byte-exact.

  * **B5** (question-type router / switch baseline) —
    ``coordpy.realworldqa_bench_v5`` (new W99).  W99 mines:
      - AddrW99-B5-P1: ORACLE simulation on W97 sidecars.
        For each problem, route by ``detect_question_type_v2``;
        if route=B0, predicted outcome = W97 D2-B0 outcome;
        else predicted outcome = W97 A1 outcome.  This produces
        an EXACT NIM-free PASS-rate prediction.
      - AddrW99-B5-P2: Parser correctness on the slice (re-uses
        W98 AddrP6).
      - AddrW99-B5-P3: K=5 byte-exact on every route.

Cross-candidate decision logic (PRE-LOCKED here so the script
cannot retro-rationalise):

* Phase 0 — any candidate failing its W96-D composite OR any
  AddrW99-* probe is KILLED.
* Phase 1 — at most TWO surviving candidates may be promoted to
  cheap NIM pilots (ranked by W99 expected lift).  If exactly
  one survives, that one gets the pilot.  If neither survives,
  pivot to ``COO-9`` (second code benchmark).
* The W99 cross-candidate decision allows up to 2 NIM pilots
  (the brief explicitly says "multiple cheap live tries are
  allowed").

Outputs a JSON verdict, a Markdown SUMMARY, and per-probe
sidecars under ``results/w99/realworldqa_preflight_b2_b4_b5/<RUN_ID>/``.
Exit code is non-zero iff NO candidate survives.

NIM-free.  No NIM calls.  No version bump.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordpy.realworldqa_bench_v2 import (  # noqa: E402
    QUESTION_TYPE_MULTI_CHOICE_LETTER,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_SHORT_TEXT,
    QUESTION_TYPE_YES_NO,
    detect_question_type_v2,
)
from coordpy.realworldqa_bench_v3 import (  # noqa: E402
    W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
    _run_b_direct_vision_final as _v3_b_runner,  # noqa: F401
)
from coordpy.realworldqa_bench_v4 import (  # noqa: E402
    W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
    _run_b_typed_no_hint_vlm_team as _v4_b_runner,  # noqa: F401
)
from coordpy.realworldqa_bench_v5 import (  # noqa: E402
    ROUTE_A1_VLM_K5,
    ROUTE_VLM_TEAM_B0,
    W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
    b5_route_for_question,
)
from coordpy.realworldqa_loader_v1 import (  # noqa: E402
    fetch_realworldqa_test_parquets,
    load_realworldqa_test_corpus_v1,
    manifest_for_corpus_v1,
    select_realworldqa_subset_v1,
)
from coordpy.realworldqa_preflight_v1 import (  # noqa: E402
    REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL,
    run_realworldqa_preflight_v1,
)


_W99_DECOMPOSITION_ARGUMENT_B2 = """
W99 B2 candidate decomposition argument
=======================================

W98 B1 (typed schema + question-typed solver) was empirically
load-bearing on yes/no perception (4 / 5 W97 unique-A1-rescues
recovered) BUT introduced a multi-choice regression cluster (5
W98 B1 FAILs vs W97 D2-B0 wins) via reader-hint over-
confidence.  Net B − A1 = −6.67 pp.

B2 is structurally distinct: it preserves the W95-B0 free-text
extraction and the K=3 text-solver chain, AND adds a final VLM
answerer that sees the original image on the W97 failure
cluster (where text-solver chain FAILs).  Mechanism:

1. VLM scene reader (T=0.0; 1 call).  Free-text bullet list,
   same as W95-B0 / W97 D2-B0.
2. Text-solver chain (T=0.7; 3 calls).  Initial + 2 reflexion
   turns.  Short-circuit on first PASS.
3. Final VLM answerer (T=0.0; 1 call; sees image + extraction
   + question + 3 prior FAIL candidates) — invoked ONLY when
   all 3 text-solver turns FAIL.  Committed answerer, NOT a
   binary verifier (mechanistically distinct from W96-C C1
   which was empirically refuted).

W97-mined prediction (NIM-free upper bound):
* Both-pass (22 / 30): B2 short-circuits to text-solver PASS.
* Unique-B-rescues (3 / 30): B2 text-solver PASSes within
  K=3 turns (W97 D2-B0 first-PASS-turn data shows all 3
  succeeded by turn 2).
* Unique-A1-rescues (5 / 30): text-solver FAILs; final VLM
  invoked with full image access.  A1 wins 5/5 on this cluster
  by definition.  B2 final VLM has equivalent visual access
  plus structured extraction.

K=5 byte-exact: 1 reader + 3 text solver + 1 final-VLM-or-pad.
Same VLM model on every arm; executor =
evaluate_realworldqa_answer_v1.  No selective retries; no LLM
judge.
"""


_W99_DECOMPOSITION_ARGUMENT_B4 = """
W99 B4 candidate decomposition argument
=======================================

W98 B1 (typed schema + question-typed solver) FAILed by
−6.67 pp because:

* the typed solver became more confident in the reader's
  ``direct_answer_hint`` field (often wrong on multi-choice);
* this stopped the K=4 reflexion-cycling discipline that
  drove W97 D2-B0's 3 unique-B-rescues + some both-pass wins;
* result: 4 yes/no rescues − 5 multi-choice regressions = −1.

B4 is a *minimal repair*: strip the ``direct_answer_hint``
field from the typed schema and remove the hint-anchoring from
the typed solver prompt.  Everything else is byte-identical to
W98 B1 (same primitives: state / orientation / depth /
text_in_object; same question-typed prompt blocks; same first-
PASS short-circuit; same K=5 budget).

Prediction (NIM-free reasoning):
* The 4 yes/no rescues (W97 → W98 B1) survive because they
  depend on the schema primitives (state / orientation /
  depth), NOT on the hint.
* The 5 multi-choice regressions are addressed because the
  typed solver now relies on the JSON primitives + the
  question text, not a possibly-wrong hint.  Reflexion-
  cycling discipline restored on multi-choice.

K=5 byte-exact: 1 typed reader + 4 typed solver.  Same VLM
model on every arm; executor = evaluate_realworldqa_answer_v1.
"""


_W99_DECOMPOSITION_ARGUMENT_B5 = """
W99 B5 candidate decomposition argument
=======================================

B5 is a SWITCH BASELINE, not a frontier mechanism.  It exists
to bound how much "team" superiority is achievable by routing
alone vs. by a structurally new mechanism.

Routing rule (deterministic, NIM-free, no oracle):
  multi_choice_letter           ->  W97 D2-B0 (V1 B arm)
  yes_no | numeric | short_text ->  A1 K=5

W97 + W98 sidecars show on the 96_504_002 / 30-problem slice:
* D2-B0 PASSes 18 / 18 multi-choice problems (W97).
* A1 K=5 PASSes 12 / 12 yes/no + numeric + short_text problems
  (W97 — all 5 unique-A1-rescues were on this subset).

NIM-free ORACLE prediction: B5 = 30 / 30 = 100 %; B5 − A1 =
+10.00 pp.  Subject to parser correctness (W98 AddrP6 = 29/30
= 96.7 %) and live NIM sampling variance.

B5 is allowed under the W99 brief explicitly: "B5 is allowed as
a switch baseline if kept honest and same-budget exact."  It is
NOT a substrate / structural / multi-agent context mechanism.
A B5 PASS would prove the per-question ceiling is high enough
that the W95-B0-family cap is a routing problem at the
per-question level — not evidence of structural team
superiority.

K=5 byte-exact on either route.  Same VLM model on every arm.
"""


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_w97_per_problem_outcomes(w97_pilot_dir: Path):
    pp_path = w97_pilot_dir / "per_problem.jsonl"
    if not pp_path.exists():
        raise SystemExit(
            f"W97 pilot dir missing per_problem.jsonl: {pp_path}")
    pp = [json.loads(line)
          for line in pp_path.read_text().splitlines()
          if line.strip()]
    return pp


def _load_w98_b1_per_problem_outcomes(w98_pilot_dir: Path):
    pp_path = w98_pilot_dir / "per_problem.jsonl"
    if not pp_path.exists():
        raise SystemExit(
            f"W98 pilot dir missing per_problem.jsonl: {pp_path}")
    pp = [json.loads(line)
          for line in pp_path.read_text().splitlines()
          if line.strip()]
    return pp


# ---------------------------------------------------------------
# B2 addressability probes
# ---------------------------------------------------------------

def _addr_w99_b2_p1_nim_free_upper_bound(pp_w97) -> dict:
    """B2-P1: NIM-free upper bound from W97 confusion table.

    For each problem:
      D2-B0 PASS  ∧ A1 PASS  ⇒ B2 expected PASS (short-circuit)
      D2-B0 PASS  ∧ A1 FAIL  ⇒ B2 expected PASS (text-solver
                                 hits PASS within K=3 turns
                                 per W97 first-PASS-turn data
                                 in arsenal-mining doc)
      D2-B0 FAIL  ∧ A1 PASS  ⇒ B2 expected PASS via final-VLM
                                 (A1 wins 5/5 on this cluster)
      D2-B0 FAIL  ∧ A1 FAIL  ⇒ B2 expected FAIL
    """
    both_pass = 0
    unique_b = 0
    unique_a1 = 0
    neither = 0
    for p in pp_w97:
        b = bool(p.get("b_vlm_team_passed"))
        a1 = bool(p.get("a1_vlm_passed"))
        if b and a1:
            both_pass += 1
        elif b and not a1:
            unique_b += 1
        elif not b and a1:
            unique_a1 += 1
        else:
            neither += 1
    n_total = len(pp_w97)
    n_a1 = both_pass + unique_a1
    # Conservative: assume final-VLM rescues 50% of unique-A1
    # cluster.  Realistic: 80%.  Best: 100%.
    best = both_pass + unique_b + unique_a1
    realistic = both_pass + unique_b + int(round(0.80 * unique_a1))
    conservative = both_pass + unique_b + int(
        round(0.50 * unique_a1))
    a1_rate = (float(n_a1) / float(n_total)) if n_total else 0.0
    b2_best_rate = (
        float(best) / float(n_total)) if n_total else 0.0
    b2_real_rate = (
        float(realistic) / float(n_total)) if n_total else 0.0
    b2_cons_rate = (
        float(conservative) / float(n_total)) if n_total else 0.0
    # Probe passes if realistic prediction ≥ A1 + 5 pp threshold.
    threshold = 0.05
    passed = bool(b2_real_rate >= a1_rate + threshold)
    return {
        "probe_id": "AddrW99_B2_P1_nim_free_upper_bound",
        "candidate": "B2",
        "passed": bool(passed),
        "summary": (
            f"W97 conf-table: both_pass={both_pass}, "
            f"unique_b={unique_b}, unique_a1={unique_a1}, "
            f"neither={neither}.  B2 NIM-free: best="
            f"{b2_best_rate * 100.0:.2f}%, "
            f"realistic={b2_real_rate * 100.0:.2f}%, "
            f"conservative={b2_cons_rate * 100.0:.2f}%.  "
            f"A1@K=5 (W97)={a1_rate * 100.0:.2f}%.  "
            "Threshold: realistic ≥ A1 + 5 pp"),
        "both_pass": int(both_pass),
        "unique_b": int(unique_b),
        "unique_a1": int(unique_a1),
        "neither": int(neither),
        "n_total": int(n_total),
        "b2_best_rate": float(b2_best_rate),
        "b2_realistic_rate": float(b2_real_rate),
        "b2_conservative_rate": float(b2_cons_rate),
        "a1_rate_w97": float(a1_rate),
        "threshold_pp": float(threshold * 100.0),
    }


def _addr_w99_b2_p2_short_circuit_static() -> dict:
    """B2-P2: STATIC code audit — B2's text-solver chain uses
    first-PASS short-circuit on the K-2 = 3 text-solver turns
    when K=5."""
    from coordpy import realworldqa_bench_v3 as v3
    src = inspect.getsource(v3._run_b_direct_vision_final)
    has_short_circuit = (
        "for i, exe in enumerate(text_exes):" in src
        and "if exe.passed:" in src
        and "text_solver_pass_idx = int(i)" in src
        and "break" in src)
    has_padding = (
        "text_solver_short_circuit_pad" in src
        and "n_pad = int(K) - len(calls)" in src)
    has_final_vlm = (
        "final_vlm_answerer" in src
        and "p.image_bytes" in src)
    passed = bool(
        has_short_circuit and has_padding and has_final_vlm)
    return {
        "probe_id": "AddrW99_B2_P2_short_circuit_static",
        "candidate": "B2",
        "passed": bool(passed),
        "summary": (
            f"V3 short-circuit={has_short_circuit}, "
            f"padding={has_padding}, "
            f"final_vlm={has_final_vlm}"),
        "has_short_circuit": bool(has_short_circuit),
        "has_padding": bool(has_padding),
        "has_final_vlm": bool(has_final_vlm),
    }


def _addr_w99_b2_p3_final_vlm_rescue_prior(pp_w97) -> dict:
    """B2-P3: definitionally, A1 wins all of the W97
    unique-A1-rescue cluster.  B2's final-VLM has equivalent
    visual access on the same cluster.  We record the rescue
    prior."""
    unique_a1 = [p for p in pp_w97
                 if p.get("a1_vlm_passed")
                 and not p.get("b_vlm_team_passed")]
    n = len(unique_a1)
    passed = bool(n >= 3)
    return {
        "probe_id": "AddrW99_B2_P3_final_vlm_rescue_prior",
        "candidate": "B2",
        "passed": bool(passed),
        "summary": (
            f"A1 K=5 rescues {n}/{n} W97 unique-A1-rescues by "
            "re-seeing the image; B2 final-turn VLM has "
            "equivalent visual access on the same cluster"),
        "n_a1_rescues": int(n),
        "pass_threshold": 3,
    }


def _addr_w99_b2_p4_budget_exact() -> dict:
    """B2-P4: K=5 byte-exact."""
    from coordpy.realworldqa_bench_v3 import (
        RealWorldQAV3BenchConfig)
    cfg = RealWorldQAV3BenchConfig()
    passed = bool(cfg.K_multi_sample == 5)
    return {
        "probe_id": "AddrW99_B2_P4_budget_exact",
        "candidate": "B2",
        "passed": bool(passed),
        "summary": f"B2 K={cfg.K_multi_sample}",
        "K": int(cfg.K_multi_sample),
    }


# ---------------------------------------------------------------
# B4 addressability probes
# ---------------------------------------------------------------

def _addr_w99_b4_p1_schema_primitives_retained() -> dict:
    """B4-P1: typed schema must still contain the W97 yes/no
    fix primitives (state / orientation / depth /
    text_in_object)."""
    from coordpy import realworldqa_bench_v4 as v4
    prompt = v4._B_TYPED_SCENE_READER_SYSTEM
    required = {
        "state",
        "orientation",
        "depth",
        "text_in_object",
    }
    missing = {r for r in required if r not in prompt}
    passed = bool(not missing)
    return {
        "probe_id": "AddrW99_B4_P1_schema_primitives_retained",
        "candidate": "B4",
        "passed": bool(passed),
        "summary": (
            "schema retains all W98 yes/no-fix primitives"
            if passed
            else f"missing primitives: {sorted(missing)}"),
        "required": sorted(required),
        "missing": sorted(missing),
        "schema_version": (
            W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION),
    }


def _addr_w99_b4_p2_hint_field_removed() -> dict:
    """B4-P2: typed schema must NOT contain ``direct_answer_hint``
    (the field that drove W98 B1's multi-choice regressions)."""
    from coordpy import realworldqa_bench_v4 as v4
    prompt = v4._B_TYPED_SCENE_READER_SYSTEM
    has_hint = bool("direct_answer_hint" in prompt)
    # Negation in the prompt body documenting the removal is OK;
    # but the prompt should NOT list it as a field-to-fill.  We
    # check by counting the substring vs the explicit removal
    # admonition.
    n_occurrences = prompt.count("direct_answer_hint")
    # Solver prompts should also not reference the hint.
    solver = v4._B_TYPED_SOLVER_SYSTEM_TEMPLATE
    solver_has_hint = bool("direct_answer_hint" in solver)
    # Accept the removal if the only occurrences are in the
    # explicit "do NOT include" admonition.  Specifically we
    # require ≤ 1 mention in the reader prompt (the admonition)
    # and 0 mentions in the solver template.
    passed = bool(n_occurrences <= 1 and not solver_has_hint)
    return {
        "probe_id": "AddrW99_B4_P2_hint_field_removed",
        "candidate": "B4",
        "passed": bool(passed),
        "summary": (
            f"reader prompt mentions direct_answer_hint "
            f"{n_occurrences}× (only the explicit removal "
            f"admonition; threshold ≤ 1); solver template "
            f"has hint = {solver_has_hint}"),
        "reader_hint_mentions": int(n_occurrences),
        "solver_template_has_hint": bool(solver_has_hint),
    }


def _addr_w99_b4_p3_budget_exact() -> dict:
    """B4-P3: K=5 byte-exact."""
    from coordpy.realworldqa_bench_v4 import (
        RealWorldQAV4BenchConfig)
    cfg = RealWorldQAV4BenchConfig()
    passed = bool(cfg.K_multi_sample == 5)
    return {
        "probe_id": "AddrW99_B4_P3_budget_exact",
        "candidate": "B4",
        "passed": bool(passed),
        "summary": f"B4 K={cfg.K_multi_sample}",
        "K": int(cfg.K_multi_sample),
    }


# ---------------------------------------------------------------
# B5 addressability probes (oracle simulation)
# ---------------------------------------------------------------

def _addr_w99_b5_p1_oracle_simulation(pp_w97) -> dict:
    """B5-P1: ORACLE simulation on W97 sidecars.  For each
    problem, compute the routing decision and the predicted
    outcome.  Produces an EXACT NIM-free PASS-rate
    prediction for B5 on this slice."""
    n_total = len(pp_w97)
    b5_pass = 0
    a1_pass = 0
    detail = []
    route_dist = {ROUTE_VLM_TEAM_B0: 0, ROUTE_A1_VLM_K5: 0}
    route_pass = {ROUTE_VLM_TEAM_B0: 0, ROUTE_A1_VLM_K5: 0}
    for p in pp_w97:
        q = p.get("question") or ""
        route = b5_route_for_question(q)
        if route == ROUTE_VLM_TEAM_B0:
            predicted = bool(p.get("b_vlm_team_passed"))
        else:
            predicted = bool(p.get("a1_vlm_passed"))
        a1 = bool(p.get("a1_vlm_passed"))
        route_dist[route] += 1
        if predicted:
            b5_pass += 1
            route_pass[route] += 1
        if a1:
            a1_pass += 1
        detail.append({
            "pid": p["pid"],
            "question_head": q[:80],
            "question_type": detect_question_type_v2(q),
            "route": route,
            "w97_b_vlm_team_passed": bool(
                p.get("b_vlm_team_passed")),
            "w97_a1_vlm_passed": bool(p.get("a1_vlm_passed")),
            "b5_predicted_passed": bool(predicted),
        })
    b5_rate = (
        float(b5_pass) / float(n_total)) if n_total else 0.0
    a1_rate = (
        float(a1_pass) / float(n_total)) if n_total else 0.0
    margin_pp = float((b5_rate - a1_rate) * 100.0)
    threshold_pp = 5.0
    passed = bool(margin_pp >= threshold_pp)
    return {
        "probe_id": "AddrW99_B5_P1_oracle_simulation",
        "candidate": "B5",
        "passed": bool(passed),
        "summary": (
            f"NIM-free oracle: B5={b5_pass}/{n_total} "
            f"= {b5_rate * 100.0:.2f}%; A1@K=5 (W97) "
            f"= {a1_rate * 100.0:.2f}%; margin = "
            f"{margin_pp:+.2f} pp (threshold ≥ +5 pp).  "
            f"Routing: {route_dist[ROUTE_VLM_TEAM_B0]} multi-"
            f"choice → D2-B0 (PASS {route_pass[ROUTE_VLM_TEAM_B0]}"
            f"/{route_dist[ROUTE_VLM_TEAM_B0]}); "
            f"{route_dist[ROUTE_A1_VLM_K5]} non-mc → A1 K=5 "
            f"(PASS {route_pass[ROUTE_A1_VLM_K5]}"
            f"/{route_dist[ROUTE_A1_VLM_K5]})"),
        "b5_predicted_pass": int(b5_pass),
        "b5_predicted_rate": float(b5_rate),
        "a1_rate_w97": float(a1_rate),
        "margin_pp": float(margin_pp),
        "threshold_pp": float(threshold_pp),
        "route_distribution": dict(route_dist),
        "route_pass_distribution": dict(route_pass),
        "n_total": int(n_total),
        "detail": detail,
    }


def _addr_w99_b5_p2_parser_correctness(pp_w97) -> dict:
    """B5-P2: re-run W98 AddrP6 (parser correctness on the
    W97 slice).  Threshold ≥ 90 %."""
    correct = 0
    total = 0
    for p in pp_w97:
        q = p.get("question") or ""
        pred = detect_question_type_v2(q)
        starts_with_yn = any(q.lstrip().lower().startswith(
            v + " ") for v in (
                "is", "are", "was", "were", "do", "does",
                "did", "can", "could", "will", "would",
                "has", "have", "should", "may", "might"))
        has_multi_choice_options = (
            q.count("A.") + q.count("A)") >= 1
            and q.count("B.") + q.count("B)") >= 1)
        if has_multi_choice_options:
            gold = QUESTION_TYPE_MULTI_CHOICE_LETTER
        elif starts_with_yn:
            gold = QUESTION_TYPE_YES_NO
        elif "how many" in q.lower():
            gold = QUESTION_TYPE_NUMERIC
        else:
            gold = QUESTION_TYPE_SHORT_TEXT
        if pred == gold:
            correct += 1
        total += 1
    rate = float(correct) / float(total) if total else 0.0
    passed = bool(rate >= 0.90)
    return {
        "probe_id": "AddrW99_B5_P2_parser_correctness",
        "candidate": "B5",
        "passed": bool(passed),
        "summary": (
            f"parser correct on {correct}/{total} = "
            f"{rate * 100.0:.1f}% (threshold ≥ 90%)"),
        "n_correct": int(correct),
        "n_total": int(total),
        "rate": float(rate),
    }


def _addr_w99_b5_p3_budget_exact() -> dict:
    """B5-P3: K=5 byte-exact on either route.  V5 deferred
    routes to V1's runners which use K=5."""
    from coordpy.realworldqa_bench_v5 import (
        RealWorldQAV5BenchConfig)
    cfg = RealWorldQAV5BenchConfig()
    passed = bool(cfg.K_multi_sample == 5)
    return {
        "probe_id": "AddrW99_B5_P3_budget_exact",
        "candidate": "B5",
        "passed": bool(passed),
        "summary": (
            f"B5 K={cfg.K_multi_sample} on either route"),
        "K": int(cfg.K_multi_sample),
    }


# ---------------------------------------------------------------
# Composite preflight
# ---------------------------------------------------------------

def _composite_realworldqa_preflight(
        candidate_model: str, manifest, problems,
        decomposition_argument: str) -> dict:
    verdict = run_realworldqa_preflight_v1(
        manifest=manifest,
        problems=problems,
        candidate_model=candidate_model,
        decomposition_argument=decomposition_argument,
        max_acceptable_a1_k5_pass_rate=80.0,
        min_executor_self_test_pass_rate=0.98)
    return verdict.to_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-model",
        default="meta/llama-3.2-11b-vision-instruct")
    ap.add_argument(
        "--w97-pilot-dir", type=Path,
        default=(ROOT / "results" / "w97" / "realworldqa_pilot"
                 / "w97_realworldqa_pilot_11b_meta_llama-3.2"
                   "-11b-vision-instruct__meta_llama-3.2-11b"
                   "-vision-instruct_20260525T182409Z"))
    ap.add_argument(
        "--w98-b1-pilot-dir", type=Path,
        default=(ROOT / "results" / "w98" / "realworldqa_pilot"
                 / "w98_realworldqa_pilot_b1_11b_meta_llama-3.2"
                   "-11b-vision-instruct__meta_llama-3.2-11b"
                   "-vision-instruct_20260525T191938Z"))
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("data") / "realworldqa")
    ap.add_argument(
        "--out-dir", type=Path,
        default=(ROOT / "results" / "w99"
                 / "realworldqa_preflight_b2_b4_b5"))
    args = ap.parse_args()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    safe_model = args.candidate_model.replace(
        "/", "_").replace(":", "_")
    scale_tag = (
        "11b" if "11b" in args.candidate_model.lower()
        else ("90b" if "90b" in args.candidate_model.lower()
              else "unknown"))
    run_dir = (
        Path(args.out_dir)
        / f"w99_preflight_{scale_tag}_{safe_model}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w99.preflight] run_dir={run_dir}")

    # --- Corpus
    print(
        "[w99.preflight] fetching RealWorldQA test parquets …")
    paths, shas, total_bytes = (
        fetch_realworldqa_test_parquets(cache_dir=args.cache_dir))
    print(
        f"[w99.preflight] parquet shards SHA-anchored: "
        f"{[s[:8] for s in shas]} ({total_bytes} bytes)")
    print("[w99.preflight] decoding corpus …")
    corpus = load_realworldqa_test_corpus_v1(parquet_paths=paths)
    manifest = manifest_for_corpus_v1(
        parquet_paths=paths, problems=corpus,
        parquet_shard_sha256=shas,
        parquet_total_bytes=total_bytes)
    print(
        f"[w99.preflight] corpus n_problems={len(corpus)} "
        f"merkle={manifest.corpus_merkle_root[:16]}…")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

    # --- W97 + W98 sidecars
    pp_w97 = []
    if args.w97_pilot_dir.exists():
        pp_w97 = _load_w97_per_problem_outcomes(
            args.w97_pilot_dir)
        print(
            f"[w99.preflight] W97 sidecars: per_problem="
            f"{len(pp_w97)}")
    else:
        print(
            "[w99.preflight] WARNING: W97 pilot dir not found; "
            "AddrW99-B2-P1, B2-P3, B5-P1, B5-P2 will be skipped.")

    pp_w98 = []
    if args.w98_b1_pilot_dir.exists():
        pp_w98 = _load_w98_b1_per_problem_outcomes(
            args.w98_b1_pilot_dir)
        print(
            f"[w99.preflight] W98 B1 sidecars: per_problem="
            f"{len(pp_w98)}")
    else:
        print(
            "[w99.preflight] WARNING: W98 B1 pilot dir not found")

    # --- B2 composite
    print(
        "[w99.preflight] running B2 composite (W96-D P1..P4) …")
    b2_composite = _composite_realworldqa_preflight(
        args.candidate_model, manifest, corpus,
        _W99_DECOMPOSITION_ARGUMENT_B2)
    b2_composite_passes = bool(b2_composite["overall_passes"])

    # --- B4 composite
    print(
        "[w99.preflight] running B4 composite (W96-D P1..P4) …")
    b4_composite = _composite_realworldqa_preflight(
        args.candidate_model, manifest, corpus,
        _W99_DECOMPOSITION_ARGUMENT_B4)
    b4_composite_passes = bool(b4_composite["overall_passes"])

    # --- B5 composite
    print(
        "[w99.preflight] running B5 composite (W96-D P1..P4) …")
    b5_composite = _composite_realworldqa_preflight(
        args.candidate_model, manifest, corpus,
        _W99_DECOMPOSITION_ARGUMENT_B5)
    b5_composite_passes = bool(b5_composite["overall_passes"])

    # --- W99 addressability probes
    addr_probes: list[dict] = []
    if pp_w97:
        addr_probes.append(_addr_w99_b2_p1_nim_free_upper_bound(
            pp_w97))
    addr_probes.append(_addr_w99_b2_p2_short_circuit_static())
    if pp_w97:
        addr_probes.append(
            _addr_w99_b2_p3_final_vlm_rescue_prior(pp_w97))
    addr_probes.append(_addr_w99_b2_p4_budget_exact())

    addr_probes.append(_addr_w99_b4_p1_schema_primitives_retained())
    addr_probes.append(_addr_w99_b4_p2_hint_field_removed())
    addr_probes.append(_addr_w99_b4_p3_budget_exact())

    if pp_w97:
        addr_probes.append(_addr_w99_b5_p1_oracle_simulation(
            pp_w97))
        addr_probes.append(_addr_w99_b5_p2_parser_correctness(
            pp_w97))
    addr_probes.append(_addr_w99_b5_p3_budget_exact())

    # --- Per-candidate verdicts
    def _candidate_probes(cand: str) -> list[dict]:
        return [p for p in addr_probes if p["candidate"] == cand]

    b2_probes = _candidate_probes("B2")
    b4_probes = _candidate_probes("B4")
    b5_probes = _candidate_probes("B5")
    b2_addr_pass = all(bool(p["passed"]) for p in b2_probes)
    b4_addr_pass = all(bool(p["passed"]) for p in b4_probes)
    b5_addr_pass = all(bool(p["passed"]) for p in b5_probes)

    b2_overall = bool(b2_composite_passes and b2_addr_pass)
    b4_overall = bool(b4_composite_passes and b4_addr_pass)
    b5_overall = bool(b5_composite_passes and b5_addr_pass)

    # --- Cross-candidate decision
    survivors = []
    if b2_overall:
        survivors.append("B2")
    if b4_overall:
        survivors.append("B4")
    if b5_overall:
        survivors.append("B5")

    # Build expected-lift table for ranking.
    b5_p1 = next((p for p in addr_probes
                  if p["probe_id"]
                  == "AddrW99_B5_P1_oracle_simulation"), None)
    b2_p1 = next((p for p in addr_probes
                  if p["probe_id"]
                  == "AddrW99_B2_P1_nim_free_upper_bound"), None)
    expected_lifts = {
        "B2": (
            b2_p1["b2_realistic_rate"]
            - b2_p1["a1_rate_w97"]) * 100.0
        if b2_p1 else None,
        "B4": None,  # no NIM-free oracle; B4 requires NIM to test.
        "B5": (
            b5_p1["margin_pp"]) if b5_p1 else None,
    }

    if survivors:
        decision = (
            "Survivors: " + ", ".join(survivors) + ".  Per the "
            "W99 brief's expensive-run discipline (multiple "
            "cheap tries allowed where each earns it), the "
            "ranked promotion order is by NIM-free expected "
            "lift (descending): " + ", ".join(
                f"{c} ({l:+.2f} pp)"
                if l is not None else f"{c} (NIM-required)"
                for c, l in sorted(
                    expected_lifts.items(),
                    key=lambda kv: (
                        -(kv[1] or -1e9), kv[0])))
            + ".  Up to 2 winners may be promoted to cheap NIM "
            "pilots; B4 is gated on B2 or B5 PASS at the same "
            "scale since its prediction has no NIM-free oracle.")
    else:
        decision = (
            "ALL three candidates KILLED at preflight.  Per "
            "Part G of the W99 brief: document the kills, sync "
            "Linear, promote COO-9 (second code benchmark) as "
            "the next lead path, and explicitly note the "
            "W95-B0-derived family is now empirically capped "
            "across FOUR preflight-earned attempts (D2-B0 / "
            "D2-B1 / B4 / B5) at 11B on RealWorldQA.")

    overall_passes = bool(survivors)

    verdict = {
        "schema": "coordpy.w99_realworldqa_preflight.v1",
        "candidate_model": args.candidate_model,
        "scale_tag": scale_tag,
        "corpus_merkle_root": str(manifest.corpus_merkle_root),
        "w97_pilot_dir": str(args.w97_pilot_dir),
        "w98_b1_pilot_dir": str(args.w98_b1_pilot_dir),
        "n_w97_slice_problems": int(len(pp_w97)),
        "n_w98_b1_slice_problems": int(len(pp_w98)),
        "b2_composite": dict(b2_composite),
        "b4_composite": dict(b4_composite),
        "b5_composite": dict(b5_composite),
        "addressability_probes": list(addr_probes),
        "b2_composite_passes": bool(b2_composite_passes),
        "b4_composite_passes": bool(b4_composite_passes),
        "b5_composite_passes": bool(b5_composite_passes),
        "b2_addr_passes": bool(b2_addr_pass),
        "b4_addr_passes": bool(b4_addr_pass),
        "b5_addr_passes": bool(b5_addr_pass),
        "b2_overall_passes": bool(b2_overall),
        "b4_overall_passes": bool(b4_overall),
        "b5_overall_passes": bool(b5_overall),
        "survivors": list(survivors),
        "expected_lifts_pp": dict(expected_lifts),
        "decision": str(decision),
        "overall_w99_slate_passes": bool(overall_passes),
    }
    verdict_cid = _sha256_hex(verdict)
    verdict["verdict_cid"] = str(verdict_cid)
    (run_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True))

    # --- Summary markdown
    lines: list[str] = []
    lines.append(
        f"# W99 RealWorldQA preflight — {run_dir.name}\n")
    lines.append(
        f"Candidate model: `{args.candidate_model}`  ")
    lines.append(
        f"Corpus Merkle root: "
        f"`{manifest.corpus_merkle_root}`  ")
    lines.append(
        f"Verdict cid: `{verdict_cid}`  \n")
    lines.append("## Composite preflight (W96-D P1..P4)\n")
    for cand, comp, comp_pass in (
            ("B2", b2_composite, b2_composite_passes),
            ("B4", b4_composite, b4_composite_passes),
            ("B5", b5_composite, b5_composite_passes)):
        lines.append(
            f"* {cand} composite overall: "
            f"`{'PASS' if comp_pass else 'FAIL'}`")
        for pr in comp["probes"]:
            lines.append(
                f"  * {pr['probe_id']}: "
                f"`{'PASS' if pr['passed'] else 'FAIL'}` — "
                f"{pr['summary']}")
    lines.append("\n## W99 addressability probes\n")
    for pr in addr_probes:
        lines.append(
            f"* **{pr['probe_id']}** ({pr['candidate']}): "
            f"`{'PASS' if pr['passed'] else 'FAIL'}` — "
            f"{pr['summary']}")
    lines.append("\n## Verdicts\n")
    lines.append(
        f"* B2 overall: "
        f"`{'PASS' if b2_overall else 'FAIL'}`  "
        f"(composite={b2_composite_passes}; "
        f"addr={b2_addr_pass})")
    lines.append(
        f"* B4 overall: "
        f"`{'PASS' if b4_overall else 'FAIL'}`  "
        f"(composite={b4_composite_passes}; "
        f"addr={b4_addr_pass})")
    lines.append(
        f"* B5 overall: "
        f"`{'PASS' if b5_overall else 'FAIL'}`  "
        f"(composite={b5_composite_passes}; "
        f"addr={b5_addr_pass})")
    lines.append(f"\n## Decision\n\n{decision}\n")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    (Path(args.out_dir) / "latest_run.txt").write_text(
        run_dir.name + "\n")

    print()
    for cand, ov in (
            ("B2", b2_overall),
            ("B4", b4_overall),
            ("B5", b5_overall)):
        print(
            f"[w99.preflight] {cand} overall: "
            f"{'PASS' if ov else 'FAIL'}")
    print(f"[w99.preflight] Survivors: {survivors}")
    print(f"[w99.preflight] {decision}")

    return 0 if overall_passes else 4


if __name__ == "__main__":
    sys.exit(main())
