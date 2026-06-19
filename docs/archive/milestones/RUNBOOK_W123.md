# RUNBOOK — W123: large-n matched ICPC closure (≥100/field) attempt

**Milestone:** W123 (COO-9 lead path; child of COO-6).
**Date locked:** 2026-05-31.
**Stable boundary:** `coordpy.__version__ == "0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI, `coordpy/__init__.py` untouched, advanced work explicit-import-only. Decision CID `258b6ed7` invariant.

W122 closed the single-seed caveat but left the matched resistant-vs-exposed ICPC contrast **unresolvable at n=30** (3-seed means resistant **+4.44pp** / exposed **+8.89pp**, both out-of-band, both non-mechanism-driven, branch **B4**). The pre-committed escalation (RUNBOOK_W122 §8) is **larger n PER FIELD (≥100/field), NOT more n=30 seeds, NO 4th seed**. W123 attempts exactly that on the SAME official `github.com/icpc` family, SAME evaluator line, SAME pass-fail discipline.

This runbook locks the α/β/γ branch logic BEFORE any expensive (NIM) call. The Lane-α supply census is a **deterministic, $0, NIM-free** GitHub-API count — running it is not a spend and is the first application of the rule below.

---

## 1. α/β/γ branch logic (pre-committed)

- **Lane α (supply census + construction):** census the official `github.com/icpc` family for the maximum reachable tier-1 pure pass-fail supply on BOTH sides (post-cutoff resistant; pre-cutoff exposed), under the unchanged W120/W121 family + grader + pass-fail rules.
  - **α-PASS** ⟺ BOTH sides can reach ≥100 tier-1 from the official family → build both ≥100 battlefields (§2) and proceed to Lane β.
  - **α-CAP** ⟺ at least one side cannot reach ≥100 → land the census as a machine-checkable blocker, name the exact deficit + the only unblock path, and DO NOT spend. (Lane α point 10.)
- **Lane β (large-n Maverick matched pilot):** fires **ONLY** if α-PASS **and** Maverick certifies on BOTH ≥100 fields. Then run ONE fresh paired seed on both (§4), interpret under §4, earn a second paired large-n seed ONLY if the first lands in the pre-committed ambiguity band.
- **Lane γ (stronger-model gate / graphify / truth):** mandatory regardless of α/β. Re-check primary-source cutoffs (§6); refresh graphify at start+end; tighten the truth surface so the outcome is defensible (§8).

---

## 2. ≥100 matched-battlefield construction rule

Identical to W120/W121 except the count target rises from 30 to 100 PER FIELD:

1. **Same official family:** `github.com/icpc` repos / official ICPC archive folders in the ICPC/Kattis Problem Package Format — NOT mirrors/aggregators/scrapers.
2. **Same evaluator line:** the verbatim W108 `_evaluate_phase2_gates` + `_mlb_rates`; grading = `run_icpc_stdin_executor_v1` over official `data/secret/*.in|*.ans`; NO LLM judge; reflexion feedback = public samples + verdict bit + stderr tail only.
3. **Same tiers:** tier-1 pure pass-fail is the CORE and the ONLY tier counted toward the 100 gate AND used in the pilot. (tier-2 float / tier-3 shipped-validator admitted for breadth only.) Excluded kinds (interactive, custom-without-validator, scoring) are never admitted to cross a count.
4. **Same date rule:** resistant = contest date STRICTLY AFTER `2024-08-31`; exposed = AT OR BEFORE `2024-08-31`.
5. **No curation:** inclusion + ordering are a total deterministic machine function of the official payload (sorted by `(contest_date, source_repo, short_name)`); every exclusion typed; CIDs/SHA pinned before any NIM.
6. **Grader self-test per surface:** every newly admitted surface must pass the R8 accepted-reference self-test before admission.
7. **Two expansion routes:** α1 = aggregate more official surfaces / years / regions; α2 = admit additional deterministic validator-mediated pass-fail tasks from already-included surfaces only if evaluator-clean and secret-data-backed.

---

## 3. validator-mediated admission rule (α2)

A task from an already-included surface may be admitted beyond tier-1 ONLY if it ships an official `output_validators/` deterministic checker (tier-3) or is float-tolerant deterministic (tier-2). These do NOT count toward the tier-1 ≥100 core gate; they widen breadth only. No operator-synthesised graders, ever.

---

## 4. large-n null band and margin rule (pre-committed, tighter than n=30)

At n=100 one K=5 rescue ≈ **1.00pp** (vs 3.33pp at n=30), so the bands tighten:

- `NULL_BAND_PP_N100 = 1.50` (≈ 1.5 rescues). Mean |B−A1| ≤ 1.50 ⇒ null.
- `MARGIN_PASS_PP = 5.00` (unchanged retirement-grade margin, = W89/W105).
- Per-field large-n branch (mirror of W122 B1/B2/B3, precedence B3>B2>B1>B4):
  - **B1 (matched null closes sharply):** |R̄|≤1.50 AND |Ē|≤1.50 ⇒ matched-family caveat CLOSED; bounded ceiling stands as HumanEval-family-specific.
  - **B2 (contamination re-strengthens):** Ē≥5.00 AND |R̄|≤1.50 ⇒ register contamination confound re-strengthened sharply.
  - **B3 (resistant reopening):** R̄≥5.00 (+ clean per-seed mechanism gates) ⇒ candidate THIRD retirement on resistant code (needs the W89/W105 bar: 3+ seeds × ≥100).
  - **B4 (ambiguous):** anything else (mean in the (1.50, 5.00) gap, or mixed) ⇒ earn EXACTLY ONE more paired large-n seed.

---

## 5. paired large-n seed earn/no-earn rule

- Default: ONE fresh paired seed on BOTH ≥100 fields.
- Earn a SECOND paired large-n seed ONLY if the first lands in branch **B4** (the (1.50, 5.00) gap or a mixed pattern). Do NOT default to multiple seeds. Do NOT buy n=30 seeds. A close or confounded edge is NOT closure.

---

## 6. per-model disclosure-status & certification rule

Reuse `certify_model_v1` C1∧C2∧C3∧C4 (resistant) / C2e (exposed), KNOWN-cutoff-only. A stronger-than-Maverick model can supersede Maverick as lead ONLY if it is reachable AND discloses a **primary-KNOWN** cutoff that leaves ≥100 problems on the relevant side of the SAME battlefield. Current registry (re-confirmed in-repo; primary sources last directly re-fetched W120 2026-05-30): `{KNOWN:1, UNKNOWN:4}` — Maverick (Aug-2024) KNOWN; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 all UNKNOWN from primary. Gate STRUCTURALLY CLOSED.

---

## 7. spend rules

- **Large-n NIM:** earned ONLY if Lane α is α-PASS (BOTH fields ≥100 from the official family) AND Maverick certifies on both.
- **Second large-n seed:** earned ONLY if the first large-n pass lands in branch B4.
- **Mechanism NIM:** none unless the ICPC signal regime genuinely changes (M3 stays closed).
- **Stronger-model NIM:** none unless a primary-KNOWN reachable stronger model clears §6.
- **No** n=30 seeds, **no** 405B, **no** dirty exposed benchmark, **no** reopening MBPP+ V2 / cross-modal / Llama-3.1 / APPS-main-lane.

---

## 8. graphify deliverables

- Refresh `graphify update .` at START (built from current HEAD) and END (after code/doc changes).
- Use `graphify explain` on the W120/W121/W122 entry points + the new census entry point; `graphify affected` to confirm the census module is leaf/standalone (no coupling into the pilot path).

---

## 9. W124 branch logic (pre-committed)

- If W123 is **α-CAP** (expected): W124 = **accept the bounded HumanEval-family ceiling as the standing claim** (W89+W105), OR fire only when a NEW official post-cutoff ICPC regional drop (RMRC/ECNA 2026-2027+) lifts the resistant supply toward 100, OR a primary-KNOWN reachable stronger-than-Maverick model opens §6 on the existing ≥30 battlefield.
- If W123 were **α-PASS** and the large-n pilot ran: W124 carries the large-n verdict (close the caveat / register re-strengthening / register resistant reopening per §4).
- COO-9 stays lead unless evidence forces a code-line move.
