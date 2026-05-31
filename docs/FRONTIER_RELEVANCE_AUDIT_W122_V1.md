# Frontier-Relevance Audit — W122 matched-family multi-seed closure + ICPC mechanism probe + stronger-model gate

*(Outcome-independent: this audit justifies the W122 closure DESIGN, the Lane-β
mechanism KILL, and the Lane-γ gate-closure argument. The paired-seed RESULT is recorded
separately in `RESULTS_W122_*`.)*

## The question

W120 (RESISTANT ICPC) and W121 (matched EXPOSED ICPC) each ran the W89 mechanism on the
SAME official ICPC package family at the SAME scale (Maverick), opposite cutoff sides:
B−A1 = +0.00 pp (resistant) and +3.33 pp (exposed), both within the pre-committed ±3.34 pp
null band ⇒ `CONFOUND_WEAKENS / bounded ceiling HARDENS`. The ONE remaining live
statistical caveat was **single seed each side** (120001). Is running ONE paired new seed
on BOTH fields the honest way to remove that caveat — or is it seed-chasing?

## Why the paired-seed closure IS a clean, decisive operation (not seed-chasing)

1. **It targets the one named caveat, under a pre-committed symmetric rule.** The closure
   rule (`docs/RUNBOOK_W122.md` § 2, `interpret_paired_closure_v1`) is locked BEFORE any
   NIM and is symmetric across both fields: both-null ⇒ caveat closed (B1); exposed-rises
   ⇒ contamination re-strengthens (B2); resistant-rises ⇒ candidate third retirement (B3);
   anything-else ⇒ earn exactly one more paired seed (B4). The same ±3.34 / +5.00
   thresholds W120/W121 used are applied to the 2-seed MEANS. No outcome is privileged.

2. **The 2-seed aggregate is structurally decisive.** At n=30, K=5, a per-problem rescue
   moves B−A1 by exactly 3.333 pp, so 2-seed means lie on the lattice {0, 1.67, 3.33,
   5.00, …}. The null band (≤3.34) and the margin (≥5.00) are lattice points; the
   ambiguous open interval (3.34, 5.00) contains NO lattice point. So each field's 2-seed
   aggregate is decisive unless the two seeds disagree in direction — the only realistic B4
   trigger. The 3rd seed is bought ONLY then. This is the opposite of open-ended
   seed-chasing: the stopping rule is mechanical.

3. **It reuses the EXACT instruments, byte-identically.** Both 30-slices are CID-guarded:
   resistant `01bf9ef8…` (W120), exposed `32d15db5…` (W121), each re-derived NIM-free and
   confirmed == provenance == pilot guard (2026-05-31). The package loaders, the
   stdin/stdout reflexion bench, the verbatim W108 9-gate + MLB-1/MLB-2 evaluator, K=5, the
   secret-case grader, and the public-only reflexion feedback are all unchanged. The ONLY
   thing that varies vs W120/W121 is the seed (120002, then optionally 120003). So a
   multi-seed shift is attributable to sampling variance alone, not to any instrument drift.

4. **It is the pre-committed W121 § 9 tightening.** W121's runbook and its COO-9 comment
   both named "(optional) one paired seed on BOTH battlefields to tighten the single-seed
   caveat" as a W122 option. W122 promotes it to the main lane because closing a live
   caveat directly is the honest aggressive move (the milestone brief: "do NOT settle for
   'weakened' if the remaining caveats can be closed directly").

## Why Lane β (different mechanism) is honestly KILLED NIM-free

The strongest non-reflexion mechanism in the arsenal is M3, the executor-grounded
structured-failure patcher (`coordpy.executor_grounded_patcher_v1`, W111). Its
load-bearing differentiator vs reflexion (W111 § 2.3) is an **explicit expected/actual
contract extracted from the FAILING test**. The W122 NIM-free audit
(`audit_icpc_mechanism_signal_v1` over the real 660-call W120+W121 sidecars, 240 reflexion
turns) found:

* On official ICPC the hidden oracle is **SECRET token-diff** — it returns only "wrong
  answer on a hidden case", NEVER the expected value (anti-cheat). So the hidden expected
  M3 needs is **structurally unavailable**.
* The only expected-value signal on ICPC is the **public samples**, which the existing
  reflexion bench ALREADY feeds (`WRONG (expected …)`) on 63 % of failing turns.
* `m3_exclusive_signal_fraction = 0.000` (< the 0.33 floor) ⇒ M3 holds **no** actionable
  contract reflexion lacks ⇒ it reduces to a reflexion prompt-variant on this family. And
  M3 was already sub-reflexion (12.5 % rescue < reflexion 25 % < the 33 % floor) on
  BigCodeBench where it had its FULL unittest edge (W111).

⇒ M3 cannot earn a fair probe on ICPC; the lane is killed NIM-free ($0). This is itself a
**finding**: the ICPC null is NOT merely reflexion-specific — the family's secret-grading
regime makes the same-budget ceiling **mechanism-robust**. M3 is genuinely different *only*
where the grader reveals the hidden expected; ICPC does not. (Falsifiable: a regime that
reveals the hidden expected, or a sidecar with ≥33 % hidden-only-with-expected turns, flips
the audit to BUILD — tested.)

## Why Lane γ (stronger model) gate is STRUCTURALLY CLOSED

The matched battlefields are anchored to Maverick's KNOWN Aug-2024 cutoff: the RESISTANT
field is every official ICPC problem dated 2024-11-11 .. 2025-11-13 (strictly after the
cutoff). For ANY stronger model M to be RESISTANT-certified on the SAME battlefield, M
needs ≥30 problems strictly after M's cutoff — i.e. M's cutoff must pre-date 2025-11. The
reachable stronger-than-Maverick models (re-checked LIVE 2026-05-31 from PRIMARY sources):

| model | primary cutoff | resistant-certifiable on the matched field? |
|---|---|---|
| Llama-4-Maverick | **Aug-2024 KNOWN** | YES — the unique anchor (the field is resistant FOR it) |
| Qwen3-Coder-480B | none stated (HF card) | no — UNKNOWN |
| DeepSeek-V4-pro | none stated (V4 card PDF re-fetched directly — confirms NO cutoff; the "Apr-2026" figure is a non-primary aggregator) | no — UNKNOWN, and the aggregator reading would put it AFTER the whole field ⇒ EXPOSED, not resistant |
| Mistral-Small-4-119B | none stated (Mistral docs) | no — UNKNOWN; released 2026-03 ⇒ post-dates the field |
| GLM-5 | none primary | no — UNKNOWN + not NIM-reachable |

⇒ only a model with a primary-KNOWN cutoff ≤ ~Aug-2024 could be resistant-certified on this
matched battlefield, and Maverick is the unique such reachable model. **The gate does not
open; Maverick stays the unique matched-family closure target; $0 Lane-γ NIM.** A
stronger-than-Maverick model with a primary-KNOWN earlier cutoff would re-open BOTH
battlefields in a future milestone (it does not exist today).

## What each closure outcome licenses

*(ACTUAL outcome: the 2-seed aggregate was **B4** [resistant null +1.67 / exposed out-of-band
+8.33], which earned the 3rd paired seed [120003]. The FINAL **3-seed** aggregate is **ALSO
B4 `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`** — seed 120003 spiked **+10.00 pp on BOTH fields**
[`PASS_NON_MECHANISM_DRIVEN`], so the 3-seed means are RESISTANT +4.44 / EXPOSED +8.89, BOTH
out of band and neither clean ⇒ B1/B2/B3 all off. The hedged "B1 expected" guess below did NOT
hold — and not in the guessed direction: the RESISTANT field did NOT stay null, it spiked too,
so the contrast is unresolvable at n=30. Exactly why the pre-committed rule, not the guess,
decides. B4-after-the-3rd-seed is terminal [no 4th seed]; W123 = larger n PER FIELD. Realised
result: `docs/RESULTS_W122_PAIRED_SEED_CLOSURE_V1.md`.)*

* **B1 (both means null):** the matched-family null SURVIVES multiple seeds ⇒ single-seed
  caveat CLOSED; contamination-confound stays WEAKENED, now multi-seed; bounded ceiling
  hardens to multi-seed HumanEval-family-(ease/structure)-specific @ 70B. (Hedged pre-run
  guess; did NOT hold — actual was B4.)
* **B2 (exposed margin, resistant null):** W121's "weakened" read REVISES back up ⇒
  contamination re-strengthens via a de-noised exposed margin. Reported as a reversal.
* **B3 (resistant margin):** the bounded resistant ceiling is OVERTURNED multi-seed ⇒
  candidate THIRD retirement on resistant code; W123 = full resistant retirement bench.
* **B4 (ambiguous):** earn one more paired seed (cap = 3). B4-after-the-3rd-seed = register
  the residual ambiguity, W123 = accept the bounded claim or escalate to larger n PER FIELD
  (no 4th seed). ← **REALISED at 3 seeds** (resistant +4.44 / exposed +8.89, both out of band,
  neither clean).

Either way the result is defensible because the design is honest: same instruments, same
evaluator, pre-committed symmetric rule, mechanical stopping, and $0 on the two lanes whose
gates did not open.
