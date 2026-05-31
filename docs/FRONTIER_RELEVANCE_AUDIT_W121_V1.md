# Frontier-Relevance Audit — W121 matched EXPOSED official-ICPC control

*(Outcome-independent: this audit justifies the matched-CONTROL design; the pilot RESULT
is recorded separately in `RESULTS_W121_*`.)*

## The question

W120 ran the W89 mechanism on a ≥30 official-ICPC **RESISTANT** battlefield ⇒ B−A1 =
+0.00 pp (FAIL). But the two confirmed retirements (W89/W105) are on contamination-EXPOSED
**HumanEval-family** code, while the resistant null is on **ICPC-regional** code. So the
exposed-vs-resistant dissociation across the programme has so far also been a
benchmark-FAMILY and benchmark-DIFFICULTY difference. Is a matched EXPOSED ICPC control —
the SAME official package family, dated BEFORE the cutoff — a clean way to hold difficulty
+ family fixed and flip ONLY exposure, or is it a soft target that would inflate an
exposed "win"?

## Why it IS a clean matched control

1. **Same official source families, opposite cutoff side.** The exposed control is built
   from the SAME two `github.com/icpc` surface families W120 used (the `na-ecna-archive`
   archive + the `na-rocky-mountain-*-public` repos), on the immediately-preceding
   pre-cutoff year-drops (RMRC 2021 2022-03-14 + ECNA 2022-2023 2022-11-12 + RMRC
   2022-2023 2023-02-25 + ECNA 2023-2024 2023-11-11). Every problem is dated **at or
   before** Maverick's KNOWN August-2024 cutoff (EXPOSED), the exact complement of W120's
   strictly-after (RESISTANT). The contest *series* (Rocky Mountain Regional; East-Central
   NA / NA East Division) are identical to W120's — these are the literal prior editions
   of the same regionals.

2. **Same grader, same evaluator, same modality, same model.** Grading is the SAME
   executable hidden-test oracle (`run_icpc_stdin_executor_v1`; token-diff on official
   `data/secret/*.in`+`*.ans`; NO LLM judge), self-test-verified on EACH exposed surface
   (30 all-pass problems / 637 official secret cases). The bench, the K=5 budget, the
   seed (120001), and the 9 Phase-2 gates + MLB-1/MLB-2 are byte-identical to W120
   (verbatim W108 evaluator). The target model (Maverick) and the stdin/stdout modality
   (the original W89/W105 modality) are the same. The ONLY systematic difference vs W120
   is the contest date relative to the cutoff.

3. **Same difficulty class.** ICPC regional problems are set to a regional-contest
   difficulty target that is stable year to year within a series; the exposed editions are
   the SAME regionals' prior years. So "ICPC-regional difficulty" — the confound the W120
   results doc flagged as not-separated from contamination — is held approximately FIXED
   across the exposed/resistant contrast. (The A0/A1 absolute pass rates on each side are
   reported so the reader can check difficulty comparability empirically.)

4. **Genuinely EXPOSED for the target.** All four exposed contests were published 9–29
   months BEFORE the Aug-2024 cutoff, i.e. squarely inside Maverick's pretraining window —
   public official packages with accepted reference solutions on GitHub well before the
   cutoff. This is the in-distribution property the EXPOSED retirements rode on.

## Objections considered (and why they do not inflate an exposed win)

* **"You swapped RMRC 2023-2024 for RMRC 2021."** RMRC 2023-2024 is more recent
  (2024-01-16, still pre-cutoff) but ships a MINIMAL package — secret data only, **no
  `problem_statement/*.tex`** — so it cannot present a statement to the model, which every
  W120 resistant problem could. Admitting it would break comparability (the model would be
  solving from no statement). The deterministic rule advances to the next pre-cutoff
  artifact-complete RMRC drop (2021), keeping all four exposed surfaces statement+grader
  complete and mirroring W120's 2-ECNA + 2-RMRC structure. RMRC 2021 is the same contest
  series at the same difficulty class — a typed comparability swap, not cherry-picking.

* **"Exposed problems might be systematically easier."** They are the same regionals'
  prior editions (same setters' difficulty targets). The audit reports A0/A1 on BOTH
  sides; if exposed A0/A1 were dramatically higher purely from ease (not exposure), that
  would itself be visible. The contrast of interest is B−A1 (the mechanism margin), which
  difficulty alone does not manufacture.

* **"Python may TLE / figures unavailable."** Identical to W120: these bind ALL arms
  equally (every arm emits Python; figures withheld from A0/A1/B alike), so they can only
  depress absolute pass rates, never manufacture a B−A1 margin. Conservative, not
  inflationary.

* **"Reflexion could be peeking at hidden tests."** It cannot — reflexion feedback is
  restricted to PUBLIC samples + the judge verdict bit + the Python stderr tail; the
  secret `.in`/`.ans` are used ONLY to score, never shown. Byte-identical anti-cheat to
  W120.

* **"Aggregating four surfaces / two years per family is cherry-picking."** Inclusion +
  ordering are a total deterministic machine rule over the official payload; the slice is
  surface×year stratified and outcome-blind (CID `32d15db5…` pinned BEFORE any NIM call).
  No problem was hand-picked.

* **"Float / custom problems slipped in to reach 30."** No: the ≥30 gate AND the pilot use
  the STRICT tier-1 pure-pass-fail tier only (42 ≥ 30). Float (5) and custom-with-validator
  (1) are documented breadth; 2 custom-no-validator are excluded (typed).

## What a clean result licenses

* **Exposed ≥ +5 pp margin while resistant ~0** ⇒ the difficulty/family loophole is
  CLOSED: on the SAME family at matched difficulty, only exposure flips, and the mechanism
  margin appears on the exposed side ⇒ the contamination reading of W89/W105 is sharply
  supported (still observational, single-seed each side).
* **Exposed null too (within ±3.34 pp)** ⇒ the contamination reading WEAKENS materially:
  exposure within ICPC does not reproduce the HumanEval-family margin, so ICPC difficulty
  (not exposure) is the more likely driver of the resistant null ⇒ the bounded
  contamination-EXPOSED-HumanEval-family-at-70B ceiling HARDENS.

Either way the result is defensible because the control is honest: same family, same
grader, same evaluator, same difficulty class, same model — opposite cutoff sides.

## Contamination-confound status

W121 is the within-family within-model exposure contrast the programme had not yet run.
It does not by itself PROVE the confound (single seed each side; ICPC ≠ HumanEval family,
so the cross-family margin transfer is still an inference), but it removes the
"exposed-vs-resistant is confounded with benchmark family/difficulty" objection in the
ICPC setting specifically. See `CONTAMINATION_CONTROL_FRAMING_W121_V1.md`.
