FAILed). See `docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md`.

## Post-W109 update — the second contamination-RESISTANT test ran (W110)

W110 ran a SECOND, genuinely-different contamination-RESISTANT benchmark to ask whether the W108 LiveCodeBench FAIL was benchmark-specific or general:

* **W110 — BigCodeBench (contamination-RESISTANT 2024-06, post-cutoff)**: the W89 mechanism FAILed AGAIN (B − A1 = +0.00 pp; A0 63.33 / A1 70.00 / B 70.00 %; MLB-1 = 40 % PASS but MLB-2 = 25 % FAIL — reflexion was genuinely invoked and still did not help; net 1 rescue / 1 regression). The SAME weak 25 % rescue rate as W108 LiveCodeBench.

So contamination-resistant same-budget code superiority via the W89 mechanism is now **0/2** at 70B (LiveCodeBench 2025 + BigCodeBench 2024), against **3/3** on contamination-EXPOSED benchmarks (W89 HumanEval, W105 HumanEval+, W109 APPS). **The W108 FAIL is GENERAL, not LCB-specific.** The contamination-confound moves from SUPPORTED (W109) to **STRENGTHENED toward a finding (W110), but still NOT proven** (two single-seed resistant points; the resistant benchmarks could share a difficulty property orthogonal to contamination). Net effect on this narrative: the two confirmed retirements (W89, W105) **STAND**, now explicitly bounded as **contamination-EXPOSED HumanEval-family at 70B**; a contamination-RESISTANT same-budget code superiority is **unproven and has now FAILed on both resistant benchmarks tested**. (SWE-bench-lite + LiveBench-coding were evaluated and REJECTED at $0.) See `docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md` + `docs/RESULTS_W110_BIGCODEBENCH_PHASE2_70B_V1.md`.

## Post-W110 update — a genuinely DIFFERENT mechanism was tried (W111)

Because reflexion is 0/2 on resistant code, W111 asked the sharper question: is the resistant ceiling specific to *reflexion*, or does it hold for *any* same-budget mechanism at 70B? A NIM-free re-execution census of all 300 W110 BigCodeBench candidates localised the resistant failure to **81.6 % SEMANTIC hidden-test coupling / 1.8 % API-grounding** — which **killed M1 (library/spec planner) and M2 (local symbol/doc introspection) at $0** (they attack near-absent failure classes) and admitted **M3 — an executor-grounded structured-failure patcher** (typed expected/actual contract + minimal-patch, materially different from prose reflexion, never the hidden test source).

* **W111 — M3 patcher (contamination-RESISTANT BigCodeBench, smallest-decisive 143-call probe)**: M3's patch loop rescued ONE hard-core problem reflexion could not (`/13`), but its rescue rate was **12.5 % (1/8) — BELOW reflexion's 25 % and the 33 % floor**; its +15.38 pp on the rescue-concentrated probe slice is an UPPER BOUND inflated by one attempt-0 sampling win (`/20`). M3 did NOT earn a fair pilot (it did not hold reflexion's `/51` rescue; a sub-floor non-load-bearing mechanism cannot clear +5 pp mechanism-driven on a fair slice — W104→W105 erosion + W106 margin-cap discipline).

So a contamination-RESISTANT same-budget code superiority is now unproven for **two mechanisms** at 70B: the W89 reflexion mechanism (0/2 benchmarks) and a genuinely-different executor-grounded patcher (M3, fair pilot not earned). The resistant ceiling is **NOT reflexion-specific** — it is a property of same-budget multi-call mechanisms at 70B against hidden-test-coupling difficulty. The two confirmed retirements (W89, W105) **STAND**, bounded as **contamination-EXPOSED HumanEval-family at 70B**; the bounded-claim is the honest code ceiling. W111 adds NO retirement and proves NOTHING about the contamination confound (it tests a mechanism, not the confound). The honest positive: M3's mechanism is not vacuous (it rescued `/13`), just sub-reflexion. See `docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W111_V1.md`.

## Post-W111 update — the stronger-model gate went LIVE (W112)

Because the resistant ceiling is not reflexion-specific at 70B, W112 asked whether SCALE reopens it. For the first time since the cross-scale-UP axis opened, a genuinely STRONGER, same-budget-comparable, non-reasoning code model was reachable on NIM — `meta/llama-4-maverick-17b-128e-instruct` (405B returned its 6th consecutive 404). The earned BigCodeBench pilot ran on the EXACT W110 fair 30-slice (only the model changed):

* **W112 — Llama-4-Maverick × BigCodeBench (1 seed × 30 × K=5)**: A0 73.33 / A1 73.33 / B 83.33 %; **B − A1 = +10.00 pp**; 9/9 core Phase-2 gates; MLB-2 = 37.5 % PASS but **MLB-1 = 26.67 % FAIL ⇒ `PASS_NON_MECHANISM_DRIVEN`** (3 clean rescues `/15`,`/26`,`/51`, 0 regressions). The reflexion margin REOPENED where 70B was +0.00 pp.

But this is **NOT** a clean reopening of contamination-RESISTANT superiority, and the consolidated narrative is bounded accordingly. **Contamination-resistance is MODEL-CUTOFF-RELATIVE**: BigCodeBench 2024-06 is resistant for Llama-3.3-70B (~2024-01 cutoff) but plausibly EXPOSED for Llama-4-Maverick (**Aug-2024 pretraining cutoff** > release). The result is a structural twin of the W109 APPS contamination-EXPOSED control (A0 = 73.33 % identical; PASS_NON_MECHANISM_DRIVEN; MLB-1-fail/MLB-2-pass), and the IDENTICAL slice flips +0.00 pp → +10.00 pp as the model's cutoff crosses the benchmark's release date. So W112 lands in the contamination-EXPOSED column, alongside W89/W105/W109 — a **third dissociation point, the first WITHIN a single benchmark**, that STRENGTHENS the contamination-confound (still not proof; single-seed; capability not fully excluded). The two confirmed retirements (W89, W105) **STAND**; W112 adds NONE. Separately, a NIM-free harder mining (W112 Lane β) showed the fair M3-strengthening design space is structurally sub-floor (reliably fair-reachable ceiling 8.3 %; 58 % of invoked failures mock/fixture-coupled), so no fair strengthening earns NIM. **W113 = a benchmark verifiably contamination-resistant FOR Llama-4 (problem dates > Aug 2024; date-filtered LiveCodeBench) to separate capability from exposure.** See `docs/RESULTS_W112_STRONGER_MODEL_BIGCODEBENCH_PILOT_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W112_V1.md`.

## Post-W112 update — the clean resistant-FOR-Llama-4 test ran (W113); the +10 pp was exposure

W112 left one question that could overturn or confirm the bounded claim: does the +10 pp SURVIVE on a benchmark verifiably resistant for Maverick? W113 built that instrument and ran it. A machine-checkable rule (`contest_date > 2024-08-31`, the entire ambiguous August window excluded) found the date-filtered LiveCodeBench `release_v6` functional subset is **63/63 resistant** for Maverick (all 2025-01..04; 0 ambiguous/missing dates), and the deterministic resistant 30-slice CID is IDENTICAL to the W108 slice — so the Maverick pilot ran on the EXACT problems 70B ran, with model scale the only variable.

* **W113 — Llama-4-Maverick × RESISTANT-LiveCodeBench (1 seed × 30 × K=5)**: A0 30.00 / A1 50.00 / B 50.00 %; **B − A1 = +0.00 pp**; 7/9 gates; **MLB-1 = 63.33 % PASS but MLB-2 = 21.05 % FAIL ⇒ `FAIL` → `EXPOSURE_CONFIRMED`** (reflexion genuinely invoked — more than at 70B — but rescued 4 / regressed 2 / net 0).

This is the clean disambiguation. The W112 **+10.00 pp on EXPOSED BigCodeBench COLLAPSED to +0.00 pp on resistant LiveCodeBench at the same Maverick scale** ⇒ the +10 pp was **contamination EXPOSURE**, not a capability reopening. The 2×2 (model scale × slice resistance, same mechanism) is now complete: the RESISTANT column is **0 clean across BOTH scales** (70B −3.33 / +0.00; Maverick +0.00), the EXPOSED column is all margin. Within the SAME model + mechanism the margin flips +10.00 (exposed) → +0.00 (resistant) purely on slice resistance — the **sharpest contamination dissociation in the programme** (corroborated by Maverick's resistant-slice A1 of 50 % being BELOW 70B's 63.33 %, removing the capability alternative for the +10 pp). The contamination-confound is STRENGTHENED a fourth time but **still NOT proven** (single-seed; BigCodeBench-2024 vs LiveCodeBench-2025 differ in construction, not only vintage). The two confirmed retirements (W89, W105) **STAND**; W113 adds NONE; resistant superiority is unproven AND has now FAILed at the stronger scale. Lane β: all three reachable tier-2 stronger models (Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B) have UNKNOWN 2025-2026 cutoffs ⇒ NO certifiably-resistant slice on the pinned corpus ⇒ no tier-2 pilot is spend-eligible (a later release_v7+ instrument is needed). **W114 = accept the bounded contamination-EXPOSED-HumanEval-family-at-70B claim as the honest code ceiling and pursue a GENUINELY DIFFERENT axis.** See `docs/RESULTS_W113_RESISTANT_PILOT_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W113_V1.md`.

## Post-W113 update — W114 registered the bounded ceiling + named the certification-supply blocker

W114 did the move W113 set up. It did NOT run another exposed rerun or another same-scale resistant reflexion pilot. It (α) REGISTERED the bounded ceiling as the honest code-superiority FLOOR across the canonical truth surfaces, and (β/γ) asked the genuinely different question: **can a NEW instrument be built that is certifiably contamination-resistant for a model STRONGER than Maverick, from the latest REAL release and OFFICIAL cutoffs?**

A NIM-free per-model certification layer (`coordpy.stronger_model_cutoff_certification_v1`; decision CID `258b6ed7…`) answered it from primary sources verified 2026-05-29:

* **Latest instrument (HF dataset file tree):** LiveCodeBench `release_v6` is the latest (`test6.jsonl` is the highest-numbered; no `test7`+). Its FUNCTIONAL subset — the part the W89 mechanism can attack — is 63 problems, all dated **2025-01-11..2025-04-05** (frontier **2025-04-05**). A ≥30 functional resistant slice requires a KNOWN cutoff ≤ ~Jan-2025.
* **Official model cutoffs:** Llama-4-Maverick = **Aug-2024 KNOWN** (official Llama-4 model card) — the ONLY reachable model with a KNOWN cutoff, and already SETTLED here (W113 resistant FAIL ⇒ a second pilot is redundant). Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-2603 = **cutoff OFFICIALLY UNDISCLOSED** (their model cards/blogs state none) ⇒ cannot be certified resistant against any instrument (KNOWN-cutoff-only rule); and where estimable, their cutoffs meet/post-date the Apr-2025 frontier.

* **W114 — certification verdict `NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM.** The latest resistant FUNCTIONAL instrument has **AGED OUT** relative to the reachable frontier-model class: its newest problems (Apr-2025) do not post-date a single reachable stronger-than-Maverick model's verifiable cutoff, and those models' cutoffs are officially undisclosed (the gaps compound). This is the genuinely-different-axis finding — a **certification-supply analysis**, not another benchmark rerun.

The two confirmed retirements (W89, W105) **STAND**; W114 adds NONE and retires NONE. The bounded two-retirement contamination-EXPOSED-HumanEval-family-at-70B claim is now the **registered code ceiling** (`W114-T-BOUNDED-EXPOSED-CODE-CEILING-REGISTERED`). The blocker is a precise, dated spend gate (`W114-L-RESISTANT-INSTRUMENT-FRONTIER-LAGS-MODEL-FRONTIER-CAP` + `W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-UNDISCLOSED`): **W115 fires only when a resistant FUNCTIONAL instrument with ≥30 problems dated strictly after a reachable frontier model's KNOWN cutoff exists** (a future LCB release_v7+ with post-Apr-2025 functional problems AND a frontier model disclosing a KNOWN cutoff < those problems). Until then the bounded ceiling STANDS and resistant-code NIM is BLOCKED on the missing instrument. See `docs/RESULTS_W114_STRONGER_MODEL_CERTIFICATION_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W114_V1.md`.

## Post-W114 update — W115 re-verified the frontier LIVE + operationalised the supply chain

Because the W114 blocker is supply-side and inherently time-sensitive, W115 RE-VERIFIED the external frontier LIVE from PRIMARY sources (2026-05-29) and turned the supply chain into a push-button pipeline. It did NOT run another exposed rerun or another same-scale resistant reflexion pilot.

* **Latest instrument (HF dataset file tree, re-checked live):** LiveCodeBench `release_v6` is STILL the latest (highest `testN.jsonl` = `test6.jsonl`; "add v6" ~1yr ago; **no `test7`+**). Functional frontier 2025-04-05 UNCHANGED.
* **Official model cutoffs (re-checked live):** Llama-4-Maverick Aug-2024 KNOWN (reconfirmed; settled at W113); Qwen3-Coder-480B official HF card states NO cutoff (UNKNOWN); **the DeepSeek V4 official model card now EXISTS (published 2026-04-27, Pro = 1.6T/49B) but discloses NO training cutoff ⇒ still UNKNOWN**; Mistral-Small-4 UNKNOWN. No newly-reachable stronger model with a KNOWN cutoff ≤ ~Jan-2025.

* **W115 — verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM.** Both binding conditions still fail (no newer instrument; no reachable stronger model with a KNOWN cutoff ≤ ~Jan-2025). The one genuine external change since W114 — the DeepSeek V4 card's *publication* — does NOT move the verdict (it discloses no cutoff). The blocker is now LIVE-re-verified + dated (`W115-L-EXTERNAL-FRONTIER-UNCHANGED-NO-CERTIFIABLE-SLICE-REVERIFIED-CAP`).

W115 ALSO shipped a durable future-fire supply-chain pipeline (`coordpy.frontier_certification_pipeline_v1`; `W115-T-FUTURE-FIRE-CERTIFICATION-PIPELINE-SHIPS`): a latest-official-release detector, a generalised frontier-date summary + threshold table (max KNOWN cutoff month admitting a ≥30 slice = 2025-01), a per-model go/no-go matrix (reusing the W114 gate, no duplication), a disclosure-consistency guard, and a structured W116 fire condition — driven by a `FrontierSnapshotV1` so the next clean shot is push-button (decision CID `258b6ed7…`, byte-identical to W114). The two confirmed retirements (W89, W105) **STAND**; W115 adds NONE and retires NONE; the contamination-confound is UNCHANGED. **W116 fires the moment the pipeline trigger flips** (a newer admitted release_v7+ with ≥30 post-frontier functional problems for a KNOWN-cutoff stronger model, OR a reachable stronger-than-Maverick model disclosing a KNOWN cutoff month ≤ 2025-01). See `docs/RESULTS_W115_FRONTIER_CERTIFICATION_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W115_V1.md`.

## Post-W115 update — W116 ATTACKED the upstream supply (not passive waiting) + confirmed the model side

W115 re-checked one surface (the lite file tree) and operationalised the re-check. W116 went one level UPSTREAM and ATTACKED the supply side at FOUR authoritative surfaces, and confirmed the model-disclosure side from PRIMARY sources. It did NOT run another exposed rerun or another same-scale resistant reflexion pilot.

* **Instrument supply (Lane α; four upstream surfaces, LIVE 2026-05-30):** (1) HF `code_generation_lite` file tree — highest `test6.jsonl`, no `test7`+, lastModified 2025-06-05; (2) the loader `code_generation_lite.py` `ALLOWED_FILES` — `v_list=[v1..v6]`, `release_latest` resolves to the SAME files as `release_v6`; (3) the full `code_generation` dataset README — `release_v6`=May-2023..Apr-2025, 1055 problems, frontier 2025-04-05; (4) the LCB GitHub README — no v7 tag. ⇒ **NO admissible new instrument** beyond `release_v6`. The "planned v7" mentioned only by a non-primary search summary is INADMISSIBLE under the pre-committed A1..A5 rule (no artifact, no SHA) and is recorded only as the W117 watch signal.
* **Model cutoffs (Lane β; PRIMARY sources, LIVE 2026-05-30):** Maverick Aug-2024 KNOWN (settled, C4); Qwen3-Coder-480B + DeepSeek-V4-pro UNKNOWN (official cards: NO CUTOFF STATED); **`mistralai/mistral-small-4-119b-2603` now CONFIRMED REAL — 119B MoE, released 2026-03-16 — with its official Mistral docs card + announcement disclosing NO cutoff** (the only figure is a non-primary aggregator "2025-06" that is itself C2-exposed); Mistral-Small-3.2-24B KNOWN ~Oct-2023 but sub-70B (C3). ⇒ no reachable stronger-than-Maverick model has a primary-KNOWN cutoff ≤ 2025-01.

* **W116 — verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM.** A pilot needs BOTH an admissible new instrument AND a non-redundant primary-KNOWN-cutoff stronger model; neither exists (`W116-L-UPSTREAM-SUPPLY-NO-ADMISSIBLE-NEW-INSTRUMENT-FOUR-SURFACE-CAP` + `W116-T-MISTRAL-SMALL-4-CONFIRMED-REAL-PRIMARY-NO-CUTOFF`).

W116 ALSO shipped a durable upstream-ADMISSION pipeline (`coordpy.upstream_instrument_admission_v1`; `W116-T-UPSTREAM-ADMISSION-PIPELINE-SHIPS`): a pre-committed A1..A5 admissibility rule (REFUSES aggregator/mirror/website/rumor instruments), a multi-surface upstream-change detector (the W117 signal), a certifiable-slice builder, a four-way disclosure-status matrix, and a structured W117 fire condition — reusing the W113 registry + W114 gate + W115 pipeline with NO duplication (decision CID `258b6ed7…`, byte-identical to W114/W115; result CID `193164c4…`). The two confirmed retirements (W89, W105) **STAND**; W116 adds NONE and retires NONE; the contamination-confound is UNCHANGED. **W117 fires the moment `detect_upstream_change_v1` flags an admissible change** (a newer admitted release_v7+ / `release_latest` re-point / new upstream functional dataset with ≥30 post-frontier problems for a KNOWN-cutoff stronger model, OR a reachable stronger-than-Maverick model disclosing a primary-KNOWN cutoff month ≤ 2025-01). See `docs/RESULTS_W116_UPSTREAM_ADMISSION_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W116_V1.md`.

## Post-W116 update — W117 proved no inheritance; W118 BUILT the post-v6 identities and isolated the grader as the blocker

W116 attacked the *packaged-release* supply; **W117** escalated to the *construction provenance* at EIGHT authoritative surfaces (HF commit/revision log, refs, discussions; GitHub commits, tags, repo pipeline structure; dataset README provenance; runner loader) and proved that **no post-v6 instrument can be CONSTRUCTED from authoritative LCB provenance** — LCB publishes only packaged releases (no collection pipeline, no forward problem-id manifest), so the only post-v6 path (raw-contest hand-assembly) is refused by the pre-committed B1 (authoritative-LCB-provenance) ∧ B2 (no-operator-curation) criteria (`W117-L-NO-CONSTRUCTION-ADMISSIBLE-POST-V6-INSTRUMENT-EIGHT-SURFACE-CAP` + `W117-T-LCB-CONSTRUCTION-PROVENANCE-IS-PACKAGED-RELEASE`). Verdict re-derived `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID `258b6ed7…`), $0 NIM.

**W118** stopped waiting for a packaged `release_v7` and **built a CoordPy-OWNED post-v6 functional instrument directly from the official source family**:

* **Lane α (LIVE construction):** a real constructor (`coordpy.coordpy_frontier_functional_v1`) ran against the official Codeforces API and produced — deterministically, reproducibly, SHA + CID-pinned — **894 post-v6 functional problem IDENTITIES** (2025-04-07..2026-05-30, 130 contests; manifest CID `fb4185a6…`). The date/IDENTITY axis (O1..O6) is **SOLVED at scale** (894 ≫ 30) — a real advance over W117's "cannot inherit" (`W118-T-COORDPY-OWNED-POST-V6-FUNCTIONAL-IDENTITY-CONSTRUCTIBLE`). BUT the executable functional GRADER (O7) is **ABSENT family-wide**: the Codeforces API has no test field/endpoint; LeetCode hidden tests are deliberately private; AtCoder system tests are Dropbox-only (no official API). Sample-only grading is non-credible and operator-synthesised tests are operator curation (refused) ⇒ the instrument is **identity-admissible but NOT pilot-admissible** (`W118-L-OFFICIAL-SOURCE-FAMILY-NO-EXECUTABLE-GRADER-CAP`). Maverick (KNOWN Aug-2024) is even C1∧C2∧C3∧C4 **identity-CERTIFIABLE** on this genuinely-new instrument — the ONLY blocker is the missing grader (`W118-T-MAVERICK-IDENTITY-CERTIFIABLE-GRADER-BLOCKED`).
* **Lane β:** primary cutoffs re-checked DEEPER (DeepSeek V4 PDF re-fetched directly — no cutoff; Maverick "August 2024" verbatim; Qwen3-Coder-480B + Mistral-Small-4-v26.03 no cutoff); GLM-5 newly noted but UNKNOWN-from-primary + C2-exposed + reachability-unverified ⇒ nothing newly disclosed.
* **Lane γ:** shipped the durable constructor/admission/pilot-readiness pipeline (O1..O7 rule + official-source-family grader registry + reused C1..C4 + O7 gate + W119 fire condition + falsifiability test), reusing the W113/W114/W116/W117 chain with NO duplication (LCB decision CID `258b6ed7…` re-derives byte-identically).

**The blocker MOVED** — from W117's "no post-v6 problem identities can be constructed" to W118's "**abundant official post-v6 identities (894), no official executable grader**". The two confirmed retirements (W89, W105) **STAND**; W118 adds NONE and retires NONE; the contamination-confound is UNCHANGED. **W119 fires the moment** an OFFICIAL executable per-problem test suite for ≥30 post-v6 functional problems appears on a clean official surface (Maverick is already identity-certifiable ⇒ a grader alone unlocks the cheapest honest verdict-changing pilot), OR a packaged `release_v7`+ / LCB-published construction provenance appears, OR a reachable stronger-than-Maverick model discloses a primary-KNOWN cutoff ≤ the manifest frontier. See `docs/RESULTS_W118_FRONTIER_FUNCTIONAL_CONSTRUCTION_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W118_V1.md`.

## Post-W119 update — W120 CLOSED the count gap, certified Maverick, ran the earned pilot (clean FAIL)

W119 dissolved the grader blocker but left a count-only gap (24 < 30 from one surface).
**W120 closed it on official surfaces only** — a problem-by-problem RMRC exclusion audit
(correcting `draftlottery` → float) **plus a NEW official surface** (`icpc/na-ecna-archive`,
NA East Division 2024+2025) ⇒ **45 tier-1 pure pass-fail ≥ 30** (49 gradeable), grader
self-test 165/165 each surface. That made **Maverick certifiable** (C2 flips 24→45). With
`docs/RUNBOOK_W120.md` locked and a clean canary, the **earned pilot RAN** (330 NIM calls):
**B − A1 = +0.00 pp, FAIL** (A0 20.00 / A1 23.33 / B 23.33; MLB-1 83.33% PASS / MLB-2 8.00%
FAIL; 6/9). 

**Net:** no third retirement — **W89 (+5.56) + W105 (+7.00) remain the only two**, both
contamination-EXPOSED-HumanEval-family @ 70B. Resistant superiority is now **0 clean across
FOUR settings** (W108/W110/W113/W120). The decisive advance over W114–W119 is that the
resistant column is no longer *untestable*: W120 BUILT the certifiable ≥30 grader-clean
instrument, certified Maverick, ran the pilot, and the mechanism **did not transfer** — a
strictly stronger statement of the bounded ceiling. Contamination-confound STRENGTHENED
(cleanest resistant null), NOT proven (single seed; difficulty + Python-TLE floor). decision
CID `258b6ed7` invariant. COO-9 lead; W121 = accept the bounded claim / different axis
(optional multi-seed) or a primary-KNOWN stronger-than-Maverick cutoff.

## Post-W120 update — W121 ran the matched-family exposure control; exposure within ICPC did NOT reproduce the margin

W120's resistant null left ONE confound: the EXPOSED retirements (W89/W105) are on EASY
HumanEval-family code while the resistant nulls are on HARD ICPC code, so "exposed vs
resistant" across the programme was entangled with "easy vs hard / HumanEval-family vs
other". **W121 removed the confound the only clean way — it held the family + difficulty
FIXED (official ICPC, same regional series) and flipped ONLY exposure.** It built a matched
EXPOSED battlefield from the SAME two `github.com/icpc` families W120 used, on the
pre-Aug-2024 editions of the same regionals (RMRC 2021 + ECNA 2022-2023 + RMRC 2022-2023 +
ECNA 2023-2024) ⇒ **42 tier-1 pure pass-fail ≥ 30**, grader self-test 30 all-pass / 637
official secret cases each surface; certified Maverick on the EXPOSED side (W114 gate, C2→
C2e); and ran the earned same-model same-mechanism pilot (330 NIM calls):

**EXPOSED B − A1 = +3.33 pp, FAIL** (A0 6.67 / A1 26.67 / B 30.00; MLB-1 93.33% PASS /
MLB-2 25.00% FAIL; 8/9) **vs the LOCKED RESISTANT +0.00 pp (W120)** — both within the
pre-committed ±3.34 pp null band.

**Net:** the matched-family exposure flip did **NOT** reproduce the retirement-grade
HumanEval-family margins (+5.56 / +7.00). The contamination hypothesis predicted a clean
exposed margin (it did not appear); the difficulty/family-ease hypothesis predicted a null
even when exposed (it matched) ⇒ the **strong contamination reading WEAKENS**, difficulty/
family-ease is implicated, and the bounded ceiling **HARDENS** to
**HumanEval-family-(ease/structure)-specific @ 70B**. NOT refuted (faint sub-floor
exposure-consistent gradient: exposed +3.33 > resistant +0.00; exposed reflexion-rescue
25% > resistant 8%; single seed each side). Difficulty comparability is empirically
supported (exposed A0 6.67% ≤ resistant A0 20.00% ⇒ exposed not easier). **W89 (+5.56) +
W105 (+7.00) remain the only two retirements**; W121 adds none; the mechanism gets a clean
retirement-grade margin ONLY on HumanEval-family code and FAILS on official ICPC code
regardless of exposure. Paired seed NOT earned (null-side of band; W106 discipline).
Decision CID `258b6ed7` invariant. COO-9 lead; W122 = accept the hardened bounded ceiling /
genuinely different axis, OR a primary-KNOWN stronger-than-Maverick model on BOTH matched
ICPC battlefields, OR (optional) one paired seed on BOTH battlefields.

## Post-W121 update — W122 ran the FULL paired-seed closure; FINAL 3-seed aggregate = B4 AMBIGUOUS (both fields spiked); closure NOT achieved

W121 left exactly one live caveat: both the resistant (+0.00) and exposed (+3.33) ICPC
results were single-seed (120001). **W122 took the optional paired-seed tightening as its
MAIN lane.** It ran the pre-committed paired-seed closure — seed 120002, then the earned
tiebreaker seed 120003 — on BOTH the EXACT W120 resistant 30-slice and the EXACT W121 exposed
30-slice (CID-guarded, re-derived NIM-free == provenance), under a SYMMETRIC rule
(`interpret_paired_closure_v1`) locked before any NIM — 1320 NIM calls total, Maverick, same
grader/evaluator/K, only the seed changed.

* **Seed 120002, BOTH fields:** RESISTANT B−A1 +3.33 pp (FAIL; 8/9; MLB-2 8.7%); EXPOSED B−A1
  **+13.33 pp** (`PASS_NON_MECHANISM_DRIVEN`; 9/9; MLB-2 28.6%; 4 rescues / 0 regr; A1 dipped
  to 20%). 2-seed aggregate = B4 ⇒ earned the 3rd paired seed.
* **Seed 120003, BOTH fields:** RESISTANT B−A1 **+10.00 pp** (`PASS_NON_MECHANISM_DRIVEN`;
  9/9; MLB-2 16.7%; 3 rescues / 0 regr) and EXPOSED B−A1 **+10.00 pp**
  (`PASS_NON_MECHANISM_DRIVEN`; 9/9; MLB-2 18.5%; 3 rescues / 0 regr). **FINAL 3-seed means:
  RESISTANT +4.44 (from [+0.00, +3.33, +10.00], OUT of band, in the 3.34–5.00 gap), EXPOSED
  +8.89 (from [+3.33, +13.33, +10.00], OUT of band)**; both `all_seeds_clean_pass=false` ⇒
  B1/B2/B3 all off ⇒ **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4)**; `caveat_closed=false`.
  B4-after-the-3rd-seed is terminal (no 4th).

The single-seed caveat is **NOT closed by a clean resolution**. The literal "single seed"
objection is retired (3 seeds each side), but it is replaced by a small-n-variance limitation:
at the 3rd seed the RESISTANT field also spiked +10.00 pp (non-mechanism-driven), so the
2-seed "resistant null vs exposed popped" asymmetry **dissolved** and both fields now carry
the same rescue-concentrated `PASS_NON_MECHANISM_DRIVEN` signature (no `PASS_MECHANISM_DRIVEN`
seed anywhere). At n=30, K=5 the per-field B−A1 swings ±10 pp on ~3 rescues, so the matched
contrast is **unresolvable at this n**. The W121 single-seed "weakened" read is NOT
multi-seed-confirmed (the exposed field did not stay flat) AND contamination is NOT
established — in fact the 3-seed data **further undercuts** a clean contamination read, since
contamination-RESISTANT code spiked just like exposed code (the spikes are sampling variance,
not exposure); the faint exposed > resistant ordering (+8.89 > +4.44) is exposure-CONSISTENT
but non-mechanism-driven and noisy. In parallel, W122 KILLED the strongest non-reflexion
mechanism (M3) NIM-free — ICPC's SECRET token-diff grading denies M3 its expected/actual
differentiator (`m3_exclusive_signal_fraction = 0.000`) ⇒ the ceiling is **mechanism-robust**,
not merely reflexion-specific — and confirmed the stronger-model gate is **structurally
closed** (no reachable stronger-than-Maverick model has a primary-KNOWN cutoff ≤ ~Aug-2024).
**W89 (+5.56) + W105 (+7.00) remain the only two retirements;** W122 adds none (every non-FAIL
seed is non-mechanism-driven). Decision CID `258b6ed7` invariant. COO-9 lead; W123 =
B4-after-3rd-seed ⇒ accept the bounded HumanEval-family ceiling OR escalate to larger n PER
FIELD (≥100/field; not more n=30 seeds; no 4th). See
`docs/RESULTS_W122_PAIRED_SEED_CLOSURE_V1.md` + `docs/CONTAMINATION_CONTROL_FRAMING_W122_V1.md`.

## Post-W122 update — W123 (battlefield supply-capped) + W124 (transformer-native arsenal signal-poor) — still exactly two retirements

W122's "unresolvable at n=30" set up two honest follow-ups, both NIM-free and both
ending **$0** with **no third retirement**. **W123** proved the obvious escalation
(≥100 problems PER FIELD) is **supply-UNREACHABLE** from the official
`github.com/icpc` family: the resistant (post-cutoff) side is hard-capped at 51
raw / ~45 tier-1 (exactly 4 post-cutoff package surfaces, all already mined by
W120; no 5th), so the n=30 caveat is **post-cutoff-supply-bound, not
method-bound**. **W124** then stopped treating the saturated battlefield as the
only lever and, for the first time, ran the repo's **transformer/substrate/
learned-memory arsenal** against the code-closure problem locally. On the only
locally-loadable real transformer (`distilgpt2`, a tiny general LM — no
code-competent model is available), an AST-boundary **hidden-state probe** over
1,570 official-grader-labelled Maverick ICPC generations scored AUC **0.6345 ≈**
the surface-feature baseline **0.6343** (Δ=+0.0001 ≪ +0.05): the transformer's
hidden state adds **nothing** beyond surface confounds for code-correctness
(`M4_CLOSE_BLIP_NOT_A_GAIN`), so the transformer-native line is **blocked at the
precursor** by **local code-model-encoder supply** (the model-axis sibling of
W123's battlefield-supply cap) — not a refutation of the idea. The learned-memory
controller line scored **at chance** (0.502) on the reflexion-rescue decision
(`TOO_SYNTHETIC_NOT_WARRANTED`). No hosted Maverick probe was earned. The
stronger-model gate is unchanged (`NO_CERTIFIABLE_STRONGER_MODEL`; decision CID
`258b6ed7` invariant). **W89 (+5.56) + W105 (+7.00) remain the only two
retirements;** W123 + W124 add none. COO-9 lead; W125 = re-test the M4 precursor
if a code-competent local model becomes loadable / accept the bounded ceiling +
encoder-supply limitation / a primary-KNOWN reachable stronger model. See
`docs/RESULTS_W123_LARGEN_SUPPLY_CENSUS_V1.md` +
`docs/RESULTS_W124_TRANSFORMER_NATIVE_CODE_INTERVENTION_V1.md`.

## Post-W124 update — W125 (hosted controller arsenal is REAL but the resistant field is generation-capped) — still exactly two retirements

W124 localised the local-encoder blocker; **W125** mined the **third, unused
lever** — the hosted/controller arsenal — and asked the sharpest version of the
question: *can our own hosted controller stack beat same-budget self-consistency
(A1) on the official resistant ICPC family where reflexion (B) failed?* It built
`coordpy.controller_native_code_mechanism_v1`, the first module to bridge the
hosted-controller stack (`hosted_router_controller_v12` / `hosted_logprob_router_v12`
/ `hosted_cache_aware_planner_v12`) to the audited `tool_call_substrate_v1` plane
and the `executor_grounded_patcher_v1` digest, promoting the W124 M6
PATCH/REPLAN/ABSTAIN contract into an executable C1/C2/C3 slate. **On the mechanism
question the answer is YES**: the pre-committed structural fake-different test bites
(reflexion B and a deliberately-degenerate C0 control classify `FAKE_DIFFERENT`;
C1/C2/C3 classify `REAL`; lead C3) and all four NIM-free contract checks pass
(audit-chain re-hash + tamper, idempotency, never-reads-secret, routing
determinism, same-budget) — the controller stack is a real new mechanism on code,
not fake-different. **On the spend question the answer is NOT EARNED**: a $0 replay
over the 330 already-paid Maverick generations on the W120 resistant 30-slice,
re-graded on the official secret/sample cases, finds A1 = **7/30**, the **entire**
generation pool reaches only **8/30** (oracle ceiling +1, inside the ±3.34 null
band), and a hidden-test-blind controller (C2 selection + C3 digest-routed walk)
recovers **zero** A1-fail problems (`blind_selection_headroom = 0`). The two
diagnostics that explain it: reflexion is **stuck** on **23/30** problems (a
digest-router would diverge) but **10** generations pass all public samples yet
fail the hidden cases (the only blind in-loop signal is **non-discriminating**) —
so there is **nothing to route to**. The resistant field is **generation-capped**
for $0 controller re-routing (`W125-L-RESISTANT-GENERATION-CAP`, the mechanism-lever
sibling of W123's battlefield-supply cap and W124's local-encoder-supply cap); the
controller's value lives only in **new trajectories** on the 22 uniformly-unsolved
problems, unfundable without a precursor signal. The fresh hosted pilot is therefore
**not earned** (`$0 NIM`); the exposed control is not bought (resistant-first). The
stronger-model gate is unchanged (`NO_CERTIFIABLE_STRONGER_MODEL`; decision CID
`258b6ed7` invariant). **W89 (+5.56) + W105 (+7.00) remain the only two
retirements;** W125 adds none. COO-9 lead; W126 = accept the bounded ceiling + the
generation-cap / an operator-greenlit fresh controller pilot on the unsolved
problems (NOT precursor-earned) / a code-competent local model / a primary-KNOWN
stronger model. See `docs/RESULTS_W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1.md` +
`docs/RUNBOOK_W125.md`.
