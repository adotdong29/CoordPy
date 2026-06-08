# RESULTS W140 — primary-source research (Lane γ)

**Status: research lane, wired to design.** Four parallel primary-source mining lanes (arXiv / OpenReview / official venue pages only; no blogs/secondary). Every cited finding is WIRED to a W140 tutor format, a leak-gate assertion, the field/band rule, or an architecture requirement — no literature-summary-as-output. All papers were verified-fetched by the mining agents; unverifiable leads are flagged as such and NOT cited as load-bearing.

## A. Teaching / explanation compression for weaker models — does compiling the witness into a tutor help?

The W140 thesis (compile the strong witness into a weak-tier-usable teaching object) is supported, with sharp constraints on HOW:

- **Small Model Learnability Gap — arXiv:2502.12143.** Models ≤3B do NOT benefit from long/complex CoT distilled from a strong model; they do better on shorter, simpler, capacity-matched reasoning. **Wired:** the tutor must DOWN-COMPILE the witness to a short form (the TC3 compressed tutor); the bench A/B-tests raw-witness `C0` vs the short tutor arms on the weak tier (predict `C0` ties-or-hurts, short tutor helps).
- **Small Models Need Strong Verifiers to Self-Correct — arXiv:2404.17140 (ACL Findings 2024).** Small models REFINE well once an error is flagged but FAIL at VERIFICATION (they cannot self-judge whether a fix is correct). **Wired (decisive):** the tutor CARRIES the execution-oracle routing (TC2 witness→rewrite keyed on the observed TLE/WRONG_ANSWER), it does not merely name a technique; this also explains the W125–W129 selection cap from the inside.
- **SuperCorrect (thought-template distillation) — arXiv:2410.09008 (ICLR 2025).** A large teacher's hierarchical thought templates (high-level + detailed) lift a 7B student +7.8% MATH / +5.3–15.1% GSM8K. **Wired:** the W140 family card is two-layer (technique NAME + a worked holed SKELETON), validated at 7B scale — the win is plausibly reachable.
- **GPT-generated programming hints for novices — arXiv:2404.02213 (CHI 2024 LBW).** High-level natural-language hints ALONE are "helpless or even misleading" for weaker learners; adding concrete code examples helps. **Wired:** principle-only TC1 is predicted to underperform; the bench includes TC2 (card + worked skeleton) to measure the marginal value of the scaffold.
- **PGS minimal property-oriented feedback — arXiv:2506.18315** and **FeedbackEval — arXiv:2504.06939.** Minimal property+counterexample feedback beats verbose (up to +13.4% pass@1); "mixed" feedback (test signal + expert explanation) beats either alone. **Wired:** TC3 minimality knob + TC2 fuses the execution witness with the technique.
- **RLAD reasoning abstractions — arXiv:2510.02263.** Concise NL procedural abstractions improve generalization to harder problems; spending test compute on abstractions can beat more solution samples. **Wired:** validates the compiler(=abstraction-generator)/8B(=solution-generator) split; the tutor is a procedural abstraction.
- **CoT-distillation capacity gap — arXiv:2604.08880.** CoT distillation often DEGRADES the student below its own pre-distillation baseline (prior work hid this by comparing only post-distillation numbers). **Wired:** the W140 non-negativity gate is correctly the PRIMARY gate and must reference the 8B's OWN no-tutor baseline (A1), flagging any sub-baseline arm as harm — exactly what `T4`'s KEEP does.
- **AlgoSimBench — arXiv:2507.15378.** LLMs are bad at recognizing shared algorithmic technique under surface variation; matching by solution attempts beats matching by problem text. **Wired:** the tutor's family/observed-kind routing is derived from the owned-oracle FAILURE MODE (discriminator), not the problem statement.

## B. Teaching vs answer-leakage — assertions for the tutor-leak gate

- **Hint-leakage detectors — arXiv:2510.21087.** Exact-match leak detection has recall ~0.075 (misses indirect leaks); an LLM-judge has precision ~0.266 (noisy). **Wired (decisive):** the authoritative leak signal is BEHAVIORAL — grading on a DISJOINT hidden bank — not text inspection; the W140 grader stays `grade_on_secret_v1` on disjoint secret cases, and the text gate is corroboration. Normalize tokens before comparison (the gate lowercases/strips to alphanumeric tokens).
- **EAL — arXiv:2402.02823** and **LLM-Decontaminator — arXiv:2311.04850.** Text-based leak detectors are defeatable by mild rephrasing; n-gram decontamination misses 8–18% of semantic overlap. **Wired:** no purely-text detector is trusted as final; the decisive guarantee is the disjoint hidden grade; the gate uses the contiguous-token-run (semantic-robust to restyle) tripwire AND a discriminator-presence check, NOT raw string match.
- **Leak-Judge dual gate — arXiv:2601.14560.** A tutor passes only if it is BOTH non-leaking AND helpful; forbid "full answer upfront / all steps in one message." **Wired:** the W140 gate pairs `tutor_leak_gate_v1` (non-leak) with `tutor_is_genuinely_new_v1` (carries real teaching) — a content-free decoration fails the latter (T6), a discriminator-bearing card fails the former.
- **SWE-Bench Illusion — arXiv:2506.12286.** Consecutive-5-gram verbatim overlap is a memorization/leak diagnostic (35% seen vs 18% unseen). **Wired:** the gate's `MIN_DISCRIMINATOR_RUN_TOKENS=5` contiguous-run check on the discriminating expression is exactly this diagnostic, applied to the ANSWER-distinguishing logic.
- **GSM-Symbolic — arXiv:2410.05229.** Template/instance invariance: a teaching object must not embed instance constants. **Wired:** the gate's `public_only_literals` (no secret-only numeric literal) + `template_invariant` (no long verbatim run with THIS statement) + `discriminator_shape_only` (no concrete array literal).

**The 7 leak-gate assertions** (`tutor_leak_gate_v1`): (1, DECISIVE) `no_discriminator_leak` — the tutor never contains the discriminating expression (`spec.correct_fill`) as a ≥5-token run; (2, DECISIVE) `holes_are_substantive` — the tutor's own skeleton trivially stubbed FAILS public (and a hole-free "skeleton" is rejected); (3) `no_reference_paste` — bounded gross verbatim run vs `ref_source`; (4) `public_only_literals`; (5) `is_procedure_not_answer`; (6) `one_liner_family_ok`; (7) `discriminator_shape_only` + `template_invariant`. Decisive guarantee remains behavioral (disjoint hidden grade).

## C. Procedural generation + per-model difficulty calibration — the field & band rule

- **DynaCode — arXiv:2503.10452.** Procedural code drop scales inversely with model strength (Llama-3.1-8B −45.7pp vs GPT-4o −16.8pp); a single global difficulty is uninformative for some tier. **Wired:** the per-tier band is MANDATORY (the 8B is fielded at its own ≈0.5 knob), exactly the W139 R4′ construction W140 reuses.
- **Confident Rankings (continuous-score CAT) — arXiv:2601.13885** and **PSN-IRT — arXiv:2505.15055.** IRT variance / Fisher information is MAXIMAL at pass≈0.5; discrimination COLLAPSES at difficulty extremes (saturation ⇒ no discrimination). **Wired:** this is the theoretical proof behind the per-tier ≈0.5 band, and the mechanism behind the W139 "mid +0 = saturated on complexity"; W140 keeps the per-tier band that avoids the discrimination collapse.
- **Memorize or Generalize? — arXiv:2503.02296.** Structural code-rewriting breaks memorization; surface paraphrase does NOT. **Wired:** the resistant-by-construction structural minting (fresh seeds, knob-parameterized factories) is the right resistance lever; paraphrase-only would not be.
- **Self-Taught Self-Correction — arXiv:2503.08681** and **The Valley of Code Reasoning — arXiv:2510.06101.** The winning self-correction/curriculum mechanism DIFFERS by capability (weak needs broad search, strong needs refinement); a hard-problem curriculum that helps a strong model can HARM a weak one (−50% dip; 7–11% gains on hard vs 33–41% on easy/medium). **Wired (the central caveat):** a strong-tier tutor may be the WRONG mechanism for the weak tier and may null/harm by construction — which is precisely W138's 8B −25. W140's honest prior is that cross-tier transfer may FAIL; the capability-matched controller (`T4`) KEEPs where the tutor does not lift usability, so the floor is non-negativity, and a weak-tier FAIL is a likely, registrable outcome.
- **Code2Bench PBT — arXiv:2508.07180.** Property-based testing synthesizes a hidden oracle and catches 7–21% false-passes. **Wired:** corroborates the exact-oracle disjoint-hidden-grade discipline W140 already uses (the witness probes are byte-disjoint from secret cases).

## D. Stronger-model cutoff disclosure re-check — gate `258b6ed7` UNCHANGED

Verified from PRIMARY official sources (vendor HF cards / official blogs / official technical-report PDFs):

| Model | Official source | Cutoff disclosed? | Verdict | Changed? |
|---|---|---|---|---|
| Llama-4 Maverick | meta-llama HF card + llama.com | YES — "cutoff of August 2024" | KNOWN (Aug-2024) | no |
| Qwen3-Coder-480B-A35B | Qwen HF card + blog + arXiv 2505.09388 | no | UNKNOWN | no |
| DeepSeek-V4-pro | official DeepSeek V4 model-card PDF (full read) | no (data categories only) | UNKNOWN | no |
| Mistral-Small-4-119B-2603 | Mistral docs + HF + official news | no | UNKNOWN | no |
| GLM-5 | zai-org HF card + arXiv 2602.15763 | no | UNKNOWN | no |

**Gate confirmed UNCHANGED: `{KNOWN:1 (Maverick), UNKNOWN:4 (Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4, GLM-5)}`, decision CID `258b6ed7`.** New observation (recorded, not gate-opening): **Gemma 4** now carries a primary-disclosed Jan-2025 cutoff (ai.google.dev), but AT (not before) the ~Jan-2025 boundary and with code-competence-vs-the-bench unestablished, so it does not open a certifiable-stronger-model path for a resistant benchmark dated after a pre-Jan-2025 cutoff. All other 2025-data frontier models (GPT-5.x, Claude 4.x, Gemini 2.5+/3) disclose cutoffs of Jan-2025 or LATER, i.e. not earlier than the reachable resistant instrument's frontier.

## Net effect on W140 design

1. The win is **plausibly reachable** (SuperCorrect lifts a 7B with hierarchical templates) **but** the tutor must be **two-layer (name + worked skeleton), carry the execution oracle, and be compressed** — built exactly so (TC1/TC2/TC3).
2. The **honest prior is FAIL-leaning**: three code/reasoning results say a strong-tier mechanism can be wrong-for / harm the weak tier; the capability-matched `T4` makes non-negativity the floor and a weak-tier teaching-compilation cap a clean, expected outcome.
3. The **leak gate is behavioral-decisive** (disjoint hidden grade) with text/structure corroboration (discriminator-presence + hole-substance), not defeatable string-match.
4. The **per-tier band is theoretically grounded** (Fisher info maximal at 0.5; discrimination collapses at extremes) — reused verbatim from W139.
5. The **stronger-model gate stays closed** (`258b6ed7`), so no stronger-model frontier spend is eligible; the frontier target stays `meta/llama-3.3-70b-instruct`.
