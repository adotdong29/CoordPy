# W94 cross-modal battlefield scouting (no NIM runs)

> 2026-05-24.  Documentation-only scouting analysis of
> candidate multimodal benchmarks for W95+ cross-modal team
> retirement attempts.  No new NIM calls; uses **published
> SOTA VLM scores** + W88–W92 evidence on the current
> battlefield's saturation.

## Why HumanEval-Visual K=5 is retired as a serious cross-modal battlefield

W88 → W92 produced 7 cross-modal configurations across 3
architecture families on HumanEval-Visual K=5 with Llama-3.2-
{11B, 90B}-Vision-Instruct.  All 7 LOST to unified-VLM K=5
on the team-organisation direction:

| Architecture family | Best B − A1_vlm | Best per-seed |
|---|---:|---:|
| Split (VLM-extract + code-LM) | −5.56 pp | 0 / 3 (W88 V1) |
| VLM-in-loop | +0.00 pp (W90 P2 doctest tie) | 2 / 3 (W91 P2 3-seed; disconfirmed) |
| Role-specialized (W92) | −10.71 pp | 0 / 7 |

The aggregate signal is that the unified-VLM K=5 baseline on
HumanEval-Visual reaches **88–92 %** at the all_docstring /
doctest_only regimes — i.e., the failure-residual for B's
multi-agent structure to rescue is only 8–12 %.  Even a strong
60 % rescue rate within the residual gives a +5–7 pp ceiling,
which is exactly where W90/W91 VLM-in-loop has landed at peak.
And W91 P2b's 7-seed disconfirmation shows the 3-seed +2.78 pp
is variance, not signal.

**Conclusion**: HumanEval-Visual K=5 is no longer a credible
battlefield for retiring `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
under the W88 retirement bars (+5 pp margin + per-seed
majority).  W95+ cross-modal work must move to a benchmark
where unified-VLM K=5 does NOT approach ceiling.

## Candidate battlefields (published SOTA reference)

| Benchmark | Subset / format | Approx. SOTA single-shot VLM | Headroom estimate | Executor signal | Selection |
|---|---|---:|---:|---|---|
| **MathVista** | testmini (1000 problems; mixed multi-choice + numeric) | ~55–65 % (frontier VLMs 2024) | **35–45 %** | Partial (numeric answer-match for some subsets; multi-choice exact match) | **TOP CANDIDATE** |
| **MMVet** | 218 problems, 6 sub-tasks, GPT-4-judge grading | ~50–70 % | 30–50 % | NO clean executor (judge-dependent); skip for fair benching | Skip — judge dependency violates W88 anti-cheat shape |
| **ChartQA** | 32k Q&A on charts; numeric or short-text answers | ~70–85 % (frontier VLMs) | 15–30 % | Clean executor (exact / relaxed numeric match) | Strong secondary |
| **DocVQA** | 50k doc QA; short-text answers | ~75–90 % | 10–25 % | Clean executor (ANLS string match) | Lower priority (high ceiling) |
| **TextVQA** | 45k Q&A on scene-text | ~70–80 % | 20–30 % | Clean executor (string match) | Secondary |
| **SEED-Bench** | 19k Q&A, multi-choice only | ~70–80 % | 20–30 % | Clean executor (multi-choice) | Lower priority |
| **RealWorldQA** | 765 problems, multi-choice + free-form | ~60–75 % | 25–40 % | Mixed; partial clean | Secondary |

## Why MathVista is the recommended W95 battlefield

1. **Lowest unified-VLM K=5 ceiling among clean-executor candidates.**
   Frontier VLMs (GPT-4o, Claude-4-Vision, Gemini-2.5-Vision)
   reach ~60-65 % single-shot on MathVista-testmini.  At K=5
   the ceiling rises to maybe 75–80 %, leaving ~20–25 % failure-
   residual — 2–3× the headroom of HumanEval-Visual.
2. **Multi-modal structure rewards team decomposition.**
   MathVista problems pair an image (chart / diagram / geometric
   figure) with a math question.  A vision-extractor agent
   parsing the image into structured data (numbers, shapes,
   relations) → a math-solver agent computing the answer is a
   plausible team decomposition with non-trivial advantage over
   unified VLM single-shot.
3. **Public + reproducible.**  Dataset is on HuggingFace
   (`AI4Math/MathVista`); SOTA scores are widely published;
   the testmini subset (1000 problems) is the canonical
   small-eval slice.
4. **Executor friendly.**  Numerical answer match (with
   tolerance) for math-result problems; exact-match for
   multi-choice.  No GPT-4-judge dependency.
5. **Cross-modal benchmark with a 'natural' architecture**:
   VLM-Geometry-Reader + Code-Math-Solver is a textbook
   pipeline that is hard to express as a single unified VLM
   forward.

## W95+ infrastructure that would be needed

Building MathVista bench in CoordPy would require:

1. **Corpus loader** for `AI4Math/MathVista` testmini (load via
   HuggingFace datasets or direct JSON; SHA-anchor the corpus).
2. **Answer-match executor** with tolerance (atol/rtol for
   numerical; exact for multi-choice; canonical normalization
   for free-form short answers).
3. **Per-problem schema** mirroring W88 cross-modal:
   {task_id, image_bytes, question_text, gold_answer,
   answer_type, stripped_prompt}.
4. **Cross-modal team architecture** specifically designed for
   math + vision:
   - VLM-Vision-Reader: extracts numerical / geometric content
     from the image into structured text.
   - Code-Math-Solver: receives the structured extraction +
     question; computes the answer (possibly via running Python).
   - Optional VLM-Verifier: cross-checks the candidate answer
     against the image.

The corpus loader + executor is ~1–2h of build work.  The
team architecture inherits most of its structure from W88/W92
cross-modal benches.

## W94 cross-modal recommendation

* **DO NOT** launch any cross-modal NIM run in W94.
* **DO** select MathVista (or ChartQA as secondary) as the W95
  battlefield.  Document this selection NOW so W95 is not
  delayed by a 1-2 hour build-from-scratch sprint.
* **DO** preserve all W88–W92 cross-modal evidence as the
  canonical "HumanEval-Visual K=5 is empirically the wrong
  battlefield" body of work; it is not erased by the W95
  pivot, only scoped.

## Honest framing

The W94 cross-modal deliverable is **a documented battlefield
selection, not a benchmark result**.  The W93 discipline says:
"do not pay full benchmark price on a presumptively-hostile
battlefield".  W94 obeys that discipline by:

1. Refusing to launch another HumanEval-Visual K=5 run.
2. Documenting WHY HumanEval-Visual K=5 is retired as a
   serious cross-modal retirement battleground (W88–W92
   evidence).
3. Selecting MathVista as the W95 battlefield with a clear
   structural rationale.
4. Recording the W95 infrastructure that needs to be built.

The W88-L / W87-L cross-modal carry-forwards STAY in W94.
The W95+ research direction is now scoped to MathVista or
analogously low-ceiling multimodal benchmarks; no new W94
data is required to make this commitment honest.
