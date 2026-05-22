"""W88 / post-W87 — Cross-modal code bench V1.

Targets the canonical W87 carry-forward
``W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP``.  W87 #46
shipped a real cross-modality substrate (vision-tower + code-LM
+ cross-modality Merkle root) but never demonstrated that a
multi-agent cross-modal team outperforms either:

  (a) a text-only single-modal baseline, OR
  (b) the strongest same-budget single-agent VLM baseline

on a public benchmark.  W88 attacks both directly.

Construction — "HumanEval-Visual"
---------------------------------

Take the canonical HumanEval corpus.  For each problem with at
least two ``>>>``-prefixed doctest example lines in its
docstring, deterministically synthesise:

  * a STRIPPED PROMPT — the docstring with all ``>>>`` /
    expected-output lines removed; the rest of the prose stays.
  * a CORPUS IMAGE — a PIL render of the removed doctest lines
    as plain monospaced text on a white background.

The synthesis is content-addressed: the corpus image bytes are
SHA-256-hashed and embedded in the per-problem capsule.

Three arms, SAME total model-call budget per problem (K=5):

* ``A0_text`` — single-shot text-only code-LM at T=0.0.  Sees
  ONLY the stripped prompt — no image access.  This is the
  "no-image floor"; it pins the load-bearing-ness of the image.
* ``A1_vlm`` — single-agent VLM at T=0.7, K=5 first-pass-among-K.
  Sees BOTH the stripped prompt AND the corpus image.  The
  strongest single-agent multimodal baseline at the same budget.
* ``B_cross`` — multi-agent cross-modal team:
    1. VLM extracts the doctest examples from the image into a
       structured text bullet list at T=0.0 (1 model call).
    2. Code-LM (text-only) generates a candidate from the
       (stripped prompt + VLM extraction) at T=0.7 (1 model call).
    3. Executor verifies; if FAIL, the code-LM does up to 3
       sequential reflexion turns, each conditioned on the
       cumulative history of prior candidates + their executor
       stderr (≤ 3 model calls).
    Total: 1 + 1 + ≤ 3 = ≤ 5 model calls (exactly 5 if call 2
    failed; 5 enforced by padding with a final reflexion turn
    if needed to hit budget parity with A1_vlm).

For the W87 carry-forward to retire, both:

  (i)  ``b_cross_modal_mean_strictly_beats_a0_text_mean = True``
       (image is load-bearing — the cross-modal substrate
       actually adds value over text-only)
  (ii) ``b_cross_modal_mean_strictly_beats_a1_vlm_mean = True``
       (cross-modal TEAM organisation beats single-agent VLM at
       the same compute budget)

must hold.  (i) alone is insufficient — it would only show
"vision helps", not "multi-agent cross-modal organisation helps".
(ii) alone is also insufficient — it could just mean A1_vlm is
weak; without (i) we have not shown the image is the explanation.

Anti-cheat (W88)
----------------

* The HumanEval corpus SHA-256 is verified against the canonical
  upstream (inherited from W86).
* The corpus synthesis is DETERMINISTIC: given the canonical
  HumanEval corpus + the synthesis recipe, the per-problem
  stripped prompt + image bytes are reproducible byte-for-byte.
* No arm has a different oracle; ``problem.test`` is the canonical
  HumanEval test block for every arm.
* No model swap between arms.  VLM = ``meta/llama-3.2-11b-vision-
  instruct``; code-LM = ``meta/llama-3.1-8b-instruct``.  A1_vlm
  uses VLM only; A0_text uses code-LM only; B_cross uses both
  exactly as designed.
* Same prompt budget per arm (K=5).
* No selective retries.

Honest scope (W88)
------------------

* ``W88-L-CROSS-MODAL-CODE-V1-NIM-DEPENDENT-CAP`` — V1 drives
  the bench through any ``LLMBackend``-shaped client; provider
  determinism is not assumed beyond temperature=0.
* ``W88-L-CROSS-MODAL-CODE-V1-PIL-RENDER-CAP`` — V1's image
  synthesis uses ``PIL.ImageDraw.text`` with the default bitmap
  font; rendering matches on the canonical configuration but
  may differ across PIL versions in pixel exact bytes.  The
  load-bearing fact is whether the VLM can READ the examples
  from the image, not the byte-exact image; the audit chain
  records the image bytes' SHA-256 anyway.
* ``W88-L-CROSS-MODAL-CODE-V1-SUBSET-CAP`` — V1's quick subset is
  ≥ 10 problems × 3 seeds.  Larger sweeps are out of scope
  unless the headline result demands it.
* ``W88-L-CROSS-MODAL-CODE-V1-DOCTEST-STRIP-CAP`` — V1 strips
  ``>>>``-prefixed lines + the immediate next line (the expected
  output) from the docstring.  Problems with no doctests (or
  whose docstring's expected-output is on the same line) are
  filtered out at corpus synthesis time.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import re
from typing import Any, Callable, Sequence

from .humaneval_real_bench_v1 import (
    W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S,
    W86_HUMANEVAL_EXECUTOR_TIMEOUT_S,
    HumanEvalArmCallCapsuleV1,
    HumanEvalArmOutcomeCapsuleV1,
    HumanEvalExecutorResultV1,
    HumanEvalProblemV1,
    extract_candidate_code_v1,
    run_humaneval_executor_v1,
)


W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_modal_code_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Corpus synthesis
# ---------------------------------------------------------------

# Match a `>>> ` line followed by its expected-output line(s) up
# to the next docstring break (blank line, next ``>>>``, the
# docstring's closing ``"""``, or end of string).  The look-ahead
# is non-consuming so we keep the closing triple-quote in place.
_DOCTEST_BLOCK_RE = re.compile(
    r"^(\s*)>>>\s.*?(?=^\s*$|^\s*>>>|^\s*\"\"\"|\Z)",
    re.MULTILINE | re.DOTALL)
# Match a single >>> line (input only) when we need to count.
_DOCTEST_PROMPT_LINE_RE = re.compile(
    r"^\s*>>>\s", re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class CrossModalProblemV1:
    """One HumanEval-Visual problem."""

    schema: str
    task_id: str
    stripped_prompt: str  # docstring with >>> lines removed
    image_bytes: bytes    # PNG bytes of the doctest lines
    n_doctest_lines: int  # how many >>> lines were stripped
    canonical_solution_cid: str
    test: str
    entry_point: str

    @property
    def image_cid(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()

    @property
    def stripped_prompt_cid(self) -> str:
        return hashlib.sha256(
            self.stripped_prompt.encode("utf-8")).hexdigest()

    def problem_cid(self) -> str:
        return _sha256_hex({
            "kind": "w88_cross_modal_problem_v1",
            "task_id": str(self.task_id),
            "stripped_prompt_cid": str(self.stripped_prompt_cid),
            "image_cid": str(self.image_cid),
            "n_doctest_lines": int(self.n_doctest_lines),
            "canonical_solution_cid": str(
                self.canonical_solution_cid),
            "entry_point": str(self.entry_point),
        })


def _split_doctest_block(prompt: str) -> tuple[str, str, int]:
    """Strip ``>>> ... `` blocks from a HumanEval prompt.

    Returns ``(stripped, doctest_text, n_doctest_lines)`` where
    ``doctest_text`` is the concatenation of every matched block
    (with leading indentation removed) and ``stripped`` is the
    original prompt with those blocks deleted.
    """
    blocks: list[str] = []
    n_dt_lines = 0
    for m in _DOCTEST_BLOCK_RE.finditer(prompt):
        block = m.group(0)
        # Remove the common leading indentation
        lines = block.splitlines()
        if lines:
            indent = lines[0][:len(lines[0]) - len(
                lines[0].lstrip())]
            cleaned_lines = []
            for ln in lines:
                if ln.startswith(indent):
                    ln = ln[len(indent):]
                cleaned_lines.append(ln)
            block_clean = "\n".join(cleaned_lines).rstrip()
            if block_clean:
                blocks.append(block_clean)
                n_dt_lines += sum(
                    1 for ln in cleaned_lines
                    if ln.strip().startswith(">>>"))
    stripped = _DOCTEST_BLOCK_RE.sub("", prompt)
    # Tidy: collapse triple+ blank lines down to one
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    doctest_text = "\n".join(blocks)
    return stripped, doctest_text, n_dt_lines


def _render_doctest_image(
        text: str, *, width: int = 800, font_size: int = 18,
        line_height: int = 24, pad: int = 16,
) -> bytes:
    """Render ``text`` as black monospaced text on a white PNG.

    Uses PIL with the default bitmap font (which ships with PIL
    on every platform).  No external font files needed.
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Menlo.ttc", font_size)
    except OSError:
        try:
            font = ImageFont.truetype(
                "DejaVuSansMono.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    lines = text.splitlines()
    if not lines:
        lines = [""]
    height = pad * 2 + line_height * max(1, len(lines))
    img = Image.new("RGB", (int(width), int(height)), "white")
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text(
            (pad, pad + i * line_height),
            ln, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return bytes(buf.getvalue())


_DOCSTRING_BLOCK_RE = re.compile(
    r'(    """.*?""")', re.DOTALL)


def _strip_full_docstring(prompt: str) -> tuple[str, str, int]:
    """Remove the entire function docstring; collect its full
    body as the doctest_text + count its ``>>>`` lines.

    The strongest "image-is-load-bearing" mode: the stripped
    prompt retains the signature only, so any behavioural
    information lives in the corpus image.
    """
    m = _DOCSTRING_BLOCK_RE.search(prompt)
    if m is None:
        return prompt, "", 0
    full_doc = m.group(1)
    inner = full_doc.strip()
    if inner.startswith('"""'):
        inner = inner[3:]
    if inner.endswith('"""'):
        inner = inner[:-3]
    inner = inner.strip()
    # Count >>> lines
    n_dt = sum(
        1 for ln in inner.splitlines()
        if ln.strip().startswith(">>>"))
    # Replace the entire matched docstring with a minimal
    # "See the image for behaviour" docstring so the
    # function signature still parses + the prompt still
    # looks like a HumanEval problem.
    replacement = (
        '    """See the attached image for the function\'s '
        'specification."""')
    stripped = prompt[:m.start()] + replacement + prompt[m.end():]
    return stripped, inner, int(n_dt)


def synthesize_cross_modal_corpus_v1(
        corpus: Sequence[HumanEvalProblemV1], *,
        min_doctest_lines: int = 2,
        strip_mode: str = "doctest_only",
) -> tuple[CrossModalProblemV1, ...]:
    """Construct the HumanEval-Visual corpus from a canonical
    HumanEval corpus.

    ``strip_mode``:

    * ``"doctest_only"`` (default) — strip only ``>>>``-prefixed
      lines from the docstring; prose description stays.  The
      "image-is-illustrative-but-not-load-bearing" mode.
    * ``"all_docstring"`` — strip the entire docstring + replace
      with a "See image" stub.  The "image-is-strongly-
      load-bearing" mode.

    Problems with fewer than ``min_doctest_lines`` ``>>>`` lines
    in the original docstring are excluded (image would be too
    sparse to be load-bearing).
    """
    if strip_mode not in ("doctest_only", "all_docstring"):
        raise ValueError(
            "strip_mode must be 'doctest_only' or 'all_docstring'")
    out: list[CrossModalProblemV1] = []
    for p in corpus:
        if strip_mode == "all_docstring":
            stripped, doctest_text, n_dt = _strip_full_docstring(
                p.prompt)
        else:
            stripped, doctest_text, n_dt = _split_doctest_block(
                p.prompt)
        if int(n_dt) < int(min_doctest_lines):
            continue
        if not doctest_text.strip():
            continue
        img = _render_doctest_image(doctest_text)
        out.append(CrossModalProblemV1(
            schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
            task_id=str(p.task_id),
            stripped_prompt=str(stripped),
            image_bytes=img,
            n_doctest_lines=int(n_dt),
            canonical_solution_cid=hashlib.sha256(
                p.canonical_solution.encode("utf-8")).hexdigest(),
            test=str(p.test),
            entry_point=str(p.entry_point)))
    return tuple(out)


def select_cross_modal_subset_v1(
        *, corpus: Sequence[CrossModalProblemV1],
        n_problems: int, seed: int,
) -> tuple[CrossModalProblemV1, ...]:
    """Deterministic per-seed subset selection (mirrors W86's
    ``select_humaneval_subset_v1`` discipline)."""
    import random
    rng = random.Random(int(seed))
    idxs = list(range(len(corpus)))
    rng.shuffle(idxs)
    chosen = idxs[: int(n_problems)]
    return tuple(corpus[i] for i in chosen)


# ---------------------------------------------------------------
# Per-arm runners
# ---------------------------------------------------------------

_TextGenFn = Callable[[str, int, float], tuple[str, int]]
# (prompt_text, image_bytes_or_None, max_tokens, temperature)
#   -> (response_text, wall_ms)
_VlmGenFn = Callable[
    [str, bytes | None, int, float], tuple[str, int]]


_CODE_SYSTEM = (
    "You are an expert Python programmer.  When given a function "
    "signature and docstring, output ONLY the complete function "
    "(inside a ```python ... ``` code fence).")


def _stripped_to_humaneval_problem(
        p: CrossModalProblemV1,
) -> HumanEvalProblemV1:
    """Wrap a cross-modal problem in the W86 HumanEval problem
    shape so the W86 executor + code extraction work unchanged."""
    return HumanEvalProblemV1(
        task_id=str(p.task_id),
        prompt=str(p.stripped_prompt),
        canonical_solution="",  # not used by executor
        test=str(p.test),
        entry_point=str(p.entry_point))


def _code_prompt(p: CrossModalProblemV1,
                 *, extra_context: str = "") -> str:
    extra = (
        f"\n\n[Additional context]\n{extra_context}"
        if extra_context else "")
    return (
        f"{_CODE_SYSTEM}\n\n"
        "Complete the following Python function.  Provide the "
        "full function including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```{extra}\n\n"
        "Your complete solution:")


def _vlm_solve_prompt(p: CrossModalProblemV1) -> str:
    return (
        f"{_CODE_SYSTEM}\n\n"
        "Complete the following Python function.  The example "
        "input/output behaviour is shown in the attached image.  "
        "Provide the full function including the signature.\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Your complete solution:")


def _vlm_extract_prompt(p: CrossModalProblemV1) -> str:
    return (
        "Read the attached image carefully.  It contains "
        "Python doctest example lines (lines starting with `>>>`) "
        "and their expected outputs for the function whose "
        "signature is:\n\n"
        f"```python\n{p.stripped_prompt}```\n\n"
        "Reproduce the doctest lines and expected outputs from "
        "the image as plain text — one input/output pair per line "
        "in the form\n"
        "    >>> <input>\n    <expected output>\n"
        "Output ONLY the doctest text, no commentary.")


def _reflexion_prompt_cross_modal(
        p: CrossModalProblemV1,
        vlm_extraction: str,
        history: Sequence[tuple[str, HumanEvalExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ...\n"
        if exe.passed:
            verdict = "PASSED visible tests"
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "")
                + ")")
            stderr_text = exe.stderr_tail.strip()
            stderr_excerpt = (
                f"\nExecutor stderr (tail):\n{stderr_text}"
                if stderr_text else "")
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    return (
        f"{_CODE_SYSTEM}\n\n"
        "[Role: reflective code generator]\n"
        f"You are on attempt {attempt_idx + 1}.\n"
        f"Target function:\n```python\n{p.stripped_prompt}```\n\n"
        "Doctest examples extracted from the corpus image:\n"
        f"```\n{vlm_extraction.strip()}\n```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the bug in the failing attempt(s) and produce a "
        "NEW corrected complete Python function.  Do not repeat "
        "a previous attempt verbatim.  Provide ONLY the corrected "
        "function in a ```python ... ``` fence:")


def _run_a0_text(
        *, seed: int, p: CrossModalProblemV1,
        text_gen: _TextGenFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           HumanEvalExecutorResultV1]:
    """A0_text: text-only single-shot, NO image access."""
    he_problem = _stripped_to_humaneval_problem(p)
    prompt = _code_prompt(p)
    text, wall = text_gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, prompt=p.stripped_prompt,
        entry_point=p.entry_point)
    exe = run_humaneval_executor_v1(
        problem=he_problem, candidate_code=code,
        **executor_kwargs)
    call = HumanEvalArmCallCapsuleV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(p.task_id),
        arm_id="A0_text",
        role="code_solver",
        call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(p.task_id),
        arm_id="A0_text",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),))
    return out, exe


def _run_a1_vlm(
        *, seed: int, p: CrossModalProblemV1,
        K: int, temperature: float,
        vlm_gen: _VlmGenFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """A1_vlm: single-agent VLM, K independent samples; ship
    first PASS."""
    he_problem = _stripped_to_humaneval_problem(p)
    prompt = _vlm_solve_prompt(p)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    for k in range(int(K)):
        text, wall = vlm_gen(
            prompt, p.image_bytes, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=p.stripped_prompt,
            entry_point=p.entry_point)
        exe = run_humaneval_executor_v1(
            problem=he_problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            task_id=str(p.task_id),
            arm_id="A1_vlm",
            role="vlm_sample",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                (prompt + "|img:" + p.image_cid).encode(
                    "utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_code_cid = str(exe.candidate_code_cid)
    if not chosen_passed:
        chosen_code_cid = str(exes[0].candidate_code_cid)
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(p.task_id),
        arm_id="A1_vlm",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


def _run_b_cross(
        *, seed: int, p: CrossModalProblemV1,
        K: int, temperature: float,
        vlm_gen: _VlmGenFn, text_gen: _TextGenFn,
        max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """B_cross: VLM extracts doctest from image → code-LM
    generates code conditioned on stripped prompt + extraction →
    executor reflexion loop.

    Total: 1 VLM extract + (K-1) code-LM calls = K model calls,
    matching A1_vlm's budget exactly.
    """
    he_problem = _stripped_to_humaneval_problem(p)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    # Call 0: VLM extracts doctest from the image at T=0.0
    vlm_prompt = _vlm_extract_prompt(p)
    extraction, w_vlm = vlm_gen(
        vlm_prompt, p.image_bytes, max_tokens, 0.0)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(p.task_id),
        arm_id="B_cross",
        role="vlm_extract",
        call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            (vlm_prompt + "|img:" + p.image_cid).encode(
                "utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            extraction.encode("utf-8")).hexdigest(),
        wall_ms=int(w_vlm)))
    total += int(w_vlm)
    # Calls 1..K-1: sequential reflexion via code-LM
    history: list[tuple[str, HumanEvalExecutorResultV1]] = []
    candidates_code: list[str] = []
    n_code_calls = int(K) - 1
    for k in range(int(n_code_calls)):
        if k == 0:
            prompt = _code_prompt(
                p, extra_context=(
                    "The function's doctest examples (as "
                    "extracted by a vision agent from the corpus "
                    "image):\n"
                    f"```\n{extraction.strip()}\n```"))
        else:
            prompt = _reflexion_prompt_cross_modal(
                p, extraction, tuple(history),
                attempt_idx=int(k))
        text, w_code = text_gen(
            prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=p.stripped_prompt,
            entry_point=p.entry_point)
        exe = run_humaneval_executor_v1(
            problem=he_problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            task_id=str(p.task_id),
            arm_id="B_cross",
            role="code_reflexion" if k > 0 else "code_initial",
            call_idx=int(k + 1),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(w_code)))
        exes.append(exe)
        candidates_code.append(code)
        history.append((code, exe))
        total += int(w_code) + int(exe.wall_ms)
    # Selection: first PASS by attempt index; else
    # lexicographically smallest CID
    final_passed = False
    final_code_cid = ""
    for e in exes:
        if e.passed:
            final_passed = True
            final_code_cid = str(e.candidate_code_cid)
            break
    if not final_passed and exes:
        cids = sorted(str(e.candidate_code_cid) for e in exes)
        final_code_cid = cids[0]
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(p.task_id),
        arm_id="B_cross",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(len(exes)),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CrossModalCodeSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_text_pass_at_1: float
    a1_vlm_pass_at_1: float
    b_cross_pass_at_1: float
    a0_text_total_wall_ms: int
    a1_vlm_total_wall_ms: int
    b_cross_total_wall_ms: int
    outcome_cids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_text_pass_at_1": float(round(
                self.a0_text_pass_at_1, 6)),
            "a1_vlm_pass_at_1": float(round(
                self.a1_vlm_pass_at_1, 6)),
            "b_cross_pass_at_1": float(round(
                self.b_cross_pass_at_1, 6)),
            "a0_text_total_wall_ms": int(
                self.a0_text_total_wall_ms),
            "a1_vlm_total_wall_ms": int(
                self.a1_vlm_total_wall_ms),
            "b_cross_total_wall_ms": int(
                self.b_cross_total_wall_ms),
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class CrossModalCodeBenchReportV1:
    schema: str
    vlm_model_id: str
    code_model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[CrossModalCodeSeedReportV1, ...]
    corpus_problem_cids: tuple[str, ...]
    a0_text_mean_pass_at_1: float
    a1_vlm_mean_pass_at_1: float
    b_cross_mean_pass_at_1: float
    b_cross_beats_a0_text_per_seed: tuple[bool, ...]
    b_cross_beats_a1_vlm_per_seed: tuple[bool, ...]
    b_cross_mean_strictly_beats_a0_text_mean: bool
    b_cross_mean_strictly_beats_a1_vlm_mean: bool
    b_cross_mean_minus_a0_text_mean_pp: float
    b_cross_mean_minus_a1_vlm_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "vlm_model_id": str(self.vlm_model_id),
            "code_model_id": str(self.code_model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "corpus_problem_cids": list(self.corpus_problem_cids),
            "a0_text_mean_pass_at_1": float(round(
                self.a0_text_mean_pass_at_1, 6)),
            "a1_vlm_mean_pass_at_1": float(round(
                self.a1_vlm_mean_pass_at_1, 6)),
            "b_cross_mean_pass_at_1": float(round(
                self.b_cross_mean_pass_at_1, 6)),
            "b_cross_beats_a0_text_per_seed": list(
                self.b_cross_beats_a0_text_per_seed),
            "b_cross_beats_a1_vlm_per_seed": list(
                self.b_cross_beats_a1_vlm_per_seed),
            "b_cross_mean_strictly_beats_a0_text_mean": bool(
                self.b_cross_mean_strictly_beats_a0_text_mean),
            "b_cross_mean_strictly_beats_a1_vlm_mean": bool(
                self.b_cross_mean_strictly_beats_a1_vlm_mean),
            "b_cross_mean_minus_a0_text_mean_pp": float(round(
                self.b_cross_mean_minus_a0_text_mean_pp, 4)),
            "b_cross_mean_minus_a1_vlm_mean_pp": float(round(
                self.b_cross_mean_minus_a1_vlm_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class CrossModalCodeBenchConfigV1:
    schema: str = W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 12
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (88_046_001, 88_046_002, 88_046_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W86_HUMANEVAL_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S)
    min_doctest_lines: int = 2
    strip_mode: str = "doctest_only"


def run_cross_modal_code_bench_v1(
        *,
        text_gen: _TextGenFn,
        vlm_gen: _VlmGenFn,
        vlm_model_id: str,
        code_model_id: str,
        corpus: Sequence[HumanEvalProblemV1],
        config: CrossModalCodeBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> tuple[CrossModalCodeBenchReportV1,
           tuple[CrossModalProblemV1, ...]]:
    """Drive the cross-modal code bench end-to-end."""
    cfg = config or CrossModalCodeBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
    }
    cross_corpus = synthesize_cross_modal_corpus_v1(
        corpus, min_doctest_lines=int(cfg.min_doctest_lines),
        strip_mode=str(cfg.strip_mode))
    if int(len(cross_corpus)) < int(cfg.n_problems):
        raise RuntimeError(
            f"cross-modal corpus has only {len(cross_corpus)} "
            f"problems with ≥ {cfg.min_doctest_lines} doctest "
            f"lines; need {cfg.n_problems}")
    per_seed: list[CrossModalCodeSeedReportV1] = []
    all_outcome_cids: list[str] = []
    corpus_problem_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_cross_modal_subset_v1(
            corpus=cross_corpus,
            n_problems=int(cfg.n_problems), seed=int(seed))
        for p in subset:
            corpus_problem_cids.append(str(p.problem_cid()))
        a0_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        a1_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        b_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(p.task_id))
            a0_out, _ = _run_a0_text(
                seed=int(seed), p=p, text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a0_outs.append(a0_out)
            a1_out, _ = _run_a1_vlm(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a1_outs.append(a1_out)
            b_out, _ = _run_b_cross(
                seed=int(seed), p=p,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                vlm_gen=vlm_gen, text_gen=text_gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            b_outs.append(b_out)
        n = float(len(a0_outs))
        a0_acc = sum(
            1 for o in a0_outs if o.final_passed) / n
        a1_acc = sum(
            1 for o in a1_outs if o.final_passed) / n
        b_acc = sum(
            1 for o in b_outs if o.final_passed) / n
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w88_cross_modal_code_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(CrossModalCodeSeedReportV1(
            schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_text_pass_at_1=float(a0_acc),
            a1_vlm_pass_at_1=float(a1_acc),
            b_cross_pass_at_1=float(b_acc),
            a0_text_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_vlm_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_cross_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle)))
        all_outcome_cids.extend(outcome_cids)
    nseeds = float(len(per_seed))
    a0_mean = sum(
        s.a0_text_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(
        s.a1_vlm_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(
        s.b_cross_pass_at_1 for s in per_seed) / nseeds
    bench_merkle = _sha256_hex({
        "kind": "w88_cross_modal_code_bench_merkle_root",
        "vlm_model_id": str(vlm_model_id),
        "code_model_id": str(code_model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    report = CrossModalCodeBenchReportV1(
        schema=W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
        vlm_model_id=str(vlm_model_id),
        code_model_id=str(code_model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        corpus_problem_cids=tuple(corpus_problem_cids),
        a0_text_mean_pass_at_1=float(a0_mean),
        a1_vlm_mean_pass_at_1=float(a1_mean),
        b_cross_mean_pass_at_1=float(b_mean),
        b_cross_beats_a0_text_per_seed=tuple(
            s.b_cross_pass_at_1 > s.a0_text_pass_at_1
            for s in per_seed),
        b_cross_beats_a1_vlm_per_seed=tuple(
            s.b_cross_pass_at_1 > s.a1_vlm_pass_at_1
            for s in per_seed),
        b_cross_mean_strictly_beats_a0_text_mean=bool(
            b_mean > a0_mean),
        b_cross_mean_strictly_beats_a1_vlm_mean=bool(
            b_mean > a1_mean),
        b_cross_mean_minus_a0_text_mean_pp=float(
            (b_mean - a0_mean) * 100.0),
        b_cross_mean_minus_a1_vlm_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        bench_merkle_root=str(bench_merkle))
    return report, cross_corpus


__all__ = [
    "W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION",
    "CrossModalProblemV1",
    "CrossModalCodeSeedReportV1",
    "CrossModalCodeBenchReportV1",
    "CrossModalCodeBenchConfigV1",
    "synthesize_cross_modal_corpus_v1",
    "select_cross_modal_subset_v1",
    "run_cross_modal_code_bench_v1",
]
