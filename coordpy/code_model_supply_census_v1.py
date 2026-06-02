"""W131 Lane α — code-competent model SUPPLY census (the main diagnosis lane).

The W130 generation-ceiling attack found the hard-cluster pool is GENERATION-bound and the
dominant ``WRONG_ALGORITHM_ADMISSIBLE`` mode is *capability*-bound (a named technique != a correct
algorithm).  W129 had already moved the binding cap from SELECTION to GENERATION.  So the next
honest lever is **stronger MODEL capability**, not more same-budget prompt engineering.

This module answers the W131 Lane α question *before* any serious dev spend:

    is there any reachable code-competent local / nearby / hosted model supply that can materially
    raise the hard-cluster generation ceiling — and which candidate is the best honest one?

It is the FIRST module to unify the three model-supply surfaces the programme has used —

  * local-HF transformer runtime (``transformers_runtime_v1`` / ``substrate_adapter_v25`` /
    ``code_substrate_v1`` — torch/transformers required),
  * local / nearby OpenAI-compatible endpoints (Ollama at ``http://localhost:11434``),
  * hosted NIM (``nim_frontier_text_runtime_v1`` → ``https://integrate.api.nvidia.com``),

— into ONE machine-checkable reachability / capability matrix, cross-cut with the
``stronger_model_cutoff_certification_v1`` disclosure gate (decision CID ``258b6ed7…``) so each
model is classed FRONTIER_ELIGIBLE (honest for a resistant claim) vs DEV_ONLY (useful for the
EXPOSED dev bench only).

Honest scope / what this is NOT
-------------------------------
* A model is NOT admitted to the dev bench just because it loads / is reachable.  It must pass a
  tiny SAME-FAMILY code smoke gate (``code_smoke_gate_v1``): emit a runnable program that solves a
  trivial competitive-programming task on stdin/stdout.  Reachable-but-not-code-emitting ⇒ not a
  candidate.
* A model's cutoff disclosure governs the RESISTANT lane only.  A stronger UNKNOWN-cutoff model is
  fine for EXPOSED dev validation (the W127–W130 operator-greenlit lane) but is DEV_ONLY — it can
  never license a resistant claim, because the resistant ICPC battlefield post-dates Maverick's
  Aug-2024 cutoff and an UNKNOWN-cutoff model may have trained on those problems (contamination).
* This census spends $0 on the hosted lane (reachability is a free ``GET /v1/models``; the hosted
  code smoke is deferred to the dev-bench canary, which is the operator-greenlit lane).  Local
  Ollama smoke is $0 (local compute).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional, Sequence

CODE_MODEL_SUPPLY_CENSUS_V1_SCHEMA = "coordpy.code_model_supply_census_v1.v1"

# ---------------------------------------------------------------------------
# LOCKED enums (the census schema — fixed before any results are interpreted).
# ---------------------------------------------------------------------------

ACCESS_PATHS = ("LOCAL_HF", "LOCAL_OLLAMA", "HOSTED_NIM")
CODE_PRIORS = ("CODE_TUNED", "REASONING", "GENERAL_LM", "EMBED_OR_OTHER")
# Cutoff disclosure (primary-source), governs the RESISTANT lane.
CUTOFF_DISCLOSURES = ("PRIMARY_KNOWN", "UNKNOWN")
# Usage class derived from disclosure + strength + reachability.
USAGE_CLASSES = ("FRONTIER_ELIGIBLE", "DEV_ONLY", "SETTLED", "NOT_A_GENERATOR")

# The resistant ICPC battlefield (W120) post-dates Maverick's Aug-2024 cutoff; a model is
# FRONTIER_ELIGIBLE for a resistant probe only if its cutoff is primary-KNOWN AND <= that frontier.
RESISTANT_INSTRUMENT_FRONTIER = "2024-08 (Maverick cutoff; W120 resistant slice is post-cutoff)"

# Name-keyed code-competence priors.  Transparent, never model-facing; an UPPER BOUND on "code
# competence" (a code-tuned name != a strong code model), reported as a heuristic.
_CODE_TUNED_RE = re.compile(
    r"(coder|code-|codestral|starcoder|codellama|codegemma|granite.*code|deepseek-coder"
    r"|nv-?embedcode)", re.I)
_REASONING_RE = re.compile(r"(deepseek-r1|reason|-r1\b|thinking|qwq|o1|nemotron.*reason)", re.I)
_EMBED_RE = re.compile(r"(embed|rerank|reward|guard|safety|parse\b|nemotron-parse)", re.I)


def classify_code_prior_v1(model_id: str) -> str:
    """Transparent name-keyed code-competence prior (heuristic, UPPER BOUND)."""
    m = str(model_id)
    if _EMBED_RE.search(m):
        return "EMBED_OR_OTHER"
    if _CODE_TUNED_RE.search(m):
        return "CODE_TUNED"
    if _REASONING_RE.search(m):
        return "REASONING"
    return "GENERAL_LM"


@dataclasses.dataclass(frozen=True)
class CodeModelSupplyRecordV1:
    """One reachable (or blocked) model in the supply census."""

    model_id: str
    access_path: str               # ACCESS_PATHS
    code_prior: str                # CODE_PRIORS
    param_hint: str                # e.g. "32B", "480b-a35b", "17b-128e", "" if unknown
    context_hint: int              # advertised context window tokens (0 if unknown)
    load_success: bool             # local: weights/endpoint reachable; hosted: in /v1/models
    blocked_reason: str            # "" if not blocked
    smoke_ran: bool                # was the code smoke gate actually executed
    smoke_pass: bool               # emitted a runnable program that solved the smoke task
    smoke_detail: str
    cutoff_disclosure: str         # CUTOFF_DISCLOSURES
    cutoff_boundary: str           # primary-source cutoff or "" if UNKNOWN
    stronger_than_maverick: Optional[bool]  # None if unknown/not-comparable
    usage_class: str               # USAGE_CLASSES
    realistic_for_dev_bench: bool  # code-competent + reachable + smoke-pass (or smoke deferred)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "access_path": str(self.access_path),
            "code_prior": str(self.code_prior),
            "param_hint": str(self.param_hint),
            "context_hint": int(self.context_hint),
            "load_success": bool(self.load_success),
            "blocked_reason": str(self.blocked_reason),
            "smoke_ran": bool(self.smoke_ran),
            "smoke_pass": bool(self.smoke_pass),
            "smoke_detail": str(self.smoke_detail),
            "cutoff_disclosure": str(self.cutoff_disclosure),
            "cutoff_boundary": str(self.cutoff_boundary),
            "stronger_than_maverick": self.stronger_than_maverick,
            "usage_class": str(self.usage_class),
            "realistic_for_dev_bench": bool(self.realistic_for_dev_bench),
            "note": str(self.note),
        }


# ---------------------------------------------------------------------------
# Minimal stdin/stdout smoke executor (inlined; isolated -I subprocess).
# ---------------------------------------------------------------------------

def _smoke_run_v1(code: str, stdin_text: str, *, timeout_s: float = 5.0) -> str:
    """Run ``code`` as an isolated ``python -I -c`` subprocess on ``stdin_text``.

    Returns stdout (whitespace-stripped) or a ``<…>`` sentinel on failure.
    """
    if not code or not code.strip():
        return "<NO_CODE>"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input=stdin_text.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>"
    except Exception as e:  # noqa: BLE001
        return f"<ERR:{type(e).__name__}>"
    if proc.returncode != 0:
        return f"<RC:{proc.returncode}>"
    return proc.stdout.decode("utf-8", "replace").strip()


# Some code models (e.g. Qwen2.5/3-Coder) emit the fence info-string on its OWN line:
#   ```\npython\n<code>\n```   instead of   ```python\n<code>\n```
# which leaves a stray bare ``python`` first line that both this smoke extractor AND the audited
# ``extract_candidate_code_v1`` keep, crashing the program (NameError → RC:1).  This is a pure
# FORMAT-NORMALIZATION (parsing fairness across models), NEVER a capability lever and NEVER a change
# to any algorithm: it only moves a misplaced python language tag onto the fence line.  Applied
# uniformly at the generation seam so every downstream consumer sees a clean fence; the raw model
# output is preserved verbatim in the call sidecar.  W130 honesty rule: a parsing fix is reported AS
# a parsing fix (it removes an UNDER-statement of capability, it does not add capability).
_FENCE_LANG_FIX_RE = re.compile(r"```[ \t]*\r?\n[ \t]*(python3?|py)\b[ \t]*\r?\n", re.I)


def normalize_fence_v1(text: str) -> str:
    """Move a misplaced ``python`` language tag onto its opening fence line (format-only)."""
    if not text:
        return text
    return _FENCE_LANG_FIX_RE.sub("```python\n", text)


def _extract_code_block_v1(text: str) -> str:
    """Pull a python code block from a chat response; fall back to whole text."""
    if not text:
        return ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    # bare fence
    m = re.search(r"```\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


# The locked smoke task: a trivial competitive-programming stdin/stdout problem.  A code-competent
# model must read N and print N squared.  Two cases guard against constant-output cheats.
SMOKE_TASK_PROMPT = (
    "Solve this competitive-programming problem in Python 3.\n\n"
    "Read a single integer N from standard input and print N*N (N squared) to standard output.\n\n"
    "Output ONLY one fenced python code block with the complete program. No explanation."
)
SMOKE_CASES = (("7\n", "49"), ("12\n", "144"))


def code_smoke_gate_v1(gen: Callable[[str, int, float], Any], *,
                       max_tokens: int = 300, temperature: float = 0.2,
                       run_timeout_s: float = 5.0) -> tuple[bool, str]:
    """Tiny SAME-FAMILY code smoke gate.

    Calls ``gen(prompt, max_tokens, temperature)`` (the standard
    ``(text, wall_ms)`` contract), extracts the program, and runs it on
    ``SMOKE_CASES``.  Returns ``(passed, detail)``.  A model passes iff its
    emitted program produces the exact expected stdout on BOTH cases.
    """
    try:
        out = gen(SMOKE_TASK_PROMPT, max_tokens, temperature)
        text = out[0] if isinstance(out, (tuple, list)) else str(out)
    except Exception as e:  # noqa: BLE001
        return False, f"gen_error:{type(e).__name__}:{e}"
    code = _extract_code_block_v1(text)
    if not code:
        return False, "no_code_emitted"
    results = []
    for stdin_text, expected in SMOKE_CASES:
        got = _smoke_run_v1(code, stdin_text, timeout_s=run_timeout_s)
        results.append((expected, got))
        if got != expected:
            return False, f"case_fail stdin={stdin_text!r} want={expected!r} got={got!r}"
    return True, f"ok {results}"


# ---------------------------------------------------------------------------
# Unified OpenAI-compatible generation builder (local Ollama AND hosted NIM).
# ---------------------------------------------------------------------------

def build_openai_compat_gen_v1(model: str, *, base_url: str, api_key: Optional[str] = None,
                               read_timeout_s: float = 120.0, max_retries: int = 6,
                               sidecar_writer: Optional[Callable[[dict], None]] = None,
                               backend_tag: str = "") -> Callable[[str, int, float], tuple]:
    """Return a ``gen(prompt, max_tokens, temperature) -> (text, wall_ms)`` closure that POSTs to
    any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Drives BOTH the hosted NIM endpoint (``base_url=https://integrate.api.nvidia.com``,
    ``api_key=$NVIDIA_API_KEY``) and a local Ollama endpoint
    (``base_url=http://localhost:11434``, ``api_key`` ignored by the server) with one contract,
    matching the ``GenFn`` seam the W128/W130 generator arms consume.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
    tag = backend_tag or ("nim" if "nvidia" in base_url else
                          ("ollama" if "11434" in base_url else "openai_compat"))

    def _gen(prompt: str, max_tokens: int = 1536, temperature: float = 0.2) -> tuple:
        body = {
            "model": str(model),
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode("utf-8")
        last_err = ""
        t0 = time.time()
        for attempt in range(1, int(max_retries) + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=float(read_timeout_s)) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                choices = payload.get("choices") or []
                msg = (choices[0].get("message") or {}) if choices else {}
                text = str(msg.get("content") or "")
                wall_ms = int((time.time() - t0) * 1000.0)
                if sidecar_writer is not None:
                    usage = payload.get("usage") or {}
                    sidecar_writer({
                        "model_id": str(model), "backend": tag,
                        "prompt_len": len(prompt), "response_len": len(text),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "temperature": float(temperature), "max_tokens": int(max_tokens),
                        "wall_ms": wall_ms, "attempt": attempt,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "prompt": prompt, "response_text": text,
                    })
                # sidecar keeps the RAW model output (provenance); downstream consumers get the
                # fence-normalized text so the audited extractor parses every model fairly.
                return normalize_fence_v1(text), wall_ms
            except urllib.error.HTTPError as e:
                last_err = f"HTTPError:{e}"
                if e.code == 429:  # rate limit — honor Retry-After, else escalating long backoff
                    try:
                        ra = int((e.headers.get("Retry-After", "0") if e.headers else "0") or 0)
                    except Exception:  # noqa: BLE001
                        ra = 0
                    time.sleep(max(ra, min(8.0 + 6.0 * attempt, 45.0)))
                else:
                    time.sleep(min(2.0 * attempt, 12.0))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"{type(e).__name__}:{e}"
                time.sleep(min(2.0 * attempt, 12.0))
        raise RuntimeError(f"gen failed after {max_retries} retries: {last_err}")

    return _gen


# ---------------------------------------------------------------------------
# Surface probes.
# ---------------------------------------------------------------------------

def probe_local_hf_v1() -> tuple[bool, str]:
    """Probe whether the local transformer runtime can run a model (torch + transformers).

    Returns ``(loadable, reason)``.  This is the W124 line: it stays DEAD if torch/transformers
    cannot be imported in this interpreter.
    """
    missing = []
    for mod in ("torch", "transformers"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}({type(e).__name__})")
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "torch+transformers importable"


def _ollama_tags(base_url: str, *, timeout_s: float = 5.0) -> list[dict]:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=float(timeout_s)) as r:
            return json.loads(r.read().decode("utf-8")).get("models", []) or []
    except Exception:  # noqa: BLE001
        return []


def probe_local_ollama_v1(*, base_url: str = "http://localhost:11434",
                          smoke: bool = True, smoke_models: Optional[Sequence[str]] = None,
                          smoke_timeout_s: float = 120.0,
                          run_timeout_s: float = 5.0) -> list[CodeModelSupplyRecordV1]:
    """Census the local Ollama OpenAI-compatible endpoint.  $0 (local compute)."""
    tags = _ollama_tags(base_url)
    records: list[CodeModelSupplyRecordV1] = []
    for m in tags:
        mid = str(m.get("name", ""))
        details = m.get("details", {}) or {}
        psize = str(details.get("parameter_size", "") or "")
        prior = classify_code_prior_v1(mid)
        smoke_ran = smoke_pass = False
        smoke_detail = "not_run"
        do_smoke = bool(smoke) and prior in ("CODE_TUNED", "REASONING", "GENERAL_LM") and (
            smoke_models is None or mid in set(smoke_models))
        if do_smoke:
            gen = build_openai_compat_gen_v1(mid, base_url=base_url, api_key="ollama",
                                             read_timeout_s=smoke_timeout_s, max_retries=2)
            smoke_pass, smoke_detail = code_smoke_gate_v1(gen, run_timeout_s=run_timeout_s)
            smoke_ran = True
        usage = ("DEV_ONLY" if prior in ("CODE_TUNED", "REASONING", "GENERAL_LM")
                 else "NOT_A_GENERATOR")
        records.append(CodeModelSupplyRecordV1(
            model_id=mid, access_path="LOCAL_OLLAMA", code_prior=prior, param_hint=psize,
            context_hint=0, load_success=True, blocked_reason="",
            smoke_ran=smoke_ran, smoke_pass=smoke_pass, smoke_detail=smoke_detail,
            cutoff_disclosure="UNKNOWN", cutoff_boundary="",
            stronger_than_maverick=False,  # local <=32B code models are not stronger than Maverick
            usage_class=usage,
            realistic_for_dev_bench=bool(prior == "CODE_TUNED" and (smoke_pass or not smoke_ran)),
            note="local $0; UNKNOWN cutoff ⇒ DEV_ONLY (resistant-ineligible)",
        ))
    return records


# Hosted-NIM cutoff disclosures keyed by the certification registry + the W131 supply additions.
# Verified-KNOWN/UNKNOWN follows ``stronger_model_cutoff_certification_v1.W114_CUTOFF_PROVENANCE``;
# additions are conservative UNKNOWN (no primary card disclosing a training cutoff).
def _hosted_cutoff_v1(model_id: str) -> tuple[str, str]:
    """Return ``(disclosure, boundary)`` for a hosted model id, primary-source-grounded."""
    try:
        from .stronger_model_cutoff_certification_v1 import W114_CUTOFF_PROVENANCE
        for mid, prov in W114_CUTOFF_PROVENANCE.items():
            if mid == model_id or mid.split("/")[-1] == model_id.split("/")[-1]:
                conf = getattr(prov, "verified_confidence", "UNKNOWN")
                boundary = getattr(prov, "note", "") if conf == "KNOWN" else ""
                return ("PRIMARY_KNOWN" if conf == "KNOWN" else "UNKNOWN"), boundary
    except Exception:  # noqa: BLE001
        pass
    # Older Llama-2/Code-Llama-era code models carry an approximate KNOWN cutoff but are WEAKER
    # than Maverick (fail the stronger-than-Maverick comparability gate ⇒ not a stronger generator).
    if model_id == "meta/llama-4-maverick-17b-128e-instruct":
        return "PRIMARY_KNOWN", "2024-08 (Meta llama4 MODEL_CARD.md)"
    return "UNKNOWN", ""


# Hosted code-relevant model id → (param_hint, stronger_than_maverick, context_hint).
# stronger_than_maverick is a transparent capability prior: True for >=70B-class / >=A35B MoE
# frontier code/general models; None for comparable; False for clearly weaker.
_HOSTED_STRENGTH = {
    "meta/llama-4-maverick-17b-128e-instruct": ("17b-128e", None, 1048576),  # the settled anchor
    "qwen/qwen3-coder-480b-a35b-instruct": ("480b-a35b", True, 262144),
    "deepseek-ai/deepseek-v4-pro": ("v4-pro", True, 131072),
    "deepseek-ai/deepseek-v4-flash": ("v4-flash", True, 131072),
    "qwen/qwen3.5-397b-a17b": ("397b-a17b", True, 262144),
    "qwen/qwen3.5-122b-a10b": ("122b-a10b", True, 262144),
    "qwen/qwen3-next-80b-a3b-instruct": ("80b-a3b", True, 262144),
    "mistralai/mistral-large-3-675b-instruct-2512": ("675b", True, 131072),
    "mistralai/mistral-medium-3.5-128b": ("128b", True, 131072),
    "mistralai/mistral-small-4-119b-2603": ("119b", True, 131072),
    "z-ai/glm-5.1": ("glm-5.1", True, 131072),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": ("253b", True, 131072),
    "nvidia/nemotron-4-340b-instruct": ("340b", True, 4096),
    "nvidia/nemotron-3-super-120b-a12b": ("120b-a12b", True, 131072),
    "meta/codellama-70b": ("70b", None, 16384),
    "mistralai/codestral-22b-instruct-v0.1": ("22b", False, 32768),
    "ibm/granite-34b-code-instruct": ("34b", False, 8192),
    "ibm/granite-8b-code-instruct": ("8b", False, 4096),
    "bigcode/starcoder2-15b": ("15b", False, 16384),
    "deepseek-ai/deepseek-coder-6.7b-instruct": ("6.7b", False, 16384),
    "google/codegemma-7b": ("7b", False, 8192),
    "google/codegemma-1.1-7b": ("7b", False, 8192),
}


def probe_hosted_nim_v1(*, api_key: Optional[str] = None,
                        endpoint: str = "https://integrate.api.nvidia.com",
                        timeout_s: float = 30.0) -> tuple[bool, str, list[CodeModelSupplyRecordV1]]:
    """Census reachable hosted NIM code-relevant models.  $0 (free ``GET /v1/models``).

    Hosted code smoke is deferred to the dev-bench canary (the operator-greenlit NIM lane), so
    ``smoke_ran=False`` here; reachability + cutoff disclosure + strength prior are recorded.
    Returns ``(reachable, blocked_reason, records)``.
    """
    from .nim_frontier_text_runtime_v1 import probe_nim_frontier_runtime_v1
    rep = probe_nim_frontier_runtime_v1(nim_endpoint=endpoint, api_key=api_key, timeout=timeout_s)
    if not rep.reachable:
        return False, rep.blocked_reason, []
    avail = set(rep.available_models)
    records: list[CodeModelSupplyRecordV1] = []
    for mid in sorted(avail):
        prior = classify_code_prior_v1(mid)
        if prior == "EMBED_OR_OTHER":
            continue  # not a generator
        strength = _HOSTED_STRENGTH.get(mid)
        # keep code-tuned models + the explicitly-strong general frontier models
        if strength is None and prior == "GENERAL_LM":
            continue
        param_hint, stronger, ctx = (strength if strength else ("", None, 0))
        disclosure, boundary = _hosted_cutoff_v1(mid)
        settled = (mid == "meta/llama-4-maverick-17b-128e-instruct")
        if settled:
            usage = "SETTLED"
        elif disclosure == "PRIMARY_KNOWN" and stronger:
            usage = "FRONTIER_ELIGIBLE"  # KNOWN cutoff + stronger ⇒ eligible (none currently)
        else:
            usage = "DEV_ONLY"
        realistic = bool(prior in ("CODE_TUNED", "REASONING") or stronger is True)
        records.append(CodeModelSupplyRecordV1(
            model_id=mid, access_path="HOSTED_NIM", code_prior=prior, param_hint=param_hint,
            context_hint=int(ctx), load_success=True, blocked_reason="",
            smoke_ran=False, smoke_pass=False, smoke_detail="deferred_to_dev_bench_canary_nim",
            cutoff_disclosure=disclosure, cutoff_boundary=boundary,
            stronger_than_maverick=stronger, usage_class=usage,
            realistic_for_dev_bench=realistic,
            note=("settled (Maverick — KNOWN but exhausted as resistant anchor)" if settled
                  else "UNKNOWN cutoff ⇒ DEV_ONLY (resistant-ineligible); EXPOSED-dev OK"
                  if disclosure == "UNKNOWN" else "KNOWN cutoff"),
        ))
    return True, "", records


# ---------------------------------------------------------------------------
# Census assembly.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CodeModelSupplyCensusV1:
    schema: str
    local_hf_loadable: bool
    local_hf_reason: str
    ollama_reachable: bool
    hosted_nim_reachable: bool
    hosted_blocked_reason: str
    records: tuple[CodeModelSupplyRecordV1, ...]

    def best_local_dev_candidate(self) -> Optional[str]:
        """Strongest local code-tuned model that smoke-passed (best $0 dev generator)."""
        cands = [r for r in self.records
                 if r.access_path == "LOCAL_OLLAMA" and r.code_prior == "CODE_TUNED"
                 and (r.smoke_pass or not r.smoke_ran)]
        cands.sort(key=lambda r: _param_billions(r.param_hint), reverse=True)
        return cands[0].model_id if cands else None

    def best_hosted_dev_candidate(self) -> Optional[str]:
        """Strongest reachable code-competent hosted model (best strong dev generator).

        Prefers code-tuned + stronger-than-Maverick; else the strongest stronger-than-Maverick.
        """
        code = [r for r in self.records if r.access_path == "HOSTED_NIM"
                and r.code_prior == "CODE_TUNED" and r.stronger_than_maverick]
        if code:
            code.sort(key=lambda r: _param_billions(r.param_hint), reverse=True)
            return code[0].model_id
        strong = [r for r in self.records if r.access_path == "HOSTED_NIM"
                  and r.stronger_than_maverick]
        strong.sort(key=lambda r: _param_billions(r.param_hint), reverse=True)
        return strong[0].model_id if strong else None

    def frontier_eligible(self) -> list[str]:
        return [r.model_id for r in self.records if r.usage_class == "FRONTIER_ELIGIBLE"]

    def dev_only(self) -> list[str]:
        return [r.model_id for r in self.records if r.usage_class == "DEV_ONLY"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "local_hf_loadable": bool(self.local_hf_loadable),
            "local_hf_reason": str(self.local_hf_reason),
            "ollama_reachable": bool(self.ollama_reachable),
            "hosted_nim_reachable": bool(self.hosted_nim_reachable),
            "hosted_blocked_reason": str(self.hosted_blocked_reason),
            "n_records": len(self.records),
            "best_local_dev_candidate": self.best_local_dev_candidate(),
            "best_hosted_dev_candidate": self.best_hosted_dev_candidate(),
            "frontier_eligible": self.frontier_eligible(),
            "dev_only_count": len(self.dev_only()),
            "records": [r.to_dict() for r in self.records],
        }

    def cid(self) -> str:
        payload = json.dumps(
            [r.to_dict() for r in self.records], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _param_billions(param_hint: str) -> float:
    """Best-effort parameter count in billions from a hint like '32B' / '480b-a35b' / '17b-128e'."""
    if not param_hint:
        return 0.0
    m = re.search(r"([\d.]+)\s*b", param_hint, re.I)
    return float(m.group(1)) if m else 0.0


def build_census_v1(*, smoke_local: bool = True,
                    local_smoke_models: Optional[Sequence[str]] = None,
                    ollama_base_url: str = "http://localhost:11434",
                    nim_api_key: Optional[str] = None,
                    local_smoke_timeout_s: float = 120.0) -> CodeModelSupplyCensusV1:
    """Assemble the full three-surface census.  $0 on the hosted lane; local Ollama smoke is $0."""
    hf_loadable, hf_reason = probe_local_hf_v1()
    ollama_records = probe_local_ollama_v1(
        base_url=ollama_base_url, smoke=smoke_local, smoke_models=local_smoke_models,
        smoke_timeout_s=local_smoke_timeout_s)
    hosted_reachable, hosted_reason, hosted_records = probe_hosted_nim_v1(api_key=nim_api_key)
    records: list[CodeModelSupplyRecordV1] = []
    # Local-HF appears as a single blocked record (the W124 line).
    records.append(CodeModelSupplyRecordV1(
        model_id="(local transformers_runtime_v1 / code_substrate_v1)", access_path="LOCAL_HF",
        code_prior="GENERAL_LM", param_hint="", context_hint=0,
        load_success=hf_loadable, blocked_reason=("" if hf_loadable else hf_reason),
        smoke_ran=False, smoke_pass=False, smoke_detail="not_run",
        cutoff_disclosure="UNKNOWN", cutoff_boundary="",
        stronger_than_maverick=None, usage_class="NOT_A_GENERATOR",
        realistic_for_dev_bench=False,
        note=("torch+transformers importable" if hf_loadable
              else "DEAD — torch/transformers not importable in this interpreter (W124 line)"),
    ))
    records.extend(ollama_records)
    records.extend(hosted_records)
    return CodeModelSupplyCensusV1(
        schema=CODE_MODEL_SUPPLY_CENSUS_V1_SCHEMA,
        local_hf_loadable=hf_loadable, local_hf_reason=hf_reason,
        ollama_reachable=bool(ollama_records), hosted_nim_reachable=hosted_reachable,
        hosted_blocked_reason=hosted_reason, records=tuple(records))


__all__ = [
    "CODE_MODEL_SUPPLY_CENSUS_V1_SCHEMA", "ACCESS_PATHS", "CODE_PRIORS",
    "CUTOFF_DISCLOSURES", "USAGE_CLASSES", "RESISTANT_INSTRUMENT_FRONTIER",
    "classify_code_prior_v1", "CodeModelSupplyRecordV1", "code_smoke_gate_v1",
    "SMOKE_TASK_PROMPT", "SMOKE_CASES", "build_openai_compat_gen_v1",
    "probe_local_hf_v1", "probe_local_ollama_v1", "probe_hosted_nim_v1",
    "CodeModelSupplyCensusV1", "build_census_v1",
]
