"""General-purpose triggers for cross-surface reuse (Phase 18).

Phase 17 isolated the deployment tax: routing transferred across tasks, but
disagreement detection didn't. The schema-key trigger is blind to numerical
drift; the behavior-probe trigger needs a hand-engineered probe battery per
pair. This module supplies two task-agnostic alternatives that the Phase 18
benchmark uses to measure whether a more general trigger can close the gap:

  * `LLMJudgeTrigger`   — asks a small LLM to rate convention-disagreement
                          between draft snippets. No task-specific code paths.
  * `HybridStructuralTrigger`
                        — pure-static analysis: combines dict-key, string-
                          literal, numeric-constant, and same-name function
                          fuzz signals. No task-specific pair tables.
  * `GeneralTrigger`    — try LLM judge; on transport / parse failure fall
                          back to the hybrid heuristic. This is the trigger
                          phase18 reports as "general".

Both implementations satisfy the `Trigger` protocol from `core/trigger.py`.

Design notes:
  * The LLM judge's prompt is intentionally task-neutral: it talks about
    "shared conventions" and "drafts", not dict keys or rounding. The same
    prompt is used for ProtocolKit and NumericLedger.
  * The hybrid heuristic combines several Jaccard distances via `max`,
    matching the operational semantic that ANY axis of drift should fire
    the trigger.
  * Failures (LLM unavailable, parse failure, exec crash on probe) degrade
    gracefully — they never raise, they fall back to the next layer. The
    last-resort default is `score=0.5, refine=True` (refine on uncertainty).
"""

from __future__ import annotations

import ast
import math
import re
import traceback
from dataclasses import dataclass
from typing import Any

from .trigger import (
    CallableTrigger, Trigger, TriggerDecision, register_trigger,
)


# =============================================================================
# Hybrid structural trigger — task-agnostic AST + behavior signal
# =============================================================================

_NUMBER_QUANT_BUCKETS = (
    # bucket numeric constants by order of magnitude so 99 vs 100 is "same",
    # but 100 vs 1000 differs (catches scale conventions: cents vs mils)
    lambda x: f"int_{int(math.log10(abs(x)))}" if isinstance(x, int) and x != 0
              else f"float_{int(math.log10(abs(x)))}"
              if isinstance(x, float) and x != 0 and not math.isnan(x)
              else "zero"
              if x == 0
              else "nan",
)


def _safe_parse(src: str) -> ast.AST | None:
    if not src:
        return None
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _extract_dict_keys(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
        elif isinstance(node, ast.Subscript):
            s = node.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                keys.add(s.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (node.func.attr in ("get", "pop", "setdefault")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
    return keys


def _extract_string_literals(tree: ast.AST | None) -> set[str]:
    """Strings used in expression contexts (not docstrings / comments)."""
    if tree is None:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if 1 <= len(v) <= 32 and not v.startswith("\n"):
                out.add(v)
    # remove function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            doc = ast.get_docstring(node)
            if doc:
                out.discard(doc)
    return out


def _extract_number_buckets(tree: ast.AST | None) -> set[str]:
    """Bucket numeric constants by order of magnitude. Catches scale-convention
    drift (cents vs mils -> different OOM bucket) without spuriously firing
    on tiny perturbations."""
    if tree is None:
        return set()
    bucket_fn = _NUMBER_QUANT_BUCKETS[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            try:
                out.add(bucket_fn(node.value))
            except Exception:
                pass
    return out


def _extract_def_signatures(tree: ast.AST | None) -> dict[str, tuple[str, ...]]:
    """Top-level function name -> tuple of argument names."""
    if tree is None:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                out[node.name] = tuple(a.arg for a in node.args.args)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


# Generic numeric / string fuzz inputs. Intentionally wide enough to surface
# any common convention drift (rounding, scale, NaN, sign, dict-key naming
# differences will already have been caught upstream by the AST signals).
_FUZZ_INPUTS_NUMERIC = [
    (0,), (1,), (-1,), (0.5,), (-0.5,), (1.5,),
    (100,), (1000,),
]
# Pairwise inputs include bit-boundary cases that distinguish two's-complement
# from sign-magnitude encoding: (-1, 8) diverges because 0xFF (two's-comp) ≠
# 0x81 (sign-mag). (-128, 8) is unrepresentable in sign-magnitude 8-bit,
# and (-127, 8) maps to 0xFF in sign-magnitude but 0x81 in two's-complement.
_FUZZ_INPUTS_PAIRWISE = [
    (1, 1), (5, 3), (-5, 3), (99, 50), (100, 0),
    # signed-encoding boundary cases
    (-1, 8), (-127, 8), (-128, 8), (-1, 16), (-5, 8),
]


def _exec_module(src: str) -> dict | None:
    """Exec a source snippet into an isolated namespace. Returns None on any
    failure — never raises. `math` and the builtins are exposed."""
    if not src:
        return None
    ns: dict = {"__builtins__": __builtins__, "math": math}
    try:
        exec(src, ns)
    except Exception:
        return None
    return ns


def _shared_function_disagreement(
    own_src: str, bulletin_srcs: list[str]
) -> float | None:
    """For each function name defined in BOTH own and at least one bulletin
    draft, fuzz-evaluate it on a small generic input grid and measure how
    often the outputs match across the two definitions. Returns mean
    disagreement in [0, 1], or None if no co-defined function was found.

    This is the only behavior-level signal in the hybrid trigger. It is
    task-agnostic: the inputs are generic numerics, not pair-specific
    probes. It catches numerical-convention drift that AST signals miss
    (e.g. half-up vs banker's rounding produces identical AST keys but
    different fuzz outputs at 0.5)."""
    own_ns = _exec_module(own_src)
    if own_ns is None:
        return None
    own_fns = {
        k: v for k, v in own_ns.items()
        if callable(v) and not k.startswith("_") and k not in ("math",)
        and getattr(v, "__module__", None) is None  # locally defined
    }
    if not own_fns:
        return None

    disagreements: list[float] = []
    for bul_src in bulletin_srcs:
        bul_ns = _exec_module(bul_src)
        if bul_ns is None:
            continue
        for name, own_fn in own_fns.items():
            if name not in bul_ns:
                continue
            bul_fn = bul_ns[name]
            if not callable(bul_fn):
                continue
            n_arg = _arity(own_fn)
            if n_arg is None or _arity(bul_fn) != n_arg:
                continue
            inputs = (
                _FUZZ_INPUTS_NUMERIC if n_arg == 1
                else _FUZZ_INPUTS_PAIRWISE if n_arg == 2
                else None
            )
            if inputs is None:
                continue
            differ = 0
            total = 0
            for args in inputs:
                try:
                    a = own_fn(*args)
                    b = bul_fn(*args)
                except Exception:
                    continue
                total += 1
                if not _approx_equal(a, b):
                    differ += 1
            if total > 0:
                disagreements.append(differ / total)
    if not disagreements:
        return None
    return sum(disagreements) / len(disagreements)


def _arity(fn) -> int | None:
    try:
        import inspect
        sig = inspect.signature(fn)
        params = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
        return len(params)
    except (TypeError, ValueError):
        return None


def _approx_equal(a, b, tol: float = 1e-6) -> bool:
    try:
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            return abs(a - b) < tol
        return a == b
    except Exception:
        return False


@dataclass
class HybridDecisionInfo:
    components: dict[str, float]


class HybridStructuralTrigger:
    """Task-agnostic structural + fuzz trigger.

    Disagreement = max over four axes:
      * dict-key Jaccard               (catches schema-naming drift)
      * string-literal Jaccard         (catches enum-tag / status-string drift)
      * numeric-constant bucket Jaccard (catches scale drift: cents vs mils)
      * shared-function fuzz disagreement
                                       (catches numerical-convention drift
                                        when AST shape is identical)

    Combining via max means ANY axis of disagreement fires the trigger,
    matching the safety-first event-trigger semantic.
    """

    name = "hybrid-structural"

    def should_refine(
        self,
        own_draft: str,
        bulletin_drafts: list[str],
        threshold: float = 0.34,
    ) -> TriggerDecision:
        if not own_draft or not bulletin_drafts:
            return TriggerDecision(
                refine=False, score=0.0, threshold=threshold,
                info={"reason": "no_input"},
            )

        own_tree = _safe_parse(own_draft)
        bul_trees = [_safe_parse(s) for s in bulletin_drafts]

        own_keys = _extract_dict_keys(own_tree)
        bul_keys: set[str] = set()
        for t in bul_trees:
            bul_keys |= _extract_dict_keys(t)
        d_keys = _jaccard(own_keys, bul_keys) if (own_keys or bul_keys) else 0.0

        own_strs = _extract_string_literals(own_tree)
        bul_strs: set[str] = set()
        for t in bul_trees:
            bul_strs |= _extract_string_literals(t)
        d_strs = (_jaccard(own_strs, bul_strs)
                  if (own_strs and bul_strs) else 0.0)

        own_nums = _extract_number_buckets(own_tree)
        bul_nums: set[str] = set()
        for t in bul_trees:
            bul_nums |= _extract_number_buckets(t)
        d_nums = (_jaccard(own_nums, bul_nums)
                  if (own_nums and bul_nums) else 0.0)

        d_fuzz = _shared_function_disagreement(own_draft, bulletin_drafts)

        components = {
            "dict_keys": d_keys,
            "string_literals": d_strs,
            "number_buckets": d_nums,
        }
        if d_fuzz is not None:
            components["fuzz_behavior"] = d_fuzz

        score = max(components.values()) if components else 0.0
        return TriggerDecision(
            refine=score >= threshold,
            score=score,
            threshold=threshold,
            info={"components": components},
        )


# =============================================================================
# LLM-judge trigger — call a small local LLM to rate disagreement
# =============================================================================

_JUDGE_PROMPT_TEMPLATE = """\
You are a code-coordination auditor. Below are Python drafts written by
DIFFERENT agents on the same team. They must agree on shared conventions
(e.g. dict key names, numeric encodings, return shapes, status strings).

Your job: rate how strongly the OWN draft DISAGREES with the BULLETIN drafts
on those shared conventions. Output a SINGLE NUMBER in [0.00, 1.00]:
  0.00 = perfect agreement (same keys, same numerical conventions, same shapes)
  1.00 = total disagreement (uses incompatible conventions everywhere)

Output ONLY the number. No commentary, no JSON, no explanation.

=== OWN DRAFT ===
{own}

=== BULLETIN DRAFT(S) ===
{bulletin}

Disagreement score (0.00-1.00):"""


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str) -> float | None:
    """Extract the first number in [0, 1] from a model response. Returns
    None if no number is found or it's out of range."""
    if not text:
        return None
    for m in _NUMBER_RE.finditer(text):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
        if 0.0 <= v <= 100.0:
            # Tolerate "75" meaning 0.75
            return v / 100.0
    return None


class LLMJudgeTrigger:
    """Task-agnostic LLM-judge disagreement detector.

    Uses the project's `LLMClient` to talk to a local Ollama model. The
    judge prompt is fixed and never references dict keys, rounding, NaN
    policy, or any task-specific concept — so the same trigger works on
    ProtocolKit and NumericLedger without modification.

    On any failure (transport error, parse failure, empty response) the
    decision falls back to `fallback`, which the `GeneralTrigger` wires
    to the hybrid structural trigger so the experiment still produces a
    decision rather than crashing mid-run.
    """

    name = "llm-judge"

    def __init__(
        self,
        client: Any,
        max_tokens: int = 16,
        max_chars_per_draft: int = 800,
        fallback: Trigger | None = None,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.max_chars = max_chars_per_draft
        self.fallback = fallback

    def _truncate(self, src: str) -> str:
        return src if len(src) <= self.max_chars else src[: self.max_chars] + "\n# ..."

    def should_refine(
        self,
        own_draft: str,
        bulletin_drafts: list[str],
        threshold: float = 0.34,
    ) -> TriggerDecision:
        if not own_draft or not bulletin_drafts:
            return TriggerDecision(
                refine=False, score=0.0, threshold=threshold,
                info={"reason": "no_input"},
            )

        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            own=self._truncate(own_draft),
            bulletin="\n\n---\n\n".join(self._truncate(s) for s in bulletin_drafts),
        )

        raw = ""
        err: str | None = None
        try:
            raw = self.client.generate(
                prompt, max_tokens=self.max_tokens, temperature=0.0,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        score = _parse_score(raw) if raw else None

        if score is None:
            if self.fallback is not None:
                fb = self.fallback.should_refine(
                    own_draft, bulletin_drafts, threshold=threshold,
                )
                fb.info = {
                    "reason": "llm_parse_failed_fellback",
                    "fallback": getattr(self.fallback, "name", "unknown"),
                    "fallback_info": fb.info,
                    "raw_response": raw[:200],
                    "error": err,
                }
                return fb
            # No fallback — refine on uncertainty (safety bias).
            return TriggerDecision(
                refine=True, score=0.5, threshold=threshold,
                info={"reason": "llm_parse_failed_no_fallback",
                      "raw_response": raw[:200], "error": err},
            )

        return TriggerDecision(
            refine=score >= threshold,
            score=score,
            threshold=threshold,
            info={"reason": "llm_judge",
                  "raw_response": raw[:200],
                  "judge_tokens_max": self.max_tokens},
        )


class GeneralTrigger:
    """LLM-judge with hybrid-structural fallback.

    This is the "general" trigger Phase 18 reports on. Practically:
      - if `client` is provided AND the LLM call succeeds, use the judge;
      - on any failure path (no client, transport error, unparseable
        response) fall through to `HybridStructuralTrigger`.

    Either path is task-agnostic: zero hardcoded pair tables, zero
    references to ProtocolKit / NumericLedger concepts.
    """

    name = "general"

    def __init__(self, client: Any | None = None, **judge_kwargs):
        self._hybrid = HybridStructuralTrigger()
        if client is not None:
            self._judge: Trigger | None = LLMJudgeTrigger(
                client=client, fallback=self._hybrid, **judge_kwargs,
            )
        else:
            self._judge = None

    def should_refine(
        self,
        own_draft: str,
        bulletin_drafts: list[str],
        threshold: float = 0.34,
    ) -> TriggerDecision:
        if self._judge is not None:
            return self._judge.should_refine(
                own_draft, bulletin_drafts, threshold=threshold,
            )
        return self._hybrid.should_refine(
            own_draft, bulletin_drafts, threshold=threshold,
        )


# ---------------- Registry hooks ------------------------------------------

# These let `phase18_general_trigger.py` request a trigger by name. The
# LLM-judge form requires a client, so we only register the no-client
# (heuristic-only) variant by default; phase18 instantiates the LLM form
# explicitly.

register_trigger("hybrid-structural", lambda: HybridStructuralTrigger())
register_trigger("general-heuristic", lambda: GeneralTrigger(client=None))
