"""Synthetic LLM clients — deterministic, in-process, no network.

The capsule-native PROMPT / LLM_RESPONSE slice (SDK v3.4) is wired
into the LLM-backed sweep path in ``vision_mvp.wevra.runtime``. To
exercise that wiring without real LLM access — for unit tests,
contract checks, and the cross-model parser-boundary research
experiment — this module provides ``SyntheticLLMClient``: a tiny
duck-typed substitute for ``vision_mvp.core.llm_client.LLMClient``
that returns canned strings keyed by ``(instance_id_hint,
model_tag)``.

The synthetic client is **deterministic** by construction; two
independent runs against the same client configuration return
byte-identical responses for byte-identical prompts. It carries a
``model_tag`` so the runtime's PROMPT / LLM_RESPONSE capsules
record which "model" the canned distribution is meant to
represent (e.g. ``"synthetic.clean"``, ``"synthetic.unclosed"``,
``"synthetic.prose"``).

Scope discipline
----------------

This module is research-grade infrastructure, not a production
LLM driver. It exists so:

  * Capsule-layer contract tests can drive the LLM-backed path
    without an Ollama endpoint.
  * The cross-model parser-boundary research
    (``vision_mvp.experiments.parser_boundary_cross_model``) can
    sweep a small set of synthetic distributions through the
    real parser pipeline with reproducible seeds.
  * The PROMPT / LLM_RESPONSE capsule slice has empirical
    coverage of the real ``failure_kind`` distribution shapes —
    not just the ``oracle`` sentinel that the deterministic
    oracle path produces on ``local_smoke``.

It is NOT:

  * A model. The "model" returns pre-canned strings; there is no
    inference.
  * A correctness benchmark. It cannot evaluate whether a patch
    is right; only whether the PARSE_OUTCOME / capsule chain
    handles the response shape correctly.
  * A replacement for ``LLMClient``. Real-LLM runs continue to
    use the ``LLMClient`` HTTP wrapper.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable


# A canned-response producer takes (prompt, instance_id_guess) and
# returns the response string. ``instance_id_guess`` is extracted
# from the prompt by the runtime ("INSTANCE: <id>") and is
# best-effort; canned producers may ignore it.
ResponseFn = Callable[[str, str], str]


@dataclasses.dataclass
class SyntheticLLMClient:
    """Deterministic, in-process duck-typed substitute for
    ``vision_mvp.core.llm_client.LLMClient``.

    Construct with ``model_tag`` (records which synthetic
    distribution this client represents) and either a
    ``responses`` dict (``{instance_id: response_str}``) or a
    ``response_fn`` callable.

    The client exposes a ``generate(prompt, max_tokens,
    temperature) -> str`` method matching ``LLMClient.generate``.
    """

    model_tag: str = "synthetic.default"
    responses: dict[str, str] | None = None
    response_fn: ResponseFn | None = None
    default_response: str = ""
    n_calls: int = 0

    def __post_init__(self) -> None:
        # Required so the runtime's _real_cells dispatch can also
        # see ``model`` and ``base_url`` as if this were a real
        # LLMClient. The capsule layer reads ``model_tag`` directly
        # but the runtime constructs an LLMClient via
        # ``LLMClient(model=spec.model, base_url=spec.endpoint)``,
        # so we mirror those fields for compat.
        self.model: str = self.model_tag
        self.base_url: str | None = None
        if self.responses is None:
            self.responses = {}

    def generate(self, prompt: str,
                  max_tokens: int = 80,
                  temperature: float = 0.0) -> str:
        """Return the canned response for ``prompt``.

        Resolution order:
          1. ``response_fn(prompt, instance_id_guess)`` if set.
          2. ``responses[instance_id_guess]`` if the guess was
             recoverable from the prompt header.
          3. ``default_response``.
        """
        self.n_calls += 1
        instance_id = _extract_instance_id_from_prompt(prompt)
        if self.response_fn is not None:
            return self.response_fn(prompt, instance_id)
        if instance_id and instance_id in (self.responses or {}):
            return self.responses[instance_id]
        return self.default_response


def _extract_instance_id_from_prompt(prompt: str) -> str:
    """Extract the ``INSTANCE: <id>`` annotation that
    ``build_patch_generator_prompt`` emits as the first line after
    the role header.

    Falls back to ``""`` if the prompt does not match. Conservative
    by design — the canned producer is responsible for handling
    unknown instances.
    """
    if not prompt:
        return ""
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("INSTANCE:"):
            rest = line[len("INSTANCE:"):].strip()
            # Format is "INSTANCE: <id>  REPO: <repo>"
            if "REPO:" in rest:
                rest = rest.split("REPO:", 1)[0].strip()
            return rest
    return ""


# =============================================================================
# Pre-canned response distributions for the cross-model parser
# boundary experiment (Conjecture W3-C4).
#
# These are NOT real LLM outputs. They are calibrated synthetic
# distributions that exercise distinct regions of
# ``swe_patch_parser.ALL_PARSE_KINDS`` so the parser's failure
# taxonomy distribution can be measured with reproducible seeds.
# =============================================================================


def _wrap_block(old: str, new: str) -> str:
    return (f"OLD>>>\n{old}\n<<<NEW>>>\n{new}\n<<<\n")


def _wrap_block_unclosed(old: str, new: str) -> str:
    """Block missing the trailing ``<<<`` delimiter.

    Triggers ``failure_kind=unclosed_new``; the ``robust`` parser
    recovers via ``recovery=closed_at_eos`` while ``strict`` fails.
    """
    return f"OLD>>>\n{old}\n<<<NEW>>>\n{new}\n"


def _prose_only(_old: str, _new: str) -> str:
    """Pure prose, no block, no diff — exercises ``prose_only``."""
    return ("The bug is on the function header. You need to "
             "replace the misspelled identifier with the correct "
             "one to make the tests pass.\n")


def _empty_response(_old: str, _new: str) -> str:
    return ""


def _fenced_only(_old: str, _new: str) -> str:
    """Code fence without OLD/NEW labels — exercises
    ``fenced_only`` (and ``fenced_code_heuristic`` when the fence
    has compatible shape)."""
    return ("```python\n"
             "def foo():\n"
             "    return 1\n"
             "```\n")


def _multi_block(old: str, new: str) -> str:
    """Two valid blocks — exercises ``multi_block``."""
    return (f"OLD>>>\n{old}\n<<<NEW>>>\n{new}\n<<<\n"
             f"OLD>>>\n{old}1\n<<<NEW>>>\n{new}1\n<<<\n")


# A "model profile" is a function that takes a (clean_old,
# clean_new) pair and returns a canned response of the chosen
# shape. The pair is supplied by the test harness that knows the
# correct edit; the canned producer decides how to format (or
# misformat) it.

ModelProfile = Callable[[str, str], str]


SYNTHETIC_MODEL_PROFILES: dict[str, ModelProfile] = {
    # The "clean" model emits the byte-perfect block on every
    # call. ``failure_kind=ok``, ``recovery=""``.
    "synthetic.clean": _wrap_block,
    # The "unclosed" model omits the trailing ``<<<`` —
    # ``failure_kind=unclosed_new`` under strict;
    # ``ok + recovery=closed_at_eos`` under robust.
    "synthetic.unclosed": _wrap_block_unclosed,
    # The "prose" model writes only prose — ``failure_kind=prose_only``.
    "synthetic.prose": _prose_only,
    # The "empty" model returns an empty string —
    # ``failure_kind=empty_output``.
    "synthetic.empty": _empty_response,
    # The "fenced" model emits a code fence without OLD/NEW —
    # ``failure_kind=fenced_only``.
    "synthetic.fenced": _fenced_only,
    # The "multi_block" model emits two blocks — ``failure_kind=multi_block``.
    "synthetic.multi_block": _multi_block,
    # The "mixed" model emits a 60/30/10 mix of clean / unclosed /
    # prose, partitioned by hash(instance_id). Used to exercise
    # the parser on a non-degenerate distribution.
    "synthetic.mixed": "_mixed_handled_below",  # type: ignore[dict-item]
}


def _mixed_response(old: str, new: str, instance_id: str) -> str:
    """Hash-partitioned 60/30/10 mix of clean / unclosed / prose.

    Hashing makes the assignment deterministic across runs while
    distributing failure kinds non-trivially over the bank.
    """
    bucket = sum(ord(c) for c in instance_id) % 10
    if bucket < 6:
        return _wrap_block(old, new)
    if bucket < 9:
        return _wrap_block_unclosed(old, new)
    return _prose_only(old, new)


def make_synthetic_response_fn(
        model_tag: str,
        gold_patches: dict[str, tuple[str, str]],
        ) -> ResponseFn:
    """Build a ``ResponseFn`` for ``model_tag`` over a bank of
    ``gold_patches`` ({instance_id: (old, new)}).

    Returns a closure suitable for
    ``SyntheticLLMClient(response_fn=...)``. The closure ignores
    the prompt content (only ``instance_id_guess`` is used to
    look up the gold pair).
    """
    if model_tag not in SYNTHETIC_MODEL_PROFILES:
        raise KeyError(
            f"unknown synthetic model_tag {model_tag!r}; "
            f"valid: {sorted(SYNTHETIC_MODEL_PROFILES)}")

    if model_tag == "synthetic.mixed":
        def _fn(_prompt: str, instance_id: str) -> str:
            old, new = gold_patches.get(instance_id, ("", ""))
            return _mixed_response(old, new, instance_id)
        return _fn

    profile = SYNTHETIC_MODEL_PROFILES[model_tag]

    def _fn(_prompt: str, instance_id: str) -> str:
        old, new = gold_patches.get(instance_id, ("", ""))
        return profile(old, new)
    return _fn


__all__ = [
    "SyntheticLLMClient",
    "ResponseFn",
    "ModelProfile",
    "SYNTHETIC_MODEL_PROFILES",
    "make_synthetic_response_fn",
]
