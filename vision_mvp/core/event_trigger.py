"""Event-triggered refinement — only call the LLM when the agent's current
draft genuinely disagrees with the routed bulletin on observable structure.

Motivation: Phase 13 showed that simultaneous round-2 refinement causes LLMs
to over-correct. An agent whose round-1 draft was already consistent with its
teammates' conventions would re-draft anyway and accidentally desync. The fix
is to suppress the round-2 LLM call whenever there is nothing to refine.

This is the "event-triggered control" pattern (Tabuada; Heemels): act only
when the local deviation crosses a Lyapunov-style threshold. Here the
"deviation" is a schema-level disagreement score: how different are the dict
keys the agent is using from the dict keys its bulletin implies?

The score is computed by static analysis on the source code — no LLM involved,
deterministic, fast. Jaccard distance is the natural choice: it's symmetric,
scale-invariant (immune to draft length), and 0 exactly when the key sets agree.

When disagreement < threshold, we emit a "skip" event — the agent keeps its
round-1 draft unchanged, and we charge zero tokens for that agent's round 2.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


def extract_dict_keys(code: str) -> set[str]:
    """Return every string literal used as a dict key or dict-subscript key.

    Catches:
      - dict literals:    `{"ok": True, "value": x}`        → {"ok", "value"}
      - dict subscripts:  `d["foo"]`                         → {"foo"}
      - dict get calls:   `d.get("bar", None)`               → {"bar"}
      - ** unpacks kw:    N/A (we skip these — no key text)

    Parsing errors return an empty set (rather than raising) so this is safe
    to call on partially-valid agent outputs.
    """
    if not code:
        return set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    keys: set[str] = set()

    for node in ast.walk(tree):
        # Dict literals
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)

        # Subscript access: d["foo"]
        elif isinstance(node, ast.Subscript):
            s = node.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                keys.add(s.value)

        # .get("foo") and .pop("foo") calls
        elif isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop", "setdefault")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)

    return keys


def disagreement_score(own_draft: str,
                        bulletin_drafts: list[str]) -> float:
    """Jaccard distance between own draft's keys and union-of-bulletin keys.

    Returns a value in [0, 1]:
      - 0.0 means the key sets are identical (no disagreement)
      - 1.0 means they are disjoint (total disagreement)
      - If either side has no keys, returns 0.0 (no signal → don't trigger)
    """
    own_keys = extract_dict_keys(own_draft)
    bulletin_keys: set[str] = set()
    for src in bulletin_drafts:
        bulletin_keys |= extract_dict_keys(src)

    if not own_keys or not bulletin_keys:
        return 0.0

    intersection = len(own_keys & bulletin_keys)
    union = len(own_keys | bulletin_keys)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


@dataclass
class TriggerDecision:
    refine: bool
    score: float
    threshold: float
    own_keys: set[str]
    bulletin_keys: set[str]


def should_refine(own_draft: str, bulletin_drafts: list[str],
                   threshold: float = 0.34) -> TriggerDecision:
    """Decide whether to invoke the LLM for round-2 refinement.

    threshold=0.34 is just above the 1/3 mark: if two-thirds of the combined
    key-set is shared, agent keeps its draft; if less than two-thirds overlap,
    agent refines. Chosen so that a single disagreeing key out of 3-4 total
    does NOT trigger refinement, but a wholesale convention mismatch does.
    """
    own_keys = extract_dict_keys(own_draft)
    bulletin_keys: set[str] = set()
    for src in bulletin_drafts:
        bulletin_keys |= extract_dict_keys(src)

    score = disagreement_score(own_draft, bulletin_drafts)
    refine = score >= threshold

    return TriggerDecision(
        refine=refine,
        score=score,
        threshold=threshold,
        own_keys=own_keys,
        bulletin_keys=bulletin_keys,
    )
