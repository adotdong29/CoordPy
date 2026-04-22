"""Phase 34 — union / ensemble claim extractor.

Phase 33 Conjecture C33-4 stated an informal union bound on the
composition of two claim extractors: for an ``e_regex`` with noise
``(δ_r, ε_r, μ_r, π_r)`` and an ``e_llm`` with noise
``(δ_l, ε_l, μ_l, π_l)``, the **union** extractor ``e_regex ∪ e_llm``
has:

  * drop rate     ``δ_u ≤ δ_r · δ_l``   (a claim is only missed when
    both extractors miss it — independence assumed).
  * spurious rate ``ε_u ≤ ε_r + ε_l``   (union bound on independent
    spurious emissions).

Phase 33 left the conjecture *unmeasured* because the existing
scenario catalogues (incident / compliance / security) have regex
precision and recall = 1 on the causal events by construction, so
the LLM extractor's output was strictly *dominated* by the regex
output and the ensemble was never worth building.

Phase 34 adds a light auxiliary scenario shape (see
``experiments/phase34_ensemble_extractor``) where regex and LLM
extractors have *genuinely complementary* coverage — specifically,
a subset of the causal documents use *narrative* (natural-language)
phrasing that the Phase-32 regex cannot parse, and the LLM-style
extractor keyword-matches on narrative phrasing. The union is
measurable there.

This module provides the minimum machinery to compose extractors:

  * ``UnionExtractor``   — concatenates two extractor outputs and
    dedups on (claim_kind, payload_cid).
  * ``ExtractorFlavor``  — a small typed tag for reporting.

Scope discipline
----------------

  * The dedup is *content*-level on ``(kind, evids-sorted)``: if both
    extractors emit the same claim with identical evidence, we count
    it once. This matches the ``RoleInbox`` dedup-by-cid semantic.
  * Payload differences (e.g. different substring of the same event)
    are preserved under different dedup keys (they land as two
    handoffs from the same source, which the Phase-31 ``RoleInbox``
    also deduplicates by payload_cid).
  * The union extractor does NOT try to reconcile *conflicting*
    claims (e.g. different kinds on the same event). This is
    intentional — mislabel noise is modelled at the per-extractor
    layer; the union is an additive combiner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


ClaimTuple = tuple[str, str, tuple[int, ...]]
Extractor = Callable[[str, Sequence[Any], Any], list[ClaimTuple]]


EXTRACTOR_REGEX = "regex"
EXTRACTOR_LLM = "llm"
EXTRACTOR_ENSEMBLE = "ensemble"


@dataclass
class UnionExtractor:
    """Callable that returns the deduped union of two extractors.

    Construction:

        e_union = UnionExtractor(
            e_regex, e_llm,
            label_primary="regex",
            label_secondary="llm")

    Call:

        claims = e_union(role, events, scenario)

    Dedup is on ``(kind, tuple(sorted(evids)))``. Claims from the
    primary extractor appear first in the output; secondary-only
    claims are appended.
    """

    primary: Extractor
    secondary: Extractor
    label_primary: str = "primary"
    label_secondary: str = "secondary"
    stats: dict = field(default_factory=lambda: {
        "n_primary_only": 0,
        "n_secondary_only": 0,
        "n_shared": 0,
        "n_calls": 0,
    })

    def __call__(self, role: str, events: Sequence[Any],
                 scenario: Any) -> list[ClaimTuple]:
        primary_emissions = self.primary(role, events, scenario)
        secondary_emissions = self.secondary(role, events, scenario)
        self.stats["n_calls"] += 1

        # Dedup key: (kind, sorted evids).
        seen: dict[tuple[str, tuple[int, ...]], ClaimTuple] = {}
        primary_keys: set[tuple[str, tuple[int, ...]]] = set()
        secondary_keys: set[tuple[str, tuple[int, ...]]] = set()

        out: list[ClaimTuple] = []
        for (kind, payload, evids) in primary_emissions:
            key = (kind, tuple(sorted(evids)))
            primary_keys.add(key)
            if key not in seen:
                seen[key] = (kind, payload, tuple(evids))
                out.append(seen[key])
        for (kind, payload, evids) in secondary_emissions:
            key = (kind, tuple(sorted(evids)))
            secondary_keys.add(key)
            if key not in seen:
                seen[key] = (kind, payload, tuple(evids))
                out.append(seen[key])

        shared = primary_keys & secondary_keys
        self.stats["n_primary_only"] += len(primary_keys - shared)
        self.stats["n_secondary_only"] += len(secondary_keys - shared)
        self.stats["n_shared"] += len(shared)
        return out


def union_of(primary: Extractor, secondary: Extractor,
             *, label_primary: str = EXTRACTOR_REGEX,
             label_secondary: str = EXTRACTOR_LLM,
             ) -> UnionExtractor:
    """Convenience constructor."""
    return UnionExtractor(primary, secondary,
                           label_primary=label_primary,
                           label_secondary=label_secondary)
