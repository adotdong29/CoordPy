"""W113 / COO-9 — LiveCodeBench contamination-resistance-FOR-A-MODEL rule.

W112 established the methodological finding that contamination-resistance is
**MODEL-CUTOFF-RELATIVE** (``W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-
RELATIVE``): BigCodeBench 2024-06 is contamination-RESISTANT for
``meta/llama-3.3-70b-instruct`` (~2024-01 cutoff) but contamination-EXPOSED for
``meta/llama-4-maverick-17b-128e-instruct`` (August-2024 cutoff), and the SAME
30-slice flips B-A1 = +0.00pp -> +10.00pp as the model cutoff crosses the
benchmark release date.  So "contamination-resistant" is NOT a property of a
benchmark alone — it is a property of the PAIR ``(benchmark slice, model
cutoff)``.

This module makes that relation **machine-checkable** for the W113 main lane.
A LiveCodeBench functional problem is RESISTANT-FOR a model iff its
``contest_date`` is STRICTLY AFTER that model's training cutoff.  For
Llama-4-Maverick (Meta-stated cutoff *August 2024*) the boundary is the last
day of that month, ``2024-08-31`` — RESISTANT iff ``contest_date >=
2024-09-01``.  This deliberately excludes the ENTIRE ambiguous August-2024
window (the cutoff is published at month granularity), which is the
conservative, defensible choice: a problem dated in August 2024 cannot be
*certified* strictly-after an August-2024 cutoff, so it is EXCLUDED rather than
counted.

Design (pure / deterministic / NIM-free / read-only):

* ``MODEL_TRAINING_CUTOFFS`` — a small registry of ``(boundary_date,
  confidence)`` per model.  ``confidence`` is one of ``KNOWN`` (vendor/Meta-
  stated or repo-established), ``ESTIMATED`` (inferred from release date), or
  ``UNKNOWN`` (undocumented).  The resistance / spend rules REFUSE to certify a
  slice resistant for a model whose cutoff is not ``KNOWN`` — you cannot claim a
  slice is resistant for a model whose cutoff you cannot verify (the W112 lesson
  as discipline).
* ``normalize_contest_date_v1`` — parse the upstream ``contest_date`` (e.g.
  ``"2025-01-11T18:30:00"``) to a ``YYYY-MM-DD`` day string, or ``None`` if
  missing / unparseable (the ambiguity-EXCLUSION rule: an unparseable date is
  never counted resistant).  Comparison is at DAY granularity; the upstream
  field carries no timezone offset and the W113 slice is months past the
  Maverick cutoff, so sub-day / tz normalization is immaterial and is documented
  as out of scope (``W113-L-CONTEST-DATE-DAY-GRANULARITY-CAP``).
* ``partition_resistant_v1`` — split a functional subset into the RESISTANT
  problems (date strictly > boundary) and the EXCLUDED ones, with a typed
  breakdown of WHY each was excluded (missing / unparseable / not-after-cutoff).
* ``cutoff_boundary_for_model_v1`` / ``resistant_partition_for_model_v1`` —
  apply the registry by model id (raise on unknown model).
* ``slice_resistant_for_model_v1`` — the tier-2 APPLICABILITY predicate: a
  pre-built slice (given its min kept date) is resistant for a model iff that
  model's cutoff is KNOWN and the min date is strictly after the boundary.

Honest scope (W113):

* ``W113-L-CONTEST-DATE-DAY-GRANULARITY-CAP`` — resistance is decided at day
  granularity from ``contest_date[:10]``; no intra-day / timezone normalization.
* ``W113-L-MODEL-CUTOFF-REGISTRY-CONFIDENCE-CAP`` — the registry's boundary
  dates are only as good as the vendor disclosure; ``ESTIMATED`` / ``UNKNOWN``
  entries are NOT certification-grade and the spend rule refuses them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Sequence

W113_RESISTANT_SLICE_V1_SCHEMA_VERSION: str = (
    "coordpy.livecodebench_resistant_slice_v1.v1")

# Confidence tags for a registry cutoff entry.
CONFIDENCE_KNOWN: str = "KNOWN"          # vendor/Meta-stated or repo-established
CONFIDENCE_ESTIMATED: str = "ESTIMATED"  # inferred from release date
CONFIDENCE_UNKNOWN: str = "UNKNOWN"      # undocumented

# Minimum resistant-slice size for a cheap pilot (mirrors W108/W110/W112).
MIN_RESISTANT_SLICE: int = 30


@dataclasses.dataclass(frozen=True)
class ModelCutoffV1:
    """One model's training-cutoff boundary.

    ``boundary_date`` is the last in-distribution day (``YYYY-MM-DD``); a
    problem is RESISTANT for this model iff ``contest_date_day > boundary_date``
    (string compare on ISO day strings == chronological compare).
    """

    boundary_date: str
    confidence: str
    note: str = ""

    def is_resistant_grade(self) -> bool:
        """Only a KNOWN boundary can CERTIFY a slice resistant for a model."""
        return self.confidence == CONFIDENCE_KNOWN


# Registry of training-cutoff boundaries.  Resistant iff contest_date strictly
# AFTER boundary_date.
#
# * Llama-3.3-70B: real cutoff December 2023; boundary 2023-12-31 makes
#   ``date >= 2024-01-01`` resistant — exactly the W108 preflight convention
#   (``LLAMA_3X_CUTOFF_DATE = "2024-01-01"`` counted ``d >= cutoff`` as
#   post-cutoff).  KNOWN (repo-established + Meta-stated).
# * Llama-4-Maverick: Meta-stated cutoff *August 2024*; boundary 2024-08-31 ->
#   resistant iff ``date >= 2024-09-01`` (the W112 fact, conservative).  KNOWN.
# * The three reachable tier-2 stronger models (W112 sweep): all RELEASED in
#   2025-2026 with UNDOCUMENTED cutoffs (Qwen3-Coder-480B released 2025-07,
#   cutoff undisclosed; DeepSeek-V4-pro 2025+; Mistral-Small-4 "2603" =
#   2026-03 release).  Their cutoffs plausibly OVERLAP or POST-DATE the
#   LiveCodeBench 2025-01..04 release_v6 window, so the test6 slice is NOT
#   certifiably resistant for them.  Recorded UNKNOWN -> the spend rule refuses
#   to certify resistance (W113-L-MODEL-CUTOFF-REGISTRY-CONFIDENCE-CAP).
MODEL_TRAINING_CUTOFFS: dict[str, ModelCutoffV1] = {
    "meta/llama-3.3-70b-instruct": ModelCutoffV1(
        boundary_date="2023-12-31", confidence=CONFIDENCE_KNOWN,
        note="cutoff Dec-2023; matches W108 LLAMA_3X_CUTOFF 2024-01-01 post-rule"),
    "meta/llama-4-maverick-17b-128e-instruct": ModelCutoffV1(
        boundary_date="2024-08-31", confidence=CONFIDENCE_KNOWN,
        note="Meta-stated cutoff August 2024 (W112); resistant iff >= 2024-09-01"),
    "qwen/qwen3-coder-480b-a35b-instruct": ModelCutoffV1(
        boundary_date="2025-07-01", confidence=CONFIDENCE_UNKNOWN,
        note="released 2025-07-22; cutoff UNDOCUMENTED; ESTIMATE only — not "
             "certification-grade; likely exposed on a 2025-01..04 slice"),
    "deepseek-ai/deepseek-v4-pro": ModelCutoffV1(
        boundary_date="2025-01-01", confidence=CONFIDENCE_UNKNOWN,
        note="V4 post-dates V3 (2024-12); cutoff UNDOCUMENTED; likely overlaps "
             "the 2025-01..04 slice"),
    "mistralai/mistral-small-4-119b-2603": ModelCutoffV1(
        boundary_date="2026-03-01", confidence=CONFIDENCE_UNKNOWN,
        note="'2603' release tag => 2026-03; cutoff UNDOCUMENTED but POST-dates "
             "the entire test6 window => test6 slice EXPOSED for it"),
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_ISO_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def normalize_contest_date_v1(raw: str | None) -> str | None:
    """Normalize an upstream ``contest_date`` to a ``YYYY-MM-DD`` day string.

    Returns ``None`` when the value is missing, blank, or does not begin with a
    valid ISO date — the AMBIGUITY-EXCLUSION rule (an unparseable date is never
    counted resistant).  Validates month/day ranges so e.g. ``2025-13-40`` is
    rejected rather than silently passed through.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    m = _ISO_DAY_RE.match(s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def is_resistant_for_boundary_v1(
        date_norm: str | None, boundary_date: str) -> bool:
    """RESISTANT iff the normalized day is STRICTLY AFTER the boundary.

    ISO ``YYYY-MM-DD`` strings sort chronologically, so ``>`` is the
    strictly-after test.  A ``None`` (unparseable / missing) day is never
    resistant.
    """
    if date_norm is None:
        return False
    return date_norm > str(boundary_date)


@dataclasses.dataclass(frozen=True)
class ResistantPartitionV1:
    """The result of partitioning a functional subset by resistance."""

    schema: str
    boundary_date: str
    n_total: int
    n_resistant: int
    resistant_question_ids: tuple[str, ...]
    excluded_missing_date: tuple[str, ...]
    excluded_unparseable_date: tuple[str, ...]
    excluded_not_after_cutoff: tuple[str, ...]
    resistant_date_min: str
    resistant_date_max: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_date": str(self.boundary_date),
            "n_total": int(self.n_total),
            "n_resistant": int(self.n_resistant),
            "n_excluded_missing_date": len(self.excluded_missing_date),
            "n_excluded_unparseable_date": len(self.excluded_unparseable_date),
            "n_excluded_not_after_cutoff": len(self.excluded_not_after_cutoff),
            "resistant_question_ids": list(self.resistant_question_ids),
            "excluded_missing_date": list(self.excluded_missing_date),
            "excluded_unparseable_date": list(self.excluded_unparseable_date),
            "excluded_not_after_cutoff": list(self.excluded_not_after_cutoff),
            "resistant_date_min": str(self.resistant_date_min),
            "resistant_date_max": str(self.resistant_date_max),
        }

    def partition_cid(self) -> str:
        return _sha256_hex({
            "kind": "w113_resistant_partition_v1",
            "boundary_date": str(self.boundary_date),
            "resistant_question_ids": list(self.resistant_question_ids),
        })


def partition_resistant_v1(
        subset: Sequence[Any], *, boundary_date: str) -> ResistantPartitionV1:
    """Split a LiveCodeBench functional subset by resistance to ``boundary_date``.

    Each problem needs ``.question_id`` and ``.contest_date`` attributes (the
    ``LiveCodeBenchProblemV1`` shape).  RESISTANT iff its normalized day is
    strictly after the boundary; otherwise EXCLUDED with a typed reason.  The
    resistant ids preserve the input order (the loader's order) so a downstream
    deterministic slice selector is stable.
    """
    resistant_ids: list[str] = []
    miss: list[str] = []
    unparse: list[str] = []
    not_after: list[str] = []
    resistant_days: list[str] = []
    for p in subset:
        qid = str(getattr(p, "question_id", ""))
        raw = getattr(p, "contest_date", "")
        if not str(raw or "").strip():
            miss.append(qid)
            continue
        day = normalize_contest_date_v1(raw)
        if day is None:
            unparse.append(qid)
            continue
        if is_resistant_for_boundary_v1(day, boundary_date):
            resistant_ids.append(qid)
            resistant_days.append(day)
        else:
            not_after.append(qid)
    return ResistantPartitionV1(
        schema=W113_RESISTANT_SLICE_V1_SCHEMA_VERSION,
        boundary_date=str(boundary_date),
        n_total=len(list(subset)),
        n_resistant=len(resistant_ids),
        resistant_question_ids=tuple(resistant_ids),
        excluded_missing_date=tuple(miss),
        excluded_unparseable_date=tuple(unparse),
        excluded_not_after_cutoff=tuple(not_after),
        resistant_date_min=(min(resistant_days) if resistant_days else ""),
        resistant_date_max=(max(resistant_days) if resistant_days else ""),
    )


def cutoff_boundary_for_model_v1(model_id: str) -> ModelCutoffV1:
    """Look up a model's cutoff boundary.  Raises ``KeyError`` on unknown id."""
    key = str(model_id)
    if key not in MODEL_TRAINING_CUTOFFS:
        raise KeyError(
            f"unknown model {key!r}; no training-cutoff registered. Refusing "
            "to guess a cutoff (W113 contamination-resistance is model-cutoff-"
            "relative and must be certified, not assumed).")
    return MODEL_TRAINING_CUTOFFS[key]


def resistant_partition_for_model_v1(
        subset: Sequence[Any], *, model_id: str) -> ResistantPartitionV1:
    """Partition a functional subset by resistance for a specific model."""
    cutoff = cutoff_boundary_for_model_v1(model_id)
    return partition_resistant_v1(subset, boundary_date=cutoff.boundary_date)


def slice_resistant_for_model_v1(
        *, slice_date_min: str, model_id: str) -> tuple[bool, str]:
    """Tier-2 APPLICABILITY predicate.

    A pre-built slice (identified by its minimum ``contest_date`` day) is
    CERTIFIABLY resistant for ``model_id`` iff the model's cutoff is KNOWN AND
    every problem (so, the minimum) is strictly after the boundary.  Returns
    ``(is_resistant, reason)``.  Refuses (``False``) for ESTIMATED / UNKNOWN
    cutoffs — the W112 lesson: you cannot certify resistance against a cutoff
    you cannot verify.
    """
    try:
        cutoff = cutoff_boundary_for_model_v1(model_id)
    except KeyError as e:
        return False, f"UNREGISTERED_MODEL: {e}"
    if not cutoff.is_resistant_grade():
        return False, (
            f"CUTOFF_{cutoff.confidence}: cannot certify resistance against a "
            f"non-KNOWN cutoff (boundary≈{cutoff.boundary_date}; {cutoff.note})")
    day = normalize_contest_date_v1(slice_date_min)
    if day is None:
        return False, "SLICE_DATE_MIN_UNPARSEABLE"
    if day > cutoff.boundary_date:
        return True, (
            f"RESISTANT: slice_min {day} > KNOWN cutoff boundary "
            f"{cutoff.boundary_date}")
    return False, (
        f"EXPOSED: slice_min {day} <= cutoff boundary {cutoff.boundary_date}")


__all__ = [
    "W113_RESISTANT_SLICE_V1_SCHEMA_VERSION",
    "CONFIDENCE_KNOWN",
    "CONFIDENCE_ESTIMATED",
    "CONFIDENCE_UNKNOWN",
    "MIN_RESISTANT_SLICE",
    "ModelCutoffV1",
    "MODEL_TRAINING_CUTOFFS",
    "normalize_contest_date_v1",
    "is_resistant_for_boundary_v1",
    "ResistantPartitionV1",
    "partition_resistant_v1",
    "cutoff_boundary_for_model_v1",
    "resistant_partition_for_model_v1",
    "slice_resistant_for_model_v1",
]
