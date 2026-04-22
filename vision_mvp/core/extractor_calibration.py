"""Phase 33 — empirical extractor-noise calibration.

Phase 32's ``core/extractor_noise`` injects *synthetic* i.i.d. Bernoulli
noise at a controlled rate (``drop_prob``, ``spurious_prob``,
``mislabel_prob``, ``payload_corrupt_prob``) and runs the substrate under
the resulting noisy extractor. Phase 33 closes the loop: we *measure*
the empirical noise profile of an LLM-driven extractor against a
domain's gold causal chain, and compare that profile against the
synthetic Phase-32 sweep to answer:

  (a) does the synthetic degradation model approximate the real
      extractor-induced degradation, or
  (b) where does it fail — which noise axes are over- or under-
      predicted?

This module is strictly a measurement / comparison layer. It does NOT
modify the substrate, the handoff protocol, or the extractors
themselves.

What this module provides
-------------------------

* ``ClaimComparator`` — given a ground-truth causal-chain set and a
  list of extractor emissions on the same events, computes the Phase-
  32 noise attribution: per (role, scenario) counts of
  ``dropped`` / ``mislabeled`` / ``spurious`` / ``payload_corrupted``
  emissions. The definitions match the Phase-33 contract in
  ``core/llm_extractor`` docstring.

* ``ExtractorAudit`` — dataclass holding the pooled empirical
  (δ̂, μ̂, ε̂, π̂) quadruple along with per-role breakdowns, total
  counts, and sample sizes. ``as_dict`` emits a JSON artifact.

* ``calibrate_extractor`` — the orchestrator. Given a scenario bank,
  a ``run_handoff_protocol``-style extractor harness, and a gold-
  chain predicate, runs the extractor across scenarios and computes
  the ``ExtractorAudit``.

* ``closest_synthetic_config`` — maps an ``ExtractorAudit`` to the
  nearest ``NoiseConfig`` point on a supplied sweep grid. Used by the
  Phase-33 calibration benchmark to answer "which synthetic point
  best approximates this real extractor?"

* ``compare_to_synthetic_curve`` — given an ``ExtractorAudit`` plus a
  synthetic Phase-32 sweep result (``results_phase32_noise_sweep.json``
  shape), returns a small dict comparing the real extractor's
  measured accuracy/recall/precision against the closest synthetic
  point. This is the headline figure of Phase-33 Part B.

Scope discipline (what this module does NOT do)
----------------------------------------------

  * It does NOT re-run the substrate for its own sake: accuracy /
    recall / precision come from the handoff harness. This module is
    the *attribution* layer.
  * It does NOT try to fit a parametric model to the real noise
    profile; Phase 33 reports the *measured* (δ̂, μ̂, ε̂, π̂) and the
    closest Phase-32 grid point. A future phase may fit a continuous
    model.

Theoretical anchor: RESULTS_PHASE33.md § C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


ClaimTuple = tuple[str, str, tuple[int, ...]]


# =============================================================================
# ClaimComparator — per-emission attribution
# =============================================================================


@dataclass
class ClaimComparator:
    """Classify each extractor emission against the gold causal chain.

    Definitions (matching the Phase-32 noise taxonomy):

      * **dropped**       — a gold causal ``(role, kind, evids)`` that is
        NOT present in the extractor's emissions for that role. Recall
        failure.
      * **mislabeled**    — an extractor emission whose ``evids`` overlap
        a gold emission's ``evids`` but whose ``kind`` does not match.
        Type-confusion failure.
      * **correct**       — an extractor emission whose ``(kind, evids)``
        matches (kind equal; evids overlap) a gold emission.
      * **spurious**      — an extractor emission whose ``evids`` do not
        overlap ANY gold emission's ``evids`` (or the overlap is a non-
        causal event). Precision failure.
      * **payload_corrupted** — a correct / mislabeled emission whose
        emitted payload fails to contain every whitespace token of the
        gold witness. Emitting a *superset* of the gold tokens (the
        regex extractor's usual behaviour — it emits the full event
        body) is NOT corruption; Phase-32 ``_payload_corrupt`` drops a
        middle token, which breaks the superset invariant and is
        therefore flagged.

    The comparator is deliberately *per-role*: the Phase-32 noise model
    is a per-role parameterisation; we report the empirical rates the
    same way.
    """

    def classify(self,
                 role: str,
                 emissions: Sequence[ClaimTuple],
                 gold_chain: Sequence[tuple[str, str, str, tuple[int, ...]]],
                 causal_event_ids: Sequence[int],
                 ) -> dict[str, int]:
        gold_for_role = [(kind, payload, tuple(evids))
                         for (r, kind, payload, evids) in gold_chain
                         if r == role]
        causal_ids = set(causal_event_ids)

        n_dropped = 0
        n_correct = 0
        n_mislabeled = 0
        n_spurious = 0
        n_payload_corrupted = 0

        # Score each gold emission — did the extractor produce a
        # matching (kind, overlapping evids) emission?
        matched_gold_idx: set[int] = set()
        used_emission_idx: set[int] = set()
        for gi, (gk, gpay, gevs) in enumerate(gold_for_role):
            gset = set(gevs)
            found = False
            for ei, (ek, epay, eevs) in enumerate(emissions):
                if ei in used_emission_idx:
                    continue
                if ek != gk:
                    continue
                if gset & set(eevs):
                    matched_gold_idx.add(gi)
                    used_emission_idx.add(ei)
                    gold_tokens = set(gpay.split())
                    emit_tokens = set(epay.split())
                    if gold_tokens and not gold_tokens.issubset(
                            emit_tokens):
                        n_payload_corrupted += 1
                    found = True
                    break
            if not found:
                n_dropped += 1

        for gi in range(len(gold_for_role)):
            if gi in matched_gold_idx:
                n_correct += 1

        # Score remaining emissions.
        for ei, (ek, epay, eevs) in enumerate(emissions):
            if ei in used_emission_idx:
                continue
            # Does it overlap any gold role emission (wrong kind)?
            mislabel = False
            for (gk, gpay, gevs) in gold_for_role:
                if gk == ek:
                    continue
                if set(eevs) & set(gevs):
                    mislabel = True
                    break
            if mislabel:
                n_mislabeled += 1
                continue
            # Does the emission cite a causal event (even though no gold
            # claim of this kind was expected on that event)?
            if set(eevs) & causal_ids:
                # Emission on a causal event but neither matching nor
                # a gold-kind mislabel — treat as mislabeled-on-causal.
                n_mislabeled += 1
            else:
                n_spurious += 1

        return {
            "n_dropped": n_dropped,
            "n_correct": n_correct,
            "n_mislabeled": n_mislabeled,
            "n_spurious": n_spurious,
            "n_payload_corrupted": n_payload_corrupted,
            "n_gold": len(gold_for_role),
            "n_emissions": len(emissions),
            # ``n_distractor_events`` is set by ``calibrate_extractor``
            # because the comparator does not see the full role-event
            # stream — only the gold causal_event_ids. Default to 0.
            "n_distractor_events": 0,
        }


# =============================================================================
# ExtractorAudit — pooled calibration result
# =============================================================================


@dataclass
class ExtractorAudit:
    """Pooled measurement of one extractor's empirical noise profile.

    The four headline numbers (pooled across role × scenario):

      * ``drop_rate``           = dropped / gold
        — empirical δ̂
      * ``mislabel_rate``        = mislabeled / (correct + mislabeled)
        — empirical μ̂ (per-emission rate, matching the Phase-32
        parameterisation where mislabel is a per-emission transformation)
      * ``spurious_per_event``   = spurious / distractor_events
        — empirical ε̂ (per-distractor rate, matching the Phase-32
        parameterisation where spurious emissions are per-event coin
        flips on non-causal events)
      * ``payload_corrupt_rate`` = payload_corrupted /
        (correct + mislabeled)
        — empirical π̂

    Additional fields:
      * ``by_role``       — same quadruple per role.
      * ``counts``        — raw counts dict.
      * ``n_scenarios``   — sample size.
    """

    drop_rate: float = 0.0
    mislabel_rate: float = 0.0
    spurious_per_event: float = 0.0
    payload_corrupt_rate: float = 0.0
    by_role: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    n_scenarios: int = 0
    extractor_label: str = "unlabeled"

    def as_dict(self) -> dict:
        return {
            "extractor_label": self.extractor_label,
            "drop_rate": round(self.drop_rate, 4),
            "mislabel_rate": round(self.mislabel_rate, 4),
            "spurious_per_event": round(self.spurious_per_event, 4),
            "payload_corrupt_rate": round(self.payload_corrupt_rate, 4),
            "counts": dict(self.counts),
            "by_role": {r: {k: round(v, 4) for k, v in d.items()}
                        for r, d in self.by_role.items()},
            "n_scenarios": self.n_scenarios,
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def pool_comparisons(per_role_counts: dict[str, dict[str, int]],
                     extractor_label: str = "llm",
                     n_scenarios: int = 0,
                     ) -> ExtractorAudit:
    """Turn a per-(role, scenario) count dict into an ``ExtractorAudit``.

    ``per_role_counts`` is keyed by role; the inner dict has *summed*
    counts across scenarios with keys matching ``ClaimComparator.classify``'s
    output. The function pools across roles and reports both the
    pooled quadruple and per-role rates.
    """
    total_drop = 0
    total_mis = 0
    total_corr = 0
    total_spur = 0
    total_payload_corrupt = 0
    total_gold = 0
    total_emit = 0
    total_distractor = 0
    by_role: dict[str, dict[str, float]] = {}
    for role, c in per_role_counts.items():
        d = c.get("n_dropped", 0)
        mi = c.get("n_mislabeled", 0)
        co = c.get("n_correct", 0)
        sp = c.get("n_spurious", 0)
        pc = c.get("n_payload_corrupted", 0)
        n_gold = c.get("n_gold", 0)
        n_emit = c.get("n_emissions", 0)
        n_dist = c.get("n_distractor_events", 0)
        total_drop += d
        total_mis += mi
        total_corr += co
        total_spur += sp
        total_payload_corrupt += pc
        total_gold += n_gold
        total_emit += n_emit
        total_distractor += n_dist
        by_role[role] = {
            "drop_rate": _safe_div(d, max(1, n_gold)),
            "mislabel_rate": _safe_div(mi, max(1, co + mi)),
            "spurious_per_event": _safe_div(sp, max(1, n_dist)),
            "payload_corrupt_rate": _safe_div(
                pc, max(1, co + mi)),
        }
    return ExtractorAudit(
        drop_rate=_safe_div(total_drop, max(1, total_gold)),
        mislabel_rate=_safe_div(total_mis,
                                max(1, total_corr + total_mis)),
        spurious_per_event=_safe_div(total_spur,
                                     max(1, total_distractor)),
        payload_corrupt_rate=_safe_div(total_payload_corrupt,
                                        max(1, total_corr + total_mis)),
        by_role=by_role,
        counts={
            "dropped": total_drop,
            "mislabeled": total_mis,
            "correct": total_corr,
            "spurious": total_spur,
            "payload_corrupted": total_payload_corrupt,
            "gold": total_gold,
            "emissions": total_emit,
            "distractor_events": total_distractor,
        },
        n_scenarios=n_scenarios,
        extractor_label=extractor_label,
    )


# =============================================================================
# calibrate_extractor — full pipeline
# =============================================================================


def calibrate_extractor(
        extractor_cb,
        scenarios: Sequence[Any],
        role_events_getter,
        causal_events_getter,
        gold_chain_getter,
        roles_to_probe: Sequence[str],
        extractor_label: str = "llm",
        ) -> ExtractorAudit:
    """Run an extractor across a scenario bank and compute the audit.

    * ``extractor_cb``          — the extractor, same signature as
      ``LLMExtractor.__call__``.
    * ``scenarios``             — list of scenario objects.
    * ``role_events_getter``    — ``fn(scenario, role) -> list[event]``.
    * ``causal_events_getter``  — ``fn(scenario, role) -> iterable of
      event_ids on causal events only``.
    * ``gold_chain_getter``     — ``fn(scenario) -> gold causal_chain``.
    * ``roles_to_probe``        — which roles to evaluate (skip the
      aggregator).
    """
    comparator = ClaimComparator()
    per_role_counts: dict[str, dict[str, int]] = {}
    for scenario in scenarios:
        chain = gold_chain_getter(scenario)
        for role in roles_to_probe:
            events = list(role_events_getter(scenario, role))
            emissions = extractor_cb(role, events, scenario)
            causal_ids = list(causal_events_getter(scenario, role))
            c = comparator.classify(role, emissions, chain, causal_ids)
            # Track the role's distractor-event count for spurious rate.
            distractor_count = sum(
                1 for ev in events
                if getattr(ev, "event_id", getattr(ev, "doc_id", None))
                   not in set(causal_ids))
            c["n_distractor_events"] = distractor_count
            slot = per_role_counts.setdefault(role, {
                "n_dropped": 0, "n_correct": 0, "n_mislabeled": 0,
                "n_spurious": 0, "n_payload_corrupted": 0,
                "n_gold": 0, "n_emissions": 0, "n_distractor_events": 0,
            })
            for k, v in c.items():
                if k in slot:
                    slot[k] += v
    return pool_comparisons(per_role_counts,
                            extractor_label=extractor_label,
                            n_scenarios=len(scenarios))


# =============================================================================
# Closest-synthetic match
# =============================================================================


@dataclass
class SyntheticMatch:
    """Closest Phase-32 sweep grid point to an ``ExtractorAudit``."""

    drop_prob: float
    spurious_prob: float
    mislabel_prob: float
    payload_corrupt_prob: float
    l1_distance: float
    synthetic_accuracy: float
    synthetic_recall: float
    synthetic_precision: float
    synthetic_tokens: float

    def as_dict(self) -> dict:
        return {
            "drop_prob": self.drop_prob,
            "spurious_prob": self.spurious_prob,
            "mislabel_prob": self.mislabel_prob,
            "payload_corrupt_prob": self.payload_corrupt_prob,
            "l1_distance": round(self.l1_distance, 4),
            "synthetic_accuracy": round(self.synthetic_accuracy, 4),
            "synthetic_recall": round(self.synthetic_recall, 4),
            "synthetic_precision": round(self.synthetic_precision, 4),
            "synthetic_tokens": round(self.synthetic_tokens, 2),
        }


def closest_synthetic_config(audit: ExtractorAudit,
                              sweep_pooled: Mapping[str, dict],
                              domain: str | None = None,
                              ) -> SyntheticMatch | None:
    """Find the Phase-32 sweep grid point closest to the audit.

    ``sweep_pooled`` is the ``pooled`` dict from
    ``results_phase32_noise_sweep.json`` (keys like
    ``incident_drop0.1_sp0.05_mis0.0_corr0.0``).

    Distance is L1 in (drop, spurious, mislabel, payload_corrupt)
    space, weighted equally.
    """
    best: SyntheticMatch | None = None
    for key, row in sweep_pooled.items():
        if domain is not None and row.get("domain") != domain:
            continue
        dp = float(row["drop_prob"])
        sp = float(row["spurious_prob"])
        mi = float(row["mislabel_prob"])
        co = float(row.get("payload_corrupt_prob", 0.0))
        dist = (abs(dp - audit.drop_rate)
                + abs(sp - audit.spurious_per_event)
                + abs(mi - audit.mislabel_rate)
                + abs(co - audit.payload_corrupt_rate))
        cand = SyntheticMatch(
            drop_prob=dp, spurious_prob=sp, mislabel_prob=mi,
            payload_corrupt_prob=co,
            l1_distance=dist,
            synthetic_accuracy=float(row.get("accuracy_mean", 0.0)),
            synthetic_recall=float(row.get("recall_mean", 0.0)),
            synthetic_precision=float(row.get("precision_mean", 0.0)),
            synthetic_tokens=float(row.get("tokens_mean", 0.0)),
        )
        if best is None or cand.l1_distance < best.l1_distance:
            best = cand
    return best


def compare_to_synthetic_curve(audit: ExtractorAudit,
                                sweep_pooled: Mapping[str, dict],
                                real_measured_accuracy: float,
                                real_measured_recall: float,
                                real_measured_precision: float,
                                domain: str,
                                ) -> dict:
    """Report the real vs synthetic comparison as a small dict.

    Keys:
      * ``real``         — the ``ExtractorAudit`` as_dict plus the
        real measured (accuracy, recall, precision).
      * ``synthetic_match`` — the closest Phase-32 grid point.
      * ``residual``     — accuracy/recall/precision gap (real - synth).
      * ``verdict``      — "approximates" if |accuracy_gap| < 0.15 and
        |recall_gap| < 0.15 and |precision_gap| < 0.15; else
        "partial" if within 0.25; else "diverges".
    """
    match = closest_synthetic_config(audit, sweep_pooled, domain=domain)
    if match is None:
        return {
            "real": audit.as_dict(),
            "synthetic_match": None,
            "residual": None,
            "verdict": "no_match",
        }
    acc_gap = real_measured_accuracy - match.synthetic_accuracy
    rec_gap = real_measured_recall - match.synthetic_recall
    prec_gap = real_measured_precision - match.synthetic_precision
    max_gap = max(abs(acc_gap), abs(rec_gap), abs(prec_gap))
    if max_gap < 0.15:
        verdict = "approximates"
    elif max_gap < 0.25:
        verdict = "partial"
    else:
        verdict = "diverges"
    return {
        "real": {
            **audit.as_dict(),
            "measured_accuracy": round(real_measured_accuracy, 4),
            "measured_recall": round(real_measured_recall, 4),
            "measured_precision": round(real_measured_precision, 4),
            "domain": domain,
        },
        "synthetic_match": match.as_dict(),
        "residual": {
            "accuracy_gap": round(acc_gap, 4),
            "recall_gap": round(rec_gap, 4),
            "precision_gap": round(prec_gap, 4),
            "max_abs_gap": round(max_gap, 4),
        },
        "verdict": verdict,
    }


# =============================================================================
# Phase 34 — per-role heterogeneity analysis
# =============================================================================


def _weakest_role(by_role: Mapping[str, Mapping[str, float]],
                   axis: str = "drop_rate") -> tuple[str, float] | None:
    """Return ``(role, rate)`` of the role with the highest noise on
    ``axis`` (higher = weaker). ``None`` if the dict is empty."""
    best_role = None
    best_rate = -1.0
    for role, rates in by_role.items():
        r = float(rates.get(axis, 0.0))
        if r > best_rate:
            best_role = role
            best_rate = r
    if best_role is None:
        return None
    return (best_role, best_rate)


def _strongest_role(by_role: Mapping[str, Mapping[str, float]],
                      axis: str = "drop_rate") -> tuple[str, float] | None:
    best_role = None
    best_rate = 2.0
    for role, rates in by_role.items():
        r = float(rates.get(axis, 0.0))
        if r < best_rate:
            best_role = role
            best_rate = r
    if best_role is None:
        return None
    return (best_role, best_rate)


def per_role_heterogeneity(audit: ExtractorAudit) -> dict:
    """Summarise per-role noise spread from an ``ExtractorAudit``.

    Returns a dict with:

    * ``axes``            — per axis: pooled rate, min across roles, max
      across roles, spread = max - min, weakest role (argmax),
      strongest role (argmin).
    * ``pooled_masks_per_role``  — True iff the max-min spread on any
      axis exceeds 0.25 (i.e. some role is > 0.25 worse than another
      on the same noise dimension; that is the Conjecture C33-3
      signal).
    * ``worst_role_by_axis`` — mapping axis → role.

    The function never fails; it just reports zeros when the audit has
    no per-role data.
    """
    axes = ("drop_rate", "mislabel_rate", "spurious_per_event",
            "payload_corrupt_rate")
    axis_pooled = {
        "drop_rate": audit.drop_rate,
        "mislabel_rate": audit.mislabel_rate,
        "spurious_per_event": audit.spurious_per_event,
        "payload_corrupt_rate": audit.payload_corrupt_rate,
    }
    per_axis: dict[str, dict] = {}
    worst_role_by_axis: dict[str, str] = {}
    max_spread = 0.0
    for ax in axes:
        rates = [float(r.get(ax, 0.0)) for r in audit.by_role.values()]
        if not rates:
            per_axis[ax] = {
                "pooled": round(axis_pooled[ax], 4),
                "min": 0.0, "max": 0.0, "spread": 0.0,
                "weakest_role": None, "strongest_role": None,
            }
            continue
        lo = min(rates)
        hi = max(rates)
        spread = hi - lo
        max_spread = max(max_spread, spread)
        weakest = _weakest_role(audit.by_role, ax)
        strongest = _strongest_role(audit.by_role, ax)
        per_axis[ax] = {
            "pooled": round(axis_pooled[ax], 4),
            "min": round(lo, 4),
            "max": round(hi, 4),
            "spread": round(spread, 4),
            "weakest_role": weakest[0] if weakest else None,
            "weakest_rate": round(weakest[1], 4) if weakest else None,
            "strongest_role": strongest[0] if strongest else None,
            "strongest_rate": round(strongest[1], 4) if strongest else None,
        }
        if weakest:
            worst_role_by_axis[ax] = weakest[0]
    return {
        "per_axis": per_axis,
        "worst_role_by_axis": worst_role_by_axis,
        "max_spread_any_axis": round(max_spread, 4),
        "pooled_masks_per_role": max_spread > 0.25,
    }


def per_role_closest_synthetic(audit: ExtractorAudit,
                                 sweep_pooled: Mapping[str, dict],
                                 domain: str | None = None,
                                 ) -> dict[str, SyntheticMatch]:
    """For each role in ``audit.by_role``, return the closest Phase-32
    synthetic grid point as a ``SyntheticMatch``.

    The per-role match is the instrument for Phase 34's per-role
    calibration report: it shows that different roles land on different
    sweep grid points, i.e. the pooled match is an average over
    qualitatively different regimes.
    """
    out: dict[str, SyntheticMatch] = {}
    for role, rates in audit.by_role.items():
        role_audit = ExtractorAudit(
            drop_rate=float(rates.get("drop_rate", 0.0)),
            mislabel_rate=float(rates.get("mislabel_rate", 0.0)),
            spurious_per_event=float(rates.get("spurious_per_event", 0.0)),
            payload_corrupt_rate=float(
                rates.get("payload_corrupt_rate", 0.0)),
            by_role={}, counts={},
            n_scenarios=audit.n_scenarios,
            extractor_label=f"{audit.extractor_label}:{role}",
        )
        m = closest_synthetic_config(role_audit, sweep_pooled,
                                       domain=domain)
        if m is not None:
            out[role] = m
    return out


def per_role_audit_summary(audit: ExtractorAudit,
                             sweep_pooled: Mapping[str, dict] | None = None,
                             domain: str | None = None,
                             ) -> dict:
    """Pretty-printable per-role calibration summary.

    Returns:

    * ``pooled``  — the pooled quadruple + pooled synthetic match
      (if sweep is provided).
    * ``by_role`` — per-role quadruple + per-role synthetic match
      (if sweep is provided).
    * ``heterogeneity`` — output of ``per_role_heterogeneity``.
    * ``role_limited`` — the role with the highest drop rate is the
      likely bottleneck on the substrate's accuracy.
    """
    pooled_match = (closest_synthetic_config(audit, sweep_pooled,
                                                domain=domain)
                    if sweep_pooled else None)
    by_role: dict[str, dict] = {}
    per_role_matches = (per_role_closest_synthetic(audit, sweep_pooled,
                                                      domain=domain)
                        if sweep_pooled else {})
    for role, rates in audit.by_role.items():
        by_role[role] = {
            "rates": {k: round(float(v), 4) for k, v in rates.items()},
            "synthetic_match": (per_role_matches[role].as_dict()
                                  if role in per_role_matches else None),
        }
    het = per_role_heterogeneity(audit)
    weakest_drop = _weakest_role(audit.by_role, "drop_rate")
    return {
        "pooled": {
            "rates": {
                "drop_rate": round(audit.drop_rate, 4),
                "mislabel_rate": round(audit.mislabel_rate, 4),
                "spurious_per_event": round(audit.spurious_per_event, 4),
                "payload_corrupt_rate": round(
                    audit.payload_corrupt_rate, 4),
            },
            "synthetic_match": (pooled_match.as_dict()
                                 if pooled_match else None),
        },
        "by_role": by_role,
        "heterogeneity": het,
        "role_limited_by": (weakest_drop[0]
                              if weakest_drop else None),
        "role_limited_rate": (round(weakest_drop[1], 4)
                                if weakest_drop else None),
        "domain": domain,
        "extractor_label": audit.extractor_label,
        "n_scenarios": audit.n_scenarios,
    }
