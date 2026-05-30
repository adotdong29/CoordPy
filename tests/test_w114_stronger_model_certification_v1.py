"""W114 / COO-9 — tests for the per-model post-cutoff certification layer.

Covers ``coordpy.stronger_model_cutoff_certification_v1``:

* the instrument's month-granular ``n_functional_resistant_after`` count;
* the C1..C4 certification gate per candidate (KNOWN-cutoff-only; >=30
  functional resistant; reachable/stronger/comparable; not-already-settled);
* the LOCKED W114 decision = ``NO_CERTIFIABLE_STRONGER_MODEL`` on the real data,
  with Maverick certifiable-but-settled (C4) and the frontier models C1-blocked;
* the W113<->W114 confidence consistency guard;
* a FALSIFIABILITY test: a synthetic KNOWN-cutoff, not-settled candidate on a
  synthetic post-cutoff instrument DOES certify (the rule is not hard-wired to
  no-go).
"""
from __future__ import annotations

import dataclasses

from coordpy.stronger_model_cutoff_certification_v1 import (
    LATEST_RESISTANT_INSTRUMENT,
    LatestResistantInstrumentV1,
    STRONGER_MODEL_CANDIDATES,
    StrongerModelCandidateV1,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W114_CUTOFF_PROVENANCE,
    certify_model_v1,
    decide_certification_v1,
)
from coordpy.livecodebench_resistant_slice_v1 import (
    MIN_RESISTANT_SLICE,
    MODEL_TRAINING_CUTOFFS,
)


# ---------------------------------------------------------------- instrument

def test_instrument_is_the_dataclass_not_a_tuple():
    # regression guard: the module constant must be the dataclass instance.
    assert isinstance(LATEST_RESISTANT_INSTRUMENT, LatestResistantInstrumentV1)
    assert LATEST_RESISTANT_INSTRUMENT.release == "release_v6"
    assert LATEST_RESISTANT_INSTRUMENT.n_functional == 63
    assert LATEST_RESISTANT_INSTRUMENT.functional_date_max == "2025-04-05"


def test_month_granular_resistant_counts_match_corpus_histogram():
    inst = LATEST_RESISTANT_INSTRUMENT
    # boundary in Aug-2024 (Maverick): all 63 functional are 2025-01..04.
    assert inst.n_functional_resistant_after("2024-08-31") == 63
    # boundary end of Dec-2024: still all 63 (months > 2024-12 = 2025-*).
    assert inst.n_functional_resistant_after("2024-12-31") == 63
    # boundary in Jan-2025: drop Jan (14) -> 49.
    assert inst.n_functional_resistant_after("2025-01-31") == 49
    assert inst.n_functional_resistant_after("2025-01-01") == 49
    # boundary in Feb-2025: keep Mar(27)+Apr(2) = 29 (< 30 -> sub-floor).
    assert inst.n_functional_resistant_after("2025-02-28") == 29
    # boundary in Apr-2025 (Mistral-Small-4-ish): nothing after -> 0.
    assert inst.n_functional_resistant_after("2025-04-30") == 0
    assert inst.n_functional_resistant_after("2026-03-01") == 0


def test_30_slice_requires_cutoff_at_or_before_january_2025():
    # The binding fact: >=30 functional resistant requires a KNOWN cutoff
    # month <= 2025-01 (Jan-2025 -> 49; Feb-2025 -> 29 < 30).
    inst = LATEST_RESISTANT_INSTRUMENT
    assert inst.n_functional_resistant_after("2025-01-31") >= MIN_RESISTANT_SLICE
    assert inst.n_functional_resistant_after("2025-02-28") < MIN_RESISTANT_SLICE


# ----------------------------------------------------------- per-model gate

def _cert(model_id):
    cand = next(c for c in STRONGER_MODEL_CANDIDATES if c.model_id == model_id)
    return certify_model_v1(cand)


def test_maverick_is_certifiable_but_settled_c4_blocks():
    m = _cert("meta/llama-4-maverick-17b-128e-instruct")
    assert m.c1_cutoff_known is True          # Aug-2024 KNOWN
    assert m.c2_enough_resistant is True      # 63 >= 30
    assert m.c3_reachable_stronger_comparable is True
    assert m.c4_not_already_settled is False  # W113 settled
    assert m.certifiable_for_new_pilot is False
    assert "already settled" in m.reason.lower()


def test_qwen3_coder_blocked_by_unknown_cutoff_c1():
    m = _cert("qwen/qwen3-coder-480b-a35b-instruct")
    assert m.cutoff_confidence == "UNKNOWN"
    assert m.c1_cutoff_known is False
    assert m.certifiable_for_new_pilot is False
    assert "C1" in m.reason


def test_deepseek_v4_blocked_by_c1_even_though_estimate_count_passes():
    # DeepSeek's registry ESTIMATE boundary (2025-01-01) would give 49 (C2 ok),
    # but UNKNOWN confidence blocks via C1 -> proves KNOWN-cutoff-only discipline.
    m = _cert("deepseek-ai/deepseek-v4-pro")
    assert m.c2_enough_resistant is True      # 49 under the estimate
    assert m.c1_cutoff_known is False         # but UNKNOWN
    assert m.certifiable_for_new_pilot is False


def test_mistral_small_4_blocked_by_c1_and_c2():
    m = _cert("mistralai/mistral-small-4-119b-2603")
    assert m.c1_cutoff_known is False
    assert m.c2_enough_resistant is False     # 2026-03 -> 0 resistant
    assert m.certifiable_for_new_pilot is False


# ------------------------------------------------------------ overall verdict

def test_decision_is_no_certifiable_stronger_model_on_real_data():
    d = decide_certification_v1()
    assert d.verdict == VERDICT_NONE
    assert d.target_model == ""
    assert d.maverick_certifiable_but_settled is True
    assert "instrument frontier" in d.w115_blocker.lower() \
        or "does not post-date" in d.w115_blocker.lower()
    assert "release_v7" in d.next_instrument_requirement
    assert d.cid()  # stable content id present


def test_w113_w114_confidence_consistency_guard():
    # Every W114 provenance entry's verified_confidence must equal the W113
    # registry confidence for the same model (no divergence).
    for model_id, prov in W114_CUTOFF_PROVENANCE.items():
        assert model_id in MODEL_TRAINING_CUTOFFS
        assert prov.verified_confidence == MODEL_TRAINING_CUTOFFS[
            model_id].confidence, model_id
    # and the decision reports all consistent
    for m in decide_certification_v1().per_model:
        assert m.confidence_consistent is True, m.model_id


def test_every_candidate_has_primary_source_provenance():
    for c in STRONGER_MODEL_CANDIDATES:
        assert c.model_id in W114_CUTOFF_PROVENANCE
        assert W114_CUTOFF_PROVENANCE[c.model_id].primary_source
        assert W114_CUTOFF_PROVENANCE[c.model_id].verified_on == "2026-05-29"


# ------------------------------------------------------------ falsifiability

def test_synthetic_known_cutoff_not_settled_model_DOES_certify():
    """The rule is not hard-wired to no-go: a synthetic KNOWN-cutoff, reachable,
    not-settled candidate on a synthetic instrument with >=30 post-cutoff
    functional problems certifies and becomes the target."""
    synthetic_inst = LatestResistantInstrumentV1(
        release="release_v9_synthetic",
        jsonl_sha256="0" * 64,
        n_functional=60,
        functional_date_min="2026-01-01",
        functional_date_max="2026-04-30",
        functional_month_histogram={
            "2026-01": 15, "2026-02": 15, "2026-03": 15, "2026-04": 15},
        note="synthetic post-2025 instrument for falsifiability")
    # Reuse Maverick's KNOWN cutoff (2024-08-31) but mark NOT settled so C4 passes
    cand = dataclasses.replace(
        STRONGER_MODEL_CANDIDATES[0], already_settled_on_instrument=False)
    m = certify_model_v1(cand, instrument=synthetic_inst)
    assert m.c1_cutoff_known is True
    assert m.c2_enough_resistant is True      # 60 >= 30 (all 2026 > Aug-2024)
    assert m.c4_not_already_settled is True
    assert m.certifiable_for_new_pilot is True

    d = decide_certification_v1(candidates=(cand,), instrument=synthetic_inst)
    assert d.verdict == VERDICT_CERTIFIABLE
    assert d.target_model == cand.model_id
