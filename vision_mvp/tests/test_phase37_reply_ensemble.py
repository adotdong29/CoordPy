"""Phase 37 Part B — reply-axis ensemble unit tests."""

from __future__ import annotations

import pytest

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import (
    DeterministicMockReplier, LLMReplyConfig, LLMThreadReplier,
)
from vision_mvp.core.reply_ensemble import (
    EnsembleReplier, MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK,
    MODE_VERIFIED, causality_extractor_from_ensemble,
    verifier_accept_ir_only_on_payload_marker,
    verifier_from_oracle, verifier_from_payload_classifier,
)


class _FakeScenario:
    scenario_id = "fake"


def _replier(mapping: dict[tuple[str, str], str],
              malformed_prob: float = 0.0,
              ) -> LLMThreadReplier:
    return LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies=mapping, malformed_prob=malformed_prob),
        config=LLMReplyConfig(witness_token_cap=12),
    )


def test_dual_agree_when_both_agree():
    # Both repliers say IR — ensemble emits IR.
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    s = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    ens = EnsembleReplier(mode=MODE_DUAL_AGREE, primary=p,
                            secondary=s)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is True
    assert rk == REPLY_INDEPENDENT_ROOT
    assert ens.stats.n_agree == 1
    assert ens.stats.n_disagree == 0


def test_dual_agree_refuses_on_disagreement():
    # Primary says IR, secondary says DOWNSTREAM → ensemble
    # refuses with UNCERTAIN, well_formed=False.
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    s = _replier({("network", "TLS_EXPIRED"):
                     REPLY_DOWNSTREAM_SYMPTOM})
    ens = EnsembleReplier(mode=MODE_DUAL_AGREE, primary=p,
                            secondary=s)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is False
    assert rk == REPLY_UNCERTAIN
    assert ens.stats.n_agree == 0
    assert ens.stats.n_disagree == 1


def test_primary_fallback_uses_primary_when_well_formed():
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    s = _replier({("network", "TLS_EXPIRED"):
                     REPLY_DOWNSTREAM_SYMPTOM})
    ens = EnsembleReplier(mode=MODE_PRIMARY_FALLBACK,
                            primary=p, secondary=s)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is True
    assert rk == REPLY_INDEPENDENT_ROOT
    assert ens.stats.n_fallback_used == 0


def test_primary_fallback_kicks_in_on_malformed_primary():
    # Primary always malformed; fallback deterministic.
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT},
                  malformed_prob=1.0)
    s = _replier({("network", "TLS_EXPIRED"):
                     REPLY_DOWNSTREAM_SYMPTOM})
    ens = EnsembleReplier(mode=MODE_PRIMARY_FALLBACK,
                            primary=p, secondary=s)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is True
    assert rk == REPLY_DOWNSTREAM_SYMPTOM
    assert ens.stats.n_fallback_used == 1


def test_verified_accepts_when_verifier_true():
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})

    def _ver(scenario, role, kind, payload, replied_kind):
        return replied_kind == REPLY_INDEPENDENT_ROOT

    ens = EnsembleReplier(mode=MODE_VERIFIED, primary=p,
                            verifier=_ver)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is True
    assert rk == REPLY_INDEPENDENT_ROOT
    assert ens.stats.n_verified == 1


def test_verified_rejects_when_verifier_false():
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})

    def _ver(scenario, role, kind, payload, replied_kind):
        return False

    ens = EnsembleReplier(mode=MODE_VERIFIED, primary=p,
                            verifier=_ver)
    rk, wit, wf = ens(_FakeScenario(), "network", "TLS_EXPIRED",
                       "tls service=api reason=expired")
    assert wf is False
    assert rk == REPLY_UNCERTAIN
    assert ens.stats.n_rejected == 1


def test_verifier_from_oracle_accepts_correct():
    def _oracle(scenario, role, kind, payload):
        if role == "network" and kind == "TLS_EXPIRED":
            return "INDEPENDENT_ROOT"
        return "UNCERTAIN"

    v = verifier_from_oracle(_oracle)
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "tls service=api reason=expired",
             REPLY_INDEPENDENT_ROOT) is True
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "tls service=api reason=expired",
             REPLY_DOWNSTREAM_SYMPTOM) is False


def test_verifier_accept_ir_on_marker():
    v = verifier_accept_ir_only_on_payload_marker({
        "TLS_EXPIRED": ("reason=expired",),
    })
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "tls reason=expired", REPLY_INDEPENDENT_ROOT) is True
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "tls no-marker-here",
             REPLY_INDEPENDENT_ROOT) is False
    # DOWNSTREAM always accepted.
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "no marker",
             REPLY_DOWNSTREAM_SYMPTOM) is True


def test_verifier_from_payload_classifier():
    def _cls(role, kind, payload):
        if kind == "TLS_EXPIRED":
            return REPLY_INDEPENDENT_ROOT
        return REPLY_UNCERTAIN

    v = verifier_from_payload_classifier(_cls)
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "any", REPLY_INDEPENDENT_ROOT) is True
    assert v(_FakeScenario(), "network", "TLS_EXPIRED",
             "any", REPLY_DOWNSTREAM_SYMPTOM) is False


def test_causality_extractor_from_ensemble_shape():
    p = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    s = _replier({("network", "TLS_EXPIRED"):
                     REPLY_INDEPENDENT_ROOT})
    ens = EnsembleReplier(mode=MODE_DUAL_AGREE, primary=p,
                            secondary=s)
    ext = causality_extractor_from_ensemble(ens)
    out = ext(_FakeScenario(), "network", "TLS_EXPIRED",
              "tls service=api reason=expired")
    assert out == "INDEPENDENT_ROOT"


def test_ensemble_rejects_unknown_mode():
    p = _replier({})
    with pytest.raises(ValueError):
        EnsembleReplier(mode="bogus", primary=p)


def test_dual_agree_requires_secondary():
    p = _replier({})
    with pytest.raises(ValueError):
        EnsembleReplier(mode=MODE_DUAL_AGREE, primary=p,
                          secondary=None)


def test_verified_requires_verifier():
    p = _replier({})
    with pytest.raises(ValueError):
        EnsembleReplier(mode=MODE_VERIFIED, primary=p,
                          verifier=None)
