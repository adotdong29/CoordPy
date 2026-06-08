"""W140 unit tests — family tutor compiler + deterministic no-leakage gate (Lane α).

$0 (no NIM).  Asserts: every shared-family tutor compiles + passes the leak gate; the leak gate is
FALSIFIABLE (planted leaks bite each decisive assertion); skeletons are completable to a correct
program (correct-fill passes the hidden secret cases) AND the holes are load-bearing (trivially
stubbed fails public); the negative control is not genuine; tutor compilation is deterministic.
"""
import dataclasses

import pytest

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, FUNC_FACTORIES
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy import family_tutor_compiler_v1 as T

# the two W140 shared families (from the reused W139 per-tier calibration), with anchor knobs
SHARED = [
    ("count_pairs_sum_le_t", CX_FACTORIES["count_pairs_sum_le_t"], 50000, 140_100_010),
    ("subarrays_sum_and_range", FUNC_FACTORIES["subarrays_sum_and_range"], 1500, 140_100_011),
]
TC_KINDS = [T.TC1_CARD, T.TC2_REWRITE, T.TC3_COMPRESSED, T.TC5_STAGED]


def _mint(fac, knob, seed):
    tmpl = fac(knob)
    prob = mint_problem_v1(tmpl.minted, global_seed=seed, timeout_s=8.0)
    return tmpl, prob


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_all_tc_kinds_compile_and_pass_leak_gate(fam, fac, knob, seed):
    tmpl, prob = _mint(fac, knob, seed)
    assert prob.gates.admitted, f"{fam} must be HC2-admitted"
    for kind in TC_KINDS:
        tut = T.COMPILERS_BY_KIND[kind](tmpl)
        rep = T.tutor_leak_gate_v1(tut, tmpl, prob, timeout_s=8.0)
        assert not rep.leaked, f"{fam}/{kind} must NOT leak: {rep.to_dict()}"
        assert rep.no_discriminator_leak and rep.holes_are_substantive
        assert rep.public_only_literals and rep.is_procedure_not_answer
        assert rep.template_invariant and rep.one_liner_family_ok


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_skeleton_completable_to_correct_program(fam, fac, knob, seed):
    """The correct-fill of the holed scaffold PASSES the hidden secret cases (the scaffold teaches a
    real technique; the holes are the whole gap between scaffold and solution)."""
    tmpl, prob = _mint(fac, knob, seed)
    out = T.skeleton_is_completable_v1(tmpl, prob, timeout_s=8.0)
    assert out["completable"] is True, f"{fam}: correct-fill must pass hidden cases: {out}"
    assert out["n_holes"] >= 2


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_planted_reference_paste_bites(fam, fac, knob, seed):
    """A 'skeleton' that is the reference source (no holes) MUST be flagged leaked."""
    tmpl, prob = _mint(fac, knob, seed)
    leaky = dataclasses.replace(T.compile_witness_rewrite_tutor_v1(tmpl),
                                skeleton=tmpl.minted.ref_source)
    rep = T.tutor_leak_gate_v1(leaky, tmpl, prob, timeout_s=8.0)
    assert rep.leaked
    assert (not rep.no_discriminator_leak) or (not rep.holes_are_substantive)


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_planted_discriminator_in_card_bites(fam, fac, knob, seed):
    """A card that states the discriminating expression verbatim MUST be flagged leaked."""
    tmpl, prob = _mint(fac, knob, seed)
    spec = T.TECHNIQUE_LIBRARY[tmpl.minted.algo_sig]
    disc = max(spec.correct_fill.values(), key=len)  # the longest (most discriminating) expr
    leaky = dataclasses.replace(T.compile_family_card_v1(tmpl),
                                key_move=f"the exact rule is: {disc}")
    rep = T.tutor_leak_gate_v1(leaky, tmpl, prob, timeout_s=8.0)
    assert rep.leaked and not rep.no_discriminator_leak
    assert disc in rep.leaked_discriminators


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_holes_are_load_bearing(fam, fac, knob, seed):
    """The TC2 skeleton with holes trivially stubbed FAILS the public samples (the holes carry the
    answer-logic)."""
    tmpl, prob = _mint(fac, knob, seed)
    rep = T.tutor_leak_gate_v1(T.compile_witness_rewrite_tutor_v1(tmpl), tmpl, prob, timeout_s=8.0)
    assert rep.holes_are_substantive


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_negative_control_not_genuine(fam, fac, knob, seed):
    tmpl, prob = _mint(fac, knob, seed)
    neg = T.make_negative_control_tutor_v1(tmpl)
    assert T.tutor_is_genuinely_new_v1(neg)["genuinely_new"] is False
    # a content-free instruction is not a leak (it carries no answer) but it is not genuine teaching
    assert T.tutor_leak_gate_v1(neg, tmpl, prob).leaked is False


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_real_tutors_are_genuine(fam, fac, knob, seed):
    tmpl, _ = _mint(fac, knob, seed)
    for kind in (T.TC1_CARD, T.TC2_REWRITE, T.TC3_COMPRESSED):
        assert T.tutor_is_genuinely_new_v1(T.COMPILERS_BY_KIND[kind](tmpl))["genuinely_new"] is True


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_compilation_is_deterministic(fam, fac, knob, seed):
    """Same family -> byte-identical tutor (CID stable) across recompiles (the tutor is a family-level
    asset, not a per-instance object)."""
    tmpl, _ = _mint(fac, knob, seed)
    tmpl2 = fac(knob)
    for kind in TC_KINDS:
        a = T.COMPILERS_BY_KIND[kind](tmpl).cid()
        b = T.COMPILERS_BY_KIND[kind](tmpl2).cid()
        assert a == b, f"{fam}/{kind} tutor CID must be deterministic"


@pytest.mark.parametrize("fam,fac,knob,seed", SHARED)
def test_tc3_is_smallest(fam, fac, knob, seed):
    """The compressed tutor is strictly smaller than the full card (the minimality knob is real)."""
    tmpl, _ = _mint(fac, knob, seed)
    t1 = T.compile_family_card_v1(tmpl).token_count()
    t3 = T.compile_compressed_tutor_v1(tmpl).token_count()
    t2 = T.compile_witness_rewrite_tutor_v1(tmpl).token_count()
    assert t3 < t1 < t2, f"{fam}: expected TC3 < TC1 < TC2, got {t3}/{t1}/{t2}"
    assert t3 <= T.TC3_TOKEN_BUDGET


def test_one_liner_family_guard_disables_cards_for_trivial_ref():
    """The one-liner guard fires when the reference is trivial (naming == coding)."""
    tmpl, prob = _mint(CX_FACTORIES["count_pairs_sum_le_t"], 50000, 140_100_012)
    # forge a trivial 1-statement reference -> the guard must trip (card not permitted)
    trivial_minted = dataclasses.replace(tmpl.minted, ref_source="print(0)\n")
    trivial_tmpl = dataclasses.replace(tmpl, minted=trivial_minted)
    rep = T.tutor_leak_gate_v1(T.compile_family_card_v1(tmpl), trivial_tmpl, prob)
    assert rep.one_liner_family_ok is False  # trivial ref -> guard trips -> leaked
    assert rep.leaked
