# Contamination-control framing — W117 (upstream-DERIVED instrument CONSTRUCTION attack + deeper primary-source cutoff attack + construction/admission pipeline)

> Companion to `CONTAMINATION_CONTROL_FRAMING_W113_V1.md` / `…_W114_V1.md` /
> `…_W115_V1.md` / `…_W116_V1.md`. W117 tests **construction supply**, NOT the
> contamination confound itself; the confound status is UNCHANGED (STRENGTHENED-
> not-proven, per W113). This doc records what W117's construction-side attack does
> and does not license about contamination.

## What W117 did

W117 attacked the *construction* half of the W114/W115/W116 supply blocker at its
source:

* **Instrument/construction side (Lane α)** — instead of only checking for a *packaged*
  `release_v7` (W116's four surfaces), W117 attacked the upstream CONSTRUCTION
  provenance at EIGHT surfaces (HF commit/revision log / refs / discussions, GitHub
  commits / tags / repo pipeline structure, README provenance, runner loader). Result:
  the authoritative provenance IS the packaged HF release; LCB publishes no collection
  pipeline or forward problem-id manifest; no post-v6 LCB-published artifact exists;
  the only post-v6 path (raw-contest hand-assembly) is CONSTRUCTION-INADMISSIBLE
  (refused by the pre-committed B1 + B2 criteria).
* **Model side (Lane β)** — re-checked official cutoff disclosures DEEPER from PRIMARY
  sources. Result: the DeepSeek V4 official model-card PDF, re-checked at primary,
  still discloses NO cutoff (the only figure is a non-primary aggregator that is itself
  C2-exposed); Maverick "August 2024" re-confirmed verbatim; no reachable
  stronger-than-Maverick model has a primary-KNOWN cutoff ≤ 2025-01; nothing
  newly-disclosed since W116.

Verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID `258b6ed7…`,
byte-identical to W114/W115/W116); $0 NIM.

## What this licenses about contamination — NOTHING NEW

W117 is a **construction-supply** result, not a contamination result. It does not run
any model and does not touch the confound:

* The contamination confound remains **STRENGTHENED-not-proven** (W109 double
  dissociation by vintage; W113 within-model exposed→resistant flip). W117 adds no
  empirical contamination evidence either way.
* The two retirements (W89 base HumanEval +5.56 pp; W105 HumanEval+ +7.00 pp) remain
  **contamination-EXPOSED-HumanEval-family at 70B** — the registered ceiling (W114).
  W117 neither strengthens nor weakens them.
* Resistant same-budget code superiority remains **0 clean across both 70B and Maverick
  scales** (W108 / W110 / W113). W117 ran no pilot, so this is unchanged.

## What W117 sharpens (supply, not confound)

* The supply blocker is now characterised as a **CONSTRUCTION-provenance blocker**, not
  merely a packaging blocker: a post-v6 instrument cannot be *constructed* from
  authoritative provenance because LCB's authoritative provenance IS the packaged
  release (no published collection pipeline / manifest), and the only post-v6 path
  (raw-contest hand-assembly) is refused by B1 (not LCB-published) ∧ B2 (operator
  curation). Selling a hand-assembled slice as a LiveCodeBench-grade resistant
  instrument would be exactly the vibes-based cherry-picking the discipline forbids.
* The model-disclosure blocker is sharpened: DeepSeek V4's *primary PDF* (not just an
  aggregator) re-checked and still no cutoff; the "Apr 2026" figure is non-primary AND
  C2-exposed — now matching Mistral Small 4's pattern.

## Do NOT claim (unchanged from W113–W116 + W117 additions)

* That contamination is PROVEN to drive the W89/W105 retirements (it is
  strengthened-not-proven).
* That resistant same-budget code superiority exists at any scale (0 clean).
* That a post-v6 LiveCodeBench instrument can be CONSTRUCTED today (it cannot, from
  authoritative provenance — only hand-curated, which is refused).
* That a raw-contest hand-assembly is a LiveCodeBench-grade resistant instrument (it
  fails A1 ∧ B1 ∧ B2).
* That any reachable stronger-than-Maverick model has a usable primary-KNOWN cutoff (it
  does not; nothing newly-disclosed since W116).

## The honest bounded claim (W117 truth floor)

Same-budget multi-agent code superiority is confirmed at 70B on the
contamination-EXPOSED HumanEval family (W89 + W105) and is contamination-confounded;
the resistant-code question is **UPSTREAM-CONSTRUCTION-PROVENANCE + CUTOFF-DISCLOSURE-
BLOCKED**, not closed. W117 proves the construction blocker is real and current at
eight provenance surfaces and on the model side, and ships the construction/admission
pipeline that turns the W118 trigger into a push-button operation.
