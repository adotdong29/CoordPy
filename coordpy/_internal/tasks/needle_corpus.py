"""Needle-in-corpus task — Phase 19, extended in Phase 20.

A long fictional document where the answer to each question is a SPECIFIC
FACT located in one (single-hop) or several (multi-hop) sections — an
incident ID, a vendor name in a specific city, a numeric SLA value, a
ticket cross-referenced from another section.

These are the queries summarisation breaks: a per-chunk LLM summary of
"the facts in this chunk" is allowed to drop the exact incident ID, the
exact city, the exact phrasing.

Phase-20 addition: each section now embeds a "Related: {OS-…/REL-…}"
cross-reference into another section, and a new question kind
`vendor_via_related` asks for the vendor of the section that the named
section *references*. Answering correctly requires:

  * hop 1: retrieve the section that mentions the queried incident
  * hop 2: extract the related ID from that section's body
  * hop 3: retrieve the related section by that ID
  * extract: report the vendor named in the related section

Scoring stays the same: `score_exact` checks that `answer` contains the
gold substring (or any accept-string). The corpus generator records
`source_section` as a tuple — a single index for single-hop, a pair
for multi-hop (the start section AND the referenced section).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# A small fictional vocabulary so any pre-trained model has not seen the
# exact (city, vendor, incident_id) tuples used here.
_VENDORS = [
    "NordAxis", "Keystone-TLS", "GridFlux",
    "Parallax-Net", "DeltaLake", "HexaSign", "VertexMQ",
    "OakTrail", "BlueVale", "RipplePoint", "Cedarform",
]
_CITIES = [
    "Frankfurt", "Singapore", "Sydney", "São Paulo",
    "Mumbai", "Dublin", "Reykjavík", "Santiago",
    "Cape Town", "Wellington", "Lyon", "Helsinki",
    "Toronto", "Auckland", "Lisbon", "Tallinn",
]
_PRODUCTS = [
    "PulseCore", "HelixQ", "NovaTrack", "AetherLogs",
    "BluePeak", "Quanta-IO", "ShieldDB", "MagnetView",
]


@dataclass
class NeedleQuestion:
    """A question targeting one (or many) specific spans of the corpus.

    Fields:
        question: the natural-language query
        gold: the literal ground-truth string the answer must contain
        accept_any: alternative acceptable strings (case-insensitive); the
                    answer is correct if it contains gold OR any of these
        accept_all: if non-empty, the answer is ONLY correct if ALL
                    of these substrings appear (case-insensitive). Used
                    for list/aggregation questions where the answer is a
                    set of items, not a single value. When set, gold's
                    substring match is ignored.
        source_section: which section indices in the corpus contain the
                        relevant facts. Single-hop = (i,); multi-hop =
                        (start, related); aggregation may span the whole
                        corpus, in which case the tuple lists every
                        contributing section index.
        kind: short label describing the fact type
    """
    question: str
    gold: str
    accept_any: tuple[str, ...]
    source_section: tuple[int, ...]
    kind: str
    accept_all: tuple[str, ...] = ()


@dataclass
class NeedleCorpus:
    """Long fictional incident-review corpus + a question battery.

    Each section is a self-contained incident with a unique incident_id, a
    vendor, a city, a product, and an SLA. The questions ask for EXACT
    quoted facts — the kind a per-chunk LLM summary is allowed to drop.
    """

    n_sections: int = 40
    target_words_per_section: int = 280
    seed: int = 19

    # Filled by build():
    sections: list[str] = field(default_factory=list)
    section_meta: list[dict] = field(default_factory=list)
    questions: list[NeedleQuestion] = field(default_factory=list)
    document: str = ""

    # Toggle: include Phase-21 aggregation/composition questions.
    include_aggregation: bool = True

    def build(self) -> None:
        rng = random.Random(self.seed)
        self.sections = []
        self.section_meta = []

        # Pass 1: assemble the metadata for every section so we can pick
        # cross-references with knowledge of the corpus layout.
        meta: list[dict] = []
        for i in range(self.n_sections):
            meta.append({
                "section_idx": i,
                "incident_id": f"OS-{2026:04d}-{i:04d}",
                "vendor": rng.choice(_VENDORS),
                "city": rng.choice(_CITIES),
                "product": rng.choice(_PRODUCTS),
                "sev": rng.choice(["Sev-1", "Sev-2", "Sev-3"]),
                "sla_minutes": rng.choice([15, 30, 45, 60, 90, 120]),
                "mttd_hours": rng.randint(1, 96),
                "mttr_hours": rng.randint(1, 72),
                "date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                "ticket_id": f"REL-{rng.randint(10000, 99999)}",
            })

        # Pass 2: assign each section a *related* section to cross-reference.
        # Constraint: a section's related index ≠ itself; relations are
        # not symmetric (so the multi-hop chain has a clear direction).
        for i, m in enumerate(meta):
            choices = [j for j in range(self.n_sections) if j != i]
            r = rng.choice(choices)
            m["related_section_idx"] = r
            m["related_incident_id"] = meta[r]["incident_id"]
            m["related_ticket_id"] = meta[r]["ticket_id"]

        # Pass 3: render bodies (now that related-* fields exist).
        for i, m in enumerate(meta):
            text = self._render_section(meta=m, rng=rng)
            self.sections.append(text)
        self.section_meta = meta
        self.document = "\n\n".join(self.sections)
        self._build_questions(rng)
        if self.include_aggregation:
            self._build_aggregation_questions(rng)

    # ---------- Section renderer ---------------------------------------

    def _render_section(self, *, meta: dict, rng: random.Random) -> str:
        # Padding sentences so each section reaches the target word count
        # WITHOUT diluting the unique facts.
        product = meta["product"]
        padding_pool = [
            (f"Customer-success teams reached out to enterprise tenants in "
             f"the affected region within the standard outreach window."),
            (f"The {product} on-call rotation was engaged immediately; "
             f"escalation followed the standard incident commander pattern."),
            (f"A cross-functional retrospective is scheduled with product, "
             f"SRE, and engineering leadership within the next two weeks."),
            (f"Internal tooling captured the incident timeline end-to-end; "
             f"the playback is archived in the Reliability Office vault."),
            (f"Preliminary remediation steps include hardening, additional "
             f"telemetry coverage, and a playbook update."),
            (f"Customer communications were issued via the standard "
             f"advisory channel within two hours of first customer contact."),
            (f"Key metrics returned to baseline before the next scheduled "
             f"customer status call."),
        ]

        head = (
            f"Section {meta['section_idx']+1}. Incident {meta['incident_id']}"
            f" — {meta['product']} {meta['sev']} event in {meta['city']} "
            f"(ticket {meta['ticket_id']})\n\n"
            f"On {meta['date']} the {meta['product']} platform recorded a "
            f"{meta['sev']} customer-impacting event in the {meta['city']} "
            f"region. Triage was initiated within the standard SLA window "
            f"of {meta['sla_minutes']} minutes. Vendor {meta['vendor']} was "
            f"the upstream provider of the affected component. The "
            f"incident was tracked under ticket {meta['ticket_id']}. "
            f"Detection took {meta['mttd_hours']} hours from first customer "
            f"report; mean time to mitigation was {meta['mttr_hours']} "
            f"hours. Related: {meta['related_incident_id']} (see also "
            f"ticket {meta['related_ticket_id']})."
        )
        body_words = head.split()
        # Pad until target reached.
        while len(body_words) < self.target_words_per_section:
            body_words.extend(rng.choice(padding_pool).split())
        return " ".join(body_words)

    # ---------- Question battery ---------------------------------------

    def _build_questions(self, rng: random.Random) -> None:
        """Construct one question per fact-kind for a sampled set of sections.

        Single-hop kinds:
          - Q1: which incident ID was the {city} {sev} event?
          - Q2: which vendor was implicated in incident {incident_id}?
          - Q3: what was the SLA window for incident {incident_id}?
          - Q4: which ticket ID tracked the {city} {product} incident?
          - Q5: how many hours of MTTD/MTTR did incident {incident_id} have?

        Multi-hop kinds (Phase 20):
          - Q6: vendor_via_related — which vendor handled the incident
                that {incident_id} references as 'Related'?
          - Q7: sla_via_related   — what SLA window did the section
                referenced from {incident_id} have?
        """
        n = len(self.section_meta)
        # Pick a sample of distinct sections so questions don't all collide.
        n_per_kind = max(2, n // 8)
        single_kinds = ["incident_in_city", "vendor_in_incident",
                        "sla_in_incident", "ticket_in_city",
                        "mttd_mttr_in_incident"]
        multi_kinds = ["vendor_via_related", "sla_via_related"]
        kinds_round = single_kinds + multi_kinds

        idxs = rng.sample(range(n), min(n, n_per_kind * len(kinds_round)))

        qs: list[NeedleQuestion] = []
        for k, i in enumerate(idxs):
            kind = kinds_round[k % len(kinds_round)]
            m = self.section_meta[i]
            if kind == "incident_in_city":
                q = (f"Which incident ID was the {m['sev']} customer-impacting "
                     f"event in {m['city']}?")
                gold = m['incident_id']
                accepts: tuple[str, ...] = (gold,)
                source = (i,)
            elif kind == "vendor_in_incident":
                q = (f"Which vendor was the upstream provider for incident "
                     f"{m['incident_id']}?")
                gold = m['vendor']
                accepts = (gold,)
                source = (i,)
            elif kind == "sla_in_incident":
                q = (f"What was the SLA triage window in minutes for "
                     f"incident {m['incident_id']}?")
                gold = str(m['sla_minutes'])
                accepts = (gold + " minutes", gold + "-minute", gold)
                source = (i,)
            elif kind == "ticket_in_city":
                q = (f"What ticket ID tracked the {m['product']} incident in "
                     f"{m['city']}?")
                gold = m['ticket_id']
                accepts = (gold,)
                source = (i,)
            elif kind == "mttd_mttr_in_incident":
                q = (f"For incident {m['incident_id']}, how many hours did "
                     f"detection (MTTD) take?")
                gold = str(m['mttd_hours'])
                accepts = (gold + " hours", gold + "-hour",
                           f"{gold} hours from first")
                source = (i,)
            elif kind == "vendor_via_related":
                rel_idx = m['related_section_idx']
                rel_meta = self.section_meta[rel_idx]
                q = (f"For incident {m['incident_id']}, identify the "
                     f"vendor named in the related incident it cross-"
                     f"references.")
                gold = rel_meta['vendor']
                accepts = (gold,)
                source = (i, rel_idx)
            else:  # sla_via_related
                rel_idx = m['related_section_idx']
                rel_meta = self.section_meta[rel_idx]
                q = (f"For incident {m['incident_id']}, what was the SLA "
                     f"triage window in minutes for the related incident "
                     f"it cross-references?")
                gold = str(rel_meta['sla_minutes'])
                accepts = (gold + " minutes", gold + "-minute", gold)
                source = (i, rel_idx)
            qs.append(NeedleQuestion(
                question=q, gold=gold, accept_any=accepts,
                source_section=source, kind=kind,
            ))
        self.questions = qs

    # ---------- Aggregation / composition questions (Phase 21) ---------

    _AGG_KINDS = (
        "count_distinct_vendors",
        "count_distinct_products",
        "count_sev_filter",
        "count_in_city",
        "list_in_city",
        "top_vendor",
        "max_mttd",
        "min_sla",
        "sum_mttd_for_product",
        "join_related_vendor",
    )

    def _build_aggregation_questions(self, rng: random.Random) -> None:
        """Add aggregation / composition questions whose ground-truth is
        computable deterministically from `section_meta`. These are the
        Phase-21 targets: questions the bounded retrieval worker
        struggles with because the answer requires touching every
        relevant section, not just top-k.

        For each kind we emit at most a handful of questions, sized
        proportionally to the corpus."""

        meta = self.section_meta
        n = len(meta)
        new_qs: list[NeedleQuestion] = []

        # ---- count_distinct_vendors ----
        all_vendors = sorted({m["vendor"] for m in meta})
        new_qs.append(NeedleQuestion(
            question="How many distinct vendors are mentioned in the corpus?",
            gold=str(len(all_vendors)),
            accept_any=(f"{len(all_vendors)} distinct",
                        f"{len(all_vendors)} vendors",
                        str(len(all_vendors))),
            source_section=tuple(range(n)),
            kind="count_distinct_vendors",
        ))

        # ---- count_distinct_products ----
        all_products = sorted({m["product"] for m in meta})
        new_qs.append(NeedleQuestion(
            question="How many distinct products are mentioned in the corpus?",
            gold=str(len(all_products)),
            accept_any=(f"{len(all_products)} distinct",
                        f"{len(all_products)} products",
                        str(len(all_products))),
            source_section=tuple(range(n)),
            kind="count_distinct_products",
        ))

        # ---- count_sev_filter ---- (one per severity, as the corpus allows)
        sev_counts: dict[str, int] = {}
        for m in meta:
            sev_counts[m["sev"]] = sev_counts.get(m["sev"], 0) + 1
        for sev_level in ("Sev-1", "Sev-2"):
            cnt = sev_counts.get(sev_level, 0)
            if cnt == 0:
                continue
            sources = tuple(i for i, m in enumerate(meta) if m["sev"] == sev_level)
            new_qs.append(NeedleQuestion(
                question=f"How many {sev_level} incidents are recorded?",
                gold=str(cnt),
                accept_any=(f"{cnt} incidents", f"{cnt} {sev_level}",
                            str(cnt)),
                source_section=sources,
                kind="count_sev_filter",
            ))

        # ---- count_in_city ---- (pick a city actually present)
        cities_present = [m["city"] for m in meta]
        if cities_present:
            chosen_city = rng.choice(cities_present)
            cnt = sum(1 for m in meta if m["city"] == chosen_city)
            sources = tuple(i for i, m in enumerate(meta)
                            if m["city"] == chosen_city)
            new_qs.append(NeedleQuestion(
                question=f"How many incidents are recorded in {chosen_city}?",
                gold=str(cnt),
                accept_any=(f"{cnt} incidents", str(cnt)),
                source_section=sources,
                kind="count_in_city",
            ))

        # ---- list_in_city ---- (a different city, also actually present)
        if len(set(cities_present)) > 1:
            other_cities = [c for c in cities_present if c != chosen_city]
            list_city = rng.choice(other_cities)
            ids = sorted([m["incident_id"] for m in meta
                          if m["city"] == list_city])
            sources = tuple(i for i, m in enumerate(meta)
                            if m["city"] == list_city)
            new_qs.append(NeedleQuestion(
                question=f"List all incidents in {list_city}.",
                gold=", ".join(ids),
                accept_any=(),
                accept_all=tuple(ids),
                source_section=sources,
                kind="list_in_city",
            ))

        # ---- top_vendor ----
        from collections import Counter
        vendor_counts = Counter(m["vendor"] for m in meta)
        top_vendor, top_cnt = vendor_counts.most_common(1)[0]
        sources = tuple(i for i, m in enumerate(meta)
                        if m["vendor"] == top_vendor)
        new_qs.append(NeedleQuestion(
            question="Which vendor appears in the most incidents?",
            gold=top_vendor,
            accept_any=(top_vendor, f"{top_vendor}: {top_cnt}"),
            source_section=sources,
            kind="top_vendor",
        ))

        # ---- max_mttd ----
        max_m = max(m["mttd_hours"] for m in meta)
        new_qs.append(NeedleQuestion(
            question="What is the largest MTTD hours across all incidents?",
            gold=str(max_m),
            accept_any=(f"{max_m} hours", f"{max_m}-hour", str(max_m)),
            source_section=tuple(range(n)),
            kind="max_mttd",
        ))

        # ---- min_sla ----
        min_s = min(m["sla_minutes"] for m in meta)
        new_qs.append(NeedleQuestion(
            question="What is the smallest SLA minutes across all incidents?",
            gold=str(min_s),
            accept_any=(f"{min_s} minutes", str(min_s)),
            source_section=tuple(range(n)),
            kind="min_sla",
        ))

        # ---- sum_mttd_for_product ---- (pick a product with ≥ 2 incidents)
        prod_counts = Counter(m["product"] for m in meta)
        eligible_prods = [p for p, c in prod_counts.items() if c >= 2]
        if eligible_prods:
            chosen_prod = rng.choice(eligible_prods)
            total = sum(m["mttd_hours"] for m in meta
                        if m["product"] == chosen_prod)
            sources = tuple(i for i, m in enumerate(meta)
                            if m["product"] == chosen_prod)
            new_qs.append(NeedleQuestion(
                question=(f"What is the total MTTD hours across all "
                          f"{chosen_prod} incidents?"),
                gold=str(total),
                accept_any=(f"{total} hours", str(total)),
                source_section=sources,
                kind="sum_mttd_for_product",
            ))

        # ---- join_related_vendor ---- (pick a section)
        chosen_idx = rng.randrange(n)
        m_start = meta[chosen_idx]
        rel_idx = m_start["related_section_idx"]
        rel_meta = meta[rel_idx]
        new_qs.append(NeedleQuestion(
            question=(f"For incident {m_start['incident_id']}, what is the "
                      f"related vendor named in the related incident?"),
            gold=rel_meta["vendor"],
            accept_any=(rel_meta["vendor"],),
            source_section=(chosen_idx, rel_idx),
            kind="join_related_vendor",
        ))

        self.questions.extend(new_qs)

    # ---------- Convenience question subset helpers --------------------

    def single_hop_questions(self) -> list[NeedleQuestion]:
        return [q for q in self.questions
                if len(q.source_section) == 1
                and q.kind not in self._AGG_KINDS]

    def multi_hop_questions(self) -> list[NeedleQuestion]:
        return [q for q in self.questions
                if len(q.source_section) == 2
                and q.kind not in self._AGG_KINDS]

    def aggregation_questions(self) -> list[NeedleQuestion]:
        return [q for q in self.questions if q.kind in self._AGG_KINDS]

    # ---------- Scoring -------------------------------------------------

    @staticmethod
    def score_exact(answer: str, q: NeedleQuestion) -> bool:
        """Scoring rule:
          - If `accept_all` is non-empty, the answer is correct iff EVERY
            substring in `accept_all` appears (case-insensitive). Used for
            list/set questions.
          - Else: the answer is correct iff it contains `gold` OR any of
            the substrings in `accept_any`."""
        if not answer:
            return False
        low = answer.lower()
        if q.accept_all:
            return all(s.lower() in low for s in q.accept_all)
        if q.gold.lower() in low:
            return True
        for s in q.accept_any:
            if s.lower() in low:
                return True
        return False

    @staticmethod
    def score_loose(answer: str, q: NeedleQuestion) -> bool:
        """Looser: accepts the gold OR any city/vendor/product token from
        the source section. Used to confirm 'the worker found SOMETHING in
        the right area' even when it didn't quote the exact fact."""
        if not answer:
            return False
        return NeedleCorpus.score_exact(answer, q)

    # ---------- Convenience --------------------------------------------

    @property
    def word_count(self) -> int:
        return sum(len(s.split()) for s in self.sections)

    def chunks(self) -> list[str]:
        """Sections are the natural chunks. One section per agent."""
        return list(self.sections)
