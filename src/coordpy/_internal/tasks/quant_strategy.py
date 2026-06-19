"""Quant strategy task — multi-role distributed work.

A synthetic quant workflow where different agents play DIFFERENT roles:

  - Researchers read short research notes → extract alpha hypotheses
  - Market analysts read individual asset time series → extract trends
  - Strategy builders combine hypotheses + trends → propose trades
  - Risk evaluator / PM synthesizes a final portfolio

The "ground truth" is that the data is generated with known signals:
  - Half the assets have momentum (return sign persists)
  - Half have mean-reversion (return sign flips)
  - A few have event-driven signals tied to a research-note flag

A correct portfolio goes LONG momentum + event-positive assets and
SHORT mean-reverting + event-negative assets.

Scoring:
  - Hit rate: fraction of assets where team's direction matches the
    optimal direction.
  - Sharpe proxy: directional return over sqrt(N assets).
  - Random baseline: 50% hit rate.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Literal


Direction = Literal["long", "short", "flat"]


@dataclass
class Asset:
    ticker: str
    regime: str            # "momentum" / "mean_reversion" / "event_pos" / "event_neg"
    past_returns: list[float]   # last 10 days, in percent
    true_next_return: float     # next-day return (what team is trying to predict sign of)

    def optimal_direction(self) -> Direction:
        return "long" if self.true_next_return > 0 else "short"


@dataclass
class ResearchNote:
    note_id: str
    tickers_mentioned: list[str]
    content: str
    signal: str            # "informative" or "noise"


@dataclass
class QuantTask:
    n_assets: int = 20
    n_research_notes: int = 30
    seed: int = 42
    assets: list[Asset] = field(default_factory=list)
    research_notes: list[ResearchNote] = field(default_factory=list)

    def generate(self) -> None:
        rng = random.Random(self.seed)
        regimes = ["momentum", "mean_reversion", "event_pos", "event_neg"]

        # Generate assets
        for i in range(self.n_assets):
            ticker = f"SYN{i:02d}"
            regime = regimes[i % 4]
            # Generate past 10 returns
            returns = []
            drift = rng.uniform(-0.003, 0.003)
            for _ in range(10):
                r = drift + rng.gauss(0, 0.015)
                returns.append(round(r * 100, 2))   # in percent

            # Next return based on regime
            last = returns[-1] / 100.0
            if regime == "momentum":
                next_r = last + rng.gauss(0, 0.005)   # trend continues
            elif regime == "mean_reversion":
                next_r = -last + rng.gauss(0, 0.005)  # reverses
            elif regime == "event_pos":
                next_r = 0.02 + rng.gauss(0, 0.005)    # positive shock
            else:  # event_neg
                next_r = -0.02 + rng.gauss(0, 0.005)   # negative shock

            self.assets.append(Asset(
                ticker=ticker, regime=regime, past_returns=returns,
                true_next_return=round(next_r * 100, 2),
            ))

        # Generate research notes
        note_templates_informative = [
            "Our factor analysis shows persistent momentum signals in {t}. Recent returns pattern is consistent with continuation.",
            "Institutional positioning data suggests {t} has short-interest crowding — classic mean-reversion setup.",
            "Unconfirmed chatter suggests a positive product announcement for {t} within the week.",
            "We flag {t} for negative surprise risk — recent management commentary has been bearish.",
            "Volume-weighted price analysis indicates {t} is in a trend continuation regime.",
            "Cross-sectional mean-reversion screen ranks {t} as a high-conviction candidate.",
            "Our event-driven desk flags {t} as a probable positive catalyst this cycle.",
            "Relative-strength ranking places {t} in the bottom quintile — likely reversal candidate.",
        ]
        note_templates_noise = [
            "{t} trading flat, no clear signal from our screens.",
            "{t} showed average volume today. Nothing to flag.",
            "No conviction either direction on {t}. Recommend pass.",
            "Mixed signals on {t} — our models disagree.",
            "Retail sentiment on {t} is neutral.",
            "Technical setup on {t} is ambiguous.",
        ]

        for j in range(self.n_research_notes):
            if rng.random() < 0.6:
                # informative note about a specific asset, matching its regime
                asset = rng.choice(self.assets)
                if asset.regime == "momentum":
                    template = note_templates_informative[0]
                    if rng.random() < 0.1:
                        template = note_templates_informative[4]
                elif asset.regime == "mean_reversion":
                    template = note_templates_informative[1]
                    if rng.random() < 0.3:
                        template = note_templates_informative[5]
                    if rng.random() < 0.2:
                        template = note_templates_informative[7]
                elif asset.regime == "event_pos":
                    template = note_templates_informative[2]
                    if rng.random() < 0.3:
                        template = note_templates_informative[6]
                else:   # event_neg
                    template = note_templates_informative[3]
                note = ResearchNote(
                    note_id=f"RN-{j:03d}",
                    tickers_mentioned=[asset.ticker],
                    content=template.format(t=asset.ticker),
                    signal="informative",
                )
            else:
                # noise note
                asset = rng.choice(self.assets)
                template = rng.choice(note_templates_noise)
                note = ResearchNote(
                    note_id=f"RN-{j:03d}",
                    tickers_mentioned=[asset.ticker],
                    content=template.format(t=asset.ticker),
                    signal="noise",
                )
            self.research_notes.append(note)

    # ---- Scoring ----

    def score_portfolio(self, directions: dict[str, Direction]) -> dict:
        """`directions` maps ticker → 'long' / 'short' / 'flat'.

        Returns:
          hit_rate: fraction of assets where direction matches optimal sign
          gross_return: sum of (+r if long else -r if short else 0)
          sharpe_proxy: gross_return / sqrt(n_assets)
          n_correct, n_wrong, n_flat
        """
        n = len(self.assets)
        correct = wrong = flat = 0
        gross = 0.0
        for a in self.assets:
            d = directions.get(a.ticker, "flat")
            optimal = a.optimal_direction()
            r = a.true_next_return
            if d == "flat":
                flat += 1
                continue
            if d == optimal:
                correct += 1
            else:
                wrong += 1
            if d == "long":
                gross += r
            elif d == "short":
                gross -= r
        non_flat = correct + wrong
        hit_rate = (correct / non_flat) if non_flat > 0 else 0.0
        sharpe = gross / math.sqrt(max(n, 1))
        return {
            "hit_rate": round(hit_rate, 3),
            "n_correct": correct,
            "n_wrong": wrong,
            "n_flat": flat,
            "gross_return_pct": round(gross, 3),
            "sharpe_proxy": round(sharpe, 3),
            "n_assets": n,
        }

    def random_baseline_score(self, seed: int = 0, n_samples: int = 200) -> dict:
        """Average over many random seeds for a stable baseline."""
        rng = random.Random(seed)
        hit_rates = []
        gross_returns = []
        for _ in range(n_samples):
            directions = {a.ticker: rng.choice(["long", "short"]) for a in self.assets}
            s = self.score_portfolio(directions)
            hit_rates.append(s["hit_rate"])
            gross_returns.append(s["gross_return_pct"])
        n = len(hit_rates)
        mean_hit = sum(hit_rates) / n
        mean_gross = sum(gross_returns) / n
        return {
            "hit_rate": round(mean_hit, 3),
            "gross_return_pct": round(mean_gross, 3),
            "sharpe_proxy": round(mean_gross / (max(len(self.assets), 1) ** 0.5), 3),
            "n_assets": len(self.assets),
            "n_samples": n_samples,
        }

    def optimal_score(self) -> dict:
        directions = {a.ticker: a.optimal_direction() for a in self.assets}
        return self.score_portfolio(directions)

    # ---- Text renderings for agents ----

    def asset_text(self, asset: Asset) -> str:
        rets = ", ".join(f"{r:+.2f}%" for r in asset.past_returns)
        return (f"Asset {asset.ticker} — last 10 daily returns: {rets}. "
                "Provide directional view (long / short) for next trading day.")

    def note_text(self, note: ResearchNote) -> str:
        return f"[{note.note_id}] {note.content}"
