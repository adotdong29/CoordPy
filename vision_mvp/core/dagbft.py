"""DAG-BFT — a single-process Bullshark-style simulator.

Spiegelman, Giridharan, Sonnino, Kokoris-Kogias (2022). DAG-BFT consensus
separates *data dissemination* (a causal DAG of blocks) from *ordering*
(leader election over the DAG). Byzantine fault tolerance at N = 3f+1 with
throughput that scales near-linearly with network bandwidth.

Simplified model we implement:
  - `N` validators, each round every honest validator *proposes* one block.
  - A block references ≥ 2f+1 blocks from the prior round (its "certificate").
  - Every 2 rounds, a deterministic leader is picked; that leader's round-r
    block is the *anchor*. The committed ordering at round r is the causal
    closure of the anchor, minus anything committed in prior anchors.

No cryptography or networking — just a single-process DAG of blocks and a
deterministic commit rule. Enough to benchmark throughput and DAG depth
growth against classical consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DAGBlock:
    round: int
    author: int
    payload: object = None
    parents: tuple = ()    # (round, author) tuples
    block_id: int = 0

    def key(self) -> tuple:
        return (self.round, self.author)


@dataclass
class BullsharkSimulator:
    """Stripped-down Bullshark.

    Validators: 0..N-1 (all honest in this simulator). Faulty validators
    can be simulated by having them skip rounds.
    """

    n_validators: int
    f: int                       # byzantine tolerance: n ≥ 3f + 1
    _blocks: dict[tuple, DAGBlock] = field(default_factory=dict)
    _round: int = 0
    _committed: list[DAGBlock] = field(default_factory=list)

    def __post_init__(self):
        if self.n_validators < 3 * self.f + 1:
            raise ValueError(f"need n ≥ 3f+1; got n={self.n_validators}, f={self.f}")

    # --- block construction ---

    def propose_round(self, payloads: list[object] | None = None) -> None:
        """Every validator emits one block referencing ≥ 2f+1 parents."""
        r = self._round
        prev_keys = [k for k in self._blocks.keys() if k[0] == r - 1]
        if r == 0 or not prev_keys:
            certificate = ()
        else:
            # Take any 2f+1 parents (deterministic: lowest-author ids)
            needed = min(2 * self.f + 1, len(prev_keys))
            certificate = tuple(sorted(prev_keys)[:needed])

        for author in range(self.n_validators):
            block = DAGBlock(
                round=r,
                author=author,
                payload=(payloads[author] if payloads else f"r{r}-a{author}"),
                parents=certificate,
                block_id=len(self._blocks),
            )
            self._blocks[block.key()] = block
        self._round += 1

        # Try to commit: every 2 rounds, elect a leader and commit its causal
        # history (minus what's already committed).
        if self._round >= 2 and self._round % 2 == 0:
            leader_round = self._round - 2
            leader = (self._round // 2) % self.n_validators
            anchor_key = (leader_round, leader)
            if anchor_key in self._blocks:
                self._commit_causal_closure(anchor_key)

    # --- commit ---

    def _commit_causal_closure(self, anchor_key: tuple) -> None:
        """Commit all ancestors of `anchor_key` that haven't been committed yet."""
        committed_ids = {b.block_id for b in self._committed}
        to_visit = [anchor_key]
        to_commit: list[DAGBlock] = []
        seen = set()
        while to_visit:
            k = to_visit.pop()
            if k in seen or k not in self._blocks:
                continue
            seen.add(k)
            b = self._blocks[k]
            if b.block_id in committed_ids:
                continue
            to_commit.append(b)
            to_visit.extend(b.parents)
        # Deterministic order: by (round, author)
        to_commit.sort(key=lambda b: (b.round, b.author))
        self._committed.extend(to_commit)

    # --- queries ---

    @property
    def round(self) -> int:
        return self._round

    @property
    def committed(self) -> list[DAGBlock]:
        return list(self._committed)

    def throughput_per_round(self) -> float:
        if self._round == 0:
            return 0.0
        return len(self._committed) / self._round
