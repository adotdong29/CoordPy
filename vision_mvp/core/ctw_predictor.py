"""Context-Tree Weighting — universal binary source prediction (OQ3).

Open Question 3 asks how to bootstrap a Stage-3 world model when you need the
world model to generate training data. The classical CASR answer is a curriculum
that gradually raises τ_i as M_i improves. CTW (Willems, Shtarkov, Tjalkens,
1995) dissolves the question: it achieves the entropy rate of *any* binary
stationary tree source without being told the source or the model order.

For a depth-D binary context tree, each node keeps counts (a, b) of 0s and 1s
observed after that context. The Krichevsky–Trofimov estimator is

    P_KT(next = 0 | a, b) = (a + ½) / (a + b + 1)

At each internal node (depth < D), a weighted mixture gives the CTW estimate:

    P_w = ½ P_KT + ½ ∏ P_w(children)

At depth D (the max), P_w = P_KT (the node acts as a leaf). Missing (never-
allocated) children contribute P_w = 1 to the product — this matches the full-
tree convention with lazy allocation.

Total predictor size O(2^D · log T), update per symbol O(D). Converges to the
entropy rate of any tree source of order ≤ D with redundancy O(log T / T).

For CASR, binarise the agent's prediction-error sign and run CTW on the symbol
stream. The CTW predicted-probability for the next sign is the "unsurprising"
mass; 1 − p_ctw is the surprise signal. No training curriculum required — CTW
converges automatically.

References: Willems et al., "The Context-Tree Weighting Method: Basic Properties"
(IEEE TIT 1995).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class _CTWNode:
    """One node in the depth-bounded binary context tree."""
    a: int = 0                          # zeros observed after this context
    b: int = 0                          # ones observed after this context
    log_pkt: float = 0.0                # Σ log P_KT(symbol | prior a,b) so far
    log_pw: float = 0.0                 # weighted estimator for subtree rooted here
    left: "_CTWNode | None" = None      # child for next symbol = 0
    right: "_CTWNode | None" = None     # child for next symbol = 1

    def get_or_create_child(self, symbol: int) -> "_CTWNode":
        if symbol == 0:
            if self.left is None:
                self.left = _CTWNode()
            return self.left
        if symbol == 1:
            if self.right is None:
                self.right = _CTWNode()
            return self.right
        raise ValueError("symbol must be 0 or 1")


def _log_kt_update(a: int, b: int, symbol: int) -> float:
    """ln P_KT(new_symbol | a, b) under the Krichevsky-Trofimov rule.

    P_KT(next=0) = (a + ½) / (a + b + 1)
    P_KT(next=1) = (b + ½) / (a + b + 1)
    """
    denom = a + b + 1.0
    numer = (a + 0.5) if symbol == 0 else (b + 0.5)
    return float(np.log(numer / denom))


def _recompute_log_pw(node: _CTWNode, is_max_depth: bool) -> float:
    """Return log P_w for `node` given its current children and log_pkt.

    At max depth the node acts as a leaf: log_pw = log_pkt. Otherwise the
    mixture formula is used, with missing children contributing log_pw = 0.
    """
    if is_max_depth:
        return node.log_pkt
    left_pw = node.left.log_pw if node.left is not None else 0.0
    right_pw = node.right.log_pw if node.right is not None else 0.0
    a = node.log_pkt
    b = left_pw + right_pw
    m = max(a, b)
    # log(½(e^a + e^b)) stably:
    return m + float(np.log(0.5 * (np.exp(a - m) + np.exp(b - m))))


class CTW:
    """Binary Context-Tree Weighting predictor.

    Usage:
        ctw = CTW(depth=6)
        for s in stream:            # stream of 0/1 ints
            p1 = ctw.predict_prob_one()
            ctw.observe(s)

    `depth` bounds the tree; longer context is truncated. Tree grows lazily.
    """

    def __init__(self, depth: int = 6):
        if depth < 0:
            raise ValueError("depth must be ≥ 0")
        self.depth = depth
        self._past: list[int] = []
        self._root = _CTWNode()

    # --- path traversal --------------------------------------------------

    def _context(self) -> list[int]:
        """Context of length exactly `depth`, most-recent-first, prepended-
        zeros convention (Willems 1995): when history is shorter than depth,
        pad with 0 symbols at the deep end so every observation descends to
        a depth-D leaf. This gives correct probability normalization starting
        from the empty history (P(0) = P(1) = 0.5).
        """
        past_recent_first = list(reversed(self._past))[:self.depth]
        pad = self.depth - len(past_recent_first)
        return past_recent_first + [0] * pad

    def _descend(self, context: list[int], create: bool) -> list[_CTWNode]:
        """Return [root, ..., deepest_reachable_or_created_node]."""
        path = [self._root]
        node = self._root
        for s in context:
            if create:
                node = node.get_or_create_child(s)
            else:
                nxt = node.left if s == 0 else node.right
                if nxt is None:
                    break
                node = nxt
            path.append(node)
        return path

    # --- mutating update -------------------------------------------------

    def _apply_update(self, path: list[_CTWNode], symbol: int) -> None:
        """Update counts, log_pkt, log_pw along `path` for `symbol`.

        Modifies `path` in place. Does NOT touch self._past.
        """
        # KT and count update, bottom-up (order doesn't matter since each
        # node's update is local to its own fields).
        for node in path:
            node.log_pkt += _log_kt_update(node.a, node.b, symbol)
            if symbol == 0:
                node.a += 1
            else:
                node.b += 1
        # log_pw recompute, bottom-up (depth len(path)-1 down to 0).
        n = len(path)
        for idx in range(n - 1, -1, -1):
            actual_depth = idx
            is_max = actual_depth == self.depth
            path[idx].log_pw = _recompute_log_pw(path[idx], is_max)

    def observe(self, symbol: int) -> None:
        if symbol not in (0, 1):
            raise ValueError("symbol must be 0 or 1")
        context = self._context()
        path = self._descend(context, create=True)
        self._apply_update(path, symbol)
        self._past.append(symbol)

    # --- non-mutating query ---------------------------------------------

    def predict_prob_one(self) -> float:
        """Pr[next symbol = 1 | history] under CTW."""
        context = self._context()
        path = self._descend(context, create=True)

        # Save mutable fields on the path (we only touch these nodes).
        saved = [(n.a, n.b, n.log_pkt, n.log_pw) for n in path]
        log_p_before = self._root.log_pw

        self._apply_update(path, 1)
        log_p_after = self._root.log_pw

        for node, (a, b, pkt, pw) in zip(path, saved):
            node.a, node.b, node.log_pkt, node.log_pw = a, b, pkt, pw

        return float(np.exp(log_p_after - log_p_before))

    def surprise_bits(self, symbol: int) -> float:
        """Information content (bits) of `symbol` under the current predictor."""
        if symbol not in (0, 1):
            raise ValueError("symbol must be 0 or 1")
        p1 = self.predict_prob_one()
        p = p1 if symbol == 1 else 1.0 - p1
        p = max(p, 1e-12)
        return float(-np.log2(p))

    # --- bulk helpers ---------------------------------------------------

    def observe_sequence(self, symbols: Iterable[int]) -> None:
        for s in symbols:
            self.observe(int(s))

    def code_length(self, symbols: Iterable[int]) -> float:
        """Total bits used to stream-encode `symbols`."""
        total = 0.0
        for s in symbols:
            s = int(s)
            total += self.surprise_bits(s)
            self.observe(s)
        return total


# ----------------------------------------------------------------------
# Surprise wrapper — drop-in replacement for Stage-3 surprise thresholding
# that sidesteps OQ3 (no trained world model required).
# ----------------------------------------------------------------------

class CTWSurpriseDetector:
    """Threshold-based binary surprise detector fed by real-valued prediction
    errors. The sign of the error is the observed symbol; surprise is the bit
    content under the evolving CTW predictor.

    Convergence to the entropy rate of the sign process is guaranteed by CTW
    universality — no hand-tuned threshold, no training phase.
    """

    def __init__(self, depth: int = 6, surprise_bits: float = 2.0):
        if surprise_bits <= 0:
            raise ValueError("surprise_bits must be > 0")
        self.ctw = CTW(depth=depth)
        self.surprise_bits = surprise_bits

    def update(self, error: float) -> bool:
        """Feed one error; return True iff it was surprising.

        error < 0 → symbol 0; error ≥ 0 → symbol 1.
        """
        symbol = 0 if error < 0 else 1
        bits = self.ctw.surprise_bits(symbol)
        self.ctw.observe(symbol)
        return bits >= self.surprise_bits
