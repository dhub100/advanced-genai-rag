"""
Adaptive (Q-learning) orchestrator — bonus strategy.

Strategy
--------
Maintains a Q-table keyed on (query_label, strategy_choice) and updates it
via epsilon-greedy online learning using MRR as the reward signal.

On each call the orchestrator:
  1. Classifies the query.
  2. Selects an action (weight preset) via epsilon-greedy.
  3. Retrieves and fuses.
  4. After a reward is supplied via ``update()``, updates the Q-table.

NOTE: Reward feedback must be provided externally after relevance judgements
are available.  See ``update(query_label, action, reward)``.
"""

import random
from collections import defaultdict
from typing import List, Tuple

from rag.retrieval.classifier import classify_query
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.orchestrators.base import BaseOrchestrator

_ACTIONS = {
    "bm25_heavy":   {"bm25": 1.4, "dense": 0.9, "graph": 0.5},
    "dense_heavy":  {"bm25": 1.0, "dense": 1.3, "graph": 0.8},
    "balanced":     {"bm25": 1.2, "dense": 1.0, "graph": 0.6},
    "graph_heavy":  {"bm25": 0.8, "dense": 1.0, "graph": 1.4},
}


class AdaptiveOrchestrator(BaseOrchestrator):
    """Adaptive (Q-learning) orchestrator that improves weights online from MRR feedback.

    Maintains a Q-table keyed on ``(query_label, action)`` and updates it
    via epsilon-greedy exploration using MRR as the reward signal.  Reward
    must be provided externally via :meth:`update` after relevance judgements
    are available.

    Attributes:
        epsilon: Exploration probability (default 0.1).
        alpha: Q-learning step size (default 0.3).
        gamma: Discount factor for future rewards (default 0.9).
        pre_k: Candidates fetched per agent before fusion.
        q_table: Q-value table ``{(label, action): float}``.
    """

    def __init__(self, bm25, dense, graph_rag,
                 epsilon: float = 0.1, alpha: float = 0.3, gamma: float = 0.9,
                 pre_k: int = 30):
        """Initialise the adaptive orchestrator with Q-learning hyperparameters.

        Args:
            bm25: BM25Agent instance.
            dense: DenseAgent instance.
            graph_rag: GraphAgent instance.
            epsilon: Probability of choosing a random action instead of the
                greedy best (exploration rate, default 0.1).
            alpha: Q-table learning rate (default 0.3).
            gamma: Discount factor applied to the best future Q-value
                (default 0.9).
            pre_k: Candidates to fetch per agent before fusion (default 30).
        """
        super().__init__(bm25, dense, graph_rag)
        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma
        self.pre_k   = pre_k
        self.q_table: dict = defaultdict(float)
        self._last: tuple | None = None

    def _select_action(self, label: str) -> str:
        """Select a weight preset via epsilon-greedy exploration.

        Args:
            label: Query label string (``"factoid"``, ``"semantic"``, or
                ``"balanced"``).

        Returns:
            Action name string from ``_ACTIONS`` (e.g. ``"bm25_heavy"``).
        """
        if random.random() < self.epsilon:
            return random.choice(list(_ACTIONS))
        return max(_ACTIONS, key=lambda a: self.q_table[(label, a)])

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Classify query, select action via epsilon-greedy, and return fused results.

        Args:
            query: Natural-language question.
            top_k: Number of documents to return after fusion.

        Returns:
            Tuple of (ranked documents, trace dict).  Trace contains
            ``strategy``, ``query_label``, ``action``, ``weights``,
            ``epsilon``, and ``fused_total``.
        """
        features = classify_query(query)
        label    = features["label"]
        action   = self._select_action(label)
        self._last = (label, action)

        weights  = _ACTIONS[action]
        bm25_docs  = self.bm25.search(query,       top_k=self.pre_k)
        dense_docs = self.dense.search(query,       top_k=self.pre_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=self.pre_k)

        ranked_lists = [("bm25", bm25_docs), ("dense", dense_docs), ("graph", graph_docs)]
        fused        = reciprocal_rank_fusion(ranked_lists, weights=weights)

        trace = {
            "strategy":    "adaptive",
            "query_label": label,
            "action":      action,
            "weights":     weights,
            "epsilon":     self.epsilon,
            "fused_total": len(fused),
        }
        return fused[:top_k], trace

    def update(self, query_label: str, action: str, reward: float) -> None:
        """Update the Q-table with an observed reward signal.

        Should be called after MRR (or another relevance metric) is computed
        for the most recent :meth:`retrieve` call.

        Args:
            query_label: Label of the query that generated the reward
                (``"factoid"``, ``"semantic"``, or ``"balanced"``).
            action: Action string that was executed (key in ``_ACTIONS``).
            reward: Observed reward, typically the MRR for the query.
        """
        key         = (query_label, action)
        best_next   = max(self.q_table[(query_label, a)] for a in _ACTIONS)
        self.q_table[key] += self.alpha * (reward + self.gamma * best_next - self.q_table[key])
