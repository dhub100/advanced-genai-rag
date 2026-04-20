"""
Confidence-based orchestrator.

Strategy
--------
Classify the query type, then assign asymmetric RRF weights:
  - factoid  → BM25-heavy  (1.4 / 0.9 / 0.5)
  - semantic → Dense-heavy (1.0 / 1.3 / 0.8)
  - balanced → default     (1.2 / 1.0 / 0.6)
"""

from typing import List, Tuple

from rag.retrieval.classifier import classify_query
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.orchestrators.base import BaseOrchestrator

_WEIGHT_PRESETS = {
    "factoid":  {"bm25": 1.4, "dense": 0.9, "graph": 0.5},
    "semantic": {"bm25": 1.0, "dense": 1.3, "graph": 0.8},
    "balanced": {"bm25": 1.2, "dense": 1.0, "graph": 0.6},
}


class ConfidenceOrchestrator(BaseOrchestrator):
    """Confidence-based orchestrator that assigns query-type-aware RRF weights.

    Classifies the query as ``"factoid"``, ``"semantic"``, or ``"balanced"``
    using :func:`~rag.retrieval.classifier.classify_query`, then applies the
    corresponding weight preset from ``_WEIGHT_PRESETS``.
    """

    def __init__(self, bm25, dense, graph_rag, pre_k: int = 30):
        """Initialise the confidence orchestrator.

        Args:
            bm25: BM25Agent instance.
            dense: DenseAgent instance.
            graph_rag: GraphAgent instance.
            pre_k: Candidates to fetch per agent before fusion (default 30).
        """
        super().__init__(bm25, dense, graph_rag)
        self.pre_k = pre_k

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Classify the query, select weights, and return fused results.

        Args:
            query: Natural-language question.
            top_k: Number of documents to return after fusion.

        Returns:
            Tuple of (ranked documents, trace dict).  Trace contains
            ``strategy``, ``query_label``, ``weights``, ``features``, and
            ``fused_total``.
        """
        features = classify_query(query)
        weights  = _WEIGHT_PRESETS[features["label"]]

        bm25_docs  = self.bm25.search(query,       top_k=self.pre_k)
        dense_docs = self.dense.search(query,       top_k=self.pre_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=self.pre_k)

        ranked_lists = [("bm25", bm25_docs), ("dense", dense_docs), ("graph", graph_docs)]
        fused        = reciprocal_rank_fusion(ranked_lists, weights=weights)

        trace = {
            "strategy":    "confidence",
            "query_label": features["label"],
            "weights":     weights,
            "features":    features,
            "fused_total": len(fused),
        }
        return fused[:top_k], trace
