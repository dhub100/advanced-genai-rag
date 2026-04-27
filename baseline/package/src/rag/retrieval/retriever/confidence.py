"""
Confidence-based orchestrator.

Strategy
--------
Classify the query type, then assign asymmetric RRF weights:
  - factoid  → BM25-heavy  (1.4 / 0.9 / 0.5)
  - semantic → Dense-heavy (1.0 / 1.3 / 0.8)
  - balanced → default     (1.2 / 1.0 / 0.6)
"""

from rag.retrieval.agents.query_classifier import QueryClassifierAgent
from rag.retrieval.retriever.base import BaseOrchestrator

_WEIGHT_PRESETS = {
    "factoid": {"bm25": 1.4, "dense": 0.9, "graph": 0.5},
    "semantic": {"bm25": 1.0, "dense": 1.3, "graph": 0.8},
    "balanced": {"bm25": 1.2, "dense": 1.0, "graph": 0.6},
}


class ConfidenceRetriever(BaseOrchestrator):
    """Confidence-based orchestrator that assigns query-type-aware RRF weights.

    Classifies the query as ``"factoid"``, ``"semantic"``, or ``"balanced"``
    using :func:`~rag.retrieval.classifier.classify_query`, then applies the
    corresponding weight preset from ``_WEIGHT_PRESETS``.
    """

    name = "Confidence"

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

    def confidence_orchestrate(self, query: str, top_k: int = 5, use_prf: bool = False):
        """
        We choose weights dynamically based on the type of query, classified by the QueryClassifierAgent.
        """

        classifier = QueryClassifierAgent()

        # classify query
        query_class = classifier.classify(query)

        if query_class == "FACTOID":
            w = {"bm25": 1.4, "dense": 0.9, "graph": 0.5}
            mode = "factoid -> BM25 boosted"
        elif query_class == "SEMANTIC":
            w = {"bm25": 0.9, "dense": 1.3, "graph": 0.6}
            mode = "semantic -> Dense boosted"
        # HYBRID and fallback if classifier fails
        else:
            w = {"bm25": 1.0, "dense": 1.1, "graph": 0.5}
            mode = "hybrid -> balanced"

        docs, trace = self.orchestrate_parallel_fusion(
            query=query,
            top_k=top_k,
            pre_k=max(30, top_k * 10),
            use_graph=True,
            use_prf=use_prf,
            weights=w,
            apply_overlap_rerank=False,
        )
        trace.insert(0, f"Confidence router: {mode}")
        return docs, trace

    def search(self, query, top_k=100, use_prf: bool = False):
        docs, _ = self.confidence_orchestrate(query, top_k=top_k, use_prf=use_prf)
        return docs
