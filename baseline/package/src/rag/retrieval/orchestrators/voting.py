"""
Voting orchestrator.

Strategy
--------
Run all three agents in parallel, then fuse with equal-weight RRF.
Simple, parallelisable, and robust across query types.
"""

from typing import List, Tuple

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.orchestrators.base import BaseOrchestrator


class VotingOrchestrator(BaseOrchestrator):
    """Voting orchestrator that fuses all three agents with equal-weight RRF.

    The simplest multi-agent strategy: run BM25, Dense, and GraphRAG in
    parallel, then fuse with equal weights.  Robust across query types and
    easy to parallelise.
    """

    def __init__(self, bm25, dense, graph_rag, pre_k: int = 30):
        """Initialise the voting orchestrator.

        Args:
            bm25: BM25Agent instance.
            dense: DenseAgent instance.
            graph_rag: GraphAgent instance.
            pre_k: Number of candidates to fetch per agent before fusion
                (default 30).
        """
        super().__init__(bm25, dense, graph_rag)
        self.pre_k = pre_k

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Run all three agents and fuse with equal-weight RRF.

        Args:
            query: Natural-language question.
            top_k: Number of documents to return after fusion.

        Returns:
            Tuple of (ranked documents, trace dict).  Trace contains
            ``strategy``, per-agent hit counts, and ``fused_total``.
        """
        bm25_docs  = self.bm25.search(query,         top_k=self.pre_k)
        dense_docs = self.dense.search(query,         top_k=self.pre_k)
        graph_docs = self.graph_rag.retrieve(query,   top_k=self.pre_k)

        ranked_lists = [("bm25", bm25_docs), ("dense", dense_docs), ("graph", graph_docs)]
        fused        = reciprocal_rank_fusion(ranked_lists)

        trace = {
            "strategy":   "voting",
            "bm25_hits":   len(bm25_docs),
            "dense_hits":  len(dense_docs),
            "graph_hits":  len(graph_docs),
            "fused_total": len(fused),
        }
        return fused[:top_k], trace
