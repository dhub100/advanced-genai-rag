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


class VotingRetriever(BaseOrchestrator):
    """Voting orchestrator that fuses all three agents with equal-weight RRF.

    The simplest multi-agent strategy: run BM25, Dense, and GraphRAG in
    parallel, then fuse with equal weights.  Robust across query types and
    easy to parallelise.
    """

    name = "Voting"

    def __init__(self, bm25, dense, graph_rag, pre_k: int = 30):
        super().__init__(bm25, dense, graph_rag)
        self.pre_k = pre_k

    def voting_orchestrate(self, query: str, top_k: int = 5):
        return self.orchestrate_parallel_fusion(
            query=query,
            top_k=top_k,
            pre_k=max(30, top_k * 10),
            use_graph=True,
            weights={"bm25": 1.2, "dense": 1.0, "graph": 0.6},
            apply_overlap_rerank=False,
        )

    def search(self, query, top_k=100):
        docs, _ = self.voting_orchestrate(query, top_k=top_k)
        return docs
