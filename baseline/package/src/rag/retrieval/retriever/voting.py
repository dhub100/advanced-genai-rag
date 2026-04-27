"""
Voting orchestrator.

Strategy
--------
Run all three agents in parallel, then fuse with equal-weight RRF.
Simple, parallelisable, and robust across query types.
"""

from rag.retrieval.retriever.base import BaseOrchestrator


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

    def voting_orchestrate(self, query: str, top_k: int = 5, use_prf: bool = False):
        return self.orchestrate_parallel_fusion(
            query=query,
            top_k=top_k,
            pre_k=max(30, top_k * 10),
            use_graph=True,
            use_prf=use_prf,
            weights={"bm25": 1.2, "dense": 1.0, "graph": 0.6},
            apply_overlap_rerank=False,
        )

    def search(self, query, top_k=100, use_prf: bool = False):
        docs, _ = self.voting_orchestrate(query, top_k=top_k, use_prf=use_prf)
        return docs
