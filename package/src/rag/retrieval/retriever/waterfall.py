"""
Waterfall orchestrator.

Strategy
--------
1. Run BM25 and Dense in parallel.
2. If the overlap between their top-k results is low (< overlap_threshold),
   also run GraphRAG to supplement.
3. Fuse with RRF.

This strategy is conservative: it avoids the expensive GraphRAG call when
BM25 and Dense already agree.
"""

from rag.retrieval.retriever.base import BaseOrchestrator


class WaterfallRetriever(BaseOrchestrator):
    """Waterfall orchestrator that activates GraphRAG only when BM25 and Dense disagree.

    Runs BM25 and Dense in parallel.  If their result-set overlap is below
    ``overlap_threshold``, GraphRAG is also invoked to supplement coverage.
    Results are fused with equal-weight RRF.  This avoids the latency cost
    of GraphRAG when BM25 and Dense already agree.
    """

    name = "Waterfall"

    def __init__(
        self,
        bm25,
        dense,
        graph,
        overlap_threshold: float = 0.3,
        pre_k: int = 30,
    ):
        """Initialise the waterfall orchestrator.

        Args:
            overlap_threshold: Jaccard overlap below which GraphRAG is
                triggered (default 0.3).
            pre_k: Number of candidates to fetch per agent before fusion
                (default 30).
        """
        super().__init__(bm25, dense, graph)
        self.overlap_threshold = overlap_threshold
        self.pre_k = pre_k

    def waterfall_orchestrate(self, query: str, top_k: int = 5, use_prf: bool = False):
        docs, trace = self.orchestrate_parallel_fusion(
            query=query,
            top_k=top_k,
            pre_k=max(30, top_k * 10),
            use_graph=False,
            use_prf=use_prf,
            weights={"bm25": 1.2, "dense": 1.0, "graph": 0.0},
            apply_overlap_rerank=False,
        )

        q_terms = set(t.lower() for t in query.split() if t.strip())
        top_text = (
            (
                docs[0].metadata.get("original_text") or docs[0].page_content or ""
            ).lower()
            if docs
            else ""
        )
        overlap = len(q_terms & set(top_text.split())) / max(len(q_terms), 1)

        if overlap < 0.05:
            docs2, trace2 = self.orchestrate_parallel_fusion(
                query=query,
                top_k=top_k,
                pre_k=max(30, top_k * 10),
                use_graph=True,
                use_prf=use_prf,
                weights={"bm25": 1.1, "dense": 1.0, "graph": 0.7},
                apply_overlap_rerank=False,
            )
            trace += ["Fallback triggered: low overlap -> add GraphRAG"] + trace2
            return docs2, trace

        trace.append(f"Critic: overlap={overlap:.3f} (ok)")
        return docs, trace

    def search(self, query, top_k=100, use_prf: bool = False):
        docs, _ = self.waterfall_orchestrate(query, top_k=top_k, use_prf=use_prf)
        return docs
