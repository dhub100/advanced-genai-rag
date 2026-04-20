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

from typing import List, Tuple

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.orchestrators.base import BaseOrchestrator


class WaterfallOrchestrator(BaseOrchestrator):
    """Waterfall orchestrator that activates GraphRAG only when BM25 and Dense disagree.

    Runs BM25 and Dense in parallel.  If their result-set overlap is below
    ``overlap_threshold``, GraphRAG is also invoked to supplement coverage.
    Results are fused with equal-weight RRF.  This avoids the latency cost
    of GraphRAG when BM25 and Dense already agree.
    """

    def __init__(self, bm25, dense, graph_rag, overlap_threshold: float = 0.3, pre_k: int = 30):
        """Initialise the waterfall orchestrator.

        Args:
            bm25: BM25Agent instance.
            dense: DenseAgent instance.
            graph_rag: GraphAgent instance.
            overlap_threshold: Jaccard overlap below which GraphRAG is
                triggered (default 0.3).
            pre_k: Number of candidates to fetch per agent before fusion
                (default 30).
        """
        super().__init__(bm25, dense, graph_rag)
        self.overlap_threshold = overlap_threshold
        self.pre_k             = pre_k

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Run the waterfall strategy and return fused documents with a trace.

        Args:
            query: Natural-language question.
            top_k: Number of documents to return after fusion.

        Returns:
            Tuple of (ranked documents, trace dict).  Trace contains
            ``strategy``, agent hit counts, ``overlap``,
            ``graph_triggered``, and ``fused_total``.
        """
        bm25_docs  = self.bm25.search(query,  top_k=self.pre_k)
        dense_docs = self.dense.search(query, top_k=self.pre_k)

        def _ids(docs):
            return {d.metadata.get("chunk_id") or d.metadata.get("record_id") for d in docs}

        bm25_ids  = _ids(bm25_docs)
        dense_ids = _ids(dense_docs)
        overlap   = len(bm25_ids & dense_ids) / max(len(bm25_ids | dense_ids), 1)

        trace = {
            "strategy":  "waterfall",
            "bm25_hits":  len(bm25_docs),
            "dense_hits": len(dense_docs),
            "overlap":    overlap,
        }

        ranked_lists = [("bm25", bm25_docs), ("dense", dense_docs)]
        if overlap < self.overlap_threshold:
            graph_docs = self.graph_rag.retrieve(query, top_k=self.pre_k)
            ranked_lists.append(("graph", graph_docs))
            trace["graph_triggered"] = True
            trace["graph_hits"]      = len(graph_docs)
        else:
            trace["graph_triggered"] = False

        fused = reciprocal_rank_fusion(ranked_lists)
        trace["fused_total"] = len(fused)
        return fused[:top_k], trace
