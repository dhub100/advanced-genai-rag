"""
Abstract base class for all orchestration strategies.

Every concrete orchestrator receives three pre-built retrieval agents and
must implement ``retrieve(query, top_k)`` which returns a tuple of
(ranked_docs, trace_dict).
"""

from collections import defaultdict

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _uid(doc):
    return doc.metadata.get("chunk_id") or doc.metadata.get("record_id")


def _safe_unique(docs):
    """Dedup by uid, keep first occurrence."""
    out = []
    seen = set()
    for d in docs:
        u = _uid(d)
        if u is None or u in seen:
            continue
        seen.add(u)
        out.append(d)
    return out


def _rrf_fuse(runs: dict, k_rrf: int = 60, weights: dict | None = None):
    """
    runs: {"bm25": [docs...], "dense": [docs...], "graph": [docs...]}
    Returns: docs sorted by weighted RRF score.
    """
    weights = weights or {"bm25": 1.0, "dense": 1.0, "graph": 0.7}
    scores: dict = defaultdict(float)
    store: dict = {}

    for name, docs in runs.items():
        w = float(weights.get(name, 1.0))
        for rank, d in enumerate(docs, start=1):
            u = _uid(d)
            if u is None:
                continue
            store.setdefault(u, d)
            scores[u] += w * (1.0 / (k_rrf + rank))

    fused = sorted(store.values(), key=lambda d: scores[_uid(d)], reverse=True)
    for d in fused:
        d.metadata["fused_score"] = float(scores[_uid(d)])
    return fused


def rerank(docs, query: str, top_k: int = 10):
    q_terms = set(t.lower() for t in query.split() if t.strip())
    scored = []
    for d in docs:
        text = (d.metadata.get("original_text") or d.page_content or "").lower()
        d_terms = set(text.split())
        overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
        scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


class BaseOrchestrator:
    """Base class for common orchestration strategies."""

    def __init__(self, bm25, dense, graph):
        self.bm25_fixed_qe = bm25
        self.dense_fixed = dense
        self.graph_rag = graph

    def orchestrate_parallel_fusion(
        self,
        query: str,
        top_k: int = 5,
        pre_k: int | None = None,
        use_graph: bool = True,
        weights: dict | None = None,
        apply_overlap_rerank: bool = False,
    ):
        trace = []
        pre_k = pre_k or max(30, top_k * 10)

        bm25_docs = self.bm25_fixed_qe.search(query, top_k=pre_k)
        trace.append(f"BM25 retrieved {len(bm25_docs)}")

        dense_docs = self.dense_fixed.search(query, top_k=pre_k)
        trace.append(f"Dense retrieved {len(dense_docs)}")

        graph_docs = []
        if use_graph:
            graph_docs = self.graph_rag.retrieve(query, top_k=pre_k)
            trace.append(f"GraphRAG retrieved {len(graph_docs)}")

        bm25_docs = _safe_unique(bm25_docs)
        dense_docs = _safe_unique(dense_docs)
        graph_docs = _safe_unique(graph_docs)

        # Fuse
        fused = _rrf_fuse(
            runs={"bm25": bm25_docs, "dense": dense_docs, "graph": graph_docs},
            k_rrf=60,
            weights=weights or {"bm25": 1.2, "dense": 1.0, "graph": 0.6},
        )
        trace.append("Fusion: weighted RRF")

        # Optional rerank for precision
        if apply_overlap_rerank:
            fused = rerank(
                fused[: max(50, top_k * 10)], query, top_k=max(50, top_k * 10)
            )
            trace.append("Post-rerank: overlap")

        return fused[:top_k], trace
