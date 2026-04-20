"""
Reciprocal Rank Fusion (RRF) and other rank-aggregation utilities.

RRF formula: score(d) = Σ_agent  weight / (k + rank(d, agent))
where k=60 is the standard smoothing constant.
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[str, list]],
    weights: Dict[str, float] | None = None,
    k: int = 60,
) -> list:
    """Fuse multiple ranked document lists using weighted Reciprocal Rank Fusion.

    For each document ``d`` the fusion score is:
    ``score(d) = Σ_agent  weight(agent) / (k + rank(d, agent))``

    Documents absent from an agent's list contribute 0 for that agent.

    Args:
        ranked_lists: List of ``(agent_name, docs)`` tuples where ``docs``
            is an ordered list of LangChain Document objects (rank 1 first).
        weights: Optional dict mapping agent name to a positive float weight.
            Defaults to 1.0 for every agent.
        k: RRF smoothing constant that prevents high scores for rank-1
            documents dominating too strongly (default 60).

    Returns:
        Sorted list of Document objects, highest fusion score first.
    """
    weights    = weights or {}
    scores     = defaultdict(float)
    doc_store  = {}

    for agent_name, docs in ranked_lists:
        w = weights.get(agent_name, 1.0)
        for rank, doc in enumerate(docs, 1):
            uid = doc.metadata.get("chunk_id") or doc.metadata.get("record_id")
            if uid:
                doc_store[uid] = doc
                scores[uid]   += w / (k + rank)

    return sorted(doc_store.values(),
                  key=lambda d: scores[d.metadata.get("chunk_id") or d.metadata.get("record_id")],
                  reverse=True)
