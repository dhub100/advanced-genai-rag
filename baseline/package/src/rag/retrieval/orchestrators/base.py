"""
Abstract base class for all orchestration strategies.

Every concrete orchestrator receives three pre-built retrieval agents and
must implement ``retrieve(query, top_k)`` which returns a tuple of
(ranked_docs, trace_dict).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseOrchestrator(ABC):
    """Common interface for orchestration strategies.

    All concrete orchestrators receive the three pre-built retrieval agents
    and must implement :meth:`retrieve`.  The ``__call__`` alias lets
    orchestrators be passed to :class:`~rag.evaluation.evaluator.ComprehensiveEvaluator`
    as plain callables.

    Attributes:
        bm25: BM25Agent instance.
        dense: DenseAgent instance.
        graph_rag: GraphAgent instance.
    """

    def __init__(self, bm25, dense, graph_rag):
        """Initialise with the three retrieval agents.

        Args:
            bm25: :class:`~rag.retrieval.agents.bm25.BM25Agent` instance.
            dense: :class:`~rag.retrieval.agents.dense.DenseAgent` instance.
            graph_rag: :class:`~rag.retrieval.agents.graph.GraphAgent` instance.
        """
        self.bm25      = bm25
        self.dense     = dense
        self.graph_rag = graph_rag

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Retrieve and fuse documents, returning a ranked list and a trace.

        Args:
            query: Natural-language question.
            top_k: Maximum number of documents to return.

        Returns:
            Tuple of:

            - ``docs``: Ordered list of LangChain Document objects.
            - ``trace``: Dict with orchestration metadata (``strategy``,
              weights used, agent hit counts, rationale, etc.).
        """

    def __call__(self, query: str, top_k: int = 10) -> Tuple[List, dict]:
        """Make the orchestrator callable; delegates to :meth:`retrieve`.

        Args:
            query: Natural-language question.
            top_k: Maximum number of documents to return.

        Returns:
            Same as :meth:`retrieve`.
        """
        return self.retrieve(query, top_k)
