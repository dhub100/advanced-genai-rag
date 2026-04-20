"""
GraphRAG retrieval agent.

Wraps the knowledge-graph-based retriever built in the Step 2 notebook.
The graph is stored and loaded via ``load_graphrag.py`` in the notebook's
storage folder.

Usage
-----
Load the graph retriever from storage, then wrap it:

    # Inside a notebook or script that has already loaded the graph:
    from rag.retrieval.agents.graph import GraphAgent

    agent   = GraphAgent(graph_retriever_obj)
    results = agent.retrieve("What research initiatives exist at ETH?", top_k=10)
"""

from __future__ import annotations

from typing import List


class GraphAgent:
    """Thin wrapper around a knowledge-graph retriever."""

    def __init__(self, graph_retriever):
        """Wrap a knowledge-graph retriever built in the Step 2 notebook.

        Args:
            graph_retriever: Object with a ``.retrieve(query, top_k)``
                method that returns a list of Document-like objects.
        """
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """Retrieve documents by traversing the knowledge graph.

        Args:
            query: Natural-language search query.
            top_k: Number of documents to return.

        Returns:
            Ordered list of up to ``top_k`` Document-like objects as
            returned by the underlying graph retriever.
        """
        return self.graph_retriever.retrieve(query, top_k=top_k)

    def search(self, query: str, top_k: int = 10) -> list:
        """Alias for :meth:`retrieve` so GraphAgent is interchangeable with other agents.

        Args:
            query: Natural-language search query.
            top_k: Number of documents to return.

        Returns:
            Same result as :meth:`retrieve`.
        """
        return self.retrieve(query, top_k)
