"""
Dense (semantic) retrieval agent backed by ChromaDB.

Usage
-----
Load the ChromaDB collection produced in the Step 2 notebook, then wrap it:

    import chromadb
    from rag.retrieval.agents.dense import DenseAgent

    client     = chromadb.PersistentClient(path="storage/vectordb_dense")
    collection = client.get_collection("dense_fixed")
    agent      = DenseAgent(collection)
    results    = agent.search("Who is the rector of ETH Zurich?", top_k=10)
"""

from __future__ import annotations

from typing import List


class DenseAgent:
    """Thin wrapper around a ChromaDB collection for semantic retrieval."""

    def __init__(self, collection, embedding_fn=None):
        """Wrap a ChromaDB collection for semantic similarity search.

        Args:
            collection: ``chromadb.Collection`` object with stored embeddings.
            embedding_fn: Optional callable ``(text: str) -> List[float]``.
                When provided, embeddings are computed locally and passed to
                ChromaDB as ``query_embeddings``; otherwise ChromaDB uses its
                own embedding function.
        """
        self.collection   = collection
        self.embedding_fn = embedding_fn

    def search(self, query: str, top_k: int = 10) -> list:
        """Return the top-k semantically similar documents from ChromaDB.

        Converts ChromaDB's dict-based response into :class:`_SimpleDoc`
        objects so that the evaluator can access ``doc.metadata["chunk_id"]``
        uniformly across all agent types.

        Args:
            query: Natural-language search query.
            top_k: Number of documents to return.

        Returns:
            Ordered list of up to ``top_k`` :class:`_SimpleDoc` objects,
            closest embedding first.
        """
        kwargs = {"query_texts": [query], "n_results": top_k}
        if self.embedding_fn:
            kwargs = {"query_embeddings": [self.embedding_fn(query)], "n_results": top_k}

        result    = self.collection.query(**kwargs)
        docs_out  = []
        for doc_text, meta in zip(result["documents"][0], result["metadatas"][0]):
            docs_out.append(_SimpleDoc(page_content=doc_text, metadata=meta))
        return docs_out


class _SimpleDoc:
    """Minimal Document-like object compatible with the LangChain Document interface.

    Used internally by :class:`DenseAgent` so that dense retrieval results
    carry the same ``page_content`` / ``metadata`` attributes as LangChain
    Documents used by BM25Agent and GraphAgent.

    Attributes:
        page_content: Raw text content of the chunk.
        metadata: Dict of chunk metadata (must contain ``"chunk_id"`` or
            ``"record_id"`` for the evaluator to identify the document).
    """

    def __init__(self, page_content: str, metadata: dict):
        """Initialise with chunk text and metadata.

        Args:
            page_content: Plain text of the chunk.
            metadata: Metadata dict from ChromaDB.
        """
        self.page_content = page_content
        self.metadata     = metadata
