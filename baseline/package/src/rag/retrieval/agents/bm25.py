"""
BM25 retrieval agent with bilingual query expansion.

Wraps a pre-built ``rank_bm25.BM25Okapi`` index and expands queries to both
English and German before scoring, so that German documents are retrievable
from English queries and vice versa.

Usage
-----
Load your pickle from Step 2 / Step_2 notebook, then wrap it:

    import pickle
    from rag.retrieval.agents.bm25 import BM25Agent

    with open("bm25_fixed_qe.pkl", "rb") as f:
        bm25_index = pickle.load(f)

    agent = BM25Agent(bm25_index, corpus_docs)
    results = agent.search("How many students are at ETH?", top_k=10)
"""

from __future__ import annotations

from typing import List

from rag.retrieval.translator import expand_query


class BM25Agent:
    """Thin wrapper around a ``BM25Okapi`` index."""

    def __init__(self, bm25_index, corpus_docs: list, device: str = "cpu"):
        """Wrap a pre-built BM25 index together with its aligned document corpus.

        Args:
            bm25_index: ``rank_bm25.BM25Okapi`` (or compatible) index built
                over ``corpus_docs``.
            corpus_docs: List of LangChain Document objects whose order must
                match the index token lists.
            device: Torch device string for the M2M100 translation model
                (``"cpu"`` or ``"cuda"``).
        """
        self.index       = bm25_index
        self.corpus      = corpus_docs
        self.device      = device

    def search(self, query: str, top_k: int = 10) -> list:
        """Return the top-k documents scored by BM25 with bilingual query expansion.

        Translates the query to the other language (EN ↔ DE), scores the
        corpus against both query variants, sums the scores, and returns the
        highest-scoring documents.

        Args:
            query: Natural-language search query.
            top_k: Number of documents to return.

        Returns:
            Ordered list of up to ``top_k`` LangChain Document objects,
            highest BM25 score first.
        """
        import numpy as np
        from nltk.tokenize import word_tokenize

        expanded = expand_query(query, device=self.device)
        scores   = np.zeros(len(self.corpus))

        for q in expanded:
            tokens  = word_tokenize(q.lower())
            scores += np.array(self.index.get_scores(tokens))

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in top_indices]
