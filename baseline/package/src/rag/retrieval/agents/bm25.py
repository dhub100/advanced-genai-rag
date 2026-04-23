from __future__ import annotations

from typing import Dict, List

import nltk
import numpy as np
from langchain_core.documents import Document
from langdetect import detect
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from rag.retrieval.translator import translator

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
STOP_EN = set(nltk.corpus.stopwords.words("english"))
STOP_DE = set(nltk.corpus.stopwords.words("german"))


class BilingualBM25:
    """Two BM25Okapi indices (EN / DE) with optional query translation."""

    def __init__(self, docs: List[Document]) -> None:
        self.docs_by_lang = {"en": [], "de": []}
        self.toks_by_lang = {"en": [], "de": []}
        for d in docs:
            lang = d.metadata.get("language", "en")
            lang = lang if lang in ("en", "de") else "en"
            self.docs_by_lang[lang].append(d)
            self.toks_by_lang[lang].append(word_tokenize(d.page_content))
        self.bm25 = {l: BM25Okapi(tok) for l, tok in self.toks_by_lang.items() if tok}

    def _rank_lang(self, q: str, lang: str, k: int) -> List[Document]:
        scores = self.bm25[lang].get_scores(word_tokenize(q))
        idx = np.argsort(scores)[::-1][:k]
        hits = []
        for i in idx:
            d = self.docs_by_lang[lang][i]
            d.metadata["bm25_score"] = float(scores[i])
            hits.append(d)
        return hits

    def search(self, query: str, top_k: int = 100) -> List[Document]:
        src = detect(query) if query.strip() else "en"
        src = src if src in ("en", "de") else "en"
        bag = []
        for lang in ("en", "de"):
            q_lang = translator.translate(query, lang) if lang != src else query
            bag.extend(self._rank_lang(q_lang, lang, top_k))
        # deduplicate by record/chunk id (keep highest score)
        best: Dict[str, Document] = {}
        for d in bag:
            uid = d.metadata.get("chunk_id") or d.metadata.get("record_id")
            if (
                uid not in best
                or d.metadata["bm25_score"] > best[uid].metadata["bm25_score"]
            ):
                best[uid] = d
        return sorted(
            best.values(), key=lambda d: d.metadata["bm25_score"], reverse=True
        )[:top_k]


class QEBM25:
    """BM25 + PRF wrapper"""

    def __init__(self, base: BilingualBM25) -> None:
        self.base = base

    def _expand_query(
        self, query: str, base_retriever, fb_docs: int = 5, fb_terms: int = 5
    ) -> str:
        """Simple pseudo-relevance feedback (token-frequency expansion)."""
        hits = base_retriever.search(query, top_k=fb_docs)
        tokens = [
            t.lower()
            for h in hits
            for t in word_tokenize(h.page_content.lower())
            if t.isalpha() and t not in STOP_EN and t not in STOP_DE
        ]
        extra = " ".join(w for w, _ in FreqDist(tokens).most_common(fb_terms))
        return f"{query} {extra}" if extra else query

    def search(self, query: str, top_k: int = 100) -> List[Document]:
        return self.base.search(self._expand_query(query, self.base), top_k)
