"""
Evidence Sufficiency Estimation (Mechanism A).

Determines whether retrieved documents contain enough evidence to answer
the query reliably, before answer generation is attempted.

Reliability signals:
    semantic_coverage     — avg cosine similarity between query and top-k docs
    chunk_support_count   — number of docs whose similarity exceeds a threshold
    aspect_coverage_ratio — fraction of query key-terms found in retrieved text
    sufficiency_score     — weighted combination of the above (0–1)

Thresholds (defaults, tunable):
    semantic_coverage     >= 0.35
    chunk_support_count   >= 3
    aspect_coverage_ratio >= 0.50
    sufficiency_score     >= 0.45
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Weights for sufficiency_score composite
_W_SEMANTIC = 0.40
_W_CHUNK = 0.35
_W_ASPECT = 0.25

_SEMANTIC_THRESHOLD = 0.35
_CHUNK_THRESHOLD_SIM = 0.40   # per-doc similarity to count as "supporting"
_CHUNK_MIN_COUNT = 3
_ASPECT_THRESHOLD = 0.50
_SUFFICIENCY_THRESHOLD = 0.45
_HARD_FLOOR = 0.20            # below this → skip recovery, abstain immediately


@dataclass
class EvidenceReport:
    """Result produced by EvidenceSufficiencyChecker."""

    sufficient: bool
    score: float                          # composite sufficiency score (0–1)
    reason: str                           # human-readable explanation
    failure_type: str                     # "ok" | "coverage_low" | "few_docs" | "low_similarity"
    missing_aspects: list[str] = field(default_factory=list)

    # Raw signal values (useful for TrustScorer / H)
    semantic_coverage: float = 0.0
    chunk_support_count: int = 0
    aspect_coverage_ratio: float = 0.0

    @property
    def below_hard_floor(self) -> bool:
        return self.score < _HARD_FLOOR


class EvidenceSufficiencyChecker:
    """
    Checks whether retrieved evidence is sufficient to answer the query.

    Reuses the sentence-transformer model already loaded by the Dense retrieval
    agent (multilingual-E5-large) — no additional model required.

    Args:
        embed_fn: callable that encodes a list of strings → np.ndarray of shape (N, D).
                  Pass the DenseRetriever's encode method, or any compatible encoder.
        semantic_threshold: minimum avg cosine similarity to consider coverage adequate.
        chunk_min_count: minimum number of supporting chunks required.
        aspect_threshold: minimum key-term coverage ratio.
        sufficiency_threshold: composite score threshold.
    """

    def __init__(
        self,
        embed_fn,
        semantic_threshold: float = _SEMANTIC_THRESHOLD,
        chunk_min_count: int = _CHUNK_MIN_COUNT,
        aspect_threshold: float = _ASPECT_THRESHOLD,
        sufficiency_threshold: float = _SUFFICIENCY_THRESHOLD,
    ):
        self._embed = embed_fn
        self.semantic_threshold = semantic_threshold
        self.chunk_min_count = chunk_min_count
        self.aspect_threshold = aspect_threshold
        self.sufficiency_threshold = sufficiency_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query: str, docs: list) -> EvidenceReport:
        """
        Evaluate evidence sufficiency for *query* given *docs*.

        Args:
            query: the user query string.
            docs:  list of LangChain Document objects (as returned by retrievers).

        Returns:
            EvidenceReport with all signals populated.
        """
        if not docs:
            return EvidenceReport(
                sufficient=False,
                score=0.0,
                reason="No documents were retrieved.",
                failure_type="few_docs",
                missing_aspects=self._extract_aspects(query),
            )

        semantic_cov = self._semantic_coverage(query, docs)
        support_count = self._chunk_support_count(query, docs)
        aspect_ratio, missing = self._aspect_coverage(query, docs)

        score = self._composite_score(semantic_cov, support_count, aspect_ratio, len(docs))
        failure_type, reason = self._diagnose(semantic_cov, support_count, aspect_ratio, score)

        return EvidenceReport(
            sufficient=score >= self.sufficiency_threshold,
            score=round(score, 4),
            reason=reason,
            failure_type=failure_type,
            missing_aspects=missing,
            semantic_coverage=round(semantic_cov, 4),
            chunk_support_count=support_count,
            aspect_coverage_ratio=round(aspect_ratio, 4),
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _semantic_coverage(self, query: str, docs: list) -> float:
        """Average cosine similarity between query embedding and each doc."""
        q_emb = self._embed([f"query: {query}"])[0]
        doc_texts = [
            f"passage: {(d.metadata.get('original_text') or d.page_content or '')[:512]}"
            for d in docs
        ]
        doc_embs = self._embed(doc_texts)
        sims = np.dot(doc_embs, q_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        return float(np.mean(sims))

    def _chunk_support_count(self, query: str, docs: list) -> int:
        """Number of docs with per-doc cosine similarity above _CHUNK_THRESHOLD_SIM."""
        q_emb = self._embed([f"query: {query}"])[0]
        doc_texts = [
            f"passage: {(d.metadata.get('original_text') or d.page_content or '')[:512]}"
            for d in docs
        ]
        doc_embs = self._embed(doc_texts)
        sims = np.dot(doc_embs, q_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        return int(np.sum(sims >= _CHUNK_THRESHOLD_SIM))

    def _aspect_coverage(self, query: str, docs: list) -> tuple[float, list[str]]:
        """
        Fraction of query key-terms found in the combined retrieved text.
        Returns (ratio, missing_terms).
        """
        aspects = self._extract_aspects(query)
        if not aspects:
            return 1.0, []

        combined = " ".join(
            (d.metadata.get("original_text") or d.page_content or "").lower()
            for d in docs
        )
        missing = [a for a in aspects if a not in combined]
        ratio = 1.0 - len(missing) / len(aspects)
        return ratio, missing

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_aspects(self, query: str) -> list[str]:
        """Extract meaningful key-terms from the query (stopword-filtered)."""
        _STOPWORDS = {
            "who", "what", "when", "where", "why", "how", "is", "are", "was",
            "were", "the", "a", "an", "of", "in", "at", "to", "for", "on",
            "and", "or", "did", "does", "do", "has", "have", "had", "be",
        }
        tokens = query.lower().split()
        return [t.strip("?.,;:") for t in tokens if t not in _STOPWORDS and len(t) > 2]

    def _composite_score(
        self,
        semantic_cov: float,
        support_count: int,
        aspect_ratio: float,
        total_docs: int,
    ) -> float:
        """Weighted composite of the three signals, normalised to [0, 1]."""
        norm_chunk = min(support_count / max(self.chunk_min_count, 1), 1.0)
        return (
            _W_SEMANTIC * min(semantic_cov / self.semantic_threshold, 1.0)
            + _W_CHUNK * norm_chunk
            + _W_ASPECT * aspect_ratio
        )

    def _diagnose(
        self,
        semantic_cov: float,
        support_count: int,
        aspect_ratio: float,
        score: float,
    ) -> tuple[str, str]:
        """Return (failure_type, human-readable reason) for the dominant weakness."""
        if score >= self.sufficiency_threshold:
            return "ok", "Evidence is sufficient to proceed with answer generation."

        # Rank weakest signal
        if support_count < self.chunk_min_count:
            return (
                "few_docs",
                f"Only {support_count} supporting chunks found (need ≥ {self.chunk_min_count}).",
            )
        if semantic_cov < self.semantic_threshold:
            return (
                "low_similarity",
                f"Avg semantic similarity {semantic_cov:.3f} is below threshold {self.semantic_threshold}.",
            )
        return (
            "coverage_low",
            f"Aspect coverage {aspect_ratio:.2%} is below threshold {self.aspect_threshold:.2%}.",
        )
