"""
Evidence Sufficiency Estimation (Mechanism A).

Determines whether retrieved documents contain enough evidence to answer
the query reliably, before answer generation is attempted.

Reliability signals:
    semantic_coverage     — avg cosine similarity between query and top-k docs
    chunk_support_count   — number of docs whose similarity exceeds a threshold
    aspect_coverage_ratio — fraction of query key-terms found in retrieved text
    temporal_validity     — if query contains a year, at least one doc must reference it
    sufficiency_score     — weighted combination of the above (0–1)

Thresholds (defaults, tunable):
    semantic_coverage     >= 0.35
    chunk_support_count   >= 3
    aspect_coverage_ratio >= 0.50
    sufficiency_score     >= 0.45
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Weights for sufficiency_score composite
_W_SEMANTIC = 0.40
_W_CHUNK = 0.35
_W_ASPECT = 0.25

_SEMANTIC_THRESHOLD = 0.50
_CHUNK_THRESHOLD_SIM = 0.40   # per-doc similarity to count as "supporting"
_CHUNK_MIN_COUNT = 3
_ASPECT_THRESHOLD = 0.50
_SUFFICIENCY_THRESHOLD = 0.45
_HARD_FLOOR = 0.20            # below this → skip recovery, abstain immediately

# Years plausibly referenceable in the corpus (outside this range → always invalid)
_YEAR_MIN = 1850
_YEAR_MAX = 2030


@dataclass
class EvidenceReport:
    """Result produced by EvidenceSufficiencyChecker."""

    sufficient: bool
    score: float                          # composite sufficiency score (0–1)
    reason: str                           # human-readable explanation
    failure_type: str                     # "ok" | "coverage_low" | "few_docs" | "low_similarity" | "temporal_mismatch"
    missing_aspects: list[str] = field(default_factory=list)

    # Raw signal values (useful for TrustScorer / H)
    semantic_coverage: float = 0.0
    chunk_support_count: int = 0
    aspect_coverage_ratio: float = 0.0
    temporal_valid: bool = True           # False when query year absent from all retrieved docs

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
        temporal_valid, query_years = self._temporal_validity(query, docs)

        score = self._composite_score(semantic_cov, support_count, aspect_ratio, len(docs))
        failure_type, reason = self._diagnose(
            semantic_cov, support_count, aspect_ratio, score, temporal_valid, query_years
        )

        # Temporal mismatch forces insufficient regardless of composite score
        if not temporal_valid:
            score = min(score, self.sufficiency_threshold - 0.01)

        return EvidenceReport(
            sufficient=temporal_valid and score >= self.sufficiency_threshold,
            score=round(score, 4),
            reason=reason,
            failure_type=failure_type,
            missing_aspects=missing + [str(y) for y in query_years if not temporal_valid],
            semantic_coverage=round(semantic_cov, 4),
            chunk_support_count=support_count,
            aspect_coverage_ratio=round(aspect_ratio, 4),
            temporal_valid=temporal_valid,
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

    def _temporal_validity(self, query: str, docs: list) -> tuple[bool, list[int]]:
        """
        If the query contains a year, check whether at least one retrieved
        document references that year.

        Returns:
            (valid, query_years) where valid is False when a query year is
            present but absent from all retrieved documents.

        Years outside [_YEAR_MIN, _YEAR_MAX] are considered out-of-corpus
        range and immediately return (False, [year]).
        """
        query_years = self._extract_years(query)
        if not query_years:
            return True, []

        combined = " ".join(
            (d.metadata.get("original_text") or d.page_content or "")
            for d in docs
        )
        doc_years = set(self._extract_years(combined))

        for year in query_years:
            # Year outside plausible corpus range → definitely not in corpus
            if not (_YEAR_MIN <= year <= _YEAR_MAX):
                return False, [year]
            # Year in plausible range but absent from all retrieved docs
            if year not in doc_years:
                return False, [year]

        return True, query_years

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_years(self, text: str) -> list[int]:
        """Extract 4-digit year candidates from text."""
        return [int(y) for y in re.findall(r"\b(1\d{3}|2\d{3})\b", text)]

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
        temporal_valid: bool,
        query_years: list[int],
    ) -> tuple[str, str]:
        """Return (failure_type, human-readable reason) for the dominant weakness."""

        # Temporal mismatch takes priority — it overrides a passing composite score
        if not temporal_valid:
            years_str = ", ".join(str(y) for y in query_years)
            if any(y > _YEAR_MAX for y in query_years):
                return (
                    "temporal_mismatch",
                    f"Query references year(s) {years_str} which are outside the corpus "
                    f"range ({_YEAR_MIN}–{_YEAR_MAX}). No document can contain this information.",
                )
            return (
                "temporal_mismatch",
                f"Query references year(s) {years_str} but none of the retrieved documents "
                f"mention this period. The specific fact may not exist in the corpus.",
            )

        if score >= self.sufficiency_threshold:
            return "ok", "Evidence is sufficient to proceed with answer generation."

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
