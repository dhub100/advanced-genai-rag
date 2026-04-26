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

The initial scoring layer is preserved, then refined by a gating / routing
layer that distinguishes topical support from answer-bearing support.
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
_CHUNK_THRESHOLD_SIM = 0.40
_CHUNK_MIN_COUNT = 3
_ASPECT_THRESHOLD = 0.50
_SUFFICIENCY_THRESHOLD = 0.45
_HARD_FLOOR = 0.20

_YEAR_MIN = 1850
_YEAR_MAX = 2030

_STOPWORDS = {
    "who", "what", "when", "where", "why", "how", "is", "are", "was",
    "were", "the", "a", "an", "of", "in", "at", "to", "for", "on",
    "and", "or", "did", "does", "do", "has", "have", "had", "be",
}

_CRITICAL_ASPECTS = {
    "grant", "grants", "erc", "president", "rector", "year", "date", "who", "when",
}
_AMBIGUOUS_MARKERS = (
    "famous", "best", "top", "main", "important", "well known", "known for", "leading",
    "strongest",
)
_AMBIGUITY_CRITERION_MARKERS = (
    "known for", "famous for", "best known", "renowned for", "leading research",
)
_ROLE_TERMS = {
    "president", "rector", "director", "head", "leader", "dean", "chair",
}
_WHO_ACTION_TERMS = {
    "received", "awarded", "won", "grant", "grants", "erc", "president", "rector",
    "professor", "director", "chair",
}


@dataclass
class EvidenceReport:
    """Result produced by EvidenceSufficiencyChecker."""

    sufficient: bool
    score: float
    reason: str
    failure_type: str
    missing_aspects: list[str] = field(default_factory=list)

    semantic_coverage: float = 0.0
    chunk_support_count: int = 0
    aspect_coverage_ratio: float = 0.0
    temporal_valid: bool = True

    routing_decision: str = "recover"
    decision_reason: str = ""
    decision_trace: list[str] = field(default_factory=list)

    @property
    def below_hard_floor(self) -> bool:
        return self.score < _HARD_FLOOR

    @property
    def sufficiency_score(self) -> float:
        """Compatibility alias for notebooks / reports."""
        return self.score


class EvidenceSufficiencyChecker:
    """
    Checks whether retrieved evidence is sufficient to answer the query.

    Reuses the sentence-transformer model already loaded by the Dense retrieval
    agent (multilingual-E5-large) — no additional model required.
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

    def check(self, query: str, docs: list) -> EvidenceReport:
        """Evaluate evidence sufficiency for *query* given *docs*."""
        if not docs:
            missing_aspects = self._extract_aspects(query)
            return EvidenceReport(
                sufficient=False,
                score=0.0,
                reason="No documents were retrieved.",
                failure_type="few_supporting_chunks",
                missing_aspects=missing_aspects,
                routing_decision="abstain",
                decision_reason="No documents were retrieved, so there is no basis for a reliable answer.",
                decision_trace=[
                    "No retrieved documents available.",
                    "Hard floor triggered because chunk_support_count is zero.",
                ],
            )

        semantic_cov = self._semantic_coverage(query, docs)
        support_count = self._chunk_support_count(query, docs)
        aspect_ratio, missing = self._aspect_coverage(query, docs)
        temporal_valid, query_years = self._temporal_validity(query, docs)
        base_score = self._composite_score(semantic_cov, support_count, aspect_ratio)

        decision = self._route_decision(
            query=query,
            docs=docs,
            semantic_cov=semantic_cov,
            support_count=support_count,
            aspect_ratio=aspect_ratio,
            base_score=base_score,
            temporal_valid=temporal_valid,
            query_years=query_years,
            missing_aspects=missing,
        )

        final_score = decision["score"]
        routing_decision = decision["routing_decision"]
        failure_type = decision["failure_type"]
        decision_reason = decision["decision_reason"]
        decision_trace = decision["decision_trace"]
        sufficient = routing_decision == "answer"

        return EvidenceReport(
            sufficient=sufficient,
            score=round(final_score, 4),
            reason=decision_reason,
            failure_type=failure_type,
            missing_aspects=missing + [str(y) for y in query_years if not temporal_valid],
            semantic_coverage=round(semantic_cov, 4),
            chunk_support_count=support_count,
            aspect_coverage_ratio=round(aspect_ratio, 4),
            temporal_valid=temporal_valid,
            routing_decision=routing_decision,
            decision_reason=decision_reason,
            decision_trace=decision_trace,
        )

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
        """Fraction of query key-terms found in the combined retrieved text."""
        aspects = self._extract_aspects(query)
        if not aspects:
            return 1.0, []

        combined = self._combined_text(docs)
        missing = [a for a in aspects if a not in combined]
        ratio = 1.0 - len(missing) / len(aspects)
        return ratio, missing

    def _temporal_validity(self, query: str, docs: list) -> tuple[bool, list[int]]:
        """
        If the query contains a year, check whether at least one retrieved
        document references that year.
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
            if not (_YEAR_MIN <= year <= _YEAR_MAX):
                return False, [year]
            if year not in doc_years:
                return False, [year]

        return True, query_years

    def _extract_years(self, text: str) -> list[int]:
        """Extract 4-digit year candidates from text."""
        return [int(y) for y in re.findall(r"\b(1\d{3}|2\d{3})\b", text)]

    def _extract_aspects(self, query: str) -> list[str]:
        """Extract meaningful key-terms from the query (stopword-filtered)."""
        tokens = query.lower().split()
        return [t.strip("?.,;:") for t in tokens if t not in _STOPWORDS and len(t) > 2]

    def _composite_score(
        self,
        semantic_cov: float,
        support_count: int,
        aspect_ratio: float,
    ) -> float:
        """Weighted composite of the three signals, normalised to [0, 1]."""
        norm_chunk = min(support_count / max(self.chunk_min_count, 1), 1.0)
        return (
            _W_SEMANTIC * min(semantic_cov / self.semantic_threshold, 1.0)
            + _W_CHUNK * norm_chunk
            + _W_ASPECT * aspect_ratio
        )

    def _route_decision(
        self,
        query: str,
        docs: list,
        semantic_cov: float,
        support_count: int,
        aspect_ratio: float,
        base_score: float,
        temporal_valid: bool,
        query_years: list[int],
        missing_aspects: list[str],
    ) -> dict:
        """
        Apply post-score gating logic that maps evidence quality to an action.
        """
        trace = [
            (
                "Scoring layer: "
                f"semantic_coverage={semantic_cov:.3f}, "
                f"chunk_support_count={support_count}, "
                f"aspect_coverage_ratio={aspect_ratio:.3f}, "
                f"temporal_valid={temporal_valid}, "
                f"base_score={base_score:.3f}"
            )
        ]
        combined_text = self._combined_text(docs)
        final_score = base_score

        if semantic_cov < 0.15 or support_count == 0:
            final_score = min(final_score, _HARD_FLOOR - 0.01)
            trace.append("Hard floor: semantic coverage is extremely weak or there are zero supporting chunks.")
            return self._decision(
                routing_decision="abstain",
                failure_type="hard_floor",
                decision_reason="Retrieved evidence is too weak to support recovery or answer generation.",
                score=final_score,
                trace=trace,
            )

        if not temporal_valid:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            years_str = ", ".join(str(y) for y in query_years)
            if any(y > _YEAR_MAX or y < _YEAR_MIN for y in query_years):
                reason = (
                    f"Query references year(s) {years_str} outside the plausible corpus range "
                    f"({_YEAR_MIN}–{_YEAR_MAX})."
                )
                trace.append("Temporal validity failed because the query year is out of corpus scope.")
                return self._decision("abstain", "temporal_mismatch", reason, final_score, trace)

            trace.append("Temporal validity failed because the retrieved evidence never mentions the query year.")
            return self._decision(
                "recover",
                "temporal_mismatch",
                f"Retrieved evidence is topically related but does not mention query year(s) {years_str}.",
                final_score,
                trace,
            )

        critical_missing = [aspect for aspect in missing_aspects if aspect in _CRITICAL_ASPECTS]
        if critical_missing:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append(
                "Critical aspect override triggered for missing query aspect(s): "
                + ", ".join(critical_missing)
            )
            failure_type = "missing_critical_aspect"
            reason = (
                "Retrieved evidence is topically related, but it is missing critical query aspect(s): "
                + ", ".join(critical_missing)
                + "."
            )
            return self._decision("recover", failure_type, reason, final_score, trace)

        if self._is_ambiguous_query(query, combined_text):
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append("Ambiguity detection triggered because the query uses broad or subjective criteria.")
            return self._decision(
                "clarify",
                "ambiguous_query",
                "The query is broad or subjective and the retrieved evidence does not define a clear criterion.",
                final_score,
                trace,
            )

        if self._is_rare_entity_definition_query(query, missing_aspects):
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append(
                "Rare-entity definition query detected: treating missing term as likely retrieval mismatch."
            )
            return self._decision(
                "recover",
                "low_semantic_coverage",
                "The query is a specific definition-style request, but the key entity is missing from the retrieved evidence. "
                "This likely reflects lexical or cross-lingual mismatch rather than user ambiguity.",
                final_score,
                trace,
            )

        if missing_aspects:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append(
                "Missing aspect override triggered for: " + ", ".join(missing_aspects[:5])
            )
            return self._decision(
                "recover",
                "missing_query_aspect",
                "Retrieved evidence is missing one or more query aspects needed for a reliable answer.",
                final_score,
                trace,
            )

        if self._needs_temporal_role_support(query, query_years):
            if not self._has_temporal_role_support(docs, query_years, query):
                final_score = min(final_score, self.sufficiency_threshold - 0.01)
                trace.append("Temporal answer-support check failed: no local link between the year and the role/action.")
                return self._decision(
                    "recover",
                    "temporal_answer_not_supported",
                    "Retrieved chunks mention the query year, but they do not locally connect it to the required role or action.",
                    final_score,
                    trace,
                )
            trace.append("Temporal answer-support check passed.")

        answer_support = self._answer_support_status(query, docs)
        if not answer_support["supported"]:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append(answer_support["trace"])
            return self._decision(
                "recover",
                "answer_not_explicitly_supported",
                answer_support["reason"],
                final_score,
                trace,
            )
        trace.append(answer_support["trace"])

        if support_count < self.chunk_min_count:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append("Support-count threshold failed after answer-support checks.")
            return self._decision(
                "recover",
                "few_supporting_chunks",
                f"Only {support_count} supporting chunks found (need ≥ {self.chunk_min_count}).",
                final_score,
                trace,
            )

        if semantic_cov < self.semantic_threshold:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append("Semantic coverage threshold failed after answer-support checks.")
            return self._decision(
                "recover",
                "low_semantic_coverage",
                f"Average semantic similarity {semantic_cov:.3f} is below threshold {self.semantic_threshold:.2f}.",
                final_score,
                trace,
            )

        if base_score < self.sufficiency_threshold:
            final_score = min(final_score, self.sufficiency_threshold - 0.01)
            trace.append("Composite sufficiency score is below threshold after gating checks.")
            return self._decision(
                "recover",
                "missing_query_aspect",
                "Composite evidence quality remains below the sufficiency threshold.",
                final_score,
                trace,
            )

        trace.append("Gating layer passed: evidence appears explicitly sufficient for answer generation.")
        return self._decision(
            "answer",
            "ok",
            "Evidence is sufficient to proceed with answer generation.",
            final_score,
            trace,
        )

    def _decision(
        self,
        routing_decision: str,
        failure_type: str,
        decision_reason: str,
        score: float,
        trace: list[str],
    ) -> dict:
        return {
            "routing_decision": routing_decision,
            "failure_type": failure_type,
            "decision_reason": decision_reason,
            "decision_trace": trace,
            "score": score,
        }

    def _combined_text(self, docs: list) -> str:
        return " ".join(
            (d.metadata.get("original_text") or d.page_content or "").lower()
            for d in docs
        )

    def _is_ambiguous_query(self, query: str, combined_text: str) -> bool:
        query_lower = query.lower()
        markers = [marker for marker in _AMBIGUOUS_MARKERS if marker in query_lower]
        if not markers:
            return False
        return not any(marker in combined_text for marker in _AMBIGUITY_CRITERION_MARKERS)

    def _needs_temporal_role_support(self, query: str, query_years: list[int]) -> bool:
        if not query_years:
            return False
        query_lower = query.lower()
        return any(term in query_lower for term in _ROLE_TERMS) or query_lower.startswith("who")

    def _has_temporal_role_support(self, docs: list, query_years: list[int], query: str) -> bool:
        query_terms = self._query_support_terms(query, question_type="temporal")
        for doc in docs:
            text = (doc.metadata.get("original_text") or doc.page_content or "")
            for span in self._support_spans(text):
                span_lower = span.lower()
                if not any(str(year) in span_lower for year in query_years):
                    continue
                if any(term in span_lower for term in query_terms):
                    return True
        return False

    def _answer_support_status(self, query: str, docs: list) -> dict:
        query_lower = query.lower().strip()
        question_type = self._question_type(query_lower)
        support_terms = self._query_support_terms(query, question_type)
        spans = []
        for doc in docs:
            text = (doc.metadata.get("original_text") or doc.page_content or "")
            spans.extend(self._support_spans(text))

        if question_type == "who":
            for span in spans:
                span_lower = span.lower()
                if any(term in span_lower for term in support_terms) and self._contains_person_like_mention(span):
                    return {
                        "supported": True,
                        "reason": "Answer-bearing support detected for the who-question.",
                        "trace": "Answer-support check passed for a who-question.",
                    }
            return {
                "supported": False,
                "reason": "Retrieved chunks are topically related, but they do not explicitly support a person-bearing answer to the query.",
                "trace": "Answer-support check failed for a who-question.",
            }

        if question_type == "when":
            for span in spans:
                span_lower = span.lower()
                if any(term in span_lower for term in support_terms) and re.search(r"\b(1\d{3}|2\d{3})\b", span):
                    return {
                        "supported": True,
                        "reason": "Answer-bearing temporal support detected.",
                        "trace": "Answer-support check passed for a when-question.",
                    }
            return {
                "supported": False,
                "reason": "Retrieved chunks do not explicitly tie the relevant event or role to a time expression.",
                "trace": "Answer-support check failed for a when-question.",
            }

        if question_type == "what_definition":
            if any(any(term in span.lower() for term in support_terms) for span in spans):
                return {
                    "supported": True,
                    "reason": "Definition-like evidence is present.",
                    "trace": "Answer-support check passed for a what-question.",
                }
            return {
                "supported": False,
                "reason": "Retrieved chunks mention the topic but do not explicitly define or characterize it.",
                "trace": "Answer-support check failed for a what-question.",
            }

        return {
            "supported": True,
            "reason": "Generic answer-support check passed.",
            "trace": "Answer-support check passed for a generic query.",
        }

    def _question_type(self, query_lower: str) -> str:
        if query_lower.startswith("who"):
            return "who"
        if query_lower.startswith("when"):
            return "when"
        if query_lower.startswith("what is") or query_lower.startswith("what are"):
            return "what_definition"
        return "generic"

    def _query_support_terms(self, query: str, question_type: str) -> list[str]:
        query_lower = query.lower()
        terms = set()
        if question_type == "temporal":
            terms.update(_ROLE_TERMS)
            if "received" in query_lower or "grant" in query_lower or "erc" in query_lower:
                terms.update({"received", "awarded", "grant", "grants", "erc"})
            return sorted(terms)

        if question_type == "who":
            terms.update(term for term in _WHO_ACTION_TERMS if term in query_lower)
            terms.update(term for term in _ROLE_TERMS if term in query_lower)
            if not terms:
                terms.update({"received", "awarded", "president", "rector"})
            return sorted(terms)

        if question_type == "when":
            terms.update(term for term in _ROLE_TERMS if term in query_lower)
            terms.update(term for term in {"received", "awarded", "founded", "started"} if term in query_lower)
            return sorted(terms or {"date", "year"})

        if question_type == "what_definition":
            aspects = self._extract_aspects(query)
            return aspects[:4]

        return self._extract_aspects(query)[:4]

    def _is_rare_entity_definition_query(self, query: str, missing_aspects: list[str]) -> bool:
        """
        Heuristic for definition queries like "What is e-Sling?" where the
        missing term is likely a lexical / cross-lingual retrieval problem.
        """
        query_lower = query.lower().strip()
        if self._question_type(query_lower) != "what_definition":
            return False
        if len(missing_aspects) != 1:
            return False

        missing = missing_aspects[0]
        query_surface = query.strip(" ?.")

        # Preserve the specific "what is X" pattern and favor recovery when X
        # looks like a named or hyphenated entity rather than a generic concept.
        if missing in {"famous", "best", "important", "top", "main"}:
            return False

        if "-" in missing or any(ch.isdigit() for ch in missing):
            return True

        # Look for a compact surface form after "what is"/"what are".
        surface_match = re.match(r"what\s+(?:is|are)\s+(.+)$", query_surface, re.IGNORECASE)
        if not surface_match:
            return False
        subject = surface_match.group(1).strip()
        subject_tokens = [tok.strip("?.,;:") for tok in subject.split() if tok.strip("?.,;:")]

        if len(subject_tokens) <= 2:
            return any(any(ch.isupper() for ch in token[1:]) for token in subject_tokens)

        return False

    def _support_spans(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _contains_person_like_mention(self, text: str) -> bool:
        return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
