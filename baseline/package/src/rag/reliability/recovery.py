"""
Recovery Mechanism (Mechanism G) — the ADAPTIVE requirement.

When evidence sufficiency fails, RecoveryAgent chooses a recovery action
that changes the orchestration behavior of the system based on which
signal was weakest.

Recovery policy (maps failure_type → action):
    "missing_query_aspect" / "missing_critical_aspect" → rewrite the query
    "few_supporting_chunks"                           → switch retrieval strategy
    "low_semantic_coverage"                          → switch to PRF-expanded BM25
    "temporal_answer_not_supported"                  → rewrite the query
    "answer_not_explicitly_supported"                → rewrite the query
    "temporal_mismatch"                              → rewrite the query
    "exhausted"                                      → signal abstention
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.reliability.evidence_sufficiency import EvidenceReport

_STRATEGY_FALLBACKS: dict[str, str] = {
    "confidence": "voting",
    "voting": "waterfall",
    "waterfall": "confidence",
}

_MAX_RETRIES_DEFAULT = 2


@dataclass
class RecoveryAction:
    """
    Describes the action chosen by RecoveryAgent for a given failure.
    """

    type: str
    new_query: str = ""
    new_strategy: str = ""
    use_prf: bool = False
    trace_entry: str = ""

    def apply(self, current_query: str, current_strategy: str) -> tuple[str, str, bool]:
        q = self.new_query if self.new_query else current_query
        s = self.new_strategy if self.new_strategy else current_strategy
        return q, s, self.use_prf


class RecoveryAgent:
    """
    Adaptive recovery controller.
    """

    def __init__(self, max_retries: int = _MAX_RETRIES_DEFAULT, openai_client=None):
        self.max_retries = max_retries
        self._openai = openai_client

    def choose_action(
        self,
        report: EvidenceReport,
        attempt: int,
        current_strategy: str = "confidence",
    ) -> RecoveryAction:
        if attempt >= self.max_retries:
            return RecoveryAction(
                type="abstain",
                trace_entry=f"Recovery exhausted after {attempt} attempt(s) → abstaining.",
            )

        failure = report.failure_type

        if failure in {
            "missing_query_aspect",
            "missing_critical_aspect",
            "temporal_answer_not_supported",
            "answer_not_explicitly_supported",
            "temporal_mismatch",
        }:
            return self._rewrite_action(report, attempt)

        if failure == "few_supporting_chunks":
            return self._switch_strategy_action(current_strategy, attempt)

        if failure == "low_semantic_coverage":
            return self._prf_expansion_action(attempt)

        return self._switch_strategy_action(current_strategy, attempt)

    def _rewrite_action(self, report: EvidenceReport, attempt: int) -> RecoveryAction:
        entry = (
            f"Recovery attempt {attempt + 1}: {report.failure_type} → query rewrite scheduled "
            f"(missing aspects: {', '.join(report.missing_aspects[:3]) or 'unknown'})."
        )
        return RecoveryAction(type="rewrite_query", trace_entry=entry)

    def _switch_strategy_action(self, current_strategy: str, attempt: int) -> RecoveryAction:
        next_strategy = _STRATEGY_FALLBACKS.get(current_strategy, "voting")
        entry = (
            f"Recovery attempt {attempt + 1}: few_supporting_chunks → switching strategy "
            f"from '{current_strategy}' to '{next_strategy}'."
        )
        return RecoveryAction(type="switch_strategy", new_strategy=next_strategy, trace_entry=entry)

    def _prf_expansion_action(self, attempt: int) -> RecoveryAction:
        entry = (
            f"Recovery attempt {attempt + 1}: low_semantic_coverage → enabling PRF query expansion (QEBM25)."
        )
        return RecoveryAction(type="prf_expansion", use_prf=True, trace_entry=entry)

    def rewrite_query(self, query: str, missing_aspects: list[str]) -> str:
        """
        Rewrite query to be more specific, addressing missing aspects.
        """
        if self._openai and missing_aspects:
            return self._llm_rewrite(query, missing_aspects)
        return self._rule_rewrite(query, missing_aspects)

    def _llm_rewrite(self, query: str, missing_aspects: list[str]) -> str:
        aspects_str = ", ".join(missing_aspects[:3])
        prompt = (
            f"Rewrite the following question to be more specific and searchable, "
            f"making sure to include information about: {aspects_str}.\n\n"
            f"Original question: {query}\n\nRewritten question:"
        )
        response = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _rule_rewrite(self, query: str, missing_aspects: list[str]) -> str:
        if not missing_aspects:
            return query
        qualifier = " ".join(missing_aspects[:2])
        return f"{query} {qualifier}".strip()
