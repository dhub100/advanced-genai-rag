"""
Recovery Mechanism (Mechanism G) — the ADAPTIVE requirement.

When evidence sufficiency fails, RecoveryAgent chooses a recovery action
that *changes the orchestration behavior* of the system based on which
signal was weakest. This is not just analysis — it dynamically routes
the next retrieval attempt differently.

Recovery policy (maps failure_type → action):
    "coverage_low"    → rewrite the query to be more specific (LLM-based)
    "few_docs"        → switch retrieval strategy
    "low_similarity"  → switch to PRF-expanded BM25 (QEBM25)
    "exhausted"       → signal abstention

Each action is logged to the trace for interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.reliability.evidence_sufficiency import EvidenceReport

# Strategy rotation order when switching due to "few_docs"
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

    Fields:
        type:          "rewrite_query" | "switch_strategy" | "prf_expansion" | "abstain"
        new_query:     rewritten query (set when type == "rewrite_query")
        new_strategy:  strategy to use next (set when type == "switch_strategy")
        use_prf:       if True, the orchestrator should use QEBM25 (PRF variant)
        trace_entry:   human-readable log entry for this decision
    """

    type: str
    new_query: str = ""
    new_strategy: str = ""
    use_prf: bool = False
    trace_entry: str = ""

    def apply(self, current_query: str, current_strategy: str) -> tuple[str, str, bool]:
        """
        Return (next_query, next_strategy, use_prf) after applying this action.
        Falls back to current values for fields not set by this action type.
        """
        q = self.new_query if self.new_query else current_query
        s = self.new_strategy if self.new_strategy else current_strategy
        return q, s, self.use_prf


class RecoveryAgent:
    """
    Adaptive recovery controller.

    Chooses a different recovery action on each attempt so that consecutive
    failures explore orthogonal strategies rather than repeating the same fix.

    Args:
        max_retries:    maximum number of recovery attempts before abstaining.
        openai_client:  optional OpenAI client for LLM-based query rewriting.
                        If None, a rule-based fallback is used for rewrites.
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
        """
        Select the recovery action for this attempt given the EvidenceReport.

        Args:
            report:           EvidenceReport from the last sufficiency check.
            attempt:          0-indexed attempt counter (0 = first recovery).
            current_strategy: the retrieval strategy used in the failed attempt.

        Returns:
            RecoveryAction describing what to do next.
        """
        if attempt >= self.max_retries:
            return RecoveryAction(
                type="abstain",
                trace_entry=f"Recovery exhausted after {attempt} attempt(s) → abstaining.",
            )

        failure = report.failure_type

        if failure == "coverage_low":
            return self._rewrite_action(report, attempt)

        if failure == "few_docs":
            return self._switch_strategy_action(current_strategy, attempt)

        if failure == "low_similarity":
            return self._prf_expansion_action(attempt)

        # Fallback: cycle through all strategies
        return self._switch_strategy_action(current_strategy, attempt)

    # ------------------------------------------------------------------
    # Action builders
    # ------------------------------------------------------------------

    def _rewrite_action(self, report: EvidenceReport, attempt: int) -> RecoveryAction:
        """Rewrite the query to improve aspect coverage."""
        # LLM-based rewrite if client available, else rule-based fallback
        # (actual rewrite happens in ReliableOrchestrator where the query is accessible)
        entry = (
            f"Recovery attempt {attempt + 1}: coverage_low → query rewrite scheduled "
            f"(missing aspects: {', '.join(report.missing_aspects[:3]) or 'unknown'})."
        )
        return RecoveryAction(type="rewrite_query", trace_entry=entry)

    def _switch_strategy_action(self, current_strategy: str, attempt: int) -> RecoveryAction:
        """Switch to the next retrieval strategy in the fallback chain."""
        next_strategy = _STRATEGY_FALLBACKS.get(current_strategy, "voting")
        entry = (
            f"Recovery attempt {attempt + 1}: few_docs → switching strategy "
            f"from '{current_strategy}' to '{next_strategy}'."
        )
        return RecoveryAction(type="switch_strategy", new_strategy=next_strategy, trace_entry=entry)

    def _prf_expansion_action(self, attempt: int) -> RecoveryAction:
        """Enable PRF (pseudo-relevance feedback) BM25 expansion."""
        entry = (
            f"Recovery attempt {attempt + 1}: low_similarity → enabling PRF query expansion (QEBM25)."
        )
        return RecoveryAction(type="prf_expansion", use_prf=True, trace_entry=entry)

    # ------------------------------------------------------------------
    # LLM-based query rewriter (used by ReliableOrchestrator)
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, missing_aspects: list[str]) -> str:
        """
        Rewrite query to be more specific, addressing missing aspects.

        Uses OpenAI if available; falls back to a simple rule-based expansion.
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
        """Simple fallback: append missing aspects as a qualifier."""
        if not missing_aspects:
            return query
        qualifier = " ".join(missing_aspects[:2])
        return f"{query} {qualifier}".strip()
