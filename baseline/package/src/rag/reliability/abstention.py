"""
Abstention Mechanism (Mechanism E).

Produces a structured AbstentionResponse when the system cannot reliably
answer the query — either because evidence was insufficient and all recovery
attempts failed, or because the sufficiency score is below the hard floor.

Abstention is always linked to explicit signals (EvidenceReport), never
used arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.reliability.evidence_sufficiency import EvidenceReport


@dataclass
class AbstentionResponse:
    """
    Returned instead of an answer when the system decides to abstain.

    Consumed by:
        - TrustScorer (H): sets trust_score = 0.0 when abstained = True
        - Evaluation: counted as correct abstention or false abstention
    """

    abstained: bool = True
    reason: str = ""
    missing_aspects: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[str] = field(default_factory=list)
    evidence_score: float = 0.0   # the final sufficiency_score for logging


class AbstentionMechanism:
    """
    Decides to abstain and constructs the AbstentionResponse.

    Triggered by ReliableOrchestrator when:
        1. EvidenceReport.sufficient is False AND recovery attempts are exhausted.
        2. EvidenceReport.score < hard floor (skip recovery entirely).
    """

    def abstain(self, query: str, report: EvidenceReport, trace: list[str] | None = None) -> AbstentionResponse:
        """
        Produce an AbstentionResponse for the given query and evidence report.

        Args:
            query:  the original user query.
            report: EvidenceReport from the final sufficiency check.
            trace:  accumulated orchestration trace entries so far.

        Returns:
            AbstentionResponse with populated reason and missing_aspects.
        """
        trace = list(trace or [])
        reason = self._build_reason(query, report)
        trace.append(f"Abstention triggered: {report.reason}")

        return AbstentionResponse(
            abstained=True,
            reason=reason,
            missing_aspects=report.missing_aspects,
            confidence=0.0,
            trace=trace,
            evidence_score=report.score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reason(self, query: str, report: EvidenceReport) -> str:
        base = (
            f"The system could not find sufficient evidence to answer: '{query}'. "
            f"{report.reason}"
        )
        if report.missing_aspects:
            aspects = ", ".join(f"'{a}'" for a in report.missing_aspects[:5])
            base += f" Missing information about: {aspects}."
        return base
