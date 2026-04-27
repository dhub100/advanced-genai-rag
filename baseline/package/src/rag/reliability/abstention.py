"""
Abstention Mechanism (Mechanism E).

Unified terminal failure handler. Produces a structured AbstentionResponse
regardless of where in the pipeline the failure originated.

Two trigger paths:
    1. Evidence failure  — EvidenceReport.sufficient is False and recovery is
                           exhausted, or score is below the hard floor.
                           Called by ReliableOrchestrator via abstain_evidence().
    2. Trust failure     — TrustScorer (H) computed a trust_score below the
                           acceptance threshold after synthesis.
                           Called by ReliableOrchestrator via abstain_low_trust().

Abstention is always linked to explicit signal values, never used arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.reliability.evidence_sufficiency import EvidenceReport


@dataclass
class AbstentionResponse:
    """
    Returned instead of an answer when the system decides to abstain.

    Fields:
        trigger:        "evidence_failure" | "trust_failure"
        evidence_score: sufficiency_score at time of abstention (0.0 if trust failure)
        trust_score:    trust_score at time of abstention (None if evidence failure)
    """

    abstained: bool = True
    trigger: str = "evidence_failure"   # "evidence_failure" | "trust_failure"
    reason: str = ""
    missing_aspects: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[str] = field(default_factory=list)
    evidence_score: float = 0.0
    trust_score: float | None = None


class AbstentionMechanism:
    """
    Unified abstention handler (Mechanism E).

    Reached from two paths in the pipeline:
        1. abstain_evidence() — retrieval failure (A → G exhausted, or hard floor)
        2. abstain_low_trust() — synthesis failure (B → H below threshold)
    """

    def abstain_evidence(
        self,
        query: str,
        report: EvidenceReport,
        trace: list[str] | None = None,
    ) -> AbstentionResponse:
        """
        Abstain because retrieved evidence was insufficient.

        Args:
            query:  the original user query.
            report: EvidenceReport from the final sufficiency check.
            trace:  accumulated orchestration trace entries so far.
        """
        trace = list(trace or [])
        reason = self._evidence_reason(query, report)
        trace.append(f"AbstentionGate [evidence_failure]: {report.reason}")

        return AbstentionResponse(
            abstained=True,
            trigger="evidence_failure",
            reason=reason,
            missing_aspects=report.missing_aspects,
            confidence=0.0,
            trace=trace,
            evidence_score=report.score,
            trust_score=None,
        )

    def abstain_low_trust(
        self,
        query: str,
        trust_score: float,
        groundedness_score: float,
        trace: list[str] | None = None,
    ) -> AbstentionResponse:
        """
        Abstain because the generated answer did not meet the trust threshold.

        Args:
            query:              the original user query.
            trust_score:        combined trust score from TrustScorer (H).
            groundedness_score: groundedness score from GroundednessVerifier (B).
            trace:              accumulated orchestration trace entries so far.
        """
        trace = list(trace or [])
        reason = (
            f"The system generated an answer for '{query}' but could not "
            f"sufficiently verify that it is grounded in the retrieved evidence "
            f"(trust_score={trust_score:.3f}, groundedness={groundedness_score:.3f})."
        )
        trace.append(
            f"AbstentionGate [trust_failure]: trust_score={trust_score:.3f} "
            f"below threshold — answer suppressed."
        )

        return AbstentionResponse(
            abstained=True,
            trigger="trust_failure",
            reason=reason,
            missing_aspects=[],
            confidence=0.0,
            trace=trace,
            evidence_score=0.0,
            trust_score=trust_score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evidence_reason(self, query: str, report: EvidenceReport) -> str:
        base = (
            f"The system could not find sufficient evidence to answer: '{query}'. "
            f"{report.reason}"
        )
        if report.missing_aspects:
            aspects = ", ".join(f"'{a}'" for a in report.missing_aspects[:5])
            base += f" Missing information about: {aspects}."
        return base
