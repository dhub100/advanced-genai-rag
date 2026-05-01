"""Module that implements the TrustScorer.
This class combine the evidence report of the Mechanism A (Evidence Analyste, and the Mechanism B (GroundedVerifier).

During the reliability analysis, it has been shown that a high self-sufficiency score doesn't necessarily implies grounded retrieved docugrounded retrieved documents.s."""


class TrustScorerStub:
    """Stub - raises NotImplementedError when called."""

    def score(self, evidence_report, groundedness_score) -> float:
        raise NotImplementedError("Mechanism H not yet implemented.")
