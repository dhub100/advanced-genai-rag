"""Module that implements the TrustScorer (Mechanism H).

Combines the evidence report from Mechanism A (EvidenceAnalyst) and the
groundedness score from Mechanism B (GroundednessVerifier) into a single
trust score that acts as the final quality gate before returning an answer.
"""

from rag.reliability.evidence_sufficiency import EvidenceReport


class TrustScorer:
    """
    Trust Scorer (Mechanism H).

    Aggregates evidence sufficiency (A) and groundedness (B) into a
    weighted trust score. The score determines whether the generated
    answer is returned or withheld via the AbstentionGate (E).

    Trust score formula:
        trust_score = 0.4 * sufficiency_score + 0.6 * groundedness_score
    """

    def score(self, evidence_report: EvidenceReport, groundedness_score: float) -> float:
        return 0.4 * evidence_report.sufficiency_score + 0.6 * groundedness_score
