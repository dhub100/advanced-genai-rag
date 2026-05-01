"""
Reliable Orchestrator — policy controller for the reliability-aware pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.reliability.abstention import AbstentionMechanism, AbstentionResponse
from rag.reliability.evidence_sufficiency import EvidenceReport, EvidenceSufficiencyChecker
from rag.reliability.recovery import RecoveryAgent
from rag.reliability.groundedness import GroundednessVerifier

_TRUST_THRESHOLD = 0.45


@dataclass
class ReliableResponse:
    """
    Unified response object returned by ReliableOrchestrator.
    """

    query: str
    strategy: str
    answer: str
    abstained: bool
    evidence_report: EvidenceReport | None
    documents: list = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    recovery_attempts: int = 0
    abstention: AbstentionResponse | None = None
    groundedness_score: float | None = None
    trust_score: float | None = None


class ReliableOrchestrator:
    """
    Policy controller that adds evidence-based reliability on top of the
    existing Orchestrator.
    """

    def __init__(
        self,
        orchestrator,
        embed_fn,
        max_retries: int = 2,
        openai_client=None,
        groundedness_verifier=None,
        trust_scorer=None,
    ):
        self._orchestrator = orchestrator
        self._checker = EvidenceSufficiencyChecker(embed_fn)
        self._abstainer = AbstentionMechanism()
        self._recovery = RecoveryAgent(
            max_retries=max_retries,
            openai_client=openai_client
        )
        self._grounder = groundedness_verifier
        self._trust = trust_scorer

    def run(
        self,
        query: str,
        strategy: str = "confidence",
        top_k: int = 5,
    ) -> ReliableResponse:
        """
        Run the full reliability-aware pipeline for a query.
        """
        trace: list[str] = []
        current_query = query
        current_strategy = strategy
        use_prf = False
        recovery_attempts = 0

        for attempt in range(self._recovery.max_retries + 1):
            retrieval_result = self._orchestrator.run(
                strategy=current_strategy,
                query=current_query,
                top_k=top_k,
                use_prf=use_prf,
            )
            docs = retrieval_result["documents"]
            trace.extend(retrieval_result.get("trace", []))
            trace.append(
                f"Attempt {attempt}: retrieved {len(docs)} docs "
                f"(strategy='{current_strategy}', query='{current_query[:60]}', use_prf={use_prf})"
            )

            report = self._checker.check(current_query, docs)
            trace.append(
                f"EvidenceCheck: score={report.score:.3f}, sufficient={report.sufficient}, "
                f"failure_type='{report.failure_type}', route='{report.routing_decision}'"
            )
            trace.extend(f"[A] {entry}" for entry in report.decision_trace)

            if report.routing_decision == "abstain" or report.below_hard_floor:
                trace.append(f"Routing decision is '{report.routing_decision}' → abstaining immediately.")
                abstention = self._abstainer.abstain_evidence(query, report, trace)
                return ReliableResponse(
                    query=query,
                    strategy=current_strategy,
                    answer="",
                    abstained=True,
                    evidence_report=report,
                    documents=docs,
                    trace=trace,
                    recovery_attempts=recovery_attempts,
                    abstention=abstention,
                )

            if report.routing_decision == "clarify":
                trace.append("Routing decision is 'clarify' → clarification required; abstaining until clarified.")
                abstention = self._abstainer.abstain_evidence(query, report, trace)
                return ReliableResponse(
                    query=query,
                    strategy=current_strategy,
                    answer="",
                    abstained=True,
                    evidence_report=report,
                    documents=docs,
                    trace=trace,
                    recovery_attempts=recovery_attempts,
                    abstention=abstention,
                )

            if report.routing_decision == "answer":
                answer = self._orchestrator.synthesizer.generate(current_query, docs)
                trace.append("Answer synthesized successfully.")

                groundedness = self._run_grounder(current_query, answer, docs, trace)
                trust = self._run_trust(report, groundedness, trace)

                if trust is not None and trust < _TRUST_THRESHOLD:
                    abstention = self._abstainer.abstain_low_trust(
                        query, trust, groundedness or 0.0, trace
                    )
                    return ReliableResponse(
                        query=query,
                        strategy=current_strategy,
                        answer="",
                        abstained=True,
                        evidence_report=report,
                        documents=docs,
                        trace=trace,
                        recovery_attempts=recovery_attempts,
                        groundedness_score=groundedness,
                        trust_score=trust,
                        abstention=abstention,
                    )

                return ReliableResponse(
                    query=query,
                    strategy=current_strategy,
                    answer=answer,
                    abstained=False,
                    evidence_report=report,
                    documents=docs,
                    trace=trace,
                    recovery_attempts=recovery_attempts,
                    groundedness_score=groundedness,
                    trust_score=trust,
                )

            action = self._recovery.choose_action(report, attempt, current_strategy)
            trace.append(action.trace_entry)

            if action.type == "clarify":
                trace.append(
                    "Recovery exhausted for rare specific entity → clarification required before answering."
                )
                return ReliableResponse(
                    query=query,
                    strategy=current_strategy,
                    answer="",
                    abstained=True,
                    evidence_report=report,
                    documents=docs,
                    trace=trace,
                    recovery_attempts=recovery_attempts,
                    abstention=self._abstainer.abstain_evidence(query, report, trace),
                )

            if action.type == "abstain":
                break

            recovery_attempts += 1

            if action.type == "rewrite_query":
                current_query = self._recovery.rewrite_query(current_query, report.missing_aspects)
                trace.append(f"Query rewritten to: '{current_query[:80]}'")
            elif action.type == "switch_strategy":
                current_strategy = action.new_strategy
            elif action.type == "prf_expansion":
                use_prf = True
                trace.append("PRF expansion enabled for next retrieval attempt.")

        abstention = self._abstainer.abstain_evidence(query, report, trace)
        return ReliableResponse(
            query=query,
            strategy=current_strategy,
            answer="",
            abstained=True,
            evidence_report=report,
            documents=docs,
            trace=trace,
            recovery_attempts=recovery_attempts,
            abstention=abstention,
        )

    def _run_grounder(self, query: str, answer: str, docs: list, trace: list) -> float | None:
        if self._grounder is None:
            return None
        score = self._grounder.check(answer, docs)
        trace.append(f"GroundednessVerifier [B]: groundedness_score={score:.3f}")
        return score

    def _run_trust(self, report: EvidenceReport, groundedness: float | None, trace: list) -> float | None:
        if self._trust is None:
            return None
        score = self._trust.score(report, groundedness)
        trace.append(f"TrustScorer [H]: trust_score={score:.3f}")
        return score
