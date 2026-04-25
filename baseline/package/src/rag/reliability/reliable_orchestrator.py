"""
Reliable Orchestrator — policy controller for the reliability-aware pipeline.

Wraps the existing Orchestrator and inserts the reliability loop between
retrieval and answer synthesis:

    Query
      → [1] Retrieve (existing strategies)
      → [2] EvidenceSufficiencyChecker (A)
            ├─ sufficient  → [3] Synthesize → GroundednessVerifier (B) → TrustScorer (H)
            └─ insufficient → [4] RecoveryAgent (G)
                                   ├─ action chosen → retry from [1] with new query/strategy
                                   └─ exhausted → [5] AbstentionMechanism (E)

Integration points for teammates:
    - GroundednessVerifier (B): receives (query, answer, docs) from step [3]
    - TrustScorer (H): receives (evidence_report, groundedness_score, answer) from step [3/5]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.reliability.abstention import AbstentionMechanism, AbstentionResponse
from rag.reliability.evidence_sufficiency import EvidenceReport, EvidenceSufficiencyChecker
from rag.reliability.recovery import RecoveryAgent


@dataclass
class ReliableResponse:
    """
    Unified response object returned by ReliableOrchestrator.

    Either contains a generated answer (abstained=False) or an abstention
    (abstained=True, answer="").

    Consumed by:
        - GroundednessVerifier (B): passes answer + docs for grounding check
        - TrustScorer (H): uses evidence_report.score + groundedness_score
        - Evaluation: checks abstained flag against benchmark ground truth
    """

    query: str
    strategy: str
    answer: str
    abstained: bool
    evidence_report: EvidenceReport | None
    documents: list = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    recovery_attempts: int = 0

    # Set by AbstentionMechanism when abstained=True
    abstention: AbstentionResponse | None = None

    # Placeholders for teammate outputs (B and H fill these in)
    groundedness_score: float | None = None
    trust_score: float | None = None


class ReliableOrchestrator:
    """
    Policy controller that adds evidence-based reliability on top of the
    existing Orchestrator.

    Args:
        orchestrator:   the existing rag.retrieval.orchestrator.Orchestrator instance.
        embed_fn:       encoding function from the Dense retriever (reused for A).
        max_retries:    maximum recovery attempts before abstaining (default 2).
        openai_client:  optional OpenAI client for LLM query rewriting in G.
    """

    def __init__(self, orchestrator, embed_fn, max_retries: int = 2, openai_client=None):
        self._orchestrator = orchestrator
        self._checker = EvidenceSufficiencyChecker(embed_fn)
        self._abstainer = AbstentionMechanism()
        self._recovery = RecoveryAgent(max_retries=max_retries, openai_client=openai_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        strategy: str = "confidence",
        top_k: int = 5,
    ) -> ReliableResponse:
        """
        Run the full reliability-aware pipeline for a query.

        Returns a ReliableResponse containing either a grounded answer or
        a structured abstention.
        """
        trace: list[str] = []
        current_query = query
        current_strategy = strategy
        use_prf = False
        recovery_attempts = 0

        for attempt in range(self._recovery.max_retries + 1):
            # [1] Retrieve
            retrieval_result = self._orchestrator.run(
                strategy=current_strategy,
                query=current_query,
                top_k=top_k,
            )
            docs = retrieval_result["documents"]
            trace.extend(retrieval_result.get("trace", []))
            trace.append(
                f"Attempt {attempt}: retrieved {len(docs)} docs "
                f"(strategy='{current_strategy}', query='{current_query[:60]}')"
            )

            # [2] Evidence sufficiency check (A)
            report = self._checker.check(current_query, docs)
            trace.append(
                f"EvidenceCheck: score={report.score:.3f}, sufficient={report.sufficient}, "
                f"failure_type='{report.failure_type}'"
            )

            # Hard floor → skip recovery
            if report.below_hard_floor:
                trace.append(f"Hard floor hit (score={report.score:.3f}) → abstaining immediately.")
                abstention = self._abstainer.abstain(query, report, trace)
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

            if report.sufficient:
                # [3] Synthesize
                answer = self._orchestrator.synthesizer.generate(current_query, docs)
                trace.append("Answer synthesized successfully.")
                return ReliableResponse(
                    query=query,
                    strategy=current_strategy,
                    answer=answer,
                    abstained=False,
                    evidence_report=report,
                    documents=docs,
                    trace=trace,
                    recovery_attempts=recovery_attempts,
                )

            # [4] Recovery (G)
            action = self._recovery.choose_action(report, attempt, current_strategy)
            trace.append(action.trace_entry)

            if action.type == "abstain":
                break

            recovery_attempts += 1

            # Apply action: update query / strategy / PRF flag
            if action.type == "rewrite_query":
                current_query = self._recovery.rewrite_query(current_query, report.missing_aspects)
                trace.append(f"Query rewritten to: '{current_query[:80]}'")

            elif action.type == "switch_strategy":
                current_strategy = action.new_strategy

            elif action.type == "prf_expansion":
                use_prf = True
                # PRF is handled inside BM25 via QEBM25; signal to orchestrator here
                # (full integration with QEBM25 swap happens in Step 3 implementation)
                trace.append("PRF expansion enabled for next retrieval attempt.")

        # [5] Abstain
        abstention = self._abstainer.abstain(query, report, trace)
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
