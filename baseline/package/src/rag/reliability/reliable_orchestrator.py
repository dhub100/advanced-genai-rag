"""
Reliable Orchestrator — policy controller for the reliability-aware pipeline.

Implements a two-gate architecture:

    Query
      → [1] Retrieve (existing strategies)
      → [2] EvidenceAnalyst (A)  ← PRE-SYNTHESIS GATE
            ├─ sufficient  → [3] Synthesize
            │                     → GroundednessVerifier (B)
            │                     → TrustScorer (H)  ← POST-SYNTHESIS GATE
            │                           ├─ trust OK  → Final Response
            │                           └─ trust low → [5] AbstentionGate (E) [trust_failure]
            └─ insufficient → [4] RecoveryAgent (G)
                                   ├─ action chosen → retry from [1]
                                   └─ exhausted → [5] AbstentionGate (E) [evidence_failure]

AbstentionGate (E) is the single terminal failure state, reachable from both gates.

Integration points for teammates:
    - GroundednessVerifier (B): implement check(query, answer, docs) → float
    - TrustScorer (H): implement score(evidence_report, groundedness_score) → float
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.reliability.abstention import AbstentionMechanism, AbstentionResponse
from rag.reliability.evidence_sufficiency import EvidenceReport, EvidenceSufficiencyChecker
from rag.reliability.recovery import RecoveryAgent

_TRUST_THRESHOLD = 0.45   # H routes to abstention below this value


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

    # Set when abstained=True; carries trigger, reason, and trace
    abstention: AbstentionResponse | None = None

    # Populated after the post-synthesis path (B → H)
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
        self._recovery = RecoveryAgent(max_retries=max_retries, openai_client=openai_client)
        # Teammate components — injected when available, skipped gracefully if None
        self._grounder = groundedness_verifier   # B: check(query, answer, docs) → float
        self._trust = trust_scorer               # H: score(evidence_report, groundedness) → float

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

                # [4] Post-synthesis gate: B → H
                groundedness = self._run_grounder(current_query, answer, docs, trace)
                trust = self._run_trust(report, groundedness, trace)

                # H gates exit: route to abstention if trust is too low
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

        # [5] Abstain — evidence failure (G exhausted)
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

    # ------------------------------------------------------------------
    # Teammate integration helpers (graceful no-ops when not yet injected)
    # ------------------------------------------------------------------

    def _run_grounder(self, query: str, answer: str, docs: list, trace: list) -> float | None:
        """Call B (GroundednessVerifier) if available; return None otherwise."""
        if self._grounder is None:
            return None
        score = self._grounder.check(query, answer, docs)
        trace.append(f"GroundednessVerifier [B]: groundedness_score={score:.3f}")
        return score

    def _run_trust(self, report: EvidenceReport, groundedness: float | None, trace: list) -> float | None:
        """Call H (TrustScorer) if available; return None otherwise."""
        if self._trust is None:
            return None
        score = self._trust.score(report, groundedness)
        trace.append(f"TrustScorer [H]: trust_score={score:.3f}")
        return score
