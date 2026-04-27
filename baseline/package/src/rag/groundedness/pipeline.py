"""Groundedness evaluation pipeline.

Implements the Answer-to-Source Alignment strategy that decomposes a generated
answer into individual claims, matches each claim to relevant source document
chunks, measures entailment with a zero-shot NLI model, aggregates the scores
into a single decision, and produces an interpretable trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import spacy
from langchain_core.documents import Document
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class GroundednessDecision(str, Enum):
    """Final groundedness verdict for an answer."""

    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"


@dataclass
class EntailmentScores:
    """Zero-shot NLI label probabilities for a single claim–chunk pair."""

    claim: str
    chunk_ids: list[str]
    chunk_texts: list[str]
    entailment: list[float]
    neutral: list[float]
    contradiction: list[float]


@dataclass
class ClaimVerdict:
    """Per-claim groundedness information exposed in the trace."""

    claim: str
    matched_chunks: List[Tuple[str, str]]  # (chunk_id, chunk_text)
    entailment_scores: List[EntailmentScores]
    status: str  # e.g. "supported", "contradicted", "unsupported"


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

# TODO: Do we need this really? Can't we just load the stuff using the default:
# nlp = spacy.load("en_core_web_lg")
_nlp: spacy.Language | None = None


def _get_spacy() -> spacy.Language:
    """Lazy-load the spaCy English sentence splitter."""
    global _nlp
    if _nlp is None:
        # In production this would be a real model, e.g. "en_core_web_sm"
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
    return _nlp


# ===========================================================================
# Step 1 : Answer Decomposition into Claims
# ===========================================================================

def decompose_answer_into_claims(answer: str) -> List[str]:
    """Use `spacy` to split a generated answer into individual,
    verifiable claims.

    Parameters
    ----------
    answer : str
        The raw generated answer produced by the downstream LLM.

    Returns
    -------
    List[str]
        A list of self-contained claim sentences extracted from *answer*.
    """
    nlp = _get_spacy()
    doc = nlp(answer)
    return [
        sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10
    ]


# ===========================================================================
# Step 2 : Claim-to-Relevance-Span Matching
# ===========================================================================

# TODO: the relevance score should be a semantic search between
# the claim and all chunks.
#   * Claim embedding <- should be computed.
#   * Chunks embedding <- vectordb using the chunk_id
def match_claim_to_relevance_spans(
    claim: str,
    retrieved_chunks: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """Associate a decomposed claim with the most relevant source text spans.

    The function consumes the *k* document chunks already retrieved by the
    upstream retrieval pipeline (BM25, dense, or graph) and returns the subset
    deemed most relevant to the individual *claim*.

    Parameters
    ----------
    claim : str
        A single verifiable claim produced by Step 1.
    retrieved_chunks : List[Document]
        Pool of candidate chunks output by the heterogeneous retriever backends.
    top_k : int, optional
        Maximum number of chunks to keep per claim, by default 5.

    Returns
    -------
    List[Document]
        The *top_k* chunks best matching the claim according to a lightweight
        overlap heuristic (placeholder for a dedicated re-ranker).
    """
    claim_tokens = set(claim.lower().split())
    scored: List[Tuple[int, Document]] = []
    for chunk in retrieved_chunks:
        text = chunk.metadata.get("original_text") or chunk.page_content
        score = len(claim_tokens & set(text.lower().split()))
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]

# ===========================================================================
# Step 3 : Entailment Measurement
# ===========================================================================


def measure_entailment(
    claim: str,
    chunks: List[Document],
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
) -> EntailmentScores:
    """Quantify how strongly a claim is supported by a source chunk.

    Uses the NLI model (`model_name`)
    to compute the textual entailment relationship between
    **chunk** (premise) and *claim* (hypothesis)

    Parameters
    ----------
    claim : str
        The verifiable claim to evaluate.
    chunk : Documents
        relevant sources chunk matched in Step 2.

    Returns
    -------
    EntailmentScores
        Structured probabilities for the three NLI labels:
        *entailment*, *neutral*, and *contradiction*.
    """
    chunk_ids = [
        str(chunk.metadata.get("chunk_id"))
        or str(chunk.metadata.get("record_id"))
        or "unknown"
        for chunk in chunks
    ]
    chunk_texts = [
        chunk.metadata.get("original_text")
        or chunk.page_content
        for chunk in chunks
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    results = {
        "entailment": [],
        "neutral": [],
        "contradiction": [],
    }
    hypothesis = claim
    for text in chunk_texts:
        # premise -> hypothesis inference
        premise = text
        input = tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        )
        # As per the model HF page.
        output = model(input["input_ids"].to(device))
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {
            name: round(float(pred) * 100, 1)
            for pred, name in zip(prediction, label_names)
        }
        results["entailment"].append(prediction["entailment"])
        results["neutral"].append(prediction["neutral"])
        results["contradiction"].append(prediction["contradiction"])

    return EntailmentScores(
        claim=claim,
        chunk_ids=chunk_ids,
        chunk_text=chunk_texts,
        entailment=results["entailment"],
        neutral=results["neutral"],
        contradiction=results["contradiction"],
    )


# ===========================================================================
# Step 4 : Decision Logic
# ===========================================================================

def aggregate_groundedness(
    per_claim_best_scores: List[float],
    strict: bool = True,
) -> GroundednessDecision:
    """Aggregate individual claim–chunk entailment scores into a decision.
    Use the top-k (step 1) scores computed by the NLI model (step 2).

    Parameters
    ----------
    per_claim_best_scores : List[float] ('EntailmentScores.entailment')
        For each claim, the highest entailment probability observed across
        all matched chunks.
    strict : bool, optional
        When ``True`` a single unsupported claim downgrades the whole answer
        to *ungrounded*; when ``False`` a majority vote is used.

    Returns
    -------
    GroundednessDecision
        One of ``GROUNDED``, ``PARTIALLY_GROUNDED``, or ``UNGROUNDED``.
    """
    if not per_claim_best_scores:
        return GroundednessDecision.UNGROUNDED

    # Look at entailment probabilies instead of class probabilities
    if strict:
        min_score = min(per_claim_best_scores)
        if min_score >= 0.7:
            return GroundednessDecision.GROUNDED
        elif min_score >= 0.3:
            return GroundednessDecision.PARTIALLY_GROUNDED
        return GroundednessDecision.UNGROUNDED
    else:
        avg_score = sum(per_claim_best_scores) / len(per_claim_best_scores)
        if avg_score >= 0.6:
            return GroundednessDecision.GROUNDED
        elif avg_score >= 0.3:
            return GroundednessDecision.PARTIALLY_GROUNDED
        return GroundednessDecision.UNGROUNDED

# ===========================================================================
# Step 5 : Interpretability & Trace Generation
# ===========================================================================

def generate_groundedness_trace(
    claims: List[str],
    matched_chunks_per_claim: List[List[Document]],
    entailment_scores_per_claim: List[List[EntailmentScores]],
    final_decision: GroundednessDecision,
) -> Dict[str, object]:
    """Make the groundedness decision interpretable by producing a trace.

    The trace exposes which claims were supported, contradicted, or left
    unsupported by which source chunks so that downstream consumers (e.g.
    ``AnswerSynthesizerAgent`` / ``Orchestrator``) can surface evidence
    to the user.

    Parameters
    ----------
    claims : List[str]
        The ordered list of claims extracted in Step 1.
    matched_chunks_per_claim : List[List[Document]]
        For each claim, the list of relevance-span chunks matched in Step 2.
    entailment_scores_per_claim : List[List[EntailmentScores]]
        For each claim, the NLI scores produced in Step 3.
    final_decision : GroundednessDecision
        The aggregated verdict produced in Step 4.

    Returns
    -------
    Dict[str, object]
        A serialisable trace dictionary compatible with the existing
        ``trace`` field in the ``AnswerSynthesizerAgent`` / ``Orchestrator``
        output schema.
    """
    claim_verdicts: List[Dict[str, object]] = []
    for claim, chunks, scores in zip(
        claims, matched_chunks_per_claim, entailment_scores_per_claim
    ):
        # Derive a per-claim status from the best entailment score
        best = max((s.entailment for s in scores), default=0.0)
        if best >= 0.7:
            status = "supported"
        elif any(s.contradiction > 0.5 for s in scores):
            status = "contradicted"
        else:
            status = "unsupported"

        claim_verdicts.append(
            {
                "claim": claim,
                "status": status,
                "matched_chunks": [
                    {
                        "chunk_id": c.metadata.get("chunk_id")
                        or c.metadata.get("record_id")
                        or "unknown",
                        "chunk_text": c.metadata.get("original_text") or c.page_content,
                    }
                    for c in chunks
                ],
                "entailment_scores": [
                    {
                        "chunk_id": s.chunk_id,
                        "entailment": s.entailment,
                        "neutral": s.neutral,
                        "contradiction": s.contradiction,
                    }
                    for s in scores
                ],
            }
        )

    return {
        "final_decision": final_decision.value,
        "num_claims": len(claims),
        "claim_verdicts": claim_verdicts,
    }
