"""Groundedness evaluation pipeline.

Implements the Answer-to-Source Alignment strategy that decomposes a generated
answer into individual claims, matches each claim to relevant source document
chunks, measures entailment with a zero-shot NLI model, aggregates the scores
into a single decision, and produces an interpretable trace.
"""

from __future__ import annotations

import spacy
from lingua import Language, LanguageDetectorBuilder
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# TODO: Ensure optimal implementation and fit on Alin's pipeline

DETECTOR = (
    LanguageDetectorBuilder
    .from_languages(
        Language.ENGLISH,
        Language.GERMAN)
    .with_preloaded_language_models()
    .build()
)

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
NLP_EN = spacy.load("en_core_web_sm")
NLP_DE = spacy.load("de_core_news_sm")


class GroundednessDecision(str, Enum):
    """Final groundedness verdict for an answer."""

    GROUNDED = 1.0
    PARTIALLY_GROUNDED = 0.5
    UNGROUNDED = 0.0


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


def detect_language(text: str) -> str:
    """Detect the language of a text string using the Lingua detector.

    Args:
        text: Plain text to classify.

    Returns:
        Lowercase ISO 639-1 language code (e.g. ``"en"``, ``"de"``), or an
        empty string if detection fails or text is blank.
    """
    if not text.strip():
        return ""
    try:
        lang = DETECTOR.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else ""
    except Exception:
        return ""


class GroundednessVerifier:
    def __init__(
        self,
        vectordb_path=(
            "/content/drive/MyDrive/Adv_GenAI/storage/full_corpus/vectordb_dense/fixed_e5"
            ),
        embedding_model=HuggingFaceEmbeddings(),
    ) -> None:
        """Initialize the Groundness Verifier.

        Params:
            * vectordb_path (str): Path pointing to a Chroma database instance.
                Must be stored to disk.
            * embedding_model: A langchain.embeddings.Embeddings instances."""
        self.nlp_en = NLP_EN
        self.nlp_de = NLP_DE
        self.vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=embedding_model,
        )
        self._embedding_model = embedding_model

    def decompose_answer_into_claims(
        self,
        answer: str
    ) -> List[str]:
        """Use `spacy` to split a generated answer into individual,
        verifiable claims.

        Parameters:
            * answer (str): The raw generated answer produced
                by the downstream LLM.

        Returns:
            * List[str]: A list of self-contained claim sentences
                extracted from *answer*.
        """
        lang = detect_language(answer)
        if lang == "en":
            doc = self.nlp_en(answer)
        elif lang == "de":
            doc = self.nlp_de(answer)
        else:
            raise AttributeError("Undetected '{lang}' language.")
        return [
            sent.text.strip()
            for sent in doc.sents
            if len(sent.text.strip()) > 10
        ]

    def match_claim_to_relevance_spans(
        self,
        claim: str,
        retrieved_chunks: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Associate a decomposed claim with the most relevant
        source text spans.

        The function consumes the *k* document chunks already retrieved by the
        upstream retrieval pipeline (BM25, dense, or graph) and
        returns the subset deemed most relevant to the individual *claim*.

        Parameters:
        * claim (str): A single verifiable claim produced by Step 1.
        * retrieved_chunks (List[Document]): Pool of candidate chunks
            output by the heterogeneous retriever backends.
        * top_k (int, optional): Maximum number of chunks to keep per claim,
            by default 5.

        Returns:
        * List[Document] The *top_k* chunks best matching the claim
            according to a lightweight overlap heuristic
            (placeholder for a dedicated re-ranker).
        """
        # chunk-id → Document mapping
        chunk_id_to_doc: dict[str, Document] = {}
        chunk_ids: list[str] = []
        for doc in retrieved_chunks:
            cid = str(doc.metadata.get("chunk_id", "")).strip()
            if cid:
                chunk_ids.append(cid)
                chunk_id_to_doc[cid] = doc

        if not chunk_ids:
            return retrieved_chunks[:top_k]

        # Access the underlying chromadb Collection to fetch by document ID.
        collection = self._vectordb._collection
        result = collection.get(ids=chunk_ids, include=["embeddings"])

        #  Chroma ID → numpy embedding mapping
        id_to_embedding: dict[str, np.ndarray] = {}
        for cid, emb in zip(result["ids"], result.get("embeddings") or []):
            if emb is not None:
                id_to_embedding[cid] = np.asarray(emb, dtype=np.float32)

        if not id_to_embedding:
            return retrieved_chunks[:top_k]

        # Embed the claim
        claim_emb = np.asarray(
            self._embedding_model.embed_query(claim), dtype=np.float32
        )
        claim_norm = np.linalg.norm(claim_emb) + 1e-8

        # Cosine similarity & top-k selection
        scored: list[tuple[float, Document]] = []
        for cid, doc in chunk_id_to_doc.items():
            chunk_emb = id_to_embedding.get(cid)
            if chunk_emb is None:
                continue
            sim = float(
                np.dot(claim_emb, chunk_emb)
                / (claim_norm * (np.linalg.norm(chunk_emb) + 1e-8))
            )
            scored.append((sim, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def measure_entailment(
        self,
        claim: str,
        chunks: List[Document],
        model_name: str = (
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        )
    ) -> EntailmentScores:
        """Quantify how strongly a claim is supported by a source chunk.

        Uses the NLI model (`model_name`)
        to compute the textual entailment relationship between
        **chunk** (premise) and *claim* (hypothesis)

        Parameters:
            * claim : str
                The verifiable claim to evaluate.
            *chunk : Documents
                relevant sources chunk matched in Step 2.

        Returns:
            * EntailmentScores
                Structured probabilities for the three NLI labels:
                    *entailment*, *neutral*, and *contradiction*
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
            input = TOKENIZER(
                premise, hypothesis, truncation=True, return_tensors="pt"
            )
            # As per the model HF page.
            output = MODEL(input["input_ids"].to(device))
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

    def aggregate_groundedness(
        per_claim_best_scores: List[float],
        strict: bool = True,
    ) -> GroundednessDecision:
        """Aggregate individual claim–chunk entailment scores into a decision.
        Use the top-k (step 1) scores computed by the NLI model (step 2).

        Parameters:
            *  per_claim_best_scores (List[float])
                For each claim the highest entailment probability observed
                across all matched chunks.
            *  strict (bool | None): When ``True`` a single unsupported
                claim downgrades the whole answer to *ungrounded*;
                when ``False`` a majority vote is used.

        Returns:
            * GroundednessDecision: One of ``GROUNDED``,
                ``PARTIALLY_GROUNDED``, or ``UNGROUNDED``.
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

    def check(
            self,
            answer: str,
            retrieved_documents: List[Document]
    ) -> float:
        """Main method compute the average groundedness score
        between all claim of a single answer."""
        claims: list[str] = self.decompose_answer_into_claims(answer)
        # Claim - Groundedness Score Mapping
        claim_to_score: dict[str, list[float]] = {}

        # Chain the pipeline steps:
        # step 2 (match_claim_to_relevance),
        # step 3 (measure_entailment)
        # step 4 (aggregated_groundedness)
        for claim in claims:
            matched_docs = self.match_claim_to_relevance_spans(claim)
            entailment_score = self.measure_entailment(
                claim, matched_docs
            )
            avg_entailment = np.mean([
                score for score in entailment_score.entailment
            ])
            groundedness = self.aggregate_groundedness(avg_entailment)
            claim_to_score[claim] = groundedness

        return np.mean(claim_to_score.values())
