"""
Document ingestion and enrichment pipeline (pipeline step 1).

This sub-package transforms raw HTML files from the ETH Zurich news archive
into structured, semantically enriched JSON chunks that downstream retrieval
agents can index.

Why it matters:
    Retrieval quality is tightly coupled to document quality.  Raw HTML
    contains navigation markup, boilerplate, and OCR artefacts that would
    pollute lexical and semantic indexes.  This stage removes that noise,
    adds linguistic metadata, and produces relevance judgements so that the
    evaluation stage can compute gold-standard IR metrics.

Module overview:
    html_parser – Step 1a: HTML → minimal JSON (raw_text + paragraphs).
    cleaner     – Step 1b: boilerplate removal, NLP enrichment (spaCy,
                   YAKE, Lingua), and structured record creation.
    validator   – Step 1c: quality filter that discards empty documents.
    benchmark   – Step 1d: Q&A pair extraction from the benchmark PDF.
    metadata    – Step 1e: LLM-based (GPT-4o-mini) structured metadata
                   extraction per chunk (entities, topic tags, events, …).
    relevance   – Step 1f: LLM-based chunk-to-question relevance scoring
                   that produces the qrels used in evaluation.

Relationship to other sub-packages:
    Output JSON chunks are consumed by ``rag.retrieval`` agents for
    indexing, and the relevance scores are consumed by ``rag.evaluation``
    to construct pytrec_eval-compatible qrels.
"""
