"""
Advanced multi-agent Retrieval-Augmented Generation (RAG) system.

This package implements a three-stage RAG pipeline designed for information
retrieval over ETH Zurich news articles.  Each stage is a self-contained
sub-package that can be run independently via CLI entry points or imported as
a library.

Pipeline stages:
    1. preprocessing  – Converts raw HTML pages into clean, enriched JSON
                        chunks ready for indexing.  Sub-modules handle HTML
                        parsing (html_parser), boilerplate removal and NLP
                        enrichment (cleaner), quality filtering (validator),
                        benchmark Q&A extraction (benchmark), LLM-based
                        metadata tagging (metadata), and relevance scoring
                        against benchmark questions (relevance).
    2. retrieval      – Provides three complementary retrieval agents (BM25,
                        dense/semantic, GraphRAG) and four orchestration
                        strategies that combine them: waterfall, voting,
                        confidence-weighted, and adaptive Q-learning.
    3. evaluation     – Measures retrieval quality with IR metrics (P@k,
                        Recall@k, MRR, NDCG@k via pytrec_eval), latency
                        statistics, paired statistical tests, agent
                        complementarity analysis, and failure pattern analysis.

Shared utilities (rag.utils) provide file I/O and NLP helpers used across all
three stages, ensuring a single model instance is loaded per process.

    4. reliability     – Reliability-aware extensions (Step 2/3 of the FS26
                         capstone).  Adds evidence sufficiency checking (A),
                         abstention (E), and adaptive recovery (G) on top of
                         the existing retrieval pipeline.  See rag.reliability
                         for EvidenceSufficiencyChecker, AbstentionMechanism,
                         RecoveryAgent, and ReliableOrchestrator.
"""
