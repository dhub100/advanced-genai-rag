"""
Retrieval agents and orchestration strategies (pipeline step 2).

This sub-package provides three complementary retrieval paradigms and four
strategies for combining them into a single ranked result list.

Why it matters:
    No single retrieval method dominates across all query types.  BM25
    excels on factoid, keyword-dense queries; dense semantic search handles
    paraphrase and conceptual queries; GraphRAG captures entity relationships
    that neither lexical nor embedding-based methods surface well.
    Combining all three with learned or heuristic weights achieves
    consistently higher recall and precision than any agent alone.

Sub-package structure:
    agents/
        bm25.py   – BM25Agent: lexical retrieval with bilingual query
                    expansion (EN ↔ DE via M2M100).
        dense.py  – DenseAgent: semantic retrieval backed by ChromaDB.
        graph.py  – GraphAgent: knowledge-graph traversal retrieval.
    orchestrators/
        base.py       – BaseOrchestrator abstract class (common interface).
        waterfall.py  – Conditional GraphRAG activation based on BM25/Dense
                        overlap; avoids expensive graph calls when not needed.
        voting.py     – Equal-weight RRF over all three agents.
        confidence.py – Query-type-aware asymmetric RRF weights.
        adaptive.py   – Epsilon-greedy Q-learning that improves weights
                        online from MRR feedback.
    classifier.py – Classifies queries as factoid / semantic / balanced.
    fusion.py     – Weighted Reciprocal Rank Fusion (RRF) implementation.
    translator.py – Lazy-loaded M2M100 EN ↔ DE translation.

Relationship to other sub-packages:
    Agents index JSON chunks produced by ``rag.preprocessing``.
    Orchestrator outputs are evaluated by ``rag.evaluation``.
"""
