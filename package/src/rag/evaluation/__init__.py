"""
Evaluation framework for comparing retrieval strategies (pipeline step 3).

This sub-package measures the quality and efficiency of all retrieval agents
and orchestration strategies against gold-standard relevance judgements
(qrels) produced in ``rag.preprocessing.relevance``.

Why it matters:
    Without a rigorous evaluation harness it is impossible to know whether
    a more complex orchestration strategy actually outperforms a simple BM25
    baseline.  This package provides the statistical evidence needed to make
    that determination, including per-query IR metrics, latency percentiles,
    paired significance tests, agent complementarity analysis, and failure
    pattern inspection.

Modules:
    evaluator.py – ComprehensiveEvaluator: runs any retriever against a
                   benchmark, computes IR metrics via pytrec_eval, and
                   renders comparison plots.
    analyzer.py  – AgentComplementarityAnalyzer: quantifies the unique
                   signal contributed by each agent.
                   FailureAnalyzer: surfaces low-performing queries and
                   common failure patterns.
    metrics.py   – METRICS constant and load_qrels() helper that converts
                   per-chunk relevance score files into a pytrec_eval dict.

Relationship to other sub-packages:
    Consumes retriever objects from ``rag.retrieval`` and qrels files
    produced by ``rag.preprocessing.relevance``.
"""
from rag.evaluation.evaluator import ComprehensiveEvaluator
from rag.evaluation.analyzer import AgentComplementarityAnalyzer, FailureAnalyzer
from rag.evaluation.metrics import METRICS, load_qrels

__all__ = [
    "ComprehensiveEvaluator",
    "AgentComplementarityAnalyzer",
    "FailureAnalyzer",
    "METRICS",
    "load_qrels",
]
