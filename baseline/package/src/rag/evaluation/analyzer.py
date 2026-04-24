"""
Post-evaluation analysis: agent complementarity and failure patterns.
"""

from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Agent complementarity
# ---------------------------------------------------------------------------

class AgentComplementarityAnalyzer:
    """Measure how much unique signal each retrieval agent contributes.

    Runs all three agents on the same query and computes set-overlap
    statistics so that one can see whether agents are redundant or whether
    each surfaces documents the others miss.

    Attributes:
        bm25: BM25Agent instance.
        dense: DenseAgent instance.
        graph_rag: GraphAgent instance.
    """

    def __init__(self, bm25, dense, graph_rag):
        """Initialise with pre-built retrieval agent instances.

        Args:
            bm25: BM25Agent (or any object with ``.search(query, top_k)``).
            dense: DenseAgent (or any object with ``.search(query, top_k)``).
            graph_rag: GraphAgent (or any object with ``.retrieve(query, top_k)``).
        """
        self.bm25      = bm25
        self.dense     = dense
        self.graph_rag = graph_rag

    @staticmethod
    def _ids(docs) -> set:
        """Extract a set of document IDs from a list of Document objects.

        Args:
            docs: List of Document-like objects with a ``metadata`` dict
                containing either ``"chunk_id"`` or ``"record_id"``.

        Returns:
            Set of non-None document identifier strings.
        """
        return {d.metadata.get("chunk_id") or d.metadata.get("record_id") for d in docs}

    def analyze_overlap(self, query: str, top_k: int = 10) -> dict:
        """Compute pairwise and triple set-overlap statistics for a single query.

        Args:
            query: Natural-language question to retrieve for.
            top_k: Number of documents to retrieve per agent.

        Returns:
            Dict with keys: ``query``, ``total_unique``, ``all_three``,
            ``bm25_dense``, ``bm25_graph``, ``dense_graph``, ``bm25_only``,
            ``dense_only``, ``graph_only`` — all expressed as raw document counts.
        """
        bm25_ids  = self._ids(self.bm25.search(query, top_k=top_k))
        dense_ids = self._ids(self.dense.search(query, top_k=top_k))
        graph_ids = self._ids(self.graph_rag.retrieve(query, top_k=top_k))

        all_three  = bm25_ids & dense_ids & graph_ids
        bm25_dense = bm25_ids & dense_ids
        bm25_graph = bm25_ids & graph_ids
        dense_graph = dense_ids & graph_ids

        return {
            "query":       query,
            "total_unique": len(bm25_ids | dense_ids | graph_ids),
            "all_three":   len(all_three),
            "bm25_dense":  len(bm25_dense),
            "bm25_graph":  len(bm25_graph),
            "dense_graph": len(dense_graph),
            "bm25_only":   len(bm25_ids - dense_ids - graph_ids),
            "dense_only":  len(dense_ids - bm25_ids - graph_ids),
            "graph_only":  len(graph_ids - bm25_ids - dense_ids),
        }

    def visualize_overlap(self, overlap: dict) -> None:
        """Render an interactive Plotly bar chart from an ``analyze_overlap`` result.

        Args:
            overlap: Dict as returned by :meth:`analyze_overlap`.
        """
        categories = ["All 3", "BM25+Dense", "BM25+Graph", "Dense+Graph",
                      "BM25 only", "Dense only", "Graph only"]
        values = [
            overlap["all_three"],
            overlap["bm25_dense"]  - overlap["all_three"],
            overlap["bm25_graph"]  - overlap["all_three"],
            overlap["dense_graph"] - overlap["all_three"],
            overlap["bm25_only"],
            overlap["dense_only"],
            overlap["graph_only"],
        ]
        fig = go.Figure(go.Bar(
            x=categories, y=values,
            marker_color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"],
            text=values, textposition="auto",
        ))
        fig.update_layout(
            title=f"Agent Complementarity: {overlap['query'][:60]}",
            xaxis_title="Agent Combination",
            yaxis_title="Documents",
            template="plotly_white",
            height=500,
        )
        fig.show()

    def batch_analyze(self, queries: List[str], top_k: int = 10) -> pd.DataFrame:
        """Run overlap analysis over a list of queries and return a summary DataFrame.

        Args:
            queries: List of natural-language questions.
            top_k: Documents per agent per query.

        Returns:
            DataFrame with columns ``query``, ``total_unique``,
            ``all_three_pct``, ``bm25_only_pct``, ``dense_only_pct``,
            and ``graph_only_pct``.
        """
        rows = []
        for q in tqdm(queries, desc="Complementarity analysis"):
            ov = self.analyze_overlap(q, top_k)
            n  = ov["total_unique"] or 1
            rows.append({
                "query":          q[:50],
                "total_unique":   ov["total_unique"],
                "all_three_pct":  ov["all_three"]  / n * 100,
                "bm25_only_pct":  ov["bm25_only"]  / n * 100,
                "dense_only_pct": ov["dense_only"]  / n * 100,
                "graph_only_pct": ov["graph_only"]  / n * 100,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Identify low-performing queries and surface common failure patterns.

    Attributes:
        qrels: pytrec_eval-compatible relevance judgements dict used to
            cross-reference failing queries with their expected answers.
    """

    def __init__(self, qrels: dict):
        """Initialise with relevance judgements.

        Args:
            qrels: pytrec_eval-compatible qrels dict
                ``{query_id: {doc_id: relevance_int}}``.
        """
        self.qrels = qrels

    def identify_failures(self, per_query: dict, threshold: float = 0.5) -> List[str]:
        """Return query IDs whose NDCG@10 score falls below a threshold.

        Args:
            per_query: pytrec_eval per-query result dict
                ``{query_id: {metric: score}}``.
            threshold: Minimum acceptable NDCG@10 (default 0.5).

        Returns:
            List of query ID strings that failed the threshold.
        """
        return [qid for qid, m in per_query.items() if m.get("ndcg_cut_10", 0) < threshold]

    def analyze_failure_patterns(self, failures: List[str], qa_data: list) -> dict:
        """Compute aggregate statistics about the failed queries.

        Args:
            failures: List of query IDs as returned by
                :meth:`identify_failures`.
            qa_data: Full list of question dicts (each with ``"id"`` and
                ``"question"`` keys).

        Returns:
            Dict with keys ``total_failures``, ``avg_length`` (tokens),
            ``has_digits``, ``has_names``, ``question_types``
            (Counter), and ``examples`` (first 5 failing question dicts).
        """
        fqs = [q for q in qa_data if str(q["id"]) in failures]
        return {
            "total_failures": len(failures),
            "avg_length":     np.mean([len(q["question"].split()) for q in fqs]) if fqs else 0,
            "has_digits":     sum(1 for q in fqs if any(c.isdigit() for c in q["question"])),
            "has_names":      sum(1 for q in fqs if any(w[0].isupper() for w in q["question"].split() if len(w) > 1)),
            "question_types": Counter([self._classify(q["question"]) for q in fqs]),
            "examples":       fqs[:5],
        }

    @staticmethod
    def _classify(question: str) -> str:
        """Classify a question by its opening interrogative word.

        Args:
            question: Natural-language question string.

        Returns:
            Capitalised interrogative word (``"Who"``, ``"What"``, …) or
            ``"Other"`` if the question does not start with a known word.
        """
        q = question.lower()
        for word in ("who", "what", "when", "where", "why", "how"):
            if q.startswith(word):
                return word.title()
        return "Other"

    def print_summary(self, patterns: dict) -> None:
        """Print a human-readable failure analysis summary to stdout.

        Args:
            patterns: Dict as returned by :meth:`analyze_failure_patterns`.
        """
        print(f"\n{'='*60}")
        print(f"FAILURE ANALYSIS  ({patterns['total_failures']} queries below threshold)")
        print(f"{'='*60}")
        print(f"Avg query length : {patterns['avg_length']:.1f} tokens")
        print(f"Queries w/ digits: {patterns['has_digits']}")
        print(f"Queries w/ names : {patterns['has_names']}")
        print("\nQuestion-type distribution:")
        for qtype, count in patterns["question_types"].most_common():
            print(f"  {qtype}: {count}")
        print("\nTop failure examples:")
        for i, ex in enumerate(patterns["examples"], 1):
            print(f"\n{i}. Q: {ex['question']}")
            print(f"   A: {ex['answer']}")
