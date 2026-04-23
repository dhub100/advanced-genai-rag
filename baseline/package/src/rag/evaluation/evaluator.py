"""
Comprehensive evaluation framework for multi-agent RAG systems.

Quantitative metrics (pytrec_eval): P@k, Recall@k, MRR, NDCG@k
Efficiency metrics: latency percentiles
Statistical tests: paired t-test between strategies
"""

import time
import warnings
from collections import defaultdict
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytrec_eval
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from rag.evaluation.metrics import METRICS

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


class ComprehensiveEvaluator:
    """Evaluate and compare retrieval strategies using standard IR metrics.

    Wraps pytrec_eval to compute precision, recall, MRR, and NDCG at
    various cut-offs, and adds per-query latency statistics.  Results from
    multiple retrievers are stored internally so they can be compared with
    ``compare_strategies`` or visualised with the plot helpers.

    Attributes:
        qrels: pytrec_eval-compatible relevance judgements dict
            ``{query_id: {doc_id: relevance_int}}``.
        metrics: Set of pytrec_eval metric strings to compute.
        results: Accumulated evaluation results keyed by retriever name.
    """

    def __init__(self, qrels: dict, metrics: Optional[set] = None):
        """Initialise the evaluator with relevance judgements and metric set.

        Args:
            qrels: pytrec_eval-compatible qrels dict mapping query IDs to
                dicts of ``{doc_id: relevance_int}``.
            metrics: Set of pytrec_eval metric strings.  Defaults to the
                standard set defined in ``rag.evaluation.metrics.METRICS``.
        """
        self.qrels = qrels
        self.metrics = metrics or METRICS
        self.results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_retriever(self, retriever, qa_data: list, name: str) -> dict:
        """Run a retriever over all benchmark questions and record IR metrics.

        Accepts both agent-style retrievers (exposing ``.search(query, top_k)``)
        and orchestrator-style callables that return ``(docs, trace)``.

        Args:
            retriever: Retrieval object.  Either an agent with a
                ``.search(query, top_k)`` method, or a callable that returns
                ``(List[Document], dict)`` where the dict is a trace.
            qa_data: List of question dicts with at least ``"id"`` and
                ``"question"`` keys.
            name: Display name used as the key in ``self.results`` and in
                progress output.

        Returns:
            Result dict with keys ``metrics_mean``, ``metrics_std``,
            ``efficiency``, ``per_query``, ``timing_data``, and ``traces``.
        """
        run: dict = defaultdict(dict)
        timing: list = []
        traces: list = []

        print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")

        for q in tqdm(qa_data, desc=name):
            qid = str(q["id"])
            t0 = time.time()

            if hasattr(retriever, "search"):
                docs = retriever.search(q["question"], top_k=100)
                trace = None
            else:
                docs, trace = retriever(q["question"], top_k=100)
                traces.append({"qid": qid, "question": q["question"], "trace": trace})

            timing.append(time.time() - t0)

            for rank, doc in enumerate(docs, 1):
                did = doc.metadata.get("chunk_id") or doc.metadata.get("record_id")
                if did:
                    run[qid][did] = 100 - rank

        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, self.metrics)
        per_query = evaluator.evaluate(run)
        df = pd.DataFrame(per_query).T

        self.results[name] = {
            "metrics_mean": df.mean().to_dict(),
            "metrics_std": df.std().to_dict(),
            "efficiency": self._efficiency(timing),
            "per_query": per_query,
            "timing_data": timing,
            "traces": traces,
        }
        return self.results[name]

    @staticmethod
    def _efficiency(timing: list) -> dict:
        """Compute latency statistics from a list of per-query timings.

        Args:
            timing: List of elapsed seconds, one entry per query.

        Returns:
            Dict with keys ``avg_latency``, ``std_latency``,
            ``median_latency``, ``p95_latency``, ``p99_latency``,
            and ``total_time``.
        """
        a = np.array(timing)
        return {
            "avg_latency": float(a.mean()),
            "std_latency": float(a.std()),
            "median_latency": float(np.median(a)),
            "p95_latency": float(np.percentile(a, 95)),
            "p99_latency": float(np.percentile(a, 99)),
            "total_time": float(a.sum()),
        }

    # ------------------------------------------------------------------
    # Comparison & statistics
    # ------------------------------------------------------------------

    def compare_strategies(self) -> pd.DataFrame:
        """Build a summary table of mean IR metrics and latency for all evaluated strategies.

        Returns:
            DataFrame with one row per strategy and columns for each metric,
            average latency, and P95 latency.
        """
        rows = []
        for name, data in self.results.items():
            row = {"Strategy": name}
            row.update({m: f"{v:.4f}" for m, v in data["metrics_mean"].items()})
            row["Avg_Latency(s)"] = f"{data['efficiency']['avg_latency']:.4f}"
            row["P95_Latency(s)"] = f"{data['efficiency']['p95_latency']:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def statistical_significance_test(
        self, strategy1: str, strategy2: str, metric: str = "recip_rank"
    ) -> dict:
        """Run a paired t-test comparing two evaluated strategies on a single metric.

        Args:
            strategy1: Name of the first strategy (must already be in
                ``self.results``).
            strategy2: Name of the second strategy.
            metric: pytrec_eval metric string to compare on (e.g.
                ``"recip_rank"``, ``"ndcg_cut_10"``).

        Returns:
            Dict with keys ``strategy1``, ``strategy2``, ``metric``,
            ``t_statistic``, ``p_value``, ``significant`` (bool, p < 0.05),
            and ``mean_diff``.
        """
        s1 = [q[metric] for q in self.results[strategy1]["per_query"].values()]
        s2 = [q[metric] for q in self.results[strategy2]["per_query"].values()]
        t_stat, p_value = stats.ttest_rel(s1, s2)
        return {
            "strategy1": strategy1,
            "strategy2": strategy2,
            "metric": metric,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "mean_diff": float(np.mean(s1) - np.mean(s2)),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_metric_distributions(self, metric: str = "recip_rank") -> None:
        """Render a box-plot of per-query metric scores for each evaluated strategy.

        Args:
            metric: pytrec_eval metric string to visualise (default
                ``"recip_rank"``).
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        data = [
            [q[metric] for q in res["per_query"].values()]
            for res in self.results.values()
        ]
        labels = list(self.results.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_title(
            f"Distribution of {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_latency_comparison(self) -> None:
        """Render a grouped bar chart comparing average and P95 latency across strategies."""
        strategies = list(self.results.keys())
        avg_latencies = [
            self.results[s]["efficiency"]["avg_latency"] for s in strategies
        ]
        p95_latencies = [
            self.results[s]["efficiency"]["p95_latency"] for s in strategies
        ]
        x, w = np.arange(len(strategies)), 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        r1 = ax.bar(x - w / 2, avg_latencies, w, label="Average", color="steelblue")
        r2 = ax.bar(x + w / 2, p95_latencies, w, label="P95", color="coral")

        def _label(rects):
            for rect in rects:
                h = rect.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        _label(r1)
        _label(r2)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.set_ylabel("Latency (s)", fontsize=12)
        ax.set_title("Latency Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()


class ExplainableOrchestrator:
    """
    Enhanced orchestrator that provides detailed rationales for decisions.
    """

    def __init__(self, bm25, dense, graph_rag):
        self.bm25 = bm25
        self.dense = dense
        self.graph_rag = graph_rag

    def analyze_query_features(self, query: str) -> dict:
        """Extract features from query to guide orchestration."""
        q_lower = query.lower()
        tokens = query.split()

        features = {
            "length": len(tokens),
            "has_digits": any(ch.isdigit() for ch in query),
            "has_quotes": ('"' in query) or ("'" in query),
            "has_names": any(t[0].isupper() for t in tokens if len(t) > 1),
            "question_words": sum(
                1
                for w in ["who", "what", "when", "where", "why", "how"]
                if w in q_lower
            ),
            "complexity": "simple" if len(tokens) <= 6 else "complex",
        }

        # Classify query type
        features["is_factoid"] = (
            features["has_digits"] or features["has_quotes"] or features["length"] <= 6
        )
        features["is_semantic"] = features["length"] > 10 and not features["has_digits"]

        return features

    def explainable_route(self, query: str, top_k: int = 5) -> tuple[list, dict]:
        """Route query with detailed explanation."""
        features = self.analyze_query_features(query)
        explanation = {
            "query": query,
            "features": features,
            "decisions": [],
            "weights": {},
            "rationale": "",
        }

        # Determine weights based on features
        if features["is_factoid"]:
            weights = {"bm25": 1.4, "dense": 0.9, "graph": 0.5}
            rationale = f"Query is FACTOID (digits={features['has_digits']}, quotes={features['has_quotes']}, length={features['length']}). "
            rationale += "BM25 weighted higher (1.4) for exact matching. Dense reduced (0.9). Graph minimal (0.5)."
            explanation["decisions"].append(
                "Route: Factoid query detected → BM25-heavy strategy"
            )
        elif features["is_semantic"]:
            weights = {"bm25": 1.0, "dense": 1.3, "graph": 0.8}
            rationale = f"Query is SEMANTIC (length={features['length']}, no digits). "
            rationale += "Dense weighted higher (1.3) for semantic similarity. BM25 standard (1.0). Graph elevated (0.8) for context."
            explanation["decisions"].append(
                "Route: Semantic query detected → Dense-heavy strategy"
            )
        else:
            weights = {"bm25": 1.2, "dense": 1.0, "graph": 0.6}
            rationale = "Query is BALANCED. Using default weights: BM25 (1.2), Dense (1.0), Graph (0.6)."
            explanation["decisions"].append("Route: Balanced query → Standard fusion")

        explanation["weights"] = weights
        explanation["rationale"] = rationale

        # Retrieve from all agents
        pre_k = max(30, top_k * 10)
        explanation["decisions"].append(f"Retrieving top-{pre_k} from each agent")

        bm25_docs = self.bm25.search(query, top_k=pre_k)
        dense_docs = self.dense.search(query, top_k=pre_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=pre_k)

        explanation["decisions"].append(
            f"BM25: {len(bm25_docs)} docs, Dense: {len(dense_docs)} docs, Graph: {len(graph_docs)} docs"
        )

        # Simple fusion (placeholder - should use proper RRF fusion)
        # Note: This is a simplified version. In production, use proper RRF fusion
        from collections import defaultdict

        scores = defaultdict(float)
        doc_store = {}
        k_rrf = 60

        for name, docs, weight in [
            ("bm25", bm25_docs, weights["bm25"]),
            ("dense", dense_docs, weights["dense"]),
            ("graph", graph_docs, weights["graph"]),
        ]:
            for rank, d in enumerate(docs, 1):
                uid = d.metadata.get("chunk_id") or d.metadata.get("record_id")
                if uid:
                    doc_store[uid] = d
                    scores[uid] += weight * (1.0 / (k_rrf + rank))

        fused = sorted(
            doc_store.values(),
            key=lambda d: scores[
                d.metadata.get("chunk_id") or d.metadata.get("record_id")
            ],
            reverse=True,
        )
        explanation["decisions"].append(
            f"Fusion complete: {len(fused)} unique documents"
        )

        final_docs = fused[:top_k]
        explanation["decisions"].append(f"Returning top-{top_k} documents")

        return final_docs, explanation

    def print_explanation(self, explanation: dict):
        """Print explanation in readable format."""
        print("\n" + "=" * 80)
        print(f"QUERY: {explanation['query']}")
        print("=" * 80)
        print("\nQUERY FEATURES:")
        for k, v in explanation["features"].items():
            print(f"  {k}: {v}")
        print("\nAGENT WEIGHTS:")
        for k, v in explanation["weights"].items():
            print(f"  {k}: {v}")
        print("\nRATIONALE:")
        print(f"  {explanation['rationale']}")
        print("\nDECISION FLOW:")
        for i, dec in enumerate(explanation["decisions"], 1):
            print(f"  {i}. {dec}")
        print("=" * 80)
