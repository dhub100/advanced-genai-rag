"""
Comprehensive evaluation framework for multi-agent RAG systems.

Quantitative metrics (pytrec_eval): P@k, Recall@k, MRR, NDCG@k
Answer quality metrics: BLEU, ROUGE-1, ROUGE-L, semantic similarity
Efficiency metrics: latency percentiles
Statistical tests: paired t-test between strategies
"""

import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

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

    # ------------------------------------------------------------------
    # Answer quality
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_answer_quality(
        results: List[dict],
        embed_fn=None,
    ) -> dict:
        """Compute answer quality metrics comparing generated answers to gold references.

        Three complementary metrics are computed so that different aspects of
        quality are captured:

        * **BLEU** (bilingual evaluation understudy) — n-gram precision between
          the generated answer and the reference.  Uses 1- and 2-gram weights
          with smoothing so short answers are not penalised unfairly.
        * **ROUGE-1 / ROUGE-L** — recall-oriented overlap.  ROUGE-L uses the
          longest common subsequence and is more robust to paraphrase than
          n-gram exact match.
        * **Semantic similarity** — cosine similarity of multilingual-E5
          embeddings.  Captures meaning even when surface wording differs.
          Requires ``embed_fn`` to be provided; omitted when ``None``.

        All scores are averaged over the full input list.

        Args:
            results: List of dicts, each with keys ``"answer"`` (generated text)
                and ``"gold"`` (reference text).  Empty answers are scored 0.
            embed_fn: Optional callable ``(texts: list[str]) -> np.ndarray``
                returning L2-normalised sentence embeddings.  When provided,
                semantic similarity is computed; otherwise that metric is
                returned as ``None``.

        Returns:
            Dict with keys ``bleu``, ``rouge1``, ``rougeL``, and
            ``semantic_similarity`` (``None`` when ``embed_fn`` is absent).
        """
        try:
            from rouge_score import rouge_scorer as _rouge_scorer
        except ImportError as exc:
            raise ImportError(
                "rouge-score is required for answer quality evaluation. "
                "Install it with: pip install rouge-score"
            ) from exc

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError as exc:
            raise ImportError(
                "nltk is required for BLEU computation. "
                "Install it with: pip install nltk && python -m nltk.downloader punkt"
            ) from exc

        scorer = _rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
        smoother = SmoothingFunction().method1

        bleu_scores, rouge1_scores, rougeL_scores = [], [], []
        sem_scores: list[float] = []

        for item in results:
            generated = str(item.get("answer") or "").strip().lower()
            gold = str(item.get("gold") or "").strip().lower()

            if not generated or not gold:
                bleu_scores.append(0.0)
                rouge1_scores.append(0.0)
                rougeL_scores.append(0.0)
                if embed_fn is not None:
                    sem_scores.append(0.0)
                continue

            # BLEU — 1-gram + 2-gram equally weighted
            ref_tokens = gold.split()
            hyp_tokens = generated.split()
            bleu = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoother,
            )
            bleu_scores.append(bleu)

            # ROUGE
            rouge = scorer.score(gold, generated)
            rouge1_scores.append(rouge["rouge1"].fmeasure)
            rougeL_scores.append(rouge["rougeL"].fmeasure)

        # Semantic similarity via embedding cosine
        if embed_fn is not None and results:
            generated_texts = [str(r.get("answer") or "") for r in results]
            gold_texts = [str(r.get("gold") or "") for r in results]
            all_texts = generated_texts + gold_texts
            embeddings = np.array(embed_fn(all_texts))
            n = len(results)
            gen_emb = embeddings[:n]
            gold_emb = embeddings[n:]

            # Normalise rows
            gen_norms = np.linalg.norm(gen_emb, axis=1, keepdims=True) + 1e-8
            gold_norms = np.linalg.norm(gold_emb, axis=1, keepdims=True) + 1e-8
            gen_emb = gen_emb / gen_norms
            gold_emb = gold_emb / gold_norms

            sem_scores = np.sum(gen_emb * gold_emb, axis=1).tolist()

        def _mean(lst: list) -> Optional[float]:
            return float(np.mean(lst)) if lst else None

        return {
            "bleu": round(_mean(bleu_scores) or 0.0, 4),
            "rouge1": round(_mean(rouge1_scores) or 0.0, 4),
            "rougeL": round(_mean(rougeL_scores) or 0.0, 4),
            "semantic_similarity": (
                round(_mean(sem_scores), 4) if sem_scores else None
            ),
            "n_evaluated": len(results),
        }

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
