"""
Step 3: Evaluation and Analysis Script
This script can be run in Google Colab or converted to a Jupyter notebook.

To convert to notebook:
1. Copy sections to new Colab/Jupyter notebook cells
2. Or use: jupyter nbconvert --to notebook Step_3_Evaluation_Script.py
"""

# =============================================================================
# SECTION 1: SETUP AND IMPORTS
# =============================================================================

print("Installing required packages...")
# !pip install -q langdetect nltk rank_bm25
# !pip install -q langchain langchain-community langchain-core langchain-huggingface chromadb
# !pip install pytrec_eval scikit-learn seaborn plotly

# Standard library
import pickle
import os
import json
import time
import pathlib
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML/Transformers
import torch
import nltk
from tqdm import tqdm

# Evaluation
import pytrec_eval

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports complete")

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_EN = set(nltk.corpus.stopwords.words("english"))
STOP_DE = set(nltk.corpus.stopwords.words("german"))

# Paths
ROOT = pathlib.Path("/content/drive/MyDrive/Adv_GenAI").resolve()
PATH_BM25_PICKLE = "/content/drive/MyDrive/Adv_GenAI/storage/subsample/retrieval_downstream/bm25_fixed_qe.pkl"
PATH_DENSE_LOADER = ROOT / "storage/subsample/vectordb_dense/load_dense_fixed.py"
PATH_GRAG_LOADER = ROOT / "storage/subsample/retrieval_graph/load_graphrag.py"
PATH_QA = ROOT / "benchmark/benchmark_qa_bilingual.json"
PATH_QRELS_FIXED = pathlib.Path("/content/drive/MyDrive/Adv_GenAI/benchmark/score/fixed_size")

# Evaluation metrics
METRICS = {"P_1", "P_3", "P_5", "P_10", "recall_5", "recall_10", "recall_100", 
           "recip_rank", "ndcg_cut_5", "ndcg_cut_10"}

print("✓ Configuration complete")

# =============================================================================
# SECTION 3: COMPREHENSIVE EVALUATOR
# =============================================================================

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for multi-agent RAG systems.
    Supports quantitative, qualitative, and efficiency metrics.
    """
    
    def __init__(self, qrels: dict, metrics: set = None):
        self.qrels = qrels
        self.metrics = metrics or METRICS
        self.results = {}
        
    def evaluate_retriever(self, retriever, qa_data: list, name: str) -> dict:
        """
        Evaluate a single retriever/orchestrator.
        Returns detailed metrics and per-query results.
        """
        run = defaultdict(dict)
        timing_data = []
        trace_data = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        for q in tqdm(qa_data, desc=f"{name} - Retrieval"):
            qid = str(q["id"])
            
            # Measure retrieval time
            start_time = time.time()
            
            # Check if orchestrator returns trace
            if hasattr(retriever, 'search'):
                results = retriever.search(q["question"], top_k=100)
                trace = None
            else:
                results, trace = retriever(q["question"], top_k=100)
                trace_data.append({
                    'qid': qid,
                    'question': q["question"],
                    'trace': trace
                })
            
            elapsed = time.time() - start_time
            timing_data.append(elapsed)
            
            # Build run
            for rank, d in enumerate(results, 1):
                did = d.metadata.get("chunk_id") or d.metadata.get("record_id")
                if did:
                    run[qid][did] = 100 - rank
        
        # Compute IR metrics
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, self.metrics)
        per_query_results = evaluator.evaluate(run)
        
        # Aggregate results
        df = pd.DataFrame(per_query_results).T
        macro_avg = df.mean()
        std_dev = df.std()
        
        # Efficiency metrics
        efficiency = {
            'avg_latency': np.mean(timing_data),
            'std_latency': np.std(timing_data),
            'median_latency': np.median(timing_data),
            'p95_latency': np.percentile(timing_data, 95),
            'p99_latency': np.percentile(timing_data, 99),
            'total_time': sum(timing_data)
        }
        
        self.results[name] = {
            'metrics_mean': macro_avg.to_dict(),
            'metrics_std': std_dev.to_dict(),
            'efficiency': efficiency,
            'per_query': per_query_results,
            'timing_data': timing_data,
            'traces': trace_data
        }
        
        return self.results[name]
    
    def compare_strategies(self) -> pd.DataFrame:
        """Compare all evaluated strategies in a comprehensive table."""
        comparison = []
        
        for name, data in self.results.items():
            row = {'Strategy': name}
            
            # Add mean metrics
            for metric, value in data['metrics_mean'].items():
                row[metric] = f"{value:.4f}"
            
            # Add efficiency
            row['Avg_Latency(s)'] = f"{data['efficiency']['avg_latency']:.4f}"
            row['P95_Latency(s)'] = f"{data['efficiency']['p95_latency']:.4f}"
            
            comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def statistical_significance_test(self, strategy1: str, strategy2: str, 
                                     metric: str = 'recip_rank') -> dict:
        """Perform paired t-test between two strategies."""
        results1 = [q[metric] for q in self.results[strategy1]['per_query'].values()]
        results2 = [q[metric] for q in self.results[strategy2]['per_query'].values()]
        
        t_stat, p_value = stats.ttest_rel(results1, results2)
        
        return {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(results1) - np.mean(results2)
        }
    
    def plot_metric_distributions(self, metric: str = 'recip_rank'):
        """Plot distribution of a metric across all strategies."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        data_to_plot = []
        labels = []
        
        for name, result in self.results.items():
            values = [q[metric] for q in result['per_query'].values()]
            data_to_plot.append(values)
            labels.append(name)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()} Across Strategies', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_latency_comparison(self):
        """Compare latency across strategies."""
        strategies = list(self.results.keys())
        avg_latencies = [self.results[s]['efficiency']['avg_latency'] for s in strategies]
        p95_latencies = [self.results[s]['efficiency']['p95_latency'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, avg_latencies, width, label='Average', color='steelblue')
        rects2 = ax.bar(x + width/2, p95_latencies, width, label='P95', color='coral')
        
        ax.set_ylabel('Latency (seconds)', fontsize=12)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_title('Latency Comparison Across Strategies', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.show()

print("✓ ComprehensiveEvaluator loaded")

# =============================================================================
# SECTION 4: AGENT COMPLEMENTARITY ANALYZER
# =============================================================================

class AgentComplementarityAnalyzer:
    """
    Analyze how different retrieval agents complement each other.
    """
    
    def __init__(self, bm25, dense, graph_rag):
        self.bm25 = bm25
        self.dense = dense
        self.graph_rag = graph_rag
        
    def analyze_overlap(self, query: str, top_k: int = 10) -> dict:
        """Analyze overlap between different agents' results."""
        bm25_docs = self.bm25.search(query, top_k=top_k)
        dense_docs = self.dense.search(query, top_k=top_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=top_k)
        
        def get_ids(docs):
            return set(d.metadata.get("chunk_id") or d.metadata.get("record_id") for d in docs)
        
        bm25_ids = get_ids(bm25_docs)
        dense_ids = get_ids(dense_docs)
        graph_ids = get_ids(graph_docs)
        
        # Calculate overlaps
        all_three = bm25_ids & dense_ids & graph_ids
        bm25_dense = bm25_ids & dense_ids
        bm25_graph = bm25_ids & graph_ids
        dense_graph = dense_ids & graph_ids
        
        # Unique to each
        bm25_only = bm25_ids - dense_ids - graph_ids
        dense_only = dense_ids - bm25_ids - graph_ids
        graph_only = graph_ids - bm25_ids - dense_ids
        
        return {
            'query': query,
            'total_unique': len(bm25_ids | dense_ids | graph_ids),
            'all_three': len(all_three),
            'bm25_dense': len(bm25_dense),
            'bm25_graph': len(bm25_graph),
            'dense_graph': len(dense_graph),
            'bm25_only': len(bm25_only),
            'dense_only': len(dense_only),
            'graph_only': len(graph_only),
        }
    
    def visualize_overlap(self, overlap_data: dict):
        """Create visualization of agent overlap."""
        categories = ['All 3 Agents', 'BM25+Dense', 'BM25+Graph', 'Dense+Graph', 
                     'BM25 Only', 'Dense Only', 'Graph Only']
        values = [
            overlap_data['all_three'],
            overlap_data['bm25_dense'] - overlap_data['all_three'],
            overlap_data['bm25_graph'] - overlap_data['all_three'],
            overlap_data['dense_graph'] - overlap_data['all_three'],
            overlap_data['bm25_only'],
            overlap_data['dense_only'],
            overlap_data['graph_only']
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"Agent Complementarity: {overlap_data['query'][:50]}...",
            xaxis_title="Agent Combination",
            yaxis_title="Number of Documents",
            template="plotly_white",
            height=500
        )
        
        fig.show()
    
    def batch_analyze(self, queries: List[str], top_k: int = 10) -> pd.DataFrame:
        """Analyze complementarity across multiple queries."""
        results = []
        
        for q in tqdm(queries, desc="Analyzing agent complementarity"):
            overlap = self.analyze_overlap(q, top_k)
            results.append({
                'query': q[:50],
                'total_unique': overlap['total_unique'],
                'all_three_pct': (overlap['all_three'] / overlap['total_unique'] * 100) if overlap['total_unique'] > 0 else 0,
                'bm25_only_pct': (overlap['bm25_only'] / overlap['total_unique'] * 100) if overlap['total_unique'] > 0 else 0,
                'dense_only_pct': (overlap['dense_only'] / overlap['total_unique'] * 100) if overlap['total_unique'] > 0 else 0,
                'graph_only_pct': (overlap['graph_only'] / overlap['total_unique'] * 100) if overlap['total_unique'] > 0 else 0,
            })
        
        return pd.DataFrame(results)

print("✓ AgentComplementarityAnalyzer loaded")

# =============================================================================
# SECTION 5: EXPLAINABLE ORCHESTRATOR
# =============================================================================

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
            'length': len(tokens),
            'has_digits': any(ch.isdigit() for ch in query),
            'has_quotes': ('"' in query) or ("'" in query),
            'has_names': any(t[0].isupper() for t in tokens if len(t) > 1),
            'question_words': sum(1 for w in ['who', 'what', 'when', 'where', 'why', 'how'] if w in q_lower),
            'complexity': 'simple' if len(tokens) <= 6 else 'complex'
        }
        
        # Classify query type
        features['is_factoid'] = features['has_digits'] or features['has_quotes'] or features['length'] <= 6
        features['is_semantic'] = features['length'] > 10 and not features['has_digits']
        
        return features
    
    def explainable_route(self, query: str, top_k: int = 5) -> Tuple[List, dict]:
        """Route query with detailed explanation."""
        features = self.analyze_query_features(query)
        explanation = {
            'query': query,
            'features': features,
            'decisions': [],
            'weights': {},
            'rationale': ""
        }
        
        # Determine weights based on features
        if features['is_factoid']:
            weights = {'bm25': 1.4, 'dense': 0.9, 'graph': 0.5}
            rationale = f"Query is FACTOID (digits={features['has_digits']}, quotes={features['has_quotes']}, length={features['length']}). "
            rationale += "BM25 weighted higher (1.4) for exact matching. Dense reduced (0.9). Graph minimal (0.5)."
            explanation['decisions'].append("Route: Factoid query detected → BM25-heavy strategy")
        elif features['is_semantic']:
            weights = {'bm25': 1.0, 'dense': 1.3, 'graph': 0.8}
            rationale = f"Query is SEMANTIC (length={features['length']}, no digits). "
            rationale += "Dense weighted higher (1.3) for semantic similarity. BM25 standard (1.0). Graph elevated (0.8) for context."
            explanation['decisions'].append("Route: Semantic query detected → Dense-heavy strategy")
        else:
            weights = {'bm25': 1.2, 'dense': 1.0, 'graph': 0.6}
            rationale = "Query is BALANCED. Using default weights: BM25 (1.2), Dense (1.0), Graph (0.6)."
            explanation['decisions'].append("Route: Balanced query → Standard fusion")
        
        explanation['weights'] = weights
        explanation['rationale'] = rationale
        
        # Retrieve from all agents
        pre_k = max(30, top_k * 10)
        explanation['decisions'].append(f"Retrieving top-{pre_k} from each agent")
        
        bm25_docs = self.bm25.search(query, top_k=pre_k)
        dense_docs = self.dense.search(query, top_k=pre_k)
        graph_docs = self.graph_rag.retrieve(query, top_k=pre_k)
        
        explanation['decisions'].append(f"BM25: {len(bm25_docs)} docs, Dense: {len(dense_docs)} docs, Graph: {len(graph_docs)} docs")
        
        # Simple fusion (placeholder - should use proper RRF fusion)
        # Note: This is a simplified version. In production, use proper RRF fusion
        from collections import defaultdict
        scores = defaultdict(float)
        doc_store = {}
        k_rrf = 60
        
        for name, docs, weight in [('bm25', bm25_docs, weights['bm25']), 
                                   ('dense', dense_docs, weights['dense']), 
                                   ('graph', graph_docs, weights['graph'])]:
            for rank, d in enumerate(docs, 1):
                uid = d.metadata.get("chunk_id") or d.metadata.get("record_id")
                if uid:
                    doc_store[uid] = d
                    scores[uid] += weight * (1.0 / (k_rrf + rank))
        
        fused = sorted(doc_store.values(), 
                      key=lambda d: scores[d.metadata.get("chunk_id") or d.metadata.get("record_id")], 
                      reverse=True)
        explanation['decisions'].append(f"Fusion complete: {len(fused)} unique documents")
        
        final_docs = fused[:top_k]
        explanation['decisions'].append(f"Returning top-{top_k} documents")
        
        return final_docs, explanation
    
    def print_explanation(self, explanation: dict):
        """Print explanation in readable format."""
        print("\n" + "="*80)
        print(f"QUERY: {explanation['query']}")
        print("="*80)
        print("\nQUERY FEATURES:")
        for k, v in explanation['features'].items():
            print(f"  {k}: {v}")
        print("\nAGENT WEIGHTS:")
        for k, v in explanation['weights'].items():
            print(f"  {k}: {v}")
        print("\nRATIONALE:")
        print(f"  {explanation['rationale']}")
        print("\nDECISION FLOW:")
        for i, dec in enumerate(explanation['decisions'], 1):
            print(f"  {i}. {dec}")
        print("="*80)

print("✓ ExplainableOrchestrator loaded")

# =============================================================================
# SECTION 6: FAILURE ANALYZER
# =============================================================================

class FailureAnalyzer:
    """
    Analyze failure cases and identify patterns.
    """
    
    def __init__(self, qrels: dict):
        self.qrels = qrels
    
    def identify_failures(self, per_query_results: dict, threshold: float = 0.5) -> List[str]:
        """Identify queries where performance is below threshold."""
        failures = []
        for qid, metrics in per_query_results.items():
            if metrics.get('ndcg_cut_10', 0) < threshold:
                failures.append(qid)
        return failures
    
    def analyze_failure_patterns(self, failures: List[str], qa_data: list) -> dict:
        """Analyze common patterns in failure cases."""
        failure_queries = [q for q in qa_data if str(q['id']) in failures]
        
        patterns = {
            'total_failures': len(failures),
            'avg_length': np.mean([len(q['question'].split()) for q in failure_queries]) if failure_queries else 0,
            'has_digits': sum(1 for q in failure_queries if any(ch.isdigit() for ch in q['question'])),
            'has_names': sum(1 for q in failure_queries if any(w[0].isupper() for w in q['question'].split() if len(w) > 1)),
            'question_types': Counter([self._classify_question(q['question']) for q in failure_queries]),
            'examples': failure_queries[:5]
        }
        
        return patterns
    
    def _classify_question(self, question: str) -> str:
        """Classify question type."""
        q_lower = question.lower()
        if q_lower.startswith('who'):
            return 'Who'
        elif q_lower.startswith('what'):
            return 'What'
        elif q_lower.startswith('when'):
            return 'When'
        elif q_lower.startswith('where'):
            return 'Where'
        elif q_lower.startswith('why'):
            return 'Why'
        elif q_lower.startswith('how'):
            return 'How'
        else:
            return 'Other'
    
    def print_failure_analysis(self, failure_patterns: dict):
        """Print failure analysis."""
        print("\n" + "="*80)
        print(f"FAILURE ANALYSIS ({failure_patterns['total_failures']} failures)")
        print("="*80)
        print(f"\nAvg Query Length: {failure_patterns['avg_length']:.1f} words")
        print(f"Queries with Digits: {failure_patterns['has_digits']}")
        print(f"Queries with Names: {failure_patterns['has_names']}")
        print("\nQuestion Type Distribution:")
        for qtype, count in failure_patterns['question_types'].most_common():
            print(f"  {qtype}: {count}")
        print("\nExample Failure Cases:")
        for i, example in enumerate(failure_patterns['examples'], 1):
            print(f"\n{i}. Q: {example['question']}")
            print(f"   A: {example['answer']}")
        print("="*80)

print("✓ FailureAnalyzer loaded")

# =============================================================================
# MAIN EVALUATION SCRIPT
# =============================================================================

def load_qrels(folder: pathlib.Path) -> dict:
    """Load relevance judgments."""
    qrels = defaultdict(dict)
    for fp in folder.glob("*.json"):
        did = fp.stem
        for qid, pay in json.loads(fp.read_text()).items():
            if pay["relevance_score"] >= 0.5:
                qrels[qid][did] = 1
    return qrels

def run_evaluation():
    """
    Main evaluation function.
    
    NOTE: You need to load your retrievers from Step 2 before running this.
    This includes: bm25_fixed_qe, dense_fixed, graph_rag
    And retriever wrappers: WaterfallRetriever, VotingRetriever, ConfidenceRetriever
    """
    
    print("\n" + "="*100)
    print("STEP 3: COMPREHENSIVE EVALUATION AND ANALYSIS")
    print("="*100)
    
    # Load data
    print("\nLoading data...")
    with open(PATH_QA, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    QRELS = load_qrels(PATH_QRELS_FIXED)
    
    print(f"✓ Loaded {len(qa_data)} QA pairs")
    print(f"✓ Loaded {len(QRELS)} queries with relevance judgments")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(QRELS)
    
    # TODO: Load your strategies here
    # strategies = {
    #     "Waterfall": WaterfallRetriever(),
    #     "Voting": VotingRetriever(),
    #     "Confidence": ConfidenceRetriever()
    # }
    
    # Evaluate all strategies
    # for name, retriever in strategies.items():
    #     evaluator.evaluate_retriever(retriever, qa_data, name)
    
    # Display results
    # comparison_df = evaluator.compare_strategies()
    # print("\n" + "="*100)
    # print("QUANTITATIVE RESULTS")
    # print("="*100)
    # print(comparison_df)
    
    # Visualizations
    # for metric in ['recip_rank', 'ndcg_cut_10', 'P_5']:
    #     evaluator.plot_metric_distributions(metric)
    # evaluator.plot_latency_comparison()
    
    # Statistical tests
    # comparisons = [
    #     ('Waterfall', 'Voting'),
    #     ('Waterfall', 'Confidence'),
    #     ('Voting', 'Confidence')
    # ]
    # for s1, s2 in comparisons:
    #     result = evaluator.statistical_significance_test(s1, s2, 'recip_rank')
    #     print(f"\n{s1} vs {s2}:")
    #     print(f"  Mean Difference: {result['mean_diff']:.4f}")
    #     print(f"  p-value: {result['p_value']:.4f}")
    #     print(f"  Significant: {'YES' if result['significant'] else 'NO'}")
    
    print("\n✓ Evaluation framework ready!")
    print("\nTo use this script:")
    print("1. Load your retrievers from Step 2")
    print("2. Uncomment and run the evaluation sections")
    print("3. Analyze results and generate visualizations")

if __name__ == "__main__":
    print("\n" + "="*100)
    print("Step 3 Evaluation Framework - Ready for Use")
    print("="*100)
    print("\nThis script provides:")
    print("  ✓ Comprehensive quantitative evaluation (P@k, Recall@k, MRR, NDCG)")
    print("  ✓ Statistical significance testing")
    print("  ✓ Agent complementarity analysis") 
    print("  ✓ Explainable orchestration")
    print("  ✓ Failure analysis")
    print("  ✓ Efficiency metrics (latency, throughput)")
    print("\nImport your Step 2 components and call run_evaluation() to begin!")

