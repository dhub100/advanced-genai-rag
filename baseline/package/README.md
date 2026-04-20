# Advanced GenAI RAG — Python Package

An installable Python package implementing an agentic, multi-strategy Retrieval-Augmented Generation (RAG) system.
The system ingests ETH Zurich news articles (HTML), builds BM25, dense-vector, and knowledge-graph indexes, and evaluates
four orchestration strategies against a benchmark Q&A set using standard IR metrics.

---

## Table of Contents

- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
  - [`src/rag/preprocessing/`](#srcragpreprocessing)
  - [`src/rag/retrieval/`](#srcragretrieval)
  - [`src/rag/evaluation/`](#srcragevaluation)
  - [`src/rag/utils/`](#srcragutils)
  - [`notebooks/`](#notebooks)
  - [`scripts/`](#scripts)
  - [`tests/`](#tests)
- [How to use](#how-to-use)
  - [Installation](#installation)
  - [Environment setup](#environment-setup)
  - [Step 1 — Preprocessing](#step-1--preprocessing)
  - [Step 2 — Retrieval](#step-2--retrieval)
  - [Step 3 — Evaluation](#step-3--evaluation)
- [Dependencies](#dependencies)

---

## Pipeline overview

``` mermaid
flowchart TD
    A([Raw HTML files]) --> B["html_parser\nBS4 + Docling → minimal JSON"]
    B --> C["cleaner\nBoilerplate removal · Language · NER · Keywords"]
    C --> D["validator\nDrop empty documents"]
    D --> E([Validated JSON chunks])

    F([Benchmark PDF]) --> G["benchmark\npdfplumber + OCR fix → Q&A pairs"]
    G --> H([benchmark_qa.json])

    E --> I["metadata\nGPT-4o-mini structured metadata per chunk"]
    I --> J([metadata/*.json])

    E --> K["relevance\nGPT-4o-mini chunk↔question scores"]
    H --> K
    K --> L([score/*.json = qrels])

    E --> M["BM25Agent\nrank-bm25 + M2M100 bilingual expansion"]
    E --> N["DenseAgent\nChromaDB semantic search"]
    E --> O["GraphAgent\nKnowledge-graph retriever"]

    M --> P{Orchestrator}
    N --> P
    O --> P

    P -->|Waterfall| Q[BM25+Dense → conditional Graph]
    P -->|Voting| R[All three · equal-weight RRF]
    P -->|Confidence| S[Query-type-adaptive weights]
    P -->|Adaptive| T[Q-learning epsilon-greedy]

    Q --> U[Reciprocal Rank Fusion]
    R --> U
    S --> U
    T --> U

    U --> V([Ranked documents])

    V --> W["ComprehensiveEvaluator\nP@k · Recall@k · MRR · NDCG@k · Latency"]
    L --> W
    W --> X([Metrics DataFrame + Plots])

    W --> Y["AgentComplementarityAnalyzer\nPairwise agent overlap"]
    W --> Z["FailureAnalyzer\nLow-NDCG query patterns"]
```

---

## Repository structure

```
package/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── preprocessing/          # Step 1 — document ingestion & enrichment
│       │   ├── __init__.py
│       │   ├── html_parser.py      # 1a: HTML → minimal JSON
│       │   ├── cleaner.py          # 1b: boilerplate removal, NLP enrichment
│       │   ├── validator.py        # 1c: drop empty documents
│       │   ├── benchmark.py        # 1d: PDF Q&A extraction
│       │   ├── metadata.py         # 1e: GPT-4o-mini structured metadata
│       │   └── relevance.py        # 1f: GPT-4o-mini chunk-question scores
│       │
│       ├── retrieval/              # Step 2 — agents and orchestration
│       │   ├── __init__.py
│       │   ├── agents/
│       │   │   ├── __init__.py
│       │   │   ├── bm25.py         # BM25 with bilingual query expansion
│       │   │   ├── dense.py        # ChromaDB semantic retrieval
│       │   │   └── graph.py        # Knowledge-graph retrieval wrapper
│       │   ├── orchestrators/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # Abstract BaseOrchestrator
│       │   │   ├── waterfall.py    # BM25+Dense → conditional GraphRAG
│       │   │   ├── voting.py       # All agents · equal-weight RRF
│       │   │   ├── confidence.py   # Query-type-adaptive weights
│       │   │   └── adaptive.py     # Q-learning (epsilon-greedy)
│       │   ├── fusion.py           # Weighted Reciprocal Rank Fusion
│       │   ├── classifier.py       # Query label: factoid / semantic / balanced
│       │   └── translator.py       # M2M100 EN ↔ DE translation
│       │
│       ├── evaluation/             # Step 3 — metrics and analysis
│       │   ├── __init__.py
│       │   ├── evaluator.py        # ComprehensiveEvaluator (P@k, MRR, NDCG, latency)
│       │   ├── analyzer.py         # AgentComplementarityAnalyzer, FailureAnalyzer
│       │   └── metrics.py          # METRICS constant + load_qrels()
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py               # load_json / save_json helpers
│           └── nlp.py              # Shared spaCy, YAKE, Lingua singletons
│
├── notebooks/
│   ├── 02_retrieval_orchestration.ipynb   # Step 2: end-to-end retrieval walkthrough
│   └── 03_evaluation_and_analysis.ipynb   # Step 3: metrics, analysis, and bonus experiments
│
├── scripts/
│   └── run_pipeline.py             # CLI runner for all preprocessing steps
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
│
├── .env.example                    # Environment variable template
└── pyproject.toml                  # Package metadata and dependencies
```

---

## Module descriptions

### `src/rag/preprocessing/`

Transforms raw HTML files into structured, enriched JSON documents ready for indexing.
Each sub-step is independently runnable as a CLI command or chained via `run_pipeline.py`.

| Module | CLI entry point | Purpose |
|---|---|---|
| `html_parser.py` | `rag-preprocess` | Strips navigation/boilerplate HTML tags with BeautifulSoup, then extracts clean plain text via Docling. Outputs one minimal JSON per HTML file with fields `doc_id`, `filename`, `raw_text`, and `paragraphs`. |
| `cleaner.py` | `rag-clean` | Two-pass enrichment. Pass 1 removes known boilerplate lines (regex) and builds a corpus-wide paragraph frequency table. Pass 2 drops paragraphs repeated across ≥ N documents, detects document language (Lingua), extracts named entities (spaCy), keywords (YAKE), and a short summary. Outputs enriched JSON with `language`, `named_entities`, `keywords`, `summary`, and `text_stats`. |
| `validator.py` | `rag-validate` | Quality gate — discards any document whose `paragraphs_cleaned` list is empty after cleaning. Files are renamed to `<doc_id>.json` in the output directory for stable downstream referencing. |
| `benchmark.py` | `rag-benchmark` | Extracts numbered Q&A pairs from `BenchmarkQuestionsAnswers.pdf` using pdfplumber. Applies a curated OCR correction table (handles `Z`-substitution artefacts common in scanned PDFs) and separates answer text from grading notes. Outputs a JSON array `[{"id": int, "question": str, "answer": str}, ...]`. |
| `metadata.py` | `rag-metadata` | Calls GPT-4o-mini via the OpenAI structured-output API with a strict Pydantic schema to extract entities, topic tags, event dates, role annotations, numeric facts, department, document type, and content date per chunk. Skips already-processed chunks (idempotent). Requires `OPENAI_API_KEY`. |
| `relevance.py` | `rag-score` | Calls GPT-4o-mini for each unscored chunk, asking it to assign a `relevance_score` (0–1) and a short reason for every one of the 25 benchmark questions. Results are saved as `<chunk_id>.json` in the score directory and later consumed by `load_qrels()` to build pytrec_eval-compatible relevance judgements. Requires `OPENAI_API_KEY`. |

---

### `src/rag/retrieval/`

Implements three retrieval agents and four orchestration strategies that combine them.

#### Agents (`retrieval/agents/`)

| Module | Class | Description |
|---|---|---|
| `bm25.py` | `BM25Agent` | Wraps a pre-built `rank_bm25.BM25Okapi` index. Before scoring, the query is expanded to both English and German using M2M100 (via `translator.py`), so German documents are reachable from English queries and vice versa. BM25 scores from both language variants are summed. |
| `dense.py` | `DenseAgent` | Wraps a ChromaDB collection for embedding-based semantic search. Accepts an optional custom `embedding_fn`; falls back to the collection's own embedding function. Returns `_SimpleDoc` objects that share the same `doc.metadata["chunk_id"]` interface as LangChain Documents, allowing uniform handling across agents. |
| `graph.py` | `GraphAgent` | Thin adapter around an external knowledge-graph retriever built in the notebook. Provides a `.search()` alias so that all three agents share the same calling convention. |

#### Orchestrators (`retrieval/orchestrators/`)

All orchestrators extend `BaseOrchestrator`, which enforces the `retrieve(query, top_k) → (docs, trace)` contract.
The `trace` dict carries orchestration metadata (strategy name, agent hit counts, weights, etc.) for debugging and logging.

| Module | Class | Strategy |
|---|---|---|
| `base.py` | `BaseOrchestrator` | Abstract base class — defines the shared constructor and the `retrieve` interface. |
| `waterfall.py` | `WaterfallOrchestrator` | Runs BM25 and Dense first. Only invokes the (slower) GraphRAG agent when the Jaccard overlap between the two result sets falls below a configurable threshold (default 0.3). Fuses the final lists with RRF. Conservative and efficient for queries where keyword and semantic signals already agree. |
| `voting.py` | `VotingOrchestrator` | Runs all three agents in parallel with equal weights and fuses via RRF. The simplest strategy — robust and unbiased, but always pays the full cost of all three agents. |
| `confidence.py` | `ConfidenceOrchestrator` | Classifies the query (factoid / semantic / balanced) and selects a pre-defined weight preset before calling all three agents. Factoid queries up-weight BM25; semantic queries up-weight Dense; balanced queries use a neutral split. |
| `adaptive.py` | `AdaptiveOrchestrator` | Maintains a Q-table keyed on `(query_label, action)` and selects weight presets via epsilon-greedy exploration. After a retrieval round, call `update(query_label, action, reward)` with the MRR score as the reward signal to update the table. Learns over time which weight configuration works best per query type. |

#### Supporting modules

| Module | Description |
|---|---|
| `fusion.py` | Implements weighted Reciprocal Rank Fusion: `score(d) = Σ_agent weight / (k + rank(d, agent))` with `k=60`. Accepts a list of `(agent_name, docs)` tuples and an optional weight dict. Returns documents sorted by descending fused score. |
| `classifier.py` | Heuristic query labeller. A query is **factoid** if it is short (≤ 6 tokens) or contains digits/quoted terms; **semantic** if it is long (> 10 tokens) without numeric anchors; **balanced** otherwise. Returns a feature dict consumed by the Confidence and Adaptive orchestrators. |
| `translator.py` | Lazily loads `facebook/m2m100_418M` on first call (CPU or CUDA). Provides `translate(text, src, tgt)` and `expand_query(query)` which returns `[original, translation]` for bilingual BM25 expansion. |

---

### `src/rag/evaluation/`

Evaluation and analysis tools for comparing retrieval strategies.

| Module | Classes / Functions | Description |
|---|---|---|
| `metrics.py` | `METRICS`, `load_qrels()` | `METRICS` is the set of pytrec_eval metric strings computed by default (P@1/3/5/10, Recall@5/10/100, MRR, NDCG@5/10). `load_qrels(folder, min_score)` reads the per-chunk score files produced by `relevance.py` and builds a `{query_id: {doc_id: 1}}` dict compatible with pytrec_eval. Only chunks with `relevance_score >= min_score` (default 0.5) are marked relevant. |
| `evaluator.py` | `ComprehensiveEvaluator` | Central evaluation class. `evaluate_retriever(retriever, qa_data, name)` runs the retriever over all benchmark questions, records per-query timings, and stores pytrec_eval results internally. `compare_strategies()` returns a summary DataFrame. `statistical_significance_test()` runs a paired t-test. `plot_metric_distributions()` and `plot_latency_comparison()` produce matplotlib/seaborn visualisations. |
| `analyzer.py` | `AgentComplementarityAnalyzer`, `FailureAnalyzer` | `AgentComplementarityAnalyzer` runs all three agents on the same query and computes set-overlap statistics (how many documents each agent retrieves exclusively vs. in common with the others). Supports single-query analysis, interactive Plotly charts, and batch analysis across a full question set. `FailureAnalyzer` identifies queries whose NDCG@10 falls below a threshold and clusters them by query length, digit/name presence, and question-word type to surface systematic weaknesses. |

---

### `src/rag/utils/`

Shared low-level utilities.

| Module | Description |
|---|---|
| `io.py` | `load_json(path)` and `save_json(data, path)` wrappers with UTF-8 handling. |
| `nlp.py` | Module-level singletons for spaCy (en/de), YAKE, and Lingua so that model loading happens at most once per process, regardless of how many pipeline steps import these helpers. |

---

### `notebooks/`

End-to-end Jupyter notebooks that demonstrate the full pipeline using the package.
Each notebook is self-contained: it declares its own dependencies, provides a configuration section for local paths, and can be re-run independently.

| Notebook | Description |
|---|---|
| `02_retrieval_orchestration.ipynb` | Builds and exercises the multi-agent retrieval system. Covers bilingual BM25 (M2M100 query expansion + pseudo-relevance feedback), dense retrieval with multilingual E5 embeddings (ChromaDB), and GraphRAG traversal. Shows how Weighted RRF fuses ranked lists and demonstrates three orchestration strategies — **Waterfall** (lazy GraphRAG invocation), **Voting** (fixed equal weights), and **Confidence** (query-type-adaptive weights). Includes a per-strategy evaluation using P@k, MRR, and NDCG@10. |
| `03_evaluation_and_analysis.ipynb` | Comprehensive evaluation and analysis on top of the retrievers from Notebook 02. Covers standard IR metrics (P@k, Recall@k, MRR, NDCG), paired t-tests for statistical significance, agent complementarity analysis (exclusive vs. shared document overlap), orchestrator explainability, failure analysis (queries with low NDCG@10 and their linguistic patterns), and latency profiling (P95). Also includes bonus sections on **Adaptive orchestration** via Q-learning and **adversarial query robustness**. |

---

### `scripts/`

| File | Description |
|---|---|
| `run_pipeline.py` | Orchestrates all six preprocessing steps (1a–1d) as sequential subprocesses. Accepts command-line arguments for every input/output directory. Aborts immediately if any step returns a non-zero exit code. Steps 1e (metadata) and 1f (relevance scoring) require manual invocation since they make paid API calls and are designed to be run incrementally. |

---

### `tests/`

pytest test suite with one file per pipeline stage:

| File | Covers |
|---|---|
| `test_preprocessing.py` | HTML parsing, cleaning, validation, and benchmark extraction |
| `test_retrieval.py` | Agent search interfaces, orchestrators, fusion, and classifier |
| `test_evaluation.py` | Metric computation, qrels loading, and evaluator output shapes |

---

## How to use

### Installation

Python 3.10+ is required. Install the package in editable mode with all development dependencies:

```bash
cd package
pip install -e ".[dev]"
```

Download the spaCy language models used by `cleaner.py`:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Environment setup

Copy the example env file and add your OpenAI API key (required only for steps 1e and 1f):

```bash
cp .env.example .env
# then edit .env and set:
# OPENAI_API_KEY=sk-...
```

---

### Step 1 — Preprocessing

**Run all steps in sequence (1a–1d):**

```bash
python scripts/run_pipeline.py \
  --html-dir    data/raw \
  --minimal-dir data/processed/minimal \
  --clean-dir   data/processed/clean \
  --valid-dir   data/processed/valid \
  --pdf         data/raw/BenchmarkQuestionsAnswers.pdf \
  --qa-out      data/benchmark/benchmark_qa.json
```

**Or run each step individually:**

```bash
# 1a: HTML → minimal JSON
rag-preprocess data/raw data/processed/minimal

# 1b: Clean + NLP enrichment (--threshold drops paragraphs seen in ≥ N docs)
rag-clean data/processed/minimal data/processed/clean --threshold 5

# 1c: Drop empty documents
rag-validate data/processed/clean data/processed/valid

# 1d: Extract Q&A pairs from benchmark PDF
rag-benchmark data/raw/BenchmarkQuestionsAnswers.pdf data/benchmark/benchmark_qa.json

# 1e: LLM metadata extraction (requires OPENAI_API_KEY; idempotent — safe to re-run)
rag-metadata --chunks subsample/semantic_chunk --meta-dir benchmark/metadata/semantic

# 1f: LLM relevance scoring (requires OPENAI_API_KEY; use --list-missing to check progress)
rag-score --chunks subsample/semantic_chunk \
          --qa-path benchmark/benchmark_qa_bilingual_with_semantic_chunks.json \
          --score-dir benchmark/score/semantic
rag-score --list-missing   # inspect which chunks still need scoring
```

---

### Step 2 — Retrieval

The retrieval agents wrap indexes built in `notebooks/02_retrieval.ipynb`.
Load your pre-built artifacts, then wrap them:

```python
import pickle
import chromadb
from rag.retrieval.agents.bm25  import BM25Agent
from rag.retrieval.agents.dense import DenseAgent
from rag.retrieval.agents.graph import GraphAgent

# Load pre-built indexes from notebook outputs
with open("storage/bm25_fixed_qe.pkl", "rb") as f:
    bm25_index = pickle.load(f)

chroma_client = chromadb.PersistentClient(path="storage/vectordb_dense")
collection    = chroma_client.get_collection("dense_fixed")

# graph_retriever is the object produced by load_graphrag.py in the notebook
# from rag.retrieval.agents.graph import GraphAgent

bm25  = BM25Agent(bm25_index, corpus_docs)   # corpus_docs: list of LangChain Documents
dense = DenseAgent(collection)
graph = GraphAgent(graph_retriever)
```

**Choose an orchestrator:**

```python
from rag.retrieval.orchestrators.waterfall  import WaterfallOrchestrator
from rag.retrieval.orchestrators.voting     import VotingOrchestrator
from rag.retrieval.orchestrators.confidence import ConfidenceOrchestrator
from rag.retrieval.orchestrators.adaptive   import AdaptiveOrchestrator

# Waterfall: cheap — only calls GraphRAG when BM25 and Dense diverge
orchestrator = WaterfallOrchestrator(bm25, dense, graph, overlap_threshold=0.3)

# Voting: all three agents, equal weights — simplest baseline
orchestrator = VotingOrchestrator(bm25, dense, graph)

# Confidence: auto-detects query type and adjusts weights
orchestrator = ConfidenceOrchestrator(bm25, dense, graph)

# Adaptive: learns optimal weights online via Q-learning
orchestrator = AdaptiveOrchestrator(bm25, dense, graph, epsilon=0.1)

# All orchestrators share the same interface:
docs, trace = orchestrator.retrieve("Who is the rector of ETH Zurich?", top_k=10)
print(trace)   # {"strategy": "voting", "bm25_hits": 30, ...}
```

**Standalone agent usage** (no orchestrator):

```python
results = bm25.search("How many students are at ETH?", top_k=10)
results = dense.search("renewable energy research initiatives", top_k=10)
results = graph.retrieve("What departments collaborate on AI research?", top_k=10)
```

---

### Step 3 — Evaluation

```python
import json
from rag.evaluation.metrics   import load_qrels
from rag.evaluation.evaluator import ComprehensiveEvaluator
from rag.evaluation.analyzer  import AgentComplementarityAnalyzer, FailureAnalyzer

# Load relevance judgements produced by rag-score (step 1f)
qrels = load_qrels("benchmark/score/semantic", min_score=0.5)

# Load benchmark questions
with open("data/benchmark/benchmark_qa.json") as f:
    qa_data = json.load(f)

# Evaluate one or more strategies
evaluator = ComprehensiveEvaluator(qrels)
evaluator.evaluate_retriever(VotingOrchestrator(bm25, dense, graph), qa_data, name="Voting")
evaluator.evaluate_retriever(ConfidenceOrchestrator(bm25, dense, graph), qa_data, name="Confidence")

# Compare strategies
print(evaluator.compare_strategies())

# Statistical significance between two strategies
print(evaluator.statistical_significance_test("Voting", "Confidence", metric="ndcg_cut_10"))

# Visualise metric distributions and latency
evaluator.plot_metric_distributions("ndcg_cut_10")
evaluator.plot_latency_comparison()

# Analyse agent complementarity across all queries
comp = AgentComplementarityAnalyzer(bm25, dense, graph)
df   = comp.batch_analyze([q["question"] for q in qa_data])
print(df.describe())

# Surface failure patterns (queries with NDCG@10 < 0.5)
results = evaluator.results["Voting"]
fa      = FailureAnalyzer(qrels)
failures = fa.identify_failures(results["per_query"], threshold=0.5)
patterns = fa.analyze_failure_patterns(failures, qa_data)
fa.print_summary(patterns)
```

---

## Dependencies

| Category | Key packages |
|---|---|
| Document processing | `beautifulsoup4`, `lxml`, `docling`, `pdfplumber`, `dateparser` |
| NLP | `spacy` (en + de models), `yake`, `lingua-language-detector` |
| Retrieval | `rank-bm25`, `chromadb`, `langchain`, `langchain-community`, `langchain-huggingface` |
| ML / Embeddings | `torch`, `transformers` (M2M100), `nltk` |
| LLM APIs | `openai`, `python-dotenv`, `pydantic>=2.0` |
| Evaluation | `pytrec-eval`, `scikit-learn`, `scipy` |
| Data / Viz | `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `tqdm` |
| Dev | `pytest`, `mypy`, `ruff` |

All runtime dependencies are declared in `pyproject.toml` and installed automatically by `pip install -e ".[dev]"`.
