# Advanced GenAI RAG — Python Package

An installable Python package implementing an agentic, multi-strategy Retrieval-Augmented Generation (RAG) system.
The system ingests ETH Zurich news articles (HTML), builds BM25, dense-vector, and knowledge-graph indexes, and evaluates
three orchestration strategies against a benchmark Q&A set using standard IR metrics.

---

## Table of Contents

- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
  - [`src/rag/preprocessing/`](#srcragpreprocessing)
  - [`src/rag/retrieval/`](#srcragretrieval)
  - [`src/rag/evaluation/`](#srcragevaluation)
  - [`src/rag/utils/`](#srcragutils)
  - [`notebooks/`](#notebooks)
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

    E --> M["BilingualBM25 / QEBM25\nrank-bm25 + M2M100 bilingual expansion"]
    E --> N["DenseRetriever\nChromaDB semantic search"]
    E --> O["GraphAgent\nKnowledge-graph retriever"]

    M --> P{Orchestrator}
    N --> P
    O --> P

    P -->|Waterfall| Q[BM25+Dense → conditional Graph]
    P -->|Voting| R[All three · equal-weight RRF]
    P -->|Confidence| S[Query-type-adaptive weights]

    Q --> U[Reciprocal Rank Fusion]
    R --> U
    S --> U

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
│       │   │   ├── bm25.py             # BilingualBM25 + QEBM25 (PRF wrapper)
│       │   │   ├── dense.py            # DenseRetriever — ChromaDB semantic search
│       │   │   ├── graph.py            # GraphAgent — knowledge-graph retrieval
│       │   │   ├── query_classifier.py # QueryClassifierAgent — flan-t5 query typing
│       │   │   └── answer_synthesizer.py # AnswerSynthesizerAgent — Mistral answer gen
│       │   ├── retriever/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # BaseOrchestrator + RRF fusion helpers
│       │   │   ├── waterfall.py    # WaterfallRetriever — conditional GraphRAG
│       │   │   ├── voting.py       # VotingRetriever — all agents, equal-weight RRF
│       │   │   └── confidence.py   # ConfidenceRetriever — query-type-adaptive weights
│       │   ├── orchestrator.py     # Orchestrator — unified strategy runner
│       │   └── translator.py       # EnDeTranslator — M2M100 EN ↔ DE translation
│       │
│       ├── evaluation/             # Step 3 — metrics and analysis
│       │   ├── __init__.py
│       │   ├── evaluator.py        # ComprehensiveEvaluator (P@k, MRR, NDCG, latency)
│       │   ├── analyzer.py         # AgentComplementarityAnalyzer, FailureAnalyzer
│       │   ├── metrics.py          # METRICS constant + load_qrels()
│       │   └── rational.py         # ExplainableOrchestrator — decision rationale
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
Each sub-step is independently runnable as a CLI command.

| Module | CLI entry point | Purpose |
|---|---|---|
| `html_parser.py` | `rag-preprocess` | Strips navigation/boilerplate HTML tags with BeautifulSoup, then extracts clean plain text via Docling. Outputs one minimal JSON per HTML file with fields `doc_id`, `filename`, `raw_text`, and `paragraphs`. |
| `cleaner.py` | `rag-clean` | Two-pass enrichment. Pass 1 removes known boilerplate lines (regex) and builds a corpus-wide paragraph frequency table. Pass 2 drops paragraphs repeated across ≥ N documents, detects document language (Lingua), extracts named entities (spaCy), keywords (YAKE), and a short summary. Outputs enriched JSON with `language`, `named_entities`, `keywords`, `summary`, and `text_stats`. |
| `validator.py` | `rag-validate` | Quality gate — discards any document whose `paragraphs_cleaned` list is empty after cleaning. Files are renamed to `<doc_id>.json` in the output directory for stable downstream referencing. |
| `benchmark.py` | `rag-benchmark` | Extracts numbered Q&A pairs from `BenchmarkQuestionsAnswers.pdf` using pdfplumber. Applies a curated OCR correction table (handles `Z`-substitution artefacts common in scanned PDFs) and separates answer text from grading notes. Outputs a JSON array `[{"id": int, "question": str, "answer": str}, ...]`. |
| `metadata.py` | `rag-metadata` | Calls GPT-4o-mini via the OpenAI structured-output API with a strict Pydantic schema (`MetadataSchema`) to extract entities, topic tags, event dates, role annotations, numeric facts, department, document type, and content date per chunk. Skips already-processed chunks (idempotent). Requires `OPENAI_API_KEY`. |
| `relevance.py` | `rag-score` | Calls GPT-4o-mini for each unscored chunk, asking it to assign a `relevance_score` (0–1) and a short reason for every one of the 25 benchmark questions. Results are saved as `<chunk_id>.json` in the score directory and later consumed by `load_qrels()` to build pytrec_eval-compatible relevance judgements. Requires `OPENAI_API_KEY`. |

---

### `src/rag/retrieval/`

Implements three retrieval agents, four agent-support modules, and three orchestration strategies that combine them.

#### Agents (`retrieval/agents/`)

| Module | Class | Description |
|---|---|---|
| `bm25.py` | `BilingualBM25` | Wraps two `rank_bm25.BM25Okapi` indexes (one per language). Before scoring, the query is translated to both English and German using `EnDeTranslator`, so German documents are reachable from English queries and vice versa. Scores from both language variants are summed and deduplicated by `chunk_id`. |
| `bm25.py` | `QEBM25` | PRF (pseudo-relevance feedback) wrapper around `BilingualBM25`. Retrieves an initial candidate set and expands the query with top-document terms before a second retrieval pass. |
| `dense.py` | `DenseRetriever` | Wraps a ChromaDB collection for embedding-based semantic search. Prepends the `query:` prefix required by multilingual-E5 embeddings. Factory function `load_dense_fixed()` loads the HuggingFace embedding model and connects to the persistent vector store. |
| `graph.py` | `GraphAgent` | Thin adapter around an external knowledge-graph retriever built in the notebook. Provides a `.retrieve(query, level, k_comms, top_k)` interface for community-level graph traversal. |
| `query_classifier.py` | `QueryClassifierAgent` | Classifies an incoming query as `"FACTOID"`, `"SEMANTIC"`, or `"HYBRID"` using a fine-tuned `google/flan-t5-base` model. Used by `ConfidenceRetriever` to select weight presets. |
| `answer_synthesizer.py` | `AnswerSynthesizerAgent` | Generates a natural-language answer from a ranked list of retrieved documents using `Mistral-7B-Instruct`. Called by `Orchestrator.run()` after retrieval. |

#### Retrievers (`retrieval/retriever/`)

All retrievers extend `BaseOrchestrator`, which implements shared RRF fusion and parallel agent execution.

| Module | Class | Strategy |
|---|---|---|
| `base.py` | `BaseOrchestrator` | Base class — wraps the three agents, exposes `orchestrate_parallel_fusion(query, top_k, pre_k, use_graph, weights, apply_overlap_rerank)`, and provides `_rrf_fuse()` (weighted Reciprocal Rank Fusion with `k=60`) and `rerank()` (overlap-based re-scoring). |
| `waterfall.py` | `WaterfallRetriever` | Runs BM25 and Dense first. Only invokes the (slower) GraphRAG agent when the Jaccard overlap between the two result sets falls below a configurable threshold (default 0.3). Exposes `waterfall_orchestrate()` and a `search()` alias. |
| `voting.py` | `VotingRetriever` | Runs all three agents in parallel with equal weights and fuses via RRF. The simplest strategy — robust and unbiased. Exposes `voting_orchestrate()` and a `search()` alias. |
| `confidence.py` | `ConfidenceRetriever` | Uses `QueryClassifierAgent` to classify the query and selects a pre-defined weight preset before calling all three agents. Factoid queries up-weight BM25 (1.4/0.9/0.5); semantic queries up-weight Dense (0.9/1.3/0.6); hybrid queries use a neutral split (1.0/1.1/0.5). Exposes `confidence_orchestrate()` and a `search()` alias. |

#### Top-level modules

| Module | Class / function | Description |
|---|---|---|
| `orchestrator.py` | `Orchestrator` | Unified entry point that holds all three retrievers and an `AnswerSynthesizerAgent`. `run(strategy, query, top_k)` dispatches to the requested strategy and returns `{"query", "strategy", "trace", "documents", "answer"}`. |
| `translator.py` | `EnDeTranslator` | Lazily loads `facebook/m2m100_418M` on first call (CPU or CUDA). `translate(text, tgt)` returns a cached translation. Module-level singleton `translator` is shared across all callers so the model is loaded at most once. |

---

### `src/rag/evaluation/`

Evaluation and analysis tools for comparing retrieval strategies.

| Module | Classes / Functions | Description |
|---|---|---|
| `metrics.py` | `METRICS`, `load_qrels()` | `METRICS` is the set of pytrec_eval metric strings computed by default (P@1/3/5/10, Recall@5/10/100, MRR, NDCG@5/10). `load_qrels(folder, min_score)` reads the per-chunk score files produced by `relevance.py` and builds a `{query_id: {doc_id: 1}}` dict compatible with pytrec_eval. Only chunks with `relevance_score >= min_score` (default 0.5) are marked relevant. |
| `evaluator.py` | `ComprehensiveEvaluator` | Central evaluation class. `evaluate_retriever(retriever, qa_data, name)` runs the retriever over all benchmark questions, records per-query timings, and stores pytrec_eval results internally. `compare_strategies()` returns a summary DataFrame. `statistical_significance_test()` runs a paired t-test. `plot_metric_distributions()` and `plot_latency_comparison()` produce matplotlib/seaborn visualisations. |
| `analyzer.py` | `AgentComplementarityAnalyzer`, `FailureAnalyzer` | `AgentComplementarityAnalyzer` runs all three agents on the same query and computes set-overlap statistics (how many documents each agent retrieves exclusively vs. in common with the others). Supports single-query analysis, interactive Plotly charts, and batch analysis. `FailureAnalyzer` identifies queries whose NDCG@10 falls below a threshold and clusters them by query length, digit/name presence, and question-word type to surface systematic weaknesses. |
| `rational.py` | `ExplainableOrchestrator` | Standalone explainability wrapper that mirrors the `ConfidenceRetriever` weight logic but emits a step-by-step `explanation` dict alongside results. `explainable_route(query, top_k)` returns `(docs, explanation)`; `print_explanation(explanation)` renders it as a human-readable decision trace. |

---

### `src/rag/utils/`

Shared low-level utilities.

| Module | Description |
|---|---|
| `io.py` | `load_json(path)` and `save_json(data, path)` wrappers with UTF-8/UTF-8-BOM handling and automatic parent-directory creation. |
| `nlp.py` | Module-level singletons for spaCy (en/de), YAKE, and Lingua so that model loading happens at most once per process, regardless of how many pipeline steps import these helpers. Exposes `detect_language()`, `get_spacy()`, and `extract_keywords()`. |

---

### `notebooks/`

End-to-end Jupyter notebooks that demonstrate the full pipeline using the package.
Each notebook is self-contained: it declares its own dependencies, provides a configuration section for local paths, and can be re-run independently.

| Notebook | Description |
|---|---|
| `02_retrieval_orchestration.ipynb` | Builds and exercises the multi-agent retrieval system. Covers bilingual BM25 (M2M100 query expansion + pseudo-relevance feedback), dense retrieval with multilingual E5 embeddings (ChromaDB), and GraphRAG traversal. Shows how Weighted RRF fuses ranked lists and demonstrates three orchestration strategies — **Waterfall** (lazy GraphRAG invocation), **Voting** (fixed equal weights), and **Confidence** (query-type-adaptive weights). Includes a per-strategy evaluation using P@k, MRR, and NDCG@10. |
| `03_evaluation_and_analysis.ipynb` | Comprehensive evaluation and analysis on top of the retrievers from Notebook 02. Covers standard IR metrics (P@k, Recall@k, MRR, NDCG), paired t-tests for statistical significance, agent complementarity analysis (exclusive vs. shared document overlap), orchestrator explainability, failure analysis (queries with low NDCG@10 and their linguistic patterns), and latency profiling (P95). |

---

### `tests/`

pytest test suite with one file per pipeline stage:

| File | Covers |
|---|---|
| `test_preprocessing.py` | HTML parsing, cleaning, validation, and benchmark extraction |
| `test_retrieval.py` | Agent search interfaces, retrievers, fusion, and classifier |
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

**Run each step individually:**

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

The retrieval agents wrap indexes built in `notebooks/02_retrieval_orchestration.ipynb`.
Load your pre-built artifacts, then wrap them:

```python
import pickle
import chromadb
from rag.retrieval.agents.bm25  import BilingualBM25, QEBM25
from rag.retrieval.agents.dense import DenseRetriever, load_dense_fixed
from rag.retrieval.agents.graph import GraphAgent

# Load pre-built BM25 index from notebook outputs
with open("storage/bm25_fixed_qe.pkl", "rb") as f:
    bm25_index = pickle.load(f)

bm25  = BilingualBM25(corpus_docs)   # corpus_docs: list of LangChain Documents
# or with PRF expansion:
qebm25 = QEBM25(base=bm25)

# Load ChromaDB dense retriever
dense = load_dense_fixed(device="cpu")

# graph_retriever is the object produced by load_graphrag.py in the notebook
graph = GraphAgent(graph_retriever)
```

**Choose a retriever:**

```python
from rag.retrieval.retriever.waterfall  import WaterfallRetriever
from rag.retrieval.retriever.voting     import VotingRetriever
from rag.retrieval.retriever.confidence import ConfidenceRetriever

# Waterfall: cheap — only calls GraphRAG when BM25 and Dense diverge
retriever = WaterfallRetriever(bm25, dense, graph, overlap_threshold=0.3)

# Voting: all three agents, equal weights — simplest baseline
retriever = VotingRetriever(bm25, dense, graph)

# Confidence: auto-detects query type and adjusts weights
retriever = ConfidenceRetriever(bm25, dense, graph)

# All retrievers share a search() interface:
docs = retriever.search("Who is the rector of ETH Zurich?", top_k=10)
```

**Unified orchestrator** (retrieval + answer synthesis):

```python
from rag.retrieval.orchestrator import Orchestrator
from rag.retrieval.agents.answer_synthesizer import AnswerSynthesizerAgent

orchestrator = Orchestrator(bm25, dense, graph, synthesizer=AnswerSynthesizerAgent())
result = orchestrator.run("confidence", "Who is the rector of ETH Zurich?", top_k=5)
# result = {"query": ..., "strategy": "confidence", "trace": [...], "documents": [...], "answer": "..."}
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
evaluator.evaluate_retriever(VotingRetriever(bm25, dense, graph), qa_data, name="Voting")
evaluator.evaluate_retriever(ConfidenceRetriever(bm25, dense, graph), qa_data, name="Confidence")

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
| ML / Embeddings | `torch`, `transformers` (M2M100, flan-t5, Mistral-7B), `nltk` |
| LLM APIs | `openai`, `python-dotenv`, `pydantic>=2.0` |
| Evaluation | `pytrec-eval`, `scikit-learn`, `scipy` |
| Data / Viz | `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `tqdm` |
| Dev | `pytest`, `mypy`, `ruff` |

All runtime dependencies are declared in `pyproject.toml` and installed automatically by `pip install -e ".[dev]"`.
