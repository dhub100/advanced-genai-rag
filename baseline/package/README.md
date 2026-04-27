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
  - [`notebooks/`](#notebooks)
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
│       └── evaluation/             # Step 3 — metrics and analysis
│           ├── __init__.py
│           ├── evaluator.py        # ComprehensiveEvaluator (P@k, MRR, NDCG, latency)
│           ├── analyzer.py         # AgentComplementarityAnalyzer, FailureAnalyzer
│           ├── metrics.py          # METRICS constant + load_qrels()
│           └── rational.py         # ExplainableOrchestrator — decision rationale
│
├── notebooks/
│   ├── 02_retrieval_orchestration.ipynb   # Step 2: end-to-end retrieval walkthrough
│   └── 03_evaluation_and_analysis.ipynb   # Step 3: metrics, analysis, and bonus experiments
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

### `notebooks/`

End-to-end Jupyter notebooks that demonstrate the full pipeline using the package.
Each notebook is self-contained: it declares its own dependencies, provides a configuration section for local paths, and can be re-run independently.

| Notebook | Description |
|---|---|
| `02_retrieval_orchestration.ipynb` | Builds and exercises the multi-agent retrieval system. Covers bilingual BM25 (M2M100 query expansion + pseudo-relevance feedback), dense retrieval with multilingual E5 embeddings (ChromaDB), and GraphRAG traversal. Shows how Weighted RRF fuses ranked lists and demonstrates three orchestration strategies — **Waterfall** (lazy GraphRAG invocation), **Voting** (fixed equal weights), and **Confidence** (query-type-adaptive weights). Includes a per-strategy evaluation using P@k, MRR, and NDCG@10. |
| `03_evaluation_and_analysis.ipynb` | Comprehensive evaluation and analysis on top of the retrievers from Notebook 02. Covers standard IR metrics (P@k, Recall@k, MRR, NDCG), paired t-tests for statistical significance, agent complementarity analysis (exclusive vs. shared document overlap), orchestrator explainability, failure analysis (queries with low NDCG@10 and their linguistic patterns), and latency profiling (P95). |

---

## How to use

### Installation

Python 3.10+ is required. Install the package in editable mode with all development dependencies:

```bash
cd package
pip install -e
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

### Pipeline

See the notebook to have a look at both the retrieval pipeline and the evaluation.

---

## Dependencies

| Category | Key packages |
|---|---|
| Document processing | `beautifulsoup4`, `lxml`, `docling`, `pdfplumber`, `dateparser` |
| NLP | `spacy` (en + de models), `yake`, `lingua-language-detector` |
| Retrieval | `rank-bm25`, `chromadb`, `langchain`, `langchain-community`, `langchain-huggingface`, `faiss-cpu` |
| ML / Embeddings | `torch`, `transformers` (M2M100, flan-t5, Mistral-7B), `sentence-transformers`, `nltk` |
| LLM APIs | `openai`, `python-dotenv`, `pydantic>=2.0` |
| Evaluation | `pytrec-eval`, `scikit-learn`, `scipy` |
| Data / Viz | `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `tqdm` |
| Dev | `pytest`, `mypy`, `ruff` |

All runtime dependencies are declared in `pyproject.toml` and installed automatically by `pip install -e ".[dev]"`.
