# Advanced Multi-Agent RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that combines multiple retrieval agents (BM25, Dense, and GraphRAG) with intelligent orchestration strategies. This project implements a complete pipeline from document preprocessing to evaluation, featuring adaptive learning and explainable AI capabilities.

## 🎯 Project Overview

This project implements a sophisticated multi-agent RAG system designed for bilingual (English/German) question-answering tasks. The system uses three complementary retrieval agents and multiple orchestration strategies to achieve optimal retrieval performance.

### Key Features

- **Multi-Agent Retrieval**: BM25 (keyword-based), Dense (semantic), and GraphRAG (knowledge graph)
- **Intelligent Orchestration**: Waterfall, Voting, and Confidence-based strategies
- **Adaptive Learning**: Q-learning based orchestrator that learns optimal strategies
- **Comprehensive Evaluation**: Quantitative metrics (P@k, Recall@k, MRR, NDCG) and qualitative analysis
- **Explainability**: Detailed rationales for orchestration decisions
- **Bilingual Support**: English and German query processing with automatic translation

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Features](#features)
- [Contributing](#contributing)

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Processing                       │
│  (Step 1: Cleaning, Metadata Extraction, Chunking)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Multi-Agent Retrieval System                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │   BM25   │  │  Dense   │  │ GraphRAG │                  │
│  │ (Keyword)│  │(Semantic)│  │  (Graph) │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │             │                         │
│       └─────────────┼─────────────┘                         │
│                     │                                         │
│       ┌─────────────▼─────────────┐                         │
│       │   Orchestration Layer     │                         │
│       │  (Waterfall/Voting/Conf)  │                         │
│       └─────────────┬─────────────┘                         │
└──────────────────────┼──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Evaluation & Analysis                          │
│  (Quantitative, Qualitative, Efficiency Metrics)            │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Agents

1. **BM25 (Keyword-based)**
   - Traditional sparse retrieval with query expansion
   - Bilingual support with automatic translation
   - Best for factoid queries with specific terms

2. **Dense (Semantic)**
   - Vector-based semantic search using embeddings
   - ChromaDB vector store
   - Best for semantic and paraphrased queries

3. **GraphRAG**
   - Knowledge graph-based retrieval
   - Entity and relationship extraction
   - Best for complex queries requiring context

### Orchestration Strategies

1. **Waterfall Strategy**
   - Starts with BM25 + Dense fusion
   - Conditionally adds GraphRAG if overlap is low
   - Efficient for most queries

2. **Voting Strategy**
   - Parallel retrieval from all agents
   - Weighted Reciprocal Rank Fusion (RRF)
   - Balanced performance across query types

3. **Confidence Strategy**
   - Dynamic weight adjustment based on query features
   - BM25-heavy for factoid queries
   - Dense-heavy for semantic queries

4. **Adaptive Strategy** (Bonus)
   - Q-learning based orchestrator
   - Learns optimal strategies per query type
   - Epsilon-greedy exploration/exploitation

## 📁 Project Structure

```
advanced_genAI-main/
│
├── Step 1: Document Processing
│   ├── step_1_hybrid.py                    # Hybrid HTML parsing (BS + Docling)
│   ├── step_1_2_advanced_cleaning_and_metadata.py  # Advanced cleaning & metadata
│   ├── step_1_3_validation_filter.py       # Document validation
│   ├── step_1_4_benchmark_extraction.py    # Benchmark QA extraction
│   ├── 2_1_llm_metadataextraction.py       # LLM-based metadata extraction
│   ├── 2_1_2_relevance_score.py            # Relevance scoring
|   ├── Step_1_Subsample_Baseline_Setup.ipynb
|   └── Step_1_Full_corpus_Baseline_Setup.ipynb
│
├── Step 2: Multi-Agent System Design
│   ├── Step_2_Subsample_Multi_agent_system_design.ipynb  # Main orchestration notebook
│   └── Step_2_Full_corpus_Multi_agent_system_design.ipynb
|
├── Step 3: Evaluation and Analysis
|   ├── Step_3_Evaluation_Script.py         # Evaluation framework (Python)
│   ├── Step_3_Subsample_Evaluation_and_Analysis.ipynb
│   └── Step_3_Full_corpus_Evaluation_and_Analysis.ipynb        
│
└── README.md                               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or Jupyter Notebook
- Google Drive (for data storage)

### Required Packages

```bash
# Core dependencies
pip install langchain langchain-community langchain-core langchain-huggingface
pip install chromadb rank-bm25 langdetect nltk
pip install pytrec-eval scikit-learn seaborn plotly
pip install transformers torch

# Document processing
pip install beautifulsoup4 docling pdfplumber
pip install spacy yake lingua

# Optional: For LLM metadata extraction
pip install openai pydantic python-dotenv
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Trista1208/advanced_genAI.git
   cd advanced_genAI
   ```

2. **Mount Google Drive** (if using Colab)
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

3. **Set up data paths**
   - Update paths in configuration sections to point to your data directory
   - Expected structure:
     ```
     /content/drive/MyDrive/Adv_GenAI/
     ├── storage/
     │   ├── subsample/
     │   │   ├── retrieval_downstream/
     │   │   ├── vectordb_dense/
     │   │   └── retrieval_graph/
     ├── benchmark/
     │   ├── benchmark_qa_bilingual.json
     │   └── score/
     └── ...
     ```

## 📖 Workflow

### Step 1: Document Processing

- Process raw HTML/PDF documents into structured JSON with metadata.
- Retrievers are built

**Output**: Cleaned JSON documents with metadata (entities, keywords, summaries, etc.) + Best Retriever Models

### Step 2: Multi-Agent System Design

Build and configure retrieval agents and orchestration strategies.

1. **Open `Step_2_Full_corpus_Multi_agent_system_design.ipynb` or `Step_2_Subsample_Multi_agent_system_design.ipynb`**
2. **Load retrievers**:
   - BM25 with query expansion
   - Dense vector retriever (ChromaDB)
   - GraphRAG retriever
3. **Implement orchestration strategies**:
   - `waterfall_orchestrate()`
   - `voting_orchestrate()`
   - `confidence_orchestrate()`
4. **Test and validate** retrieval performance

**Key Components**:
- Bilingual BM25 with query expansion
- Dense retriever with semantic embeddings
- GraphRAG for knowledge graph traversal
- RRF (Reciprocal Rank Fusion) for result merging

### Step 3: Evaluation and Analysis

Comprehensive evaluation of the multi-agent system.

1. **Open `Step_3_Evaluation_and_Analysis.ipynb`**
2. **Load Step 2 components** (orchestrators and retrievers)
3. **Run quantitative evaluation**:
   - Precision@k, Recall@k
   - MRR (Mean Reciprocal Rank)
   - NDCG@k
4. **Perform qualitative analysis**:
   - Agent complementarity
   - Failure pattern analysis
   - Explainability demonstrations
5. **Evaluate bonus features**:
   - Adaptive orchestration with RL
   - Adversarial query robustness
   - Human-in-the-loop simulation

## 💻 Usage

### Basic Retrieval

```python
from Multi_agent_system_design_Step_2 import (
    waterfall_orchestrate,
    voting_orchestrate,
    confidence_orchestrate
)

# Waterfall strategy
docs, trace = waterfall_orchestrate("Who was president of ETH in 2003?", top_k=5)

# Voting strategy
docs, trace = voting_orchestrate("What are the main research areas at ETH?", top_k=5)

# Confidence strategy
docs, trace = confidence_orchestrate("How does ETH support entrepreneurship?", top_k=5)
```

### Adaptive Orchestration

```python
from Step_3_Evaluation_and_Analysis import AdaptiveOrchestrator

# Initialize adaptive orchestrator
adaptive_orch = AdaptiveOrchestrator(
    bm25_fixed_qe, 
    dense_fixed, 
    graph_rag,
    learning_rate=0.3,
    epsilon=0.2
)

# Retrieve with learning
docs, explanation = adaptive_orch.retrieve("Your query here", top_k=5)

# Update Q-values based on feedback
adaptive_orch.update_q_value(query_type, strategy, feedback_score)
```

### Evaluation

```python
from Step_3_Evaluation_Script import ComprehensiveEvaluator

# Initialize evaluator
evaluator = ComprehensiveEvaluator(qrels)

# Evaluate a strategy
results = evaluator.evaluate_retriever(
    retriever=WaterfallRetriever(),
    qa_data=qa_data,
    name="Waterfall"
)

# Compare all strategies
comparison_df = evaluator.compare_strategies()
print(comparison_df)

# Statistical significance testing
test_result = evaluator.statistical_significance_test(
    "Waterfall", 
    "Voting", 
    metric='recip_rank'
)
```

## 📊 Evaluation

### Quantitative Metrics

- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents retrieved
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain

### Qualitative Analysis

- **Agent Complementarity**: Overlap analysis between retrieval agents
- **Failure Analysis**: Pattern identification in low-performing queries
- **Explainability**: Detailed rationales for orchestration decisions

### Efficiency Metrics

- **Latency**: Average, P95, P99 response times
- **Throughput**: Queries processed per second
- **Cost Analysis**: Computational resource usage

## ✨ Features

### Core Features

- ✅ Multi-agent retrieval (BM25, Dense, GraphRAG)
- ✅ Three orchestration strategies (Waterfall, Voting, Confidence)
- ✅ Bilingual support (English/German)
- ✅ Query expansion for BM25
- ✅ Weighted RRF fusion
- ✅ Comprehensive evaluation framework

### Bonus Features

- 🎯 **Adaptive Orchestration with RL**
  - Q-learning based strategy selection
  - Epsilon-greedy exploration
  - Query-type specific learning

- 🔍 **Explainable Orchestration**
  - Detailed decision rationales
  - Query feature analysis
  - Strategy selection explanations

- 🛡️ **Adversarial Query Evaluation**
  - Ambiguous query generation
  - Code-switched queries (EN/DE)
  - Paraphrased queries
  - Robustness testing

- 👤 **Human-in-the-Loop Simulation**
  - Simulated human feedback
  - Iterative refinement
  - Learning from feedback

## 🔧 Configuration

### Key Parameters

```python
# Retrieval parameters
TOP_K = 5                    # Number of documents to retrieve
PRE_K = 30                   # Pre-retrieval count for fusion
K_RRF = 60                   # RRF constant

# BM25 weights
BM25_WEIGHT = 1.2
DENSE_WEIGHT = 1.0
GRAPH_WEIGHT = 0.6

# Adaptive learning
LEARNING_RATE = 0.3
EPSILON = 0.2                # Exploration rate
```

### Path Configuration

Update paths in configuration sections:

```python
ROOT = pathlib.Path("/content/drive/MyDrive/Adv_GenAI")
PATH_BM25_PICKLE = ".../bm25_fixed_qe.pkl"
PATH_DENSE_LOADER = ".../load_dense_fixed.py"
PATH_GRAG_LOADER = ".../load_graphrag.py"
PATH_QA = ".../benchmark_qa_bilingual.json"
```

## 🐛 Troubleshooting

### Common Issues

1. **AttributeError: 'AdaptiveOrchestrator' object has no attribute 'update_q_value'**
   - **Solution**: Re-run cell 31 (AdaptiveOrchestrator class definition) before using it

2. **Low success rate in Human-in-the-Loop**
   - **Solution**: The matching logic has been improved. Re-run cells 37 and 38

3. **Import errors**
   - **Solution**: Ensure all required packages are installed (see Installation section)

4. **Path errors**
   - **Solution**: Update paths in configuration sections to match your Google Drive structure

## 📝 Key Improvements

### Recent Fixes

- ✅ Fixed critical Q-learning bug where feedback was applied to wrong strategy
- ✅ Improved answer matching with lenient thresholds and entity detection
- ✅ Added warm-start Q-table initialization
- ✅ Enhanced learning parameters for faster adaptation
- ✅ Increased retrieval coverage (top_k: 5→10)

## 🤝 Contributing

This is an academic/research project. For improvements or bug fixes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request with detailed description

## 📄 License

This project is an assignment for Course: Advanced Generative AI at IDS-HSLU

## 🙏 Acknowledgments

- LangChain for RAG framework
- ChromaDB for vector storage
- pytrec_eval for evaluation metrics
- ETH Zurich for the dataset

## 📚 References

- **RAG**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **RRF**: Reciprocal Rank Fusion for Multiple Retrieval Systems
- **GraphRAG**: Knowledge Graph Enhanced Retrieval
- **Q-Learning**: Reinforcement Learning for Adaptive Systems

---

**Note**: This project is designed to run in Google Colab. Adjust paths and configurations for local execution.

