# Advanced Generative AI — Reliable and Adaptive Agentic RAG Systems

HSLU Applied Information and Data Science, Spring 2026  
Course: Advanced Generative AI  
Team: Sever Alin, Girardin Robin, Huber Daniel

## Overview

This repository contains the third semester capstone project, building on the bilingual multi-agent RAG system developed in the previous two semesters. The focus this semester is on reliability and adaptiveness: the system should not only retrieve and generate answers, but also determine whether its evidence is sufficient, whether the answer is properly supported, and whether it should abstain instead of producing an unreliable answer.

## Repository Structure

```
advanced-genai-rag/
├── baseline/               # Semester 2 baseline code and benchmark
├── package/                # Refactored Python package + reliability mechanisms
│   ├── src/rag/            # Installable package source
│   │   ├── retrieval/      # BM25, Dense, GraphRAG, Orchestrator
│   │   └── reliability/    # Mechanisms A, B, D, E, G, H
│   ├── notebooks/          # Jupyter notebooks (04, 05)
│   └── benchmark/          # Extended benchmark (15 challenging cases)
└── report/                 # Quarto book report
    ├── data/               # Evaluation results (CSV, JSON)
    └── images/             # Figures embedded in the report
```

## Getting Started

The notebooks are designed to run on Google Colab with GPU (A100 recommended). Each notebook installs the package automatically from this repository:

```python
REPO_REF = "main"
```

The notebooks require pre-built retrieval indices (BM25, Dense, GraphRAG) and evaluation results stored on Google Drive. These are not included in the repository due to size constraints (~1GB). The graders can request access to the shared Drive folder `Adv_GenAI_FS26` from the team.

## Report

The full report is written as a Quarto book in `report/`. To render locally:

```bash
quarto render report/
```

## Submission

Deadline: 14 June 2026