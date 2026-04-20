"""
Shared utilities used across all pipeline stages.

This sub-package centralises helpers that would otherwise be duplicated in
``rag.preprocessing``, ``rag.retrieval``, and ``rag.evaluation``.  Keeping
them here ensures that expensive models (spaCy, YAKE, Lingua) are loaded
at most once per process.

Why it matters:
    Loading large NLP models in each sub-module independently would waste
    memory and add start-up overhead.  The singleton pattern used in
    ``utils.nlp`` means that preprocessing and any retrieval-time NLP share
    the same loaded model instance.

Modules:
    io.py  – UTF-8-safe JSON load / save helpers.
    nlp.py – Singleton language detector (Lingua), lazy spaCy loader, and
             YAKE keyword extractor shared by preprocessing and retrieval.
"""
