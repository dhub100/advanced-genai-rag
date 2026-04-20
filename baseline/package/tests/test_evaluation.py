"""Unit tests for the evaluation module."""

import pathlib
import json
import tempfile

from rag.evaluation.metrics import load_qrels, METRICS
from rag.evaluation.analyzer import FailureAnalyzer


# ---------------------------------------------------------------------------
# Metrics / qrels loader
# ---------------------------------------------------------------------------

def test_load_qrels_filters_below_threshold(tmp_path):
    score_file = tmp_path / "chunk_abc.json"
    score_file.write_text(json.dumps({
        "1": {"relevance_score": 0.8, "relevance_reason": "exact"},
        "2": {"relevance_score": 0.2, "relevance_reason": "weak"},
    }), encoding="utf-8")

    qrels = load_qrels(tmp_path, min_score=0.5)
    assert "chunk_abc" in qrels.get("1", {})
    assert "chunk_abc" not in qrels.get("2", {})


def test_metrics_set_contains_ndcg():
    assert "ndcg_cut_10" in METRICS
    assert "recip_rank"  in METRICS


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------

def test_failure_analyzer_identify():
    per_query = {
        "1": {"ndcg_cut_10": 0.9},
        "2": {"ndcg_cut_10": 0.1},
        "3": {"ndcg_cut_10": 0.3},
    }
    analyzer = FailureAnalyzer(qrels={})
    failures = analyzer.identify_failures(per_query, threshold=0.5)
    assert set(failures) == {"2", "3"}


def test_failure_analyzer_classify_question():
    assert FailureAnalyzer._classify("Who is the rector?") == "Who"
    assert FailureAnalyzer._classify("How many grants?")   == "How"
    assert FailureAnalyzer._classify("Describe the policy") == "Other"
