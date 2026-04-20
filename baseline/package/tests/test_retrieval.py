"""Unit tests for retrieval utilities."""

from rag.retrieval.classifier import classify_query
from rag.retrieval.fusion import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

def test_classify_factoid_short():
    result = classify_query("ETH rector")
    assert result["label"] == "factoid"


def test_classify_factoid_with_digits():
    result = classify_query("How many students enrolled in 2023?")
    assert result["label"] == "factoid"


def test_classify_semantic_long():
    result = classify_query(
        "What are the main research initiatives and strategic goals "
        "of ETH Zurich in the field of sustainability and climate change?"
    )
    assert result["label"] == "semantic"


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, chunk_id: str):
        self.metadata = {"chunk_id": chunk_id}
        self.page_content = chunk_id


def test_rrf_returns_correct_count():
    a = [_FakeDoc("d1"), _FakeDoc("d2"), _FakeDoc("d3")]
    b = [_FakeDoc("d2"), _FakeDoc("d4")]
    fused = reciprocal_rank_fusion([("a", a), ("b", b)])
    ids = {d.metadata["chunk_id"] for d in fused}
    assert ids == {"d1", "d2", "d3", "d4"}


def test_rrf_top_doc_is_consistently_ranked():
    # d1 appears at rank 1 in both lists → should be top
    a = [_FakeDoc("d1"), _FakeDoc("d2")]
    b = [_FakeDoc("d1"), _FakeDoc("d3")]
    fused = reciprocal_rank_fusion([("a", a), ("b", b)])
    assert fused[0].metadata["chunk_id"] == "d1"
