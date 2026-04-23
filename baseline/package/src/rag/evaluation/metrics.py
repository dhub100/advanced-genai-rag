"""
IR metric constants and relevance-judgement loader.
"""

import json
import pathlib
from collections import defaultdict

METRICS = {
    "P_1",
    "P_3",
    "P_5",
    "P_10",
    "recall_5",
    "recall_10",
    "recall_100",
    "recip_rank",
    "ndcg_cut_5",
    "ndcg_cut_10",
}


def load_qrels(folder: pathlib.Path, min_score: float = 0.5) -> dict:
    """
    Build a pytrec_eval-compatible qrels dict from per-chunk score files.

    Each file is named ``<chunk_id>.json`` and contains
    ``{question_id: {"relevance_score": float, ...}, ...}``.
    A chunk is considered relevant for a question when its score >= min_score.
    """
    qrels: dict = defaultdict(dict)
    for fp in pathlib.Path(folder).glob("*.json"):
        doc_id = fp.stem
        scores = json.loads(fp.read_text(encoding="utf-8"))
        for qid, payload in scores.items():
            if payload["relevance_score"] >= min_score:
                qrels[qid][doc_id] = 1
    return dict(qrels)
