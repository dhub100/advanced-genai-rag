"""
Reliability-oriented evaluation metrics for the reliable RAG pipeline.

These metrics go beyond standard IR quality (P@k, MRR) and measure how well
the system handles uncertainty, avoids hallucination, and knows when to abstain.

All functions are pure Python — no model calls, no GPU required.  They consume
the result records produced by ``ReliableOrchestrator.run()`` (as collected in
the evaluation loop) together with gold-label annotations per query.
"""

from __future__ import annotations

from typing import Any


def compute_reliability_metrics(
    results: list[dict[str, Any]],
    gold_labels: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute all reliability-oriented metrics from a batch of pipeline results.

    Parameters
    ----------
    results:
        One dict per query, as written by the evaluation loop in the notebook.
        Required keys: ``query``, ``abstained`` (bool), ``trigger`` (str|None),
        ``recoveries`` (int), ``groundedness_score`` (float|None),
        ``trust_score`` (float|None), ``answer`` (str).
        Optional key: ``rouge_l`` (float) — pre-computed per-query ROUGE-L score,
        used for the trust–correctness alignment metric.
    gold_labels:
        One dict per query.  Required keys: ``query`` (str, must match the query
        field in ``results``), ``is_answerable`` (bool — True when a correct
        grounded answer exists in the corpus).

    Returns
    -------
    dict
        Flat mapping of metric name → float value (all between 0 and 1 unless
        noted).  Returns ``float("nan")`` for metrics that cannot be computed
        (e.g. Spearman correlation when fewer than 3 data points are available).
    """
    answerable_map: dict[str, bool] = {
        g["query"]: g["is_answerable"] for g in gold_labels
    }

    answered = [r for r in results if not r.get("abstained", False) and r.get("answer")]
    abstained = [r for r in results if r.get("abstained", False)]
    recovered = [r for r in results if r.get("recoveries", 0) > 0]
    recovered_and_answered = [r for r in recovered if not r.get("abstained", False)]

    n_total = len(results)
    n_answered = len(answered)
    n_abstained = len(abstained)

    # ── Grounded answer rate ──────────────────────────────────────────────────
    # Among answered queries: fraction where groundedness_score >= 0.5.
    # Captures how often the synthesized answer is supported by retrieved evidence.
    grounded = [
        r for r in answered
        if r.get("groundedness_score") is not None and r["groundedness_score"] >= 0.5
    ]
    grounded_answer_rate = len(grounded) / n_answered if n_answered else float("nan")

    # ── Unsupported claim rate ────────────────────────────────────────────────
    # Complement of grounded_answer_rate: fraction of answered queries where the
    # answer is not fully supported by the retrieved evidence.
    unsupported_claim_rate = 1.0 - grounded_answer_rate if n_answered else float("nan")

    # ── Contradiction rate ────────────────────────────────────────────────────
    # Fraction of answered queries where the groundedness score is exactly 0.0,
    # which the GroundednessVerifier assigns when the answer CONTRADICTS the
    # retrieved evidence (UNGROUNDED decision).  This approximates "contradiction
    # handling quality" — a low rate means the system rarely produces answers
    # that actively contradict its own evidence.
    contradicted = [
        r for r in answered
        if r.get("groundedness_score") is not None and r["groundedness_score"] == 0.0
    ]
    contradiction_rate = len(contradicted) / n_answered if n_answered else float("nan")

    # ── Abstention rate ───────────────────────────────────────────────────────
    abstention_rate = n_abstained / n_total if n_total else float("nan")

    # ── Correct abstention rate ───────────────────────────────────────────────
    # Among truly unanswerable queries (is_answerable=False in gold_labels):
    # fraction where the system correctly abstained.
    unanswerable = [r for r in results if not answerable_map.get(r["query"], True)]
    correct_abstentions = [r for r in unanswerable if r.get("abstained", False)]
    correct_abstention_rate = (
        len(correct_abstentions) / len(unanswerable) if unanswerable else float("nan")
    )

    # ── False abstention rate ─────────────────────────────────────────────────
    # Among truly answerable queries (is_answerable=True in gold_labels):
    # fraction where the system incorrectly abstained.  A high rate means the
    # system is too conservative and loses useful coverage.
    answerable_queries = [r for r in results if answerable_map.get(r["query"], True)]
    false_abstentions = [r for r in answerable_queries if r.get("abstained", False)]
    false_abstention_rate = (
        len(false_abstentions) / len(answerable_queries) if answerable_queries else float("nan")
    )

    # ── Recovery attempt rate ─────────────────────────────────────────────────
    recovery_attempt_rate = len(recovered) / n_total if n_total else float("nan")

    # ── Recovery success rate ─────────────────────────────────────────────────
    # Among queries that triggered at least one recovery attempt: fraction that
    # ultimately produced an answer (i.e. recovery prevented abstention).
    recovery_success_rate = (
        len(recovered_and_answered) / len(recovered) if recovered else float("nan")
    )

    # ── Average trust score ───────────────────────────────────────────────────
    trust_scores = [
        r["trust_score"]
        for r in answered
        if r.get("trust_score") is not None
    ]
    avg_trust_score = sum(trust_scores) / len(trust_scores) if trust_scores else float("nan")

    # ── Trust–correctness alignment (Spearman correlation) ────────────────────
    # Measures whether the system's confidence (trust_score) predicts answer
    # quality (ROUGE-L vs. gold).  A positive correlation means high-confidence
    # answers are indeed more accurate — the system is well-calibrated.
    # Requires pre-computed "rouge_l" field in results (added by the notebook).
    trust_rouge_pairs = [
        (r["trust_score"], r["rouge_l"])
        for r in answered
        if r.get("trust_score") is not None and r.get("rouge_l") is not None
    ]
    trust_correctness_correlation = _spearman(
        [p[0] for p in trust_rouge_pairs],
        [p[1] for p in trust_rouge_pairs],
    )

    return {
        "grounded_answer_rate": round(grounded_answer_rate, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "abstention_rate": round(abstention_rate, 4),
        "correct_abstention_rate": round(correct_abstention_rate, 4),
        "false_abstention_rate": round(false_abstention_rate, 4),
        "recovery_attempt_rate": round(recovery_attempt_rate, 4),
        "recovery_success_rate": round(recovery_success_rate, 4),
        "avg_trust_score": round(avg_trust_score, 4),
        "trust_correctness_correlation": round(trust_correctness_correlation, 4),
    }


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation between two equal-length sequences.

    Returns ``float('nan')`` when fewer than 3 pairs are available, since the
    correlation is meaningless or undefined for very small samples.
    """
    n = len(xs)
    if n < 3:
        return float("nan")

    def _rank(seq: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: seq[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and seq[sorted_idx[j]] == seq[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)
