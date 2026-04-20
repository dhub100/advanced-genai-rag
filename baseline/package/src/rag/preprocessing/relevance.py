#!/usr/bin/env python3
"""
LLM-based chunk-to-question relevance scoring (pipeline step 1f).

For each unscored chunk, calls GPT-4o-mini with all 25 benchmark Q&A pairs
and receives a relevance_score (0–1) and a short reason for each pair.

Requires OPENAI_API_KEY in the environment (or a .env file).

CLI
---
    rag-score [--list-missing] [--chunks <dir>] [--qa-path <path>] [--score-dir <dir>]
    python -m rag.preprocessing.relevance --list-missing
"""

import argparse
import json
import os
import pathlib
from typing import Dict, List, Optional

import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

dotenv.load_dotenv()
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PairScore(BaseModel):
    """Relevance score assigned by the LLM to one (chunk, question) pair.

    Attributes:
        question_id: Integer ID of the benchmark question.
        relevance_score: Float in [0, 1] indicating how relevant the chunk
            is to the question (1.0 = exact answer, 0.0 = unrelated).
        relevance_reason: Optional short explanation (≤ 200 characters).
    """

    question_id:      int
    relevance_score:  float = Field(..., ge=0, le=1)
    relevance_reason: Optional[str]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYS_PROMPT = """
Return ONE JSON array. Each element MUST be:
{
  "question_id": <int>,
  "relevance_score": <float 0-1>,
  "relevance_reason": "<≤200 chars>"
}
Do not add other keys. Do not output markdown.
Scoring rubric:
  1.0 exact answer · 0.8 verbatim · 0.5 partial · 0.2 tangential · 0.0 unrelated
""".strip()


def _build_user_prompt(chunk: dict, qa_items: List[dict], with_answer: bool) -> str:
    """Build the user-turn prompt for the relevance-scoring LLM call.

    Args:
        chunk: Chunk JSON dict containing at least ``metadata.chunk_id``
            and ``text`` fields.
        qa_items: List of benchmark question dicts (bilingual, with ``"id"``,
            ``"question"``, ``"question_de"``, ``"answer"``, ``"answer_de"``).
        with_answer: Whether to include gold answers in the prompt so the
            model can judge relevance more accurately.

    Returns:
        Formatted prompt string ready to send as the ``"user"`` message.
    """
    lines = [
        "### Chunk ID", chunk["metadata"]["chunk_id"], "",
        "### Chunk text", chunk["text"], "",
        "### Questions (25)",
    ]
    for q in qa_items:
        lines.append(f"{q['id']}. EN: {q['question']} / DE: {q['question_de']}")
        if with_answer:
            lines.append(f"   answers → EN: {q['answer']} / DE: {q['answer_de']}")
    return "\n".join(lines).strip()


def _missing_chunks(chunk_dir: pathlib.Path, score_dir: pathlib.Path) -> List[pathlib.Path]:
    """Find chunk files that do not yet have a corresponding score file.

    Args:
        chunk_dir: Directory containing raw chunk JSON files.
        score_dir: Directory where per-chunk score JSON files are written.

    Returns:
        Sorted list of chunk file paths that have not been scored yet.
    """
    todo = []
    for fp in chunk_dir.glob("*.json"):
        cid = json.loads(fp.read_text(encoding="utf-8"))["metadata"]["chunk_id"]
        if not (score_dir / f"{cid}.json").exists():
            todo.append(fp)
    return sorted(todo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for relevance scoring.

    Finds unscored chunks, sends each to GPT-4o-mini with all 25 benchmark
    questions, validates the 25-item response array against
    :class:`PairScore`, and writes a score file per chunk.  Idempotent —
    already-scored chunks are skipped.
    """
    ap = argparse.ArgumentParser(description="Step 1f: score chunk-question relevance pairs.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list-missing", action="store_true",
                      help="List unscored chunks without making API calls.")
    ap.add_argument("--with-answer", dest="with_answer", action="store_true", default=True)
    ap.add_argument("--no-answer",   dest="with_answer", action="store_false")
    ap.add_argument("--chunks",    default="subsample/semantic_chunk")
    ap.add_argument("--qa-path",   default="benchmark/benchmark_qa_bilingual_with_semantic_chunks.json")
    ap.add_argument("--score-dir", default=None)
    args = ap.parse_args()

    chunk_dir = pathlib.Path(args.chunks).resolve()
    if args.score_dir:
        score_dir = pathlib.Path(args.score_dir)
    else:
        score_dir = pathlib.Path("benchmark/score") / chunk_dir.name.removesuffix("_chunk")
    score_dir.mkdir(parents=True, exist_ok=True)

    todo = _missing_chunks(chunk_dir, score_dir)

    if args.list_missing:
        if not todo:
            print("Everything already scored.")
        else:
            print(f"{len(todo)} chunk(s) need scoring:")
            for fp in todo:
                print("-", json.loads(fp.read_text(encoding="utf-8"))["metadata"]["chunk_id"])
        return

    if not todo:
        print("All chunks already scored.")
        return

    qa_items = json.loads(pathlib.Path(args.qa_path).read_text(encoding="utf-8"))
    print(f"Scoring {len(todo)} chunk(s)…\n")

    for chunk_fp in todo:
        raw = json.loads(chunk_fp.read_text(encoding="utf-8"))
        cid = raw["metadata"]["chunk_id"]
        out_path = score_dir / f"{cid}.json"

        user_prompt = _build_user_prompt(raw, qa_items, args.with_answer)
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        try:
            arr = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError as exc:
            print("✗ JSON decode error for", cid, "→", exc)
            continue

        result: Dict[str, dict] = {}
        for item in arr:
            try:
                ps = PairScore(**item)
            except ValidationError as exc:
                print("✗ validation error for", cid, item, "→", exc)
                break
            result[str(ps.question_id)] = {
                "relevance_score":  ps.relevance_score,
                "relevance_reason": ps.relevance_reason,
            }
        else:
            if len(result) != 25:
                print(f"✗ {cid}: expected 25 items, got {len(result)} – skipping")
                continue
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print("✓ scored", cid)


if __name__ == "__main__":
    main()
