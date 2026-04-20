#!/usr/bin/env python3
"""
LLM-based structured metadata extraction for semantic chunks (pipeline step 1e).

Calls GPT-4o-mini with a strict Pydantic schema to extract entities, topic tags,
event dates, role annotations, numeric facts, and more from each chunk.

Requires OPENAI_API_KEY in the environment (or a .env file).

CLI
---
    rag-metadata --chunks <dir> [--meta-dir <dir>]
    python -m rag.preprocessing.metadata --chunks subsample/semantic_chunk
"""

import argparse
import json
import os
import pathlib
from typing import Any, List, Optional

import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Role(BaseModel):
    """A person-role annotation extracted from a text chunk.

    Attributes:
        person: Full name of the person.
        role: Normalised role title (e.g. ``"rector"``, ``"professor"``).
        from_: Start year of the role tenure (may be None).
        to: End year of the role tenure (may be None).
    """

    person: str
    role: str
    from_: Optional[int] = Field(None, alias="from")
    to:    Optional[int]


class Event(BaseModel):
    """A dated event mentioned in a text chunk.

    Attributes:
        label: Short description of the event.
        year: Four-digit year.
        month: Month number (1–12), or None if unknown.
        day: Day of month, or None if unknown.
    """

    label: str
    year: int
    month: Optional[int]
    day:   Optional[int]


class NumericFact(BaseModel):
    """A numeric measurement or statistic extracted from a text chunk.

    Attributes:
        name: Human-readable label for the fact.
        value: Numeric value.
        unit: Unit of measurement (e.g. ``"CHF"``, ``"%"``, ``"km"``).
        year: Reference year for the fact (may be None).
    """

    name: str
    value: float
    unit: str
    year: Optional[int]


class Entities(BaseModel):
    """Structured named-entity lists for a text chunk.

    Attributes:
        person: List of person name strings.
        org: List of organisation name strings.
        location: List of location name strings.
    """

    person:   List[str] = []
    org:      List[str] = []
    location: List[str] = []


class MetadataSchema(BaseModel):
    """Full LLM-extracted metadata record for a single text chunk.

    This schema is used both as the structured output format for the
    GPT-4o-mini API call and as the persisted JSON record on disk.

    Attributes:
        id: Chunk identifier (mirrors ``chunk_id`` from the source chunk).
        chunk_summary: ≤ 30-word plain-text summary of the chunk.
        entities: Structured named entities.
        topic_tags: Free-form topic label strings.
        event_dates: Dated events mentioned in the chunk.
        role_annotations: Person-role pairs mentioned in the chunk.
        numeric_facts: Numeric measurements or statistics.
        department: ETH department names mentioned (may be empty).
        document_type: Coarse document category (e.g. ``"news article"``).
        content_year: Primary publication or reference year.
        content_month: Primary publication month (1–12).
        initiative: Named research or strategic initiatives.
        grant_type: Grant or funding instrument names.
    """

    id:               str
    chunk_summary:    str
    entities:         Entities
    topic_tags:       List[str] = []
    event_dates:      List[Event] = []
    role_annotations: List[Role] = []
    numeric_facts:    List[NumericFact] = []
    department:       List[str] = []
    document_type:    str
    content_year:     int
    content_month:    int
    initiative:       List[str] = []
    grant_type:       List[str] = []
    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYSTEM_MSG = """
Return ONE JSON object that validates against the schema.
• Do NOT add or delete keys; use "" or [] if unknown.
• chunk_summary ≤ 30 words, no quotes.
• role_annotations.role must be a normalised title (president, rector, …).
• event_dates numbers only {"year":2024,"month":5,"day":17}, month/day null if unknown.
• content_month integer 1-12.
Output must be pure JSON – no markdown – and must not invent facts.
""".strip()


def _load_json(path: pathlib.Path) -> dict:
    """Read a JSON file, trying UTF-8 and UTF-8-BOM encodings.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        RuntimeError: If the file cannot be decoded as UTF-8.
    """
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} as UTF-8")


def _dedup(seq: list[str]) -> list[str]:
    """Return a deduplicated copy of a list, preserving first-occurrence order.

    Args:
        seq: Input list of strings.

    Returns:
        New list with duplicates removed.
    """
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _deep_lower(obj: Any) -> Any:
    """Recursively lower-case all strings in a nested dict/list structure.

    String lists are also deduplicated after lowercasing.

    Args:
        obj: Arbitrary JSON-compatible Python object.

    Returns:
        New object of the same structure with all strings lower-cased.
    """
    if isinstance(obj, str):
        return obj.lower()
    if isinstance(obj, list):
        lowered = [_deep_lower(v) for v in obj]
        return _dedup(lowered) if all(isinstance(v, str) for v in lowered) else lowered
    if isinstance(obj, dict):
        return {k: _deep_lower(v) for k, v in obj.items()}
    return obj


def _normalise(meta: MetadataSchema) -> MetadataSchema:
    """Normalise a MetadataSchema by lower-casing strings and collapsing whitespace.

    Args:
        meta: Parsed MetadataSchema instance from the LLM response.

    Returns:
        New MetadataSchema with all string fields lower-cased and
        ``chunk_summary`` whitespace-collapsed.
    """
    data = _deep_lower(meta.model_dump())
    data["chunk_summary"] = " ".join(data["chunk_summary"].split())
    return MetadataSchema(**data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for LLM metadata extraction.

    Iterates over all chunk JSON files in the specified directory.  Skips
    chunks that already have a metadata file.  For each remaining chunk,
    sends a structured prompt to GPT-4o-mini, validates the response
    against :class:`MetadataSchema`, normalises the output, and writes it
    to disk.
    """
    ap = argparse.ArgumentParser(description="Step 1e: LLM metadata extraction per chunk.")
    ap.add_argument("--chunks",   default="subsample/semantic_chunk",
                    help="Folder with raw chunk JSON files.")
    ap.add_argument("--meta-dir", default=None,
                    help="Output directory (default: benchmark/metadata/<chunk_folder>).")
    args = ap.parse_args()

    chunk_dir = pathlib.Path(args.chunks).resolve()
    if args.meta_dir:
        meta_dir = pathlib.Path(args.meta_dir)
    else:
        subfolder = chunk_dir.name.removesuffix("_chunk")
        meta_dir  = pathlib.Path("benchmark/metadata") / subfolder
    meta_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(chunk_dir.glob("*.json")):
        raw = _load_json(fp)
        cid = raw["metadata"]["chunk_id"]
        out_path = meta_dir / f"{cid}.json"
        if out_path.exists():
            continue

        user_prompt = f"### Chunk ID\n{raw['id']}\n\n### Chunk text\n{raw['text']}"
        response = _client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=MetadataSchema,
        )
        meta = _normalise(response.choices[0].message.parsed)
        out_path.write_text(json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("✓", cid)


if __name__ == "__main__":
    main()
