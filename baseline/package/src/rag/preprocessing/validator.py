#!/usr/bin/env python3
"""
Quality filter: discard documents whose ``paragraphs_cleaned`` list is empty.

CLI
---
    rag-validate <input_dir> <output_dir>
    python -m rag.preprocessing.validator <input_dir> <output_dir>
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_valid(record: dict) -> bool:
    """Check whether an enriched document record has usable content.

    Args:
        record: Enriched JSON record as produced by ``rag.preprocessing.cleaner``.

    Returns:
        True if ``paragraphs_cleaned`` is a non-empty list.
    """
    cleaned = record.get("paragraphs_cleaned", [])
    return isinstance(cleaned, list) and len(cleaned) > 0


def main() -> None:
    """CLI entry point for the document validator.

    Recursively scans ``input_dir`` for JSON files, passes each through
    :func:`is_valid`, and copies valid records to ``output_dir`` using the
    ``doc_id`` as the filename.
    """
    parser = argparse.ArgumentParser(description="Step 1c: filter empty documents.")
    parser.add_argument("input_dir",  help="Directory with enriched JSON files.")
    parser.add_argument("output_dir", help="Directory for validated JSON files.")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total, kept = 0, 0
    for path in in_dir.rglob("*.json"):
        if not path.is_file():
            continue
        total += 1
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Cannot read %s: %s", path, exc)
            continue

        if not isinstance(record, dict) or not is_valid(record):
            continue

        doc_id   = record.get("doc_id", path.stem)
        out_path = out_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        kept += 1

    logging.info("Scanned %d files; kept %d valid documents.", total, kept)


if __name__ == "__main__":
    main()
