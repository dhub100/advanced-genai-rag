#!/usr/bin/env python3
"""
HTML → minimal JSON using BeautifulSoup (boilerplate removal) + Docling (text extraction).

Output schema per document
--------------------------
{
  "doc_id":    str,   # SHA-1 of the relative file path
  "filename":  str,
  "raw_text":  str,
  "paragraphs": [str, ...]
}

CLI
---
    rag-preprocess <input_dir> <output_dir>
    python -m rag.preprocessing.html_parser <input_dir> <output_dir>
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import unicodedata
import argparse
from pathlib import Path

from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_converter = DocumentConverter()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def strip_boilerplate_with_bs(html_text: str) -> str:
    """Remove navigation / decorative tags before passing to Docling."""
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()
    return str(soup)


def docling_extraction(cleaned_html: str) -> str:
    """Extract plain text via Docling; fall back to BS4 get_text on failure."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp.write(cleaned_html.encode("utf-8"))
        tmp_path = tmp.name

    try:
        result = _converter.convert(tmp_path)
        return result.document.export_to_text()
    except Exception as exc:
        logging.error("Docling extraction error: %s", exc)
        return BeautifulSoup(cleaned_html, "lxml").get_text(separator="\n")
    finally:
        os.remove(tmp_path)


def normalize_lines(text: str) -> str:
    """NFC-normalize, strip each line, drop empty lines, re-join."""
    lines = [
        unicodedata.normalize("NFC", ln.strip())
        for ln in text.splitlines()
        if ln.strip()
    ]
    return "\n".join(lines)


def naive_paragraph_split(text: str) -> list[str]:
    """Split on blank lines; fall back to per-line split for single-paragraph docs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) <= 1:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paras


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_single_file(html_file: Path, in_base: Path) -> dict:
    """Parse one HTML file into a minimal JSON record.

    Reads the file, strips boilerplate with BeautifulSoup, extracts plain
    text via Docling (with BS4 fallback), normalises Unicode, and splits
    into paragraphs.

    Args:
        html_file: Absolute path to the ``.html`` source file.
        in_base: Root input directory used to compute the relative path for
            the ``doc_id`` SHA-1 hash.

    Returns:
        Dict with keys ``"doc_id"``, ``"filename"``, ``"raw_text"``,
        and ``"paragraphs"``.  Returns an empty dict on read failure.
    """
    rel_path = html_file.relative_to(in_base).as_posix()
    doc_id = hashlib.sha1(rel_path.encode()).hexdigest()

    try:
        raw_html = html_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logging.error("Failed to read %s: %s", html_file, exc)
        return {}

    stripped = strip_boilerplate_with_bs(raw_html)
    extracted = docling_extraction(stripped)
    raw_text = normalize_lines(extracted)

    return {
        "doc_id": doc_id,
        "filename": html_file.name,
        "raw_text": raw_text,
        "paragraphs": naive_paragraph_split(raw_text),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the HTML parser.

    Recursively discovers ``.html`` files under ``input_dir``, processes
    each with :func:`process_single_file`, and writes output JSON files
    mirroring the source directory tree under ``output_dir``.
    """
    parser = argparse.ArgumentParser(description="Step 1a: HTML → minimal JSON.")
    parser.add_argument("input_dir",  help="Directory containing .html files.")
    parser.add_argument("output_dir", help="Directory for output JSON files.")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0, count = time.time(), 0
    for html_file in in_dir.rglob("*.html"):
        if not html_file.is_file():
            continue
        record = process_single_file(html_file, in_dir)
        if not record:
            continue

        rel     = html_file.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        logging.info("[html_parser] %s => %s", html_file, out_path)
        count += 1

    logging.info("Parsed %d files in %.2fs.", count, time.time() - t0)


if __name__ == "__main__":
    main()
