#!/usr/bin/env python3
"""
Script: step_1_hybrid.py
Purpose:
  Minimal HTML parsing using a hybrid approach:
    - BeautifulSoup to remove large boilerplate elements (scripts, styles, nav, etc.)
    - Docling for robust text extraction from the stripped HTML.
    - Outputs only minimal JSON fields.

Resulting JSON schema:
{
  "doc_id": str,        # unique identifier (SHA-1 or file_path stem)
  "filename": str,      # original filename
  "raw_text": str,      # entire cleaned text
  "paragraphs": [ ... ] # naive list of paragraphs
}

Usage:
    python step_1_hybrid.py [input_dir] [output_dir]
Example:
    python step_1_hybrid.py data data_cleaned/minimal_hybrid
"""

import os
import re
import json
import time
import logging
import argparse
import unicodedata
import hashlib
from pathlib import Path
import tempfile

from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

converter = DocumentConverter()

def strip_boilerplate_with_bs(html_text: str) -> str:
    """
    Remove known boilerplate tags using BeautifulSoup, returning an HTML string
    that's lighter for Docling to process.
    """
    soup = BeautifulSoup(html_text, "lxml")

    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()

    # If you'd like, you could also remove known repetitive text or disclaimers here
    # but let's keep it minimal for now.

    return str(soup)

def docling_extraction(cleaned_html: str) -> str:
    """
    Feed the stripped HTML to docling.DocumentConverter and return extracted text.
    If Docling fails, fallback to plain text extraction from BeautifulSoup.
    """
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp.write(cleaned_html.encode("utf-8"))
        tmp.flush()
        tmp_path = tmp.name

    try:
        conv_result = converter.convert(tmp_path)
        doc_text = conv_result.document.export_to_text()
    except Exception as e:
        logging.error(f"Docling extraction error: {e}")
        # fallback: extract text with BeautifulSoup's get_text
        doc_text = BeautifulSoup(cleaned_html, "lxml").get_text(separator="\n")
    finally:
        os.remove(tmp_path)

    return doc_text

def normalize_lines(text: str) -> str:
    """
    Normalize unicode, strip each line, remove empty lines,
    and re-join with single newlines.
    """
    lines = [
        unicodedata.normalize("NFC", ln.strip())
        for ln in text.splitlines()
        if ln.strip()
    ]
    return "\n".join(lines)

def naive_paragraph_split(text: str) -> list[str]:
    """
    Split on blank lines. If there's only one chunk, fallback to splitting on each line.
    """
    paras = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in paras if p.strip()]
    if len(paras) <= 1:
        # fallback: each line is its own paragraph
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paras

def process_single_file(html_file: Path, in_base: Path) -> dict:
    """
    1. Compute doc_id (via SHA-1 of relative path).
    2. Read HTML file.
    3. Strip boilerplate with BeautifulSoup.
    4. Extract text with Docling.
    5. Normalize lines & split into paragraphs.
    6. Return minimal record.
    """
    # 1) doc_id
    rel_path = html_file.relative_to(in_base).as_posix()
    doc_id = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()

    # 2) read the HTML
    try:
        raw_html = html_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Failed to read {html_file}: {e}")
        return {}

    # 3) strip boilerplate
    stripped_html = strip_boilerplate_with_bs(raw_html)

    # 4) docling extraction
    extracted = docling_extraction(stripped_html)

    # 5) normalize & paragraph split
    raw_text = normalize_lines(extracted)
    paragraphs = naive_paragraph_split(raw_text)

    # 6) build record
    record = {
        "doc_id": doc_id,
        "filename": html_file.name,
        "raw_text": raw_text,
        "paragraphs": paragraphs
    }
    return record

def main():
    parser = argparse.ArgumentParser(description="Step 1: Hybrid minimal parse (BS+Docling).")
    parser.add_argument("input_dir", help="Directory containing .html files.")
    parser.add_argument("output_dir", help="Directory to store minimal JSON.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    count = 0

    for html_file in in_dir.rglob("*.html"):
        if not html_file.is_file():
            continue

        record = process_single_file(html_file, in_dir)
        if not record:
            # indicates an error or empty result
            continue

        # output path mirrors relative structure
        rel = html_file.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # write JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        logging.info(f"[STEP1] {html_file} => {out_path}")
        count += 1

    dur = time.time() - start_time
    logging.info(f"Completed minimal hybrid parsing of {count} files in {dur:.2f}s.")

if __name__ == "__main__":
    main()
