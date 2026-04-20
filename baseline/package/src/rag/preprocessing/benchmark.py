#!/usr/bin/env python3
"""
Extract question-answer pairs from the benchmark PDF.

Handles OCR artefacts with targeted regex replacements, then parses numbered
Q&A blocks from the corrected plain text.

Output schema
-------------
[{"id": int, "question": str, "answer": str, "notes": str (optional)}, ...]

CLI
---
    rag-benchmark <pdf_path> <output_json_path>
    python -m rag.preprocessing.benchmark <pdf_path> <output_json_path>
"""

import argparse
import json
import os
import re

import pdfplumber


# ---------------------------------------------------------------------------
# OCR correction table
# ---------------------------------------------------------------------------

_OCR_REPLACEMENTS = [
    (r"compeZZon",               "competition"),
    (r"compeZZve",               "competitive"),
    (r"iniZaZve",                "initiative"),
    (r"iniZaZves",               "initiatives"),
    (r"addiZonal",               "additional"),
    (r"addiZon",                 "addition"),
    (r"mulZple",                 "multiple"),
    (r"opZon",                   "option"),
    (r"miZgaZng",                "mitigating"),
    (r"producZvity",             "productivity"),
    (r"projecZons",              "projections"),
    (r"direcZon",                "direction"),
    (r"capabiliZes",             "capabilities"),
    (r"deveZop",                 "develop"),
    (r"seZngs",                  "settings"),
    (r"explaZion",               "explanation"),
    (r"quesZon",                 "question"),
    (r"quesZons",                "questions"),
    (r"elevaZon",                "elevation"),
    (r"elevaZons",               "elevations"),
    (r"organizaZon",             "organization"),
    (r"acZvely",                 "actively"),
    (r"experZse",                "expertise"),
    (r"collaboraZons",           "collaborations"),
    (r"condiZons",               "conditions"),
    (r"insZtuZonal",             "institutional"),
    (r"insZtuZon",               "institution"),
    (r"insZtute",                "institute"),
    (r"parZcular",               "particular"),
    (r"conZnues",                "continues"),
    (r"acceleraZon",             "acceleration"),
    (r"deflecZon",               "deflection"),
    (r"\|",                      "l"),
    (r"PaTern",                  "Pattern"),
    (r"EllH",                    "ETH"),
    (r"tiurich",                 "Zurich"),
    (r"lladeus",                 "Tadeus"),
    (r"llilman",                 "Tilman"),
    (r"llumor",                  "Tumor"),
    (r" llanDEM-X",              " TanDEM-X"),
    (r"Engimmune llherapeutics", "Engimmune Therapeutics"),
    (r"swiwly",                  "swiftly"),
    (r"culng-edge",              "cutting-edge"),
    (r"shiw",                    "shift"),
    (r"pallern",                 "pattern"),
    (r"Anthony Pall",            "Anthony Patt"),
    (r"Net tiero",               "Net Zero"),
    (r"llCRs",                   "TCRs"),
    (r"Elioll Ash",              "ElioT Ash"),
    (r"llobias Donner",          "Tobias Donner"),
]


def fix_ocr_typos(text: str) -> str:
    """Apply the pre-built OCR correction table to a text string.

    Iterates through ``_OCR_REPLACEMENTS`` and applies each regex
    substitution in order.

    Args:
        text: Raw text extracted from a PDF page, potentially containing
            OCR artefacts such as ``"Z"`` stand-ins for ``"ti"`` or ``"Z"``.

    Returns:
        Text with known OCR artefacts replaced.
    """
    for pattern, replacement in _OCR_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_note_line(line: str) -> bool:
    """Return True if the line looks like a scoring note rather than an answer.

    Args:
        line: A single text line from the benchmark PDF.

    Returns:
        True if the line contains any note/scoring keyword.
    """
    note_keywords = [
        "score", "point", "criterion", "criteria", "explanation", "should",
        "note", "this is about", "deflect", "ambiguous", "answer should",
    ]
    return any(kw in line.lower() for kw in note_keywords)


def _parse_qa_blocks(lines: list[str]) -> list[dict]:
    """Parse numbered Q&A blocks from a list of text lines.

    Identifies question lines via the pattern ``<digit>. <text>`` and
    collects subsequent lines as the answer until the next numbered item.
    Lines that look like scoring notes are separated into an optional
    ``"notes"`` field.

    Args:
        lines: Cleaned, OCR-corrected text lines from the PDF.

    Returns:
        List of dicts with keys ``"id"`` (int), ``"question"`` (str),
        ``"answer"`` (str), and optionally ``"notes"`` (str).
    """
    qa_pairs: list[dict] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\d+)\.\s+(.+)", lines[i])
        if not m:
            i += 1
            continue

        q_id     = int(m.group(1))
        question = m.group(2).strip()

        answer_lines, j = [], i + 1
        while j < len(lines):
            if re.match(r"^\d+\.\s", lines[j]):
                break
            if lines[j]:
                answer_lines.append(lines[j])
            j += 1

        answer, notes, is_note = [], [], False
        for line in answer_lines:
            if _is_note_line(line):
                is_note = True
            (notes if is_note else answer).append(line)

        # If the first "note" line doesn't look like a scoring note, merge back
        if notes and not _is_note_line(notes[0]):
            answer += notes
            notes   = []

        clean_answer = " ".join(answer).strip()
        if clean_answer and not clean_answer.endswith((".", "?", "!", '"', "\u201d", "'")) \
                and len(clean_answer.split()) > 3:
            clean_answer += "."

        qa: dict = {"id": q_id, "question": question, "answer": clean_answer}
        if notes:
            qa["notes"] = " ".join(notes).strip()
        qa_pairs.append(qa)
        i = j

    return qa_pairs


def _merge_split_q12(qa_pairs: list[dict]) -> list[dict]:
    """Fix Q12 whose answer text bleeds into the question field due to PDF layout.

    Args:
        qa_pairs: List of Q&A dicts as produced by ``_parse_qa_blocks``.

    Returns:
        The same list with Q12's question and answer corrected in-place.
    """
    for qa in qa_pairs:
        if qa["id"] == 12 and qa["answer"].startswith("over the years"):
            qa["question"] = qa["question"].rstrip("?") + " over the years?"
            qa["answer"]   = re.sub(r"^over the years\?\s*", "", qa["answer"])
    return qa_pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_qa_from_pdf(pdf_path: str) -> list[dict]:
    """Extract and clean all Q&A pairs from the benchmark PDF.

    Opens the PDF with pdfplumber, concatenates page text, applies OCR
    corrections, parses numbered Q&A blocks, and applies known layout fixes.

    Args:
        pdf_path: Path to ``BenchmarkQuestionsAnswers.pdf``.

    Returns:
        List of Q&A dicts with keys ``"id"``, ``"question"``, ``"answer"``,
        and optionally ``"notes"``.
    """
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    lines    = [fix_ocr_typos(ln.strip()) for ln in raw_text.splitlines()]
    qa_pairs = _parse_qa_blocks(lines)
    qa_pairs = _merge_split_q12(qa_pairs)

    for qa in qa_pairs:
        qa["question"] = qa["question"].strip()
        qa["answer"]   = qa["answer"].strip()
        if "notes" in qa:
            qa["notes"] = qa["notes"].strip()

    return qa_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the benchmark extractor.

    Parses ``pdf_path`` and ``output_path`` arguments, extracts Q&A pairs
    via :func:`extract_qa_from_pdf`, and writes the result as JSON.
    """
    parser = argparse.ArgumentParser(description="Step 1d: extract Q&A pairs from benchmark PDF.")
    parser.add_argument("pdf_path",    help="Path to BenchmarkQuestionsAnswers.pdf")
    parser.add_argument("output_path", help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    qa_pairs = extract_qa_from_pdf(args.pdf_path)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(qa_pairs)} Q&A pairs → {args.output_path}")


if __name__ == "__main__":
    main()
