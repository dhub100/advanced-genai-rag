"""Unit tests for the preprocessing module."""

import json
import textwrap
from pathlib import Path

import pytest

from rag.preprocessing.html_parser import (
    naive_paragraph_split,
    normalize_lines,
    strip_boilerplate_with_bs,
)
from rag.preprocessing.validator import is_valid
from rag.preprocessing.benchmark import fix_ocr_typos, _merge_split_q12


# ---------------------------------------------------------------------------
# html_parser
# ---------------------------------------------------------------------------

def test_normalize_lines_strips_and_joins():
    text = "  hello  \n\n  world  \n"
    assert normalize_lines(text) == "hello\nworld"


def test_naive_paragraph_split_blank_lines():
    text = "para one\n\npara two\n\npara three"
    assert naive_paragraph_split(text) == ["para one", "para two", "para three"]


def test_naive_paragraph_split_fallback_to_lines():
    text = "only one paragraph here"
    result = naive_paragraph_split(text)
    assert result == ["only one paragraph here"]


def test_strip_boilerplate_removes_nav():
    html = "<html><nav>NAV</nav><p>Content</p></html>"
    result = strip_boilerplate_with_bs(html)
    assert "NAV" not in result
    assert "Content" in result


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def test_is_valid_accepts_nonempty():
    assert is_valid({"paragraphs_cleaned": ["some text"]}) is True


def test_is_valid_rejects_empty_list():
    assert is_valid({"paragraphs_cleaned": []}) is False


def test_is_valid_rejects_missing_key():
    assert is_valid({}) is False


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def test_fix_ocr_typos_basic():
    assert fix_ocr_typos("EllH Zurich") == "ETH Zurich"
    assert fix_ocr_typos("compeZZve") == "competitive"


def test_merge_split_q12_noop_when_no_split():
    qa = [{"id": 1, "question": "q?", "answer": "a"}]
    assert _merge_split_q12(qa) == qa
