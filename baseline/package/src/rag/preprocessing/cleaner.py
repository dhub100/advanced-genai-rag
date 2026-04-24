#!/usr/bin/env python3
"""
Advanced cleaning and metadata enrichment (pipeline step 1b).

Reads minimal JSON produced by ``html_parser``, then:
  1. Removes boilerplate lines matching known patterns.
  2. Drops paragraphs repeated across >= ``threshold`` documents.
  3. Detects document language (Lingua).
  4. Extracts named entities (spaCy), keywords (YAKE), and a short summary.
  5. Writes enriched JSON.

Output schema additions
-----------------------
domain, language, title, date, year, month, source,
main_content, paragraphs_original, paragraphs_cleaned,
named_entities, keywords, summary, text_stats, semantic_chunk_hints,
embedding_vector, doc_embedding

CLI
---
    rag-clean <input_dir> <output_dir> [--threshold N]
    python -m rag.preprocessing.cleaner <input_dir> <output_dir>
"""

import json
import logging
import re
import time
import argparse
from pathlib import Path

import spacy
import yake
from lingua import Language, LanguageDetectorBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Language detection (built once at import time)
# ---------------------------------------------------------------------------

_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN)
    .with_preloaded_language_models()
    .build()
)

# Lazy-loaded caches
_spacy_models: dict = {}
_yake_extractors: dict = {}

# ---------------------------------------------------------------------------
# Boilerplate patterns (line-level, case-insensitive)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r"^Staffnet\s*$",
    r"^Newsletter\sabonnieren\s*$",
    r"^call_made\s*$",
    r"^externe\sSeite\s*$",
    r"^vertical_align_bottom\s*$",
    r"^Download\s*$",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

def _get_spacy(lang: str):
    """Return a cached spaCy model for the given language, or None.

    Loads ``en_core_web_sm`` or ``de_core_news_sm`` on first call and
    caches the result so subsequent calls are instant.

    Args:
        lang: BCP-47 language code (``"en"`` or ``"de"``).

    Returns:
        Loaded ``spacy.Language`` object, or ``None`` if the language is
        unsupported or the model fails to load.
    """
    model_map = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang not in model_map:
        return None
    if lang not in _spacy_models:
        try:
            _spacy_models[lang] = spacy.load(model_map[lang])
        except Exception as exc:
            logging.error("spaCy load failed for %s: %s", lang, exc)
            _spacy_models[lang] = None
    return _spacy_models[lang]


def detect_language(text: str) -> str:
    """Detect the language of a text string using the Lingua detector.

    Args:
        text: Plain text to classify.

    Returns:
        Lowercase ISO 639-1 language code (e.g. ``"en"``, ``"de"``), or an
        empty string if detection fails or text is blank.
    """
    if not text.strip():
        return ""
    try:
        lang = _detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else ""
    except Exception:
        return ""


def extract_entities(text: str, lang: str) -> list[dict]:
    """Extract named entities from text using spaCy.

    Args:
        text: Plain text to process.
        lang: Language code used to select the spaCy model (``"en"``/``"de"``).

    Returns:
        List of dicts with keys ``"text"`` (entity surface form) and
        ``"label"`` (spaCy entity type, e.g. ``"PERSON"``, ``"ORG"``).
        Duplicates are removed while preserving first-occurrence order.
        Returns an empty list if no suitable spaCy model is available.
    """
    nlp = _get_spacy(lang)
    if not nlp:
        return []
    seen, ents = set(), []
    for ent in nlp(text).ents:
        if ent.text not in seen:
            seen.add(ent.text)
            ents.append({"text": ent.text, "label": ent.label_})
    return ents


def extract_keywords(text: str, lang: str, top_k: int = 10) -> list[str]:
    """Extract the top-k keywords from text using YAKE.

    Args:
        text: Plain text to analyse.
        lang: Language code for YAKE (``"en"``, ``"de"``, ``"fr"``, ``"it"``).
            Falls back to ``"en"`` for unsupported codes.
        top_k: Maximum number of keywords to return (default 10).

    Returns:
        List of keyword strings sorted by YAKE score (most relevant first).
        Returns an empty list if text is blank.
    """
    if not text.strip():
        return []
    yake_lang = lang if lang in ("en", "de", "fr", "it") else "en"
    if yake_lang not in _yake_extractors:
        try:
            _yake_extractors[yake_lang] = yake.KeywordExtractor(lan=yake_lang, n=3, top=top_k)
        except Exception:
            _yake_extractors[yake_lang] = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    kw_scores = _yake_extractors[yake_lang].extract_keywords(text)
    kw_scores.sort(key=lambda x: x[1])
    return [kw for kw, _ in kw_scores[:top_k]]


def simple_summary(text: str, lang: str, max_bullets: int = 3) -> str:
    """Generate a short extractive summary from text.

    Prefers bullet points if present, otherwise uses the first two spaCy
    sentences, falling back to a naive sentence splitter if spaCy is
    unavailable.

    Args:
        text: Plain text to summarise.
        lang: Language code used to select the spaCy model.
        max_bullets: Maximum bullet-point lines to include when the text
            contains bullet lists (default 3).

    Returns:
        Summary string (one or two sentences, or up to ``max_bullets``
        bullet points).
    """
    lines = text.splitlines()
    bullets = [ln.strip() for ln in lines if ln.strip().startswith("-")]
    if bullets:
        return "\n".join(bullets[:max_bullets])

    nlp = _get_spacy(lang)
    if nlp:
        sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        return " ".join(sents[:2]) if sents else ""

    parts = re.split(r"(?<=[.!?]) +", text)
    return " ".join(p.strip() for p in parts[:2]) if parts else ""


# ---------------------------------------------------------------------------
# Boilerplate removal
# ---------------------------------------------------------------------------

def _remove_boilerplate(paragraphs: list[str]) -> list[str]:
    """Strip boilerplate lines from each paragraph and return non-empty paragraphs.

    Iterates over ``paragraphs``, removes lines matching ``_BOILERPLATE_RE``,
    and drops any paragraph that becomes empty after removal.

    Args:
        paragraphs: List of raw paragraph strings.

    Returns:
        List of cleaned, non-empty paragraph strings.
    """
    cleaned = []
    for para in paragraphs:
        lines = [ln for ln in para.splitlines() if not _BOILERPLATE_RE.search(ln.strip())]
        merged = " ".join(ln.strip() for ln in lines if ln.strip())
        if merged:
            cleaned.append(merged)
    return cleaned


def _parse_date_from_path(path: Path) -> tuple[str, int | None, int | None]:
    """Extract publication date from a file path that encodes YYYY/MM in its directories.

    Args:
        path: Relative file path whose directory components may include a
            four-digit year and a two-digit month.

    Returns:
        Tuple of (iso_date_str, year_int, month_int).  ``iso_date_str`` is
        formatted as ``"YYYY-MM-DD"`` (day always ``01``), or empty string
        if no year is found.  ``year_int`` and ``month_int`` are ``None``
        when absent.
    """
    year = month = None
    for part in path.parts:
        if re.fullmatch(r"\d{4}", part):
            year = part
        elif re.fullmatch(r"(0[1-9]|1[0-2])", part) and year:
            month = part.zfill(2)
    if year:
        return f"{year}-{month or '01'}-01", int(year), int(month) if month else 1
    return "", None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the document cleaner and enricher.

    Two-pass algorithm:

    1. Load all JSON files, remove boilerplate lines, and build a paragraph
       frequency table.
    2. Filter paragraphs that appear in >= ``threshold`` documents (corpus-level
       boilerplate), run NLP enrichment, and write enriched JSON records.
    """
    parser = argparse.ArgumentParser(description="Step 1b: advanced cleaning and metadata enrichment.")
    parser.add_argument("input_dir",  help="Directory with minimal JSON from html_parser.")
    parser.add_argument("output_dir", help="Directory for enriched JSON.")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Drop paragraphs appearing in >= N documents (default: 5).")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Pass 1: load all docs, remove boilerplate lines, build frequency dict
    docs: list[tuple[Path, dict, list[str]]] = []
    para_freq: dict[str, int] = {}

    for js_path in in_dir.rglob("*.json"):
        try:
            data = json.loads(js_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.error("Cannot read %s: %s", js_path, exc)
            continue
        refined = _remove_boilerplate(data.get("paragraphs", []))
        docs.append((js_path, data, refined))
        for p in refined:
            para_freq[p] = para_freq.get(p, 0) + 1

    # Pass 2: filter repeated paragraphs, extract metadata, write output
    threshold = args.threshold
    out_count = 0

    for js_path, data, refined in docs:
        cleaned = [p for p in refined if para_freq[p] < threshold]
        final_text = "\n".join(cleaned)

        lang = detect_language(final_text)
        iso_date, year_int, month_int = _parse_date_from_path(js_path.relative_to(in_dir))

        record = {
            "doc_id":    data.get("doc_id", ""),
            "filename":  data.get("filename", ""),
            "domain":    "ethz.ch",
            "language":  lang,
            "title":     data.get("title", ""),
            "date":      iso_date,
            "year":      year_int,
            "month":     month_int,
            "source":    "ETH News",
            "main_content":          final_text,
            "paragraphs_original":   data.get("paragraphs", []),
            "paragraphs_cleaned":    cleaned,
            "named_entities":        extract_entities(final_text, lang),
            "keywords":              extract_keywords(final_text, lang),
            "summary":               simple_summary(final_text, lang),
            "text_stats": {
                "char_count":      len(final_text),
                "word_count":      len(final_text.split()),
                "paragraph_count": len(cleaned),
            },
            "semantic_chunk_hints": [{"type": "paragraph_boundaries", "count": len(cleaned)}],
            "embedding_vector": [],
            "doc_embedding":    [],
        }

        rel      = js_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        out_count += 1
        logging.info("[cleaner] %s", out_path)

    logging.info("Enriched %d documents in %.2fs.", out_count, time.time() - t0)


if __name__ == "__main__":
    main()
