"""
Shared NLP utilities: lazy spaCy model loading, YAKE keyword extraction,
and Lingua language detection.

These are thin re-exports so that both preprocessing and retrieval modules
can share a single model instance rather than loading duplicates.
"""

import logging
import re

import spacy
import yake
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detector (singleton)
# ---------------------------------------------------------------------------

_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN)
    .with_preloaded_language_models()
    .build()
)


def detect_language(text: str) -> str:
    """Detect the dominant language of a text string using the Lingua detector.

    Args:
        text: Plain text to classify.

    Returns:
        Lowercase BCP-47 language code (``"en"``, ``"de"``, ``"fr"``,
        ``"it"``), or an empty string if detection fails or text is blank.
    """
    if not text.strip():
        return ""
    try:
        lang = _detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else ""
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# spaCy (lazy per-language)
# ---------------------------------------------------------------------------

_spacy_cache: dict = {}
_SPACY_MODELS = {"en": "en_core_web_sm", "de": "de_core_news_sm"}


def get_spacy(lang: str):
    """Return a cached spaCy NLP model for the given language, or None.

    Loads the model on first call; subsequent calls return the cached
    instance.  Supports English (``"en"`` → ``en_core_web_sm``) and German
    (``"de"`` → ``de_core_news_sm``).

    Args:
        lang: BCP-47 language code.

    Returns:
        Loaded ``spacy.Language`` object, or ``None`` if the language is
        unsupported or the model fails to load.
    """
    if lang not in _SPACY_MODELS:
        return None
    if lang not in _spacy_cache:
        try:
            _spacy_cache[lang] = spacy.load(_SPACY_MODELS[lang])
        except Exception as exc:
            logger.error("spaCy load failed for %s: %s", lang, exc)
            _spacy_cache[lang] = None
    return _spacy_cache[lang]


# ---------------------------------------------------------------------------
# YAKE keyword extraction (lazy per-language)
# ---------------------------------------------------------------------------

_yake_cache: dict = {}


def extract_keywords(text: str, lang: str, top_k: int = 10) -> list[str]:
    """Extract the top-k keywords from text using a cached YAKE extractor.

    Args:
        text: Plain text to analyse.
        lang: Language code for YAKE (``"en"``, ``"de"``, ``"fr"``, ``"it"``).
            Falls back to ``"en"`` for unsupported codes.
        top_k: Maximum number of keywords to return (default 10).

    Returns:
        List of keyword strings sorted by YAKE relevance score (most
        relevant first).  Returns an empty list if text is blank.
    """
    if not text.strip():
        return []
    yake_lang = lang if lang in ("en", "de", "fr", "it") else "en"
    if yake_lang not in _yake_cache:
        try:
            _yake_cache[yake_lang] = yake.KeywordExtractor(lan=yake_lang, n=3, top=top_k)
        except Exception:
            _yake_cache[yake_lang] = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    kw_scores = _yake_cache[yake_lang].extract_keywords(text)
    kw_scores.sort(key=lambda x: x[1])
    return [kw for kw, _ in kw_scores[:top_k]]
