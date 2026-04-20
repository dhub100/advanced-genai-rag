"""
Bilingual EN ↔ DE query translation using the M2M100 model.

The model is loaded lazily the first time ``translate`` is called so that
importing this module does not trigger a GPU/CPU load.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_model     = None
_tokenizer = None
_device    = None


def _load_model(device: str = "cpu"):
    """Load the M2M100 translation model and tokenizer if not already loaded.

    Idempotent — subsequent calls return immediately if the model is cached.
    Sets module-level ``_model``, ``_tokenizer``, and ``_device`` globals.

    Args:
        device: Torch device string (``"cpu"`` or ``"cuda"``).  When
            ``None`` or empty, auto-selects CUDA if available.
    """
    global _model, _tokenizer, _device
    if _model is not None:
        return
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    import torch

    _device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/m2m100_418M"
    logger.info("Loading M2M100 translation model on %s …", _device)
    _tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    _model     = M2M100ForConditionalGeneration.from_pretrained(model_name).to(_device)
    logger.info("M2M100 loaded.")


def translate(text: str, src_lang: str, tgt_lang: str, device: str = "cpu") -> str:
    """Translate text between languages using the M2M100 model.

    Args:
        text: Source text to translate.
        src_lang: BCP-47 source language code (e.g. ``"en"``, ``"de"``).
        tgt_lang: BCP-47 target language code.
        device: Torch device string passed to :func:`_load_model`.

    Returns:
        Translated string.  Falls back to the original text if translation
        raises an exception.
    """
    _load_model(device)
    try:
        _tokenizer.src_lang = src_lang
        encoded = _tokenizer(text, return_tensors="pt").to(_device)
        tgt_id  = _tokenizer.get_lang_id(tgt_lang)
        out     = _model.generate(**encoded, forced_bos_token_id=tgt_id)
        return _tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    except Exception as exc:
        logger.warning("Translation failed (%s → %s): %s", src_lang, tgt_lang, exc)
        return text


def expand_query(query: str, src_lang: str = "en", device: str = "cpu") -> list[str]:
    """Return the original query plus its translation for bilingual BM25 expansion.

    Args:
        query: Source query string.
        src_lang: Language code of ``query`` (``"en"`` or ``"de"``).
        device: Torch device string forwarded to :func:`translate`.

    Returns:
        Two-element list ``[original_query, translated_query]``.
    """
    tgt = "de" if src_lang == "en" else "en"
    return [query, translate(query, src_lang, tgt, device)]
