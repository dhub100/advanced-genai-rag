"""
Bilingual EN ↔ DE query translation using the M2M100 model.

The model is loaded lazily the first time ``translate`` is called so that
importing this module does not trigger a GPU/CPU load.
"""

from __future__ import annotations

import functools
import logging
from functools import lru_cache
from typing import Optional

import torch
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

logger = logging.getLogger(__name__)


class EnDeTranslator:
    """Greedy EN⇄DE translation (cached – 1st call downloads weights)."""

    def __init__(self, model="facebook/m2m100_418M", device: str | None = None) -> None:
        self.device = device
        self.tok = M2M100Tokenizer.from_pretrained(model)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model).to(
            self.device
        )

    @functools.lru_cache(maxsize=512)
    def translate(self, text: str, tgt: str) -> str:
        src = detect(text) if text.strip() else "en"
        src = src if src in ("en", "de") else "en"
        if src == tgt:
            return text
        self.tok.src_lang = src
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids,
            forced_bos_token_id=self.tok.get_lang_id(tgt),
            num_beams=1,
            max_new_tokens=128,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)


translator = EnDeTranslator(
    model="facebook/m2m100_418M", device="cuda" if torch.cuda.is_available() else "cpu"
)
