import re
from typing import List

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

synth_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
synth_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16,  # half-precision to fit in 14 GB VRAM
)


def split_sentences(text: str) -> List[str]:
    """Split text on sentence-ending punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def select_best_sentences(text: str, query: str, max_sentences: int = 5) -> List[str]:
    """
    Return the `max_sentences` sentences from `text` with the highest
    token overlap with `query`.  Used to trim context before prompting.
    """
    q_tokens = set(query.lower().split())
    scored = [
        (len(q_tokens & set(s.lower().split())), s)
        for s in split_sentences(text)
        if len(s) > 20  # skip very short fragments
    ]
    scored.sort(reverse=True)
    return [s for _, s in scored[:max_sentences]]


class AnswerSynthesizerAgent:
    """
    Generates a short factual answer from retrieved documents.

    Pipeline:
    1. For each of the top-7 documents, extract the 2 most query-relevant sentences
    2. Concatenate into a compact context block
    3. Prompt Mistral-7B-Instruct with a grounded QA template
    4. Return the generated answer (or "NOT FOUND IN CONTEXT")
    """

    def __init__(self, max_new_tokens: int = 128):
        self.tokenizer = synth_tokenizer
        self.model = synth_model
        self.max_new_tokens = max_new_tokens

    def generate(self, query: str, docs: List[Document]) -> str:
        # Build a compact, query-relevant context from top-7 documents
        context_blocks = []
        for d in docs[:7]:
            raw = d.metadata.get("original_text") or d.page_content
            sents = select_best_sentences(raw, query, max_sentences=2)
            if sents:
                context_blocks.append(" ".join(sents))
        context = "\n".join(context_blocks)

        prompt = (
            "[INST]\n"
            "Answer the question using the context.\n"
            "Give a short, factual answer.\n"
            'If the answer is not clearly supported by the context, say "NOT FOUND IN CONTEXT".\n\n'
            f"Context:\n{context}\n\nQuestion: {query}\n[/INST]\n"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return decoded.split("[/INST]")[-1].strip()
