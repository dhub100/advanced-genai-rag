import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class QueryClassifierAgent:
    """Lightweight query classifier using flan-t5."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, query: str) -> str:
        """Short prompt that flan-t5 can handle."""
        return f"""Classify as FACTOID, SEMANTIC, or HYBRID.

FACTOID = names, numbers, dates, who/when/where
SEMANTIC = why, how, explain, conceptual
HYBRID = mix of both

Query: {query}

Answer:"""

    def classify(self, query: str):
        """Main entry point: classify query and return analysis."""
        prompt = self._build_prompt(query)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
