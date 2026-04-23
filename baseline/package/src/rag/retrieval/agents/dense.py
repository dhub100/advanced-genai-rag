import pathlib

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
INDEX_DIR = pathlib.Path(
    "/content/drive/MyDrive/Adv_GenAI/storage/full_corpus/vectordb_dense/fixed_e5"
)
LOADER_FILE = INDEX_DIR.parent / "load_dense_fixed.py"


class DenseRetriever:
    """Light wrapper that adds the e5 'query:' prefix and returns similarity"""

    def __init__(self, vectordb, k: int = 100):
        self.store, self.k = vectordb, k

    def _prep(self, q: str) -> str:  # e5 query format
        return "query: " + q.strip()

    def search(self, query: str, top_k: int | None = None):
        k = top_k or self.k
        hits = self.store.similarity_search_with_score(self._prep(query), k=k)
        docs = []
        for doc, dist in hits:  # cosine *distance*
            doc.metadata["dense_score"] = 1.0 - float(dist)
            docs.append(doc)
        return docs


def load_dense_fixed(device: str | None = None, k: int = 100) -> DenseRetriever:
    """Factory – returns a DenseRetriever ready for inference"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    embeds = HuggingFaceEmbeddings(
        model_name="{EMBED_MODEL}",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )
    vectordb = Chroma(
        persist_directory=str(pathlib.Path(r"{INDEX_DIR}")),
        embedding_function=embeds,
    )
    return DenseRetriever(vectordb, k=k)
