import importlib
import pathlib

import torch

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


def load_dense_fixed(
    dense_loader_path: str = "/content/drive/MyDrive/Adv_GenAI_FS26/storage/subsample/vectordb_dense/load_dense_fixed.py",
) -> DenseRetriever:
    DEVICE = "cuda" if torch.is_available() else "cpu"
    dense_loader = importlib.machinery.SourceFileLoader(
        "dense_mod", dense_loader_path
    ).load_module()
    dense_fixed = dense_loader.load_dense_fixed(device=DEVICE, k=100)
    return dense_fixed
